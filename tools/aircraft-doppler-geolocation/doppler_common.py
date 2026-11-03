"""Shared physics, geodesy and I/O helpers for the aircraft Doppler geolocation scripts.

Conventions
-----------
* Times are seconds. "Receive time" is when sound reaches the microphone;
  "emission time" is when the aircraft made it. They differ by range / c.
* Positions are converted to a local East-North-Up frame. The conversion is a
  rigid transform of WGS84 ECEF coordinates, so 3D distances are exact (no
  flat-earth approximation).
* c is the speed of sound in still air (343 m/s at about 20 C). Wind is ignored.
"""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np

C_SOUND = 343.0

# --------------------------------------------------------------------------- geodesy

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
FT = 0.3048


def geodetic_to_ecef(lat, lon, h):
    lat = np.radians(np.asarray(lat, float))
    lon = np.radians(np.asarray(lon, float))
    h = np.asarray(h, float)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + h) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz):
    xyz = np.asarray(xyz, float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(8):
        n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - _WGS84_E2 * n / (n + h)))
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def _enu_rotation(lat0, lon0):
    la, lo = math.radians(lat0), math.radians(lon0)
    return np.array([
        [-math.sin(lo), math.cos(lo), 0.0],
        [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
        [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)],
    ])


class LocalFrame:
    """East-North-Up frame anchored at (lat0, lon0, h0)."""

    def __init__(self, lat0, lon0, h0=0.0):
        self.lat0, self.lon0, self.h0 = float(lat0), float(lon0), float(h0)
        self._r = _enu_rotation(self.lat0, self.lon0)
        self._o = geodetic_to_ecef(self.lat0, self.lon0, self.h0)

    def to_enu(self, lat, lon, h):
        d = geodetic_to_ecef(lat, lon, h) - self._o
        return d @ self._r.T

    def to_geodetic(self, enu):
        return ecef_to_geodetic(np.asarray(enu, float) @ self._r + self._o)


# --------------------------------------------------------------------------- physics

def straight_pass_curve(t_recv, f0, speed, slant, t_ca, c=C_SOUND):
    """Received frequency for a source moving in a straight line at constant speed.

    The source passes its closest point of approach (slant range `slant`) at
    emission time `t_ca`. For each receive time we solve exactly for the
    emission time tau (relative to t_ca) from

        T - tau = sqrt(slant^2 + speed^2 tau^2) / c,   T = t_recv - t_ca

    which is a quadratic with one physical (causal, subsonic) root. The Doppler
    factor for a moving source and a still observer is then

        f = f0 / (1 - v_toward / c),   v_toward = -speed^2 tau / r.
    """
    t = np.asarray(t_recv, float) - t_ca
    a = c * c - speed * speed
    tau = (c * c * t - np.sqrt(c * c * speed * speed * t * t + a * slant * slant)) / a
    r = np.sqrt(slant * slant + (speed * tau) ** 2)
    return f0 / (1.0 + speed * speed * tau / (c * r))


def straight_pass_emission_time(t_recv, speed, slant, t_ca, c=C_SOUND):
    """Absolute emission time for each receive time (same geometry as above)."""
    t = np.asarray(t_recv, float) - t_ca
    a = c * c - speed * speed
    tau = (c * c * t - np.sqrt(c * c * speed * speed * t * t + a * slant * slant)) / a
    return t_ca + tau


def initial_guess_from_ridge(t, f, c=C_SOUND):
    """Rough (f0, speed, slant, t_ca) from the ends and steepest part of a ridge."""
    n = len(f)
    k = max(3, n // 10)
    f_hi = float(np.median(f[:k]))
    f_lo = float(np.median(f[-k:]))
    if f_hi <= f_lo:
        f_hi, f_lo = f_lo * 1.001 + 1e-3, f_lo
    f0 = 2.0 * f_hi * f_lo / (f_hi + f_lo)
    speed = float(np.clip(c * (f_hi - f_lo) / (f_hi + f_lo), 10.0, 0.9 * c))
    # time where the curve crosses f0 is (receive time of) closest approach
    idx = int(np.argmin(np.abs(f - f0)))
    t_ca_recv = float(t[idx])
    # slope at CPA: df/dt ~= -f0 v^2 / (c d)
    lo, hi = max(0, idx - k), min(n - 1, idx + k)
    if t[hi] > t[lo]:
        slope = (f[hi] - f[lo]) / (t[hi] - t[lo])
    else:
        slope = -1.0
    slant = float(np.clip(f0 * speed * speed / (c * max(abs(slope), 1e-3)), 30.0, 3.0e4))
    return f0, speed, slant, t_ca_recv - slant / c


# --------------------------------------------------------------------------- audio

def decode_audio(path, sr, start=None, end=None):
    """Mono float32 audio at `sr` Hz. Uses ffmpeg (no shell) or falls back to soundfile."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-nostdin", "-v", "error"]
        if start:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(path)]
        if end is not None:
            cmd += ["-t", f"{end - (start or 0.0):.3f}"]
        cmd += ["-vn", "-ac", "1", "-ar", str(int(sr)), "-f", "f32le", "-"]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")
        return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)

    import soundfile as sf
    from scipy.signal import resample_poly

    x, fs = sf.read(str(path), always_2d=True)
    x = x.mean(axis=1)
    i0 = int(round((start or 0.0) * fs))
    i1 = int(round(end * fs)) if end is not None else len(x)
    x = x[i0:i1]
    if fs != sr:
        g = math.gcd(int(fs), int(sr))
        x = resample_poly(x, int(sr) // g, int(fs) // g)
    return x.astype(np.float64)


# --------------------------------------------------------------------------- CSV helpers

def read_csv_dicts(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def pick_col(columns, wanted, required=True, label=""):
    lower = {c.lower().strip(): c for c in columns}
    for w in wanted:
        if w in lower:
            return lower[w]
    if required:
        raise ValueError(f"no {label or wanted[0]} column; expected one of {wanted}, got {list(columns)}")
    return None


def parse_time(value):
    """Unix seconds from a number (s or ms) or an ISO-8601 string (UTC if no zone)."""
    s = str(value).strip()
    try:
        v = float(s)
        return v / 1000.0 if v > 1e11 else v
    except ValueError:
        pass
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def fmt_utc(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-4] + "Z"

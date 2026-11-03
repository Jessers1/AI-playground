#!/usr/bin/env python3
"""Offline end-to-end self-test on a synthetic flyover. It needs no network or real data.

    python make_synthetic_test.py [--outdir synthetic_out] [--seed 7]

It builds:
  * flyover.wav  60 s of noise (pink background + a steady 120 Hz hum) with a
                 harmonic 110 Hz engine tone from an aircraft at 80 m/s, 900 m up,
                 passing 520 m to the side. The audio is the emitted waveform
                 delayed by range/c sample by sample, so the Doppler shift is exact.
  * track.csv    "ADS-B" for that aircraft (1 Hz, jittered, 3 m position noise,
                 altitude in 25 ft steps) plus a decoy aircraft elsewhere
  * candidates   200 m grid, ±3 km, with the true microphone position on a node
  * noise.wav    the same background without the aircraft (must be rejected)

It then runs the real CLIs (doppler_track.py, make_grid.py and score_candidates.py)
and checks the results. The exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

from doppler_common import FT, LocalFrame, straight_pass_emission_time

HERE = os.path.dirname(os.path.abspath(__file__))

LAT0, LON0, GROUND = 45.0, 7.0, 300.0
REC_START = 1725827820.0       # 2024-09-08 20:37:00 UTC
FS, DUR = 8000, 60.0
F0, SPEED, AGL, LATERAL, HEADING = 110.0, 80.0, 900.0, 520.0, 60.0
T_CA = REC_START + 30.0         # emission time of closest approach
MIC = 1.5
CLOCK_ERROR = 1.3               # the "known" start time we pass is off by this much


def pink(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def run(cmd):
    print("$", " ".join(os.path.relpath(c) if os.path.exists(c) else c for c in cmd))
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, cwd=HERE)
    sys.stdout.write("  " + p.stdout.replace("\n", "\n  ").rstrip() + "\n")
    if p.returncode:
        sys.stderr.write(p.stderr)
        raise SystemExit(f"command failed: {cmd[0]}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.join(HERE, "synthetic_out"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    o = lambda name: os.path.join(a.outdir, name)  # noqa: E731
    rng = np.random.default_rng(a.seed)
    frame = LocalFrame(LAT0, LON0, GROUND)

    # ---- geometry (ENU metres around the true microphone)
    mic = np.array([0.0, 0.0, MIC])
    h = np.radians(HEADING)
    along = np.array([np.sin(h), np.cos(h), 0.0])
    side = np.array([np.cos(h), -np.sin(h), 0.0])          # horizontal, perpendicular to track
    cpa = mic + LATERAL * side + np.array([0, 0, AGL - MIC])
    slant = float(np.linalg.norm(cpa - mic))
    mirror = mic + 2 * LATERAL * side                        # same distance to the whole flight line

    # ---- audio
    n = int(FS * DUR)
    t_r = REC_START + np.arange(n) / FS
    t_e = straight_pass_emission_time(t_r, SPEED, slant, T_CA)
    r = np.sqrt(slant ** 2 + (SPEED * (t_e - T_CA)) ** 2)
    ph = 2 * np.pi * F0 * (t_e - T_CA)
    jitter = 1 + 0.15 * np.sin(2 * np.pi * 0.37 * (t_e - T_CA))  # engine amplitude wobble
    tone = (np.cos(ph) + 0.5 * np.cos(2 * ph + 1.0) + 0.25 * np.cos(3 * ph + 2.0)) * jitter * slant / r
    background = 4.0 * pink(n, rng) + 0.3 * np.cos(2 * np.pi * 120.0 * np.arange(n) / FS)
    for name, x in (("flyover.wav", tone + background), ("noise.wav", background)):
        sf.write(o(name), (0.9 * x / np.abs(x).max()).astype(np.float32), FS, subtype="FLOAT")

    # ---- ADS-B track (+ decoy)
    rows = []
    tt = np.arange(T_CA - 120, T_CA + 120, 1.0) + rng.uniform(-0.15, 0.15, 240)
    p = cpa[None, :] + SPEED * (tt - T_CA)[:, None] * along[None, :]
    p[:, :2] += rng.normal(0, 3.0, (len(tt), 2))
    la, lo, hh = frame.to_geodetic(p)
    for ti, a_, b_, c_ in zip(tt, la, lo, hh):
        rows.append([f"{ti:.2f}", "a1b2c3", "TEST123", f"{a_:.6f}", f"{b_:.6f}", int(round(c_ / FT / 25) * 25)])
    h2 = np.radians(150.0)
    along2 = np.array([np.sin(h2), np.cos(h2), 0.0])
    cpa2 = np.array([-4500.0, 3500.0, 3000.0])
    p2 = cpa2[None, :] + 220.0 * (tt - T_CA - 5)[:, None] * along2[None, :]
    la, lo, hh = frame.to_geodetic(p2)
    for ti, a_, b_, c_ in zip(tt, la, lo, hh):
        rows.append([f"{ti:.2f}", "d4e5f6", "DECOY9", f"{a_:.6f}", f"{b_:.6f}", int(round(c_ / FT / 25) * 25)])
    rows.sort(key=lambda r_: float(r_[0]))
    with open(o("track.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "icao24", "callsign", "lat", "lon", "alt_geom"])
        w.writerows(rows)

    # ---- candidate grid: true mic sits on a node, not at the centre
    gc = frame.to_geodetic(np.array([400.0, -600.0, 0.0]))
    truth_ll = frame.to_geodetic(np.array([0.0, 0.0, 0.0]))
    mirror_ll = frame.to_geodetic(mirror * [1, 1, 0])
    truth_arg = f"{float(truth_ll[0]):.7f},{float(truth_ll[1]):.7f}"
    run(["make_grid.py", "--center", f"{float(gc[0]):.7f},{float(gc[1]):.7f}", "--radius", "3000",
         "--step", "200", "--elev", f"{GROUND}", "--out", o("candidates.csv")])

    # ---- pipeline
    run(["doppler_track.py", o("flyover.wav"), "--fmin", "60", "--fmax", "250", "--out", o("ev1")])
    run(["doppler_track.py", o("noise.wav"), "--fmin", "60", "--fmax", "250", "--out", o("noise")])
    run(["score_candidates.py", o("track.csv"), o("candidates.csv"), o("ev1.csv"),
         "--rec-start", f"{REC_START + CLOCK_ERROR}", "--dt-tol", "2",
         "--out", o("scores_known_time.csv"), "--plot", o("scores_known_time.png"), "--truth", truth_arg, "--top", "5"])
    run(["score_candidates.py", o("track.csv"), o("candidates.csv"), o("ev1.csv"),
         "--out", o("scores_unknown_time.csv"), "--plot", o("scores_unknown_time.png"), "--truth", truth_arg,
         "--top", "5"])

    # ---- checks
    ev = json.load(open(o("ev1.json")))
    nz = json.load(open(o("noise.json")))
    fit = ev["fit"]

    def ranked(path):
        with open(path) as fh:
            return list(csv.DictReader(fh))

    def enu_of(row):
        return frame.to_enu(float(row["lat"]), float(row["lon"]), GROUND)

    known, unknown = ranked(o("scores_known_time.csv")), ranked(o("scores_unknown_time.csv"))
    b = enu_of(known[0])
    d_truth = float(np.hypot(*b[:2]))
    d_mirror = float(np.hypot(*(b - mirror)[:2]))
    truth_rank = min(int(r_["rank"]) for r_ in known if np.hypot(*enu_of(r_)[:2]) < 1.0)
    bu = enu_of(unknown[0])
    lat_off = abs(float((bu - cpa) @ side))  # distance from the ground track
    along_err = float((bu - mic) @ along)

    checks = [
        ("flyover detected", ev["detection"]["detected"], f"ridge_snr_z={ev['detection']['ridge_snr_z']:.2f}"),
        ("noise-only clip rejected", not nz["detection"]["detected"],
         f"ridge_snr_z={nz['detection']['ridge_snr_z']:.2f}; {nz['detection']['rejected_because']}"),
        ("f0 within 0.5 Hz", abs(fit["f0_hz"] - F0) < 0.5, f"{fit['f0_hz']:.2f} vs {F0}"),
        ("speed within 5 %", abs(fit["speed_mps"] / SPEED - 1) < 0.05, f"{fit['speed_mps']:.1f} vs {SPEED}"),
        ("slant range within 5 %", abs(fit["slant_range_m"] / slant - 1) < 0.05,
         f"{fit['slant_range_m']:.0f} vs {slant:.0f} m"),
        ("closest approach time within 0.5 s", abs(fit["t_ca_recv_s"] - (T_CA - REC_START + slant / 343)) < 0.5,
         f"{fit['t_ca_recv_s']:.2f} vs {T_CA - REC_START + slant / 343:.2f} s"),
        ("known time: rank 1 within 1 cell of truth or its mirror", min(d_truth, d_mirror) <= 200.0,
         f"{d_truth:.0f} m from truth, {d_mirror:.0f} m from mirror; truth is rank {truth_rank}"),
        ("known time: recovered clock error", abs(float(known[0]["dt_offset_s"]) + CLOCK_ERROR) < 0.3,
         f"{float(known[0]['dt_offset_s']):+.2f} s (true {-CLOCK_ERROR:+.2f})"),
        ("unknown time: right distance from the flight line", abs(lat_off - LATERAL) <= 200.0,
         f"{lat_off:.0f} m from ground track (true {LATERAL:.0f}); {along_err:+.0f} m along track"),
    ]
    width = max(len(c[0]) for c in checks)
    print("\nchecks")
    for name, ok, info in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {info}")
    summary = {
        "truth": {"lat": float(truth_ll[0]), "lon": float(truth_ll[1]), "mirror_lat": float(mirror_ll[0]),
                  "mirror_lon": float(mirror_ll[1]), "f0_hz": F0, "speed_mps": SPEED, "slant_range_m": slant},
        "fit": fit, "checks": [{"name": c[0], "pass": bool(c[1]), "info": c[2]} for c in checks],
    }
    json.dump(summary, open(o("summary.json"), "w"), indent=2)
    print(f"\noutputs in {a.outdir}")
    return 0 if all(c[1] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())

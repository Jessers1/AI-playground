#!/usr/bin/env python3
"""Rank candidate observer positions by how well an ADS-B track explains a traced Doppler curve.

    python score_candidates.py track.csv candidates.csv ev1.csv \
        --rec-start 1725827820 --dt-tol 2 --plot scores.png

For every candidate point and every allowed recording start time, the exact
curve that point would have heard is predicted from the aircraft's ADS-B track:

  * the track is smoothed (smoothing spline, GCV) and differentiated for velocity
  * for each emission time t_e:  receive time  t_r = t_e + |p(t_e) - x| / c
                                 Doppler factor g = 1 / (1 - v(t_e) . u / c)
    (u = unit vector from aircraft to observer; uses velocity at emission time)
  * g is interpolated at the observed receive times, so there is no iteration
  * f0 is fitted in closed form per hypothesis (or fixed with --f0)
  * misfit = RMS(observed - f0 * g) in Hz; rank 1 is the lowest misfit

Inputs
  track.csv       ADS-B samples: time (unix s, ms or ISO-8601), lat, lon and an
                  altitude column. Optional icao24/callsign column to score several
                  aircraft (each candidate keeps its best aircraft).
  candidates.csv  lat, lon, elev_m (ground elevation, m above the WGS84 ellipsoid
                  or MSL, consistent with the track's altitude)
  ev1.csv         output of doppler_track.py (rows with used=1 are scored)

Without --rec-start the time offset is searched over the whole track. One straight
pass then only fixes your distance from the flight line, not where along it you are.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from doppler_common import FT, C_SOUND, LocalFrame, fmt_utc, parse_time, pick_col, read_csv_dicts

ALT_M = ["alt_m", "geoaltitude", "geo_altitude", "baroaltitude", "baro_altitude", "altitude_m"]
ALT_FT = ["alt_ft", "alt_geom", "alt_geom_ft", "alt_baro", "altitude", "altitude_ft"]


# --------------------------------------------------------------------------- inputs

def load_tracks(path, alt_col=None, alt_units=None, group_col=None):
    rows = read_csv_dicts(path)
    cols = rows[0].keys()
    tcol = pick_col(cols, ["time", "timestamp", "t", "ts", "unix_time"], label="time")
    lacol = pick_col(cols, ["lat", "latitude"], label="latitude")
    locol = pick_col(cols, ["lon", "lng", "longitude"], label="longitude")
    if alt_col:
        acol = pick_col(cols, [alt_col.lower()], label="altitude")
        units = alt_units or ("ft" if acol.lower() in ALT_FT else "m")
    else:
        acol = pick_col(cols, ALT_M, required=False)
        units = "m"
        if acol is None:
            acol = pick_col(cols, ALT_FT, label="altitude")
            units = "ft"
        units = alt_units or units
    gcol = pick_col(cols, [group_col.lower()] if group_col else ["icao24", "icao", "hex", "callsign"],
                    required=False)
    print(f"track: time={tcol} lat={lacol} lon={locol} alt={acol} [{units}]"
          + (f" group={gcol}" if gcol else "") + "  (override with --alt-col/--alt-units)")
    if acol.lower() in ("baroaltitude", "baro_altitude", "alt_baro", "altitude"):
        print("warning: barometric altitude can be off by tens of metres; prefer geometric altitude")

    groups = {}
    for r in rows:
        try:
            t = parse_time(r[tcol])
            lat, lon, alt = float(r[lacol]), float(r[locol]), float(r[acol])
        except (ValueError, TypeError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(alt)):
            continue
        key = r[gcol].strip() if gcol else "aircraft"
        groups.setdefault(key, []).append((t, lat, lon, alt * (FT if units == "ft" else 1.0)))
    out = {}
    for k, v in groups.items():
        a = np.array(sorted(v))
        _, keep = np.unique(a[:, 0], return_index=True)
        a = a[keep]
        if len(a) >= 6:
            out[k] = a
    if not out:
        raise ValueError("no usable track rows (need >= 6 samples per aircraft)")
    hi = max(v[:, 3].max() for v in out.values())
    if hi > 16000:
        print(f"warning: max altitude {hi:.0f} m looks like feet; pass --alt-units ft")
    return out


def load_candidates(path):
    rows = read_csv_dicts(path)
    cols = rows[0].keys()
    la = pick_col(cols, ["lat", "latitude"], label="latitude")
    lo = pick_col(cols, ["lon", "lng", "longitude"], label="longitude")
    el = pick_col(cols, ["elev_m", "elevation", "elev", "ground_m", "alt_m", "h"], required=False)
    if el is None:
        print("warning: no elev_m column in candidates; assuming ground elevation 0 m")
    data = np.array([[float(r[la]), float(r[lo]), float(r[el]) if el else 0.0] for r in rows])
    return data


def load_event(path, max_points):
    rows = read_csv_dicts(path)
    t = np.array([float(r["t_s"]) for r in rows])
    f = np.array([float(r["f_hz"]) for r in rows])
    used = np.array([r.get("used", "1") in ("1", "True", "true") for r in rows])
    t, f = t[used], f[used]
    if len(t) < 8:
        raise ValueError(f"{path}: fewer than 8 used ridge points (was the flyover detected?)")
    if len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points).round().astype(int)
        t, f = t[idx], f[idx]
    meta = {}
    jpath = os.path.splitext(path)[0] + ".json"
    if os.path.exists(jpath):
        with open(jpath) as fh:
            meta = json.load(fh)
    return t, f, meta


# --------------------------------------------------------------------------- track model

class Track:
    """Smoothed ENU track resampled on a uniform emission-time grid."""

    def __init__(self, samples, frame, dt=0.2, max_gap=15.0):
        t = samples[:, 0]
        enu = frame.to_enu(samples[:, 1], samples[:, 2], samples[:, 3])
        self.t_raw = t
        self.t = np.arange(t[0], t[-1], dt)
        splines = [make_smoothing_spline(t, enu[:, k]) for k in range(3)]
        self.p = np.stack([s(self.t) for s in splines], axis=1)
        self.v = np.stack([s.derivative()(self.t) for s in splines], axis=1)
        # mask emission times far from any real sample (gaps in coverage)
        j = np.clip(np.searchsorted(t, self.t), 1, len(t) - 1)
        nearest = np.minimum(np.abs(self.t - t[j - 1]), np.abs(t[j] - self.t))
        gap = (t[j] - t[j - 1]) > max_gap
        self.valid = ~(gap & (nearest > max_gap / 2))
        self.speed = np.linalg.norm(self.v, axis=1)

    def doppler_table(self, x, c):
        """Receive time and Doppler factor at every grid emission time for observer x."""
        d = x[None, :] - self.p
        rng = np.linalg.norm(d, axis=1)
        vt = np.einsum("ij,ij->i", self.v, d) / rng  # velocity toward the observer
        g = 1.0 / (1.0 - vt / c)
        g[~self.valid] = np.nan
        return self.t + rng / c, g


# --------------------------------------------------------------------------- scoring

def misfit_for_offsets(t_r_grid, g_grid, t_obs, f_obs, offsets, f0_fixed, min_cov):
    tq = offsets[:, None] + t_obs[None, :]
    g = np.interp(tq.ravel(), t_r_grid, g_grid, left=np.nan, right=np.nan).reshape(tq.shape)
    ok = np.isfinite(g)
    cov = ok.mean(axis=1)
    g0 = np.where(ok, g, 0.0)
    fo = np.where(ok, f_obs[None, :], 0.0)
    n = np.maximum(ok.sum(axis=1), 1)
    if f0_fixed is None:
        f0 = (fo * g0).sum(axis=1) / np.maximum((g0 * g0).sum(axis=1), 1e-12)
    else:
        f0 = np.full(len(offsets), float(f0_fixed))
    res = np.where(ok, fo - f0[:, None] * g0, 0.0)
    rms = np.sqrt((res * res).sum(axis=1) / n)
    rms[cov < min_cov] = np.inf
    return rms, f0, cov


def score_one(track, x, t_obs, f_obs, offsets, f0_fixed, c, min_cov, refine_step, coarse_idx):
    t_r_grid, g_grid = track.doppler_table(x, c)
    if coarse_idx is not None:  # cheap first pass on a subset of points
        rms_c, _, _ = misfit_for_offsets(t_r_grid, g_grid, t_obs[coarse_idx], f_obs[coarse_idx],
                                         offsets, f0_fixed, min_cov)
        order = np.argsort(rms_c)[:3]
        seeds = offsets[order[np.isfinite(rms_c[order])]]
        if len(seeds) == 0:
            return np.inf, np.nan, np.nan, 0.0
        fine = np.unique(np.concatenate([s + np.arange(-2, 2.01, 0.25) * refine_step for s in seeds]))
    else:
        fine = offsets
    rms, f0, cov = misfit_for_offsets(t_r_grid, g_grid, t_obs, f_obs, fine, f0_fixed, min_cov)
    i = int(np.argmin(rms))
    if not np.isfinite(rms[i]):
        return np.inf, np.nan, np.nan, 0.0
    lo, hi = fine[max(i - 1, 0)], fine[min(i + 1, len(fine) - 1)]
    best = (rms[i], fine[i], f0[i], cov[i])
    if hi > lo:
        def obj(o):
            return misfit_for_offsets(t_r_grid, g_grid, t_obs, f_obs, np.array([o]), f0_fixed, min_cov)[0][0]

        r = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
        if r.fun < best[0]:
            rr, ff, cc = misfit_for_offsets(t_r_grid, g_grid, t_obs, f_obs, np.array([r.x]), f0_fixed, min_cov)
            best = (rr[0], r.x, ff[0], cc[0])
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track")
    ap.add_argument("candidates")
    ap.add_argument("event", help="ridge CSV written by doppler_track.py")
    ap.add_argument("--f0", type=float, default=None, help="fix the base tone (Hz) instead of fitting per point")
    ap.add_argument("--rec-start", type=float, default=None,
                    help="unix time of the recording's first sample (else searched over the whole track)")
    ap.add_argument("--dt-tol", type=float, default=2.0, help="± uncertainty on --rec-start, s")
    ap.add_argument("--dt-step", type=float, default=0.25, help="offset search step with --rec-start, s")
    ap.add_argument("--coarse-step", type=float, default=1.0, help="offset search step without --rec-start, s")
    ap.add_argument("--obs-height", type=float, default=1.5, help="microphone height above ground, m")
    ap.add_argument("--c", type=float, default=C_SOUND, help="speed of sound, m/s")
    ap.add_argument("--alt-col", default=None, help="altitude column name in track.csv")
    ap.add_argument("--alt-units", choices=["m", "ft"], default=None)
    ap.add_argument("--group-col", default=None, help="aircraft id column (default icao24/hex/callsign)")
    ap.add_argument("--max-gap", type=float, default=15.0, help="treat track gaps longer than this as missing, s")
    ap.add_argument("--min-coverage", type=float, default=0.8, help="fraction of ridge points the track must cover")
    ap.add_argument("--max-points", type=int, default=400, help="decimate the ridge to at most this many points")
    ap.add_argument("--out", default=None, help="ranked CSV (default <event>_scores.csv)")
    ap.add_argument("--plot", default=None, help="write a misfit heatmap PNG")
    ap.add_argument("--truth", default=None, help="lat,lon to mark on the plot (for tests)")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)

    t_obs, f_obs, meta = load_event(a.event, a.max_points)
    cands = load_candidates(a.candidates)
    frame = LocalFrame(cands[:, 0].mean(), cands[:, 1].mean(), float(np.median(cands[:, 2])))
    xs = frame.to_enu(cands[:, 0], cands[:, 1], cands[:, 2] + a.obs_height)
    tracks = {k: Track(v, frame, max_gap=a.max_gap) for k, v in load_tracks(
        a.track, a.alt_col, a.alt_units, a.group_col).items()}
    f0_fixed = a.f0
    if f0_fixed is None and meta.get("fit", {}) and meta["fit"].get("f0_fixed"):
        f0_fixed = meta["fit"]["f0_hz"]

    rng_ = np.arange(-1.0, 1.0 + 1e-9, a.dt_step / max(a.dt_tol, 1e-9)) * a.dt_tol
    coarse_idx = None
    if a.rec_start is None:
        mode = "unknown time (searched over the track)"
        coarse_idx = np.linspace(0, len(t_obs) - 1, min(60, len(t_obs))).round().astype(int)
    else:
        mode = f"rec_start {fmt_utc(a.rec_start)} ± {a.dt_tol:g} s"

    print(f"{len(cands)} candidates x {len(tracks)} aircraft, {len(t_obs)} ridge points, {mode}")
    t_start = time.time()
    best = np.full(len(cands), np.inf)
    rec = [None] * len(cands)
    for name, tr in tracks.items():
        if a.rec_start is None:
            # every start time for which the ridge overlaps what the track could have produced
            # (sound arrives up to ~60 s late from 20 km); coverage is checked per hypothesis
            lo = tr.t[0] - t_obs.max()
            hi = tr.t[-1] + 60.0 - t_obs.min()
            offsets = np.arange(lo, hi + a.coarse_step, a.coarse_step)
            refine = a.coarse_step
        else:
            offsets = a.rec_start + rng_
            refine = a.dt_step
        for i, x in enumerate(xs):
            r = score_one(tr, x, t_obs, f_obs, offsets, f0_fixed, a.c, a.min_coverage, refine, coarse_idx)
            if r[0] < best[i]:
                best[i] = r[0]
                rec[i] = (name,) + tuple(r[1:])
    print(f"scored in {time.time() - t_start:.1f} s")

    order = np.argsort(best)
    out = a.out or os.path.splitext(a.event)[0] + "_scores.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "lat", "lon", "elev_m", "misfit_hz", "f0_hz", "rec_start_unix",
                    "rec_start_utc", "dt_offset_s", "aircraft", "coverage"])
        for k, i in enumerate(order, 1):
            if rec[i] is None or not np.isfinite(best[i]):
                w.writerow([k, f"{cands[i, 0]:.6f}", f"{cands[i, 1]:.6f}", f"{cands[i, 2]:.1f}",
                            "inf", "", "", "", "", "", 0])
                continue
            name, off, f0, cov = rec[i]
            dto = "" if a.rec_start is None else f"{off - a.rec_start:+.3f}"
            w.writerow([k, f"{cands[i, 0]:.6f}", f"{cands[i, 1]:.6f}", f"{cands[i, 2]:.1f}",
                        f"{best[i]:.4f}", f"{f0:.3f}", f"{off:.3f}", fmt_utc(off), dto, name, f"{cov:.2f}"])

    print(f"{'rank':>4} {'lat':>11} {'lon':>11} {'misfit Hz':>10} {'f0 Hz':>8} {'rec start (UTC)':>24} aircraft")
    for k, i in enumerate(order[:a.top], 1):
        if rec[i] is None or not np.isfinite(best[i]):
            break
        name, off, f0, _ = rec[i]
        print(f"{k:>4} {cands[i, 0]:>11.6f} {cands[i, 1]:>11.6f} {best[i]:>10.4f} {f0:>8.3f} "
              f"{fmt_utc(off):>24} {name}")
    print(f"wrote {out}")

    if a.plot:
        truth = tuple(float(v) for v in a.truth.split(",")) if a.truth else None
        plot(a.plot, cands, best, order, tracks, frame, mode, truth)
        print(f"wrote {a.plot}")
    return 0


def plot(path, cands, best, order, tracks, frame, mode, truth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ok = np.isfinite(best)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if ok.any():
        vmin = max(np.nanmin(best[ok]), 1e-3)
        vmax = max(np.nanpercentile(best[ok], 98), vmin * 10)
        lat, lon = cands[:, 0], cands[:, 1]
        # marker side ~ typical spacing between neighbouring candidates, in points
        xy = np.stack([lon * math.cos(math.radians(lat.mean())), lat], axis=1)
        spacing = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]) if len(xy) > 1 else 1.0
        extent = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), spacing) * 1.1
        size = max((0.95 * 400.0 * spacing / extent) ** 2, 9.0)
        sc = ax.scatter(lon[ok], lat[ok], c=best[ok], s=size, marker="s", cmap="viridis_r",
                        norm=LogNorm(vmin=vmin, vmax=vmax), linewidths=0)
        fig.colorbar(sc, ax=ax, label="RMS misfit (Hz), lower = better")
        ax.scatter(lon[~ok], lat[~ok], s=4, c="0.85", marker="s", linewidths=0)
    x0, x1 = cands[:, 1].min(), cands[:, 1].max()
    y0, y1 = cands[:, 0].min(), cands[:, 0].max()
    colors = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]
    for k, (name, tr) in enumerate(tracks.items()):
        la, lo, _ = frame.to_geodetic(tr.p)
        col = colors[k % len(colors)]
        ax.plot(lo, la, "-", color=col, lw=1.2, alpha=0.85, label=f"ADS-B {name}")
        mid = len(lo) // 2  # direction-of-flight arrow
        ax.annotate("", xy=(lo[mid + 1], la[mid + 1]), xytext=(lo[mid], la[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
    b = order[0]
    if np.isfinite(best[b]):
        ax.plot(cands[b, 1], cands[b, 0], "*", ms=16, mfc="white", mec="black", label="rank 1")
    if truth:
        ax.plot(truth[1], truth[0], "o", ms=12, mfc="none", mec="#ff7f0e", mew=2, label="truth")
    pad_x, pad_y = 0.05 * (x1 - x0 + 1e-6), 0.05 * (y1 - y0 + 1e-6)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_aspect(1.0 / math.cos(math.radians(cands[:, 0].mean())))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Candidate misfit: {mode}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

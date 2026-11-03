#!/usr/bin/env python3
"""Trace an aircraft flyover tone in a recording and fit a straight-pass Doppler model.

    python doppler_track.py clip.mp4 --fmin 60 --fmax 250 --out ev1

Steps
  1. decode audio with ffmpeg (no shell), optionally only --start..--end
  2. narrow-band spectrogram (default 1 s Hann window, 0.1 s hop, 4x zero padding)
  3. remove stationary lines (per-bin median over time) and z-score each frame
  4. Viterbi ridge: highest-z path whose frame-to-frame change is <= --max-slope Hz/s
  5. keep the longest run where the (median-smoothed) ridge stays above --trim-z
  6. robust least-squares fit of f0, speed, closest-approach slant range and time
  7. detection guard: ridge_snr_z, fit RMS, duration and physical plausibility

Writes <out>.csv (ridge points), <out>.json (fit + detection) and <out>.png.
The band (--fmin/--fmax) and time window (--start/--end) are chosen by eye from a
spectrogram. Look for a tone that slides downward over tens of seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

from doppler_common import (
    C_SOUND,
    decode_audio,
    fmt_utc,
    initial_guess_from_ridge,
    straight_pass_curve,
)


def spectrogram(x, sr, win_s, hop_s, zpad, fmin, fmax):
    nper = int(round(win_s * sr))
    hop = max(1, int(round(hop_s * sr)))
    if len(x) < nper:
        raise ValueError(f"audio is {len(x) / sr:.1f} s, shorter than one {win_s} s window")
    nfft = 1 << int(math.ceil(math.log2(nper * zpad)))
    nfr = 1 + (len(x) - nper) // hop
    idx = np.arange(nper)[None, :] + hop * np.arange(nfr)[:, None]
    w = np.hanning(nper)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    spec = np.empty((nfr, band.sum()))
    step = 256  # frames per chunk, bounds memory
    for i in range(0, nfr, step):
        fr = x[idx[i:i + step]] * w
        spec[i:i + step] = np.abs(np.fft.rfft(fr, n=nfft, axis=1)[:, band]) ** 2
    times = (np.arange(nfr) * hop + nper / 2.0) / sr
    return times, freqs[band], 10.0 * np.log10(spec + 1e-20)


def frame_zscore(p_db, remove_lines=True):
    p = p_db.copy()
    if remove_lines:
        p -= np.median(p, axis=0, keepdims=True)
    med = np.median(p, axis=1, keepdims=True)
    mad = np.median(np.abs(p - med), axis=1, keepdims=True) * 1.4826 + 1e-9
    return (p - med) / mad


def viterbi_ridge(z, df, hop_s, max_slope, jump_penalty):
    """Max-sum path through z with |delta f| <= max_slope*hop per frame."""
    nt, nf = z.shape
    k = max(1, int(math.ceil(max_slope * hop_s / df)))
    shifts = np.arange(-k, k + 1)
    pen = jump_penalty * (shifts * df / (max_slope * hop_s)) ** 2
    emis = np.clip(z, -3.0, 8.0)
    score = emis[0].copy()
    back = np.zeros((nt, nf), dtype=np.int16)
    for i in range(1, nt):
        best = np.full(nf, -np.inf)
        arg = np.zeros(nf, dtype=np.int16)
        for s, pz in zip(shifts, pen):
            cand = np.full(nf, -np.inf)
            if s >= 0:
                cand[s:] = score[:nf - s] if s else score
            else:
                cand[:s] = score[-s:]
            cand -= pz
            better = cand > best
            best[better] = cand[better]
            arg[better] = s
        score = best + emis[i]
        back[i] = arg
    path = np.empty(nt, dtype=int)
    path[-1] = int(np.argmax(score))
    for i in range(nt - 1, 0, -1):
        path[i - 1] = path[i] - back[i, path[i]]
    return path


def refine_peak(p_db, path, freqs):
    """Parabolic sub-bin interpolation of each ridge point."""
    df = freqs[1] - freqs[0]
    f = freqs[path].astype(float)
    rows = np.arange(len(path))
    ok = (path > 0) & (path < len(freqs) - 1)
    a = p_db[rows[ok], path[ok] - 1]
    b = p_db[rows[ok], path[ok]]
    c = p_db[rows[ok], path[ok] + 1]
    den = a - 2 * b + c
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    f[ok] += np.clip(delta, -0.5, 0.5) * df
    return f


def longest_run(mask):
    best, cur, best_end = 0, 0, -1
    for i, m in enumerate(mask):
        cur = cur + 1 if m else 0
        if cur > best:
            best, best_end = cur, i
    return (best_end - best + 1, best_end + 1) if best else (0, 0)


def fit_curve(t, f, c, f0_fixed=None, f_scale=0.5, starts=None):
    g = initial_guess_from_ridge(t, f, c)
    if f0_fixed is not None:
        g = (f0_fixed,) + g[1:]
    span = t[-1] - t[0]
    lo = np.array([f.min() * 0.5, 5.0, 5.0, t[0] - 2 * span - 60])
    hi = np.array([f.max() * 1.5, 0.95 * c, 5.0e4, t[-1] + 2 * span + 60])

    def model(p):
        f0 = f0_fixed if f0_fixed is not None else p[0]
        return straight_pass_curve(t, f0, p[1], p[2], p[3], c)

    best = None
    for dmul in starts or (0.5, 1.0, 2.0):
        for tshift in (-0.15 * span, 0.0, 0.15 * span):
            p0 = np.array([g[0], g[1], g[2] * dmul, g[3] + tshift])
            p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
            free = slice(1, 4) if f0_fixed is not None else slice(0, 4)

            def resid(q, p0=p0, free=free):
                p = p0.copy()
                p[free] = q
                return model(p) - f

            r = least_squares(resid, p0[free], bounds=(lo[free], hi[free]),
                              loss="soft_l1", f_scale=f_scale, x_scale="jac", max_nfev=2000)
            if best is None or r.cost < best[0].cost:
                p = p0.copy()
                p[free] = r.x
                best = (r, p, free)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="audio or video file (anything ffmpeg can read)")
    ap.add_argument("--fmin", type=float, required=True, help="band low edge, Hz")
    ap.add_argument("--fmax", type=float, required=True, help="band high edge, Hz")
    ap.add_argument("--out", required=True, help="output prefix: writes <out>.csv/.json/.png")
    ap.add_argument("--start", type=float, default=0.0, help="analyse from this many seconds into the file")
    ap.add_argument("--end", type=float, default=None, help="analyse up to this many seconds into the file")
    ap.add_argument("--sr", type=int, default=None, help="decode sample rate (default: max(2000, 4*fmax))")
    ap.add_argument("--win", type=float, default=1.0, help="STFT window, s (frequency resolution ~ 1/win)")
    ap.add_argument("--hop", type=float, default=0.1, help="STFT hop, s")
    ap.add_argument("--zpad", type=int, default=4, help="FFT zero-padding factor")
    ap.add_argument("--max-slope", type=float, default=15.0,
                    help="max ridge slope, Hz/s (tracker limit; steeper fitted curves are rejected)")
    ap.add_argument("--jump-penalty", type=float, default=2.0, help="Viterbi penalty (z units) at max slope")
    ap.add_argument("--trim-z", type=float, default=1.5, help="keep longest run with smoothed ridge z above this")
    ap.add_argument("--trim-smooth", type=float, default=2.0, help="median-smoothing window for trimming, s")
    ap.add_argument("--min-snr", type=float, default=1.8,
                    help="reject if median ridge z (ridge_snr_z) is below; pure noise gives ~1.1-1.3")
    ap.add_argument("--max-rms", type=float, default=1.0, help="reject if fit RMS residual (Hz) is above")
    ap.add_argument("--min-dur", type=float, default=8.0, help="reject if the traced segment is shorter, s")
    ap.add_argument("--f0", type=float, default=None, help="fix the base (rest) tone instead of fitting it")
    ap.add_argument("--c", type=float, default=C_SOUND, help="speed of sound, m/s")
    ap.add_argument("--rec-start", type=float, default=None, help="unix time of the file's first sample")
    ap.add_argument("--keep-lines", action="store_true", help="do not remove stationary spectral lines")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    sr = a.sr or int(max(2000, 4 * a.fmax))
    x = decode_audio(a.input, sr, a.start or None, a.end)
    if x.size == 0:
        sys.exit("no audio decoded")
    times, freqs, p_db = spectrogram(x, sr, a.win, a.hop, a.zpad, a.fmin, a.fmax)
    times = times + a.start
    df = freqs[1] - freqs[0]
    z = frame_zscore(p_db, remove_lines=not a.keep_lines)

    path = viterbi_ridge(z, df, a.hop, a.max_slope, a.jump_penalty)
    f_ridge = refine_peak(p_db, path, freqs)
    z_ridge = z[np.arange(len(path)), path]
    nsm = max(3, int(round(a.trim_smooth / a.hop)) | 1)
    z_smooth = median_filter(z_ridge, size=nsm, mode="nearest")
    i0, i1 = longest_run(z_smooth >= a.trim_z)

    reasons, warnings = [], []
    used = np.zeros(len(times), dtype=bool)
    fit_hz = np.full(len(times), np.nan)
    result = {
        "input": a.input, "band_hz": [a.fmin, a.fmax], "window_s": [float(times[0]), float(times[-1])],
        "settings": {k: v for k, v in vars(a).items() if k not in ("input", "out")} | {"sr": sr, "bin_hz": df},
    }
    if i1 - i0 < 2:
        i0, i1 = 0, len(times)
        reasons.append(f"ridge never stays above trim_z={a.trim_z}")
    seg = slice(i0, i1)
    t_seg, f_seg = times[seg], f_ridge[seg]
    seg_dur = float(t_seg[-1] - t_seg[0])
    snr_z = float(np.median(z_ridge[seg]))
    if seg_dur < a.min_dur:
        reasons.append(f"traced segment {seg_dur:.1f} s < min_dur {a.min_dur} s")

    fit = None
    if len(t_seg) >= 8:
        r, p, free = fit_curve(t_seg, f_seg, a.c, a.f0, f_scale=max(df, 0.25))
        res = straight_pass_curve(t_seg, p[0], p[1], p[2], p[3], a.c) - f_seg
        sig = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        inl = np.abs(res) <= max(3.0 * sig, df)
        if inl.sum() >= 8:  # refit on inliers with plain least squares for clean errors
            t_in, f_in = t_seg[inl], f_seg[inl]
            p0 = p.copy()

            def resid(q):
                pp = p0.copy()
                pp[free] = q
                return straight_pass_curve(t_in, pp[0], pp[1], pp[2], pp[3], a.c) - f_in

            r2 = least_squares(resid, p0[free], x_scale="jac", max_nfev=2000)
            p[free] = r2.x
            res_in = r2.fun
            dof = max(1, len(res_in) - len(r2.x))
            s2 = float(res_in @ res_in) / dof
            try:
                cov = np.linalg.inv(r2.jac.T @ r2.jac) * s2 * (a.win / a.hop)  # frames overlap
                err = np.sqrt(np.clip(np.diag(cov), 0, None))
            except np.linalg.LinAlgError:
                err = np.full(len(r2.x), np.nan)
            stderr = np.full(4, np.nan)
            stderr[free] = err
            rms = float(np.sqrt(np.mean(res_in ** 2)))
            used[np.arange(i0, i1)[inl]] = True
            fit_hz[seg] = straight_pass_curve(t_seg, p[0], p[1], p[2], p[3], a.c)
            f0, v, d, tca = (float(q) for q in p)
            tdense = np.linspace(t_seg[0], t_seg[-1], 2000)
            fd = straight_pass_curve(tdense, f0, v, d, tca, a.c)
            max_fit_slope = float(np.max(np.abs(np.diff(fd) / np.diff(tdense))))
            fit = {
                "f0_hz": f0, "speed_mps": v, "slant_range_m": d,
                "t_ca_emit_s": tca, "t_ca_recv_s": tca + d / a.c,
                "stderr": {"f0_hz": float(stderr[0]), "speed_mps": float(stderr[1]),
                           "slant_range_m": float(stderr[2]), "t_ca_emit_s": float(stderr[3])},
                "f0_fixed": a.f0 is not None,
                "rms_hz": rms, "n_points": int(inl.sum()), "n_outliers": int((~inl).sum()),
                "max_slope_hz_s": max_fit_slope,
                "f_start_hz": float(fd[0]), "f_end_hz": float(fd[-1]),
            }
            if a.rec_start is not None:
                fit["t_ca_recv_utc"] = fmt_utc(a.rec_start + fit["t_ca_recv_s"])
            if rms > a.max_rms:
                reasons.append(f"fit rms {rms:.2f} Hz > max_rms {a.max_rms}")
            if max_fit_slope > a.max_slope:
                reasons.append(f"fitted slope {max_fit_slope:.1f} Hz/s > max_slope {a.max_slope}")
            if not 15.0 <= v <= 0.95 * a.c:
                reasons.append(f"implausible speed {v:.0f} m/s")
            if fd[0] - fd[-1] < 2 * df:
                reasons.append("no downward Doppler sweep")
            if not t_seg[0] <= tca + d / a.c <= t_seg[-1]:
                warnings.append("closest approach is outside the traced segment; speed and range are poorly "
                                "constrained")
            if min(f_seg.min() - a.fmin, a.fmax - f_seg.max()) < 2 * df:
                warnings.append("ridge touches the band edge; widen --fmin/--fmax")
        else:
            reasons.append("too few inlier points for a fit")
    else:
        reasons.append("too few ridge points")
    if snr_z < a.min_snr:
        reasons.append(f"ridge_snr_z {snr_z:.2f} < min_snr {a.min_snr}")

    detected = not reasons
    result["fit"] = fit
    result["detection"] = {
        "detected": detected, "ridge_snr_z": snr_z, "segment_s": [float(t_seg[0]), float(t_seg[-1])],
        "segment_duration_s": seg_dur, "rejected_because": reasons, "warnings": warnings,
    }
    if a.rec_start is not None:
        result["rec_start_unix"] = a.rec_start
        result["rec_start_utc"] = fmt_utc(a.rec_start)

    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "t_unix", "f_hz", "z", "used", "fit_hz", "resid_hz"])
        for i in range(len(times)):
            tu = "" if a.rec_start is None else f"{a.rec_start + times[i]:.3f}"
            fh_ = "" if np.isnan(fit_hz[i]) else f"{fit_hz[i]:.4f}"
            rs = "" if np.isnan(fit_hz[i]) else f"{f_ridge[i] - fit_hz[i]:.4f}"
            w.writerow([f"{times[i]:.3f}", tu, f"{f_ridge[i]:.4f}", f"{z_ridge[i]:.3f}", int(used[i]), fh_, rs])
    with open(a.out + ".json", "w") as fh:
        json.dump(result, fh, indent=2)

    if not a.no_plot:
        plot(a.out + ".png", times, freqs, z, f_ridge, used, fit_hz, result)

    if fit:
        e = fit["stderr"]
        print(f"f0 {fit['f0_hz']:.2f} Hz (±{e['f0_hz']:.2f})  speed {fit['speed_mps']:.1f} m/s "
              f"(±{e['speed_mps']:.1f})  slant range {fit['slant_range_m']:.0f} m (±{e['slant_range_m']:.0f})  "
              f"closest approach heard at t={fit['t_ca_recv_s']:.2f} s  rms {fit['rms_hz']:.3f} Hz")
    print(f"ridge_snr_z {snr_z:.2f}  segment {t_seg[0]:.1f}-{t_seg[-1]:.1f} s  "
          f"detected={detected}" + (f"  rejected_because={reasons}" if reasons else ""))
    for wmsg in warnings:
        print("warning:", wmsg)
    print(f"wrote {a.out}.csv {a.out}.json" + ("" if a.no_plot else f" {a.out}.png"))
    return 0


def plot(path, times, freqs, z, f_ridge, used, fit_hz, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    ax.imshow(np.clip(z.T, -2, 8), origin="lower", aspect="auto", cmap="magma",
              extent=[times[0], times[-1], freqs[0], freqs[-1]])
    ax.plot(times[~used], f_ridge[~used], ".", ms=2, color="#7fb3ff", alpha=0.6, label="ridge (unused)")
    ax.plot(times[used], f_ridge[used], ".", ms=3, color="#00e5ff", label="ridge (fit points)")
    if np.isfinite(fit_hz).any():
        ax.plot(times, fit_hz, "-", lw=1.6, color="#7CFC00", label="straight-pass fit")
    det = result["detection"]
    fit = result["fit"]
    title = f"{'DETECTED' if det['detected'] else 'REJECTED'}  ridge_snr_z={det['ridge_snr_z']:.2f}"
    if fit:
        title += (f"  f0={fit['f0_hz']:.1f} Hz  v={fit['speed_mps']:.0f} m/s  "
                  f"d={fit['slant_range_m']:.0f} m  rms={fit['rms_hz']:.2f} Hz")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("frequency (Hz)")
    ax.legend(loc="upper right", fontsize=8)
    resid = f_ridge - fit_hz
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.plot(times[used], resid[used], ".", ms=3, color="#1f77b4")
    ax2.plot(times[~used], resid[~used], ".", ms=2, color="0.7")
    lim = np.nanpercentile(np.abs(resid[used]), 99) * 2 if used.any() else 1
    ax2.set_ylim(-max(lim, 0.5), max(lim, 0.5))
    ax2.set_ylabel("resid (Hz)")
    ax2.set_xlabel("time in file (s)")
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

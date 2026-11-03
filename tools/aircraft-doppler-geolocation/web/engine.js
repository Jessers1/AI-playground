/* Doppler flyover engine. One file shared by the web page, its workers and the Node tests.
 *
 * Pipeline
 *   audio -> narrow-band spectrogram -> whitened z-scores
 *         -> automatic flyover search (Viterbi ridges in overlapping log-spaced bands)
 *         -> straight-pass Doppler fit per ridge (robust Levenberg-Marquardt)
 *         -> matched-curve detection score, calibrated against the same search run
 *            on the time-reversed spectrogram (a real flyover only ever falls in pitch)
 *         -> ridges that share speed / range / closest-approach time grouped as one pass
 *   ADS-B -> smoothed ENU tracks (Whittaker for dense data, Hermite with reported
 *            velocity for sparse data), cropped to the search area
 *   score -> for every candidate point, every aircraft and every recording start time,
 *            the exact curve that point would hear (velocity at emission time, wind,
 *            temperature); all selected passes share one clock offset
 *         -> 95% confidence region from the misfit, split into separate clusters
 */
(function (root) {
  'use strict';
  const E = {};
  const DEG = Math.PI / 180;
  const FT = 0.3048, KT = 0.514444, FPM = 0.00508;

  E.soundSpeed = (tempC) => 331.3 * Math.sqrt(1 + tempC / 273.15);

  // ------------------------------------------------------------------ small utils
  function rng(seed) {
    let a = seed >>> 0;
    const u = () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    u.normal = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let x, y, r;
      do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
      const m = Math.sqrt(-2 * Math.log(r) / r);
      spare = y * m;
      return x * m;
    };
    return u;
  }
  E.rng = rng;

  function median(arr) {
    const a = Float64Array.from(arr).sort();
    const n = a.length;
    if (!n) return NaN;
    return n % 2 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
  }
  E.median = median;

  function medianFilter(x, w) {
    const n = x.length, h = Math.floor(w / 2), out = new Float64Array(n), buf = [];
    for (let i = 0; i < n; i++) {
      buf.length = 0;
      for (let j = Math.max(0, i - h); j <= Math.min(n - 1, i + h); j++) buf.push(x[j]);
      buf.sort((p, q) => p - q);
      out[i] = buf[buf.length >> 1];
    }
    return out;
  }

  // Best segment of a ridge: maximum-sum run of (z - thr), where dips below thr cost 4x.
  // Short dips (a crossing, a gust) are bridged; long stretches of noise are not.
  function bestSegment(sm, thr) {
    let best = 0, bs = 0, be = 0, cur = 0, cs = 0;
    for (let i = 0; i < sm.length; i++) {
      if (cur <= 0) { cur = 0; cs = i; }
      const d = sm[i] - thr;
      cur += d >= 0 ? d : 4 * d;
      if (cur > best) { best = cur; bs = cs; be = i + 1; }
    }
    return [bs, be];
  }

  function longestRun(mask) {
    let best = 0, cur = 0, end = -1;
    for (let i = 0; i < mask.length; i++) {
      cur = mask[i] ? cur + 1 : 0;
      if (cur > best) { best = cur; end = i; }
    }
    return best ? [end - best + 1, end + 1] : [0, 0];
  }

  function solveLinear(A, b, n) { // Gaussian elimination with partial pivoting, A is n*n row-major
    const M = Float64Array.from(A), x = Float64Array.from(b);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r;
      if (Math.abs(M[p * n + c]) < 1e-300) return null;
      if (p !== c) {
        for (let k = 0; k < n; k++) { const t = M[c * n + k]; M[c * n + k] = M[p * n + k]; M[p * n + k] = t; }
        const t = x[c]; x[c] = x[p]; x[p] = t;
      }
      for (let r = c + 1; r < n; r++) {
        const f = M[r * n + c] / M[c * n + c];
        if (!f) continue;
        for (let k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k];
        x[r] -= f * x[c];
      }
    }
    for (let c = n - 1; c >= 0; c--) {
      let s = x[c];
      for (let k = c + 1; k < n; k++) s -= M[c * n + k] * x[k];
      x[c] = s / M[c * n + c];
    }
    return x;
  }

  function invert(A, n) {
    const out = new Float64Array(n * n);
    for (let c = 0; c < n; c++) {
      const e = new Float64Array(n); e[c] = 1;
      const col = solveLinear(A, e, n);
      if (!col) return null;
      for (let r = 0; r < n; r++) out[r * n + c] = col[r];
    }
    return out;
  }

  // ------------------------------------------------------------------ FFT
  const fftCache = new Map();
  function fftTables(n) {
    let t = fftCache.get(n);
    if (t) return t;
    const bits = Math.round(Math.log2(n));
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0, x = i;
      for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
      rev[i] = r;
    }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
    t = { rev, cos, sin };
    fftCache.set(n, t);
    return t;
  }
  function fft(re, im) {
    const n = re.length, { rev, cos, sin } = fftTables(n);
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = i + j, b = a + half, wr = cos[k], wi = -sin[k];
          const tr = wr * re[b] - wi * im[b], ti = wr * im[b] + wi * re[b];
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        }
      }
    }
  }
  E.fft = fft;

  // ------------------------------------------------------------------ resampling
  function lowpassDecimate(x, factor) {
    const taps = 32 * factor + 1, m = (taps - 1) / 2, fc = 0.45 / factor, h = new Float64Array(taps);
    let s = 0;
    for (let i = 0; i < taps; i++) {
      const k = i - m;
      const sinc = k === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * k) / (Math.PI * k);
      const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (taps - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (taps - 1));
      h[i] = sinc * w; s += h[i];
    }
    for (let i = 0; i < taps; i++) h[i] /= s;
    const nOut = Math.floor(x.length / factor), y = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {
      const j0 = o * factor - m;
      let acc = 0;
      const i0 = Math.max(0, -j0), i1 = Math.min(taps, x.length - j0);
      for (let i = i0; i < i1; i++) acc += h[i] * x[j0 + i];
      y[o] = acc;
    }
    return y;
  }
  E.resample = function (x, srIn, srOut) {
    if (srIn === srOut) return Float32Array.from(x);
    const r = srIn / srOut;
    if (Math.abs(r - Math.round(r)) < 1e-9) return lowpassDecimate(x, Math.round(r));
    const f = Math.max(1, Math.floor(r));
    const y = f > 1 ? lowpassDecimate(x, f) : x;
    const sr1 = srIn / f, n = Math.floor(y.length * srOut / sr1), out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * sr1 / srOut, k = Math.floor(p), a = p - k;
      out[i] = (y[k] || 0) * (1 - a) + (y[k + 1] || 0) * a;
    }
    return out;
  };

  // ------------------------------------------------------------------ spectrogram
  E.spectrogram = function (x, sr, o = {}) {
    const win = o.win || 1.0, zpad = o.zpad || 2;
    const dur = x.length / sr;
    const hop = o.hop || (dur > 300 ? 0.2 : 0.1);
    const fmin = o.fmin || 20, fmax = Math.min(o.fmax || 900, 0.47 * sr);
    const nper = Math.round(win * sr), hs = Math.max(1, Math.round(hop * sr));
    if (x.length < nper * 4) throw new Error(`The recording is ${dur.toFixed(1)} s long. It needs at least ${4 * win} s of audio.`);
    let nfft = 1;
    while (nfft < nper * zpad) nfft <<= 1;
    const nT = 1 + Math.floor((x.length - nper) / hs);
    const df = sr / nfft, k0 = Math.ceil(fmin / df), k1 = Math.floor(fmax / df), nF = k1 - k0 + 1;
    const freqs = new Float64Array(nF), times = new Float64Array(nT);
    for (let k = 0; k < nF; k++) freqs[k] = (k0 + k) * df;
    for (let t = 0; t < nT; t++) times[t] = (t * hs + nper / 2) / sr;
    const w = new Float64Array(nper);
    for (let i = 0; i < nper; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 0.5) / nper);
    const db = new Float32Array(nT * nF), re = new Float64Array(nfft), im = new Float64Array(nfft);
    for (let t = 0; t < nT; t += 2) { // two real frames per complex FFT
      re.fill(0); im.fill(0);
      const s0 = t * hs, two = t + 1 < nT;
      for (let i = 0; i < nper; i++) re[i] = x[s0 + i] * w[i];
      if (two) { const s1 = (t + 1) * hs; for (let i = 0; i < nper; i++) im[i] = x[s1 + i] * w[i]; }
      fft(re, im);
      for (let k = 0; k < nF; k++) {
        const kk = k0 + k, nk = (nfft - kk) % nfft;
        const ar = 0.5 * (re[kk] + re[nk]), ai = 0.5 * (im[kk] - im[nk]);
        db[t * nF + k] = 10 * Math.log10(ar * ar + ai * ai + 1e-20);
        if (two) {
          const br = 0.5 * (im[kk] + im[nk]), bi = -0.5 * (re[kk] - re[nk]);
          db[(t + 1) * nF + k] = 10 * Math.log10(br * br + bi * bi + 1e-20);
        }
      }
      if (o.onProgress && (t & 127) === 0) o.onProgress(t / nT);
    }
    return { nT, nF, times, freqs, df, hop: hs / sr, win: nper / sr, sr, db, duration: dur };
  };

  // Remove steady lines (per-bin median over time), then z-score each frame (median / MAD).
  E.whiten = function (spec) {
    const { nT, nF, db } = spec, z = new Float32Array(nT * nF);
    const stride = Math.max(1, Math.floor(nT / 500)), col = new Float32Array(Math.ceil(nT / stride));
    const binMed = new Float32Array(nF);
    for (let b = 0; b < nF; b++) {
      let m = 0;
      for (let t = 0; t < nT; t += stride) col[m++] = db[t * nF + b];
      const s = col.subarray(0, m).slice().sort();
      binMed[b] = s[m >> 1];
    }
    const row = new Float32Array(nF), tmp = new Float32Array(nF);
    for (let t = 0; t < nT; t++) {
      for (let b = 0; b < nF; b++) row[b] = db[t * nF + b] - binMed[b];
      tmp.set(row); tmp.sort();
      const med = tmp[nF >> 1];
      for (let b = 0; b < nF; b++) tmp[b] = Math.abs(row[b] - med);
      tmp.sort();
      const mad = 1.4826 * tmp[nF >> 1] + 1e-6;
      for (let b = 0; b < nF; b++) z[t * nF + b] = (row[b] - med) / mad;
    }
    return z;
  };

  function reverseFrames(arr, nT, nF) {
    const out = new Float32Array(arr.length);
    for (let t = 0; t < nT; t++) out.set(arr.subarray((nT - 1 - t) * nF, (nT - t) * nF), t * nF);
    return out;
  }

  // ------------------------------------------------------------------ ridge tracking
  // Max-sum path through z with a quadratic jump penalty (penW at a jump of K bins). The inner max
  // over previous bins is a distance transform (lower envelope of parabolas), so each frame is O(bins).
  function viterbi(z, nF, t0, t1, b0, b1, K, penW) {
    const nb = b1 - b0, nt = t1 - t0, a = penW / (K * K);
    let score = new Float64Array(nb), next = new Float64Array(nb);
    const back = new Int16Array(nt * nb), cost = new Float64Array(nb), v = new Int32Array(nb), zz = new Float64Array(nb + 1);
    const clip = (x) => (x < -3 ? -3 : x > 8 ? 8 : x);
    for (let j = 0; j < nb; j++) score[j] = clip(z[t0 * nF + b0 + j]);
    for (let i = 1; i < nt; i++) {
      for (let j = 0; j < nb; j++) cost[j] = -score[j];
      let k = 0;
      v[0] = 0; zz[0] = -Infinity; zz[1] = Infinity;
      for (let q = 1; q < nb; q++) {
        let r = v[k], sct = (cost[q] + a * q * q - (cost[r] + a * r * r)) / (2 * a * (q - r));
        while (sct <= zz[k]) { k--; r = v[k]; sct = (cost[q] + a * q * q - (cost[r] + a * r * r)) / (2 * a * (q - r)); }
        k++; v[k] = q; zz[k] = sct; zz[k + 1] = Infinity;
      }
      k = 0;
      const row = (t0 + i) * nF + b0;
      for (let j = 0; j < nb; j++) {
        while (zz[k + 1] < j) k++;
        const q = v[k], d = j - q;
        next[j] = -(a * d * d + cost[q]) + clip(z[row + j]);
        back[i * nb + j] = d;
      }
      const tmp = score; score = next; next = tmp;
    }
    const path = new Int32Array(nt);
    let j = 0, bv = -Infinity;
    for (let q = 0; q < nb; q++) if (score[q] > bv) { bv = score[q]; j = q; }
    path[nt - 1] = j;
    for (let i = nt - 1; i > 0; i--) { j -= back[i * nb + j]; path[i - 1] = j; }
    for (let i = 0; i < nt; i++) path[i] += b0;
    return path;
  }

  // ------------------------------------------------------------------ Doppler model
  function spCurve(t, f0, v, d, tca, c) {
    const T = t - tca, a = c * c - v * v;
    const tau = (c * c * T - Math.sqrt(c * c * v * v * T * T + a * d * d)) / a;
    const r = Math.sqrt(d * d + v * v * tau * tau);
    return f0 / (1 + v * v * tau / (c * r));
  }
  E.spCurve = spCurve;
  E.spEmission = function (t, v, d, tca, c) {
    const T = t - tca, a = c * c - v * v;
    return tca + (c * c * T - Math.sqrt(c * c * v * v * T * T + a * d * d)) / a;
  };

  function initialGuess(t, f, c) {
    const n = f.length, k = Math.max(3, Math.floor(n / 10));
    let fh = median(f.subarray(0, k)), fl = median(f.subarray(n - k));
    if (fh <= fl) fh = fl * 1.002 + 1e-3;
    const f0 = 2 * fh * fl / (fh + fl);
    const v = Math.min(Math.max(c * (fh - fl) / (fh + fl), 10), 0.9 * c);
    let idx = 0, bd = Infinity;
    for (let i = 0; i < n; i++) { const d = Math.abs(f[i] - f0); if (d < bd) { bd = d; idx = i; } }
    const lo = Math.max(0, idx - k), hi = Math.min(n - 1, idx + k);
    const slope = t[hi] > t[lo] ? (f[hi] - f[lo]) / (t[hi] - t[lo]) : -1;
    const d = Math.min(Math.max(f0 * v * v / (c * Math.max(Math.abs(slope), 1e-3)), 30), 3e4);
    return [f0, v, d, t[idx] - d / c];
  }

  // Levenberg-Marquardt on the straight-pass model, optional soft-L1 reweighting.
  function lm(t, f, p0, lo, hi, c, robust, fscale, maxIter) {
    const n = t.length, m = 4;
    const w = new Float64Array(n).fill(1);
    let p = p0.slice();
    const resid = (pp, out) => { for (let i = 0; i < n; i++) out[i] = spCurve(t[i], pp[0], pp[1], pp[2], pp[3], c) - f[i]; return out; };
    let r = resid(p, new Float64Array(n));
    const rt = new Float64Array(n), cols = [0, 1, 2, 3].map(() => new Float64Array(n));
    for (let outer = 0; outer < (robust ? 3 : 1); outer++) {
      if (robust) for (let i = 0; i < n; i++) { const q = r[i] / fscale; w[i] = 1 / Math.sqrt(1 + q * q); }
      let cost = 0;
      for (let i = 0; i < n; i++) cost += w[i] * r[i] * r[i];
      let lam = 1e-3;
      for (let it = 0; it < maxIter; it++) {
        for (let a = 0; a < m; a++) {
          const h = 1e-6 * Math.max(Math.abs(p[a]), 1), pp = p.slice();
          pp[a] += h; resid(pp, rt);
          for (let i = 0; i < n; i++) cols[a][i] = (rt[i] - r[i]) / h;
        }
        const JTJ = new Float64Array(16), JTr = new Float64Array(4);
        for (let a = 0; a < m; a++) {
          let s = 0;
          for (let i = 0; i < n; i++) s += w[i] * cols[a][i] * r[i];
          JTr[a] = -s;
          for (let b = a; b < m; b++) {
            let q = 0;
            for (let i = 0; i < n; i++) q += w[i] * cols[a][i] * cols[b][i];
            JTJ[a * 4 + b] = JTJ[b * 4 + a] = q;
          }
        }
        let improved = false;
        for (let tries = 0; tries < 12; tries++) {
          const A = JTJ.slice();
          for (let a = 0; a < m; a++) A[a * 5] *= 1 + lam;
          const delta = solveLinear(A, JTr, 4);
          if (!delta) { lam *= 10; continue; }
          const pn = p.slice();
          for (let a = 0; a < m; a++) pn[a] = Math.min(hi[a], Math.max(lo[a], p[a] + delta[a]));
          const rn = resid(pn, new Float64Array(n));
          let cn = 0;
          for (let i = 0; i < n; i++) cn += w[i] * rn[i] * rn[i];
          if (cn < cost) {
            improved = (cost - cn) / Math.max(cost, 1e-30) > 1e-9;
            p = pn; r = rn; cost = cn; lam = Math.max(lam / 3, 1e-9);
            break;
          }
          lam *= 4;
        }
        if (!improved) break;
      }
    }
    let loss = 0;
    for (let i = 0; i < n; i++) { const q = r[i] / fscale; loss += robust ? 2 * (Math.sqrt(1 + q * q) - 1) : q * q; }
    return { p, r, loss };
  }

  function jacobian(t, p, c) {
    const n = t.length, J = [0, 1, 2, 3].map(() => new Float64Array(n));
    for (let a = 0; a < 4; a++) {
      const h = 1e-6 * Math.max(Math.abs(p[a]), 1), pp = p.slice(); pp[a] += h;
      for (let i = 0; i < n; i++) J[a][i] = (spCurve(t[i], pp[0], pp[1], pp[2], pp[3], c) - spCurve(t[i], p[0], p[1], p[2], p[3], c)) / h;
    }
    return J;
  }

  E.fitRidge = function (t, f, c, df) {
    const n = t.length;
    if (n < 10) return null;
    const span = t[n - 1] - t[0];
    let fmin = Infinity, fmax = -Infinity;
    for (let i = 0; i < n; i++) { fmin = Math.min(fmin, f[i]); fmax = Math.max(fmax, f[i]); }
    const lo = [fmin * 0.5, 5, 5, t[0] - 2 * span - 60], hi = [fmax * 1.5, 0.95 * c, 5e4, t[n - 1] + 2 * span + 60];
    const g = initialGuess(t, f, c);
    // multi-start on a subsample, then polish the winner on every point
    const step = Math.max(1, Math.floor(n / 150));
    const ts = [], fs = [];
    for (let i = 0; i < n; i += step) { ts.push(t[i]); fs.push(f[i]); }
    const tsub = Float64Array.from(ts), fsub = Float64Array.from(fs), fscale = Math.max(df, 0.25);
    let best = null;
    for (const dm of [0.5, 1, 2]) for (const ts_ of [-0.15, 0, 0.15]) {
      const p0 = [g[0], g[1], g[2] * dm, g[3] + ts_ * span].map((v, a) => Math.min(hi[a] - 1e-6, Math.max(lo[a] + 1e-6, v)));
      const res = lm(tsub, fsub, p0, lo, hi, c, true, fscale, 30);
      if (!best || res.loss < best.loss) best = res;
    }
    let fit = lm(t, f, best.p, lo, hi, c, true, fscale, 60);
    const absr = Float64Array.from(fit.r, Math.abs);
    const sig = 1.4826 * median(absr) + 1e-9;
    const keep = new Uint8Array(n);
    let nin = 0;
    for (let i = 0; i < n; i++) if (Math.abs(fit.r[i]) <= Math.max(3 * sig, df)) { keep[i] = 1; nin++; }
    if (nin < 10) return null;
    const ti = new Float64Array(nin), fi = new Float64Array(nin);
    for (let i = 0, k = 0; i < n; i++) if (keep[i]) { ti[k] = t[i]; fi[k] = f[i]; k++; }
    fit = lm(ti, fi, fit.p, lo, hi, c, false, 1, 80);
    const r = fit.r;
    let sse = 0;
    for (let i = 0; i < nin; i++) sse += r[i] * r[i];
    // overlapping frames make neighbouring residuals correlated: inflate by 1 + 2*sum(acf)
    let sumRho = 0;
    for (let lag = 1; lag <= Math.min(300, Math.floor(nin / 3)); lag++) {
      let num = 0;
      for (let i = lag; i < nin; i++) num += r[i] * r[i - lag];
      const rho_ = sse > 0 ? num / sse : 0;
      if (rho_ <= 0.02) break;
      sumRho += rho_;
    }
    const infl = Math.min(nin / 5, 1 + 2 * sumRho), rho = sumRho;
    const s2 = sse / Math.max(1, nin - 4);
    const J = jacobian(ti, fit.p, c), JTJ = new Float64Array(16);
    for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) { let q = 0; for (let i = 0; i < nin; i++) q += J[a][i] * J[b][i]; JTJ[a * 4 + b] = q; }
    const cov = invert(JTJ, 4);
    const se = [0, 1, 2, 3].map((a) => (cov ? Math.sqrt(Math.max(0, cov[a * 5] * s2 * infl)) : NaN));
    const [f0, v, d, tca] = fit.p;
    return {
      f0, v, d, tca, tcaRecv: tca + d / c, se: { f0: se[0], v: se[1], d: se[2], tca: se[3] },
      rms: Math.sqrt(sse / nin), rho, neff: Math.max(5, nin / infl),
      t: ti, f: fi, nIn: nin, nOut: n - nin,
    };
  };

  // ------------------------------------------------------------------ detection
  function refinePeaks(spec, dbRow, path, i0) {
    const { nF, freqs, df } = spec, out = new Float64Array(path.length);
    for (let k = 0; k < path.length; k++) {
      const b = path[k], row = dbRow(i0 + k);
      let f = freqs[b];
      if (b > 0 && b < nF - 1) {
        const a = row[b - 1], m = row[b], c = row[b + 1], den = a - 2 * m + c;
        if (Math.abs(den) > 1e-12) f += Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / den)) * df;
      }
      out[k] = f;
    }
    return out;
  }

  function curveMeanZ(spec, z, frames, curve, scale) {
    const { nF, freqs, df } = spec;
    let s = 0, n = 0;
    for (let k = 0; k < curve.length; k++) {
      const b = Math.round((curve[k] * scale - freqs[0]) / df);
      if (b < 0 || b >= nF) continue;
      s += z[frames[k] * nF + b]; n++;
    }
    return n > curve.length * 0.6 ? s / n : NaN;
  }

  // Matched-curve score: mean z along the fitted Doppler curve, in units of the spread of
  // the same curve slid up and down in frequency (where there is no tone).
  function matchedScore(spec, z, frames, curve) {
    const S = curveMeanZ(spec, z, frames, curve, 1);
    const nulls = [];
    for (let k = 0; k < 48; k++) {
      const u = 0.045 + (0.22 - 0.045) * (k >> 1) / 23;
      const v = curveMeanZ(spec, z, frames, curve, Math.exp(k & 1 ? u : -u));
      if (Number.isFinite(v)) nulls.push(v);
    }
    if (nulls.length < 10 || !Number.isFinite(S)) return { score: 0, S: NaN };
    const med = median(nulls), mad = 1.4826 * median(nulls.map((v) => Math.abs(v - med))) + 1e-3;
    return { score: (S - med) / mad, S };
  }

  function buildDetection(spec, z, dbRow, path, i0, i1, o) {
    const freqs = refinePeaks(spec, dbRow, path.subarray(i0, i1), i0);
    const times = spec.times.subarray(i0, i1);
    return evaluateRidge(spec, z, times, freqs, o);
  }

  function frameOf(spec, t) { return Math.max(0, Math.min(spec.nT - 1, Math.round((t - spec.times[0]) / spec.hop))); }

  function evaluateRidge(spec, z, times, freqs, o) {
    const fit = E.fitRidge(times, freqs, o.c, spec.df);
    if (!fit) return null;
    const n = fit.t.length, frames = new Int32Array(n), curve = new Float64Array(n);
    for (let k = 0; k < n; k++) { frames[k] = frameOf(spec, fit.t[k]); curve[k] = spCurve(fit.t[k], fit.f0, fit.v, fit.d, fit.tca, o.c); }
    let maxSlope = 0;
    for (let k = 1; k < n; k++) if (fit.t[k] > fit.t[k - 1]) maxSlope = Math.max(maxSlope, Math.abs(curve[k] - curve[k - 1]) / (fit.t[k] - fit.t[k - 1]));
    const ms = matchedScore(spec, z, frames, curve);
    const times_ = fit.t;
    const reasons = [], warnings = [];
    const sweep = curve[0] - curve[n - 1];
    if (fit.v < 15 || fit.v > 0.95 * o.c) reasons.push(`implausible speed ${fit.v.toFixed(0)} m/s`);
    if (fit.d < 20 || fit.d > 3e4) reasons.push(`implausible range ${fit.d.toFixed(0)} m`);
    if (sweep < 2 * spec.df) reasons.push('no falling Doppler sweep');
    const maxRms = Math.max(0.6, 0.008 * fit.f0);
    if (fit.rms > maxRms) reasons.push(`fit residual ${fit.rms.toFixed(2)} Hz > ${maxRms.toFixed(2)} Hz`);
    if (maxSlope > o.maxSlopeRel * fit.f0 * 1.05) reasons.push('steeper than the tracker allows');
    const cpaInside = fit.tcaRecv >= times_[0] + 2 && fit.tcaRecv <= times_[n - 1] - 2;
    if (!cpaInside) warnings.push('closest approach is outside the heard segment, so speed and range are loose');
    return {
      t0: times_[0], t1: times_[n - 1], i0: frames[0], i1: frames[n - 1] + 1, f0: fit.f0, v: fit.v, d: fit.d, tca: fit.tca, tcaRecv: fit.tcaRecv,
      se: fit.se, rms: fit.rms, neff: fit.neff, rho: fit.rho, t: fit.t, f: fit.f, nIn: fit.nIn,
      fStart: curve[0], fEnd: curve[n - 1], score: ms.score, meanZ: ms.S, plausible: reasons.length === 0,
      reasons, warnings, cpaInside, dRel: cpaInside, vRel: cpaInside || sweep > 0.15 * fit.f0,
    };
  }

  function maskAlong(zw, spec, i0, i1, fOf, halfBins) {
    const { nF, freqs, df } = spec;
    for (let i = Math.max(0, i0); i < Math.min(spec.nT, i1); i++) {
      const b = Math.round((fOf(i) - freqs[0]) / df);
      for (let k = b - halfBins; k <= b + halfBins; k++) if (k >= 0 && k < nF) zw[i * nF + k] = 0;
    }
  }

  function trackInBox(spec, zScore, zWork, dbRow, i0, i1, f0, f1, o) {
    const { nF, freqs, df, hop } = spec;
    const b0 = Math.max(0, Math.floor((f0 - freqs[0]) / df)), b1 = Math.min(nF, Math.ceil((f1 - freqs[0]) / df) + 1);
    if (b1 - b0 < 8 || i1 - i0 < 10) return null;
    const fc = Math.sqrt(Math.max(f0, 1) * f1);
    const K = Math.max(1, Math.min(120, Math.ceil(o.maxSlopeRel * fc * hop / df)));
    const path = viterbi(zWork, nF, i0, i1, b0, b1, K, o.penW);
    const zr = new Float64Array(i1 - i0);
    for (let k = 0; k < zr.length; k++) zr[k] = zWork[(i0 + k) * nF + path[k]];
    const sm = medianFilter(zr, Math.max(3, Math.round(o.trimSmooth / hop) | 1));
    const [a, b] = bestSegment(sm, o.trimZ);
    if ((b - a) * hop < o.minDur) return { path, run: [i0 + a, i0 + b], det: null };
    const full = new Int32Array(spec.nT);
    full.set(path, i0);
    const det = buildDetection(spec, zScore, dbRow, full, i0 + a, i0 + b, o);
    return { path, run: [i0 + a, i0 + b], det };
  }

  function sameRidgeUnused(a, b, c) {
    const lo = Math.max(a.t0, b.t0), hi = Math.min(a.t1, b.t1);
    if (hi - lo < 0.3 * Math.min(a.t1 - a.t0, b.t1 - b.t0)) return false;
    const diffs = [];
    for (let k = 0; k <= 20; k++) {
      const t = lo + (hi - lo) * k / 20;
      diffs.push(Math.abs(spCurve(t, a.f0, a.v, a.d, a.tca, c) - spCurve(t, b.f0, b.v, b.d, b.tca, c)));
    }
    return median(diffs) < Math.max(1.5, 0.015 * Math.min(a.f0, b.f0));
  }

  function search(spec, zScore, dbRow, o, progress) {
    const { nT, nF, freqs, df, hop } = spec;
    const zw = Float32Array.from(zScore);
    const found = [];
    const bands = [];
    for (let fl = o.fmin; fl * o.bandRatio <= freqs[nF - 1] * 1.02; fl *= o.bandStep) bands.push([fl, Math.min(fl * o.bandRatio, freqs[nF - 1])]);
    bands.forEach(([fl, fh], bi) => {
      for (let iter = 0; iter < o.perBand; iter++) {
        const res = trackInBox(spec, zScore, zw, dbRow, 0, nT, fl, fh, o);
        if (!res || res.run[1] - res.run[0] < 2) break;
        let det = res.det;
        const [r0, r1] = res.run;
        // mask what was traced so the next pass (and later bands) look elsewhere
        maskAlong(zw, spec, r0, r1, (i) => freqs[res.path[i]], Math.ceil(2.5 / df));
        if (!det) continue;
        // re-track in a box around the fitted curve: lets a ridge run past the band edges
        const len = r1 - r0;
        const j0 = Math.max(0, r0 - Math.round(0.4 * len)), j1 = Math.min(nT, r1 + Math.round(0.4 * len));
        const ca = spCurve(spec.times[j0], det.f0, det.v, det.d, det.tca, o.c), cb = spCurve(spec.times[j1 - 1], det.f0, det.v, det.d, det.tca, o.c);
        const box = trackInBox(spec, zScore, zScore, dbRow, j0, j1, Math.min(ca, cb) / 1.08, Math.max(ca, cb) * 1.08, o);
        if (box && box.det && box.det.plausible && box.det.score > det.score) det = box.det;
        maskAlong(zw, spec, Math.max(0, det.i0 - len), Math.min(nT, det.i1 + len), (i) => spCurve(spec.times[i], det.f0, det.v, det.d, det.tca, o.c), Math.ceil(2.5 / df));
        found.push(det);
      }
      if (progress) progress((bi + 1) / bands.length);
    });
    let dets = dedupe(spec, found);
    dets = dets.map((d) => (d.plausible ? guidedExtend(spec, zScore, d, o) : d));
    return mergeFragments(spec, zScore, dedupe(spec, dets), o);
  }

  // Harmonic bookkeeping against stronger, accepted ridges (their fitted curves, scaled by simple
  // ratios). A ridge that one ratio explains is a harmonic of that pass and takes its geometry; a
  // ridge explained piecewise by several ratios was stitched across crossing tones and is dropped.
  const RATIOS = [1, 2, 3, 4, 5, 1 / 2, 1 / 3, 1 / 4, 3 / 2, 2 / 3, 4 / 3, 3 / 4, 5 / 2, 2 / 5, 5 / 3, 3 / 5];
  function attributeHarmonics(dets, threshold, c) {
    const parents = [], out = [];
    for (const d of dets.sort((a, b) => b.score - a.score)) {
      if (!d.plausible) { out.push(d); continue; }
      let best = null, union = 0;
      const n = d.t.length;
      const hitAny = new Uint8Array(n);
      for (const P of parents) for (const q of RATIOS) {
        let hit = 0;
        for (let k = 0; k < n; k++) {
          const fp = q * spCurve(d.t[k], P.f0, P.v, P.d, P.tca, c);
          if (Math.abs(fp - d.f[k]) <= Math.max(1.5, 0.012 * fp)) { hit++; hitAny[k] = 1; }
        }
        if (!best || hit > best.hit) best = { P, q, hit };
      }
      for (let k = 0; k < n; k++) union += hitAny[k];
      if (best && best.hit >= 0.7 * n) {
        if (best.q === 1) continue; // the same ridge again
        const P = best.P;
        { // keep only the points on the parent's scaled curve
          const keep = [];
          for (let k = 0; k < n; k++) { const fp = best.q * spCurve(d.t[k], P.f0, P.v, P.d, P.tca, c); if (Math.abs(fp - d.f[k]) <= Math.max(1.5, 0.012 * fp)) keep.push(k); }
          d.t = Float64Array.from(keep, (k) => d.t[k]); d.f = Float64Array.from(keep, (k) => d.f[k]);
          d.t0 = d.t[0]; d.t1 = d.t[d.t.length - 1]; d.nIn = d.t.length;
        }
        const twin = out.find((r) => r.harmonicOf === P && Math.abs(r.ratio - best.q) < 1e-9);
        if (twin) { // another piece of a harmonic we already have: pool the points
          const m = new Map();
          for (const R of [twin, d]) for (let k = 0; k < R.t.length; k++) m.set(Math.round(R.t[k] * 1000), R.f[k]);
          const keys = [...m.keys()].sort((a, b) => a - b);
          twin.t = Float64Array.from(keys, (k) => k / 1000); twin.f = Float64Array.from(keys, (k) => m.get(k));
          twin.t0 = twin.t[0]; twin.t1 = twin.t[twin.t.length - 1]; twin.nIn = twin.t.length;
          twin.score = Math.max(twin.score, d.score);
          continue;
        }
        Object.assign(d, { harmonicOf: P, ratio: best.q, v: P.v, d: P.d, tca: P.tca, tcaRecv: P.tcaRecv, f0: best.q * P.f0, se: P.se, dRel: P.dRel, vRel: P.vRel });
        d.accepted = P.accepted && d.score >= 0.5 * threshold;
        out.push(d);
        continue;
      }
      if (parents.length && union >= 0.5 * n) continue; // stitched from other tones
      out.push(d);
      if (d.accepted) parents.push(d);
    }
    return out;
  }

  // Re-collect ridge points along a narrow corridor around the fitted curve over the whole clip
  // (wider where a fast sweep smears the tone across bins), then refit. Bridges the gap at
  // closest approach and picks up the faint approach and departure.
  function guidedExtend(spec, z, det, o) {
    const { nT, nF, freqs, df, times, win } = spec;
    for (let iter = 0; iter < 3; iter++) {
      // keep the traced points; add corridor peaks only in frames the trace did not cover
      const have = new Set();
      const ts = Array.from(det.t), fs = Array.from(det.f);
      for (let k = 0; k < det.t.length; k++) have.add(frameOf(spec, det.t[k]));
      for (let i = 0; i < nT; i++) {
        if (have.has(i)) continue;
        const t = times[i], fm = spCurve(t, det.f0, det.v, det.d, det.tca, o.c);
        if (fm < freqs[0] || fm > freqs[nF - 1]) continue;
        const slope = Math.abs(spCurve(t + 0.05, det.f0, det.v, det.d, det.tca, o.c) - spCurve(t - 0.05, det.f0, det.v, det.d, det.tca, o.c)) / 0.1;
        const w = Math.max(2 * df, 3 * det.rms, 0.5 * slope * win);
        const b0 = Math.max(1, Math.floor((fm - w - freqs[0]) / df)), b1 = Math.min(nF - 2, Math.ceil((fm + w - freqs[0]) / df));
        let bb = -1, bz = -Infinity;
        for (let b = b0; b <= b1; b++) { const v = z[i * nF + b]; if (v > bz) { bz = v; bb = b; } }
        if (bb < 0 || bz < o.guideZ + 0.5 * Math.log(1 + (b1 - b0) / 8)) continue; // wider corridor, higher bar
        const a = z[i * nF + bb - 1], m = z[i * nF + bb], c = z[i * nF + bb + 1], den = a - 2 * m + c;
        ts.push(t); fs.push(freqs[bb] + (Math.abs(den) > 1e-12 ? Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / den)) * df : 0));
      }
      if (ts.length < det.t.length + 5) break;
      const ord = ts.map((_, k) => k).sort((p, q) => ts[p] - ts[q]);
      const U = evaluateRidge(spec, z, Float64Array.from(ord, (k) => ts[k]), Float64Array.from(ord, (k) => fs[k]), o);
      if (!U || !U.plausible || U.nIn < 0.9 * det.nIn || U.rms > 1.5 * det.rms + 0.15 || U.score < 0.6 * det.score) break;
      const grew = U.nIn > det.nIn * 1.02;
      det = U;
      if (!grew) break;
    }
    return det;
  }

  // Two detections are the same ridge when most points of the smaller one sit on the other.
  function overlapFrac(spec, a, b) {
    const m = new Map();
    for (let k = 0; k < a.t.length; k++) m.set(frameOf(spec, a.t[k]), a.f[k]);
    let hit = 0;
    for (let k = 0; k < b.t.length; k++) {
      const fa = m.get(frameOf(spec, b.t[k]));
      if (fa !== undefined && Math.abs(fa - b.f[k]) <= Math.max(1.5, 0.01 * fa)) hit++;
    }
    return hit / Math.min(a.t.length, b.t.length);
  }

  function dedupe(spec, found) {
    found.sort((a, b) => b.score - a.score);
    const out = [];
    for (const d of found) if (!out.some((k) => overlapFrac(spec, k, d) > 0.4 || overlapFrac(spec, d, k) > 0.4)) out.push(d);
    return out;
  }

  // Pieces of one ridge broken by a crossing tone or a quiet stretch: refit the union and keep it
  // when the union fits about as well as the pieces did.
  function mergeFragments(spec, z, dets, o) {
    let changed = true;
    while (changed) {
      changed = false;
      dets.sort((a, b) => b.score - a.score);
      outer: for (let i = 0; i < dets.length; i++) {
        for (let j = i + 1; j < dets.length; j++) {
          const A = dets[i], B = dets[j];
          // candidates: one piece ends shortly before the other starts, and pitch only falls across the gap
          const [early, late] = A.t0 <= B.t0 ? [A, B] : [B, A];
          const gap = late.t0 - early.t1;
          const fe = spCurve(early.t1, early.f0, early.v, early.d, early.tca, o.c), fl = spCurve(late.t0, late.f0, late.v, late.d, late.tca, o.c);
          if (gap < -5 || gap > 40 || fe < 0.97 * fl || fe > 2.0 * fl) continue;
          if (gap < 0 && overlapFrac(spec, early, late) < 0.2) {
            // overlapping in time at different pitch is two tones, not one broken ridge
            const fm = spCurve(late.t0, early.f0, early.v, early.d, early.tca, o.c);
            if (Math.abs(fm - fl) > Math.max(2, 0.03 * fl)) continue;
          }
          const pts = [];
          for (const D of [A, B]) for (let k = 0; k < D.t.length; k++) pts.push([D.t[k], D.f[k]]);
          pts.sort((p, q) => p[0] - q[0]);
          const U = evaluateRidge(spec, z, Float64Array.from(pts, (p) => p[0]), Float64Array.from(pts, (p) => p[1]), o);
          if (!U || !U.plausible || U.nIn < 0.8 * (A.nIn + B.nIn)) continue;
          if (U.rms > 1.4 * Math.max(A.rms, B.rms) + 0.1) continue;
          dets.splice(j, 1); dets[i] = U; changed = true;
          break outer;
        }
      }
    }
    return dets.sort((a, b) => b.score - a.score);
  }

  E.detectDefaults = {
    fmin: 25, bandRatio: 2.5, bandStep: Math.pow(2, 1 / 3), perBand: 3, maxSlopeRel: 0.18, penW: 2,
    trimZ: 1.5, trimSmooth: 2, minDur: 8, c: 343, minScore: 8, nullFactor: 1.3, guideZ: 2.2,
  };

  // Full automatic search plus the time-reversed null run.
  E.detectFlyovers = function (spec, z, opts, progress) {
    const o = Object.assign({}, E.detectDefaults, opts);
    const { nT, nF } = spec;
    // peaks are refined on the whitened spectrum, so steady hum lines do not pull them
    const rowFwd = (i) => z.subarray(i * nF, (i + 1) * nF);
    const dets = search(spec, z, rowFwd, o, progress && ((p) => progress(0.5 * p)));
    const zRev = reverseFrames(z, nT, nF);
    const rowRev = (i) => zRev.subarray(i * nF, (i + 1) * nF);
    const nullDets = search(spec, zRev, rowRev, o, progress && ((p) => progress(0.5 + 0.5 * p)));
    let nullMax = 0;
    for (const d of nullDets) if (d.plausible) nullMax = Math.max(nullMax, d.score);
    const threshold = Math.max(o.minScore, o.nullFactor * nullMax);
    for (const d of dets) {
      d.margin = d.score / Math.max(nullMax, 1);
      d.accepted = d.plausible && d.score >= threshold;
    }
    const kept = attributeHarmonics(dets, threshold, o.c);
    return { detections: kept, nullMax, threshold, nullCount: nullDets.filter((d) => d.plausible).length };
  };

  E.detectInBox = function (spec, z, box, opts) {
    const o = Object.assign({}, E.detectDefaults, opts, { minDur: 4 });
    const { nF, times } = spec;
    const rowFwd = (i) => z.subarray(i * nF, (i + 1) * nF);
    let i0 = 0, i1 = spec.nT;
    while (i0 < spec.nT && times[i0] < box.t0) i0++;
    while (i1 > 0 && times[i1 - 1] > box.t1) i1--;
    const res = trackInBox(spec, z, z, rowFwd, i0, i1, box.f0, box.f1, o);
    if (!res || !res.det) return { error: 'No steady tone was traced in that box. Draw it around one falling line, at least 4 s long.' };
    return { detection: res.det };
  };

  // Ridges with the same speed, range and closest-approach time are one aircraft (its harmonics).
  E.groupPasses = function (dets) {
    const groups = [];
    for (const d of [...dets].sort((a, b) => (a.harmonicOf ? 1 : 0) - (b.harmonicOf ? 1 : 0))) {
      if (d.harmonicOf) {
        const g = groups.find((g) => g.includes(d.harmonicOf));
        if (g) { g.push(d); continue; }
      }
      const g = groups.find((g) => {
        const r = g[0];
        const sv = Math.hypot(r.se.v || 0, d.se.v || 0), sd = Math.hypot(r.se.d || 0, d.se.d || 0), st = Math.hypot(r.se.tca || 0, d.se.tca || 0);
        return Math.abs(r.v - d.v) < 0.08 * r.v + 2 * sv && Math.abs(r.d - d.d) < 0.15 * r.d + 2 * sd && Math.abs(r.tcaRecv - d.tcaRecv) < 1.5 + 2 * st;
      });
      if (g) g.push(d); else groups.push([d]);
    }
    return groups
      .map((ridges) => {
        ridges.sort((a, b) => (a.harmonicOf ? 1 : 0) - (b.harmonicOf ? 1 : 0) || b.score - a.score);
        const lead = ridges[0];
        const vs = ridges.map((r) => r.v);
        return {
          ridges, v: lead.v, d: lead.d, tcaRecv: lead.tcaRecv, f0: lead.f0, score: lead.score,
          vSpread: ridges.length > 1 ? (Math.max(...vs) - Math.min(...vs)) / lead.v : null,
          // ADS-B, altitude and atmosphere errors are correlated along a whole pass, so one pass
          // counts as at most ~10 independent measurements when sizing the confidence region
          dRel: lead.dRel, vRel: lead.vRel, neff: Math.min(10, ridges.reduce((s, r) => s + r.neff, 0)),
          t0: Math.min(...ridges.map((r) => r.t0)), t1: Math.max(...ridges.map((r) => r.t1)),
        };
      })
      .sort((a, b) => a.tcaRecv - b.tcaRecv);
  };

  // ------------------------------------------------------------------ geodesy
  const WA = 6378137.0, WF = 1 / 298.257223563, WE2 = WF * (2 - WF);
  function toECEF(lat, lon, h) {
    const la = lat * DEG, lo = lon * DEG, s = Math.sin(la);
    const N = WA / Math.sqrt(1 - WE2 * s * s);
    return [(N + h) * Math.cos(la) * Math.cos(lo), (N + h) * Math.cos(la) * Math.sin(lo), (N * (1 - WE2) + h) * s];
  }
  function fromECEF(x, y, z) {
    const lon = Math.atan2(y, x), p = Math.hypot(x, y);
    let lat = Math.atan2(z, p * (1 - WE2)), h = 0;
    for (let i = 0; i < 6; i++) {
      const s = Math.sin(lat), N = WA / Math.sqrt(1 - WE2 * s * s);
      h = p / Math.cos(lat) - N;
      lat = Math.atan2(z, p * (1 - WE2 * N / (N + h)));
    }
    return [lat / DEG, lon / DEG, h];
  }
  class LocalFrame {
    constructor(lat0, lon0, h0 = 0) {
      this.lat0 = lat0; this.lon0 = lon0; this.h0 = h0;
      const la = lat0 * DEG, lo = lon0 * DEG;
      this.R = [
        [-Math.sin(lo), Math.cos(lo), 0],
        [-Math.sin(la) * Math.cos(lo), -Math.sin(la) * Math.sin(lo), Math.cos(la)],
        [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)],
      ];
      this.O = toECEF(lat0, lon0, h0);
    }
    toENU(lat, lon, h) {
      const p = toECEF(lat, lon, h), d = [p[0] - this.O[0], p[1] - this.O[1], p[2] - this.O[2]], R = this.R;
      return [R[0][0] * d[0] + R[0][1] * d[1] + R[0][2] * d[2], R[1][0] * d[0] + R[1][1] * d[1] + R[1][2] * d[2], R[2][0] * d[0] + R[2][1] * d[1] + R[2][2] * d[2]];
    }
    toGeo(e, n, u) {
      const R = this.R;
      return fromECEF(
        R[0][0] * e + R[1][0] * n + R[2][0] * u + this.O[0],
        R[0][1] * e + R[1][1] * n + R[2][1] * u + this.O[1],
        R[0][2] * e + R[1][2] * n + R[2][2] * u + this.O[2],
      );
    }
  }
  E.LocalFrame = LocalFrame;

  // ------------------------------------------------------------------ flight data parsing
  function parseTime(v) {
    if (v === undefined || v === null) return NaN;
    const s = String(v).trim();
    if (!s) return NaN;
    if (/^-?\d+(\.\d+)?$/.test(s)) { const x = parseFloat(s); return x > 1e11 ? x / 1000 : x; }
    let iso = s.replace(' ', 'T');
    if (!/[zZ]|[+-]\d\d:?\d\d$/.test(iso)) iso += 'Z';
    const ms = Date.parse(iso);
    return Number.isFinite(ms) ? ms / 1000 : NaN;
  }
  E.parseTime = parseTime;

  function splitCSV(text) {
    const first = text.slice(0, text.indexOf('\n') + 1 || text.length);
    const delim = [',', ';', '\t'].map((d) => [d, first.split(d).length]).sort((a, b) => b[1] - a[1])[0][0];
    const rows = [];
    let row = [], field = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) {
        if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += ch;
      } else if (ch === '"') q = true;
      else if (ch === delim) { row.push(field); field = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(field); field = '';
        if (row.length > 1 || row[0] !== '') rows.push(row);
        row = [];
      } else field += ch;
    }
    if (field !== '' || row.length) { row.push(field); rows.push(row); }
    return rows;
  }

  const COLS = {
    time: ['lastposupdate', 'time', 'timestamp', 't', 'ts', 'unix_time', 'datetime', 'utc', 'date_time'],
    lat: ['lat', 'latitude'], lon: ['lon', 'lng', 'long', 'longitude'],
    altGeoM: ['geoaltitude', 'geo_altitude', 'alt_m', 'altitude_m', 'gnss_altitude_m'],
    altGeoFt: ['alt_geom', 'alt_geom_ft', 'geom_altitude', 'alt_ft_geom'],
    altBaroM: ['baroaltitude', 'baro_altitude'],
    altBaroFt: ['alt_baro', 'altitude', 'alt_ft', 'altitude_ft', 'alt'],
    id: ['icao24', 'icao', 'hex', 'icao_hex', 'mode_s'],
    callsign: ['callsign', 'flight', 'call_sign', 'registration'],
    gsMs: ['velocity', 'groundspeed_ms', 'speed_ms'], gsKt: ['gs', 'groundspeed', 'speed', 'ground_speed', 'speed_kt'],
    trk: ['track', 'heading', 'direction', 'true_track', 'trk', 'course'],
    vrMs: ['vertrate', 'vertical_rate'], vrFpm: ['vert_rate', 'baro_rate', 'geom_rate', 'vertical_rate_fpm', 'vs'],
    onground: ['onground', 'on_ground', 'ground'],
  };

  function pick(header, names) {
    for (const n of names) { const i = header.indexOf(n); if (i >= 0) return i; }
    return -1;
  }

  function finalizeAircraft(map, notes) {
    const out = [];
    for (const [key, a] of map) {
      const idx = a.t.map((_, i) => i).filter((i) => Number.isFinite(a.t[i]) && Number.isFinite(a.lat[i]) && Number.isFinite(a.lon[i]) && Number.isFinite(a.alt[i]));
      idx.sort((i, j) => a.t[i] - a.t[j]);
      const keep = [];
      for (const i of idx) if (!keep.length || a.t[i] > a.t[keep[keep.length - 1]] + 1e-3) keep.push(i);
      if (keep.length < 4) continue;
      const col = (arr) => Float64Array.from(keep, (i) => (arr ? arr[i] : NaN));
      out.push({ key, id: a.id, label: a.label || a.id, altKind: a.altKind, t: col(a.t), lat: col(a.lat), lon: col(a.lon), alt: col(a.alt), gs: col(a.gs), trk: col(a.trk), vr: col(a.vr) });
    }
    if (!out.length) notes.push('No aircraft with at least 4 usable positions.');
    return out;
  }

  function newAc(id, label, altKind) { return { id, label, altKind, t: [], lat: [], lon: [], alt: [], gs: [], trk: [], vr: [] }; }

  function parseCSVFlights(text, altUnits, notes) {
    const rows = splitCSV(text);
    if (rows.length < 2) throw new Error('The CSV has no data rows.');
    const header = rows[0].map((h) => h.trim().toLowerCase().replace(/^﻿/, ''));
    let iT = pick(header, COLS.time);
    const iPos = header.indexOf('position');
    let iLat = pick(header, COLS.lat), iLon = pick(header, COLS.lon);
    if ((iLat < 0 || iLon < 0) && iPos < 0) throw new Error(`No latitude/longitude columns. Found: ${header.join(', ')}`);
    if (iT < 0) throw new Error(`No time column. Found: ${header.join(', ')}`);
    // FlightRadar24 exports have both Timestamp (unix) and UTC; prefer the number
    const iTs = header.indexOf('timestamp');
    if (iTs >= 0) iT = iTs;
    let iAlt, unit, kind;
    const geoM = pick(header, COLS.altGeoM), geoFt = pick(header, COLS.altGeoFt), baroM = pick(header, COLS.altBaroM), baroFt = pick(header, COLS.altBaroFt);
    // the `traffic` library writes geoaltitude in feet next to an `altitude` column
    const trafficStyle = geoM >= 0 && header[geoM] === 'geoaltitude' && header.includes('altitude') && !header.includes('baroaltitude');
    if (geoFt >= 0) [iAlt, unit, kind] = [geoFt, 'ft', 'geometric'];
    else if (geoM >= 0) [iAlt, unit, kind] = [geoM, trafficStyle ? 'ft' : 'm', 'geometric'];
    else if (baroM >= 0) [iAlt, unit, kind] = [baroM, 'm', 'barometric'];
    else if (baroFt >= 0) [iAlt, unit, kind] = [baroFt, 'ft', 'barometric'];
    else throw new Error(`No altitude column. Found: ${header.join(', ')}`);
    if (altUnits === 'ft' || altUnits === 'm') unit = altUnits;
    const iId = pick(header, COLS.id), iCs = pick(header, COLS.callsign);
    const iGsMs = pick(header, COLS.gsMs), iGsKt = pick(header, COLS.gsKt), iTrk = pick(header, COLS.trk);
    const iVrMs = pick(header, COLS.vrMs), iVrF = pick(header, COLS.vrFpm), iGnd = pick(header, COLS.onground);
    const map = new Map();
    let maxAlt = 0;
    for (let r = 1; r < rows.length; r++) {
      const row = rows[r];
      if (iGnd >= 0 && /^(true|1|t)$/i.test((row[iGnd] || '').trim())) continue;
      let lat, lon;
      if (iLat >= 0 && iLon >= 0) { lat = parseFloat(row[iLat]); lon = parseFloat(row[iLon]); }
      else { const p = (row[iPos] || '').split(','); lat = parseFloat(p[0]); lon = parseFloat(p[1]); }
      const altRaw = parseFloat(row[iAlt]);
      if (!Number.isFinite(altRaw)) continue;
      const id = ((iId >= 0 ? row[iId] : '') || (iCs >= 0 ? row[iCs] : '') || 'aircraft').trim().toLowerCase();
      const cs = (iCs >= 0 ? row[iCs] : '').trim();
      if (!map.has(id)) map.set(id, newAc(id, cs, kind));
      const a = map.get(id);
      if (cs && !a.label) a.label = cs;
      a.t.push(parseTime(row[iT])); a.lat.push(lat); a.lon.push(lon);
      a.alt.push(altRaw); maxAlt = Math.max(maxAlt, altRaw);
      a.gs.push(iGsMs >= 0 ? parseFloat(row[iGsMs]) : iGsKt >= 0 ? parseFloat(row[iGsKt]) * KT : NaN);
      a.trk.push(iTrk >= 0 ? parseFloat(row[iTrk]) : NaN);
      a.vr.push(iVrMs >= 0 ? parseFloat(row[iVrMs]) : iVrF >= 0 ? parseFloat(row[iVrF]) * FPM : NaN);
    }
    if (unit === 'm' && altUnits !== 'm' && maxAlt > 16000) { unit = 'ft'; notes.push(`Altitudes up to ${Math.round(maxAlt)} look like feet, so they were read as feet.`); }
    const scale = unit === 'ft' ? FT : 1;
    for (const a of map.values()) a.alt = a.alt.map((v) => v * scale);
    notes.push(`Read ${header[iAlt]} as ${kind} altitude in ${unit === 'ft' ? 'feet' : 'metres'}.`);
    return finalizeAircraft(map, notes);
  }

  function parseJSONFlights(obj, notes) {
    const map = new Map();
    const list = Array.isArray(obj) ? obj : [obj];
    for (const o of list) {
      if (o && Array.isArray(o.trace)) { // readsb / tar1090 trace_full_<hex>.json (adsb.lol, ADS-B Exchange)
        const id = String(o.icao || 'aircraft').toLowerCase();
        const a = newAc(id, '', 'geometric');
        let anyGeom = false;
        for (const p of o.trace) {
          if (!Array.isArray(p) || p[3] === 'ground') continue;
          const geom = p[10], baro = p[3];
          const alt = Number.isFinite(geom) ? geom : Number.isFinite(baro) ? baro : NaN;
          if (Number.isFinite(geom)) anyGeom = true;
          a.t.push(o.timestamp + p[0]); a.lat.push(p[1]); a.lon.push(p[2]); a.alt.push(alt * FT);
          a.gs.push(Number.isFinite(p[4]) ? p[4] * KT : NaN); a.trk.push(Number.isFinite(p[5]) ? p[5] : NaN);
          a.vr.push(Number.isFinite(p[11]) ? p[11] * FPM : Number.isFinite(p[7]) ? p[7] * FPM : NaN);
          if (p[8] && p[8].flight && !a.label) a.label = String(p[8].flight).trim();
        }
        a.altKind = anyGeom ? 'geometric' : 'barometric';
        map.set(id + '#' + map.size, a);
      } else if (o && Array.isArray(o.path)) { // OpenSky /tracks: [time, lat, lon, baro_alt_m, track, on_ground]
        const id = String(o.icao24 || 'aircraft').toLowerCase();
        const a = newAc(id, (o.callsign || '').trim(), 'barometric');
        for (const p of o.path) {
          if (p[5]) continue;
          a.t.push(p[0]); a.lat.push(p[1]); a.lon.push(p[2]); a.alt.push(p[3]);
          a.gs.push(NaN); a.trk.push(Number.isFinite(p[4]) ? p[4] : NaN); a.vr.push(NaN);
        }
        map.set(id + '#' + map.size, a);
      }
    }
    if (!map.size) throw new Error('Unrecognised JSON. Expected a readsb/tar1090 trace (with "trace") or an OpenSky track (with "path").');
    return finalizeAircraft(map, notes);
  }

  function parseKMLFlights(text, notes) {
    const map = new Map();
    const marks = text.split(/<Placemark[\s>]/i).slice(1);
    for (const pm of marks) {
      const whens = [...pm.matchAll(/<when>([^<]+)<\/when>/gi)].map((m) => parseTime(m[1]));
      const coords = [...pm.matchAll(/<gx:coord>([^<]+)<\/gx:coord>/gi)].map((m) => m[1].trim().split(/\s+/).map(Number));
      if (whens.length < 4 || whens.length !== coords.length) continue;
      const nm = (pm.match(/<name>([^<]*)<\/name>/i) || [])[1] || 'aircraft';
      const a = newAc(nm.trim().toLowerCase(), nm.trim(), 'unknown');
      whens.forEach((t, i) => { a.t.push(t); a.lon.push(coords[i][0]); a.lat.push(coords[i][1]); a.alt.push(coords[i][2]); a.gs.push(NaN); a.trk.push(NaN); a.vr.push(NaN); });
      map.set(a.id + '#' + map.size, a);
    }
    if (!map.size) throw new Error('No timed track (<gx:Track> with <when> and <gx:coord>) found in the KML.');
    notes.push('KML altitudes were read as metres.');
    return finalizeAircraft(map, notes);
  }

  E.parseFlights = function (name, text, altUnits = 'auto') {
    const notes = [];
    const s = text.trimStart();
    let aircraft;
    if (/\.kml$/i.test(name) || s.startsWith('<')) aircraft = parseKMLFlights(text, notes);
    else if (s.startsWith('{') || s.startsWith('[')) aircraft = parseJSONFlights(JSON.parse(text), notes);
    else aircraft = parseCSVFlights(text, altUnits, notes);
    if (aircraft.some((a) => a.altKind === 'barometric')) notes.push('Some altitudes are barometric. They can be off by tens of metres, so prefer geometric (GNSS) altitude.');
    return { aircraft, notes };
  };

  // ------------------------------------------------------------------ track preparation
  function whittaker(y, lam) {
    const n = y.length;
    if (n < 5) return Float64Array.from(y);
    const d0 = new Float64Array(n).fill(1), d1 = new Float64Array(n), d2 = new Float64Array(n), c = [1, -2, 1];
    for (let k = 0; k < n - 2; k++) for (let a = 0; a < 3; a++) for (let b = a; b < 3; b++) {
      const v = lam * c[a] * c[b], i = k + a, j = k + b;
      if (j === i) d0[i] += v; else if (j === i + 1) d1[i] += v; else d2[i] += v;
    }
    const l0 = new Float64Array(n), l1 = new Float64Array(n), l2 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      l2[i] = i >= 2 ? d2[i - 2] / l0[i - 2] : 0;
      l1[i] = i >= 1 ? (d1[i - 1] - l2[i] * (i >= 2 ? l1[i - 1] : 0)) / l0[i - 1] : 0;
      l0[i] = Math.sqrt(Math.max(1e-12, d0[i] - l1[i] * l1[i] - l2[i] * l2[i]));
    }
    const u = new Float64Array(n), x = new Float64Array(n);
    for (let i = 0; i < n; i++) u[i] = (y[i] - (i >= 1 ? l1[i] * u[i - 1] : 0) - (i >= 2 ? l2[i] * u[i - 2] : 0)) / l0[i];
    for (let i = n - 1; i >= 0; i--) x[i] = (u[i] - (i + 1 < n ? l1[i + 1] * x[i + 1] : 0) - (i + 2 < n ? l2[i + 2] * x[i + 2] : 0)) / l0[i];
    return x;
  }

  // Returns one or more prepared passes (ENU, 0.25 s grid) inside the crop radius.
  E.prepareTrack = function (ac, frame, opt) {
    const dt = 0.25, n = ac.t.length, cropR = opt.cropRadius, maxGapOpt = opt.maxGap || 20;
    const P = [];
    for (let i = 0; i < n; i++) P.push(frame.toENU(ac.lat[i], ac.lon[i], ac.alt[i]));
    const near = Array.from({ length: n }, (_, i) => Math.hypot(P[i][0], P[i][1]) <= cropR);
    const gaps = [];
    for (let i = 1; i < n; i++) gaps.push(ac.t[i] - ac.t[i - 1]);
    const medDt = gaps.length ? median(gaps) : 1;
    const maxGap = Math.max(maxGapOpt, 4 * medDt);
    // contiguous runs of samples near the area (keeping one sample either side) and without long gaps
    const runs = [];
    let s = -1;
    for (let i = 0; i < n; i++) {
      const inside = near[i] || (i > 0 && near[i - 1]) || (i + 1 < n && near[i + 1]);
      const broken = i > 0 && ac.t[i] - ac.t[i - 1] > maxGap;
      if (inside && s < 0) s = i;
      else if (s >= 0 && (!inside || broken)) { runs.push([s, i]); s = inside ? i : -1; }
    }
    if (s >= 0) runs.push([s, n]);
    const passes = [];
    runs.forEach(([a, b], k) => {
      if (b - a < 3) return;
      const t0 = ac.t[a], t1 = ac.t[b - 1], N = Math.floor((t1 - t0) / dt) + 1;
      if (N < 8) return;
      const tg = new Float64Array(N);
      for (let j = 0; j < N; j++) tg[j] = t0 + j * dt;
      const px = new Float64Array(N), py = new Float64Array(N), pz = new Float64Array(N);
      const vx = new Float64Array(N), vy = new Float64Array(N), vz = new Float64Array(N);
      const dense = medDt <= 3;
      let i = a;
      for (let j = 0; j < N; j++) {
        const t = tg[j];
        while (i < b - 2 && ac.t[i + 1] < t) i++;
        const h = ac.t[i + 1] - ac.t[i], u = Math.min(1, Math.max(0, (t - ac.t[i]) / h));
        const A = P[i], B = P[i + 1];
        if (dense) {
          px[j] = A[0] + (B[0] - A[0]) * u; py[j] = A[1] + (B[1] - A[1]) * u; pz[j] = A[2] + (B[2] - A[2]) * u;
        } else { // cubic Hermite; tangents from reported velocity where available
          const tan = (q) => {
            const gs = ac.gs[q], tr = ac.trk[q];
            if (Number.isFinite(gs) && Number.isFinite(tr)) return [gs * Math.sin(tr * DEG), gs * Math.cos(tr * DEG), Number.isFinite(ac.vr[q]) ? ac.vr[q] : 0];
            const qa = Math.max(a, q - 1), qb = Math.min(b - 1, q + 1);
            const ht = ac.t[qb] - ac.t[qa] || 1;
            return [0, 1, 2].map((c) => (P[qb][c] - P[qa][c]) / ht);
          };
          const mA = tan(i), mB = tan(i + 1);
          const h00 = 2 * u ** 3 - 3 * u ** 2 + 1, h10 = u ** 3 - 2 * u ** 2 + u, h01 = -2 * u ** 3 + 3 * u ** 2, h11 = u ** 3 - u ** 2;
          const d00 = (6 * u * u - 6 * u) / h, d10 = 3 * u * u - 4 * u + 1, d01 = (-6 * u * u + 6 * u) / h, d11 = 3 * u * u - 2 * u;
          const out = [px, py, pz], vel = [vx, vy, vz];
          for (let c = 0; c < 3; c++) {
            out[c][j] = h00 * A[c] + h10 * h * mA[c] + h01 * B[c] + h11 * h * mB[c];
            vel[c][j] = d00 * A[c] + d10 * mA[c] + d01 * B[c] + d11 * mB[c];
          }
        }
      }
      if (dense) { // smooth noisy 1 Hz positions, then differentiate
        const lam = Math.pow(6.4, 4);
        for (const [p, v] of [[px, vx], [py, vy], [pz, vz]]) {
          p.set(whittaker(p, lam));
          for (let j = 0; j < N; j++) v[j] = (p[Math.min(N - 1, j + 1)] - p[Math.max(0, j - 1)]) / (dt * (Math.min(N - 1, j + 1) - Math.max(0, j - 1)));
        }
      }
      let smin = Infinity, smax = 0;
      for (let j = 0; j < N; j += 4) { const sp = Math.hypot(vx[j], vy[j], vz[j]); smin = Math.min(smin, sp); smax = Math.max(smax, sp); }
      passes.push({ id: ac.id, label: ac.label, pass: k, t0, dt, N, px, py, pz, vx, vy, vz, smin, smax });
    });
    return passes;
  };

  E.makeGrid = function (radius, step) {
    const n = Math.round(radius / step), pts = [];
    for (let j = -n; j <= n; j++) for (let i = -n; i <= n; i++) pts.push([i * step, j * step]);
    return pts;
  };

  // ------------------------------------------------------------------ scoring
  // Doppler factor on a uniform receive-time grid for observer X.
  function receiveTable(tr, X, c, wind, dtU) {
    const N = tr.N, trecv = new Float64Array(N), g = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      const dx = X[0] - tr.px[j], dy = X[1] - tr.py[j], dz = X[2] - tr.pz[j];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
      const ux = dx / r, uy = dy / r, uz = dz / r;
      const vt = tr.vx[j] * ux + tr.vy[j] * uy + tr.vz[j] * uz;
      const wu = wind[0] * ux + wind[1] * uy;
      g[j] = (c + wu) / (c - vt + wu);
      trecv[j] = tr.t0 + j * tr.dt + r / (c + wu);
    }
    const T0 = Math.ceil(trecv[0] / dtU) * dtU, M = Math.floor((trecv[N - 1] - T0) / dtU) + 1;
    if (M < 2) return null;
    const G = new Float64Array(M);
    let j = 0;
    for (let k = 0; k < M; k++) {
      const T = T0 + k * dtU;
      while (j < N - 2 && trecv[j + 1] < T) j++;
      const a = (T - trecv[j]) / (trecv[j + 1] - trecv[j]);
      G[k] = g[j] + (g[j + 1] - g[j]) * Math.min(1, Math.max(0, a));
    }
    return { T0, M, G, dtU };
  }

  // Mean squared misfit of one pass (all its ridges; each ridge gets its own best f0).
  function passMisfit(ev, tab, off, sub) {
    let sse = 0, n = 0, tot = 0;
    const { T0, M, G, dtU } = tab;
    for (const rd of ev.ridges) {
      const t = sub ? rd.ts : rd.t, f = sub ? rd.fs : rd.f;
      let Sfg = 0, Sgg = 0, Sff = 0, cnt = 0;
      for (let k = 0; k < t.length; k++) {
        const u = (off + t[k] - T0) / dtU, i = Math.floor(u);
        if (i < 0 || i >= M - 1) continue;
        const g = G[i] + (G[i + 1] - G[i]) * (u - i);
        Sfg += f[k] * g; Sgg += g * g; Sff += f[k] * f[k]; cnt++;
      }
      tot += t.length;
      if (cnt >= 5) { sse += Math.max(0, Sff - Sfg * Sfg / Sgg); n += cnt; }
    }
    return n ? { mse: sse / n, cov: n / tot } : { mse: Infinity, cov: 0 };
  }

  E.prepareEvents = function (events) {
    return events.map((ev) => ({
      v: ev.v, d: ev.d, vRel: ev.vRel, dRel: ev.dRel, neff: ev.neff,
      ridges: ev.ridges.map((r) => {
        const step = Math.max(1, Math.floor(r.t.length / 40));
        const ts = [], fs = [];
        for (let k = 0; k < r.t.length; k += step) { ts.push(r.t[k]); fs.push(r.f[k]); }
        return { t: Float64Array.from(r.t), f: Float64Array.from(r.f), ts: Float64Array.from(ts), fs: Float64Array.from(fs), f0: r.f0 };
      }),
      tMin: Math.min(...ev.ridges.map((r) => r.t[0])), tMax: Math.max(...ev.ridges.map((r) => r.t[r.t.length - 1])),
    }));
  };

  // Score candidate points X (array of [e,n,u]) against prepared passes and events.
  E.scoreCandidates = function (job, progress) {
    const { cands, tracks, events, c, wind, mode, recStart, tol, minCov } = job;
    const dtU = 0.25, nE = events.length, nc = cands.length;
    const rms = new Float64Array(nc).fill(Infinity), off = new Float64Array(nc).fill(NaN);
    const evTrack = new Int32Array(nc * nE).fill(-1), evRms = new Float64Array(nc * nE).fill(NaN);
    const evF0 = new Float64Array(nc * nE).fill(NaN);
    const nTot = events.reduce((s, e) => s + e.ridges.reduce((q, r) => q + r.t.length, 0), 0);
    for (let ci = 0; ci < nc; ci++) {
      const X = cands[ci];
      const usable = []; // per track: table + which events it may explain
      for (let a = 0; a < tracks.length; a++) {
        const tr = tracks[a];
        let dmin = Infinity;
        for (let j = 0; j < tr.N; j += 8) dmin = Math.min(dmin, Math.hypot(X[0] - tr.px[j], X[1] - tr.py[j], X[2] - tr.pz[j]));
        const ok = events.map((e) => {
          const dOk = e.dRel ? dmin < 2.2 * e.d + 400 && dmin > 0.35 * e.d - 400 : dmin < 20000;
          const vOk = e.vRel ? tr.smax >= 0.6 * e.v && tr.smin <= 1.6 * e.v : true;
          return dOk && vOk;
        });
        if (!ok.some(Boolean)) continue;
        const tab = receiveTable(tr, X, c, wind, dtU);
        if (tab) usable.push({ a, tab, ok });
      }
      if (!usable.length) continue;
      const joint = (o, sub, detail) => {
        let total = 0, n = 0;
        for (let e = 0; e < nE; e++) {
          let best = Infinity, ba = -1;
          for (const u of usable) {
            if (!u.ok[e]) continue;
            const m = passMisfit(events[e], u.tab, o, sub);
            if (m.cov >= minCov && m.mse < best) { best = m.mse; ba = u.a; }
          }
          if (!Number.isFinite(best)) return Infinity;
          const ne = events[e].ridges.reduce((q, r) => q + r.t.length, 0);
          total += best * ne; n += ne;
          if (detail) { detail[e] = { a: ba, rms: Math.sqrt(best) }; }
        }
        return total / n;
      };
      let offs;
      const known = mode === 'known' || !!job.hints;
      if (known) {
        const rs = job.hints ? job.hints[ci] : recStart, tl = job.hints ? job.hintTol || 3 : tol;
        offs = [];
        for (let o = -tl; o <= tl + 1e-9; o += 0.25) offs.push(rs + o);
      } else {
        let lo = Infinity, hi = -Infinity;
        for (const u of usable) { lo = Math.min(lo, u.tab.T0); hi = Math.max(hi, u.tab.T0 + u.tab.M * dtU); }
        const tmin = Math.min(...events.map((e) => e.tMin)), tmax = Math.max(...events.map((e) => e.tMax));
        offs = [];
        for (let o = Math.floor(lo - tmax); o <= hi - tmin; o += 1) offs.push(o);
      }
      const sub = !known;
      const vals = offs.map((o) => joint(o, sub));
      // candidate offsets: best few local minima (coarse), then a fine pass and golden-section polish
      const order = vals.map((v, i) => i).filter((i) => Number.isFinite(vals[i])).sort((i, j) => vals[i] - vals[j]);
      if (!order.length) continue;
      const seeds = [];
      for (const i of order) { if (seeds.every((s) => Math.abs(offs[s] - offs[i]) > 3)) seeds.push(i); if (seeds.length >= 3) break; }
      let bestO = NaN, bestV = Infinity;
      for (const s of seeds) {
        const span = sub ? 1.5 : 0.25, stepF = sub ? 0.25 : 0.25;
        let o0 = offs[s], v0 = sub ? joint(o0, false) : vals[s];
        if (sub) for (let o = offs[s] - span; o <= offs[s] + span + 1e-9; o += stepF) { const v = joint(o, false); if (v < v0) { v0 = v; o0 = o; } }
        // golden-section on [o0 - 0.25, o0 + 0.25]
        let A = o0 - 0.25, B = o0 + 0.25;
        const gr = 0.6180339887;
        let x1 = B - gr * (B - A), x2 = A + gr * (B - A), f1 = joint(x1, false), f2 = joint(x2, false);
        for (let it = 0; it < 18; it++) {
          if (f1 < f2) { B = x2; x2 = x1; f2 = f1; x1 = B - gr * (B - A); f1 = joint(x1, false); }
          else { A = x1; x1 = x2; f1 = f2; x2 = A + gr * (B - A); f2 = joint(x2, false); }
        }
        const om = 0.5 * (A + B), vm = joint(om, false);
        if (vm < v0) { v0 = vm; o0 = om; }
        if (v0 < bestV) { bestV = v0; bestO = o0; }
      }
      if (!Number.isFinite(bestV)) continue;
      const detail = [];
      joint(bestO, false, detail);
      rms[ci] = Math.sqrt(bestV); off[ci] = bestO;
      for (let e = 0; e < nE; e++) { evTrack[ci * nE + e] = detail[e].a; evRms[ci * nE + e] = detail[e].rms; }
      if (progress && (ci & 15) === 0) progress(ci / nc);
    }
    return { rms, off, evTrack, evRms, nTot };
  };

  // Predicted curve for one event at observer X and start time `off` (for display).
  E.predictCurve = function (tr, X, off, c, wind, ridge, tGrid) {
    const tab = receiveTable(tr, X, c, wind, 0.25);
    if (!tab) return null;
    const gAt = (t) => { const u = (off + t - tab.T0) / tab.dtU, i = Math.floor(u); return i < 0 || i >= tab.M - 1 ? NaN : tab.G[i] + (tab.G[i + 1] - tab.G[i]) * (u - i); };
    let Sfg = 0, Sgg = 0;
    for (let k = 0; k < ridge.t.length; k++) { const g = gAt(ridge.t[k]); if (Number.isFinite(g)) { Sfg += ridge.f[k] * g; Sgg += g * g; } }
    if (!Sgg) return null;
    const f0 = Sfg / Sgg;
    return { f0, f: tGrid.map((t) => f0 * gAt(t)) };
  };

  // Cells worth a closer look (not clearly worse than the best), each split into a 4x4 sub-grid.
  // Sub-points inherit the parent's best start time, so they are scored in "known time" mode.
  E.refineCandidates = function (pts, rms, off, step, maxCells = 300) {
    let rb = Infinity;
    for (const v of rms) if (v < rb) rb = v;
    if (!Number.isFinite(rb)) return { pts: [], hints: [] };
    const lim = 2.5 * rb + 0.3, idx = [];
    for (let i = 0; i < pts.length; i++) if (rms[i] <= lim) idx.push(i);
    idx.sort((a, b) => rms[a] - rms[b]);
    idx.length = Math.min(idx.length, maxCells);
    const q = [-0.375, -0.125, 0.125, 0.375], out = [], hints = [];
    for (const i of idx) for (const a of q) for (const b of q) { out.push([pts[i][0] + a * step, pts[i][1] + b * step, pts[i][2]]); hints.push(off[i]); }
    return { pts: out, hints, truncated: idx.length === maxCells };
  };

  // ------------------------------------------------------------------ confidence region / clusters
  E.analyze = function (pts, rms, neff, step) {
    const n = pts.length;
    let best = -1, rb = Infinity;
    for (let i = 0; i < n; i++) if (rms[i] < rb) { rb = rms[i]; best = i; }
    if (best < 0) return { best: -1, inRegion: new Uint8Array(n), clusters: [], chi2: new Float64Array(n).fill(Infinity) };
    const sig2 = Math.max(rb, 0.05) ** 2;
    const chi2 = new Float64Array(n), inRegion = new Uint8Array(n);
    for (let i = 0; i < n; i++) { chi2[i] = Number.isFinite(rms[i]) ? neff * (rms[i] ** 2 - rb ** 2) / sig2 : Infinity; inRegion[i] = chi2[i] <= 5.99 ? 1 : 0; }
    // connected components of the region (spatial hash, neighbours within 1.5 steps)
    const cell = 1.5 * step, hash = new Map(), idx = [];
    for (let i = 0; i < n; i++) if (inRegion[i]) {
      idx.push(i);
      const k = Math.floor(pts[i][0] / cell) + ',' + Math.floor(pts[i][1] / cell);
      if (!hash.has(k)) hash.set(k, []);
      hash.get(k).push(i);
    }
    const parent = new Map(idx.map((i) => [i, i]));
    const find = (i) => { while (parent.get(i) !== i) { parent.set(i, parent.get(parent.get(i))); i = parent.get(i); } return i; };
    for (const i of idx) {
      const cx = Math.floor(pts[i][0] / cell), cy = Math.floor(pts[i][1] / cell);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        for (const j of hash.get(cx + dx + ',' + (cy + dy)) || []) {
          if (j <= i) continue;
          if (Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) <= cell) { const a = find(i), b = find(j); if (a !== b) parent.set(a, b); }
        }
      }
    }
    const groups = new Map();
    for (const i of idx) { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(i); }
    const clusters = [...groups.values()].map((m) => {
      let bi = m[0];
      for (const i of m) if (rms[i] < rms[bi]) bi = i;
      let mx = 0, my = 0;
      for (const i of m) { mx += pts[i][0]; my += pts[i][1]; }
      mx /= m.length; my /= m.length;
      let sxx = 0, syy = 0, sxy = 0;
      for (const i of m) { const dx = pts[i][0] - mx, dy = pts[i][1] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
      sxx /= m.length; syy /= m.length; sxy /= m.length;
      const tr = sxx + syy, det = sxx * syy - sxy * sxy, l1 = tr / 2 + Math.sqrt(Math.max(0, tr * tr / 4 - det)), l2 = Math.max(0, tr / 2 - Math.sqrt(Math.max(0, tr * tr / 4 - det)));
      const major = 4 * Math.sqrt(l1) + step, minor = 4 * Math.sqrt(l2) + step; // ~full extent of a uniform blob, plus one cell
      return { members: m, best: bi, rms: rms[bi], area: m.length * step * step, major, minor, cx: mx, cy: my };
    }).sort((a, b) => a.rms - b.rms);
    return { best, rmsBest: rb, chi2, inRegion, clusters };
  };

  // ------------------------------------------------------------------ synthetic demo
  E.makeDemo = function (seed = 7, opts = {}) {
    const R = rng(seed), sr = 2000, dur = 120, n = sr * dur;
    const tempC = 15, c = E.soundSpeed(tempC);
    const lat0 = 45.0, lon0 = 7.0, ground = 300, frame = new LocalFrame(lat0, lon0, ground);
    const recStart = 1725827820; // 2024-09-08 20:37:00 UTC
    const mic = [0, 0, 1.5];
    const planes = [
      { id: 'a1b2c3', cs: 'DEMO11', f0: 110, harm: [1, 0.5, 0.25], amp: 1.0, v: 80, agl: 900, hdg: 60, lateral: 520, tcaRel: 35 },
      { id: 'c0ffee', cs: 'DEMO22', f0: 320, harm: [1, 0.35], amp: 0.8, v: 130, agl: 1400, hdg: 160, lateral: -800, tcaRel: 82 },
    ];
    // pink background, steady hum, then the two engine tones (emitted waveform delayed by range / c)
    let N2 = 1;
    while (N2 < n) N2 <<= 1;
    const re = new Float64Array(N2), im = new Float64Array(N2);
    for (let i = 0; i < N2; i++) re[i] = R.normal();
    fft(re, im);
    for (let k = 1; k < N2; k++) { const fk = Math.min(k, N2 - k) * sr / N2, s = 1 / Math.sqrt(Math.max(fk, 1)); re[k] *= s; im[k] *= s; }
    re[0] = im[0] = 0;
    for (let i = 0; i < N2; i++) im[i] = -im[i];
    fft(re, im); // inverse via conjugation
    let ss = 0;
    for (let i = 0; i < n; i++) ss += re[i] * re[i];
    const nscale = 2.2 / Math.sqrt(ss / n);
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = re[i] * nscale + 0.3 * Math.cos(2 * Math.PI * 120 * i / sr);
    const geo = planes.map((p) => {
      const h = p.hdg * DEG, along = [Math.sin(h), Math.cos(h), 0], side = [Math.cos(h), -Math.sin(h), 0];
      const cpa = [mic[0] + p.lateral * side[0], mic[1] + p.lateral * side[1], p.agl];
      const d = Math.hypot(cpa[0] - mic[0], cpa[1] - mic[1], cpa[2] - mic[2]);
      return { ...p, along, cpa, d, tca: recStart + p.tcaRel };
    });
    for (const p of opts.noPlanes ? [] : geo.filter((q, k) => !opts.only || opts.only.includes(k))) {
      for (let i = 0; i < n; i++) {
        const tr = recStart + i / sr, te = E.spEmission(tr, p.v, p.d, p.tca, c), tau = te - p.tca;
        const r = Math.sqrt(p.d * p.d + p.v * p.v * tau * tau), ph = 2 * Math.PI * p.f0 * tau;
        let s = 0;
        p.harm.forEach((a, k) => { s += a * Math.cos((k + 1) * ph + k); });
        x[i] += p.amp * s * (1 + 0.15 * Math.sin(2 * Math.PI * 0.37 * tau)) * p.d / r;
      }
    }
    const decoy = { id: 'd4e5f6', cs: 'DECOY9', v: 220, agl: 3000, hdg: 300, cpa: [-4500, 3500, 3000], tca: recStart + 40 };
    decoy.along = [Math.sin(decoy.hdg * DEG), Math.cos(decoy.hdg * DEG), 0];
    const lines = ['time,icao24,callsign,lat,lon,alt_geom,gs,track'];
    const rows = [];
    for (const p of [...geo, decoy]) {
      for (let t = recStart - 150; t < recStart + dur + 150; t += 1) {
        const tt = t + (R() - 0.5) * 0.3, s = p.v * (tt - p.tca);
        const e = p.cpa[0] + s * p.along[0] + R.normal() * 3, nn = p.cpa[1] + s * p.along[1] + R.normal() * 3, u = p.cpa[2];
        const [la, lo, hh] = frame.toGeo(e, nn, u);
        rows.push([tt, `${tt.toFixed(2)},${p.id},${p.cs},${la.toFixed(6)},${lo.toFixed(6)},${Math.round(hh / FT / 25) * 25},${Math.round(p.v / KT)},${p.hdg}`]);
      }
    }
    rows.sort((a, b) => a[0] - b[0]);
    for (const r of rows) lines.push(r[1]);
    const [tlat, tlon] = frame.toGeo(0, 0, 0);
    const [clat, clon] = frame.toGeo(700, -500, 0);
    return {
      sr, samples: x, csv: lines.join('\n'), recStart, tempC, groundElev: ground, micHeight: 1.5,
      center: { lat: clat, lon: clon }, truth: { lat: tlat, lon: tlon },
      planes: geo.map((p) => ({ id: p.id, cs: p.cs, f0: p.f0, v: p.v, d: p.d, tcaRecv: p.tcaRel + p.d / c })),
    };
  };

  root.DopplerEngine = E;
  if (typeof module !== 'undefined' && module.exports) module.exports = E;
})(typeof self !== 'undefined' ? self : typeof globalThis !== 'undefined' ? globalThis : this);

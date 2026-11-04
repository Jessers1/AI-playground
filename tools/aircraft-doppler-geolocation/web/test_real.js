// Real-recording check of the direct-match mode:  node test_real.js <audio.f32 @ 8 kHz> <flights.csv>
//   <true lat> <true lon> <ground elevation, ellipsoid m> <logged start, ISO UTC> <± minutes> <search centre lat,lon> <radius m> <step m>
// Parallel over 4 worker threads. Prints where the truth ranks and how far the best match is from it.
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const E = require('./engine.js');
const fs = require('fs');

function build(d) {
  const buf = fs.readFileSync(d.audio);
  const x = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const L = E.guidedSpectrogram(x, 8000, {});
  const frame = new E.LocalFrame(d.cLat, d.cLon, d.elev);
  const { aircraft } = E.parseFlights(d.flights, fs.readFileSync(d.flights, 'utf8'));
  const tracks = aircraft.flatMap((a) => E.prepareTrack(a, frame, { cropRadius: d.radius + 25000 }));
  return { L, frame, tracks };
}
function parallel(d, cands, extra) {
  const W = 4, chunk = Math.ceil(cands.length / W);
  return Promise.all([...Array(W).keys()].map((w) => new Promise((res, rej) => {
    const part = { cands: cands.slice(w * chunk, (w + 1) * chunk), hints: extra.hints ? extra.hints.slice(w * chunk, (w + 1) * chunk) : null };
    const wk = new Worker(__filename, { workerData: { d, part, extra } });
    wk.on('message', res); wk.on('error', rej);
  }))).then((rs) => ({ score: rs.flatMap((r) => Array.from(r.score)), off: rs.flatMap((r) => Array.from(r.off)), trk: rs.flatMap((r) => Array.from(r.trk)), f0a: rs.flatMap((r) => Array.from(r.f0a)), f0b: rs.flatMap((r) => Array.from(r.f0b)) }));
}
if (!isMainThread) {
  const { d, part, extra } = workerData;
  const B = build(d);
  const r = E.guidedScore({ cands: part.cands, tracks: B.tracks, L: B.L, c: E.soundSpeed(d.temp), wind: [0, 0], mode: extra.mode, recStart: d.t, tol: d.tolMin * 60, hints: part.hints, hintTol: 4, bands: d.bands, diff: d.diff });
  parentPort.postMessage(r);
} else {
  const a = process.argv.slice(2);
  const [cLat, cLon] = a[7].split(',').map(Number);
  const d = { audio: a[0], flights: a[1], tLat: +a[2], tLon: +a[3], elev: +a[4], t: Date.parse(a[5]) / 1000, tolMin: +a[6], cLat, cLon, radius: +a[8], step: +a[9], temp: Number(process.env.TEMP || 12), bands: process.env.BANDS ? process.env.BANDS.split(';').map((b) => b.split('-').map(Number)) : undefined, diff: process.env.DIFF === '1' };
  const mode = process.env.MODE || 'known';
  const B = build(d);
  const truth = B.frame.toENU(d.tLat, d.tLon, d.elev);
  const grid = E.makeGrid(d.radius, d.step).map(([e, n]) => [e, n, 1.5]);
  const t0 = Date.now();
  (async () => {
    const coarse = await parallel(d, grid, { mode });
    const fine = E.refineByScore(grid, Float64Array.from(coarse.score), coarse.off, d.step, 60);
    const fr = await parallel(d, fine.pts, { mode: 'known', hints: fine.hints });
    const an = E.analyzeScores(fine.pts, Float64Array.from(fr.score), d.step / 4);
    const b = fine.pts[an.best], [blat, blon] = B.frame.toGeo(b[0], b[1], 0);
    const err = Math.hypot(b[0] - truth[0], b[1] - truth[1]);
    const order = coarse.score.map((s, i) => [s, i]).sort((p, q) => q[0] - p[0]);
    const it = grid.map((g, i) => [Math.hypot(g[0] - truth[0], g[1] - truth[1]), i]).sort((p, q) => p[0] - q[0])[0][1];
    const rank = order.findIndex((o) => o[1] === it) + 1;
    let nearIn = Infinity;
    fine.pts.forEach((p, i) => { if (an.inRegion[i]) nearIn = Math.min(nearIn, Math.hypot(p[0] - truth[0], p[1] - truth[1])); });
    const tr = B.tracks[fr.trk[an.best]];
    console.log(`${grid.length} + ${fine.pts.length} points in ${((Date.now() - t0) / 1000).toFixed(0)} s, start time ${mode === 'known' ? `${a[5]} ± ${d.tolMin} min` : 'unknown'}`);
    console.log(`best match ${blat.toFixed(5)}, ${blon.toFixed(5)}: ${err.toFixed(0)} m from the recordist; aircraft ${tr ? tr.label + ' (' + tr.id + ')' : '-'}; start ${new Date(fr.off[an.best] * 1000).toISOString().slice(11, 21)}Z; tones ${fr.f0a[an.best].toFixed(1)}${Number.isFinite(fr.f0b[an.best]) ? ' & ' + fr.f0b[an.best].toFixed(1) : ''} Hz`);
    console.log(`truth's grid cell ranks ${rank}/${grid.length}; best is ${an.sigmaAbove.toFixed(1)} MAD above the area median; ${an.clusters.length} area(s) in the near-best region; nearest region point to truth ${nearIn.toFixed(0)} m`);
    an.clusters.slice(0, 4).forEach((cl, k) => { const X = fine.pts[cl.best]; console.log(`   area ${k + 1}: ${Math.hypot(X[0] - truth[0], X[1] - truth[1]).toFixed(0)} m from truth, score ${cl.score.toFixed(3)}, ${cl.major.toFixed(0)} x ${cl.minor.toFixed(0)} m, ${B.tracks[fr.trk[cl.best]]?.label}`); });
    if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify({ grid, coarse: coarse.score, fine: fine.pts, fineScore: fr.score, inRegion: Array.from(an.inRegion), truth, best: b, labels: B.tracks.map((t) => t.label), tracks: B.tracks.map((t) => ({ label: t.label, x: Array.from(t.px).filter((_, i) => i % 8 === 0), y: Array.from(t.py).filter((_, i) => i % 8 === 0) })) }));
  })();
}

// Headless end-to-end test of engine.js:  node test_engine.js [seed]
// Exit code 1 if any check fails.
const E = require('./engine.js');

const seed = Number(process.argv[2] || 7);
const checks = [];
const check = (name, ok, info) => { checks.push({ name, ok, info }); console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}  ${info}`); };
const tick = () => { const t = Date.now(); return () => ((Date.now() - t) / 1000).toFixed(1) + ' s'; };

function analyseAudio(samples, sr, c) {
  const spec = E.spectrogram(samples, sr, {});
  const z = E.whiten(spec);
  const det = E.detectFlyovers(spec, z, { c });
  return { spec, z, ...det };
}

function locate(demo, events, mode, extra = {}) {
  const c = E.soundSpeed(demo.tempC);
  const frame = new E.LocalFrame(demo.center.lat, demo.center.lon, demo.groundElev);
  const radius = extra.radius || 4000, step = extra.step || 200;
  const grid = E.makeGrid(radius, step).map(([e, n]) => [e, n, demo.micHeight]);
  const { aircraft } = E.parseFlights('demo.csv', demo.csv);
  const tracks = aircraft.flatMap((a) => E.prepareTrack(a, frame, { cropRadius: radius + 25000 }));
  const evs = E.prepareEvents(events);
  const job = { cands: grid, tracks, events: evs, c, wind: [0, 0], mode, recStart: extra.recStart, tol: extra.tol || 2, minCov: 0.8 };
  const res = E.scoreCandidates(job);
  const fine = E.refineCandidates(grid, res.rms, res.off, step);
  const res2 = E.scoreCandidates(Object.assign({}, job, { cands: fine.pts, hints: fine.hints, hintTol: 3 }));
  const neff = events.reduce((s, e) => s + e.neff, 0);
  const an = E.analyze(fine.pts, res2.rms, neff, step / 4);
  const truth = frame.toENU(demo.truth.lat, demo.truth.lon, demo.groundElev);
  const b = fine.pts[an.best];
  // truth counts as inside when an in-region sub-point lies within one sub-cell diagonal of it
  let near = Infinity;
  fine.pts.forEach((g, i) => { if (an.inRegion[i]) near = Math.min(near, Math.hypot(g[0] - truth[0], g[1] - truth[1])); });
  return { res: res2, an, grid: fine.pts, truth, best: b, err: Math.hypot(b[0] - truth[0], b[1] - truth[1]), tracks, frame, off: res2.off[an.best], truthIn: near <= Math.SQRT2 * step / 4, nFine: fine.pts.length };
}

console.log(`seed ${seed}`);
let t = tick();
const demo = E.makeDemo(seed);
const c = E.soundSpeed(demo.tempC);
console.log(`demo synthesised in ${t()}`);
t = tick();
const A = analyseAudio(demo.samples, demo.sr, c);
console.log(`spectrogram ${A.spec.nT}x${A.spec.nF}, detection in ${t()}; null max ${A.nullMax.toFixed(2)} (${A.nullCount} null ridges), threshold ${A.threshold.toFixed(2)}`);
for (const d of A.detections) console.log(`   ${d.accepted ? 'ACCEPT' : 'reject'} f0 ${d.f0.toFixed(2)} v ${d.v.toFixed(1)}±${d.se.v.toFixed(1)} d ${d.d.toFixed(0)}±${d.se.d.toFixed(0)} cpa ${d.tcaRecv.toFixed(2)} score ${d.score.toFixed(1)} meanZ ${d.meanZ.toFixed(2)} rms ${d.rms.toFixed(2)} ${d.t0.toFixed(0)}-${d.t1.toFixed(0)}s ${d.reasons.join('; ')}`);
const passes = E.groupPasses(A.detections.filter((d) => d.accepted));
console.log(`passes: ${passes.length}`);
for (const p of passes) console.log(`   pass ridges=${p.ridges.length} f0s=${p.ridges.map((r) => r.f0.toFixed(1)).join('/')} v ${p.v.toFixed(1)} d ${p.d.toFixed(0)} cpa ${p.tcaRecv.toFixed(2)} vSpread ${p.vSpread === null ? '-' : (100 * p.vSpread).toFixed(1) + '%'}`);
check('two passes found', passes.length === 2, `${passes.length} passes`);
demo.planes.forEach((pl, k) => {
  const p = passes.find((q) => Math.abs(q.tcaRecv - pl.tcaRecv) < 2);
  const lead = p && p.ridges[0];
  const okV = p && Math.abs(p.v - pl.v) < Math.max(0.03 * pl.v, 3 * lead.se.v), okD = p && Math.abs(p.d - pl.d) < Math.max(0.04 * pl.d, 3 * lead.se.d);
  check(`pass ${k + 1} recovered (within 3% / 4% or 3 sigma)`, !!(okV && okD),
    p ? `v ${p.v.toFixed(1)}±${lead.se.v.toFixed(1)} (true ${pl.v}), d ${p.d.toFixed(0)}±${lead.se.d.toFixed(0)} (true ${pl.d.toFixed(0)}), CPA ${p.tcaRecv.toFixed(2)} (true ${pl.tcaRecv.toFixed(2)}), heard ${lead.t0.toFixed(0)}-${lead.t1.toFixed(0)} s` : 'missing');
});
const harmonicPass = passes.find((p) => p.ridges.length > 1);
check('harmonics grouped into one pass', !!harmonicPass, harmonicPass ? `${harmonicPass.ridges.length} ridges in one pass` : 'no multi-ridge pass');

if (passes.length >= 2) {
  t = tick();
  const both = locate(demo, passes, 'unknown');
  console.log(`unknown time, both passes: scored 1681 + ${both.nFine} points in ${t()}; best err ${both.err.toFixed(0)} m, rms ${both.an.rmsBest.toFixed(3)} Hz, clusters ${both.an.clusters.length}, region ${both.an.clusters.map((q) => q.members.length).join('/')} cells, start ${(both.off - demo.recStart).toFixed(2)} s`);
  check('unknown time + 2 passes: rank 1 within 150 m', both.err <= 150, `${both.err.toFixed(0)} m`);
  check('unknown time + 2 passes: truth inside 95% region', both.truthIn, '');
  check('unknown time + 2 passes: start time within 1 s', Math.abs(both.off - demo.recStart) < 1, `${(both.off - demo.recStart).toFixed(2)} s`);
  check('unknown time + 2 passes: single cluster', both.an.clusters.length === 1, `${both.an.clusters.length} clusters`);

  t = tick();
  const one = locate(demo, [passes[0]], 'unknown');
  const c0 = one.an.clusters[0];
  console.log(`unknown time, pass 1 only: ${t()}; clusters ${one.an.clusters.length}; biggest major ${c0.major.toFixed(0)} m minor ${c0.minor.toFixed(0)} m`);
  check('unknown time + 1 pass: reported as ambiguous', one.an.clusters.length > 1 || c0.major > 4 * c0.minor, `${one.an.clusters.length} clusters, elongation ${(c0.major / c0.minor).toFixed(1)}`);

  t = tick();
  const known = locate(demo, [passes[0]], 'known', { recStart: demo.recStart + 1.3, tol: 2 });
  console.log(`known time, pass 1 only: ${t()}; best err ${known.err.toFixed(0)} m; clusters ${known.an.clusters.length}`);
  check('known time + 1 pass: truth inside 95% region', known.truthIn, `best ${known.err.toFixed(0)} m from truth, ${known.an.clusters.length} cluster(s)`);
}

t = tick();
const nz = analyseAudio(E.makeDemo(seed + 100, { noPlanes: true }).samples, demo.sr, c);
check('noise + hum only: nothing accepted', nz.detections.every((d) => !d.accepted), `${nz.detections.filter((d) => d.accepted).length} accepted, null max ${nz.nullMax.toFixed(2)}, best raw score ${Math.max(0, ...nz.detections.map((d) => d.score)).toFixed(2)} (${t()})`);

const failed = checks.filter((c) => !c.ok).length;
console.log(failed ? `${failed} check(s) failed` : 'all checks passed');
process.exit(failed ? 1 : 0);

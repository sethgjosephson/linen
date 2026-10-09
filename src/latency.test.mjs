// The timing side of binding: does one sense drive the conjunctive cells in the order the pair drives them?
// Planted both ways, because a measure for something never observed has to be shown to move when the effect is there and to sit at zero when it is not.
import { latencyAgreement, ItemRecorder } from './analysis.js';

let fail = 0;
const ok = (n, c, x) => { console.log((c?'ok   ':'FAIL ')+n+(x?'  '+x:'')); if(!c) fail++; };

const NI = 6, NN = 200, K = 12;
const sets = Array.from({ length:NI }, (_, i) =>
  Array.from({ length:K }, (_, k) => i*K + k));

let z = 987654321;
const rnd = () => { z = (z*1664525 + 1013904223) >>> 0; return z/4294967296; };

// each conjunctive cell has its own characteristic latency under the pair
const trueLat = new Float64Array(NN);
for(let j=0;j<NN;j++) trueLat[j] = 5 + rnd()*30;

// copy is how faithfully the probe reproduces that order; noise is jitter
const mk = (item, copy, noise, reps = 6) => {
  const out = [];
  for(let r=0;r<reps;r++){
    const lat = new Float32Array(NN).fill(-1);
    for(const j of sets[item])
      lat[j] = copy*trueLat[j] + (1-copy)*(5 + rnd()*30) + (rnd()-0.5)*noise;
    out.push({ label:item, lat });
  }
  return out;
};
const build = (copy, noise) => {
  const both = [], probe = [];
  for(let i=0;i<NI;i++){
    both.push(...mk(i, 1, noise));
    probe.push(...mk(i, copy, noise));
  }
  return { both, probe };
};

{
  const { both, probe } = build(0, 2);        // probe timing unrelated
  const r = latencyAgreement(sets, both, probe, NI, NN);
  ok('unrelated timing reads about zero', Math.abs(r.r) < 0.25,
    `r ${r.r} over ${r.items} items, ${r.cells} cells each`);
}
{
  const { both, probe } = build(1, 2);        // probe reproduces the order
  const r = latencyAgreement(sets, both, probe, NI, NN);
  ok('reproduced timing reads clearly positive', r.r > 0.7, `r ${r.r}`);
}
{
  // ordered, not banded: guessing the value of a partial copy is guessing the sampling distribution of a correlation over twelve cells, and the property that matters is that more copying reads higher
  const un = latencyAgreement(sets, ...Object.values(build(0, 2)), NI, NN);
  const half = latencyAgreement(sets, ...Object.values(build(0.5, 2)), NI, NN);
  const full = latencyAgreement(sets, ...Object.values(build(1, 2)), NI, NN);
  ok('more reproduction reads higher', un.r < half.r && half.r <= full.r,
    `unrelated ${un.r}, half ${half.r}, full ${full.r}`);
}
{
  // a cell that never fires under the probe cannot be compared, and must not be silently counted as agreement
  const { both, probe } = build(1, 2);
  for(const t of probe) t.lat.fill(-1);
  const r = latencyAgreement(sets, both, probe, NI, NN);
  ok('silent probe yields no measurement rather than a number', r === null,
    r ? `got r ${r.r}` : 'null');
}

// The recorder: a trial's latency is each cell's first spike from the presentation's onset, taken once, not the within-tick step averaged over every tick the cell fired in.
// Three cells over four 50 ms ticks: cell 0 first fires 7 ms into tick 0 and again later, cell 1 first fires 12 ms into tick 2, cell 2 never fires.
{
  const rec = new ItemRecorder(Uint32Array.from([0, 1, 2]), new Map([['a', 0]]), 50);
  const tick = (key, f, l) => rec.sample(Uint8Array.from(f), key, 0, 0, Uint8Array.from(l));
  tick('a#1', [1, 0, 0], [7, 255, 255]);
  tick('a#1', [1, 0, 0], [40, 255, 255]);
  tick('a#1', [2, 1, 0], [3, 12, 255]);
  tick('a#1', [0, 1, 0], [255, 30, 255]);
  rec.flush();
  const lat = rec.trials[0] && rec.trials[0].lat;
  ok('recorder latency is the first spike from onset', lat && lat[0] === 7 && lat[1] === 112 && lat[2] === -1,
    lat ? Array.from(lat).join(' ') : 'no lat');
  // the next presentation starts its own clock
  tick('a#2', [0, 1, 0], [255, 5, 255]);
  tick('a#2', [1, 0, 0], [9, 255, 255]);
  rec.flush();
  const l2 = rec.trials[1] && rec.trials[1].lat;
  ok('each presentation measures from its own onset', l2 && l2[1] === 5 && l2[0] === 59 && l2[2] === -1,
    l2 ? Array.from(l2).join(' ') : 'no lat');
}

console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);

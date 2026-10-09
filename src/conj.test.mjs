// The conjunctive binding metric, against data where the answer is known.
//
// A metric for something that has never been observed has to be shown to move when the effect is planted and to stay at zero when it is not, or a null result says nothing about the network and everything about the measurement.
import { conjunctiveSets, conjunctiveRecall } from './analysis.js';

let fail = 0;
const ok = (name, cond, extra) => {
  console.log((cond ? 'ok   ' : 'FAIL ') + name + (extra ? '  ' + extra : ''));
  if(!cond) fail++;
};

const NI = 8, NN = 600, K = 20;   // NN must exceed NI*3*K or the items overlap

// Layout per item i: a visual block, an audio block, and a conjunctive block that only the pair drives.
// Blocks never overlap between items.
const vCells = i => Array.from({ length:K }, (_, k) => (i*3*K) + k);
const aCells = i => Array.from({ length:K }, (_, k) => (i*3*K) + K + k);
const cCells = i => Array.from({ length:K }, (_, k) => (i*3*K) + 2*K + k);

// leak lets a probe partially drive the conjunctive cells: 0 is no binding
function build(leak){
  const both = new Float32Array(NI*NN);
  const vis = new Float32Array(NI*NN);
  const aud = new Float32Array(NI*NN);
  for(let i=0;i<NI;i++){
    const b = i*NN;
    for(const j of vCells(i)){ both[b+j] = 1; vis[b+j] = 1; }
    for(const j of aCells(i)){ both[b+j] = 1; aud[b+j] = 1; }
    for(const j of cCells(i)){ both[b+j] = 1; vis[b+j] = leak; aud[b+j] = leak; }
  }
  return { both, vis, aud };
}
const cnt = new Float32Array(NI).fill(4);

{
  const { both, vis, aud } = build(0);
  const sets = conjunctiveSets(both, vis, aud, NI, NN, 0.2);
  const exact = sets.every((s, i) => {
    const want = new Set(cCells(i));
    return s.length === K && s.every(j => want.has(j));
  });
  ok('conjunctive sets are exactly the cells neither modality drives', exact,
    `sizes ${sets.map(s => s.length).join(',')}`);
}

{
  const { both, vis, aud } = build(0);
  const sets = conjunctiveSets(both, vis, aud, NI, NN, 0.2);
  const r = conjunctiveRecall(sets, vis, cnt, NI, NN);
  ok('no binding reads about zero', Math.abs(r.bind) < 0.02,
    `bind ${r.bind} own ${r.own} other ${r.other}`);
}

{
  const { both, vis, aud } = build(0.5);
  // sets must come from the unbound state, as calibration would give them
  const cal = build(0);
  const sets = conjunctiveSets(cal.both, cal.vis, cal.aud, NI, NN, 0.2);
  const r = conjunctiveRecall(sets, vis, cnt, NI, NN);
  ok('planted binding reads clearly positive', r.bind > 0.3,
    `bind ${r.bind} own ${r.own} other ${r.other}`);
}

// the drift control: scaling every response leaves the answer alone, which is the reason the measure is in rank units
{
  const cal = build(0);
  const sets = conjunctiveSets(cal.both, cal.vis, cal.aud, NI, NN, 0.2);
  const a = build(0.5);
  const scaled = new Float32Array(a.vis.length);
  for(let i=0;i<a.vis.length;i++) scaled[i] = a.vis[i]*7.5 + 3;   // gain and offset
  const r1 = conjunctiveRecall(sets, a.vis, cnt, NI, NN);
  const r2 = conjunctiveRecall(sets, scaled, cnt, NI, NN);
  ok('immune to a global gain and offset', Math.abs(r1.bind - r2.bind) < 1e-6,
    `${r1.bind} vs ${r2.bind}`);
}

// a probe that drives every item's conjunction equally is arousal, not recognition
{
  const cal = build(0);
  const sets = conjunctiveSets(cal.both, cal.vis, cal.aud, NI, NN, 0.2);
  const { vis } = build(0);
  for(let i=0;i<NI;i++)
    for(let j=0;j<NI;j++)
      for(const c of cCells(j)) vis[i*NN + c] = 0.5;   // all conjunctions, all items
  const r = conjunctiveRecall(sets, vis, cnt, NI, NN);
  ok('non-specific drive does not count as binding', Math.abs(r.bind) < 0.02,
    `bind ${r.bind}`);
}

// A limitation of the measure: for a cell with no signal the three conditions are independent noise, so a rank cut alone places some cells in the top of one and not the others, more often as the firing rate rises, and a unimodal layer then shows a conjunction it cannot have.
// A margin on the rank gap cannot close it: tie-averaging puts the silent majority near rank 0.5, so a genuinely conjunctive cell reaches a gap of about 0.58 while suppressing noise needs more than 0.8.
// The calibration sets, taken while the network is quiet, and the binding score against those fixed sets are sound; comparing a later set against the calibration one confounds drift with rate.

console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);

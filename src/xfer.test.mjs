// The cross-modal learning measures against a world whose answer is known.
//
// Two drive sets, A (sight) and B (sound), each item driving its own cells in its own set.
// The input geometry is built in as the confound the weaves have: presenting item k's sight also puts a little activity on the sound cells of a different item, sigma(k), through the wiring as wired, before anything is learned.
// That is what makes the plain cross-decoder read below chance on a fixed confusion map with plasticity off.
// A learned association is then planted: item k's sight reaching item k's own sound cells.
// The other measures have to read nothing on the geometry alone and see the planted association; the plain decoder is checked to be fooled, so the test says why the other ones exist.
import { itemCellSets, evokedTransfer, itemCoupling, crossDecode } from './analysis.js';
import { ok, report } from '../tools/harness.mjs';

let seed = 12345;
const rnd = () => { seed = (seed*1103515245 + 12345) >>> 0; return seed/4294967296; };
const nN = 240, nI = 6, A0 = 0, A1 = 90, B0 = 90, B1 = 180;   // 180..239 in neither set
const pick = (lo, hi, n) => { const s = new Set(); while(s.size < n) s.add(lo + Math.floor(rnd()*(hi - lo))); return [...s]; };
const cellsA = Array.from({ length:nI }, () => pick(A0, A1, 14));
const cellsB = Array.from({ length:nI }, () => pick(B0, B1, 14));
const sigma = [2, 3, 4, 5, 0, 1];                      // the geometry: k's sight leaks onto sigma(k)'s sound cells
const inA = Uint8Array.from({ length:nN }, (_, j) => j >= A0 && j < A1 ? 1 : 0);
const inB = Uint8Array.from({ length:nN }, (_, j) => j >= B0 && j < B1 ? 1 : 0);

// a trial: counts per cell for one presentation
function trial(cond, k, learned){
  const v = new Float32Array(nN);
  for(let j=0;j<nN;j++) v[j] = rnd() < 0.05 ? 1 : 0;      // background
  if(cond === 1){                                          // sight alone
    for(const j of cellsA[k]) v[j] += 6 + Math.floor(rnd()*3);
    for(const j of cellsB[sigma[k]]) v[j] += rnd() < 0.5 ? 1 : 0;   // geometry
    if(learned) for(const j of cellsB[k]) v[j] += rnd() < 0.6 ? 2 : 0;
  } else if(cond === 2){                                   // sound alone
    for(const j of cellsB[k]) v[j] += 6 + Math.floor(rnd()*3);
    for(const j of cellsA[sigma[k]]) v[j] += rnd() < 0.5 ? 1 : 0;
    if(learned) for(const j of cellsA[k]) v[j] += rnd() < 0.6 ? 2 : 0;
  }
  return v;
}
function pack(cond, learned, per = 20){
  const n = nI*per, trials = new Float32Array(n*nN), labels = new Int32Array(n);
  let t = 0;
  for(let r=0;r<per;r++) for(let k=0;k<nI;k++){ trials.set(trial(cond, k, learned), t*nN); labels[t] = k; t++; }
  return { trials, labels, n };
}
function centroids(p){
  const cent = new Float32Array(nI*nN), cnt = new Float32Array(nI);
  for(let t=0;t<p.n;t++){ const k = p.labels[t]; cnt[k]++; for(let j=0;j<nN;j++) cent[k*nN + j] += p.trials[t*nN + j]; }
  for(let k=0;k<nI;k++) for(let j=0;j<nN;j++) cent[k*nN + j] /= cnt[k] || 1;
  return { cent, cnt };
}

// the calibration: before learning, sets from each sense alone
const cal1 = pack(1, false), cal2 = pack(2, false);
const c1 = centroids(cal1), c2 = centroids(cal2);
const SA = itemCellSets(c1.cent, c1.cnt, nI, nN, inA, 0.16);
const SB = itemCellSets(c2.cent, c2.cnt, nI, nN, inB, 0.16);
const hit = (set, truth) => set.filter(j => truth.includes(j)).length/set.length;
ok('item sets find each item\'s own cells in its own drive set',
  SA.every((s, k) => hit(s, cellsA[k]) > 0.9) && SB.every((s, k) => hit(s, cellsB[k]) > 0.9),
  SA.map((s, k) => hit(s, cellsA[k]).toFixed(2)).join(' ') + ' | ' + SB.map((s, k) => hit(s, cellsB[k]).toFixed(2)).join(' '));
ok('item sets never take a cell from the other drive set',
  SA.every(s => s.every(j => inA[j])) && SB.every(s => s.every(j => inB[j])));

// the plain decoder is fooled by the geometry: below chance on a fixed partner
const unl1 = pack(1, false), unl2 = pack(2, false);
const old = crossDecode(unl1, unl2, nI, nN, 0);
ok('the plain cross-decoder reads the geometry as a consistent wrong partner with nothing learned',
  old.ba && old.ba.acc < 1/nI && old.ba.shape && old.ba.shape.partners.every((p, k) => p >= 0 && p !== k),
  'sound trained, sight tested: ' + JSON.stringify({ acc:old.ba && old.ba.acc, partners:old.ba && old.ba.shape && old.ba.shape.partners }));

// the evoked measure: at baseline it carries the geometry (negative, since the leak goes to another item's cells), and without learning it stays there
const base = evokedTransfer(cal1, SB, nI, nN);
const later = evokedTransfer(unl1, SB, nI, nN);
ok('the evoked transfer baseline carries the geometry as a negative own-minus-other',
  base.d < 0, JSON.stringify(base));
ok('with nothing learned, the evoked transfer stays at its baseline',
  Math.abs(later.d - base.d) < 3*Math.hypot(base.se, later.se) + 0.02, 'base ' + base.d + ' later ' + later.d);
const lrn = evokedTransfer(pack(1, true), SB, nI, nN);
ok('a planted association raises the evoked transfer well above its baseline',
  lrn.d - base.d > 0.5 && lrn.d - base.d > 5*Math.hypot(base.se, lrn.se), 'base ' + base.d + ' learned ' + lrn.d);
const lrnBA = evokedTransfer(pack(2, true), SA, nI, nN), baseBA = evokedTransfer(cal2, SA, nI, nN);
ok('and in the other direction', lrnBA.d - baseBA.d > 0.5, 'base ' + baseBA.d + ' learned ' + lrnBA.d);

// the weight measure: synapses between the two codes, dense enough to cover every item pair
const s = [], pre = [], post = [], dir = [];
let q = 0;
for(let i=A0;i<A1;i++) for(let j=B0;j<B1;j++) if(rnd() < 0.3){ s.push(q++); pre.push(i); post.push(j); dir.push(1); }
for(let i=B0;i<B1;i++) for(let j=A0;j<A1;j++) if(rnd() < 0.3){ s.push(q++); pre.push(i); post.push(j); dir.push(2); }
const between = { s:Int32Array.from(s), pre:Int32Array.from(pre), post:Int32Array.from(post), dir:Uint8Array.from(dir) };
const w0 = Float32Array.from({ length:q }, () => 0.2 + rnd()*0.6);
const same = itemCoupling(between, SA, SB, w0.slice(), w0);
ok('with no weight changed the coupling is exactly zero in both directions',
  same.ab && same.ba && same.ab.d === 0 && same.ba.d === 0, JSON.stringify(same));
// generic change: every between synapse moves by the same random amount law, with no regard to items (what a scrambled pairing can write at most)
const wGen = w0.map(x => x + (rnd() - 0.4)*0.05);
const gen = itemCoupling(between, SA, SB, wGen, w0);
ok('item-blind weight change leaves own and other alike',
  Math.abs(gen.ab.d) < 0.004 && Math.abs(gen.ba.d) < 0.004, JSON.stringify(gen));
// planted: synapses from item k's sight cells onto item k's sound cells grow
const trueA = new Map(), trueB = new Map();
cellsA.forEach((cs, k) => cs.forEach(c => trueA.set(c, k)));
cellsB.forEach((cs, k) => cs.forEach(c => trueB.set(c, k)));
const wLrn = w0.slice();
for(let r=0;r<q;r++){
  const a = dir[r] === 1 ? trueA.get(pre[r]) : trueB.get(pre[r]);
  const b = dir[r] === 1 ? trueB.get(post[r]) : trueA.get(post[r]);
  if(a !== undefined && a === b) wLrn[r] += 0.1;
}
const lc = itemCoupling(between, SA, SB, wLrn, w0);
ok('a planted pairing makes own exceed other in both directions, by far more than item-blind change does',
  lc.ab.d > 0.02 && lc.ba.d > 0.02 && lc.ab.d > 10*Math.abs(gen.ab.d) && lc.ba.d > 10*Math.abs(gen.ba.d), JSON.stringify(lc));
// the geometry's partner is not the learned one: planting sigma's pairs shows up as other, not own
const wSig = w0.slice();
for(let r=0;r<q;r++){
  const a = dir[r] === 1 ? trueA.get(pre[r]) : undefined;
  const b = dir[r] === 1 ? trueB.get(post[r]) : undefined;
  if(a !== undefined && b !== undefined && sigma[a] === b) wSig[r] += 0.1;
}
const sc = itemCoupling(between, SA, SB, wSig, w0);
ok('growth onto another item\'s cells counts against own, not for it', sc.ab.d < 0, JSON.stringify(sc.ab));
report('xfer');

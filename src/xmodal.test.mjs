// The field's cross-modal measures (ANALYSIS.md section 12) on synthetic data with a known answer: shared and unshared representations, so each measure has to separate the two.
// Run: node src/xmodal.test.mjs
import assert from 'node:assert/strict';
import { rng } from './rand.js';
import { multisensoryIndex, crossDecode, rdm, rdmCorrelation, crossRdm,
  populationSparseness, blockStructure, decodeAccuracy,
  pairClasses, setWeightStats, confusionShape } from './analysis.js';

const nI = 5, nN = 40, PER = 16;
const r = rng(7);
// item i drives cells 6i..6i+7 under a map, so neighboring items overlap by two cells and the items have a geometry (i is nearer i+1 than i+3); the map is identity for a shared code and a derangement for an unshared one
const make = (map, noise = 0.6) => {
  const trials = new Float32Array(nI*PER*nN), labels = new Int32Array(nI*PER);
  let t = 0;
  for(let i=0;i<nI;i++) for(let k=0;k<PER;k++, t++){
    labels[t] = i;
    const m = map[i];
    for(let j=0;j<nN;j++) trials[t*nN + j] = r()*noise + (j >= 6*m && j < 6*m + 8 ? 3 + r() : 0);
  }
  return { trials, labels, n:t };
};
const cent = pack => {
  const c = new Float32Array(nI*nN), cnt = new Float32Array(nI);
  for(let t=0;t<pack.n;t++){ const l = pack.labels[t]; cnt[l]++;
    for(let j=0;j<nN;j++) c[l*nN + j] += pack.trials[t*nN + j]; }
  for(let i=0;i<nI;i++) for(let j=0;j<nN;j++) c[i*nN + j] /= cnt[i];
  return { cent:c, cnt };
};
const ident = [0,1,2,3,4], derange = [1,2,3,4,0];
const A = make(ident), B = make(ident), Bx = make(derange);

// cross-classification separates a shared code from an unshared one
{
  const s = crossDecode(A, B, nI, nN, 40, 3);
  assert.ok(s.ab.acc > 0.9 && s.ba.acc > 0.9, 'shared code cross-decodes: ' + JSON.stringify(s));
  assert.ok(s.ab.p <= 0.05, 'shared code p small: ' + s.ab.p);
  assert.ok(s.withinA.acc > 0.9, 'within-modality decodes');
  const u = crossDecode(A, Bx, nI, nN, 40, 3);
  assert.ok(u.ab.acc < 0.4, 'unshared code stays near chance: ' + u.ab.acc);
  assert.ok(u.ab.p > 0.2, 'unshared code p not small: ' + u.ab.p);
  assert.ok(u.withinA.acc > 0.9, 'within does not imply across');
  console.log('ok  crossDecode: shared ab ' + s.ab.acc + ' p ' + s.ab.p + ', unshared ab ' + u.ab.acc + ' p ' + u.ab.p);
}

// representational similarity: identical geometry correlates, deranged does not
{
  const cA = cent(A), cB = cent(B), cX = cent(Bx);
  const rA = rdm(cA.cent, nI, nN), rB = rdm(cB.cent, nI, nN), rX = rdm(cX.cent, nI, nN);
  const same = rdmCorrelation(rA, rB), diff = rdmCorrelation(rA, rX);
  assert.ok(same.rho > 0.6, 'same geometry: rho ' + same.rho);
  // a derangement of a near-symmetric geometry is not strongly negative; it only has to be clearly below the shared case
  assert.ok(diff.rho < same.rho - 0.3, 'deranged geometry lower: ' + diff.rho + ' vs ' + same.rho);
  const x = crossRdm(cA.cent, cB.cent, nI, nN), xx = crossRdm(cA.cent, cX.cent, nI, nN);
  assert.ok(x.dominance > 0.5, 'cross RDM diagonal dominates for a shared code: ' + x.dominance);
  assert.ok(xx.dominance < 0.1, 'no dominance for an unshared code: ' + xx.dominance);
  console.log('ok  rsa: rho same ' + same.rho + ' deranged ' + diff.rho + '; cross dominance ' + x.dominance + ' vs ' + xx.dominance);
}

// enhancement index and additivity on known numbers
{
  const both = new Float32Array([10, 7, 3]), vis = new Float32Array([4, 4, 3]), aud = new Float32Array([3, 3, 0]);
  const m = multisensoryIndex(both, vis, aud, [[0, 1, 2]], 1, 3);
  assert.equal(m.cells, 3);
  // CRE per cell: (10-4)/4 = 1.5, (7-4)/4 = 0.75, (3-3)/3 = 0
  assert.ok(Math.abs(m.cre - (1.5 + 0.75 + 0)/3) < 1e-6, 'mean CRE ' + m.cre);
  // 10 > 1.1*7 superadditive; 7 = 7 additive; 3 = 3 additive
  assert.equal(m.superadditive, +(1/3).toFixed(3)); assert.equal(m.additive, +(2/3).toFixed(3));
  const none = multisensoryIndex(both, new Float32Array(3), new Float32Array(3), [[0]], 1, 3);
  assert.equal(none.cre, null); assert.equal(none.undefined, 1);
  console.log('ok  multisensoryIndex: cre ' + m.cre + ' super ' + m.superadditive + ' add ' + m.additive);
}

// population sparseness: one active cell is 1, a flat response is 0
{
  const one = new Float32Array(nN); one[3] = 5;
  const flat = new Float32Array(nN).fill(2);
  const c = new Float32Array(2*nN); c.set(one, 0); c.set(flat, nN);
  const s1 = populationSparseness(c, [1, 0], 2, nN), s2 = populationSparseness(c, [0, 1], 2, nN);
  assert.ok(Math.abs(s1.mean - 1) < 1e-6, 'one-hot is 1: ' + s1.mean);
  assert.ok(Math.abs(s2.mean) < 1e-6, 'flat is 0: ' + s2.mean);
  console.log('ok  populationSparseness: one-hot ' + s1.mean + ', flat ' + s2.mean);
}

// block structure: within-set links at 5, between at 1
{
  const sets = [[0, 1, 2], [3, 4, 5]];
  const a = [0, 1, 3, 4, 0, 3], b = [1, 2, 4, 5, 3, 0], slot = [0, 1, 2, 3, 4, 5];
  const w = new Float32Array([5, 5, 5, 5, 1, 1]);
  const bs = blockStructure(sets, { a:Int32Array.from(a), b:Int32Array.from(b), slot:Int32Array.from(slot) }, w, 6);
  assert.equal(bs.within, 5); assert.equal(bs.between, 1); assert.equal(bs.ratio, 5);
  assert.equal(bs.nWithin, 4); assert.equal(bs.nBetween, 2);
  // the set-to-set matrix behind the sorted weight figure
  assert.equal(bs.sets, 2);
  assert.deepEqual(Array.from(bs.matrix), [5, 1, 1, 5]);
  // inhibitory links are not part of the block structure
  const wi = new Float32Array([5, 5, 5, 5, -1, -1]);
  assert.equal(blockStructure(sets, { a:Int32Array.from(a), b:Int32Array.from(b), slot:Int32Array.from(slot) }, wi, 6), null);
  console.log('ok  blockStructure: within ' + bs.within + ' between ' + bs.between + ' ratio ' + bs.ratio);
}
// the decoder's confusion matrix: a shared code confuses nothing
{
  const d = decodeAccuracy(A.trials, A.labels, nI, nN);
  assert.ok(d.confusion && d.confusion.length === nI*nI, 'confusion matrix present');
  let diag = 0, off = 0;
  for(let i=0;i<nI;i++) for(let j=0;j<nI;j++) (i === j ? (diag += d.confusion[i*nI + j]) : (off += d.confusion[i*nI + j]));
  assert.ok(diag > 0 && diag >= 5*off, 'diagonal dominates: ' + diag + ' vs ' + off);
  console.log('ok  confusion: diagonal ' + diag + ', off ' + off);
}
// Association asked of the synapses.
// A hand-built network, four cells, two per set, with one synapse of each kind: the labelling has to put each synapse in the right pathway, leave the frozen one out, and the summary has to be positive exactly when the between-code synapses grew more than the within-code ones.
{
  const net = { count:4, synCount:6,
    preStart:Uint32Array.from([0, 2, 4, 5, 6]),
    post:Uint32Array.from([1, 2,  0, 3,  3,  1]),
    //                     A>A A>B A>A A>B B>B(frozen) B>A
    pmask:Uint8Array.from([1, 1, 1, 1, 0, 1]) };
  const A = Uint8Array.from([1, 1, 0, 0]), B = Uint8Array.from([0, 0, 1, 1]);
  const pc = pairClasses(net, A, B);
  assert.deepEqual(Array.from(pc.cls), [0, 1, 0, 1, 255, 2]);
  assert.deepEqual(pc.counts, [2, 2, 1, 0], 'the frozen B>B synapse is not counted');
  console.log('ok  pairClasses: ' + pc.counts.join('/') + ' (AA/AB/BA/BB), frozen excluded');

  const w0 = Float32Array.from([1, 1, 1, 1, 1, 1]);
  const grew = setWeightStats(pc.cls, Float32Array.from([1.1, 1.5, 1.0, 1.6, 9, 1.2]), w0);
  assert.equal(grew.ab.n, 2);
  assert.ok(grew.betweenMinusWithin > 0.3, 'between-code growth reads positive');
  const flat = setWeightStats(pc.cls, Float32Array.from([1.5, 1.1, 1.6, 1.0, 9, 1.0]), w0);
  assert.ok(flat.betweenMinusWithin < 0, 'within-code growth reads negative');
  const none = setWeightStats(pc.cls, w0, w0);
  assert.equal(none.betweenMinusWithin, 0, 'no change reads zero');
  // an inhibitory synapse is not part of the comparison
  const inh = setWeightStats(pc.cls, Float32Array.from([1, 1, 1, 1, 1, 1]),
    Float32Array.from([-1, -1, -1, -1, -1, -1]));
  assert.equal(inh.betweenMinusWithin, undefined, 'nothing excitatory, nothing to compare');
  console.log('ok  setWeightStats: grew ' + grew.betweenMinusWithin +
    ', within-heavy ' + flat.betweenMinusWithin + ', unchanged ' + none.betweenMinusWithin);
}

// Below chance is not one thing.
// A decoder that sends every item to one particular other item has associated them wrongly; a decoder whose errors scatter has only pushed each response off its own.
// Both read as low accuracy, and the difference is in the confusion matrix.
{
  const nI = 4;
  const consistent = new Float64Array(nI*nI);
  for(let i=0;i<nI;i++) consistent[i*nI + (i + 1) % nI] = 10;
  const cs = confusionShape(consistent, nI);
  assert.equal(cs.topOffMass, 1);
  assert.equal(cs.topOffRatio, 3);          // even spread would be 1/3 of the row
  assert.deepEqual(cs.partners, [1, 2, 3, 0]);

  const scattered = new Float64Array(nI*nI);
  for(let i=0;i<nI;i++) for(let k=0;k<nI;k++) if(k !== i) scattered[i*nI + k] = 10;
  const sc = confusionShape(scattered, nI);
  assert.equal(sc.topOffRatio, 1, 'evenly spread errors sit at the even-spread ratio');
  assert.equal(confusionShape(new Float64Array(nI*nI), nI), null, 'no errors, nothing to shape');
  // a row with one wrong answer has a favorite whatever it does, so it must not be allowed to read as concentration
  const thin = new Float64Array(nI*nI);
  for(let i=0;i<nI;i++) thin[i*nI + (i + 1) % nI] = 1;
  const th = confusionShape(thin, nI);
  assert.equal(th.rows, 0, 'no row has enough errors to judge');
  assert.equal(th.topOffRatio, undefined, 'and so no ratio is offered');
  assert.equal(th.off, nI, 'the errors are still counted');
  console.log('ok  confusionShape: consistent ' + cs.topOffRatio +
    'x even, scattered ' + sc.topOffRatio + 'x');
}

console.log('xmodal: all checks passed');

// The generalization has to contain the thing it generalizes.
// If setsFromExpr with the conjunction expression does not reproduce conjunctiveSets exactly, then this is a rewrite wearing the same name and every earlier number stops being comparable.
import { conjunctiveSets, setsFromExpr } from './analysis.js';
import { parse } from './setexpr.js';

let fail = 0;
const ok = (n, c, x) => { console.log((c?'ok   ':'FAIL ')+n+(x?'  '+x:'')); if(!c) fail++; };

const NI = 6, NN = 400, K = 15;
let z = 4242;
const rnd = () => { z = (z*1664525 + 1013904223) >>> 0; return z/4294967296; };
const mk = () => {
  const b = new Float32Array(NI*NN), v = new Float32Array(NI*NN), a = new Float32Array(NI*NN);
  for(let i=0;i<NI;i++){
    const o = i*NN;
    for(let k=0;k<K;k++){ b[o + i*3*K + k] = 1; v[o + i*3*K + k] = 1; }
    for(let k=0;k<K;k++){ b[o + i*3*K + K + k] = 1; a[o + i*3*K + K + k] = 1; }
    for(let k=0;k<K;k++){ b[o + i*3*K + 2*K + k] = 1; }
    for(let j=0;j<NN;j++){ b[o+j] += rnd()*0.01; v[o+j] += rnd()*0.01; a[o+j] += rnd()*0.01; }
  }
  return { both:b, sight:v, sound:a };
};
const C = mk();

{
  const oldWay = conjunctiveSets(C.both, C.sight, C.sound, NI, NN, 0.2, 0);
  const newWay = setsFromExpr(parse('both and not sight and not sound'),
    C, NI, NN, 0.2, 0);
  const same = oldWay.every((s, i) =>
    s.length === newWay[i].length && s.every((x, k) => x === newWay[i][k]));
  ok('the expression form reproduces conjunctiveSets exactly', same,
    'sizes ' + oldWay.map(s => s.length).join(',') + ' vs ' +
    newWay.map(s => s.length).join(','));
}
{
  // questions the fixed code could not ask at all
  const only = setsFromExpr(parse('sight and not sound'), C, NI, NN, 0.2, 0);
  ok('a one-sense question works', only.every(s => s.length >= K*0.8),
    'mean ' + (only.reduce((a,s) => a+s.length, 0)/NI).toFixed(1) + ' cells');
  const any = setsFromExpr(parse('sight or sound'), C, NI, NN, 0.2, 0);
  ok('a union works', any.every(s => s.length > only[0].length),
    'mean ' + (any.reduce((a,s) => a+s.length, 0)/NI).toFixed(1) + ' cells');
}
{
  // and one an experiment with different conditions entirely would ask
  const D = { attended:C.both, ignored:C.sight };
  const r = setsFromExpr(parse('attended and not ignored'), D, NI, NN, 0.2, 0);
  ok('conditions need not be modalities', r.some(s => s.length > 0),
    'mean ' + (r.reduce((a,s) => a+s.length, 0)/NI).toFixed(1) + ' cells');
}

console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);

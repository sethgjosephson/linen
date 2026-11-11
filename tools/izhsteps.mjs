// How far each membrane step sits from Izhikevich's equations (MODEL.md 1), on every quadratic row of the type table.
// Each row runs as one cell on the reference engine under the half steps (integ 0) and the equations' step (integ 2), and on a 1 microsecond Euler clock that stands for the equations.
// The input is the row's rheobase on that clock times 1.5, 2 and 4, held, and the same with a Poisson kick the size of the bias each way at 2 kHz, piecewise constant per millisecond as the engine takes it.
// The rate is counted over the second half of 4 s.
//
//   node tools/izhsteps.mjs
import { PROTOCOL } from '../src/protocol.js';
import { initState } from '../src/rand.js';
import { NEURON_TYPES } from '../src/nodes.js';

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const handle = globalThis.onmessage;
const send = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };
const VMIN = -90, MS = 4000;

// the input, one value per millisecond
function drive(bias, ms, noise, seed = 7){
  const I = new Float64Array(ms).fill(bias);
  if(!noise) return I;
  let s = seed >>> 0;
  const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s/4294967296; };
  for(let t = 0; t < ms; t++){ let k = 0; for(let j = 0; j < 4; j++){ const r = rnd(); if(r < 0.5) k += r < 0.25 ? 1 : -1; } I[t] += k*noise*bias; }
  return I;
}
// the engine, the bias retuned every millisecond; the spike times in ms
function engine(f7, I, integ){
  const types = [{ f7 }], ntype = Uint8Array.from([0]), st = initState(3, ntype, types);
  const cfg = { syn:0, plast:0, refrac:0, vmin:VMIN, eRevI:-80, integ };
  send({ cmd:'init', protocol:PROTOCOL, count:1, ntype, bias:Float32Array.from([I[0]]), preStart:Int32Array.from([0, 0]), post:new Int32Array(0),
    w:new Float32Array(0), delay:new Uint8Array(0), pmask:new Uint8Array(0), ...cfg, types, seed:3, ...st, v0:Float32Array.from([f7.vr]), u0:new Float32Array(1) });
  const sp = [];
  for(let t = 0; t < I.length; t++){
    if(t) send({ cmd:'tune', bias:Float32Array.from([I[t]]), ...cfg });
    if(send({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state').fired[0]) sp.push(t);
  }
  return sp;
}
// the equations on a 1 microsecond Euler clock
function fine(p, I){
  const N = 1000, h = 1/N, sp = [];
  let v = p.vr, u = 0;
  for(let t = 0; t < I.length; t++) for(let q = 0; q < N; q++){
    const vn = v + h*(p.k*(v - p.vr)*(v - p.vt) - u + I[t])/p.C;
    u += h*p.a*(p.b*(v - p.vr) - u);
    v = vn < VMIN ? VMIN : vn;
    if(v >= p.vpeak){ v = p.c; u += p.d; sp.push(t); }
  }
  return sp;
}
const hz = sp => sp.filter(x => x >= MS/2).length/(MS/2000);
function rheobase(p){
  // the bias at which the cell keeps firing through the second of its two seconds, not the onset spike alone
  const fires = b => fine(p, drive(b, 2000, 0)).some(x => x >= 1000);
  let lo = 0, hi = 1;
  while(!fires(hi) && hi < 1e6) hi *= 2;
  for(let k = 0; k < 14; k++){ const m = (lo + hi)/2; if(fires(m)) hi = m; else lo = m; }
  return hi;
}
const rows = NEURON_TYPES.filter(t => t.f7 && !t.f7.graded && t.f7.vt - t.f7.vpeak <= 100);
const err = { 0:[], 2:[] };
console.log('| row | input | drive | 1 us clock Hz | half steps Hz | equations\' step Hz |');
console.log('|---|---|---|---|---|---|');
for(const t of rows){
  const rb = rheobase(t.f7);
  for(const noise of [0, 1]) for(const m of [1.5, 2, 4]){
    const I = drive(m*rb, MS, noise), f = hz(fine(t.f7, I)), e0 = hz(engine(t.f7, I, 0)), e2 = hz(engine(t.f7, I, 2));
    if(f > 0){ err[0].push(Math.abs(e0 - f)/f); err[2].push(Math.abs(e2 - f)/f); }
    console.log('| ' + t.key + ' | ' + (noise ? 'kicks' : 'held') + ' | ' + m + 'x | ' + f.toFixed(1) + ' | ' + e0.toFixed(1) + ' | ' + e2.toFixed(1) + ' |');
  }
}
console.log('\n| step | mean error | worst |\n|---|---|---|');
for(const [k, name] of [[0, 'half steps (integ 0)'], [2, 'equations\' step (integ 2)']]){
  const e = err[k];
  console.log('| ' + name + ' | ' + (100*e.reduce((a, b) => a + b, 0)/e.length).toFixed(1) + '% | ' + (100*Math.max(...e)).toFixed(1) + '% |');
}

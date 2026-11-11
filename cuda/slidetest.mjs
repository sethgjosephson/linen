// Does the CUDA child apply the sliding threshold as the reference does (MODEL.md 4: depression scaled by the square of the cell's slow rate estimate over the target, the estimate a float32 trace started where slideInit says)?
// The spikes of src/slide.test.mjs on both engines through the same input map: post at 5, pre at 10, the weight read back and compared to the reference's and to the arithmetic.
//
//   node cuda/slidetest.mjs [exePath]
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from '../src/protocol.js';
import { initState } from '../src/rand.js';
import { engineTypes, NEURON_TYPES, engineConfig } from '../src/nodes.js';
import { CudaEngine } from '../host/cudaengine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXE = process.argv[2] || path.join(HERE, 'engine.exe');
if(!fs.existsSync(EXE)){ console.error('slidetest: no engine at ' + EXE); process.exit(1); }

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const handle = globalThis.onmessage;
const refSend = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };
async function child(){
  const eng = new CudaEngine(EXE, () => {});
  const inbox = [];
  eng.on('message', d => inbox.push(d));
  const next = async cmd => { for(let spin = 0; spin < 20000; spin++){ const k = inbox.findIndex(r => r.cmd === cmd); if(k >= 0) return inbox.splice(0, k + 1)[k]; const e = inbox.find(r => r.cmd === 'error'); if(e) throw new Error(e.message); await new Promise(r => setTimeout(r, 5)); } throw new Error('no ' + cmd); };
  eng.postMessage({ cmd:'hello' }); const hello = await next('hello');
  return { hello, send:(d, reply) => { eng.postMessage(d); return reply ? next(reply) : null; }, close:() => eng.terminate() };
}

const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07');
const TAU = 20, AM = 0.1;
function init({ slide = 0, slideRate = 5, slideTau = 1, slideInit = 5 }){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias, preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([1]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0, aM:AM, tauS:TAU, tauM:TAU, wmax:30, wdep:0, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    slide, slideRate, slideTau, slideInit, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
const gains = t => Float32Array.from([t === 10 ? 1 : 0, t === 5 ? 1 : 0]);
function refRun(c){
  refSend(init(c));
  for(let t = 0; t < 20; t++){ refSend({ cmd:'inputFrame', mi:0, gains:gains(t) }); refSend({ cmd:'tick', steps:1 }); }
  return refSend({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0];
}
async function childRun(c){
  const ch = await child();
  ch.send(init(c));
  for(let t = 0; t < 20; t++){ ch.send({ cmd:'inputFrame', mi:0, gains:gains(t) }); await ch.send({ cmd:'tick', steps:1 }, 'state'); }
  const wm = await ch.send({ cmd:'getWeights' }, 'weights');
  ch.close();
  return { w:wm.w[0], terms:ch.hello.terms || [] };
}
const f32 = x => Math.fround(x);
const near = (a, b, tol = 1e-5) => Math.abs(a - b) < tol;
function khomAt(slideInit, tauMs, postAt, readAt){
  const dec = Math.exp(-1/tauMs);
  let k = f32(slideInit*tauMs/1000);
  for(let t = 0; t < readAt; t++){ if(t === postAt) k = f32(k + 1); k = f32(k*dec); }
  return k;
}
const kq = Math.exp(-5/TAU);
const factor = (slideInit, slideRate) => { const kh = khomAt(slideInit, 1000, 5, 10), k = 1000/(1000*slideRate); return (kh*k)*(kh*k); };
const cases = [
  ['with the threshold off the pre spike depresses by A- times the fast trace', { slide:0 }, f32(1 - AM*kq)],
  ['an estimate above the target depresses more', { slide:1, slideRate:5, slideTau:1, slideInit:5 }, f32(1 - AM*kq*factor(5, 5))],
  ['an estimate under the target depresses less', { slide:1, slideRate:5, slideTau:1, slideInit:2.5 }, f32(1 - AM*kq*factor(2.5, 5))],
];
let terms = null;
for(const [label, c, expect] of cases){
  const r = refRun(c), ch = await childRun(c); terms = ch.terms;
  ok(label + ' (reference)', near(r, expect), 'reference ' + r + ' expected ' + expect);
  ok(label + ' (child agrees)', near(ch.w, r, 2e-5), 'child ' + ch.w + ' reference ' + r);
}
ok('the child names slide among its terms', terms.includes('slide'), terms.join(' '));
report('slidetest');

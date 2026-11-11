// Does the CUDA child apply the heterosynaptic term as the reference does (MODEL.md 4: beta times the cube of the fast postsynaptic trace, read before the spike, times the distance from the reference), and start the consolidation reference where refInit says?
// The pairings of src/het.test.mjs on both engines through the same input map: pre at 2, post at 7 and again at 12, the weight read back and compared to the reference's and to the arithmetic.
//
//   node cuda/hettest.mjs [exePath]
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
if(!fs.existsSync(EXE)){ console.error('hettest: no engine at ' + EXE); process.exit(1); }

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
const TAU = 20;
function init({ het, w = 1, cons = 0, refInit = 0, tauCons = 20 }){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias, preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([w]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0.1, aM:0, tauS:TAU, tauM:TAU, wmax:30, wdep:1, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    het, cons, consW:0.5, consP:10, tauCons, consStep:1.2, refInit, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
const gains = t => Float32Array.from([t === 2 ? 1 : 0, t === 7 || t === 12 ? 1 : 0]);
function refRun(c){
  refSend(init(c));
  if(c.idle) refSend({ cmd:'tick', steps:c.idle });
  for(let t = 0; t < 30; t++){ refSend({ cmd:'inputFrame', mi:0, gains:gains(t) }); refSend({ cmd:'tick', steps:1 }); }
  return refSend({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0];
}
async function childRun(c){
  const ch = await child();
  ch.send(init(c));
  if(c.idle) await ch.send({ cmd:'tick', steps:c.idle }, 'state');
  for(let t = 0; t < 30; t++){ ch.send({ cmd:'inputFrame', mi:0, gains:gains(t) }); await ch.send({ cmd:'tick', steps:1 }, 'state'); }
  const wm = await ch.send({ cmd:'getWeights' }, 'weights');
  ch.close();
  return { w:wm.w[0], terms:ch.hello.terms || [] };
}
const f32 = x => Math.fround(x);
const near = (a, b, tol = 1e-5) => Math.abs(a - b) < tol;
const lift = 0.1*Math.exp(-5/TAU), lift2 = 0.1*Math.exp(-10/TAU), kq = Math.exp(-5/TAU), cube = kq*kq*kq, beta = 2;
const cases = [
  ['with beta 0 the second spike only potentiates again', { het:0 }, f32(f32(1 + lift) + lift2)],
  ['with beta the second spike pulls the weight toward its reference by beta Kpost^3 times the lift', { het:beta }, f32(f32(1 + lift) + lift2 - beta*cube*(f32(1 + lift) - 1))],
  ['with the reference started at zero the pull aims at zero', { het:beta, cons:1, refInit:1 }, f32(f32(1 + lift) + lift2 - beta*cube*f32(1 + lift))],
  ['started at the consolidated weight it aims at 0.5', { het:beta, cons:1, refInit:2 }, f32(f32(1 + lift) + lift2 - beta*cube*(f32(1 + lift) - 0.5))],
];
let terms = null;
for(const [label, c, expect] of cases){
  const r = refRun(c), ch = await childRun(c); terms = ch.terms;
  ok(label + ' (reference)', near(r, expect), 'reference ' + r + ' expected ' + expect);
  ok(label + ' (child agrees)', near(ch.w, r), 'child ' + ch.w + ' reference ' + r);
}
// a synapse held above the upper point: after a minute the reference has settled above it, and the child's reference has to be where the reference engine's is
{ const c = { het:beta, cons:1, refInit:2, tauCons:0.2, idle:60000 }, r = refRun(c), ch = await childRun(c);
  ok('a reference above the upper point agrees between the engines', near(ch.w, r), 'child ' + ch.w + ' reference ' + r);
  const atUpper = f32(f32(1 + lift) + lift2 - beta*cube*(f32(1 + lift) - 0.5));
  ok('and it is not the upper point', r > atUpper + 0.01, 'reference ' + r + ' at the upper point ' + atUpper);
}
ok('the child names refInit among its terms', terms.includes('refInit'), terms.join(' '));
report('hettest');

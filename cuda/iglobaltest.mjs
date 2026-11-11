// Does the CUDA child apply the global factor on the inhibitory rule as the reference does (MODEL.md 4)?
// The three-cell protocol of src/iglobal.test.mjs on both engines through the same input map, the inhibitory weight read back after every step and compared.
//
//   node cuda/iglobaltest.mjs [exePath]
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
if(!fs.existsSync(EXE)){ console.error('iglobaltest: no engine at ' + EXE); process.exit(1); }

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

const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07'), FS = NEURON_TYPES.findIndex(t => t.key === 'FS07');
function init(iGlobal){
  const ntype = Uint8Array.from([FS, RS, RS]), bias = new Float32Array(3);
  const net = { count:3, ntype, bias, preStart:Int32Array.from([0, 1, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([-1]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0, aM:0, tauS:20, tauM:20, wmax:30, wdep:0, stdpAt:0, iEta:0.5, iRho:5, refrac:2, vmin:-90,
    iGlobal, iTauG:1, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:3, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2, 3]), chIdx:Int32Array.from([0, 1, 2]), chW:Float32Array.from([1, 1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
const fireAt = { 0:[10, 30, 50], 1:[20, 45], 2:[25] };
const gains = t => Float32Array.from([0, 1, 2].map(c => fireAt[c].includes(t) ? 1 : 0));
function refRun(iGlobal, upTo){
  refSend(init(iGlobal));
  const ws = [];
  for(let t = 0; t < upTo; t++){ refSend({ cmd:'inputFrame', mi:0, gains:gains(t) }); refSend({ cmd:'tick', steps:1 }); ws.push(refSend({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0]); }
  return ws;
}
async function childRun(iGlobal, upTo){
  const ch = await child();
  ch.send(init(iGlobal));
  const ws = [];
  for(let t = 0; t < upTo; t++){ ch.send({ cmd:'inputFrame', mi:0, gains:gains(t) }); await ch.send({ cmd:'tick', steps:1 }, 'state'); ws.push((await ch.send({ cmd:'getWeights' }, 'weights')).w[0]); }
  ch.close();
  return { ws, terms:ch.hello.terms || [] };
}
const near = (a, b, tol = 2e-5) => Math.abs(a - b) < tol;
let terms = null;
for(const [label, g] of [['with the global factor', 1], ['with the Vogels form', 0]]){
  const r = refRun(g, 60), c = await childRun(g, 60); terms = c.terms;
  let worst = 0, at = -1; for(let t = 0; t < 60; t++){ const d = Math.abs(r[t] - c.ws[t]); if(d > worst){ worst = d; at = t; } }
  ok(label + ' the child\'s weight follows the reference\'s through 60 steps', worst < 2e-5, 'worst ' + worst + ' at step ' + at + ': ' + r[at] + ' vs ' + c.ws[at]);
  ok(label + ' the weight moved', !near(r[59], -1), String(r[59]));
}
ok('the child names iGlobal among its terms', terms.includes('iGlobal'), terms.join(' '));
report('iglobaltest');

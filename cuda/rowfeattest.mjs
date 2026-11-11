// Does the CUDA child run the three row features as the reference does (MODEL.md 1: the dynamic threshold, the conductance adaptation, the Poisson output)?
// The one-cell cases of src/rowfeat.test.mjs on both engines: a constant bias, the spike times read back tick by tick and compared whole.
//
//   node cuda/rowfeattest.mjs [exePath]
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from '../src/protocol.js';
import { initState } from '../src/rand.js';
import { CudaEngine } from '../host/cudaengine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXE = process.argv[2] || path.join(HERE, 'engine.exe');
if(!fs.existsSync(EXE)){ console.error('rowfeattest: no engine at ' + EXE); process.exit(1); }

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

const LIF = { C:250, k:12.5/1070, vr:-70, vt:1000, vpeak:-50, a:0.01, b:0, c:-70, d:0 };
function init(f7, bias){
  const types = [{ f7 }], ntype = Uint8Array.from([0]);
  const st = initState(3, ntype, types);
  return { cmd:'init', protocol:PROTOCOL, count:1, ntype, bias:Float32Array.from([bias]),
    preStart:Int32Array.from([0, 0]), post:new Int32Array(0), w:new Float32Array(0), delay:new Uint8Array(0), pmask:new Uint8Array(0),
    syn:0, plast:0, refrac:0, vmin:-90, eRevI:-80, types, seed:3, ...st, v0:Float32Array.from([-70]), u0:new Float32Array(1) };
}
function refRun(f7, bias, steps){
  refSend(init(f7, bias));
  const times = [];
  for(let t = 0; t < steps; t++){ const r = refSend({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state'); if(r.fired[0] > 0) times.push(t); }
  return times;
}
async function childRun(f7, bias, steps){
  const ch = await child();
  ch.send(init(f7, bias));
  const times = [];
  for(let t = 0; t < steps; t++){ const r = await ch.send({ cmd:'tick', steps:1 }, 'state'); if(r.fired[0] > 0) times.push(t); }
  ch.close();
  return { times, terms:ch.hello.terms || [] };
}
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
let terms = null;
for(const [label, f7, bias, steps] of [
  ['a plain cell', LIF, 400, 1000],
  ['the dynamic threshold', { ...LIF, thrJump:100, thrTau:5 }, 400, 1000],
  ['the conductance adaptation', { ...LIF, d:1.25, adaptCond:1 }, 400, 2000],
  ['the Poisson row', { ...LIF, poisson:1 }, 30, 5000]]){
  const r = refRun(f7, bias, steps), c = await childRun(f7, bias, steps); terms = c.terms;
  ok(label + ': the child spikes when the reference does (' + r.length + ' spikes)', r.length > 0 && same(r, c.times), 'reference ' + r.slice(0, 6) + ' child ' + c.times.slice(0, 6));
}
ok('the child names thr, adaptCond and poisson among its terms', ['thr', 'adaptCond', 'poisson'].every(t => terms.includes(t)), terms.join(' '));
report('rowfeattest');

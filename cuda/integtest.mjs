// Does the CUDA child run the membrane steps of the checkpoint's integ as the reference does (MODEL.md 1: under 1 two exponential half steps and the reset at the crossing inside the millisecond, under 2 the equations solved exactly over each half step)?
// The one-cell cases of src/integ.test.mjs on both engines, and a conductance synapse from a driven cell onto a quiet one; the spike times read back tick by tick and compared.
//
//   node cuda/integtest.mjs [exePath]
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from '../src/protocol.js';
import { initState } from '../src/rand.js';
import { CudaEngine } from '../host/cudaengine.mjs';
import { NEURON_TYPES } from '../src/nodes.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXE = process.argv[2] || path.join(HERE, 'engine.exe');
if(!fs.existsSync(EXE)){ console.error('integtest: no engine at ' + EXE); process.exit(1); }

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

const row = key => NEURON_TYPES.find(t => t.key === key).f7;
const LIF = { C:250, k:12.5/1070, vr:-70, vt:1000, vpeak:-50, a:0.01, b:0, c:-70, d:0 };
// one cell under a constant bias, or (syn 2) cell 0 under the bias driving cell 1 through one conductance synapse of w nS
function init(f7, bias, syn = 0, w = 0, bias1 = 0, integ = 1){
  const n = syn ? 2 : 1, types = [{ f7 }], ntype = new Uint8Array(n);
  const st = initState(3, ntype, types);
  return { cmd:'init', protocol:PROTOCOL, count:n, ntype, bias:Float32Array.from(syn ? [bias, bias1] : [bias]),
    preStart:Int32Array.from(syn ? [0, 1, 1] : [0, 0]), post:Int32Array.from(syn ? [1] : []), w:Float32Array.from(syn ? [w] : []),
    delay:Uint8Array.from(syn ? [1] : []), pmask:Uint8Array.from(syn ? [0] : []),
    syn, tauE:5, tauI:10, eRevE:0, eRevI:-80, psc:0, plast:0, refrac:0, vmin:-90, integ, types, seed:3, ...st,
    v0:new Float32Array(n).fill(-70), u0:new Float32Array(n) };
}
const cell = syn => syn ? 1 : 0;
function refRun(c){
  refSend(init(...c.args));
  const times = [];
  for(let t = 0; t < c.steps; t++){ const r = refSend({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state'); if(r.fired[cell(c.args[2])] > 0) times.push(t); }
  return times;
}
async function childRun(c){
  const ch = await child();
  ch.send(init(...c.args));
  const times = [];
  for(let t = 0; t < c.steps; t++){ const r = await ch.send({ cmd:'tick', steps:1 }, 'state'); if(r.fired[cell(c.args[2])] > 0) times.push(t); }
  ch.close();
  return { times, terms:ch.hello.terms || [] };
}
// float32 on the child against float64 here: a spike may land a millisecond apart when the crossing sits on a step boundary
const close = (a, b) => a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) <= 1) && a.filter((x, i) => x !== b[i]).length <= Math.max(1, 0.02*a.length);
let terms = null;
for(const [label, c] of [
  ['a plain cell', { args:[LIF, 400], steps:2000 }],
  ['the dynamic threshold', { args:[{ ...LIF, thrJump:100, thrTau:5 }, 600], steps:2000 }],
  ['the conductance adaptation', { args:[{ ...LIF, d:1.25, adaptCond:1, thrJump:100, thrTau:5 }, 600], steps:2000 }],
  ['a conductance synapse onto a quiet cell', { args:[{ ...LIF, thrJump:100, thrTau:5 }, 600, 2, 30, 150], steps:3000 }],
  // the equations' step (integ 2) on quadratic rows, in kick mode and through a conductance synapse
  ['integ 2, a 2007 regular spiking cell', { args:[row('RS07'), 104, 0, 0, 0, 2], steps:2000 }],
  ['integ 2, a 2007 fast-spiking cell', { args:[row('FS07'), 116, 0, 0, 0, 2], steps:2000 }],
  ['integ 2, a 2003 chattering cell', { args:[row('CH'), 5.42, 0, 0, 0, 2], steps:2000 }],
  ['integ 2, the dynamic threshold on a quadratic row', { args:[{ ...row('RS07'), thrJump:10, thrTau:5 }, 150, 0, 0, 0, 2], steps:2000 }],
  ['integ 2, a conductance synapse between 2007 regular spiking cells', { args:[row('RS07'), 150, 2, 20, 40, 2], steps:3000 }]]){
  const r = refRun(c), ch = await childRun(c); terms = ch.terms;
  ok(label + ': the child spikes when the reference does (' + r.length + ' spikes)', r.length > 0 && close(r, ch.times), 'reference ' + r.slice(0, 6) + ' child ' + ch.times.slice(0, 6) + ', ' + r.length + ' against ' + ch.times.length);
}
ok('the child names integ and integ:2 among its terms', terms.includes('integ') && terms.includes('integ:2'), terms.join(' '));
report('integtest');

// Does the CUDA child give each rule its own short-term plasticity (MODEL.md 2)?
//
// src/stprules.test.mjs holds the reference engine to it; the child lives behind a pipe and has no pulse protocol, so the source cell here is driven through an input map, one frame per spike.
// One presynaptic cell sends one synapse under a depressing rule and one under a facilitating rule (Markram, Wang and Tsodyks 1998) onto two targets held below threshold; each delivery's jump in each target's membrane, less the same network's with zero weights, is compared between the child and the reference.
//
//   node cuda/stprulestest.mjs [exePath]
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
if(!fs.existsSync(EXE)){ console.error('stprulestest: no engine at ' + EXE); process.exit(1); }

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
  const next = async cmd => {
    for(let spin = 0; spin < 20000; spin++){
      const k = inbox.findIndex(r => r.cmd === cmd);
      if(k >= 0) return inbox.splice(0, k + 1)[k];
      const e = inbox.find(r => r.cmd === 'error');
      if(e) throw new Error('child: ' + e.message);
      await new Promise(r => setTimeout(r, 1));
    }
    throw new Error('child: no ' + cmd);
  };
  eng.postMessage({ cmd:'hello' }); const hello = await next('hello');
  return { hello, send:(d, reply) => { eng.postMessage(d); return reply ? next(reply) : null; }, close:() => eng.terminate() };
}

const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07');
const DEP = { stpOwn:2, stpU:0.5, stpTauD:800, stpTauF:1 }, FAC = { stpOwn:2, stpU:0.03, stpTauD:130, stpTauF:530 };
const SPIKES = [5, 25, 45, 65, 85, 105, 125, 145, 165, 185];
function init(w, syn){
  const ntype = Uint8Array.from([RS, RS, RS]), bias = Float32Array.from([0, -40, -40]);
  const net = { count:3, ntype, bias, preStart:Int32Array.from([0, 2, 2, 2]), post:Int32Array.from([1, 2]),
    w:Float32Array.from([w, w]), delay:Uint8Array.from([1, 1]), pmask:Uint8Array.from([2, 3]),
    syn, tauE:3, tauI:8, psc:0, plast:0, refrac:2, vmin:-90, stp:0, stpNorm:0, stpOrder:0, stpScope:0,
    rules:[{ name:'a', ...DEP }, { name:'b', ...FAC }], protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:3, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1]), chIdx:Int32Array.from([0]), chW:Float32Array.from([1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
const frame = t => ({ cmd:'inputFrame', mi:0, gains:Float32Array.from([SPIKES.includes(t) ? 1 : 0]) });
function refTrace(w, syn){
  refSend(init(w, syn)); refSend({ cmd:'sendV', on:true });
  const v = [], fired = [];
  for(let t = 0; t < 200; t++){
    refSend(frame(t));
    const st = refSend({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state');
    v.push(Float32Array.from(st.v)); if(st.fired[0]) fired.push(t);
  }
  return { v, fired };
}
async function childTrace(w, syn){
  const ch = await child();
  ch.send(init(w, syn)); ch.send({ cmd:'sendV', on:true });
  const v = [], fired = [];
  for(let t = 0; t < 200; t++){
    ch.send(frame(t));
    const st = await ch.send({ cmd:'tick', steps:1 }, 'state');
    v.push(Float32Array.from(st.v)); if(st.fired[0]) fired.push(t);
  }
  const terms = ch.hello.terms || [];
  ch.close();
  return { v, fired, terms };
}
const jumps = (a, z) => {
  const d = (t, j) => a.v[t][j] - z.v[t][j], out = [[], []];
  for(const t of a.fired) if(t + 1 < a.v.length) for(const j of [1, 2]) out[j - 1].push(d(t + 1, j) - d(t, j));
  return out;
};

for(const syn of [0, 1]){
  const label = syn ? 'exponential synapses' : 'kick synapses';
  const ref = jumps(refTrace(200, syn), refTrace(0, syn));
  const a = await childTrace(200, syn), z = await childTrace(0, syn);
  if(!syn) ok('the child names per-rule short-term plasticity in its hello', a.terms.includes('stpRules'), a.terms.join(' '));
  const cu = jumps(a, z);
  ok(label + ': the child fires the source on every frame', a.fired.length === SPIKES.length, a.fired.join(' '));
  const ratio = x => x[x.length - 1]/x[0];
  ok(label + ': on the child the depressing rule\'s target receives less each delivery and the facilitating one\'s more',
    ratio(cu[0]) < 0.3 && ratio(cu[1]) > 3, 'depressing ' + ratio(cu[0]).toFixed(3) + ', facilitating ' + ratio(cu[1]).toFixed(3));
  const err = Math.max(...[0, 1].flatMap(j => cu[j].map((x, k) => Math.abs(x - ref[j][k])/Math.max(1e-6, Math.abs(ref[j][k])))));
  ok(label + ': every delivery on the child within 1 percent of the reference', err < 0.01, 'largest relative difference ' + err.toExponential(2));
}
report('stprulestest');

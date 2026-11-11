// Does the CUDA child read back the consolidation reference as the reference engine does, and take it back with setRef (engine contract 15)?
// The case of src/brainref.test.mjs on both engines: one synapse idle for twenty seconds until it commits, the reference read back and compared, then a fresh child given that reference, where a pairing must leave the committed synapse where it was.
//
//   node cuda/reftest.mjs [exePath]
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
if(!fs.existsSync(EXE)){ console.error('reftest: no engine at ' + EXE); process.exit(1); }

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
function init(w){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias, preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([w]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0.1, aM:0.01, tauS:20, tauM:20, wmax:30, wdep:1, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    het:0, cons:1, consW:1.2, consP:10, tauCons:0.5, consStep:1.2, commit:1, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
const gains = t => Float32Array.from([t === 2 ? 1 : 0, t === 7 ? 1 : 0]);

// the reference engine: the trained reference
refSend(init(1));
refSend({ cmd:'tick', steps:20000 });
const want = refSend({ cmd:'getWeights' }).find(x => x.cmd === 'weights');

// the child: the same twenty seconds
let ch = await child();
ch.send(init(1));
await ch.send({ cmd:'tick', steps:20000 }, 'state');
const got = await ch.send({ cmd:'getWeights' }, 'weights');
ch.close();
ok('the child\'s weights reply carries the reference', got.ref instanceof Float32Array && got.ref.length === 1, String(got.ref));
ok('and it matches the reference engine\'s', got.ref && Math.abs(got.ref[0] - want.ref[0]) < 1e-5, 'cuda ' + (got.ref && got.ref[0]) + ' reference ' + want.ref[0]);

// a fresh child from the saved weight, with and without the reference
async function restored(withRef){
  const c = await child();
  c.send(init(got.w[0]));
  if(withRef) c.send({ cmd:'setRef', wRef:got.ref });
  const back = withRef ? (await c.send({ cmd:'getWeights' }, 'weights')).ref : null;
  for(let t = 0; t < 40; t++){ c.send({ cmd:'inputFrame', mi:0, gains:gains(t) }); await c.send({ cmd:'tick', steps:1 }, 'state'); }
  const w = (await c.send({ cmd:'getWeights' }, 'weights')).w[0];
  c.close();
  return { w, back };
}
const kept = await restored(true);
ok('setRef puts the reference in place', kept.back && kept.back[0] === got.ref[0], String(kept.back));
ok('restored with its reference, the committed synapse does not potentiate', kept.w === got.w[0], 'w ' + kept.w + ' saved ' + got.w[0]);
const lost = await restored(false);
ok('restored without it, the same pairing potentiates', lost.w > got.w[0] + 0.05, 'w ' + lost.w);
report('cuda reftest');
process.exit(0);

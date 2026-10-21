// Does the CUDA child carry a noise stimulus's memory and shared part (MODEL.md 2)?
//
// The battery holds the two page engines to it; the child lives behind a pipe, so this holds it to the reference the same way.
// N cells with a bias and a noise drive, no synapses between them, run white, with a 20 ms memory, and with the memory and a fully shared part, on the reference in this process and on the child: the child's rates match the reference's in each case, and the shared part raises population synchrony on both.
//
//   node cuda/noisetest.mjs [exePath]
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from '../src/protocol.js';
import { initState } from '../src/rand.js';
import { engineTypes, NEURON_TYPES } from '../src/nodes.js';
import { CudaEngine } from '../host/cudaengine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXE = process.argv[2] || path.join(HERE, 'engine.exe');
if(!fs.existsSync(EXE)){ console.error('noisetest: no engine at ' + EXE); process.exit(1); }
const N = 400, MS = 2000;
function mk(tau, common){
  const ntype = new Uint8Array(N), bias = new Float32Array(N).fill(5);
  // one weak synapse per cell onto its neighbor, so every engine has a wiring to hold
  const preStart = new Int32Array(N + 1); for(let i = 0; i <= N; i++) preStart[i] = i;
  const post = new Int32Array(N); for(let i = 0; i < N; i++) post[i] = (i + 1) % N;
  const types = engineTypes();
  return { cmd:'init', protocol:PROTOCOL, count:N, ntype, bias, syn:1, tauE:3, tauI:8, psc:0, plast:0, refrac:2,
    protocols:[{ idx:Uint32Array.from({ length:N }, (_, i) => i), amp:12, mode:3, period:100, width:10, t0:0, duration:0, tau, common }], inputs:[],
    preStart, post, w:new Float32Array(N).fill(0.1), delay:new Uint8Array(N).fill(1), pmask:new Uint8Array(N).fill(0),
    types, seed:9, ...initState(9, ntype, NEURON_TYPES) };
}
const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const handle = globalThis.onmessage;
function refRun(init){
  replies.length = 0; handle({ data:structuredClone(init) });
  const err = replies.find(r => r.cmd === 'error'); if(err) throw new Error(err.message);
  const frames = [];
  for(let t = 0; t < MS; t += 2){ replies.length = 0; handle({ data:{ cmd:'tick', steps:2 } }); frames.push(replies.find(r => r.cmd === 'state').spikes); }
  return frames;
}
async function cudaRun(init){
  const eng = new CudaEngine(EXE, () => {});
  const inbox = [];
  eng.on('message', d => inbox.push(d));
  const next = async cmd => { for(let spin = 0; spin < 20000; spin++){ const k = inbox.findIndex(r => r.cmd === cmd); if(k >= 0) return inbox.splice(0, k + 1)[k]; const e = inbox.find(r => r.cmd === 'error'); if(e) throw new Error(e.message); await new Promise(r => setTimeout(r, 1)); } throw new Error('the child stopped answering'); };
  eng.postMessage({ cmd:'hello' }); await next('hello');
  eng.postMessage(init);
  const frames = [];
  for(let t = 0; t < MS; t += 2){ eng.postMessage({ cmd:'tick', steps:2 }); const st = await next('state'); frames.push(st.spikes); }
  eng.terminate();
  return frames;
}
const rate = f => f.reduce((a, b) => a + b, 0)/N/(MS/1000);
const sync = f => { const m = f.reduce((a, b) => a + b, 0)/f.length; const v = f.reduce((a, b) => a + (b - m)*(b - m), 0)/f.length; return m > 0 ? v/m : 0; };
const close = (a, b) => Math.abs(a - b) <= Math.max(0.3, 0.2*b);
const cases = [['white', 0, 0], ['tau 20 ms', 20, 0], ['tau 20 ms, shared', 20, 1]];
const R = {}, C = {};
for(const [name, tau, common] of cases){ R[name] = refRun(mk(tau, common)); C[name] = await cudaRun(mk(tau, common)); }
for(const [name] of cases) ok('the child matches the reference, ' + name, close(rate(C[name]), rate(R[name])), rate(C[name]).toFixed(2) + ' vs ' + rate(R[name]).toFixed(2) + ' Hz');
ok('the memory moves the rate (reference)', Math.abs(rate(R['tau 20 ms']) - rate(R.white))/rate(R.white) > 0.05, rate(R.white).toFixed(2) + ' to ' + rate(R['tau 20 ms']).toFixed(2) + ' Hz');
ok('the shared part raises synchrony on the reference', sync(R['tau 20 ms, shared']) > 2*sync(R['tau 20 ms']), sync(R['tau 20 ms']).toFixed(2) + ' to ' + sync(R['tau 20 ms, shared']).toFixed(2));
ok('and on the child', sync(C['tau 20 ms, shared']) > 2*sync(C['tau 20 ms']), sync(C['tau 20 ms']).toFixed(2) + ' to ' + sync(C['tau 20 ms, shared']).toFixed(2));
report('noisetest');

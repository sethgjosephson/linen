// Does the CUDA child carry the magnesium block (MODEL.md 2)?
//
// The battery holds the two page engines to it; the child lives behind a pipe, so this holds it to the reference the same way.
// N sources each onto a target through one synapse on a receptor channel, run with the channel open and with it gated at 1 mM, on the reference engine in this process and on the child: the gated targets fire well below the open ones, and the child's rates match the reference's.
//
//   node cuda/mgtest.mjs [exePath]
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
if(!fs.existsSync(EXE)){ console.error('mgtest: no engine at ' + EXE); process.exit(1); }
const N = 200;
function mk(mg){
  const ntype = new Uint8Array(2*N), bias = new Float32Array(2*N);
  for(let i = 0; i < N; i++) bias[i] = 6 + 2*(i/N);
  const preStart = new Int32Array(2*N + 1); for(let i = 0; i < N; i++) preStart[i+1] = i + 1; for(let i = N; i <= 2*N; i++) preStart[i] = N;
  const post = new Int32Array(N); for(let i = 0; i < N; i++) post[i] = N + i;
  const types = engineTypes();
  return { cmd:'init', protocol:PROTOCOL, count:2*N, ntype, bias, syn:1, tauE:3, tauI:8, psc:0, plast:0, refrac:2,
    chanTau:[30], chanErev:[0], chanMg:[mg], protocols:[], inputs:[],
    preStart, post, w:new Float32Array(N).fill(4), delay:new Uint8Array(N).fill(1), pmask:new Uint8Array(N).fill(1 | (2 << 5)),
    types, seed:5, ...initState(5, ntype, NEURON_TYPES) };
}
const MS = 1000;
// the reference engine, in this process
const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const handle = globalThis.onmessage;
function refRun(init){
  replies.length = 0; handle({ data:structuredClone(init) });
  const counts = new Float64Array(2*N);
  for(let t = 0; t < MS; t++){ replies.length = 0; handle({ data:{ cmd:'tick', steps:1 } }); const st = replies.find(r => r.cmd === 'state'); for(let i = 0; i < 2*N; i++) counts[i] += st.fired[i]; }
  return counts;
}
async function cudaRun(init){
  const eng = new CudaEngine(EXE, () => {});
  const inbox = [];
  eng.on('message', d => inbox.push(d));
  const next = async cmd => { for(let spin = 0; spin < 20000; spin++){ const k = inbox.findIndex(r => r.cmd === cmd); if(k >= 0) return inbox.splice(0, k + 1)[k]; const e = inbox.find(r => r.cmd === 'error'); if(e) throw new Error(e.message); await new Promise(r => setTimeout(r, 1)); } throw new Error('the child stopped answering'); };
  eng.postMessage({ cmd:'hello' }); await next('hello');
  eng.postMessage(init);
  const counts = new Float64Array(2*N);
  for(let t = 0; t < MS; t += 10){ eng.postMessage({ cmd:'tick', steps:10 }); const st = await next('state'); for(let i = 0; i < 2*N; i++) counts[i] += st.fired[i]; }
  eng.terminate();
  return counts;
}
const rate = (c, lo, hi) => { let s = 0; for(let i = lo; i < hi; i++) s += c[i]; return s/(hi - lo)/(MS/1000); };
const rOpen = refRun(mk(0)), rGated = refRun(mk(1));
const cOpen = await cudaRun(mk(0)), cGated = await cudaRun(mk(1));
ok('sources fire alike open and gated (reference)', Math.abs(rate(rOpen, 0, N) - rate(rGated, 0, N)) < 0.5, rate(rOpen, 0, N).toFixed(1) + ' vs ' + rate(rGated, 0, N).toFixed(1) + ' Hz');
ok('the block cuts the targets well below the open channel (reference)', rate(rGated, N, 2*N) < 0.5*rate(rOpen, N, 2*N), rate(rGated, N, 2*N).toFixed(1) + ' vs ' + rate(rOpen, N, 2*N).toFixed(1) + ' Hz');
ok('the child matches the reference with the channel open', Math.abs(rate(cOpen, N, 2*N) - rate(rOpen, N, 2*N)) <= Math.max(0.5, 0.1*rate(rOpen, N, 2*N)), rate(cOpen, N, 2*N).toFixed(1) + ' vs ' + rate(rOpen, N, 2*N).toFixed(1) + ' Hz');
ok('and with the channel gated', Math.abs(rate(cGated, N, 2*N) - rate(rGated, N, 2*N)) <= Math.max(0.5, 0.1*rate(rGated, N, 2*N)), rate(cGated, N, 2*N).toFixed(1) + ' vs ' + rate(rGated, N, 2*N).toFixed(1) + ' Hz');
report('mgtest');

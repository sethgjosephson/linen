// Does the CUDA child time the pair rule at the synapse (stdpAt 1, MODEL.md 4)?
//
// src/stdpat.test.mjs holds the reference engine to the pairing arithmetic and the battery holds the two page engines to each other; the child lives behind a pipe and runs constant and noise drive only, so its pairing is driven through an input map, one channel per cell, with a frame of gain before the step each cell is to fire in.
// The same pairings on the reference engine in this process and on the child, with the weight after each read back: pre 5 ms before post through a 12 ms delay, which the soma reading potentiates and the synapse reading depresses, and a causal pairing timed from the arrival.
// Then a seeded recurrent network of 200 cells under noise, learning for two seconds on both engines at both readings: the weight changes of the two engines correlate as closely at the synapse reading as they do at the soma reading, and the two readings learn different weights on each engine.
//
//   node cuda/stdpattest.mjs [exePath]
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
if(!fs.existsSync(EXE)){ console.error('stdpattest: no engine at ' + EXE); process.exit(1); }

// the reference engine, in this process
const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const handle = globalThis.onmessage;
const refSend = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };
// the child over its pipe, the same messages
async function child(){
  const eng = new CudaEngine(EXE, () => {});
  const inbox = [];
  eng.on('message', d => inbox.push(d));
  const next = async cmd => { for(let spin = 0; spin < 20000; spin++){ const k = inbox.findIndex(r => r.cmd === cmd); if(k >= 0) return inbox.splice(0, k + 1)[k]; const e = inbox.find(r => r.cmd === 'error'); if(e) throw new Error(e.message); await new Promise(r => setTimeout(r, 1)); } throw new Error('the child stopped answering'); };
  eng.postMessage({ cmd:'hello' }); const hello = await next('hello');
  return { hello, send:(d, reply) => { eng.postMessage(d); return reply ? next(reply) : null; }, close:() => eng.terminate() };
}

// ---- the pairing: cell 0 sends to cell 1 through one plastic synapse of delay d; each cell is driven over the peak in the step of its own frame (20 nA for one step on an RS07 row)
const RS07 = NEURON_TYPES.findIndex(t => t.key === 'RS07');
function pairInit({ d, stdpAt }){
  const ntype = Uint8Array.from([RS07, RS07]), bias = new Float32Array(2);
  return { cmd:'init', protocol:PROTOCOL, count:2, ntype, bias, syn:0, psc:0, refrac:2, vmin:-90,
    plast:1, aP:0.1, aM:0.1, tauS:20, tauM:20, wmax:30, wdep:0, iEta:0, iRho:5, stdpAt,
    protocols:[], inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]), w:Float32Array.from([0.5]), delay:Uint8Array.from([d]), pmask:Uint8Array.from([1]),
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// the weight after 60 steps on the reference, with the frames at the steps given
function refPair(c, tPre, tPost){
  refSend(pairInit(c));
  const fired = [[], []];
  for(let t = 0; t < 60; t++){
    refSend({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t === tPre ? 1 : 0, t === tPost ? 1 : 0]) });
    const st = refSend({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state');
    for(let i = 0; i < 2; i++) if(st.fired[i]) fired[i].push(t);
  }
  return { w:refSend({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0], fired };
}
async function childPair(c, tPre, tPost){
  const ch = await child();
  ch.send(pairInit(c));
  const fired = [[], []];
  for(let t = 0; t < 60; t++){
    ch.send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t === tPre ? 1 : 0, t === tPost ? 1 : 0]) });
    const st = await ch.send({ cmd:'tick', steps:1 }, 'state');
    for(let i = 0; i < 2; i++) if(st.fired[i]) fired[i].push(t);
  }
  const wm = await ch.send({ cmd:'getWeights' }, 'weights');
  ch.close();
  return { w:wm.w[0], fired, terms:ch.hello.terms || [] };
}
const f32 = x => Math.fround(x);
const near = (a, b) => Math.abs(a - b) < 1e-6;
{
  // the critique's pairing: pre 5 ms before post through a 12 ms delay (pre at 2, post at 7, the arrival at 14)
  const rSoma = refPair({ d:12, stdpAt:0 }, 2, 7), cSoma = await childPair({ d:12, stdpAt:0 }, 2, 7);
  const rSyn = refPair({ d:12, stdpAt:1 }, 2, 7), cSyn = await childPair({ d:12, stdpAt:1 }, 2, 7);
  ok('the child names the term', cSyn.terms.includes('stdpAt'), cSyn.terms.join(' '));
  ok('each cell fires once, in the step of its frame, on both engines', JSON.stringify(rSoma.fired) === '[[2],[7]]' && JSON.stringify(cSoma.fired) === '[[2],[7]]',
    'reference ' + JSON.stringify(rSoma.fired) + ' child ' + JSON.stringify(cSoma.fired));
  ok('at the soma both engines potentiate by aP e^(-5/tau)', near(rSoma.w, f32(0.5 + 0.1*Math.exp(-5/20))) && near(cSoma.w, rSoma.w), 'reference ' + rSoma.w + ' child ' + cSoma.w);
  ok('at the synapse both engines depress by aM e^(-7/tau) from the arrival', near(rSyn.w, f32(0.5 - 0.1*Math.exp(-7/20))) && near(cSyn.w, rSyn.w), 'reference ' + rSyn.w + ' child ' + cSyn.w);
  // a causal pairing timed from the arrival: pre at 2 through a 3 ms delay arrives at 5, post at 9
  const rC = refPair({ d:3, stdpAt:1 }, 2, 9), cC = await childPair({ d:3, stdpAt:1 }, 2, 9);
  ok('a causal pairing is timed from the arrival on both engines', near(rC.w, f32(0.5 + 0.1*Math.exp(-4/20))) && near(cC.w, rC.w), 'reference ' + rC.w + ' child ' + cC.w);
  // an arrival in the step of the post spike: potentiation at the peak, no depression
  const rE = refPair({ d:5, stdpAt:1 }, 2, 7), cE = await childPair({ d:5, stdpAt:1 }, 2, 7);
  ok('an arrival in the step of the post spike potentiates at the peak on both engines', near(rE.w, f32(0.6)) && near(cE.w, rE.w), 'reference ' + rE.w + ' child ' + cE.w);
}

// ---- a recurrent network under noise, both readings on both engines
function mulberry(seed){
  let s = seed >>> 0;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const N = 200, NE = 160, MS = 2000;
const RS = NEURON_TYPES.findIndex(t => t.key === 'RS'), FS = NEURON_TYPES.findIndex(t => t.key === 'FS');
const rnd = mulberry(4321);
const ntype = Uint8Array.from({ length:N }, (_, i) => i < NE ? RS : FS);
const bias = Float32Array.from({ length:N }, () => 2 + 5*rnd());
const lists = [];
for(let i = 0; i < N; i++){ const o = [];
  for(let j = 0; j < N; j++) if(j !== i && rnd() < 0.08) o.push([j, i < NE ? 0.5 : -2, 1 + Math.floor(10*rnd())]);
  lists.push(o); }
const preStart = new Int32Array(N + 1);
for(let i = 0; i < N; i++) preStart[i + 1] = preStart[i] + lists[i].length;
const M = preStart[N];
const post = new Int32Array(M), w0 = new Float32Array(M), delay = new Uint8Array(M);
{ let k = 0; for(const o of lists) for(const [j, wt, d] of o){ post[k] = j; w0[k] = wt; delay[k] = d; k++; } }
function netInit(stdpAt){
  return { cmd:'init', protocol:PROTOCOL, count:N, ntype, bias, syn:0, psc:0, refrac:3, vmin:-90,
    plast:1, aP:0.05, aM:0.03, tauS:16.8, tauM:33.7, wmax:5, wdep:1, iEta:0.05, iRho:5, stdpAt,
    protocols:[{ idx:Uint32Array.from({ length:N }, (_, i) => i), amp:2, mode:3, period:100, width:10, t0:0, duration:0 }], inputs:[],
    preStart, post, w:w0.slice(), delay, pmask:new Uint8Array(M).fill(1),
    types:engineTypes(), seed:5, ...initState(5, ntype, NEURON_TYPES) };
}
function refNet(stdpAt){
  refSend(netInit(stdpAt));
  let spikes = 0;
  for(let t = 0; t < MS; t += 10) spikes += refSend({ cmd:'tick', steps:10 }).find(r => r.cmd === 'state').spikes;
  return { w:Float32Array.from(refSend({ cmd:'getWeights' }).find(r => r.cmd === 'weights').w), spikes };
}
async function childNet(stdpAt){
  const ch = await child();
  ch.send(netInit(stdpAt));
  let spikes = 0;
  for(let t = 0; t < MS; t += 10) spikes += Number((await ch.send({ cmd:'tick', steps:10 }, 'state')).spikes);
  const wm = await ch.send({ cmd:'getWeights' }, 'weights');
  ch.close();
  return { w:Float32Array.from(wm.w), spikes };
}
function corr(a, b){
  let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for(let s = 0; s < M; s++){ const x = a[s] - w0[s], y = b[s] - w0[s]; n++; sa += x; sb += y; saa += x*x; sbb += y*y; sab += x*y; }
  const cov = sab/n - (sa/n)*(sb/n), va = saa/n - (sa/n)*(sa/n), vb = sbb/n - (sb/n)*(sb/n);
  return va > 0 && vb > 0 ? cov/Math.sqrt(va*vb) : 0;
}
const moved = (a, b) => { let k = 0; for(let s = 0; s < M; s++) if(a[s] !== b[s]) k++; return k; };
const rSyn = refNet(1), rSoma = refNet(0), cSyn = await childNet(1), cSoma = await childNet(0);
const hz = r => (r.spikes/N/(MS/1000)).toFixed(1);
ok('the network fires and learns on both engines', rSyn.spikes > 0 && cSyn.spikes > 0 && moved(rSyn.w, w0) > M/4 && moved(cSyn.w, w0) > M/4,
  'reference ' + hz(rSyn) + ' Hz, child ' + hz(cSyn) + ' Hz; moved ' + moved(rSyn.w, w0) + ' and ' + moved(cSyn.w, w0) + ' of ' + M);
// Two seconds of a self-driven net under strong learning spread the engines' spike trains a little (delivery sums in a different order), so the bar is the soma reading's own agreement on this network, not a fixed figure: the new reading has to track the reference as well as the established one does.
const rAt = corr(rSyn.w, cSyn.w), rSo = corr(rSoma.w, cSoma.w);
ok('at the synapse the child learns what the reference learns, as closely as at the soma', rAt >= 0.95 && rAt >= rSo - 0.02, 'r ' + rAt.toFixed(4) + ' at the synapse, ' + rSo.toFixed(4) + ' at the soma');
ok('the two readings learn different weights on the reference', moved(rSyn.w, rSoma.w) > M/4, moved(rSyn.w, rSoma.w) + ' of ' + M + ' differ');
ok('and on the child', moved(cSyn.w, cSoma.w) > M/4, moved(cSyn.w, cSoma.w) + ' of ' + M + ' differ');
report('stdpattest');

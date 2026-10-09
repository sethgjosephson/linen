// Builds a deterministic test network, writes it as testnet.bin, then runs the reference engine (src/simworker.js, imported as-is through a global shim) on it and writes the per-neuron spike counts.
// The CUDA driver consumes the same binary, so the two engines are fed identical tissue and the only thing under comparison is the step arithmetic.
//
//   node cuda/dumpnet.mjs [n] [seconds] [outPrefix] [syn]
//
// syn 0 is kick synapses, syn 1 exponential with tauE 3 and tauI 8, the mode every real scenario runs.
//
// Binary layout, little-endian:
//   u32 n, u32 m, u32 syn, f32 tauE, f32 tauI,
//   f32 a[n], b[n], c[n], d[n], bias[n],
//   i32 preStart[n+1], i32 post[m], f32 w[m], u8 delay[m]
//
// a to d are each cell's classic RS or FS parameters; they stay in the layout so the readers and the saved dumps still read, and no reader passes them to an engine, which takes its rows from types (engineTypes, classicRow).
import fs from 'node:fs';
import { PROTOCOL } from '../src/protocol.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { initState } from '../src/rand.js';
import { engineTypes } from '../src/nodes.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const N = +(process.argv[2] || 4000);
const SECONDS = +(process.argv[3] || 10);
const PREFIX = process.argv[4] || 'testnet';
const SYN = +(process.argv[5] || 0);
const PLAST = +(process.argv[6] || 0);
// cons > 0 enables reference-weight consolidation (Zenke 2015 eq 16) in the reference run, so the CUDA port is checked against this engine with the term on, not only against itself with it off.
const CONS = +(process.argv[7] || 0);
const STP = +(process.argv[8] || 0);
// scale > 0 enables multiplicative synaptic scaling in the reference run, for the same reason as CONS.
const SCALE = +(process.argv[9] || 0);
// rhoMode > 0 measures a per-neuron homeostatic target over the first calS seconds instead of using the fixed one; the targets come back with the weights and are dumped for the CUDA comparison.
const RHOMODE = +(process.argv[10] || 0);
const DRIVE = +(process.argv[11] || 1);
// The engine seed, distinct from the tissue seed: it sets the initial membrane jitter.
// Varying it shows how much of a measurement is the network and how much is where the neurons happened to start.
const ESEED = +(process.argv[12] || 1);
const PSC = +(process.argv[13] || 0);      // exp current scale: 0 peak, 1 charge, 2 exact
// commit > 0 takes consolidated synapses out of the plastic pool, and tauCons is how long the reference has to be held past the midpoint of the well before that happens.
// The time is settable because the published twenty minutes would leave a ten second parity run with nothing consolidated to compare.
const COMMIT = +(process.argv[14] || 0);
const TAUCONS = +(process.argv[15] || 0);    // ms, 0 = the published 20 min
// pool < 0 puts the excitatory cells in two feedback inhibition pools at this gain (ENGINE.md section 14), so the CUDA port of the pool is checked against the reference with the term on; the arrays ride at the end of the dump and the driver reads them when they are there
const POOL = +(process.argv[16] || 0);
// ISCALE multiplies every current in the dump (bias and weights): the measured rows (MEASURED=1) divide currents by 20 to 100 pF, so tissue that fires on the classic rows is silent there without it
const ISCALE = +(process.env.ISCALE || 1) || 1;
const TAUE = 3, TAUI = 8;
// the learning lab's committed rule, so plasticity parity is measured on the configuration the real experiments run
const RULE = { aP:0.003, aM:0.001, tauS:20, wmax:60, wdep:1, iEta:0.01, iRho:2,
  trip:0.002, tauY:114, het:0.01, tin:0.00005 };

let z = 1234567;
const rnd = () => (z = (z*1664525 + 1013904223) >>> 0) / 4294967296;

// 80/20 excitatory/inhibitory, sparse random wiring, kick synapses, constant bias drive, no noise: a deterministic asynchronous regime is not needed for parity, only a stable nonzero one.
const NE = Math.round(N*0.8);
const a = new Float32Array(N), b = new Float32Array(N),
  c = new Float32Array(N), d = new Float32Array(N), bias = new Float32Array(N);
for(let i=0;i<N;i++){
  const exc = i < NE;
  a[i] = exc ? 0.02 : 0.1; b[i] = 0.2;
  c[i] = -65; d[i] = exc ? 8 : 2;
  // heterogeneous drive so rates spread rather than locking into one ISI.
  // DRIVE scales it: measured set points are only an informative comparison when the rates land between the 0.25 and 30 Hz clamps, and at the default drive this tissue runs near 50 Hz, where almost every neuron pins to the ceiling and the two engines agree for the wrong reason.
  bias[i] = ISCALE*DRIVE*((exc ? 3.5 : 2.5) + 3.0*rnd());
}
const pre = [], post = [], w = [], del = [];
if(N <= 10000){
  // pairwise Bernoulli, kept for reproducibility of the parity network
  const P = Math.min(0.04, 80/N);
  for(let i=0;i<N;i++) for(let j=0;j<N;j++){
    if(i === j || rnd() >= P) continue;
    pre.push(i); post.push(j);
    w.push(ISCALE*(i < NE ? 5*(0.5 + rnd()) : -9*(0.5 + rnd())));
    del.push(1 + Math.floor(rnd()*8));
  }
} else {
  // fixed out-degree sampling, O(N*K) rather than O(N*N), for the throughput networks where the pairwise loop would take hours
  const K = 80;
  for(let i=0;i<N;i++) for(let k=0;k<K;k++){
    let j = Math.floor(rnd()*N);
    if(j === i) j = (j+1)%N;
    pre.push(i); post.push(j);
    w.push(ISCALE*(i < NE ? 5*(0.5 + rnd()) : -9*(0.5 + rnd())));
    del.push(1 + Math.floor(rnd()*8));
  }
}
const m = pre.length;
const order = [...Array(m).keys()].sort((x, y) => pre[x]-pre[y] || post[x]-post[y]);
const preStart = new Int32Array(N+1);
for(const s of order) preStart[pre[s]+1]++;
for(let i=0;i<N;i++) preStart[i+1] += preStart[i];
const postA = Int32Array.from(order.map(s => post[s]));
const wA = Float32Array.from(order.map(s => w[s]));
// The synthetic weights above are charges.
// Read as peak currents (psc 0, the default) the same numbers deliver tau times the charge and put this network into a chaotic regime where any perturbation decorrelates per-neuron spike counts (measured 2026-09-01: the reference against itself at a second noise seed, count r 0.57 against 0.998 in charge scaling), which makes a parity number meaningless.
// Converting them keeps both conventions testing the same dynamics.
if(SYN === 1 && PSC === 0)
  for(let s=0;s<wA.length;s++)
    wA[s] *= wA[s] > 0 ? (1 - Math.exp(-1/TAUE)) : (1 - Math.exp(-1/TAUI));
// under psc 2 the charge is w tau exactly, so a charge becomes a peak by tau
if(SYN === 1 && PSC === 2)
  for(let s=0;s<wA.length;s++)
    wA[s] /= wA[s] > 0 ? TAUE : TAUI;
const delayA = Uint8Array.from(order.map(s => del[s]));
const poolA = new Uint16Array(N);
const poolK = new Float32Array(POOL < 0 ? 3 : 1);
if(POOL < 0){
  const half = NE >> 1;
  for(let i=0;i<NE;i++) poolA[i] = i < half ? 1 : 2;
  poolK[1] = POOL*1000/half; poolK[2] = POOL*1000/(NE - half);
}

const parts = [];
const head = new ArrayBuffer(68);
const hv = new DataView(head);
hv.setUint32(0, N, true); hv.setUint32(4, m, true); hv.setUint32(8, SYN, true);
hv.setFloat32(12, TAUE, true); hv.setFloat32(16, TAUI, true);
hv.setUint32(20, PLAST, true); hv.setUint32(24, RULE.wdep, true);
hv.setFloat32(28, RULE.aP, true); hv.setFloat32(32, RULE.aM, true);
hv.setFloat32(36, RULE.tauS, true); hv.setFloat32(40, RULE.wmax, true);
hv.setFloat32(44, RULE.iEta, true); hv.setFloat32(48, RULE.iRho, true);
hv.setFloat32(52, RULE.trip, true); hv.setFloat32(56, RULE.tauY, true);
hv.setFloat32(60, RULE.het, true); hv.setFloat32(64, RULE.tin, true);
parts.push(Buffer.from(head));
for(const arr of [a, b, c, d, bias, preStart, postA, wA, delayA,
    Uint32Array.of(poolK.length), poolA, poolK])
  parts.push(Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
fs.writeFileSync(path.join(HERE, PREFIX + '.bin'), Buffer.concat(parts));
console.log('wrote ' + PREFIX + '.bin: ' + N + ' neurons, ' + m + ' synapses, ' + (SYN ? 'exp' : 'kick') + ' synapses');

// ---- reference run: the engine itself, in-process ----------------------
const replies = [];
globalThis.postMessage = (d) => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const send = (d) => globalThis.onmessage({ data: d });

const ntype = new Uint8Array(N);
// GRADED=1 makes the second class the graded fly relay (FG)
const GRADED = +(process.env.GRADED || 0) === 1;
// CHAN=<tau> puts every excitatory synapse on receptor channel 2 with that
// time constant (the top three bits of the plasticity byte), both sides
const CHAN = +(process.env.CHAN || 0);
// COND=1 runs the conductance synapse (mode 2) with the checkpoint reversals
const COND = +(process.env.COND || 0) === 1;
const chanBytes = (n, ntype, preStart, m) => { const pm = new Uint8Array(m).fill(1);
  if(CHAN > 0) for(let i = 0; i < n; i++) if(ntype[i] === 0) pm.fill(1 | (2 << 5), preStart[i], preStart[i+1]);
  return pm; };
// The rows are RS and FS from the type table: the classic rows (the 2003 presets as 2007 rows, src/form.js) by default.
// MEASURED=1 runs the measured chapter 8 rows instead (rows 10 and 13), whose currents are picoamps; ISCALE lifts the drive for them.
const MEASURED = +(process.env.MEASURED || 0) === 1;
const CLASSES = [engineTypes()[MEASURED ? 10 : 0], GRADED ? engineTypes().find((t, i) => i === 8) : engineTypes()[MEASURED ? 13 : 3]];
for(let i=0;i<N;i++) ntype[i] = i < NE ? 0 : 1;

send({ cmd:'init', protocol:PROTOCOL, count:N, ntype, bias,
  syn:COND ? 2 : SYN, eRevE:0, eRevI:-70, psc:PSC, tauE:TAUE, tauI:TAUI, protocols:[], plast:PLAST,
  aP:RULE.aP, aM:RULE.aM, tauS:RULE.tauS, wmax:RULE.wmax, wdep:RULE.wdep,
  iEta:RULE.iEta, iRho:RULE.iRho, rhoMode:RHOMODE, calS:10, scale:SCALE, sEta:0.001,
  trip:RULE.trip, tauY:RULE.tauY, het:RULE.het, tin:RULE.tin,
  cons:CONS, consW:0, consP:10,   // 0 = the scaled default, a tenth of wmax
  commit:COMMIT, tauCons:TAUCONS,
  stp:STP, stpOrder:+(process.env.STPORDER || 0), stpU:0.2, stpTauD:200, stpTauF:600,
  seed:ESEED, ...initState(ESEED, ntype, CLASSES),
  pmask:chanBytes(N, ntype, preStart, m), ...(CHAN > 0 ? { chanTau:[CHAN] } : {}),
  ...(POOL < 0 ? { pool:poolA, poolK } : {}),
  inputs:[], preStart, post:postA, w:wA.slice(), delay:delayA, types:CLASSES });

// one-step ticks so the fired flags are exact per-step spike counts, the same thing the CUDA driver accumulates
const MS = SECONDS*1000;
const counts = new Float64Array(N);
let spikes = 0;
const t0 = performance.now();
for(let t=0;t<MS;t++){
  replies.length = 0;
  send({ cmd:'tick', steps:1 });
  const st = replies.find(r => r.cmd === 'state');
  spikes += st.spikes;
  for(let i=0;i<N;i++) if(st.fired[i]) counts[i]++;
}
const wall = (performance.now()-t0)/1000;
const rate = spikes/N/SECONDS;
console.log('reference: ' + spikes + ' spikes, ' + rate.toFixed(2) + ' Hz mean, ' +
  (MS/1000/wall).toFixed(1) + 'x realtime, ' + (N*MS/wall/1e6).toFixed(1) + ' M neuron-steps/s');
fs.writeFileSync(path.join(HERE, PREFIX + '-ref.json'), JSON.stringify({
  n:N, m, seconds:SECONDS, spikes, rateHz:+rate.toFixed(4),
  counts:[...counts] }));
console.log('wrote ' + PREFIX + '-ref.json');
if(PLAST){
  replies.length = 0;
  send({ cmd:'getWeights' });
  const wr = replies.find(r => r.cmd === 'weights');
  fs.writeFileSync(path.join(HERE, PREFIX + '-ref-w.bin'),
    Buffer.from(wr.w.buffer, wr.w.byteOffset, wr.w.byteLength));
  console.log('wrote ' + PREFIX + '-ref-w.bin');
  if(wr && wr.rho){
    fs.writeFileSync(path.join(HERE, PREFIX + '-ref-rho.bin'),
      Buffer.from(wr.rho.buffer, wr.rho.byteOffset, wr.rho.byteLength));
    console.log('wrote ' + PREFIX + '-ref-rho.bin');
  }
}

// Does the reference engine's code do what its rules say?
//
// This runs small networks on the reference engine (src/simworker.js, in process) and on a second implementation of its step (tools/brian_ref.py: Python transcribed from simworker.js, run on Brian 2's spike queues, delays, synapse indexing and scheduling), and requires every spike train, and under plasticity every learned weight, to be identical.
// Both store float32 and step the same arithmetic, so exact agreement is the expected result, and any difference is a difference between the engine's code and that reading of it rather than rounding.
//
// Agreement shows the engine's delivery timing, routing, trace timing, update order and rule lookups match a separately structured implementation, and the suite catches any change to the engine's results.
// It does not show the equations match the literature (validate.html's published-result groups ask that), and it does not exercise the wiring or the initial state: the networks are made here and rand.js initState is reused.
//
// Cases, one mechanism each, then some together:
//   membrane        every spiking type row, from hyperpolarizing to strong drive
//   floor           a membrane floor other than the default, under strong
//                   hyperpolarizing drive
//   refractory      the absolute refractory count under strong drive
//   kick            a 400-cell recurrent network, kick synapses, delays 1 to 10 ms
//   kick+refrac     the same network with a 2 ms refractory period
//   noise           the per-ms noise alone, on three rows
//   kick+noise      the kick network with noise on every cell
//   protocols       a DC step, a pulse train with per-cell gains, a ramp and
//                   noise gated to a window, on the kick network
//   input frames    two framed input maps with gains changed between ticks
//   pools           feedback inhibition pools on the kick network
//   exp peak        exponential synapses, w the peak current (psc 0)
//   exp charge      the same weights, w the charge per spike (psc 1)
//   exp exact       the same weights, the current integrated exactly over each step (psc 2)
//   pools exp       pools feeding the inhibitory conductance
//   conductance     every channel a conductance toward its reversal (syn 2)
//   channels        two receptor channels beyond E and I, exponential synapses
//   channels cond   the same channels as conductances
//   everything      conductance, channels, noise and a refractory period
//   graded off      graded relay cells between a spiking network and targets,
//                   their output weights zero
//   graded          the same with graded release reaching the targets
// and on a 200-cell network with delays 1 to 4 ms and a 3 ms refractory count, a tenth of its synapses frozen:
//   static short    no plasticity, the base the next cases differ from
//   stdp soft       pair STDP with soft bounds, tin, the Vogels rule
//   stdp hard       the same with hard bounds
//   triplet+het     the triplet term and the heterosynaptic gate
//   het saturating  the gate past its ceiling of 1
//   stp             short-term plasticity alone
//   stp norm        short-term plasticity normalized to the rested release
//   stdp+stp exp    plasticity and short-term plasticity on exponential synapses
//   scaling         synaptic scaling once per simulated second
//   set points      measured set points after a calibration window
//   consolidation   the reference weight through the double well
//   commit          consolidated synapses leaving the plastic pool
//   two rules       two rows of the rule table on one network
//
// A case that switches a term on also checks that its spike trains or its weights differ from the case without it.
// Agreement alone would pass if both sides ignored the term.
//
// Brian 2 runs in .venv-analysis and is not a dependency of the app: with it absent this says the comparison DID NOT RUN rather than passing quietly.
// It takes a few minutes, so tools/test.mjs runs it in the full suite and not in the pre-commit set.
//
//   node tools/brianref.mjs [substring]   (runs the matching cases and the
//                                          cases they are compared against)

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ok, report } from './harness.mjs';
import { PROTOCOL } from '../src/protocol.js';
import { engineTypes, NEURON_TYPES, RULE_FIELDS, gradedArrays } from '../src/nodes.js';
import { initState, noiseDraw } from '../src/rand.js';
import { caseFromInit, floorOf } from '../src/briancase.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const PY = [path.join(ROOT, '.venv-analysis', 'Scripts', 'python.exe'),
            path.join(ROOT, '.venv-analysis', 'bin', 'python')].find(p => fs.existsSync(p));
const ONLY = process.argv[2] || '';

let brianVersion = null;
if(PY){
  try { brianVersion = execFileSync(PY, ['-c', 'import brian2; print(brian2.__version__)'],
    { encoding:'utf8', stdio:['ignore', 'pipe', 'ignore'] }).trim(); }
  catch(e){ brianVersion = null; }
}
if(!brianVersion){
  console.log('brianref: Brian 2 not found in .venv-analysis, so the comparison DID NOT RUN.');
  process.exit(0);
}

// ---- the reference engine, in process (as cuda/dumpnet.mjs runs it) --------
const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const send = d => globalThis.onmessage({ data:d });

const TYPES = engineTypes();
const RS = RULE_FIELDS.length;
const key = k => NEURON_TYPES.findIndex(t => t.key === k);
const SPIKING = TYPES.map((t, i) => i).filter(i => TYPES[i].f7 && !TYPES[i].f7.graded);
const isDynamic = c => !!(c.rule && (c.rule.plast || c.rule.stp));

// a case's rule table, when it declares rules: row 0 unused, rows from 1
function ruleTableOf(r){
  if(!r || !r.rules) return null;
  const RT = new Float32Array((r.rules.length + 1)*RS);
  r.rules.forEach((row, k) => RULE_FIELDS.forEach((f, x) => { RT[(k + 1)*RS + x] = +(row[f] ?? 0); }));
  return RT;
}

function runEngine(c){
  const n = c.ntype.length;
  const st = initState(c.seed, c.ntype, TYPES);
  const net = c.net || { preStart:new Int32Array(n + 1), post:new Int32Array(0),
    w:new Float32Array(0), delay:new Uint8Array(0), pmask:null };
  const { rules, ...ruleFields } = c.rule || {};
  const RT = ruleTableOf(c.rule);
  replies.length = 0;
  // the init message, kept: Brian's case is built from this same object
  const init = { cmd:'init', protocol:PROTOCOL, count:n, ntype:c.ntype, bias:c.bias,
    syn:c.syn || 0, psc:c.psc || 0, tauE:c.tauE || 3, tauI:c.tauI || 8,
    eRevE:c.eRevE ?? 0, eRevI:c.eRevI ?? -70,
    ...(c.chanTau ? { chanTau:c.chanTau, chanErev:c.chanErev } : {}),
    protocols:c.protocols || [], inputs:c.inputs || [],
    ...(c.pools ? { pool:c.pools.pool, poolK:c.pools.poolK } : {}),
    plast:0, ...ruleFields,
    ...(RT ? { ruleTable:RT, ruleCount:RT.length/RS } : {}),
    refrac:c.refrac, ...(c.vmin === undefined ? {} : { vmin:c.vmin }),
    seed:c.seed, ...st, preStart:net.preStart, post:net.post, w:net.w.slice(),
    delay:net.delay, ...(net.pmask ? { pmask:net.pmask } : {}), types:TYPES };
  // the engine runs in this process and learns in place on the weight array it is handed, so it gets a copy and init keeps the wired weights
  send({ ...init, w:init.w.slice() });
  const err = replies.find(r => r.cmd === 'error');
  if(err) throw new Error(c.name + ': ' + err.message);
  send({ cmd:'sendV', on:true });
  // one-millisecond ticks, so every spike has its own step; an input frame sent before tick t sets the gains step t runs with
  const times = Array.from({ length:n }, () => []);
  const vLow = new Float64Array(n).fill(Infinity);
  const frames = c.frames || [];
  let fi = 0;
  for(let t = 0; t < c.ms; t++){
    while(fi < frames.length && frames[fi].t === t){
      send({ cmd:'inputFrame', mi:frames[fi].mi, gains:frames[fi].gains });
      fi++;
    }
    replies.length = 0;
    send({ cmd:'tick', steps:1 });
    const s = replies.find(r => r.cmd === 'state');
    for(let i = 0; i < n; i++){
      if(s.fired[i]) times[i].push(t);
      if(s.v[i] < vLow[i]) vLow[i] = s.v[i];
    }
  }
  let weights = null;
  if(isDynamic(c)){
    replies.length = 0;
    send({ cmd:'getWeights' });
    const r = replies.find(x => x.cmd === 'weights');
    weights = { w:r.w, rho:r.rho };
  }
  return { times, vLow, st, weights, init };
}

// ---- Brian 2 on the same network ------------------------------------------
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'brianref-'));
const NOISE_PROBE = [[0, 0, 1], [5, 17, 3], [399, 1999, 3], [123456, 987654, 4294967295]];

function runBrian(c, init){
  // the case from the init the engine ran (briancase.js), which is what the app exports; the frames are the harness's input frames
  const file = caseFromInit(init, { ms:c.ms, frames:c.frames, noiseProbe:NOISE_PROBE });
  if(!c.net){ delete file.pre; delete file.post; delete file.w; delete file.delay; delete file.chan; delete file.rule; }
  const inFile = path.join(TMP, c.name.replace(/\W+/g, '_') + '.json');
  const outFile = inFile.replace(/\.json$/, '-out.json');
  fs.writeFileSync(inFile, JSON.stringify(file));
  try {
    execFileSync(PY, [path.join(HERE, 'brian_ref.py'), inFile, outFile],
      { stdio:['ignore', 'ignore', 'pipe'], timeout:900000, encoding:'utf8' });
  } catch(e){
    // the lines that say what went wrong, not Brian's advice or the traceback frames
    const said = String(e.stderr || e.message).split(/\r?\n/)
      .filter(l => /Error|error:|cannot|invalid/i.test(l) && !/^\s*File /.test(l) && !/brian\.discourse/.test(l));
    throw new Error('Brian 2 failed on case "' + c.name + '" (input kept at ' + inFile + '):\n  ' +
      (said.length ? said.slice(-4).join('\n  ') : String(e.stderr || e.message).slice(-800)));
  }
  return JSON.parse(fs.readFileSync(outFile, 'utf8'));
}

// ---- cases ------------------------------------------------------------------
// Currents per row as multiples of the quadratic's rheobase with u held, k (vt - vr)^2 / 4, so every row is driven from well below firing to far above it on its own scale (picoamps on a measured row, the classic unit on a classic one).
// RZ's rest and threshold coincide, so it gets half its capacitance as the unit instead of zero.
const unitOf = t => { const f = TYPES[t].f7; return Math.max(f.k*(f.vt - f.vr)**2/4, 0.5*f.C); };
function grid(types, levels){
  const ntype = [], bias = [], level = [];
  for(const t of types) for(const m of levels){ ntype.push(t); bias.push(m*unitOf(t)); level.push(m); }
  return { ntype:Uint8Array.from(ntype), bias:Float32Array.from(bias), level };
}
// a noise protocol per row, amplitude a multiple of that row's unit
function noiseGrid(types, levels, ampMul){
  const g = grid(types, levels), byRow = new Map();
  g.ntype.forEach((t, i) => { if(!byRow.has(t)) byRow.set(t, []); byRow.get(t).push(i); });
  g.protocols = [...byRow].map(([t, idx]) => ({ idx:Uint32Array.from(idx), amp:ampMul*unitOf(t),
    mode:3, period:100, width:10, t0:0, duration:0 }));
  return g;
}
const range = (a, b) => Uint32Array.from({ length:b - a }, (_, i) => a + i);
const noiseOn = (n, amp) => [{ idx:range(0, n), amp, mode:3, period:100, width:10, t0:0, duration:0 }];

function mulberry(seed){
  let s = seed >>> 0;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// CSR from per-presynaptic lists of [post, weight, delay]
function csr(out, pmaskOf, NE){
  const N = out.length, preStart = new Int32Array(N + 1);
  for(let i = 0; i < N; i++) preStart[i + 1] = preStart[i] + out[i].length;
  const m = preStart[N];
  const post = new Int32Array(m), w = new Float32Array(m), delay = new Uint8Array(m);
  let k = 0;
  for(const o of out) for(const [j, wt, d] of o){ post[k] = j; w[k] = wt; delay[k] = d; k++; }
  return { preStart, post, w, delay, pmask:pmaskOf ? pmaskOf(preStart, N, NE, m) : null };
}

// A recurrent network: seeded, so it is the same tissue on every run; the draws do not depend on the weights, so every variant is the same wiring.
function network({ N = 400, NE = 320, P = 0.05, seed = 12345, maxDelay = 10, we, wi, pmaskOf = null }){
  const rnd = mulberry(seed), RSr = key('RS'), FSr = key('FS');
  const ntype = Uint8Array.from({ length:N }, (_, i) => i < NE ? RSr : FSr);
  const bias = Float32Array.from({ length:N }, () => 2 + 5*rnd());
  const out = [];
  for(let i = 0; i < N; i++){ const o = [];
    for(let j = 0; j < N; j++) if(j !== i && rnd() < P) o.push([j, i < NE ? we : wi, 1 + Math.floor(maxDelay*rnd())]);
    out.push(o); }
  return { ntype, bias, net:csr(out, pmaskOf, NE) };
}
// every other excitatory cell sends on channel 2, every other inhibitory one on 3
const channelBits = (preStart, N, NE, m) => {
  const pm = new Uint8Array(m).fill(1);
  for(let i = 1; i < N; i += 2) pm.fill(1 | ((i < NE ? 2 : 3) << 5), preStart[i], preStart[i + 1]);
  return pm;
};
// a tenth of the synapses frozen (rule 0); with two rules, a third on rule 2
const ruleBits = two => (preStart, N, NE, m) =>
  Uint8Array.from({ length:m }, (_, q) => q % 10 === 0 ? 0 : (two && q % 3 === 1 ? 2 : 1));

// Graded relays between a spiking network and targets: 100 spiking cells (recurrent) drive 60 graded FG cells; each graded cell sends one synapse, delay 1 to 3, to a target of its own that receives nothing else, so a target's input in a step is one release and no order of summation enters.
function gradedNet({ wg }){
  const rnd = mulberry(4242), NA = 100, NG = 60, NT = 60, N = NA + NG + NT;
  const RSr = key('RS'), FGr = key('FG');
  const ntype = Uint8Array.from({ length:N }, (_, i) => i < NA ? RSr : (i < NA + NG ? FGr : RSr));
  const bias = Float32Array.from({ length:N }, (_, i) => i < NA ? 2 + 5*rnd() : (i < NA + NG ? 1 + 4*rnd() : 1.5 + rnd()));
  const out = Array.from({ length:N }, () => []);
  for(let i = 0; i < NA; i++){
    for(let j = 0; j < NA; j++) if(j !== i && rnd() < 0.05) out[i].push([j, 0.5, 1 + Math.floor(10*rnd())]);
    for(let g = 0; g < NG; g++) if(rnd() < 0.1) out[i].push([NA + g, 3, 1 + Math.floor(5*rnd())]);
  }
  for(let g = 0; g < NG; g++) out[NA + g].push([NA + NG + g, wg, 1 + (g % 3)]);
  return { ntype, bias, net:csr(out, null, 0), gradedFrom:NA, targetsFrom:NA + NG };
}

// Two framed input maps: 8 channels over 160 cells, two entries a cell, and 4 channels over 80 cells; gains change every 100 ms on the first map and every 200 ms on the second, about a third of them zero.
function inputMaps(){
  const r = mulberry(99);
  const mk = (NC, cells, perCell, amp) => {
    const lists = Array.from({ length:NC }, () => []);
    for(const i of cells) for(let q = 0; q < perCell; q++) lists[Math.floor(NC*r())].push([i, 0.5 + r()]);
    const chStart = new Int32Array(NC + 1);
    lists.forEach((l, ch) => { chStart[ch + 1] = chStart[ch] + l.length; });
    const chIdx = new Int32Array(chStart[NC]), chW = new Float32Array(chStart[NC]);
    let k = 0;
    for(const l of lists) for(const [i, wv] of l){ chIdx[k] = i; chW[k] = wv; k++; }
    return { chStart, chIdx, chW, amp };
  };
  return [mk(8, range(0, 160), 2, 2), mk(4, range(160, 240), 1, 1.5)];
}
function inputFrames(maps, ms){
  const r = mulberry(123), frames = [];
  for(let t = 0; t < ms; t += 100) maps.forEach((m, mi) => {
    if(mi === 1 && t % 200) return;
    frames.push({ t, mi, gains:Float32Array.from({ length:m.chStart.length - 1 }, () => r() < 0.3 ? 0 : 2*r()) });
  });
  return frames;
}

const ROWS3 = [key('RS'), key('RS07'), key('FL')];
const kick = network({ we:0.5, wi:-2 });
const expNet = network({ we:0.17, wi:-0.25 });
const condNet = network({ we:0.003, wi:-0.05 });
const chanExp = network({ we:0.17, wi:-0.25, pmaskOf:channelBits });
const chanCond = network({ we:0.003, wi:-0.05, pmaskOf:channelBits });
const CHANNELS = { chanTau:[20, 100], chanErev:[0, -90] };
const PROTOCOLS = [
  { idx:range(0, 100), amp:3, mode:0, period:100, width:10, t0:500, duration:500 },
  { idx:range(100, 200), gain:Float32Array.from({ length:100 }, (_, q) => 0.5 + (q % 7)/7), amp:5,
    mode:1, period:100, width:20, t0:200, duration:0 },
  { idx:range(200, 300), amp:4, mode:2, period:1500, width:10, t0:0, duration:0 },
  { idx:range(0, 400), amp:3, mode:3, period:100, width:10, t0:300, duration:1000 },
];
const POOLS = { pool:Uint16Array.from({ length:400 }, (_, i) => i < 160 ? 1 : (i < 320 ? 2 : 0)),
  poolK:Float32Array.from([0, -0.3, -0.5]) };
const MAPS = inputMaps(), FRAMES = inputFrames(MAPS, 2000);
// the plastic network: delays up to 4 ms under a 3 ms refractory count, so every presynaptic interval covers the longest delay
const PLASTIC = { N:200, NE:160, P:0.1, seed:777, maxDelay:4, we:0.5, wi:-2 };
const pnet = network({ ...PLASTIC, pmaskOf:ruleBits(false) });
const pnet2 = network({ ...PLASTIC, pmaskOf:ruleBits(true) });
const STDP = { plast:1, aP:0.05, aM:0.03, tauS:16.8, tauM:33.7, wmax:5, wdep:1, iEta:0.05, iRho:5, tin:0.001 };
const TRIPHET = { ...STDP, trip:0.02, het:0.01 };
const CONS = { ...TRIPHET, cons:1, consW:1, consP:10, tauCons:2000, consStep:50 };
const STP = { stp:1, stpU:0.2, stpTauD:200, stpTauF:600 };
const P2 = { ms:2000, refrac:3, seed:3 };

const ALL = [
  { name:'membrane', ms:3000, refrac:0, seed:7, ...grid(SPIKING, [-40, -3, 0, 0.9, 1.5, 3, 8, 40]) },
  { name:'floor', ms:2000, refrac:0, vmin:-75, seed:11, ...grid(ROWS3, [-40, -3, 3, 8]) },
  { name:'refractory', ms:2000, refrac:3, seed:5, ...grid(SPIKING, [3, 8, 40]) },
  { name:'kick', ms:2000, refrac:0, seed:3, ...kick },
  { name:'kick+refrac', ms:2000, refrac:2, seed:3, ...kick },
  { name:'noise', ms:2000, refrac:0, seed:9, ...noiseGrid(ROWS3, [0, 0.5, 1], 4) },
  { name:'kick+noise', ms:2000, refrac:0, seed:3, protocols:noiseOn(400, 2), differsFrom:'kick', ...kick },
  { name:'protocols', ms:2000, refrac:0, seed:3, protocols:PROTOCOLS, differsFrom:'kick', ...kick },
  { name:'input frames', ms:2000, refrac:0, seed:3, inputs:MAPS, frames:FRAMES, differsFrom:'kick', ...kick },
  { name:'pools', ms:2000, refrac:0, seed:3, pools:POOLS, differsFrom:'kick', ...kick },
  { name:'exp peak', ms:2000, refrac:0, seed:3, syn:1, psc:0, ...expNet },
  { name:'exp charge', ms:2000, refrac:0, seed:3, syn:1, psc:1, differsFrom:'exp peak', ...expNet },
  { name:'exp exact', ms:2000, refrac:0, seed:3, syn:1, psc:2, differsFrom:'exp peak', ...expNet },
  { name:'pools exp', ms:2000, refrac:0, seed:3, syn:1, psc:0, pools:POOLS, differsFrom:'exp peak', ...expNet },
  { name:'conductance', ms:2000, refrac:0, seed:3, syn:2, ...condNet },
  { name:'channels', ms:2000, refrac:0, seed:3, syn:1, psc:0, ...CHANNELS, differsFrom:'exp peak', ...chanExp },
  { name:'channels cond', ms:2000, refrac:0, seed:3, syn:2, ...CHANNELS, differsFrom:'conductance', ...chanCond },
  { name:'everything', ms:2000, refrac:2, seed:3, syn:2, ...CHANNELS, protocols:noiseOn(400, 2),
    differsFrom:'channels cond', ...chanCond },
  { name:'graded off', ms:2000, refrac:0, seed:5, graded:true, ...gradedNet({ wg:0 }) },
  { name:'graded', ms:2000, refrac:0, seed:5, graded:true, differsFrom:'graded off', ...gradedNet({ wg:5 }) },
  { name:'static short', ...P2, ...pnet },
  { name:'stdp soft', ...P2, rule:STDP, coincide:true, differsFrom:'static short', ...pnet },
  { name:'stdp hard', ...P2, rule:{ ...STDP, wdep:0 }, weightsDifferFrom:'stdp soft', ...pnet },
  { name:'triplet+het', ...P2, rule:TRIPHET, weightsDifferFrom:'stdp soft', ...pnet },
  { name:'het saturating', ...P2, rule:{ ...STDP, trip:0.1, het:5 }, weightsDifferFrom:'triplet+het', ...pnet },
  { name:'stp', ...P2, rule:STP, differsFrom:'static short', ...pnet },
  { name:'stp norm', ...P2, rule:{ ...STP, stpNorm:1 }, differsFrom:'stp', ...pnet },
  { name:'stp order', ...P2, rule:{ ...STP, stpOrder:1 }, differsFrom:'stp', ...pnet },
  { name:'stp order norm', ...P2, rule:{ ...STP, stpOrder:1, stpNorm:1 }, differsFrom:'stp order', ...pnet },
  { name:'stdp+stp exp', ...P2, syn:1, psc:1, rule:{ ...STDP, ...STP }, ...pnet },
  { name:'scaling', ...P2, rule:{ ...STDP, scale:1, sEta:0.05 }, weightsDifferFrom:'stdp soft', ...pnet },
  { name:'set points', ...P2, rule:{ ...STDP, scale:1, sEta:0.05, rhoMode:1, calS:0.5 },
    weightsDifferFrom:'scaling', ...pnet },
  { name:'consolidation', ...P2, rule:CONS, weightsDifferFrom:'triplet+het', ...pnet },
  { name:'commit', ...P2, rule:{ ...CONS, commit:1 }, weightsDifferFrom:'consolidation', ...pnet },
  { name:'two rules', ...P2, rule:{ ...STDP, rules:[
      { aP:0.05, aM:0.03, wmax:5, wdep:1, tin:0.001, iEta:0.05, consW:0.5, consP:10 },
      { aP:0.02, aM:0.05, wmax:3, wdep:0, iEta:0.02, consW:0.3, consP:10 }] },
    weightsDifferFrom:'stdp soft', ...pnet2 },
];
// a substring runs its matches and the cases they are compared against
const wanted = new Set(ALL.filter(c => !ONLY || c.name.includes(ONLY)).map(c => c.name));
for(const c of ALL) if(wanted.has(c.name)){
  if(c.differsFrom) wanted.add(c.differsFrom);
  if(c.weightsDifferFrom) wanted.add(c.weightsDifferFrom);
}
const CASES = ALL.filter(c => wanted.has(c.name));

const sameTrains = (A, B) => A.length === B.length &&
  A.every((a, i) => a.length === B[i].length && a.every((t, j) => t === B[i][j]));
const sameArray = (A, B) => A.length === B.length && A.every((x, i) => x === B[i]);

console.log('brianref: the reference engine against brian2 ' + brianVersion);
const trainsOf = new Map(), weightsOf = new Map();
for(const c of CASES){
  const n = c.ntype.length;
  if(c.graded){
    // a graded synapse's target receives nothing else, so no summation order enters
    const G = gradedArrays(c.ntype, TYPES), inbound = new Map();
    for(let i = 0; i < n; i++) for(let q = c.net.preStart[i]; q < c.net.preStart[i + 1]; q++){
      const j = c.net.post[q];
      if(!inbound.has(j)) inbound.set(j, { graded:0, other:0 });
      inbound.get(j)[G.grd[i] ? 'graded' : 'other']++;
    }
    const mixed = [...inbound.values()].filter(v => v.graded && (v.graded > 1 || v.other)).length;
    ok(c.name + ': every graded target receives one release and nothing else', mixed === 0,
      mixed + ' targets mix a release with other input: the case cannot compare their sums');
  }
  const eng = runEngine(c);
  trainsOf.set(c.name, eng.times);
  if(eng.weights) weightsOf.set(c.name, eng.weights.w);
  const br = runBrian(c, eng.init);
  let same = 0, first = Infinity, spikes = 0;
  for(let i = 0; i < n; i++){
    const e = eng.times[i], b = br.times[i];
    spikes += e.length;
    if(e.length === b.length && e.every((t, j) => t === b[j])){ same++; continue; }
    const j = e.findIndex((t, x) => t !== b[x]);
    first = Math.min(first, j < 0 ? (b[e.length] ?? Infinity) : Math.min(e[j], b[j] ?? Infinity));
  }
  const hz = spikes/n/(c.ms/1000);
  let wline = '';
  if(eng.weights){
    const W = eng.weights.w, B = br.w || [];
    let wsame = 0, moved = 0, firstW = -1;
    for(let q = 0; q < W.length; q++){
      if(W[q] === B[q]) wsame++; else if(firstW < 0) firstW = q;
      if(W[q] !== c.net.w[q]) moved++;
    }
    wline = '   weights ' + wsame + '/' + W.length + ' (' + moved + ' moved)';
    ok(c.name + ': every weight identical', wsame === W.length,
      (W.length - wsame) + ' differ, the first synapse ' + firstW + ': engine ' + W[firstW] + ' brian ' + B[firstW]);
    if(c.rule.plast) ok(c.name + ': weights move', moved > 0, 'no weight changed, so the rules were not exercised');
    if(eng.weights.rho){
      ok(c.name + ': every measured set point identical', sameArray([...eng.weights.rho], br.rho || []),
        'engine ' + [...eng.weights.rho].slice(0, 4).join(', ') + ' brian ' + (br.rho || []).slice(0, 4).join(', '));
    }
    // the stash that carries a delivery is exact only while intervals cover delays
    let minIsi = Infinity;
    for(const t of eng.times) for(let j = 1; j < t.length; j++) minIsi = Math.min(minIsi, t[j] - t[j - 1]);
    const maxDelay = Math.max(...c.net.delay);
    ok(c.name + ': every presynaptic interval covers the longest delay', minIsi >= maxDelay,
      'shortest interval ' + minIsi + ' ms against a delay of ' + maxDelay + ' ms: the case cannot compare deliveries');
    if(c.coincide){
      // same-millisecond pre and post on a plastic synapse, the path the masks reproduce
      const sets = eng.times.map(ts => new Set(ts));
      let low = 0, high = 0;
      for(let i = 0; i < n; i++) for(let q = c.net.preStart[i]; q < c.net.preStart[i + 1]; q++){
        if(c.net.pmask && !(c.net.pmask[q] & 31)) continue;
        for(const t of eng.times[i]) if(sets[c.net.post[q]].has(t)){ if(i < c.net.post[q]) low++; else high++; }
      }
      wline += '   same-ms pairs ' + low + '/' + high;
      ok(c.name + ': same-millisecond pre and post spikes occur in both index orders', low > 0 && high > 0,
        'pre lower ' + low + ', post lower ' + high);
    }
  }
  console.log('  ' + c.name.padEnd(15) + String(n).padStart(4) + ' cells ' +
    String(spikes).padStart(7) + ' spikes ' + hz.toFixed(1).padStart(6) + ' Hz   identical ' +
    same + '/' + n + wline + '   brian ' + br.wallS.toFixed(1) + ' s');
  ok(c.name + ': every spike train identical', same === n,
    (n - same) + ' differ, the first at ' + first + ' ms');
  ok(c.name + ': the case fires', spikes > 0, 'no spikes, so nothing was compared');

  if(br.noiseProbe){
    const ours = NOISE_PROBE.map(([i, t, s]) => noiseDraw(i, t, s));
    ok(c.name + ': the Python noise draw equals rand.js noiseDraw',
      ours.every((x, k) => x === br.noiseProbe[k]),
      'rand.js ' + ours.join(', ') + ' against ' + br.noiseProbe.join(', '));
  }
  if(c.graded){
    const silent = eng.times.slice(c.gradedFrom, c.targetsFrom).every(t => t.length === 0);
    ok(c.name + ': graded cells never spike', silent, 'a graded cell spiked');
    if(c.differsFrom) ok(c.name + ': targets fire on graded release alone',
      eng.times.slice(c.targetsFrom).some(t => t.length > 0), 'no target fired, so no release was compared');
  }
  if(c.differsFrom && trainsOf.has(c.differsFrom))
    ok(c.name + ': the term changes the spike trains (against ' + c.differsFrom + ')',
      !sameTrains(eng.times, trainsOf.get(c.differsFrom)), 'identical to ' + c.differsFrom + ', so the term was not exercised');
  if(c.weightsDifferFrom && weightsOf.has(c.weightsDifferFrom))
    ok(c.name + ': the term changes the learned weights (against ' + c.weightsDifferFrom + ')',
      !sameArray(eng.weights.w, weightsOf.get(c.weightsDifferFrom)), 'identical to ' + c.weightsDifferFrom + ', so the term was not exercised');
  if(c.level && c.level.some(m => m < 0)){
    // What the floor promises: no membrane goes below it, and strong enough hyperpolarizing drive reaches it.
    // Not every row reaches it at the same multiple of its current unit, so this asks for one cell, not all.
    const floor = floorOf(c.vmin);
    const lowest = Math.min(...eng.vLow);
    ok(c.name + ': no membrane goes below the floor', lowest >= floor,
      'lowest ' + lowest + ' against ' + floor);
    ok(c.name + ': hyperpolarizing drive reaches the floor', eng.vLow.some(v => v === floor),
      'lowest ' + lowest + ' against ' + floor);
    ok(c.name + ': some cells stay silent', eng.times.some(t => t.length === 0), 'every cell fired');
  }
  if(c.refrac > 0){
    let minIsi = Infinity;
    for(const t of eng.times) for(let j = 1; j < t.length; j++) minIsi = Math.min(minIsi, t[j] - t[j - 1]);
    ok(c.name + ': no interval shorter than the refractory count allows',
      minIsi >= c.refrac + 1, 'shortest ' + minIsi + ' ms');
    if(c.name === 'refractory') ok(c.name + ': the refractory count binds at saturating drive',
      minIsi === c.refrac + 1, 'shortest ' + minIsi + ' ms, so the term was never tested');
  }
}
fs.rmSync(TMP, { recursive:true, force:true });
report('brianref');

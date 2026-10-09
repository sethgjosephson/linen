// Does the model reproduce published results?
//
// tools/brianref.mjs checks that the reference engine does what its own code says.
// This checks the model against what the literature reports, on the reference engine (src/simworker.js, in process), with every number taken from the paper it is attributed to and every criterion fixed before the run.
// Where a paper prints its own program, the program is transcribed and run on the same inputs, so a criterion that the published program itself fails is reported as a misreading of the paper rather than a fault in the engine.
//
// Izhikevich 2003 (IEEE Trans Neural Networks 14:1569), the network of Fig. 3.
// The classic rows are the 2003 model with the 2003 integration scheme, so the engine can run the printed program as printed: 800 excitatory and 200 inhibitory neurons with per-neuron parameters from the program's draws, all-to-all weights (0.5 rand from excitatory, -rand from inhibitory, self connections included), Gaussian thalamic input (5 randn excitatory, 2 randn inhibitory) every millisecond, v = -65 and u = b v at the start.
// A spike the program finds at iteration t acts on iteration t's update, which is the engine's 1 ms delay.
// The engine's membrane floor is off, since the program has none.
// The thalamic input reaches the engine as an input frame per millisecond, one channel per neuron. tools/izh2003.py runs the printed program in double precision on the same draws.
//
// The paper reports Poisson spike trains around 8 Hz with occasional synchronized episodes in the alpha (10 Hz) and gamma (40 Hz) ranges.
// Criteria, fixed before the first run:
//   rate          6 to 10 Hz                    ("around 8 Hz")
//   rhythms       mean power over 8 to 12 Hz and over 30 to 50 Hz each at
//                 least 1.5 times the median over 150 to 450 Hz, in the
//                 population spike count's periodogram ("synchronized
//                 firings in the alpha and gamma frequency range")
// applied to both the engine and the printed program, and between them:
//   rate within 5 percent, median CV within 0.05, each band ratio within 25
//   percent, and the time their spike trains first diverge reported (the
//   engine stores float32 and the program float64, so identity is not
//   expected to last).
//
// A third criterion was fixed with them and is recorded, not gated: "Poisson spike trains" read as a median ISI CV of 0.7 to 1.3.
// The printed program gives 0.38 on the same draws (2026-09-14, 5 s; the engine 0.38), more regular than Poisson, since the alpha episodes entrain the cells.
// A criterion the published program fails is a misreading of the paper, so the statement is reported with the program's value and the engine is held to the program instead.
//
// Runs 5 s where the paper ran 1 s, so the periodogram has 0.2 Hz resolution and the band means rest on 20 or more bins.
//
// Izhikevich 2003, Fig. 2, the cortical cell classes.
// The figure shows each class's response to "a step of dc-current I = 10" at a "time resolution" of 0.1 ms, finer than the engine's 1 ms, so the reference is the published equations at 0.1 ms (tools/izh2003_cells.py, forward Euler as in the author's figure code) and the engine is compared with it.
// Stimuli from the figure: RS, IB, CH, FS and LTS from rest, a step of 10.
// TC held at -63 mV (printed) by the current the equations need, then a step the figure draws but does not number, scanned 0.5 to 10; and held at -87 mV (printed), then released to zero.
// RZ from rest, a 4 ms pulse (the author's resonator code uses 4 ms), which the figure draws but does not number, scanned 0.25 to 5.
// Rheobase scans in steps of 0.25.
// The paper's words and the criterion each is read as, fixed before this design's first run (a first design, 1 ms against the paper's words directly with steps from -65 mV, was replaced on 2026-09-14 after its first run):
//   RS   "the period increases": the last interval longer than the first
//   IB   "a stereotypical burst ... followed by repetitive single spikes": the
//        first interval under 10 ms, every interval after 200 ms above 20
//   CH   "stereotypical bursts of closely spaced spikes": at least three bursts
//        (two or more spikes split by gaps above 15 ms); "the inter-burst
//        frequency can be as high as 40 Hz": 40 Hz or less
//   FS   "extremely high frequency": a higher rate than RS; "practically
//        without any adaptation": the last interval within 20 percent of the first
//   LTS  "high-frequency trains": a higher rate than RS; "a noticeable spike
//        frequency adaptation": the last interval at least 1.2 times the first;
//        "low firing thresholds": a lower rheobase than RS
//   TC   "at rest ... and then depolarized, they exhibit tonic firing": some
//        step gives three or more spikes with no interval under 10 ms; "a
//        rebound burst": after release, two or more spikes within 100 ms, every
//        interval under 10 ms
//   RZ   "damped or sustained subthreshold oscillations": some pulse makes the
//        potential cross rest at least twice in the 300 ms after it, no spike
// Each criterion is judged on the reference first.
// If the published equations at 0.1 ms fail it, the reading is wrong and the criterion is recorded, not gated; if they pass, the engine at 1 ms must pass it too.
// Each class's rate at 1 ms against 0.1 ms is reported, flagged past 20 percent, and not gated: that is the time step's effect, not a fault.
//
// Pfister and Gerstner 2006 (J Neurosci 26:9673), the triplet rule against the pairing-frequency data of Sjostrom et al. 2001.
// The paper's visual cortex fit, minimal all-to-all model (Table 3): A2+ = 0, A3+ = 6.5e-3, A2- = 7.1e-3, A3- = 0, tau_y = 114 ms (the one slow constant the minimal model uses; its table row lists one of tau_x and tau_y), with tau+ = 16.8 ms and tau- = 33.7 ms fixed from Bi and Poo.
// On the engine: aP = A2+, trip = A3+, aM = A2-, tauS = tau+, tauM = tau-, tauY = tau_y, additive updates (wdep 0) from w = 1, so an absolute change is the relative one the data report.
// Protocol: 60 pairs at Delta t = +10 and -10 ms, at 0.1 and 50 Hz, each spike forced by a 1 ms pulse that fires the cell within its step.
// Sjostrom's changes as the paper gives them: +10 ms at 0.1 Hz -0.04 +- 0.05,
// -10 ms at 0.1 Hz -0.29 +- 0.08, +10 ms at 50 Hz 0.56 +- 0.26, -10 ms at
// 50 Hz 0.75 +- 0.19.
// Criteria, fixed before the first run:
//   the engine's change within 0.01 of the published rule evaluated on the
//   engine's own spike times (continuous traces, all-to-all);
//   the engine's change, and the rule's, within twice the reported error of
//   each data point. A point the published rule itself misses is recorded
//   rather than gated; the paper reports a fit error of 0.34.
//
// Tsodyks, Pawelzik and Markram 1998 (Neural Computation 10:821), short-term plasticity.
// Equation 2.1: resources x recover with tau_rec and each spike releases U1 x; equation 2.2: U1 decays to zero with tau_facil and each spike raises it by U (1 - U1), and "U_SE ... coincides with the value of U1 reached upon the arrival of the first spike", so the release uses U1 after the spike's increment.
// Inactivation (tau_in) is taken as instantaneous, the paper's own simplification for its mean-field equations.
// The engine's variables differ (its release probability rests at U and the release comes before the jump, as in Zenke et al. 2015 and Auryn) but give the same release on every spike.
// Parameters from Fig. 1: depressing, U 0.5 and tau_rec 800 ms with no facilitation (the engine's tau_F at 1e-6 ms, so the probability is back at U every millisecond); facilitating, U 0.03, tau_rec 130 ms, tau_facil 530 ms. The engine's release is read directly: a probe cell of the 2007 form with k = 0 integrates its input exactly, so its potential rises by w times the release the millisecond it arrives; presynaptic spikes are forced by 1 ms inputs on probe cells that fire at 1 and reset to 0.
// Criteria, fixed before the first run:
//   every train fires exactly as intended, and each spike's release on the
//   engine is within 1e-4 of equations 2.1 and 2.2 on the same spike times,
//   with no movement of the probe that no spike explains (regular trains at
//   2 to 80 Hz depressing, 1 to 100 Hz facilitating, and the Poisson trains);
//   the first release from rest equals U within 1e-6 (both synapses);
//   depressing, Poisson trains of 300 s at 1, 2.5, 10 and 40 Hz: the release
//   per second after the first 5 s within 5 percent of the first term of
//   equation 3.4, r U / (1 + r U tau_rec), at the realized rate; above the
//   limiting frequency 1/(U tau_rec) (2.5 Hz) rate signaling saturates: at
//   40 Hz within 10 percent of 1/tau_rec, and below it, at 1 Hz, at least
//   0.6 r U;
//   facilitating, regular trains at 1 to 100 Hz: the release per second over
//   3 to 4 s (the stationary level Fig. 1D plots, up to the membrane's scale)
//   peaks within 25 percent of equation 3.6's theta, 1/tau_facil +
//   sqrt(2/tau_facil^2 + (1 + U)/(U tau_rec tau_facil)), 24.4 Hz, and the
//   engine's peak within 1 Hz of the equations' peak.
// Each criterion but the first two is judged on equations 2.1 and 2.2 on the same trains first; one they fail is recorded, not gated.
// The Fig. 1D reading failed that way on its first run (2026-09-14): the release per second rises to 100 Hz on the equations, so the figure's tuning curve is not that quantity.
// The paper's text at equation 3.6 says the average EPSP amplitude, x U1, is what the balance of facilitation and depression holds, so a second reading was added after that run and is marked as such: the mean release per spike over 3 to 4 s peaks within 25 percent of theta, and the engine's peak within 1 Hz of the equations'.
//
// Vogels, Sprekeler, Zenke, Clopath and Gerstner 2011 (Science 334:1569), inhibitory plasticity.
// The authors' supplementary program (inhib_plasticity.m, ModelDB 143751) is transcribed in tools/vogels2011.py and run as printed: eta 0.001, alpha 0.25 eta, traces of 20 ms that add eta per spike, on an inhibitory presynaptic spike w += post - alpha, on a postsynaptic spike w += pre.
// With presynaptic spikes independent of the postsynaptic cell the mean change vanishes when the postsynaptic rate is alpha/(2 eta tau), 6.25 Hz for the program's numbers; the engine's target, iRho, is the same rate by construction (alpha = iRho (tau+ + tau-)/1000 in trace units).
// The engine: one classic RS cell held at a bias of 20 and 1000 inhibitory probe cells firing independent 10 Hz Bernoulli trains through plastic kick synapses from -0.01, iRho 6.25, iEta 0.02, tau+ = tau- = 20 ms, hard bounds at 100, 150 s.
// Two designs came before this one, both run on 2026-09-14 with the same criteria.
// The first, 200 inputs from -0.05, iEta 0.01, bound 10: the weights reached the bound (mean -9.78) with the cell at 17.3 Hz, a target the bound made unreachable.
// The second, the bound at 100 and iEta 0.05: 9.48 Hz over the last 25 s, weights at -22.
// A 600 s run of it outside the suite was still falling (24.4 Hz in the first 25 s, 7 to 7.6 Hz after 400 s): not converged, and settling above the target, since with 200 strong kicks an inhibitory spike visibly delays the next postsynaptic spike and the independence the target assumes does not hold.
// With 1000 weak inputs the same run settled at 6.2 to 6.5 Hz from 50 s on.
// Criteria, fixed before the first run:
//   the program's output rate, mean over its last 20 runs (30 s), within 15
//   percent of alpha/(2 eta tau); judged first, recorded if it fails;
//   the engine's rate over the last 25 s within 15 percent of iRho;
//   the engine's rate over the first 2 s at least twice iRho, so the rate
//   the rule reaches is not where the cell started.
//
//   node tools/published.mjs

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ok, report } from './harness.mjs';
import { PROTOCOL } from '../src/protocol.js';
import { classicRow } from '../src/form.js';
import { engineTypes, NEURON_TYPES } from '../src/nodes.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const PY = [path.join(ROOT, '.venv-analysis', 'Scripts', 'python.exe'),
            path.join(ROOT, '.venv-analysis', 'bin', 'python')].find(p => fs.existsSync(p));
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'published-'));

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const send = d => globalThis.onmessage({ data:d });

function mulberry(seed){
  let s = seed >>> 0;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
// Box-Muller from the uniform stream, for the program's randn
function gaussian(rnd){
  let spare = null;
  return () => {
    if(spare !== null){ const g = spare; spare = null; return g; }
    let u1 = 0; while(u1 === 0) u1 = rnd();
    const u2 = rnd(), r = Math.sqrt(-2*Math.log(u1));
    spare = r*Math.sin(2*Math.PI*u2);
    return r*Math.cos(2*Math.PI*u2);
  };
}

// ---- measures -----------------------------------------------------------------
function rateHz(times, ms){ let s = 0; for(const t of times) s += t.length; return s/times.length/(ms/1000); }
function medianCv(times){
  const cvs = [];
  for(const t of times){
    if(t.length < 10) continue;
    const isi = []; for(let j = 1; j < t.length; j++) isi.push(t[j] - t[j - 1]);
    const m = isi.reduce((a, b) => a + b, 0)/isi.length;
    const sd = Math.sqrt(isi.reduce((a, b) => a + (b - m)**2, 0)/isi.length);
    cvs.push(sd/m);
  }
  cvs.sort((a, b) => a - b);
  return cvs.length ? cvs[Math.floor(cvs.length/2)] : NaN;
}
// periodogram of the population spike count, one bin per 1/T Hz up to 450 Hz
function bands(times, ms){
  const x = new Float64Array(ms);
  for(const t of times) for(const s of t) if(s >= 0 && s < ms) x[s]++;
  const mean = x.reduce((a, b) => a + b, 0)/ms;
  for(let t = 0; t < ms; t++) x[t] -= mean;
  const df = 1000/ms, power = [];
  for(let k = 1; k*df <= 450; k++){
    const w = 2*Math.PI*k/ms;
    let re = 0, im = 0;
    for(let t = 0; t < ms; t++){ re += x[t]*Math.cos(w*t); im -= x[t]*Math.sin(w*t); }
    power.push([k*df, (re*re + im*im)/ms]);
  }
  const bandMean = (lo, hi) => { const p = power.filter(([f]) => f >= lo && f <= hi).map(([, v]) => v);
    return p.reduce((a, b) => a + b, 0)/p.length; };
  const base = power.filter(([f]) => f >= 150 && f <= 450).map(([, v]) => v).sort((a, b) => a - b);
  const median = base[Math.floor(base.length/2)];
  return { alpha:bandMean(8, 12)/median, gamma:bandMean(30, 50)/median };
}

// ---- Izhikevich 2003, Fig. 3 ------------------------------------------------------
function izhikevich2003(seed, ms){
  const Ne = 800, Ni = 200, N = Ne + Ni;
  const rnd = mulberry(seed), randn = gaussian(mulberry(seed + 1000));
  const re = Float64Array.from({ length:Ne }, rnd), ri = Float64Array.from({ length:Ni }, rnd);
  // the program's parameter lines
  const a = i => i < Ne ? 0.02 : 0.02 + 0.08*ri[i - Ne];
  const b = i => i < Ne ? 0.2 : 0.25 - 0.05*ri[i - Ne];
  const c = i => i < Ne ? -65 + 15*re[i]**2 : -65;
  const d = i => i < Ne ? 8 - 6*re[i]**2 : 2;
  const types = Array.from({ length:N }, (_, i) => ({ f7:classicRow({ a:a(i), b:b(i), c:c(i), d:d(i) }) }));
  // S[i, j] from j to i: 0.5 rand from excitatory, -rand from inhibitory
  const S = new Float32Array(N*N);
  for(let i = 0; i < N; i++) for(let j = 0; j < N; j++) S[i*N + j] = j < Ne ? 0.5*rnd() : -rnd();
  const input = new Float32Array(ms*N);
  for(let t = 0; t < ms; t++) for(let i = 0; i < N; i++) input[t*N + i] = (i < Ne ? 5 : 2)*randn();
  return { Ne, Ni, N, re, ri, types, S, input };
}

function runEngine(net, ms){
  const { N, types, S, input } = net;
  // every neuron sends to every neuron with delay 1, grouped by presynaptic neuron
  const preStart = Int32Array.from({ length:N + 1 }, (_, j) => j*N);
  const post = new Int32Array(N*N), w = new Float32Array(N*N);
  for(let j = 0; j < N; j++) for(let i = 0; i < N; i++){ post[j*N + i] = i; w[j*N + i] = S[i*N + j]; }
  const v0 = new Float32Array(N).fill(-65);
  // the 2007 row's recovery variable is measured from the rest: U = u - b vr
  const u0 = Float32Array.from(types, t => t.f7.b*(-65 - t.f7.vr));
  const chStart = Int32Array.from({ length:N + 1 }, (_, k) => k);
  replies.length = 0;
  send({ cmd:'init', protocol:PROTOCOL, count:N, ntype:Uint16Array.from({ length:N }, (_, i) => i),
    bias:new Float32Array(N), syn:0, psc:0, tauE:3, tauI:8, protocols:[], plast:0, refrac:0, vmin:0,
    inputs:[{ chStart, chIdx:Int32Array.from({ length:N }, (_, i) => i), chW:new Float32Array(N).fill(1), amp:1 }],
    seed:1, v0, u0, preStart, post, w, delay:new Uint8Array(N*N).fill(1), types });
  const err = replies.find(r => r.cmd === 'error');
  if(err) throw new Error('engine refused the network: ' + err.message);
  const times = Array.from({ length:N }, () => []);
  for(let t = 0; t < ms; t++){
    send({ cmd:'inputFrame', mi:0, gains:input.subarray(t*N, (t + 1)*N) });
    replies.length = 0;
    send({ cmd:'tick', steps:1 });
    const s = replies.find(r => r.cmd === 'state');
    for(let i = 0; i < N; i++) if(s.fired[i]) times[i].push(t);
  }
  return times;
}

function runProgram(net, ms){
  const file = name => path.join(TMP, name);
  fs.writeFileSync(file('re.bin'), Buffer.from(net.re.buffer));
  fs.writeFileSync(file('ri.bin'), Buffer.from(net.ri.buffer));
  fs.writeFileSync(file('S.bin'), Buffer.from(net.S.buffer));
  fs.writeFileSync(file('input.bin'), Buffer.from(net.input.buffer));
  fs.writeFileSync(file('case.json'), JSON.stringify({ Ne:net.Ne, Ni:net.Ni, ms,
    re:file('re.bin'), ri:file('ri.bin'), S:file('S.bin'), input:file('input.bin') }));
  execFileSync(PY, [path.join(HERE, 'izh2003.py'), file('case.json'), file('out.json')],
    { stdio:['ignore', 'ignore', 'pipe'], timeout:900000 });
  return JSON.parse(fs.readFileSync(file('out.json'), 'utf8')).times;
}

const MS = 5000;
console.log('published: Izhikevich 2003, Fig. 3 (1000 neurons, ' + MS/1000 + ' s)');
const net = izhikevich2003(2003, MS);
const eng = runEngine(net, MS);
const measured = { engine:eng };
if(PY) measured.program = runProgram(net, MS);
else console.log('  NumPy not found in .venv-analysis, so the printed program DID NOT RUN; the engine is checked against the paper alone.');

const stats = {};
for(const [who, times] of Object.entries(measured)){
  const bp = bands(times, MS);
  stats[who] = { rate:rateHz(times, MS), cv:medianCv(times), ...bp };
  const s = stats[who];
  console.log('  ' + who.padEnd(8) + ' rate ' + s.rate.toFixed(2) + ' Hz   median CV ' + s.cv.toFixed(3) +
    '   alpha ' + s.alpha.toFixed(2) + 'x   gamma ' + s.gamma.toFixed(2) + 'x baseline');
  ok(who + ': rate around 8 Hz (6 to 10)', s.rate >= 6 && s.rate <= 10, s.rate.toFixed(2) + ' Hz');
  ok(who + ': alpha-range synchrony (8 to 12 Hz power at least 1.5x baseline)', s.alpha >= 1.5, s.alpha.toFixed(2) + 'x');
  ok(who + ': gamma-range synchrony (30 to 50 Hz power at least 1.5x baseline)', s.gamma >= 1.5, s.gamma.toFixed(2) + 'x');
}
if(stats.program){
  const e = stats.engine, p = stats.program;
  console.log('  the paper calls the trains Poisson; the printed program gives a median ISI CV of ' +
    p.cv.toFixed(3) + ' (Poisson is 1), so that statement is recorded, not gated');
  const rel = (x, y) => Math.abs(x - y)/Math.max(1e-9, Math.abs(y));
  ok('engine against the printed program: rate within 5 percent', rel(e.rate, p.rate) <= 0.05,
    e.rate.toFixed(2) + ' against ' + p.rate.toFixed(2) + ' Hz');
  ok('engine against the printed program: median CV within 0.05', Math.abs(e.cv - p.cv) <= 0.05,
    e.cv.toFixed(3) + ' against ' + p.cv.toFixed(3));
  ok('engine against the printed program: alpha ratio within 25 percent', rel(e.alpha, p.alpha) <= 0.25,
    e.alpha.toFixed(2) + ' against ' + p.alpha.toFixed(2));
  ok('engine against the printed program: gamma ratio within 25 percent', rel(e.gamma, p.gamma) <= 0.25,
    e.gamma.toFixed(2) + ' against ' + p.gamma.toFixed(2));
  // when the float32 engine and the float64 program part company
  let diverge = MS;
  for(let i = 0; i < net.N; i++){
    const A = measured.engine[i], B = measured.program[i];
    const k = A.findIndex((t, x) => t !== B[x]);
    const at = k >= 0 ? Math.min(A[k], B[k] ?? Infinity) : (B.length > A.length ? B[A.length] : MS);
    diverge = Math.min(diverge, at);
  }
  console.log('  spike trains identical until ' + diverge + ' ms (float32 engine, float64 program)');
}

// ---- Izhikevich 2003, Fig. 2: the cell classes, 1 ms against 0.1 ms --------------
{
  const T = engineTypes();
  const key = k => NEURON_TYPES.findIndex(t => t.key === k);
  const MS2 = 1100, ON = 100;
  // the current that holds a 2003 cell at potential v: 0.04 v^2 + (5 - b) v + 140 + I = 0
  const holdAt = (b, v) => -(0.04*v*v + (5 - b)*v + 140);
  const cells = [];
  const add = (name, k, { v0, bias = 0, steps = [] } = {}) => {
    const row = NEURON_TYPES[key(k)];
    cells.push({ name, type:key(k), a:row.a, b:row.b, c:row.c, d:row.d,
      v0:v0 ?? T[key(k)].f7.vr, bias, steps });
  };
  const STEP10 = [{ amp:10, t0:ON, t1:MS2 }];
  for(const k of ['RS', 'IB', 'CH', 'FS', 'LTS']) add(k, k, { steps:STEP10 });
  const tcB = NEURON_TYPES[key('TC')].b;
  for(let q = 1; q <= 20; q++)
    add('TC tonic ' + q/2, 'TC', { v0:-63, bias:holdAt(tcB, -63), steps:[{ amp:q/2, t0:ON, t1:MS2 }] });
  add('TC rebound', 'TC', { v0:-87, bias:holdAt(tcB, -87), steps:[{ amp:-holdAt(tcB, -87), t0:ON, t1:MS2 }] });
  for(let q = 1; q <= 20; q++) add('RZ pulse ' + q/4, 'RZ', { steps:[{ amp:q/4, t0:ON, t1:ON + 4 }] });
  for(let q = 1; q <= 40; q++){
    add('RS rheobase ' + q/4, 'RS', { steps:[{ amp:q/4, t0:ON, t1:MS2 }] });
    add('LTS rheobase ' + q/4, 'LTS', { steps:[{ amp:q/4, t0:ON, t1:MS2 }] });
  }
  const n = cells.length;

  // the engine, at 1 ms
  const v0 = Float32Array.from(cells, cl => cl.v0);
  const u0 = Float32Array.from(cells, (cl, i) => T[cl.type].f7.b*(v0[i] - T[cl.type].f7.vr));
  const protocols = [];
  cells.forEach((cl, i) => cl.steps.forEach(st => protocols.push({ idx:Uint32Array.of(i), amp:st.amp,
    mode:0, period:100, width:10, t0:st.t0, duration:st.t1 - st.t0 })));
  replies.length = 0;
  send({ cmd:'init', protocol:PROTOCOL, count:n, ntype:Uint8Array.from(cells, cl => cl.type),
    bias:Float32Array.from(cells, cl => cl.bias), syn:0, psc:0, tauE:3, tauI:8, protocols, inputs:[],
    plast:0, refrac:0, vmin:0, seed:1, v0, u0, preStart:new Int32Array(n + 1), post:new Int32Array(0),
    w:new Float32Array(0), delay:new Uint8Array(0), types:T });
  const err = replies.find(r => r.cmd === 'error');
  if(err) throw new Error('engine refused the cell classes: ' + err.message);
  send({ cmd:'sendV', on:true });
  const linen = { times:cells.map(() => []), v:cells.map(() => []) };
  for(let t = 0; t < MS2; t++){
    replies.length = 0;
    send({ cmd:'tick', steps:1 });
    const st = replies.find(r => r.cmd === 'state');
    for(let i = 0; i < n; i++){ if(st.fired[i]) linen.times[i].push(t); linen.v[i].push(st.v[i]); }
  }

  // the published equations, at 0.1 ms
  let reference = null;
  if(PY){
    const inFile = path.join(TMP, 'cells.json'), outFile = path.join(TMP, 'cells-out.json');
    fs.writeFileSync(inFile, JSON.stringify({ ms:MS2, cells:cells.map(cl => ({ a:cl.a, b:cl.b, c:cl.c, d:cl.d,
      v0:cl.v0, bias:cl.bias, steps:cl.steps })) }));
    execFileSync(PY, [path.join(HERE, 'izh2003_cells.py'), inFile, outFile], { stdio:['ignore', 'ignore', 'pipe'], timeout:900000 });
    reference = JSON.parse(fs.readFileSync(outFile, 'utf8'));
  }

  console.log('published: Izhikevich 2003, Fig. 2, the cell classes (the engine at 1 ms, the published equations at 0.1 ms)');
  if(!reference){
    console.log('  NumPy not found in .venv-analysis, so the 0.1 ms reference DID NOT RUN and the cell classes were not compared.');
  } else {
    const at = name => cells.findIndex(cl => cl.name === name);
    const within = (side, i, t0, t1 = MS2) => side.times[i].filter(t => t >= t0 && t < t1);
    const intervals = ts => ts.slice(1).map((t, j) => t - ts[j]);
    const rate = (side, i) => within(side, i, ON).length/((MS2 - ON)/1000);
    const fmt = x => Number.isInteger(x) ? String(x) : x.toFixed(1);
    const summary = (side, i) => { const ts = within(side, i, ON); const iv = intervals(ts);
      return rate(side, i).toFixed(0) + ' Hz, intervals ' + iv.slice(0, 6).map(fmt).join(' ') + (iv.length > 6 ? ' ...' : ''); };
    const rheobase = (side, prefix) => { for(let q = 1; q <= 40; q++) if(within(side, at(prefix + ' ' + q/4), ON).length) return q/4; return Infinity; };
    const crossings = (side, i, rest) => { let count = 0, sign = 0;
      for(let t = ON + 4; t < ON + 304; t++){ const dv = side.v[i][t] - rest; if(Math.abs(dv) < 1e-3) continue;
        const sg = dv > 0 ? 1 : -1; if(sign && sg !== sign) count++; sign = sg; }
      return count; };
    const sides = { reference, linen };

    for(const k of ['RS', 'IB', 'CH', 'FS', 'LTS']){
      const r = rate(reference, at(k)), l = rate(linen, at(k));
      const off = Math.abs(l - r)/Math.max(1e-9, r);
      console.log('  ' + k.padEnd(4) + ' 0.1 ms ' + summary(reference, at(k)).padEnd(44) + ' 1 ms ' + summary(linen, at(k)) +
        (off > 0.2 ? '   rate differs by ' + (100*off).toFixed(0) + ' percent' : ''));
    }

    const CRITERIA = [
      ['RS: "the period increases" (last interval longer than the first)', side => {
        const iv = intervals(within(side, at('RS'), ON));
        return { pass:iv.length >= 3 && iv[iv.length - 1] > iv[0], detail:'first ' + fmt(iv[0]) + ', last ' + fmt(iv[iv.length - 1]) + ' ms' }; }],
      ['IB: "a stereotypical burst ... followed by repetitive single spikes" (first interval under 10 ms, all after 200 ms above 20)', side => {
        const iv = intervals(within(side, at('IB'), ON)), late = intervals(within(side, at('IB'), ON + 200));
        return { pass:iv.length >= 3 && iv[0] < 10 && late.length >= 2 && Math.min(...late) > 20,
          detail:'first ' + fmt(iv[0]) + ' ms, shortest after 200 ms ' + (late.length ? fmt(Math.min(...late)) : 'none') }; }],
      ['CH: "stereotypical bursts of closely spaced spikes" (at least three bursts)', side => {
        const groups = []; for(const t of within(side, at('CH'), ON)){ const g = groups[groups.length - 1];
          if(g && t - g[g.length - 1] <= 15) g.push(t); else groups.push([t]); }
        const bursts = groups.filter(g => g.length >= 2);
        return { pass:bursts.length >= 3, detail:bursts.length + ' bursts' }; }],
      ['CH: "the inter-burst frequency can be as high as 40 Hz" (40 Hz or less)', side => {
        const groups = []; for(const t of within(side, at('CH'), ON)){ const g = groups[groups.length - 1];
          if(g && t - g[g.length - 1] <= 15) g.push(t); else groups.push([t]); }
        const hz = groups.filter(g => g.length >= 2).length/((MS2 - ON)/1000);
        return { pass:hz > 0 && hz <= 40, detail:hz.toFixed(1) + ' bursts per second' }; }],
      ['FS: "extremely high frequency" (a higher rate than RS)', side => ({
        pass:rate(side, at('FS')) > rate(side, at('RS')), detail:rate(side, at('FS')).toFixed(0) + ' against ' + rate(side, at('RS')).toFixed(0) + ' Hz' })],
      ['FS: "practically without any adaptation" (last interval within 20 percent of the first)', side => {
        const iv = intervals(within(side, at('FS'), ON));
        return { pass:iv.length >= 3 && iv[iv.length - 1] <= 1.2*iv[0], detail:'first ' + fmt(iv[0]) + ', last ' + fmt(iv[iv.length - 1]) + ' ms' }; }],
      ['LTS: "high-frequency trains" (a higher rate than RS)', side => ({
        pass:rate(side, at('LTS')) > rate(side, at('RS')), detail:rate(side, at('LTS')).toFixed(0) + ' against ' + rate(side, at('RS')).toFixed(0) + ' Hz' })],
      ['LTS: "a noticeable spike frequency adaptation" (last interval at least 1.2 times the first)', side => {
        const iv = intervals(within(side, at('LTS'), ON));
        return { pass:iv.length >= 3 && iv[iv.length - 1] >= 1.2*iv[0], detail:'first ' + fmt(iv[0]) + ', last ' + fmt(iv[iv.length - 1]) + ' ms' }; }],
      ['LTS: "low firing thresholds" (a lower rheobase than RS)', side => ({
        pass:rheobase(side, 'LTS rheobase') < rheobase(side, 'RS rheobase'),
        detail:'LTS ' + rheobase(side, 'LTS rheobase') + ', RS ' + rheobase(side, 'RS rheobase') })],
      ['TC: "at rest and then depolarized, they exhibit tonic firing" (some step from -63 mV gives three or more spikes, none under 10 ms apart)', side => {
        const steps = []; for(let q = 1; q <= 20; q++){ const ts = within(side, at('TC tonic ' + q/2), ON);
          if(ts.length >= 3 && Math.min(...intervals(ts)) >= 10) steps.push(q/2); }
        return { pass:steps.length > 0, detail:'tonic at steps ' + (steps.length ? steps.join(' ') : 'none') }; }],
      ['TC: "a rebound burst of action potentials" after release from -87 mV (two or more spikes within 100 ms, every interval under 10 ms)', side => {
        const ts = within(side, at('TC rebound'), ON, ON + 100);
        return { pass:ts.length >= 2 && intervals(ts).every(x => x < 10), detail:ts.length + ' spikes, intervals ' + intervals(ts).map(fmt).join(' ') }; }],
      ['RZ: "damped or sustained subthreshold oscillations" (some pulse makes the potential cross rest twice in 300 ms, no spike)', side => {
        const rest = T[key('RZ')].f7.vr, amps = [];
        for(let q = 1; q <= 20; q++){ const i = at('RZ pulse ' + q/4);
          if(within(side, i, ON).length === 0 && crossings(side, i, rest) >= 2) amps.push(q/4); }
        return { pass:amps.length > 0, detail:'oscillating without a spike at pulses ' + (amps.length ? amps.join(' ') : 'none') }; }],
    ];
    for(const [label, judge] of CRITERIA){
      const ref = judge(reference), lin = judge(linen);
      if(ref.pass)
        ok(label + ': the engine at 1 ms agrees with the published equations at 0.1 ms', lin.pass,
          '1 ms: ' + lin.detail + '; 0.1 ms: ' + ref.detail);
      else
        console.log('  recorded, not gated: ' + label + '. The published equations at 0.1 ms do not show it (' + ref.detail +
          '); the engine at 1 ms: ' + lin.detail);
    }
  }
}

// ---- Pfister and Gerstner 2006, the triplet rule against Sjostrom 2001 --------------
{
  const T = engineTypes();
  const RSr = NEURON_TYPES.findIndex(t => t.key === 'RS');
  const RULE = { aP:0, trip:6.5e-3, aM:7.1e-3, tauS:16.8, tauM:33.7, tauY:114 };
  const CONDITIONS = [
    { dt:10, hz:0.1, data:-0.04, err:0.05 },
    { dt:-10, hz:0.1, data:-0.29, err:0.08 },
    { dt:10, hz:50, data:0.56, err:0.26 },
    { dt:-10, hz:50, data:0.75, err:0.19 },
  ];
  const PAIRS = 60, T0 = 1000, PULSE = 300;
  // four independent pairs, pre 2k and post 2k + 1, one synapse each
  const n = 2*CONDITIONS.length;
  const protocols = [];
  CONDITIONS.forEach((c, k) => {
    const period = 1000/c.hz, duration = (PAIRS - 1)*period + 1;
    const preAt = T0 + Math.max(0, -c.dt), postAt = T0 + Math.max(0, c.dt);
    protocols.push({ idx:Uint32Array.of(2*k), amp:PULSE, mode:1, period, width:1, t0:preAt, duration });
    protocols.push({ idx:Uint32Array.of(2*k + 1), amp:PULSE, mode:1, period, width:1, t0:postAt, duration });
  });
  const MS3 = T0 + Math.max(...CONDITIONS.map(c => (PAIRS - 1)*1000/c.hz)) + 100;
  const preStart = Int32Array.from({ length:n + 1 }, (_, i) => Math.ceil(i/2));
  const post = Int32Array.from({ length:CONDITIONS.length }, (_, k) => 2*k + 1);
  const ntype = new Uint8Array(n).fill(RSr);
  const v0 = new Float32Array(n).fill(-70);
  replies.length = 0;
  send({ cmd:'init', protocol:PROTOCOL, count:n, ntype, bias:new Float32Array(n), syn:0, psc:0, tauE:3, tauI:8,
    protocols, inputs:[], plast:1, ...RULE, wmax:10, wdep:0, iEta:0, iRho:5, het:0, tin:0, refrac:0, vmin:0,
    seed:1, v0, u0:new Float32Array(n), preStart, post, w:new Float32Array(CONDITIONS.length).fill(1),
    delay:new Uint8Array(CONDITIONS.length).fill(1), types:T });
  const err = replies.find(r => r.cmd === 'error');
  if(err) throw new Error('engine refused the pairing protocol: ' + err.message);
  const times = Array.from({ length:n }, () => []);
  for(let t = 0; t < MS3; t++){
    replies.length = 0;
    send({ cmd:'tick', steps:1 });
    const st = replies.find(r => r.cmd === 'state');
    for(let i = 0; i < n; i++) if(st.fired[i]) times[i].push(t);
  }
  replies.length = 0;
  send({ cmd:'getWeights' });
  const w = replies.find(r => r.cmd === 'weights').w;

  // the published rule, all-to-all, with continuous traces on given spike times
  const rule = (pre, postT) => {
    const events = [...pre.map(t => [t, 0]), ...postT.map(t => [t, 1])].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let dw = 0;
    for(const [t, kind] of events){
      if(kind === 0){
        let o1 = 0; for(const s of postT) if(s < t) o1 += Math.exp(-(t - s)/RULE.tauM);
        dw -= RULE.aM*o1;
      } else {
        let r1 = 0, o2 = 0;
        for(const s of pre) if(s < t) r1 += Math.exp(-(t - s)/RULE.tauS);
        for(const s of postT) if(s < t) o2 += Math.exp(-(t - s)/RULE.tauY);
        dw += r1*(RULE.aP + RULE.trip*o2);
      }
    }
    return dw;
  };

  console.log('published: Pfister and Gerstner 2006, the triplet rule against Sjostrom 2001 (60 pairs)');
  CONDITIONS.forEach((c, k) => {
    const pre = times[2*k], postT = times[2*k + 1];
    const offsets = pre.map((t, j) => postT[j] - t);
    const forced = pre.length === PAIRS && postT.length === PAIRS && offsets.every(o => o === c.dt);
    const engine = w[k] - 1, model = rule(pre, postT);
    const label = (c.dt > 0 ? '+' : '') + c.dt + ' ms at ' + c.hz + ' Hz';
    console.log('  ' + label.padEnd(16) + ' engine ' + engine.toFixed(3) + '   rule ' + model.toFixed(3) +
      '   data ' + c.data.toFixed(2) + ' +- ' + c.err.toFixed(2));
    ok(label + ': 60 pairs at exactly the intended offset', forced,
      pre.length + ' pre, ' + postT.length + ' post spikes, offsets ' + [...new Set(offsets)].join(' '));
    ok(label + ': the engine applies the published rule (within 0.01 on its own spike times)',
      Math.abs(engine - model) <= 0.01, engine.toFixed(4) + ' against ' + model.toFixed(4));
    const ruleFits = Math.abs(model - c.data) <= 2*c.err;
    if(ruleFits)
      ok(label + ': the engine reproduces Sjostrom 2001 (within twice the reported error)',
        Math.abs(engine - c.data) <= 2*c.err, engine.toFixed(3) + ' against ' + c.data + ' +- ' + c.err);
    else
      console.log('    the published rule itself gives ' + model.toFixed(3) + ' against ' + c.data + ' +- ' + c.err +
        ', outside twice the error, so this point is recorded, not gated');
  });
}
// ---- Tsodyks, Pawelzik and Markram 1998, short-term plasticity --------------------
{
  const PULSE = 300;
  // Probe rows of the 2007 form with k = 0 and a = b = 0: the membrane integrates its input exactly.
  // A spiker fires at 1 and resets to 0; a meter never fires, so its potential steps by each release as it arrives.
  const stpRun = (rule, trains, ms) => {
    const T = engineTypes();
    const SP = T.length, ME = T.length + 1;
    T.push({ f7:{ C:1, k:0, vr:0, vt:0, vpeak:1, a:0, b:0, c:0, d:0 } });
    T.push({ f7:{ C:1, k:0, vr:0, vt:0, vpeak:1e30, a:0, b:0, c:0, d:0 } });
    const k = trains.length, n = 2*k;
    const at = trains.map(tr => { const s = new Uint8Array(ms); for(const t of tr) s[t] = 1; return s; });
    replies.length = 0;
    send({ cmd:'init', protocol:PROTOCOL, count:n, ntype:Uint8Array.from({ length:n }, (_, i) => i % 2 ? ME : SP),
      bias:new Float32Array(n), syn:0, psc:0, tauE:3, tauI:8, protocols:[], plast:0, refrac:0, vmin:0, seed:1,
      inputs:[{ chStart:Int32Array.from({ length:k + 1 }, (_, j) => j), chIdx:Int32Array.from({ length:k }, (_, j) => 2*j),
        chW:new Float32Array(k).fill(1), amp:1 }],
      stp:1, stpU:rule.U, stpTauD:rule.tauRec, stpTauF:rule.tauF > 0 ? rule.tauF : 1e-6, stpOrder:0, stpNorm:0,
      v0:new Float32Array(n), u0:new Float32Array(n),
      preStart:Int32Array.from({ length:n + 1 }, (_, i) => Math.ceil(i/2)),
      post:Int32Array.from({ length:k }, (_, j) => 2*j + 1),
      w:new Float32Array(k).fill(1), delay:new Uint8Array(k).fill(1), types:T });
    const err = replies.find(r => r.cmd === 'error');
    if(err) throw new Error('engine refused the short-term plasticity probes: ' + err.message);
    send({ cmd:'sendV', on:true });
    const fired = trains.map(() => []), rel = trains.map(() => []);
    const last = new Float64Array(k), prevFired = new Uint8Array(k), gains = new Float32Array(k);
    let stray = 0;
    for(let t = 0; t < ms; t++){
      for(let j = 0; j < k; j++) gains[j] = at[j][t] ? PULSE : 0;
      send({ cmd:'inputFrame', mi:0, gains });
      replies.length = 0;
      send({ cmd:'tick', steps:1 });
      const st = replies.find(r => r.cmd === 'state');
      for(let j = 0; j < k; j++){
        const vm = st.v[2*j + 1], dv = vm - last[j];
        if(prevFired[j]) rel[j].push(dv); else if(dv !== 0) stray++;
        last[j] = vm;
        prevFired[j] = st.fired[2*j] ? 1 : 0;
        if(prevFired[j]) fired[j].push(t);
      }
    }
    return { fired, rel, stray };
  };
  // equations 2.1 and 2.2 (tau_in instantaneous) on given spike times, in the paper's variables: U1 decays to zero and the release uses it after the jump
  const paper = (times, rule) => {
    let x = 1, u1 = 0, prev = null;
    return times.map(t => {
      if(prev !== null){
        x = 1 - (1 - x)*Math.exp(-(t - prev)/rule.tauRec);
        u1 = rule.tauF > 0 ? u1*Math.exp(-(t - prev)/rule.tauF) : 0;
      }
      u1 += rule.U*(1 - u1);
      const r = u1*x;
      x -= r; prev = t;
      return r;
    });
  };
  const regular = (hz, t0, t1) => { const out = []; for(let j = 0; ; j++){ const t = Math.round(t0 + j*1000/hz); if(t >= t1) break; out.push(t); } return out; };
  const poisson = (hz, t0, t1, seed) => { const rnd = mulberry(seed), out = []; for(let t = t0; t < t1; t++) if(rnd() < hz/1000) out.push(t); return out; };
  const exact = (label, rule, trains, res) => {
    const forced = trains.every((tr, j) => tr.length === res.fired[j].length && tr.every((t, q) => t === res.fired[j][q]));
    ok(label + ': every train fires exactly as intended', forced, trains.map((tr, j) => res.fired[j].length + '/' + tr.length).join(' '));
    let worst = 0;
    trains.forEach((tr, j) => { const ref = paper(tr, rule); ref.forEach((r, q) => { const e = Math.abs((res.rel[j][q] ?? NaN) - r); worst = Number.isNaN(e) ? Infinity : Math.max(worst, e); }); });
    ok(label + ': each release matches equations 2.1 and 2.2 (within 1e-4), nothing else moves the probe',
      worst <= 1e-4 && res.stray === 0, 'largest difference ' + worst.toExponential(2) + ', unexplained steps ' + res.stray);
    const first = res.rel.map(r => r[0]).filter(r => r !== undefined);
    const firstWorst = Math.max(...first.map(r => Math.abs(r - rule.U)));
    ok(label + ': the first release from rest is U ("coincides with the value of U1 reached upon the arrival of the first spike")',
      firstWorst <= 1e-6, 'U ' + rule.U + ', largest difference ' + firstWorst.toExponential(2));
  };
  // judged on the paper's equations first; gates the engine only if they pass
  const judged = (label, refPass, engPass, detail) => {
    if(refPass) ok(label, engPass, detail);
    else console.log('    ' + label + ': the published equations fail it (' + detail + '), recorded, not gated');
  };
  console.log('published: Tsodyks, Pawelzik and Markram 1998, short-term plasticity');

  // depressing synapse, Fig. 1A
  const DEP = { U:0.5, tauRec:800, tauF:0 };
  const LONG = 300000, SKIP = 5000;
  const depRates = [1, 2.5, 10, 40];
  const depRegular = [2, 5, 10, 20, 40, 80].map(hz => regular(hz, 100, 4100));
  const depPoisson = depRates.map((hz, q) => poisson(hz, 100, LONG, 1998 + q));
  const depTrains = [...depRegular, ...depPoisson];
  const dep = stpRun(DEP, depTrains, LONG + 2);
  exact('depressing', DEP, depTrains, dep);
  const lambda = 1000/(DEP.U*DEP.tauRec);
  console.log('  depressing, U 0.5, tau_rec 800 ms; limiting frequency ' + lambda.toFixed(1) + ' Hz');
  console.log('    rate      engine  equations  eq. 3.4   (released per second, Poisson, after 5 s)');
  depRates.forEach((hz, q) => {
    const j = depRegular.length + q, tr = depTrains[j];
    const secs = (LONG - SKIP - 100)/1000;
    const keep = tr.map((t, i) => t >= SKIP + 100 ? i : -1).filter(i => i >= 0);
    const r = keep.length/secs;
    const eng = keep.reduce((s, i) => s + dep.rel[j][i], 0)/secs;
    const refRel = paper(tr, DEP);
    const ref = keep.reduce((s, i) => s + refRel[i], 0)/secs;
    const mf = r*DEP.U/(1 + r*DEP.U*DEP.tauRec/1000);
    console.log('    ' + (hz + ' Hz').padEnd(9) + eng.toFixed(3).padStart(7) + ref.toFixed(3).padStart(11) + mf.toFixed(3).padStart(10));
    judged('depressing, Poisson ' + hz + ' Hz: released per second within 5 percent of equation 3.4',
      Math.abs(ref - mf) <= 0.05*mf, Math.abs(eng - mf) <= 0.05*mf,
      'engine ' + eng.toFixed(3) + ', equations ' + ref.toFixed(3) + ', eq. 3.4 ' + mf.toFixed(3));
    if(hz === 40){
      const sat = 1000/DEP.tauRec;
      judged('depressing: above the limiting frequency transmission saturates (40 Hz within 10 percent of 1/tau_rec)',
        Math.abs(ref - sat) <= 0.1*sat, Math.abs(eng - sat) <= 0.1*sat, eng.toFixed(3) + ' against ' + sat.toFixed(3));
    }
    if(hz === 1)
      judged('depressing: below the limiting frequency transmission follows the rate (1 Hz at least 0.6 r U)',
        ref >= 0.6*r*DEP.U, eng >= 0.6*r*DEP.U, eng.toFixed(3) + ' against r U ' + (r*DEP.U).toFixed(3));
  });

  // facilitating synapse, Fig. 1B to D
  const FAC = { U:0.03, tauRec:130, tauF:530 };
  const facHz = Array.from({ length:100 }, (_, q) => q + 1);
  const facTrains = facHz.map(hz => regular(hz, 100, 4100));
  const fac = stpRun(FAC, facTrains, 4102);
  exact('facilitating', FAC, facTrains, fac);
  const level = (tr, rels) => tr.reduce((s, t, i) => s + (t >= 3100 && t < 4100 ? rels[i] : 0), 0);
  const engLevel = facTrains.map((tr, j) => level(tr, fac.rel[j]));
  const refLevel = facTrains.map(tr => level(tr, paper(tr, FAC)));
  const argmax = a => a.reduce((b, x, i) => x > a[b] ? i : b, 0);
  const engPeak = facHz[argmax(engLevel)], refPeak = facHz[argmax(refLevel)];
  const tf = FAC.tauF/1000, tr_ = FAC.tauRec/1000;
  const theta = 1/tf + Math.sqrt(2/(tf*tf) + (1 + FAC.U)/(FAC.U*tr_*tf));
  console.log('  facilitating, U 0.03, tau_rec 130 ms, tau_facil 530 ms; released per second over 3 to 4 s');
  console.log('    ' + [5, 10, 15, 20, 25, 30, 40, 50, 70, 100].map(hz => hz + ' Hz ' + engLevel[hz - 1].toFixed(3)).join(', '));
  console.log('    peak: engine ' + engPeak + ' Hz, equations ' + refPeak + ' Hz, eq. 3.6 theta ' + theta.toFixed(1) + ' Hz');
  ok('facilitating: the engine peaks where equations 2.1 and 2.2 do (within 1 Hz)', Math.abs(engPeak - refPeak) <= 1,
    engPeak + ' against ' + refPeak + ' Hz');
  judged('facilitating: the stationary level peaks within 25 percent of equation 3.6 (Fig. 1D)',
    Math.abs(refPeak - theta) <= 0.25*theta, Math.abs(engPeak - theta) <= 0.25*theta,
    'engine ' + engPeak + ' Hz, equations ' + refPeak + ' Hz, theta ' + theta.toFixed(1) + ' Hz');
  // the second reading, added after the first run: release per spike
  const count = tr => tr.filter(t => t >= 3100 && t < 4100).length;
  const engAmp = facTrains.map((tr, j) => engLevel[j]/Math.max(1, count(tr)));
  const refAmp = facTrains.map((tr, j) => refLevel[j]/Math.max(1, count(tr)));
  const engAmpPeak = facHz[argmax(engAmp)], refAmpPeak = facHz[argmax(refAmp)];
  console.log('    release per spike: ' + [5, 10, 15, 20, 25, 30, 40, 50, 70, 100].map(hz => hz + ' Hz ' + engAmp[hz - 1].toFixed(3)).join(', '));
  console.log('    peak per spike: engine ' + engAmpPeak + ' Hz, equations ' + refAmpPeak + ' Hz (second reading, added after the first run)');
  ok('facilitating: the release per spike peaks where equations 2.1 and 2.2 do (within 1 Hz)', Math.abs(engAmpPeak - refAmpPeak) <= 1,
    engAmpPeak + ' against ' + refAmpPeak + ' Hz');
  judged('facilitating: the release per spike peaks within 25 percent of equation 3.6 (second reading of Fig. 1D)',
    Math.abs(refAmpPeak - theta) <= 0.25*theta, Math.abs(engAmpPeak - theta) <= 0.25*theta,
    'engine ' + engAmpPeak + ' Hz, equations ' + refAmpPeak + ' Hz, theta ' + theta.toFixed(1) + ' Hz');
}

// ---- Vogels et al. 2011, inhibitory plasticity sets the postsynaptic rate ----------
{
  const OUT = path.join(TMP, 'vogels.json');
  execFileSync(PY, [path.join(HERE, 'vogels2011.py'), OUT], { stdio:['ignore', 'ignore', 'inherit'] });
  const prog = JSON.parse(fs.readFileSync(OUT, 'utf8'));
  const target = prog.alpha/(2*prog.eta*prog.tau/1000);
  const tail = prog.rates.slice(-20);
  const progRate = tail.reduce((s, x) => s + x, 0)/tail.length;

  const T = engineTypes();
  const SP = T.length;
  T.push({ f7:{ C:1, k:0, vr:0, vt:0, vpeak:1, a:0, b:0, c:0, d:0 } });
  const RSr = NEURON_TYPES.findIndex(t => t.key === 'RS');
  const NI = 1000, POST = NI, n = NI + 1, MS = 150000, HZ = 10, DRIVE = 20, PULSE = 300;
  const ntype = Uint8Array.from({ length:n }, (_, i) => i < NI ? SP : RSr);
  const bias = new Float32Array(n); bias[POST] = DRIVE;
  const v0 = new Float32Array(n); v0[POST] = T[RSr].f7.vr;
  replies.length = 0;
  send({ cmd:'init', protocol:PROTOCOL, count:n, ntype, bias, syn:0, psc:0, tauE:3, tauI:8, protocols:[],
    inputs:[{ chStart:Int32Array.from({ length:NI + 1 }, (_, j) => j), chIdx:Int32Array.from({ length:NI }, (_, j) => j),
      chW:new Float32Array(NI).fill(1), amp:1 }],
    plast:1, aP:0, aM:0, tauS:20, tauM:20, wmax:100, wdep:0, iEta:0.02, iRho:target, rhoMode:0,
    trip:0, het:0, tin:0, cons:0, scale:0, refrac:2, seed:1, v0, u0:new Float32Array(n),
    preStart:Int32Array.from({ length:n + 1 }, (_, i) => Math.min(i, NI)), post:new Int32Array(NI).fill(POST),
    w:new Float32Array(NI).fill(-0.01), delay:new Uint8Array(NI).fill(1), types:T });
  const err = replies.find(r => r.cmd === 'error');
  if(err) throw new Error('engine refused the inhibitory plasticity cell: ' + err.message);
  const rnd = mulberry(2011), gains = new Float32Array(NI);
  let early = 0, late = 0;
  for(let t = 0; t < MS; t++){
    for(let j = 0; j < NI; j++) gains[j] = rnd() < HZ/1000 ? PULSE : 0;
    send({ cmd:'inputFrame', mi:0, gains });
    replies.length = 0;
    send({ cmd:'tick', steps:1 });
    const st = replies.find(r => r.cmd === 'state');
    if(st.fired[POST]){ if(t < 2000) early++; if(t >= MS - 25000) late++; }
  }
  replies.length = 0;
  send({ cmd:'getWeights' });
  const w = replies.find(r => r.cmd === 'weights').w;
  const wMean = w.reduce((s, x) => s + x, 0)/w.length;
  const earlyHz = early/2, lateHz = late/25;
  console.log('published: Vogels et al. 2011, inhibitory plasticity sets the postsynaptic rate');
  console.log('  program: rate per 1.5 s run ' + prog.rates.slice(0, 3).map(x => x.toFixed(1)).join(', ') + ' ... ' +
    prog.rates.slice(-3).map(x => x.toFixed(1)).join(', ') + ' Hz; last 30 s ' + progRate.toFixed(2) +
    ' Hz; mean inhibitory weight ' + prog.winh[0].toFixed(3) + ' to ' + prog.winh[prog.winh.length - 1].toFixed(3));
  console.log('  engine: first 2 s ' + earlyHz.toFixed(1) + ' Hz, last 25 s ' + lateHz.toFixed(2) + ' Hz; mean inhibitory weight ' +
    wMean.toFixed(3) + '; target ' + target.toFixed(2) + ' Hz');
  const progFits = Math.abs(progRate - target) <= 0.15*target;
  if(progFits)
    ok('Vogels: the published program settles within 15 percent of alpha/(2 eta tau)', true, progRate.toFixed(2) + ' against ' + target.toFixed(2) + ' Hz');
  else
    console.log('    the published program settles at ' + progRate.toFixed(2) + ' Hz against ' + target.toFixed(2) +
      ', outside 15 percent, so the rate criteria are recorded, not gated');
  const engLabel = 'Vogels: the engine\'s rule brings the rate within 15 percent of its target';
  if(progFits) ok(engLabel, Math.abs(lateHz - target) <= 0.15*target, lateHz.toFixed(2) + ' against ' + target.toFixed(2) + ' Hz');
  else console.log('    ' + engLabel + ': ' + lateHz.toFixed(2) + ' against ' + target.toFixed(2) + ' Hz, recorded');
  ok('Vogels: the cell started well away from the target (first 2 s at least twice it)', earlyHz >= 2*target,
    earlyHz.toFixed(1) + ' against ' + target.toFixed(2) + ' Hz');
}

fs.rmSync(TMP, { recursive:true, force:true });
report('published');

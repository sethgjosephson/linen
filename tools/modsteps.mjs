// The modulator field against its equation (MODEL.md 4, neuromodulation), on the reference engine.
// The field at a cell is read through a pairing: under the gain form with the learning base at 0, one pre-post pairing moves the weight by the level at the postsynaptic cell times the pair rule's increment, so the weight after the pairing gives the level as the step before the post spike left it.
// Three cells: a presynaptic and a postsynaptic cell fired through an input map at chosen steps, and a third cell, fired by a pulse train, that releases onto the postsynaptic cell when a releasing population is measured.
// The first table reads the field under each kind of source at chosen steps against the equation stepped in float64.
// The second pairs the cells once and follows the pairing with a 50 ms pulse of the modulator at a delay, under the gain form and under the eligibility form, against the sweeps stepped in float64.
// The third holds a level through the run under both forms.
//
//   node tools/modsteps.mjs
import { PROTOCOL } from '../src/protocol.js';
import { initState } from '../src/rand.js';
import { engineTypes, NEURON_TYPES, engineConfig } from '../src/nodes.js';

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const handle = globalThis.onmessage;
const send = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };

const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07');
const AP = 0.1, TAU = 20, MODTAU = 200, W0 = 0.5, GAP = 5, ELIG_TAU = 1000, SWEEP = 10;
const DEC = Math.exp(-1/MODTAU), KP = Math.exp(-GAP/TAU), DECE = Math.exp(-SWEEP/ELIG_TAU), PAY = 1 - DECE;
const C0 = Math.fround(AP*KP);   // the pair rule's increment for the pairing, as the engine stores it

// cell 0 sends to cell 1 through one plastic synapse; cell 2 fires on a pulse train and releases onto cell 1 when `release` is given
function init(extra, release){
  const ntype = Uint8Array.from([RS, RS, RS]), bias = new Float32Array(3);
  const net = { count:3, ntype, bias, preStart:Int32Array.from([0, 1, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([W0]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:AP, aM:AP, tauS:TAU, tauM:TAU, wmax:30, wdep:0, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    modTau:MODTAU, modPlast:1, modBase:0, modStep:SWEEP, eligTau:ELIG_TAU,
    protocols:release ? [{ idx:Uint32Array.from([2]), amp:3000, mode:1, period:release.period, width:2, t0:0, duration:0 }] : [],
    ...(release ? { modGrp:Uint8Array.from([0, 0, 1]), modTgt:Uint16Array.from([0, 1, 0]), modK:Float32Array.from([0, release.level*1000]) } : {}),
    ...extra };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:3, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// pre fires at tPre, post at tPost; the weight after `steps` steps and the steps at which cell 2 fired
function run(extra, tPre, tPost, steps, release){
  send(init(extra, release));
  const rel = [];
  for(let t = 0; t < steps; t++){
    send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t === tPre ? 1 : 0, t === tPost ? 1 : 0]) });
    const st = send({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state');
    if(st.fired[2]) rel.push(t);
  }
  return { w:send({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0], rel };
}
// the source's scale at step t, as the engine's scaleOf has it
function scaleOf(p, t){
  const dt = t - p.t0;
  if(dt < 0) return 0;
  if(p.duration > 0 && dt >= p.duration) return 0;
  if(p.mode === 1) return (dt % p.period) < p.width ? 1 : 0;
  if(p.mode === 2) return Math.min(1, dt/p.period);
  return 1;
}
// the equation in float64: the field after the update of step T, under a source and the releasing cell's spikes (counted the step after)
function field(src, rel, relSpikes, T){
  let m = 0;
  const fired = new Set(relSpikes);
  for(let t = 0; t <= T; t++){
    m *= DEC;
    if(src) m += src.amp*(1 - DEC)*scaleOf(src, t);
    if(rel && fired.has(t - 1)) m += rel.level*1000*(1 - DEC);
  }
  return m;
}
const onPost = (amp, extra = {}) => ({ idx:Uint32Array.from([1]), amp, mode:0, period:100, width:10, t0:0, duration:0, ...extra });
// the field after the update of step t, read by a pairing whose post spike is at t + 1
function read(src, rel, t){
  const r = run({ modSrc:src ? [src] : [] }, t + 1 - GAP, t + 1, t + 2, rel);
  return { m:(r.w - W0)/(AP*KP), rel:r.rel };
}
const fmt = (x, d = 6) => x.toFixed(d);

// 1. the field under each kind of source
console.log('| source | step | equation | engine | difference |');
console.log('|---|---|---|---|---|');
const SOURCES = [
  ['constant, level 1', onPost(1), null, [10, 20, 50, 100, 200, 400, 600, 1000]],
  ['pulse train, level 1, 10 of every 100 ms', onPost(1, { mode:1, period:100, width:10 }), null, [5, 9, 10, 20, 50, 99, 105, 109, 110, 200]],
  ['ramp, level 1 over 500 ms', onPost(1, { mode:2, period:500 }), null, [50, 100, 250, 500, 750, 1000]],
  ['released, 0.05 per Hz, one cell at 20 Hz', null, { level:0.05, period:50 }, [100, 500, 1000, 1500, 2000]],
];
let worst = 0;
for(const [name, src, rel, times] of SOURCES){
  for(const t of times){
    const got = read(src, rel, t);
    const want = field(src, rel, got.rel, t);
    worst = Math.max(worst, Math.abs(got.m - want));
    console.log('| ' + name + ' | ' + t + ' | ' + fmt(want) + ' | ' + fmt(got.m) + ' | ' + (got.m - want).toExponential(1) + ' |');
  }
}
console.log('\nlargest difference ' + worst.toExponential(1) + ' (float32 stores against the equation in float64)');
{
  // the released field over a second, against the level times the releasing cell's rate
  const rel = { level:0.05, period:50 };
  let sum = 0, n = 0, rate = 0, first = [];
  for(let t = 1000; t < 1050; t++){ const got = read(null, rel, t); sum += got.m; n++; rate = got.rel.filter(s => s < 1000).length; first = got.rel.slice(0, 3); }
  console.log('released field over steps 1000 to 1049: mean ' + fmt(sum/n, 4) + '; the releasing cell fired ' + rate + ' times in the first second (at steps ' + first.join(', ') + ', ...), level times rate ' + fmt(rel.level*rate, 4));
}

// 2. one pairing, then a pulse of the modulator at a delay
const PRE = 100, POST = PRE + GAP, PW = 50, LVL = 2;
// the sweeps in float64: the trace C0 written at the post spike, the sweep at every step t with t mod 10 = 9 after it paying c (1 - e^(-10/1000)) (base + M(t)) and decaying c
function sweeps(src, base, steps){
  let c = C0, dw = 0, m = 0;
  for(let t = 0; t < steps; t++){
    m *= DEC;
    if(src) m += src.amp*(1 - DEC)*scaleOf(src, t);
    if(t >= POST && t % SWEEP === SWEEP - 1){ dw += c*PAY*(base + m); c *= DECE; }
  }
  return dw;
}
console.log('\n| pulse onset after the pairing (ms) | gain form | eligibility form | eligibility, the sweeps in float64 | e^(-delay / trace tau) |');
console.log('|---|---|---|---|---|');
for(const d of [-50, 0, 100, 250, 500, 1000, 2000, 4000]){
  const src = onPost(LVL, { mode:1, period:1e7, width:PW, t0:POST + d });
  const steps = POST + Math.max(d, 0) + PW + 2000;
  const gain = run({ modPlast:1, modBase:0, modSrc:[src] }, PRE, POST, steps).w - W0;
  const elig = run({ modPlast:2, modBase:0, modSrc:[src] }, PRE, POST, steps).w - W0;
  console.log('| ' + d + ' | ' + fmt(gain) + ' | ' + fmt(elig) + ' | ' + fmt(sweeps(src, 0, steps)) + ' | ' + fmt(Math.exp(-Math.max(d, 0)/ELIG_TAU), 3) + ' |');
}
console.log('\nthe pairing\'s increment is ' + fmt(C0) + '; a level of ' + LVL + ' for ' + PW + ' ms leaves ' + fmt(LVL*(1 - Math.exp(-PW/MODTAU)), 4) + ' at the end of the pulse');

// 3. a level held through the run
console.log('\n| level | gain form | eligibility form | eligibility, the sweeps in float64 |');
console.log('|---|---|---|---|');
for(const [name, extra, src] of [
  ['none, learning base 1', { modBase:1 }, null],
  ['1 held from the start, base 0', { modBase:0 }, onPost(1)],
  ['1 held from the start, base 1', { modBase:1 }, onPost(1)],
]){
  const steps = 8000, mods = src ? [src] : [];
  const gain = run({ modPlast:1, ...extra, modSrc:mods }, PRE, POST, steps).w - W0;
  const elig = run({ modPlast:2, ...extra, modSrc:mods }, PRE, POST, steps).w - W0;
  console.log('| ' + name + ' | ' + fmt(gain) + ' | ' + fmt(elig) + ' | ' + fmt(sweeps(src, extra.modBase, steps)) + ' |');
}

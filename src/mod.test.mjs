// Neuromodulation on the reference engine (MODEL.md 4): the modulator field per cell, raised by a time-course source or by a releasing population's spikes, read by the Hebbian terms as a gain or through an eligibility trace, by the synaptic current as a gain, and as a current.
// Two cells through one plastic synapse, driven over the peak in chosen steps through an input map, as src/plastgain.test.mjs drives them; the field is emulated in float32 beside the run.
// Run: node src/mod.test.mjs
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from './protocol.js';
import { initState } from './rand.js';
import { engineTypes, NEURON_TYPES, engineConfig } from './nodes.js';

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('./simworker.js');
const handle = globalThis.onmessage;
const send = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };

const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07');
const AP = 0.1, TAU = 20, MODTAU = 200, W0 = 0.5;
// cell 0 sends to cell 1 through one plastic synapse; the modulation settings and sources are fields of the init as engineConfig carries them
function init(extra = {}, { w = W0, synapse = true } = {}){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias,
    preStart:Int32Array.from(synapse ? [0, 1, 1] : [0, 0, 0]), post:Int32Array.from(synapse ? [1] : []),
    w:Float32Array.from(synapse ? [w] : []), delay:Uint8Array.from(synapse ? [1] : []), pmask:Uint8Array.from(synapse ? [1] : []),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:AP, aM:AP, tauS:TAU, tauM:TAU, wmax:30, wdep:0, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    modTau:MODTAU, protocols:[], ...extra };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// pre fires at tPre, post at tPost; the weight after `steps` steps, and the spike count of each cell
function run(c, tPre, tPost, steps = 40, opts){
  send(init(c, opts));
  const spikes = [0, 0];
  for(let t = 0; t < steps; t++){
    send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t === tPre ? 1 : 0, t === tPost ? 1 : 0]) });
    const st = send({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state');
    spikes[0] += st.fired[0]; spikes[1] += st.fired[1];
  }
  const wm = send({ cmd:'getWeights' }).find(x => x.cmd === 'weights');
  return { w:wm.w.length ? wm.w[0] : NaN, spikes };
}
const f32 = x => Math.fround(x);
const near = (a, b, tol = 1e-6) => Math.abs(a - b) < tol;
const DEC = f32(Math.exp(-1/MODTAU));
// the field at a cell after `steps` updates, under a constant level (the increment amp (1 - dec) per step) from step from0, in the engine's float32 stores
function fieldAt(level, steps, from0 = 0, until = 1e9){
  const inc = f32(level*(1 - DEC));
  let m = 0;
  for(let t = 0; t < steps; t++) m = f32(m*DEC + (t >= from0 && t < until ? inc : 0));
  return m;
}
const kp = Math.exp(-5/TAU);   // the pre trace at the post spike, pre at 2 and post at 7
const src = (level, extra = {}) => ({ idx:Uint32Array.from([1]), amp:level, mode:0, period:100, width:10, t0:0, duration:0, ...extra });

// 1. the gain form: the Hebbian terms scale with modBase + the field at the postsynaptic cell, the field as the step before left it
{
  const plain = run({ modPlast:1, modBase:1 }, 2, 7).w;
  ok('gain form with base 1 and no source is the plain rule', near(plain, f32(W0 + AP*kp)), 'w ' + plain);
  const L = 20;
  const g = run({ modPlast:1, modBase:0, modSrc:[src(L)] }, 2, 7).w;
  const m6 = fieldAt(L, 7);
  ok('with base 0 the pairing scales with the field at the post spike', near(g, f32(W0 + m6*AP*kp)), 'w ' + g + ' expected ' + f32(W0 + m6*AP*kp) + ' (field ' + m6 + ')');
  const d = run({ modPlast:1, modBase:0, modSrc:[src(L)] }, 7, 2).w;
  ok('depression scales the same way, with the field at the pre spike', near(d, f32(W0 - m6*AP*kp)), 'w ' + d + ' expected ' + f32(W0 - m6*AP*kp));
  const off = run({ modPlast:0, modBase:0, modSrc:[src(L)] }, 2, 7).w;
  ok('with learning unmodulated the source changes nothing', off === f32(W0 + AP*kp), 'w ' + off);
}
// 2. the eligibility form: the pairing writes the trace, a later pulse of the modulator pays it into the weight at the sweeps
{
  const none = run({ modPlast:2, modBase:0, modStep:10, eligTau:1000 }, 2, 7).w;
  ok('eligibility with base 0 and no modulator leaves the weight where it was', none === f32(W0), 'w ' + none);
  // with base 1 and no modulator each sweep pays (1 - dec) of the trace: four sweeps pay 1 - dec^4 of the pairing, a long run all of it
  const decE = f32(Math.exp(-10/1000)), pay = f32(1 - decE);
  const base1 = run({ modPlast:2, modBase:1, modStep:10, eligTau:1000 }, 2, 7).w;
  { let c = f32(AP*kp), w = f32(W0);
    for(let k = 0; k < 4; k++){ w = f32(w + c*pay*1); c = f32(c*decE); }
    ok('eligibility with base 1 pays 1 - dec^4 of the pairing over four sweeps', near(base1, w, 1e-6), 'w ' + base1 + ' expected ' + w); }
  const long = run({ modPlast:2, modBase:1, modStep:10, eligTau:1000 }, 2, 7, 8000).w;
  ok('and the whole pairing over the life of the trace', near(long, f32(W0 + AP*kp), 2e-4), 'w ' + long + ' against ' + f32(W0 + AP*kp));
  // a pulse from step 20 to 29: the sweeps at 9 and 19 see no field, the ones at 29 and 39 pay the decayed trace in
  const L = 20;
  const got = run({ modPlast:2, modBase:0, modStep:10, eligTau:1000, modSrc:[src(L, { mode:1, period:100000, width:10, t0:20 })] }, 2, 7).w;
  let c = f32(AP*kp), w = f32(W0);
  for(const t of [9, 19, 29, 39]){
    const m = fieldAt(L, t + 1, 20, 30);
    if(c !== 0 && m !== 0) w = f32(w + c*pay*m);
    c = f32(c*decE);
  }
  ok('a later pulse pays the decayed trace into the weight at each sweep', near(got, w, 1e-6), 'w ' + got + ' expected ' + w);
  ok('(and the pulse moved it)', got > W0 + 1e-3, 'w ' + got);
}
// 3. a releasing population: the presynaptic cell's spike raises the field at the target one step later, by level times 1000 / N per spike
{
  const level = 1;
  const rel = { modGrp:Uint8Array.from([1, 0]), modTgt:Uint16Array.from([0, 1]), modK:Float32Array.from([0, level*1000/1]) };
  const g = run({ modPlast:1, modBase:0, ...rel }, 2, 7).w;
  // the spike at step 2 is counted into step 3's update: M(3) = K (1 - dec), decaying after, read at the post spike as M(6)
  const k = f32(level*1000*(1 - DEC));
  let m = 0;
  for(let t = 0; t < 7; t++) m = f32(m*DEC + (t === 3 ? k : 0));
  ok('a releasing cell raises the field at its target the step after its spike', near(g, f32(W0 + m*AP*kp), 1e-6), 'w ' + g + ' expected ' + f32(W0 + m*AP*kp));
}
// 4. the synaptic gain and the current: the field scales what a cell receives through its synapses, and adds a current of its own
{
  // the pre cell fires every 20 ms through a pulse stimulus; the post cell's spikes count what the synapse delivers
  const pulse = { idx:Uint32Array.from([0]), amp:3000, mode:1, period:20, width:2, t0:0, duration:0 };
  const flat = run({ plast:0, protocols:[pulse] }, -1, -1, 2000, { w:1500 }).spikes;
  const gained = run({ plast:0, protocols:[pulse], modDrive:1, modSrc:[src(2)] }, -1, -1, 2000, { w:1500 }).spikes;
  ok('the pre cell fires on its pulses either way', flat[0] > 50 && gained[0] === flat[0], flat[0] + ' and ' + gained[0] + ' pre spikes');
  ok('a synaptic gain of 1 + 2 makes the post cell fire on input that left it silent', gained[1] > 20 && gained[1] > 2*flat[1], flat[1] + ' post spikes against ' + gained[1]);
  const quiet = run({ plast:0 }, -1, -1, 1000, { synapse:false }).spikes;
  const driven = run({ plast:0, modCur:100, modSrc:[src(1)] }, -1, -1, 1000, { synapse:false }).spikes;
  ok('a current per level drives a cell with no synapses', quiet[1] === 0 && driven[1] > 5, quiet[1] + ' then ' + driven[1] + ' spikes');
}
report('mod');

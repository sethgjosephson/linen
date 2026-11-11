// The global factor on the inhibitory rule (Zenke, Agnes and Gerstner 2015, their GlobalPFConnection; MODEL.md 4): every inhibitory update is the rule's eta times (the population rate over the target minus one), the rate the low-passed spike count of the cells the rule inhibits, and the presynaptic term is the trace plus one.
// Three cells on the reference engine in this process: an inhibitory cell onto one excitatory cell, and a second excitatory cell no inhibition reaches, which must not count toward the population.
// Spikes are forced through an input map at known steps; the weight after each is compared with the arithmetic (float64 factor, float32 traces and weight, the factor read as the step before left it).
// Run: node src/iglobal.test.mjs
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

const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07'), FS = NEURON_TYPES.findIndex(t => t.key === 'FS07');
const TAU = 20, ETA = 0.5, RHO = 5, TAUG = 1;   // seconds
function init(iGlobal){
  const ntype = Uint8Array.from([FS, RS, RS]), bias = new Float32Array(3);
  // cell 0 (inhibitory) onto cell 1 at -1; cell 2 gets nothing
  const net = { count:3, ntype, bias, preStart:Int32Array.from([0, 1, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([-1]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0, aM:0, tauS:TAU, tauM:TAU, wmax:30, wdep:0, stdpAt:0, iEta:ETA, iRho:RHO, refrac:2, vmin:-90,
    iGlobal, iTauG:TAUG, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:3, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2, 3]), chIdx:Int32Array.from([0, 1, 2]), chW:Float32Array.from([1, 1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// the forced spikes: the inhibitory cell at 10 and 30, the inhibited cell at 20, the free cell at 25
const fireAt = { 0:[10, 30], 1:[20], 2:[25] };
function run(iGlobal, upTo){
  send(init(iGlobal));
  const ws = {};
  for(let t = 0; t < upTo; t++){
    send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([0, 1, 2].map(c => fireAt[c].includes(t) ? 1 : 0)) });
    send({ cmd:'tick', steps:1 });
    ws[t] = send({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0];
  }
  return ws;
}
const f32 = Math.fround;
const near = (a, b, tol = 1e-6) => Math.abs(a - b) < tol;
const decS = Math.exp(-1/TAU), decG = Math.exp(-1/(TAUG*1000));
// the factor's state after step t: the low-passed count of cell 1's spikes, started at the target times the population of one
// the engine keeps the count and the ratio in float32, and the population's inverse times the target as one float32 number
const gInv = f32(1/(1*RHO));
function ratioAfter(t){
  let pf = f32(RHO*1);
  for(let s = 0; s <= t; s++){ pf = f32(pf*decG + (fireAt[1].includes(s) ? 1 : 0)/TAUG); }
  return f32(pf*gInv - 1);
}
const ws = run(1, 40);
// at 10: the inhibitory cell fires with the inhibited cell's trace at zero: d = -eta ratio(9) (0 + 1)
let w = f32(-1 + (-ETA*ratioAfter(9)*(0 + 1)));
ok('at the first inhibitory spike the weight moves by -eta times the ratio times (trace + 1)', near(ws[10], w), ws[10] + ' expected ' + w);
ok('and the ratio is negative, so the inhibition weakens', ws[10] > -1, String(ws[10]));
// at 20: the inhibited cell fires; the presynaptic trace is one bumped at 10 and decayed ten steps
const kpre20 = f32(Math.pow(decS, 10));
w = f32(w + (-ETA*ratioAfter(19)*kpre20));
ok('at the postsynaptic spike the weight moves by -eta times the ratio times the presynaptic trace', near(ws[20], w), ws[20] + ' expected ' + w);
ok('the free cell\'s spike at 25 changes nothing', near(ws[25], ws[20]) && near(ws[29], ws[20]), ws[25] + ' against ' + ws[20]);
// at 30: the inhibitory cell fires again; the ratio now carries the inhibited cell's spike of 20, and its trace is one decayed ten steps
const kpost30 = f32(Math.pow(decS, 10));
w = f32(w + (-ETA*ratioAfter(29)*(kpost30 + 1)));
ok('at the second inhibitory spike the ratio carries the population spike and the trace plus one', near(ws[30], w), ws[30] + ' expected ' + w);
ok('the population rate counts only the inhibited cell (the ratio would differ with the free cell in it)', !near(ws[30], f32(ws[20] + (-ETA*((((RHO*2) + Math.pow(decG, 4)/TAUG + Math.pow(decG, 9)/TAUG)/2/RHO - 1))*(kpost30 + 1)))));
// the Vogels form is untouched when the switch is off
const wv = run(0, 12);
const alpha = RHO*0.001*(TAU + TAU);
ok('with the switch off the first inhibitory spike applies the Vogels term, -eta (trace - alpha)', near(wv[10], f32(-1 + (-ETA*(0 - alpha)))), wv[10] + ' expected ' + f32(-1 + ETA*alpha));
report('iglobal');

// The consolidation reference starts inside its well (MODEL.md 4): a synapse wired above consW under a consolidating rule starts its reference at consW, so with commitment on it is committed from the first step; a synapse inside the well commits only once its reference has climbed to nine tenths of consW.
// One pre-before-post pairing on the reference engine in this process through an input map, after a stretch of idle time long enough for the consolidation cadence to run.
// Run: node src/consref.test.mjs
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
function init({ w, commit = 1, cons = 1, het = 0, consW = 1.2, consP = 10, tauCons = 0.5 }){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  // tauCons is in simulated minutes on the net and milliseconds in the engine; half a minute makes the rate 0.04 per 1.2 s step
  const net = { count:2, ntype, bias, preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([w]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0.1, aM:0.01, tauS:20, tauM:20, wmax:30, wdep:1, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    het, cons, consW, consP, tauCons, consStep:1.2, commit, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// idle for idleMs, then pre at +2 and post at +7: potentiation by aP e^(-5/tau) unless the synapse is committed
function run(c, idleMs){
  send(init(c));
  send({ cmd:'tick', steps:idleMs });
  for(let t = 0; t < 40; t++){
    send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t === 2 ? 1 : 0, t === 7 ? 1 : 0]) });
    send({ cmd:'tick', steps:1 });
  }
  return send({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0];
}
const f32 = x => Math.fround(x);
const near = (a, b) => Math.abs(a - b) < 1e-6;
const dP = 0.1*Math.exp(-5/20);
// wired at 16, far above consW 1.2: the reference starts at 1.2 and the synapse is committed before the first consolidation step
const strong = run({ w:16 }, 2000);
ok('a synapse wired above consW is committed from the first step', strong === f32(16), 'w ' + strong);
const strongOpen = run({ w:16, commit:0 }, 2000);
ok('the same synapse potentiates with commitment off', near(strongOpen, f32(16 + dP)), 'w ' + strongOpen);
// wired at 1.0, inside the well above its midpoint: one consolidation step moves the reference to 1.032, short of 1.08, so the pairing still potentiates
const early = run({ w:1 }, 2000);
ok('inside the well the synapse is still plastic after one consolidation step', near(early, f32(1 + dP)), 'w ' + early);
// after twenty seconds the reference has climbed past nine tenths of consW and the synapse has committed
const late = run({ w:1 }, 20000);
ok('and committed once the reference has climbed', late === f32(1), 'w ' + late);
// a synapse held above consW: the reference settles above consW too, where the weight's pull balances the well's (their P10Connection has no limit on it) Read through the heterosynaptic pull: post alone at 7 and 12, no pre, so the second spike moves the weight by beta times the cubed trace times its distance from the reference.
function heldAbove(idleMs){
  const c = { w:1, commit:0, het:2, consW:0.5, consP:10, tauCons:0.2 };
  send(init(c));
  send({ cmd:'tick', steps:idleMs });
  for(let t = 0; t < 30; t++){
    send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([0, t === 7 || t === 12 ? 1 : 0]) });
    send({ cmd:'tick', steps:1 });
  }
  return send({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0];
}
{ // the reference from zero under the same steps, in the engine's float32
  let r = 0.5; const rate = 1.2/12;
  for(let k = 0; k < 50; k++) r = Math.fround(r + rate*(Math.fround(1) - r - 10*r*(0.25 - r)*(0.5 - r)));
  const cube = Math.exp(-15/20), expect = f32(1 - 2*cube*(1 - r));
  ok('the reference of a synapse at 1 settles above the upper point 0.5, at ' + r.toFixed(3), r > 0.55, 'r ' + r);
  const got = heldAbove(60000);
  ok('and the heterosynaptic pull aims at it', Math.abs(got - expect) < 1e-4, 'w ' + got + ' expected ' + expect);
}
report('consref');

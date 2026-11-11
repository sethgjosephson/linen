// The heterosynaptic term in the form of Zenke, Agnes and Gerstner 2015 (MODEL.md 4): at a postsynaptic spike every incoming plastic excitatory weight moves toward its reference by beta times the cube of the cell's fast postsynaptic trace, read before that spike is added; no normalization, no ceiling.
// One synapse on the reference engine in this process, driven through an input map: a pre-before-post pairing lifts the weight above its reference (the wired weight, since consolidation is off), then a second postsynaptic spike five steps later pulls it back by beta times (e^(-5/tauM))^3 times the lift, beside the pairing's own potentiation.
// Run: node src/het.test.mjs
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
const TAU = 20;
function init({ het, w = 1, cons = 0, refInit = 0 }){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias, preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([w]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0.1, aM:0, tauS:TAU, tauM:TAU, wmax:30, wdep:1, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    het, cons, consW:0.5, consP:10, tauCons:20, consStep:1.2, refInit, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// pre at 2, post at 7 and again at 12 (or only at 7)
function run(c, secondPost){
  send(init(c));
  for(let t = 0; t < 30; t++){
    send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t === 2 ? 1 : 0, t === 7 || (secondPost && t === 12) ? 1 : 0]) });
    send({ cmd:'tick', steps:1 });
  }
  return send({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0];
}
const f32 = x => Math.fround(x);
const near = (a, b, tol = 2e-6) => Math.abs(a - b) < tol;
const lift = 0.1*Math.exp(-5/TAU);                 // the pairing at 7: aP Kpre, Kpre five steps old
const w1 = run({ het:0 }, false);
ok('the first pairing lifts the weight by aP e^(-5/tau)', near(w1, f32(1 + lift)), 'w ' + w1);
// At the second post spike Kpre is ten steps old and Kpost, from the spike at 7, five steps old; the reference is the wired weight 1.
const kq = Math.exp(-5/TAU), cube = kq*kq*kq;
const lift2 = 0.1*Math.exp(-10/TAU);
const w2off = run({ het:0 }, true);
ok('with beta 0 the second spike only potentiates again', near(w2off, f32(f32(1 + lift) + lift2)), 'w ' + w2off);
const beta = 2;
const w2 = run({ het:beta }, true);
const expect2 = f32(f32(1 + lift) + lift2 - beta*cube*(f32(1 + lift) - 1));
ok('with beta the second spike also pulls the weight toward its reference by beta Kpost^3 times the lift', near(w2, expect2, 1e-5), 'w ' + w2 + ' expected ' + expect2);
ok('and the pull is not capped at a full return', (() => { const big = run({ het:200 }, true); return big < 1 - 0.5*lift; })(), 'a beta of 200 overshoots the reference');
// the reference's start under consolidation: at zero the first pull aims at nothing, at the consolidated weight it aims at 0.5
const z = run({ het:beta, cons:1, refInit:1 }, true);
ok('with the reference started at zero the pull aims at zero', near(z, f32(f32(1 + lift) + lift2 - beta*cube*f32(1 + lift)), 1e-5), 'w ' + z);
const c = run({ het:beta, cons:1, refInit:2 }, true);
ok('started at the consolidated weight it aims at 0.5', near(c, f32(f32(1 + lift) + lift2 - beta*cube*(f32(1 + lift) - 0.5)), 1e-5), 'w ' + c);
report('het');

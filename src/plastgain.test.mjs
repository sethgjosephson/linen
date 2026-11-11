// The third factor (MODEL.md 4): plastGain multiplies the Hebbian terms of the pair rule (potentiation, depression, the triplet and transmitter terms) and leaves the homeostatic inhibitory rule alone.
// Two pairings on the reference engine in this process, each cell driven over the peak in a chosen step through an input map: an excitatory synapse at gain 1, 0.5 and 0, and an inhibitory synapse under the Vogels rule at gain 0.
// Run: node src/plastgain.test.mjs
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
// cell 0 sends to cell 1 through one plastic synapse; the gain is a field of the init as engineConfig carries it
function init({ gain, pre = RS, w = 0.5, iEta = 0 }){
  const ntype = Uint8Array.from([pre, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias, preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([w]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0.1, aM:0.1, tauS:20, tauM:20, wmax:30, wdep:0, stdpAt:0, iEta, iRho:5, refrac:2, vmin:-90,
    plastGain:gain, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// pre fires at tPre, post at tPost (frames of gain before each step); the weight after 40 steps
function run(c, tPre, tPost){
  send(init(c));
  for(let t = 0; t < 40; t++){
    send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t === tPre ? 1 : 0, t === tPost ? 1 : 0]) });
    send({ cmd:'tick', steps:1 });
  }
  return send({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0];
}
const f32 = x => Math.fround(x);
const near = (a, b) => Math.abs(a - b) < 1e-6;
const w1 = run({ gain:1 }, 2, 7), wHalf = run({ gain:0.5 }, 2, 7), w0 = run({ gain:0 }, 2, 7);
ok('at gain 1 the pairing potentiates by aP e^(-5/tau)', near(w1, f32(0.5 + 0.1*Math.exp(-5/20))), 'w ' + w1);
ok('at gain 0.5 it potentiates by half', near(wHalf, f32(0.5 + 0.5*0.1*Math.exp(-5/20))), 'w ' + wHalf);
ok('at gain 0 the weight does not move', w0 === f32(0.5), 'w ' + w0);
const d1 = run({ gain:1 }, 7, 2), d0 = run({ gain:0 }, 7, 2);
ok('depression follows the gain the same way', near(d1, f32(0.5 - 0.1*Math.exp(-5/20))) && d0 === f32(0.5), 'w ' + d1 + ' and ' + d0);
// the inhibitory rule is homeostatic and runs at gain 0: a pre spike with no post spike weakens the synapse by iEta alpha (-iEta (Kpost - alpha) with Kpost 0)
const i0 = run({ gain:0, pre:FS, w:-0.5, iEta:0.05 }, 2, 60);
ok('the inhibitory rule moves at gain 0', i0 !== f32(-0.5) && i0 > -0.5, 'w ' + i0);
report('plastgain');

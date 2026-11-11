// The sliding threshold of Zenke, Agnes and Gerstner 2015 (their P11Connection; MODEL.md 4): each cell keeps a slow trace of its own spikes, and the depression of its incoming excitatory synapses is scaled by the square of that rate estimate over a target rate.
// One synapse on the reference engine in this process, driven through an input map: a post spike at 5 and a pre spike at 10 depress the weight by A- times the fast trace (e^(-5/tauM)) times the factor, with the slow trace followed step by step in float32 as the engine keeps it.
// Run: node src/slide.test.mjs
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
const TAU = 20, AM = 0.1;
// slideTau in seconds here, as the checkpoint carries it; engineConfig turns it into ms
function init({ slide = 0, slideRate = 5, slideTau = 1, slideInit = 5 }){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias, preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([1]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0, aM:AM, tauS:TAU, tauM:TAU, wmax:30, wdep:0, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    slide, slideRate, slideTau, slideInit, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// post (cell 1) at 5, pre (cell 0) at 10
function run(c){
  send(init(c));
  for(let t = 0; t < 20; t++){
    send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t === 10 ? 1 : 0, t === 5 ? 1 : 0]) });
    send({ cmd:'tick', steps:1 });
  }
  return send({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0];
}
const f32 = x => Math.fround(x);
const near = (a, b, tol = 1e-5) => Math.abs(a - b) < tol;
// the slow trace as the engine keeps it: started at slideInit times the time constant, one added at the post spike, decayed by e^(-1/tau) each step in float32, read at the pre spike before that step's decay
function khomAt(slideInit, tauMs, postAt, readAt){
  const dec = Math.exp(-1/tauMs);
  let k = f32(slideInit*tauMs/1000);
  for(let t = 0; t < readAt; t++){ if(t === postAt) k = f32(k + 1); k = f32(k*dec); }
  return k;
}
const kq = Math.exp(-5/TAU);
const w0 = run({ slide:0 });
ok('with the threshold off the pre spike depresses by A- times the fast trace', near(w0, f32(1 - AM*kq)), 'w ' + w0);
for(const [label, slideInit, slideRate] of [['an estimate above the target depresses more', 5, 5], ['an estimate under the target depresses less', 2.5, 5], ['at a higher target the same estimate depresses less', 5, 10]]){
  const tauMs = 1000, slideK = 1000/(tauMs*slideRate);
  const kh = khomAt(slideInit, tauMs, 5, 10), sl = (kh*slideK)*(kh*slideK);
  const w = run({ slide:1, slideRate, slideTau:1, slideInit });
  ok(label + ' (factor ' + sl.toFixed(3) + ')', near(w, f32(1 - AM*kq*sl)), 'w ' + w + ' expected ' + f32(1 - AM*kq*sl));
}
// the estimate over a long trace barely moves within the test, so the factor is the starting rate over the target, squared
const wl = run({ slide:1, slideRate:10, slideTau:600, slideInit:2 });
ok('on a 600 s trace started at 2 Hz against a 10 Hz target the factor is near 0.04', near(wl, f32(1 - AM*kq*0.04), 2e-4), 'w ' + wl);
report('slide');

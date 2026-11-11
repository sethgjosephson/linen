// The weight exponent of soft depression (MODEL.md 4): depression scales with w^mu, 1 proportional to the weight, 0 additive, between the two the interpolation of Gutig et al. 2003.
// One post-before-pre pairing on the reference engine in this process through an input map, at weight 4: the depression at mu 1 is four times the additive one, at mu 0.5 twice, at mu 0 the same; potentiation does not depend on mu.
// Run: node src/wmu.test.mjs
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
function init({ mu, w = 4 }){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias, preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([w]), delay:Uint8Array.from([1]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0.1, aM:0.01, tauS:20, tauM:20, wmax:30, wdep:1, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    wmu:mu, protocols:[] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
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
// post at 2, pre at 7: depression by aM e^(-5/tau) w^mu
const d1 = run({ mu:1 }, 7, 2), dHalf = run({ mu:0.5 }, 7, 2), d0 = run({ mu:0 }, 7, 2);
const e = 0.01*Math.exp(-5/20);
ok('at mu 1 depression is proportional to the weight', near(d1, f32(4 - e*4)), 'w ' + d1);
ok('at mu 0.5 it scales with the square root', near(dHalf, f32(4 - e*2)), 'w ' + dHalf);
ok('at mu 0 it is additive', near(d0, f32(4 - e)), 'w ' + d0);
// pre at 2, post at 7: potentiation is additive whatever mu
const p1 = run({ mu:1 }, 2, 7), p0 = run({ mu:0 }, 2, 7);
ok('potentiation does not depend on mu', near(p1, f32(4 + 0.1*Math.exp(-5/20))) && p1 === p0, 'w ' + p1 + ' and ' + p0);
report('wmu');

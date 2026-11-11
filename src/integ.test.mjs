// The exponential membrane step (MODEL.md 1, the checkpoint's integ): two exponential half steps and the reset at the crossing inside the millisecond, against the same cell integrated on a 1 microsecond clock.
// One leaky integrate-and-fire cell per case on the reference engine in this process, driven by a constant bias; the Euler half steps round every interval up to the next millisecond, the exponential step does not.
// Run: node src/integ.test.mjs
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from './protocol.js';
import { initState } from './rand.js';
import { NEURON_TYPES } from './nodes.js';

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('./simworker.js');
const handle = globalThis.onmessage;
const send = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };

// the cell of src/rowfeat.test.mjs: a 12.5 nS leak on 250 pF, the quadratic's second root a volt away, the spike cut at -50 and reset to -70
const LIF = { C:250, k:12.5/1070, vr:-70, vt:1000, vpeak:-50, a:0.01, b:0, c:-70, d:0 };
function run(f7, bias, steps, integ){
  const types = [{ f7 }], ntype = Uint8Array.from([0]);
  const st = initState(3, ntype, types);
  send({ cmd:'init', protocol:PROTOCOL, count:1, ntype, bias:Float32Array.from([bias]),
    preStart:Int32Array.from([0, 0]), post:new Int32Array(0), w:new Float32Array(0), delay:new Uint8Array(0), pmask:new Uint8Array(0),
    syn:0, plast:0, refrac:0, vmin:-90, eRevI:-80, integ, types, seed:3, ...st, v0:Float32Array.from([-70]), u0:new Float32Array(1) });
  let n = 0;
  for(let t = 0; t < steps; t++) n += send({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state').fired[0];
  return n;
}
// the same cell on a 1 microsecond Euler clock: the threshold and the adaptation decay continuously, the spike is wherever the potential crosses
function fine(f7, bias, steps, eRevI = -80){
  const h = 0.001, dec = Math.exp(-h/(f7.thrTau > 0 ? f7.thrTau : 5));
  let v = -70, u = 0, thr = 0, n = 0;
  for(let q = 0; q < steps/h; q++){
    if(f7.thrJump > 0) thr *= dec;
    v += h*(f7.k*(v - f7.vr)*(v - f7.vt) - (f7.adaptCond ? u*(v - eRevI) : u) + bias)/f7.C;
    u += h*f7.a*(f7.b*(v - f7.vr) - u);
    if(v >= f7.vpeak + (f7.thrJump > 0 ? thr : 0)){ v = f7.c; u += f7.d; n++; if(f7.thrJump > 0) thr = f7.thrJump; }
  }
  return n;
}
for(const [label, f7, bias] of [
  ['a plain cell', LIF, 400],
  ['the dynamic threshold', { ...LIF, thrJump:100, thrTau:5 }, 600],
  ['the conductance adaptation', { ...LIF, d:1.25, adaptCond:1, thrJump:100, thrTau:5 }, 600]]){
  const steps = 4000, f = fine(f7, bias, steps), e = run(f7, bias, steps, 1), o = run(f7, bias, steps, 0);
  ok(label + ': the exponential step fires as the fine clock does (' + e + ' against ' + f + ' spikes in 4 s)', Math.abs(e - f) <= Math.max(1, 0.003*f), 'exponential ' + e + ', fine ' + f);
  ok(label + ': and the Euler half steps fire fewer, every interval rounded up (' + o + ')', f - o > Math.max(3, 0.01*f), 'Euler ' + o + ', fine ' + f);
}
// below threshold the two steps agree to within the Euler error, and integ 0 is the engine as it was
ok('a cell held below threshold fires under neither step', run(LIF, 200, 2000, 1) === 0 && run(LIF, 200, 2000, 0) === 0);

// The equations' step (integ 2): a linear row runs as under integ 1, and a quadratic row is solved exactly over each half step.
// Each quadratic case at twice its rheobase on the 1 microsecond clock (RS07 and FS07 the 2007 rows, CH and FS the 2003 rows in the 2007 form).
ok('integ 2 runs a linear row as integ 1 does', run(LIF, 400, 4000, 2) === run(LIF, 400, 4000, 1));
for(const [key, bias] of [['RS07', 104], ['FS07', 116], ['CH', 5.42], ['FS', 6.66]]){
  const f7 = NEURON_TYPES.find(t => t.key === key).f7, steps = 4000;
  const f = fine(f7, bias, steps), e = run(f7, bias, steps, 2), o = run(f7, bias, steps, 0);
  ok(key + ': the step on the equations fires as the fine clock does (' + e + ' against ' + f + ' spikes in 4 s; the half steps ' + o + ')', Math.abs(e - f) <= Math.max(2, 0.01*f), 'equations ' + e + ', fine ' + f + ', half steps ' + o);
}
{
  // the half steps fall far behind on the fast-spiking row, which the equations' step does not
  const f7 = NEURON_TYPES.find(t => t.key === 'FS07').f7, f = fine(f7, 116, 2000);
  ok('FS07: the half steps fire at least a fifth fewer than the fine clock', run(f7, 116, 2000, 0) < 0.8*f);
}
ok('a quadratic cell held below threshold fires under neither step', run(NEURON_TYPES.find(t => t.key === 'RS07').f7, 40, 2000, 2) === 0);
report('integ');

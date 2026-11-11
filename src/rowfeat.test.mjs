// The three row features of Zenke, Agnes and Gerstner 2015's cells (MODEL.md 1): the dynamic threshold that jumps at a spike and relaxes, the adaptation as a conductance toward the inhibitory reversal, and the Poisson output that fires at the input current read in hertz.
// One cell per case on the reference engine in this process, driven by a constant bias, its spike times against a mirror of the engine's own arithmetic (two half steps, the float32 stores) and, for the Poisson row, against the counter-based draw.
// Run: node src/rowfeat.test.mjs
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from './protocol.js';
import { initState, poissonDraw } from './rand.js';

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('./simworker.js');
const handle = globalThis.onmessage;
const send = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };

// a leaky integrate-and-fire cell in the 2007 form: a 12.5 nS leak on 250 pF, the quadratic's second root a volt away, the spike cut at -50 and reset to -70
const LIF = { C:250, k:12.5/1070, vr:-70, vt:1000, vpeak:-50, a:0.01, b:0, c:-70, d:0 };
function run(f7, bias, steps){
  const types = [{ f7 }], ntype = Uint8Array.from([0]);
  const st = initState(3, ntype, types);
  const v0 = Float32Array.from([-70]), u0 = new Float32Array(1);   // from rest, so the mirror starts where the engine does
  send({ cmd:'init', protocol:PROTOCOL, count:1, ntype, bias:Float32Array.from([bias]),
    preStart:Int32Array.from([0, 0]), post:new Int32Array(0), w:new Float32Array(0), delay:new Uint8Array(0), pmask:new Uint8Array(0),
    syn:0, plast:0, refrac:0, vmin:-90, eRevI:-80, types, seed:3, ...st, v0, u0 });
  const times = [];
  for(let t = 0; t < steps; t++){
    const r = send({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state');
    if(r.fired[0] > 0) times.push(t);
  }
  return times;
}
const f32 = Math.fround;
// the engine's step for one cell with no synapses: the threshold relaxes, two Euler half steps with the floor, the recovery, the spike test against the peak plus the threshold state
function mirror(f7, bias, steps, eRevI = -80){
  let v = f32(-70), u = 0, thr = 0;
  const dec = Math.exp(-1/(f7.thrTau > 0 ? f7.thrTau : 5)), times = [];
  for(let t = 0; t < steps; t++){
    if(f7.thrJump > 0) thr = f32(thr*dec);
    let vi = v, ui = u;
    const I = bias;
    const cur = () => f7.adaptCond ? ui*(vi - eRevI) : ui;
    vi += 0.5*(f7.k*(vi - f7.vr)*(vi - f7.vt) - cur() + I)/f7.C; if(vi < -90) vi = -90;
    vi += 0.5*(f7.k*(vi - f7.vr)*(vi - f7.vt) - cur() + I)/f7.C; if(vi < -90) vi = -90;
    ui += f7.a*(f7.b*(vi - f7.vr) - ui);
    if(vi >= f7.vpeak + (f7.thrJump > 0 ? thr : 0)){ vi = f7.c; ui += f7.d; times.push(t); if(f7.thrJump > 0) thr = f32(f7.thrJump); }
    v = f32(vi); u = f32(ui);
  }
  return times;
}
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

// 1. the dynamic threshold: a jump of 100 mV relaxing over 5 ms holds the cell off for a while after each spike
const plain = run(LIF, 400, 1000), plainM = mirror(LIF, 400, 1000);
ok('a plain cell at 400 pA spikes as the mirror says (' + plain.length + ' spikes)', same(plain, plainM) && plain.length > 20, plain.slice(0, 5) + ' vs ' + plainM.slice(0, 5));
const thrRow = { ...LIF, thrJump:100, thrTau:5 };
const thr = run(thrRow, 400, 1000), thrM = mirror(thrRow, 400, 1000);
ok('with the dynamic threshold the same cell spikes as the mirror says (' + thr.length + ' spikes)', same(thr, thrM), thr.slice(0, 5) + ' vs ' + thrM.slice(0, 5));
ok('and fewer times than without it', thr.length < plain.length, thr.length + ' against ' + plain.length);
ok('its first interval is longer than the plain cell\'s', thr[1] - thr[0] > plain[1] - plain[0], (thr[1] - thr[0]) + ' against ' + (plain[1] - plain[0]));

// 2. the conductance adaptation: a jump of 1.25 nS toward -80 mV, decaying over 100 ms, slows the cell where the same number as a current barely does
const adRow = { ...LIF, d:1.25, adaptCond:1 }, curRow = { ...LIF, d:1.25 };
const ad = run(adRow, 400, 2000), adM = mirror(adRow, 400, 2000), cu = run(curRow, 400, 2000);
ok('the conductance adaptation spikes as the mirror says (' + ad.length + ' spikes)', same(ad, adM), ad.slice(0, 5) + ' vs ' + adM.slice(0, 5));
ok('and slows the cell more than the same jump as a current (' + cu.length + ')', ad.length < cu.length, ad.length + ' against ' + cu.length);
ok('its intervals grow as the conductance accumulates', ad.length > 3 && ad[ad.length - 1] - ad[ad.length - 2] > ad[1] - ad[0], (ad[1] - ad[0]) + ' then ' + (ad[ad.length - 1] - ad[ad.length - 2]));

// 3. the Poisson row: a cell at 30 pA fires at 30 Hz on the counter-based draw, and one at 0 pA never
const poiRow = { ...LIF, poisson:1 };
const N = 20000;
const po = run(poiRow, 30, N);
const expect = []; for(let t = 0; t < N; t++) if(poissonDraw(0, t, 3) < 0.03) expect.push(t);
ok('a Poisson cell at 30 pA fires exactly where the draw falls under 0.03 (' + po.length + ' spikes in 20 s)', same(po, expect), po.length + ' against ' + expect.length);
ok('at about 30 Hz', po.length > 480 && po.length < 720, po.length + ' spikes in 20 s');
ok('and never at 0 pA', run(poiRow, 0, 2000).length === 0);
report('rowfeat');

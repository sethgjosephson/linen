// Short-term plasticity per rule (MODEL.md 2): one presynaptic cell, two rules onto two targets, one class depressing and one facilitating (Markram, Wang and Tsodyks 1998), on the reference engine in this process.
// The release is read as the jump in each target's membrane on every delivery of a 50 Hz train; the targets are held below threshold so the jump is the synapse alone.
// The battery's short-term plasticity group, cuda/ruletest.mjs and the Brian case "stp rules" hold the other engines to the reference.
// Run: node src/stprules.test.mjs
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from './protocol.js';
import { initState } from './rand.js';
import { engineTypes, NEURON_TYPES, engineConfig, stpClassTable } from './nodes.js';

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('./simworker.js');
const handle = globalThis.onmessage;
const send = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };

const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07');
const DEP = { stpOwn:2, stpU:0.5, stpTauD:800, stpTauF:1 }, FAC = { stpOwn:2, stpU:0.03, stpTauD:130, stpTauF:530 };
// cell 0 fires every 20 ms from a pulse protocol; it sends one synapse to cell 1 under rule row 2 and one to cell 2 under rule row 3
function init(ruleA, ruleB, base = { stp:0 }, w = 200){
  const ntype = Uint8Array.from([RS, RS, RS]), bias = Float32Array.from([0, -40, -40]);
  const rules = [{ name:'a', ...ruleA }, { name:'b', ...ruleB }];
  const net = { count:3, ntype, bias, preStart:Int32Array.from([0, 2, 2, 2]), post:Int32Array.from([1, 2]),
    w:Float32Array.from([w, w]), delay:Uint8Array.from([1, 1]), pmask:Uint8Array.from([2, 3]),
    syn:0, tauE:3, tauI:8, psc:0, plast:0, refrac:2, vmin:-90, stpNorm:0, stpOrder:0, stpScope:0, rules,
    ...base,
    protocols:[ { idx:Uint32Array.from([0]), amp:20000, mode:1, period:20, width:1, t0:5 } ] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:3, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask, inputs:[],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// each target's membrane, every step, and the spike times of cell 0
function trace(ruleA, ruleB, base, w){
  send(init(ruleA, ruleB, base, w));
  send({ cmd:'sendV', on:true });
  const v = [], spikes = [];
  for(let t = 0; t < 230; t++){
    const st = send({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state');
    v.push(Float32Array.from(st.v));
    if(st.fired[0]) spikes.push(t);
  }
  return { v, spikes };
}
// each target's jump on every delivery: the change across the delivery step of its membrane less the same network's with zero weights, so the membrane's own drift cancels
function run(ruleA, ruleB, base){
  const a = trace(ruleA, ruleB, base, 200), z = trace(ruleA, ruleB, base, 0);
  const d = (t, j) => a.v[t][j] - z.v[t][j];
  const jumps = [[], []];
  for(const t of a.spikes) if(t + 1 < a.v.length) for(const j of [1, 2]) jumps[j - 1].push(d(t + 1, j) - d(t, j));
  return { jumps, spikes:a.spikes };
}

const tbl = stpClassTable({ stp:0 }, [{ stpOwn:0 }, { stpOwn:2, ...DEP }, { stpOwn:2, ...DEP }, { stpOwn:1 }]);
ok('rules with the same settings share a class, and a rule that is off shares the off checkpoint\'s', tbl.stpClasses === 2 && tbl.stpCls[3] === tbl.stpCls[4] && tbl.stpCls[5] === 0 && tbl.stpCls[2] === 0,
  tbl.stpClasses + ' classes, map ' + Array.from(tbl.stpCls).join(','));

const r = run(DEP, FAC);
ok('cell 0 fires on every pulse', r.spikes.length >= 10, r.spikes.length + ' spikes');
const [dep, fac] = r.jumps, n = Math.min(dep.length, fac.length);
const ratio = a => a[n - 1]/a[0];
ok('the depressing rule\'s target receives less on each delivery of the train', dep[0] > 0 && ratio(dep) < 0.3, dep.slice(0, 4).map(x => x.toFixed(3)).join(' ') + ' ... last/first ' + ratio(dep).toFixed(3));
ok('the facilitating rule\'s target receives more, from the same spikes of the same axon', fac[0] > 0 && ratio(fac) > 3, fac.slice(0, 4).map(x => x.toFixed(3)).join(' ') + ' ... last/first ' + ratio(fac).toFixed(3));

// the first release is U of the weight under the published order, so the two first jumps are in the ratio of the two U
ok('the first releases are in the ratio of the two release probabilities', Math.abs(dep[0]/fac[0] - 0.5/0.03) < 0.5, (dep[0]/fac[0]).toFixed(3));

// two rules with the checkpoint's own settings are one class and deliver what the checkpoint alone delivers
const one = run({}, {}, { stp:1, stpU:0.5, stpTauD:800, stpTauF:1 });
const own = run(DEP, DEP, { stp:1, stpU:0.5, stpTauD:800, stpTauF:1 });
ok('a rule with the checkpoint\'s settings delivers exactly what the checkpoint\'s class does', one.jumps[0].every((x, k) => x === own.jumps[0][k]),
  one.jumps[0].slice(0, 3).join(' ') + ' against ' + own.jumps[0].slice(0, 3).join(' '));

// a rule that is off delivers the full weight on every spike while the checkpoint's class depresses
const off = run({ stpOwn:1 }, {}, { stp:1, stpU:0.5, stpTauD:800, stpTauF:1 });
ok('a rule that is off is not depressed by the checkpoint\'s class', Math.abs(ratio(off.jumps[0]) - 1) < 0.05 && ratio(off.jumps[1]) < 0.3,
  'off ' + ratio(off.jumps[0]).toFixed(3) + ', checkpoint ' + ratio(off.jumps[1]).toFixed(3));
report('stprules');

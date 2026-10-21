// Where the pair rule is timed (MODEL.md 4): at the soma the traces are read when a cell fires, so a pre spike at 0 with a 12 ms delay and a post spike at 5 reads as pre before post and potentiates; at the synapse the pre spike counts when it arrives, at 12, after the post spike, and the pairing depresses.
// Two cells driven to spike at chosen steps by pulse protocols, on the reference engine in this process; the reading is a setting on the checkpoint (stdpAt) and reaches the engine through engineConfig, so the test drives it through the init message's field.
// The CUDA parity gate (cuda/parity.mjs with STDPAT=1), cuda/stdpattest.mjs and the battery's cross-engine group hold the other two engines to the reference.
// Run: node src/stdpat.test.mjs
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

// cell 0 sends to cell 1 through one plastic synapse of delay d; each cell is kicked over threshold by a pulse at its own time (a 1 ms pulse of 20 nA on an RS07 row takes it over the peak within the step)
const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07');
function init({ d, tPre, tPost, stdpAt, w = 0.5, aP = 0.1, aM = 0.1 }){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias, preStart:Int32Array.from([0, 1, 1]), post:Int32Array.from([1]),
    w:Float32Array.from([w]), delay:Uint8Array.from([d]), pmask:Uint8Array.from([1]),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP, aM, tauS:20, tauM:20, wmax:30, wdep:0, stdpAt, iEta:0, iRho:5, refrac:2, vmin:-90,
    protocols:[ { idx:Uint32Array.from([0]), amp:20000, mode:1, period:100000, width:1, t0:tPre, duration:1 },
                { idx:Uint32Array.from([1]), amp:20000, mode:1, period:100000, width:1, t0:tPost, duration:1 } ] };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask, inputs:[],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
// the spike times of both cells and the weight after 60 ms
function run(c){
  send(init(c));
  const times = [[], []];
  for(let t = 0; t < 60; t++){
    const r = send({ cmd:'tick', steps:1, sendT:true });
    const st = r.find(x => x.cmd === 'state');
    if(st && st.fired) for(let i = 0; i < 2; i++) if(st.fired[i]) times[i].push(t);
  }
  const w = send({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w[0];
  return { times, w };
}

// the critique's example: pre at 0 with a 12 ms delay, post at 5
const soma = run({ d:12, tPre:2, tPost:7, stdpAt:0 }), syn = run({ d:12, tPre:2, tPost:7, stdpAt:1 });
ok('each cell fires once, when its pulse lands', soma.times[0].length === 1 && soma.times[1].length === 1 && soma.times[0][0] === 2 && soma.times[1][0] === 7,
  JSON.stringify(soma.times));
ok('at the soma the pairing reads pre before post and potentiates', soma.w > 0.5, 'w ' + soma.w);
ok('at the synapse the spike arrives after the post spike and the pairing depresses', syn.w < 0.5, 'w ' + syn.w);
// the exact amounts: at the soma dt is 5 ms (potentiation aP e^(-5/20) with pre at 2, post at 7); at the synapse the arrival at 14 is 7 ms after the post spike (depression aM e^(-7/20)), and the post spike at 7 sees no arrival yet
const f32 = x => Math.fround(x);
ok('the soma reading potentiates by aP e^(-5/tau)', Math.abs(soma.w - f32(0.5 + 0.1*Math.exp(-5/20))) < 1e-6, 'w ' + soma.w);
ok('the synapse reading depresses by aM e^(-7/tau) from the arrival', Math.abs(syn.w - f32(0.5 - 0.1*Math.exp(-7/20))) < 1e-6, 'w ' + syn.w);
// a causal pairing at the synapse: pre at 2 with delay 3 arrives at 5, post at 9: potentiation by aP e^(-4/tau), timed from the arrival, not aP e^(-7/tau) from the soma
const causal = run({ d:3, tPre:2, tPost:9, stdpAt:1 });
ok('a causal pairing at the synapse is timed from the arrival', Math.abs(causal.w - f32(0.5 + 0.1*Math.exp(-4/20))) < 1e-6, 'w ' + causal.w);
// an arrival in the step of the post spike is read as pre before post: potentiation at the peak, no depression
const coincident = run({ d:5, tPre:2, tPost:7, stdpAt:1 });
ok('an arrival in the step of the post spike potentiates at the peak of the window', Math.abs(coincident.w - f32(0.6)) < 1e-6, 'w ' + coincident.w);
report('stdpat');

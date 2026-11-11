// A brain file carries the consolidation reference, and a run restored from it keeps what had consolidated (engine contract 15).
// One synapse inside its well, idle long enough for its reference to climb past nine tenths of consW, so it is committed; the reference read back, written into each kind of brain file, read again and sent to a fresh engine with setRef.
// A pre-before-post pairing then leaves a committed synapse where it was, and potentiates one whose engine started its reference afresh.
// Run: node src/brainref.test.mjs
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from './protocol.js';
import { initState } from './rand.js';
import { engineTypes, NEURON_TYPES, engineConfig } from './nodes.js';
import { buildBrainFile, buildWeightsFile, buildPlasticFile, parseBrainFile, applyBrainWeights } from './brain.js';

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('./simworker.js');
const handle = globalThis.onmessage;
const send = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };

const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07');
// two cells and one synapse, plus a frozen synapse back from the second cell so the plastic-only file has something to leave out
const net = () => ({ count:2, ntype:Uint8Array.from([RS, RS]), bias:new Float32Array(2), src:Int32Array.from([0, 0]), lidx:Uint32Array.from([0, 1]),
  preStart:Int32Array.from([0, 1, 2]), post:Int32Array.from([1, 0]), w:Float32Array.from([1, 0.25]), delay:Uint8Array.from([1, 1]), pmask:Uint8Array.from([1, 0]), synCount:2,
  syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:0.1, aM:0.01, tauS:20, tauM:20, wmax:30, wdep:1, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
  het:0, cons:1, consW:1.2, consP:10, tauCons:0.5, consStep:1.2, commit:1, protocols:[] });
const initMsg = n => ({ cmd:'init', protocol:PROTOCOL, ...engineConfig(n), count:2, ntype:n.ntype, bias:n.bias,
  preStart:n.preStart, post:n.post, w:n.w, delay:n.delay, pmask:n.pmask,
  inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
  types:engineTypes(), seed:3, ...initState(3, n.ntype, NEURON_TYPES) });
const readback = () => send({ cmd:'getWeights' }).find(x => x.cmd === 'weights');
function pairing(){
  for(let t = 0; t < 40; t++){
    send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t === 2 ? 1 : 0, t === 7 ? 1 : 0]) });
    send({ cmd:'tick', steps:1 });
  }
  return readback().w[0];
}

// the trained state: twenty seconds idle, the reference past 1.08
const n0 = net();
send(initMsg(n0));
send({ cmd:'tick', steps:20000 });
const saved = readback();
ok('the weights reply carries the reference', saved.ref instanceof Float32Array && saved.ref.length === 2, String(saved.ref));
ok('and the reference has climbed past nine tenths of consW', saved.ref[0] >= 1.08, 'ref ' + saved.ref[0]);
const fresh = readback();
ok('a reply without plasticity in between repeats it', fresh.ref[0] === saved.ref[0]);

// the file round trip, each kind
const graph = { v:2, nodes:[] };
for(const [kind, build] of [['pair-keyed', buildBrainFile], ['weights-only', buildWeightsFile], ['plastic-only', buildPlasticFile]]){
  const b = parseBrainFile(build(graph, n0, saved.w, 20000, null, saved.ref));
  ok(kind + ' file says it holds the reference', b.meta.ref === true);
  const r = applyBrainWeights(net(), b);
  ok(kind + ' restores the reference of the plastic synapse', r.net.wRef0 && r.net.wRef0[0] === saved.ref[0], String(r.net.wRef0));
  const none = applyBrainWeights(net(), parseBrainFile(build(graph, n0, saved.w, 20000, null)));
  ok(kind + ' file without it restores none', none.net.wRef0 === null && !parseBrainFile(build(graph, n0, saved.w, 20000, null)).meta.ref);
}
{ // a pair-keyed file onto a wiring with a synapse it does not have: the reference is not restored
  const b = parseBrainFile(buildBrainFile(graph, n0, saved.w, 20000, null, saved.ref));
  const grown = { ...net(), count:3, ntype:Uint8Array.from([RS, RS, RS]), src:Int32Array.from([0, 0, 0]), lidx:Uint32Array.from([0, 1, 2]),
    preStart:Int32Array.from([0, 2, 3, 3]), post:Int32Array.from([1, 2, 0]), w:Float32Array.from([1, 0.25, 0.5]), synCount:3 };
  const r = applyBrainWeights(grown, b);
  ok('a pair-keyed file onto a grown wiring restores the weights and not the reference', r.matched === 2 && r.net.wRef0 === null);
}

// a fresh engine from the weights-only file, with and without the reference
const back = applyBrainWeights(net(), parseBrainFile(buildWeightsFile(graph, n0, saved.w, 20000, null, saved.ref))).net;
send(initMsg(back));
send({ cmd:'setRef', wRef:back.wRef0 });
ok('setRef puts the reference in place', readback().ref[0] === saved.ref[0]);
const kept = pairing();
ok('restored with its reference, the committed synapse does not potentiate', kept === saved.w[0], 'w ' + kept + ' saved ' + saved.w[0]);
send(initMsg(back));
const lost = pairing();
ok('restored without it, the reference starts again and the same pairing potentiates', lost > saved.w[0] + 0.05, 'w ' + lost);
report('brainref');

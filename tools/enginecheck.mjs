// Conformance check for an engine that speaks the protocol over a WebSocket (ENGINE.md, writing an engine).
//
//   node tools/enginecheck.mjs ws://localhost:8890 [--seconds S]
//
// It sends hello and checks the contract number and the shape of the reply, runs a small fixed scene on the candidate and on the reference engine (src/simworker.js, in this process) from the same init message, checks the shape of the state replies and of every optional query the candidate says it answers, and prints the rate, ISI CV and population synchrony of each population side by side.
// Each measure passes or fails on the band of the battery's cross-engine equivalence group (src/equivalence.js), which is what the WebGPU engine is held to.
// The scene is built through the scenario API with every node named and every setting on its node.
// Exit status 0 when every check passes.
import os from 'node:os';
import { NODE_DEFS, computeNode, setResolution, setMaxSyn, wireConnect, assembleConnect } from '../src/nodes.js';
import { api } from '../src/scenarios.js';
import { initMessage } from '../src/document.js';
import { PROTOCOL, checkHello, missingTerms, KNOWN_TERMS, KNOWN_OPTIONAL } from '../src/protocol.js';
import { RemoteEngine } from '../src/remoteworker.js';
import { isi, mean, std, populationSynchrony } from '../src/spikestats.js';
import { EQUIV, rateWithin, cvWithin, syncWithin } from '../src/equivalence.js';
import { inProcessWorker } from './harness.mjs';

const argv = process.argv.slice(2);
const url = argv.find(a => /^wss?:\/\//.test(a));
const si = argv.indexOf('--seconds');
const MS = Math.round(1000*(si >= 0 ? +argv[si + 1] : 3));
if(!url || !(MS > 0)){ console.error('usage: node tools/enginecheck.mjs ws://host:port [--seconds S]'); process.exit(2); }

let failed = 0;
const check = (name, pass, detail) => {
  if(!pass) failed++;
  console.log((pass ? 'ok    ' : 'FAIL  ') + name + (detail ? '  (' + detail + ')' : ''));
  return pass;
};
const wait = (eng, cmd, ms) => new Promise(res => {
  const t = setTimeout(() => { eng.onmessage = null; res({ cmd:'timeout' }); }, ms);
  eng.onmessage = e => {
    const d = e.data;
    if(d.cmd === cmd || d.cmd === 'error'){ clearTimeout(t); eng.onmessage = null; res(d); }
  };
});

// ---- the candidate says what it is
const cand = new RemoteEngine(url);
cand.postMessage({ cmd:'hello' });
const hello = await wait(cand, 'hello', 10000);
if(hello.cmd !== 'hello'){
  check('the engine answers hello', false, hello.cmd === 'error' ? hello.message : 'no reply within 10 s');
  cand.terminate(); process.exit(1);
}
console.log('engine: ' + hello.engine + ' at ' + url + ', contract ' + hello.protocol);
check('the contract number is this checkout\'s (' + PROTOCOL + ')', hello.protocol === PROTOCOL, 'the engine answers ' + hello.protocol);
check('the hello reply has its fields', checkHello({ ...hello, protocol:PROTOCOL }) === null, checkHello({ ...hello, protocol:PROTOCOL }));
const strange = [...(hello.terms || []).filter(t => !KNOWN_TERMS.includes(t)), ...(hello.optional || []).filter(q => !KNOWN_OPTIONAL.includes(q))];
check('every term and query it names is one the protocol defines', !strange.length, strange.join(', '));
console.log('terms: ' + (hello.terms || []).join(' ') + '\noptional: ' + (hello.optional || []).join(' '));
if(failed){ cand.terminate(); process.exit(1); }

// ---- the scene: a small balanced network, noise driven, kick synapses
inProcessWorker(wireConnect, assembleConnect);
setMaxSyn(Math.floor(os.totalmem()*0.5/13));
setResolution(1);
const nodes = [];
const ed = { nodes, nextId:1, groups:[],
  byId(id){ return nodes.find(n => n.id === id); },
  addNode(type, x, y){
    const def = NODE_DEFS[type], params = {};
    def.params.forEach(p => params[p.k] = structuredClone(p.def));
    const n = { id:this.nextId++, type, x, y, params, name:'', on:true, inputs:new Array(def.inputs).fill(null) };
    nodes.push(n); return n;
  } };
const { add, wire } = api(ed);
const vol = add('sphere', 40, 20, { center:[0, 0, 0], radius:200 }, 'volume');
const e = add('scatter', 40, 75, { count:800, type:0, seed:1, tag:'exc' }, 'excitatory cells'); wire(e, 0, vol);
const i = add('scatter', 190, 75, { count:200, type:3, seed:2, tag:'inh' }, 'inhibitory cells'); wire(i, 0, vol);
const all = add('gather', 115, 155, {}, 'both populations'); wire(all, 0, e); wire(all, 1, i);
const cn = add('connect', 115, 220, { radius:400, sigma:4000, prob:0.1, wExc:4, wInh:-30, wdist:1, velocity:100, seed:1 }, 'random wiring'); wire(cn, 0, all);
const dE = add('stimulus', 40, 280, { mode:3, current:14, tag:'exc', radius:100000, center:[0, 0, 0] }, 'noise into exc'); wire(dE, 0, cn);
const dI = add('stimulus', 190, 280, { mode:3, current:12, tag:'inh', radius:100000, center:[0, 0, 0] }, 'noise into inh'); wire(dI, 0, dE);
const out = add('checkpoint', 115, 460, { steps:1, syn:0 }, 'the run'); wire(out, 0, dI);
const unnamed = nodes.filter(n => !n.name);
if(unnamed.length) throw new Error('enginecheck: every node of the scene is named, and ' + unnamed.map(n => n.type).join(', ') + ' is not');
const net = await computeNode(out, id => ed.byId(id));
const init = { ...initMessage(net), inputs:[] };
const n = net.count;
const pops = new Map();
for(let c = 0; c < n; c++){
  const t = net.tags[net.src[c]];
  if(!pops.has(t)) pops.set(t, []);
  pops.get(t).push(c);
}
console.log('scene: ' + n + ' cells (' + [...pops].map(([t, c]) => c.length + ' ' + t).join(', ') + '), ' +
  net.synCount.toLocaleString() + ' synapses, kick synapses, noise drive, ' + MS/1000 + ' s at a 1 ms tick');
const miss = missingTerms(init, hello);
if(!check('the engine implements every term the scene uses', !miss.length, miss.map(m => m[1]).join(', '))){ cand.terminate(); process.exit(1); }

// ---- run one engine for MS ms, one tick a millisecond, keeping every spike time
async function run(post, next, label){
  const trains = Array.from({ length:n }, () => []);
  let shapeBad = '';
  for(let t = 0; t < MS; t++){
    post({ cmd:'tick', steps:1 });
    const st = await next();
    if(st.cmd !== 'state') return { error:label + ': ' + (st.message || 'no state reply at ' + t + ' ms') };
    if(!shapeBad){
      let sum = 0;
      if(!(st.fired instanceof Uint8Array) || st.fired.length !== n) shapeBad = 'fired is not a Uint8Array of ' + n;
      else for(let c = 0; c < n; c++) sum += st.fired[c];
      if(!shapeBad && st.steps !== 1) shapeBad = 'steps ' + st.steps + ' for a tick of 1';
      if(!shapeBad && st.spikes !== sum) shapeBad = 'spikes ' + st.spikes + ' against ' + sum + ' in fired';
    }
    const f = st.fired;
    for(let c = 0; c < n; c++) for(let k = 0; k < f[c]; k++) trains[c].push(t);
  }
  return { trains, shapeBad };
}
const measure = (trains, cells) => {
  const tr = cells.map(c => trains[c]);
  const spikes = tr.reduce((a, t) => a + t.length, 0);
  const isis = tr.flatMap(t => [...isi(t)]);
  return { rate:spikes/cells.length/(MS/1000), cv:isis.length > 30 ? std(isis)/mean(isis) : NaN,
    sync:populationSynchrony(tr, EQUIV.syncBinMs, 0, MS) };
};

// the candidate, over its socket
const replies = [];
let wake = null;
cand.onmessage = e => { replies.push(e.data); if(wake){ const w = wake; wake = null; w(); } };
const nextRemote = async () => {
  const t0 = Date.now();
  while(!replies.length){
    if(Date.now() - t0 > 10000) return { cmd:'timeout', message:'no reply within 10 s' };
    await new Promise(r => { wake = r; setTimeout(r, 200); });
  }
  return replies.shift();
};
cand.postMessage(init);
const t0 = Date.now();
const C = await run(d => cand.postMessage(d), async () => {
  for(;;){ const d = await nextRemote(); if(d.cmd === 'state' || d.cmd === 'error' || d.cmd === 'timeout') return d; }
}, 'candidate');
const candWall = (Date.now() - t0)/1000;
if(C.error){ check('the engine runs the scene', false, C.error); cand.terminate(); process.exit(1); }
check('state replies carry fired (one byte a cell), spikes and steps', !C.shapeBad, C.shapeBad);

// the optional queries the candidate names, each asked once
const ask = async (msg, cmd) => { replies.length = 0; cand.postMessage(msg); for(;;){ const d = await nextRemote(); if(d.cmd === cmd || d.cmd === 'error' || d.cmd === 'timeout') return d; } };
const has = q => (hello.optional || []).includes(q);
if(has('sendV')){
  cand.postMessage({ cmd:'sendV', on:true });
  const st = await ask({ cmd:'tick', steps:2 }, 'state');
  check('sendV: a state reply carries v, a Float32Array of every cell', st.v instanceof Float32Array && st.v.length === n);
  cand.postMessage({ cmd:'sendV', on:false });
}
if(has('watch')){
  cand.postMessage({ cmd:'watch', idx:0 });
  const st = await ask({ cmd:'tick', steps:3 }, 'state');
  check('watch: a state reply carries vtrace, one value a step', st.vtrace instanceof Float32Array && st.vtrace.length === 3);
  cand.postMessage({ cmd:'watch', idx:-1 });
}
if(has('sendT')){
  cand.postMessage({ cmd:'sendT', on:true });
  const st = await ask({ cmd:'tick', steps:5 }, 'state');
  check('sendT: a state reply carries lat, a Uint8Array of every cell', st.lat instanceof Uint8Array && st.lat.length === n);
  cand.postMessage({ cmd:'sendT', on:false });
}
if(has('query')){
  const q = await ask({ cmd:'query', idx:0, cap:2000 }, 'queryResult');
  const outN = net.preStart[1] - net.preStart[0];
  check('query: the reply lists cell 0\'s outgoing synapses', q.cmd === 'queryResult' && q.idx === 0 && q.outTotal === outN, q.cmd === 'queryResult' ? q.outTotal + ' against ' + outN : q.message);
}
if(has('getWeights')){
  const w = await ask({ cmd:'getWeights' }, 'weights');
  check('getWeights: the reply carries every weight', w.cmd === 'weights' && w.w instanceof Float32Array && w.w.length === net.synCount);
}
cand.terminate();

// the reference, in this process, from the same init
const refOut = [];
globalThis.postMessage = d => refOut.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
globalThis.onmessage({ data:init });
if(refOut.some(d => d.cmd === 'error')) throw new Error('the reference engine refused the scene: ' + refOut.find(d => d.cmd === 'error').message);
const R = await run(d => globalThis.onmessage({ data:d }), async () => refOut.pop(), 'reference');

// ---- side by side
const f = (x, d) => Number.isFinite(x) ? x.toFixed(d) : 'n/a';
console.log('\n' + 'population'.padEnd(12) + 'measure'.padEnd(12) + 'candidate'.padStart(11) + 'reference'.padStart(11) + '   band');
let differ = 0;
for(const [tag, cells] of pops){
  const a = measure(C.trains, cells), b = measure(R.trains, cells);
  const rows = [
    ['rate Hz', a.rate, b.rate, 2, rateWithin(a.rate, b.rate), 'within ' + EQUIV.rate*100 + '%'],
    ['ISI CV', a.cv, b.cv, 3, cvWithin(a.cv, b.cv), 'within ' + EQUIV.cv],
    ['synchrony', a.sync, b.sync, 2, syncWithin(a.sync, b.sync), 'within a factor of ' + EQUIV.sync],
  ];
  for(const [name, x, y, d, pass, band] of rows){
    if(!pass){ failed++; differ++; }
    console.log(tag.padEnd(12) + name.padEnd(12) + f(x, d).padStart(11) + f(y, d).padStart(11) + '   ' + band + (pass ? '  ok' : '  DIFFERENT'));
  }
}
console.log('\ncandidate ' + candWall.toFixed(1) + ' s of wall clock for ' + MS/1000 + ' s simulated');
if(differ && hello.engine === 'lif-template')
  console.log('expected: tools/engine_template.py runs a leaky integrate-and-fire membrane, not the Izhikevich membrane the reference runs, so its rates are not the reference\'s. ' +
    'The protocol checks above are what it is there to pass.');
else if(differ)
  console.log('the candidate differs from the reference beyond the bands two implementations of the same model meet');
console.log(failed ? failed + ' check' + (failed > 1 ? 's' : '') + ' failed' : 'every check passed');
process.exit(failed ? 1 : 0);

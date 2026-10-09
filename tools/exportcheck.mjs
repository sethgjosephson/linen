// Does the Brian 2 export run the same network the engine runs?
//
// Builds a Tab-menu scene at a small resolution through the battery's fake editor, composes the init message the page composes, exports it as a Brian 2 case (src/briancase.js, the same call the checkpoint's "save for Brian 2" makes), runs the case on Brian 2 with tools/brian_ref.py, runs the same init on the reference engine in process, and requires every cell's spike train to be identical.
//
//   node tools/exportcheck.mjs ["balanced random net"] [resolution] [ms]
//
// Needs Brian 2 in .venv-analysis; says DID NOT RUN without it.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { NODE_DEFS, computeNode, setResolution, wireConnect, assembleConnect } from '../src/nodes.js';
import { caseFromInit } from '../src/briancase.js';
import { SCENARIOS } from '../src/scenarios.js';
import { ok, report } from './harness.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url)), ROOT = path.join(HERE, '..');
const PY = [path.join(ROOT, '.venv-analysis', 'Scripts', 'python.exe'), path.join(ROOT, '.venv-analysis', 'bin', 'python')].find(p => fs.existsSync(p));
if(!PY){ console.log('exportcheck: Brian 2 not found in .venv-analysis, so the check DID NOT RUN.'); process.exit(0); }

const [name = 'balanced random net', res = '0.05', msArg = '400'] = process.argv.slice(2);
// node has no web Worker: the wiring runs in this process
import { inProcessWorker } from './harness.mjs';
import { initMessage } from '../src/document.js';
inProcessWorker(wireConnect, assembleConnect);
function fakeEditor(){
  const nodes = [];
  return { nodes, nextId:1, groups:[], addNode(type, x, y){
    const def = NODE_DEFS[type], params = {};
    def.params.forEach(p => params[p.k] = structuredClone(p.def));
    const n = { id:this.nextId++, type, x, y, params, inputs:new Array(def.inputs).fill(null) };
    nodes.push(n); return n; } };
}
const sc = SCENARIOS.find(s => s.name.toLowerCase().includes(name.toLowerCase()));
if(!sc){ console.error('no scene ' + name); process.exit(1); }
const ed = fakeEditor(); sc.build(ed);
setResolution(+res);
const out = ed.nodes.find(n => n.type === 'checkpoint');
const net = await computeNode(out, id => ed.nodes.find(n => n.id === id));
const n = net.count, ms = +msArg;
console.log(sc.name + ' at resolution ' + res + ': ' + n.toLocaleString() + ' cells, ' + net.synCount.toLocaleString() + ' synapses, ' + ms + ' ms');

// the page's own init message, without input nodes
const init = { ...initMessage(net), inputs:[] };

const file = caseFromInit(init, { ms });
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'exportcheck-'));
const inFile = path.join(tmp, 'case.json'), outFile = path.join(tmp, 'out.json');
const json = JSON.stringify(file);
fs.writeFileSync(inFile, json);
console.log('case: ' + (json.length/1e6).toFixed(2) + ' MB at ' + inFile);
const t0 = performance.now();
try {
  execFileSync(PY, [path.join(HERE, 'brian_ref.py'), inFile, outFile], { stdio:['ignore', 'ignore', 'pipe'], timeout:1800000, encoding:'utf8' });
} catch(e){
  console.error('Brian 2 failed:\n' + String(e.stderr || e.message).split(/\r?\n/).filter(l => /Error|error:/.test(l)).slice(-4).join('\n'));
  process.exit(1);
}
const br = JSON.parse(fs.readFileSync(outFile, 'utf8'));
console.log('Brian 2 ' + br.versions.brian2 + ' ran the case in ' + Math.round((performance.now() - t0)/1000) + ' s');

// the engine on the same init, one-millisecond ticks so every spike has its own step
const replies = []; globalThis.postMessage = d => replies.push(d); globalThis.onmessage = null;
await import('../src/simworker.js');
const send = d => globalThis.onmessage({ data:d });
send({ ...init, w:init.w.slice() });
const err = replies.find(r => r.cmd === 'error'); if(err){ console.error('engine: ' + err.message); process.exit(1); }
const times = Array.from({ length:n }, () => []);
for(let t = 0; t < ms; t++){
  replies.length = 0; send({ cmd:'tick', steps:1 });
  const s = replies.find(r => r.cmd === 'state');
  for(let i = 0; i < n; i++) if(s.fired[i]) times[i].push(t);
}
let same = 0, differ = 0, spikes = 0, first = null;
for(let i = 0; i < n; i++){
  const a = times[i], b = br.times[i] || [];
  spikes += a.length;
  if(a.length === b.length && a.every((t, k) => t === b[k])) same++;
  else { differ++; if(!first) first = { cell:i, engine:a.slice(0, 6), brian:b.slice(0, 6) }; }
}
ok('the engine fired at all', spikes > 0, spikes + ' spikes');
ok('every cell’s spike train is identical on Brian 2', differ === 0, differ + ' of ' + n + ' cells differ' + (first ? '; first: cell ' + first.cell + ' engine ' + JSON.stringify(first.engine) + ' brian ' + JSON.stringify(first.brian) : ''));
console.log(same + ' of ' + n + ' cells identical, ' + spikes + ' spikes in ' + ms + ' ms');
report('export check');

// Old scene files still open and still compute the same network.
// The folder examples/compat holds the Tab-menu scenes as every earlier build saved them, one set per scene format, each written by that build's own code.
// Each is loaded the way the page loads a file (migrated, its node modules checked), computed at a small resolution, and held to the cell count, the synapse count and a hash of the wiring recorded in examples/compat/expect.json.
// A format with no fixture fails, so the corpus cannot lag a bump; a file from a newer build must be refused, not guessed at.
//
//   node src/compat.test.mjs            one scene per format (the quick suite)
//   node src/compat.test.mjs --all      every scene (the full suite)
//   node src/compat.test.mjs --record   write expect.json from this build, after a deliberate change to the wiring
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ok, report } from '../tools/harness.mjs';
import { inProcessWorker } from '../tools/harness.mjs';
import { computeNode, setResolution, setMaxSyn, wireConnect, assembleConnect } from './nodes.js';
import { migrateScene, SCENE_FORMAT } from './migrate.js';
import { PROJECT_FORMAT, readProject } from './project.js';
import { parseBrainFile, applyBrainWeights } from './brain.js';
import { LAYOUT_FORMAT, layoutMismatch } from './layoutfile.js';
import { loadScene, editorFromScene } from '../tools/scene.mjs';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIR = path.join(ROOT, 'examples', 'compat');
// fixtures of scenes a checkout keeps to itself sit in local/, with their own expect.json; a checkout without the folder has none
const LOCAL = path.join(DIR, 'local');
const SETS = [DIR, ...(fs.existsSync(LOCAL) ? [LOCAL] : [])];
const ALL = process.argv.includes('--all'), RECORD = process.argv.includes('--record');
const RES = 0.05;                               // the same small resolution every time; the hash is of this network

// FNV-1a over the wiring: every target, every delay, every weight's bits
function wiringHash(net){
  let h = 0x811c9dc5 >>> 0;
  const mix = b => { h ^= b; h = Math.imul(h, 0x01000193) >>> 0; };
  const u8 = (view) => { const b = new Uint8Array(view.buffer, view.byteOffset, view.byteLength); for(let i = 0; i < b.length; i++) mix(b[i]); };
  u8(net.post); u8(net.delay); u8(net.w);
  return h.toString(16).padStart(8, '0');
}

inProcessWorker(wireConnect, assembleConnect);
setMaxSyn(4e8);
setResolution(RES);

// a file from a newer build is refused, not guessed at
{
  let msg = '';
  try { migrateScene({ format:SCENE_FORMAT + 1, nodes:[] }); } catch(e){ msg = e.message; }
  ok('a scene of a newer format is refused', /newer/.test(msg), msg);
  msg = '';
  const store = { read: async () => new TextEncoder().encode(JSON.stringify({ format:PROJECT_FORMAT + 1, scenes:[] })) };
  try { await readProject(store); } catch(e){ msg = e.message; }
  ok('a project of a newer format is refused', /understands/.test(msg), msg);
}

// a layout file of another format is refused and the computed layout stands; there is no migration to get wrong
ok('a layout of another format is refused', /not a layout file of this format/.test(layoutMismatch([], { format:LAYOUT_FORMAT - 1, nodes:[], build:[] }) || ''));

// Old brain files: the header reads, the graph inside migrates and computes, and the weights refuse a wiring that is not the one they were saved from, with a message that says so.
// The weights themselves need the saved graph at full resolution, which is a training run's size; what is held here is the file format and the refusal.
for(const [dir, f] of SETS.flatMap(d => fs.readdirSync(d).filter(f => /\.npb$/.test(f)).sort().map(f => [d, f]))){
  const label = f.replace(/\.npb$/, '');
  try {
    const b = fs.readFileSync(path.join(dir, f));
    const brain = parseBrainFile(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    ok(label + ' parses', ['pairs', 'weights', 'plastic'].includes(brain.kind) && brain.w.length === (brain.kind === 'plastic' ? brain.k : brain.m), brain.kind + ' ' + brain.w.length);
    const { ed } = editorFromScene(structuredClone(brain.graph));
    const cp = ed.nodes.find(n => n.type === 'checkpoint');
    ok(label + "'s graph migrates to one with a checkpoint", !!cp, 'format ' + (brain.graph.format || 1) + ', ' + ed.nodes.length + ' nodes');
    const net = await computeNode(cp, id => ed.nodes.find(n => n.id === id));
    let msg = '';
    try { applyBrainWeights(net, brain); } catch(e){ msg = e.message; }
    ok(label + ' refuses a wiring it was not saved from', /saved from|need the graph|synapses but this wiring/.test(msg), msg.slice(0, 120));
  } catch(e){ ok(label + ' is handled', false, e.message.slice(0, 200)); }
}

const fixtures = d => fs.readdirSync(d).filter(f => /^f\d\d-.*\.json$/.test(f)).sort();
const files = fixtures(DIR);
const byFormat = new Map();
for(const f of files){ const k = +f.slice(1, 3); if(!byFormat.has(k)) byFormat.set(k, []); byFormat.get(k).push(f); }
for(let k = 1; k < SCENE_FORMAT; k++) ok('format ' + k + ' has fixtures', byFormat.has(k), 'none in examples/compat');
// the quick suite takes one scene per format from the shipped set; --all takes every file of every set
const chosen = ALL ? SETS.flatMap(d => fixtures(d).map(f => [d, f])) : [...byFormat.values()].map(list => [DIR, list.find(f => /guided-tour/.test(f)) || list[0]]);

const expect = new Map(), got = new Map();
for(const d of SETS){
  const file = path.join(d, 'expect.json');
  let e = {};
  try { e = JSON.parse(fs.readFileSync(file, 'utf8')); } catch(err){ if(!RECORD) ok(path.relative(ROOT, file) + ' is present', false, file); }
  expect.set(d, e); got.set(d, {});
}
for(const [dir, f] of chosen){
  const label = f.replace(/\.json$/, '');
  let r;
  try {
    const { ed } = await loadScene(path.relative(ROOT, path.join(dir, f)), { root:ROOT });
    const cp = ed.nodes.find(n => n.type === 'checkpoint');
    if(!cp) throw new Error('no checkpoint node after migration');
    const net = await computeNode(cp, id => ed.nodes.find(n => n.id === id));
    r = { cells:net.count, synapses:net.synCount, hash:wiringHash(net) };
  } catch(e){
    // a scene that reads project files (points, connections, meshes) has nothing to read here; that is recorded, so a new failure cannot hide behind it
    if(/file reader|no file|project file|ENOENT|not found/i.test(e.message)) r = { skip:e.message.slice(0, 80) };
    else r = { error:e.message.slice(0, 200) };
  }
  got.get(dir)[label] = r;
  if(RECORD) continue;
  const want = expect.get(dir)[label];
  if(r.error){ ok(label + ' opens and computes', false, r.error); continue; }
  if(!want){ ok(label + ' is in expect.json', false, 'not recorded; run with --record after checking the change'); continue; }
  if(want.skip || r.skip){ ok(label + ' is skipped as recorded', !!want.skip && !!r.skip, (want.skip || 'computed') + ' vs ' + (r.skip || 'computed')); continue; }
  ok(label + ' computes the recorded network', r.cells === want.cells && r.synapses === want.synapses && r.hash === want.hash,
    `cells ${r.cells} vs ${want.cells}, synapses ${r.synapses} vs ${want.synapses}, hash ${r.hash} vs ${want.hash}`);
}
if(RECORD){
  const errors = [];
  for(const d of SETS){
    const g = got.get(d), file = path.join(d, 'expect.json');
    fs.writeFileSync(file, JSON.stringify({ ...expect.get(d), ...g }, null, 1) + '\n');
    const skipped = Object.values(g).filter(r => r.skip).length;
    errors.push(...Object.entries(g).filter(([, r]) => r.error));
    console.log('recorded ' + Object.keys(g).length + ' scenes (' + skipped + ' skipped) into ' + file);
  }
  for(const [k, r] of errors) console.log('  error ' + k + ': ' + r.error);
  ok('every scene opened and computed, or is skipped for its files', errors.length === 0, errors.length + ' errors');
}
report('compat');

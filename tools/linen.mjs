// Run a scene without the browser: a scene file or the name of a scene from the Tab menu is computed, run on the reference engine, on the CUDA engine or on an engine at a WebSocket address, and what happened is written into a folder.
//
//   node tools/linen.mjs run "mouse cortical column" --seconds 10 --out runs/col
//   node tools/linen.mjs run scenes/my-scene.json --seconds 60 --engine cuda
//   node tools/linen.mjs run "balanced random net" --over "connect:*:wInh=2.5"
//   node tools/linen.mjs run "guided tour" --engine remote --host ws://localhost:8890
//
//   --seconds N     simulated seconds (default 10)
//   --seed K        the seed on every connect node, which is where the engines
//                   take their noise and their initial state from
//   --res R         resolution, 0 to 1 (default 1, the authored size)
//   --over SPEC     settings on nodes: type:tag-or-name:key=value, several
//                   separated by semicolons, * for every node of that type.
//                   Repeatable. Settings are read from the nodes, so a run
//                   here is the run the app would make of the same scene.
//                   An input node fed by a test signal is driven on
//                   simulated time, the way a training run drives it; an
//                   input fed by a curriculum, a microphone, a webcam or
//                   footage gets no frames, which the run says and run.json
//                   records (inputsDriven false, with those maps named).
//   --engine NAME   cpu (the reference engine, in this process), cuda, or
//                   remote (an engine at --host that speaks the protocol over
//                   a WebSocket: host/host.mjs, or a custom engine)
//   --host URL      the remote engine's address (default ws://localhost:8801)
//   --bin MS        the rate bin, default 1000
//   --tick MS       simulated milliseconds per tick, default 1. Spike times
//                   are recorded at this resolution.
//   --cells N       how many cells' spike trains to write, sampled evenly
//                   across the network (default 2000; all for every cell)
//   --out DIR       where to write (default runs/<scene>-<stamp>)
//   --quiet         no progress line
//   --syn M         the synapse budget in millions (default: half the machine's memory at 13 B a synapse)
//   --nodes DIR     a folder of node modules to load before the scene (repeatable).
//                   A scene file inside a project's scenes folder loads that
//                   project's nodes folder without being asked.
//
// The folder it writes:
//
//   run.json     the scene, the settings that were overridden, the size, the
//                engine, the commit, the versions and the wall time
//   rates.csv    one row per bin, one column per population, in Hz
//   release.csv  one row per bin, one column per graded population (cells
//                that never spike), the mean release in percent of its
//                maximum, read from the potentials the way a probe reads it;
//                written only when the scene has a graded population
//   spikes.txt   spike times in milliseconds, one line per cell, the layout
//                Neo's AsciiSpikeTrainIO reads
//   cells.csv    the cell behind each line of spikes.txt: population, source
//                node, index within it, cell type and position
//   settings.json  every node in the computed graph with the settings it ran
//
// tools/linen_read.py loads that folder into pandas and Neo.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { NODE_DEFS, NEURON_TYPES, computeNode, setResolution, setMaxSyn, wireConnect, assembleConnect, setFileReader, gradedArrays, engineTypes } from '../src/nodes.js';
import os from 'node:os';
import { initMessage, engineInputs } from '../src/document.js';
import { checkHello, missingTerms, ignoredCustom, customNote, answers } from '../src/protocol.js';
import { IORuntime } from '../src/io.js';
import { migrateScene } from '../src/migrate.js';
import { loadPlugin, importer, checkScenePlugins } from '../src/plugins.js';
import { ALL_SCENARIOS } from '../src/experiments.js';
import { csv, multiSeriesCsv, spikeTrainsAscii } from '../src/figure.js';
import { inProcessWorker } from './harness.mjs';
import { splitTerms, applyOverrides } from './overrides.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url)), ROOT = path.join(HERE, '..');

// ---- the command line
const argv = process.argv.slice(2);
const cmd = argv.shift();
if(cmd !== 'run'){
  console.error('usage: node tools/linen.mjs run <scene.json or scene name> [options]\n'
    + '       see the comment at the top of this file for the options');
  process.exit(1);
}
const opt = { seconds:10, res:1, engine:'cpu', host:'ws://localhost:8801', bin:1000, tick:1, cells:2000, out:'', seed:null, quiet:false, over:[], syn:0, nodes:[] };
const target = argv.shift();
if(!target){ console.error('linen: name a scene file or a scene'); process.exit(1); }
for(let i = 0; i < argv.length; i++){
  const a = argv[i], next = () => { const v = argv[++i]; if(v === undefined){ console.error('linen: ' + a + ' needs a value'); process.exit(1); } return v; };
  if(a === '--seconds') opt.seconds = +next();
  else if(a === '--seed') opt.seed = +next();
  else if(a === '--res') opt.res = +next();
  else if(a === '--over') opt.over.push(next());
  else if(a === '--engine') opt.engine = next();
  else if(a === '--host') opt.host = next();
  else if(a === '--bin') opt.bin = +next();
  else if(a === '--tick') opt.tick = Math.max(1, Math.round(+next()));
  else if(a === '--cells'){ const v = next(); opt.cells = v === 'all' ? Infinity : +v; }
  else if(a === '--out') opt.out = next();
  else if(a === '--quiet') opt.quiet = true;
  else if(a === '--syn') opt.syn = +next()*1e6;
  else if(a === '--nodes') opt.nodes.push(next());
  else { console.error('linen: unknown option ' + a); process.exit(1); }
}
if(!(opt.seconds > 0)){ console.error('linen: --seconds must be above zero'); process.exit(1); }
if(!(opt.res > 0 && opt.res <= 1)){ console.error('linen: --res is a fraction between 0 and 1'); process.exit(1); }
if(!['cpu', 'cuda', 'remote'].includes(opt.engine)){ console.error('linen: --engine is cpu, cuda or remote'); process.exit(1); }
if(opt.engine === 'remote' && !/^wss?:\/\/\S+$/.test(opt.host)){ console.error('linen: --host is a ws:// address'); process.exit(1); }
const ms = Math.round(opt.seconds*1000);
// a bin closes on a tick boundary and its rate divides by the bin width, so a tick that does not divide the bin made every bin a multiple of the width long and every rate that multiple too high
if(!(opt.bin > 0) || opt.bin % opt.tick || ms % opt.tick){
  console.error('linen: --tick must divide --bin and the run (' + opt.tick + ' ms against a ' +
    opt.bin + ' ms bin and ' + ms + ' ms)'); process.exit(1); }
const say = s => { if(!opt.quiet) console.log(s); };

// ---- the graph: a scene file, or a scene by name
inProcessWorker(wireConnect, assembleConnect);       // node has no Worker; the wiring runs here
// the synapse budget: --syn in millions, else half the machine's memory at 13 bytes a synapse while wiring (the page derives it from the device)
setMaxSyn(opt.syn > 0 ? opt.syn : Math.floor(os.totalmem()*0.5/13));
function blankNode(type, id, x, y){
  const def = NODE_DEFS[type], params = {};
  def.params.forEach(p => params[p.k] = structuredClone(p.def));
  return { id, type, x, y, params, name:'', on:true, inputs:new Array(def.inputs).fill(null) };
}
function fakeEditor(){
  const nodes = [];
  return { nodes, nextId:1, groups:[],
    byId(id){ return nodes.find(n => n.id === id); },
    addNode(type, x, y){ const n = blankNode(type, this.nextId++, x, y); nodes.push(n); return n; } };
}
// Node modules (src/plugins.js): every .js in each folder, loaded the way the page loads a project's nodes folder.
async function loadNodeFolder(dir){
  if(!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()){ console.error('linen: no folder of node modules at ' + dir); process.exit(1); }
  for(const name of fs.readdirSync(dir).sort()){
    if(!/\.m?js$/.test(name)) continue;
    const r = await loadPlugin(name, fs.readFileSync(path.join(dir, name), 'utf8'), importer());
    if(r.error){ console.error('linen: ' + r.error); process.exit(1); }
    say('node module ' + name + ': ' + r.keys.join(', '));
  }
}
// This is what editor.load does, without a canvas under it.
function editorFromScene(data){
  data = migrateScene(data);
  try { checkScenePlugins(data); }
  catch(e){ console.error('linen: ' + e.message + '\n  (node modules load from --nodes DIR, and from the nodes folder beside the scenes folder the scene file is in)'); process.exit(1); }
  const ed = fakeEditor();
  ed.nextId = data.nextId || 1;
  for(const sn of data.nodes || []){
    if(!NODE_DEFS[sn.type]) continue;
    const n = blankNode(sn.type, sn.id, sn.x, sn.y);
    const def = NODE_DEFS[sn.type];
    def.params.forEach(pd => { if(sn.params && sn.params[pd.k] !== undefined) n.params[pd.k] = structuredClone(sn.params[pd.k]); });
    n.name = sn.name || '';
    n.on = sn.on === false ? false : true;
    const nIn = Math.max(def.inputs, sn.inputs ? sn.inputs.length : 0);
    n.inputs = new Array(nIn).fill(null).map((_, k) => sn.inputs && sn.inputs[k] ? { id:sn.inputs[k].id } : null);
    ed.nodes.push(n);
  }
  for(const n of ed.nodes) n.inputs = n.inputs.map(c => c && ed.byId(c.id) ? c : null);
  return { ed, name:data.scenarioName || '' };
}
let ed, sceneName, source;
const asFile = path.resolve(ROOT, target);
if(/\.json$/i.test(target) && fs.existsSync(asFile)){
  const own = path.join(path.dirname(asFile), '..', 'nodes');
  if(path.basename(path.dirname(asFile)) === 'scenes' && fs.existsSync(own) && !opt.nodes.some(d => path.resolve(ROOT, d) === path.resolve(own))) opt.nodes.push(own);
  // A scene in a project's scenes folder reads the project's files (points files, connections files, meshes) the way the page does.
  if(path.basename(path.dirname(asFile)) === 'scenes'){
    const projDir = path.join(path.dirname(asFile), '..');
    setFileReader(async p => fs.readFileSync(path.join(projDir, p)));
  }
  for(const d of opt.nodes) await loadNodeFolder(path.resolve(ROOT, d));
  const loaded = editorFromScene(JSON.parse(fs.readFileSync(asFile, 'utf8')));
  ed = loaded.ed;
  sceneName = loaded.name || path.basename(target, '.json');
  source = path.relative(ROOT, asFile).replace(/\\/g, '/');
} else {
  const sc = ALL_SCENARIOS.find(s => s.name.toLowerCase() === target.toLowerCase())
    || ALL_SCENARIOS.find(s => s.name.toLowerCase().includes(target.toLowerCase()));
  if(!sc){
    console.error('linen: no scene file and no scene called ' + target + '. The scenes are:\n  '
      + ALL_SCENARIOS.map(s => s.name).join('\n  '));
    process.exit(1);
  }
  ed = fakeEditor(); sc.build(ed);
  sceneName = sc.name; source = 'scene ' + sc.name;
}

// ---- settings on nodes, which is where every setting lives
const overrides = [...opt.over.flatMap(s => s.split(';')).map(s => s.trim()).filter(Boolean)];
if(opt.seed !== null) overrides.push('connect:*:seed=' + opt.seed);
let applied;
try { applied = applyOverrides(ed.nodes, overrides); }
catch(e){ console.error('linen: ' + e.message); process.exit(1); }
for(const a of applied) say('setting ' + a.term + ' on ' + a.nodes + ' node' + (a.nodes === 1 ? '' : 's'));

// ---- compute
setResolution(opt.res);
const outs = ed.nodes.filter(n => n.type === 'checkpoint');
if(!outs.length){ console.error('linen: the scene has no checkpoint node, so there is nothing to run'); process.exit(1); }
const computeStart = Date.now();
const net = await computeNode(outs[0], id => ed.nodes.find(n => n.id === id));
const n = net.count;
say(sceneName + ': ' + n.toLocaleString() + ' cells, ' + net.synCount.toLocaleString() + ' synapses'
  + (opt.res < 1 ? ' at resolution ' + opt.res : '') + ', computed in ' + ((Date.now() - computeStart)/1000).toFixed(1) + ' s');
if(net.delayClamped) say('note: ' + net.delayClamped.toLocaleString() + ' synapses were longer than the '
  + net.delayMax + ' ms delay ceiling and are clamped to it');

// ---- the engine
// An input map fed by the test signal (source 0, the bar sweep) is driven on simulated time through the stimulus runtime a training run uses, so its frames are the ones the trainer would post.
// A curriculum needs the glyph atlas only a browser can draw, and a microphone, a webcam or footage needs a device or a folder, so those maps get no frame: said on the console and in run.json rather than left to be found.
// The engine receives only the driven maps, so the map index of every frame is its index in that list.
const allMaps = net.inputMaps || [];
const drivenMaps = allMaps.filter(m => m.source === 0);
const undriven = allMaps.filter(m => m.source !== 0).map((m, i) => m.tag || ('map ' + i));
if(undriven.length)
  console.warn('linen: this scene has ' + undriven.length + ' input map' + (undriven.length > 1 ? 's' : '') +
    ' (' + undriven.join(', ') + ') fed by a curriculum, a device or footage, which this tool does not drive: those input nodes get no drive');
if(drivenMaps.length) say('driving ' + drivenMaps.length + ' input map' + (drivenMaps.length > 1 ? 's' : '') +
  ' from the test signal (' + drivenMaps.map(m => m.tag).join(', ') + ')');
const init = { ...initMessage(net), inputs:engineInputs({ inputMaps:drivenMaps }) };
let post, close;
const onState = [];
if(opt.engine === 'cpu'){
  const replies = [];
  globalThis.postMessage = d => replies.push(d);
  globalThis.onmessage = null;
  await import('../src/simworker.js');
  const send = d => { replies.length = 0; globalThis.onmessage({ data:d }); return replies; };
  post = d => { for(const r of send(d)) onState.push(r); };
  close = () => {};
} else if(opt.engine === 'remote'){
  // the same adapter the page uses, over Node's own WebSocket
  const { RemoteEngine } = await import('../src/remoteworker.js');
  const engine = new RemoteEngine(opt.host);
  engine.onmessage = e => onState.push(e.data);
  post = d => engine.postMessage(d);
  close = () => engine.terminate();
} else {
  const exe = path.join(ROOT, 'cuda', 'engine.exe');
  if(!fs.existsSync(exe)){
    console.error('linen: cuda/engine.exe is not built. Build it with\n'
      + '  nvcc -O3 -arch=sm_89 -o cuda/engine.exe cuda/engine.cu');
    process.exit(1);
  }
  const { CudaEngine } = await import('../host/cudaengine.mjs');
  const engine = new CudaEngine(exe, () => {});
  engine.on('message', d => onState.push(d));
  post = d => engine.postMessage(d);
  close = () => engine.stop ? engine.stop() : engine.proc.kill();
}
// The engine answers a tick with a state frame and a hello with a hello.
// On the CPU that answer is already in the array when post returns; the CUDA child answers over a pipe, so a message waits for its reply.
const waitReply = async (cmd, what) => {
  for(let spin = 0; spin < 200000; spin++){
    const k = onState.findIndex(r => r.cmd === cmd);
    if(k >= 0) return onState.splice(0, k + 1)[k];
    const err = onState.find(r => r.cmd === 'error');
    if(err){ console.error('linen: ' + what + ': ' + err.message); close(); process.exit(1); }
    await new Promise(r => setTimeout(r, 1));
  }
  console.error('linen: the engine stopped answering');
  close(); process.exit(1);
};
const waitFrame = () => waitReply('state', 'the engine refused the run');
// hello first: the engine's contract, its name, and the terms it implements (ENGINE.md section 2); a network using a term it does not name is not run
post({ cmd:'hello' });
const hello = await waitReply('hello', 'no hello from the engine');
{
  const bad = checkHello(hello), miss = bad ? [] : missingTerms(init, hello);
  if(bad || miss.length){
    console.error('linen: ' + (bad || 'the ' + hello.engine + ' engine does not implement ' + miss.map(m => m[1]).join(', ')) + '; the scene was not run');
    close(); process.exit(1);
  }
  say('engine: ' + hello.engine + (opt.engine === 'remote' ? ' at ' + opt.host : ''));
  const ign = ignoredCustom(init, hello);
  if(ign) console.warn('linen: ' + customNote(ign, hello));
}
post(init);
{
  const err = onState.find(r => r.cmd === 'error');
  if(err){ console.error('linen: the engine refused the network: ' + err.message); close(); process.exit(1); }
}

// ---- run, keeping spike times per cell and spikes per bin per population
const tagOf = i => (net.tags && net.tags[net.src[i]]) || NEURON_TYPES[net.ntype[i]].key;
const pops = new Map();
for(let i = 0; i < n; i++){
  const t = tagOf(i);
  if(!pops.has(t)) pops.set(t, { cells:[], bin:0, bins:[] });
  pops.get(t).cells.push(i);
}
// A graded cell never spikes, so a population of them reads as its mean release, (v - thr) / slope clamped to [0, 1], from the potentials the engine streams when asked: the probe's readout (probeview.js), sampled every tick.
// A population that mixes graded and spiking cells reads as spikes, as the probe reads it.
const G = gradedArrays(net.ntype, engineTypes());
const gradedPops = G ? [...pops.values()].filter(p => p.cells.every(i => G.grd[i])) : [];
for(const p of gradedPops){ p.rel = 0; p.relBins = []; }
let releaseRead = gradedPops.length > 0;
if(releaseRead && !answers(hello, 'sendV')){
  console.warn('linen: the ' + hello.engine + ' engine does not stream membrane potentials, so the ' + gradedPops.length +
    ' graded population' + (gradedPops.length > 1 ? 's' : '') + ' (cells that never spike) get no release readout');
  releaseRead = false;
}
if(releaseRead) post({ cmd:'sendV', on:true });
let io = null;
if(drivenMaps.length){
  io = new IORuntime(e => { console.error('linen: input: ' + e); close(); process.exit(1); });
  io.attach({ postMessage:d => post(d) }, drivenMaps, init.seed);
  io.last = -1e9;                                    // the stimulus clock is simulated time and starts at zero
}
const want = Math.min(opt.cells, n);
const stride = want >= n ? 1 : Math.max(1, Math.floor(n/want));
const kept = [];
for(let i = 0; i < n; i += stride) kept.push(i);
const keptRow = new Map(kept.map((i, r) => [i, r]));
const events = [];                                   // [t_ms, row] into kept
const runStart = Date.now();
let spikes = 0;
for(let t = 0; t < ms; t += opt.tick){
  if(io) io.frame(t);
  post({ cmd:'tick', steps:opt.tick });
  const st = await waitFrame();
  const fired = st.fired;
  if(releaseRead){
    if(!st.v){ console.error('linen: the engine was asked for membrane potentials and sent none'); close(); process.exit(1); }
    const v = st.v;
    for(const p of gradedPops){
      let r = 0;
      for(const i of p.cells) r += Math.min(1, Math.max(0, (v[i] - G.thr[i])/G.slope[i]));
      p.rel += opt.tick*r/p.cells.length;
    }
  }
  for(let i = 0; i < n; i++){
    const f = fired[i];
    if(!f) continue;
    spikes += f;
    pops.get(tagOf(i)).bin += f;
    const row = keptRow.get(i);
    if(row !== undefined) for(let k = 0; k < f; k++) events.push([t, row]);
  }
  if((t + opt.tick) % opt.bin === 0){
    for(const p of pops.values()){
      p.bins.push(p.bin/(p.cells.length*opt.bin/1000));
      p.bin = 0;
      if(releaseRead && p.relBins){ p.relBins.push(100*p.rel/opt.bin); p.rel = 0; }
    }
    if(!opt.quiet && ((t + opt.tick) % 5000 === 0))
      process.stdout.write('\r' + ((t + opt.tick)/1000) + ' s of ' + (ms/1000) + ' simulated');
  }
}
if(!opt.quiet) process.stdout.write('\r');
const wallS = (Date.now() - runStart)/1000;
close();
say('ran ' + (ms/1000) + ' s in ' + wallS.toFixed(1) + ' s of wall clock ('
  + (ms/1000/wallS).toFixed(2) + 'x realtime), ' + spikes.toLocaleString() + ' spikes, '
  + (spikes/(n*ms/1000)).toFixed(2) + ' Hz mean');

// ---- what it wrote
const stamp = (() => { const d = new Date(), p2 = x => String(x).padStart(2, '0');
  return '' + d.getFullYear() + p2(d.getMonth() + 1) + p2(d.getDate()) + 'T' + p2(d.getHours()) + p2(d.getMinutes()); })();
const slug = sceneName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const dir = path.resolve(ROOT, opt.out || path.join('runs', slug + '-' + stamp));
fs.mkdirSync(dir, { recursive:true });

const git = a => { try { return execFileSync('git', a, { cwd:ROOT, encoding:'utf8' }).trim(); } catch(e){ return ''; } };
const commit = git(['rev-parse', '--short', 'HEAD']);
const dirty = !!git(['status', '--porcelain']);

const binCount = pops.values().next().value.bins.length;
const series = [...pops].map(([tag, p]) => ({ label:tag,
  points:p.bins.map((hz, k) => [(k + 1)*opt.bin, hz]) }));
fs.writeFileSync(path.join(dir, 'rates.csv'), multiSeriesCsv(series));
if(releaseRead){
  const gp = [...pops].filter(([, p]) => p.relBins);
  fs.writeFileSync(path.join(dir, 'release.csv'), csv(['t_ms', ...gp.map(([tag]) => tag + '_pct')],
    gp[0][1].relBins.map((_, k) => [(k + 1)*opt.bin, ...gp.map(([, p]) => +p.relBins[k].toFixed(4))])));
}
fs.writeFileSync(path.join(dir, 'spikes.txt'), spikeTrainsAscii(events, kept.length));
fs.writeFileSync(path.join(dir, 'cells.csv'), csv(
  ['row', 'cell', 'population', 'source_node', 'index_in_source', 'cell_type', 'x_um', 'y_um', 'z_um'],
  kept.map((i, r) => [r, i, tagOf(i), net.src[i], net.lidx ? net.lidx[i] : '',
    NEURON_TYPES[net.ntype[i]].key, net.pos[i*3], net.pos[i*3 + 1], net.pos[i*3 + 2]])));
fs.writeFileSync(path.join(dir, 'settings.json'), JSON.stringify(
  { scene:sceneName, nodes:ed.nodes.map(q => ({ id:q.id, type:q.type, name:q.name || '', on:q.on !== false,
    params:q.params, inputs:q.inputs.map(c => c ? c.id : null) })) }, null, 1));
const run = {
  scene:sceneName, source, resolution:opt.res, seconds:opt.seconds, binMs:opt.bin, tickMs:opt.tick,
  inputsDriven:!undriven.length, ...(undriven.length ? { undrivenInputs:undriven } : {}),
  drivenInputs:drivenMaps.map(m => m.tag),
  overrides:applied, engine:opt.engine === 'cpu' ? 'cpu (src/simworker.js, the reference)'
    : opt.engine === 'cuda' ? 'cuda (cuda/engine.cu)' : 'remote (' + hello.engine + ' at ' + opt.host + ')',
  cells:n, synapses:net.synCount, seed:init.seed,
  populations:[...pops].map(([tag, p]) => ({ tag, cells:p.cells.length,
    meanHz:+(p.bins.reduce((a, b) => a + b, 0)/Math.max(1, binCount)).toFixed(3),
    ...(p.relBins ? { graded:true, releasePct:releaseRead
      ? +(p.relBins.reduce((a, b) => a + b, 0)/Math.max(1, p.relBins.length)).toFixed(3) : null } : {}) })),
  spikes, meanHz:+(spikes/(n*ms/1000)).toFixed(3),
  cellsWritten:kept.length, everyNthCell:stride,
  delayClamped:net.delayClamped || 0,
  wallSeconds:+wallS.toFixed(1), realtimeFactor:+(ms/1000/wallS).toFixed(3),
  commit:commit + (dirty ? ' (with uncommitted changes)' : ''),
  node:process.version, when:new Date().toString(),
};
fs.writeFileSync(path.join(dir, 'run.json'), JSON.stringify(run, null, 1));
say('wrote ' + path.relative(ROOT, dir).replace(/\\/g, '/') + ': run.json, rates.csv, ' + (releaseRead ? 'release.csv, ' : '') +
  'spikes.txt, cells.csv, settings.json');
for(const p of run.populations)
  say('  ' + p.tag.padEnd(16) + String(p.cells).padStart(8) + ' cells' + (p.graded
    ? (p.releasePct === null ? '   graded, no readout' : p.releasePct.toFixed(1).padStart(9) + ' % release (graded)')
    : p.meanHz.toFixed(2).padStart(9) + ' Hz'));

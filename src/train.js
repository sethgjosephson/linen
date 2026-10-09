// Overnight training runner.
// Computes a scenario headlessly, keeps the checkpoint's plasticity switch as it is (plast=0 on the address bar turns learning off whatever the switch says), and free-runs the engine with a message-driven pump (no animation frames, so a hidden tab keeps training at full speed).
// The input encoder is driven by simulated time, so the stimulus timeline is identical no matter how fast the engine runs.
// Brain-file checkpoints and metrics land in the project folder under runs/<tag>/ when serve.py is serving one, and in browser storage otherwise (src/store.js decides, and the file pane says which); load a checkpoint with LOAD BRAIN in the playground.
// URL params: scenario (name substring, default visual), engine (gpu|cpu), hours (default 8; min=N overrides for short runs), steps (per tick, 50), ckptmin (sim-minutes between checkpoints, 30).
import { engineConfig, NODE_DEFS, NEURON_TYPES, computeNode, setResolution, setComputeProgress, setMaxSyn,
  buildRuleTable, RULE_STRIDE, engineTypes } from './nodes.js';
import { PROTOCOL } from './protocol.js';
import { initState } from './rand.js';
import { ALL_SCENARIOS, setFootprint, getFootprint,
  setLabScale, setLabAmp, setLabSeed,
  setBlockAmp, setBlockDrive, setBlockLayers,
  setBlockXl4ei, setBlockXrelay, setBlockXdrive,
  setBlockBalance, setBlockLag, setBlockXrec, setBlockXrecp, setBlockXrecw, setBlockXdisp, setBlockXl23rec, setBlockUrelay, setBlockXsigma,
  setBlockXshare, setBlockXprob, setBlockSpread,
  setBlockRelay } from './experiments.js';
import { buildWeightsFile, buildPlasticFile, weightFileBytes, plasticFileBytes,
  plasticCount, parseBrainFile, applyBrainWeights } from './brain.js';
import { IORuntime } from './io.js';
import { SETS, scheduleAt, setVoice, buildAtlas, setGlyphAtlas } from './curriculum.js';
import { multisensoryIndex, crossDecode, rdm, rdmCorrelation, crossRdm, populationSparseness, blockStructure, ItemRecorder, selectivityProfile, decodeAccuracy,
  completionScore, transferAccuracy,
  responseProfile, ensembleCorrelation, decodeTopAccuracy,
  likeToLike, popLinks, completionTrajectory,
  decodeShape, conjunctiveSets, conjunctiveRecall, sectorHistogram,
  latencyAgreement, fanoPerCell, packTrialFile,
  pairClasses, setWeightStats, itemCellSets, evokedTransfer, betweenSynapses, itemCoupling } from './analysis.js';
import { Viewer } from './viewer.js';
import { attachProbes, setupProbes, updateProbes } from './probeview.js';
import { attachInputs, setupInputs, updateInputs } from './inputview.js';
import { RemoteEngine } from './remoteworker.js';
import { store, whereLabel, browserRuns, importBrowserRuns } from './store.js';
import { renderRuns } from './runlist.js';
import * as P from './project.js';

// The run's configuration comes from the address bar when this module is the page, and from an injected object when the playground drives it in place.
// Same keys either way, so a run started from the building UI and one started from a URL are the same run.
export async function startRun(params){
const Q = params instanceof URLSearchParams ? params
  : new URLSearchParams(params !== undefined && params !== null ? String(params) : location.search);
const HOURS = Q.get('min') ? +Q.get('min')/60 : +(Q.get('hours') || 8);
const ENGINE = Q.get('engine') || 'gpu';
const STEPS = Math.max(1, parseInt(Q.get('steps') || '50', 10));
const CKPT_MS = Math.max(0.2, +(Q.get('ckptmin') || 30)) * 60000;
const BRAIN_MS = Math.max(CKPT_MS, (+Q.get('brainmin') || 60) * 60000);
const KEEP = Math.max(1, +Q.get('keep') || 6);
// A replicate sweep varies the engine's noise seed per run without touching the graph; absent, the seed is the one the wiring carries.
const SEED_Q = Q.get('seed') !== null ? (+Q.get('seed') | 0) : null;
// Retention is bounded by bytes as well as count: a weights file for a full-density column is over a gigabyte, so eight of them do not fit in the storage quota no matter what keep says.
const KEEP_BYTES = Math.max(0.5, +Q.get('keepgb') || 6) * 1073741824;
// Plasticity starts after the network has settled.
// Switching it on at t=0 lets the first synchronous transient, when every neuron is near reset and the background current hits them together, set the weights: inhibitory plasticity reads that burst as a layer far above target and drives inhibition up several fold.
// If it silences the layer's interneurons too, nothing can undo it, because the Vogels depression term needs presynaptic inhibitory spikes that no longer happen. brain=<run>/<file> or brain=latest resumes from a saved checkpoint.
// The curriculum scheduler is a pure function of simulated time, so restoring the clock restores the lesson position exactly and nothing else has to be bookkept for it. footprint=0.25 builds a quarter square millimeter of cortical surface at unchanged density, distances and delays.
// Population counts are per square millimeter, so this is a smaller piece of the same tissue rather than a coarser model of the whole one, and each footprint carries its own tuned background drive.
const FOOTPRINT = Math.max(0.05, Math.min(1, +Q.get('footprint') || 1));
const BRAIN = Q.get('brain');
// brainurl=<path> loads a brain file over HTTP instead of out of browser storage.
// Storage is per browser, so a checkpoint written by one browser cannot be evaluated in another without this; a downloaded file served from the project folder can.
const BRAIN_URL = Q.get('brainurl');
// clock=<min> starts the simulated clock somewhere other than zero without loading weights.
// The curriculum is a pure function of that clock, so a baseline run set to the same clock as a trained one sees the identical items in the identical order with the identical poses and probes.
// That turns a comparison of two networks on the same distribution into a comparison on the same stimuli.
const CLOCK = Q.get('clock') === null ? null : Math.max(0, +Q.get('clock') || 0) * 60000;
// usegraph=1 computes the graph stored inside the brain file instead of a named scenario.
// Brain files carry the graph that built them, so a checkpoint outlives the scenario code it came from and can still be evaluated after the scenario has been rewritten or removed.
const USE_GRAPH = Q.get('usegraph') === '1';
// usegraph=2 computes the graph the building UI is showing, handed over in __npTrainGraph; only meaningful when the run shares this page.
const LIVE_GRAPH = Q.get('usegraph') === '2'
  ? (globalThis.__npTrainGraph || null) : null;
// plastx=<n> multiplies both STDP rates on the checkpoint node.
// Scaling them together leaves the fixed point, which is their ratio, where it was, so this compresses the timescale rather than changing the rule.
// For asking whether a configuration learns at all, before spending a night finding out at production rates.
const PLASTX = Math.max(0, +Q.get('plastx') || 0);
// pump=host (remote engine only) moves the tick loop, the stimulus clock, the warm-up switch and the brain-file cadence into the engine host, so a suspended tab does not stall the run.
// The page becomes a viewer: it draws and measures whatever state frames reach it and resynchronizes its clock from the host's, and the host writes checkpoints to host/runs/.
const PUMP = Q.get('pump') === 'host';
// attach=1 opens a live detached run as a viewer: the page wires the same network locally for the scene (determinism makes it identical), then asks the host to stream the run's state frames instead of starting an engine of its own. run=<dir> picks a run when several are live.
// The viewer sends nothing that steers the run.
const ATTACH = Q.get('attach') === '1';
// modes in which this page does not drive the engine loop itself
const PASSIVE = PUMP || ATTACH;
// preflight=1 wires the scenario, prints the manifest and the computed-net census, and stops before any engine is touched.
// Configuration questions are compute-time questions: an assembly defect is visible in the census at zero simulated minutes, and a thirty second wiring answers them without spending a training run.
const PREFLIGHT = Q.get('preflight') === '1';
// useweights=0 takes the graph from the brain file and leaves the weights as built.
// That is the baseline pass for a checkpoint whose scenario no longer exists in the code: same tissue, same lesson position, untrained.
const USE_WEIGHTS = Q.get('useweights') !== '0';
// Cross-decode trials pool across checkpoints from this sim-minute on (poolfrom on the address bar); the readout reports the pooled test beside the per-window one.
const POOL_FROM_MS = Math.max(0, +(Q.get('poolfrom') || 90)) * 60000;
const WARMUP_RAW = Q.get('warmupsec');    // absent means the default, 0 means disabled
const WARMUP_MS = Math.max(0, WARMUP_RAW === null ? 30 : +WARMUP_RAW || 0) * 1000;
const TOTAL_MS = Math.round(HOURS * 3600000);
const PLAST = Q.get('plast') !== '0';        // plast=0: evaluation run, no learning
let NODE_PLAST = true;                       // the checkpoint's switch, read at compute time
// probe=<n> overrides the scenario's unimodal probe rate.
// An evaluation run wants far more probe trials than a training run does, because completion is measured per item per modality and probes alternate, so probe=8 yields one trial per item per modality every 16 presentations.
// With learning off there is no reason to be sparing.
const PROBE = Q.get('probe') === null ? null : Math.max(0, parseInt(Q.get('probe'), 10) || 0);
const RES = +Q.get('res') || 0;             // override the scenario's suggestion
// Long runs are allowed a bigger synapse budget than the interactive guard: the guard exists to stop an interactive user freezing the tab, which does not apply here.
// The budget is set in memory for this run and cleared when it finishes.
// Not in localStorage, which is the machine's own SYN setting from the top bar and would be replaced permanently with 1.2e9 by every run.
// A run that is stopped rather than finished keeps the raised budget until the page is reloaded.
setMaxSyn(+Q.get('maxsyn') || 1.2e9);

const logEl = document.getElementById('log');
const statEl = document.getElementById('stat');
// Everything printed before the engine starts describes how the run was configured and never changes again; everything after is what the run did.
// In one scroll the configuration pushes the events out of sight and the events bury the configuration. log() writes to whichever the run is currently in: setup until the engine exists, events after.
const setupEl = document.getElementById('setup') || logEl;
let logDest = setupEl;
const log = s => {
  logDest.textContent += s + String.fromCharCode(10);
  if(logDest === logEl && globalThis.__npTrainTab) globalThis.__npTrainTab('log');
};
// A second run in the same page appends to the first one's panes, so a divider keeps its settings from reading as the previous run's.
for(const el of new Set([setupEl, logEl]))
  if(el.textContent.trim())
    el.textContent += String.fromCharCode(10) + '---- new run ----' + String.fromCharCode(10);

// Only one training run at a time.
// A page left open in another tab keeps simulating after its server is gone and quietly competes for the GPU, which makes throughput and rate readings untrustworthy.
// A new run announces itself and older ones stand down.
const RUN_ID = Math.random().toString(36).slice(2) + Date.now().toString(36);
const runChan = ('BroadcastChannel' in self) ? new BroadcastChannel('np-train') : null;
let superseded = false, liveWorker = null;
// The claim protocol protects the local GPU from tabs competing for it.
// A remote engine runs on the host's card, where several runs at once is deliberate (triage grids, twin controls), so remote pages neither claim nor stand down whether or not the host drives the pump.
if(runChan && !PASSIVE && Q.get('engine') !== 'remote'){
  runChan.onmessage = e => {
    if(e.data && e.data.cmd === 'claim' && e.data.id !== RUN_ID){
      superseded = true;
      if(liveWorker) liveWorker.terminate();
      statEl.textContent = 'STOPPED: a newer training run took over';
      log('stopped: another run claimed the engine');
    }
  };
  runChan.postMessage({ cmd:'claim', id:RUN_ID });
}

// The playground already owns a viewer, a gizmo and a selection; making a second one on the same element would fight it.
// When it hands this module a viewer, use that, so the graph, properties panel and neuron inspection keep working while the run trains.
const viewer = globalThis.__npViewer
  || new Viewer(document.getElementById('viewer'), document.getElementById('hud'));
if(!globalThis.__npViewer){
  attachProbes(document.getElementById('probes'));
  attachInputs(document.getElementById('inputs'));
}
const rasterCv = document.getElementById('raster');   // may be absent
window.viewer = viewer;   // debug access, mirrors the playground
let rasterIdx = null;
function drawRaster(fired, n){
  if(!rasterIdx || rasterIdx.n !== n){
    const K = Math.min(260, n);
    rasterIdx = { n, idx:new Int32Array(K) };
    for(let k=0;k<K;k++) rasterIdx.idx[k] = Math.floor(k*n/K);
    const c0 = rasterCv.getContext('2d');
    c0.fillStyle = '#000'; c0.fillRect(0, 0, rasterCv.width, rasterCv.height);
  }
  if(rasterCv.width !== rasterCv.clientWidth) rasterCv.width = rasterCv.clientWidth;
  const c = rasterCv.getContext('2d'), w = rasterCv.width, h = rasterCv.height;
  c.drawImage(rasterCv, -1, 0);
  c.fillStyle = '#000'; c.fillRect(w-1, 0, 1, h);
  c.fillStyle = '#fff';
  const K = rasterIdx.idx.length;
  for(let k=0;k<K;k++)
    if(fired[rasterIdx.idx[k]]) c.fillRect(w-1, Math.floor(k*h/K), 1, 1);
}
// render on animation frames only: a hidden tab stops drawing while the message-driven training pump keeps running at full speed
(function render(){ requestAnimationFrame(render); viewer.frame(); updateInputs(); })();

function fakeEditor(){
  const nodes = [];
  return { nodes, nextId:1,
    addNode(type, x, y){
      const def = NODE_DEFS[type], params = {};
      def.params.forEach(p => params[p.k] = structuredClone(p.def));
      const n = { id:this.nextId++, type, x, y, params, inputs:new Array(def.inputs).fill(null) };
      nodes.push(n); return n;
    } };
}

// One subdirectory per run, since a flat directory of checkpoints does not say which run a file came from; the tag is also the only place the start time is recorded.
// Second resolution, so two runs launched in the same minute do not share a host-side checkpoint directory.
const RUN_TAG = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
// Paths, not directory handles.
// The store resolves them against the project folder or against browser storage, and every caller here is the same either way: a run directory is a string, and nothing in this file knows which of the two it is writing into.
let ST = null;
const runDir = tag => P.runDir(tag || RUN_TAG);
async function writeFile(dir, name, data){
  return ST.write(dir + '/' + name, data);
}
// Runs written before the project folder existed are still in browser storage, where nothing in a file manager shows they exist.
// They are counted and named rather than moved: copying gigabytes is a decision, so the IMPORT link makes it one.
// Nothing is deleted from browser storage by this, so an import that goes wrong costs time and not data.
async function reportLegacy(){
  if(ST.mode !== 'project') return;
  const runs = await browserRuns();
  if(!runs.length) return;
  const bytes = runs.reduce((t, r) => t + r.bytes, 0);
  const line = document.createElement('li');
  line.className = 'flegacy';
  line.textContent = runs.length + ' run(s) still in browser storage (' +
    (bytes/1073741824).toFixed(2) + ' GB), not in the project folder \u00b7 ';
  const a = document.createElement('a');
  a.href = '#'; a.textContent = 'IMPORT';
  a.onclick = async e => {
    e.preventDefault();
    a.textContent = 'importing...'; a.onclick = null;
    const r = await importBrowserRuns(ST, m => log(m));
    log('imported ' + r.copied + ' file(s), ' + r.skipped + ' already present, ' +
      (r.bytes/1073741824).toFixed(2) + ' GB into ' + ST.root);
    log('browser storage was not touched; clear it yourself once the copies check out');
    await listRuns();
  };
  line.appendChild(a);
  // under the line naming the project folder, since it is a remark about that folder rather than a heading of its own
  const ul = document.getElementById('files');
  ul.insertBefore(line, ul.firstChild ? ul.firstChild.nextSibling : null);
}

// Find a saved brain in the run store.
// "latest" takes the most recently written checkpoint across every run, which is what a resume after a pause almost always wants.
// It returns the bytes rather than a File, because the two stores have no file object in common.
async function findBrain(spec){
  // "<tag>/<file>" names one checkpoint.
  // It is written without the brains/ segment because that is a detail of the layout rather than something anyone should have to type, and the run tag is what identifies it.
  if(spec.includes('/')){
    const [tag, file] = spec.split('/');
    const bytes = await ST.read(P.runBrainPath(tag, file));
    return bytes ? { name:spec, bytes } : null;
  }
  let best = null;
  for(const d of await ST.list(P.runsDir())){
    if(!d.dir) continue;
    for(const f of await ST.list(P.runBrainsDir(d.name))){
      if(f.dir || !f.name.endsWith('.npb')) continue;
      if(spec !== 'latest' && f.name !== spec) continue;
      if(!best || f.mtime > best.mtime)
        best = { name: d.name + '/' + f.name, mtime:f.mtime, tag:d.name, file:f.name };
    }
  }
  if(!best) return null;
  return { name:best.name, bytes: await ST.read(P.runBrainPath(best.tag, best.file)) };
}

async function listRuns(){
  return renderRuns(document.getElementById('files'), ST,
    { currentTag:RUN_TAG, whereText: await whereLabel(),
      onChart:(rows, tag) => globalThis.__npChartRun && globalThis.__npChartRun(rows, tag),
      onError:msg => log(msg) });
}

// A run that only writes files at checkpoint boundaries is invisible between them: while waiting on the first checkpoint of a full-density run there is no way to tell wiring from training from a dead engine without reading the tab's own log.
// The status file is written from the first moment of the run and refreshed on a wall-clock timer, so progress and errors are readable from outside the tab.
let statusDir = null, lastStatus = 0, lastError = null, RESUMED_FROM = null;
async function writeStatus(phase, extra){
  if(!statusDir) return;
  try {
    await writeFile(statusDir, P.RUN_STATUS, JSON.stringify({
      run:RUN_TAG, phase, at:new Date().toISOString(), resumedFrom:RESUMED_FROM,
      footprint:FOOTPRINT,
      wallMin:+((performance.now() - START_WALL)/60000).toFixed(2),
      error:lastError, ...extra }, null, 1));
  } catch(e){ /* status is best effort, never break the run for it */ }
}
const START_WALL = performance.now();

async function main(){
  ST = await store();
  statusDir = runDir();
  // Before anything is listed, so a folder written by the first layout reads as a folder rather than as a set of runs with no checkpoints in them.
  const mig = await P.migrate(ST, m => log(m));
  if(mig.moved) log('moved ' + mig.moved + ' file(s) in ' + mig.runs +
    ' run(s) into the current layout');
  await P.openProject(ST, ST.mode === 'project' ? undefined : 'browser');
  await listRuns();
  await reportLegacy();
  log('files go to ' + (await whereLabel()));
  await writeStatus('starting');
  const want = (Q.get('scenario') || 'visual').toLowerCase();
  const sc = ALL_SCENARIOS.find(s => s.name.toLowerCase().includes(want));
  if(!sc && !USE_GRAPH && !LIVE_GRAPH){
    log('no scenario matching "' + want + '"'); return; }
  if(Q.get('usegraph') === '2' && !LIVE_GRAPH){
    log('usegraph=2 needs the building UI to hand over its graph'); return; }
  // the brain file is read before the build when its graph is the build
  const loadBrain = async () => {
    if(BRAIN_URL){
      const resp = await fetch(BRAIN_URL);
      if(!resp.ok) throw new Error(`brainurl ${BRAIN_URL} returned ${resp.status}`);
      return { name:BRAIN_URL.split('/').pop(), buf:await resp.arrayBuffer() };
    }
    const hit = await findBrain(BRAIN);
    if(!hit) throw new Error('no saved brain matching "' + BRAIN + '" in ' + ST.label);
    return { name:hit.name, buf:hit.bytes.buffer.slice(
      hit.bytes.byteOffset, hit.bytes.byteOffset + hit.bytes.byteLength) };
  };
  let pre = null;
  if(USE_GRAPH){
    if(!BRAIN && !BRAIN_URL) throw new Error('usegraph needs brain or brainurl');
    const got = await loadBrain();
    pre = { name:got.name, brain:parseBrainFile(got.buf) };
    if(!pre.brain.graph || !pre.brain.graph.nodes)
      throw new Error('this brain file carries no graph');
  }
  log(`network: ${LIVE_GRAPH ? 'the graph on screen, ' + LIVE_GRAPH.nodes.length + ' nodes'
    : sc.name} · engine ${ENGINE} · ${(TOTAL_MS/3600000).toFixed(2)} sim-hours · checkpoints every ${(CKPT_MS/60000).toFixed(1)} sim-min` +
    (FOOTPRINT < 1 ? ` · footprint ${FOOTPRINT} mm2` : ''));
  setFootprint(FOOTPRINT);
  if(Q.get('labscale')) setLabScale(+Q.get('labscale'));
  if(Q.get('labamp')) setLabAmp(+Q.get('labamp'));
  if(Q.get('labseed')) setLabSeed(+Q.get('labseed'));
  if(Q.get('blockamp')) setBlockAmp(+Q.get('blockamp'));
  if(Q.get('blockdrive')) setBlockDrive(+Q.get('blockdrive'));
  // blocklayers=L2/3,L4,L5,L6 multiplies the tuned background drive per layer
  if(Q.get('blocklayers')) setBlockLayers(Q.get('blocklayers'));
  // binding architecture switches: plastic E/I at the convergence site, and the unfrozen L4 to L2/3 relay in the association territory
  if(Q.get('xl4ei')) setBlockXl4ei(+Q.get('xl4ei'));
  if(Q.get('xrelay')) setBlockXrelay(+Q.get('xrelay'));
  if(Q.get('xdrive')) setBlockXdrive(+Q.get('xdrive'));
  setBlockXrecp(Q.get('xrecp'));
  setBlockXrecw(Q.get('xrecw'));
  setBlockXdisp(Q.get('xdisp'));
  setBlockXl23rec(Q.get('xl23rec'));
  setBlockUrelay({ p:Q.get('urelayp'), w:Q.get('urelayw') });
  setBlockXsigma(Q.get('xsigma'));
  setBlockBalance({ vamp:Q.get('vamp'), aamp:Q.get('aamp'),
    xvw:Q.get('xvw'), xaw:Q.get('xaw') });
  if(Q.get('lag') !== null && Q.get('lag') !== '') setBlockLag(+Q.get('lag'));
  if(Q.get('xrec')) setBlockXrec(+Q.get('xrec'));
  if(Q.get('xshare')) setBlockXshare(+Q.get('xshare'));
  if(Q.get('xprob')) setBlockXprob(+Q.get('xprob'));
  if(Q.get('spread')) setBlockSpread(+Q.get('spread'));
  if(Q.get('relay')) setBlockRelay(+Q.get('relay'));
  const ed = fakeEditor();
  ed.suggestResolution = 0;
  if(LIVE_GRAPH){
    for(const n of LIVE_GRAPH.nodes) ed.nodes.push(structuredClone(n));
    ed.nextId = LIVE_GRAPH.nextId ||
      (ed.nodes.reduce((m, n) => Math.max(m, n.id), 0) + 1);
    log(`graph on screen: ${ed.nodes.length} nodes`);
  } else if(pre){
    for(const n of pre.brain.graph.nodes) ed.nodes.push(n);
    ed.nextId = pre.brain.graph.nextId ||
      (ed.nodes.reduce((m, n) => Math.max(m, n.id), 0) + 1);
    log(`graph from ${pre.name}: ${ed.nodes.length} nodes, ` +
      `saved resolution ${pre.brain.graph.resolution === undefined ? 'unset' : pre.brain.graph.resolution}`);
  } else sc.build(ed);
  setComputeProgress(f => {
    // Short: this line lives in a 300 px pane with nowrap and an ellipsis, so anything longer is cut off mid-word.
    // It names no density: a run trains the graph on screen, at whatever resolution it was built at.
    statEl.textContent = `wiring synapses ${Math.round(f*100)}%`;
    // The building UI owns the progress bar in the top bar, and a run replaces its progress hook with this one, so this one has to drive the bar or it goes blank during the wiring that matters.
    if(globalThis.__npComputeBar) globalThis.__npComputeBar(f);
    const now = performance.now();
    if(now - lastStatus > 15000){ lastStatus = now; writeStatus('wiring', { wireFrac:+f.toFixed(3) }); }
  });
  // scramble=1 redraws which tone accompanies which place every presentation: the decisive control for an association claim, since the stimulus statistics are unchanged and only the pairing is destroyed.
  if(Q.get('scramble') === '1'){
    const cur = ed.nodes.filter(n => n.type === 'curriculum');
    cur.forEach(n => n.params.scramble = 1);
    if(cur.length) log('pairing scrambled: sound no longer predicts sight');
  }
  if(PROBE !== null){
    const cur = ed.nodes.filter(n => n.type === 'curriculum');
    cur.forEach(n => n.params.probeEvery = PROBE);
    if(cur.length) log(`unimodal probe every ${PROBE || 'never'} (scenario override)`);
  } else if(!PLAST){
    // An evaluation run has no reason to be sparing with probes: learning is off, so a probe disturbs nothing, and every readout measure is starved at the training rate (about two trials per item per condition in a thirty-minute window at 5).
    // Every other presentation, unless the scene already probes at least that often, so the bimodal readout keeps its trials too. probe=N on the run overrides either way.
    const cur = ed.nodes.filter(n => n.type === 'curriculum');
    let changed = 0;
    for(const n of cur){
      const was = n.params.probeEvery | 0;
      if(was === 0 || was > 2){ n.params.probeEvery = 2; changed++; }
    }
    if(changed) log('evaluation run: unimodal probe every 2 on ' + changed +
      ' curriculum node(s), since learning is off; probe=N on the run overrides');
  }
  // Curriculum retargeting without editing the scenario: curset picks the lesson set (0 is the alphabet), curon/curoff the presentation timing, curjitter/curscalevar/currotvar the pose variation, voice=1 mixes the recorded speech clips into the audio channel.
  {
    const cur = ed.nodes.filter(n => n.type === 'curriculum');
    if(Q.get('curset') !== null && cur.length){
      cur.forEach(n => n.params.set = +Q.get('curset'));
      log(`lesson set ${+Q.get('curset')} (scenario override)`);
    }
    if(Q.get('curon') !== null && cur.length)
      cur.forEach(n => n.params.onMs = +Q.get('curon'));
    if(Q.get('curoff') !== null && cur.length)
      cur.forEach(n => n.params.offMs = +Q.get('curoff'));
    // Pose variation.
    // A lesson set presented at one fixed pose is 26 fixed images, and what a network can learn from that is a lookup: the curriculum's own design notes say interleaving and jittered pose are what decorrelate the input the way developmental experience does.
    const POSE = { curjitter:'jitter', curscalevar:'scaleVar', currotvar:'rotVar' };
    const posed = [];
    for(const k in POSE){
      const v = Q.get(k);
      if(v !== null && v !== '' && cur.length){
        cur.forEach(n => n.params[POSE[k]] = +v);
        posed.push(POSE[k] + ' ' + (+v));
      }
    }
    if(posed.length) log('pose variation: ' + posed.join(', '));
    if(Q.get('voice') === '1' && cur.length){
      cur.forEach(n => n.params.voice = 1);
      try {
        const bank = await (await fetch(new URL('../speech/clips/voice.json', import.meta.url))).json();
        setVoice(bank);
        log(`recorded voice mixed in: ${Object.keys(bank.items).length} items` +
          (PUMP ? ' (the host loads its own copy)' : ''));
      } catch(e){ log('voice bank not found; synthetic audio only'); }
    }
  }
  const res = RES || (pre && pre.brain.graph.resolution) || ed.suggestResolution || 100;
  // Proxy downscaling preserves rates and spectra, not correlations (van Albada, Helias and Diesmann 2015), and spike-timing plasticity is a function of correlations.
  // A run below full resolution learns on statistics the full network does not have, so it is refused unless asked for explicitly, which marks the result as a non-biological intervention.
  if(res !== 100 && Q.get('allowproxy') !== '1')
    throw new Error(`training refused at ${res}% resolution: proxy compensation ` +
      'does not preserve the correlations plasticity depends on; train at 100%, ' +
      'or pass allowproxy=1 to accept a non-biological run');
  setResolution(res/100);
  if(res !== 100) log(`resolution ${res}% (allowproxy=1: a non-biological run)`);
  const outNode = ed.nodes.find(n => n.type === 'checkpoint');
  if(!outNode) throw new Error('the graph has no checkpoint node');
  // Direct overrides of the checkpoint's plasticity parameters, for isolating which term is responsible for a behavior without editing the scenario between runs.
  for(const k of ['aP', 'aM', 'wmax', 'wdep', 'iEta', 'iRho', 'scale', 'sEta', 'calS',
      'trip', 'tauY', 'het', 'tin', 'refrac', 'cons', 'consW', 'consP',
      'stp', 'stpU', 'stpTauD', 'stpTauF', 'stpNorm', 'stpOrder']){
    const v = Q.get(k.toLowerCase());
    if(v !== null && v !== ''){ outNode.params[k] = +v; log(`override ${k} = ${+v}`); }
  }
  if(PLASTX > 0){
    outNode.params.aP *= PLASTX;
    outNode.params.aM *= PLASTX;
    // the fast mechanism set compresses with the pair rule: triplet and transmitter terms are Hebbian-side, and heterosynaptic competition must stay fast relative to them (the Zenke stability argument), so all three scale together and the relative timescales survive
    for(const k of ['trip', 'het', 'tin'])
      if(outNode.params[k] > 0) outNode.params[k] *= PLASTX;
    log(`learning rates x${PLASTX}: aP ${outNode.params.aP.toPrecision(3)} ` +
      `aM ${outNode.params.aM.toPrecision(3)}, fixed point unchanged at ` +
      (outNode.params.aP/outNode.params.aM).toFixed(3));
  }
  // plast=0 on the address bar evaluates the circuit as built; otherwise the checkpoint's own switch stands.
  if(!PLAST) outNode.params.plast = 0;
  NODE_PLAST = (outNode.params.plast|0) !== 0;
  const byId = id => ed.nodes.find(n => n.id === id);
  // Show the anatomy before the wiring: placing neurons takes seconds while wiring hundreds of millions of synapses takes minutes, and an empty viewer for the first few minutes of a run is indistinguishable from a broken one.
  try {
    const cn = ed.nodes.find(n => n.type === 'connect');
    if(cn && cn.inputs[0]){
      const pts = await computeNode(byId(cn.inputs[0].id), byId);
      if(pts && pts.kind === 'points'){
        viewer.showPoints(pts);
        viewer.fitToPoints(pts.pos, pts.count);
        statEl.textContent = `${pts.count.toLocaleString()} neurons placed, wiring synapses…`;
      }
    }
  } catch(e){ /* preview is a convenience; never block the run on it */ }
  const wired = await computeNode(outNode, byId);
  // The building UI shows the size of the network on screen, and during a run the network on screen is this one; starting a run supersedes the interactive computation before it finishes, so the size has to come from here.
  if(globalThis.__npNetSize) globalThis.__npNetSize(wired.count, wired.synCount);
  log(`${wired.count.toLocaleString()} neurons · ${wired.synCount.toLocaleString()} synapses · ` +
    (PLAST ? `plast aP ${wired.aP} aM ${wired.aM} tauS ${wired.tauS} wmax ${wired.wmax} iEta ${wired.iEta} ` +
      ((wired.rhoMode|0) === 1 ? `set points measured (${wired.calS || 10}s cal)` : `iRho ${wired.iRho}`) +
      ` scale ${wired.scale ? 'on ' + wired.sEta : 'off'}` : 'evaluation (no learning)'));
  viewer.setNetwork(wired);
  viewer.fitToNet();
  setupProbes(wired);
  // resolution travels with the graph: weights-only checkpoints restore by index, so loading one has to rewire the same number of synapses
  const graph = { v:2, nextId:ed.nextId, resolution:res, nodes:ed.nodes.map(n =>
    ({ id:n.id, type:n.type, x:n.x, y:n.y, params:n.params, inputs:n.inputs })) };
  // Resume. applyBrainWeights restores a weights-only file by index, so a mismatched wiring throws with the reason rather than training something subtly wrong; a missing file stops the run rather than silently starting from scratch and wasting the night.
  let net = wired, resumeMs = 0, resumedFrom = null;
  if((BRAIN || BRAIN_URL) && !USE_WEIGHTS){
    resumeMs = CLOCK !== null ? CLOCK : (pre ? pre.brain.meta.simMs || 0 : 0);
    log(`graph only, weights as built, clock ${(resumeMs/60000).toFixed(1)} sim-min ` +
      `(baseline for the saved brain, same lesson position)`);
  } else if(BRAIN || BRAIN_URL){
    let name, brain;
    if(pre){ name = pre.name; brain = pre.brain; }
    else { const got = await loadBrain(); name = got.name; brain = parseBrainFile(got.buf); }
    const r = applyBrainWeights(net, brain);
    net = r.net;
    resumeMs = brain.meta.simMs || 0;
    resumedFrom = name; RESUMED_FROM = name;
    log(`resumed from ${name}: ${r.matched.toLocaleString()} of ` +
      `${r.saved.toLocaleString()} weights onto ${r.wired.toLocaleString()} wired, ` +
      `sim clock ${(resumeMs/60000).toFixed(1)} min`);
    if(brain.meta.curriculum)
      log(`curriculum: ${JSON.stringify(brain.meta.curriculum)}`);
  }
  if(CLOCK !== null && !BRAIN && !BRAIN_URL){
    resumeMs = CLOCK;
    log(`clock set to ${(CLOCK/60000).toFixed(1)} sim-min with baseline weights ` +
      `(same lesson position as a trained run at that clock)`);
  }
  const w0 = net.w.slice();

  // ---- run manifest and computed-net preflight -------------------------
  // Configuration defects (the wrong lesson set, zero pose variation, raw luminance, drive masks clipping territory edges) live in the assembled configuration, a layer no battery group examines: the battery validates mechanisms against literature and engines against each other, not what a computed scenario is.
  // This census prints what the network in front of the engine contains, per launch, so a wrong assembly is visible on the first screen instead of after a night of training.
  {
    const over = [];
    for(const [k, v] of Q.entries())
      if(!['scenario','engine','host'].includes(k)) over.push(k + '=' + v);
    log('overrides: ' + (over.join(' ') || 'none'));
    for(const m of (net.inputMaps || [])){
      const g = m.signal || {};
      log(`input ${m.sheet ? 'sheet' : 'bands'}: ${m.cols}${m.sheet ? 'x' + m.rows : ''}` +
        ` ch · ${m.sense || 'sight'}` + (m.tag ? ` · population ${m.tag}` : '') +
        ` · ${m.chIdx.length.toLocaleString()} connections` +
        ` · code ${m.code === 1 ? 'contrast ON/OFF' : 'luminance'} · amp ${m.amp}` +
        (m.lagMs ? ` · lag ${m.lagMs} ms` : '') +
        (g.offset ? ` · offset ${g.offset} ms from the followed stream` : '') +
        (g.src === 'curriculum' ? ` · set ${g.set} (${(SETS[g.set] || SETS[0]).name})` +
          ` · on/off ${g.onMs}/${g.offMs} ms · jitter ${g.jitter} scale ${g.scaleVar}` +
          ` rot ${g.rotVar} · acuity ${g.acuity} · probe ${g.probeEvery}` +
          (g.scramble ? ' · SCRAMBLED' : '') + (g.voice ? ' · voice' : '') : ''));
    }
    // per-population coverage: who gets background drive and a noise floor
    const noiseSet = new Uint8Array(net.count);
    for(const pr of (net.protocols || []))
      if((pr.mode|0) === 3) for(const i of pr.idx) noiseSet[i] = 1;
    const covLines = [];
    for(const pb of (net.probes || [])){
      if(!pb.idx || !pb.idx.length) continue;
      let drv = 0, noi = 0;
      for(const i of pb.idx){ if(net.bias[i] !== 0) drv++; if(noiseSet[i]) noi++; }
      covLines.push(pb.label + ' ' + Math.round(100*drv/pb.idx.length) + '/' +
        Math.round(100*noi/pb.idx.length) + '%');
    }
    if(covLines.length) log('drive/noise coverage: ' + covLines.join(' · '));
    // plastic synapse census by tag pair, from the mask actually computed
    if(net.pmask){
      const tags = net.tags || {};
      const tagOf = i => tags[net.src[i]] || ('pop' + net.src[i]);
      const pairs = new Map();
      let tot = 0;
      for(let pre = 0; pre < net.count; pre++){
        const s0 = net.preStart[pre], s1 = net.preStart[pre + 1];
        for(let sn = s0; sn < s1; sn++){
          if(!(net.pmask[sn] & 31)) continue;   // the rule is the low five bits
          tot++;
          const k = tagOf(pre) + '>' + tagOf(net.post[sn]);
          pairs.set(k, (pairs.get(k) || 0) + 1);
        }
      }
      const top = [...pairs.entries()].sort((x, y) => y[1] - x[1])
        .map(([k, n2]) => k + ' ' + n2);
      log('plastic synapses: ' + tot.toLocaleString() + ' of ' +
        net.synCount.toLocaleString() + ' · ' + (top.join(' · ') || 'NONE'));
      if(PLAST && tot === 0)
        log('PREFLIGHT WARNING: plasticity is on and no synapse is plastic');
    }
  }
  if(PREFLIGHT){
    statEl.textContent = 'PREFLIGHT ONLY · nothing was simulated';
    log('preflight complete; no engine started');
    writeStatus('preflight');
    return;
  }
  const dir = statusDir;
  const runTag = RUN_TAG;
  globalThis.__npRunTag = RUN_TAG;      // the page's sweep table names the run by it

  // engine=remote runs the reference engine in a native host process over a WebSocket (host/host.mjs), same protocol, host=ws://... overrides the default local port.
  // Everything downstream holds a thing that looks like a Worker and cannot tell the difference, which is the seam REMOTE.md section 2 describes.
  const file = ENGINE === 'gpu' ? './gpuworker.js' : './simworker.js';
  const worker = ENGINE === 'remote'
    ? new RemoteEngine(Q.get('host') || 'ws://localhost:8801')
    : new Worker(new URL(file + '?p=' + PROTOCOL, import.meta.url), { type:'module' });
  liveWorker = worker;
  // From here the run is doing rather than being configured, so the lines go to the events pane.
  logDest = logEl;
  // The building UI inspects neurons through the engine that owns them.
  // When the playground is driving this run it stopped its own engine, so without this its watch and query messages go nowhere and clicking a neuron waits forever on a reply that cannot come.
  // Latency is asked for once, for every engine and both run modes: the readout needs it and an engine that cannot supply it simply omits the block, which the reader detects by length.
  worker.postMessage({ cmd:'sendT', on:true });
  globalThis.__npEngine = worker;
  // debug bridge: the network under training and its latest weight readback
  globalThis.__npNet = net;
  if(globalThis.__npOnEngine) globalThis.__npOnEngine(worker);
  const engineInputs = (net.inputMaps || []).map(m =>
    ({ chStart:m.chStart, chIdx:m.chIdx, chW:m.chW, amp:m.amp }));
  // tune replaces the whole configuration rather than merging, so the settings are kept here and resent in full whenever one of them changes
  const engineCfg = { bias:net.bias, ...engineConfig(net),
    plast:(PLAST && NODE_PLAST && !WARMUP_MS) ? 1 : 0 };
  // Watched populations are thinned here, before the init message, because the like-to-like measure needs the synapse arrays and the transfer below detaches them.
  // Capturing the within-population link lists now costs one pass over the CSR and a few thousand entries per population; reading net.post after the transfer would silently yield nothing.
  // The per-population sample is the resolution of the whole measurement, so it is raised until the conjunctive sets are big enough to be quiet.
  // Cost is linear here and in the trial buffers, which the checkpoint clears.
  // From the analysis node when the graph has one, otherwise the values it defaults to.
  // The query parameter still wins, for a sweep driven from the address bar.
  const AN = Object.assign({ cap:8192, topQ:0.1, margin:0.15, minCells:4,
    minTrials:2, nPerm:40, doDecode:true, doBind:true, doTiming:true, doLike:true,
    exportTrials:false },
    net.analysis || {});
  // binding, timing and like-to-like run inside the decode pass, so with decode off they are absent whatever their own switches say; said here rather than found missing from every row
  if(!AN.doDecode && (AN.doBind || AN.doTiming || AN.doLike))
    log('analysis: decode is off, so binding, timing and like-to-like are not measured either');
  const CAP = Math.max(64, parseInt(Q.get('cap') || '', 10) || AN.cap);
  const thin = idx => {
    if(idx.length <= CAP) return Int32Array.from(idx);
    const step = idx.length/CAP, sub = new Int32Array(CAP);
    for(let k=0;k<CAP;k++) sub[k] = idx[Math.floor(k*step)];
    return sub;
  };
  // The probes that opted in are recorded for analysis; a graph with no probe opted in measures them all, since that is what a scene without the flag expects.
  const optedIn = (net.probes || []).filter(p => p.record);
  const pool = optedIn.length ? optedIn : (net.probes || []);
  const watched = pool.filter(p => p.idx && p.idx.length)
    .map(p => { const idx = thin(p.idx);
      return { label:p.label, idx, links:popLinks(idx, net.preStart, net.post) }; });
  // The init below transfers the synapse arrays to the engine, which detaches them here.
  // The save path still needs the plasticity mask, to know which weights a plastic checkpoint should carry, so it is copied first for the same reason w0 above keeps a baseline copy of the weights.
  // One byte a synapse, a quarter of what w0 already costs.
  const pmask0 = net.pmask ? net.pmask.slice() : null;
  const runSeed = SEED_Q !== null ? SEED_Q : net.seed;
  if(SEED_Q !== null) log('engine seed ' + runSeed + ' (replicate sweep)');
  // Which of the two drive sets each end of a synapse belongs to, resolved now because the init message below transfers the synapse arrays to the engine and detaches them here.
  //
  // A decoder answers whether one channel's response resembles the other's.
  // This asks the synapses instead, and needs no decoder to be able to read anything: did the connections running between the two codes grow more than the connections running within them?
  // The two within-code pairs are a control that sits inside every run.
  const driveSets = (net.inputMaps || []).map(im => {
    const f = new Uint8Array(net.count);
    for(let e=0;e<im.chIdx.length;e++) f[im.chIdx[e]] = 1;
    let n = 0; for(let i=0;i<f.length;i++) n += f[i];
    return { id:im.id, flag:f, n, sense:im.sense };
  });
  // 0 AA, 1 AB, 2 BA, 3 BB, 255 not in the comparison.
  // Cells touched by both channels are excluded from every pair: a synapse between two of them belongs to no direction, and counting it in both would put the overlap on both sides of the comparison.
  let pairClass = null;
  if(driveSets.length >= 2){
    const pc = pairClasses(net, driveSets[0].flag, driveSets[1].flag);
    if(pc){
      pairClass = pc.cls;
      const [aa, ab, ba, bb] = pc.counts;
      log(`drive sets: A ${driveSets[0].n} cells, B ${driveSets[1].n} cells; ` +
        `plastic synapses A>A ${aa}, A>B ${ab}, B>A ${ba}, B>B ${bb}. ` +
        `Weight between and within the codes is reported at each weight readback.`);
    }
  }
  // The same two sets by sense, and the synapses running between them with their two ends, for the cross-modal learning measures: captured here for the same reason as the pathway labels above, before the init message takes the synapse arrays away.
  const sightSet = driveSets.find(d => d.sense === 'sight');
  const soundSet = driveSets.find(d => d.sense === 'sound');
  let betweenSyn = null;
  if(sightSet && soundSet){
    const xp = pairClasses(net, sightSet.flag, soundSet.flag);
    if(xp) betweenSyn = betweenSynapses(net, xp.cls);
  }

  if(!ATTACH) worker.postMessage({ cmd:'init', count:net.count, ntype:net.ntype,
    inputs:engineInputs, pmask:net.pmask, ...engineCfg,
    seed:runSeed, ...initState(runSeed, net.ntype, NEURON_TYPES),
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay,
    types:engineTypes() },
    // Transfer rather than copy past the size the page itself transfers at: at full density the synapse arrays are gigabytes, and structured cloning them runs the tab out of memory.
    // Below it, copy, so the arrays are still here when the run ends and the page can put the trained network back on screen without rewiring. w0 above already holds the baseline copy the metrics need.
    (net.synCount*13 + net.count*25 > 1.6e9)
      ? [net.preStart.buffer, net.post.buffer, net.w.buffer, net.delay.buffer,
          ...(net.pmask ? [net.pmask.buffer] : [])]
      : []);
  net.pmask = pmask0;                  // put the readable copy back in its place
  // the computed network is exposed for console inspection: anyone with the page can interrogate exactly what was computed, per the replicability principle
  window.__net = net;
  const io = new IORuntime(msg => log('io: ' + msg));
  io.attach(worker, net.inputMaps, runSeed);
  setupInputs(net, io);
  io.last = -1e9;                           // sim-time clock starts at zero

  // Design rule: a brain file carries the curriculum position. scheduleAt is a pure function of simulated time, so the clock is the position; the rest is recorded so a resume can be checked against the lesson stream it claims to continue.
  const curriculumPosition = t => {
    const cur = ed.nodes.filter(n => n.type === 'curriculum');
    if(!cur.length) return null;
    // a curriculum that follows another runs on that node's schedule, so the record carries the schedule it actually runs on
    const leadOf = n => {
      const seen = new Set();
      while(n.inputs && n.inputs[0] && !seen.has(n.id)){
        seen.add(n.id);
        const up = ed.nodes.find(m => m.id === n.inputs[0].id);
        if(!up || up.type !== 'curriculum') break;
        n = up;
      }
      return n;
    };
    return { simMs:t, nodes:cur.map(n => { const L = leadOf(n);
      return { id:n.id, set:L.params.set|0, order:L.params.order|0,
        onMs:L.params.onMs, offMs:L.params.offMs, seed:L.params.seed|0 }; }) };
  };

  const probes = (net.probes || []).map(p => ({ label:p.label, idx:p.idx, spikes:0 }));

  // Observer-side readout, one recorder per probe population rather than one over all of them together.
  // The union hides the answer: retina and the thalamic relays are driven by the stimulus by construction, so they decode the item whether or not anything has been learned, and pooling them with cortex buries any cortical structure under a signal that cannot change.
  // Learning, if it happens, happens in L4 and L2/3, and those have to be scored on their own.
  const csig = (net.inputMaps || []).map(m => m.signal)
    .find(g => g && g.src === 'curriculum');
  const itemList = csig ? (SETS[csig.set] || SETS[0]).items : [];
  // Canvas-drawn stimuli (glyphs, words, objects) become an atlas of canonical rasters sampled through the pose transform, so the same rendering runs in the page and on the engine host, which has no canvas.
  // Spots are already analytic and need none of this.
  let atlas = null;
  if(csig && itemList.length && (SETS[csig.set] || SETS[0]).kind !== 'spot'){
    atlas = buildAtlas(itemList, (SETS[csig.set] || SETS[0]).kind);
    setGlyphAtlas(atlas);
  }
  const itemLabel = new Map(itemList.map((it, i) => [it, i]));
  let recs = [];
  if(csig && itemList.length){
    recs = watched.map(p => ({ label:p.label, links:p.links,
      rec:new ItemRecorder(p.idx, itemLabel, STEPS) }));
    if(recs.length) log(`readout: ${recs.length} populations ` +
      `(${recs.map(r => `${r.label} ${r.rec.nNeurons}`).join(', ')}), ` +
      `${itemList.length} items` +
      (csig.probeEvery ? `, unimodal probe every ${csig.probeEvery}` : ', no unimodal probes'));
    // A blocked run is a different experiment from an unblocked one, and the log is what a reader has months later when the scene has moved on.
    if(csig.phasesText)
      log(`training blocks: ${csig.phasesText} (simulated minutes from the end ` +
        `of the opening sweep; probes rotate through both, A alone and B alone ` +
        `in every block)`);
  }

  // The conjunctive sets, per population, fixed at their first measurable window.
  // This is the run's calibration: which cells were conjunctive before learning had time to move them.
  // Everything the binding measure says is relative to it, so it is taken once and never rebuilt.
  const conjCal = new Map();
  const conjWarned = new Set();   // say once per population, not every checkpoint
  // The cross-modal learning measures' calibration, per population: each item's cells in each sense's drive set and the evoked transfer before anything learned.
  // Taken at the same moment as the conjunctive sets and for the same reason, and pooled trials for the end-of-run figure.
  const xferCal = new Map();
  const xferPools = new Map();
  let calAtMs = 0;                // sim time the host flips plasticity, if pumped
  let sweepActive = false;        // inside any sweep window, opening or repeat
  let reSweepRan = 0;             // how many repeat sweeps have been measured
  let wasReSweep = false;         // the window that is closing was a repeat
  // warmup means "the switch still needs scheduling" and a host-pumped run clears it at startup, so it cannot be read as "not learning yet". plastOn tracks the engine's actual state, and the calibration warning depends on it.
  // It is set once HOLD_MS exists; reading HOLD_MS here is a temporal dead zone throw.
  let plastOn = false;
  let calWindowShut = false;      // the opening sweep has been measured
  // Build the conjunctive sets from what the sweep recorded, without clearing the trials: the readout keeps accumulating toward its first decode, and this only reads.
  const calibrateNow = () => {
    for(const r of recs){
      r.rec.flush();
      measure(r.rec, r.links, r.label, true);
    }
    calWindowShut = true;
    saveCalibration();
  };
  // The baseline has to outlive the tab.
  // Held only in memory, a reload or a resume rebuilds the conjunctive sets from a network that has already learned, which is a different question and not comparable with what came before it.
  // Written once, with the cell indices themselves, so the run can be re-analyzed later and so the file's existence is the proof that the sweep did its job before sixteen hours were committed to it.
  let calSaved = false;
  // What the conjunction looks like now, against what it looked like before anything learned.
  // Drift is the overlap of the two sets of cells, per item: 1 means the same cells are still the conjunctive ones, 0 means the set has moved entirely.
  // Reported beside the binding score so a change in binding can be told from a change in what is being scored.
  const reSweepReport = () => {
    reSweepRan++;
    for(const r of recs){
      const cal = conjCal.get(r.label);
      if(!cal) continue;
      r.rec.flush();
      const nI = itemList.length, nN = r.rec.nNeurons;
      const cc = r.rec.centroids(0, nI), c1 = r.rec.centroids(1, nI), c2 = r.rec.centroids(2, nI);
      let full = 0;
      for(let i=0;i<nI;i++) if(cc.cnt[i] > 0 && c1.cnt[i] > 0 && c2.cnt[i] > 0) full++;
      if(full < nI){
        log(`re-sweep ${reSweepRan} on ${r.label}: only ${full} of ${nI} items covered, skipped`);
        continue;
      }
      // with the node's own thresholds, as the calibration it is compared with was built; the defaults read as drift wherever the node set others
      const now = conjunctiveSets(cc.cent, c1.cent, c2.cent, nI, nN, AN.topQ, AN.margin);
      let jac = 0, pairs = 0, cellsNow = 0;
      for(let i=0;i<nI;i++){
        const a = new Set(cal.sets[i] || []), b = new Set(now[i] || []);
        cellsNow += b.size;
        if(!a.size && !b.size) continue;
        let inter = 0; for(const x of b) if(a.has(x)) inter++;
        jac += inter/(a.size + b.size - inter); pairs++;
      }
      // recall from each sense alone, as the checkpoint reports it
      const bind = sets => {
        const v = conjunctiveRecall(sets, c1.cent, c1.cnt, nI, nN), a = conjunctiveRecall(sets, c2.cent, c2.cnt, nI, nN);
        return `sight ${v ? v.bind : 'n/a'} sound ${a ? a.bind : 'n/a'}`;
      };
      log(`re-sweep ${reSweepRan} on ${r.label} at sim ${(simMs/60000).toFixed(0)} min: ` +
        `drift ${pairs ? (jac/pairs).toFixed(3) : 'n/a'} overlap with calibration, ` +
        `cells ${cal.mean} then / ${(cellsNow/nI).toFixed(1)} now, ` +
        `bind on original sets ${bind(cal.sets)}, on current ${bind(now)}`);
    }
  };
  const saveCalibration = async () => {
    if(calSaved || !statusDir || !conjCal.size) return;
    calSaved = true;
    const pops = {};
    for(const [label, c] of conjCal)
      pops[label] = { atSimMin:+(c.atMs/60000).toFixed(2), items:c.sets.length,
        nonEmpty:c.sized, meanCells:c.mean,
        // probe-local indices, the same space the readout counts in
        sets:c.sets.map(x => Array.from(x)) };
    try {
      await writeFile(statusDir, P.RUN_CALIBRATION, JSON.stringify({
        run:RUN_TAG, at:new Date().toISOString(), simMs,
        items:itemList.length, populations:Object.keys(pops).length,
        note:'conjunctive sets from the opening sweep, before plasticity; ' +
          'held fixed for the run and not comparable across runs',
        pops }, null, 1));
      log(`calibration saved: ${P.RUN_CALIBRATION}, ` +
        `${Object.keys(pops).length} populations`);
    } catch(e){ log('calibration save failed: ' + e.message); }
  };

  // Filled during a checkpoint's measure pass when the analysis node asks for it, drained by the writer below.
  // Cleared each checkpoint, so the file always describes the window it was written for.
  const trialExport = new Map();
  // What the trial condition codes mean, taken from the curriculum node when it names them so an export says "sight" rather than 1.
  const CONDS = (() => {
    // from the computed signal, which carries the names of the schedule every stream runs on, whichever curriculum node holds it
    const named = csig && Array.isArray(csig.condNames) ? csig.condNames : null;
    return named && named.length >= 3 ? named : ['both', 'sight', 'sound'];
  })();

  // Turns one population's accumulated trials into numbers, then clears them, so each row describes the window it belongs to rather than everything since the run began.
  const pools = new Map();                 // popKey -> pooled probe trials
  const measure = (rec, links, popKey, calOnly) => {
    rec.flush();
    const nI = itemList.length, nN = rec.nNeurons;
    const both = rec.pack(0);
    const out = { neurons:nN, trials:both.n, probeTrials:rec.trials.length - both.n };
    // How many single-sense probes evoke nothing at all in this population.
    // A trial with no spikes carries no pattern to compare, so the decoders skip it and every accuracy below is a fraction of the trials that did respond.
    // That denominator is a result in itself: a third of sound-alone probes leave the association area silent (measured 2026-09-03), and a rule change that moves the silent fraction moves what the accuracy is an accuracy of.
    {
      const cnt = [0, 0, 0], sil = [0, 0, 0];
      for(const tr of rec.trials){
        const c = tr.cond | 0;
        if(c < 0 || c > 2) continue;
        cnt[c]++;
        let s = 0;
        for(let j=0;j<nN;j++) s += tr.vec[j];
        if(s <= 0) sil[c]++;
      }
      out.silentFrac = { both:cnt[0] ? +(sil[0]/cnt[0]).toFixed(4) : null,
        vis:cnt[1] ? +(sil[1]/cnt[1]).toFixed(4) : null,
        aud:cnt[2] ? +(sil[2]/cnt[2]).toFixed(4) : null,
        trials:[cnt[0], cnt[1], cnt[2]] };
    }
    // Decoding needs two bimodal trials per item.
    // A checkpoint window that cannot hold that many measures nothing, so trials carry over between windows: with 26 letters at one second on and two off, a two-minute checkpoint sees about thirty bimodal presentations against the fifty-two needed, and a readout that started from nothing each window could never appear at any point in a sixteen-hour run.
    // The first readout arrives as soon as the statistics allow at whatever cadence and item count the run happens to use.

    // Calibration runs before the decode gate below, because it needs coverage rather than trial count: the sweep gives every item in all three conditions but fewer bimodal trials than decoding wants, and gating it on that count meant the sets were never built during the one window where nothing had learned yet.
    // Coverage is reached part way through the sweep, because the first pass already shows every item in every condition.
    // Calibrating there would throw away the passes that follow and build the baseline from one or two trials per item, which is what the extra passes exist to avoid.
    // So while a sweep is configured, wait for it to finish.
    const sweepPending = calMs > 0 && simMs < resumeMs + HOLD_MS;
    // The baseline is taken once, when the opening sweep ends.
    // A population that does not qualify then does not get one later: retina and cochlea cannot have a conjunction, so they fall short at the sweep and would otherwise calibrate at the next checkpoint on noise, with plasticity already on.
    // Having no baseline is the correct answer for them.
    if(AN.doBind && popKey && !conjCal.has(popKey) && !sweepPending && !calWindowShut){
      const cc = rec.centroids(0, nI), c1 = rec.centroids(1, nI), c2 = rec.centroids(2, nI);
      let full = 0;
      for(let i=0;i<nI;i++)
        if(cc.cnt[i] > 0 && c1.cnt[i] > 0 && c2.cnt[i] > 0) full++;
      // Every item, not a couple of them.
      // The sets are the baseline the whole run is scored against, so calibrating on partial coverage would measure binding for some letters and not others, and the missing ones would read as an absence of learning.
      if(full < nI && !conjWarned.has(popKey)){
        conjWarned.add(popKey);
        // Which condition is missing matters: no unimodal trials at all means the schedule never produced them or the recorder never saw them, and a few missing items means the sweep was simply cut short.
        const tot = c => { let n = 0; for(let i=0;i<nI;i++) n += c.cnt[i]; return n; };
        const cov = c => { let n = 0; for(let i=0;i<nI;i++) if(c.cnt[i] > 0) n++; return n; };
        log(`conjunction on ${popKey}: ${full} of ${nI} items seen in all three ` +
          `conditions, waiting. trials both ${tot(cc)}/${cov(cc)} items, ` +
          `sight ${tot(c1)}/${cov(c1)}, sound ${tot(c2)}/${cov(c2)}; ` +
          `sweep ${calMs ? (calMs/60000).toFixed(1) + ' min' : 'off'} at sim ` +
          `${(simMs/60000).toFixed(1)}`);
      }
      if(full === nI){
        const sets = conjunctiveSets(cc.cent, c1.cent, c2.cent, nI, nN, AN.topQ, AN.margin);
        const sized = sets.filter(x => x.length).length;
        if(sized >= 2){
          conjCal.set(popKey, { sets, atMs:simMs, sized,
            mean:+(sets.reduce((a, x) => a + x.length, 0)/nI).toFixed(1) });
          // Handed to the page as well, so the viewer can draw them, mapped to network indices here because the sets index this recorder's own sample of the population and nothing outside knows that sampling.
          if(globalThis.__npConjSets)
            globalThis.__npConjSets(popKey, {
              sets:sets.map(set => Array.from(set || [], k => rec.idx[k])),
              items:itemList.slice(), at:simMs });
          log(`conjunction calibrated on ${popKey}: all ${nI} items, ${sized} with ` +
            `a non-empty set, ${conjCal.get(popKey).mean} cells each, at sim ` +
            `${(simMs/60000).toFixed(1)} min` +
            (plastOn ? ' (WARNING: plasticity was already on)' : ''));
        } else {
          log(`conjunction on ${popKey}: every item covered but only ${sized} sets ` +
            `are non-empty; the joint response is not separable from the ` +
            `single-modality ones in this population`);
        }
      }
    }
    // the cross-modal learning measures' calibration, here because a calibration pass returns just below
    if(sightSet && soundSet && popKey){
      if(!xferCal.has(popKey) && !sweepPending && !calWindowShut){
        const c1 = rec.centroids(1, nI), c2 = rec.centroids(2, nI);
        let full = 0;
        for(let i=0;i<nI;i++) if(c1.cnt[i] > 0 && c2.cnt[i] > 0) full++;
        if(full === nI){
          const inA = Uint8Array.from(rec.idx, g => sightSet.flag[g] && !soundSet.flag[g] ? 1 : 0);
          const inB = Uint8Array.from(rec.idx, g => soundSet.flag[g] && !sightSet.flag[g] ? 1 : 0);
          const SA = itemCellSets(c1.cent, c1.cnt, nI, nN, inA, AN.topQ);
          const SB = itemCellSets(c2.cent, c2.cnt, nI, nN, inB, AN.topQ);
          if(SA && SB){
            const baseAB = evokedTransfer(rec.pack(1), SB, nI, nN);
            const baseBA = evokedTransfer(rec.pack(2), SA, nI, nN);
            xferCal.set(popKey, { SA, SB, baseAB, baseBA, atMs:simMs,
              SAnet:SA.map(set => set.map(k => rec.idx[k])),
              SBnet:SB.map(set => set.map(k => rec.idx[k])) });
            const size = sets => (sets.reduce((a, x) => a + x.length, 0)/nI).toFixed(1);
            log(`transfer calibrated on ${popKey}: ${size(SA)} sight cells and ${size(SB)} ` +
              `sound cells per item; evoked own minus other before learning, ` +
              `sight to sound ${baseAB ? baseAB.d : '-'}, sound to sight ${baseBA ? baseBA.d : '-'}` +
              (plastOn ? ' (WARNING: plasticity was already on)' : ''));
          }
        }
      }
    }
    // A calibration pass only builds the sets.
    // It must not reset the recorder: the readout is still accumulating toward its first decode and those trials are the same ones.
    if(calOnly) return out;
    if(both.n < nI*2 && rec.trials.length < nI*8){
      out.waitingFor = nI*2 - both.n;
      return out;                          // keep accumulating, do not reset
    }
    if(AN.doDecode && both.n >= nI*AN.minTrials){
      const d = decodeAccuracy(both.trials, both.labels, nI, nN);
      out.decode = +d.acc.toFixed(3); out.decodeN = d.n;
      if(globalThis.__npMatrix && d.confusion && popKey)
        globalThis.__npMatrix(popKey + ':confusion', { rows:nI, cols:nI, values:d.confusion,
          items:itemList, simMin:+(simMs/60000).toFixed(2) });
      const c = rec.centroids(0, nI);
      // the layer-transformation measures: how many cells are tuned, how reliably they answer, and how correlated the active population is
      const rp = responseProfile(both.trials, both.labels, nI, nN);
      if(rp){
        out.responsiveFrac = +rp.responsiveFrac.toFixed(3);
        out.reliability = +rp.reliability.toFixed(3);
      }
      out.ensembleCorr = +ensembleCorrelation(both.trials, both.labels, nN).toFixed(3);
      // does connectivity within this population track tuning similarity, beyond what distance explains (MICrONS, Ding et al. 2025)
      const l2l = AN.doLike ? likeToLike(both.trials, both.labels, nI, nN, net.pos,
        rec.idx, links, { w:lastW || w0, w0 }) : null;
      if(l2l){
        out.likeDelta = +l2l.deltaCorr.toFixed(3);
        out.likeW = +l2l.deltaW.toFixed(3);
        out.likePairs = l2l.pairs;
      }
      const dsh = decodeShape(both.trials, both.labels, nI, nN);
      out.decodeShape = +dsh.acc.toFixed(3);
      const dt = decodeTopAccuracy(both.trials, both.labels, nI, nN);
      out.decodeTop = +dt.acc.toFixed(3); out.decodeTopN = dt.used;
      const sel = selectivityProfile(c.cent, nI, nN);
      out.selectivity = +sel.mean.toFixed(3);
      out.activeFrac = +(sel.active/nN).toFixed(3);
      // Trial-to-trial variability in the form the field reports it and Elephant computes it, per cell across repeated presentations, 1 for a Poisson process.
      // Not the number audit.html calls Fano, which is a synchrony measure over time windows; see src/spikestats.js.
      const ff = fanoPerCell(both.trials, both.labels, nI, nN);
      if(Number.isFinite(ff.mean)){
        out.fano = +ff.mean.toFixed(3);
        out.fanoCells = ff.cells;
      }
      // The item by cell response matrix every scalar above was reduced from, handed to the building UI so a chart can draw it and the arithmetic can be checked.
      // Capped at a readable size: a heatmap of eight thousand columns is a smear, and the first few hundred cells are a fair sample because the recorder's own index is already a spread across the population rather than the first N of it.
      if(globalThis.__npMatrix){
        const MAXC = 512;
        const step = Math.max(1, Math.floor(nN/MAXC));
        const cols = Math.min(nN, Math.floor(nN/step));
        const mat = new Float32Array(nI*cols);
        for(let i = 0; i < nI; i++)
          for(let k = 0; k < cols; k++) mat[i*cols + k] = c.cent[i*nN + k*step];
        globalThis.__npMatrix(popKey, { rows:nI, cols, values:mat,
          items:itemList, simMin:+(simMs/60000).toFixed(2) });
      }
      // The whole trial matrix, all conditions, since the unimodal probes are what the binding measures are built from; without it every question nobody thought to ask in advance is unanswerable.
      if(AN.exportTrials) trialExport.set(popKey, {
        neurons:nN, idx:Array.from(rec.idx),
        trials:rec.trials.map(t => ({ cond:t.cond, label:t.label, vec:t.vec })),
      });
      // Completion against the full-modality centroids, each direction: 1 = vision alone, 2 = sound alone.
      // Read this with care in a feedforward patch.
      // One modality drives a subset of the cells the pair drives, and a subset resembles its own joint pattern more than another item's whatever the weights are doing, so a high score here is partly structural.
      const cp1 = rec.centroids(1, nI), cp2 = rec.centroids(2, nI);
      for(const [cp, key] of [[cp1, 'visualOnly'], [cp2, 'audioOnly']]){
        let have = 0;
        for(let i=0;i<nI;i++) if(cp.cnt[i] > 0 && c.cnt[i] > 0) have++;
        if(have >= 2){
          // The direct question: on every presentation where both senses showed this item, which cells fired?
          // That set is the item's template.
          // Does one sense alone reproduce it? rank1 answers in steps of 1/nItems and is partly structural, since a subset of the joint pattern resembles its own joint pattern whatever the weights do.
          // The centered cosine removes the common component every centroid shares and asks only whether the item-specific residual matches, which is the part learning has to build.
          // (full, partial): the joint template is the reference and the one-sense pattern is scored against every item's template.
          const cs = completionScore(c.cent, cp.cent, nI, nN);
          out[key] = { rank1:+cs.rank1.toFixed(3), margin:+cs.margin.toFixed(3),
            ownC:+cs.ownC.toFixed(3), otherC:+cs.otherC.toFixed(3),
            rank1C:+cs.rank1C.toFixed(3), n:cs.n };
        }
      }
      // Cross-modal transfer, and the one that is not structural.
      // Does the pattern sound alone produces resemble the pattern this item's sight alone produces, more than it resembles any other item's?
      // The two are driven by disjoint afferents, so before anything is learned there is no reason for them to match beyond chance, and any match has to be carried by what training built between them.
      const cal = popKey && conjCal.get(popKey);
      if(cal){
        // Does one sense alone now reach this item's conjunction, and its own rather than another item's?
        // Own minus other, in rank units, both measured now, so drift over sixteen sim-hours cancels.
        const bv = conjunctiveRecall(cal.sets, cp1.cent, cp1.cnt, nI, nN);
        const ba = conjunctiveRecall(cal.sets, cp2.cent, cp2.cnt, nI, nN);
        // The timing question, which counts cannot answer: does one sense wake the conjunctive cells in the order the pair wakes them?
        const tv = AN.doTiming ? latencyAgreement(cal.sets, rec.trials.filter(t => t.cond === 0),
          rec.trials.filter(t => t.cond === 1), nI, nN, AN.minCells) : null;
        const ta = AN.doTiming ? latencyAgreement(cal.sets, rec.trials.filter(t => t.cond === 0),
          rec.trials.filter(t => t.cond === 2), nI, nN, AN.minCells) : null;
        if(tv || ta) out.timing = { vis:tv ? tv.r : null, aud:ta ? ta.r : null,
          cells:(tv || ta).cells, items:(tv || ta).items };
        if(bv || ba) out.bind = {
          vis: bv ? bv.bind : null, aud: ba ? ba.bind : null,
          visOwn: bv ? bv.own : null, audOwn: ba ? ba.own : null,
          cells: cal.mean, calMin:+(cal.atMs/60000).toFixed(1) };
      }
      let paired = 0;
      for(let i=0;i<nI;i++) if(cp1.cnt[i] > 0 && cp2.cnt[i] > 0) paired++;
      if(paired >= 2){
        // sight is the reference and sound the probe, the direction the comment above and transferAccuracy below both ask
        const x = completionScore(cp1.cent, cp2.cent, nI, nN);
        // trial-resolved: every sound-alone presentation classified against the sight-alone centroids, hundreds of measurements instead of twelve binary item verdicts
        const tx = transferAccuracy(cp1.cent, cp1.cnt, rec.pack(2), nI, nN);
        // Noise ceiling: the same modality against its own other half.
        // If this is near zero the unimodal centroids carry no reproducible item pattern at all and the cross-modal number above is measuring noise against noise rather than two unrelated codes.
        // Does a single unimodal presentation move toward its item's joint template over the course of the trial?
        // A slow recurrent completion shows here while summed-count transfer stays at chance.
        const trj1 = completionTrajectory(c.cent, rec.trials.filter(t => t.cond === 1), nI, nN);
        const trj2 = completionTrajectory(c.cent, rec.trials.filter(t => t.cond === 2), nI, nN);
        if(trj1) out.driftVis = { gap:+trj1.gap.toFixed(4), se:+trj1.se.toFixed(4), n:trj1.n };
        if(trj2) out.driftAud = { gap:+trj2.gap.toFixed(4), se:+trj2.se.toFixed(4), n:trj2.n };
        const sp0 = rec.centroidsSplit(0, nI);
        const rel0 = completionScore(sp0.a, sp0.b, nI, nN);
        out.templateRel = +rel0.ownC.toFixed(3);
        const sp1 = rec.centroidsSplit(1, nI), sp2 = rec.centroidsSplit(2, nI);
        const rel1 = completionScore(sp1.a, sp1.b, nI, nN);
        const rel2 = completionScore(sp2.a, sp2.b, nI, nN);
        out.crossModal = { rank1:+x.rank1.toFixed(3), margin:+x.margin.toFixed(3),
          relVis:+rel1.ownC.toFixed(3), relAud:+rel2.ownC.toFixed(3),
          own:+x.own.toFixed(3), other:+x.other.toFixed(3),
          ownC:+x.ownC.toFixed(3), otherC:+x.otherC.toFixed(3),
          rank1C:+x.rank1C.toFixed(3),
          perItem:(x.perItem || []).map(q => `${q.own}/${q.best}`).join(' '),
          n:x.n, trialAcc:+tx.acc.toFixed(3), trialN:tx.n,
          trialSe:+tx.se.toFixed(3) };
        // The field's forms of the same questions (ANALYSIS.md section 12): cross-classification both ways with a permutation null and the within-modality accuracies beside it, representational similarity between the modalities and across them, and population sparseness.
        const pk1 = rec.pack(1), pk2 = rec.pack(2);
        const xd = crossDecode(pk1, pk2, nI, nN, AN.nPerm, 1);
        out.crossDecode = xd;
        // The same test pooled over every window from POOL_FROM_MS on: a thirty-minute window holds about forty probe trials per direction and its label-shuffle p floors near 0.02, so the number a reader can use is correct-over-total across the run against a binomial at chance.
        // The windows before the start are the plasticity transient and are left out.
        if(simMs >= POOL_FROM_MS){
          let pool = pools.get(popKey);
          if(!pool) pools.set(popKey, pool = { a:[], b:[] });
          for(let t=0;t<pk1.n;t++) pool.a.push({ vec:pk1.trials.slice(t*nN, (t+1)*nN), label:pk1.labels[t] });
          for(let t=0;t<pk2.n;t++) pool.b.push({ vec:pk2.trials.slice(t*nN, (t+1)*nN), label:pk2.labels[t] });
          const packOf = arr => { const trials = new Float32Array(arr.length*nN), labels = new Int32Array(arr.length);
            arr.forEach((t, i) => { trials.set(t.vec, i*nN); labels[i] = t.label; }); return { trials, labels, n:arr.length }; };
          const px = crossDecode(packOf(pool.a), packOf(pool.b), nI, nN, 0, 1);
          const tail = (k, n, q) => { // P(X >= k), X ~ Binomial(n, q)
            let pr = Math.pow(1 - q, n), sum = 0;
            for(let i=0;i<=n;i++){ if(i >= k) sum += pr; pr *= (n - i)/(i + 1) * q/(1 - q); }
            return Math.min(1, sum); };
          const side = d => d && d.n ? { acc:d.acc, n:d.n, k:Math.round(d.acc*d.n),
            p:+tail(Math.round(d.acc*d.n), d.n, 1/Math.max(1, nI)).toFixed(4),
            // where the wrong answers went: an accuracy under chance means one thing if each item is sent consistently to one other item and another if the errors scatter
            shape:d.shape || null } : null;
          out.crossDecodePooled = { fromMin:POOL_FROM_MS/60000, ab:side(px.ab), ba:side(px.ba) };
        }
        const mask = Array.from({ length:nI }, (_, i) => cp1.cnt[i] > 0 && cp2.cnt[i] > 0 && c.cnt[i] > 0);
        const rV = rdm(cp1.cent, nI, nN, mask), rA = rdm(cp2.cent, nI, nN, mask), rB = rdm(c.cent, nI, nN, mask);
        const xr = crossRdm(cp1.cent, cp2.cent, nI, nN, mask);
        out.rsa = { visAud:rdmCorrelation(rV, rA), visBoth:rdmCorrelation(rV, rB),
          audBoth:rdmCorrelation(rA, rB),
          cross: xr ? { dominance:xr.dominance, onDiag:xr.onDiag, offDiag:xr.offDiag, items:xr.k } : null };
        out.popSparseness = populationSparseness(c.cent, c.cnt, nI, nN).mean;
        // the similarity matrices themselves, for the chart node
        if(globalThis.__npMatrix && popKey){
          const at = +(simMs/60000).toFixed(2);
          const names = mask.map((on, i) => on ? itemList[i] : null).filter(Boolean);
          for(const [k, R] of [['both', rB], ['vis', rV], ['aud', rA], ['cross', xr]])
            if(R && R.k) globalThis.__npMatrix(popKey + ':rdm:' + k,
              { rows:R.k, cols:R.k, values:R.d, items:names, simMin:at });
        }
        // Single-neuron enhancement and additivity on the conjunctive cells, and the weight block structure those cells carry.
        if(cal){
          // where the conjunctive cells sit around the vertical axis
          if(net.pos) out.sector = sectorHistogram(cal.sets, rec.idx, net.pos, 12);
          const msi = multisensoryIndex(c.cent, cp1.cent, cp2.cent, cal.sets, nI, nN);
          if(msi.cells) out.msi = { cre:msi.cre, superadditive:msi.superadditive,
            additive:msi.additive, subadditive:msi.subadditive, cells:msi.cells };
          if(links && (lastW || w0)){
            const bs = blockStructure(cal.sets, links, lastW || w0, nN);
            if(bs){
              out.block = { within:bs.within, between:bs.between, ratio:bs.ratio,
                nWithin:bs.nWithin, nBetween:bs.nBetween };
              if(globalThis.__npMatrix && popKey)
                globalThis.__npMatrix(popKey + ':wblock', { rows:bs.sets, cols:bs.sets,
                  values:bs.matrix, items:itemList, simMin:+(simMs/60000).toFixed(2) });
            }
          }
        }
      }
    }
    // Never mid-sweep.
    // The sweep's trials are the conjunctive baseline and they accumulate across all three passes, so a checkpoint landing inside it would reset the recorder and throw the earlier passes away, leaving the calibration the tail after the checkpoint rather than anything the sweep had presented.
    // Cross-modal learning, measured where the input does not reach.
    // The decoders above compare whole response patterns, which on the weaves are set by which cells each input drives directly, so they read the geometry of the two codes with plasticity off as with it on.
    // Here: on a trial presenting one sense alone, the response in the other sense's cells for the same item against the other items', where only the recurrent synapses can carry anything, as its change from the same number at the calibration sweep.
    // The weight form of the same question is itemCoupling at each readback.
    if(sightSet && soundSet && popKey){
      const xc = xferCal.get(popKey);
      if(xc && !calOnly){
        const side = (now, base) => now ? { ...now, base:base ? base.d : null,
          delta:base ? +(now.d - base.d).toFixed(4) : null } : null;
        const p1 = rec.pack(1), p2 = rec.pack(2);
        out.xfer = { ab:side(evokedTransfer(p1, xc.SB, nI, nN), xc.baseAB),
          ba:side(evokedTransfer(p2, xc.SA, nI, nN), xc.baseBA),
          calMin:+(xc.atMs/60000).toFixed(1) };
        if(simMs >= POOL_FROM_MS){
          let pool = xferPools.get(popKey);
          if(!pool) xferPools.set(popKey, pool = { a:[], b:[] });
          for(let t=0;t<p1.n;t++) pool.a.push({ vec:p1.trials.slice(t*nN, (t+1)*nN), label:p1.labels[t] });
          for(let t=0;t<p2.n;t++) pool.b.push({ vec:p2.trials.slice(t*nN, (t+1)*nN), label:p2.labels[t] });
          const packOf = arr => { const trials = new Float32Array(arr.length*nN), labels = new Int32Array(arr.length);
            arr.forEach((t, i) => { trials.set(t.vec, i*nN); labels[i] = t.label; }); return { trials, labels, n:arr.length }; };
          out.xferPooled = { fromMin:POOL_FROM_MS/60000,
            ab:side(evokedTransfer(packOf(pool.a), xc.SB, nI, nN), xc.baseAB),
            ba:side(evokedTransfer(packOf(pool.b), xc.SA, nI, nN), xc.baseBA) };
        }
      }
    }
    // The PSTH per item and condition, handed over before the recorder resets.
    if(globalThis.__npMatrix && popKey){
      for(const [k, cond] of [['both', 0], ['vis', 1], ['aud', 2]]){
        const pm = rec.psthMatrix(cond, nI);
        if(pm) globalThis.__npMatrix(popKey + ':psth:' + k,
          { ...pm, items:itemList, tickMs:STEPS, simMin:+(simMs/60000).toFixed(2) });
      }
    }
    if(!sweepPending && !sweepActive) rec.reset();
    return out;
  };
  // What the cadence can actually measure, said once at the start rather than discovered by a silent readout column at three in the morning.
  {
    const sig0 = (net.inputMaps || []).map(m => m.signal).find(Boolean);
    if(sig0 && itemList.length){
      const per = Math.max(20, sig0.onMs) + Math.max(0, sig0.offMs);
      const bimodalShare = sig0.probeEvery > 0 ? 1 - 1/sig0.probeEvery : 1;
      const needMs = Math.ceil(itemList.length*2/Math.max(0.01, bimodalShare))*per;
      log(`readout needs ${itemList.length*2} bimodal trials (2 per item): ` +
        `about ${(needMs/60000).toFixed(1)} sim-min of presentations. ` +
        (needMs > CKPT_MS
          ? `The ${(CKPT_MS/60000).toFixed(1)} min checkpoint cannot hold that, so ` +
            `trials carry over and the first readout lands near sim ` +
            `${(needMs/60000).toFixed(1)} min.`
          : `Each checkpoint holds enough.`));
    }
  }
  const readout = () => {
    if(!recs.length) return null;
    const out = { items:itemList.length,
      chance:itemList.length ? +(1/itemList.length).toFixed(3) : 0, pops:{} };
    for(const r of recs) out.pops[r.label] = measure(r.rec, r.links, r.label);
    return out;
  };
  let simMs = resumeMs, spikes = 0, segSpikes = 0, ckptN = 0, pumpStopped = false;
  let paused = false, pauseAt = 0, pausedMs = 0;   // wall time spent paused
  let saveNow = false;                    // an explicit brain-file request
  let endDone = false;                      // end-of-run sequence latch
  let doneSignalled = false;                // the page is told once, not twice
  let nextCkpt = simMs + CKPT_MS;
  let segSum = 0, segSum2 = 0, segN = 0;    // per-tick spike counts -> population Fano
  let nextBrain = simMs + BRAIN_MS;
  const brainFiles = [];
  const wall0 = performance.now();
  const metrics = [];
  let waitingWeights = false;


  // Metrics are cheap and frequent; brain files are neither.
  // At full density a brain file is gigabytes and reading the weights back stalls the run, so rates and synchrony are logged every ckptmin while weights and the saved brain follow the slower brainmin cadence, keeping the most recent few plus the first so an overnight run does not fill the disk.
  // The weight-dependent fields of a checkpoint row, from a weights frame.
  const weightFields = (row, w) => {
    let dE = 0, nE = 0, dI = 0, nI = 0, moved = 0;
    for(let s=0;s<net.synCount;s++){
      const d = Math.abs(w[s] - w0[s]);
      if(d > Math.abs(w0[s])*0.01) moved++;
      if(w0[s] > 0){ dE += d; nE++; } else { dI += d; nI++; }
    }
    row.meanAbsDwExc = +(dE/Math.max(1,nE)).toFixed(4);
    row.meanAbsDwInh = +(dI/Math.max(1,nI)).toFixed(4);
    row.fracMoved = +(moved/net.synCount).toFixed(3);
    // how many inhibitory weights are pinned at the clamp: the number that turns a mistuned target into a dead column
    let pinned = 0, nInh = 0;
    for(let s2=0;s2<net.synCount;s2++) if(w0[s2] < 0){
      nInh++;
      if(-w[s2] >= net.wmax*0.999) pinned++;
    }
    row.inhAtLimit = +(pinned/Math.max(1, nInh)).toFixed(4);
    if(lastRho) row.setPoints = lastRho;
    // Weight between and within the two codes (analysis.js), the question asked of the synapses rather than of a decoder.
    if(pairClass) row.setWeights = setWeightStats(pairClass, w, w0);
    // and item by item: the change onto the same item's cells in the other code against the change onto other items' cells, zero exactly when nothing is learned
    if(betweenSyn && xferCal.size){
      row.itemCoupling = {};
      for(const [pop, xc] of xferCal)
        row.itemCoupling[pop] = itemCoupling(betweenSyn, xc.SAnet, xc.SBnet, w, w0);
    }
  };
  const saveCkpt = (w) => {
    // A host-pumped run writes its rates row at the checkpoint (saveCkpt with no weights) and the weights arrive later, with the host's brain save or the final readback.
    // The weights join the row they belong to rather than making a second row with every rate zero and an empty readout, and wSimMin says when they were read.
    const last = metrics[metrics.length - 1];
    if(w && last && last.meanAbsDwExc === undefined &&
       (PASSIVE || last.simMin === +(simMs/60000).toFixed(1))){
      weightFields(last, w);
      last.wSimMin = +(simMs/60000).toFixed(1);
      log(`ckpt ${ckptN} weights · sim ${last.wSimMin} min · |dW| exc ${last.meanAbsDwExc} inh ${last.meanAbsDwInh}` +
        ` · ${(last.fracMoved*100).toFixed(1)}% moved · ${(last.inhAtLimit*100).toFixed(1)}% inh at limit`);
      return writeBrain(w, [writeFile(dir, P.RUN_METRICS, JSON.stringify(metrics, null, 1))]);
    }
    ckptN++;
    const mu = segSum/Math.max(1, segN);
    // Population synchrony: the variance over the mean of the whole population's spike count per time window.
    // It is NOT the Fano factor, which is per cell across repeated presentations of one stimulus and is set below from fanoPerCell.
    const sync = mu > 0 ? (segSum2/Math.max(1, segN) - mu*mu)/mu : 0;
    segSum = 0; segSum2 = 0; segN = 0;
    const row = { simMin:+(simMs/60000).toFixed(1),
      wallMin:+((performance.now()-wall0)/60000).toFixed(1),
      rateHz:+(segSpikes/net.count/(CKPT_MS/1000)).toFixed(2),
      probes:probes.map(p => ({ l:p.label, hz:+(p.spikes/p.idx.length/(CKPT_MS/1000)).toFixed(2) })),
      sync:+sync.toFixed(1) };
    if(w) weightFields(row, w);
    const ro = readout();
    if(ro) row.readout = ro;
    metrics.push(row);
    segSpikes = 0; probes.forEach(p => p.spikes = 0);
    log(`ckpt ${ckptN} · sim ${row.simMin} min (wall ${row.wallMin}) · ${row.rateHz} Hz · ` +
      row.probes.map(p => `${p.l} ${p.hz} Hz`).join(' · ') +
      (w ? ` · |dW| exc ${row.meanAbsDwExc} inh ${row.meanAbsDwInh} · ${(row.fracMoved*100).toFixed(1)}% moved` +
        ` · ${(row.inhAtLimit*100).toFixed(1)}% inh at limit` : '') +
      (row.setWeights && row.setWeights.betweenMinusWithin !== undefined
        ? ` · between-code dW minus within ${row.setWeights.betweenMinusWithin}` : '') +
      ` · sync ${row.sync}` +
      (row.readout
        ? ` · chance ${(row.readout.chance*100).toFixed(0)}% · ` +
          // say what it is waiting for rather than printing an empty column
          (Object.values(row.readout.pops).every(v => v.decode === undefined)
            ? `readout accumulating, ${Math.max(0, ...Object.values(row.readout.pops)
                .map(v => v.waitingFor || 0))} more bimodal trials needed · `
            : '') +
          Object.entries(row.readout.pops)
            .filter(([, v]) => v.decode !== undefined)
            .map(([k, v]) => `${k} ${(v.decode*100).toFixed(0)}%/sel ${v.selectivity}` +
              // active fraction rides with selectivity: a high mean over a nearly silent population is the artifact, not a sparse code
              `@${Math.round((v.activeFrac || 0)*100)}%act` +
              (v.responsiveFrac !== undefined
                ? `/resp ${v.responsiveFrac}/rel ${v.reliability}/corr ${v.ensembleCorr}` +
                  `/top ${Math.round((v.decodeTop || 0)*100)}%` +
                  (v.decodeShape !== undefined
                    ? `/shape ${Math.round(v.decodeShape*100)}%` : '') : '') +
              (v.likeDelta !== undefined ? `/l2l ${v.likeDelta}|w ${v.likeW}` : '') +
              // the conjunction measure: own minus other in rank units, zero is chance, and it is the one that asks about binding
              (v.bind ? `/bind v${v.bind.vis === null ? '-' : v.bind.vis}` +
                `|a${v.bind.aud === null ? '-' : v.bind.aud}` +
                `|c${v.bind.cells}` : '') +
              // timing agreement: 0 is an unrelated order, 1 is the same one
              (v.timing ? `/lat v${v.timing.vis === null ? '-' : v.timing.vis}` +
                `|a${v.timing.aud === null ? '-' : v.timing.aud}` : '') +
              (v.crossModal ? `/xmodal ${v.crossModal.rank1}` +
                (v.crossModal.ownC !== undefined
                  ? `|cos ${v.crossModal.own}v${v.crossModal.other}` +
                    `|ctr ${v.crossModal.ownC}v${v.crossModal.otherC}` +
                    `/r1c ${v.crossModal.rank1C}` +
                    (v.crossModal.relVis !== undefined
                      ? `|rel ${v.crossModal.relVis}/${v.crossModal.relAud}` : '') : '') +
                (v.crossModal.trialN ? `|t${Math.round(v.crossModal.trialAcc*100)}%~${Math.round(2*v.crossModal.trialSe*100)} n${v.crossModal.trialN}` : '') : '') +
              (v.visualOnly ? `/vis ${v.visualOnly.rank1}` +
                (v.visualOnly.ownC !== undefined ? `c${v.visualOnly.ownC}` : '') : '') +
              (v.audioOnly ? `/aud ${v.audioOnly.rank1}` +
                (v.audioOnly.ownC !== undefined ? `c${v.audioOnly.ownC}` : '') : '') +
              (v.templateRel !== undefined ? `/tpl ${v.templateRel}` : '') +
              (v.driftVis ? `/drift v${v.driftVis.gap}~${(2*v.driftVis.se).toFixed(3)}` : '') +
              (v.driftAud ? `a${v.driftAud.gap}~${(2*v.driftAud.se).toFixed(3)}` : '') +
              (v.crossModal && v.crossModal.perItem
                ? `/items ${v.crossModal.perItem}` : ''))
            .join(' · ')
        : ''));
    // Hand the row to the building UI as well as the file, so a probe can show what it measured without anyone opening a JSON.
    if(globalThis.__npMetrics) globalThis.__npMetrics(row);
    const jobs = [writeFile(dir, P.RUN_METRICS, JSON.stringify(metrics, null, 1))];
    // One file, overwritten each checkpoint rather than one per checkpoint: a recording probe of 8192 cells over a couple of hundred trials is about six megabytes, which is fine once and is not fine ninety times.
    // It always describes the most recent window, which is the one whose numbers are on screen.
    if(trialExport.size){
      const header = { format:'np-trials-1', run:runTag, simMs, simMin:+(simMs/60000).toFixed(2),
        // 0, 1 and 2 are the codes the recorder stamps on every trial; the curriculum node names them, and a scene that has not renamed them gets the three the recorder was built around
        items:itemList, conditions:CONDS, populations:[] };
      const arrays = [];
      for(const [pop, d] of trialExport){
        const nT = d.trials.length, m = d.neurons;
        const vecs = new Float32Array(nT*m), labels = new Int32Array(nT),
          conds = new Int32Array(nT);
        d.trials.forEach((t, i) => { vecs.set(t.vec, i*m); labels[i] = t.label; conds[i] = t.cond; });
        header.populations.push({ name:pop, neurons:m, trials:nT,
          neuronIndex:d.idx,
          arrays:[
            { name:'counts', dtype:'float32', shape:[nT, m] },
            { name:'label', dtype:'int32', shape:[nT] },
            { name:'cond', dtype:'int32', shape:[nT] },
          ] });
        arrays.push(vecs, labels, conds);
      }
      jobs.push(writeFile(dir, P.RUN_TRIALS, packTrialFile(header, arrays)));
      trialExport.clear();
    }
    return writeBrain(w, jobs);
  };
  // The brain file for a weights frame, with its retention, joined to the other writes of the same checkpoint.
  const writeBrain = (w, jobs) => {
    if(w){
      const name = P.ckptName(runTag, simMs);
      // A gated scenario freezes most of its synapses, and a frozen weight is reproduced exactly by the wiring, so writing it is writing a number the loader is about to compute anyway; a plastic-only file is a tenth of the size and of the writing time.
      // Ungated networks keep the weights-only file.
      const pk = plasticCount(net);
      // pk of 0 means nothing is plastic, and a plastic-only file would then hold no weights at all.
      // That is the weights-file case, not a smaller plastic one.
      const usePlastic = pk !== null && pk > 0 && pk < net.synCount*0.9;
      const bytes = usePlastic ? plasticFileBytes(pk) : weightFileBytes(net.synCount);
      brainFiles.push(name);
      jobs.push(ST.write(P.runBrainPath(runTag, name), usePlastic
        ? buildPlasticFile(graph, net, w, simMs, curriculumPosition(simMs))
        : buildWeightsFile(graph, net, w, simMs, curriculumPosition(simMs))));
      // retention: always keep the first, then the most recent that fit both the count and the byte budget
      while(brainFiles.length > KEEP + 1 ||
            (brainFiles.length > 2 && brainFiles.length*bytes > KEEP_BYTES)){
        const drop = brainFiles.splice(1, 1)[0];
        jobs.push(ST.remove(P.runBrainPath(runTag, drop)).catch(() => {}));
      }
    }
    // The listing after a save is a courtesy to the FILES pane, two requests per run folder; with two hundred run folders it is the biggest burst the page sends, and a failed request in it must not reject the checkpoint save, which sits inside the tick handler, or the run stops after its last checkpoint with the engine finished and the sweep waiting.
    return Promise.all(jobs).then(() => listRuns().catch(() => {}));
  };

  let lastRho = null;
  // last weights read back from the engine, for the like-to-like measure.
  // Weights only come back at brain saves, so between them this is the state at the previous save rather than the current one; connectivity does not change, so only the similarity-against-weight figure is affected and it is labeled as of the last readback.
  let lastW = null;
  // The warm-up has to outlast the calibration sweep.
  // The conjunctive sets are what the whole binding measure is scored against, and a set built from a network that has already been learning for a few minutes is not a baseline.
  // So plasticity stays off for the settle time or the full sweep, whichever is longer, and the sweep is sized by the curriculum itself.
  const csig0 = (net.inputMaps && net.inputMaps[0] && net.inputMaps[0].signal) || null;
  const calMs = csig0 && csig0.calCycles
    ? csig0.calCycles * 3 * itemList.length *
      (Math.max(20, csig0.onMs) + Math.max(0, csig0.offMs))
    : 0;
  const HOLD_MS = Math.max(WARMUP_MS, calMs);
  let warmup = PLAST && HOLD_MS > 0;
  plastOn = !warmup;                      // on from the start when there is no hold
  if(warmup && calMs)
    log(`calibration sweep: every one of ${itemList.length} items in all three ` +
      `conditions, ${csig0.calCycles} passes, ${(calMs/60000).toFixed(1)} sim-min, ` +
      `plasticity off throughout`);
  if(warmup) log(`warm-up: plasticity off for the first ${(HOLD_MS/1000).toFixed(0)} sim-seconds`);

  worker.onmessage = async e => {
    const m = e.data;
    if(m.__attached){
      simMs = m.__attached.simMs || 0;
      resumeMs = simMs;
      log('attached to live run ' + (m.__attached.dir || '?') + ' at sim ' +
        (simMs/60000).toFixed(1) + ' min, running to ' +
        (m.__attached.untilMs/60000).toFixed(0) + ' min');
      return;
    }
    // an engine that cannot stream first-spike latency says so, and the timing readout is then absent; said once in the log rather than dropped
    if(m.cmd === 'unsupported'){
      log(`the engine does not stream ${m.of === 'sendT' ? 'first-spike latency; the timing readout is absent from this run' : m.of}`);
      return;
    }
    if(m.cmd === 'error'){
      log('ENGINE ERROR: ' + m.message);
      lastError = m.message;
      writeStatus('error');
      // Say it where the run is being watched, not only in the log.
      // A dead engine leaving the bar reading TRAINING with the clock at zero is the most misleading thing the page can do.
      statEl.textContent = 'ENGINE ERROR: ' + m.message;
      if(globalThis.__npTrainFailed) globalThis.__npTrainFailed(m.message);
      return;
    }
    if(m.cmd === 'weights'){
      waitingWeights = false;
      lastW = m.w;
      globalThis.__npWeights = m.w;
      // Measured homeostatic set points, when the engine has them.
      // An unreachable target drives inhibition into the clamp, and is invisible while these live only in the engine.
      // Logging the spread against the rates actually achieved makes that diagnosable from the metrics alone.
      if(m.rho && m.rho.length){
        const r = m.rho;
        let lo = Infinity, hi = -Infinity, sum = 0;
        for(let i=0;i<r.length;i++){ const v = r[i];
          if(v < lo) lo = v; if(v > hi) hi = v; sum += v; }
        lastRho = { mean:+(sum/r.length).toFixed(2), min:+lo.toFixed(2), hi:+hi.toFixed(2) };
      }
      await saveCkpt(m.w);
      if(simMs >= resumeMs + TOTAL_MS){
        worker.terminate();
        setMaxSyn(0);                        // the run's budget ends with it
        statEl.textContent = 'DONE · ' + (simMs/3600000).toFixed(2) + ' sim-hours trained';
        writeStatus('done', { simMin:+(simMs/60000).toFixed(2), ckpts:ckptN });
        log('training complete; download checkpoints above and LOAD BRAIN in the playground');
        // hand the page back: while a run owns the engine the building UI refuses to rebuild, and without this it stays that way for the rest of the session.
        // Latched, because the remote engine can deliver a second weights frame after the terminate and the completion block would run again: in a sweep the second signal lands on the next run's handler and starts a third run beside it, two runs sharing one folder.
        if(!doneSignalled){
          doneSignalled = true;
          // the final weights go back with the signal, so the page can put the trained network on screen rather than an empty viewer
          if(globalThis.__npOnTrainDone) globalThis.__npOnTrainDone(m.w);
        }
        return;
      }
      if(!PASSIVE && !paused) worker.postMessage({ cmd:'tick', steps:STEPS });
      return;
    }
    if(m.cmd !== 'state') return;
    if(superseded) return;   // a newer run has taken the engine
    simMs = (PASSIVE && m.__simMs !== undefined) ? m.__simMs : simMs + m.steps;
    spikes += m.spikes; segSpikes += m.spikes;
    // the engine keeps running at full speed with the tab hidden, so the display work is skipped entirely rather than drawn into nothing
    if(!document.hidden){
      viewer.onFired(m.fired);
      drawRaster(m.fired, net.count);
      updateProbes(m.fired, m.steps);
    }
    segSum += m.spikes; segSum2 += m.spikes*m.spikes; segN++;
    for(const p of probes){
      let c = 0;
      // fired is a spike count per cell for the tick (up to 255); counted as a flag it would cap every probe at one spike per tick, 20 Hz at 50 steps.
      for(let k=0;k<p.idx.length;k++) c += m.fired[p.idx[k]];
      p.spikes += c;
    }
    // stimulus follows simulated time; when the host drives the pump the page still renders it for the input monitor but sends nothing
    io.frame(simMs, PASSIVE);
    // The first seconds after init are an onset transient: every neuron is near reset and the background current arrives together.
    // Recording it would put a synchronous burst into the first trials of every item and make them look alike.
    // The settle window is the same one plasticity waits out, measured from the start of this run rather than from the restored clock, since a resumed run transients too.
    // Recording waits out the startup transient only, not the whole plasticity hold: the calibration sweep runs inside that hold and its trials are exactly what the conjunctive sets are built from, so gating recording on HOLD_MS would leave nothing to calibrate from.
    if(recs.length && simMs >= resumeMs + WARMUP_MS){
      const st = scheduleAt(csig, simMs);
      const lab = st.on ? (itemLabel.has(st.item) ? itemLabel.get(st.item) : -1) : -1;
      const key = (st.on ? 'p' : 'g') + st.k;
      for(const r of recs) r.rec.sample(m.fired, key, lab, st.cond, m.lat);
      // A sweep window holds the recorder open, the same way the opening one does, because its trials only mean something together.
      // The report runs as the window closes, while they are all still there.
      const inSweep = !!st.sweep;
      if(sweepActive && !inSweep){
        // Only a repeat clears up after itself.
        // The opening sweep's trials are read by calibrateNow, which runs further down this same handler, so a reset here would destroy the calibration a few lines before it is taken.
        if(wasReSweep){
          reSweepReport();
          for(const r of recs) r.rec.reset();
        }
        wasReSweep = false;
      }
      if(inSweep && !sweepActive){
        wasReSweep = !!st.resweep;
        // A repeat measures its own window and nothing else.
        // Left to accumulate, the recorder still holds the ordinary presentations since the last checkpoint, almost all bimodal, so the both-senses centroid is built from many more trials than the single-sense ones and the noise difference alone invents a conjunction, even in the retina where one is structurally impossible.
        // The opening sweep is clean only because it starts from an empty recorder at t=0.
        if(wasReSweep) for(const r of recs) r.rec.reset();
      }
      sweepActive = inSweep;
    }
    // A host-pumped run does not run the branch below, since the host owns the plasticity switch, so the calibration moment is caught here.
    if(calAtMs && simMs >= calAtMs){ calAtMs = 0; calibrateNow(); plastOn = true; }
    // relative to the start of this run, not to the restored clock: a resumed run starts from a fresh init and transients exactly like a cold one, and comparing against the absolute clock would skip the settle window on every resume.
    // Learning through that first synchronous burst is the failure the settle window prevents.
    if(!PASSIVE && warmup && simMs >= resumeMs + HOLD_MS){   // settle first, then learn
      warmup = false;
      // Calibrate here, on the sweep that has just finished, while nothing has learned yet.
      // Waiting for the next checkpoint would take the baseline from a network that had already been learning for however long the cadence is, which is not a baseline.
      calibrateNow();
      plastOn = true;
      worker.postMessage({ cmd:'tune', ...engineCfg, plast:(PLAST && NODE_PLAST) ? 1 : 0 });
      log(`plasticity ${(PLAST && NODE_PLAST) ? 'on' : 'off, as the checkpoint says,'} at sim ${(simMs/1000).toFixed(0)} s (warm-up complete)`);
    }
    if((simMs/STEPS & 63) === 0){
      const wallS = (performance.now() - wall0 -
        (pausedMs + (pauseAt ? performance.now() - pauseAt : 0)))/1000;
      // against time elapsed in this run, not against the restored clock: a run resumed at sim 120 has not simulated two hours in its first second, and the spikes counted here started at zero with it
      const ranS = (simMs - resumeMs)/1000;
      statEl.textContent = `sim ${(simMs/60000).toFixed(1)} / ${((resumeMs+TOTAL_MS)/60000).toFixed(0)} min · ` +
        `${(ranS/wallS).toFixed(2)}x realtime · ${(spikes/net.count/Math.max(1e-9, ranS)).toFixed(2)} Hz mean`;
      // The same headline as fields rather than a sentence, for the strip in the viewer.
      // A run is the loudest thing the app can be doing, so the state belongs where the eye already is, not only in a side panel.
      // Fields, not the string above, because parsing a sentence back apart is a bug waiting to happen.
      if(globalThis.__npTrainHead) globalThis.__npTrainHead({
        // plastOn, not warmup: warmup means "the switch still needs scheduling" and a host-pumped run clears it at startup, so on warmup the bar would read TRAINING through the whole calibration sweep, the one stretch where the network is deliberately not learning.
        phase: plastOn ? 'training' : 'warm-up',
        simMin: simMs/60000, targetMin: (resumeMs+TOTAL_MS)/60000,
        xRealtime: ranS/wallS, meanHz: spikes/net.count/Math.max(1e-9, ranS),
        engine: ENGINE, pumped: PUMP });
      const now = performance.now();
      if(now - lastStatus > 15000){
        lastStatus = now;
        writeStatus(warmup ? 'warm-up' : 'training', {
          simMin:+(simMs/60000).toFixed(2), targetMin:+(TOTAL_MS/60000).toFixed(0),
          xRealtime:+(ranS/wallS).toFixed(3),
          meanHz:+(spikes/net.count/Math.max(1e-9, ranS)).toFixed(2),
          nextCkptMin:+(nextCkpt/60000).toFixed(2), ckpts:ckptN });
      }
    }
    // The end-of-run test is level triggered on simMs, so once the run is long enough every further state frame re-enters it and repeats a full weight readback and readout, gigabytes each at full density; endDone latches the sequence to exactly one pass.
    const atEnd = simMs >= resumeMs + TOTAL_MS;
    if(endDone) return;
    if(simMs >= nextCkpt || atEnd){
      nextCkpt += CKPT_MS;
      if(!PASSIVE && (simMs >= nextBrain || atEnd)){
        nextBrain += BRAIN_MS;
        if(atEnd) endDone = true;
        worker.postMessage({ cmd:'getWeights' });   // pump resumes after the save
        return;
      }
      saveCkpt(null);                        // rates and synchrony only
      if(PUMP && atEnd){
        if(!pumpStopped){
          pumpStopped = true;
          endDone = true;
          worker.postMessage({ __host:'pump', on:false });
          worker.postMessage({ cmd:'getWeights' });   // final readback for the page
        }
        return;
      }
    }
    // A paged run stops by simply not asking for the next tick; a host-pumped one stops at the host, which owns the loop.
    // Either way nothing is torn down: the engine keeps the network, the stimulus clock stops with the simulation because it advances on simulated time, and resuming carries on from where it stopped.
    // An on-demand save joins the tick chain here rather than posting getWeights out of band, which would leave two chains running.
    if(saveNow && !PASSIVE && !waitingWeights){
      saveNow = false; waitingWeights = true;
      worker.postMessage({ cmd:'getWeights' });
      return;
    }
    if(!PASSIVE && !paused) worker.postMessage({ cmd:'tick', steps:STEPS });
  };
  // Write a brain file now rather than at the next cadence, for stepping away from a run with hours left on it.
  // Same on every engine from the caller's side; underneath, a host-pumped run is saved by the host and a paged one by this loop, because in each case that is who owns the engine.
  // Paused is the case that matters and it works in both: nothing is in flight, so the request is served immediately.
  globalThis.__npTrainSave = () => {
    if(PUMP){ worker.postMessage({ __host:'saveNow' }); return 'requested'; }
    if(waitingWeights || saveNow) return 'busy';
    if(paused){                            // no tick chain to join
      waitingWeights = true;
      worker.postMessage({ cmd:'getWeights' });
      return 'requested';
    }
    saveNow = true;
    return 'requested';
  };
  // Exposed to the page so the viewer bar can drive it.
  // Returns the state it settled in, so a button cannot disagree with the run about what happened.
  globalThis.__npTrainPause = want => {
    const to = want === undefined ? !paused : !!want;
    if(to === paused) return paused;
    paused = to;
    if(PUMP) worker.postMessage({ __host:'pump', on:!paused, steps:STEPS });
    if(paused){ pauseAt = performance.now(); }
    else {
      // Wall time spent paused is not time the engine failed to keep up in, so it comes out of the realtime factor.
      // Without this an hour paused would read as an engine that had slowed to a crawl.
      if(pauseAt){ pausedMs += performance.now() - pauseAt; pauseAt = 0; }
      if(!PASSIVE) worker.postMessage({ cmd:'tick', steps:STEPS });
    }
    log((paused ? 'paused' : 'resumed') + ' at sim ' +
      (simMs/60000).toFixed(1) + ' min');
    return paused;
  };
  if(PASSIVE && ENGINE !== 'remote')
    log((ATTACH ? 'attach=1' : 'pump=host') + ' needs engine=remote; running with the page pump instead');
  if(ATTACH && ENGINE === 'remote'){
    worker.postMessage({ __host:'attach', dir: Q.get('run') || '' });
    log('viewer mode: asking the host for the live run' +
      (Q.get('run') ? ' ' + Q.get('run') : ''));
  } else if(PUMP && ENGINE === 'remote'){
    // hand the run to the host: identity for checkpoint files, the input maps and clock for the stimulus, the warm-up switch as a scheduled message, and the brain cadence; then start its pump
    worker.postMessage({ __host:'net', count:net.count, synCount:net.synCount,
      graph, curriculum: curriculumPosition(resumeMs) });
    if(atlas) worker.postMessage({ __host:'assets', atlas });
    worker.postMessage({ __host:'maps', maps: net.inputMaps || [], seed: runSeed });
    worker.postMessage({ __host:'clock', simMs: resumeMs });
    if(warmup){
      worker.postMessage({ __host:'at', simMs: resumeMs + HOLD_MS,
        msg:{ cmd:'tune', ...engineCfg, plast:(PLAST && NODE_PLAST) ? 1 : 0 } });
      // The host flips plasticity on its own clock, so this page has to calibrate when that moment passes rather than at its next checkpoint.
      calAtMs = resumeMs + HOLD_MS;
      warmup = false;
      log('warm-up switch scheduled at the host');
    }
    worker.postMessage({ __host:'save', everyMs: BRAIN_MS, dir: runTag });
    worker.postMessage({ __host:'detach', untilMs: resumeMs + TOTAL_MS });
    worker.postMessage({ __host:'pump', on:true, steps:STEPS });
    log('host pump on: this run survives a frozen tab; brains land in host/runs/' + runTag);
  } else worker.postMessage({ cmd:'tick', steps:STEPS });
}
await main().catch(e => log('ERROR: ' + e.message));
}

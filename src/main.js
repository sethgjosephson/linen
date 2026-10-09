import { NodeEditor } from './editor.js';
import { PROTOCOL, checkHello, missingTerms, answers, ignoredCustom, customNote } from './protocol.js';
import { Viewer } from './viewer.js';
import { sliceNormalize, sliceDefault } from './slice.js';
import { renderProps, renderGroupProps, setPropsAction, setPropsDocs,
  setPropsLocked, setPropsMetrics, setPropsGraph, setPropsCharts, setPropsRuns, setPropsProbeSet,
  setPropsDriveSet, setPropsInputSet, setPropsConjSet,
  setPropsWeights, setPropsMatrices, setPropsLive, setHueLookup, chartSources, setPropsCaption, setPropsEngine } from './props.js';
import { chooseDir, chooseProjectDir, rememberedProjectDir, grantProjectDir } from './fsstore.js';
import { store, whereLabel, resetStore, useFolder, setProjectHandleSource } from './store.js';
import { icon, iconify, installTips } from './icons.js';
import { applyOverrides as applyOverridesTo, applyVariant, gatherVariants } from './sweepvariant.js';
// overrides on the editor's own nodes mark them dirty and report on the status line
const applyOverrides = (spec, nodes = editor.nodes, onEdit = n => editor.markDirty(n), sep = /[;,]/) =>
  applyOverridesTo(spec, nodes, onEdit, sep, msg => status(msg, true));
// on the hosted site the project is a folder picked in the browser, kept as a handle; the store asks here for it when there is no server
setProjectHandleSource(rememberedProjectDir);
import { renderRuns, listRunTags, readRunMetrics, listSweepTags, readSweep } from './runlist.js';
import * as P from './project.js';
import { loadProjectPlugins, addPluginFromUrl, importer, scenePluginsMissing } from './plugins.js';
import { engineConfig, engineTypes, lockHues, tagMatch, NEURON_TYPES, NODE_DEFS, computeNode, setComputeProgress, setResolution, setRateHint,
  tagHues, hueCss, hueRgb, buildRuleTable, RULE_STRIDE, deviceSynLimit, sameTag,
  setFileReader, forgetMesh, forgetPoints, forgetConnections } from './nodes.js';
import { initState } from './rand.js';
import { expandPopulations, collapsePopulations } from './expand.js';
import { SCENARIOS, guidedTour } from './scenarios.js';
import { ALL_SCENARIOS } from './experiments.js';
import { ACTIONS, GROUPS, binding, isCustom, setBinding, resetBindings,
  eventToBinding, matches as keyMatches } from './keymap.js';
import { RemoteEngine } from './remoteworker.js';
import { layoutTopDown } from './layout.js';
import { buildBrainFile, buildWeightsFile, parseBrainFile, applyBrainWeights,
  pairFileBytes, MAX_BUFFER } from './brain.js';
import { IORuntime } from './io.js';
import { attachProbes, setupProbes, updateProbes, probesWantV } from './probeview.js';
import { Recorder } from './record.js';
import { encodeSceneFragment, decodeSceneFragment } from './link.js';
import { caseFromInit } from './briancase.js';
import { layoutSlug, layoutRecord, applyLayoutRecord, markBuilt, builtTypes, isBuilt } from './layoutfile.js';
import { nameNodes } from './nodenames.js';
import { Recording, weightSummary, changedKeys } from './recording.js';
import { SceneTabs, initMessage, tuneMessage, engineInputs, engineTerms, engineKind, wantGpu, gpuMissing } from './document.js';
import { neuronsCsv, synapsesCsv } from './export.js';
import { csv, download, downloadText, canvasPng, seriesCsv, multiSeriesCsv, bandSeriesCsv, rasterCsv, spikeTrainsAscii, histogramCsv, matrixCsv, figureName } from './figure.js';
import { withMeta } from './png.js';
import { attachInputs, setupInputs, updateInputs } from './inputview.js';

const statusEl = document.getElementById('status');
const rateEl = document.getElementById('rate');
const rtfEl = document.getElementById('rtf');
const propsEl = document.getElementById('props');
const rasterCv = document.getElementById('raster');

const viewer = new Viewer(document.getElementById('viewer'), document.getElementById('hud'));
let propsNode = null;                      // props follow dblclick (inspect), not selection
let propsGroup = null;                     // the panel shows a node or a group, never both
function showProps(n){
  propsNode = n; propsGroup = null;
  renderProps(propsEl, n, onParamEdit, editor.nodes);
}
// A group carries no parameters and never reaches a computation, so an edit here redraws (the editor repaints every frame anyway) and records history without scheduling rewiring.
function showGroupProps(g){
  propsNode = null; propsGroup = g;
  renderGroupProps(propsEl, g,
    commit => { if(commit){ pushHistory(); saveGraph(); } },
    () => { editor.ungroup(g); showProps(null); saveGraph(); },
    front => editor.raiseGroup(g, front),
    () => editor.fitGroup(g));
}

// downstream transform-node params between an edited node and the network, so viewer gizmos can preview shapes where the rendered points actually are.
// Affine transforms only; noise warp / twist / gauss blur cannot be represented and are passed through as identity.
function transformChainFor(node){
  const consumers = id => editor.nodes.filter(m => m.inputs.some(c => c && c.id === id));
  let cur = node;
  if(NODE_DEFS[node.type] && NODE_DEFS[node.type].cat === 'regions'){
    cur = consumers(node.id).find(m => m.type === 'scatter');
    if(!cur) return [];
  }
  const chain = [];
  let guard = 50;
  while(cur && guard-- > 0){
    const next = consumers(cur.id)[0];
    if(!next || next.type === 'connect' || next.type === 'checkpoint') break;
    if(next.type === 'move' && next.on !== false)
      chain.push(structuredClone(next.params));
    cur = next;
  }
  return chain;
}
const editor = new NodeEditor(document.getElementById('graph'), {
  onError: msg => status(msg, true),   // a handler that threw, named, instead of a pane gone quiet
  onSelect: n => viewer.focusPopulationBySrc(n && n.id),   // its box, if it made one
  onInspect: n => showProps(n),
  onInspectGroup: g => showGroupProps(g),
  onChange: () => {
    if(propsNode && !editor.nodes.includes(propsNode)) showProps(null);
    if(propsGroup && !editor.groups.includes(propsGroup)) showProps(null);
    scheduleCompute();
  },
  // position, grouping and depth: record history, never rewire.
  // Ungrouping reaches here rather than onChange, so the panel is dropped here too or it goes on showing a group that no longer exists.
  onMoved: () => {
    if(propsGroup && !editor.groups.includes(propsGroup)) showProps(null);
    pushHistory();
  },
  onEditGeo: n => viewer.editGeometry(n, node => {
    editor.markDirty(node); scheduleCompute();
    if(propsNode === node) throttleProps(node);
  }, transformChainFor(n)),
  onScenario: s => {
    editor.clear();
    setSceneSettings({ slice:null, hues:null });   // a scenario is a whole tissue, with fresh hues
    editor.viewSlots = {}; editor.activeView = null;
    editor.suggestResolution = 0;
    fitNextCompute = true;   // frame the camera on a freshly loaded scenario
    s.build(editor);
    markBuilt(editor.nodes);             // node k of the build is built k, the key a saved layout uses
    editor.scenarioName = s.name;
    // Position is display only and never reaches the computation, so it is derived from the graph, and a saved arrangement (src/layouts/<slug>.json, the scene menu writes one) replaces the computed one when it matches this build.
    layoutTopDown(editor.nodes, { groups: editor.groups, trunk:true, route:(x, y) => editor.addNode('pin', x, y) });
    applySavedLayout(s.name);
    // scenarios authored at real density can suggest a working resolution: the authored counts stay full scale, the slider does the trading
    if(editor.suggestResolution > 0){
      setSceneSettings({ resolution:editor.suggestResolution });
      status(`resolution set to ${editor.suggestResolution}% for this scenario (authored at full density)`);
    }
    assignDefaultSlots();
    editor.frameAll();
    showProps(null);
    recompute();
  },
});
// The pair table draws rules declared by plasticity and project nodes upstream, so the panel needs the graph, not just the node it is showing.
setPropsGraph(() => ({
  byId: id => editor.nodes.find(n => n.id === id),
  nodes: () => editor.nodes,
  select: n => { editor.sel = n; editor.draw(); showProps(n); },
  togglePathway: (from, to, rule) => togglePathway(from, to, rule),
  pathwayState,
  clearPathways,
  showAllPathways: pairs => {
    clearPathways();
    let n = 0;
    for(const [a, b, rule] of pairs){
      if(n >= PATH_MAX) break;
      if(activePaths.has(pathKey(a, b))) continue;
      togglePathway(a, b, rule); n++;
    }
    if(pairs.length > n)
      status(`showing ${n} of ${pairs.length} pathways, the most colors that stay tellable apart`);
  },
}));
editor.scenarios = SCENARIOS;
// The viewer bar wraps in a narrow pane; the viewer and the scope sit below whatever height it takes, through a CSS variable the stylesheet reads.
{
  const vb = document.getElementById('viewerbar');
  const setH = () => {
    const h = vb.offsetHeight + 'px';
    if(document.documentElement.style.getPropertyValue('--vbh') === h) return;
    document.documentElement.style.setProperty('--vbh', h);
    // the viewer's canvas sizes itself on resize; a taller strip is one
    window.dispatchEvent(new Event('resize'));
  };
  if('ResizeObserver' in window) new ResizeObserver(setH).observe(vb);
  window.addEventListener('resize', setH);
  setH();
}
window.editor = editor;   // debug access, as window.viewer already is
let _lastProps = 0;
function throttleProps(node){
  const now = performance.now();
  if(now - _lastProps > 150){ _lastProps = now; renderProps(propsEl, node, onParamEdit, editor.nodes); }
}
// what is picked in the viewer belongs to the shown scene: its tab comes forward first when another one is being edited
const inShownTab = then => { if(tabs.cur === tabs.shown) then(); else switchTab(tabs.shown).then(ok => { if(ok) then(); }); };
viewer.onPickSource = id => inShownTab(() => {
  const n = editor.byId(id);
  if(n) editor.selectExternal(n);
});
// the panel row for whatever the pointer is over in the viewer, so the two lists of regions agree about which one is live
viewer.onHoverPopulation = tag => {
  const body = document.querySelector('#regions .ovbody');
  if(!body) return;
  for(const row of body.children) row.classList.toggle('on', row.dataset.tag === tag);
};
// Clicking a population's label selects the node that made it.
// The source id is the scatter for a plain population; a repeat's copies and a points file's populations carry derived ids that no node has, so those fall back to the node whose tag names the population.
viewer.onPickPopulation = (src, tag) => inShownTab(() => {
  let n = editor.byId(src);
  if(!n) n = editor.nodes.find(q => q.params && typeof q.params.tag === 'string' &&
    q.params.tag.toLowerCase() === String(tag).toLowerCase());
  if(n) editor.selectExternal(n);
  else status('no node names the population ' + tag, true);
});

const scopeCv = document.getElementById('scope');
let scopeBuf = [];
// Inspection goes to whichever engine currently owns the network: this page's own during interactive use, and the training run's once it has taken over. trainEngine is set when a run starts in this page.
let trainEngine = null;
function inspectEngine(){ return worker || trainEngine; }
viewer.onSelect = idx => {
  scopeBuf = [];
  scopeCv.style.display = idx >= 0 ? 'block' : 'none';
  const eng = inspectEngine();
  if(!eng) return;
  if(offers(eng, 'watch', 'a cell\'s membrane trace')) eng.postMessage({ cmd:'watch', idx });
  if(idx >= 0 && offers(eng, 'query', 'connection queries')) eng.postMessage({ cmd:'query', idx, cap:QUERY_CAP });
};

// ---- pathway highlight ---------------------------------------------
// Clicking a row of the pair table draws that pathway in the viewer.
// It uses the per-neuron 'query' the protocol already has rather than adding a pair query to it: ENGINE.md is a frozen contract, all three engines already answer 'query', and a handful of presynaptic cells is enough to show where a pathway goes.
// Sampling is the honest word for it and the HUD says so.
// One request in flight at a time, the rest queued. queryResult echoes only the neuron index, so two pathways sampling the same neuron are indistinguishable in the reply and a second request started before the first finished would eat its answers.
let pathReq = null;
const pathQueue = [];
// A bundle is drawn from many source cells with a few synapses each, not from a few cells with everything: a dozen cells with five thousand lines apiece read as a pathway from almost nowhere.
// Up to PATH_SOURCES cells spread evenly over the source population, each contributing at most PATH_PER_SOURCE of its synapses into the target, chosen by a seeded hash so the same pathway draws the same lines.
const PATH_SOURCES = 400, PATH_PER_SOURCE = 4, PATH_QUERY_CAP = 1024;
// How many pathways can be lit at once.
// Each one costs PATH_SOURCES queries and a line buffer.
// Color belongs to the population rather than to the pathway (below), so more pathways do not mean more colors to tell apart; the limit is the query cost.
const PATH_MAX = 48;
// Connection lines drawn per neuron query; the totals stay exact.
const QUERY_CAP = 5000;
// Color belongs to the population, not to the pathway.
// A tag is already the stable name for a population, so a color derived from it means the same thing in the table, in the viewer and across scenes, instead of depending on what order things were switched on.
// A pathway has two ends, so its lines run as a gradient from the source color to the target color, which also makes the direction of a bundle visible without needing an arrowhead.
let tagHueMap = new Map();
// a pattern (src*) takes the hue of the first population it matches
const hueOf = tag => {
  if(tagHueMap.has(tag)) return tagHueMap.get(tag);
  if(String(tag).includes('*')) for(const [k, h] of tagHueMap) if(tagMatch(k, tag) !== null) return h;
  return 0;
};
// the scene's locked hues (lockHues in nodes.js): tag -> hue, saved with the scene, so a population keeps its color across reloads
let sceneHues = {};
// hues chosen on nodes (scatter, points file, population), read from the graph rather than from the wired net so a color change recolors the running network without touching the wiring
function hueOverrides(net){
  const out = {}, tags = (net && net.tags) ? [...new Set(Object.values(net.tags).filter(Boolean))] : [];
  for(const n of editor.nodes){
    if(n.on === false || n.params === undefined) continue;
    const hv = n.params.hue === undefined || n.params.hue === '' ? -1 : (n.params.hue|0);
    if(hv < 0) continue;
    if(n.type === 'scatter' || (n.type === 'pointfile' && !String(n.params.tagColumn || '').trim())){ if(n.params.tag) out[n.params.tag] = hv; }
    else if(n.type === 'population'){ const t = String(n.params.tag || '').trim(); if(!t || t === 'E' || t === 'I') continue;
      for(const x of tags) if(tagMatch(x, t) !== null) out[x] = hv; }
  }
  return out;
}
const sceneTagHues = net => lockHues(sceneHues, (net && net.tags) || {}, hueOverrides(net));
setHueLookup(n => {
  const t = n.type === 'population' ? String(n.params.tag || '').trim() : String(n.params.tag || '').trim();
  if(!t) return -1;
  if(tagHueMap.has(t)) return tagHueMap.get(t);
  for(const [k, h] of tagHueMap) if(tagMatch(k, t) !== null) return h;
  return -1;
});
function applyHues(h){ sceneHues = h && typeof h === 'object' ? { ...h } : {}; }
const activePaths = new Map();          // key -> { from, to, fromHue, toHue }
// The eye in each row carries its pathway's color, so the table is the legend and there is nothing separate to read.
// That means the panel has to be redrawn whenever the lit set changes.
const refreshProps = () => { if(propsNode) showProps(propsNode); };
const pathKey = (a, b) => a + '→' + b;
// What the panel needs in order to color each eye: both ends of every lit pathway, so the icon can carry the same gradient the bundle does.
function pathwayState(){
  const out = {};
  for(const [k, p] of activePaths)
    out[k] = { from:hueCss(p.fromHue), to:hueCss(p.toHue) };
  return out;
}
// rule is the pair row's fifth column: '' the checkpoint's own rule, '0' frozen, or a plasticity node's name.
// A frozen bundle is drawn gray rather than in the population colors, so whether a pathway learns is visible on screen.
const FROZEN_RGB = [0.42, 0.42, 0.45];
function togglePathway(from, to, rule){
  if(!needShown('pathways are drawn on the network in the viewer')) return;
  const key = pathKey(from, to);
  if(activePaths.has(key)){
    viewer.removePathway(key);
    activePaths.delete(key);
    const q = pathQueue.findIndex(j => j.key === key);
    if(q >= 0) pathQueue.splice(q, 1);
    if(pathReq && pathReq.key === key){ pathReq = null; pumpPathQueue(); }
    refreshProps();
    return;
  }
  if(activePaths.size >= PATH_MAX){
    status(`showing ${PATH_MAX} pathways already; switch one off first`);
    return;
  }
  const fromHue = hueOf(from), toHue = hueOf(to);
  const frozen = String(rule === undefined || rule === null ? '' : rule).trim() === '0';
  activePaths.set(key, { from, to, fromHue, toHue, frozen });
  refreshProps();
  pathQueue.push({ from, to, key,
    rgbFrom: frozen ? FROZEN_RGB : hueRgb(fromHue),
    rgbTo: frozen ? FROZEN_RGB : hueRgb(toHue) });
  pumpPathQueue();
}
function clearPathways(){
  pathQueue.length = 0; pathReq = null;
  viewer.clearPathways(); activePaths.clear(); refreshProps();
}
// The lit set, redrawn against a new network: the drawn lines go, the names stay, and each is requested again.
// Deferred a frame so the sim worker the queries go to has been started by the caller first.
function relightPathways(){
  if(!activePaths.size) return;
  const keep = [...activePaths.values()].map(p => [p.from, p.to, p.frozen ? '0' : '']);
  clearPathways();
  setTimeout(() => { for(const [a, b, r] of keep) togglePathway(a, b, r); }, 0);
}
function neuronsMatching(net, key){
  // the same vocabulary the pair table resolves with: a population tag, E or I for the sign, * for anything
  const out = [];
  if(!net) return out;
  const tags = net.tags || {};
  for(let i = 0; i < net.count; i++){
    const t = tags[net.src[i]] || '';
    const sign = NEURON_TYPES[net.ntype[i]].sign > 0 ? 'E' : 'I';
    // a pattern with one * (src*, tier1.ray*.arm) as the pair table reads it
    if(key === '*' || key === sign || sameTag(key, t) || (key.includes('*') && tagMatch(t, key) !== null)) out.push(i);
  }
  return out;
}
function pumpPathQueue(){
  if(pathReq || !pathQueue.length) return;
  const j = pathQueue.shift();
  highlightPathway(j.from, j.to, j.key, j.rgbFrom, j.rgbTo);
  // a job that never got as far as a request must not stall the queue
  if(!pathReq) pumpPathQueue();
}
function highlightPathway(fromKey, toKey, key, rgbFrom, rgbTo){
  const net = viewer.net, eng = inspectEngine();
  if(!net || !eng){ status('no network to show the pathway in'); return; }
  if(!offers(eng, 'query', 'connection queries, so no pathway can be drawn')){ activePaths.delete(key); refreshProps(); return; }
  const pre = neuronsMatching(net, fromKey);
  const post = neuronsMatching(net, toKey);
  if(!pre.length || !post.length){
    status(`${fromKey} to ${toKey}: no synapses between these populations`);
    activePaths.delete(key); refreshProps();
    return;                       // pathReq stays null, so the queue pumps on
  }
  const postSet = new Set(post);
  // spread the sample across the population rather than taking the first few, which would all sit in one corner of the sheet
  const step = Math.max(1, Math.floor(pre.length / PATH_SOURCES));
  const picks = [];
  for(let k = 0; k < pre.length && picks.length < PATH_SOURCES; k += step) picks.push(pre[k]);
  pathReq = { key, rgbFrom, rgbTo, postSet, want:new Set(picks), pairs:[],
    sampled:picks.length, ofSources:pre.length, perSource:PATH_PER_SOURCE };
  // outgoing only: a pathway is drawn from the source's axons, and asking for the incoming side would make each of these walk the whole network
  for(const idx of picks) eng.postMessage({ cmd:'query', idx, cap:PATH_QUERY_CAP, dir:'out' });
}
// Returns true when the reply belonged to a pathway request rather than to a neuron the user clicked, so the single-neuron path does not also fire.
function pathwayReply(d){
  if(!pathReq || !pathReq.want.has(d.idx)) return false;
  pathReq.want.delete(d.idx);
  // this cell's synapses into the target, then a seeded few of them
  const into = [];
  for(const j of d.out) if(pathReq.postSet.has(j)) into.push(j);
  const keep = Math.min(into.length, pathReq.perSource);
  for(let k = 0; k < keep; k++){
    const h = (Math.imul(d.idx + 1, 2654435761) ^ Math.imul(k + 1, 2246822519)) >>> 0;
    const pick = h % into.length;
    pathReq.pairs.push(d.idx, into[pick]);
    into.splice(pick, 1);
  }
  if(d.out.length < d.outTotal) pathReq.capped = true;
  if(!pathReq.want.size){
    viewer.addPathway(pathReq.key, pathReq.pairs, pathReq.rgbFrom, pathReq.rgbTo,
      { sampled:pathReq.sampled, ofSources:pathReq.ofSources, perSource:pathReq.perSource });
    pathReq = null;
    pumpPathQueue();
  }
  return true;
}

// A training engine is a Worker or a RemoteEngine; both deliver replies as 'message' events, so one listener serves either.
// Only the two inspection replies are taken here: the run owns everything else about its own state.
globalThis.__npOnEngine = eng => {
  trainEngine = eng;
  const onMsg = e => {
    const d = e.data || e;
    if(!d || !d.cmd) return;
    if(d.cmd === 'queryResult'){
      if(!pathwayReply(d))
        viewer.applyQuery(d.idx, d.out, d.inn, d.outTotal, d.inTotal);
    }
    else if(d.cmd === 'unsupported' && d.of === 'sendV'){
      // the engine that owns the network cannot stream potentials, so the filter goes out with the reason rather than sitting there inert
      vmUnsupported = d.message || 'this engine does not stream membrane potentials';
      applyVmLock();
      status(vmUnsupported);
    }
    else if(d.cmd === 'state' && d.v)
      // potentials for the display filter.
      // The run owns the point cloud, so this only feeds the per-neuron value the shader clips on.
      viewer.setPotentials(d.v, d.fired);
    if(d.cmd === 'state' && d.vtrace && viewer.selected >= 0){
      for(const v of d.vtrace) scopeBuf.push(v);
      while(scopeBuf.length > scopeCv.width) scopeBuf.shift();
      drawScope();
    }
  };
  if(eng.addEventListener) eng.addEventListener('message', onMsg);
  else if(eng.on) eng.on('message', d => onMsg({ data:d }));
  pushVmFilter(eng);      // a filter set before the run started still applies
};
function drawScope(){
  const c = scopeCv.getContext('2d'), w = scopeCv.width, h = scopeCv.height;
  c.fillStyle = '#000'; c.fillRect(0, 0, w, h);
  const y = v => h - (v + 95) * h / 135;
  c.strokeStyle = '#333'; c.lineWidth = 1;
  for(const lv of [30, -65]){ c.beginPath(); c.moveTo(0, y(lv)); c.lineTo(w, y(lv)); c.stroke(); }
  c.strokeStyle = '#fff';
  c.beginPath();
  scopeBuf.forEach((v, i) => i ? c.lineTo(i, y(v)) : c.moveTo(i, y(v)));
  c.stroke();
  c.fillStyle = '#666'; c.font = '9px ui-monospace, Menlo, monospace';
  c.fillText('peak 30', 4, y(30) - 3);
  c.fillText('reset -65', 4, y(-65) - 3);
  const last = scopeBuf.length ? scopeBuf[scopeBuf.length-1].toFixed(0) + ' mV' : '';
  c.fillText('membrane potential  ' + last, 4, 10);
}

let worker = null, workerIsGpu = false, inFlight = false, paused = false, computeTimer = null;
let firstOpenInspect = false;   // set by the default scene on first open; the first wired network inspects one cell
let workerRuns = { kind:'cpu', host:'' };   // which engine the worker is, for planEngine
let startRefused = '';                      // why the last start ran nothing, kept for the status line the computation writes after it
let engineHello = null, helloPending = false, lackSaid = new Set();   // the running engine's hello (ENGINE.md section 2)
let refusedHello = null;   // the hello of an engine the network was refused on, so the panel can still show what it lacks
let sendInit = null;       // the init, held until the engine's hello has been checked
// ---- live-run attachment (the trainer's attach protocol, in the full UI) --
// index.html?attach=<runDir>&host=ws://... loads the run's newest brain, so the node graph sits below and every neuron is clickable, computes the identical network locally for geometry and the ins-and-outs inspector, and then streams the LIVE run's spikes from the host into the same display path.
// The local engine holds the network but is never ticked: what moves on screen is the run itself.
const ATTACH_Q = new URLSearchParams(location.search);
const ATTACH_RUN = ATTACH_Q.get('attach');
const ATTACH_HOST = ATTACH_Q.get('host') || 'ws://localhost:8801';
let attachRemote = null;
// What belongs to an open scene: the wired network, attached brains, the weight lock, undo (document.js), one per scene tab. tabs.cur is the scene in the node graph; tabs.shown is the scene whose network the viewer and the engine hold, and `doc` is that one's document.
// They differ between switching to a tab and pressing a viewer key there.
const tabs = new SceneTabs();
tabs.reset(null, '');
let doc = tabs.shown.doc;
let pendingWeights = null, fitNextCompute = false;
const io = new IORuntime(msg => status(msg, true));
// mesh files are read from the project folder, through the store the page is on (the server folder or the browser fallback)
setFileReader(async path => (await store()).read(path));
attachProbes(document.getElementById('probes'));
attachInputs(document.getElementById('inputs'));
// The input and probe overlays cover a third of the viewer at their default size.
// Each collapses to its header and resizes by the corner handle; size and state are kept per machine like the panel widths.
function overlayControls(id, def){
  const box = document.getElementById(id);
  if(!box) return;
  const key = 'np-overlay-' + id;
  // def is how the panel opens before this machine has said otherwise: the region list arrives collapsed, since it appears on its own in a scene with many populations and the pane it lands in is already crowded
  let pref = null;
  try { pref = JSON.parse(localStorage.getItem(key)); } catch(e){}
  if(!pref) pref = { ...(def || {}) };
  const save = () => { try { localStorage.setItem(key, JSON.stringify(pref)); } catch(e){} };
  const btn = box.querySelector('.ovmin');
  const apply = () => {
    box.classList.toggle('min', !!pref.min);
    if(btn){ btn.innerHTML = pref.min ? '&#9656;' : '&#9662;'; btn.title = pref.min ? 'expand this panel' : 'collapse this panel'; }
    if(!pref.min){
      if(pref.w) box.style.width = pref.w + 'px';
      if(pref.h) box.style.height = pref.h + 'px';
    }
  };
  if(btn) btn.onclick = e => { e.stopPropagation(); pref.min = !pref.min; apply(); save(); };
  // A drag on the corner handle ends in a resize event on the element.
  // Only a size reached while the pointer is down on the panel is remembered: the layout resizes the panel too, as rows arrive, and remembering that would pin its height.
  let dragging = false;
  box.addEventListener('pointerdown', () => { dragging = true; });
  window.addEventListener('pointerup', () => { dragging = false; });
  new ResizeObserver(() => {
    if(!dragging || pref.min || box.style.display === 'none') return;
    pref.w = Math.round(box.offsetWidth); pref.h = Math.round(box.offsetHeight); save();
  }).observe(box);
  apply();
}
overlayControls('inputs'); overlayControls('probes'); overlayControls('regions', { min:true });
let rateAcc = { spikes:0, steps:0, exc:0 }, lastTickAt = 0;
// which cells are excitatory, for the rate per class the compensation targets; built once per network
let signNet = null, signE = null, signCount = 0;
function excitatoryMask(net){
  if(signNet !== net){
    signNet = net; signCount = 0;
    signE = Uint8Array.from(net.ntype, t => NEURON_TYPES[t] && NEURON_TYPES[t].sign > 0 ? 1 : 0);
    for(const s of signE) signCount += s;
  }
  return signE;
}
let lastAuthoredSteps = 2;   // see setSpeed: distinguishes an author's change from a viewer's

function status(msg, err){ statusEl.textContent = msg; statusEl.className = err ? 'error' : ''; }

// node === null means a rename: it changes nothing a computation can see, so it redraws and saves without invalidating wiring that may have taken minutes
function onParamEdit(node){
  if(!node){ editor.draw(); saveGraph(); return; }
  noteEdit(node);
  editor.markDirty(node); scheduleCompute();
}

// ---- recording: what the probes read and what was done, for as long as REC is on ----
// (src/recording.js).
// The edits are found by comparing a node's settings with the copy taken at the last edit, so every path that changes a setting through onParamEdit is logged without each control saying so.
let recording = null, lastRecording = null;
const paramShadow = new Map();
const recEl = document.getElementById('rec');
function noteEdit(node){
  if(!recording || !node || tabs.cur !== tabs.shown) return;
  const before = paramShadow.get(node.id) || {}, after = node.params || {};
  for(const k of changedKeys(before, after)) recording.edit(node.name || node.type, k, before[k], after[k]);
  paramShadow.set(node.id, structuredClone(after));
}
// the engine's weights now, summarized; null when nothing answers
function weightsNow(){
  if(!worker || !offers(worker, 'getWeights', 'weight readback')) return Promise.resolve(null);
  return new Promise(res => {
    const timer = setTimeout(() => { if(pendingWeights === take) pendingWeights = null; res(null); }, 8000);
    const take = d => { clearTimeout(timer); res(weightSummary(d.w)); };
    pendingWeights = take; worker.postMessage({ cmd:'getWeights' });
  });
}
async function startRecording(){
  if(!worker || !doc.curNet){ status('nothing is running to record: compute a scene first', true); return; }
  const name = editor.scenarioName || sceneLabel(activeScene || '') || 'scene';
  recording = new Recording(name, doc.curNet.probes || []);
  paramShadow.clear();
  for(const n of editor.nodes) paramShadow.set(n.id, structuredClone(n.params || {}));
  recEl.classList.add('on'); recEl.textContent = 'REC 0 s';
  recording.weights.before = await weightsNow();
  status('recording every probe rate and every edit at its simulated time, until REC is pressed again');
}
async function stopRecording(){
  const r = recording; if(!r) return;
  recEl.textContent = 'REC saving';
  r.weights.after = await weightsNow();
  recording = null; lastRecording = r;
  recEl.classList.remove('on'); recEl.textContent = 'REC';
  r.intervals = r.labels().map(l => { const iv = recorder.intervals(l); return { label:l, counts:Array.from(iv.counts), edges:Array.from(iv.edges), total:iv.total, ceiling:iv.ceiling }; });
  const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
  const base = 'recordings/' + layoutSlug(r.scene) + '-' + stamp;
  const json = JSON.stringify(r.toJSON()), table = r.csv();
  const said = r.seconds().toFixed(1) + ' s of simulated time, ' + r.edits() + ' edit' + (r.edits() === 1 ? '' : 's');
  try {
    const st = await store();
    await st.write(base + '.json', json); await st.write(base + '.csv', table);
    status('recording saved to ' + base + '.json and .csv: ' + said);
  } catch(e){
    downloadText(base.split('/').pop() + '.json', json); downloadText(base.split('/').pop() + '.csv', table);
    status('recording downloaded (' + said + '); the project folder refused the write: ' + e.message, true);
  }
  if(propsNode && propsNode.type === 'chart') showProps(propsNode);
}
recEl.onclick = () => { if(recording) stopRecording(); else startRecording(); };

// What a saved figure or data file is: the scene, the plot, the population, the simulated time, the network it came from and when.
// Written into the PNG's text chunks, the SVG's metadata and the .json beside a CSV, and stamped under the figure as a caption unless the node says otherwise.
function figureInfo(node){
  const plot = NODE_DEFS.chart.params.find(p => p.k === 'plot').options[node.params.plot | 0];
  // a scenario opened from the address bar is not the open file; name the scenario
  const scene = (urlScene && editor.scenarioName) || sceneLabel(activeScene || '') || 'scene';
  const net = doc.curNet;
  // a sweep is several runs: the caption says how many, since a mean of two runs and a mean of ten are different claims
  const src = String(node.params.source || '').trim();
  const gs = src ? sweepGroups.get(src) : null;
  return { app:'linen', scene, plot, pop:String(node.params.pop || '').trim(), run:src || undefined,
    runs: gs ? gs.reduce((n, g) => n + g.histories.length, 0) : undefined,
    runGroups: gs ? gs.length : undefined,
    simSeconds:+(recorder.t/1000).toFixed(3), neurons:net ? net.count : undefined, synapses:net ? net.synCount : undefined,
    resolutionPercent:+document.getElementById('res').value, savedAt:new Date().toISOString() };
}
function figureCaption(node){
  const i = figureInfo(node), smooth = +node.params.smooth > 0 && ((node.params.plot|0) === 7 || (node.params.plot|0) === 11);
  const runText = !i.run ? '' : i.runs
    ? ' · ' + i.run + ', ' + i.runs + ' runs in ' + i.runGroups + ' group' + (i.runGroups === 1 ? '' : 's') + ', mean and range'
    : ' · run ' + i.run;
  return ['linen · ' + i.scene + ' · ' + i.plot + (i.pop ? ' · ' + i.pop : '') + runText + (smooth ? ' · moving average ' + (+node.params.smooth) + ' ms' : ''),
    'sim ' + i.simSeconds.toFixed(1) + ' s' + (i.neurons ? ' · ' + i.neurons.toLocaleString() + ' neurons, ' + (i.synapses || 0).toLocaleString() + ' synapses' : '') + ' · ' + i.savedAt.slice(0, 16).replace('T', ' ') + ' UTC'];
}
setPropsCaption(figureCaption);
// into the project's figures folder; a download when there is no folder to write to
async function saveFigureFile(name, blob){
  try { const st = await store(); await st.write(P.figurePath(name), blob instanceof Blob ? new Uint8Array(await blob.arrayBuffer()) : blob); return P.figurePath(name); }
  catch(e){ download(name, blob); return name + ' (downloaded: ' + e.message + ')'; }
}

// props-panel action buttons (read node's folder picker); the graph keeps only the path string, the granted handle lives in IndexedDB
setPropsAction(async (action, node) => {
  if(action === 'chartPng' || action === 'chartCsv'){
    const Q = node.params, info = figureInfo(node);
    const base = [info.scene, info.plot, info.pop];
    if(action === 'chartPng'){
      const F = await import('./chartpaint.js');
      const spec = F.figureSpec(Q), fmt = Q.format | 0;
      const cap = figureCaption(node), lines = (Q.caption | 0) === 0 ? cap : [];
      const SvgMod = fmt !== 0 ? await import('./svgctx.js') : null;
      const src = chartSources(node);
      const meta = { ...info, figure:{ widthMm:spec.mmW, heightMm:spec.mmH, dpi:spec.dpi, fontPt:spec.style.fontPx, linePt:spec.style.lineW }, settings:structuredClone(Q) };
      const saved = [], empty = r => { status('nothing to draw yet: ' + ((r && r.note) || 'the plot has no data'), true); };
      // Both formats are drawn before either is encoded: drawing does not yield to the engine, so a PNG and an SVG saved together from a running network show the same moment.
      let cv = null, svg = null;
      if(fmt !== 1){
        if(spec.tooBig){ status(`a figure of ${spec.pxW} x ${spec.pxH} pixels is more than a canvas holds: lower the resolution or the size, or save SVG`, true); return; }
        cv = document.createElement('canvas'); cv.width = spec.pxW; cv.height = spec.pxH;
        const ctx = cv.getContext('2d'); ctx.setTransform(spec.k, 0, 0, spec.k, 0, 0);
        const r = await F.paintFigure(ctx, spec, Q, src, lines);
        if(!r || !r.chart) return empty(r);
      }
      if(fmt !== 0){
        svg = new SvgMod.SvgContext(spec.wPt, spec.hPt, { widthMm:spec.mmW, heightMm:spec.mmH,
          title:info.plot + (info.pop ? ', ' + info.pop : ''), desc:cap.join(' | '), meta });
        const r = await F.paintFigure(svg, spec, Q, src, lines);
        if(!r || !r.chart) return empty(r);
      }
      if(cv){
        const png = withMeta(await canvasPng(cv), { dpi:spec.dpi, text:{ Title:info.plot + (info.pop ? ', ' + info.pop : ''),
          Description:cap.join(' | '), Software:'linen', 'Creation Time':info.savedAt, Source:info.scene, Comment:JSON.stringify(meta) } });
        saved.push(await saveFigureFile(figureName(base, 'png'), new Blob([png], { type:'image/png' })));
      }
      if(svg) saved.push(await saveFigureFile(figureName(base, 'svg'), new Blob([svg.toString()], { type:'image/svg+xml' })));
      status('saved ' + saved.join(' and ') + (fmt !== 1 ? ` (${spec.pxW} x ${spec.pxH} px, ${spec.mmW} x ${spec.mmH} mm at ${spec.dpi} dpi)` : ` (${spec.mmW} x ${spec.mmH} mm)`));
      return;
    }
    const ch = node && node._chart;
    if(!ch){ status('nothing drawn yet: open the chart and let it paint', true); return; }
    const pop = info.pop;
    let text = null, extra = '';
    if(ch.kind === 'series') text = ch.xName ? csv([ch.xName, ch.yName], ch.points) : seriesCsv(ch.points, pop || 'value');
    else if(ch.kind === 'multi') text = multiSeriesCsv(ch.series);
    else if(ch.kind === 'bands') text = bandSeriesCsv(ch.series, ch.xName, ch.yName);
    else if(ch.kind === 'raster'){
      text = rasterCsv(ch.events, ch.cells);
      await saveFigureFile(figureName([...base, 'spiketrains'], 'txt'), new Blob([spikeTrainsAscii(ch.events, ch.rows)], { type:'text/plain' }));
      extra = ' and the spike trains';
    }
    else if(ch.kind === 'hist') text = histogramCsv(ch.counts, ch.edges);
    else if(ch.kind === 'trace') text = seriesCsv(ch.points, 'mv');
    else if(ch.kind === 'recording') text = ch.csv();
    else if(ch.kind === 'matrix') text = matrixCsv(ch.values, ch.rows, ch.cols);
    if(text === null){ status('this plot has no data to save yet', true); return; }
    const where = await saveFigureFile(figureName(base, 'csv'), new Blob([text], { type:'text/csv' }));
    // what the numbers are, beside them: a CSV has no place for it
    await saveFigureFile(figureName(base, 'json'), new Blob([JSON.stringify({ ...info, data:figureName(base, 'csv'), note:ch.smoothMs ? 'the data is as recorded; the figure draws a moving average of ' + ch.smoothMs + ' ms' : undefined, settings:structuredClone(Q) }, null, 1)], { type:'application/json' }));
    status('saved ' + where + extra + ' with a .json beside it saying what it is'); return;
  }
  if(action === 'exportNet' || action === 'exportBrian'){
    if(!needShown('export')) return;
    const net = doc.curNet;
    if(!net || net.kind !== 'net' || !net.count){ status('nothing is wired to export', true); return; }
    if(net.synCount && net.post.byteLength === 0){
      status('the synapse arrays were handed to the engine rather than copied (a network this size is); edit connect to rewire, then export before starting', true); return;
    }
    const scene = (urlScene && editor.scenarioName) || sceneLabel(activeScene || '') || 'scene';
    if(action === 'exportNet'){
      const a = figureName([scene, 'neurons'], 'csv'), b = figureName([scene, 'synapses'], 'csv');
      downloadText(a, neuronsCsv(net, NEURON_TYPES));
      downloadText(b, synapsesCsv(net));
      status('saved ' + a + ' (' + net.count.toLocaleString() + ' rows) and ' + b + ' (' + net.synCount.toLocaleString() + ' rows)');
      return;
    }
    const ms = Math.max(10, Math.round((+node.params.exportSecs || 2)*1000));
    const c = caseFromInit(initMessage(net), { ms });
    const json = JSON.stringify(c);
    const name = figureName([scene, 'brian', 'case'], 'json');
    downloadText(name, json);
    let py = null;
    try { py = await (await fetch(new URL('../tools/brian_ref.py', import.meta.url))).text(); } catch(e){}
    if(py && /brian2/.test(py)) downloadText('brian_ref.py', py); else py = null;
    status('saved ' + name + ' (' + (json.length/1e6).toFixed(1) + ' MB, ' + ms + ' ms)' + (py ? ' and brian_ref.py' : '') +
      ': python brian_ref.py ' + name + ' out.json, with Brian 2 and NumPy');
    return;
  }
  if(action === 'brainSave'){ if(needShown('save brain')) saveBrain(); return; }
  if(action === 'brainLoad'){ pickBrainFile(); return; }
  if(action === 'runs'){
    const pane = document.getElementById('trainPane');
    pane.style.display = 'flex';
    document.getElementById('tsplit').style.display = 'block';
    applyPanels();
    if(!trainingRunning) document.getElementById('trainHint').textContent = 'browsing runs';
    const tab = document.querySelector('#ttabs .ttab[data-pane="files"]');
    if(tab) tab.click();
    try {
      const ST = await store();
      const n = await renderRuns(document.getElementById('files'), ST,
        { whereText: await whereLabel(), onChart:globalThis.__npChartRun,
          onError:msg => status(msg, true) });
      status(n ? `${n} run(s) listed: the chart link on a metrics file draws that run`
        : 'no runs in this project folder yet');
    } catch(e){ status('could not list runs: ' + e.message, true); }
    return;
  }
  if(action === 'train'){
    if(!needShown('training')) return;
    if(trainingRunning){
      status('a training run is already going in this page', true);
      return;
    }
    openTrainDialog(node);
    return;
  }
  if(action === 'meshReload'){
    // the parsed file is cached per path until asked; a re-export of the mesh is picked up here, then the node computes again
    forgetMesh(String(node.params.file || '').trim());
    editor.markDirty(node); scheduleCompute();
    status('mesh: ' + (node.params.file || '(no file)') + ' will be read again');
    return;
  }
  if(action === 'connectionsReload'){
    forgetConnections();
    editor.markDirty(node); scheduleCompute();
    status('connections file: ' + (node.params.file || '(no table)') + ' will be read again');
    return;
  }
  if(action === 'expandPops'){
    // the node's own compute says which populations it has; compute it here if a collapse or an edit just emptied its cache
    let cache = node._cache;
    if(!cache){ try { cache = await computeNode(node, id => editor.nodes.find(n => n.id === id)); } catch(e){ status('expand: ' + (e.message || e), true); return; } }
    const tags = cache && cache.tags ? [...new Set(Object.values(cache.tags))].filter(Boolean) : [];
    if(!tags.length){ status('expand: this node has no populations (set a population column)', true); return; }
    const { merge } = expandPopulations(editor, node, tags);
    editor.markDirty(merge); scheduleCompute(); showProps(node);
    status('expanded ' + (node.name || 'points file') + ' into ' + tags.length + ' population nodes and a merge');
    return;
  }
  if(action === 'collapsePops'){
    const fan = collapsePopulations(editor, node);
    if(!fan.merge){ status('collapse: ' + fan.why, true); return; }
    // the file node's compute is unchanged; what read the merge now reads it
    for(const n of editor.nodes) if(n.inputs && n.inputs.some(c => c && c.id === node.id)) editor.markDirty(n);
    scheduleCompute(); showProps(node);
    status('collapsed ' + fan.kids.length + ' population nodes and their merge back into ' + (node.name || 'the points file'));
    return;
  }
  if(action === 'pointsReload'){
    forgetPoints();
    editor.markDirty(node); scheduleCompute();
    status('points file: ' + (node.params.file || '(no file)') + ' will be read again');
    return;
  }
  if(action !== 'chooseFolder') return;
  try {
    node.params.path = await chooseDir();
    showProps(node);
    onParamEdit(node);
  } catch(e){
    if(e && e.name === 'AbortError') return;
    status('choose folder: ' + (e.message || e), true);
  }
});

// ---- undo/redo: snapshot history of the whole graph ----
function pushHistory(){ tabs.cur.doc.pushHistory(JSON.stringify(editor.toJSON())); }
function timeTravel(di){
  const d = tabs.cur.doc.travel(di);              // the graph, without its camera
  if(!d) return;
  editor.load(d);
  showProps(null);
  recompute(); saveGraph();
}
window.addEventListener('keydown', e => {
  if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
  if(keyMatches(e, 'redo')){
    e.preventDefault(); timeTravel(+1);
  } else if(keyMatches(e, 'undo')){
    e.preventDefault(); timeTravel(e.shiftKey ? +1 : -1);
  } else if(/^[1-9]$/.test(e.key)){
    // viewer slots: with a node selected the digit binds to it and views it; with nothing selected the digit recalls its bound node
    const d = e.key;
    if(editor.sel && editor.sel.type !== 'note'){
      for(const k in editor.viewSlots)
        if(editor.viewSlots[k] === editor.sel.id) delete editor.viewSlots[k];
      editor.viewSlots[d] = editor.sel.id;
      editor.activeView = editor.sel.id;
      if(makeShown()) recompute();
    } else if(editor.viewSlots[d] && (editor.activeView !== editor.viewSlots[d] || tabs.cur !== tabs.shown)){
      editor.activeView = editor.viewSlots[d];
      if(makeShown()) recompute();
    }
  }
});

// The population boxes and the list of them, kept here because the button on the viewer strip, the computation and the REGIONS panel all read the same two facts: whether boxes are on, and which populations the scene has.
let popBoxesOn = false, popBoxList = [];
// Above a handful of populations the names over the boxes stop being a list you can read, and the panel becomes the list instead.
const NAMES_FIT = 6;
// Every population in the scene, in its color, with its count.
// A row lights its box in the viewer; clicking one selects the node that made the population, which is what clicking its name in the viewer does.
function buildRegionsPanel(){
  const panel = document.getElementById('regions');
  if(!panel) return;
  const body = panel.querySelector('.ovbody');
  const show = popBoxesOn && popBoxList.length > NAMES_FIT;
  panel.style.display = show ? 'block' : 'none';
  body.innerHTML = '';
  if(!show) return;
  for(const b of [...popBoxList].sort((p, q) => p.tag.localeCompare(q.tag))){
    const row = document.createElement('div');
    row.className = 'rrow'; row.dataset.tag = b.tag;
    row.title = 'select the ' + b.tag + ' node';
    const sw = document.createElement('span'); sw.className = 'rsw';
    sw.style.background = hueCss(b.hue);
    const tag = document.createElement('span'); tag.className = 'rtag'; tag.textContent = b.tag;
    const ct = document.createElement('span'); ct.className = 'rct';
    ct.textContent = b.count.toLocaleString();
    row.append(sw, tag, ct);
    row.onmouseenter = () => viewer.hoverPopulation(b.tag);
    row.onmouseleave = () => viewer.hoverPopulation('');
    row.onclick = () => viewer.onPickPopulation(b.src, b.tag);
    body.appendChild(row);
  }
}

{
  const btn = document.getElementById('popboxes');
  const KEY = 'np-pop-boxes';
  popBoxesOn = localStorage.getItem(KEY) === '1';
  const apply = () => {
    viewer.showPopulationBoxes(popBoxesOn);
    btn.classList.toggle('on', popBoxesOn);
    buildRegionsPanel();
  };
  btn.onclick = () => {
    popBoxesOn = !popBoxesOn;
    try { localStorage.setItem(KEY, popBoxesOn ? '1' : '0'); } catch(e){}
    apply();
    status(!popBoxesOn ? 'population boxes off'
      : popBoxList.length > NAMES_FIT
        ? popBoxList.length + ' regions, boxed in their colors and named where the names fit. Point at a box to read its name; the REGIONS panel lists them all.'
        : 'drawing a box around each population, named above it; click a name to select its node');
  };
  apply();
}
// The slicer box: off, on with its handles, on with the handles hidden.
// One button cycles off to on-with-handles; escape (the gizmo key) hides the handles and keeps the cut; the button on a handled box turns the cut off, and on an unhandled one brings the handles back.
{
  const btn = document.getElementById('slicebtn');
  btn.onclick = () => {
    const on = !!(sceneSlice && sceneSlice.on);
    if(!on){
      if(!sceneSlice) sceneSlice = sliceDefault(viewer.cloudPos);   // half the tissue along x, the first time
      else sceneSlice.on = true;
      viewer.setSlice(sceneSlice);
      viewer.editSlice(sceneSlice, () => saveGraph());
      status('slice: cells outside the box are hidden, not removed. Drag to move, scale mode to resize, escape hides the handles, SLICE again turns it off');
    } else if(viewer.sliceEditing){
      viewer.closeGizmo();
      sceneSlice.on = false;
      viewer.setSlice(sceneSlice);
      status('slice off; the box is kept for next time');
    } else {
      viewer.editSlice(sceneSlice, () => saveGraph());
      status('slice handles shown');
    }
    btn.classList.toggle('on', !!(sceneSlice && sceneSlice.on));
    saveGraph();
  };
}
// Color by population is a per-machine preference rather than part of the document, like the other viewer controls.
{
  const btn = document.getElementById('popcolor');
  const KEY = 'np-color-by';
  let mode = localStorage.getItem(KEY) === 'population' ? 'population' : 'type';
  const apply = () => {
    viewer.setColorBy(mode);
    btn.classList.toggle('on', mode === 'population');
    btn.setAttribute('aria-label', 'color by population');
  };
  btn.onclick = () => {
    mode = mode === 'population' ? 'type' : 'population';
    try { localStorage.setItem(KEY, mode); } catch(e){}
    apply();
    status(mode === 'population'
      ? 'coloring by population: each tag wears the hue it has in the pair table'
      : 'coloring by cell type');
  };
  apply();
}

{
  const btn = document.getElementById('spikestyle');
  const KEY = 'np-spike-style';
  const STATE = {
    bright: ['bright', 'spikes flash bright',
      'a firing cell flashes toward white. Click to flash each cell in its own color from a gray rest instead, which shows which population is firing when populations coloring is on.'],
    sat: ['sat', 'spikes flash color',
      'cells rest as dim gray and flash in their own color. Click to flash toward white instead.'] };
  let s = localStorage.getItem(KEY) === 'sat' ? 'sat' : 'bright';
  const apply = () => {
    viewer.setSpikeStyle(s);
    const [ic, name, tip] = STATE[s];
    btn.innerHTML = icon(ic);
    btn.classList.add('icb', 'icon-only');
    btn.classList.toggle('on', s === 'sat');
    btn.setAttribute('aria-label', name);
    btn.title = tip;
  };
  btn.onclick = () => {
    s = s === 'sat' ? 'bright' : 'sat';
    try { localStorage.setItem(KEY, s); } catch(e){}
    apply();
    status(s === 'sat' ? 'spikes flash in their color from a gray rest'
      : 'spikes flash toward white');
  };
  apply();
}
{
  const btn = document.getElementById('spikesign');
  const KEY = 'np-spike-sign';
  const STATE = {
    '0': ['spikeEI', 'all spikes',
      'excitatory and inhibitory spikes are both shown. Click to show excitatory spikes only. Display only: the simulation and the raster are unchanged.'],
    '1': ['spikeE', 'excitatory spikes only',
      'only excitatory cells light up when they fire. Click to show inhibitory spikes only.'],
    '-1': ['spikeI', 'inhibitory spikes only',
      'only inhibitory cells light up when they fire. Click to show every spike.'] };
  let s = +localStorage.getItem(KEY) || 0;
  if(s !== 1 && s !== -1) s = 0;
  const apply = () => {
    viewer.setSpikeSign(s);
    const [ic, name, tip] = STATE[s];
    btn.innerHTML = icon(ic);
    btn.classList.add('icb', 'icon-only');
    btn.classList.toggle('on', s !== 0);
    btn.setAttribute('aria-label', name);
    btn.title = tip;
  };
  btn.onclick = () => {
    s = s === 0 ? 1 : s === 1 ? -1 : 0;
    try { localStorage.setItem(KEY, String(s)); } catch(e){}
    apply();
    status(s === 0 ? 'showing every spike'
      : s === 1 ? 'showing excitatory spikes only' : 'showing inhibitory spikes only');
  };
  apply();
}

const STORE = 'neuron-playground-graph-v2';   // v2: spatial units are micrometers
// a graph brought in by the address bar is not the open scene's content
let urlScene = false;
// True until the startup graph is on screen: the project's node modules load first (a scene may need them), and an autosave in that window would write the empty editor over the scene file.
let booting = true;
// The scene the buffer was saved for.
// Without it the buffer is a graph with no owner, and a startup that trusts it under whatever name the project holds writes one scene's content over another.
const STORE_FOR = STORE + '-for';
// The slicer box is part of the scene (slice.js): null when the scene has none, else { on, center, size } in micrometers.
let sceneSlice = null;
function applySlice(s){
  sceneSlice = sliceNormalize(s);
  if(viewer.sliceEditing) viewer.closeGizmo();
  viewer.setSlice(sceneSlice);
  const btn = document.getElementById('slicebtn');
  if(btn) btn.classList.toggle('on', !!(sceneSlice && sceneSlice.on));
}
// The viewer-side settings of the scene in the editor.
// The live ones (the slider, the slicer box, the locked hues) are the shown scene's; a tab that is not shown keeps its own on the tab.
const sceneSettings = () => tabs.cur !== tabs.shown && tabs.cur.held ? tabs.cur.held
  : { resolution: +document.getElementById('res').value, slice: sceneSlice, hues: sceneHues };
function setSceneSettings(v){
  if(tabs.cur === tabs.shown){
    if(v.resolution) applyResolution(v.resolution, false);
    if('slice' in v) applySlice(v.slice);
    if('hues' in v) applyHues(v.hues);
  } else tabs.cur.held = { resolution:100, slice:null, hues:{}, ...(tabs.cur.held || {}), ...v };
}
const sceneJSON = () => { const v = sceneSettings();
  return JSON.stringify({ ...editor.toJSON(), resolution: v.resolution, slice: v.slice, hues: v.hues }, null, 1); };
function saveGraph(){
  if(urlScene || booting) return;      // an address-bar graph is not a scene; before the first scene is on screen there is nothing to save
  try {
    localStorage.setItem(STORE, sceneJSON());
    localStorage.setItem(STORE_FOR, activeScene);
  } catch(e){}
  scheduleSceneSave();
}
setInterval(() => { saveGraph(); }, 2000);   // catches node moves / view pans

// The working graph as a file in the project, not only in this browser.
//
// localStorage stays the instant buffer: it survives a reload with no server running and costs nothing.
// The file is what makes the scene part of the project, so it can be opened somewhere else, copied, or diffed, and it is what the scene list in project.json points at.
//
// Written only when the graph actually changed.
// The interval fires whether or not anything moved, and a scene is small but a write is not free, so an idle tab should not be touching the disk every few seconds.
// A project has scenes and one of them is open.
// Which one is recorded in project.json, so it survives a reload and is the same answer for anything else that opens the folder.
//
// There is deliberately no save button and no dirty state.
// Edits go into the open scene continuously, the way the graph already went into localStorage, so opening another scene cannot lose anything and does not have to ask.
// SAVE AS therefore means fork: copy what is on screen to a new name and continue in the copy, leaving the original where it was.
let activeScene = P.scenePath('main.json');
const sceneLabel = f => String(f).replace(/^scenes\//, '').replace(/\.json$/, '');
// the scene in the editor becomes this file: the name in the bar, the tab
function retarget(file){
  activeScene = file;
  tabs.cur.file = file; tabs.cur.label = sceneLabel(file);
  urlScene = false;
  renderTabs();
}
tabs.cur.file = activeScene; tabs.cur.label = sceneLabel(activeScene);
// A scene whose node modules are not loaded is refused whole (checkScenePlugins in plugins.js, called by editor.load).
// What is on screen is then not that scene, so the tab lets go of the file and nothing is saved over it until another scene is opened.
function refuseScene(e){
  sceneSwitching = false;
  editor.clear(); showProps(null);
  urlScene = true; tabs.cur.file = null; tabs.cur.label = 'scene not opened';
  renderTabs(); showSceneName();
  status(e.message || String(e), true);
}
// Why a scene file cannot be opened for want of its node modules, or null (a missing or unreadable file is another question, answered where it is opened).
async function sceneModulesMissing(st, file){
  try {
    const bytes = await st.read(file);
    return bytes ? scenePluginsMissing(JSON.parse(new TextDecoder().decode(bytes))) : null;
  } catch(e){ return null; }
}
// One dropdown, shared by both halves of the path.
// Same black, same 1px, and the same .mitem rows as the graph's right-click menu, because a second menu that looks like a different app is worse than no menu.
let barOpen = null;
function closeBarMenu(){
  if(!barOpen) return;
  barOpen.el.remove();
  if(barOpen.anchor) barOpen.anchor.classList.remove('open');
  barOpen = null;
}
addEventListener('mousedown', e => {
  if(barOpen && !barOpen.el.contains(e.target) && e.target !== barOpen.anchor)
    closeBarMenu();
}, true);
addEventListener('keydown', e => { if(e.key === 'Escape') closeBarMenu(); });

// items: {label, note, on, off, run} | {head} | null for a rule
function barMenu(anchor, items){
  if(barOpen && barOpen.anchor === anchor){ closeBarMenu(); return; }
  closeBarMenu();
  const m = document.createElement('div');
  m.id = 'barmenu';
  for(const it of items){
    if(!it){ const s = document.createElement('div'); s.className = 'msep';
      m.appendChild(s); continue; }
    if(it.head !== undefined){
      const h = document.createElement('div');
      h.className = 'mhead'; h.textContent = it.head; h.title = it.head;
      m.appendChild(h); continue;
    }
    const el = document.createElement('div');
    el.className = 'mitem' + (it.off ? ' off' : '') + (it.on ? ' on' : '');
    el.appendChild(document.createTextNode(it.label));
    if(it.note){
      const k = document.createElement('span');
      k.className = 'k'; k.textContent = it.note; el.appendChild(k);
    }
    if(!it.off) el.onclick = () => { closeBarMenu(); it.run(); };
    m.appendChild(el);
  }
  document.body.appendChild(m);
  anchor.classList.add('open');
  const a = anchor.getBoundingClientRect(), r = m.getBoundingClientRect();
  // under the button, or over it when the button sits low on the page (the scene tabs do, in the layouts with the graph at the bottom)
  m.style.top = (a.bottom + r.height > innerHeight - 4 ? Math.max(4, a.top - r.height) : a.bottom) + 'px';
  { const tip = document.getElementById('tip'); if(tip) tip.hidden = true; }
  // right-aligned to the button, then pulled back on screen if it would run off the edge, which is the common case for a bar sitting at the right
  m.style.left = Math.max(4, Math.min(a.left, innerWidth - r.width - 4)) + 'px';
  barOpen = { el:m, anchor };
}

function showSceneName(){
  const el = document.getElementById('scenebtn');
  if(el) el.textContent = tabs.cur.file ? sceneLabel(activeScene) : (tabs.cur.label || sceneLabel(activeScene));   // a graph from the address bar is named by its tab
}
function showProjName(root){
  const el = document.getElementById('projbtn');
  if(!el) return;
  el.textContent = root
    ? String(root).replace(/[\\/]+$/, '').split(/[\\/]/).pop()
    : 'browser storage';
  el.title = root ? 'project folder: ' + root
    : 'no project folder yet: files go to browser storage. Pick a folder from this menu.';
}
// The hosted site: no server, so a project is a folder picked in the browser.
// Chrome and Edge have the picker; elsewhere the menu says so.
const hasFolderPicker = () => typeof window.showDirectoryPicker === 'function';
async function pickProjectFolder(fresh){
  let h;
  try { h = await chooseProjectDir(); }
  catch(e){
    if(e && e.name !== 'AbortError'){
      status(String(e.message || e), true);
      // the node that failed pulses red in the graph until a computation succeeds
      editor.errorNode = e && e.nodeId !== undefined && e.nodeId >= 0 ? e.nodeId : null;
      editor.draw();
    }
    return; }
  const st = useFolder(h);
  rememberProject(h.name);
  projMode = fresh ? 'new' : 'open';
  await adoptProject(h.name, async () => st);
  status('project folder ' + h.name + ' (through the browser)');
}
// A folder remembered from an earlier visit needs one click to let the page back in; the menu offers it.
async function reopenRememberedFolder(h){
  if(!(await grantProjectDir(h))){ status('no access to folder ' + h.name, true); return; }
  const st = useFolder(h);
  projMode = 'open';
  await adoptProject(h.name, async () => st);
}

let lastSceneJson = null, sceneBusy = false, sceneSwitching = false;
async function saveScene(){
  if(sceneBusy || urlScene || sceneSwitching || booting) return;
  const json = sceneJSON();
  if(json === lastSceneJson) return;
  sceneBusy = true;
  try {
    const st = await store();
    await st.write(activeScene, json);
    lastSceneJson = json;
    const proj = await P.openProject(st);
    const open = tabs.openFiles();
    if(!proj.scenes.some(sc => sc.file === activeScene) ||
       proj.activeScene !== activeScene || JSON.stringify(proj.openScenes || []) !== JSON.stringify(open)){
      P.addScene(proj, activeScene);
      proj.activeScene = activeScene;
      proj.openScenes = open;
      await P.writeProject(st, proj);
    }
  } catch(e){
    // The browser copy is current either way, so a failed write is not worth interrupting anyone over.
    // It retries on the next tick by construction, since lastSceneJson was not advanced.
  } finally { sceneBusy = false; }
}
// The file follows an edit rather than a clock.
// A short delay coalesces the many parameter events a slider drag produces into one write, and the timer stays as a backstop for changes that do not run through onParamEdit, such as dragging a node or panning the view.
// A file written only on the five second tick would lose an edit made within a few seconds of a reload.
let sceneSaveTimer = null;
function scheduleSceneSave(){
  if(sceneSaveTimer) clearTimeout(sceneSaveTimer);
  sceneSaveTimer = setTimeout(() => { sceneSaveTimer = null; saveScene(); }, 400);
}
setInterval(saveScene, 5000);
// A reload or a closed tab must not lose the last edit. pagehide fires in cases beforeunload does not, and hiding the tab is the common case for a browser that discards it later.
addEventListener('pagehide', () => { saveScene(); });
addEventListener('visibilitychange', () => { if(document.hidden) saveScene(); });

// A name has to survive being a filename and a path segment.
// Refusing the awkward ones here means the message says what is wrong with the name, rather than a write failing later with something about paths.
const SCENE_NAME = /^[A-Za-z0-9][A-Za-z0-9 _.-]{0,59}$/;
function sceneNameError(name){
  if(!name) return 'give it a name';
  if(!SCENE_NAME.test(name))
    return 'letters, digits, spaces, dot, dash and underscore only, starting with a letter or digit';
  return null;
}

async function listScenes(){
  const st = await store();
  const proj = await P.openProject(st);
  const onDisk = (await st.list(P.scenesDir())).filter(e => !e.dir &&
    e.name.endsWith('.json'));
  // project.json gives the order; anything on disk it has not heard of is still listed, because a file someone dropped in the folder is a scene whether or not the index knows about it yet.
  const seen = new Set();
  const out = [];
  for(const s of proj.scenes){
    const hit = onDisk.find(e => P.scenePath(e.name) === s.file);
    if(!hit) continue;
    seen.add(hit.name);
    out.push({ file:s.file, name:sceneLabel(s.file), size:hit.size, mtime:hit.mtime });
  }
  for(const e of onDisk){
    if(seen.has(e.name)) continue;
    out.push({ file:P.scenePath(e.name), name:sceneLabel(e.name),
      size:e.size, mtime:e.mtime });
  }
  return out;
}

// Switch the open scene.
// The autosave is pointed at the new file before anything is loaded, so a tick landing mid-switch cannot write the old graph into the new file.
async function setActiveScene(file){
  retarget(file);
  lastSceneJson = null;
  showSceneName();
  await persistProject();
}
// project.json: the scene in the editor and the tabs that are open
async function persistProject(){
  if(!tabs.cur.file) return;
  try {
    const st = await store();
    const proj = await P.openProject(st);
    for(const f of tabs.openFiles()) P.addScene(proj, f);
    proj.activeScene = tabs.cur.file;
    proj.openScenes = tabs.openFiles();
    await P.writeProject(st, proj);
  } catch(e){ /* the name in the bar is still right */ }
}

// ---- scene tabs ----------------------------------------------------------
// Opening a scene opens a tab, or goes to the one it already has.
// Switching tabs changes what the node graph edits and nothing else: the viewer and the engine keep the scene they hold until a viewer key (1 to 9) is pressed in the new tab, which is what viewing a node has always meant.
const viewHint = () => 'the viewer still shows ' + (tabs.shown.label || 'the other scene') + ': press a viewer key (1) here to view this one';
// for what needs the network of the scene in the editor
function needShown(what){
  if(tabs.cur === tabs.shown) return true;
  status(what + ': ' + viewHint(), true);
  return false;
}
async function openScene(file){
  return switchTab(tabs.byFile(file) || tabs.add(file, sceneLabel(file), tabs.list.indexOf(tabs.cur) + 1));
}
// opts.reload reads the file again even for the tab on screen (a project just adopted); opts.make builds the graph in place of reading one (a new scene)
async function switchTab(tab, opts = {}){
  const from = tabs.cur;
  if(tab === from && !opts.reload) return true;
  if(trainingRunning){ status('a training run is going on the graph on screen: tabs switch when it finishes', true); return false; }
  if(sceneSwitching) return false;
  let d = opts.reload || opts.make ? null : tab.graph, fresh = false;
  if(!d && !opts.make){
    const drop = why => { status(why, true); if(tab !== from && !tab.graph){ tabs.remove(tab); renderTabs(); } return false; };
    if(!tab.file) return drop('nothing to open');
    const st = await store();
    const bytes = await st.read(tab.file);
    if(!bytes) return drop('no such scene: ' + tab.file);
    try { d = JSON.parse(new TextDecoder().decode(bytes)); }
    catch(e){ return drop('scene ' + sceneLabel(tab.file) + ' is not readable: ' + e.message); }
    if(!d || !d.nodes) return drop('scene ' + sceneLabel(tab.file) + ' has no nodes');
    const missing = scenePluginsMissing(d);
    if(missing) return drop(sceneLabel(tab.file) + ': ' + missing);
    fresh = true;
  }
  // what is on screen goes into its file and stays with its tab
  if(from && from !== tab){
    await saveScene();
    from.graph = JSON.parse(sceneJSON());
    if(from !== tabs.shown && !from.held) from.held = { resolution:from.graph.resolution, slice:from.graph.slice, hues:from.graph.hues };
  }
  // The new graph goes on screen and the autosave is pointed at its file in one step, and no save lands while the project file is written: a save timer firing inside that await would write the previous graph into the new file.
  sceneSwitching = true;
  try {
    tabs.cur = tab;
    urlScene = !tab.file;
    if(tab.file) activeScene = tab.file;
    editor.docKey = tab.key;              // a copy remembers which scene it came from
    editor.errorNode = null;
    try { viewer.closeGizmo(); } catch(e){}
    if(tab === tabs.shown){
      if(fresh){ exitLive(); doc.brainByNode.clear();
        if(d.resolution) applyResolution(d.resolution, false);
        applySlice(d.slice); applyHues(d.hues); }
    } else if(d && (fresh || !tab.held)) tab.held = { resolution:d.resolution || 100, slice:d.slice || null, hues:d.hues || {} };
    if(opts.make){
      if(tab !== tabs.shown) tab.held = { resolution:100, slice:null, hues:{} };
      opts.make();
    } else {
      editor.load(d); assignDefaultSlots(); showProps(null);
      if(tab.file) editor.scenarioName = null;   // a file on screen is not a scenario
    }
    // read from the file, what is on screen is what the file holds; held by the tab, the next save tick writes it again, which costs one write and cannot lose an edit a busy save skipped on the way out
    lastSceneJson = fresh && tab.file ? sceneJSON() : null;
    showSceneName(); renderTabs();
    await persistProject();
  } finally { sceneSwitching = false; }
  saveGraph();
  if(tab === tabs.shown){
    if(fresh || opts.make) recompute(); else computePoints();
    status((fresh ? 'opened scene ' : 'scene ') + tab.label);
  } else {
    if(!opts.make) computePoints();
    status(tab.label + ': ' + viewHint());
  }
  return true;
}
// The scene in the editor becomes the one the viewer and the engine hold.
// False when the shown scene's weights are locked: showing another scene starts another engine, which is what the lock is there to refuse.
function makeShown(){
  if(tabs.cur === tabs.shown) return true;
  if(trainingRunning){ status('a training run owns the viewer until it finishes', true); return false; }
  if(doc.livePhase){
    status('weights are locked in ' + tabs.shown.label + ': unlock them before viewing another scene, or they are lost', true);
    return false;
  }
  // the scene leaving the viewer takes its viewer-side settings with it
  const was = tabs.shown;
  was.held = { resolution:+resEl.value, slice:sceneSlice, hues:sceneHues };
  if(was.graph){
    Object.assign(was.graph, was.held);
    if(was.file) store().then(st => st.write(was.file, JSON.stringify(was.graph, null, 1))).catch(() => {});
  }
  doc.computeGen++;                          // a computation of that scene still in flight bails
  was.doc.curNet = null; was.doc.lastSimView = null;
  tabs.shown = tabs.cur; doc = tabs.cur.doc;
  doc.curNet = null; doc.lastSimView = null;   // whatever engine is running holds another scene's network
  const h = tabs.cur.held || {};
  tabs.cur.held = null;
  applyResolution(h.resolution || 100, false);
  applySlice(h.slice); applyHues(h.hues);
  for(const n of editor.nodes) n._cache = null;   // anything computed here was computed at the other scene's resolution
  clearPathways();
  fitNextCompute = true;
  if(recording) recording.mark('scene', tabs.cur.label);
  renderTabs();
  return true;
}
// The streams above the connect node, for a scene that is not being viewed: the panel's population pickers read tags off computed streams (a repeat's copies are in no scatter's settings).
// Computing points is cheap and nothing here reaches the viewer or the engine.
async function computePoints(){
  const byId = id => editor.byId(id);
  const seen = new Set(), stack = editor.nodes.filter(n => n.type === 'checkpoint');
  while(stack.length){
    const n = stack.pop();
    if(!n || seen.has(n.id)) continue;
    seen.add(n.id);
    const ins = (n.inputs || []).filter(Boolean).map(c => byId(c.id)).filter(Boolean);
    if(NODE_DEFS[n.type] && NODE_DEFS[n.type].memo){
      for(const m of ins){ try { await computeNode(m, byId); } catch(e){} }
    } else stack.push(...ins);
  }
  if(propsNode && editor.nodes.includes(propsNode)) refreshProps();
}
async function closeTab(tab){
  if(tabs.list.length < 2) return;
  if(trainingRunning){ status('a training run is going: tabs close when it finishes', true); return; }
  if(tab === tabs.shown && doc.livePhase){
    status('weights are locked in ' + tab.label + ': unlock them before closing it, or they are lost', true); return; }
  if(tab === tabs.cur && !(await switchTab(tabs.neighbour(tab)))) return;
  tabs.remove(tab);
  // the viewer cannot go on showing a scene that has no tab
  if(tab === tabs.shown && makeShown()) recompute();
  renderTabs();
  await persistProject();
}
// tabs for the scenes the project had open, once the first one is settled
async function restoreTabs(){
  if(urlScene) return;
  try {
    const st = await store();
    const proj = await P.openProject(st);
    const onDisk = new Set((await st.list(P.scenesDir())).filter(e => !e.dir).map(e => P.scenePath(e.name)));
    tabs.restore(proj.openScenes, f => onDisk.has(f));
    for(const t of tabs.list) if(!t.label && t.file) t.label = sceneLabel(t.file);
  } catch(e){}
  renderTabs();
}
function renderTabs(){
  const el = document.getElementById('scenetabs');
  if(!el) return;
  el.innerHTML = '';
  for(const t of tabs.list){
    const b = document.createElement('div');
    b.className = 'stab' + (t === tabs.cur ? ' cur' : '') + (t === tabs.shown ? ' shown' : '');
    b.title = (t.file || 'not a file: save as to keep it') +
      (t === tabs.shown ? '\nthis scene is in the viewer' : '\nnot in the viewer: press a viewer key (1) in this tab to view it');
    const eye = document.createElement('span');
    eye.className = 'seye'; eye.innerHTML = icon('eye');
    const name = document.createElement('span');
    name.className = 'sname'; name.textContent = t.label || 'scene';
    b.append(eye, name);
    if(tabs.list.length > 1){
      const x = document.createElement('span');
      x.className = 'sclose'; x.innerHTML = icon('close'); x.title = 'close this tab. The scene stays in the project.';
      x.onclick = ev => { ev.stopPropagation(); closeTab(t); };
      b.appendChild(x);
    }
    b.onclick = () => switchTab(t);
    b.onauxclick = ev => { if(ev.button === 1){ ev.preventDefault(); closeTab(t); } };
    el.appendChild(b);
  }
  const add = document.createElement('button');
  add.id = 'stabadd'; add.className = 'icb icon-only'; add.innerHTML = icon('add');
  add.title = 'open a scene in a new tab, or start a new one';
  add.setAttribute('aria-label', 'open a scene in a new tab');
  add.onclick = async ev => {
    const anchor = ev.currentTarget;
    const items = [{ label:'new scene...', run: () => showSceneDlg('new') }];
    let scenes = [];
    try { scenes = await listScenes(); } catch(e){}
    if(scenes.length){
      items.push(null, { head:'scenes in this project' });
      for(const sc of scenes) items.push({ label:sc.name, on:!!tabs.byFile(sc.file),
        note: tabs.byFile(sc.file) ? 'open' : '', run: () => openScene(sc.file) });
      items.push({ label:'open...', note:'with sizes and dates', run: () => showSceneDlg('open') });
    }
    barMenu(anchor, items);
  };
  el.appendChild(add);
}

// Fork: what is on screen, under a new name, and carry on in the copy.
async function saveSceneAs(name){
  const st = await store();
  const file = P.scenePath(name + '.json');
  if(await st.read(file)) return 'a scene called ' + name + ' is already here';
  await st.write(file, sceneJSON());
  await setActiveScene(file);
  lastSceneJson = sceneJSON();
  status('saved scene as ' + name);
  return null;
}

async function newScene(name){
  const st = await store();
  const file = P.scenePath(name + '.json');
  if(await st.read(file)) return 'a scene called ' + name + ' is already here';
  // a new tab beside this one, on the guided tour; the viewer keeps what it shows until a viewer key is pressed there
  const tab = tabs.add(file, name, tabs.list.indexOf(tabs.cur) + 1);
  if(!(await switchTab(tab, { make: buildDefaultGraph }))){ tabs.remove(tab); renderTabs(); return 'could not switch to a new tab just now'; }
  await st.write(file, sceneJSON());
  lastSceneJson = sceneJSON();
  saveGraph();
  status('new scene ' + name + ': ' + viewHint());
  return null;
}

// ---- the dialog ----------------------------------------------------------
// One box in two modes, because they differ by a list and a text field rather than by anything structural.
const sceneDlg = document.getElementById('sceneDlg');
let sceneMode = 'open', scenePick = null;
const sceneEls = () => ({
  title: document.getElementById('sceneDlgTitle'),
  hint: document.getElementById('sceneDlgHint'),
  list: document.getElementById('sceneList'),
  nameRow: document.getElementById('sceneNameRow'),
  nameIn: document.getElementById('sceneNameIn'),
  err: document.getElementById('sceneDlgErr'),
});
function closeSceneDlg(){ sceneDlg.style.display = 'none'; }

async function showSceneDlg(mode){
  const e = sceneEls();
  sceneMode = mode; scenePick = null;
  e.err.textContent = '';
  e.list.hidden = mode !== 'open';
  e.nameRow.hidden = mode === 'open';
  if(mode === 'open'){
    e.title.textContent = 'open a scene';
    e.hint.textContent = 'Everything on screen is already saved into ' +
      sceneLabel(activeScene) + ', so opening another one loses nothing.';
    e.list.innerHTML = '';
    let scenes = [];
    try { scenes = await listScenes(); }
    catch(err){ e.err.textContent = 'could not read the project: ' + err.message; }
    if(!scenes.length){
      const li = document.createElement('li');
      li.className = 'empty';
      li.textContent = 'no scenes in this project yet';
      e.list.appendChild(li);
    }
    for(const s of scenes){
      const li = document.createElement('li');
      li.className = s.file === activeScene ? 'cur' : '';
      const left = document.createElement('span');
      left.textContent = s.name + (s.file === activeScene ? '  (open)' : '');
      const right = document.createElement('span');
      right.className = 'dim';
      right.textContent = (s.size/1024).toFixed(1) + ' KB' +
        (s.mtime ? '  ' + new Date(s.mtime).toLocaleString() : '');
      li.append(left, right);
      li.onclick = () => {
        scenePick = s.file;
        for(const o of e.list.children) o.classList.remove('sel');
        li.classList.add('sel');
      };
      li.ondblclick = () => { scenePick = s.file; sceneOk(); };
      e.list.appendChild(li);
    }
  } else {
    e.title.textContent = mode === 'new' ? 'new scene' : 'save this scene as';
    e.hint.textContent = mode === 'new'
      ? 'Starts from the guided tour graph. The scene you have open now is already saved and stays where it is.'
      : 'Copies what is on screen to a new name and carries on in the copy. ' +
        sceneLabel(activeScene) + ' keeps what it has.';
    e.nameIn.value = mode === 'new' ? '' : sceneLabel(activeScene) + ' copy';
    setTimeout(() => { e.nameIn.focus(); e.nameIn.select(); }, 0);
  }
  sceneDlg.style.display = 'flex';
}

async function sceneOk(){
  const e = sceneEls();
  e.err.textContent = '';
  if(sceneMode === 'open'){
    if(!scenePick){ e.err.textContent = 'pick a scene first'; return; }
    if(scenePick === activeScene){ closeSceneDlg(); return; }
    closeSceneDlg();
    await openScene(scenePick);
    return;
  }
  const name = e.nameIn.value.trim();
  const bad = sceneNameError(name);
  if(bad){ e.err.textContent = bad; return; }
  let err = null;
  try {
    err = sceneMode === 'new' ? await newScene(name) : await saveSceneAs(name);
  } catch(ex){ err = ex.message; }
  if(err){ e.err.textContent = err; return; }
  closeSceneDlg();
}

document.getElementById('sceneOk').onclick = sceneOk;
document.getElementById('sceneCancel').onclick = closeSceneDlg;
document.getElementById('sceneNameIn').onkeydown = ev => {
  if(ev.key === 'Enter'){ ev.preventDefault(); sceneOk(); }
};
sceneDlg.onkeydown = ev => { if(ev.key === 'Escape') closeSceneDlg(); };
sceneDlg.onclick = ev => { if(ev.target === sceneDlg) closeSceneDlg(); };
showSceneName();

// ---- which folder the project lives in -----------------------------------
//
// Chosen through the server rather than through the browser's folder picker.
// The picker never reveals an absolute path, and the engine host is a separate process that needs one; asking the server keeps the page and the host pointed at the same real directory.
// The server also decides what it will adopt, since every read and write is confined to the project root and a root pointing at a home directory would confine nothing.
const projDlg = document.getElementById('projDlg');
const RECENT = 'neuron-playground-recent-projects';
const recentProjects = () => {
  try { return JSON.parse(localStorage.getItem(RECENT)) || []; }
  catch(e){ return []; }
};
function rememberProject(path){
  // Per machine, like the keymap: a list of paths on this disk is not a fact about the project, and copying one to another machine would list folders that are not there.
  try {
    const list = [path, ...recentProjects().filter(p => p !== path)].slice(0, 8);
    localStorage.setItem(RECENT, JSON.stringify(list));
  } catch(e){}
}

// The folder the app is already using counts as recent too.
// Without this the list only ever holds folders reached through this dialog, and not the one given on the command line, which is the one most sessions start in.
store().then(st => { if(st && st.root) rememberProject(st.root); })
  .catch(() => {});

let projAt = '';
async function projBrowse(path){
  const err = document.getElementById('projErr');
  const list = document.getElementById('projList');
  const box = document.getElementById('projPath');
  err.textContent = '';
  let d;
  try {
    const r = await fetch('/project/browse' +
      (path ? '?path=' + encodeURIComponent(path) : ''));
    d = await r.json();
  } catch(e){ err.textContent = 'could not reach the server: ' + e.message; return; }
  if(!d.ok){ err.textContent = d.error || 'could not read that folder'; return; }
  projAt = d.path;
  box.value = d.path;
  list.innerHTML = '';
  const row = (label, cls, onclick) => {
    const li = document.createElement('li');
    li.className = cls;
    li.textContent = label;
    li.onclick = onclick;
    list.appendChild(li);
    return li;
  };
  if(!d.roots && d.parent !== null) row('.. up one', 'up', () => projBrowse(d.parent));
  for(const name of d.dirs){
    // the separator this machine actually uses, taken from the path the server just handed back rather than assumed
    const sep = d.path.includes('\\') ? '\\' : '/';
    const full = d.roots ? name
      : d.path.replace(/[\\/]+$/, '') + sep + name;
    row(name, '', () => projBrowse(full));
  }
  if(!d.dirs.length && !d.roots) row('no subfolders here', 'empty', null);
  // Recent folders, so getting back to one does not mean clicking down the tree again.
  const recent = recentProjects().filter(p => p !== d.path);
  if(recent.length){
    const head = document.createElement('li');
    head.className = 'empty';
    head.textContent = 'recent';
    list.appendChild(head);
    for(const p of recent) row(p, 'up', () => projBrowse(p));
  }
  if(d.adoptable === false && d.why) err.textContent = d.why;
}

async function projOpen(path, create, errEl){
  const err = errEl || document.getElementById('projErr');
  err.textContent = '';
  let d;
  try {
    const r = await fetch('/project/open', { method:'POST',
      body: JSON.stringify({ path, create: !!create }) });
    d = await r.json();
  } catch(e){ err.textContent = 'could not reach the server: ' + e.message; return; }
  if(!d.ok){
    if(d.missing){
      // Creating a folder is a decision, so it is asked rather than assumed.
      err.textContent = d.error + '  ';
      const b = document.createElement('button');
      b.textContent = 'CREATE IT';
      b.onclick = () => projOpen(path, true);
      err.appendChild(b);
      return;
    }
    err.textContent = d.error || 'could not open that folder';
    return;
  }
  rememberProject(d.root);
  await adoptProject(d.root, () => { resetStore(); return store(); }, d);
}
// Whatever holds the project now (a server folder switched behind the endpoints, or a folder handle picked in the browser), open it: its active scene if it has one, else the guided tour or what is on screen as its first. info: the server's reply when the server opened the folder, for the created and engine-host notices; a browser-picked folder has none.
async function adoptProject(root, getStore, info = {}){
  showProjName(root);
  projDlg.style.display = 'none';
  newProjDlg.style.display = 'none';
  lastSceneJson = null;
  const st = await getStore();
  const proj = await P.openProject(st);
  // this project's node modules, before any of its scenes loads; a module the previous project loaded stays registered for the session
  await refreshNodeModules();
  // the tabs were the other project's; this one starts with its open scene in the viewer and brings its own list back below
  exitLive();
  const held = tabs.shown.doc;
  tabs.reset(null, '');
  doc = tabs.shown.doc;
  held.curNet = null;
  retarget(proj.activeScene || P.scenePath('main.json'));
  showSceneName();
  // refused rather than treated as a project with no scene yet, which would write what is on screen into its file
  const missing = await sceneModulesMissing(st, activeScene);
  if(missing){ refuseScene(new Error(sceneLabel(activeScene) + ': ' + missing)); return; }
  const opened = await switchTab(tabs.cur, { reload:true });
  if(!opened){
    // A fresh project has no scene yet: a new project starts on the guided tour, and a folder opened empty takes what is on screen rather than throwing it away.
    exitLive();
    if(projMode === 'new'){ doc.brainByNode.clear(); buildDefaultGraph(); }
    await st.write(activeScene, sceneJSON());
    lastSceneJson = sceneJSON();
    P.addScene(proj, activeScene);
    proj.activeScene = activeScene;
    await P.writeProject(st, proj);
  }
  await restoreTabs();
  status('project folder: ' + root +
    (info.created ? ' (created)' : '') +
    (info.hostRunning ? '. An engine host is already running and keeps the folder it was started with; restart it to follow this one.' : ''));
}

document.getElementById('projUp').onclick = () => {
  const p = document.getElementById('projPath').value.trim();
  const cut = p.replace(/[\\/]+$/, '').replace(/[\\/][^\\/]*$/, '');
  projBrowse(cut || '');
};
// A new project is a name and a location, not a folder browsed to: the folder is made from the two, inside the location, and starts on the guided tour.
// The location defaults to beside the open project.
const newProjDlg = document.getElementById('newProjDlg');
let projMode = 'open';
function showNewProjDlg(root){
  projMode = 'new';
  const sep = root.includes('\\') ? '\\' : '/';
  const parent = root ? root.replace(/[\\/]+$/, '').replace(/[\\/][^\\/]*$/, '') : '';
  document.getElementById('newProjLoc').value = parent;
  document.getElementById('newProjName').value = '';
  document.getElementById('newProjErr').textContent = '';
  const show = () => {
    const loc = document.getElementById('newProjLoc').value.trim().replace(/[\\/]+$/, '');
    const name = document.getElementById('newProjName').value.trim();
    document.getElementById('newProjPath').textContent =
      loc && name ? 'will be made as ' + loc + sep + name : '';
  };
  document.getElementById('newProjLoc').oninput = show;
  document.getElementById('newProjName').oninput = show;
  show();
  newProjDlg.style.display = 'flex';
  document.getElementById('newProjName').focus();
}
document.getElementById('newProjOk').onclick = () => {
  const loc = document.getElementById('newProjLoc').value.trim().replace(/[\\/]+$/, '');
  const name = document.getElementById('newProjName').value.trim();
  const err = document.getElementById('newProjErr');
  if(!name){ err.textContent = 'give the project a name'; return; }
  if(/[\\/:*?"<>|]/.test(name)){ err.textContent = 'a name, not a path'; return; }
  if(!loc){ err.textContent = 'where should it go?'; return; }
  const sep = loc.includes('\\') ? '\\' : '/';
  projOpen(loc + sep + name, true, err);
};
document.getElementById('newProjCancel').onclick = () => { newProjDlg.style.display = 'none'; };
newProjDlg.onclick = ev => { if(ev.target === newProjDlg) newProjDlg.style.display = 'none'; };
document.getElementById('newProjName').onkeydown = ev => {
  if(ev.key === 'Enter'){ ev.preventDefault(); document.getElementById('newProjOk').click(); }
};
// The machine's own folder picker, opened by the server (a page can open one too, but never learns the path from it).
// Resolves to the path, '' if dismissed, or null when there is no picker, in which case the in-app browser is the fallback.
async function pickFolder(start, title){
  try {
    const r = await fetch('/project/pick', { method:'POST',
      body: JSON.stringify({ start, title }) });
    const d = await r.json();
    if(!d.ok){ status(d.error, true); return null; }
    return d.cancelled ? '' : d.path;
  } catch(e){ return null; }
}
document.getElementById('newProjBrowse').onclick = async () => {
  const loc = document.getElementById('newProjLoc');
  const p = await pickFolder(loc.value.trim(), 'where the new project goes');
  if(p){ loc.value = p; loc.dispatchEvent(new Event('input')); }
};
function showProjDlg(root){
  projMode = 'open';
  document.getElementById('projErr').textContent = '';
  document.getElementById('projList').innerHTML = '';
  projDlg.style.display = 'flex';
  projBrowse(root || '');
}
document.getElementById('projOk').onclick = () =>
  projOpen(document.getElementById('projPath').value.trim());
document.getElementById('projCancel').onclick = () => {
  projDlg.style.display = 'none';
};
document.getElementById('projPath').onkeydown = ev => {
  if(ev.key === 'Enter'){ ev.preventDefault(); projBrowse(ev.target.value.trim()); }
};
projDlg.onclick = ev => { if(ev.target === projDlg) projDlg.style.display = 'none'; };

// ---- node modules: the project's nodes folder (src/plugins.js) -------------
// A module runs with the page's own access; it is code the person put in their project or chose to add, and is loaded as such.
const importModule = importer();
let nodeModules = [];                    // one record per file in the open project's nodes folder
async function refreshNodeModules(){
  let recs;
  try { recs = await loadProjectPlugins(await store(), importModule); }
  catch(e){ status('node modules: ' + (e.message || e), true); return []; }
  nodeModules = recs;
  const changed = new Set(recs.filter(r => r.changed && !r.error).flatMap(r => r.keys));
  const hit = editor.nodes.filter(n => changed.has(n.type));
  for(const n of hit){                   // a setting the new version added starts at its default
    for(const q of NODE_DEFS[n.type].params) if(n.params[q.k] === undefined) n.params[q.k] = structuredClone(q.def);
    editor.markDirty(n);
  }
  const bad = recs.filter(r => r.error);
  if(bad.length) status(bad.map(r => r.error).join('; '), true);
  return hit;
}
async function reloadNodeModules(){
  const hit = await refreshNodeModules();
  if(hit.length){ recompute(); if(propsNode) refreshProps(); }
  const good = nodeModules.filter(r => !r.error);
  if(!nodeModules.some(r => r.error))
    status(good.length ? 'node modules: ' + good.map(r => r.module + ' (' + r.keys.join(', ') + ')').join(', ') : 'no node modules in this project\'s nodes folder');
}
const nodeModDlg = document.getElementById('nodeModDlg');
function showNodeModDlg(){
  document.getElementById('nodeModErr').textContent = '';
  nodeModDlg.style.display = 'flex';
  document.getElementById('nodeModUrl').focus();
}
document.getElementById('nodeModOk').onclick = async () => {
  const err = document.getElementById('nodeModErr');
  const url = document.getElementById('nodeModUrl').value.trim();
  err.textContent = '';
  const fetchText = async u => {
    let r;
    try { r = await fetch(u); }
    catch(e){ throw new Error('could not fetch ' + u + ': the server has to allow requests from this page (CORS)'); }
    if(!r.ok) throw new Error('fetching ' + u + ' gave ' + r.status);
    return r.text();
  };
  try {
    const st = await store();
    const rec = await addPluginFromUrl(st, url, fetchText, importModule, location.href);
    nodeModDlg.style.display = 'none';
    await refreshNodeModules();
    status(rec.module + ' is in ' + st.label + ' under nodes/ and adds ' + rec.keys.join(', ') + ' to the Tab menu');
  } catch(e){ err.textContent = e.message || String(e); }
};
document.getElementById('nodeModCancel').onclick = () => { nodeModDlg.style.display = 'none'; };
nodeModDlg.onclick = ev => { if(ev.target === nodeModDlg) nodeModDlg.style.display = 'none'; };
document.getElementById('nodeModUrl').onkeydown = ev => {
  if(ev.key === 'Enter'){ ev.preventDefault(); document.getElementById('nodeModOk').click(); }
};

// ---- the path, and what each half of it can do ---------------------------
document.getElementById('projbtn').onclick = async ev => {
  const anchor = ev.currentTarget;        // gone from the event after the first await
  const st = await store().catch(() => null);
  const root = st && st.root ? st.root : '';
  const served = !!(st && st.mode === 'project');
  const items = [{ head: root ? (st.mode === 'folder' ? 'folder ' + root : root)
    : 'browser storage' }, null];
  if(served){
    items.push(
      { label:'new project...', run: () => showNewProjDlg(root) },
      { label:'open a project...', run: async () => {
          const p = await pickFolder(root, 'open a project folder');
          if(p === null) showProjDlg(root);       // no picker here: browse in the page
          else if(p) projOpen(p, false, document.getElementById('status'));
        } });
  } else if(hasFolderPicker()){
    items.push(
      { label:'new project...', note:'pick or make an empty folder',
        run: () => pickProjectFolder(true) },
      { label:'open a project...', note:'a folder with a project.json',
        run: () => pickProjectFolder(false) });
    const remembered = await rememberedProjectDir().catch(() => null);
    if(remembered && !(st && st.mode === 'folder' && st.root === remembered.name))
      items.push({ label:'reopen ' + remembered.name, note:'the folder from last time',
        run: () => reopenRememberedFolder(remembered) });
  } else {
    items.push({ head:'this browser cannot open folders; scenes stay in browser storage' });
  }
  // A brain carries its graph, so opening one is opening a scene: it arrives in a tab of its own with its weights (loadBrainBuffer).
  let brains = [];
  try { brains = await listProjectBrains(); } catch(e){}
  items.push(null, { head:'open a brain as a scene' });
  for(const b of brains.slice(0, 8)) items.push({ label:b.name,
    note:(b.size/1048576).toFixed(b.size < 10485760 ? 1 : 0) + ' MB', run: () => openBrainPath(b) });
  if(brains.length > 8) items.push({ head:(brains.length - 8) + ' older ones are in the project folder' });
  items.push({ label:'from a file...', run: pickBrainFile });
  // the node modules in the project's nodes folder, each with what it registered or why it did not load (a click puts the whole message in the status line)
  items.push(null, { head:'node modules (nodes folder)' });
  for(const r of nodeModules) items.push({ label:r.module,
    note: r.error ? 'not loaded' : r.keys.join(', '),
    run: () => status(r.error || r.module + ' registers ' + r.keys.join(', ') + ', in the Tab menu', !!r.error) });
  items.push({ label:'reload node modules', note:'after a file in nodes/ changed', run: reloadNodeModules });
  items.push({ label:'add a node module from a URL...', note:'copied into nodes/', run: showNodeModDlg });
  const recent = served ? recentProjects().filter(p => !st || p !== st.root) : [];
  if(recent.length){
    items.push(null, { head:'recent' });
    for(const p of recent) items.push({ label:p, run: () => projOpen(p) });
  }
  barMenu(anchor, items);
};

document.getElementById('scenebtn').onclick = async ev => {
  const anchor = ev.currentTarget;
  const items = [
    { label:'new scene...', run: () => showSceneDlg('new') },
    { label:'save as...', note:'a copy, and you continue in it',
      run: () => showSceneDlg('saveas') },
    { label:'copy a link to this scene', note:'restores it exactly, seeds and all',
      run: async () => {
        try {
          const frag = await encodeSceneFragment(sceneJSON());
          const link = location.origin + location.pathname + '#' + frag;
          await navigator.clipboard.writeText(link);
          status('link copied, ' + (link.length/1024).toFixed(1) + ' KB: it opens this scene exactly as it is now, at ' + resEl.value + '% resolution, without touching the open file');
        } catch(e){ status('could not make the link: ' + e.message, true); }
      } },
    null,
    { label:'reset this scene to the guided tour', run: () => {
        if(tabs.cur === tabs.shown) exitLive();
        tabs.cur.doc.brainByNode.clear();
        buildDefaultGraph();
        saveGraph();
        status('reset ' + sceneLabel(activeScene) + ' to the guided tour');
      } },
    null,
    { head:'scenario authoring' },
    // Two steps, and away from the everyday items: it writes into src/layouts, which is a change to the source tree that a mis-click should not be able to make.
    { label:'save this layout for the scenario...', note:'writes a file into the source tree',
      run: () => barMenu(anchor, [
        { head:'the arrangement on screen becomes the scenario default, written into src/layouts' },
        { label:'write the layout file', run: saveScenarioLayout },
        { label:'cancel', run: () => {} },
      ]) },
  ];
  // The scenes themselves, so switching is one click rather than a dialog.
  // The dialog is still there for the sizes and dates.
  let scenes = [];
  try { scenes = await listScenes(); } catch(e){}
  if(scenes.length){
    items.push(null, { head:'scenes in this project' });
    for(const s of scenes) items.push({
      label: s.name, on: s.file === activeScene,
      off: s.file === activeScene, note: s.file !== activeScene && tabs.byFile(s.file) ? 'open in a tab' : '',
      run: () => openScene(s.file) });
    items.push({ label:'open...', note:'with sizes and dates',
      run: () => showSceneDlg('open') });
  }
  barMenu(anchor, items);
};

// The label follows whatever folder the store settled on, including the browser-storage fallback, which has no folder to name.
store().then(st => showProjName(st && st.root)).catch(() => showProjName(null));


// Read the scene back out of the project.
// Used when this browser has nothing, which is a fresh profile, a cleared cache, or a project folder opened on a machine that has not seen it before.
async function openProjectScene(){
  try {
    const st = await store();
    const proj = await P.openProject(st);
    if(proj.activeScene) { retarget(proj.activeScene); showSceneName(); }
    const bytes = await st.read(activeScene);
    if(!bytes) return false;
    const d = JSON.parse(new TextDecoder().decode(bytes));
    if(!d || !d.nodes || !d.nodes.length) return false;
    if(d.resolution) applyResolution(d.resolution, false);
    applySlice(d.slice); applyHues(d.hues);
    editor.load(d); assignDefaultSlots(); showProps(null); recompute();
    status('opened scene ' + sceneLabel(activeScene) + ' from the project folder');
    return true;
  } catch(e){
    // true: the caller would otherwise build the guided tour under this scene's name
    if(e && e.pluginMissing){ refuseScene(new Error(sceneLabel(activeScene) + ': ' + e.message)); return true; }
    return false;
  }
}

// The browser buffer holds a graph but not which scene it belongs to, so on the fast path the two can disagree: another browser opened a different scene, and project.json is the only record of that.
// The buffer is used first because it is instant, then the project is asked, and the scene it names wins if they differ.
async function reconcileScene(){
  try {
    const st = await store();
    const proj = await P.openProject(st);
    const want = proj.activeScene;
    // Decide from which scene the buffer belongs to, not from the name the page happens to hold: the two can agree by name while the buffer holds another scene, and the autosave would write it over the file.
    let bufferFor = null;
    try { bufferFor = localStorage.getItem(STORE_FOR); } catch(e){}
    const bytes = want ? await st.read(want) : null;
    const pick = P.startupScene(want, bufferFor, !!bytes);
    let load = pick.load;
    // The buffer and the file are meant to hold the same graph.
    // When the buffer is tagged as the file's scene and the two differ, the file is the document and wins: a save that lands mid-switch can leave the buffer holding another scene's graph under this scene's name, and the autosave would then write that graph over the file at every start.
    if(!load && bytes && want === bufferFor){
      try {
        const d0 = JSON.parse(new TextDecoder().decode(bytes));
        if(d0 && d0.nodes && d0.nodes.length &&
           JSON.stringify(d0.nodes) !== JSON.stringify(editor.toJSON().nodes)) load = true;
      } catch(e){}
    }
    if(pick.scene && pick.scene !== activeScene){ retarget(pick.scene); lastSceneJson = null; }
    showSceneName();
    if(!load){
      if(want && bufferFor && want !== bufferFor)
        status('the project names ' + sceneLabel(want) + ' but it has no file yet; ' +
          'staying in ' + sceneLabel(bufferFor));
      return;
    }
    const d = JSON.parse(new TextDecoder().decode(bytes));
    if(!d || !d.nodes || !d.nodes.length){
      // an empty file is not a scene to save another scene's graph into
      if(bufferFor){ retarget(bufferFor); showSceneName(); }
      return;
    }
    if(d.resolution) applyResolution(d.resolution, false);
    applySlice(d.slice); applyHues(d.hues);
    editor.load(d); assignDefaultSlots(); showProps(null); recompute();
    lastSceneJson = sceneJSON();         // what is on screen is what the file holds
    sceneSwitching = false;
    saveGraph();                         // re-tag the buffer as this scene
    status('opened scene ' + sceneLabel(want) + ', which this project had open');
  } catch(e){
    if(e && e.pluginMissing) refuseScene(new Error(sceneLabel(activeScene) + ': ' + e.message));
    else showSceneName();
  }
  finally { sceneSwitching = false; restoreTabs(); }
}

function scheduleCompute(){ clearTimeout(computeTimer); computeTimer = setTimeout(recompute, 200); saveGraph(); }

// ---- wiring progress bar ----
const computebarEl = document.getElementById('computebar');
// The count travels with the fraction once the computation knows it.
// On a full density scene wiring takes minutes, and a bar with no number on it is the difference between waiting and wondering whether it has stopped.
const fmtSyn = v => v >= 1e6 ? (v/1e6).toFixed(1) + 'M'
  : v >= 1e3 ? Math.round(v/1e3) + 'k' : String(v);
setComputeProgress((f, syn) => {
  computebarEl.style.display = 'block';
  computebarEl.style.transform = `scaleX(${f})`;
  status(`wiring synapses ${Math.round(f*100)}%` +
    (syn > 0 ? ` · ${fmtSyn(syn)} so far` : ''));
});
// Millions and above are abbreviated: the exact synapse count is in the training panel's setup, and at 151,693,342 the digits stop being read and start being a wall.
const netSizeEl = document.getElementById('netsize');
function setNetSize(count, syn){
  if(!netSizeEl) return;
  if(!count){ netSizeEl.textContent = ''; return; }
  const big = n => n >= 1e6 ? (n/1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M'
    : n >= 1e4 ? (n/1e3).toFixed(0) + 'k' : n.toLocaleString();
  netSizeEl.textContent = `${count.toLocaleString()} neurons · ${big(syn)} synapses`;
  netSizeEl.title = `${count.toLocaleString()} neurons, ` +
    `${syn.toLocaleString()} synapses on screen`;
}
// The 3D view can be taken away by the driver while the run is untouched.
// Saying which of those two things happened is the whole job here.
viewer.onContextLost = () => {
  let el = document.getElementById('glLost');
  if(!el){
    el = document.createElement('div');
    el.id = 'glLost';
    document.getElementById('viewerPane').appendChild(el);
  }
  el.innerHTML = '<b>3D view lost</b><br>The graphics driver took the ' +
    'WebGL context away from this page, which happens when the CUDA engine ' +
    'is working the same card. <b>The run is not affected</b>: it is in ' +
    'another process and keeps going, and its numbers below keep ' +
    'updating.<br>' +
    (trainingRunning
      ? '<span class="dim">The view comes back on its own when the run ends. ' +
        'Reloading now would end a sweep, since this page starts each variant.</span> '
      : '') +
    '<button id="glRebuild">restore the 3D view</button>';
  el.style.display = 'block';
  const b = document.getElementById('glRebuild');
  if(b) b.onclick = () => restoreView();
  status('3D view lost to the graphics driver; the run is unaffected', true);
};
// A new canvas and renderer for the same scene.
// The driver does not always give a taken context back, so this is what the notice offers and what a finished run does by itself.
function restoreView(){
  try { viewer.rebuildRenderer(); } catch(e){ status('3D view could not be rebuilt: ' + e.message, true); return; }
  const el = document.getElementById('glLost');
  if(el) el.style.display = 'none';
  status('3D view restored');
}
viewer.onContextRestored = () => {
  const el = document.getElementById('glLost');
  if(el) el.style.display = 'none';
  status('3D view restored');
};
// The newest checkpoint row, kept so a probe can show what it measured. train.js hands it over rather than the panel reading a file, because the panel should not have to know where a run writes.
let lastMetrics = null;
// the newest row that carries measured numbers, which is what a sweep reports
let lastScored = null;
// the newest row that carries the item-specific weight measure, which is written at weight readbacks rather than at every readout
let lastCoupled = null;
// Every checkpoint's row, so a chart can draw a trace over the run rather than the single latest value.
// Kept in memory only: a run's file holds the same history.
// Bounded because a long run at a short checkpoint interval is thousands of rows and nothing here needs the oldest of them.
const metricsHistory = [];
const METRICS_KEEP = 4000;
globalThis.__npMetrics = row => {
  lastMetrics = row;
  if(row && row.itemCoupling) lastCoupled = row;
  // The last row of a run is written after the recorders are flushed, so it carries a readout with every measure empty.
  // A sweep wants the last row that measured something, which is the checkpoint before it.
  if(row && row.readout && row.readout.pops &&
     Object.values(row.readout.pops).some(v => v && v.decode !== undefined && v.decode !== null))
    lastScored = row;
  metricsHistory.push(row);
  if(metricsHistory.length > METRICS_KEEP) metricsHistory.shift();
  if(propsNode && (propsNode.type === 'probe' || propsNode.type === 'chart'))
    showProps(propsNode);
};
setPropsCharts(() => metricsHistory);

// What a probe reads, by population.
// A probe is a mask over a network, so which cells it covers is not readable from the geometry feeding it, and a probe over the wrong population looks like a probe over the right one.
// The panel names them and the viewer marks them, both from the computed network rather than from the graph, because the mask decides.
function probeSetInfo(node){
  if(tabs.cur !== tabs.shown || !doc.curNet || !doc.curNet.probes || !node) return null;
  const pr = doc.curNet.probes.find(q => q.id === node.id);
  if(!pr) return null;
  const tagOf = i => (doc.curNet.tags && doc.curNet.tags[doc.curNet.src[i]]) || ('src ' + doc.curNet.src[i]);
  // how big each population is in the whole network, so the panel can say how much of one a probe covers rather than only how many cells it reads
  const whole = new Map();
  for(let i=0;i<doc.curNet.count;i++){ const t = tagOf(i); whole.set(t, (whole.get(t) || 0) + 1); }
  const per = new Map();
  for(const i of pr.idx){ const t = tagOf(i); per.set(t, (per.get(t) || 0) + 1); }
  const hues = sceneTagHues(doc.curNet);
  const rows = [...per.entries()].sort((a, b) => b[1] - a[1])
    .map(([tag, n]) => ({ tag, n, of: whole.get(tag) || n, hue: hues.get(tag) ?? 0 }));
  return { label:pr.label, total:pr.idx.length, rows, idx:pr.idx, hues };
}
setPropsProbeSet(probeSetInfo);
// What a stimulus drives, from the same network.
// A stimulus picks its cells by a mask, a radius and a population tag together, so the set is not readable from any one of them.
function driveSetInfo(node){
  if(tabs.cur !== tabs.shown || !doc.curNet || !doc.curNet.drives || !node) return null;
  const d = doc.curNet.drives.find(q => q.id === node.id);
  if(!d) return null;
  const tagOf = i => (doc.curNet.tags && doc.curNet.tags[doc.curNet.src[i]]) || ('src ' + doc.curNet.src[i]);
  const whole = new Map(), per = new Map();
  for(let i=0;i<doc.curNet.count;i++){ const t = tagOf(i); whole.set(t, (whole.get(t) || 0) + 1); }
  for(const i of d.idx){ const t = tagOf(i); per.set(t, (per.get(t) || 0) + 1); }
  const hues = sceneTagHues(doc.curNet);
  const rows = [...per.entries()].sort((a, b) => b[1] - a[1])
    .map(([tag, n]) => ({ tag, n, of: whole.get(tag) || n, hue: hues.get(tag) ?? 0 }));
  return { total:d.idx.length, rows, idx:d.idx, hues, amp:d.amp, mode:d.mode, gain:d.gain };
}
// Where an input lands.
// The encoder's channels are a map onto neurons, and the monitor draws what is being sent without saying where it goes.
function inputSetInfo(node){
  if(tabs.cur !== tabs.shown || !doc.curNet || !doc.curNet.inputMaps || !node) return null;
  const im = doc.curNet.inputMaps.find(q => q.id === node.id);
  if(!im || !im.chIdx || !im.chIdx.length) return null;
  const chans = im.chStart.length - 1;
  // channel index per driven neuron, so the mark can color by it
  const chan = new Int32Array(im.chIdx.length);
  for(let c=0;c<chans;c++)
    for(let k=im.chStart[c]; k<im.chStart[c+1]; k++) chan[k] = c;
  const seen = new Set(im.chIdx);
  return { idx:im.chIdx, chan, chans, cells:seen.size, cols:im.cols, rows:im.rows,
    sheet:!!im.sheet, amp:im.amp, sense:im.sense, tag:im.tag, masked:!!im.masked,
    source:im.source };
}
// The conjunctive cells an analysis found: the cells a pair drives that neither sense drives alone.
// They are the object every binding number in this project is about.
let conjSets = new Map();                 // population label -> { items, sets, at }
globalThis.__npConjSets = (label, info) => {
  if(!label) return;
  conjSets.set(label, info);
  if(propsNode && (propsNode.type === 'analysis' || propsNode.type === 'probe'))
    showProps(propsNode);
  const view = editor.activeView ? editor.byId(editor.activeView) : null;
  if(view && view.type === 'analysis') markViewSet(view);
};
function conjSetInfo(node){
  if(tabs.cur !== tabs.shown || !doc.curNet || !node) return null;
  // an analysis reads the probe upstream of it, so the sets it named are that probe's population
  let up = node.inputs && node.inputs[0] ? editor.byId(node.inputs[0].id) : null;
  let guard = 8;
  while(up && up.type !== 'probe' && guard-- > 0)
    up = up.inputs && up.inputs[0] ? editor.byId(up.inputs[0].id) : null;
  const label = up && up.type === 'probe' ? (up.params.label || '') : '';
  const info = conjSets.get(label);
  if(!info || !info.sets) return null;
  // the trainer sends network indices, since the sets index its own sample
  const items = [], idx = [], item = [];
  info.sets.forEach((set, i) => {
    if(!set || !set.length) return;
    items.push({ i, name:(info.items && info.items[i]) || String(i), n:set.length });
    for(const g of set) if(g < doc.curNet.count){ idx.push(g); item.push(i); }
  });
  if(!idx.length) return null;
  return { label, items, idx:Uint32Array.from(idx), item:Int32Array.from(item),
    nItems:info.sets.length, at:info.at };
}
setPropsDriveSet(driveSetInfo);
setPropsInputSet(inputSetInfo);
setPropsConjSet(conjSetInfo);

// Population coloring for the whole cloud, from the tags the scatter nodes carry.
// Made once per computation because it is one pass over the network.
function applyPopulationColors(net){
  if(!net || !net.count){ viewer.setPopulationColors(null); return; }
  const hues = sceneTagHues(net);
  const rgb = new Float32Array(net.count*3);
  const cache = new Map();
  for(let i=0;i<net.count;i++){
    const t = (net.tags && net.tags[net.src[i]]) || '';
    let c = cache.get(t);
    if(!c){ c = hueRgb(hues.get(t) ?? 0); cache.set(t, c); }
    rgb[i*3] = c[0]; rgb[i*3+1] = c[1]; rgb[i*3+2] = c[2];
  }
  viewer.setPopulationColors(rgb);
}
// A box around each population, from the same tags.
// Built with the colors because both are one pass over the network, and a scene of eight populations is otherwise a cloud with no regions in it.
function applyPopulationBoxes(net){
  if(!net || !net.count){ popBoxList = []; viewer.setPopulationBoxes([]); buildRegionsPanel(); return; }
  const hues = sceneTagHues(net);
  const box = new Map();
  for(let i=0;i<net.count;i++){
    const t = (net.tags && net.tags[net.src[i]]) || '';
    if(!t) continue;                      // an untagged scatter names nothing
    const x = net.pos[i*3], y = net.pos[i*3+1], z = net.pos[i*3+2];
    let b = box.get(t);
    if(!b){ b = { tag:t, hue:hues.get(t) ?? 0, count:0, src:net.src[i],
      min:[x, y, z], max:[x, y, z] }; box.set(t, b); }
    b.count++;
    if(x < b.min[0]) b.min[0] = x; if(x > b.max[0]) b.max[0] = x;
    if(y < b.min[1]) b.min[1] = y; if(y > b.max[1]) b.max[1] = y;
    if(z < b.min[2]) b.min[2] = z; if(z > b.max[2]) b.max[2] = z;
  }
  // a margin, so the outermost cells sit inside the box rather than on it
  for(const b of box.values())
    for(let k=0;k<3;k++){
      const pad = Math.max(4, (b.max[k] - b.min[k])*0.04);
      b.min[k] -= pad; b.max[k] += pad;
    }
  popBoxList = [...box.values()];
  viewer.setPopulationBoxes(popBoxList);
  buildRegionsPanel();
}
// The viewer draws the set the node it is bound to is about.
// Binding the view to a probe, a stimulus, an input or an analysis is the question "which cells is this", so the answer is the picture.
function markViewSet(view){
  if(!view || !doc.curNet) { viewer.clearMarkSet(); return; }
  const put = (idx, colOf, label, size) => {
    const n = idx.length, pos = new Float32Array(n*3), col = new Float32Array(n*3);
    for(let k=0;k<n;k++){
      const i = idx[k];
      pos[k*3] = doc.curNet.pos[i*3]; pos[k*3+1] = doc.curNet.pos[i*3+1]; pos[k*3+2] = doc.curNet.pos[i*3+2];
      const rgb = colOf(k, i);
      col[k*3] = rgb[0]; col[k*3+1] = rgb[1]; col[k*3+2] = rgb[2];
    }
    viewer.showMarkSet(pos, col, label, size);
  };
  const byTag = (hues) => {
    const cache = new Map();
    return (k, i) => {
      const t = (doc.curNet.tags && doc.curNet.tags[doc.curNet.src[i]]) || '';
      let rgb = cache.get(t);
      if(!rgb){ rgb = hueRgb(hues.get(t) ?? 0); cache.set(t, rgb); }
      return rgb;
    };
  };
  if(view.type === 'probe'){
    const info = probeSetInfo(view);
    if(!info) return viewer.clearMarkSet();
    return put(info.idx, byTag(info.hues),
      `probe ${info.label} · ${info.total.toLocaleString()} neurons · ` +
      info.rows.map(r => `${r.tag} ${r.n}`).join(' · '));
  }
  if(view.type === 'stimulus'){
    const info = driveSetInfo(view);
    if(!info) return viewer.clearMarkSet();
    // brightness carries the per-neuron drive when it is heterogeneous, so a spread that leaves some cells barely driven is visible as such
    const g = info.gain;
    let gmax = 1;
    if(g) for(const x of g) if(x > gmax) gmax = x;
    const base = info.amp >= 0 ? [1, 0.85, 0.25] : [0.45, 0.7, 1];
    const colOf = g
      ? k => { const f = 0.25 + 0.75*(g[k]/gmax); return [base[0]*f, base[1]*f, base[2]*f]; }
      : () => base;
    const modes = ['constant', 'pulse', 'ramp', 'noise'];
    return put(info.idx, colOf,
      `stimulus · ${info.total.toLocaleString()} neurons · ${modes[info.mode] || ''} ` +
      `current ${info.amp}` + (g ? ' · brightness is the per-neuron spread' : ''));
  }
  if(view.type === 'input'){
    const info = inputSetInfo(view);
    if(!info) return viewer.clearMarkSet();
    // hue is the channel, so a sheet mapped onto tissue shows its map
    const colOf = k => hueRgb(Math.round(360*(info.chan[k]/Math.max(1, info.chans))));
    return put(info.idx, colOf,
      `input · ${info.cells.toLocaleString()} neurons · ${info.chans} channels · ` +
      `color is the channel` + (info.sheet ? ` (${info.cols} by ${info.rows} sheet)` : ''), 6);
  }
  if(view.type === 'analysis'){
    const info = conjSetInfo(view);
    if(!info) return viewer.clearMarkSet();
    const colOf = k => hueRgb(Math.round((info.item[k]*137.507764) % 360));
    return put(info.idx, colOf,
      `conjunctive cells in ${info.label} · ${info.items.length} of ${info.nItems} items ` +
      `have one · ${info.idx.length} cells · color is the item`, 8);
  }
  viewer.clearMarkSet();
}

// Any run in the project folder, drawable.
// A chart node names the run it draws (its `source`, empty for this session's), so there is one path from a run to a picture rather than a panel mode beside a file link: the FILES tab's chart link and the panel's own run picker both set that parameter.
// A sweep is a group of runs and is drawable the same way: its file names the runs it made, so a chart of a sweep is the mean of each variant and condition with the spread across its replicates.
// One run is one draw of a chaotic system, which is the whole reason the sweep exists.
const runRows = new Map();               // tag -> rows, once read
const sweepGroups = new Map();           // sweep tag -> [{ label, tags, histories }]
let runTags = [], sweepTags = [];        // what the folder holds, for the picker
const loadingRun = new Set();
setPropsRuns({
  tags: () => runTags,
  sweeps: () => sweepTags,
  rows: tag => runRows.get(tag) || null,
  groups: tag => sweepGroups.get(tag) || null,
  load: async tag => {
    if(loadingRun.has(tag) || runRows.has(tag)) return;
    loadingRun.add(tag);
    try {
      const ST = await store();
      if(/^sweep-/.test(tag)){
        const sw = await readSweep(ST, tag);
        sweepGroups.set(tag, sw.groups);
        // the pickers read a history for the populations and measures a run recorded; any run of the sweep answers that
        runRows.set(tag, (sw.groups[0] && sw.groups[0].histories[0]) || []);
      } else runRows.set(tag, await readRunMetrics(ST, tag));
      if(propsNode && propsNode.type === 'chart') showProps(propsNode);
    } catch(e){ status('could not read ' + tag + ': ' + e.message, true); }
    finally { loadingRun.delete(tag); }
  },
});
// The picker offers what the folder holds; read once, and again whenever a run is charted from the FILES tab, which is when a new one can have appeared.
async function refreshRunTags(){
  try { const ST = await store(); runTags = await listRunTags(ST); sweepTags = await listSweepTags(ST); }
  catch(e){ runTags = []; sweepTags = []; }
}
refreshRunTags();
// Draw a run in the chart nodes: name it on each of them and show the first.
globalThis.__npChartRun = (rows, tag) => {
  if(Array.isArray(rows) && rows.length) runRows.set(tag, rows);
  const charts = editor.nodes.filter(n => n.type === 'chart');
  if(!charts.length){ status('no chart node in this graph to draw ' + tag + ' in', true); return; }
  for(const c of charts){ c.params.source = tag; editor.markDirty(c); }
  refreshRunTags();
  editor.selectExternal(charts[0]);
  status(`chart nodes now draw run ${tag}`);
};
// The item by cell response matrices the trainer measured, newest per population.
// Kept beside the metrics history for the same reason: a chart cannot draw what the run threw away.
const matrices = new Map();
globalThis.__npMatrix = (pop, data) => {
  matrices.set(pop, data);
  if(propsNode && propsNode.type === 'chart') showProps(propsNode);
};
setPropsMatrices(() => matrices);
// The weights the engine currently holds, with the rule byte beside them so a distribution can be split into what learns and what is frozen.
// Reading the wiring's own array instead would draw the starting weights and call them the learned ones, which is the one mistake this plot must not make.
setPropsWeights(() => {
  if(!worker || !doc.curNet || !offers(worker, 'getWeights', 'weight readback')) return Promise.resolve(null);
  return new Promise(res => {
    const guard = setTimeout(() => { pendingWeights = null; res(null); }, 8000);
    pendingWeights = d => { clearTimeout(guard); res(d); };
    worker.postMessage({ cmd:'getWeights' });
  }).then(d => d && d.w ? { w:d.w, pmask:doc.curNet.pmask, t:d.t } : null);
});
setPropsMetrics(label => {
  if(!lastMetrics || !lastMetrics.readout || !lastMetrics.readout.pops) return null;
  const v = lastMetrics.readout.pops[label];
  if(!v) return null;
  const pr = (lastMetrics.probes || []).find(p => p.l === label);
  return { simMin:lastMetrics.simMin, hz:pr ? pr.hz : undefined,
    decode:v.decode, chance:v.chance !== undefined ? v.chance : 1/26,
    selectivity:v.selectivity, bind:v.bind, timing:v.timing };
});
globalThis.__npNetSize = (count, syn) => setNetSize(count, syn);
// A training run takes over the computation progress hook, so it hands the bar back through here rather than leaving the top of the window empty through the one wiring the run depends on.
globalThis.__npComputeBar = (f, syn) => {
  if(f >= 1){ wireDone(); return; }
  computebarEl.style.display = 'block';
  computebarEl.style.transform = `scaleX(${f})`;
  status(`wiring synapses ${Math.round(f*100)}%` +
    (syn > 0 ? ` · ${fmtSyn(syn)} so far` : ''));
};
// A computation that was thrown away says nothing rather than leaving its last progress reading standing as though it were the current state.
function wireCancelled(){
  wireDone();
  if(/^wiring synapses/.test(statusEl.textContent)) status('');
}
function wireDone(){ computebarEl.style.display = 'none'; computebarEl.style.transform = 'scaleX(0)'; }

async function recompute(){
  // A run owns the network and the engine.
  // Computing a second one underneath it would compete for the card and leave the viewer showing a network nobody is simulating, so edits during a run change the graph on screen and nothing else; the run keeps going on the graph it was given.
  if(trainingRunning){
    status('a training run is going: edits are not computed until it finishes');
    return;
  }
  const byId = id => editor.byId(id);
  pushHistory();
  // a scene that is not in the viewer is edited and saved and nothing more, until a viewer key is pressed in its tab
  if(tabs.cur !== tabs.shown){ computePoints(); status(tabs.cur.label + ': ' + viewHint()); return; }
  const view = doc.viewOf(editor);
  const D = doc, gen = ++D.computeGen;        // stale awaits bail; the sim keeps running the current net meanwhile
  try {
    if(!view) throw new Error('nothing to view: add a checkpoint node');
    let net = await computeNode(view, byId);
    if(editor.errorNode !== null){ editor.errorNode = null; editor.draw(); }
    // A superseded computation still owns the progress bar and the status line it was drawing.
    // Bailing without clearing them would leave the bar frozen at whatever fraction it had reached and the words 'wiring synapses 43%' on screen over a run that has long since started.
    // Starting a training run does exactly this to the interactive computation underneath it: doc.computeGen is bumped on purpose there.
    if(gen !== D.computeGen || D !== doc || tabs.cur !== tabs.shown){ wireCancelled(); return; }
    wireDone();
    if(!net) throw new Error('viewed node has no result (bypassed or unwired)');
    if(net.kind === 'points'){           // static preview; the sim keeps running underneath
      viewer.showPoints(net);
      status(`${net.count.toLocaleString()} points · preview` + (worker ? ' (sim continues)' : ''));
      return;
    }
    if(net.kind === 'geo'){
      viewer.showGeo(net);
      status('geometry preview' + (worker ? ' (sim continues)' : ''));
      return;
    }
    if(net.kind !== 'net') throw new Error('this node has no viewable result');
    // what this network means for the running engine: tune it in place, start a new one, or refuse under the weight lock (document.js, which also applies an attached or pending brain, since that decides it)
    const plan = doc.planCompute(net, view, !!worker);
    if(plan.action === 'locked'){ status(plan.why, true); return; }
    net = plan.net;
    const tune = plan.action === 'tune', brainMsg = plan.brainMsg;
    // Adopt the checkpoint's steps only when the author changed it, so rewiring does not stomp a speed the person watching has chosen.
    const authored = net.steps || 2;
    if(authored !== lastAuthoredSteps){ lastAuthoredSteps = authored; setSpeed(authored); }
    // an engine change (or enabling plasticity while on the gpu engine) needs a worker restart; dynamics reset, which an engine switch implies anyway
    const eng = tune ? doc.planEngine(net, workerRuns) : { action:'keep' };
    if(eng.action === 'locked'){ status(eng.why, true); return; }
    if(eng.action === 'restart'){
      viewer.setNetwork(net); startSim(net);
      if(viewer.selected >= 0) viewer.onSelect(viewer.selected);
    }
    else if(tune){
      // a term the running engine did not name in its hello is refused here as at the start, rather than tuned into an engine that would run without it
      const miss = engineHello ? missingTerms(engineTerms(net), engineHello) : [];
      if(miss.length){
        const h = engineHello;
        stopSim();
        refusedHello = h;
        if(propsNode && propsNode.type === 'checkpoint') showProps(propsNode);
        status('the ' + h.engine + ' engine does not implement ' + miss.map(m => m[1]).join(', ') +
          '; the simulation stopped. Switch the checkpoint to an engine that does, or turn the term off.', true);
        return;
      }
      // an engine still answering hello has had no init yet, and the init it gets carries this network's settings
      if(!helloPending){ worker.postMessage(tuneMessage(net)); io.attach(worker, net.inputMaps, net.seed); }
      if(viewer.mode !== 'net') viewer.setNetwork(net);   // back from a preview: reattach, no reset
    }
    else {
      viewer.setNetwork(net); startSim(net);
      // the selection carried over by identity, but its query went to the worker that no longer exists; ask the new one
      if(viewer.selected >= 0) viewer.onSelect(viewer.selected);
      if(fitNextCompute){ fitNextCompute = false; viewer.fitToNet(); }
      if(firstOpenInspect){ firstOpenInspect = false; firstView(net); }
    }
    doc.lastSimView = view.id;
    if(tune){ recorder.setPopulations(net.probes || []); if(recording) recording.setPopulations(net.probes || []); }   // a new engine resets the recording in startSim; a tune keeps it
    viewer.refreshTypeColors();      // a cell type row's hue may have changed
    applyPopulationColors(net);
    applyPopulationBoxes(net);
    markViewSet(view);
    setupProbes(net);
    // a probe on a graded population reads the potentials, so the engine that is already running is asked for them here as well as at init
    { const eng = inspectEngine(); if(eng && probesWantV() && offers(eng, 'sendV', 'membrane potentials')) eng.postMessage({ cmd:'sendV', on:true }); }
    setupInputs(net, io);
    tagHueMap = sceneTagHues(net);   // colors follow the set of tags
    // A lit pathway is a list of neuron indices into the network that was wired when it was switched on, so new wiring invalidates every line.
    // The pathway itself is named by tags, which survive rewiring, so each one is asked for again against the new network once the sim is up; a pathway whose populations are gone reports that and goes out.
    relightPathways();
    setNetSize(net.count, net.synCount);
    // A pair table row or a projection naming a rule that nothing declares falls back to the default rule rather than failing the wiring, since bypassing a plasticity node is an ordinary thing to do and it would otherwise take the scene down.
    // Falling back quietly is the other half of that bargain and not acceptable, so it is said here.
    const warn = (net.ruleWarn && net.ruleWarn.length
      ? `  ·  no rule named ${net.ruleWarn.map(n => `"${n}"`).join(', ')}, ` +
        'using the checkpoint default'
      : '') + (net.delayClamped
      // a delay the ring buffer cannot hold is a wrong delay, and the wiring says so rather than quietly shortening it
      ? `  ·  ${net.delayClamped.toLocaleString()} synapses clamped to the ${net.delayMax} ms delay ceiling`
      : '') + (net.connectionsReport
      // a table row whose ids are not on the stream places nothing, and the status line says how many rather than wiring what it could in silence
      ? `  ·  connections file: ${net.connectionsReport.synapses.toLocaleString()} synapses from ${net.connectionsReport.matched.toLocaleString()} rows` +
        (net.connectionsReport.unmatched ? `, ${net.connectionsReport.unmatched.toLocaleString()} rows with an id not on the stream` : '') +
        (net.connectionsReport.self ? `, ${net.connectionsReport.self.toLocaleString()} self-connections skipped` : '')
      : '');
    if(startRefused && !worker) status(startRefused, true);
    else status(`${net.count.toLocaleString()} neurons · ${net.synCount.toLocaleString()} synapses`
      + brainMsg + warn + enginePart(net));
  } catch(e){
    // the node whose computation failed pulses red in the graph until a computation succeeds
    if(e && e.nodeId !== undefined && e.nodeId >= 0){ editor.errorNode = e.nodeId; editor.draw(); }
    // Same for a computation that was canceled rather than failing: a superseded wiring throws with e.superseded.
    if(gen !== D.computeGen || D !== doc || e.superseded){ wireCancelled(); return; }
    wireDone();
    doc.curNet = null; viewer.setNetwork(null); stopSim(); io.stop(); setupProbes(null);
    // a computation names its node; anything else that threw here (the viewer, the probes, the engine start) says so, since a bare message named nothing and read as a node's fault
    const byNode = e && e.nodeId !== undefined && e.nodeId >= 0 ? editor.byId(e.nodeId) : null;
    const where = byNode ? (byNode.name || byNode.type) + ': ' : (e && e.nodeId !== undefined ? '' : 'after the computation, showing the network: ');
    status(where + e.message, true);
    console.error(e);
  }
}

function stopSim(){ if(worker){ worker.terminate(); worker = null; } inFlight = false; helloPending = false; engineHello = null; sendInit = null; io.detach(); }

// ---- what the running engine said it is (ENGINE.md section 2, hello) ----
// engineHello (declared with the worker) is the reply of this page's own engine: its name, contract, the terms it implements and the optional queries it answers.
// The init waits for it and is sent only once it has been checked against the network, and no tick goes before the init.
const HELLO_WAIT_MS = 10000;
setPropsEngine(() => engineHello || refusedHello);
// Whether an engine answers an optional query.
// Only this page's own engine has been asked; a training run's engine is the trainer's and is asked as before.
// What the engine does not answer is not sent, and the status line says once per engine what is unavailable.
function offers(eng, q, what){
  if(eng !== worker || answers(engineHello, q)) return true;
  if(engineHello && !lackSaid.has(q)){ lackSaid.add(q); status('the ' + engineHello.engine + ' engine does not answer ' + what, true); }
  return false;
}
// What the status line after a computation says about the engine: which remote engine runs the network, and how many keys of the engine settings block the running engine ignores (the stock engines read none).
// A new engine has not answered hello when that line is written, so its answer adds the note.
function enginePart(net){
  if(!engineHello || !net) return '';
  const n = ignoredCustom(net, engineHello);
  return (workerRuns.kind === 'remote' ? '  ·  on the ' + engineHello.engine + ' engine at ' + (engineHello.url || workerRuns.host) : '') +
    (n ? '  ·  ' + customNote(n, engineHello) : '');
}
function onHello(r){
  const bad = checkHello(r);
  const miss = !bad && doc.curNet ? missingTerms(engineTerms(doc.curNet), r) : [];
  if(bad || miss.length){
    stopSim();
    refusedHello = bad ? null : r;
    if(propsNode && propsNode.type === 'checkpoint') showProps(propsNode);
    status((bad || 'the ' + r.engine + ' engine does not implement ' + miss.map(m => m[1]).join(', ')) +
      '; the network was not run. Switch the checkpoint to an engine that does, or turn the term off.', true);
    return;
  }
  engineHello = r; helloPending = false; inFlight = false;
  if(sendInit){ const go = sendInit; sendInit = null; go(); }
  if((+vmEl.value > -100 || probesWantV()) && offers(worker, 'sendV', 'membrane potentials'))
    worker.postMessage({ cmd:'sendV', on:true });   // the filter, or a graded probe
  if(propsNode && propsNode.type === 'checkpoint') showProps(propsNode);   // gray what this engine lacks
  const note = enginePart(doc.curNet);
  if(note && !statusEl.textContent.includes(note)) status(statusEl.textContent.replace(/^connecting to the engine at .*$/, '') + note);
}

// The engine choice (wantGpu, gpuMissing), the init message and the tune message are the document's (document.js), where a test can reach them.
function startSim(net, forceCpu){
  stopSim();
  recorder.reset(); recorder.setPopulations(net.probes || []);
  if(recording){ recording.mark('rewire', net.count + ' cells'); recording.setPopulations(net.probes || []); }   // a new engine; the record runs on
  if(net.synCount && net.post.byteLength === 0){   // buffers already transferred away
    status('network buffers live in a previous sim; edit connect to rewire', true);
    return;
  }
  const kind = forceCpu ? 'cpu' : engineKind(net);
  workerIsGpu = kind === 'gpu';
  workerRuns = { kind, host: kind === 'remote' ? net.liveHost || '' : '' };
  startRefused = '';
  if(kind === 'remote' && !/^wss?:\/\/\S+$/.test(net.liveHost || '')){
    startRefused = 'the checkpoint\'s engine address "' + (net.liveHost || '') + '" is not a ws:// address, so the network was not run';
    status(startRefused, true);
    return;
  }
  if((net.engine|0) === 2 && !workerIsGpu && !forceCpu){
    const miss = gpuMissing(net);
    status(miss.length
      ? 'gpu engine does not implement ' + miss.join(', ') + '; using the cpu engine'
      : 'WebGPU unavailable; using the cpu engine', true);
  }
  // the remote engine is a WebSocket behind the Worker surface (remoteworker.js); every message below is the same
  worker = kind === 'remote' ? new RemoteEngine(net.liveHost)
    : new Worker(new URL((workerIsGpu ? './gpuworker.js' : './simworker.js') + '?p=' + PROTOCOL,
      import.meta.url), { type:'module' });
  if(kind === 'remote') status('connecting to the engine at ' + net.liveHost + '…');
  // Asked first: what the engine is and what it implements.
  // The init is sent when the answer has been checked (onHello), so a network an engine cannot run never reaches it.
  const w = worker;
  engineHello = null; refusedHello = null; lackSaid = new Set(); helloPending = true; inFlight = true;
  worker.postMessage({ cmd:'hello' });
  const helloTimer = setTimeout(() => {
    if(worker !== w || !helloPending) return;
    stopSim();
    status('the engine did not answer hello within ' + HELLO_WAIT_MS/1000 + ' s, so the network was not run', true);
  }, HELLO_WAIT_MS);
  sendInit = () => {
    // A tune that arrived while the engine was answering hello changed the network's settings and not its arrays, and the init carries it.
    const cur = doc.curNet && doc.curNet.preStart === net.preStart ? doc.curNet : net;
    // over-budget nets (bigger than the memo cache would keep anyway) transfer their synapse buffers so exactly one copy exists, inside the sim worker; smaller nets are copied so the memo can restore them instantly.
    // Downstream reuse of a transferred net stays valid: the detached arrays still work as identity tokens for the tune fast-path, and every real rewiring produces fresh buffers.
    const big = cur.synCount*13 + cur.count*25 > 1.6e9;
    // A big net transfers its synapse arrays instead of cloning them, and a transfer detaches them here. doc.curNet.pmask is read afterwards by the weight readback, so it is copied first; below full density this costs nothing because the branch does not transfer at all.
    const pmask0 = cur.pmask ? cur.pmask.slice() : null;
    w.postMessage(initMessage(cur),
      big ? [cur.preStart.buffer, cur.post.buffer, cur.w.buffer, cur.delay.buffer,
        ...(cur.pmask ? [cur.pmask.buffer] : [])] : []);
    if(cur.pmask){ cur.pmask = pmask0; net.pmask = pmask0; }   // put the readable copy back in its place
    io.attach(w, cur.inputMaps, cur.seed);
  };
  worker.onmessage = e => {
    if(e.data.cmd === 'hello'){ clearTimeout(helloTimer); if(worker === w) onHello(e.data); return; }
    if(e.data.cmd === 'error'){
      const msg = e.data.message || 'engine error';
      // a remote engine that cannot be reached or drops the connection is stopped and said, never replaced by another engine
      if(workerRuns.kind === 'remote'){
        clearTimeout(helloTimer);
        if(worker === w) stopSim();
        status('remote engine: ' + msg + '; nothing is running', true);
        return;
      }
      if(workerIsGpu && doc.curNet && doc.curNet.post.byteLength){
        status('gpu engine: ' + msg + '; falling back to the cpu engine', true);
        startSim(doc.curNet, true);
      }
      else status(msg, true);
      return;
    }
    if(e.data.cmd === 'weights'){
      if(pendingWeights){ pendingWeights(e.data); pendingWeights = null; }
      return;
    }
    if(e.data.cmd === 'queryResult'){
      if(!pathwayReply(e.data))
        viewer.applyQuery(e.data.idx, e.data.out, e.data.inn, e.data.outTotal, e.data.inTotal);
      return;
    }
    if(e.data.cmd !== 'state') return;
    if(attachRemote) return;   // display belongs to the live run
    inFlight = false;
    handleState(e.data);
  };
  if(ATTACH_RUN) startAttach();
}

// The live recording behind the chart node's live plots: a rate per frame per probed population, a rolling raster, interval histograms.
// Observation only (record.js); fed every state message, reset with every engine.
const recorder = new Recorder();
setPropsLive(() => ({ rec:recorder, scope:() => scopeBuf, recording:() => recording || lastRecording }));
function handleState(d){
  if(d.v) viewer.setPotentials(d.v, d.fired);
  recorder.onState(d.fired, d.steps);
  if(recording){
    recording.onState(d.fired, d.steps);
    if(recording.frames % 25 === 0) recEl.textContent = 'REC ' + recording.seconds().toFixed(0) + ' s · ' + recording.edits();
  }
  updateProbes(d.fired, d.steps, false, d.v, io.sweepDirection);
  viewer.onFired(d.fired);
  drawRaster(d.fired);
  rateAcc.spikes += d.spikes; rateAcc.steps += d.steps; lastTickAt = performance.now();
  if(doc.curNet && d.fired && d.fired.length === doc.curNet.count){
    const se = excitatoryMask(doc.curNet), f = d.fired; let e = 0;
    for(let i = 0; i < f.length; i++) if(f[i]) e += se[i];
    rateAcc.exc += e;
  }
  if(viewer.selected >= 0 && d.vtrace){
    for(const v of d.vtrace) scopeBuf.push(v);
    if(scopeBuf.length > scopeCv.width) scopeBuf.splice(0, scopeBuf.length - scopeCv.width);
    drawScope();
  }
}

function startAttach(){
  if(attachRemote){ try { attachRemote.terminate(); } catch(e){} }
  attachRemote = new RemoteEngine(ATTACH_HOST);
  attachRemote.postMessage({ __host:'attach', dir:ATTACH_RUN });
  attachRemote.onmessage = e => {
    const m = e.data;
    if(m.__attached){
      status('attached to live run ' + (m.__attached.dir || ATTACH_RUN) +
        ' at sim ' + (m.__attached.simMs/60000).toFixed(1) + ' min, running to ' +
        (m.__attached.untilMs/60000).toFixed(0) + ' min');
      return;
    }
    if(m.cmd === 'error'){ status('attach: ' + m.message, true); return; }
    if(m.cmd === 'state') handleState(m);
  };
}

// ---- raster ----
let rasterIdx = null;
function setupRaster(n){
  const K = Math.min(220, n); rasterIdx = new Int32Array(K);
  for(let k=0;k<K;k++) rasterIdx[k] = Math.floor(k*n/K);
  const c = rasterCv.getContext('2d');
  c.fillStyle = '#000'; c.fillRect(0, 0, rasterCv.width, rasterCv.height);
}
function drawRaster(fired){
  if(!doc.curNet) return;
  if(!rasterIdx || Math.floor((rasterIdx.length-1)*doc.curNet.count/rasterIdx.length) >= doc.curNet.count)
    setupRaster(doc.curNet.count);
  if(rasterCv.width !== rasterCv.clientWidth) rasterCv.width = rasterCv.clientWidth;
  const c = rasterCv.getContext('2d'), w = rasterCv.width, h = rasterCv.height;
  c.drawImage(rasterCv, -1, 0);
  c.fillStyle = '#000'; c.fillRect(w-1, 0, 1, h);
  c.fillStyle = '#fff';
  const K = rasterIdx.length;
  for(let k=0;k<K;k++)
    if(fired[rasterIdx[k]]) c.fillRect(w-1, Math.floor(k*h/K), 1, 1);
}

// ---- simulation speed ----
// Milliseconds of simulated time per frame.
// The checkpoint node's steps param is the authored default; this is how fast you happen to be watching, the same relationship the resolution slider has to the authored network.
// It changes nothing about the dynamics, which always advance 1 ms at a time, only how much of that is done between two looks at the viewer.
let simSpeed = 2;
const speedEl = document.getElementById('speed');
const speedLabel = document.getElementById('speedlabel');
function showSpeed(){
  // Named for what it sets, not for what it achieves.
  // A label of SPEED would promise something it cannot deliver: the same 10 ms per frame runs the guided tour near realtime and a 157M synapse scene at a quarter of it, because the engine's ability to keep up is what varies.
  // The realtime figure sits beside the slider and is the comparable number.
  speedLabel.textContent = 'PER FRAME ' + simSpeed + ' ms';
}
function setSpeed(v, fromUser){
  simSpeed = Math.max(1, Math.min(20, v|0));
  if(!fromUser) speedEl.value = String(simSpeed);
  showSpeed();
}
speedEl.oninput = () => setSpeed(+speedEl.value, true);
setSpeed(2);

// ---- main loop ----
// The tick rides on animation frames, which a hidden tab or a window behind another one stops.
// Everything downstream stops with them: the engine takes no steps, the raster stops scrolling, and the readouts below blank themselves because no state arrived, so the viewer looks dead in every scene while the window stays covered, and no reload brings it back.
// A timer takes over whenever frames stop arriving: slower than a frame, but the simulation keeps going and the numbers keep meaning something.
let lastStep = 0;
function step(){
  lastStep = performance.now();
  io.frame(lastStep);
  updateInputs();
  if(worker && !inFlight && !helloPending && !paused && doc.curNet && !attachRemote){
    inFlight = true;
    worker.postMessage({ cmd:'tick', steps:simSpeed });
  }
}
function loop(){
  requestAnimationFrame(loop);
  step();
  viewer.frame();
}
requestAnimationFrame(loop);
setInterval(() => { if(performance.now() - lastStep > 250) step(); }, 100);

// attach boot: the run's own brain file is the configuration, so nothing has to be retyped; the newest checkpoint in the run directory wins
if(ATTACH_RUN){
  // patient by design: a live run saves its first brain only at the first checkpoint, so the page waits and keeps saying so instead of failing; leave the tab open and it comes alive on its own
  (async () => {
    for(let tries = 0;; tries++){
      let names = [];
      try {
        const html = await (await fetch('host/runs/' + ATTACH_RUN + '/')).text();
        names = [...html.matchAll(/href="([^"]*\.npb)"/g)].map(m2 => m2[1]);
      } catch(e){ /* directory not created yet */ }
      if(names.length){
        try {
          names.sort();
          const file = names[names.length - 1].split('/').pop();
          const buf = await (await fetch('host/runs/' + ATTACH_RUN + '/' + file)).arrayBuffer();
          status('attach: graph from ' + file);
          loadBrainBuffer(buf, { replace:true });
          return;
        } catch(e){ status('attach failed: ' + e.message, true); return; }
      }
      status('attach: waiting for the first checkpoint of ' + ATTACH_RUN +
        ' (no brain saved yet; checking every 20 s, attempt ' + (tries + 1) + ')', true);
      await new Promise(r => setTimeout(r, 20000));
    }
  })();
}

setInterval(() => {
  // A blank readout would mean three things at once: no network, paused, and nothing stepped this second.
  // It says which.
  if(!doc.curNet || !rateAcc.steps){
    // A window behind another one keeps stepping off a timer, slowly, and one of those steps does not always land inside this second.
    // The readout holds what it last said for a moment rather than flickering between a rate and nothing.
    if(doc.curNet && !paused && performance.now() - lastTickAt < 3000) return;
    rateEl.textContent = !doc.curNet ? '' : paused ? 'paused' : 'no steps';
    rtfEl.textContent = '';
    return;
  }
  const hz = rateAcc.spikes / (doc.curNet.count * rateAcc.steps * 0.001);
  rateEl.textContent = `${hz.toFixed(1)} Hz mean`;
  // achieved, not requested: simulated milliseconds accumulated over one second of wall clock.
  // Paused or throttled runs read what they really did.
  const rtf = rateAcc.steps / 1000;
  rtfEl.textContent = (rtf >= 1 ? rtf.toFixed(2) : rtf.toFixed(3)) + 'x realtime';
  // only full-resolution runs define the compensation target; a downscaled sim feeding its own rate back in would chase itself.
  // A rate per class: the interneurons of a scene on the measured rows fire several times faster than its pyramids, and one rate for both under-restores the inhibitory mean input
  if(+resEl.value === 100){
    excitatoryMask(doc.curNet);
    const nE = signCount, nI = doc.curNet.count - nE, per = rateAcc.steps*0.001;
    if(nE > 0 && nI > 0) setRateHint({ E:rateAcc.exc/(nE*per), I:(rateAcc.spikes - rateAcc.exc)/(nI*per) });
    else setRateHint(hz);
  }
  rateAcc = { spikes:0, steps:0, exc:0 };
}, 1000);

// ---- resizable panels ----
const UI_STORE = 'neuron-playground-ui-v1';
const appEl = document.getElementById('app');
// Pane layouts.
// Each is a grid: a side column beside the viewer (or, in wide, beside the graph) and a lower pane under something, so every layout has the same two numbers, the side column's width and the lower pane's height, moved by the same two handles, and keeps its own pair.
// The top bar's layout button steps through them like the viewer bar's filters.
const LAYOUTS = {
  right:   { name:'viewer right', icon:'layoutRight', sideLeft:true, side:460, lower:340,
    areas:'"top top top" "props vsplit viewer" "hsplit vsplit viewer" "graph vsplit viewer"',
    tip:'the viewer fills the right; the properties sit over the node graph on the left' },
  classic: { name:'viewer over graph', icon:'layoutClassic', sideLeft:false, side:300, lower:320,
    areas:'"top top top" "viewer vsplit props" "hsplit vsplit props" "graph vsplit props"',
    tip:'the viewer over the node graph, the properties down the right' },
  left:    { name:'viewer left', icon:'layoutLeft', sideLeft:false, side:460, lower:340,
    areas:'"top top top" "viewer vsplit props" "viewer vsplit hsplit" "viewer vsplit graph"',
    tip:'the viewer fills the left; the properties sit over the node graph on the right' },
  wide:    { name:'viewer across the top', icon:'layoutWide', sideLeft:false, side:360, lower:340,
    areas:'"top top top" "viewer viewer viewer" "hsplit hsplit hsplit" "graph vsplit props"',
    tip:'the viewer across the top; the node graph and the properties side by side below' },
};
const LAYOUT_ORDER = ['right', 'classic', 'left', 'wide'];
const panels = { train:240, layout:'classic', sizes:{} };
try {
  const saved = JSON.parse(localStorage.getItem(UI_STORE)) || {};
  // sizes kept before there were layouts belong to the one there was
  if(saved.props !== undefined || saved.graph !== undefined){
    saved.sizes = saved.sizes || {};
    if(!saved.sizes.classic) saved.sizes.classic = { side:+saved.props || 300, lower:+saved.graph || 320 };
    delete saved.props; delete saved.graph;
  }
  Object.assign(panels, saved);
} catch(e){}
if(!LAYOUTS[panels.layout]) panels.layout = 'classic';
if(!panels.sizes || typeof panels.sizes !== 'object') panels.sizes = {};
const sizeOf = () => {
  const L = LAYOUTS[panels.layout];
  const z = panels.sizes[panels.layout] || (panels.sizes[panels.layout] = { side:L.side, lower:L.lower });
  return z;
};
// The text-size setting scales the panels that are made of text, so their tracks have to grow with it: a panel kept at its stored width gives its zoomed contents that width divided by the factor.
const uiScale = () =>
  +getComputedStyle(document.documentElement).getPropertyValue('--ui') || 1;
function applyPanels(){
  const ui = uiScale(), L = LAYOUTS[panels.layout], z = sizeOf();
  z.side = Math.max(180, Math.min((window.innerWidth - 360)/ui, z.side));
  z.lower = Math.max(80, Math.min(window.innerHeight - 200, z.lower));
  const side = Math.round(z.side*ui) + 'px';
  appEl.dataset.layout = panels.layout;
  appEl.style.gridTemplateAreas = L.areas;
  appEl.style.gridTemplateColumns = L.sideLeft ? `${side} 5px minmax(0,1fr)` : `minmax(0,1fr) 5px ${side}`;
  appEl.style.gridTemplateRows = `${Math.round(28*ui)}px minmax(0,1fr) 5px ${Math.round(z.lower)}px`;
  const tp = document.getElementById('trainPane');
  if(tp && tp.style.display !== 'none'){
    // Clamp against the room the properties panel needs, not against the window.
    // A bound that let the training pane take all but 220 px of the whole window would, on a 720 px screen with a stored 500, leave properties 187 px, cut off mid-parameter, while most of the training pane sat empty.
    // Properties keeps 240 px and the pane takes what is left.
    // The column is the window's height where properties has the whole side, the lower pane in wide, and what the graph leaves above it elsewhere.
    const full = window.innerHeight - 28 - 5;
    const colH = panels.layout === 'classic' ? full : panels.layout === 'wide' ? z.lower : full - z.lower - 5;
    panels.train = Math.max(90, Math.min(Math.max(90, colH - 240), panels.train));
    tp.style.height = panels.train + 'px';
  }
  try { localStorage.setItem(UI_STORE, JSON.stringify(panels)); } catch(e){}
}
function bindSplit(id, key, vertical){
  document.getElementById(id).addEventListener('mousedown', e => {
    e.preventDefault();
    const start = vertical ? e.clientY : e.clientX;
    const base = key === 'train' ? panels.train : sizeOf()[key];
    const move = ev => {
      // the properties panel is stored in its own units, which the text scale multiplies on the way to the grid, so a drag in screen pixels is divided by it on the way back.
      // A side column on the left grows as the handle moves right; one on the right, as it moves left.
      let d = start - (vertical ? ev.clientY : ev.clientX);
      if(key === 'side' && LAYOUTS[panels.layout].sideLeft) d = -d;
      if(key === 'train') panels.train = base + d;
      else sizeOf()[key] = base + (vertical ? d : d/uiScale());
      applyPanels();
      if(key !== 'train') window.dispatchEvent(new Event('resize'));
    };
    const up = () => { window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up); };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  });
}
{
  const btn = document.getElementById('layoutbtn');
  const show = () => {
    const L = LAYOUTS[panels.layout];
    btn.innerHTML = icon(L.icon);
    btn.classList.add('icb', 'icon-only');
    btn.setAttribute('aria-label', 'pane layout: ' + L.name);
    btn.title = 'pane layout: ' + L.tip + '. Click for the next one; each layout keeps the pane sizes you gave it.';
  };
  btn.onclick = () => {
    panels.layout = LAYOUT_ORDER[(LAYOUT_ORDER.indexOf(panels.layout) + 1) % LAYOUT_ORDER.length];
    applyPanels(); show();
    window.dispatchEvent(new Event('resize'));     // the viewer's canvas and the graph take their new boxes
    // the graph pane has a new shape, so what was framed for the last one is off to a side or cut off: frame it again once the grid has settled
    requestAnimationFrame(() => { editor.frameAll(); editor.draw(); });
    status('pane layout: ' + LAYOUTS[panels.layout].name);
  };
  show();
}
bindSplit('vsplit', 'side', false);
bindSplit('hsplit', 'lower', true);
bindSplit('tsplit', 'train', true);
window.addEventListener('resize', applyPanels);
applyPanels();

// The synapse budget: the largest wiring this machine is allowed to attempt, in millions.
{
  const inp = document.getElementById('synbudget');
  const KEY = 'neuron-playground-max-synapses';
  const show = () => {
    const v = +localStorage.getItem(KEY);
    inp.value = v >= 1e6 ? String(Math.round(v/1e6)) : '';
    inp.placeholder = String(Math.round(deviceSynLimit()/1e6));
  };
  inp.onchange = () => {
    const m = +inp.value;
    try {
      if(m >= 1) localStorage.setItem(KEY, String(Math.round(m*1e6)));
      else localStorage.removeItem(KEY);
    } catch(e){}
    show();
  };
  show();
}

// Text size for the whole app, kept for this machine.
// The node graph is not in it: that canvas has its own zoom on the wheel, and its labels sit inside boxes of a fixed size which larger text would overflow.
// Panels are re-rendered because a few of them measure text as they build.
{
  const rng = document.getElementById('uiscale');
  const KEY = 'np-ui-scale';
  const SIZES = [0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8];
  // a saved size off the list snaps to the nearest one on it
  const clamp = v => SIZES.reduce((a, b) => Math.abs(b - (+v || 1)) < Math.abs(a - (+v || 1)) ? b : a);
  for(const s of SIZES) rng.add(new Option(Math.round(s*100) + '%', String(s)));
  let ui = clamp(localStorage.getItem(KEY) || 1);
  const apply = () => {
    document.documentElement.style.setProperty('--ui', String(ui));
    rng.value = String(ui);
    applyPanels();
    if(propsNode) showProps(propsNode);
  };
  rng.onchange = () => {
    ui = clamp(rng.value);
    try { localStorage.setItem(KEY, String(ui)); } catch(e){}
    apply();
  };
  apply();
}


// ---- resolution slider: one knob, the compensation included ----
const resEl = document.getElementById('res'), resLabel = document.getElementById('reslabel');
function applyResolution(v, recomputeNow){
  resEl.value = v; liveRes = v;
  resLabel.textContent = `RES ${v}%`;
  setResolution(v/100);
  if(recomputeNow){
    for(const n of editor.nodes) n._cache = null;   // resolution is compute-global: invalidate
    scheduleCompute();                                 // (memo still restores visited levels instantly)
  }
}
let liveRes = +resEl.value;
resEl.oninput = () => {
  if(!needShown('the resolution in the viewer bar is ' + (tabs.shown.label || 'the shown scene') + "'s")){ resEl.value = liveRes; return; }
  applyResolution(+resEl.value, true);
};

// ---- weight lock: protect learned state from silent rewiring ----
// Not labeled WIRING / LIVE, which would be the easiest thing in the app to misread: wiring means growing the synapses everywhere else, including the progress line in this same bar, so a button reading WIRING looks like it would rewire the network when it means the opposite.
// And a label carrying the state rather than the action never makes it obvious which of the two a click gets.
// It locks the learned weights, so it says that, and it says which way round it currently is.
const liveEl = document.getElementById('live');
liveEl.onclick = () => {
  doc.livePhase = !doc.livePhase;
  if(recording) recording.mark(doc.livePhase ? 'weights locked' : 'weights unlocked');
  liveEl.innerHTML = icon(doc.livePhase ? 'lock' : 'unlock');
  liveEl.setAttribute('aria-label', doc.livePhase ? 'weights locked' : 'lock weights');
  liveEl.classList.toggle('on', doc.livePhase);
  resEl.disabled = doc.livePhase;               // proxy neurons are different neurons
};
function exitLive(){ if(doc.livePhase) liveEl.onclick(); }

// ---- potential filter: hide neurons below a membrane-potential floor ----
// per-neuron potentials stream from the engine only while the filter is active (sendV), so the default costs nothing
const vmEl = document.getElementById('vmfloor');
const vmValEl = document.getElementById('vmfloorval');
function applyVmFilter(){
  const v = +vmEl.value;
  viewer.setPotentialFloor(v);
  vmValEl.textContent = v <= -100 ? 'all' : `≥ ${v} mV`;
  // whichever engine owns the network, not just this page's own: during a training run a request to this page's worker would move the label and change nothing, because that worker is not running
  const eng = inspectEngine();
  if(eng && offers(eng, 'sendV', 'membrane potentials')) eng.postMessage({ cmd:'sendV', on: v > -100 || probesWantV() });
}
vmEl.oninput = applyVmFilter;
// a run that starts later has to be told the filter is already on
function pushVmFilter(eng){
  if(eng && +vmEl.value > -100) eng.postMessage({ cmd:'sendV', on:true });
}

// ---- what a training run takes away, and what it leaves ----
// A run owns the network and the engine, so controls that would rewire or re-pace it cannot work while it goes.
// A control left live that does nothing reads as a broken control rather than an unavailable one.
// Each carries the reason it is out, and everything that only reads or only moves things on screen stays in: selecting nodes, reading their settings, dragging them, framing, the camera, and the scope.
//
// All three engines stream membrane potentials, so the potential filter stays available; the refusal path below is kept for any engine that lacks a term, since an engine must refuse what it cannot do rather than ignore it.
const RUN_LOCK = [
  ['res', 'resolution rewires the network from scratch, which a running ' +
    'run cannot survive'],
  ['live', 'a training run already holds the weights; the lock is for ' +
    'interactive building'],
  ['speed', 'the run owns the tick loop and sets its own steps per frame'],
  ['pause', 'the run drives the engine; pausing here would only stop the ' +
    'view of it'],
  // reset, brain save and brain load are node actions, and the properties panel is already locked as a whole while a run owns the network.
];
let vmUnsupported = '';                  // set when an engine refuses sendV

// The run's headline, in the viewer bar beside the controls it has taken away. train.js hands over fields rather than the sentence it puts in the panel, because parsing that sentence back apart would break the first time its wording changed.
const fmtLeft = mins => {
  if(!isFinite(mins) || mins <= 0) return '';
  // rounding would put "0 min left" on a run with twenty seconds to go, which reads as finished rather than nearly finished
  if(mins < 1) return 'under a minute left';
  if(mins < 90) return Math.round(mins) + ' min left';
  const h = mins/60;
  return (h < 24 ? h.toFixed(1) + ' h left' : (h/24).toFixed(1) + ' days left');
};
let runPaused = false;
globalThis.__npTrainHead = d => {
  const bar = document.getElementById('viewerbar');
  if(bar) bar.classList.add('training');
  const st = document.getElementById('tbstate');
  if(st && !runPaused && !runFailed){
    const stalled = watchStall(+d.simMin.toFixed(2));
    st.classList.toggle('failed', stalled);
    st.textContent = stalled ? 'STALLED'
      : d.phase === 'warm-up' ? 'WARM-UP' : 'TRAINING';
    if(stalled) st.title = 'the simulated clock has not advanced for 45 ' +
      'seconds; the engine or the host may have stopped';
    if(!stalled) st.classList.toggle('warmup', d.phase === 'warm-up');
    st.title = d.phase === 'warm-up'
      ? 'plasticity is still off; the run is settling before it learns'
      : 'plasticity is on and the run is learning' +
        (d.pumped ? ', paced by the engine host so this tab cannot stall it' : '');
  }
  const frac = d.targetMin > 0 ? Math.min(1, d.simMin/d.targetMin) : 0;
  const cl = document.getElementById('tbclock');
  if(cl) cl.textContent = `sim ${d.simMin.toFixed(1)} / ${d.targetMin.toFixed(0)} min`;
  const fill = document.getElementById('tbfill');
  if(fill) fill.style.width = (frac*100).toFixed(2) + '%';
  const eta = document.getElementById('tbeta');
  if(eta){
    // at the rate it is actually managing, not the rate it was asked for
    const left = d.xRealtime > 0 ? (d.targetMin - d.simMin)/d.xRealtime : Infinity;
    eta.textContent = `${d.xRealtime.toFixed(2)}x · ${d.meanHz.toFixed(2)} Hz` +
      (fmtLeft(left) ? ' · ' + fmtLeft(left) : '');
  }
};
// Training panel tabs.
// The three panes are consulted at different moments: EVENTS while watching, SETUP when asking what this run actually was, FILES when collecting what it wrote.
// Each scrolls on its own, so switching does not carry one pane's position into another.
{
  const tabs = [...document.querySelectorAll('#ttabs .ttab')];
  const show = name => {
    for(const t of tabs){
      const on = t.dataset.pane === name;
      t.classList.toggle('on', on);
      if(on) t.classList.remove('unread');    // reading it is what clears it
      const el = document.getElementById(t.dataset.pane);
      if(el) el.hidden = !on;
    }
  };
  for(const t of tabs) t.onclick = () => show(t.dataset.pane);
}
// Draws attention to a tab that has something new while another is open, so a checkpoint landing behind the SETUP tab is not silently missed.
function markTab(name){
  const t = document.querySelector(`#ttabs .ttab[data-pane="${name}"]`);
  if(t && !t.classList.contains('on')) t.classList.add('unread');
}
globalThis.__npTrainTab = markTab;

// The run's own pause.
// Deliberately not the viewer's PAUSE, which stops the interactive simulation this page owns: during a run there is no such simulation, and stopping the view of a run is not the same act as stopping the run. train.js reports the state it settled in, so the label follows the run rather than the click.
{
  const btn = document.getElementById('tbpause');
  if(btn) btn.onclick = () => {
    if(!globalThis.__npTrainPause) return;
    runPaused = !!globalThis.__npTrainPause();
    btn.innerHTML = icon(runPaused ? 'play' : 'pause');
    btn.setAttribute('aria-label', runPaused ? 'resume run' : 'pause run');
    btn.classList.toggle('on', runPaused);
    const st = document.getElementById('tbstate');
    if(st){
      st.textContent = runPaused ? 'PAUSED' : 'TRAINING';
      st.classList.toggle('paused', runPaused);
      if(runPaused) st.classList.remove('warmup');
    }
    status(runPaused
      ? 'training paused: the network and its learned weights are kept'
      : 'training resumed');
  };
  const sv = document.getElementById('tbsave');
  if(sv) sv.onclick = () => {
    if(!globalThis.__npTrainSave) return;
    const r = globalThis.__npTrainSave();
    if(r === 'busy'){ status('a brain file is already being written', true); return; }
    // The write is not instant at full density, and the run reports it in the log when it lands, so this only says the request was taken.
    sv.disabled = true;
    status('brain file requested; it appears in the run log when written');
    setTimeout(() => { sv.disabled = false; }, 4000);
  };
}
// A run that has failed must not look like one that is training.
globalThis.__npTrainFailed = msg => {
  runFailed = true;
  const st = document.getElementById('tbstate');
  if(st){ st.textContent = 'ENGINE FAILED'; st.classList.add('failed');
    st.classList.remove('warmup', 'paused', 'done');
    st.title = msg || 'the engine stopped'; }
  const cl = document.getElementById('tbclock');
  if(cl) cl.textContent = 'stopped';
  status(msg || 'the engine stopped', true);
};
// And neither must one whose clock has stopped moving for any other reason.
// The dot pulses on a CSS animation, so it would keep pulsing over a frozen run.
let lastSim = -1, lastSimAt = 0, runFailed = false;
function watchStall(simMin){
  const now = Date.now();
  if(simMin !== lastSim){ lastSim = simMin; lastSimAt = now; return false; }
  return lastSimAt > 0 && now - lastSimAt > 45000;
}
function clearTrainHead(done){
  runPaused = false;
  // Both go out when the run ends: its engine is gone, so there is nothing left to pause and nothing left to read weights from.
  // SAVE BRAIN in the top bar comes back at the same moment and is the way to keep the finished network.
  // A finished run keeps its headline in the bar, but not its controls: a disabled pause beside the live pause reads as two pause buttons.
  const btn = document.getElementById('tbpause');
  if(btn){ btn.disabled = !!done; btn.hidden = !!done; btn.classList.remove('on');
    btn.innerHTML = icon('pause'); btn.setAttribute('aria-label', 'pause run');
    btn.title = 'pause the training run. The network, the learned weights and the run clock are kept.'; }
  const sv = document.getElementById('tbsave');
  if(sv){ sv.disabled = !!done; sv.hidden = !!done; }
  const st = document.getElementById('tbstate');
  if(st){ st.textContent = done ? 'RUN FINISHED' : 'TRAINING';
    st.classList.toggle('done', !!done);
    st.classList.remove('warmup'); st.classList.remove('paused'); }
  if(!done){
    const bar = document.getElementById('viewerbar');
    if(bar) bar.classList.remove('training');
  }
}
function setRunLock(on){
  for(const [id, why] of RUN_LOCK){
    const el = document.getElementById(id);
    if(!el) continue;
    el.disabled = on;
    el.classList.toggle('runlocked', on);
    if(on){ el.dataset.wasTitle = el.title || ''; el.title = why; }
    else if(el.dataset.wasTitle !== undefined){ el.title = el.dataset.wasTitle; }
  }
  const lab = document.getElementById('reslabel');
  if(lab) lab.classList.toggle('runlocked', on);
  const sl = document.getElementById('speedlabel');
  if(sl) sl.classList.toggle('runlocked', on);
  if(on){
    // The status line keeps whatever was last written to it, and anything about the previous run is wrong the moment a new one starts.
    status('training run started on the graph on screen');
    // Reveal the run's block now, not when the first status arrives.
    // That status comes after the wiring, which at full density takes twenty minutes, and until then the bar would show nothing at all: a run being set up would look exactly like no run.
    const bar0 = document.getElementById('viewerbar');
    if(bar0) bar0.classList.add('training');
    for(const id of ['tbpause', 'tbsave']){ const b = document.getElementById(id); if(b) b.hidden = false; }
    // Same for the headline: the first status only arrives once the run is past its wiring, so without this the bar reads FINISHED over a starting run.
    const st = document.getElementById('tbstate');
    if(st){ st.textContent = 'STARTING'; st.classList.remove('done', 'paused', 'warmup');
      st.title = 'the run is computing its own copy of the network; the clock ' +
        'starts when that finishes'; }
    const cl = document.getElementById('tbclock');
    if(cl) cl.textContent = 'wiring';
    const et = document.getElementById('tbeta');
    if(et) et.textContent = '';
    const fl = document.getElementById('tbfill');
    if(fl) fl.style.width = '0%';
    for(const id of ['tbpause', 'tbsave']){
      const b = document.getElementById(id);
      if(b){ b.disabled = false; b.classList.remove('on'); }
    }
    const pb = document.getElementById('tbpause');
    if(pb){ pb.innerHTML = icon('pause'); pb.setAttribute('aria-label', 'pause run'); }
    runPaused = false;
  }
  if(!on) vmUnsupported = '';
  applyVmLock();
  setPropsLocked(on, 'a training run is going');
  showProps(propsNode);                  // re-render so the state is visible
}
// The filter is out only when the engine says it cannot stream potentials.
function applyVmLock(){
  const off = !!vmUnsupported;
  for(const id of ['vmfloor', 'vmlabel']){
    const el = document.getElementById(id);
    if(!el) continue;
    el.classList.toggle('runlocked', off);
    if('disabled' in el) el.disabled = off;
    if(off) el.title = vmUnsupported;
  }
  if(off){
    vmEl.value = -100;
    viewer.setPotentialFloor(-100);
    vmValEl.textContent = 'all';
  }
}

// ---- brain files ----
function saveBrain(){
  if(!worker || !doc.curNet || !doc.curNet.synCount){
    status('no running network to save', true); return;
  }
  if(!offers(worker, 'getWeights', 'weight readback, so no brain can be saved from it')) return;
  new Promise(res => { pendingWeights = res; worker.postMessage({ cmd:'getWeights' }); })
    .then(({ w, t }) => {
      const graph = { ...editor.toJSON(), resolution:+resEl.value };
      // pair identity is 20 bytes a synapse and a single ArrayBuffer caps near 2 GB, so past that size the save drops to weights-only, which reloads onto the wiring of this same graph but not onto an edited one.
      // A network whose buffers moved into the sim is past that size too, and the weights-only file needs only what the worker just returned.
      const big = pairFileBytes(doc.curNet.synCount) > MAX_BUFFER || doc.curNet.post.byteLength === 0;
      const buf = big ? buildWeightsFile(graph, doc.curNet, w, t, null)
                      : buildBrainFile(graph, doc.curNet, w, t, null);
      const ts = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
      const name = `brain-${ts}.npb`;
      const note = big
        ? ' (weights only: too large for pair identity, reloads onto this graph)' : '';
      // Into the project folder next to the runs that produced it, rather than into Downloads.
      // A brain in Downloads is separated from the run it came from by every unrelated file the browser has saved since, and LOAD BRAIN then means finding it there by its timestamp.
      store().then(async st => {
        await st.write(P.brainPath(name), buf);
        status(`brain saved to ${st.label} as ${P.brainPath(name)}: ` +
          `${doc.curNet.synCount.toLocaleString()} synapses at sim-ms ${t}` + note);
      }).catch(e => {
        // The save is the point, so a store that will not take it still hands the file over rather than losing it.
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([buf], { type:'application/octet-stream' }));
        a.download = name; a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 5000);
        status(`brain downloaded (could not write to the project folder: ${e.message})` + note);
      });
    });
}
function loadBrain(file){
  file.arrayBuffer().then(buf => loadBrainBuffer(buf, { name:file.name }))
    .catch(e => status('brain load failed: ' + e.message, true));
}
// A brain is a document that carries a graph.
// Onto a scene with the same graph structure (the next checkpoint of the same run, say) it attaches to the active checkpoint, so brains load back to back and different checkpoints can hold different brains.
// A brain with another graph opens as a scene of its own in a new tab, since replacing the graph on screen would let the autosave write it over the open scene's file. opts.replace does replace it for the attach page, which is a view of a run and not a project being edited.
async function loadBrainBuffer(buf, opts = {}){
  try {
    const brain = parseBrainFile(buf);
    const missing = scenePluginsMissing(brain.graph);
    if(missing) throw new Error('the brain\'s graph: ' + missing);
    const sig = g => JSON.stringify((g.nodes || []).map(n => [n.id, n.type]));
    const same = sig(brain.graph) === sig(editor.toJSON());
    const put = () => {
      setSceneSettings({ resolution:brain.graph.resolution || 100, slice:brain.graph.slice || null, hues:brain.graph.hues || null });
      editor.load(brain.graph); assignDefaultSlots(); showProps(null);
    };
    if(!same && opts.replace){
      if(!makeShown()) return;
      exitLive(); doc.brainByNode.clear(); put();
    } else if(!same){
      const st = await store();
      const base = String(opts.name || 'brain').replace(/^.*[\\/]/, '').replace(/\.npb$/i, '').replace(/[^A-Za-z0-9 _.-]/g, ' ').trim().slice(0, 50) || 'brain';
      let name = base, k = 1;
      while(await st.read(P.scenePath(name + '.json')) || tabs.byFile(P.scenePath(name + '.json'))) name = base + ' ' + (++k);
      const tab = tabs.add(P.scenePath(name + '.json'), name, tabs.list.indexOf(tabs.cur) + 1);
      if(!(await switchTab(tab, { make: put }))){ tabs.remove(tab); renderTabs(); return; }
      await st.write(tab.file, sceneJSON());
      lastSceneJson = sceneJSON();
    }
    // the weights go onto the next computation of this scene; when the viewer cannot be taken now (locked weights in the scene it holds) they wait for the viewer key
    tabs.cur.doc.pendingBrain = brain;
    if(!makeShown()) return;
    exitLive();
    recompute();
    if(!same && !opts.replace) status('opened the brain as scene ' + tabs.cur.label + ': its graph, and its weights once the wiring finishes');
  } catch(e){ status('brain load failed: ' + e.message, true); }
}
// every brain in the project: the ones saved by hand, and each run's checkpoints
async function listProjectBrains(){
  const st = await store();
  const found = [];
  const take = async (dir, label) => {
    let list = []; try { list = await st.list(dir); } catch(e){}
    for(const f of list) if(!f.dir && f.name.endsWith('.npb'))
      found.push({ path:dir + '/' + f.name, name:label + f.name, mtime:f.mtime || 0, size:f.size || 0 });
  };
  await take(P.brainsDir(), '');
  let runs = []; try { runs = await st.list(P.runsDir()); } catch(e){}
  for(const d of runs) if(d.dir) await take(P.runBrainsDir(d.name), d.name + ' / ');
  return found.sort((a, b) => b.mtime - a.mtime);
}
async function openBrainPath(b){
  try {
    status('reading ' + b.path + '...');
    const st = await store();
    const bytes = await st.read(b.path);
    if(!bytes){ status('no such brain: ' + b.path, true); return; }
    const buf = bytes instanceof ArrayBuffer ? bytes : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    await loadBrainBuffer(buf, { name:b.path.split('/').pop() });
  } catch(e){ status('brain load failed: ' + e.message, true); }
}
// ---- keyboard shortcuts ----
// the hint strip reads the live bindings, so a rebound key is never contradicted by the text above the graph
function renderHint(){
  const el = document.getElementById('graphHint');
  if(!el) return;
  const k = a => binding(a);
  el.innerHTML = [
    k('addNode') + ' add', 'dblclick props/gizmo', 'drag select', 'mmb pan',
    k('copy').replace('ctrl+','') + '/' + k('paste').replace('ctrl+','') + ' copy',
    'alt-drag clone', k('undo').replace('ctrl+','') + ' undo',
    k('bypass') + ' bypass', '1-9 bind/view', 'shake unplug', 'drop insert',
    k('deleteNode') + ' delete', k('frameAll') + ' frame',
    k('findNode') + ' find',
    k('addDot') + ' reroute',
  ].join(' &nbsp; ');
}
renderHint();

// ---- documentation overlay ----
// Imported on first open, not at startup: the docs module is prose and the renderer is only needed by whoever asks for it.
// Opening it does not touch the graph, the running simulation or the wired synapses, which is the whole point; a link away to docs.html would throw all three away and coming back would rewire from storage.
const docsDlg = document.getElementById('docsDlg');
let docsMounted = false;
let docsShow = null;
// nodeType jumps to that node's entry in the reference, which is what a right click on a parameter label asks for
async function openDocs(nodeType){
  if(!docsMounted){
    const { mountDocs } = await import('./docsview.js');
    docsShow = mountDocs({
      nav: document.getElementById('docsNav'),
      content: document.getElementById('docsContent'),
      page: document.getElementById('docsPage'),
    }).show;
    docsMounted = true;
  }
  docsDlg.style.display = 'flex';
  if(nodeType && docsShow){
    docsShow('nodes');
    // after the page is in the DOM
    setTimeout(() => {
      const t = document.getElementById('n-' + nodeType);
      if(t) t.scrollIntoView();
    }, 0);
  }
}
setPropsDocs(openDocs);
document.getElementById('docsbtn').onclick = () => openDocs();
document.getElementById('docsClose').onclick = () => { docsDlg.style.display = 'none'; };
docsDlg.onclick = e => { if(e.target === docsDlg) docsDlg.style.display = 'none'; };
// Escape closes it, the way it closes everything else that opens over the graph.
// Captured before the editor sees the key so it does not also cancel a drag or a menu underneath.
window.addEventListener('keydown', e => {
  if(e.key === 'Escape' && docsDlg.style.display === 'flex'){
    docsDlg.style.display = 'none';
    e.stopPropagation();
  }
}, true);

const keysDlg = document.getElementById('keysDlg');
let armed = null;                          // the row waiting for a keypress
function renderKeys(){
  const list = document.getElementById('keysList');
  list.innerHTML = '';
  for(const [g, gl] of GROUPS){
    const h = document.createElement('div');
    h.className = 'kgroup'; h.textContent = gl.toUpperCase();
    list.appendChild(h);
    for(const k in ACTIONS){
      const a = ACTIONS[k];
      if(a.group !== g) continue;
      const row = document.createElement('div'); row.className = 'krow';
      const lab = document.createElement('span'); lab.textContent = a.label;
      const b = document.createElement('button');
      b.className = 'kbind' + (a.fixed ? ' fixed' : '') + (isCustom(k) ? ' custom' : '');
      b.textContent = binding(k) + (a.alt ? '  /  ' + a.alt : '');
      if(!a.fixed) b.onclick = () => {
        if(armed) renderKeys();
        armed = k; b.classList.add('armed'); b.textContent = 'press a key';
      };
      row.appendChild(lab); row.appendChild(b);
      list.appendChild(row);
    }
  }
}
document.getElementById('keysbtn').onclick = () => { armed = null; renderKeys();
  keysDlg.style.display = 'flex'; };
document.getElementById('keysClose').onclick = () => { armed = null;
  keysDlg.style.display = 'none'; };
document.getElementById('keysReset').onclick = () => { resetBindings(); armed = null;
  renderKeys(); renderHint(); };
keysDlg.onclick = e => { if(e.target === keysDlg){ armed = null; keysDlg.style.display = 'none'; } };
// capture phase so a key being bound never also triggers the action it is being bound away from
window.addEventListener('keydown', e => {
  if(!armed) return;
  e.preventDefault(); e.stopPropagation();
  if(e.key === 'Escape'){ armed = null; renderKeys(); return; }
  const b = eventToBinding(e);
  if(b){ setBinding(armed, b); armed = null; renderKeys(); renderHint(); }
}, true);

// ---- training launcher ----
// Saved brains are listed so a paused run can be picked up; the curriculum position rides along in the file, so a resume continues the lesson stream instead of restarting it.
const trainDlg = document.getElementById('trainDlg');
async function listSavedBrains(){
  const sel = document.getElementById('trBrain');
  sel.innerHTML = '<option value="">start fresh</option>';
  try {
    const st = await store();
    const found = [];
    for(const d of await st.list(P.runsDir())){
      if(!d.dir) continue;
      for(const f of await st.list(P.runBrainsDir(d.name))){
        if(f.dir || !f.name.endsWith('.npb')) continue;
        found.push({ path:`${d.name}/${f.name}`, mod:f.mtime,
          mb:(f.size/1048576).toFixed(0) });
      }
    }
    found.sort((a, b) => b.mod - a.mod);
    if(found.length){
      const latest = document.createElement('option');
      latest.value = 'latest';
      latest.textContent = `latest (${found[0].path})`;
      sel.appendChild(latest);
    }
    for(const f of found){
      const o = document.createElement('option');
      o.value = f.path;
      o.textContent = `${f.path} (${f.mb} MB)`;
      sel.appendChild(o);
    }
    const where = await whereLabel();
    document.getElementById('trNote').textContent = (found.length
      ? `${found.length} saved brain${found.length === 1 ? '' : 's'} found. A brain only loads onto a network with the same synapse count, so one saved at a different footprint is refused rather than restored onto different tissue.`
      : 'No saved brains yet; the first appears once a run reaches its brain-file cadence.')
      + ' Files go to ' + where + '.';
  } catch(e){
    document.getElementById('trNote').textContent = 'could not read storage: ' + e.message;
  }
}
// Training is started from a checkpoint node, not from a button in the top bar.
// A run is that checkpoint left going with plasticity on, so its settings belong on the node, travel with the saved graph, and sit next to the plasticity parameters they interact with.
// All this dialog still asks is the one thing the graph cannot carry: which saved brain to resume from.
let trainNode = null, trainingRunning = false;

// Is a host listening?
// A WebSocket that opens is the only answer that means what the run needs it to mean, so this asks the way the run will connect rather than trusting the dev server's view of a port.
const DEFAULT_HOST = 'ws://localhost:8801';
const hostPort = url => +((/:(\d+)\/?$/.exec(String(url || '')) || [])[1]) || 8801;
function hostReachable(ms = 2500, url = DEFAULT_HOST){
  return new Promise(res => {
    let done = false;
    const finish = v => { if(!done){ done = true; res(v); } };
    let ws;
    try { ws = new WebSocket(url); }
    catch(e){ return finish(false); }
    const to = setTimeout(() => { try { ws.close(); } catch(e){} finish(false); }, ms);
    ws.onopen = () => { clearTimeout(to); try { ws.close(); } catch(e){} finish(true); };
    ws.onerror = () => { clearTimeout(to); finish(false); };
  });
}

// Served from a checkout with serve.py in front of it, or from the hosted deploy?
// Only the first can be asked to start a process, and the difference decides what the CUDA option can honestly promise.
const LOCAL_DEV = ['localhost', '127.0.0.1', '[::1]', ''].includes(location.hostname);

// What the CUDA engine needs, said once so the dialog and the failure path cannot drift apart.
// It is a real install, not a download: an NVIDIA card, Node, and engine.exe compiled by nvcc for that card.
const CUDA_NEEDS = 'The CUDA engine runs on your own machine, not on the ' +
  'server: it needs the repository, Node, an NVIDIA card and cuda/engine.exe ' +
  'built for it. Without one, the gpu (WebGPU) engine trains in the browser ' +
  'and needs nothing installed.';

// Ask the dev server to start one.
// It only ever runs host/host.mjs and only for a request from that machine; serve.py holds the rest of the command.
// Returns a line to show, or null when there is a host either way.
async function ensureHost(say, url = DEFAULT_HOST){
  if(await hostReachable(2500, url)) return null;
  // A hosted page has no dev server behind it, so there is nothing to ask. ws://localhost still points at the reader's own machine, so someone running a host of their own is served by the check above rather than being told to install anything.
  if(!LOCAL_DEV)
    return 'no engine host is running on your machine. ' + CUDA_NEEDS;
  say('engine host not running, starting one');
  let r;
  try {
    r = await (await fetch('/host/start', { method:'POST',
      headers:{ 'Content-Type':'application/json' },
      body: JSON.stringify({ cuda:true, port:hostPort(url) }) })).json();
  } catch(e){
    return 'could not reach the dev server to start an engine host (' + e.message +
      '). Start one yourself: node host/host.mjs ' + hostPort(url) + ' --cuda';
  }
  if(!r || !r.ok)
    return 'could not start an engine host: ' + ((r && r.error) || 'unknown') +
      '. Start one yourself: node host/host.mjs ' + hostPort(url) + ' --cuda';
  // serve.py reports the port open; confirm the way the run will connect
  if(!(await hostReachable(4000, url)))
    return 'an engine host started but is not accepting connections yet. ' +
      'Try START again in a moment';
  say(r.started ? 'engine host started with the CUDA engine'
    : 'engine host was already listening');
  return null;
}
async function openTrainDialog(node){
  trainNode = node;
  const p = node.params;
  const ei = p.trEngine | 0;
  const eng = ei === 1 ? 'cpu (reference)' : ei === 2 ? 'cuda (remote host)' : 'gpu';
  // what the run will vary comes from the checkpoint's sweep tab, named here so the dialog says what the node says
  const variants = gatherVariants(node, editor.nodes);
  document.getElementById('trSummary').innerHTML =
    `<b>${editor.nodes.length} nodes</b> on screen, plasticity ${(p.plast|0) === 0 ? '<b>off</b>' : 'on'} (the checkpoint's switch)<br>` +
    `${(+p.trHours).toFixed(2)} sim-hours on the ${eng} engine<br>` +
    (ei === 2 ? `engine host: <span id="trHostState">checking ${p.trHost || DEFAULT_HOST}…</span><br>` : '') +
    `metrics every ${+p.trCkptMin} sim-min, brain file every ${+p.trBrainMin}, ` +
    `keeping ${p.trKeep | 0}<br>` +
    (((p.trReps | 0) > 1 || (p.trSweep | 0) === 1)
      ? `<b>sweep:</b> ${Math.max(1, p.trReps | 0)} replicate(s) × ` +
        `${(p.trSweep | 0) === 1 ? 'paired and scrambled' : 'the graph as configured'}, ` +
        `run one after another; results in the SWEEP tab<br>` : '') +
    (variants.length
      ? `<b>${variants.length} variant(s)</b>` +
        ' from the sweep tab' +
        `: ${variants.map(v => typeof v === 'object' ? v.name : v).join(' · ')}<br>` : '') +
    `<span class="dim">change any of these on the checkpoint node</span>`;
  // The cuda option depends on a process this page cannot start itself, so the dialog reports reachability up front rather than letting START discover it.
  // The dev server can start one on request, so a host that is down is a line of prose here and not an errand: START brings it up.
  if(ei === 2){
    const el = () => document.getElementById('trHostState');
    hostReachable(2500, p.trHost || DEFAULT_HOST).then(up => {
      if(!el()) return;
      el().innerHTML = up
        ? 'reachable at ' + (p.trHost || DEFAULT_HOST) + '. What computes there is decided ' +
          'at its launch: <code>--cuda</code> for the CUDA engine, without it ' +
          'the reference engine runs natively'
        : LOCAL_DEV
          ? 'not running. START will start one with <code>--cuda</code>, or run ' +
            '<code>node host/host.mjs 8801 --cuda</code> yourself to choose ' +
            'otherwise'
          : '<b>not running on your machine.</b> ' + CUDA_NEEDS;
    });
  }
  await listSavedBrains();
  trainDlg.style.display = 'flex';
}
document.getElementById('trCancel').onclick = () => { trainDlg.style.display = 'none'; };
trainDlg.onclick = e => { if(e.target === trainDlg) trainDlg.style.display = 'none'; };
document.getElementById('trGo').onclick = async () => {
  if(!trainNode) return;
  // Proxy downscaling preserves rates and spectra, not correlations (van Albada, Helias and Diesmann 2015), and spike-timing plasticity is a function of correlations, so a run below 100% learns on statistics the full network does not have.
  // The trainer refuses it; the address-bar harness can override with allowproxy=1, which marks the run as a non-biological intervention.
  if(+resEl.value !== 100){
    status(`training needs 100% resolution (the slider is at ${resEl.value}%): ` +
      'proxy compensation does not preserve the correlations plasticity depends on', true);
    return;
  }
  const p = trainNode.params;
  // A cuda run needs a host.
  // Starting one is the dev server's job and takes a couple of seconds, so it happens here rather than leaving the run to fail on a refused socket after a seven-minute wiring.
  if((p.trEngine | 0) === 2){
    const go = document.getElementById('trGo');
    const el = () => document.getElementById('trHostState');
    const say = s => { if(el()) el().textContent = s; };
    go.disabled = true;
    const problem = await ensureHost(say, p.trHost || DEFAULT_HOST);
    go.disabled = false;
    if(problem){ if(el()) el().innerHTML = '<b>' + problem + '</b>'; return; }
  }
  const q = new URLSearchParams({
    usegraph:'2',
    engine:(p.trEngine | 0) === 1 ? 'cpu' : (p.trEngine | 0) === 2 ? 'remote' : 'gpu',
    // A remote run is host-pumped, always: the host owns the tick loop, the stimulus clock and the checkpoint cadence, and this page is a viewer.
    // Without it the run is paced by this tab's event loop, and a backgrounded tab throttles to a crawl: measured 2026-08-31, six sim-seconds in six wall-minutes the moment focus went elsewhere, which no overnight run survives.
    ...((p.trEngine | 0) === 2 ? { pump:'host', host:p.trHost || DEFAULT_HOST } : {}),
    hours:String(+p.trHours),
    ckptmin:String(+p.trCkptMin),
    brainmin:String(+p.trBrainMin),
    keep:String(p.trKeep | 0),
  });
  const brain = document.getElementById('trBrain').value;
  if(brain) q.set('brain', brain);
  globalThis.__npTrainGraph = editor.toJSON();
  trainDlg.style.display = 'none';
  const variants = gatherVariants(trainNode, editor.nodes);
  if((p.trReps | 0) > 1 || (p.trSweep | 0) === 1 || variants.length) runSweep(q, p, variants);
  else startTrainingHere(q);
};

// A replicate sweep: the same graph run several times per condition, one run after another, each with its own curriculum seed and engine noise seed.
// One run is one draw of a chaotic system and its trials share that draw (measured 2026-09-02: a same-seed replicate went from 0.159 to 0.023 and a scrambled control scored 0.17), so the sweep tabulates per-run results and the spread across them is the evidence.
let sweepRows = [];
async function runSweep(q, p, variants = []){
  const reps = Math.max(1, p.trReps | 0);
  const conds = (p.trSweep | 0) === 1 ? ['paired', 'scrambled'] : ['as configured'];
  const vars = variants.length ? variants : [''];
  const base = editor.toJSON();
  const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
  sweepRows = [];
  drawSweep(stamp, conds, reps, vars);
  markTab('sweep');
  let k = 0;
  const total = vars.length*conds.length*reps;
  for(const variant of vars) for(const cond of conds) for(let r=0;r<reps;r++){
    k++;
    const g = JSON.parse(JSON.stringify(base));
    // the variant's edits, on the copy: a text line by selector, or a sweep node's rows by node id
    const applied = applyVariant(variant, g.nodes, msg => status(msg, true));
    for(const n of g.nodes) if(n.type === 'curriculum'){
      if(cond !== 'as configured') n.params.scramble = cond === 'scrambled' ? 1 : 0;
      n.params.seed = (n.params.seed | 0) + r;
    }
    const qq = new URLSearchParams(q);
    qq.set('seed', String(7919*(r + 1) + conds.indexOf(cond)));
    globalThis.__npTrainGraph = g;
    const vname = variant && typeof variant === 'object' ? variant.name : variant;
    status(`sweep: ${vname ? vname + ', ' : ''}${cond}, replicate ${r + 1} of ${reps} (run ${k} of ${total})`);
    const res = await runOnce(qq);
    const row = { variant:vname, applied, cond, rep:r + 1, tag:res.tag, failed:res.failed || null, pops:{} };
    const pick = (o, side, k) => o && o[side] && o[side][k] !== undefined && o[side][k] !== null ? o[side][k] : null;
    const ro = res.last && res.last.readout;
    if(ro) for(const [pop, v] of Object.entries(ro.pops || {})){
      const cp = v.crossDecodePooled, cd = v.crossDecode || {};
      row.pops[pop] = { decode:v.decode, selectivity:v.selectivity, activeFrac:v.activeFrac,
        ab: cp && cp.ab ? cp.ab : (cd.ab ? { acc:cd.ab.acc, n:cd.ab.n } : null),
        ba: cp && cp.ba ? cp.ba : (cd.ba ? { acc:cd.ba.acc, n:cd.ba.n } : null),
        // the denominator: probes that leave this population silent are not scored, so the accuracies beside this are of the rest
        silent: v.silentFrac || null,
        // learning where the input does not reach: the evoked transfer's change from calibration, pooled over the run's end, and the item-specific weight change from the last weight readback
        xab: pick(v.xferPooled, 'ab', 'delta') ?? pick(v.xfer, 'ab', 'delta'),
        xba: pick(v.xferPooled, 'ba', 'delta') ?? pick(v.xfer, 'ba', 'delta'),
        cab: pick(res.coupled && res.coupled.itemCoupling && res.coupled.itemCoupling[pop], 'ab', 'd'),
        cba: pick(res.coupled && res.coupled.itemCoupling && res.coupled.itemCoupling[pop], 'ba', 'd') };
    }
    sweepRows.push(row);
    drawSweep(stamp, conds, reps, vars);
    try { const ST = await store(); await ST.write('runs/sweep-' + stamp + '.json',
      JSON.stringify({ format:'np-sweep-2', stamp,
        variants:vars.map(x => x && typeof x === 'object' ? x.name : x),
        conditions:conds, replicates:reps,
        hours:+p.trHours, engine:q.get('engine'), rows:sweepRows }, null, 1)); }
    catch(e){ status('sweep file not written: ' + e.message, true); }
    if(res.failed) break;
  }
  status(`sweep finished: ${sweepRows.length} run(s); table in the SWEEP tab, file runs/sweep-${stamp}.json`);
  if(viewer._ctxLost) restoreView();
}
// One run to completion.
// The trainer signals the end through
// __npOnTrainDone and a failure through __npTrainFailed; both are page
// globals the run driver calls, so a sweep waits on them.
function runOnce(q){
  return new Promise(resolve => {
    const prevFail = globalThis.__npTrainFailed;
    let settled = false;
    const finish = failed => {
      if(settled) return;
      settled = true;
      globalThis.__npTrainFailed = prevFail;
      resolve({ tag:globalThis.__npRunTag || null, last:lastScored, coupled:lastCoupled, failed });
    };
    globalThis.__npTrainFailed = msg => { if(prevFail) prevFail(msg); finish(msg || 'failed'); };
    sweepDone = () => finish(null);
    lastMetrics = null; lastScored = null; lastCoupled = null;
    startTrainingHere(q).catch(e => finish(e.message));
  });
}
let sweepDone = null;
const fmtCell = d => d ? (d.k !== undefined ? `${d.k}/${d.n} = ${(+d.acc).toFixed(3)}` : `${(+d.acc).toFixed(3)} (n ${d.n})`) : '·';
function drawSweep(stamp, conds, reps, vars = ['']){
  const el = document.getElementById('sweep');
  if(!el) return;
  const pops = [...new Set(sweepRows.flatMap(r => Object.keys(r.pops)))];
  const hasVar = vars.some(Boolean);
  vars = vars.map(x => x && typeof x === 'object' ? x.name : x);
  let h = `<div class="dim">sweep ${stamp}: ${hasVar ? vars.length + ' variant(s), ' : ''}` +
    `${conds.join(' and ')}, ${reps} replicate(s) each. ` +
    `Pooled cross-modal test at each run's end, correct/total (vision trained, sound tested; ` +
    `then the reverse); chance 1/items. Silent is the fraction of sound-alone probes that ` +
    `evoke nothing, which the decoders skip. The last four columns ask where the input does not ` +
    `reach: evoked is the response one sense alone draws in the other sense's cells for the same ` +
    `item against other items', as its change since calibration; coupling is the weight change ` +
    `onto the same item's cells in the other code against other items', which is zero when ` +
    `nothing is learned. Read the spread across rows, not one row.</div>`;
  h += '<table class="sweep"><tr>' + (hasVar ? '<th>variant</th>' : '') + '<th>condition</th><th>rep</th><th>run</th>';
  for(const p of pops) h += `<th>${p} decode</th><th>${p} active</th><th>${p} selectivity</th><th>${p} silent</th><th>${p} vis→snd</th><th>${p} snd→vis</th>` +
    `<th>${p} evoked vis→snd</th><th>${p} evoked snd→vis</th><th>${p} coupling vis→snd</th><th>${p} coupling snd→vis</th>`;
  h += '</tr>';
  for(const r of sweepRows){
    h += `<tr>${hasVar ? `<td>${r.variant || 'as is'}</td>` : ''}<td>${r.cond}</td><td>${r.rep}</td><td>${r.tag || '·'}${r.failed ? ' (failed)' : ''}</td>`;
    for(const p of pops){ const v = r.pops[p];
      const sil = v && v.silent && v.silent.aud !== null && v.silent.aud !== undefined
        ? (+v.silent.aud).toFixed(2) : '·';
      const num = x => x !== undefined && x !== null ? (+x).toFixed(2) : '·';
      const small = x => x !== undefined && x !== null ? (+x).toFixed(4) : '·';
      h += v ? `<td>${num(v.decode)}</td><td>${num(v.activeFrac)}</td><td>${num(v.selectivity)}</td><td>${sil}</td><td>${fmtCell(v.ab)}</td><td>${fmtCell(v.ba)}</td>` +
          `<td>${small(v.xab)}</td><td>${small(v.xba)}</td><td>${small(v.cab)}</td><td>${small(v.cba)}</td>`
        : '<td>·</td>'.repeat(10); }
    h += '</tr>';
  }
  const cols = (hasVar ? 4 : 3) + 10*pops.length;
  const done = sweepRows.length, total = vars.length*conds.length*reps;
  if(done < total) h += `<tr><td colspan="${cols}" class="dim">run ${done + 1} of ${total} in progress</td></tr>`;
  // Per variant and condition, the mean and spread over the runs in it, in every column rather than the two cross-modal ones: a single run is one draw of a chaotic system, so the summary row is the result and the rows above it are its evidence.
  // With two conditions, an exact Mann-Whitney test between them within each variant, the run being the unit; a trial-level p inside one run says nothing about the configuration.
  const runsOf = (vname, cond) => sweepRows.filter(r =>
    (!hasVar || (r.variant || '') === (vname || '')) && r.cond === cond);
  const pull = (rows, p, pick) => rows.filter(r => r.pops[p]).map(r => pick(r.pops[p]))
    .map(Number).filter(Number.isFinite);
  const MEASURES = [x => x.decode, x => x.activeFrac, x => x.selectivity,
    x => x.silent && x.silent.aud, x => x.ab && x.ab.acc, x => x.ba && x.ba.acc,
    x => x.xab, x => x.xba, x => x.cab, x => x.cba];
  const msd = a => { if(!a.length) return '·'; const mu = a.reduce((s, x) => s + x, 0)/a.length;
    const sd = a.length > 1 ? Math.sqrt(a.reduce((s, x) => s + (x - mu)**2, 0)/(a.length - 1)) : 0;
    // the weight and evoked measures live near zero, where three places would print every value as 0.000
    const f = x => x !== 0 && Math.abs(x) < 0.01 ? x.toExponential(2) : x.toFixed(3);
    return a.length > 1 ? `${f(mu)} ± ${f(sd)} (n ${a.length})` : `${f(mu)} (n 1)`; };
  for(const v of (hasVar ? vars : [''])) for(const cond of conds){
    const rows = runsOf(v, cond);
    if(!rows.length) continue;
    h += `<tr class="sum">${hasVar ? `<td>${v || 'as is'}</td>` : ''}<td>${cond}</td><td>mean ± sd</td><td>${rows.length} run(s)</td>`;
    for(const p of pops) for(const m of MEASURES) h += `<td>${msd(pull(rows, p, m))}</td>`;
    h += '</tr>';
  }
  if(conds.length === 2) for(const v of (hasVar ? vars : [''])){
    const a = runsOf(v, conds[0]), b = runsOf(v, conds[1]);
    if(!a.length || !b.length) continue;
    h += `<tr class="sum">${hasVar ? `<td>${v || 'as is'}</td>` : ''}<td>${conds[0]} vs ${conds[1]}</td><td>Mann-Whitney p</td><td></td>`;
    for(const p of pops) for(const m of MEASURES) h += `<td>${mwp(pull(a, p, m), pull(b, p, m))}</td>`;
    h += '</tr>';
  }
  h += '</table>';
  el.innerHTML = h;
}
// Exact two-sided Mann-Whitney by enumeration of the rank-sum distribution, for the small run counts a sweep produces; ties share average ranks.
function mwp(a, b){
  const n1 = a.length, n2 = b.length;
  if(n1 < 2 || n2 < 2) return '·';
  const all = [...a.map(x => [x, 0]), ...b.map(x => [x, 1])].sort((p, q) => p[0] - q[0]);
  const rank = new Array(all.length);
  for(let i=0;i<all.length;){ let j = i; while(j + 1 < all.length && all[j + 1][0] === all[i][0]) j++;
    const r = (i + j)/2 + 1; for(let k=i;k<=j;k++) rank[k] = r; i = j + 1; }
  const obs = all.reduce((s, x, i) => s + (x[1] === 0 ? rank[i] : 0), 0);
  const N = all.length; let count = 0, total = 0;
  const mean = n1*(N + 1)/2;
  const rec = (i, picked, sum) => {
    if(picked === n1){ total++; if(Math.abs(sum - mean) >= Math.abs(obs - mean) - 1e-9) count++; return; }
    if(N - i < n1 - picked) return;
    rec(i + 1, picked + 1, sum + rank[i]); rec(i + 1, picked, sum);
  };
  if(N > 20) return '·';
  rec(0, 0, 0);
  return (count/total).toFixed(3);
}

// Training runs in this page rather than one of its own, so the graph, the properties panel and neuron inspection stay in the same tab.
// The run driver is the same module either way; handing it this page's viewer lets it compute and simulate while the building UI stays live around it, which is what makes it possible to click a neuron and see its inputs mid-run.
// One engine at a time: the interactive simulation stops first, since two engines on one card compete and make every rate reading untrustworthy.
async function startTrainingHere(q){
  // The run is always the graph on screen, never a named scenario built from source, so the graph being looked at is the network that is training.
  // A checkpoint trains what is upstream of it: loading a scenario is just something done before pressing start.
  stopSim();
  // An interactive computation can still be in flight, and both computations share one pending promise, so they resolve with the same network object.
  // Without this bump the interactive completion would start its own sim and transfer the synapse buffers, and the trainer's next line, net.w.slice() for the baseline copy, would die on a detached ArrayBuffer.
  // The generation check makes the stale completion bail instead.
  doc.computeGen++;
  doc.livePhase = false;
  const pane = document.getElementById('trainPane');
  pane.style.display = 'flex';
  document.getElementById('tsplit').style.display = 'block';
  applyPanels();
  document.getElementById('trainHint').textContent =
    'this page is driving the run';
  trainingRunning = true;
  setRunLock(true);
  globalThis.__npOnTrainDone = w => {
    trainingRunning = false;
    if(viewer._ctxLost) restoreView();
    trainEngine = null;
    setRunLock(false);
    // the headline stays, marked finished: a run that has ended is a fact about the network on screen, not something to clear away
    clearTrainHead(true);
    document.getElementById('trainHint').textContent = 'finished';
    // The run's engine is gone, and with it every way of looking at what it made: no pathway, no neuron query, a dead viewer until the network is rewired.
    // The trained weights come back with the signal; when they fit the network on screen, the page's own sim restarts on them, locked so an edit cannot silently discard them.
    // A network whose buffers moved into the trainer has to be rewired.
    // The network the run trained is the trainer's: the page's own computation was superseded when the run started, so doc.curNet can be stale or empty.
    const net = globalThis.__npNet || doc.curNet;
    const fits = w && net && net.synCount === w.length &&
      net.post && net.post.byteLength > 0 && !sweepDone;
    if(fits){
      net.w = w instanceof Float32Array ? w : Float32Array.from(w);
      doc.curNet = net;
      viewer.setNetwork(net); startSim(net);
      applyPopulationColors(net); applyPopulationBoxes(net);
      setupProbes(net); setupInputs(net, io);
      tagHueMap = sceneTagHues(net);
      setNetSize(net.count, net.synCount);
      if(viewer.selected >= 0) viewer.onSelect(viewer.selected);
      relightPathways();
      if(!doc.livePhase) liveEl.onclick();
      status('training finished: the trained weights are live and locked. Unlock to rewire; LOAD BRAIN brings back an earlier checkpoint');
    } else {
      status('training finished: edit the graph to rewire, or LOAD BRAIN to bring the weights back');
    }
    if(sweepDone){ const f = sweepDone; sweepDone = null; f(); }
  };
  globalThis.__npViewer = viewer;
  try { const { startRun } = await import('./train.js'); await startRun(q); }
  catch(e){
    document.getElementById('stat').textContent = 'training failed to start';
    document.getElementById('log').textContent +=
      'ERROR: ' + e.message + String.fromCharCode(10);
  }
}

// A brain is this checkpoint's weights, so saving and loading one is a thing the checkpoint does; both are actions on the node.
// The file input has nowhere else to live, so it stays in the page.
const brainFileEl = document.getElementById('brainfile');
brainFileEl.onchange = () => { if(brainFileEl.files[0]) loadBrain(brainFileEl.files[0]); };
function pickBrainFile(){ brainFileEl.value = ''; brainFileEl.click(); }

// ---- top bar ----
document.getElementById('pause').onclick = e => {
  paused = !paused;
  if(recording) recording.mark(paused ? 'pause' : 'resume');
  const b = e.currentTarget;
  b.innerHTML = icon(paused ? 'play' : 'pause');
  b.setAttribute('aria-label', paused ? 'run' : 'pause');
  b.title = paused ? 'run the simulation' : 'pause the simulation. The network is kept.';
};

// checkpoints get viewer digits 1..9 in creation order when a graph arrives without bindings (scenarios, legacy saves)
function assignDefaultSlots(){
  if(Object.keys(editor.viewSlots).length) return;
  const cps = editor.nodes.filter(n => n.type === 'checkpoint');
  cps.slice(0, 9).forEach((n, i) => { editor.viewSlots[i+1] = n.id; });
  if(!editor.activeView && cps[0]) editor.activeView = cps[0].id;
}

// A scenario's saved layout, when the site carries one for it.
// Applied after the computed layout, so a stale file (the scene has changed since) leaves the computed arrangement and says so.
async function applySavedLayout(name){
  const slug = layoutSlug(name);
  let data = null;
  try {
    const r = await fetch(new URL('./layouts/' + slug + '.json', import.meta.url), { cache:'no-cache' });
    if(r.ok) data = await r.json();
  } catch(e){ return; }
  if(!data || editor.scenarioName !== name) return;
  const res = applyLayoutRecord(editor, data);
  // nodes the saved arrangement added (a rebuilt gather) are named like the rest
  if(res.ok){ nameNodes(editor.nodes, editor.groups); assignDefaultSlots(); editor.frameAll(); editor.draw(); recompute(); }   // the graph is now the file's: its nodes, its wires
  else status('the saved layout for ' + name + ' no longer matches the scene (' + res.why + '); using the computed one', true);
}
// The scene menu: the arrangement on screen becomes the scenario's default.
// Through the dev server when this page is served by it (loopback only); otherwise as a download to drop into src/layouts.
async function saveScenarioLayout(){
  // The file is keyed on the scenario's build: every node the builder made carries its place in that order (marked when the scenario was opened, kept by the scene file), so the scenario is the one whose build has these nodes in these places, whatever was added, removed or reconnected since.
  // Settings on built nodes are the scenario's own and stay in code: a changed one is reported and not saved.
  const fresh = sc => { const ed = { nodes:[], nextId:1, groups:[], addNode(t, x, y){ const def = NODE_DEFS[t], params = {};
    def.params.forEach(p => params[p.k] = structuredClone(p.def));
    const n = { id:this.nextId++, type:t, x, y, params, inputs:new Array(def.inputs).fill(null) }; this.nodes.push(n); return n; } };
    sc.build(ed); return ed.nodes; };
  const here = builtTypes(editor.nodes);
  if(!here.length){ status('no scenario on screen: open one from the Tab menu, arrange it, then save', true); return; }
  const fits = b => here.length <= b.length && here.every((t, k) => t === undefined || t === b[k].type);
  let sc = editor.scenarioName ? ALL_SCENARIOS.find(x => x.name === editor.scenarioName) : null;
  let build = sc ? fresh(sc) : null;
  if(sc && !fits(build)){ status('this graph is not the ' + sc.name + ' scenario: its built nodes do not match that build', true); return; }
  if(!sc){
    const hits = ALL_SCENARIOS.map(x => [x, fresh(x)]).filter(([, b]) => fits(b));
    if(hits.length !== 1){ status('no scenario on screen: open one from the Tab menu, arrange it, then save', true); return; }
    [sc, build] = hits[0];
  }
  const name = sc.name;
  const changed = editor.nodes.filter(n => isBuilt(n) && build[n.built] && NODE_DEFS[n.type].params.some(p =>
    p.k !== NODE_DEFS[n.type].portsParam && JSON.stringify(n.params[p.k]) !== JSON.stringify(build[n.built].params[p.k])))
    .map(n => n.name || n.type);
  const slug = layoutSlug(name), data = layoutRecord(editor, name, build.map(n => n.type)), json = JSON.stringify(data, null, 1);
  const note = changed.length ? ' (settings changed on ' + changed.join(', ') + ' stay in the scenario\'s code, not the layout)' : '';
  try {
    const r = await fetch('/layout?name=' + encodeURIComponent(slug), { method:'POST', headers:{ 'Content-Type':'application/json' }, body:json });
    const reply = r.ok ? await r.json() : null;
    if(reply && reply.ok){ status('saved the layout for ' + name + ' to ' + reply.path + ': it is the scenario default now; commit it' + note, changed.length > 0); return; }
  } catch(e){}
  downloadText(slug + '.json', json);
  status('downloaded ' + slug + '.json: put it in src/layouts to make it the scenario default' + note, changed.length > 0);
}
function buildDefaultGraph(){
  editor.clear();
  editor.viewSlots = {}; editor.activeView = null;
  guidedTour(editor);
  markBuilt(editor.nodes);
  editor.scenarioName = 'guided tour';
  layoutTopDown(editor.nodes, { groups: editor.groups, trunk:true, route:(x, y) => editor.addNode('pin', x, y) });
  applySavedLayout('guided tour');
  assignDefaultSlots();
  editor.frameAll();
  showProps(null);
  firstOpenInspect = true;
  recompute();
}
// The first open, once the default scene's network is wired and every load has settled: the top of the graph at reading scale, the connect node's settings open, one neuron inspected.
function firstView(net){
  editor.frameTop();
  const first = editor.nodes.find(n => n.type === 'connect');
  if(first) editor.selectNode(first);
  editor.draw();
  // the inspection queries the engine, which answers only once its hello is in; until then the request would be dropped
  const idx = Math.floor(net.count/2);
  const whenReady = (tries) => {
    if(engineHello && !helloPending) viewer.inspect(idx);
    else if(tries < 40) setTimeout(() => whenReady(tries + 1), 250);
  };
  whenReady(0);
}
function initGraph(){
  try {
    const d = JSON.parse(localStorage.getItem(STORE));
    if(d && d.nodes && d.nodes.length){
      // no save may land until the buffer and the file have been compared: the buffer's computation would otherwise schedule a save that writes the buffer over the file before reconcileScene has read it
      sceneSwitching = true;
      if(d.resolution) applyResolution(d.resolution, false);
      applySlice(d.slice); applyHues(d.hues);   // the buffer carries the scene settings too
      editor.load(d); assignDefaultSlots(); showProps(null); recompute();
      editor.scenarioName = null;
      reconcileScene();
      return;
    }
  } catch(e){ sceneSwitching = false; }   // a buffer that does not load (its node modules are gone, say): the file below decides
  // Nothing in this browser.
  // The scene may still be in the project, and falling straight through to the default graph would leave it sitting there with nothing that ever opens it.
  // The graph is empty for the one tick this takes, which is cheaper than building a default and then visibly replacing it.
  openProjectScene().then(found => { if(!found) buildDefaultGraph(); restoreTabs(); });
}
// A run from the address bar, so a sweep can be a script rather than a sequence of clicks: ?scenario=<name> builds the scenario in place of the stored scene, &over=<selector>.<param>=<value>;... edits nodes before computing (a selector is a node name, a name prefix ending in *, #type for every node of a type, or @tag for the scatter carrying a population tag), and &train=1 starts the run on the checkpoint with
// &hours, &ckptmin, &engine (gpu, cpu, cuda) applied to it. The scene on
// disk is not written over: the graph is loaded without becoming the open scene's content until something is edited by hand.
const URL_Q = new URLSearchParams(location.search);
// A scene carried in the fragment (the scene menu's link): loaded the way a ?scenario= graph is, on screen and not written over the open file.
// The project's node modules load before any scene does, since a scene may hold their nodes.
const nodeModulesReady = refreshNodeModules();
if(/(^|[#&])scene=/.test(location.hash)){
  booting = false;                       // a scene from a link is not the open file and is never saved over it
  Promise.all([decodeSceneFragment(location.hash), nodeModulesReady]).then(([d]) => {
    if(!d) return;
    urlScene = true; tabs.cur.file = null; tabs.cur.label = 'scene from a link'; renderTabs(); showSceneName();
    exitLive(); doc.brainByNode.clear();
    if(d.resolution) applyResolution(d.resolution, false);
    applySlice(d.slice); applyHues(d.hues);
    editor.load(d); assignDefaultSlots(); showProps(null); recompute();
    editor.frameAll();
    status('scene from the link: ' + d.nodes.length + ' nodes at ' + (d.resolution || 100) + '% resolution; it is not the open file, save as to keep it');
  }).catch(e => status('the link does not open: ' + e.message, true));
}
else if(URL_Q.get('scenario')){
  booting = false;
  const want = URL_Q.get('scenario').toLowerCase();
  const sc = ALL_SCENARIOS.find(x => x.name.toLowerCase() === want) ||
    ALL_SCENARIOS.find(x => x.name.toLowerCase().includes(want));
  if(!sc) status('no scenario named ' + URL_Q.get('scenario'), true);
  else {
    urlScene = true; tabs.cur.file = null; tabs.cur.label = sc.name; renderTabs(); showSceneName();
    editor.cb.onScenario(sc);
    // semicolons only between address-bar overrides, since a value can be a variants list with commas inside it
    const applied = applyOverrides(URL_Q.get('over'), editor.nodes, n => editor.markDirty(n), /;/);
    if(applied.length) status('overrides: ' + applied.join(' '));
    if(URL_Q.get('train') === '1'){
      const out = editor.nodes.find(n => n.type === 'checkpoint');
      if(out){
        if(URL_Q.get('hours')) out.params.trHours = +URL_Q.get('hours');
        if(URL_Q.get('ckptmin')) out.params.trCkptMin = +URL_Q.get('ckptmin');
        const eng = { gpu:0, cpu:1, cuda:2 }[URL_Q.get('engine') || ''];
        if(eng !== undefined) out.params.trEngine = eng;
        // the computation has to finish before a run can take the network; the dialog's own start does the rest
        const tryStart = () => {
          if(!doc.curNet || doc.curNet.kind !== 'net'){ setTimeout(tryStart, 500); return; }
          openTrainDialog(out).then(() => document.getElementById('trGo').click());
        };
        setTimeout(tryStart, 1500);
      }
    }
  }
} else nodeModulesReady.then(() => { booting = false; initGraph(); });
// ?audit=settings runs the settings audit on the graph once it has computed,
// and writes the result into the project folder.
if(URL_Q.get('audit') === 'settings'){
  // &res=<percent> audits a big scene at a working resolution: the
  // question is whether each setting reaches the tissue, which a tenth of the cells answers in a tenth of the time
  if(URL_Q.get('res')) applyResolution(+URL_Q.get('res'), true);
  const go = async () => {
    if(!doc.curNet || doc.curNet.kind !== 'net'){ setTimeout(go, 500); return; }
    const { auditSettings } = await import('./audit-settings.js');
    const lines = [];
    status('settings audit running; see the console');
    const res = await auditSettings(editor, { log:s => { lines.push(s); console.log(s); } });
    const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
    const report = { format:'np-audit-settings-1', stamp, scene:editor.scenarioName || activeScene,
      resolution:+resEl.value, checked:res.checked, flagged:res.flagged, log:lines };
    try { const st = await store(); await st.write('runs/audit-settings-' + stamp + '.json',
      JSON.stringify(report, null, 1)); } catch(e){}
    globalThis.__npAudit = report;
    status(`settings audit: ${res.checked} settings moved, ${res.flagged.length} flagged (runs/audit-settings-${stamp}.json)`);
  };
  setTimeout(go, 2000);
}
// the buttons that carry an icon, dressed once; their words stay in the title, and the ones whose meaning is not a picture keep a word beside it
for(const [id, name, text] of [
    ['pause', 'pause', ''], ['live', 'unlock', ''],
    ['popboxes', 'box', ''], ['popcolor', 'palette', ''], ['slicebtn', 'slice', ''],
    ['tbpause', 'pause', ''], ['tbsave', 'save', ''],
    ['keysbtn', 'keys', ''], ['docsbtn', 'docs', ''],
    ['projUp', 'up', ''], ['newProjBrowse', 'browse', ''],
    ['docsClose', 'close', ''], ['keysClose', 'close', '']])
  iconify(document.getElementById(id), name, text);
installTips();
// the page is drawn only now that its buttons carry their icons (index.html hides it until this class is set)
document.documentElement.classList.add('ready');
// The splash shows once per browser, and again from the wordmark; OPEN or Escape puts it away.
{
  const splash = document.getElementById('splash'), SEEN = 'linen-splash-v1';
  let seen = false; try { seen = localStorage.getItem(SEEN) === '1'; } catch(e){}
  const put = () => { splash.hidden = true; try { localStorage.setItem(SEEN, '1'); } catch(e){} };
  document.getElementById('splashopen').onclick = put;
  document.getElementById('apptitle').onclick = e => { e.preventDefault(); splash.hidden = false; document.getElementById('splashopen').focus(); };
  window.addEventListener('keydown', e => { if(e.key === 'Escape' && !splash.hidden) put(); });
  if(!seen){ splash.hidden = false; document.getElementById('splashopen').focus(); }
}
window.editor = editor; window.viewer = viewer; window.io = io; window.tabs = tabs;   // debug access
renderTabs();

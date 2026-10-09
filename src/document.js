// The document: what belongs to the open scene, and nothing about how it is shown.
// The wired network the engine is running, the brains attached to its checkpoints, the weight lock, the undo history, and the decisions a computation leads to. main.js holds one of these and does the showing: the viewer, the engine's worker, the status line.
// Nothing here touches the DOM, so a test can make a document, compute a scene into it and run the engine on what it says, through the same code the page uses (document.test.mjs).
//
// As module-level state in main.js, which no suite can import (it needs a canvas and the 3D library), the glue between a computation and an engine would be the one part of the app nothing tests.
// Scenes as tabs is one of these per tab.

import { engineConfig, engineTypes, NEURON_TYPES } from './nodes.js';
import { initState } from './rand.js';
import { applyBrainWeights } from './brain.js';

// What the engine needs of an input mapping: the channel layout, not the node that made it.
export const engineInputs = net => (net.inputMaps || []).map(m =>
  ({ chStart:m.chStart, chIdx:m.chIdx, chW:m.chW, amp:m.amp }));

// The init message for a network: what the engine on the page runs, what a training run of the same graph runs (engineConfig), and what the Brian 2 export is built from, all one object.
export function initMessage(net){
  return { cmd:'init', count:net.count,
    ntype:net.ntype, bias:net.bias,
    inputs:engineInputs(net),
    // every engine-facing field from one place, so the live view runs the same model a training run of this graph runs
    ...engineConfig(net),
    pmask:net.pmask,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay,
    types: engineTypes(),
    seed: net.seed, ...initState(net.seed, net.ntype, NEURON_TYPES) };
}
// What the engine takes on a tune: the parts of a network a recomputation can change without rewiring.
export function tuneMessage(net){
  return { cmd:'tune', bias:net.bias, inputs:engineInputs(net), ...engineConfig(net) };
}

// What the hello check reads off a network (missingTerms in protocol.js): every engine-facing field, the input maps, and the type table with the cells, without the initial state an init computes.
export function engineTerms(net){
  return { ...engineConfig(net), inputs:engineInputs(net), ntype:net.ntype, types:engineTypes() };
}

// Terms the WebGPU engine would have to refuse: [key, label] pairs, read off the network.
// Empty, since the engine carries every term (ENGINE.md; the battery's cross-engine group checks each one for parity and for not being ignored).
// The guard stays so a term added to the reference without a port on the WebGPU engine is listed here the same day rather than run without; a term listed here keeps every scene with it on off the gpu engine, and the status line says the engine lacks it.
export const GPU_MISSING = [];
export function gpuMissing(net){
  return GPU_MISSING.filter(([k]) => +net[k] > 0).map(([, label]) => label);
}
// What 'auto' picks.
// The gpu engine is not simply faster: every tick costs a submit and a readback, and at a small steps/frame that fixed cost is most of the tick.
// Measured here, throughput as a multiple of realtime:
//
//   network            synapses   cpu@2  gpu@2   cpu@20  gpu@20
//   balanced random      930k     4.13   0.53     5.00    4.96
//   CA3                  674k     5.24   0.56     6.34    4.65
//   cortical column    10.06M     1.76   0.54     2.07    2.51
//   alphabet school 2 157.30M     0.163  0.441       -       -
//
// Fitting wall time per tick as fixed + steps * perStep on the column gives the gpu 2.71 ms fixed and 0.2425 ms per step, against 0.4774 ms per step on the cpu and no meaningful fixed cost.
// So the gpu is already twice as fast per step; the round trip is the entire problem.
// Both per-step costs are proportional to synapse count, which makes the crossover a single product rather than two separate thresholds: the gpu pays once
//
//   steps * synCount * (perSynCpu - perSynGpu) > fixed
//
// and the measured constants put that boundary at 1.16e8.
// It reproduces every point above: 0.93M never crosses inside the 20-step maximum (a tie at 20, as measured), the column crosses at 11.5 steps (measured between 8 and 16), and 157.30M crosses at well under one step, so alphabet school 2 takes the gpu at the default tick and runs 2.71x faster there.
//
// That last row is also the model's known weakness, in the safe direction.
// Fitting per-step cost as proportional to synapse count predicted 1.5x on the 157M scene and the measured figure is 2.71x: the gpu scales better than linearly because a larger network occupies the device more fully, while the cpu came in close to prediction.
// So the rule is conservative, and errs toward leaving a mid-size network on the cpu that the gpu would have won.
// Correcting that needs points between 10M and 157M rather than a retune from one measurement.
// Recheck GPU_BREAKEVEN if either engine changes its inner loop.
export const GPU_BREAKEVEN = 1.16e8;
export function wantGpu(net, hasGpu = typeof navigator !== 'undefined' && !!navigator.gpu){
  if(!hasGpu) return false;
  const e = net.engine|0;
  if(e === 1 || e === 3) return false;             // cpu, or the remote engine, asked for by name
  if(gpuMissing(net).length) return false;         // would throw, or worse, drift
  if(e === 2) return true;                         // gpu, asked for by name
  return net.synCount * (net.steps || 2) >= GPU_BREAKEVEN;
}

// Which engine a network asks for: 'remote' (an engine at the checkpoint's engine address), 'gpu' or 'cpu'.
export function engineKind(net, hasGpu){
  if((net.engine|0) === 3) return 'remote';
  return wantGpu(net, hasGpu) ? 'gpu' : 'cpu';
}

export const HISTORY_MAX = 80;

export class Doc {
  constructor(){
    this.curNet = null;                  // the network the engine is running
    this.brainByNode = new Map();        // checkpoint node id -> attached brain (re-applied on every rewiring)
    this.pendingBrain = null;            // a brain just loaded, for the next computation of the active view
    this.livePhase = false;              // the weight lock
    this.lastSimView = null;             // the checkpoint the running network was computed from
    this.computeGen = 0;                    // stale computations bail against this
    this.hist = []; this.hi = -1;        // undo: snapshots of the whole graph
  }
  // ---- undo ----
  // true when the snapshot was new
  pushHistory(snapshot){
    if(this.hist[this.hi] === snapshot) return false;
    this.hist = this.hist.slice(0, this.hi + 1);
    this.hist.push(snapshot);
    if(this.hist.length > HISTORY_MAX) this.hist.shift();
    this.hi = this.hist.length - 1;
    return true;
  }
  // the graph di steps away, without its camera (undo is about the graph), or null at either end
  travel(di){
    const i = this.hi + di;
    if(i < 0 || i >= this.hist.length) return null;
    this.hi = i;
    const d = JSON.parse(this.hist[i]); delete d.view;
    return d;
  }
  // ---- a computation ----
  // The node a computation shows: the active view, or the first checkpoint when there is none or it is a note.
  // Sets the editor's active view to it.
  viewOf(editor){
    let view = editor.activeView ? editor.byId(editor.activeView) : null;
    if(!view || view.type === 'note') view = editor.nodes.find(n => n.type === 'checkpoint') || null;
    if(view) editor.activeView = view.id;
    return view;
  }
  // What a freshly computed network means for the engine that is running.
  //   tune     the same synapses and cells: adjust the running engine in place
  //   start    a new network: a new engine
  //   locked   the weights are locked and this would discard them; nothing changes
  // An attached or pending brain is applied here, since it decides between tune and start.
  // The engine's side of a tune that changes which engine is wanted is planEngine, after the caller has adopted the network.
  planCompute(net, view, running){
    // bias/noise-only recomputation (a stimulus edit after connect): synapse and point arrays pass through by reference, so the live engine is tuned in place instead of being re-initialized with its dynamics reset
    const cur = this.curNet;
    let tune = !!running && !!cur && net.preStart === cur.preStart &&
      net.pos === cur.pos && net.ntype === cur.ntype && net.count === cur.count;
    // the tune fast-path would skip weight application (a memoized rewiring returns identical arrays), so a pending or attached brain forces a fresh engine
    if(this.pendingBrain || (this.brainByNode.has(view.id) && view.id !== this.lastSimView)) tune = false;
    if(!tune && this.livePhase && running)
      return { action:'locked', why:'weights are locked: this edit would rewire the network and lose them. Unlock to rewire.' };
    // brains attach per checkpoint node: a loaded file applies to the active view and re-applies every time it is rewired, so several checkpoints can each carry their own.
    // Matching pairs restore their weight, new synapses stay at baseline.
    let brainMsg = '';
    const attached = this.pendingBrain || this.brainByNode.get(view.id);
    if(attached && !tune){
      const r = applyBrainWeights(net, attached);
      net = r.net;
      this.brainByNode.set(view.id, attached);
      this.pendingBrain = null;
      brainMsg = ` · brain: ${r.matched.toLocaleString()} of ${r.saved.toLocaleString()} saved pairs`;
    }
    this.curNet = net;
    return { action: tune ? 'tune' : 'start', net, brainMsg };
  }
  // A tune that changes which engine is wanted needs a new engine, which the weight lock refuses.
  // The running engine is given as { kind, host }, what engineKind said when it started and the address it connected to; a boolean is the gpu flag alone.
  planEngine(net, running, hasGpu){
    const cur = running && typeof running === 'object' ? running : { kind: running ? 'gpu' : 'cpu' };
    const kind = engineKind(net, hasGpu);
    if(kind === cur.kind && (kind !== 'remote' || (net.liveHost || '') === (cur.host || ''))) return { action:'keep' };
    if(this.livePhase) return { action:'locked', why:'weights are locked: switching engine would lose them. Unlock first.' };
    return { action:'restart' };
  }
}

// ---- scene tabs ----
// Several scenes open at once, one document each.
// Two of them are marked: `cur` is the one in the node graph, `shown` is the one whose network the viewer and the engine hold.
// They differ after a tab is switched to and before a viewer key is pressed there: switching tabs changes what is edited and leaves what is running alone.
//
// A tab off screen keeps its scene as the object its file holds (`graph`), so coming back needs no read and keeps the camera; one restored from the project's list has none until it is first switched to.
// A tab that is not shown keeps the viewer-side settings of its scene (resolution, slice, hues) in `held`, since the live ones belong to the shown scene.
export class SceneTabs {
  constructor(){ this.list = []; this.cur = null; this.shown = null; this.n = 0; }
  // file is the scene's path in the project, or null for a graph that is not a file (one from the address bar)
  add(file, label, at){
    const t = { key:++this.n, file:file || null, label:label || '', doc:new Doc(), graph:null, held:null };
    if(at === undefined || at < 0 || at > this.list.length) this.list.push(t); else this.list.splice(at, 0, t);
    return t;
  }
  byFile(file){ return file ? this.list.find(t => t.file === file) || null : null; }
  // the tab to go to when this one closes: the one after it, else before
  neighbour(tab){
    const i = this.list.indexOf(tab);
    return i < 0 ? null : this.list[i + 1] || this.list[i - 1] || null;
  }
  remove(tab){
    const i = this.list.indexOf(tab);
    if(i >= 0) this.list.splice(i, 1);
    return i >= 0;
  }
  // one tab, as a page starts or a project is adopted
  reset(file, label){
    this.list = [];
    this.cur = this.shown = this.add(file, label);
    return this.cur;
  }
  // what project.json records
  openFiles(){ return this.list.filter(t => t.file).map(t => t.file); }
  // Tabs for the files a project had open, in its order, around the tab already here.
  // `exists` says whether a file is still in the project.
  restore(files, exists = () => true){
    const want = (files || []).filter((f, i, a) => typeof f === 'string' && a.indexOf(f) === i && exists(f));
    const here = this.cur && this.cur.file;
    const at = want.indexOf(here);
    let k = 0;
    for(const f of want){
      if(this.byFile(f)) continue;
      const before = at >= 0 && want.indexOf(f) < at;
      if(before) this.add(f, '', this.list.indexOf(this.cur)); else this.add(f, '');
      k++;
    }
    return k;
  }
}

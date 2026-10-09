// A smoke test through the path the page takes: a scenario built into a graph, computed, handed to the document, the document's init message run on the reference engine, then an edit that tunes, an edit the weight lock refuses, a brain attached, and undo.
import { Doc, SceneTabs, initMessage, tuneMessage, wantGpu, gpuMissing, engineKind } from './document.js';
import { NODE_DEFS, computeNode, setResolution, wireConnect, assembleConnect } from './nodes.js';
import { SCENARIOS } from './scenarios.js';
import { buildBrainFile, parseBrainFile } from './brain.js';
import { ok, report, inProcessWorker } from '../tools/harness.mjs';
inProcessWorker(wireConnect, assembleConnect);

function mkEditor(){
  return { nodes:[], nextId:1, groups:[], activeView:null,
    byId(id){ return this.nodes.find(n => n.id === id); },
    // an edit clears the node's cached result and every one below it, as the editor does
    markDirty(node){ node._cache = null; for(const m of this.nodes) if(m.inputs.some(c => c && c.id === node.id)) this.markDirty(m); },
    addNode(t, x, y){ const def = NODE_DEFS[t], params = {}; def.params.forEach(p => params[p.k] = structuredClone(p.def));
      const n = { id:this.nextId++, type:t, x, y, params, inputs:new Array(def.inputs).fill(null) }; this.nodes.push(n); return n; },
    toJSON(){ return { v:1, nodes:this.nodes.map(n => ({ id:n.id, type:n.type, x:n.x, y:n.y, params:structuredClone(n.params), inputs:n.inputs.map(c => c ? { id:c.id } : null) })), groups:[] }; } };
}
// the reference engine in this process, as tools/tune.mjs runs it
const replies = []; globalThis.postMessage = d => replies.push(d); globalThis.onmessage = null;
await import('./simworker.js');
const send = d => { replies.length = 0; globalThis.onmessage({ data:d }); return replies.slice(); };
const run = ms => { let spikes = 0; for(let t = 0; t < ms; t += 2){ for(const r of send({ cmd:'tick', steps:2 })){ if(r.cmd === 'error') throw new Error(r.message); if(r.cmd === 'state') spikes += r.spikes; } } return spikes; };

setResolution(0.05);
const ed = mkEditor();
SCENARIOS.find(s => s.name === 'balanced random net').build(ed);
const byId = id => ed.byId(id);
const doc = new Doc();

// the view a computation shows
const view = doc.viewOf(ed);
ok('the view is the checkpoint, and the editor is told', view && view.type === 'checkpoint' && ed.activeView === view.id);

// compute, plan, init, run
let net = await computeNode(view, byId);
let plan = doc.planCompute(net, view, false);
ok('a first network starts an engine and becomes the document\'s', plan.action === 'start' && doc.curNet === plan.net && plan.brainMsg === '');
const init = initMessage(plan.net);
ok('the init message carries the network, the engine settings, the type table and the initial state',
  init.cmd === 'init' && init.count === net.count && init.preStart === net.preStart && init.w === net.w && Array.isArray(init.types) && init.types.length >= 17 &&
  init.v0 && init.v0.length === net.count && init.u0 && init.seed === net.seed && init.refrac !== undefined && Array.isArray(init.inputs));
const initReply = send(init);
ok('the reference engine takes it', !initReply.some(r => r.cmd === 'error'), JSON.stringify(initReply.find(r => r.cmd === 'error') || ''));
const base = run(600);
ok('and the network fires', base > 0, String(base));
doc.lastSimView = view.id;

// a stimulus edit is a tune: same synapses, same cells, the engine adjusted in place
const noise = ed.nodes.find(n => n.type === 'stimulus');
noise.params.current *= 2; ed.markDirty(noise);
net = await computeNode(view, byId);
plan = doc.planCompute(net, view, true);
ok('a stimulus edit tunes the running engine', plan.action === 'tune' && plan.net.preStart === init.preStart);
ok('and leaves the engine choice alone', doc.planEngine(plan.net, false, false).action === 'keep');
ok('the engine takes the tune', !send(tuneMessage(plan.net)).some(r => r.cmd === 'error'));
const driven = run(600);
ok('and the doubled drive raises the rate', driven > base*1.2, base + ' then ' + driven);

// the weight lock refuses an edit that would rebuild the network
doc.livePhase = true;
const cn = ed.nodes.find(n => n.type === 'connect');
const before = doc.curNet;
cn.params.seed += 1; ed.markDirty(cn);
net = await computeNode(view, byId);
plan = doc.planCompute(net, view, true);
ok('under the weight lock rewiring is refused and the network kept', plan.action === 'locked' && /locked/.test(plan.why) && doc.curNet === before);
ok('and an engine switch is refused too', doc.planEngine({ ...before, engine:2 }, false, true).action === 'locked');
doc.livePhase = false;
ok('unlocked, the same switch restarts', doc.planEngine({ ...before, engine:2 }, false, true).action === 'restart');

// a brain saved from the running network attaches to the checkpoint and is applied on the next rewiring
const w = Float32Array.from(before.w, v => v*0.5);
const brain = parseBrainFile(buildBrainFile(ed.toJSON(), before, w, 1200, null));
doc.pendingBrain = brain;
cn.params.seed -= 1; ed.markDirty(cn);        // back to the network the brain was saved from
net = await computeNode(view, byId);
plan = doc.planCompute(net, view, true);
ok('a pending brain forces a new engine and is applied', plan.action === 'start' && /brain: [\d,]+ of [\d,]+ saved pairs/.test(plan.brainMsg), plan.brainMsg);
ok('its pairs all match and carry the saved weights', (() => { const m = /brain: ([\d,]+) of ([\d,]+)/.exec(plan.brainMsg); return m && m[1] === m[2]; })() && Math.abs(plan.net.w[0] - w[0]) < 1e-6);
ok('it stays attached to that checkpoint and is no longer pending', doc.pendingBrain === null && doc.brainByNode.get(view.id) === brain);

// the engine choice
ok('no gpu, no gpu engine', wantGpu({ ...before, engine:2 }, false) === false);
ok('asked for by name it is taken, and no term is missing there (the gpu engine carries them all)', wantGpu({ ...before, engine:2 }, true) === true && wantGpu({ ...before, engine:2, stp:1, trip:0.1 }, true) === true && gpuMissing({ stp:1, trip:0.1 }).length === 0);
ok('auto goes by steps times synapses', wantGpu({ engine:0, synCount:1e6, steps:2 }, true) === false && wantGpu({ engine:0, synCount:1e8, steps:2 }, true) === true);
// the remote engine, at the checkpoint's engine address
const remote = { ...before, engine:3, liveHost:'ws://localhost:8890' };
ok('remote is its own kind, never the gpu', engineKind(remote, true) === 'remote' && wantGpu({ ...remote, synCount:1e9 }, true) === false);
ok('the cpu and gpu kinds follow wantGpu', engineKind({ ...before, engine:1 }, true) === 'cpu' && engineKind({ ...before, engine:2 }, true) === 'gpu');
{
  const d3 = new Doc();
  ok('the same address keeps the running remote engine', d3.planEngine(remote, { kind:'remote', host:'ws://localhost:8890' }, true).action === 'keep');
  ok('another address restarts it', d3.planEngine({ ...remote, liveHost:'ws://localhost:8891' }, { kind:'remote', host:'ws://localhost:8890' }, true).action === 'restart');
  ok('switching from the cpu to the remote engine restarts', d3.planEngine(remote, { kind:'cpu' }, true).action === 'restart');
  d3.livePhase = true;
  ok('and the weight lock refuses it', d3.planEngine(remote, { kind:'cpu' }, true).action === 'locked');
}
{
  // the checkpoint carries the address on the network
  const cp = NODE_DEFS.checkpoint;
  const p = {}; cp.params.forEach(q => p[q.k] = structuredClone(q.def));
  const computed = cp.compute([{ ...before, kind:'net' }], { ...p, engine:3, liveHost:' ws://example.test:1 ' });
  ok('the checkpoint puts its engine address on the network, trimmed', computed.engine === 3 && computed.liveHost === 'ws://example.test:1');
  ok('the engine select offers the remote engine', cp.params.find(q => q.k === 'engine').options[3].startsWith('remote'));
}

// undo
const d2 = new Doc();
ok('a new snapshot is kept and the same one is not kept twice', d2.pushHistory('{"v":1,"view":{"x":1},"nodes":[1]}') && !d2.pushHistory('{"v":1,"view":{"x":1},"nodes":[1]}') && d2.pushHistory('{"v":1,"view":{"x":2},"nodes":[1,2]}'));
const back = d2.travel(-1);
ok('undo returns the graph before, without its camera', back && back.nodes.length === 1 && back.view === undefined && d2.travel(-1) === null);
ok('redo returns the graph after', d2.travel(+1).nodes.length === 2 && d2.travel(+1) === null);
ok('a new edit after an undo drops the redo', (d2.travel(-1), d2.pushHistory('{"nodes":[9]}'), d2.travel(+1) === null));

// scene tabs: the bookkeeping main.js leans on
const T = new SceneTabs();
const first = T.reset('scenes/b.json', 'b');
ok('a page starts on one tab that is both edited and shown', T.list.length === 1 && T.cur === first && T.shown === first && first.doc instanceof Doc);
T.restore(['scenes/a.json', 'scenes/b.json', 'scenes/gone.json', 'scenes/c.json', 'scenes/a.json'], f => f !== 'scenes/gone.json');
ok('the open scenes of the project come back in its order around the tab already here, without the missing or the doubled',
  T.list.map(t => t.file).join() === 'scenes/a.json,scenes/b.json,scenes/c.json' && T.cur === first, T.list.map(t => t.file).join());
ok('a restored tab holds no graph until it is switched to, and has a document of its own', T.list[0].graph === null && T.list[0].doc !== first.doc);
ok('a scene has one tab', T.byFile('scenes/c.json') === T.list[2] && T.byFile('scenes/x.json') === null && T.byFile(null) === null);
const link = T.add(null, 'from a link');
ok('a graph that is not a file has a tab and is not in what the project records', T.list.length === 4 && T.openFiles().join() === 'scenes/a.json,scenes/b.json,scenes/c.json');
ok('closing goes to the tab after, or the one before at the end', T.neighbour(first) === T.list[2] && T.neighbour(link) === T.list[2]);
ok('a closed tab is gone', T.remove(link) && T.list.length === 3 && !T.remove(link));
ok('a new tab can sit beside the one in the editor', T.add('scenes/d.json', 'd', T.list.indexOf(first) + 1) === T.list[2]);
ok('each tab has its own undo', (first.doc.pushHistory('{"nodes":[1]}'), T.list[0].doc.hist.length === 0 && first.doc.hist.length === 1));
setResolution(1);
report('document');

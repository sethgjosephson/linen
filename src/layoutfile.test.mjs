import { layoutSlug, layoutRecord, applyLayoutRecord, markBuilt, builtTypes, layoutMismatch, LAYOUT_FORMAT } from './layoutfile.js';
import { NODE_DEFS } from './nodes.js';
import { ok, report } from '../tools/harness.mjs';

ok('a slug is lower case with dashes', layoutSlug('CA3 pattern completion') === 'ca3-pattern-completion' && layoutSlug('the weave, letters') === 'the-weave-letters');

function mkEditor(){
  return { nodes:[], nextId:1, groups:[],
    addNode(t, x, y){ const def = NODE_DEFS[t], params = {}; def.params.forEach(p => params[p.k] = structuredClone(p.def));
      const n = { id:this.nextId++, type:t, x, y, params, inputs:new Array(def.inputs).fill(null) }; this.nodes.push(n); return n; },
    deleteNode(n){ this.nodes = this.nodes.filter(m => m !== n); for(const m of this.nodes) m.inputs = m.inputs.map(c => c && c.id === n.id ? null : c);
      for(const g of this.groups) g.members = g.members.filter(id => id !== n.id); } };
}
// the "scenario": a sphere, two scatters, a merge, a probe, a checkpoint
function build(ed){
  const sp = ed.addNode('sphere', 0, 0), a = ed.addNode('scatter', 0, 60), b = ed.addNode('scatter', 150, 60);
  a.inputs[0] = { id:sp.id }; b.inputs[0] = { id:sp.id }; a.params.count = 123;
  const m = ed.addNode('gather', 0, 120); m.inputs[0] = { id:a.id }; m.inputs[1] = { id:b.id };
  const pr = ed.addNode('probe', 0, 180); pr.inputs[0] = { id:m.id };
  const out = ed.addNode('checkpoint', 0, 240); out.inputs[0] = { id:pr.id };
  ed.groups = [{ label:'cells', hue:130, members:[sp.id, a.id, b.id, m.id] }, { label:'checkpoint', hue:0, members:[out.id] }];
  markBuilt(ed.nodes);
}

// someone arranges it: deletes the probe, replaces the merge tree with a 3-port merge of their own, drops a dot before the checkpoint, adds a note
const ed = mkEditor(); build(ed);
const [sp, a, b, m0, pr, out] = ed.nodes;
ed.deleteNode(pr); ed.deleteNode(m0);
const m = ed.addNode('gather', 40, 130); m.params.ports = 3; m.inputs = [{ id:a.id }, { id:b.id }, null];
const dot = ed.addNode('pin', 200, 200); dot.inputs[0] = { id:m.id };
out.inputs[0] = { id:dot.id }; out.x = 400; a.params.count = 999;
const note = ed.addNode('note', -300, 0); note.params.text = 'hello';
ed.groups[0].members.push(m.id, dot.id); ed.groups[0].pad = { l:10, t:0, r:0, b:0 };
const buildTypes = ['sphere', 'scatter', 'scatter', 'gather', 'probe', 'checkpoint'];
const rec = layoutRecord(ed, 'test scene', buildTypes);
ok('the record is format ' + LAYOUT_FORMAT + ' with the build order', rec.format === LAYOUT_FORMAT && rec.build.join() === buildTypes.join());
ok('built nodes carry their build index and no settings', rec.nodes[0].built === 0 && rec.nodes[0].params === undefined && rec.nodes[1].built === 1 && rec.nodes[1].params === undefined);
ok('added nodes carry their settings', rec.nodes.find(n => n.type === 'gather').params.ports === 3 && rec.nodes.find(n => n.type === 'note').params.text === 'hello');
ok('wires are by file index', rec.nodes.find(n => n.type === 'pin').inputs[0] === rec.nodes.findIndex(n => n.type === 'gather') && rec.nodes[3].inputs[0] === rec.nodes.findIndex(n => n.type === 'pin'));
ok('a hole where a built node was deleted', builtTypes(ed.nodes)[4] === undefined && builtTypes(ed.nodes)[5] === 'checkpoint');

// applied to a fresh build
const fresh = mkEditor(); build(fresh);
const r = applyLayoutRecord(fresh, rec);
const fm = fresh.nodes.filter(n => n.type === 'gather'), fd = fresh.nodes.find(n => n.type === 'pin'), fo = fresh.nodes.find(n => n.type === 'checkpoint'), fn = fresh.nodes.find(n => n.type === 'note');
ok('it applies: two added (merge, dot, note make three) and two removed', r.ok && r.added === 3 && r.removed === 2, JSON.stringify(r));
ok('the probe is gone and the build’s merge with it', !fresh.nodes.some(n => n.type === 'probe') && fm.length === 1 && fm[0].params.ports === 3);
ok('the wires are the file’s: scatters into the new merge, merge, dot, checkpoint', fm[0].inputs[0].id === fresh.nodes[1].id && fm[0].inputs[1].id === fresh.nodes[2].id && fd.inputs[0].id === fm[0].id && fo.inputs[0].id === fd.id);
ok('positions come from the file', fo.x === 400 && fn && fn.x === -300 && fn.params.text === 'hello');
ok('a built node keeps the build’s settings, not the file’s', fresh.nodes[1].params.count === 123);
ok('groups come from the file, mapped to the new ids', fresh.groups[0].pad.l === 10 && fresh.groups[0].members.includes(fd.id) && fresh.groups[0].members.includes(fm[0].id) && fresh.groups[1].members[0] === fo.id);

// the builder changed since the save
const grown = mkEditor(); build(grown); grown.addNode('chart', 0, 300); markBuilt(grown.nodes);
const bad = applyLayoutRecord(grown, rec);
ok('a build with more nodes is refused, with a reason', !bad.ok && /builds 7 nodes now and built 6/.test(bad.why), bad.why);
const retyped = mkEditor(); build(retyped); retyped.nodes[1].type = 'box';
ok('a build with a retyped node is refused', /node 1 of the build is a box now/.test(layoutMismatch(builtTypes(retyped.nodes), rec)));
ok('an old format is refused', !applyLayoutRecord(fresh, { format:2, nodes:[] }).ok && !applyLayoutRecord(fresh, { hello:1 }).ok);
report('layout file');

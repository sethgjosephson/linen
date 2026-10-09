// No note in a Tab menu scene sits on top of another note or a node.
//
// A scene is opened the way the page opens it: built, laid out by layout.js, then given its saved arrangement (src/layouts/<slug>.json) when the site carries one.
// A saved arrangement fixes where a note is, not how tall it is, so a note whose text grew can come to rest over its neighbor; the computed layout measures the text and cannot.
// A note's height here is layout.js's estimate, which is a little generous, as the editor's own wrap needs a canvas.

import fs from 'node:fs';
import { ok, report } from '../tools/harness.mjs';
import { NODE_DEFS } from './nodes.js';
import { SCENARIOS } from './scenarios.js';
import { layoutTopDown, noteHeight } from './layout.js';
import { layoutSlug, markBuilt, applyLayoutRecord } from './layoutfile.js';

const W = 130, H = 26, DOT = 18;          // editor.js node, pin
function mkEditor(){
  return { nodes:[], nextId:1, groups:[], viewSlots:{},
    addNode(t, x, y, params){ const def = NODE_DEFS[t], p = {}; def.params.forEach(q => p[q.k] = structuredClone(q.def)); Object.assign(p, params || {});
      const n = { id:this.nextId++, type:t, x, y, name:'', params:p, inputs:new Array(def.inputs).fill(null) }; this.nodes.push(n); return n; },
    deleteNode(n){ this.nodes = this.nodes.filter(m => m !== n); for(const m of this.nodes) m.inputs = m.inputs.map(c => c && c.id === n.id ? null : c);
      for(const g of this.groups) g.members = g.members.filter(id => id !== n.id); },
    createGroup(nodes, label, hue){ const g = { label, hue, members:nodes.map(n => n.id) }; this.groups.push(g); return g; },
    clear(){ this.nodes = []; this.groups = []; this.nextId = 1; } };
}
const rect = n => n.type === 'note' ? [n.x, n.y, +n.params.width || 220, noteHeight(n)] : n.type === 'pin' ? [n.x, n.y, DOT, DOT] : [n.x, n.y, W, H];
const hit = (a, b) => a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3];

export function overlaps(ed){
  const out = [], notes = ed.nodes.filter(n => n.type === 'note');
  for(const a of notes) for(const b of ed.nodes){
    if(a === b || (b.type === 'note' && b.id < a.id)) continue;
    if(hit(rect(a), rect(b))) out.push(`"${a.params.text.slice(0, 28)}" over ${b.type === 'note' ? '"' + b.params.text.slice(0, 28) + '"' : b.type + ' ' + (b.name || b.id)}`);
  }
  return out;
}

for(const s of SCENARIOS){
  const ed = mkEditor();
  try { s.build(ed); } catch(e){ ok(s.name + ' builds', false, e.message); continue; }
  markBuilt(ed.nodes);
  layoutTopDown(ed.nodes, { groups:ed.groups, trunk:true, route:(x, y) => ed.addNode('pin', x, y) });
  const computed = overlaps(ed);
  ok(s.name + ': no note overlaps in the computed layout', computed.length === 0, computed.join('; '));
  const file = new URL('./layouts/' + layoutSlug(s.name) + '.json', import.meta.url);
  if(!fs.existsSync(file)) continue;
  const res = applyLayoutRecord(ed, JSON.parse(fs.readFileSync(file, 'utf8')));
  if(!res.ok){ ok(s.name + ': its saved layout still matches the scene', false, res.why); continue; }
  const saved = overlaps(ed);
  ok(s.name + ': no note overlaps in its saved layout', saved.length === 0, saved.join('; '));
}
report('notes');

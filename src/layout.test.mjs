import { layoutTopDown } from './layout.js';
import { NODE_DEFS } from './nodes.js';
import { SCENARIOS } from './scenarios.js';
import { ok, report } from '../tools/harness.mjs';

function mkEditor(){
  return { nodes:[], nextId:1, groups:[],
    addNode(t, x, y){ const def = NODE_DEFS[t], params = {}; def.params.forEach(p => params[p.k] = structuredClone(p.def));
      const n = { id:this.nextId++, type:t, x, y, params, inputs:new Array(def.inputs).fill(null) }; this.nodes.push(n); return n; } };
}
const W = 130, H = 26;
const overlaps = nodes => {
  const box = n => [n.x, n.y, n.x + (n.type === 'pin' ? 18 : n.type === 'note' ? (n.params.width || 340) : W), n.y + (n.type === 'pin' ? 18 : n.type === 'note' ? 40 : H)];
  let k = 0;
  for(let i = 0; i < nodes.length; i++) for(let j = i + 1; j < nodes.length; j++){
    const [ax0, ay0, ax1, ay1] = box(nodes[i]), [bx0, by0, bx1, by1] = box(nodes[j]);
    if(ax0 < bx1 && bx0 < ax1 && ay0 < by1 && by0 < ay1) k++;
  }
  return k;
};

// the column: four populations in a row, one trunk, lanes with dots
{
  const ed = mkEditor(); SCENARIOS.find(s => s.name === 'mouse cortical column').build(ed);
  const before = ed.nodes.length;
  layoutTopDown(ed.nodes, { groups:ed.groups, trunk:true, route:(x, y) => ed.addNode('pin', x, y) });
  const byId = new Map(ed.nodes.map(n => [n.id, n]));
  const boxes = ed.nodes.filter(n => n.type === 'box'), stims = ed.nodes.filter(n => n.type === 'stimulus');
  const trunk = ed.nodes.filter(n => ['connect', 'stimulus', 'probe', 'checkpoint', 'chart'].includes(n.type));
  ok('the four layer boxes sit in one row', new Set(boxes.map(n => n.y)).size === 1 && boxes.every((b, i) => i === 0 || b.x > boxes[i-1].x + W));
  ok('the trunk is one straight column below the populations', new Set(trunk.map(n => n.x)).size === 1 && Math.min(...trunk.map(n => n.y)) > Math.max(...boxes.map(n => n.y)) + H);
  ok('the trunk runs top down in wire order', trunk.every(n => { const p = n.inputs[0] && byId.get(n.inputs[0].id); return !p || p.type === 'gather' || p.y < n.y; }));
  const dots = ed.nodes.filter(n => n.type === 'pin');
  ok('every mask now arrives through a dot level with its stimulus', dots.length > before/2 && stims.every(s => { const d = byId.get(s.inputs[1].id); return d.type === 'pin' && Math.abs(d.y + 18 - (s.y + H/2)) < 1; }));
  ok('a dot chain runs down one lane per box', boxes.every(b => { let d = ed.nodes.find(n => n.type === 'pin' && n.inputs[0].id === b.id); const x = d.x; let ys = [d.y]; for(;;){ const next = ed.nodes.find(n => n.type === 'pin' && n.inputs[0].id === d.id); if(!next) break; if(next.x !== x) return false; ys.push(next.y); d = next; } return ys.every((y, i) => i === 0 || y > ys[i-1]); }));
  ok('the lanes sit right of their own population and left of the next', boxes.every((b, i) => { const d = ed.nodes.find(n => n.type === 'pin' && n.inputs[0].id === b.id); return d.x > b.x + W && (!boxes[i+1] || d.x < boxes[i+1].x); }));
  ok('nothing overlaps', overlaps(ed.nodes) === 0);
  ok('the merges gather at the trunk', ed.nodes.filter(n => n.type === 'gather').every(m => m.x === trunk[0].x));
}
// every Tab-menu scene: no overlaps, finite coordinates, and without a route the wires run direct
for(const sc of SCENARIOS){
  const ed = mkEditor(); sc.build(ed);
  const wires = () => ed.nodes.map(n => n.inputs.map(c => c ? c.id : 0).join(',')).join(';');
  const w0 = wires();
  layoutTopDown(ed.nodes, { groups:ed.groups, trunk:true });
  ok(sc.name + ': laid out without dots, the wires untouched', wires() === w0 && overlaps(ed.nodes) === 0 && ed.nodes.every(n => Number.isFinite(n.x) && Number.isFinite(n.y)));
  layoutTopDown(ed.nodes, { groups:ed.groups, trunk:true, route:(x, y) => ed.addNode('pin', x, y) });
  ok(sc.name + ': with dots, nothing overlaps and every dot is on a wire', overlaps(ed.nodes) === 0 && ed.nodes.filter(n => n.type === 'pin').every(d => d.inputs[0] && ed.nodes.some(n => n.inputs.some(c => c && c.id === d.id))));
}
report('layout');

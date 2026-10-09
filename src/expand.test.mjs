// Expanding a points file tagged by column: a population node per tag reading the file node, a merge in the file node's place downstream, the file node kept; collapsing removes exactly that fan and nothing else.
import { ok, report } from '../tools/harness.mjs';
import { expandPopulations, collapsePopulations, fanOf } from './expand.js';
import { registerCellType, NODE_DEFS, NEURON_TYPES } from './nodes.js';
const { NodeEditor } = await import('./editor.js');
const ed = Object.create(NodeEditor.prototype);
Object.assign(ed, { nodes:[], groups:[], nextId:1, cb:{ onSelect(){}, onChange(){}, onMoved(){} },
  viewSlots:{}, activeView:null, suggestResolution:0, selSet:new Set(), sel:null });
const pf = ed.addNode('pointfile', 400, 100); pf.name = 'cells';
const con = ed.addNode('connect', 400, 300); con.inputs[0] = { id:pf.id };
const { children, merge } = expandPopulations(ed, pf, ['fly.a', 'fly.b', 'fly.c']);
ok('one population node per tag, reading the file node', children.length === 3 && children.every(c => c.type === 'population' && c.inputs[0].id === pf.id));
ok('each keeps its tag and a short name', children[1].params.tag === 'fly.b' && children[1].name === 'b');
ok('the file node stays', ed.nodes.includes(pf));
ok('the merge reads all of them and the consumer reads the merge', merge.params.ports === 3 && merge.inputs.map(c => c.id).join(',') === children.map(c => c.id).join(',') && con.inputs[0].id === merge.id);
ok('the fan is recognized', fanOf(ed, pf).merge === merge);
ok('the fan is grouped, named for the file node', ed.groups.length === 1 && ed.groups[0].label === 'cells populations' && ed.groups[0].members.length === 4);
const tr = ed.addNode('move', 0, 0); tr.inputs[0] = { id:children[0].id };
ok('a population feeding something else refuses the collapse', /feed 2 nodes/.test(collapsePopulations(ed, pf).why || ''), collapsePopulations(ed, pf).why);
ed.nodes = ed.nodes.filter(n => n !== tr);
const fan = collapsePopulations(ed, pf);
ok('collapse removes the populations and the merge', fan.merge === merge && !ed.nodes.some(n => n.type === 'population' || n.type === 'gather'));
ok('the consumer reads the file node again', con.inputs[0].id === pf.id);
ok('the group went with the fan', ed.groups.length === 0);

// the population node's compute keeps identity and the ids
const pts = { kind:'points', count:4, pos:Float32Array.from([0,0,0, 10,0,0, 20,0,0, 30,0,0]), ntype:Uint8Array.from([0,0,3,3]), bias:new Float32Array(4),
  src:Int32Array.from([7,7,8,8]), lidx:Uint32Array.from([0,1,0,1]), tags:{ 7:'fly.a', 8:'fly.b' }, ids:['x','y','z','w'] };
const a = NODE_DEFS.population.compute([pts], { tag:'fly.a' });
ok('the population keeps its cells, identity and ids', a.count === 2 && a.src[0] === 7 && a.lidx[1] === 1 && a.ids.join() === 'x,y' && a.tags[7] === 'fly.a');
ok('a pattern takes every match', NODE_DEFS.population.compute([pts], { tag:'fly.*' }).count === 4);
ok('a class works too', NODE_DEFS.population.compute([pts], { tag:'I' }).count === 2);
let msg = null; try { NODE_DEFS.population.compute([pts], { tag:'nope' }); } catch(e){ msg = e.message; }
ok('an unknown tag is refused with the list', /fly\.a, fly\.b/.test(msg || ''), msg);
// retype by population: one region of a stream gets its own type
const FL = NEURON_TYPES.findIndex(t => t.key === 'FL');
const rt = NODE_DEFS.celltype.compute([pts], { row:FL + 1, tag:'fly.b', frac:1 }, { id:5 });
ok('a built-in row with a tag touches that population only', rt.ntype[0] === 0 && rt.ntype[1] === 0 && rt.ntype[2] === FL && rt.ntype[3] === FL);
ok('a built-in row with a blank tag touches every cell', NODE_DEFS.celltype.compute([pts], { row:FL + 1, tag:'', frac:1 }, { id:5 }).ntype.every(t => t === FL));
let m2 = null; try { NODE_DEFS.celltype.compute([pts], { row:FL + 1, tag:'nope', frac:1 }, { id:5 }); } catch(e){ m2 = e.message; }
ok('an unknown tag is refused', /no cell is tagged nope/.test(m2 || ''), m2);

// the cell type node: a named row joins the type table, its cells point at it
const before = NEURON_TYPES.length;
const ct = NODE_DEFS.celltype.compute([pts], { name:'kc', tag:'fly.a', sign:0, kind:1, C:8, k:0.03, vr:-62, vt:-45, vpeak:10, a:0.05, b:0, c:-55, d:5, thr:-60, slope:8, base2003:0, hue:120 });
ok('a cell type row is appended and its population points at it', NEURON_TYPES.length === before + 1 && ct.ntype[0] === before && ct.ntype[1] === before && ct.ntype[2] === 3);
ok('the row carries the numbers, graded, and its sign', NEURON_TYPES[before].f7.C === 8 && NEURON_TYPES[before].f7.graded.thr === -60 && NEURON_TYPES[before].sign === 1);
ok('the declaration rides on the stream', ct.cellTypes.length === 1 && ct.cellTypes[0].key === 'kc');
const ct2 = NODE_DEFS.celltype.compute([pts], { name:'kc', tag:'fly.a', sign:1, kind:0, C:30, k:0.06, vr:-58, vt:-40, vpeak:20, a:0.05, b:0, c:-50, d:10, thr:-55, slope:10, base2003:3, hue:120 });
ok('the same name updates the row in place', NEURON_TYPES.length === before + 1 && ct2.ntype[0] === before && NEURON_TYPES[before].f7.C === 30 && !NEURON_TYPES[before].f7.graded && NEURON_TYPES[before].sign === -1);
// a wiring worker puts a row where the computation put it, whatever it has met before
{ const n0 = NEURON_TYPES.length;
  const at = registerCellType({ key:'farrow', name:'far', custom:true, sign:1, color:[1,1,1], f7:{ ...NEURON_TYPES[0].f7 }, index:n0 + 3 });
  ok('a row that carries an index lands on it, the gap filled with placeholders', at === n0 + 3 && NEURON_TYPES.length === n0 + 4 && NEURON_TYPES[n0].placeholder && NEURON_TYPES[n0 + 3].key === 'farrow');
  ok('the same row again stays where it is', registerCellType({ key:'farrow', name:'far', custom:true, sign:-1, color:[1,1,1], f7:{ ...NEURON_TYPES[0].f7 }, index:n0 + 3 }) === n0 + 3 && NEURON_TYPES[n0 + 3].sign === -1);
  ok('the row the cell type node puts on the stream says where it is', ct2.cellTypes.every(r => Number.isInteger(r.index) && NEURON_TYPES[r.index].key === r.key)); }
// a custom row rides on the stream so the wiring workers can register it: a repeat has to pass it on
{ const rp = NODE_DEFS.repeat.compute([ct2], { copies:2, pops:0, tag:'t', translate:[0,100,0], rotate:[0,0,0], scale:[1,1,1], pivot:[0,0,0], mirror:0 });
  ok('a repeat after a custom cell type keeps the row on the stream', Array.isArray(rp.cellTypes) && rp.cellTypes.some(r => r.key === 'kc') && rp.count === ct2.count*2); }
let m3 = null; try { NODE_DEFS.celltype.compute([pts], { name:'RS', tag:'', sign:0, kind:0, C:20, k:0.06, vr:-58, vt:-40, vpeak:20, a:0.05, b:0, c:-50, d:10, thr:-55, slope:10, base2003:0, hue:0 }); } catch(e){ m3 = e.message; }
ok('a built-in name is refused', /built-in row/.test(m3 || ''), m3);
// the operations node shuffles positions and nothing else
const sh = NODE_DEFS.ops.compute([pts], { expr:'bias = bias', a:0, b:0, c:0, d:0, seed:1, shuffle:1 });
const pos0 = Array.from(pts.pos), pos1 = Array.from(sh.pos);
ok('shuffled positions are a permutation of the originals with identity kept', pos1.join() !== pos0.join() && [...pos1].sort().join() === [...pos0].sort().join() && sh.src.join() === pts.src.join() && sh.lidx.join() === pts.lidx.join());
ok('the shuffle is seeded', Array.from(NODE_DEFS.ops.compute([pts], { expr:'bias = bias', a:0, b:0, c:0, d:0, seed:1, shuffle:1 }).pos).join() === pos1.join());
report('expand');

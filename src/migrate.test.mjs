// Scene migrations: an old file loads as the current shape.
// Every step in migrate.js is a promise about files already written, so each one gets a case here.
// Run: node src/migrate.test.mjs
import assert from 'node:assert/strict';
import { migrateScene, SCENE_FORMAT } from './migrate.js';

// 1 -> 2: the viewer binding was called activeOutput
{
  const d = migrateScene({ nodes:[], activeOutput:7 });
  assert.equal(d.activeView, 7, 'the binding carries over');
  assert.equal(d.activeOutput, undefined, 'activeOutput is gone');
  assert.equal(d.format, SCENE_FORMAT);
  console.log('ok  1 -> 2: activeOutput becomes activeView');
}

// 2 -> 3: a chart in the readout chain moves after the checkpoint
{
  // probe -> analysis -> chart -> checkpoint, the shape every scene had
  const scene = { format:2, nodes:[
    { id:1, type:'probe', x:0, y:0, params:{}, inputs:[{ id:0 }] },
    { id:2, type:'analysis', x:0, y:0, params:{}, inputs:[{ id:1 }] },
    { id:3, type:'chart', x:0, y:0, params:{}, inputs:[{ id:2 }] },
    { id:4, type:'output', x:100, y:200, params:{}, inputs:[{ id:3 }] },
  ] };
  const d = migrateScene(scene);
  const byId = id => d.nodes.find(n => n.id === id);
  assert.deepEqual(byId(4).inputs, [{ id:2 }], 'the checkpoint reads the analysis directly');
  assert.deepEqual(byId(3).inputs, [{ id:4 }], 'the chart reads the checkpoint');
  assert.equal(byId(3).x, 320, 'and sits to its right');
  assert.equal(d.format, SCENE_FORMAT);
  console.log('ok  2 -> 3: the chart moves after the checkpoint');
}

// two charts in a row, and a chart that fed nothing
{
  const scene = { format:2, nodes:[
    { id:1, type:'analysis', x:0, y:0, params:{}, inputs:[null] },
    { id:2, type:'chart', x:0, y:0, params:{}, inputs:[{ id:1 }] },
    { id:3, type:'chart', x:0, y:0, params:{}, inputs:[{ id:2 }] },
    { id:4, type:'output', x:0, y:0, params:{}, inputs:[{ id:3 }] },
  ] };
  const d = migrateScene(scene);
  const byId = id => d.nodes.find(n => n.id === id);
  assert.deepEqual(byId(4).inputs, [{ id:1 }], 'the checkpoint ends up on the analysis');
  assert.deepEqual(byId(2).inputs, [{ id:4 }]);
  assert.deepEqual(byId(3).inputs, [{ id:4 }], 'both charts read the checkpoint');
  console.log('ok  2 -> 3: a chain of charts unwinds onto the checkpoint');
}

// a scene with no checkpoint is left alone rather than half rewired
{
  const scene = { format:2, nodes:[
    { id:1, type:'probe', x:0, y:0, params:{}, inputs:[null] },
    { id:2, type:'chart', x:0, y:0, params:{}, inputs:[{ id:1 }] },
  ] };
  const d = migrateScene(scene);
  assert.deepEqual(d.nodes.find(n => n.id === 2).inputs, [{ id:1 }], 'nothing moves');
  assert.equal(d.format, SCENE_FORMAT);
  console.log('ok  2 -> 3: a scene without a checkpoint is untouched');
}

// 4 -> 5: the input node keeps only its mapping.
// This is the shape of a scene with two inputs on one curriculum with different senses, one of them with its side ports the other way round, and two inputs driven by the built-in sources with nothing on the signal port.
{
  const scene = { format:4, nextId:10, nodes:[
    { id:1, type:'curriculum', name:'curriculum', x:0, y:0, params:{ set:0, seed:1 }, inputs:[] },
    { id:2, type:'box', x:0, y:0, params:{}, inputs:[] },
    { id:3, type:'input', name:'input', x:400, y:100,
      params:{ source:0, sense:0, map:0, period:1500 }, inputs:[null, { id:1 }, { id:2 }] },
    { id:4, type:'input', name:'input2', x:600, y:100,
      params:{ source:0, sense:1, map:2 }, inputs:[{ id:3 }, { id:2 }, { id:1 }] },
    { id:5, type:'input', name:'input3', x:800, y:100,
      params:{ source:2, sense:0, map:0 }, inputs:[{ id:4 }, { id:2 }, null] },
    { id:6, type:'input', name:'input4', x:900, y:100,
      params:{ source:0, sense:0, map:0, period:900 }, inputs:[{ id:5 }, null, null] },
  ] };
  const d = migrateScene(scene);
  const byId = id => d.nodes.find(n => n.id === id);
  for(const id of [3, 4, 5, 6]){
    const p = byId(id).params;
    assert.equal(p.source, undefined); assert.equal(p.sense, undefined);
    assert.equal(p.period, undefined);
  }
  assert.deepEqual(byId(3).inputs, [null, { id:2 }, { id:1 }], 'mask on port 1, signal on port 2');
  assert.equal(byId(1).params.sense, 0, 'the curriculum takes the sense of the first input');
  const follower = byId(byId(4).inputs[2].id);
  assert.equal(follower.type, 'curriculum');
  assert.equal(follower.params.sense, 1, 'the sound input gets a sound curriculum');
  assert.deepEqual(follower.inputs, [{ id:1 }], 'which follows the first');
  assert.equal(follower.params.set, 0, 'with the same lesson');
  const cam = byId(byId(5).inputs[2].id);
  assert.equal(cam.type, 'live'); assert.equal(cam.params.device, 1, 'webcam becomes a live node');
  const bar = byId(byId(6).inputs[2].id);
  assert.equal(bar.type, 'testsignal'); assert.equal(bar.params.period, 900, 'bar sweep keeps its period');
  assert.deepEqual(byId(1).inputs, [null], 'the curriculum gains its side port');
  assert.equal(d.nextId, 13);
  const names = d.nodes.map(n => n.name).filter(Boolean);
  assert.equal(new Set(names).size, names.length, 'new nodes take unique names');
  console.log('ok  4 -> 5: sources and senses leave the input node');
}

// 4 -> 5 also: a population tag is one token.
// A scene can have scatters tagged with spaces and a pair table the chips have shredded into three-word populations.
{
  const scene = { format:4, nextId:10, nodes:[
    { id:1, type:'scatter', name:'scatter', params:{ tag:'RS exc 1' }, inputs:[] },
    { id:2, type:'scatter', name:'scatter2', params:{ tag:'Bulb CH exc' }, inputs:[] },
    { id:3, type:'scatter', name:'Input 1', params:{ tag:'Input 1' }, inputs:[] },
    { id:4, type:'connect', params:{ table:[
      'Input 1 RS exc 1 RS exc 1', 'Bulb CH exc Bulb CH', 'Bulb CH exc Bulb CH',
      'Bulb CH exc RS exc 1 0.6 1.3 # hi', 'RS exc 1 RS exc 1 1 1'].join(String.fromCharCode(10)) },
      inputs:[] },
    { id:5, type:'probe', params:{ tag:'rs exc 1' }, inputs:[] },
    { id:6, type:'plasticity', params:{ from:'Bulb CH exc', to:'RS exc 1' }, inputs:[] },
  ] };
  const d = migrateScene(scene);
  const byId = id => d.nodes.find(n => n.id === id);
  assert.equal(byId(1).params.tag, 'RSexc1'); assert.equal(byId(2).params.tag, 'BulbCHexc');
  assert.equal(byId(5).params.tag, 'rsexc1', 'a reference becomes a token; computation ignores case');
  assert.deepEqual([byId(6).params.from, byId(6).params.to], ['BulbCHexc', 'RSexc1']);
  const rows = byId(4).params.table.split(String.fromCharCode(10));
  assert.ok(rows.includes('BulbCHexc RSexc1 0.6 1.3 # hi'), 'a spaced row is read back whole');
  assert.ok(rows.includes('RSexc1 RSexc1 1 1'));
  assert.ok(!rows.some(r => /Input 1/.test(r)), 'a row with no number in it was never a row');
  assert.ok(!rows.includes('Bulb CH exc Bulb CH'), 'nor its duplicate');
  console.log('ok  4 -> 5: tags lose their spaces and the table follows');
}

// 5 -> 6: a dispersed projection keeps the synapse count it had
{
  const scene = { format:5, nextId:10, nodes:[
    { id:1, type:'scatter', params:{ tag:'RSexc1', count:31160 }, inputs:[] },
    { id:2, type:'project', params:{ from:'Input1', to:'RSexc1', dispersed:1, prob:0.35 }, inputs:[] },
    { id:3, type:'project', params:{ from:'Input1', to:'RSexc1', dispersed:0, prob:0.35 }, inputs:[] },
  ] };
  const d = migrateScene(scene);
  const byId = id => d.nodes.find(n => n.id === id);
  assert.equal(byId(2).params.fanout, 10906, 'probability times the target count');
  assert.equal(byId(3).params.fanout, 50, 'a topographic one takes the default');
  console.log('ok  5 -> 6: a dispersed projection is counted per source');
}

// a file from the future is refused rather than silently mangled
{
  assert.throws(() => migrateScene({ format:SCENE_FORMAT + 1, nodes:[] }), /newer than this build/);
  console.log('ok  a newer format is refused');
}
{
  // 10 -> 11: retype folded into cell type
  const d = migrateScene({ format:10, nodes:[
    { id:1, type:'ntype', params:{ type:4, tag:'inh', frac:0.3 } },
    { id:2, type:'celltype', params:{ name:'kc', C:8 } },
    { id:3, type:'checkpoint', params:{ variants:[{ name:'v', edits:[{ node:1, key:'type', value:3 }, { node:2, key:'C', value:9 }] }] } } ] });
  assert.equal(d.nodes[0].type, 'celltype', 'a retype node becomes a cell type node');
  assert.deepEqual(d.nodes[0].params, { row:5, tag:'inh', frac:0.3 }, 'on the same row (index plus one), with its population and fraction');
  assert.equal(d.nodes[0].id, 1, 'and the same id, which seeds the draw');
  assert.equal(d.nodes[1].params.row, 0, 'a cell type node from before is custom');
  assert.equal(d.nodes[1].params.C, 8, 'with its numbers');
  assert.deepEqual(d.nodes[2].params.variants[0].edits[0], { node:1, key:'row', value:4 }, 'a sweep edit of the type follows');
  assert.equal(d.nodes[2].params.variants[0].edits[1].key, 'C', 'and another edit is left alone');
  console.log('ok  10 -> 11: retype is a cell type node on a built-in row');
}
{
  // 11 -> 12: a scene lists the node modules its nodes come from; an older scene uses none
  const d = migrateScene({ format:11, nodes:[{ id:1, type:'scatter', params:{} }] });
  assert.deepEqual(d.plugins, [], 'a scene from before node modules lists none');
  assert.equal(d.format, SCENE_FORMAT);
  const kept = [{ module:'jitter.js', nodes:['jitter'] }];
  assert.deepEqual(migrateScene({ format:SCENE_FORMAT, plugins:kept, nodes:[] }).plugins, kept, 'a current scene keeps its list');
  assert.deepEqual(migrateScene({ nodes:[], activeOutput:2 }).plugins, [], 'a scene from format 1 comes through every step listing none');
  console.log('ok  11 -> 12: a scene lists its node modules');
}
{
  // 12 -> 13: the wiring node is the connections file node
  const d = migrateScene({ format:12, plugins:[], nodes:[
    { id:1, type:'pointfile', params:{ file:'points.csv', id:'id' } },
    { id:2, type:'wiring', name:'fly table', params:{ file:'connections.csv', weight:2, signMode:1 }, inputs:[{ id:1 }] },
    { id:3, type:'connect', params:{}, inputs:[{ id:2 }] } ] });
  assert.equal(d.nodes[1].type, 'connectionsfile', 'a wiring node becomes a connections file node');
  assert.deepEqual(d.nodes[1].params, { file:'connections.csv', weight:2, signMode:1 }, 'with its settings');
  assert.equal(d.nodes[1].name, 'fly table', 'its name');
  assert.deepEqual(d.nodes[1].inputs, [{ id:1 }], 'and its wires');
  assert.deepEqual(d.nodes.map(n => n.type), ['pointfile', 'connectionsfile', 'connect'], 'and nothing else changes type');
  assert.equal(d.format, SCENE_FORMAT);
  console.log('ok  12 -> 13: the wiring node is the connections file node');
}
console.log('migrate: all checks passed');


// 6 -> 7: the sweep node's tabs move onto the checkpoint, its text lines become tabs with their selectors resolved, the node is unwired and gone
{
  const scene = { format:6, nodes:[
    { id:1, type:'connect', name:'connect', x:0, y:0, params:{ prob:0.6, seed:1 }, inputs:[null] },
    { id:2, type:'analysis', name:'analysis', x:0, y:0, params:{}, inputs:[{ id:1 }] },
    { id:3, type:'sweep', name:'sweep', x:0, y:0, params:{ variants:[{ name:'wide', edits:[{ node:1, key:'prob', value:0.9 }] }] }, inputs:[{ id:2 }] },
    { id:4, type:'output', name:'checkpoint', x:0, y:0, params:{ trVariants:'connect.seed=5; #analysis.x=1' }, inputs:[{ id:3 }] },
  ], groups:[{ label:'readout', members:[2, 3, 4] }] };
  const d = migrateScene(scene);
  const byId = id => d.nodes.find(n => n.id === id);
  assert.equal(byId(3), undefined, 'the sweep node is gone');
  assert.deepEqual(byId(4).inputs, [{ id:2 }], 'the checkpoint reads what the sweep node read');
  const v = byId(4).params.variants;
  assert.equal(v.length, 2, 'text line and node tab both became tabs');
  assert.deepEqual(v[0].edits, [{ node:1, key:'seed', value:5 }], 'the text line resolved by selector (the analysis has no x, so that part is dropped)');
  assert.deepEqual(v[1], { name:'wide', edits:[{ node:1, key:'prob', value:0.9 }] }, 'the node tab carried over');
  assert.equal(byId(4).params.trVariants, undefined, 'the text field is gone');
  assert.deepEqual(d.groups[0].members, [2, 4], 'the group no longer lists it');
  assert.equal(d.format, SCENE_FORMAT);
  console.log('ok  6 -> 7: the sweep is a tab on the checkpoint');
}

// 7 -> 8: the 2003 membrane form is retired; the 2003 presets are rows 0 to 6 of the 2007 form, the measured 2007 rows moved to 10 to 16
{
  const d = migrateScene({ format:7, form:0, nodes:[
    { id:1, type:'scatter', params:{ type:3 } }, { id:2, type:'celltype', params:{ base2003:2, C:20 } } ] });
  assert.equal(d.form, undefined, 'the form field is gone');
  assert.equal(d.nodes[0].params.type, 3, 'a 2003 scene keeps its type indices');
  assert.equal(d.nodes[1].params.base2003, undefined, 'the cell type node loses its 2003 base row');
  assert.equal(d.nodes[1].params.C, 20, 'and keeps its own numbers');
  const e = migrateScene({ format:7, form:1, nodes:[
    { id:1, type:'scatter', params:{ type:0 } }, { id:2, type:'ntype', params:{ type:6 } },
    { id:3, type:'pointfile', params:{ type:8 } }, { id:4, type:'merge', params:{ type:2 } },
    { id:5, type:'output', params:{ variants:[{ name:'v', edits:[{ node:1, key:'type', value:3 }, { node:4, key:'type', value:3 }] }] } } ] });
  assert.equal(e.form, undefined);
  assert.equal(e.nodes[0].params.type, 10, 'a 2007 scene: RS measured moves to row 10');
  assert.equal(e.nodes[1].type, 'celltype', 'a retype node is a cell type node since format 11');
  assert.equal(e.nodes[1].params.row, 17, 'RZ measured to row 16, which the cell type node counts from one (zero is custom)');
  assert.equal(e.nodes[2].params.type, 8, 'a fly row stays where it is');
  assert.equal(e.nodes[3].params.type, 2, 'a type field on a node that has no type table is untouched');
  assert.equal(e.nodes[4].params.variants[0].edits[0].value, 13, 'a sweep edit of a typed node moves too');
  assert.equal(e.nodes[4].params.variants[0].edits[1].value, 3, 'and one on another node does not');
  const f = migrateScene({ format:7, nodes:[{ id:1, type:'scatter', params:{ type:5 } }] });
  assert.equal(f.nodes[0].params.type, 5, 'no form field means the 2003 form: indices kept');
  assert.equal(f.format, SCENE_FORMAT);
  console.log('ok  7 -> 8: the 2003 form is retired, its presets are rows of the 2007 form');
}

// 8 -> 9: short-term plasticity releases before the facilitation jump by default; a scene that used the term keeps the order it was tuned with
{
  const d = migrateScene({ format:8, nodes:[
    { id:1, type:'output', params:{ stp:1, stpNorm:1 } } ] });
  assert.equal(d.nodes[0].params.stpOrder, 1, 'a checkpoint with the term on keeps facilitation first');
  assert.equal(d.nodes[0].params.stpNorm, 1, 'and its delivery setting');
  const e = migrateScene({ format:8, nodes:[
    { id:1, type:'output', params:{ stp:0 } }, { id:2, type:'output', params:{ stp:0,
      variants:[{ name:'on', edits:[{ node:2, key:'stp', value:1 }] }] } },
    { id:3, type:'output', params:{ stp:1, stpOrder:0 } } ] });
  assert.equal(e.nodes[0].params.stpOrder, undefined, 'a checkpoint with the term off takes the published order');
  assert.equal(e.nodes[1].params.stpOrder, 1, 'a sweep tab that turns the term on counts');
  assert.equal(e.nodes[2].params.stpOrder, 0, 'an order already set is kept');
  assert.equal(e.format, SCENE_FORMAT);
  console.log('ok  8 -> 9: scenes using short-term plasticity keep their release order');
}

// Every node of every scenario has a name that says what it is for.
//
// A scene is opened the way the page opens it (the editor hands every new node its default name, scatter, scatter2), and after the build no node but a pin or a note may still be wearing that default.
// A new scenario fails here until its nodes are named, by its builder or by nodenames.js.

import { ok, report } from '../tools/harness.mjs';
import { NODE_DEFS } from './nodes.js';
import { ALL_SCENARIOS } from './experiments.js';
import { nameNodes, isDefaultName } from './nodenames.js';
import { defaultNodeName, tagToken } from './tagname.js';

function mkEditor(){
  return { nodes:[], nextId:1, groups:[],
    addNode(t, x, y){ const def = NODE_DEFS[t], p = {}; def.params.forEach(q => p[q.k] = structuredClone(q.def));
      const n = { id:this.nextId++, type:t, x, y, name:defaultNodeName(def.stem || def.title, this.nodes.map(m => m.name)), params:p, inputs:new Array(def.inputs).fill(null) };
      this.nodes.push(n); return n; } };
}
for(const s of ALL_SCENARIOS){
  const ed = mkEditor(); s.build(ed);
  const nodes = ed.nodes.filter(n => n.type !== 'pin' && n.type !== 'note');
  const bare = nodes.filter(isDefaultName);
  ok(s.name + ': every node is named', bare.length === 0, bare.slice(0, 5).map(n => n.type + ' "' + n.name + '"').join(', '));
  const seen = new Map(); for(const n of nodes) seen.set(n.name.toLowerCase(), (seen.get(n.name.toLowerCase()) || 0) + 1);
  const twice = [...seen].filter(([, c]) => c > 1).map(([k]) => k);
  ok(s.name + ': no two nodes share a name', twice.length === 0, twice.slice(0, 5).join(', '));
  const off = nodes.filter(n => n.type === 'scatter' && n.params.tag && tagToken(n.name) !== n.params.tag);
  ok(s.name + ': a neuron scatter is named by its population, so the tag follows the name', off.length === 0, off.slice(0, 3).map(n => n.name + ' / ' + n.params.tag).join(', '));
  const long = nodes.filter(n => n.name.length > 40);
  ok(s.name + ': names stay short enough to read beside a node', long.length === 0, long.slice(0, 3).map(n => n.name).join(' | '));
}
{
  // a name someone gave is never replaced; a default one is
  const ed = mkEditor();
  const sp = ed.addNode('sphere', 0, 0), sc = ed.addNode('scatter', 0, 60), st = ed.addNode('stimulus', 0, 120), pin = ed.addNode('pin', 0, 180);
  sc.inputs[0] = { id:sp.id }; sc.params.tag = 'relay'; st.inputs[0] = { id:sc.id }; st.inputs[1] = { id:sp.id }; st.params.mode = 3;
  st.name = 'my own drive';
  nameNodes(ed.nodes, []);
  ok('a region is named after what it holds', sp.name === 'relay sphere', sp.name);
  ok('a scatter takes its population as its name', sc.name === 'relay');
  ok('a name someone gave is kept', st.name === 'my own drive');
  ok('a pin keeps its default', pin.name === 'pin');
  st.name = 'stimulus2'; nameNodes(ed.nodes, []);
  ok('a stimulus is named by its mode and where it lands', st.name === 'noise on relay', st.name);
  ok('an old default from a renamed node type counts as unnamed', isDefaultName({ type:'gather', name:'merge3' }) && isDefaultName({ type:'pin', name:'dot' }) && !isDefaultName({ type:'gather', name:'all cells' }));
}
report('nodenames');

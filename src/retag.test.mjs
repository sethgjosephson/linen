// Renaming a population renames every reference to it: each setting of the population kind on any node, projection and plasticity scopes, a chart's population and pair-table rows, and nothing that defines a population of its own (cell type, population and pool nodes).
import { retagReferences } from './props.js';
import { ok, report } from '../tools/harness.mjs';

const nodes = [
  { id:1, type:'scatter', params:{ tag:'exc' } },          // a second population of the same name
  { id:2, type:'celltype', params:{ tag:'Exc', row:1 } },
  { id:3, type:'population', params:{ tag:'exc' } },
  { id:4, type:'pool', params:{ tag:'exc' } },
  { id:5, type:'probe', params:{ tag:'exc' } },
  { id:6, type:'project', params:{ from:'exc', to:'inh' } },
  { id:7, type:'plasticity', params:{ from:'inh', to:'exc' } },
  { id:8, type:'chart', params:{ pop:'exc' } },
  { id:9, type:'connect', params:{ table:'exc inh 0.1\ninh exc 0.2 # keep' } },
  { id:10, type:'probe', params:{ tag:'inh' } },
];
const edited = new Set();
retagReferences(nodes, 'exc', 'pyr', n => edited.add(n.id));
const at = id => nodes.find(n => n.id === id).params;
ok('cell type, population and pool follow the rename',
  at(2).tag === 'pyr' && at(3).tag === 'pyr' && at(4).tag === 'pyr', JSON.stringify([at(2).tag, at(3).tag, at(4).tag]));
ok('and the references that already did still do',
  at(5).tag === 'pyr' && at(6).from === 'pyr' && at(6).to === 'inh' && at(7).to === 'pyr' && at(8).pop === 'pyr' &&
  at(9).table === 'pyr inh 0.1\ninh pyr 0.2 # keep', JSON.stringify([at(5), at(6), at(7), at(8), at(9)]));
ok('another population tagged the same is not renamed', at(1).tag === 'exc');
ok('a reference to another population is not touched', at(10).tag === 'inh' && !edited.has(10));
ok('each changed node is edited once', [2, 3, 4, 5, 6, 7, 8, 9].every(id => edited.has(id)) && edited.size === 8,
  [...edited].join(','));
report('retag');

// The checkpoint's sweep tab against every scenario: a sweep with one edit on every node that has an editable setting expands onto a copy of the graph with every edit applied and the graph on screen untouched; text variants and address-bar overrides go through the same selectors.
import { NODE_DEFS } from './nodes.js';
import { ALL_SCENARIOS as SCENARIOS } from './experiments.js';
import { applyVariant, applyOverrides, gatherVariants } from './sweepvariant.js';
import { ok, report } from '../tools/harness.mjs';

const { NodeEditor } = await import('./editor.js');
const build = sc => {
  const ed = Object.create(NodeEditor.prototype);
  Object.assign(ed, { nodes:[], groups:[], nextId:1, cb:{ onSelect(){}, onChange(){} },
    viewSlots:{}, activeView:null, suggestResolution:0 });
  sc.build(ed);
  return ed;
};
const settable = n => NODE_DEFS[n.type].params.filter(q =>
  ['float', 'int', 'select'].includes(q.t) && !q.identity && !(n.type === 'checkpoint' && /^tr[A-Z]/.test(q.k)));
const moved = (q, cur) => q.t === 'select' ? ((cur|0) + 1) % q.options.length
  : (+cur === 0 ? 1 : +cur * 2);

for(const sc of SCENARIOS){
  const ed = build(sc);
  const out = ed.nodes.find(n => n.type === 'checkpoint');
  ok(`${sc.name}: has a checkpoint`, !!out);
  if(!out) continue;
  // the checkpoint's sweep tab, with one edit per editable node
  const edits = [];
  for(const n of ed.nodes){
    if(n.type === 'note' || n.type === 'pin') continue;
    const q = settable(n)[0]; if(!q) continue;
    edits.push({ node:n.id, key:q.k, value:moved(q, n.params[q.k]) });
  }
  out.params.variants = [{ name:'all', edits }, { name:'none', edits:[] }];
  ok(`${sc.name}: a tab with no edits alone is no sweep`,
     gatherVariants({ params:{ variants:[{ name:'sweep 1', edits:[] }] } }, ed.nodes).length === 0);
  const before = JSON.stringify(ed.nodes.map(n => n.params));
  const variants = gatherVariants(out, ed.nodes);
  ok(`${sc.name}: the checkpoint gathers the sweep`, variants.length === 2 && variants[0].name === 'all');
  const g = JSON.parse(JSON.stringify({ nodes:ed.nodes.map(n => ({ id:n.id, type:n.type, name:n.name, params:n.params })) }));
  const warns = [];
  const applied = applyVariant(variants[0], g.nodes, w => warns.push(w));
  ok(`${sc.name}: every edit lands on the copy`, applied.length === edits.length && !warns.length,
     applied.length + ' of ' + edits.length + (warns.length ? ' ' + warns[0] : ''));
  ok(`${sc.name}: every edited value is on the copy`, edits.every(e => {
    const n = g.nodes.find(x => x.id === e.node); return n.params[e.key] === e.value; }));
  ok(`${sc.name}: the graph on screen is untouched`, JSON.stringify(ed.nodes.map(n => n.params)) === before);
  // the same edits as text, by name and by type
  const g2 = JSON.parse(JSON.stringify(g));
  const cn = ed.nodes.find(n => n.type === 'connect');
  if(cn){
    const a = applyOverrides(`${cn.name}.prob=0.123; #connect.seed=77`, g2.nodes, () => {}, /;/, w => warns.push(w));
    ok(`${sc.name}: text variants by name and by type`, a.length >= 2 &&
       g2.nodes.find(x => x.id === cn.id).params.prob === 0.123, a.join(' '));
  }
}
report('sweep');

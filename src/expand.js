// Expanding a points file tagged by column into one node per population: a row of population nodes below it, each reading the file node and keeping one tag's cells, wired into a merge that takes the file node's place in whatever read it.
// The file node stays as the origin, identity is untouched (a population node keeps the cells' src and lidx), and every population is a node of its own to view or edit.
// Collapsing removes a fan that is exactly that and nothing more.
export function expandPopulations(editor, node, tags){
  const gap = 150, y0 = node.y + 120;
  const x0 = node.x - (tags.length - 1)*gap/2;
  const children = tags.map((t, k) => {
    const c = editor.addNode('population', Math.round(x0 + k*gap), y0);
    c.name = String(t).replace(/^.*\./, '');
    c.params.tag = String(t);
    c.inputs[0] = { id:node.id };
    return c;
  });
  const m = editor.addNode('gather', node.x, y0 + 120);
  m.name = (node.name || 'populations') + ' merged';
  m.params.ports = Math.max(2, tags.length);
  m.inputs = children.map(c => ({ id:c.id }));
  for(const n of editor.nodes)
    if(n !== m && !children.includes(n) && Array.isArray(n.inputs))
      n.inputs = n.inputs.map(c => c && c.id === node.id ? { id:m.id } : c);
  const group = editor.createGroup ? editor.createGroup([...children, m], (node.name || 'populations') + ' populations') : null;
  return { children, merge:m, group };
}

// The fan a file node feeds: its population children, and the one merge that reads exactly them, if the fan is untouched.
// Null otherwise, with the reason, so a fan someone has built on is never collapsed by mistake.
export function fanOf(editor, node){
  const kids = editor.nodes.filter(n => n.type === 'population' && n.inputs[0] && n.inputs[0].id === node.id);
  if(!kids.length) return { kids:[], why:'no population nodes read this one' };
  const readers = new Set();
  for(const n of editor.nodes) for(const c of n.inputs || []) if(c && kids.some(k => k.id === c.id)) readers.add(n);
  if(readers.size !== 1) return { kids, why:'the populations feed ' + readers.size + ' nodes, not one merge' };
  const m = [...readers][0];
  if(m.type !== 'gather') return { kids, why:'the populations feed a ' + m.type + ', not a merge' };
  const ins = m.inputs.filter(Boolean).map(c => c.id).sort().join(',');
  if(ins !== kids.map(k => k.id).sort().join(',')) return { kids, why:'the merge reads more than these populations' };
  return { kids, merge:m };
}
export function collapsePopulations(editor, node){
  const fan = fanOf(editor, node);
  if(!fan.merge) return fan;
  const gone = new Set([...fan.kids.map(k => k.id), fan.merge.id]);
  for(const n of editor.nodes)
    if(!gone.has(n.id) && Array.isArray(n.inputs))
      n.inputs = n.inputs.map(c => c && c.id === fan.merge.id ? { id:node.id } : c);
  for(const g of editor.groups || []) if(Array.isArray(g.members)) g.members = g.members.filter(id => !gone.has(id));
  if(editor.groups) editor.groups = editor.groups.filter(g => !Array.isArray(g.members) || g.members.length);   // the fan's own backdrop goes with it
  editor.nodes = editor.nodes.filter(n => !gone.has(n.id));
  if(editor.sel && gone.has(editor.sel.id)) editor.sel = null;
  if(editor.selSet) for(const n of [...editor.selSet]) if(gone.has(n.id)) editor.selSet.delete(n);
  return fan;
}

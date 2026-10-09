// The two shapes a sweep variant takes and how each is applied to a copy of the graph: a line of text by selector (the checkpoint's variants box, the address bar), or a sweep node's rows by node id.
// Pure, so the expansion is testable outside the page.

// Edits by selector on a node list (editor nodes or a graph's JSON, the same shape: type, name, params).
// A selector is a node name, a name prefix ending in *, #type for every node of a type, or @tag for the scatter carrying that population tag.
// Returns what was applied; a problem is reported through `warn` rather than thrown.
export function applyOverrides(spec, nodes, onEdit = () => {}, sep = /[;,]/, warn = () => {}){
  const applied = [];
  for(const item of String(spec || '').split(sep).map(x => x.trim()).filter(Boolean)){
    const m = /^([^.=]+)\.([A-Za-z0-9_]+)=(.*)$/.exec(item);
    if(!m){ warn('override not understood: ' + item); continue; }
    const [, sel, key, raw] = m;
    const val = raw === '' ? '' : (isNaN(+raw) ? raw : +raw);
    const hit = nodes.filter(n => sel.startsWith('#') ? n.type === sel.slice(1)
      : sel.startsWith('@') ? String((n.params || {}).tag || '').toLowerCase() === sel.slice(1).toLowerCase()
      : sel.endsWith('*') ? String(n.name || '').toLowerCase().startsWith(sel.slice(0, -1).toLowerCase())
      : String(n.name || '').toLowerCase() === sel.toLowerCase());
    if(!hit.length){ warn('override matched no node: ' + item); continue; }
    for(const n of hit){
      if(!(key in n.params)){ warn(`override: ${n.type} ${n.name} has no ${key}`); continue; }
      n.params[key] = val; onEdit(n); applied.push(`${n.name}.${key}=${val}`);
    }
  }
  return applied;
}

// One variant onto a graph copy's nodes.
// A variant is a text line or a sweep node's { name, edits:[{ node, key, value }] }.
export function applyVariant(variant, nodes, warn = () => {}){
  if(!variant) return [];
  if(typeof variant === 'object'){
    const applied = [];
    for(const e of variant.edits || []){
      const n = nodes.find(x => x.id === e.node);
      if(!n){ warn(`sweep ${variant.name}: node #${e.node} is gone`); continue; }
      if(!(e.key in n.params)){ warn(`sweep ${variant.name}: ${n.type} ${n.name} has no ${e.key}`); continue; }
      n.params[e.key] = e.value; applied.push(`${n.name || n.type}.${e.key}=${e.value}`);
    }
    return applied;
  }
  return applyOverrides(variant, nodes, () => {}, /[;,]/, warn);
}

// The variants a checkpoint runs: the tabs on its sweep tab.
// A variant with no edits is the graph as it stands, which is a control when it sits beside variants that edit something and nothing at all on its own, so a list in which no variant edits anything gathers as no sweep.
export function gatherVariants(checkpoint, nodes){
  const list = (checkpoint.params || {}).variants;
  if(!Array.isArray(list) || !list.some(v => (v.edits || []).length)) return [];
  return list.map(v => ({ name:v.name || 'sweep', edits:(v.edits || []).map(e => ({ ...e })) }));
}

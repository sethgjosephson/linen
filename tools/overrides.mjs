// Settings on nodes from a command line, for every headless tool: terms of the form type[@k]:who:key=value, separated by semicolons. who is a tag, a node's name, a probe's label, or * for every node of the type; @k picks the k-th node of the type in build order.
// A value in brackets is JSON, a number is a number, anything else a string, and an empty value stays empty.
// A key the node does not have, or a term that matches no node, is an error: a run that quietly went ahead on the unmodified scene is the failure this exists to prevent.

export function splitTerms(specs){
  return (Array.isArray(specs) ? specs : [specs])
    .flatMap(s => String(s || '').split(';')).map(s => s.trim()).filter(Boolean);
}

export function parseValue(raw){
  if(raw === '') return '';
  if(/^\[.*\]$/.test(raw)) return JSON.parse(raw);
  return isNaN(+raw) ? raw : +raw;
}

export function applyOverrides(nodes, terms){
  const applied = [];
  for(const term of terms){
    const m = /^([a-z]+)(?:@(\d+))?:([^:]*):([A-Za-z0-9_]+)=(.*)$/.exec(term);
    if(!m) throw new Error('an override reads type:tag-or-name:key=value, not ' + term);
    const [, type, nth, who, key, raw] = m;
    const val = parseValue(raw);
    let hits = 0, seen = -1;
    for(const n of nodes){
      if(n.type !== type) continue;
      seen++;
      if(nth !== undefined && seen !== +nth) continue;
      if(who !== '*' && String(n.params.tag || '') !== who && String(n.name || '') !== who &&
        String(n.params.label || '') !== who) continue;
      if(!(key in n.params)) throw new Error(type + ' has no setting called ' + key);
      n.params[key] = val; hits++;
    }
    if(!hits) throw new Error('no ' + type + ' node matched ' + term);
    applied.push({ term, nodes:hits });
  }
  return applied;
}

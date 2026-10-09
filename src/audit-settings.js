// The settings audit: every setting on every node, does changing it change what is computed.
// Runs in the page (the wiring needs its workers) against the graph on screen: for each node and each numeric, select or population setting, the value is moved, the checkpoint is computed again, and a signature of the result is compared with the baseline.
// A setting whose change leaves the signature identical is reported, since either it reaches nothing or it reaches something the signature does not see; both are worth a look.
// The graph is restored after every setting.
//
// Start it with ?audit=settings (with ?scenario= to pick the graph), or call auditSettings(editor) from the console; results go to the console, the status line and runs/audit-settings-<stamp>.json.

import { NODE_DEFS, computeNode, engineConfig } from './nodes.js';

function hashFloats(arr, cap = 200000){
  if(!arr || !arr.length) return 0;
  const n = Math.min(arr.length, cap), step = Math.max(1, Math.floor(arr.length / n));
  let h = 0, s = 0;
  for(let i = 0; i < arr.length; i += step){ s += arr[i]; h = (h*31 + Math.round(arr[i]*1000)) | 0; }
  return h + ':' + s.toFixed(3) + ':' + arr.length;
}

// What a computed network is, for the purpose of telling two apart.
export function netSignature(net){
  if(!net) return 'null';
  if(net.kind !== 'net'){
    return net.kind + ':' + (net.count || 0) + ':' + hashFloats(net.pos) +
      ':' + JSON.stringify(net.tags || {}) + ':' + hashFloats(net.ntype) + ':' + hashFloats(net.bias);
  }
  const cfg = engineConfig(net);
  const maps = (net.inputMaps || []).map(m => ({
    id:m.id, source:m.source, cols:m.cols, rows:m.rows, sheet:m.sheet, sense:m.sense, tag:m.tag,
    code:m.code, transient:m.transient, jitter:m.jitter, lagMs:m.lagMs, period:m.period, amp:m.amp,
    ch:hashFloats(m.chIdx), w:hashFloats(m.chW), signal:m.signal }));
  const probes = (net.probes || []).map(p => ({ ...p, idx:hashFloats(p.idx) }));
  return JSON.stringify({
    count:net.count, syn:net.synCount, pos:hashFloats(net.pos), ntype:hashFloats(net.ntype),
    bias:hashFloats(net.bias), w:hashFloats(net.w), delay:hashFloats(net.delay),
    post:hashFloats(net.post), pmask:hashFloats(net.pmask), tags:net.tags,
    cfg:{ ...cfg, ruleTable:cfg.ruleTable ? hashFloats(cfg.ruleTable) : null,
      pool:hashFloats(cfg.pool), poolK:hashFloats(cfg.poolK) },
    steps:net.steps, engine:net.engine, seed:net.seed, maps, probes,
    analysis:net.analysis || null, chart:net.chart || null,
    inputs:(net.stims || net.drives || []).length,
    extra:Object.keys(net).filter(k => !['pos','ntype','bias','w','delay','post','preStart','pmask',
      'inputMaps','probes','tags','lidx','src'].includes(k)).sort().join(','),
  });
}

// A different value for a setting, or null when the setting is not the kind this audit moves (vec3, text, actions, tables).
function moved(p, cur){
  if(p.t === 'select') return (((cur|0) + 1) % (p.options.length || 1));
  if(p.t === 'int' || p.t === 'float'){
    let v = +cur || 0;
    let w = v === 0 ? 1 : v * 1.5;
    if(p.t === 'int') w = Math.round(w) === v ? v + 1 : Math.round(w);
    if(p.min !== undefined && w < p.min) w = p.min;
    if(p.max !== undefined && w > p.max) w = p.max;
    if(w === v) w = p.min !== undefined && v > p.min ? p.min : v - 1;
    return w;
  }
  if(p.t === 'tag') return cur ? '' : 'E';
  return null;
}

export async function auditSettings(editor, opts = {}){
  const log = opts.log || (s => console.log(s));
  const out = editor.nodes.find(n => n.type === 'checkpoint');
  if(!out) throw new Error('no checkpoint to compute');
  const byId = id => editor.nodes.find(n => n.id === id);
  const compute = async () => {
    for(const n of editor.nodes) n._cache = null;
    return computeNode(out, byId);
  };
  const upstream = new Set();
  const walk = n => { if(!n || upstream.has(n.id)) return; upstream.add(n.id);
    for(const c of n.inputs || []) if(c) walk(byId(c.id)); };
  walk(out);
  const base = netSignature(await compute());
  const rows = [];
  let checked = 0;
  for(const node of editor.nodes){
    if(!upstream.has(node.id)) continue;
    const def = NODE_DEFS[node.type];
    for(const p of def.params){
      if(p.cond && !p.cond(node.params, node, editor.nodes)) continue;  // hidden: audited when its parent is on
      // run settings live on the checkpoint but are read by the trainer, not the computation, so the computation cannot see them; they are audited by the sweep
      if(node.type === 'checkpoint' && /^tr[A-Z]/.test(p.k)) continue;
      const cur = node.params[p.k];
      const to = moved(p, cur);
      if(to === null || to === cur) continue;
      checked++;
      node.params[p.k] = to;
      let sig = null, err = null;
      try { sig = netSignature(await compute()); }
      catch(e){ err = String(e.message || e); }
      let same = sig === base, far = null;
      // a small move can land inside a quantum (a delay that still rounds to one millisecond, a spacing below the jitter); a large one settles it
      if(same && !err && (p.t === 'int' || p.t === 'float')){
        far = p.s ? (Math.abs(cur - p.s[1]) > Math.abs(cur - p.s[0]) ? p.s[1] : p.s[0])
          : (p.max !== undefined && cur !== p.max ? p.max : (+cur || 1) * 8);
        if(far !== cur){
          node.params[p.k] = far;
          try { same = netSignature(await compute()) === base; }
          catch(e){ err = String(e.message || e); }
        }
      }
      node.params[p.k] = cur;
      if(same || err) rows.push({ node:node.type + ' ' + (node.name || '') + ' #' + node.id,
        setting:p.k, label:p.label, from:cur, to:far !== null ? [to, far] : to,
        result:err ? 'error: ' + err : 'no change' });
      log(`${node.type} ${node.name || ''} · ${p.label}: ${cur} -> ${to}${far !== null ? ' and ' + far : ''} · ${err ? 'ERROR ' + err : same ? 'NO CHANGE' : 'changes the result'}`);
    }
  }
  await compute();                            // leave the graph as it was
  return { checked, flagged:rows };
}

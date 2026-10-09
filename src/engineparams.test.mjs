// Every setting on the checkpoint that describes the simulation has to reach the engines.
// This moves each one off its value on a small computed network and requires the engine config (what engineConfig hands every engine, plus the bias the computation sets) to change.
// A setting the computation drops leaves the panel showing a value no engine ever ran.
//
// Settings the page reads rather than the engines are listed with the reason, and a new setting in an engine-facing tab is covered without being named.
import { NODE_DEFS, computeNode, setResolution, engineConfig, wireConnect, assembleConnect } from './nodes.js';
import { ok, report, inProcessWorker } from '../tools/harness.mjs';
inProcessWorker(wireConnect, assembleConnect);

// the checkpoint's tabs whose settings describe the simulation
const ENGINE_TABS = ['engine', 'timing', 'homeostasis', 'competition', 'short-term'];
// read by the page, not the engines
const PAGE_SIDE = {
  steps:'steps per frame: how much the page asks for per tick message',
  engine:'which engine the page starts',
  stpPreset:'a preset: it writes stpU, stpTauD and stpTauF through its apply hook, which are covered',
};
// every term on, so a setting that only matters with its term on is live
const ALL_ON = { plast:1, stp:0.5, syn:2, cons:1, scale:1, het:0.001, trip:0.001, tin:0.001, commit:1, rhoMode:1 };

function graph(edit){
  const nodes = [];
  const add = (type, params) => {
    const def = NODE_DEFS[type], p = {};
    def.params.forEach(q => p[q.k] = structuredClone(q.def));
    Object.assign(p, params);
    const n = { id:nodes.length + 1, type, x:0, y:0, params:p, inputs:new Array(def.inputs).fill(null) };
    nodes.push(n); return n;
  };
  const wire = (to, port, from) => { to.inputs[port] = { id:from.id }; };
  const box = add('box', { center:[0, 0, 0], size:[200, 200, 200] });
  const e = add('scatter', { fill:0, count:80, type:0, tag:'e', seed:1 }); wire(e, 0, box);
  const i = add('scatter', { fill:0, count:20, type:3, tag:'i', seed:2 }); wire(i, 0, box);
  const g = add('gather', {}); wire(g, 0, e); wire(g, 1, i);
  const cn = add('connect', { radius:300, sigma:3000, prob:0.2, seed:1 }); wire(cn, 0, g);
  const cp = add('checkpoint', ALL_ON); wire(cp, 0, cn);
  if(edit) edit(cp.params);
  return { cp, byId:id => nodes.find(n => n.id === id) };
}
const h32 = u8 => { let h = 2166136261 >>> 0; for(let k = 0; k < u8.length; k++){ h ^= u8[k]; h = Math.imul(h, 16777619) >>> 0; } return h; };
const dg = v => v === null || v === undefined ? String(v)
  : ArrayBuffer.isView(v) ? 'A' + v.length + ':' + h32(new Uint8Array(v.buffer, v.byteOffset, v.byteLength))
  : Array.isArray(v) ? '[' + v.map(dg).join(',') + ']'
  : typeof v === 'object' ? '{' + Object.keys(v).sort().map(k => k + ':' + dg(v[k])).join(',') + '}'
  : JSON.stringify(v);
async function config(edit){
  const { cp, byId } = graph(edit);
  const net = await computeNode(cp, byId);
  const c = engineConfig(net);
  const out = { bias:dg(net.bias) };
  for(const k of Object.keys(c)) out[k] = dg(c[k]);
  return out;
}
function moved(p, cur){
  if(p.t === 'select') return (cur | 0) === 0 ? 1 : 0;
  const c = +cur;
  let v = c === 0 ? (p.t === 'int' ? 1 : 0.5) : (p.t === 'int' ? c + 1 : c*1.37 + 0.01);
  if(p.max !== undefined && v > p.max) v = c === 0 ? p.max : c*0.63;
  if(p.min !== undefined && v < p.min) v = p.min;
  return p.t === 'int' ? Math.round(v) : v;
}

setResolution(1);
const base = await config();
const params = NODE_DEFS.checkpoint.params.filter(p => ENGINE_TABS.includes(p.g) && ['int', 'float', 'select'].includes(p.t));
ok('the engine-facing tabs have settings to move', params.length >= 30, params.length + ' settings');
for(const p of params){
  if(PAGE_SIDE[p.k]) continue;
  const start = { ...Object.fromEntries(NODE_DEFS.checkpoint.params.map(q => [q.k, q.def])), ...ALL_ON }[p.k];
  const to = moved(p, start);
  const c = await config(q => { q[p.k] = to; });
  const changed = Object.keys(c).filter(k => c[k] !== base[k]);
  ok('checkpoint ' + p.k + ' reaches the engines (' + JSON.stringify(start) + ' to ' + JSON.stringify(to) + ')',
    changed.length > 0, 'no field of the engine config changed: the setting is dropped somewhere between the node and the engine');
}
// a setting with an engine field of its own arrives as that field, not only through the bias
for(const [k, v] of [['vmin', -75], ['eRevE', 10], ['eRevI', -80]]){
  const c = await config(q => { q[k] = v; });
  ok('checkpoint ' + k + ' arrives as its own value', c[k] === JSON.stringify(v), 'engine config ' + k + ' reads ' + c[k]);
}
// a floor of 0 is off, and stays 0 rather than falling back to the default
{ const c = await config(q => { q.vmin = 0; }); ok('a membrane floor of 0 reaches the engines as 0 (off)', c.vmin === '0', 'reads ' + c.vmin); }
report('engineparams');

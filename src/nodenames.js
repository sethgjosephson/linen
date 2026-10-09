// A name for every node of a scenario, saying what the node is for.
//
// A scenario's builder may name a node itself (the fifth argument of api(ed).add in scenarios.js) and that name stands.
// Every other node is named here from its place in the graph: a neuron scatter by its population, a region by what it holds or masks, a stimulus by its mode and its target, a probe by its label, and so on.
// Pins and notes have nothing to say and are left alone.
// Names are display only: nothing in a computation reads them, and a scatter's name is its tag, so the tag still follows the name when someone renames it (tagname.js).
//
// nameNodes leaves alone any name a person or a builder gave, which is any name that is not the editor's default for that node (scatter, scatter2).

import { NODE_DEFS } from './nodes.js';
import { tagToken } from './tagname.js';

const SKIP = new Set(['pin', 'note']);
const SHAPES = new Set(['sphere', 'box', 'ellipsoid', 'cylinder', 'torus', 'mesh', 'spline', 'noisefield']);
const VERB = { move:'move', noisewarp:'warp', twist:'twist', gaussblur:'blur', cull:'cull', gradient:'gradient', ops:'formula' };
const PLOT = ['measure', 'weights', 'responses', 'similarity', 'PSTH', 'weight blocks', 'confusion', 'rate', 'raster', 'intervals', 'trace', 'recording'];
// older scenes carry the defaults of node keys since renamed
const LEGACY = { pin:['dot'], gather:['merge'], footage:['read'], move:['transform'], checkpoint:['output'], noisefield:['noise'] };

export function isDefaultName(n){
  const name = String(n.name || '').trim();
  if(!name) return true;
  const def = NODE_DEFS[n.type] || {};
  const stems = [def.stem, def.title, n.type, ...(LEGACY[n.type] || [])].filter(Boolean).map(tagToken);
  return stems.some(s => new RegExp('^' + s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\d*$', 'i').test(name));
}

// what several tags have in common, for a region that holds L4e and L4i
function common(tags){
  const t = [...new Set(tags.filter(Boolean))];
  if(!t.length) return '';
  if(t.length === 1) return t[0];
  let p = t[0];
  for(const s of t) while(p && !s.startsWith(p)) p = p.slice(0, -1);
  p = p.replace(/[._\-\/\s]+$/, '');
  if(p.length >= 2) return p;
  return t.length <= 3 ? t.join(' + ') : '';
}

export function nameNodes(nodes, groups){
  const byId = new Map(nodes.map(n => [n.id, n]));
  const up = (n, port = 0) => { const c = n.inputs && n.inputs[port]; return c ? byId.get(c.id) || null : null; };
  const users = new Map();            // node id -> [{ node, port }]
  for(const n of nodes) (n.inputs || []).forEach((c, port) => { if(c && byId.has(c.id)){ if(!users.has(c.id)) users.set(c.id, []); users.get(c.id).push({ node:n, port }); } });
  // the population tags that reach a node through its main inputs
  const memo = new Map();
  const tagsOf = n => {
    if(!n) return [];
    if(memo.has(n.id)) return memo.get(n.id);
    memo.set(n.id, []);
    let out;
    const P = n.params || {};
    if(n.type === 'scatter' || n.type === 'pointfile') out = [String(P.tag || '').trim()].filter(Boolean);
    else if(n.type === 'population') out = [String(P.tag || '').trim()].filter(Boolean);
    else { const def = NODE_DEFS[n.type] || {}; const main = def.portsParam || n.type === 'project' ? (n.inputs || []).map((_, i) => i) : [0];
      out = main.flatMap(i => tagsOf(up(n, i))); }
    out = [...new Set(out)];
    memo.set(n.id, out);
    return out;
  };
  const opt = (n, k) => { const p = (NODE_DEFS[n.type].params || []).find(q => q.k === k); const o = p && Array.isArray(p.options) ? p.options[n.params[k] | 0] : ''; return String(o || ''); };
  const todo = nodes.filter(n => !SKIP.has(n.type) && NODE_DEFS[n.type] && isDefaultName(n));
  const named = new Map();           // id -> the name given here
  const nameOf = n => named.get(n.id) || (isDefaultName(n) ? '' : n.name);

  // what a region holds: the populations scattered into it
  const held = sh => common((users.get(sh.id) || []).filter(u => u.port === 0 && !SHAPES.has(u.node.type)).flatMap(u => tagsOf(u.node)));
  // the backdrop a node sits on, as a last word on where it is
  const groupOf = n => { const g = n && (groups || []).find(g => (g.members || []).includes(n.id)); const l = g ? String(g.label || '').trim() : ''; return l.length <= 24 ? l : ''; };
  const derive = n => {
    const P = n.params || {}, tags = tagsOf(n), what = common(tags);
    switch(n.type){
      case 'scatter': case 'pointfile': return String(P.tag || '').trim() || 'cells';
      case 'population': return 'pick ' + (String(P.tag || '').trim() || 'population');
      case 'celltype': {
        const row = (P.row | 0) === 0 ? String(P.name || 'custom') : opt(n, 'row').split(' ')[0];
        const who = String(P.tag || '').trim() || what;
        const frac = +P.frac > 0 && +P.frac < 1 ? ' ' + Math.round(+P.frac*100) + '%' : '';
        return (who ? who + ' ' : '') + row + frac; }
      case 'gather': {
        const feedsOther = (users.get(n.id) || []).some(u => u.node.type !== 'gather');
        if(tags.length && tags.length <= 3) return tags.join(' + ');
        if(feedsOther && !(users.get(n.id) || []).some(u => u.node.type === 'gather')) return 'all cells';
        if(what) return what + ' cells';
        // L2/3e, L2/3i, L4e, L4i is L2/3 + L4; a longer run is its two ends
        const parts = [...new Set(tags.map(t => { const c = t.replace(/([._\-].*|[eEiI])$/, ''); return c.length >= 2 ? c : t; }))];
        return parts.length <= 3 ? parts.join(' + ') : parts[0] + ' to ' + parts[parts.length - 1]; }
      case 'connect': return tags.length && tags.length <= 2 ? tags.join(' + ') + ' wiring' : 'local wiring';
      case 'project': { const a = String(P.from || '').trim() || common(tagsOf(up(n, 0))), b = String(P.to || '').trim() || common(tagsOf(up(n, 1)));
        return (a || 'source') + ' to ' + (b || 'target'); }
      case 'connectionsfile': { const f = String(P.file || '').trim().replace(/^.*[\\/]/, '').replace(/\.[^.]*$/, '');
        return f ? f + ' table' : 'connections table'; }
      case 'receptor': return 'receptor channels';
      case 'plasticity': return 'rule ' + (String(P.name || '').trim() || 'stdp');
      case 'pool': return 'pool on ' + (String(P.tag || '').trim() || 'all cells');
      case 'repeat': return (String(P.tag || '').trim() || what || 'copies') + ' x' + (P.copies | 0);
      case 'stimulus': {
        const mode = ['bias', 'pulse', 'ramp', 'noise'][P.mode | 0] || 'drive';
        const mask = up(n, 1);
        const tag = String(P.tag || '').trim(), where = mask ? held(mask) : '';
        const neg = +P.current < 0 && (P.mode | 0) !== 3;
        const kind = neg ? (mode === 'bias' ? 'hold' : 'inhibitory ' + mode) : mode;
        // a mask that holds no population is a spot of the stimulus's own
        if(mask && !where && !tag) return groupOf(mask) || groupOf(n) ? kind + ' on ' + (groupOf(mask) || groupOf(n)) : 'spot ' + kind;
        // with no mask and no tag the region is the node's own sphere: every cell when it is huge, a spot otherwise
        if(!mask && !tag) return +P.radius >= 100000 ? kind + ' on ' + (tags.length === 1 ? tags[0] : 'all cells') : 'spot ' + kind;
        // L4 and L4e is L4e; ce + ci and ce is ce
        const both = where && tag && !tag.toLowerCase().startsWith(where.toLowerCase()) && !where.split(' + ').includes(tag);
        return kind + ' on ' + (both ? where + ' ' + tag : tag || where); }
      case 'probe': return (String(P.label || '').trim() || String(P.tag || '').trim() || what || 'all cells') + ' probe';
      case 'input': { const sig = up(n, 2), to = String(P.tag || '').trim() || (up(n, 1) ? held(up(n, 1)) : '') || what || 'all cells';
        const from = sig ? named.get(sig.id) || nameOf(sig) : '';
        return from ? from + ' into ' + to : 'input to ' + to; }
      case 'testsignal': return 'bar sweep';
      case 'curriculum': return 'lesson, ' + (opt(n, 'sense') || 'sight');
      case 'live': return opt(n, 'device') || 'device';
      case 'footage': return 'image sequence';
      case 'analysis': return 'readout measures';
      case 'checkpoint': return 'simulation';
      case 'chart': return (PLOT[P.plot | 0] || 'chart') + (String(P.pop || '').trim() ? ' of ' + String(P.pop).trim() : '');
      default:
        if(VERB[n.type]) return (what ? what + ' ' : '') + VERB[n.type];
        return '';
    }
  };
  // everything but the regions first, since a region that only masks a node is named after that node
  for(const n of todo) if(!SHAPES.has(n.type) && n.type !== 'input'){ const s = derive(n); if(s) named.set(n.id, s); }
  for(const n of todo) if(n.type === 'input'){ const s = derive(n); if(s) named.set(n.id, s); }
  for(const n of todo) if(SHAPES.has(n.type)){
    const us = users.get(n.id) || [];
    const kind = n.type === 'noisefield' ? 'field' : n.type;
    let s = held(n);
    if(s) s += ' ' + kind;
    else { const m = us.find(u => u.port > 0); s = m ? (nameOf(m.node) || m.node.type).replace(/ on .*$/, '') + ' region' : kind; }
    named.set(n.id, s);
  }
  // one name per node: a second node under a name takes a number
  const taken = new Set(nodes.filter(n => !named.has(n.id)).map(n => String(n.name || '').toLowerCase()).filter(Boolean));
  // populations claim their names first: a scatter's name stays its tag exactly, so the tag goes on following it
  const order = [...todo].sort((a, b) => (b.type === 'scatter') - (a.type === 'scatter'));
  // two stimuli under one name are told apart by when they start
  const count = new Map(); for(const n of todo){ const s = named.get(n.id); if(s) count.set(s, (count.get(s) || 0) + 1); }
  for(const n of todo) if(n.type === 'stimulus' && count.get(named.get(n.id)) > 1){
    const same = todo.filter(m => m.type === 'stimulus' && named.get(m.id) === named.get(n.id));
    if(new Set(same.map(m => +m.params.t0 || 0)).size === same.length) n._at = ' at ' + (+n.params.t0 || 0) + ' ms';
  }
  for(const n of todo) if(n._at){ named.set(n.id, named.get(n.id) + n._at); delete n._at; }
  for(const n of order){
    let s = (named.get(n.id) || '').trim(); if(!s) continue;
    let cand = s, k = 2;
    while(taken.has(cand.toLowerCase())) cand = s + ' ' + (k++);
    taken.add(cand.toLowerCase());
    n.name = cand;
  }
  return todo.length;
}

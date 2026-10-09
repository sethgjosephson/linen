import { NODE_DEFS, hueCss, nodeDoc } from './nodes.js';
import { tagToken, tagFollowsName } from './tagname.js';
import { icon } from './icons.js';
import { expandProjection, tagMatch } from './nodes.js';
import { termLabel } from './protocol.js';

// population names come from a scene and are shown as markup
const esc = t => String(t).replace(/[&<>"]/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;' }[c]));

// app-level handlers for t:'action' button params (e.g. the read node's choose folder); registered by main.js so nodes.js stays DOM-free
let actionHandler = null;
export function setPropsAction(fn){ actionHandler = fn; }

// Opens the node reference at a given node; registered by main.js.
let docsHandler = null;
export function setPropsDocs(fn){ docsHandler = fn; }

// While a training run owns the network, parameters are readable but not writable: the run is simulating the graph it was handed, and a changed value here would describe a network nobody is running.
// Selecting nodes, reading their settings and moving them around stay available, because none of that reaches a computation.
// The node's name stays editable for the same reason: it is display only and deliberately outside the memo signature.
let locked = false, lockWhy = '';
export function setPropsLocked(on, why){
  locked = !!on;
  lockWhy = why || 'a training run is going';
}
function propsLocked(){ return locked; }

// The prose for every parameter is in docs.js, keyed by the same names the params use; it arrives as a tooltip on the label, and a right click opens the full entry.
//
// docs.js is loaded on first use rather than imported at the top: it is prose and it is not small, and the overlay loads it lazily for the same reason.
let DOCS = null, docsPending = null;
function withDocs(fn){
  if(DOCS){ fn(DOCS); return; }
  if(!docsPending) docsPending = import('./docs.js').then(m => { DOCS = m.NODES; return m.NODES; });
  docsPending.then(fn).catch(() => {});
}
const stripTags = h => {
  const d = document.createElement('div');
  d.innerHTML = String(h).replace(/<br\s*\/?>/gi, '\n');
  return (d.textContent || '').replace(/\n{3,}/g, '\n\n').trim();
};
// Attaches the prose for one parameter to its label.
// Both handlers are set immediately; only the text waits on the import.
function annotate(lab, type, key){
  lab.classList.add('hasdoc');
  lab.oncontextmenu = e => {
    e.preventDefault();
    if(docsHandler) docsHandler(type);
  };
  withDocs(nodes => {
    const entry = nodeDoc(type, nodes);
    const prose = entry && entry.params && entry.params[key];
    if(!prose) return;
    lab.title = stripTags(prose) + '\n\nright click for the full entry';
  });
}

// Supplied by main.js: the running engine's hello reply (ENGINE.md section 2), or null before one has arrived.
// A setting whose term the engine does not list is grayed, with a tooltip naming the engine.
let engineFor = null;
export function setPropsEngine(fn){ engineFor = fn; }
// The engine that lacks a setting's term, or null.
function lackingEngine(p){
  if(!p.term || !engineFor) return null;
  const h = engineFor();
  const want = Array.isArray(p.term) ? p.term : [p.term];
  return h && !want.some(t => (h.terms || []).includes(t)) ? h : null;
}

// Supplied by main.js: the newest metrics row for a probe label, or null.
let latestFor = null;
const tabFor = {};   // node type -> the group last read on it
export function setPropsMetrics(fn){ latestFor = fn; }
// A plasticity node or a project node upstream also decides whether a pathway learns, so the panel is given the graph and draws every rule that applies to this connect node wherever it was declared, marking the ones that came from elsewhere and graying them when their node is bypassed.
// It reads the graph rather than the computed stream deliberately: a bypassed node contributes nothing to the computation, so a view derived from it would show its row simply absent, and the point is to see what is switched off.
let graphFor = null;
export function setPropsGraph(fn){ graphFor = fn; }
// Everything feeding this node, following every input rather than only the main one, since populations arrive through merges.
// Every population tag the graph names: the scatters' own, and the tags a computed stream carries, which is where a repeat node's numbered copies live (they exist in no scatter's parameters).
function graphTags(nodes){
  const tags = [];
  const add = t => { t = String(t || '').trim();
    if(t && !tags.some(x => x.toLowerCase() === t.toLowerCase())) tags.push(t); };
  for(const m of (nodes || [])){
    if(m.type === 'scatter' && m.params) add(m.params.tag);
    const c = m._cache;
    if(c && (c.kind === 'points' || c.kind === 'net') && c.tags)
      for(const t of Object.values(c.tags)) add(t);
  }
  return tags.sort((a, b) => a.localeCompare(b));
}
// A population renamed is renamed everywhere it is named.
// Every setting of the population kind (probes, stimuli, inputs, cell types, populations, pools, projections), plasticity scopes, charts and pair-table rows refer to a population by its tag, and a tag edited on the scatter alone would leave every one of them pointing at a name nothing carries (an input still on Input2 after its scatter became Input).
// Each node that changes is edited like any other, so its computation follows.
export function retagReferences(nodes, oldTag, newTag, onEdit){
  const was = String(oldTag || '').trim(), now = String(newTag || '').trim();
  if(!was || !now || was.toLowerCase() === now.toLowerCase()) return;
  const same = v => String(v || '').trim().toLowerCase() === was.toLowerCase();
  for(const n of nodes || []){
    const p = n.params; if(!p) continue;
    let hit = false;
    // every setting that names a population, read off the node's own definition rather than a list of node types, so a new node kind is covered as soon as it declares one.
    // A node that defines a population carries its tag as an identity, not as one of these, so it is not touched.
    const def = NODE_DEFS[n.type];
    for(const q of (def && def.params) || [])
      if(q.t === 'tag' && same(p[q.k])){ p[q.k] = now; hit = true; }
    if(['project', 'plasticity'].includes(n.type)){
      if(same(p.from)){ p.from = now; hit = true; }
      if(same(p.to)){ p.to = now; hit = true; }
    }
    if(n.type === 'chart' && same(p.pop)){ p.pop = now; hit = true; }
    if(n.type === 'connect' && p.table){
      const lines = String(p.table).split(String.fromCharCode(10)).map(line => {
        const body = line.split('#')[0], comment = line.slice(body.length);
        const f = body.trim().split(/\s+/).filter(Boolean);
        if(f.length < 3) return line;
        let ch = false;
        if(same(f[0])){ f[0] = now; ch = true; }
        if(same(f[1])){ f[1] = now; ch = true; }
        return ch ? f.join(' ') + (comment ? ' ' + comment.trim() : '') : line;
      });
      const t = lines.join(String.fromCharCode(10));
      if(t !== p.table){ p.table = t; hit = true; }
    }
    if(hit) onEdit(n);
  }
}
// The sweep node's panel: one tab per variant, and in it rows of node, setting, value.
// Built like a custom node in a compositor: pick a node from the graph, pick one of its settings, and that setting's own control appears.
// Rows sit in the order added; a node's rows can be duplicated and pointed at another node, a whole variant can be duplicated.
const sweepTabFor = new Map();            // node id -> active variant index
function sweepUI(node, p, onEdit, nodes){
  // One variant at a time, picked from a list; its name on the line below; then one row per edit (node, setting, value, remove) and one add button.
  // No per-node grouping, ids only where two nodes would otherwise read the same, and no sliders: a sweep row is a value, not a control.
  const wrap = el_('div', 'sweepbox');
  const vars = Array.isArray(node.params[p.k]) && node.params[p.k].length
    ? node.params[p.k] : (node.params[p.k] = [{ name:'sweep 1', edits:[] }]);
  let active = Math.min(sweepTabFor.get(node.id) || 0, vars.length - 1);
  const commit = () => onEdit(null);      // display only: the run reads it
  const redraw = () => { const w = sweepUI(node, p, onEdit, nodes); wrap.replaceWith(w); };
  const others = (nodes || []).filter(m => m.type !== 'note' &&
    m.type !== 'pin' && NODE_DEFS[m.type] && NODE_DEFS[m.type].params.length);
  const baseLabel = m => NODE_DEFS[m.type].title +
    (m.name && m.name !== NODE_DEFS[m.type].title ? ' ' + m.name : '');
  const labelCount = new Map();
  for(const m of others) labelCount.set(baseLabel(m), (labelCount.get(baseLabel(m)) || 0) + 1);
  const nodeLabel = m => baseLabel(m) + (labelCount.get(baseLabel(m)) > 1 ? ' #' + m.id : '');
  const settable = m => NODE_DEFS[m.type].params.filter(q =>
    ['float', 'int', 'select', 'str', 'tag', 'pairs', 'text'].includes(q.t) && !q.identity);
  const byId = id => others.find(m => m.id === id);
  const sel = (opts, value, onchange, cls) => {
    const el = document.createElement('select');
    if(cls) el.className = cls;
    for(const [val, text] of opts){ const op = document.createElement('option');
      op.value = val; op.textContent = text; el.appendChild(op); }
    el.value = value; el.onchange = () => onchange(el.value);
    return el;
  };
  const iconBtn = (name, title, fn, cls = 'icb icon-only') => {
    const b = el_('button', cls); b.innerHTML = icon(name); b.type = 'button'; b.title = title;
    b.onclick = fn; return b;
  };

  // which variant, and what can be done with it
  const pick = el_('div', 'sw-pick');
  pick.appendChild(sel(vars.map((v, i) => [i, v.name || ('sweep ' + (i + 1))]), active,
    val => { sweepTabFor.set(node.id, +val); redraw(); }, 'sw-which'));
  pick.appendChild(iconBtn('add', 'a new empty variant', () => {
    vars.push({ name:'sweep ' + (vars.length + 1), edits:[] });
    sweepTabFor.set(node.id, vars.length - 1); commit(); redraw(); }));
  pick.appendChild(iconBtn('copy', 'a copy of this variant, to change the numbers', () => {
    const v = vars[active];
    vars.splice(active + 1, 0, JSON.parse(JSON.stringify({ ...v, name:(v.name || 'sweep') + ' copy' })));
    sweepTabFor.set(node.id, active + 1); commit(); redraw(); }));
  const del = iconBtn('remove', 'remove this variant', () => {
    vars.splice(active, 1); sweepTabFor.set(node.id, Math.max(0, active - 1)); commit(); redraw(); });
  del.disabled = vars.length < 2;
  pick.appendChild(del);
  wrap.appendChild(pick);
  const v = vars[active];
  const nameRow = el_('div', 'sw-name');
  nameRow.appendChild(el_('span', 'sw-lab', 'name'));
  const name = document.createElement('input');
  name.type = 'text'; name.value = v.name || ''; name.placeholder = 'name this variant';
  name.onchange = () => { v.name = name.value.trim(); commit(); redraw(); };
  nameRow.appendChild(name);
  wrap.appendChild(nameRow);

  // one row per edit
  if(v.edits.length){
    const head = el_('div', 'sw-row sw-cols');
    for(const t of ['node', 'setting', 'value', '', '']) head.appendChild(el_('span', 'sw-lab', t));
    wrap.appendChild(head);
  }
  for(const e of v.edits){
    const r = el_('div', 'sw-row');
    const m = byId(e.node);
    r.appendChild(sel(others.map(x => [x.id, nodeLabel(x)]), e.node, val => {
      e.node = +val;
      const m2 = byId(e.node);
      // keep the setting when the new node has it, else its first
      if(m2 && !settable(m2).some(q => q.k === e.key)){
        const q = settable(m2)[0]; e.key = q ? q.k : ''; e.value = q ? m2.params[q.k] : '';
      }
      commit(); redraw();
    }));
    const opts = m ? settable(m) : [];
    r.appendChild(sel(opts.map(q => [q.k, q.label]), e.key,
      val => { e.key = val; e.value = m.params[e.key]; commit(); redraw(); }));
    const q = opts.find(x => x.k === e.key);
    if(!q) r.appendChild(el_('span', 'sw-lab', m ? '' : 'node gone'));
    else if(q.t === 'select')
      r.appendChild(sel(q.options.map((o, i) => [i, o]).filter(([i]) => !q.hideOption || !q.hideOption(i) || i === +e.value), e.value, val => { e.value = +val; commit(); }));
    else if(q.t === 'tag')
      r.appendChild(sel([['', 'all'], ['E', 'E'], ['I', 'I'], ...graphTags(nodes).map(t => [t, t])],
        e.value || '', val => { e.value = val; commit(); }));
    else if(q.t === 'str'){
      const inp = document.createElement('input');
      inp.type = 'text'; inp.value = e.value == null ? '' : e.value;
      inp.onchange = () => { e.value = inp.value; commit(); };
      r.appendChild(inp);
    } else if(q.t === 'pairs' || q.t === 'text'){
      // a whole pair table or text block as one value, so a sweep can vary the weights between classes (the E to I row)
      const ta = document.createElement('textarea');
      ta.rows = 3; ta.value = e.value == null ? '' : e.value;
      ta.onchange = () => { e.value = ta.value; commit(); };
      r.appendChild(ta);
    } else r.appendChild(num(e.value, q, val => { e.value = val; commit(); }));
    // duplicate the row in place: the same node and setting, to change the value or point the copy at another node
    r.appendChild(iconBtn('copy', 'duplicate this edit', () => {
      v.edits.splice(v.edits.indexOf(e) + 1, 0, { ...e }); commit(); redraw(); }, 'sw-dup icb icon-only'));
    r.appendChild(iconBtn('remove', 'remove this edit', () => {
      v.edits.splice(v.edits.indexOf(e), 1); commit(); redraw(); }, 'sw-del icb icon-only'));
    wrap.appendChild(r);
  }
  if(!v.edits.length) wrap.appendChild(el_('div', 'empty',
    'no edits: this variant runs the graph as it stands'));
  const addRow = el_('button', 'icb');
  addRow.innerHTML = icon('add') + '<span>edit</span>';
  addRow.type = 'button'; addRow.title = 'a row: a node, one of its settings, and the value this variant runs';
  addRow.onclick = () => {
    const last = v.edits[v.edits.length - 1];
    const m = (last && byId(last.node)) || others[0]; if(!m) return;
    const q = settable(m)[0];
    v.edits.push({ node:m.id, key:q ? q.k : '', value:q ? m.params[q.k] : '' });
    commit(); redraw();
  };
  wrap.appendChild(addRow);
  return wrap;
}
function upstreamOf(node, byId){
  const seen = new Set(), out = [];
  const walk = n => {
    if(!n || seen.has(n.id)) return;
    seen.add(n.id);
    for(const inp of n.inputs || []) if(inp){
      const s = byId(inp.id);
      if(s && !seen.has(s.id)){ out.push(s); walk(s); }
    }
  };
  walk(node);
  return out;
}
const pct = v => v === undefined || v === null ? '-' : Math.round(v*100) + '%';
const fmt3 = v => v === undefined || v === null ? '-' : (+v).toFixed(3);
function rowsFor(m){
  const r = [];
  const add = (k, v, t) => r.push('<div class="prow2" title="' + (t || '') +
    '"><span>' + k + '</span><b>' + v + '</b></div>');
  add('rate', m.hz === undefined ? '-' : m.hz + ' Hz', 'mean firing rate over the window');
  if(m.decode !== undefined)
    add('decode', pct(m.decode), 'which of the items this population identifies, ' +
      'against chance of ' + pct(m.chance));
  if(m.selectivity !== undefined)
    add('selectivity', fmt3(m.selectivity), 'how few items each cell answers to; ' +
      '0 is everything, 1 is one item');
  if(m.bind)
    add('binding v|a', fmt3(m.bind.vis) + ' | ' + fmt3(m.bind.aud),
      'does one sense alone wake this item conjunctive cells rather than ' +
      'another item ones. 0 is chance');
  if(m.timing)
    add('timing v|a', fmt3(m.timing.vis) + ' | ' + fmt3(m.timing.aud),
      'does one sense wake those cells in the order the pair does. 0 is unrelated');
  if(m.bind && m.bind.cells !== undefined)
    add('conjunctive cells', m.bind.cells, 'per item, fixed at calibration');
  return r.join('');
}


// Supplied by main.js: every metrics row seen this session, oldest first.
let chartsFor = null;
export function setPropsCharts(fn){ chartsFor = fn; }
// What a probe covers, from the computed network: the panel names the populations, the viewer draws them.
let probeSetFor = null;
export function setPropsProbeSet(fn){ probeSetFor = fn; }
// the same for the other nodes that are about a set of cells: what a stimulus drives, where an input lands, which cells an analysis called conjunctive
let driveSetFor = null, inputSetFor = null, conjSetFor = null;
export function setPropsDriveSet(fn){ driveSetFor = fn; }
export function setPropsInputSet(fn){ inputSetFor = fn; }
export function setPropsConjSet(fn){ conjSetFor = fn; }
// Where a chart gets a run other than this session's: tags() lists what the project folder holds, rows(tag) returns what is already loaded, load(tag) fetches it and redraws.
// The panel asks; main.js knows where files are.
let runsFor = null;
export function setPropsRuns(provider){ runsFor = provider; }
// Supplied by main.js: a promise of the live weights, so the distribution can be drawn from what the engine currently holds rather than from the wiring.
let weightsFor = null;
export function setPropsWeights(fn){ weightsFor = fn; }
// Supplied by main.js: the newest item by cell response matrix per population.
let matricesFor = null;
export function setPropsMatrices(fn){ matricesFor = fn; }
// the page's live recording (record.js) and the selected cell's trace
let liveFor = null;
export function setPropsLive(fn){ liveFor = fn; }

// Which populations and which measures the history actually contains, so the chart node offers what exists rather than a list someone has to keep in step with the analysis code by hand.
// This is the same principle the pair table follows: the choices come from the data, so a typo is impossible and a new measure appears without anyone editing a dropdown.
function chartFields(history){
  const pops = new Set(), fields = new Set();
  for(const row of history){
    const p = row && row.readout && row.readout.pops;
    if(!p) continue;
    for(const [name, v] of Object.entries(p)){
      pops.add(name);
      // nested measures (crossDecode.ab.acc, rsa.visAud.rho, block.ratio) are offered by dotted path, so the chart can draw them without a list
      const walk = (o, pre, depth) => {
        for(const [k, val] of Object.entries(o)){
          if(typeof val === 'number' && Number.isFinite(val)) fields.add(pre + k);
          else if(val && typeof val === 'object' && !Array.isArray(val) && depth < 3)
            walk(val, pre + k + '.', depth + 1);
        }
      };
      walk(v, '', 0);
    }
  }
  // the per-probe rate lives outside readout, and is the measure people reach for first
  for(const row of history) for(const pr of (row.probes || [])) pops.add(pr.l);
  if(history.length) fields.add('hz');
  return { pops:[...pops].sort(), fields:[...fields].sort() };
}

const el_ = (t, cls, txt) => { const e = document.createElement(t);
  if(cls) e.className = cls; if(txt !== undefined) e.textContent = txt; return e; };
// A drawn eye rather than a glyph: the nearest characters (bullseye, circled dot) do not read as one, and the panel has no icon font. currentColor so it takes its state from the cell's hover rules like everything else here.
function eyeIcon(lidColor, irisColor){
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 10');
  svg.setAttribute('width', '13'); svg.setAttribute('height', '9');
  svg.setAttribute('aria-hidden', 'true');
  const lid = document.createElementNS(NS, 'path');
  lid.setAttribute('d', 'M1 5 Q8 -0.6 15 5 Q8 10.6 1 5 Z');
  lid.setAttribute('fill', 'none');
  lid.setAttribute('stroke', lidColor || 'currentColor');
  lid.setAttribute('stroke-width', '1.1');
  const iris = document.createElementNS(NS, 'circle');
  iris.setAttribute('cx', '8'); iris.setAttribute('cy', '5'); iris.setAttribute('r', '2');
  iris.setAttribute('fill', irisColor || 'currentColor');
  svg.appendChild(lid); svg.appendChild(iris);
  return svg;
}
const RULE_BUILTIN = ['frozen', 'default'];
// The eye is a toggle, and its color is the pathway's color in the viewer.
// Gray is off.
// That makes the table the legend: the row you are reading is unmistakably the bundle you are looking at, with no separate key. rule is the row's fifth column: '' the checkpoint's rule, '0' frozen, or the name of a plasticity node's rule.
// The pathway is drawn in the color that says which, so a bundle that is not learning looks like one.
function eyeCell(from, to, litMap, onToggle, rule){
  const td = el_('td', 'pt-eye');
  const key = from + '→' + to;
  const col = litMap[key];
  // Lid in the source color, iris in the target's: the icon carries the same two colors the bundle does, so the row reads as its own legend.
  const svg = eyeIcon(col && col.from, col && col.to);
  td.appendChild(svg);
  if(col) td.classList.add('on');
  td.title = col ? `stop drawing ${from} to ${to}` : `draw ${from} to ${to} in the viewer`;
  if(from && to && onToggle) td.onclick = e => { e.stopPropagation(); onToggle(from, to, rule); };
  return td;
}
// Parse just enough to edit: the fifth column stays a string, because it can now be a name and because a value this panel does not recognize must survive a round trip rather than be silently normalized away.
function parsePairs(text){
  return String(text || '').split('\n').map(line => {
    const body = line.split('#')[0], comment = line.slice(body.length);
    const f = body.trim().split(/\s+/).filter(Boolean);
    if(f.length < 3) return { raw:line, bad:true };
    return { pre:f[0], post:f[1], pm:f[2], wm:f[3] !== undefined ? f[3] : '1',
      pl:f[4] !== undefined ? f[4] : '', comment, raw:line };
  });
}
const unparsePairs = rows => rows.map(r => r.bad ? r.raw
  : [r.pre, r.post, r.pm, r.wm, r.pl].filter((v, i) => i < 4 || (v !== '' && v !== undefined))
      .join(' ') + (r.comment || '')).join('\n');

// A text field that is also a dropdown.
// A native select cannot be typed into, and a datalist can be typed into but shows no arrow, so neither one alone is the control this wants: the population tags are known and worth picking from, and someone who knows the name should be able to just write it.
// Groups are [{label, items}], rendered as headed sections.
//
// The list is positioned fixed rather than absolute.
// The properties pane scrolls, so an absolutely positioned list is clipped by it the moment the row is near the bottom, which is exactly where a long table puts it.
// The open list is placed once, in viewport coordinates, so anything that scrolls underneath it leaves it hanging over the wrong row.
// Closing on scroll is the fix, and it is one shared listener rather than one per control: a two hundred row table builds six hundred comboboxes, and only ever one of them is open.
let openCombo = null;
function closeOpenCombo(e){
  // The list scrolls too, once it is longer than its own height, and that scroll reaches this handler in the capture phase like any other, so reaching for an option far down the list would close it.
  // Its own scrolling is the one kind that must not count, since the list moves with it.
  const t = e && e.target;
  if(t && t.nodeType === 1 && t.closest && t.closest('.cb-list')) return;
  if(openCombo){ openCombo(); openCombo = null; }
}
// guarded so the module stays importable outside a browser, which is how the battery reaches node definitions without a DOM
if(typeof window !== 'undefined'){
  window.addEventListener('scroll', closeOpenCombo, true);
  window.addEventListener('resize', closeOpenCombo);
}

function comboBox(value, groups, onPick, opts = {}){
  const wrap = el_('div', 'cb');
  const inp = document.createElement('input');
  inp.type = 'text'; inp.value = value || ''; inp.disabled = !!opts.disabled;
  inp.className = 'cb-in' + (opts.cls ? ' ' + opts.cls : '');
  if(opts.title) inp.title = opts.title;
  const arrow = el_('button', 'cb-arrow', '▾');
  arrow.type = 'button'; arrow.tabIndex = -1; arrow.disabled = !!opts.disabled;
  arrow.title = 'show every choice';
  const list = el_('div', 'cb-list');
  list.style.display = 'none';
  document.body.appendChild(list);          // fixed, so it escapes the pane
  wrap.appendChild(inp); wrap.appendChild(arrow);

  let open = false, marked = -1, flat = [];
  const place = () => {
    const r = inp.getBoundingClientRect();
    list.style.left = r.left + 'px';
    list.style.width = Math.max(r.width + 22, 120) + 'px';
    const below = window.innerHeight - r.bottom;
    if(below < 160 && r.top > below){
      list.style.top = ''; list.style.bottom = (window.innerHeight - r.top) + 'px';
    } else { list.style.bottom = ''; list.style.top = r.bottom + 'px'; }
  };
  const build = filter => {
    list.textContent = ''; flat = [];
    const q = String(filter || '').toLowerCase();
    for(const g of groups){
      const items = g.items.filter(v => !q || v.toLowerCase().includes(q));
      if(!items.length) continue;
      if(g.label) list.appendChild(el_('div', 'cb-head', g.label));
      for(const v of items){
        const row = el_('div', 'cb-item', v);
        row.onmousedown = e => { e.preventDefault(); pick(v); };
        list.appendChild(row); flat.push(row);
      }
    }
    if(!flat.length) list.appendChild(el_('div', 'cb-none', 'nothing matches'));
    marked = -1;
  };
  const show = filter => {
    if(openCombo && openCombo !== hide) closeOpenCombo();
    build(filter); place(); list.style.display = 'block'; open = true;
    openCombo = hide;
  };
  const hide = () => {
    list.style.display = 'none'; open = false;
    if(openCombo === hide) openCombo = null;
  };
  const pick = v => { inp.value = v; hide(); onPick(v); };
  const mark = d => {
    if(!flat.length) return;
    if(marked >= 0) flat[marked].classList.remove('on');
    marked = (marked + d + flat.length) % flat.length;
    flat[marked].classList.add('on');
    flat[marked].scrollIntoView({ block:'nearest' });
  };
  arrow.onclick = () => { if(open) hide(); else { inp.focus(); show(''); } };
  inp.oninput = () => show(inp.value);
  inp.onfocus = () => show(inp.value);
  inp.onblur = () => { hide(); if(inp.value !== (value || '')) onPick(inp.value.trim()); };
  inp.onkeydown = e => {
    if(e.key === 'ArrowDown'){ e.preventDefault(); if(!open) show(inp.value); else mark(1); }
    else if(e.key === 'ArrowUp'){ e.preventDefault(); mark(-1); }
    else if(e.key === 'Enter'){ e.preventDefault();
      if(open && marked >= 0) pick(flat[marked].textContent); else { hide(); onPick(inp.value.trim()); } }
    else if(e.key === 'Escape'){ e.preventDefault(); inp.value = value || ''; hide(); }
  };
  wrap._cleanup = () => { if(openCombo === hide) openCombo = null; list.remove(); };
  return wrap;
}


// The chart node's own panel: its controls, then the plot beneath them.
// Drawn on a canvas at device pixel ratio, because a line chart at CSS resolution on a high-density display is visibly soft and this is a thing people read numbers off.
// The open chart panel's repaint, so an edit to a parameter that only changes the drawing (height, bins) redraws it without rebuilding the controls: a rebuild during a slider drag replaces the slider under the pointer and the drag stops.
let chartRepaint = null;
// Where a chart's numbers come from: the run it names (or this session's), the live recorder, the engine's weights, the readout's matrices.
// The panel and the figure export both draw from this, so they draw the same thing.
export function chartSources(node){
  const tag = String(node.params.source || '').trim();
  let history = [], groups = null;
  if(!tag) history = chartsFor ? chartsFor() : [];
  else if(runsFor){
    history = runsFor.rows(tag) || [];
    // a sweep draws as a line per variant and condition, each the mean of its replicates with the spread behind it
    if(runsFor.groups) groups = runsFor.groups(tag);
  }
  return { history, groups, live:liveFor || null, weights:weightsFor || null, matrices:matricesFor || null };
}
// the caption lines the page stamps under a figure, for the preview
let captionFor = null;
export function setPropsCaption(fn){ captionFor = fn; }
function chartUI(node, onEdit, rebuild = () => {}){
  const wrap = el_('div', 'chartbox');
  const P = node.params;
  // A chart draws one run: this session's by default, or one from the project folder by tag.
  // Everything below reads that history, so the populations and measures offered are the ones that run measured.
  const src = String(P.source || '').trim();
  let history = [], loading = false, groups = null;
  if(!src) history = chartsFor ? chartsFor() : [];
  else if(runsFor){
    const rows = runsFor.rows(src);
    if(rows) history = rows;
    else { loading = true; runsFor.load(src); }
    if(runsFor.groups) groups = runsFor.groups(src);
  }
  const plot = P.plot | 0, live = plot >= 7;
  let { pops, fields } = chartFields(history);
  // a live plot offers the probed populations, whatever any run measured
  if(live && liveFor){ const L = liveFor(); pops = L && L.rec ? L.rec.labels() : []; }
  const combos = [];

  const row = el_('div', 'chart-controls');
  const pick = (label, value, items, onPick, cls) => {
    row.appendChild(el_('span', 'chart-lab', label));
    const cb = comboBox(value, [{ label:'seen this run', items }], onPick,
      { cls, title:value });
    row.appendChild(cb); combos.push(cb);
  };
  // the run first: what is being looked at decides what the rest can offer
  {
    const tags = runsFor ? runsFor.tags() : [];
    const sweeps = runsFor && runsFor.sweeps ? runsFor.sweeps() : [];
    row.appendChild(el_('span', 'chart-lab', 'from'));
    const cb = comboBox(src || 'this session',
      [{ label:'runs', items:['this session', ...tags] },
        ...(sweeps.length ? [{ label:'sweeps (mean and spread)', items:sweeps }] : [])],
      v => { P.source = (v === 'this session' ? '' : v); onEdit(node); rebuild(); },
      { cls:'chart-src', title:src || 'the run this page is doing or last did' });
    row.appendChild(cb); combos.push(cb);
  }
  // A pick changes what the other pickers can offer, so it rebuilds rather than repaints: the measures a population records are not the measures another one does, and the run decides both.
  pick('of', P.pop || '', pops, v => { P.pop = v; onEdit(node); rebuild(); },
    pops.includes(P.pop) ? '' : 'pt-unknown');
  if(plot === 0)
    pick('show', P.field || 'decode', fields, v => { P.field = v; onEdit(node); rebuild(); },
      fields.includes(P.field) ? '' : 'pt-unknown');
  wrap.appendChild(row);

  const cv = document.createElement('canvas');
  cv.className = 'chart-canvas';
  cv.style.height = (P.height | 0 || 130) + 'px';
  wrap.appendChild(cv);
  const note = el_('div', 'chart-note');
  wrap.appendChild(note);

  // The canvas has no width until it is in the document, so the draw waits for layout rather than measuring zero and drawing nothing.
  // The drawing is chartpaint.js, shared with the figure export: this only sizes the canvas and picks the style.
  // With the preview on, the panel shows the figure as it will be saved (its theme, font, size and proportions), scaled to the panel's width.
  let paintTok = 0;
  const paint = () => {
    const tok = ++paintTok;
    import('./chartpaint.js').then(async F => {
      const preview = (P.preview | 0) === 1;
      const spec = preview ? F.figureSpec(P) : null;
      const cw = cv.clientWidth;
      if(!cw) return;
      if(preview) cv.style.height = Math.round(cw*spec.hPt/spec.wPt) + 'px';
      const w = cv.clientWidth, h = cv.clientHeight;
      if(!w || !h) return;
      // the panel is zoomed by the text-size setting, so a CSS pixel here is worth more device pixels than devicePixelRatio alone says
      const ui = +getComputedStyle(document.documentElement).getPropertyValue('--ui') || 1;
      const dpr = Math.min(3, (window.devicePixelRatio || 1) * ui);
      cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
      const ctx = cv.getContext('2d');
      const src = { ...chartSources(node), stale:() => tok !== paintTok };
      let r;
      if(preview){
        const k = w/spec.wPt*dpr;
        ctx.setTransform(k, 0, 0, k, 0, 0);
        r = await F.paintFigure(ctx, spec, P, src, (P.caption | 0) === 0 && captionFor ? captionFor(node) : []);
      } else {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        r = await F.paintChart(ctx, w, h, P, src, F.screenStyle(P));
      }
      if(!r || tok !== paintTok) return;
      node._chart = r.chart ? { ...r.chart, smoothMs:r.smoothMs || 0 } : null;
      note.textContent = (loading ? 'reading the run from the project folder · ' : '') + (r.note || '') +
        (preview ? (r.note ? ' · ' : '') + `preview of the figure: ${spec.mmW} x ${spec.mmH} mm, ${spec.pxW} x ${spec.pxH} px at ${spec.dpi} dpi` : '');
    }).catch(e => {
      // silence here reads as "no data" and means "the drawing code did not load", which are opposite problems
      note.textContent = 'the chart could not be drawn: ' + e.message;
    });
  };
  // Painted by renderProps once this is in the document.
  // Every other trigger is unreliable in the case that matters: the canvas has no width until it is mounted, so painting here measures zero, and a hidden tab delivers neither a frame nor a resize, so both fallbacks stall too.
  // The mount hook is the one that always fires.
  wrap._mount = paint;
  // what a parameter edit calls: the canvas takes its height from the parameter every time, so a height change lands here too
  chartRepaint = () => { if((P.preview | 0) !== 1) cv.style.height = (P.height | 0 || 130) + 'px'; paint(); };
  requestAnimationFrame(paint);
  const ro = new ResizeObserver(() => paint());
  ro.observe(cv);
  const onVis = () => { if(!document.hidden) paint(); };
  document.addEventListener('visibilitychange', onVis);
  // a live plot follows the recording while the panel is open
  const timer = live ? setInterval(() => { if(!document.hidden) paint(); }, 500) : null;
  wrap._cleanup = () => {
    if(timer) clearInterval(timer);
    ro.disconnect();
    document.removeEventListener('visibilitychange', onVis);
    for(const c of combos) c._cleanup();
    if(chartRepaint && wrap._repaint === chartRepaint) chartRepaint = null;
  };
  wrap._repaint = chartRepaint;
  return wrap;
}

// The table's view state (rows expanded, sort column and direction) lives here, per node, rather than in the closure below.
// The whole panel is rebuilt whenever anything it shows changes, including the lit set when an eye is clicked, and a closure-held state would be reset by every rebuild: expand the rows, light one pathway, and the table collapses again.
const PT_STATE = new Map();
const ptState = id => { let st = PT_STATE.get(id); if(!st){ st = {}; PT_STATE.set(id, st); } return st; };

function pairTableUI(node, p, onEdit, opts = {}){
  const wrap = el_('div', 'pairtable');
  const st = ptState(node.id + (opts.roomy ? ':roomy' : ''));
  const g = graphFor && graphFor();
  const up = g ? upstreamOf(node, g.byId) : [];
  // the populations this connect node will wire: from the computed stream on its input when there is one (a repeat node's copies are only there), else from the scatters upstream
  const fed = node.inputs && node.inputs[0] && g ? g.byId(node.inputs[0].id) : null;
  const computed = fed && fed._cache && fed._cache.kind === 'points' ? fed._cache.tags : null;
  const tags = computed
    ? [...new Set(Object.values(computed).filter(Boolean).map(String))].sort()
    : [...new Set(up.filter(n => n.type === 'scatter' && n.params.tag)
        .map(n => String(n.params.tag)))].sort();
  const ruleNodes = up.filter(n => n.type === 'plasticity');
  const projNodes = up.filter(n => n.type === 'project' && n.params.from && n.params.to);
  const ruleNames = ruleNodes.map(n => String(n.params.name || '').trim()).filter(Boolean);
  // which pathways are lit, and in what color, so each eye can wear its own
  const lit = () => (g && g.pathwayState) ? g.pathwayState() : {};


  // --- what other nodes contribute -----------------------------------
  // The pathways as the wiring will see them come from the computed stream on this node's input: a project node inside a repeat is there once per copy, renamed, and a pattern is there once per matching pair.
  // Without a computed stream the project nodes upstream stand in.
  // A pattern on one side names every matching population at once; the table lists them one per row, since a row is a pathway an eye can light.
  const oneSided = (pr, tagsMap) => {
    const all = [...new Set(Object.values(tagsMap || {}))];
    const hits = side => all.filter(t => tagMatch(t, pr[side]) !== null);
    if(String(pr.from).includes('*') && !String(pr.to).includes('*')) return hits('from').map(t => ({ ...pr, from:t }));
    if(String(pr.to).includes('*') && !String(pr.from).includes('*')) return hits('to').map(t => ({ ...pr, to:t }));
    return [pr];
  };
  const computedProjs = fed && fed._cache && fed._cache.kind === 'points' && fed._cache.projs
    ? fed._cache.projs.flatMap(pr => expandProjection(pr, fed._cache.tags)).flatMap(pr => oneSided(pr, fed._cache.tags)) : null;
  const contributed = [
    ...(computedProjs
      ? computedProjs.map(pr => ({ node:(g && g.byId(pr.node)) || { on:true }, kind:'project',
          from:pr.from, to:pr.to, pm:'via projection',
          pl:pr.frozen ? 'frozen' : String(pr.rule || 'default') }))
      : projNodes.map(n => ({ node:n, kind:'project', from:n.params.from, to:n.params.to,
          pm:'via projection', pl:(n.params.frozen | 0) === 1 ? 'frozen'
            : String(n.params.rule || 'default') }))),
    ...ruleNodes.filter(n => n.params.from && n.params.to).map(n => ({ node:n, kind:'plasticity',
      from:n.params.from, to:n.params.to, pm:'rule only',
      pl:String(n.params.name || '').trim() || 'unnamed' })),
  ];
  // --- this node's own rows ------------------------------------------
  const rows = parsePairs(node.params[p.k]);
  const real = rows.filter(r => !r.bad);
  const LIMIT = opts.roomy ? Infinity : 40;
  let showAll = st.showAll !== undefined ? st.showAll : real.length <= LIMIT;
  const body = el_('div');
  wrap.appendChild(body);

  const commit = () => { node.params[p.k] = unparsePairs(rows); onEdit(node); };
  // Every population plugged in gets a recurrent row the moment it appears, so the table lists what is there instead of waiting to be told.
  // The row is the default (multipliers 1, default rule), and it is added only when no row could already match that pair, so it never changes what wires: a wildcard or class row that covers the pair is left to cover it.
  if(!locked){
    const covers = (key, t) => key === '*' || key === 'E' || key === 'I' ||
      key.toLowerCase() === t.toLowerCase() ||
      (key.length > 1 && key.endsWith('*') &&
        t.toLowerCase().startsWith(key.slice(0, -1).toLowerCase()));
    let added = false;
    // an empty table parses as one blank row; it must not become a blank line
    if(rows.length === 1 && rows[0].bad && !String(rows[0].raw || '').trim() && tags.length)
      rows.length = 0;
    for(const t of tags){
      if(real.some(r => covers(r.pre, t) && covers(r.post, t))) continue;
      const r = { pre:t, post:t, pm:'1', wm:'1', pl:'' };
      rows.push(r); real.push(r); added = true;
    }
    // saved without a computation: the rows say what is already wired
    if(added){ node.params[p.k] = unparsePairs(rows); onEdit(null); }
  }
  // One table, the full width of the panel, holding everything that decides this connect node's pairs: the rows it owns, and the rows a plasticity or project node upstream contributes.
  // They share a column layout so the whole thing reads as one list rather than two stacked views.
  //
  // Percentage widths and no minimum: a minimum wider than the panel makes the table scroll sideways and puts a row's numbers on screen while its two population names are off it, so every row reads "frozen" against nothing.
  const colgroup = () => { const cg = el_('colgroup');
    for(const w of ['24%', '24%', '11%', '10%', '21%', '5%', '5%']){
      const c = el_('col'); c.style.width = w; cg.appendChild(c);
    }
    return cg; };
  // Each combobox parks its list on document.body so the scrolling pane cannot clip it, which means each one has to be taken away again when the table redraws.
  // Every edit redraws, so leaking here would leave a stack of invisible lists behind after a few clicks.
  const combos = [];
  const dropCombos = () => { for(const c of combos) c._cleanup(); combos.length = 0; };
  wrap._cleanup = dropCombos;

  // Sorting is a view over the rows and never touches the stored order, which is load bearing: buildPairLUT takes the best specificity with a strict greater-than, so on a tie the row written first is the one that wins.
  // Reordering the text to match a column click would silently change which rule applies to a pair.
  // Hence a third state, "file", to get back to the order that decides ties, and hence sorting a copy while edits go on reaching the same row objects.
  let sortKey = st.sortKey || null, sortDir = st.sortDir || 1;
  const ruleLabel = r => r.pl === '' || r.pl === '1' ? 'default'
    : r.pl === '0' ? 'frozen' : r.pl;
  const sorted = list => {
    if(!sortKey) return list;
    const num = sortKey === 'pm' || sortKey === 'wm';
    const key = r => sortKey === 'pl' ? ruleLabel(r) : r[sortKey];
    return [...list].sort((a, b) => {
      const x = key(a), y = key(b);
      const c = num ? (+x || 0) - (+y || 0) : String(x).localeCompare(String(y));
      return c * sortDir;
    });
  };
  const bumpSort = k => {
    if(sortKey !== k){ sortKey = k; sortDir = 1; }
    else if(sortDir === 1) sortDir = -1;
    else { sortKey = null; sortDir = 1; }     // back to file order
    st.sortKey = sortKey; st.sortDir = sortDir;
    draw();
  };
  // Every pathway this table names between two real populations, once.
  const allPairs = () => {
    const seen = new Set(), pairs = [];
    for(const r of [...contributed.map(c => [c.from, c.to, c.pl]),
                    ...real.map(r2 => [r2.pre, r2.post, r2.pl])]){
      const [a, b] = r;
      if(!a || !b || a === '*' || b === '*') continue;
      const k = a + '→' + b;
      if(seen.has(k)) continue;
      seen.add(k); pairs.push([a, b, r[2]]);
    }
    return pairs;
  };
  // An edit redraws the table, and a redraw made inside the change event destroys the cell the browser is about to move focus to, so Tab out of a cell would go nowhere.
  // The redraw waits for focus to settle, then finds the cell that now holds it by its row and column and gives it back.
  const redraw = () => setTimeout(() => {
    const a = document.activeElement;
    const key = a && a.dataset ? a.dataset.cell : null;
    draw();
    if(key){ const el = body.querySelector(`[data-cell="${key}"]`);
      if(el){ el.focus(); if(el.select) el.select(); } }
  }, 0);
  const draw = () => {
    dropCombos();
    const litMap = lit();
    body.textContent = '';
    const t = el_('table', 'pt');
    t.appendChild(colgroup());
    const hr = el_('tr', 'pt-hr');
    for(const [h, k, tip] of [
        ['from', 'pre', 'presynaptic population, or E, I, * for any'],
        ['to', 'post', 'postsynaptic population'],
        ['p', 'pm', 'probability multiplier'],
        ['w', 'wm', 'weight multiplier'],
        ['learns by', 'pl', 'which plasticity rule applies'],
        ['', null, ''], ['', null, '']]){
      const th = el_('th', null, h);
      if(k){
        th.classList.add('sortable');
        if(sortKey === k){
          th.classList.add('on');
          th.appendChild(el_('span', 'pt-arrowdir', sortDir === 1 ? ' ▲' : ' ▼'));
        }
        th.title = tip + '.  Click to sort; sorting is a view only and does ' +
          'not change the stored order, which is what decides ties.';
        th.onclick = () => bumpSort(k);
      } else if(tip) th.title = tip;
      hr.appendChild(th);
    }
    // The eye column's header is the master switch: one click draws every pathway the table names, up to PATH_MAX, and the next clears them all.
    if(g && g.togglePathway){
      const th = hr.children[5];
      const nLit = Object.keys(litMap).length;
      th.classList.add('pt-eye');
      th.appendChild(eyeIcon(nLit ? '#ddd' : null, nLit ? '#ddd' : null));
      if(nLit) th.classList.add('on');
      th.title = nLit ? 'stop drawing every pathway' : 'draw every pathway in this table';
      th.onclick = () => nLit ? g.clearPathways() : g.showAllPathways(allPairs());
    }
    t.appendChild(hr);

    // The rows one project node contributes are one line of the graph and many of the tissue (a repeat's copies, a pattern's matches), so they sit under one summary row, collapsed, with the names' common shape, the count, and an eye that lights them all; the row opens into the per-copy rows and closes again.
    const patternOf = names => {
      const u = [...new Set(names)];
      if(u.length === 1) return u[0];
      let a = 0; while(u.every(x => x[a] === u[0][a]) && a < u[0].length) a++;
      let b = 0; while(u.every(x => x.length > a + b && x[x.length-1-b] === u[0][u[0].length-1-b])) b++;
      return u[0].slice(0, a) + '*' + (b ? u[0].slice(u[0].length - b) : '');
    };
    const groups = new Map();
    for(const c of contributed){
      const key = c.kind === 'project' && c.node && c.node.id !== undefined ? 'n' + c.node.id : 'row' + groups.size + Math.random();
      if(!groups.has(key)) groups.set(key, []);
      groups.get(key).push(c);
    }
    st.open = st.open || {};
    const drawRow = (c, indent) => {
      const off = c.node.on === false;
      const tr = el_('tr', 'pt-ext' + (off ? ' pt-off' : '') + (indent ? ' pt-copy' : ''));
      tr.appendChild(el_('td', 'pt-tag', c.from || '?'));
      tr.appendChild(el_('td', 'pt-tag', c.to || '?'));
      const note = el_('td', 'pt-note', c.pm); note.colSpan = 2;
      tr.appendChild(note);
      tr.appendChild(el_('td', 'pt-rule', off ? 'bypassed' : c.pl));
      tr.appendChild(eyeCell(c.from, c.to, litMap,
        g && g.togglePathway ? g.togglePathway : null, c.pl));
      tr.appendChild(el_('td', 'pt-src', c.kind === 'project' ? 'pr' : 'pl'));
      tr.title = 'declared by the ' + c.kind + ' node' +
        (off ? ', which is bypassed, so it applies nothing' : '') + '. Click to select it.';
      tr.onclick = () => { if(g && g.select) g.select(c.node); };
      t.appendChild(tr);
    };
    for(const [key, list] of groups){
      if(list.length < 2){ drawRow(list[0], false); continue; }
      const c0 = list[0], off = c0.node.on === false, open = !!st.open[key];
      const tr = el_('tr', 'pt-ext pt-group' + (off ? ' pt-off' : '') + (open ? ' open' : ''));
      tr.appendChild(el_('td', 'pt-tag', patternOf(list.map(c => c.from))));
      tr.appendChild(el_('td', 'pt-tag', patternOf(list.map(c => c.to))));
      const note = el_('td', 'pt-note', list.length + ' pathways via projection'); note.colSpan = 2;
      tr.appendChild(note);
      tr.appendChild(el_('td', 'pt-rule', off ? 'bypassed' : c0.pl));
      // one eye for the group: on when every pathway in it is drawn
      const litN = list.filter(c => litMap[c.from + '→' + c.to]).length;
      const td = el_('td', 'pt-eye');
      const col = litN ? litMap[(list.find(c => litMap[c.from + '→' + c.to])).from + '→' + (list.find(c => litMap[c.from + '→' + c.to])).to] : null;
      td.appendChild(eyeIcon(col && col.from, col && col.to));
      if(litN === list.length) td.classList.add('on');
      td.title = litN ? 'stop drawing these ' + list.length + ' pathways' : 'draw all ' + list.length + ' pathways in the viewer';
      if(g && g.togglePathway) td.onclick = e => { e.stopPropagation();
        for(const c of list) if(litN ? litMap[c.from + '→' + c.to] : !litMap[c.from + '→' + c.to]) g.togglePathway(c.from, c.to, c.pl); };
      tr.appendChild(td);
      tr.appendChild(el_('td', 'pt-src', open ? '▾' : '▸'));
      tr.title = 'one project node, ' + list.length + ' pathways (a repeat\'s copies or a pattern\'s matches). Click to ' +
        (open ? 'collapse' : 'expand') + '; the rows inside select the node.';
      tr.onclick = () => { st.open[key] = !open; draw(); };
      t.appendChild(tr);
      if(open) for(const c of list) drawRow(c, true);
    }

    const view = sorted(real);
    const shown = showAll ? view : view.slice(0, LIMIT);
    for(const r of shown){
      const tr = el_('tr');
      for(const k of ['pre', 'post']){
        const td = el_('td');
        const cur = r[k] || '';
        // Every population upstream, so the two ends of a row can be picked rather than spelled.
        // A tag that is merely mistyped produces a row the resolver silently never matches, since score() returns no-match and the pair falls through to the wildcard: the one failure in this table that leaves no trace anywhere.
        const known = tags.includes(cur) || ['*', 'E', 'I'].includes(cur) ||
          (cur.length > 1 && cur.endsWith('*') &&
            tags.some(t => t.toLowerCase().startsWith(cur.slice(0, -1).toLowerCase())));
        const cb = comboBox(cur, [
          { label:'populations', items:tags },
          { label:'any, or by sign', items:['*', 'E', 'I'] },
        ], v => { r[k] = v; commit(); redraw(); }, {
          disabled: locked, cls: 'pt-tag' + (known ? '' : ' pt-unknown'),
          title: known ? cur
            : 'no population upstream carries this tag, so this row never matches',
        });
        cb.querySelector('input').dataset.cell = rows.indexOf(r) + ':' + k;
        combos.push(cb);
        td.appendChild(cb); tr.appendChild(td);
      }
      for(const k of ['pm', 'wm']){
        const td = el_('td');
        const inp = document.createElement('input');
        inp.type = 'text'; inp.value = r[k]; inp.disabled = locked;
        inp.className = 'pt-num';
        inp.title = k === 'pm' ? 'probability multiplier' : 'weight multiplier';
        inp.dataset.cell = rows.indexOf(r) + ':' + k;
        inp.onchange = () => { r[k] = inp.value.trim(); commit(); redraw(); };
        td.appendChild(inp); tr.appendChild(td);
      }
      const td = el_('td');
      const cur = r.pl === '' || r.pl === '1' ? 'default'
        : r.pl === '0' ? 'frozen' : r.pl;
      const isBuiltin = RULE_BUILTIN.includes(cur);
      const declared = isBuiltin || ruleNames.includes(cur);
      const cbr = comboBox(cur, [
        { label:'built in', items:RULE_BUILTIN },
        { label:'declared by plasticity nodes', items:ruleNames },
      ], v => {
        r.pl = v === 'default' ? '1' : v === 'frozen' ? '0' : v;
        commit(); redraw();
      }, {
        disabled: locked,
        cls: (declared ? '' : 'pt-unknown ') + (declared && !isBuiltin ? 'pt-named' : ''),
        title: declared ? cur : 'no plasticity node declares a rule with this name',
      });
      cbr.querySelector('input').dataset.cell = rows.indexOf(r) + ':pl';
      combos.push(cbr);
      td.appendChild(cbr); tr.appendChild(td);
      // Show this pathway in the viewer.
      // It gets its own cell rather than living on the row, because every cell of the row is filled edge to edge by an input or a combobox and the only bare surface left is two pixels of cell padding: a row click is reachable from a script and essentially not with a mouse.
      tr.appendChild(eyeCell(r.pre, r.post, litMap,
        g && g.togglePathway ? g.togglePathway : null, r.pl));
      const del = el_('td', 'pt-del', '×');
      del.title = 'remove this row';
      if(!locked) del.onclick = () => {
        rows.splice(rows.indexOf(r), 1); real.splice(real.indexOf(r), 1); commit(); draw();
      };
      tr.appendChild(del);
      t.appendChild(tr);
    }
    body.appendChild(t);

    const foot = el_('div', 'pt-foot');
    if(!locked){
      const add = el_('button', 'icb');
      add.innerHTML = icon('add') + '<span>row</span>';
      add.onclick = () => { const r = { pre:tags[0] || '*', post:tags[0] || '*',
        pm:'1', wm:'1', pl:'' };
        rows.push(r); real.push(r); showAll = true; st.showAll = true; commit(); draw(); };
      foot.appendChild(add);
    }
    if(real.length > LIMIT){
      const more = el_('button', null,
        showAll ? 'show first ' + LIMIT : (real.length - LIMIT) + ' more rows');
      more.onclick = () => { showAll = !showAll; st.showAll = showAll; draw(); };
      foot.appendChild(more);
    }
    if(g && g.togglePathway){
      const nLit = Object.keys(litMap).length;
      const all = el_('button', null, 'show all');
      // "all" of two hundred rows is not a thing anyone can read, and each pathway costs a round of queries, so this takes the rows that name two real populations and stops at the color limit, saying so.
      all.title = 'draw every pathway this table names, up to the color limit';
      all.onclick = () => g.showAllPathways(allPairs());
      foot.appendChild(all);
      if(nLit){
        const none = el_('button', 'pt-clear', 'clear ' + nLit);
        none.title = 'stop drawing every pathway';
        none.onclick = () => g.clearPathways();
        foot.appendChild(none);
      }
    }
    if(!opts.roomy){
      const pop = el_('button', null, 'expand');
      pop.title = 'open the table in a larger panel';
      pop.onclick = () => openPairPopout(node, p, onEdit, draw);
      foot.appendChild(pop);
    }
    const txt = el_('button', null, 'text view');
    txt.onclick = () => {
      body.textContent = '';
      const ta = document.createElement('textarea');
      ta.value = node.params[p.k]; ta.rows = 12; ta.disabled = locked;
      ta.onchange = () => { node.params[p.k] = ta.value; onEdit(node);
        rows.length = 0; rows.push(...parsePairs(ta.value));
        real.length = 0; real.push(...rows.filter(r => !r.bad)); };
      body.appendChild(ta);
      const back = el_('button', null, 'table view');
      back.onclick = draw;
      const f2 = el_('div', 'pt-foot'); f2.appendChild(back); body.appendChild(f2);
    };
    foot.appendChild(txt);
    const n = el_('span', 'pt-count', real.length + ' rows');
    foot.appendChild(n);
    body.appendChild(foot);
  };
  draw();
  return wrap;
}

// The properties panel is a column, and a pair table is a wide thing with several hundred rows in it.
// Rather than make the column wider for one control, the table can be opened over the app the way the documentation is, with room for every row at once.
function openPairPopout(node, p, onEdit, refresh){
  const dlg = el_('div', 'dlg pairpop');
  const box = el_('div', 'dlgbox');
  const head = el_('div', 'pphead');
  head.appendChild(el_('b', null, 'PAIR TABLE'));
  head.appendChild(el_('span', 'ppsub',
    (node.name || node.type) + ', and every rule reaching it'));
  const close = el_('button', null, 'close');
  head.appendChild(close);
  box.appendChild(head);
  const holder = el_('div', 'ppbody');
  const table = pairTableUI(node, p, onEdit, { roomy:true });
  holder.appendChild(table);
  box.appendChild(holder);
  dlg.appendChild(box);
  document.body.appendChild(dlg);

  // The panel behind is showing the same rows, so it is redrawn on the way out rather than left describing the table as it was before this opened.
  const done = () => {
    if(table._cleanup) table._cleanup();
    dlg.remove();
    document.removeEventListener('keydown', onKey);
    if(refresh) refresh();
  };
  const onKey = e => { if(e.key === 'Escape'){ e.preventDefault(); done(); } };
  close.onclick = done;
  dlg.onclick = e => { if(e.target === dlg) done(); };
  document.addEventListener('keydown', onKey);
}

// nodes: the rest of the graph, so a row can hide itself when the mechanism it belongs to is switched off somewhere else.
// Consolidation is the case that needs it: a plasticity node can turn it on for one pathway while the checkpoint's own switch reads zero, and its timings apply either way. hueLookup is the page's lookup for a node's current hue (its tag's locked hue), for the swatch beside an automatic hue setting.
let hueLookup = null;
export function setHueLookup(fn){ hueLookup = fn; }
export function renderProps(el, node, onEdit, nodes){
  // What the last panel put outside itself has to come back: a combo box appends its dropdown to the document body so it can escape the pane's scroll, and the chart holds a resize observer and a visibility listener.
  // Clearing innerHTML orphans all of it, so every render would leave a dead list in the body, one per click on a chart panel.
  for(const kid of el.querySelectorAll('*')) if(kid._cleanup) kid._cleanup();
  el.innerHTML = '';
  // A chart draws its parameters, so changing one has to redraw it.
  // Only the drawing is redone, on the frame after the edit, because the edit can be a slider still under the pointer.
  if(node && node.type === 'chart'){
    const upstream = onEdit;
    let queued = false;
    onEdit = n => {
      upstream(n);
      if(queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; if(chartRepaint) chartRepaint(); });
    };
  }
  el.classList.remove('locked');
  let nameInput = null;
  if(!node){ el.innerHTML = '<div class="empty">no node selected</div>'; return; }
  const def = NODE_DEFS[node.type];
  const title = document.createElement('div'); title.className = 'ptitle';
  const sw = document.createElement('span'); sw.className = 'swatch'; sw.style.background = def.color;
  title.appendChild(sw); title.appendChild(document.createTextNode(def.title.toUpperCase()));
  el.appendChild(title);
  // Name first, above the parameters: it is not a parameter, it never reaches a computation, and it is deliberately outside the memo signature so renaming a node in a full-density graph does not force rewiring.
  {
    const row = document.createElement('div'); row.className = 'prow';
    const lab = document.createElement('label'); lab.textContent = 'name';
    const inp = document.createElement('input'); inp.type = 'text';
    inp.value = node.name || '';
    inp.placeholder = 'unnamed';
    // If this node carries a population tag and the tag is still following the name, a rename moves the tag with it.
    // Whether it is following is read once, here, before editing starts, because oninput mutates the name as the user types.
    // The name itself never reaches a computation (onEdit(null)); the tag does, so the tag is committed on change, which rewires and redraws the panel with the new tag shown.
    const tagP = def.params.find(p => p.k === 'tag' && p.identity);
    const followed = tagP && tagFollowsName(node.name, node.params.tag);
    inp.oninput = () => { node.name = inp.value; onEdit(null); };
    inp.onchange = () => {
      if(tagP && followed){
        const t = tagToken(node.name);
        if(node.params.tag !== t){
          const was = node.params.tag;
          node.params.tag = t; onEdit(node);
          retagReferences(nodes, was, t, onEdit);
        }
      }
    };
    nameInput = inp;
    row.appendChild(lab); row.appendChild(inp);
    el.appendChild(row);
  }
  // Tabs, when a node declares groups on its parameters.
  // This is general: any node that tags its params gets the same treatment, and one that does not is unchanged.
  const groupsIn = [];
  for(const p of def.params){
    if(p.cond && !p.cond(node.params, node, nodes)) continue;
    const g = p.g || '';
    if(g && !groupsIn.includes(g)) groupsIn.push(g);
  }
  const useTabs = groupsIn.length > 1;
  let active = groupsIn[0];
  if(useTabs){
    // remembered per node type, so clicking between two checkpoints does not throw away which tab was being read
    if(tabFor[node.type] && groupsIn.includes(tabFor[node.type]))
      active = tabFor[node.type];
    const strip = document.createElement('div');
    strip.className = 'ptabs';
    for(const g of groupsIn){
      const b = document.createElement('button');
      b.textContent = g;
      b.className = 'ptab' + (g === active ? ' on' : '');
      b.onclick = () => { tabFor[node.type] = g; renderProps(el, node, onEdit, nodes); };
      strip.appendChild(b);
    }
    el.appendChild(strip);
  }
  for(const p of def.params){
    // conditionally relevant params hide when inactive; their values persist
    if(p.cond && !p.cond(node.params, node, nodes)) continue;
    if(useTabs && (p.g || groupsIn[0]) !== active) continue;
    const row = document.createElement('div'); row.className = 'prow';
    // display-only settings stay live under the run lock (see the lock pass)
    if(p.view || def.view) row.classList.add('viewctl');
    const lab = document.createElement('label'); lab.textContent = p.label; row.appendChild(lab);
    annotate(lab, node.type, p.k);
    { const h = lackingEngine(p);
      if(h){
        row.classList.add('lacking');
        const t = Array.isArray(p.term) ? p.term : [p.term];
        row.title = 'the ' + h.engine + ' engine does not implement ' + t.map(termLabel).join(' or ') +
          '; a network that uses it is not run there';
      } }
    if(p.t === 'vec3'){
      for(let axis=0; axis<3; axis++){
        const inp = num(node.params[p.k][axis], p, v => { node.params[p.k][axis] = v; onEdit(node); });
        row.appendChild(inp);
      }
    } else if(p.t === 'action'){
      const btn = document.createElement('button');
      const ACTION_ICON = { chartPng:'save', chartCsv:'save', exportNet:'save', exportBrian:'save', brainSave:'save', brainLoad:'load', runs:'runs', train:'play', chooseFolder:'folder', meshReload:'load', pointsReload:'load', connectionsReload:'load', expandPops:'add', collapsePops:'remove' };
      const nm = ACTION_ICON[p.action];
      if(nm){ btn.innerHTML = icon(nm) + '<span>' + p.label + '</span>'; btn.classList.add('icb'); }
      else btn.textContent = p.label;
      btn.onclick = () => { if(actionHandler) actionHandler(p.action, node); };
      lab.textContent = '';
      row.appendChild(btn);
    } else if(p.t === 'str'){
      const inp = document.createElement('input');
      inp.type = 'text'; inp.value = node.params[p.k] || '';
      // a population tag is one token: the pair table is split on whitespace, so a tag with a space in it can never be written into a row
      inp.onchange = () => {
        const was = node.params[p.k];
        node.params[p.k] = p.identity ? tagToken(inp.value) : inp.value;
        if(p.identity) inp.value = node.params[p.k];
        onEdit(node);
        if(p.identity) retagReferences(nodes, was, node.params[p.k], onEdit);
      };
      row.appendChild(inp);
    } else if(p.t === 'sweeps'){
      lab.textContent = '';
      row.classList.add('wide');
      row.appendChild(sweepUI(node, p, onEdit, nodes));
    } else if(p.t === 'tag'){
      // A population picker: every scatter tag in the graph, the two cell classes, or all.
      // A value that names no scatter stays selectable and says so, since a scene can arrive with a tag its scatters lost.
      const sel = document.createElement('select');
      const cur = String(node.params[p.k] || '').trim();
      // a project node with both ports wired names populations that arrive on those ports: from is chosen among port A's tags, to among port B's
      const portTags = k => { const c = node.inputs && node.inputs[k] && (nodes || []).find(m => m.id === node.inputs[k].id);
        const cache = c && c._cache; return cache && cache.tags ? [...new Set(Object.values(cache.tags))] : null; };
      const twoPort = node.type === 'project' && node.inputs && node.inputs[1];
      const tags = twoPort && (p.k === 'from' || p.k === 'to')
        ? (portTags(p.k === 'from' ? 0 : 1) || graphTags(nodes)) : graphTags(nodes);
      const opts = [['', 'all'], ['E', 'E (excitatory)'], ['I', 'I (inhibitory)'],
        ...tags.map(t => [t, t])];
      if(cur && !opts.some(o => o[0].toLowerCase() === cur.toLowerCase()))
        opts.push([cur, cur + (cur.includes('*') ? ' (pattern)' : ' (no scatter has this tag)')]);
      // a pattern with * matches many tags at once (repeat copies): typed, since no list can hold it
      opts.push(['__pattern', 'a pattern with *']);
      for(const [v, text] of opts){
        const op = document.createElement('option');
        op.value = v; op.textContent = text; sel.appendChild(op);
      }
      sel.value = opts.find(o => o[0].toLowerCase() === cur.toLowerCase())?.[0] ?? '';
      sel.onchange = () => {
        if(sel.value === '__pattern'){
          const inp = document.createElement('input');
          inp.type = 'text'; inp.value = cur.includes('*') ? cur : (cur ? cur.replace(/\d+/, '*') : '*');
          inp.placeholder = 'tier1.bulb*.RS';
          inp.onchange = () => { node.params[p.k] = inp.value.trim(); onEdit(node); };
          sel.replaceWith(inp); inp.focus();
          return;
        }
        node.params[p.k] = sel.value; onEdit(node);
      };
      row.appendChild(sel);
    } else if(p.t === 'pairs'){
      lab.textContent = '';
      row.classList.add('wide');   // .prow.wide: full width, label suppressed
      row.appendChild(pairTableUI(node, p, onEdit));
    } else if(p.t === 'text'){
      const ta = document.createElement('textarea');
      ta.value = node.params[p.k]; ta.rows = 6;
      ta.onchange = () => { node.params[p.k] = ta.value; onEdit(node); };
      row.appendChild(ta);
    } else if(p.t === 'select'){
      const sel = document.createElement('select');
      // a hidden option (a type row that a scene reaches through the cell type node) is listed only while it is the value
      p.options.forEach((o,i) => { if(p.hideOption && p.hideOption(i) && i !== +node.params[p.k]) return;
        const op = document.createElement('option');
        op.value = i; op.textContent = o; sel.appendChild(op); });
      sel.value = node.params[p.k];
      // A select can carry the rest of its node with it: p.apply writes the other parameters that go with the choice.
      // That is what makes a preset a preset rather than a label, and the panel re-renders after it so the fields show what was written rather than what was there.
      sel.onchange = () => { node.params[p.k] = +sel.value;
        if(p.apply) p.apply(node.params, +sel.value);
        onEdit(node);
        renderProps(el, node, onEdit, nodes); };   // dependent rows may appear or hide
      row.appendChild(sel);
    } else if((p.min !== undefined && p.max !== undefined) || p.s){
      // slider over the hard bounds, or over a soft range (p.s) that typing in the numeric field can exceed
      const lo = p.s ? p.s[0] : p.min, hi = p.s ? p.s[1] : p.max;
      const set = v => { node.params[p.k] = v; onEdit(node); };
      const inp = num(node.params[p.k], p, set);
      inp.classList.add('narrow');
      const rng = document.createElement('input');
      rng.type = 'range'; rng.min = lo; rng.max = hi;
      rng.step = p.t === 'int' ? 1 : (hi - lo)/200;
      rng.value = Math.max(lo, Math.min(hi, node.params[p.k]));
      rng.oninput = () => { inp.value = rng.value; set(+rng.value); };
      inp.addEventListener('change', () => {
        rng.value = Math.max(lo, Math.min(hi, +inp.value)); });
      row.appendChild(rng); row.appendChild(inp);
      // a hue setting shows its color: the chosen hue, or the one the scene has locked for the node's tag when the setting is automatic
      if(p.k === 'hue'){
        const sw = document.createElement('span'); sw.className = 'swatch';
        const paint = () => { const hv = +inp.value; const h = hv >= 0 ? hv : (hueLookup ? hueLookup(node) : -1);
          sw.style.background = h >= 0 ? `hsl(${h},75%,62%)` : 'transparent';
          sw.title = hv >= 0 ? 'hue ' + hv : (h >= 0 ? 'automatic, currently hue ' + h : 'automatic'); };
        paint(); rng.addEventListener('input', paint); inp.addEventListener('input', paint); inp.addEventListener('change', paint);
        row.appendChild(sw);
      }
    } else {
      row.appendChild(num(node.params[p.k], p, v => { node.params[p.k] = v; onEdit(node); }));
    }
    el.appendChild(row);
  }
  if(!def.params.length) el.insertAdjacentHTML('beforeend', '<div class="empty">no parameters</div>');
  // What this probe last measured, on the probe itself.
  // Which plasticity mechanisms are actually running.
  // Every one of these is a parameter that reads 0 when it is off, and a row saying "triplet A3+ 0" looks like a tuning value at rest rather than an absent mechanism.
  if(node.type === 'checkpoint'){
    const p = node.params;
    const terms = [
      // A training run keeps the node's switch as it is (train.js), so off here is off in a run too.
      ['pair STDP', (p.plast|0) === 1, 'aP ' + p.aP + ' aM ' + p.aM],
      ['inhibitory homeostasis', +p.iEta > 0, 'target ' + p.iRho + ' Hz'],
      ['triplet', +p.trip > 0, 'A3+ ' + p.trip],
      ['heterosynaptic', +p.het > 0, 'beta ' + p.het],
      ['transmitter', +p.tin > 0, 'delta ' + p.tin],
      ['consolidation', +p.cons > 0, 'rate ' + p.cons],
      ['short-term', +p.stp > 0, 'U ' + p.stpU],
      ['synaptic scaling', (p.scale|0) === 1, 'eta ' + p.sEta],
    ];
    const on = terms.filter(t => t[1]);
    const box = document.createElement('div');
    box.className = 'pmetrics';
    box.innerHTML = '<div class="ptitle2">plasticity actually running</div>' +
      terms.map(([k, active, detail]) =>
        '<div class="prow2 ' + (active ? 'on' : 'offterm') +
        '"><span>' + k + '</span><b>' + (active ? detail : 'off') +
        '</b></div>').join('') +
      (on.length <= 2
        ? '<div class="empty" style="margin-top:6px">Only the pair rule and ' +
          'its homeostat. Assemblies are not expected to form from these ' +
          'alone (Zenke et al. 2015).</div>' : '');
    el.appendChild(box);
  }
  if(node.type === 'chart'){
    // Below the parameters, because the plot is what this node is for and the parameters are how it is aimed.
    const box = chartUI(node, onEdit, () => renderProps(el, node, onEdit));
    el.appendChild(box);
    if(box._mount) box._mount();      // it has a size now, so it can draw
  }
  // Which neurons this probe reads, by population.
  // A probe is a mask over a network and its label is free text, so nothing on the panel said whether it covers what its name claims.
  if(node.type === 'probe'){
    const info = probeSetFor ? probeSetFor(node) : null;
    const box = document.createElement('div');
    box.className = 'pprobe';
    if(!info) box.innerHTML = '<div class="empty">compute the graph to see which ' +
      'neurons this probe reads</div>';
    else {
      box.innerHTML = '<div class="ptitle2">reads ' + info.total.toLocaleString() +
        ' neuron' + (info.total === 1 ? '' : 's') + '</div>' +
        info.rows.map(r => '<div class="prow2"><span><span class="pdot" style="background:' +
          hueCss(r.hue) + '"></span>' + esc(r.tag) + '</span><span>' +
          r.n.toLocaleString() + ' of ' + r.of.toLocaleString() + '</span></div>').join('');
    }
    el.appendChild(box);
  }
  // What this stimulus drives.
  // The set comes from a mask, a radius and a population tag together, so no single field on the panel says it.
  // Which pathways this rule actually governs.
  // A named rule is referred to from a pair table somewhere else in the graph, so the node that declares it says nothing about where it lands.
  if(node.type === 'plasticity'){
    const g = graphFor ? graphFor() : null;
    const name = String(node.params.name || '').trim();
    const from = String(node.params.from || '').trim();
    const to = String(node.params.to || '').trim();
    const pairs = [];
    if(from && to) pairs.push([from, to, name]);
    for(const other of (g && g.nodes ? g.nodes() : [])){
      const tbl = other.params && typeof other.params.table === 'string' ? other.params.table : '';
      if(!tbl) continue;
      for(const line of tbl.split(String.fromCharCode(10))){
        const f = line.split('#')[0].trim().split(/\s+/).filter(Boolean);
        if(f.length < 5 || f[4] !== name) continue;
        if(!f[0] || !f[1] || f[0] === '*' || f[1] === '*') continue;
        if(!pairs.some(q => q[0] === f[0] && q[1] === f[1])) pairs.push([f[0], f[1], name]);
      }
    }
    const box = document.createElement('div');
    box.className = 'pprobe';
    box.innerHTML = '<div class="ptitle2">governs ' + pairs.length + ' pathway' +
      (pairs.length === 1 ? '' : 's') + '</div>' +
      (pairs.length
        ? pairs.map(q => '<div class="prow2"><span>' + esc(q[0]) + ' to ' + esc(q[1]) +
            '</span><span></span></div>').join('')
        : '<div class="empty">no pair table row names this rule, so it applies ' +
          'nowhere</div>');
    if(pairs.length && g && g.showAllPathways){
      const b = document.createElement('button');
      b.textContent = 'DRAW THESE PATHWAYS';
      b.onclick = () => g.showAllPathways(pairs);
      box.appendChild(b);
    }
    el.appendChild(box);
  }
  if(node.type === 'stimulus'){
    const info = driveSetFor ? driveSetFor(node) : null;
    const box = document.createElement('div');
    box.className = 'pprobe';
    if(!info) box.innerHTML = '<div class="empty">compute the graph to see which ' +
      'neurons this drives</div>';
    else box.innerHTML = '<div class="ptitle2">drives ' + info.total.toLocaleString() +
      ' neuron' + (info.total === 1 ? '' : 's') + '</div>' +
      info.rows.map(r => '<div class="prow2"><span><span class="pdot" style="background:' +
        hueCss(r.hue) + '"></span>' + esc(r.tag) + '</span><span>' +
        r.n.toLocaleString() + ' of ' + r.of.toLocaleString() + '</span></div>').join('');
    el.appendChild(box);
  }
  // Where this input lands.
  // The monitor above shows what is being sent; this says how many cells receive it and over how many channels.
  if(node.type === 'input'){
    const info = inputSetFor ? inputSetFor(node) : null;
    const box = document.createElement('div');
    box.className = 'pprobe';
    if(!info) box.innerHTML = '<div class="empty">compute the graph to see where ' +
      'this input lands</div>';
    else box.innerHTML = '<div class="ptitle2">lands on ' + info.cells.toLocaleString() +
      ' neuron' + (info.cells === 1 ? '' : 's') + '</div>' +
      '<div class="prow2"><span>channels</span><span>' + info.chans +
        (info.sheet ? ' (' + info.cols + ' by ' + info.rows + ')' : '') + '</span></div>' +
      '<div class="prow2"><span>connections</span><span>' +
        info.idx.length.toLocaleString() + '</span></div>' +
      '<div class="prow2"><span>per channel</span><span>' +
        (info.idx.length/Math.max(1, info.chans)).toFixed(1) + ' cells</span></div>' +
      // what is holding the input back from the whole network
      (info.source === -2 ? '<div class="prow2"><span>carries</span><span>' +
        (info.sense === 'sound' ? 'sound' : 'sight') + '</span></div>' : '') +
      '<div class="prow2"><span>population</span><span>' +
        (info.tag ? esc(info.tag) : 'all') + '</span></div>' +
      '<div class="prow2"><span>region mask</span><span>' +
        (info.masked ? 'on side input' : 'none') + '</span></div>';
    el.appendChild(box);
  }
  // The conjunctive cells this analysis named, per item.
  // They are the object every binding number is about.
  if(node.type === 'analysis'){
    const info = conjSetFor ? conjSetFor(node) : null;
    const box = document.createElement('div');
    box.className = 'pprobe';
    if(!info) box.innerHTML = '<div class="empty">conjunctive cells appear once a ' +
      'training run has calibrated them</div>';
    else box.innerHTML = '<div class="ptitle2">conjunctive cells in ' + esc(info.label) +
      ' · ' + info.items.length + ' of ' + info.nItems + ' items</div>' +
      info.items.map(it => '<div class="prow2"><span><span class="pdot" style="background:' +
        hueCss(Math.round((it.i*137.507764) % 360)) + '"></span>' + esc(it.name) +
        '</span><span>' + it.n + ' cells</span></div>').join('');
    el.appendChild(box);
  }
  if(node.type === 'probe' && latestFor){
    const m = latestFor(node.params.label || '');
    const box = document.createElement('div');
    box.className = 'pmetrics';
    box.innerHTML = m
      ? '<div class="ptitle2">last checkpoint  sim ' + m.simMin + ' min</div>' +
        rowsFor(m)
      : '<div class="empty">no measurements yet; they appear once a training ' +
        'run reaches its first checkpoint</div>';
    el.appendChild(box);
  }
  if(locked){
    // disable in one pass over what was built, so a new control cannot be added later that quietly stays live during a run
    el.classList.add('locked');
    for(const c of el.querySelectorAll('input, select, textarea, button')){
      if(c === nameInput) continue;
      // a tab only chooses which settings are shown; reading them during a run is the point of a read-only panel
      if(c.classList.contains('ptab')) continue;
      // a view setting changes what is shown, not what runs: a chart is for watching a run develop, and choosing the plot is the whole point
      if(c.closest('.viewctl')) continue;
      c.disabled = true;
      c.title = lockWhy + ', so settings are read only until it finishes';
    }
    const note = document.createElement('div');
    note.className = 'empty lockmsg';
    note.textContent = lockWhy + ': settings are read only';
    el.appendChild(note);
  }
}

// A group is not a node: it carries a label and a hue, neither of which reaches a computation, so it gets its own small panel rather than a node definition that would have to pretend to have parameters and ports. onEdit(commit) is called on every keystroke with commit false and once the edit settles with commit true, so history records the result rather than every letter of a rename.
const HUE_PRESETS = [[205,'visual'], [130,'association'], [25,'auditory'],
  [280,'plumbing'], [55,'drive'], [0,'checkpoint']];

export function renderGroupProps(el, group, onEdit, onUngroup, onRaise, onFit){
  el.innerHTML = '';
  if(!group){ el.innerHTML = '<div class="empty">no group selected</div>'; return; }
  const hue = () => group.hue === undefined ? 210 : group.hue;
  const paint = () => `hsl(${hue()} 45% 55%)`;

  const title = document.createElement('div'); title.className = 'ptitle';
  const sw = document.createElement('span'); sw.className = 'swatch';
  sw.style.background = paint();
  title.appendChild(sw); title.appendChild(document.createTextNode('GROUP'));
  el.appendChild(title);

  const row = text => {
    const r = document.createElement('div'); r.className = 'prow';
    const l = document.createElement('label'); l.textContent = text;
    r.appendChild(l); el.appendChild(r); return r;
  };

  const nameIn = document.createElement('input');
  nameIn.type = 'text'; nameIn.value = group.label || '';
  nameIn.placeholder = 'unnamed';
  nameIn.oninput = () => { group.label = nameIn.value; onEdit(false); };
  nameIn.onchange = () => { group.label = nameIn.value; onEdit(true); };
  row('name').appendChild(nameIn);

  const swatches = [];
  const setHue = (h, commit) => {
    group.hue = h;
    sw.style.background = paint();
    rng.value = h; numIn.value = h;
    for(const [b, ph] of swatches) b.classList.toggle('on', ph === h);
    onEdit(commit);
  };
  const colRow = row('hue');
  const rng = document.createElement('input');
  rng.type = 'range'; rng.min = 0; rng.max = 360; rng.step = 1; rng.value = hue();
  rng.oninput = () => setHue(+rng.value, false);
  rng.onchange = () => setHue(+rng.value, true);
  const numIn = document.createElement('input');
  numIn.type = 'number'; numIn.step = 1; numIn.value = hue();
  numIn.classList.add('narrow');
  numIn.onchange = () => {
    let v = parseFloat(numIn.value); if(isNaN(v)) v = 210;
    setHue(Math.max(0, Math.min(360, Math.round(v))), true);
  };
  colRow.appendChild(rng); colRow.appendChild(numIn);

  const preRow = row('preset');
  const wrap = document.createElement('div'); wrap.className = 'gswatches';
  for(const [h, name] of HUE_PRESETS){
    const b = document.createElement('button');
    b.style.background = `hsl(${h} 45% 55%)`;
    b.title = name;
    if(h === hue()) b.classList.add('on');
    b.onclick = () => setHue(h, true);
    swatches.push([b, h]);
    wrap.appendChild(b);
  }
  preRow.appendChild(wrap);

  // Depth.
  // Hand-made groups overlap on purpose, so which one reads as the container has to be sayable: the higher number paints last and so sits in front.
  // Front and back step one past the current extremes rather than clamping to a range, so a group put in front stays in front.
  const zRow = row('depth');
  const zIn = document.createElement('input');
  zIn.type = 'number'; zIn.step = 1; zIn.value = group.z || 0;
  zIn.classList.add('narrow');
  zIn.onchange = () => {
    let v = parseFloat(zIn.value); if(isNaN(v)) v = 0;
    group.z = Math.round(v); zIn.value = group.z; onEdit(true);
  };
  const front = document.createElement('button'); front.textContent = 'front';
  front.onclick = () => { zIn.value = onRaise(true); onEdit(true); };
  const back = document.createElement('button'); back.textContent = 'back';
  back.onclick = () => { zIn.value = onRaise(false); onEdit(true); };
  zRow.appendChild(zIn); zRow.appendChild(front); zRow.appendChild(back);

  // Size.
  // The backdrop is resized by dragging its edges, which adds room around the members without moving them; this puts the numbers where they can be read and typed, and closes the box back onto the nodes.
  const pad = () => (group.pad = { l:0, t:0, r:0, b:0, ...(group.pad || {}) });
  const sizeRow = row('room');
  sizeRow.classList.add('wrap');
  const padIn = {};
  for(const side of ['l', 't', 'r', 'b']){
    const i = document.createElement('input');
    i.type = 'number'; i.step = 1; i.min = 0; i.value = pad()[side];
    i.className = 'padin';
    i.title = { l:'left', t:'top', r:'right', b:'bottom' }[side];
    i.onchange = () => {
      let v = parseFloat(i.value); if(isNaN(v) || v < 0) v = 0;
      pad()[side] = Math.round(v); i.value = pad()[side]; onEdit(true);
    };
    padIn[side] = i; sizeRow.appendChild(i);
  }
  const fit = document.createElement('button'); fit.textContent = 'fit';
  fit.onclick = () => {
    onFit();
    for(const side of ['l', 't', 'r', 'b']) padIn[side].value = 0;
    onEdit(true);
  };
  sizeRow.appendChild(fit);

  const info = document.createElement('div'); info.className = 'empty';
  info.textContent = group.members.length +
    (group.members.length === 1 ? ' node' : ' nodes');
  el.appendChild(info);

  const btn = document.createElement('button'); btn.textContent = 'ungroup';
  btn.onclick = () => onUngroup();
  const btnRow = row('');
  btnRow.appendChild(btn);
}

function num(value, p, set){
  const inp = document.createElement('input');
  inp.type = 'number'; inp.value = value;
  inp.step = p.t === 'int' ? 1 : 'any';
  inp.onchange = () => {
    let v = parseFloat(inp.value); if(isNaN(v)) v = 0;
    if(p.min !== undefined) v = Math.max(p.min, v);
    if(p.max !== undefined) v = Math.min(p.max, v);
    if(p.t === 'int') v = Math.round(v);
    inp.value = v; set(v);
  };
  return inp;
}

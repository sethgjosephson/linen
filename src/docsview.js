// Renders the documentation into a nav column and a content column.
//
// Used by docs.html as a standalone page and by the app as an overlay.
// Navigating away from the app to read the docs would throw away the running simulation and the wired synapses: coming back would reload the graph from storage and rewire it from scratch, which on a full-density scene is minutes of work to answer a question about a parameter.
// The overlay costs nothing to open, because this module is imported on first use rather than at startup.
import { NODE_DEFS, nodeDoc } from './nodes.js';
import { NODES, PAGES } from './docs.js';
import { ACTIONS, GROUPS, binding } from './keymap.js';

// Every category a node declares has to appear here or its nodes render nowhere: the reference and the nav both iterate CATS, so an unlisted category silently drops the node from the page even though its documentation exists.
// The battery checks this.
const CATS = [['regions','REGIONS'],['cells','CELLS'],['arrange','ARRANGE'],['celltypes','CELL TYPES'],['wiring','WIRING'],['drive','DRIVE'],['readout','READOUT'],['run','RUN'],['routing','ROUTING'],['notes','NOTES']];

function esc(s){ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function nodeSection(type, def){
  const doc = nodeDoc(type, NODES) || { blurb:'', io:'', params:{} };
  // a node a project module registered says so, with the module's name
  const from = def.plugin ? `<p class="io">from ${esc(def.plugin.module)}, a node module this project loaded</p>` : '';
  let h = `<h3 class="node" id="n-${type}"><span class="sw" style="background:${esc(def.color)}"></span>${esc(def.title)}</h3>
    <p class="io">${esc(doc.io || '')}</p>${from}<p class="blurb">${doc.blurb || ''}</p>`;
  if(def.params.length){
    h += '<table><tr><th>control</th><th>default</th><th>range</th><th>description</th></tr>';
    for(const p of def.params)
      h += `<tr><td class="k">${esc(p.label)}</td><td class="v">${esc(defaultText(p))}</td><td class="v">${esc(rangeText(p))}</td><td class="d">${(doc.params || {})[p.k] || '(undocumented)'}</td></tr>`;
    h += '</table>';
  } else h += '<p class="io">no parameters</p>';
  return h;
}

// The default as the panel shows it: a select by its option's name, a vector by its numbers, text in quotes, an action or a table with no default at all.
function defaultText(p){
  if(p.t === 'action' || p.t === 'pairs' || p.t === 'sweeps') return '';
  if(p.t === 'select') return Array.isArray(p.options) && p.options[p.def] !== undefined ? String(p.options[p.def]) : String(p.def);
  if(p.t === 'vec3') return Array.isArray(p.def) ? p.def.join(', ') : String(p.def);
  if(p.t === 'str' || p.t === 'text' || p.t === 'tag') return p.def === '' || p.def === undefined ? '' : '"' + p.def + '"';
  return p.def === undefined ? '' : String(p.def);
}
// The slider's span where there is one, else the hard bounds; nothing for a control without either.
function rangeText(p){
  if(Array.isArray(p.s)) return p.s[0] + ' to ' + p.s[1];
  const lo = p.min !== undefined, hi = p.max !== undefined;
  if(lo && hi) return p.min + ' to ' + p.max;
  if(lo) return p.min + ' and up';
  if(hi) return 'up to ' + p.max;
  return '';
}

function nodesPage(){
  let h = `<h1>node reference</h1>
    <p>Every tool in the Tab menu. Setting tables are generated from the
    running code, so they always match the app.</p>`;
  for(const [cat, label] of CATS){
    const inCat = Object.entries(NODE_DEFS).filter(([, def]) => (def.cat || 'sim') === cat);
    if(!inCat.length) continue;
    h += `<h2 class="cat" id="c-${cat}">${esc(label)}</h2>`;
    for(const [type, def] of inCat) h += nodeSection(type, def);
  }
  return h;
}

function keysPage(){
  let h = `<h1>keyboard</h1>
    <p>Bindings shown are the ones in effect. Press KEYS in the top bar to
    change them: click a binding, then press the key. Bindings are stored per
    browser. Escape is reserved and cannot be rebound.</p>`;
  for(const [g, gl] of GROUPS){
    h += `<h2>${esc(gl)}</h2><table>`;
    for(const k in ACTIONS){
      const a = ACTIONS[k];
      if(a.group !== g) continue;
      const b = esc(binding(k)) + (a.alt ? ' or ' + esc(a.alt) : '');
      h += `<tr><td class="k">${b}</td><td class="d">${esc(a.label)}</td></tr>`;
    }
    h += '</table>';
  }
  h += `<h2>pointer</h2><table>
    <tr><td class="k">click</td><td class="d">select a node, or a neuron in the viewer</td></tr>
    <tr><td class="k">double click</td><td class="d">open properties; shapes and unmasked stimulus nodes also open a viewer gizmo</td></tr>
    <tr><td class="k">drag on empty space</td><td class="d">selection box</td></tr>
    <tr><td class="k">middle drag</td><td class="d">pan</td></tr>
    <tr><td class="k">wheel</td><td class="d">zoom</td></tr>
    <tr><td class="k">right click</td><td class="d">in the graph: group, copy, paste, clone, delete</td></tr>
    <tr><td class="k">hover a setting name</td><td class="d">its description, from this reference</td></tr>
    <tr><td class="k">right click a setting name</td><td class="d">open that node's full entry here</td></tr>
    <tr><td class="k">ctrl or cmd click a wire</td><td class="d">insert a pin</td></tr>
    <tr><td class="k">alt drag</td><td class="d">clone the dragged node</td></tr>
    <tr><td class="k">drag a node onto a wire</td><td class="d">insert it into that connection</td></tr>
    <tr><td class="k">shake while dragging</td><td class="d">disconnect</td></tr>
    </table>`;
  return h;
}

// nav and content are the two columns; useHash is for the standalone page, where a deep link like #n-connect should land on that node.
export function mountDocs({ nav, content, page, useHash = false }){
  const show = id => {
    const pg = PAGES.find(p => p.id === id) || PAGES[0];
    page.innerHTML = pg.id === 'nodes' ? nodesPage()
      : pg.id === 'keys' ? keysPage() : pg.html;
    for(const a of nav.querySelectorAll('a')) a.classList.toggle('on', a.dataset.id === pg.id);
    const hash = useHash ? location.hash.slice(1) : '';
    if(hash && hash.startsWith('n-')) page.querySelector('#' + CSS.escape(hash))?.scrollIntoView();
    else content.scrollTop = 0;
  };
  const head = t => { const d = document.createElement('div'); d.className = 'head';
    d.textContent = t; nav.appendChild(d); };
  head('GUIDE');
  for(const pg of PAGES){
    const a = document.createElement('a');
    a.textContent = pg.title; a.dataset.id = pg.id; a.href = '#';
    a.onclick = e => { e.preventDefault();
      if(useHash) history.replaceState(null, '', location.pathname);
      show(pg.id); };
    nav.appendChild(a);
  }
  head('NODES');
  for(const [cat, label] of CATS){
    const inCat = Object.entries(NODE_DEFS).filter(([, def]) => (def.cat || 'sim') === cat);
    if(!inCat.length) continue;
    const sub = document.createElement('div'); sub.className = 'sub'; sub.textContent = label; nav.appendChild(sub);
    for(const [type, def] of inCat){
        const a = document.createElement('a');
        const sw = document.createElement('span'); sw.style.background = def.color;
        a.appendChild(sw); a.appendChild(document.createTextNode(def.title));
        a.dataset.id = 'nodes'; a.href = useHash ? `#n-${type}` : '#';
        a.onclick = e => { if(!useHash) e.preventDefault();
          show('nodes');
          setTimeout(() => page.querySelector(`#n-${type}`)?.scrollIntoView(), 0); };
        nav.appendChild(a);
      }
  }
  show('overview');
  return { show };
}

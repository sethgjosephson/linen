// Node modules from outside the repository.
//
// A node module is an ES module whose default export is a function; it is called once with the plugin interface (PLUGIN_API below, with registerNode bound to the module) and registers one or more nodes through it.
//
//   export default function(linen){
//     linen.registerNode('jitter', { title:'jitter', cat:'arrange', ... compute(ins, p, node){ ... }, doc:{ ... } });
//   }
//
// The module imports nothing.
// It is loaded from its source text (a blob URL in the page, a data URL in node), so where the text came from (a project's nodes folder, a URL, a file in the repository) makes no difference to it, and the hash of that text is the module's identity: the memo key of a plugin node carries it, so an edited module never reuses what its earlier version computed.
//
// A module runs with everything the page can do.
// It is the user's own code, or code the user chose to add, and nothing here sandboxes it; what is checked is what can be checked from outside: the registered definitions (registerNode), and that the source does not draw from Math.random, which would break the promise that the same graph and seeds compute the same network.
//
// Computation runs on the page's thread (computeNode); the wiring workers run wireConnect on a finished stream and never read NODE_DEFS, so a module is imported by the page (and by tools/linen.mjs and the tests), not by the workers.

import { registerNode, restoreNode, need, clonePts, NODE_DEFS } from './nodes.js';
import { h32, pairHash, pairGauss, rng } from './rand.js';
import { nodesDir, nodePath } from './project.js';
import { migrateScene } from './migrate.js';

// The module's name: the last segment of its path or URL, without a query or fragment.
// A module added from a URL is kept in the project under this name, so the name a scene records is the same wherever the text came from.
export function moduleName(where){
  const s = String(where || '').split(/[?#]/)[0].replace(/[\\/]+$/, '');
  return s.slice(Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\')) + 1);
}
export const MODULE_NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\.m?js$/;

// FNV-1a over the UTF-8 bytes, as hex: the identity of a module's source.
export function sourceHash(text){
  const b = new TextEncoder().encode(String(text));
  let h = 2166136261 >>> 0;
  for(let i = 0; i < b.length; i++){ h ^= b[i]; h = Math.imul(h, 16777619) >>> 0; }
  return h.toString(16).padStart(8, '0');
}

// What a module is handed besides registerNode: the stream helpers the built-in compute functions use, and the seeded draws of src/rand.js.
// A draw keyed on a cell's identity (pairHash, pairGauss with the cell's src and lidx) is the same at every computation, whatever else the graph holds.
const HELPERS = { need, clonePts, h32, pairHash, pairGauss, rng };

// module name -> { module, hash, keys, error }: every module this page or process has loaded, and how it went.
const loaded = new Map();
export function pluginStates(){ return [...loaded.values()].map(r => ({ ...r, keys:[...r.keys] })); }
export function pluginState(module){ const r = loaded.get(module); return r ? { ...r, keys:[...r.keys] } : null; }

// Load one module from its source text.
// The importText argument turns text into the module's namespace (importer() below gives one for the page and one for node).
// The result is the module's record; a module that fails is recorded with its error and leaves no node of its own registered (what it registered before failing is put back), so a broken module never half-loads.
// The same text loaded again is a no-op; changed text replaces the module's nodes.
export async function loadPlugin(module, text, importText){
  const hash = sourceHash(text);
  const prev = loaded.get(module);
  if(prev && prev.hash === hash && !prev.error) return { ...prev, keys:[...prev.keys], changed:false };
  const rec = { module, hash, keys:[], error:null };
  const before = new Map();
  try {
    if(!MODULE_NAME.test(module))
      throw new Error(module + ': a node module is named like name.js (letters, digits, dot, dash, underscore)');
    if(/\bMath\s*\.\s*random\b/.test(text))
      throw new Error(module + ' calls Math.random, so the same graph would not compute the same network twice; draw from linen.pairHash or linen.pairGauss keyed on the cell (its src and lidx), or from linen.rng(seed)');
    const ns = await importText(text);
    if(!ns || typeof ns.default !== 'function')
      throw new Error(module + ' has no default export; a node module is export default function(linen){ linen.registerNode(key, definition) }');
    const api = { ...HELPERS,
      registerNode(key, def){
        if(!before.has(key)) before.set(key, NODE_DEFS[key]);
        const out = registerNode(key, def, { module, hash });
        if(!rec.keys.includes(key)) rec.keys.push(key);
        return out;
      } };
    await ns.default(api);
    if(!rec.keys.length) throw new Error(module + ' registered no node; call linen.registerNode in its default export');
  } catch(e){
    for(const [key, def] of before) restoreNode(key, def);
    rec.keys = [];
    rec.error = String(e && e.message || e);
  }
  loaded.set(module, rec);
  return { ...rec, keys:[...rec.keys], changed:true };
}

// ---- a project's nodes folder ----------------------------------------------
// Every *.js (and *.mjs) file in the project's nodes folder is a node module, loaded when the project opens, in name order.
// The project file records nothing about them: the folder is the list.
// The result is one record per file, in that order, each with its error when it did not load.
export async function loadProjectPlugins(st, importText){
  const out = [];
  for(const e of await st.list(nodesDir())){
    if(e.dir || !/\.m?js$/.test(e.name)) continue;
    const bytes = await st.read(nodePath(e.name));
    if(!bytes){ out.push({ module:e.name, hash:'', keys:[], error:e.name + ' could not be read', changed:true }); continue; }
    out.push(await loadPlugin(e.name, new TextDecoder().decode(bytes), importText));
  }
  return out;
}

// A module from a URL, kept in the project: it is fetched, loaded, and written into the nodes folder under its own name, so it loads with the project from then on, the same text every time, with or without the network.
// A module that does not load is not written.
// The fetchText argument is (url) => the text; the page passes one built on fetch, and its own address as base, so a path on the site (examples/nodes/jitter.js) is a URL too.
export async function addPluginFromUrl(st, url, fetchText, importText, base){
  let u;
  try { u = new URL(String(url || '').trim(), base); } catch(e){ throw new Error('not a URL: ' + url); }
  if(!/^https?:$/.test(u.protocol)) throw new Error('a node module is fetched over http or https, not ' + u.protocol);
  const name = moduleName(u.pathname);
  if(!MODULE_NAME.test(name)) throw new Error('the URL has to end in the module file, like https://example.org/jitter.js');
  const text = await fetchText(u.href);
  const rec = await loadPlugin(name, text, importText);
  if(rec.error) throw new Error(rec.error);
  await st.write(nodePath(name), text);
  return rec;
}

// ---- what a scene records ------------------------------------------------
// A scene file lists the modules its nodes come from: [{ module, nodes:[key, ...] }], one entry per module, the keys the scene uses, both sorted.
// The editor writes it with the scene (toJSON in editor.js) and refuses a scene whose modules are not loaded (load), whole, before anything on screen changes.
export function scenePlugins(nodes){
  const by = new Map();
  for(const n of nodes || []){
    const def = NODE_DEFS[n.type];
    if(!def || !def.plugin) continue;
    if(!by.has(def.plugin.module)) by.set(def.plugin.module, new Set());
    by.get(def.plugin.module).add(n.type);
  }
  return [...by].sort((a, b) => a[0] < b[0] ? -1 : 1).map(([module, keys]) => ({ module, nodes:[...keys].sort() }));
}

// Why a scene in the current format cannot be opened as it is, or null when every node type it holds is defined.
// It names each module the scene records with the node keys that are missing, and any type the scene holds that no module it records accounts for.
export function pluginProblem(scene){
  const recorded = Array.isArray(scene && scene.plugins) ? scene.plugins : [];
  const missing = [];
  for(const p of recorded){
    const keys = (p.nodes || []).filter(k => !NODE_DEFS[k]);
    if(keys.length) missing.push(p.module + ' (' + keys.join(', ') + ')');
  }
  const covered = new Set(recorded.flatMap(p => p.nodes || []));
  const stray = [...new Set(((scene && scene.nodes) || []).map(n => n.type))].filter(t => !NODE_DEFS[t] && !covered.has(t));
  const out = [];
  if(missing.length) out.push('this scene needs node modules that are not loaded: ' + missing.join(', ') +
    '. Put each module in the project\'s nodes folder, or add it from the project menu, and open the scene again');
  if(stray.length) out.push('this scene holds nodes of a type no loaded module defines and names no module for: ' + stray.join(', '));
  return out.length ? out.join('; ') : null;
}
// The same question about a scene as read from a file, in any format.
export function scenePluginsMissing(raw){ return pluginProblem(migrateScene(structuredClone(raw))); }
// Throw pluginProblem's answer, marked so the page can tell it from other load errors.
export function checkScenePlugins(scene){
  const why = pluginProblem(scene);
  if(why) throw Object.assign(new Error(why), { pluginMissing:true });
}

// How a module's text becomes a module.
// In the page, a blob URL: the text read is the text run, whatever store it came from, and a reload of changed text is a new URL rather than the browser's cached module.
// In node, a data URL.
export function importer(){
  if(typeof Blob !== 'undefined' && typeof URL !== 'undefined' && URL.createObjectURL && typeof document !== 'undefined')
    return async text => {
      const url = URL.createObjectURL(new Blob([text], { type:'text/javascript' }));
      try { return await import(url); } finally { URL.revokeObjectURL(url); }
    };
  return text => import('data:text/javascript;base64,' + Buffer.from(String(text), 'utf8').toString('base64'));
}

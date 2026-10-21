// A scene for the tools that run one without the browser: a scene file or the name of a scene from the Tab menu, loaded into a stand-in editor the way editor.load loads it in the page, with the settings given on the command line applied to its nodes.
// Both tools/linen.mjs and tools/enginebench.mjs load their scene here, so the two compute the same graph from the same arguments.
// A failure throws an Error whose message is ready to print.
import fs from 'node:fs';
import path from 'node:path';
import { NODE_DEFS, setFileReader } from '../src/nodes.js';
import { migrateScene } from '../src/migrate.js';
import { loadPlugin, importer, checkScenePlugins } from '../src/plugins.js';
import { ALL_SCENARIOS } from '../src/scenelist.js';
import { applyOverrides } from './overrides.mjs';

export function blankNode(type, id, x, y){
  const def = NODE_DEFS[type], params = {};
  def.params.forEach(p => params[p.k] = structuredClone(p.def));
  return { id, type, x, y, params, name:'', on:true, inputs:new Array(def.inputs).fill(null) };
}
export function fakeEditor(){
  const nodes = [];
  return { nodes, nextId:1, groups:[],
    byId(id){ return nodes.find(n => n.id === id); },
    addNode(type, x, y){ const n = blankNode(type, this.nextId++, x, y); nodes.push(n); return n; } };
}
// Node modules (src/plugins.js): every .js in each folder, loaded the way the page loads a project's nodes folder.
async function loadNodeFolder(dir, say){
  if(!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) throw new Error('no folder of node modules at ' + dir);
  for(const name of fs.readdirSync(dir).sort()){
    if(!/\.m?js$/.test(name)) continue;
    const r = await loadPlugin(name, fs.readFileSync(path.join(dir, name), 'utf8'), importer());
    if(r.error) throw new Error(r.error);
    say('node module ' + name + ': ' + r.keys.join(', '));
  }
}
// This is what editor.load does, without a canvas under it.
export function editorFromScene(data){
  data = migrateScene(data);
  try { checkScenePlugins(data); }
  catch(e){ throw new Error(e.message + '\n  (node modules load from --nodes DIR, and from the nodes folder beside the scenes folder the scene file is in)'); }
  const ed = fakeEditor();
  ed.nextId = data.nextId || 1;
  for(const sn of data.nodes || []){
    if(!NODE_DEFS[sn.type]) continue;
    const n = blankNode(sn.type, sn.id, sn.x, sn.y);
    const def = NODE_DEFS[sn.type];
    def.params.forEach(pd => { if(sn.params && sn.params[pd.k] !== undefined) n.params[pd.k] = structuredClone(sn.params[pd.k]); });
    n.name = sn.name || '';
    n.on = sn.on === false ? false : true;
    const nIn = Math.max(def.inputs, sn.inputs ? sn.inputs.length : 0);
    n.inputs = new Array(nIn).fill(null).map((_, k) => sn.inputs && sn.inputs[k] ? { id:sn.inputs[k].id } : null);
    ed.nodes.push(n);
  }
  for(const n of ed.nodes) n.inputs = n.inputs.map(c => c && ed.byId(c.id) ? c : null);
  return { ed, name:data.scenarioName || '' };
}

// The target is a .json scene file (relative to root) or a scene name, matched whole first and then as a substring.
// The nodes list is folders of node modules to load first; a scene file in a project's scenes folder adds that project's nodes folder and reads that project's files.
// The overrides are type:tag-or-name:key=value terms (tools/overrides.mjs).
// Returns the editor, the scene's name, where it came from and the overrides that were applied.
export async function loadScene(target, { root, nodes = [], overrides = [], say = () => {} }){
  let ed, sceneName, source;
  nodes = [...nodes];
  const asFile = path.resolve(root, target);
  if(/\.json$/i.test(target) && fs.existsSync(asFile)){
    const own = path.join(path.dirname(asFile), '..', 'nodes');
    if(path.basename(path.dirname(asFile)) === 'scenes' && fs.existsSync(own) && !nodes.some(d => path.resolve(root, d) === path.resolve(own))) nodes.push(own);
    // A scene in a project's scenes folder reads the project's files (points files, connections files, meshes) the way the page does.
    if(path.basename(path.dirname(asFile)) === 'scenes'){
      const projDir = path.join(path.dirname(asFile), '..');
      setFileReader(async p => fs.readFileSync(path.join(projDir, p)));
    }
    for(const d of nodes) await loadNodeFolder(path.resolve(root, d), say);
    const loaded = editorFromScene(JSON.parse(fs.readFileSync(asFile, 'utf8')));
    ed = loaded.ed;
    sceneName = loaded.name || path.basename(target, '.json');
    source = path.relative(root, asFile).replace(/\\/g, '/');
  } else {
    const sc = ALL_SCENARIOS.find(s => s.name.toLowerCase() === target.toLowerCase())
      || ALL_SCENARIOS.find(s => s.name.toLowerCase().includes(target.toLowerCase()));
    if(!sc) throw new Error('no scene file and no scene called ' + target + '. The scenes are:\n  '
      + ALL_SCENARIOS.map(s => s.name).join('\n  '));
    ed = fakeEditor(); sc.build(ed);
    sceneName = sc.name; source = 'scene ' + sc.name;
  }
  const applied = applyOverrides(ed.nodes, overrides);
  for(const a of applied) say('setting ' + a.term + ' on ' + a.nodes + ' node' + (a.nodes === 1 ? '' : 's'));
  return { ed, sceneName, source, applied };
}

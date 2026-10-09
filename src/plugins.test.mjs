// Nodes from outside the repository: the registry, the module loader and what a scene records.
// Run: node src/plugins.test.mjs
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ok, report, threw, inProcessWorker } from '../tools/harness.mjs';
import { NODE_DEFS, registerNode, computeNode, memoKey, nodeDocProblems, isBuiltinNode,
  setResolution, wireConnect, assembleConnect } from './nodes.js';
import { isDefaultName } from './nodenames.js';
import { loadPlugin, importer, moduleName, sourceHash, pluginState, loadProjectPlugins, addPluginFromUrl,
  scenePlugins, scenePluginsMissing } from './plugins.js';
import { SCENE_FORMAT } from './migrate.js';
const { NodeEditor } = await import('./editor.js');

const imp = importer();
// plugin nodes are read through a function, since src/nodedefs.test.mjs requires every node the source names on NODE_DEFS with a dot to be a built-in
const defOf = key => NODE_DEFS[key];

// a node module as a person would write one, as text
const shiftModule = (dx, key = 'shift_x') => `
export default function(linen){
  linen.registerNode('${key}', {
    title:'shift x', cat:'arrange', color:'hsl(270,75%,65%)', inputs:1,
    params:[ { k:'dx', label:'shift um', t:'float', def:${dx} } ],
    compute(ins, p){
      const pts = linen.need(ins[0], 'points', 'shift x needs points');
      const out = linen.clonePts(pts);
      for(let i = 0; i < out.count; i++) out.pos[i*3] += p.dx;
      return out;
    },
    doc:{ io:'points in, points out', blurb:'Moves every cell along x by a fixed distance.',
      params:{ dx:'The distance, micrometers.' } } });
}`;

// the example in the repository: examples/nodes/jitter.js and the scene that uses it
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const JITTER = readFileSync(join(ROOT, 'examples', 'nodes', 'jitter.js'), 'utf8');
const SCENE = JSON.parse(readFileSync(join(ROOT, 'examples', 'scenes', 'jitter.json'), 'utf8'));

{
  // before its module is loaded, the example scene is refused loudly and whole
  ok('the example scene records its module', JSON.stringify(SCENE.plugins) === JSON.stringify([{ module:'jitter.js', nodes:['jitter'] }]), JSON.stringify(SCENE.plugins));
  const why = scenePluginsMissing(SCENE);
  ok('without jitter.js loaded, the example scene is refused naming the module and the node key', /jitter\.js \(jitter\)/.test(why || ''), why);
  const ed = bareEditor(); ed.addNode('sphere', 0, 0);
  let err = null; try { ed.load(SCENE); } catch(e){ err = e; }
  ok('the editor refuses it', err && err.pluginMissing === true && /jitter\.js/.test(err.message));
  ok('and keeps what it held, with nothing of the scene half loaded', ed.nodes.length === 1 && ed.nodes[0].type === 'sphere');
}

{
  const r = await loadPlugin('shift.js', shiftModule(10), imp);
  ok('a module loads and registers its node', !r.error && r.keys.join() === 'shift_x', r.error);
  ok('the node is in NODE_DEFS with its module and source hash',
    defOf('shift_x') && defOf('shift_x').plugin.module === 'shift.js' && defOf('shift_x').plugin.hash === sourceHash(shiftModule(10)));
  ok('and its documentation passes the rule the built-in nodes pass',
    nodeDocProblems('shift_x', defOf('shift_x'), defOf('shift_x').doc).length === 0);
  ok('it is not a built-in', !isBuiltinNode('shift_x') && isBuiltinNode('move'));
  const again = await loadPlugin('shift.js', shiftModule(10), imp);
  ok('the same text again changes nothing', again.changed === false);
}

{
  const e = threw(() => registerNode('move', defOf('shift_x'), { module:'mine.js', hash:'0' }));
  ok('a key a built-in node has is refused, naming the key and the module', e && /"move"/.test(e) && /mine\.js/.test(e) && /built-in/.test(e), e);
  const r = await loadPlugin('other.js', shiftModule(5), imp);
  ok('a key another module registered is refused, naming both modules',
    r.error && /other\.js/.test(r.error) && /shift\.js/.test(r.error), r.error);
  ok('and the first module keeps its node', defOf('shift_x').plugin.module === 'shift.js');
  ok('a refused module is recorded with its error', pluginState('other.js').error === r.error);
}

{
  const bad = shiftModule(1, 'undocumented').replace("params:{ dx:'The distance, micrometers.' }", 'params:{}');
  const r = await loadPlugin('undocumented.js', bad, imp);
  ok('a node without prose for a setting is refused', r.error && /undocumented\.dx has no prose/.test(r.error), r.error);
  ok('and is not registered', !defOf('undocumented'));
  const r2 = await loadPlugin('random.js', shiftModule(1, 'rnd').replace('p.dx;', 'p.dx*Math.random();'), imp);
  ok('a module that draws from Math.random is refused, naming the draws to use', r2.error && /Math\.random/.test(r2.error) && /pairGauss/.test(r2.error), r2.error);
  const r3 = await loadPlugin('nodefault.js', 'export const x = 1;', imp);
  ok('a module with no default export is refused, saying what one looks like', r3.error && /default export/.test(r3.error), r3.error);
  const two = `export default function(linen){
    linen.registerNode('half_one', { title:'one', cat:'arrange', color:'#888', inputs:1, params:[], compute:ins => ins[0],
      doc:{ blurb:'A node that passes its input through.', params:{} } });
    linen.registerNode('half_two', { title:'two' });
  }`;
  const r4 = await loadPlugin('half.js', two, imp);
  ok('a module that fails half way is refused', !!r4.error, r4.error);
  ok('and leaves none of its nodes registered', !defOf('half_one') && !defOf('half_two'));
  ok('a module name that is not a file name is refused', !!(await loadPlugin('../x.js', shiftModule(1, 'escape'), imp)).error);
  ok('the module name is the last path segment', moduleName('https://example.org/a/b/jitter.js?v=2') === 'jitter.js' && moduleName('nodes\\jitter.js') === 'jitter.js');
}

{
  // a plugin node computes in a graph like any other
  const nodes = [
    { id:1, type:'sphere', params:{ center:[0,0,0], radius:100 }, inputs:[] },
    { id:2, type:'scatter', params:{}, inputs:[{ id:1 }] },
    { id:3, type:'shift_x', params:{ dx:25 }, inputs:[{ id:2 }] },
  ];
  for(const p of NODE_DEFS.scatter.params) if(nodes[1].params[p.k] === undefined) nodes[1].params[p.k] = structuredClone(p.def);
  nodes[1].params.fill = 0; nodes[1].params.count = 50;
  const byId = id => nodes.find(n => n.id === id);
  const a = await computeNode(nodes[1], byId), b = await computeNode(nodes[2], byId);
  ok('a plugin node computes through computeNode', b.kind === 'points' && b.count === a.count);
  ok('and does what its compute says', Math.abs(b.pos[0] - a.pos[0] - 25) < 1e-3 && b.pos[1] === a.pos[1]);
  ok('a node of an unloaded type fails loudly, naming the type',
    /nosuch: no node of this type/.test(await computeNode({ id:9, type:'nosuch', params:{}, inputs:[] }, byId).then(() => '', e => e.message)));
}

{
  // the memo key of a plugin node that asks for one carries the module's hash
  const r = await loadPlugin('heavy.js', shiftModule(3, 'heavy').replace("inputs:1,", 'inputs:1, memo:true,'), imp);
  ok('a module may ask for a memoized compute', !r.error, r.error);
  const node = { id:5, type:'heavy', params:{ dx:3 }, inputs:[] };
  const k1 = memoKey(node, defOf('heavy'), [null]);
  await loadPlugin('heavy.js', shiftModule(3, 'heavy').replace("inputs:1,", 'inputs:1, memo:true,') + '\n// edited', imp);
  const k2 = memoKey(node, defOf('heavy'), [null]);
  ok('an edited module gives its memoized node another key', k1 !== k2 && k2.includes(sourceHash(shiftModule(3, 'heavy').replace("inputs:1,", 'inputs:1, memo:true,') + '\n// edited')));
  ok('a built-in node key carries no module', !memoKey({ id:1, type:'connect', params:{} }, NODE_DEFS.connect, []).includes('@'));
}

// a project store in memory, with the same list, read and write store.js gives the page
function memStore(files = {}){
  const enc = s => typeof s === 'string' ? new TextEncoder().encode(s) : s;
  const m = new Map(Object.entries(files).map(([k, v]) => [k, enc(v)]));
  return { label:'memory', files:m,
    async list(dir){ const pre = dir + '/'; return [...m.keys()].filter(k => k.startsWith(pre) && !k.slice(pre.length).includes('/'))
      .sort().map(k => ({ name:k.slice(pre.length), dir:false, size:m.get(k).length, mtime:0 })); },
    async read(p){ return m.get(p) || null; },
    async write(p, d){ m.set(p, enc(d)); return m.get(p).length; } };
}

{
  // a project's nodes folder: every .js in it, in name order, each with how it went
  const st = memStore({ 'nodes/b-shift.js':shiftModule(4, 'proj_b'), 'nodes/a-broken.js':'export default function(linen){ linen.registerNode("proj_a", {}); }',
    'nodes/readme.txt':'not a module', 'scenes/main.json':'{}' });
  const recs = await loadProjectPlugins(st, imp);
  ok('the project loads every module in its nodes folder and nothing else', recs.map(r => r.module).join() === 'a-broken.js,b-shift.js', recs.map(r => r.module).join());
  ok('a module that loads registers its node', !recs[1].error && defOf('proj_b') && defOf('proj_b').plugin.module === 'b-shift.js');
  ok('a module that does not is recorded with what it needs', !!recs[0].error && /needs/.test(recs[0].error) && !defOf('proj_a'), recs[0].error);
  const again = await loadProjectPlugins(st, imp);
  ok('opening the project again loads nothing twice', again[1].changed === false);
  st.files.set('nodes/b-shift.js', new TextEncoder().encode(shiftModule(4, 'proj_b').replace("title:'shift x'", "title:'shift x, edited'")));
  const edited = await loadProjectPlugins(st, imp);
  ok('an edited module replaces its node', edited[1].changed === true && defOf('proj_b').title === 'shift x, edited');
  ok('an empty project loads no modules', (await loadProjectPlugins(memStore(), imp)).length === 0);
}

{
  // a module from a URL is fetched, loaded and kept in the project's nodes folder
  const st = memStore();
  const served = { 'https://example.org/mods/url-shift.js?v=1':shiftModule(6, 'url_shift') };
  const fetchText = async u => { if(!(u in served)) throw new Error('404 ' + u); return served[u]; };
  const rec = await addPluginFromUrl(st, 'https://example.org/mods/url-shift.js?v=1', fetchText, imp);
  ok('a module added from a URL registers its node', rec.keys.join() === 'url_shift' && defOf('url_shift'));
  ok('and is kept in the project under its own name', new TextDecoder().decode(await st.read('nodes/url-shift.js')) === served['https://example.org/mods/url-shift.js?v=1']);
  ok('so the project loads it from then on', (await loadProjectPlugins(st, imp))[0].module === 'url-shift.js');
  served['https://linen.example/examples/nodes/rel-shift.js'] = shiftModule(2, 'rel_shift');
  const rel = await addPluginFromUrl(st, 'examples/nodes/rel-shift.js', fetchText, imp, 'https://linen.example/index.html');
  ok('a path is read against the page\'s address', rel.keys.join() === 'rel_shift' && !!(await st.read('nodes/rel-shift.js')));
  const e0 = await addPluginFromUrl(st, 'examples/nodes/rel-shift.js', fetchText, imp).then(() => '', e => e.message);
  ok('and without one is not a URL', /not a URL/.test(e0), e0);
  const e1 = await addPluginFromUrl(st, 'ftp://example.org/x.js', fetchText, imp).then(() => '', e => e.message);
  ok('a URL that is not http or https is refused', /http or https/.test(e1), e1);
  const e2 = await addPluginFromUrl(st, 'https://example.org/mods/', fetchText, imp).then(() => '', e => e.message);
  ok('a URL that does not name a module file is refused', /module file/.test(e2), e2);
  served['https://example.org/bad.js'] = 'export default 3;';
  const e3 = await addPluginFromUrl(st, 'https://example.org/bad.js', fetchText, imp).then(() => '', e => e.message);
  ok('a module that does not load is refused', /default export/.test(e3), e3);
  ok('and is not written into the project', !(await st.read('nodes/bad.js')));
}

// an editor as the page has one, without a canvas, for its toJSON and load
function bareEditor(){
  const ed = Object.create(NodeEditor.prototype);
  Object.assign(ed, { nodes:[], groups:[], nextId:1, cb:{ onSelect(){}, onChange(){}, onMoved(){} },
    viewSlots:{}, activeView:null, view:{ x:0, y:0, scale:1 }, suggestResolution:0, selSet:new Set(), sel:null });
  return ed;
}

{
  // a scene records the modules its nodes come from, and is refused whole when one is not loaded
  const ed = bareEditor();
  const g = ed.addNode('sphere', 0, 0), sc = ed.addNode('scatter', 0, 60), sh = ed.addNode('shift_x', 0, 120);
  sc.inputs[0] = { id:g.id }; sh.inputs[0] = { id:sc.id }; sh.params.dx = 33;
  const saved = JSON.parse(JSON.stringify(ed.toJSON()));
  ok('a scene lists its node modules and the keys it uses', JSON.stringify(saved.plugins) === JSON.stringify([{ module:'shift.js', nodes:['shift_x'] }]), JSON.stringify(saved.plugins));
  ok('in the current format', saved.format === SCENE_FORMAT);
  ok('a scene of built-in nodes lists none', JSON.stringify(scenePlugins([g, sc])) === '[]');
  const back = bareEditor(); back.load(saved);
  ok('the scene loads back with the plugin node and its settings', back.nodes.length === 3 && back.nodes[2].type === 'shift_x' && back.nodes[2].params.dx === 33);
  ok('and writes the same list again', JSON.stringify(back.toJSON().plugins) === JSON.stringify(saved.plugins));

  const elsewhere = JSON.parse(JSON.stringify(saved));
  elsewhere.plugins = [{ module:'gone.js', nodes:['gone_node'] }];
  elsewhere.nodes[2].type = 'gone_node';
  const before = back.nodes.map(n => n.id + n.type).join();
  let err = null; try { back.load(elsewhere); } catch(e){ err = e; }
  ok('a scene whose module is not loaded is refused, naming the module and its node keys', err && /gone\.js \(gone_node\)/.test(err.message) && err.pluginMissing === true, err && err.message);
  ok('and nothing of it is loaded: the editor holds what it held', back.nodes.map(n => n.id + n.type).join() === before);
  ok('the refusal says how to fix it', err && /nodes folder/.test(err.message));
  ok('the page can ask the same question of a file before opening it', /gone\.js/.test(scenePluginsMissing(elsewhere) || ''));
  const stray = JSON.parse(JSON.stringify(saved));
  stray.plugins = []; stray.nodes[2].type = 'mystery';
  ok('a node type no module accounts for is refused too, by name', /mystery/.test(scenePluginsMissing(stray) || ''));
  ok('an older scene with built-in nodes only opens', scenePluginsMissing({ format:11, nodes:[{ id:1, type:'sphere', params:{} }] }) === null);
  ok('and so does one from before the checkpoint had its name', scenePluginsMissing({ format:9, nodes:[{ id:1, type:'output', params:{} }] }) === null);
}

// ---- the example: examples/nodes/jitter.js ----------------------------------
{
  const r = await loadPlugin('jitter.js', JITTER, imp);
  ok('the example module loads', !r.error && r.keys.join() === 'jitter', r.error);
  ok('its documentation passes the rule the built-in nodes pass', nodeDocProblems('jitter', defOf('jitter'), defOf('jitter').doc).length === 0,
    nodeDocProblems('jitter', defOf('jitter'), defOf('jitter').doc).join('; '));

  // a small graph through it: two populations in a sphere, gathered, jittered
  const mk = () => {
    const ed = bareEditor();
    const g = ed.addNode('sphere', 0, 0); g.params.radius = 150;
    const a = ed.addNode('scatter', 0, 60); Object.assign(a.params, { fill:0, count:300, seed:1, tag:'a' });
    const b = ed.addNode('scatter', 100, 60); Object.assign(b.params, { fill:0, count:100, seed:2, tag:'b', type:3 });
    const m = ed.addNode('gather', 0, 120), j = ed.addNode('jitter', 0, 180);
    a.inputs[0] = b.inputs[0] = { id:g.id }; m.inputs[0] = { id:a.id }; m.inputs[1] = { id:b.id }; j.inputs[0] = { id:m.id };
    return { ed, a, b, m, j, byId:id => ed.byId(id) };
  };
  const G = mk();
  const before = await computeNode(G.m, G.byId), after = await computeNode(G.j, G.byId);
  const d = []; for(let i = 0; i < after.count*3; i++) d.push(after.pos[i] - before.pos[i]);
  const mean = d.reduce((s, x) => s + x, 0)/d.length, sd = Math.sqrt(d.reduce((s, x) => s + (x - mean)**2, 0)/d.length);
  ok('jitter moves every cell, with offsets of mean near zero', after.count === before.count && d.every(x => x !== 0) && Math.abs(mean) < 3, mean);
  ok('and a standard deviation near sigma (20 um)', Math.abs(sd - 20) < 3, sd);
  ok('and keeps every identity', after.src.every((s, i) => s === before.src[i]) && after.lidx.every((l, i) => l === before.lidx[i]));
  // the same cells move the same way whatever else the graph holds: here population b is taken away
  const H = mk(); H.m.inputs[1] = null;
  const alone = await computeNode(H.j, H.byId);
  ok('a cell moves by the same offset when another population is taken away', alone.count === 300 &&
    [...alone.pos].every((v, i) => v === after.pos[i]));
  for(const n of G.ed.nodes) n._cache = null;
  const again = await computeNode(G.j, G.byId);
  ok('and at every computation', [...again.pos].every((v, i) => v === after.pos[i]));
  G.j.params.sigma = 0; G.j._cache = null;
  ok('sigma 0 leaves every cell where it was', [...(await computeNode(G.j, G.byId)).pos].every((v, i) => v === before.pos[i]));
  G.j.params.sigma = 20; G.j.params.seed = 2; G.j._cache = null;
  ok('another seed is another draw', !(await computeNode(G.j, G.byId)).pos.every((v, i) => v === after.pos[i]));
  G.j.params.seed = 1; G.j._cache = null;

  // a changed module changes the wiring key: the connect node's key is the content of the stream it is handed
  const cn = G.ed.addNode('connect', 0, 240); cn.inputs[0] = { id:G.j.id };
  const keyNow = async () => { for(const n of G.ed.nodes) n._cache = null; return memoKey(cn, NODE_DEFS.connect, [await computeNode(G.j, G.byId)]); };
  const k1 = await keyNow();
  await loadPlugin('jitter.js', JITTER, imp);
  ok('the same module loaded again keeps the wiring key', (await keyNow()) === k1);
  const edited = JITTER.replace('+ 0x6a17', '+ 0x6a18');
  ok('(the edit is to the draw)', edited !== JITTER);
  const r2 = await loadPlugin('jitter.js', edited, imp);
  const k2 = await keyNow();
  ok('an edited module is a new hash', !r2.error && r2.changed && defOf('jitter').plugin.hash === sourceHash(edited));
  ok('and a new wiring key for the connect node below it', k2 !== k1);
  await loadPlugin('jitter.js', JITTER, imp);
  ok('and loading the first version back gives the first key back', (await keyNow()) === k1);

  // the example scene: every node named, the round trip, and a computation through it at a small resolution
  const named = SCENE.nodes.filter(n => n.type !== 'pin' && n.type !== 'note');
  const bare = named.filter(isDefaultName);
  ok('every node of the example scene is named', bare.length === 0, bare.map(n => n.type + ' ' + n.name).join(', '));
  ok('and the jitter node is in it, between the gather and the connect node', (() => {
    const by = id => SCENE.nodes.find(n => n.id === id), j = SCENE.nodes.find(n => n.type === 'jitter'), c = SCENE.nodes.find(n => n.type === 'connect');
    return j && by(j.inputs[0].id).type === 'gather' && c.inputs[0].id === j.id; })());
  const ed = bareEditor(); ed.load(SCENE);
  ok('with its module loaded, the example scene loads', ed.nodes.length === SCENE.nodes.length);
  const round = JSON.parse(JSON.stringify(ed.toJSON()));
  ok('and saves the same module list and nodes back', JSON.stringify(round.plugins) === JSON.stringify(SCENE.plugins) &&
    JSON.stringify(round.nodes) === JSON.stringify(SCENE.nodes));
  inProcessWorker(wireConnect, assembleConnect);
  setResolution(0.05);
  const net = await computeNode(ed.nodes.find(n => n.type === 'checkpoint'), id => ed.byId(id));
  setResolution(1);
  ok('the example scene computes through jitter to a network', net && net.kind === 'net' && net.count > 500 && net.synCount > 0, net && net.count + ' cells, ' + net.synCount + ' synapses');
}

report('plugins');

// A saved layout for a scenario: the graph exactly as someone arranged it, kept apart from the builder so the scene's numbers stay in code and its arrangement in a file.
// The builder makes the scenario's nodes in a fixed order and each is marked with its place in that order (`built`, kept in scene files too); the file records, for every node on screen, where it is, which built node it is or, for a node the person added (a dot, a merge rebuilt as a different tree, a note, a probe), its type and settings, and every node's wires by file index.
// The groups come from the file whole: labels, hues, padding, membership.
//
// Applying a file to a fresh build keeps the build's settings on every built node (the scenario's tuned numbers, gated by the battery, are never in the file), drops the built nodes the file does not have, makes the added nodes, moves everything and rewires everything.
// The file also records the build's node types in order; when the builder has changed since (a node added or retyped in code) the file is refused with a reason and the computed layout stands, until it is saved again.
//
// Files live in src/layouts/<slug>.json, served with the site.
// The scene menu writes one through the dev server (loopback only) or downloads it.

import { NODE_DEFS } from './nodes.js';

export const LAYOUT_FORMAT = 3;

export function layoutSlug(name){
  return String(name || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

export function markBuilt(nodes){ nodes.forEach((n, k) => { n.built = k; }); }
export const isBuilt = n => Number.isInteger(n.built);
// the build's node types in build order, from a marked graph (a node the person deleted leaves a hole, which is fine for comparing with a file)
export function builtTypes(nodes){
  const out = [];
  for(const n of nodes) if(isBuilt(n)) out[n.built] = n.type;
  return out;
}

// what the file holds, from the graph on screen; `build` is the builder's node types in order (from a fresh build, so a deleted node is still listed)
export function layoutRecord(editor, name, build){
  const index = new Map(editor.nodes.map((n, i) => [n.id, i]));
  const nodes = editor.nodes.map(n => {
    const def = NODE_DEFS[n.type] || {};
    const rec = { type:n.type, x:Math.round(n.x), y:Math.round(n.y) };
    if(isBuilt(n)){
      rec.built = n.built;
      // port count is structure, not a setting: a built merge someone widened
      if(def.portsParam && n.params) rec.params = { [def.portsParam]:n.params[def.portsParam] };
    } else {
      rec.params = structuredClone(n.params || {});
      if(n.name) rec.name = n.name;
    }
    if(n.on === false) rec.off = true;
    rec.inputs = (n.inputs || []).map(c => c && index.has(c.id) ? index.get(c.id) : null);
    return rec;
  });
  const groups = (editor.groups || []).map(g => ({
    label:g.label, hue:g.hue, z:g.z || 0, pad:g.pad || null,
    members:(g.members || []).map(id => index.get(id)).filter(i => i !== undefined) }));
  return { format:LAYOUT_FORMAT, scenario:name, saved:new Date().toISOString().slice(0, 10),
    build:build ? build.slice() : builtTypes(editor.nodes), nodes, groups };
}

// does the file describe this build?
// '' when it does, else why not
export function layoutMismatch(build, data){
  if(!data || data.format !== LAYOUT_FORMAT || !Array.isArray(data.nodes) || !Array.isArray(data.build)) return 'not a layout file of this format: save it again';
  if(data.build.length !== build.length) return 'the scenario builds ' + build.length + ' nodes now and built ' + data.build.length + ' when the layout was saved';
  for(let k = 0; k < build.length; k++)
    if(data.build[k] !== build[k]) return 'node ' + k + ' of the build is a ' + build[k] + ' now and was a ' + data.build[k];
  return '';
}

// apply a file to a freshly built, marked graph (an editor with addNode and deleteNode); ok:false with a reason when the file does not describe this build
export function applyLayoutRecord(editor, data){
  const build = builtTypes(editor.nodes);
  const why = layoutMismatch(build, data);
  if(why) return { ok:false, why };
  const byBuilt = new Map(editor.nodes.filter(isBuilt).map(n => [n.built, n]));
  for(const f of data.nodes) if(f.built !== undefined){
    const n = byBuilt.get(f.built);
    if(!n) return { ok:false, why:'the file names built node ' + f.built + ' twice or out of range' };
    if(n.type !== f.type) return { ok:false, why:'built node ' + f.built + ' is a ' + n.type + ' and the file has a ' + f.type };
    byBuilt.delete(f.built);
  }
  // the graph becomes the file's: what the computed layout added (its reroute dots) goes, and so do the built nodes the file does not have
  for(const n of editor.nodes.filter(n => !isBuilt(n))) editor.deleteNode(n);
  let removed = 0;
  for(const n of byBuilt.values()){ editor.deleteNode(n); removed++; }
  // file index -> node on screen
  const nodeAt = new Array(data.nodes.length).fill(null);
  const built = new Map(editor.nodes.filter(isBuilt).map(n => [n.built, n]));
  let added = 0;
  data.nodes.forEach((f, i) => {
    let n;
    if(f.built !== undefined) n = built.get(f.built);
    else {
      if(!NODE_DEFS[f.type]) return;
      n = editor.addNode(f.type, +f.x || 0, +f.y || 0);
      if(f.params) Object.assign(n.params, structuredClone(f.params));
      if(f.name) n.name = f.name;
      added++;
    }
    const def = NODE_DEFS[n.type] || {};
    if(f.built !== undefined && def.portsParam && f.params && f.params[def.portsParam] !== undefined)
      n.params[def.portsParam] = f.params[def.portsParam];
    n.x = +f.x || 0; n.y = +f.y || 0;
    n.on = f.off ? false : true;
    nodeAt[i] = n;
  });
  // every wire from the file, sized to the node's ports
  data.nodes.forEach((f, i) => {
    const n = nodeAt[i]; if(!n) return;
    const def = NODE_DEFS[n.type] || {};
    const want = Math.max(def.inputs || 0, Array.isArray(f.inputs) ? f.inputs.length : 0);
    n.inputs = new Array(want).fill(null).map((_, p) => {
      const j = Array.isArray(f.inputs) ? f.inputs[p] : null;
      return j === null || j === undefined || !nodeAt[j] ? null : { id:nodeAt[j].id };
    });
  });
  editor.groups = (Array.isArray(data.groups) ? data.groups : []).map(g => ({
    label:g.label, hue:g.hue, z:g.z || 0, pad:{ l:0, t:0, r:0, b:0, ...(g.pad || {}) },
    members:(g.members || []).map(i => nodeAt[i]).filter(Boolean).map(n => n.id) }));
  return { ok:true, added, removed };
}

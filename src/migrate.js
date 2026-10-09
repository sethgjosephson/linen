// Scene format versions and the migrations between them, in one place.
// A scene file carries `format`; a file without one is format 1.
// Loading runs every migration from the file's format up to SCENE_FORMAT, so the rest of the editor only ever sees the current shape and the shims for older files live here rather than at whichever call site first met them.
//
// Add a version by appending a step below and bumping SCENE_FORMAT.
// Never edit an existing step: files written at that version depend on it.

import { defaultNodeName } from './tagname.js';

export const SCENE_FORMAT = 13;

import { applyOverrides } from './sweepvariant.js';

const STEPS = [
  // 1 -> 2: the active viewer binding was saved as `activeOutput` before viewer digits existed; it is `activeView`.
  d => {
    if(d.activeView === undefined && d.activeOutput !== undefined)
      d.activeView = d.activeOutput;
    delete d.activeOutput;
    return d;
  },
  // 2 -> 3: a chart node sat in the readout chain, upstream of the checkpoint, while what it draws is what the checkpoint's run produces.
  // Charts move downstream of the checkpoint: whatever fed a chart now feeds the chart's consumer, and the chart takes the checkpoint as its input.
  d => {
    const nodes = d.nodes || [];
    const out = nodes.find(x => x.type === 'output');
    if(!out) return d;
    const byId = id => nodes.find(x => x.id === id);
    for(const c of nodes.filter(x => x.type === 'chart')){
      const feeder = c.inputs && c.inputs[0] ? c.inputs[0].id : null;
      for(const other of nodes)
        for(let i=0;i<(other.inputs || []).length;i++)
          if(other.inputs[i] && other.inputs[i].id === c.id)
            other.inputs[i] = feeder !== null ? { id:feeder } : null;
      c.inputs = [{ id:out.id }];
      // and it lands to the right of the checkpoint rather than on top of it
      if(byId(out.id)) c.x = (out.x || 0) + 220;
    }
    return d;
  },
  // 3 -> 4: an input's sense was read off its mapping, a two dimensional map meaning sight and anything else sound.
  // It is a control, so each input is given the sense its mapping implied.
  d => {
    for(const nd of (d.nodes || []))
      if(nd.type === 'input' && nd.params && nd.params.sense === undefined)
        nd.params.sense = (nd.params.map | 0) === 1 ? 1 : 0;
    return d;
  },
  // 4 -> 5: the input node is the mapping only.
  // Its source dropdown (bar sweep, microphone, webcam) becomes a test signal or live node on its signal port; its sense moves to the curriculum, which is one stream of one sense, so an input that wanted the other sense from a shared curriculum gets a second curriculum that follows the first.
  // The side ports become mask (1) and signal (2) in that order.
  d => {
    const nodes = d.nodes || [];
    let nextId = d.nextId || (Math.max(0, ...nodes.map(n => n.id | 0)) + 1);
    const byId = id => nodes.find(x => x.id === id);
    const SIGNAL = new Set(['curriculum', 'read', 'live', 'testsignal']);
    const mk = (type, title, x, y, params) => {
      const n = { id:nextId++, type, x, y, params, inputs:[],
        name:defaultNodeName(title, nodes.map(m => m.name)) };
      nodes.push(n); return n;
    };
    const streams = new Map();            // curriculum id -> { sense, followers }
    for(const inp of nodes.filter(n => n.type === 'input')){
      const p = inp.params || (inp.params = {});
      const ins = inp.inputs || (inp.inputs = []);
      while(ins.length < 3) ins.push(null);
      let mask = null, sig = null;
      for(const c of [ins[1], ins[2]]){
        if(!c) continue;
        const src = byId(c.id);
        if(src && SIGNAL.has(src.type)){ if(!sig) sig = c; }
        else if(!mask) mask = c;
      }
      const sense = (p.sense | 0) === 1 ? 1 : 0;
      if(sig && byId(sig.id).type === 'curriculum'){
        const cur = byId(sig.id);
        let st = streams.get(cur.id);
        if(!st){
          cur.params.sense = sense;
          st = { sense, followers:new Map() }; streams.set(cur.id, st);
        } else if(st.sense !== sense){
          let f = st.followers.get(sense);
          if(!f){
            f = mk('curriculum', 'curriculum', (cur.x || 0), (cur.y || 0) + 56,
              { ...JSON.parse(JSON.stringify(cur.params)), sense });
            f.inputs = [{ id:cur.id }];
            st.followers.set(sense, f);
          }
          sig = { id:f.id };
        }
      } else if(!sig){
        const which = p.source | 0, x = (inp.x || 0) - 170, y = (inp.y || 0) + 44;
        const s = which === 1 || which === 2
          ? mk('live', 'live', x, y, { device:which === 2 ? 1 : 0 })
          : mk('testsignal', 'test signal', x, y, { pattern:0, period:p.period || 1500 });
        sig = { id:s.id };
      }
      ins[1] = mask; ins[2] = sig;
      delete p.source; delete p.sense; delete p.period;
    }
    for(const c of nodes.filter(n => n.type === 'curriculum')){
      if(!c.inputs || !c.inputs.length) c.inputs = [null];
      if(c.params && c.params.sense === undefined) c.params.sense = 0;
    }
    // A population tag is one token.
    // The pair table is split on whitespace, and a scatter tagged 'RS exc 1' produced rows that re-parsed as three populations and a multiplier.
    // Tags lose their spaces, every reference to one follows, and a table row that was written with a spaced tag is read back by matching the longest known tag at each position.
    const spaced = new Map();                // old tag -> token
    for(const sc of nodes.filter(n => n.type === 'scatter')){
      const t = String((sc.params || {}).tag || '');
      const tok = t.trim().replace(/\s+/g, '');
      if(tok !== t){ spaced.set(t, tok); sc.params.tag = tok; }
    }
    if(spaced.size){
      // every reference becomes a token too, whatever its case: computation compares tags without case, so 'input 2' meets 'Input2' as 'input2'
      const fix = v => String(v || '').trim().replace(/\s+/g, '');
      for(const n of nodes){
        const p = n.params || {};
        if(['probe', 'stimulus', 'input'].includes(n.type) && p.tag !== undefined) p.tag = fix(p.tag);
        if(['plasticity', 'project'].includes(n.type)){
          if(p.from !== undefined) p.from = fix(p.from);
          if(p.to !== undefined) p.to = fix(p.to);
        }
        if(n.type === 'chart' && p.pop !== undefined) p.pop = fix(p.pop);
      }
      const known = [...spaced.keys()].sort((a, b) => b.length - a.length);
      const takeTag = words => {         // longest spaced tag at the head, else one word
        for(const t of known){
          const w = t.split(/\s+/);
          if(w.length <= words.length && w.every((x, i) => x === words[i]))
            return [spaced.get(t), words.slice(w.length)];
        }
        return [words[0], words.slice(1)];
      };
      for(const cn of nodes.filter(n => n.type === 'connect' && n.params && n.params.table)){
        const out = [];
        for(const line of String(cn.params.table).split('\n')){
          const body = line.split('#')[0], comment = line.slice(body.length);
          const words = body.trim().split(/\s+/).filter(Boolean);
          if(words.length < 3){ if(line.trim()) out.push(line); continue; }
          const [pre, r1] = takeTag(words);
          const [post, r2] = takeTag(r1);
          // what remains must read as multipliers.
          // A row with no number in it at all was never a row: it is what a chip wrote when it split a spaced tag, and it is dropped.
          // Anything else is left as it is.
          if(r2.length >= 1 && r2.length <= 3 && Number.isFinite(+r2[0]) &&
             (r2.length < 2 || Number.isFinite(+r2[1])))
            out.push([pre, post, ...r2].join(' ') + (comment ? ' ' + comment : ''));
          else {
            // tile the whole line with known tags; a bare number left over means it was a row of some kind and it stays for the reader
            let rest = words, bare = false;
            while(rest.length){ const [tok, r] = takeTag(rest); rest = r;
              if(Number.isFinite(+tok)) bare = true; }
            if(bare) out.push(line);
          }
        }
        // the same row twice says nothing twice
        cn.params.table = [...new Set(out)].join('\n');
      }
    }
    d.nextId = nextId;
    return d;
  },
  // 5 -> 6: a dispersed projection is set by contacts per source rather than by a probability over the whole target population.
  // The count is what the probability came to against the target scatter's authored count, so the network a scene wires is unchanged.
  d => {
    const nodes = d.nodes || [];
    for(const pr of nodes.filter(n => n.type === 'project' && n.params)){
      const p = pr.params;
      if(p.fanout !== undefined) continue;
      const to = String(p.to || '').trim().toLowerCase();
      const target = nodes.find(n => n.type === 'scatter' && n.params &&
        String(n.params.tag || '').trim().toLowerCase() === to);
      const count = target ? (target.params.count | 0) : 0;
      p.fanout = (p.dispersed | 0) === 1 && count > 0
        ? Math.max(1, Math.round((+p.prob || 0) * count)) : 50;
    }
    return d;
  },
  // 6 -> 7: the sweep is a tab on the checkpoint.
  // A sweep node reached down into nodes below it, and a node may only read what is above it; every node is above the checkpoint.
  // Each sweep node's tabs move onto the checkpoint, the checkpoint's own text lines become tabs with their selectors resolved against this scene, and the sweep nodes are unwired (their one input goes to whatever read them) and dropped.
  d => {
    const nodes = d.nodes || [];
    const out = nodes.find(n => n.type === 'output');
    const sweeps = nodes.filter(n => n.type === 'sweep');
    if(out){
      out.params = out.params || {};
      const list = Array.isArray(out.params.variants) ? out.params.variants : [];
      for(const line of String(out.params.trVariants || '').split(/[\n|]/).map(x => x.trim()).filter(Boolean)){
        const edits = [];
        const copy = nodes.map(n => ({ id:n.id, type:n.type, name:n.name, params:{ ...(n.params || {}) } }));
        applyOverrides(line, copy, n => {
          const orig = nodes.find(m => m.id === n.id);
          for(const k of Object.keys(n.params))
            if(n.params[k] !== (orig.params || {})[k] && !edits.some(e => e.node === n.id && e.key === k))
              edits.push({ node:n.id, key:k, value:n.params[k] });
        });
        list.push({ name:line, edits });
      }
      for(const s of sweeps)
        for(const v of ((s.params || {}).variants || []))
          list.push({ name:v.name || 'sweep', edits:(v.edits || []).map(e => ({ ...e })) });
      if(list.length) out.params.variants = list;
      delete out.params.trVariants;
    }
    for(const s of sweeps){
      const src = (s.inputs || [])[0] || null;
      for(const n of nodes) if(Array.isArray(n.inputs))
        n.inputs = n.inputs.map(c => c && c.id === s.id ? (src ? { id:src.id } : null) : c);
      for(const g of d.groups || []) if(Array.isArray(g.members)) g.members = g.members.filter(id => id !== s.id);
    }
    d.nodes = nodes.filter(n => n.type !== 'sweep');
    return d;
  },
  // 7 -> 8: the 2003 membrane form is retired (form.js).
  // Rows 0 to 6 of the type table are the 2003 presets carried as 2007 rows, so a scene written for the 2003 form (form 0, or none) keeps every type index and every number, and only loses the form field.
  // A scene on the 2007 form used the measured rows under those same indices; those rows moved to 10 to 16, so its type indices move with them (scatter, points file and retype nodes, and any sweep variant that edits one of their types).
  // The cell type node's 2003 base row goes with the form.
  d => {
    const was2007 = (d.form|0) === 1;
    const TYPED = new Set(['scatter', 'pointfile', 'ntype']);
    const nodes = d.nodes || [];
    const typed = new Set(nodes.filter(n => TYPED.has(n.type)).map(n => n.id));
    const lift = v => (typeof v === 'number' && v >= 0 && v <= 6) ? v + 10 : v;
    for(const n of nodes){
      const p = n.params || {};
      if(was2007 && TYPED.has(n.type) && 'type' in p) p.type = lift(p.type);
      if(n.type === 'celltype') delete p.base2003;
      if(n.type === 'output' && Array.isArray(p.variants) && was2007)
        for(const v of p.variants) for(const e of v.edits || [])
          if(e.key === 'type' && typed.has(e.node)) e.value = lift(e.value);
    }
    delete d.form;
    return d;
  },
  // 8 -> 9: short-term plasticity releases before the facilitation jump by default, as published (stpOrder 0); scenes written before this ran with the jump first.
  // A checkpoint that has the term on, or a sweep tab that turns it on, declares the order it was tuned with (stpOrder 1), so the scene runs as it did.
  d => {
    for(const n of d.nodes || []){
      if(n.type !== 'output' || !n.params || n.params.stpOrder !== undefined) continue;
      const p = n.params;
      const viaTab = Array.isArray(p.variants) && p.variants.some(v =>
        (v.edits || []).some(e => e.node === n.id && e.key === 'stp' && +e.value > 0));
      if(+p.stp > 0 || viaTab) p.stpOrder = 1;
    }
    return d;
  },
  // 9 -> 10: node types renamed. dot is pin, merge is gather, read is footage, transform is move, the noise region is noisefield, and the checkpoint's key is checkpoint (it was output).
  // Only the type changes: parameters, wires and ids are as they were.
  d => {
    const MAP = { dot:'pin', merge:'gather', read:'footage', transform:'move', noise:'noisefield', output:'checkpoint' };
    for(const n of d.nodes || []) if(MAP[n.type]) n.type = MAP[n.type];
    return d;
  },
  // 10 -> 11: retype folded into cell type.
  // A retype node becomes a cell type node on the same built-in row (row is the type index plus one, zero being custom), with its population and fraction; the node id is kept, and the fraction's draw is seeded by it, so the same cells are drawn.
  // A cell type node from before is custom, which is all it could be.
  // A sweep tab edit of a retype's type follows.
  d => {
    const was = new Set();
    for(const n of d.nodes || []){
      const p = n.params || {};
      if(n.type === 'ntype'){
        was.add(n.id); n.type = 'celltype';
        n.params = { row:(p.type | 0) + 1, tag:p.tag || '', frac:p.frac === undefined ? 1 : p.frac };
      } else if(n.type === 'celltype' && p.row === undefined){ n.params = { ...p, row:0 }; }
    }
    for(const n of d.nodes || []){
      if(n.type !== 'checkpoint' || !n.params || !Array.isArray(n.params.variants)) continue;
      for(const v of n.params.variants) for(const e of v.edits || [])
        if(was.has(e.node) && e.key === 'type'){ e.key = 'row'; e.value = (+e.value | 0) + 1; }
    }
    return d;
  },
  // 11 -> 12: a scene lists the node modules its nodes come from (src/plugins.js scenePlugins), each with the node keys it uses.
  // A scene from before node modules uses none.
  d => {
    if(!Array.isArray(d.plugins)) d.plugins = [];
    return d;
  },
  // 12 -> 13: the wiring node is the connections file node (key connectionsfile).
  // Only the type changes: parameters, wires, ids and names are as they were.
  d => {
    for(const n of d.nodes || []) if(n.type === 'wiring') n.type = 'connectionsfile';
    return d;
  },
];

export function migrateScene(data){
  let d = { ...data };
  let from = d.format || 1;
  if(from > SCENE_FORMAT)
    throw new Error(`scene format ${from} is newer than this build understands (${SCENE_FORMAT})`);
  while(from < SCENE_FORMAT){ d = STEPS[from - 1](d); from++; }
  d.format = SCENE_FORMAT;
  return d;
}

import { NODE_DEFS } from './nodes.js';
import { defaultNodeName, tagToken, tagFollowsName, bumpName } from './tagname.js';
import { SCENE_FORMAT, migrateScene } from './migrate.js';
import { scenePlugins, checkScenePlugins } from './plugins.js';
import { matches as keyMatches, binding } from './keymap.js';

const W = 130, H = 26, PORT = 7;
// Zoom range.
// A scenario is several thousand pixels tall and wants about 0.05 to fit a short pane; a higher floor would stop the wheel going where frameAll had already gone and snap the view back on any zoom.
// One constant, used by both, low enough that the limit is the screen rather than the editor.
const MIN_SCALE = 0.01, MAX_SCALE = 2.5;
// advance per character of a group label at the 12px weight it is drawn in; layout.js keeps the same figure so its separation pass sizes boxes the way the renderer will
export const LABEL_ADV = 7.25;
// Hue for a group made from the UI, cycled so two made one after the other are told apart.
// The order is the palette the scenarios already use.
const GROUP_HUES = [205, 130, 25, 280, 55, 0];
// A dot is a reroute handle, so it is drawn as a small circle rather than a titled box: it carries no parameters and naming the wire is the point, not naming the node.
// Sized so its center stays outside the port hit radius, which keeps the body grabbable for dragging.
const DOT = 18;

export class NodeEditor {
  constructor(canvas, cb){                 // cb: { onSelect(node|null), onChange() }
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.nodes = []; this.nextId = 1;
    // A group names a set of nodes and draws behind them; it owns nothing and changes nothing about the computation, so a node can be moved out of one by dragging and the graph still computes.
    // The scenarios declare them because they know what each cluster is: the four nodes that make one cortical layer of one territory read as 'v.L4' at a glance and as four unlabelled boxes otherwise.
    this.groups = [];
    this.view = { x:60, y:30, scale:1 };
    this.viewSlots = {};                     // digit -> node id (viewer bindings)
    this.activeView = null;                  // node id currently viewed
    this.sel = null; this.selSet = new Set(); this.drag = null; this.hover = null;
    this.insertHit = null;                 // wire under a dragged free node
    this.cb = cb;
    this._bind();
    // The next frame is asked for whatever draw does.
    // Asked for after draw, one throw would end the loop for good: the pane freezes, clicks still land and move nodes nobody can see, a resize stretches the last image, and only the gestures that draw on their own flash through.
    const loop = () => {
      requestAnimationFrame(loop);
      try { this.draw(); this._drawFailed = false; }
      catch(err){
        if(!this._drawFailed){ this._drawFailed = true; console.error('graph draw', err);
          if(this.cb.onError) this.cb.onError('graph draw: ' + (err && err.message || err)); }
      }
    };
    requestAnimationFrame(loop);
  }
  byId(id){ return this.nodes.find(n => n.id===id); }
  addNode(type, x, y){
    const def = NODE_DEFS[type], params = {};
    def.params.forEach(p => params[p.k] = structuredClone(p.def));
    // A name from birth, unique among the nodes already here so two of a kind do not share it.
    // A node that carries a population tag adopts the name as its tag, so plugging one in tags it: the connect node's table can then see it without anyone typing the tag by hand.
    // The tag follows the name until it is edited to something else (tagname.js).
    const name = defaultNodeName(def.stem || def.title, this.nodes.map(m => m.name));   // a stem when the title is long (neuron scatter names itself scatter)
    const n = { id:this.nextId++, type, x, y, name, params,
      inputs:new Array(def.inputs).fill(null) };
    if(def.params.some(p => p.k === 'tag' && p.identity)) n.params.tag = name;
    this.nodes.push(n); return n;
  }
  clear(){ this.nodes = []; this.groups = []; this.sel = null; this.selSet = new Set(); this.cb.onSelect(null); }
  selectExternal(n){
    this.sel = n; this.selSet = new Set([n]); this.cb.onSelect(n);
    if(this.cb.onInspect) this.cb.onInspect(n);
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    const sx = this.view.x + (n.x+65)*this.view.scale, sy = this.view.y + (n.y+13)*this.view.scale;
    if(sx < 20 || sx > cw-20 || sy < 20 || sy > ch-20){
      this.view.x = cw/2 - (n.x+65)*this.view.scale;
      this.view.y = ch/2 - (n.y+13)*this.view.scale;
    }
  }
  toJSON(){
    return { v:1, format:SCENE_FORMAT, nextId:this.nextId, view:{ ...this.view },
      viewSlots:{ ...this.viewSlots }, activeView:this.activeView,
      scenarioName: this.scenarioName || '',
      plugins: scenePlugins(this.nodes),   // the node modules its nodes come from (src/plugins.js)
      groups: (this.groups || []).map(g =>
        ({ label:g.label, members:[...g.members], hue:g.hue, z:g.z || 0,
          pad:{ ...(g.pad || { l:0, t:0, r:0, b:0 }) } })),
      nodes: this.nodes.map(n => ({ id:n.id, type:n.type, x:n.x, y:n.y, on:n.on !== false,
        name:n.name || '', params:structuredClone(n.params),
        ...(Number.isInteger(n.built) ? { built:n.built } : {}),   // its place in a scenario's build, for saved layouts
        inputs:n.inputs.map(c => c ? { id:c.id } : null) })) };
  }
  load(data){
    data = migrateScene(data);           // older formats arrive here already current
    checkScenePlugins(data);             // a scene whose node modules are not loaded is refused whole, before anything here changes
    this.clear();
    this.nextId = data.nextId || 1;
    this.viewSlots = data.viewSlots ? { ...data.viewSlots } : {};
    this.activeView = data.activeView || null;
    if(data.view) this.view = { ...data.view };
    this.scenarioName = data.scenarioName || '';
    this.groups = (data.groups || []).map(g =>
      ({ label:g.label, members:[...(g.members || [])], hue:g.hue, z:g.z || 0,
        pad:{ l:0, t:0, r:0, b:0, ...(g.pad || {}) } }));
    for(const sn of data.nodes){
      const def = NODE_DEFS[sn.type]; if(!def) continue;
      const params = {};
      def.params.forEach(pd => params[pd.k] =
        sn.params && sn.params[pd.k] !== undefined ? structuredClone(sn.params[pd.k]) : structuredClone(pd.def));
      // keep arrays longer than the node's declared port count (a saved graph can carry a real wire there)
      const nIn = Math.max(def.inputs, sn.inputs ? sn.inputs.length : 0);
      this.nodes.push({ id:sn.id, type:sn.type, x:sn.x, y:sn.y, params,
        name:sn.name || '', on:sn.on === false ? false : true,
        ...(Number.isInteger(sn.built) ? { built:sn.built } : {}),
        inputs:new Array(nIn).fill(null).map((_,i) => sn.inputs && sn.inputs[i] ? { id:sn.inputs[i].id } : null) });
    }
    for(const n of this.nodes)
      n.inputs = n.inputs.map(c => c && this.byId(c.id) ? c : null);
  }
  deleteNode(n){
    const def = NODE_DEFS[n.type];                 // splice through MAIN inputs only, never masks
    const src = n.inputs.find((c,i) => c && (def.side===undefined || i < def.side)) || null;
    this.nodes = this.nodes.filter(m => m!==n);
    for(const m of this.nodes){
      let touched = false;
      m.inputs = m.inputs.map(c => {
        if(c && c.id===n.id){ touched = true; return src ? { id:src.id } : null; }
        return c;
      });
      if(touched) this.markDirty(m);
    }
    this.selSet.delete(n);
    if(this.sel===n){ this.sel = null; this.cb.onSelect(null); }
    // A group holds ids, so a deleted node has to leave its membership, and a group whose members have all gone stops being a group rather than lingering as an invisible entry that a later paste could revive.
    this.groups = this.groups.filter(g => {
      g.members = g.members.filter(id => this.byId(id));
      return g.members.length > 0;
    });
    this.cb.onChange();
  }
  // ---- groups ----
  // Smallest group whose backdrop covers a point.
  // The whole box counts here, not just the header band the drag uses: a right click or a double click anywhere on a backdrop should reach the group it belongs to.
  groupAt(gx, gy){
    let best = null, bestZ = -Infinity, bestArea = Infinity;
    for(const g of (this.groups || [])){
      const b = this.groupBox(g);
      if(!b || gx < b.x0 || gx > b.x1 || gy < b.y0 || gy > b.y1) continue;
      const z = g.z || 0, area = (b.x1-b.x0)*(b.y1-b.y0);
      if(z > bestZ || (z === bestZ && area < bestArea)){
        bestZ = z; bestArea = area; best = g;
      }
    }
    return best;
  }
  // Drop the room added by dragging the edges, so the backdrop closes back onto the nodes it names.
  fitGroup(g){
    g.pad = { l:0, t:0, r:0, b:0 };
    if(this.cb.onMoved) this.cb.onMoved();
  }
  // Depth is a plain number so it can also be typed, and the ends are one past the current extremes rather than a fixed range, so a group put in front stays there.
  raiseGroup(g, front){
    const zs = this.groups.map(x => x.z || 0);
    g.z = front ? Math.max(...zs) + 1 : Math.min(...zs) - 1;
    if(this.cb.onMoved) this.cb.onMoved();
    return g.z;
  }
  createGroup(nodes, label, hue){
    const members = [...new Set(nodes.filter(Boolean).map(n => n.id))];
    if(!members.length) return null;
    // A group made by hand goes in front of what is already there, since it is usually made around a cluster that sits inside a bigger backdrop.
    const zs = this.groups.map(x => x.z || 0);
    const g = { label: label || 'group', members,
      hue: hue === undefined ? GROUP_HUES[this.groups.length % GROUP_HUES.length] : hue,
      z: zs.length ? Math.max(...zs) + 1 : 0 };
    this.groups.push(g);
    // onMoved is the record-history-but-do-not-rewire callback: a group is display only and never reaches a computation, exactly like a position.
    if(this.cb.onMoved) this.cb.onMoved();
    return g;
  }
  ungroup(g){
    const i = this.groups.indexOf(g);
    if(i < 0) return false;
    this.groups.splice(i, 1);
    if(this.cb.onMoved) this.cb.onMoved();
    return true;
  }
  // Grouping is a thing you do to a selection, so one node is not enough: a single-node backdrop is noise, and the gesture would fire by accident every time a shortcut was pressed with something selected.
  groupSelected(){
    const list = this.selection();
    if(list.length < 2) return null;
    const g = this.createGroup(list);
    if(g && this.cb.onInspectGroup) this.cb.onInspectGroup(g);   // name it now
    return g;
  }
  // ---- clipboard ----
  selection(){
    return this.selSet.size ? [...this.selSet] : (this.sel ? [this.sel] : []);
  }
  // Copy a set of nodes, keeping the wires that run between them and dropping the ones that leave the set, so a copied branch pastes as a working branch rather than a pile of loose nodes.
  // A copied node is a new node: its name steps on from the original's so the two are told apart, and a node that carries a population tag gets a tag of its own, because a copy under the same tag is not a second population but the same one addressed twice by every table, probe and projection.
  // A scatter's seed steps on too: the same seed on the same geometry draws the same points, which would put every copied cell exactly on top of an original.
  _asCopy(c, n, foreign){
    c.params = structuredClone(n.params);
    if(n.on === false) c.on = false;
    const def = NODE_DEFS[n.type];
    const others = this.nodes.filter(m => m !== c);
    const base = n.name || def.title;
    // Pasted from another scene, the original is not here to be told apart from: the node keeps its name, its tag and its seed when nothing in this scene has that name or tag already.
    if(foreign && n.name){
      const low = v => String(v || '').trim().toLowerCase();
      const tagP = def.params.find(p => p.k === 'tag' && p.identity);
      const clash = others.some(m => low(m.name) === low(n.name)) ||
        (tagP && others.some(m => m.params && low(m.params.tag) && low(m.params.tag) === low(n.params.tag)));
      if(!clash){ c.name = n.name; return; }
    }
    c.name = bumpName(base, others.map(m => m.name));
    const tagP = def.params.find(p => p.k === 'tag' && p.identity);
    if(tagP){
      const tags = others.filter(m => m.type === n.type).map(m => m.params.tag);
      c.params.tag = tagFollowsName(n.name, n.params.tag)
        ? tagToken(c.name) : bumpName(n.params.tag, tags);
      if(c.params.seed !== undefined){
        const seeds = others.filter(m => m.type === n.type).map(m => m.params.seed | 0);
        c.params.seed = Math.max(0, ...seeds) + 1;
      }
    }
  }
  duplicate(list, dx = 24, dy = 24){
    const map = new Map(), made = [];
    for(const n of list){
      const c = this.addNode(n.type, n.x + dx, n.y + dy);
      this._asCopy(c, n);
      map.set(n.id, c.id); made.push(c);
    }
    made.forEach((c, i) => {
      c.inputs = list[i].inputs.map(w =>
        w ? { id:map.has(w.id) ? map.get(w.id) : w.id } : null);
    });
    return made;
  }
  cloneSelection(){
    const list = this.selection();
    if(!list.length) return null;
    const made = this.duplicate(list);
    this.sel = made[made.length-1]; this.selSet = new Set(made);
    this.cb.onSelect(this.sel);
    this.cb.onChange();
    return made;
  }
  copySelection(){
    const list = this.selection();
    if(!list.length) return;
    const idx = new Map(list.map((n, i) => [n.id, i]));
    // wires inside the set are kept by index; a wire from outside the set is kept by the source's id, so a pasted copy still reads what the original read (a copied scatter keeps its geometry) while that source exists
    this.clipboard = list.map(n => ({ type:n.type, name:n.name || '',
      params:structuredClone(n.params), on:n.on !== false,
      dx:n.x - list[0].x, dy:n.y - list[0].y,
      inputs:n.inputs.map(w => (w && idx.has(w.id)) ? idx.get(w.id) : null),
      ext:n.inputs.map(w => (w && !idx.has(w.id)) ? w.id : null) }));
    // which scene the copy came from (main.js sets docKey per scene tab): an outside wire is an id, and an id means another node in another scene
    this.clipboard.from = this.docKey;
  }
  pasteClipboard(){
    const cb = this.clipboard;
    if(!cb || !cb.length) return;
    const [gx, gy] = this.cursorGraph();
    const anchor = this.sel;
    const made = cb.map(s => {
      const n = this.addNode(s.type, Math.round(gx - W/2 + s.dx), Math.round(gy - H/2 + s.dy));
      this._asCopy(n, { type:s.type, name:s.name, params:s.params, on:s.on ? true : false }, cb.from !== this.docKey);
      return n;
    });
    made.forEach((n, i) => {
      n.inputs = cb[i].inputs.map((k, port) => k !== null ? { id:made[k].id }
        : (cb.from === this.docKey && cb[i].ext && cb[i].ext[port] !== null && this.byId(cb[i].ext[port]))
          ? { id:cb[i].ext[port] } : null);
    });
    // A single pasted node joins the pipe after the selection, as it always has, unless it kept an upstream wire of its own.
    // A pasted branch arrives with its own wires and is left loose, because there is no one node in it that the pipe should run through.
    if(made.length === 1 && !made[0].inputs.some(Boolean)) this._insertAfter(anchor, made[0]);
    this.sel = made[made.length-1]; this.selSet = new Set(made);
    this.cb.onSelect(this.sel);
    if(made.length === 1 && this.cb.onInspect) this.cb.onInspect(made[0]);
    this.cb.onChange();
  }
  deleteSelection(){
    const del = this.selection();
    if(!del.length) return;
    this.selSet = new Set();
    for(const n of del) if(this.nodes.includes(n)) this.deleteNode(n);
  }
  extractNode(n){
    const users = this.nodes.filter(m => m.inputs.some(c => c && c.id===n.id));
    if(!users.length && !n.inputs.some(c => c)) return;
    const def = NODE_DEFS[n.type];
    const src = n.inputs.find((c,i) => c && (def.side===undefined || i < def.side)) || null;
    for(const m of users){
      m.inputs = m.inputs.map(c => c && c.id===n.id ? (src ? { id:src.id } : null) : c);
      this.markDirty(m);
    }
    n.inputs = n.inputs.map(() => null);
    this.markDirty(n);
    n._flash = performance.now();
    this.cb.onChange();
  }
  markDirty(node){
    node._cache = null;
    for(const m of this.nodes)
      if(m.inputs.some(c => c && c.id===node.id)) this.markDirty(m);
  }
  dependsOn(node, targetId){
    if(node.id===targetId) return true;
    return node.inputs.some(c => c && this.dependsOn(this.byId(c.id), targetId));
  }
  // Splice free nodes into a wire, in the order they sit on screen from top to bottom (ties left to right): src -> first -> ... -> last -> dst.
  // One node is a drop-to-insert; several are a chain in one gesture.
  insertChain(wr, nodes){
    const chain = [...nodes].sort((a, b) => a.y - b.y || a.x - b.x);
    if(!chain.length) return;
    let prev = wr.src;
    for(const n of chain){ n.inputs[0] = { id:prev.id }; n._cache = null; prev = n; }
    wr.dst.inputs[wr.port] = { id:prev.id };
    this.markDirty(wr.dst);
    return chain;
  }
  isFree(n){
    return n.inputs.length > 0 && n.inputs.every(c => !c) &&
      !this.nodes.some(m => m.inputs.some(c => c && c.id===n.id));
  }
  // ---- geometry ----
  isSide(n, i){ const d = NODE_DEFS[n.type]; return d.side !== undefined && i >= d.side; }
  inPort(n, i){
    if(n.type === 'pin') return [ n.x + DOT/2, n.y ];
    if(this.isSide(n, i)){
      const d = NODE_DEFS[n.type], k = i - d.side, total = n.inputs.length - d.side;
      return [ n.x + W, n.y + H/2 + (k - (total-1)/2)*12 ];
    }
    // top ports at a 16 px pitch, closing up when there are more than the box holds so the last one still lands inside it
    const d = NODE_DEFS[n.type];
    const top = d.side !== undefined ? d.side : n.inputs.length;
    const pitch = top > 1 ? Math.min(16, (W - 32)/(top - 1)) : 16;
    return [ n.x + 16 + i*pitch, n.y ];
  }
  // A node whose port count is a parameter (merge) keeps its inputs array the length the parameter says, never dropping a wired port, and grows a spare when every port is taken so there is always somewhere to plug in.
  _syncPorts(n){
    const d = NODE_DEFS[n.type];
    if(!d.portsParam) return;
    let want = Math.max(2, n.params[d.portsParam] | 0 || d.inputs);
    let last = -1;
    n.inputs.forEach((c, i) => { if(c) last = i; });
    if(want < last + 1) want = last + 1;
    if(n.inputs.length && want <= n.inputs.length && n.inputs.every(Boolean)) want = n.inputs.length + 1;
    if(want !== (n.params[d.portsParam] | 0)) n.params[d.portsParam] = want;
    while(n.inputs.length < want) n.inputs.push(null);
    if(n.inputs.length > want) n.inputs.length = want;
  }
  outPort(n){
    if(n.type === 'pin') return [ n.x + DOT/2, n.y + DOT ];
    return [ n.x + W/2, n.y + H ];
  }
  toGraph(e){ const r = this.canvas.getBoundingClientRect();
    return [ (e.clientX-r.left-this.view.x)/this.view.scale,
             (e.clientY-r.top-this.view.y)/this.view.scale ]; }
  dims(n){
    if(n.type === 'note') return [n._noteW || n.params.width || 220, n._noteH || 26];
    if(n.type === 'pin') return [DOT, DOT];
    return [W, H];
  }
  // Dragging a wire or a node to the edge of the graph pane pans the view that way, so a connection can be made to something off screen without letting go: the pane is a window onto a graph that is usually larger than it, and the drag had no way to reach past its own edge.
  _edgePan(e){
    const r = this.canvas.getBoundingClientRect();
    const M = 36;                          // how close to the edge starts it
    const depth = v => Math.min(1, Math.max(0, v)/M);
    const vx = depth(r.left + M - e.clientX) - depth(e.clientX - (r.right - M));
    const vy = depth(r.top + M - e.clientY) - depth(e.clientY - (r.bottom - M));
    // only while the pointer is over the pane's own band, not miles away
    const near = e.clientX > r.left - M*3 && e.clientX < r.right + M*3 &&
                 e.clientY > r.top - M*3 && e.clientY < r.bottom + M*3;
    if(!near || (!vx && !vy)){ this._stopEdgePan(); return; }
    this._pan = { vx, vy };
    if(this._panRaf) return;
    const step = () => {
      if(!this.drag || !this._pan){ this._panRaf = 0; return; }
      const SPEED = 14;                    // screen pixels a frame at the very edge
      this.view.x += this._pan.vx*SPEED;
      this.view.y += this._pan.vy*SPEED;
      // the dragged thing follows the view: the pointer has not moved, but the graph under it has
      if(this.lastMouse) this._onMove(this.lastMouse);
      this.draw();
      this._panRaf = requestAnimationFrame(step);
    };
    this._panRaf = requestAnimationFrame(step);
  }
  _stopEdgePan(){
    this._pan = null;
    if(this._panRaf){ cancelAnimationFrame(this._panRaf); this._panRaf = 0; }
  }
  hitPort(gx, gy){
    const t0 = Math.min(7/this.view.scale, 9);  // 7px on screen, but never wider than the drawn port
    for(const n of this.nodes){
      if(n.type === 'note') continue;           // notes have no ports
      // and never wider than a third of the node it belongs to: a dot is 18 units across, so at a zoomed-out view its two ports covered the whole handle and dragging one started a wire instead of moving it.
      const [dw, dh] = this.dims(n);
      const t = Math.min(t0, Math.min(dw, dh)/3);
      for(let i=0;i<n.inputs.length;i++){ const [px,py]=this.inPort(n,i);
        if(Math.abs(gx-px)<t && Math.abs(gy-py)<t) return { node:n, port:i, kind:'in' }; }
      const [px,py]=this.outPort(n);
      if(Math.abs(gx-px)<t && Math.abs(gy-py)<t) return { node:n, kind:'out' };
    }
    return null;
  }
  hitNode(gx, gy){
    for(let i=this.nodes.length-1;i>=0;i--){ const n=this.nodes[i];
      const [w, h] = this.dims(n);
      if(gx>=n.x && gx<=n.x+w && gy>=n.y && gy<=n.y+h) return n; }
    return null;
  }
  wireList(){
    const out = [];
    for(const n of this.nodes) n.inputs.forEach((c,i) => {
      if(!c) return; const s = this.byId(c.id);
      if(s) out.push({ src:s, dst:n, port:i });
    });
    return out;
  }
  // The control points a wire is drawn with, so a hit test can walk the curve that is on screen.
  // A mask wire comes into its port from the side and bulges to the right of it; testing every wire against the main wire's curve meant a click on a mask wire landed on empty canvas and ctrl-click did nothing there.
  _wireCurve(wr){
    const [x1, y1] = this.outPort(wr.src), [x2, y2] = this.inPort(wr.dst, wr.port);
    return this.isSide(wr.dst, wr.port)
      ? [x1, y1, x1, y1 + 34, x2 + 34, y2, x2, y2]
      : [x1, y1, x1, y1 + 30, x2, y2 - 30, x2, y2];
  }
  hitWire(gx, gy, exclude, allowSide){
    // The threshold is a distance on screen, so it grows in graph units as the view zooms out; a fixed 14 graph units is a hairline at 0.4 zoom.
    const T = 14/Math.max(0.15, this.view.scale), T2 = T*T;
    for(const wr of this.wireList()){
      if(wr.src===exclude || wr.dst===exclude) continue;
      // only a node that passes its input through can be spliced into a mask wire, since what travels there is a geometry and everything else would hand the port something it cannot use
      if(!allowSide && this.isSide(wr.dst, wr.port)) continue;
      const [x1, y1, cx1, cy1, cx2, cy2, x2, y2] = this._wireCurve(wr);
      // Sampled along its own length rather than at a fixed twenty points: a long wire left gaps between samples wider than the threshold, so a click on the line landed between them and did nothing.
      const len = Math.hypot(cx1-x1, cy1-y1) + Math.hypot(cx2-cx1, cy2-cy1) +
                  Math.hypot(x2-cx2, y2-cy2);
      const steps = Math.max(24, Math.min(400, Math.ceil(len/Math.max(4, T/3))));
      const dt = 1/steps;
      for(let t=0; t<=1.001; t+=dt){
        const mt = 1-t;
        const bx = mt*mt*mt*x1 + 3*mt*mt*t*cx1 + 3*mt*t*t*cx2 + t*t*t*x2;
        const by = mt*mt*mt*y1 + 3*mt*mt*t*cy1 + 3*mt*t*t*cy2 + t*t*t*y2;
        const dx = gx-bx, dy = gy-by;
        if(dx*dx+dy*dy < T2) return wr;
      }
    }
    return null;
  }
  // What a mask wire carries is a geometry, so a node can be spliced into one if it hands its input straight on (a dot) or if it takes a geometry and gives one back (a transform, which moves the region).
  passesThrough(n){
    const d = n && NODE_DEFS[n.type];
    return !!(d && (d.passthrough || d.carriesGeo));
  }
  // The pin is transparent to the wiring of the network (see the bypass rule in sigOf), so this is a pure layout operation even on a graph that took minutes to compute.
  insertDot(wr, gx, gy){
    const n = this.addNode('pin', Math.round(gx - DOT/2), Math.round(gy - DOT/2));
    n.inputs[0] = { id:wr.src.id };
    wr.dst.inputs[wr.port] = { id:n.id };
    this.markDirty(wr.dst);
    this.sel = n; this.selSet = new Set();
    this.cb.onSelect(n); this.cb.onChange();
    n._flash = performance.now();
    return n;
  }
  // where the pointer is in graph space, or the middle of the view if it is outside the canvas
  cursorGraph(){
    const r = this.canvas.getBoundingClientRect(), lm = this.lastMouse;
    if(lm && lm.clientX>=r.left && lm.clientX<=r.right && lm.clientY>=r.top && lm.clientY<=r.bottom)
      return this.toGraph(lm);
    return [(r.width/2 - this.view.x)/this.view.scale, (r.height/2 - this.view.y)/this.view.scale];
  }
  frameAll(){
    if(!this.nodes.length) return;
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    if(!cw || !ch) return;                 // a pane with no size yet would frame to the minimum scale
    const xs = this.nodes.map(n=>n.x).filter(Number.isFinite), ys = this.nodes.map(n=>n.y).filter(Number.isFinite);
    if(!xs.length) return;
    const x0 = Math.min(...xs)-40, x1 = Math.max(...xs)+W+40;
    const y0 = Math.min(...ys)-40, y1 = Math.max(...ys)+H+40;
    const s = Math.max(MIN_SCALE,
      Math.min(cw/(x1-x0), ch/(y1-y0), 1.4));
    this.view.scale = s;
    this.view.x = (cw - (x0+x1)*s)/2;
    this.view.y = (ch - (y0+y1)*s)/2;
  }
  // insert a fresh node into the pipe after src: src's downstream main wires move to n, and n takes src as its input
  _insertAfter(src, n){
    if(!src || src === n || src.type === 'note') return;
    const def = NODE_DEFS[n.type];
    if(!def.inputs) return;
    const side = this.passesThrough(n);    // a dot can stand in a mask wire
    for(const m of this.nodes){
      if(m === n) continue;
      m.inputs.forEach((c, i) => {
        if(c && c.id === src.id && (side || !this.isSide(m, i))){
          m.inputs[i] = { id:n.id }; this.markDirty(m);
        }
      });
    }
    n.inputs[0] = { id:src.id };
    this.markDirty(n);
  }
  // ---- connect helpers ----
  _connect(srcNode, dstNode, port){
    if(srcNode===dstNode || this.dependsOn(srcNode, dstNode.id)) return false;
    dstNode.inputs[port] = { id:srcNode.id };
    this.markDirty(dstNode); this.cb.onChange();
    return true;
  }
  // ---- events ----
  // A handler that throws leaves the pane in whatever state it was in, with nothing said: the drag it was in the middle of stays, and every move after it throws the same way, which reads as the pane going dead.
  // Every handler runs through this: the drag is dropped, the pane redrawn, and the error goes to the status line and the console.
  _guard(what, fn){
    return e => {
      try { fn(e); }
      catch(err){
        this.drag = null; this.insertHit = null; this._stopEdgePan();
        console.error('graph ' + what, err);
        if(this.cb.onError) this.cb.onError('graph ' + what + ': ' + (err && err.message || err));
        try { this.draw(); } catch(e2){}
      }
    };
  }
  _bind(){
    const cv = this.canvas;
    cv.addEventListener('contextmenu', this._guard('menu', e => { e.preventDefault(); this._ctxMenu(e); }));
    cv.addEventListener('dblclick', this._guard('double click', e => {
      const [gx,gy] = this.toGraph(e);
      const n = this.hitNode(gx,gy);
      if(!n){                            // a backdrop opens the group's own panel
        const g = this.groupAt(gx, gy);
        if(g && this.cb.onInspectGroup) this.cb.onInspectGroup(g);
        return;
      }
      if(this.cb.onInspect) this.cb.onInspect(n);
      const editable = NODE_DEFS[n.type].cat === 'regions' || n.type === 'move' ||
        (n.type === 'stimulus' && !n.inputs[1]);   // raw stimulus: gizmo drives its own region
      if(editable && this.cb.onEditGeo) this.cb.onEditGeo(n);
    }));
    cv.addEventListener('mousedown', this._guard('press', e => {
      this._closeMenu();
      if(e.button===1){                   // middle-drag pans
        e.preventDefault();
        this.drag = { mode:'pan', sx:e.clientX, sy:e.clientY, vx:this.view.x, vy:this.view.y };
        return;
      }
      if(e.button!==0) return;
      const [gx,gy] = this.toGraph(e);
      if(e.ctrlKey || e.metaKey){          // ctrl-click a wire to reroute it
        const wr = this.hitWire(gx, gy, null, true);
        if(wr){ e.preventDefault(); this.insertDot(wr, gx, gy); return; }
      }
      const p = this.hitPort(gx,gy);
      if(p){
        if(p.kind==='out'){
          this.drag = { mode:'wire', fixed:{ kind:'out', node:p.node }, gx, gy };
        } else if(p.node.inputs[p.port]){   // unplug and keep dragging from the source
          const src = this.byId(p.node.inputs[p.port].id);
          p.node.inputs[p.port] = null; this.markDirty(p.node); this.cb.onChange();
          this.drag = { mode:'wire', fixed:{ kind:'out', node:src }, gx, gy };
        } else {                            // drag from empty input toward an output
          this.drag = { mode:'wire', fixed:{ kind:'in', node:p.node, port:p.port }, gx, gy };
        }
        return;
      }
      let n = this.hitNode(gx,gy);
      if(n){
        if(e.altKey){                       // alt-drag: clone the node and drag the copy
          const copy = this.addNode(n.type, n.x+14, n.y+14);
          copy.params = structuredClone(n.params);
          if(n.on === false) copy.on = false;
          n = copy; this.cb.onChange();
        }
        this.sel = n; this.cb.onSelect(n);
        if(!this.selSet.has(n)) this.selSet = new Set([n]);
        const group = [...this.selSet].map(m => ({ node:m, ox:gx-m.x, oy:gy-m.y }));
        this.drag = { mode:'node', node:n, ox:gx-n.x, oy:gy-n.y, group,
          // several free nodes dragged together insert as a chain; shaking a wire loose stays a single-node gesture
          canInsert:group.every(g => this.isFree(g.node)) }; }
      else if(this.hitGroup(gx, gy)){
        const hit = this.hitGroup(gx, gy);
        this.drag = { mode:'group', group:hit.group,
          members: hit.box.ms.map(m => ({ node:m, ox:gx-m.x, oy:gy-m.y })) };
      }
      else if(this.hitGroupEdge(gx, gy)){
        const hit = this.hitGroupEdge(gx, gy);
        const g = hit.group;
        g.pad = { l:0, t:0, r:0, b:0, ...(g.pad || {}) };
        this.drag = { mode:'gsize', group:g, edge:hit, gx, gy,
          pad0: { ...g.pad } };
      }
      else {
        const base = e.shiftKey ? new Set(this.selSet) : new Set();
        if(!e.shiftKey){ this.sel = null; this.selSet = base; this.cb.onSelect(null); }
        this.drag = { mode:'marquee', x0:gx, y0:gy, x1:gx, y1:gy, base }; }
    }));
    window.addEventListener('mousemove', this._guard('move', e => this._onMove(e)));
    // A release the page never saw (the button let go over another window, or a dialog took the event) left the drag running: the node followed the pointer until the next click, and an edge pan carried the view off with it.
    // A move with no button held ends the drag where the pointer is; a wire being drawn is dropped.
    window.addEventListener('blur', () => {
      if(!this.drag) return;
      this.drag = null; this.insertHit = null; this._stopEdgePan(); this.draw();
    });
    this._onMove = e => {
      this.lastMouse = { clientX:e.clientX, clientY:e.clientY };
      if(this.drag && e.buttons === 0){
        if(this.drag.mode === 'wire'){ this.drag = null; this._stopEdgePan(); this.draw(); }
        else this._endDrag(e);
        return;
      }
      if(this.drag) this._edgePan(e);
      const [gx,gy] = this.toGraph(e);
      this.hover = this.hitPort(gx,gy);
      if(!this.drag){
        // a resize edge is invisible, so the cursor is the only thing that says it is there
        const ed = this.hover || this.hitNode(gx,gy) ? null : this.hitGroupEdge(gx,gy);
        cv.style.cursor = !ed ? ''
          : (ed.l || ed.r) && (ed.t || ed.b) ? (ed.l === ed.t ? 'nwse-resize' : 'nesw-resize')
          : (ed.l || ed.r) ? 'ew-resize' : 'ns-resize';
      }
      if(!this.drag) return;
      const d = this.drag;
      if(d.mode==='group'){
        for(const m of d.members){ m.node.x = gx-m.ox; m.node.y = gy-m.oy; }
        this.draw();
        return;
      }
      if(d.mode==='gsize'){
        // padding only grows, so an edge dragged past its own nodes stops there rather than cutting the backdrop across them
        const p = d.pad0, dx = gx - d.gx, dy = gy - d.gy, e = d.edge;
        const set = { ...p };
        if(e.l) set.l = Math.max(0, p.l - dx);
        if(e.r) set.r = Math.max(0, p.r + dx);
        if(e.t) set.t = Math.max(0, p.t - dy);
        if(e.b) set.b = Math.max(0, p.b + dy);
        d.group.pad = set;
        this.draw();
        return;
      }
      if(d.mode==='node'){
        for(const g of d.group){ g.node.x = gx-g.ox; g.node.y = gy-g.oy; }
        if(d.group.length===1){           // shake / insert only for single-node drags
          const sh = d.shake || (d.shake = { lastX:gx, sign:0, flips:[] });
          const dx = gx - sh.lastX;
          if(Math.abs(dx) > 6/this.view.scale){
            const sg = Math.sign(dx);
            if(sh.sign && sg !== sh.sign){
              const now = performance.now();
              sh.flips = sh.flips.filter(t => now - t < 700);
              sh.flips.push(now);
              if(sh.flips.length >= 4){ sh.flips = []; this.extractNode(d.node); d.canInsert = this.isFree(d.node); }
            }
            sh.sign = sg; sh.lastX = gx;
          }
        }
        if(d.canInsert){
          // the dragged node's own center: a dot is 18 units across, and testing a point half a full node to its right missed every wire.
          // A group is tested by the node under the pointer, and may land on a mask wire only if every node in it passes geometry through.
          const [dw, dh] = this.dims(d.node);
          this.insertHit = this.hitWire(d.node.x+dw/2, d.node.y+dh/2, d.node,
            d.group.every(g => this.passesThrough(g.node)));
        } else this.insertHit = null;
      }
      else if(d.mode==='marquee'){ d.x1 = gx; d.y1 = gy; this._marqueeSelect(d); }
      else if(d.mode==='pan'){ this.view.x = d.vx + e.clientX-d.sx; this.view.y = d.vy + e.clientY-d.sy; }
      else if(d.mode==='wire'){ d.gx = gx; d.gy = gy; }
    };
    window.addEventListener('mouseup', this._guard('release', e => this._endDrag(e)));
    this._endDrag = e => {
      this._stopEdgePan();
      const d = this.drag; this.drag = null;
      if(!d) return;
      if(d.mode==='group' || d.mode==='gsize'){
        // a drag lands on fractional graph units at most zoom levels, and the panel shows these numbers, so they settle on whole pixels
        if(d.mode==='gsize'){
          const p = d.group.pad;
          for(const k of ['l','t','r','b']) p[k] = Math.round(p[k]);
        }
        if(this.cb.onMoved) this.cb.onMoved();
        this.draw();
        return;
      }
      if(d.mode==='marquee'){
        const list = [...this.selSet];
        this.sel = list.length ? list[list.length-1] : null;
        this.cb.onSelect(this.sel);
        return;
      }
      if(d.mode==='node'){
        if(this.cb.onMoved) this.cb.onMoved();
        const wr = this.insertHit; this.insertHit = null;
        if(wr){                             // drop onto a wire -> insert between
          this.insertChain(wr, d.group.map(g => g.node));
          this.cb.onChange();
        }
        return;
      }
      if(d.mode!=='wire') return;
      const [gx,gy] = this.toGraph(e);
      const p = this.hitPort(gx,gy);
      if(d.fixed.kind==='out'){
        if(p && p.kind==='in') this._connect(d.fixed.node, p.node, p.port);
        else if(!p && !this.hitNode(gx,gy)) this._menu(e, d.fixed);
      } else {
        if(p && p.kind==='out') this._connect(p.node, d.fixed.node, d.fixed.port);
        else if(!p && !this.hitNode(gx,gy)) this._menu(e, d.fixed);
      }
    };
    cv.addEventListener('wheel', this._guard('wheel', e => {
      e.preventDefault();
      // pinch (reported with ctrlKey) and classic mouse-wheel notches zoom; trackpad two-finger scroll (fractional pixel deltas) pans
      const notch = e.deltaMode !== 0 ||
        (e.deltaX === 0 && Math.abs(e.deltaY) >= 100 && Number.isInteger(e.deltaY));
      if(e.ctrlKey || e.metaKey || notch){
        const r = cv.getBoundingClientRect(), mx = e.clientX-r.left, my = e.clientY-r.top;
        const s0 = this.view.scale;
        const step = e.ctrlKey || e.metaKey ? Math.exp(-e.deltaY*0.01) : (e.deltaY>0 ? 0.9 : 1.1);
        const s1 = Math.min(MAX_SCALE, Math.max(MIN_SCALE, s0 * step));
        this.view.x = mx - (mx-this.view.x)*s1/s0;
        this.view.y = my - (my-this.view.y)*s1/s0;
        this.view.scale = s1;
      } else {
        this.view.x -= e.deltaX;
        this.view.y -= e.deltaY;
      }
    }), { passive:false });
    window.addEventListener('keydown', e => {
      if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
      if(keyMatches(e, 'cancel')){ this.drag = null; this.insertHit = null; this._closeMenu(); }
      else if(keyMatches(e, 'addNode')){
        e.preventDefault();
        const r = this.canvas.getBoundingClientRect();
        let pos = this.lastMouse;
        if(!pos || pos.clientX < r.left || pos.clientX > r.right || pos.clientY < r.top || pos.clientY > r.bottom)
          pos = { clientX:r.left + r.width/2, clientY:r.top + r.height/2 };
        this._menu(pos, null);
      }
      else if(keyMatches(e, 'frameAll')) this.frameAll();
      else if(keyMatches(e, 'findNode')){ e.preventDefault(); this._find(); }
      else if(keyMatches(e, 'addDot')){
        e.preventDefault();
        const [gx, gy] = this.cursorGraph();
        const wr = this.hitWire(gx, gy, null, true);
        if(wr) this.insertDot(wr, gx, gy);
        else {                             // no wire under the cursor: a loose handle
          const n = this.addNode('pin', Math.round(gx - DOT/2), Math.round(gy - DOT/2));
          this.sel = n; this.selSet = new Set();
          this.cb.onSelect(n); this.cb.onChange();
          n._flash = performance.now();
        }
      }
      else if(keyMatches(e, 'deleteNode') && (this.selSet.size || this.sel)){
        e.preventDefault(); this.deleteSelection();
      }
      else if(keyMatches(e, 'copy')) this.copySelection();
      else if(keyMatches(e, 'paste')){ e.preventDefault(); this.pasteClipboard(); }
      else if(keyMatches(e, 'clone')){ e.preventDefault(); this.cloneSelection(); }
      else if(keyMatches(e, 'groupSel')){ e.preventDefault(); this.groupSelected(); }
      else if(keyMatches(e, 'ungroup')){
        e.preventDefault();
        const g = this.groupAt(...this.cursorGraph());
        if(g) this.ungroup(g);
      }
      else if(keyMatches(e, 'bypass') && (this.sel || this.selSet.size)){
        // every selected node goes to the state the primary one is toggled to, so a mixed selection ends up in step rather than swapped
        const list = this.selSet.size ? [...this.selSet] : [this.sel];
        if(this.sel && !list.includes(this.sel)) list.push(this.sel);
        const lead = this.sel || list[0];
        const to = lead.on === false;                // toggle: bypassed -> on, on -> bypassed
        for(const n of list){ n.on = to; this.markDirty(n); }
        this.cb.onChange();
      }
    });
  }
  // What the right click menu offers depends on what the cursor is over: a selection can be grouped, a backdrop can be opened or dissolved, and the clipboard actions are listed because a menu is where they are looked for whether or not the shortcut is known.
  // Each row shows its binding, so the menu also teaches the keyboard.
  _ctxMenu(e){
    this._closeMenu();
    const [gx, gy] = this.toGraph(e);
    const n = this.hitNode(gx, gy);
    // right-clicking a node outside the selection selects it first, so the menu always acts on what is being pointed at rather than on whatever happened to be selected before
    if(n && !this.selSet.has(n)){
      this.sel = n; this.selSet = new Set([n]); this.cb.onSelect(n);
    }
    const sel = this.selection(), grp = this.groupAt(gx, gy);
    const has = sel.length > 0, held = !!(this.clipboard && this.clipboard.length);
    const items = [];
    if(sel.length > 1)
      items.push({ label:`group ${sel.length} nodes`, key:binding('groupSel'),
        run:() => this.groupSelected() });
    if(grp){
      items.push({ label:'group properties',
        run:() => { if(this.cb.onInspectGroup) this.cb.onInspectGroup(grp); } });
      items.push({ label:'ungroup ' + (grp.label || ''), key:binding('ungroup'),
        run:() => this.ungroup(grp) });
    }
    if(items.length) items.push(null);
    items.push({ label:'copy',   key:binding('copy'),   off:!has,  run:() => this.copySelection() });
    items.push({ label:'paste',  key:binding('paste'),  off:!held, run:() => this.pasteClipboard() });
    items.push({ label:'clone',  key:binding('clone'),  off:!has,  run:() => this.cloneSelection() });
    items.push({ label:'delete', key:binding('deleteNode'), off:!has, run:() => this.deleteSelection() });
    items.push(null);
    items.push({ label:'add node', key:binding('addNode'), run:() => this._menu(e, null) });

    const m = document.createElement('div'); m.id = 'ctxmenu';
    for(const it of items){
      if(!it){ const s = document.createElement('div'); s.className = 'msep'; m.appendChild(s); continue; }
      const el = document.createElement('div');
      el.className = 'mitem' + (it.off ? ' off' : '');
      el.appendChild(document.createTextNode(it.label));
      if(it.key){
        const k = document.createElement('span'); k.className = 'k';
        k.textContent = it.key; el.appendChild(k);
      }
      if(!it.off) el.onclick = () => { this._closeMenu(); it.run(); };
      m.appendChild(el);
    }
    // (this menu is not in the text-size zoom group, so client pixels are its pixels; the add menu is, see _menu)
    m.style.left = e.clientX+'px'; m.style.top = e.clientY+'px';
    document.body.appendChild(m); this._ctxEl = m;
    const r = m.getBoundingClientRect();   // keep it on screen near an edge
    m.style.left = Math.max(0, Math.min(e.clientX, innerWidth - r.width - 4))+'px';
    m.style.top = Math.max(0, Math.min(e.clientY, innerHeight - r.height - 4))+'px';
  }
  _find(){
    this._closeMenu();
    const ui = +getComputedStyle(document.documentElement).getPropertyValue('--ui') || 1;
    const r = this.canvas.getBoundingClientRect();
    const m = document.createElement('div'); m.id = 'addmenu';
    m.style.left = ((r.left + 10)/ui) + 'px';
    m.style.top = ((r.top + 10)/ui) + 'px';
    const inp = document.createElement('input');
    inp.id = 'addsearch'; inp.placeholder = 'find a node';
    m.appendChild(inp);
    const list = document.createElement('div'); m.appendChild(list);
    const entries = this.nodes.filter(n => NODE_DEFS[n.type]).map(n => ({ node:n,
      label:n.name || NODE_DEFS[n.type].title, type:NODE_DEFS[n.type].title,
      color:NODE_DEFS[n.type].color }));
    entries.sort((a, b) => a.label.localeCompare(b.label) || a.type.localeCompare(b.type));
    const LIMIT = 40;
    let first = null;
    const item = en => {
      const it = document.createElement('div'); it.className = 'mitem';
      const sw = document.createElement('span'); sw.style.background = en.color;
      it.appendChild(sw); it.appendChild(document.createTextNode(en.label));
      if(en.label !== en.type){            // the type too, when the name is not it
        const k = document.createElement('span');
        k.style.marginLeft = '12px'; k.style.color = '#666';
        k.textContent = en.type; it.appendChild(k);
      }
      it.onclick = () => { this._closeMenu(); this.revealNode(en.node); };
      return it;
    };
    const render = q => {
      list.innerHTML = ''; first = null;
      const hits = entries.filter(en => !q ||
        en.label.toLowerCase().includes(q) || en.type.toLowerCase().includes(q));
      for(const en of hits.slice(0, LIMIT)){ if(!first) first = en; list.appendChild(item(en)); }
      const note = document.createElement('div'); note.className = 'mhead';
      note.textContent = !hits.length ? 'no node of that name or type'
        : hits.length > LIMIT ? hits.length + ' found, first ' + LIMIT + ' shown'
        : hits.length + ' of ' + entries.length;
      list.appendChild(note);
    };
    inp.oninput = () => render(inp.value.trim().toLowerCase());
    inp.onkeydown = ke => {
      ke.stopPropagation();
      if(ke.key === 'Enter' && first){ this._closeMenu(); this.revealNode(first.node); }
      else if(ke.key === 'Escape') this._closeMenu();
    };
    render('');
    document.body.appendChild(m); this._menuEl = m;
    inp.focus();
  }
  // Center one node in the view, selected and flashing, at a scale it can be read at: what frameAll does for the whole graph, for one node.
  revealNode(n){
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    if(cw && ch){
      const s = Math.min(MAX_SCALE, Math.max(this.view.scale, 0.6));
      this.view.scale = s;
      this.view.x = cw/2 - (n.x + W/2)*s;
      this.view.y = ch/2 - (n.y + H/2)*s;
    }
    this.sel = n; this.selSet = new Set([n]);
    n._flash = performance.now();
    this.cb.onSelect(n);
    if(this.cb.onInspect) this.cb.onInspect(n);
    this.draw();
  }
  _menu(e, pending){                        // pending: wire endpoint to auto-connect
    this._closeMenu();
    const m = document.createElement('div'); m.id = 'addmenu';
    // the menu is zoomed by the text-size setting, and a zoomed fixed element's left and top are in its own scaled pixels: a pointer at 300 lands the menu at 390 under 1.3 unless the position is divided
    const ui = +getComputedStyle(document.documentElement).getPropertyValue('--ui') || 1;
    m.style.left = (e.clientX/ui)+'px'; m.style.top = (e.clientY/ui)+'px';
    const [gx,gy] = this.toGraph(e);
    const pickNode = type => {
      const anchor = this.sel;
      const n = this.addNode(type, gx-W/2, gy-H/2);
      if(pending){
        if(pending.kind==='out'){ n.x = gx-16; n.inputs[0] = { id:pending.node.id }; }
        else { n.x = gx-W/2; n.y = gy-H;
          pending.node.inputs[pending.port] = { id:n.id }; this.markDirty(pending.node); }
      }
      else this._insertAfter(anchor, n);     // new nodes join the pipe after the selection
      this.sel = n; this.selSet = new Set([n]); this.cb.onSelect(n);
      if(this.cb.onInspect) this.cb.onInspect(n);       // fresh nodes open their properties
      this._closeMenu(); this.cb.onChange();
    };
    const entries = [];                     // { label, color, cat, run }
    for(const [type, def] of Object.entries(NODE_DEFS)){
      if(pending && pending.kind==='out' && !def.inputs) continue;   // sources can't take a wire
      entries.push({ label:def.title, color:def.color, cat:def.cat || 'wiring',
        run:() => pickNode(type) });
    }
    if(!pending && this.scenarios && this.cb.onScenario)
      for(const sc of this.scenarios)
        entries.push({ label:sc.name, color:'#888', cat:'scenarios',
          run:() => { this._closeMenu(); this.cb.onScenario(sc); } });
    const item = en => {
      const it = document.createElement('div'); it.className = 'mitem';
      const sw = document.createElement('span'); sw.style.background = en.color;
      it.appendChild(sw); it.appendChild(document.createTextNode(en.label));
      it.onclick = en.run;
      return it;
    };
    const inp = document.createElement('input');
    inp.id = 'addsearch'; inp.placeholder = 'type to search';
    m.appendChild(inp);
    const list = document.createElement('div'); m.appendChild(list);
    let first = null;
    const renderSections = () => {
      list.innerHTML = ''; first = null;
      for(const [cat, label] of [['regions','REGIONS'],['cells','CELLS'],['arrange','ARRANGE'],['celltypes','CELL TYPES'],['wiring','WIRING'],['drive','DRIVE'],['readout','READOUT'],['run','RUN'],['routing','ROUTING'],['notes','NOTES'],['scenarios','SCENARIOS']]){
        const its = entries.filter(en => en.cat === cat);
        if(!its.length) continue;
        const sec = document.createElement('div'); sec.className = 'msection';
        const h = document.createElement('div'); h.className = 'mhead';
        h.textContent = label;
        const fly = document.createElement('div'); fly.className = 'mfly';
        its.forEach(en => fly.appendChild(item(en)));
        sec.appendChild(h); sec.appendChild(fly);
        const open = () => { for(const s of list.querySelectorAll('.msection.open'))
          if(s !== sec) s.classList.remove('open');
          sec.classList.add('open');
          // a flyout that would run off the bottom slides up until it fits (a long one scrolls), and one that would run off the right opens to the left; positions inside the menu are in its zoomed pixels, the rects are not
          fly.style.top = '-1px'; fly.classList.remove('leftward');
          const r = fly.getBoundingClientRect();
          const over = r.bottom - (innerHeight - 4);
          if(over > 0) fly.style.top = (-1 - Math.min(over, Math.max(0, r.top - 4))/ui) + 'px';
          if(r.right > innerWidth - 4 && r.left - r.width > 4) fly.classList.add('leftward'); };
        sec.onmouseenter = open; h.onclick = open;
        sec.onmouseleave = () => sec.classList.remove('open');
        list.appendChild(sec);
      }
    };
    const renderFiltered = q => {
      list.innerHTML = ''; first = null;
      for(const en of entries.filter(en => en.label.toLowerCase().includes(q))){
        if(!first) first = en;
        list.appendChild(item(en));
      }
    };
    inp.oninput = () => { const q = inp.value.trim().toLowerCase();
      q ? renderFiltered(q) : renderSections(); };
    inp.onkeydown = ke => {
      ke.stopPropagation();
      if(ke.key === 'Enter' && first) first.run();
      else if(ke.key === 'Escape') this._closeMenu();
    };
    renderSections();
    document.body.appendChild(m); this._menuEl = m;
    // near the bottom or the right edge the menu opens upward or leftward from the pointer rather than running off the screen
    { const r = m.getBoundingClientRect();
      let x = e.clientX, y = e.clientY;
      if(y + r.height > innerHeight - 4) y = Math.max(4, y - r.height);
      if(x + r.width > innerWidth - 4) x = Math.max(4, x - r.width);
      m.style.left = (x/ui) + 'px'; m.style.top = (y/ui) + 'px'; }
    inp.focus();
  }
  _closeMenu(){
    if(this._menuEl){ this._menuEl.remove(); this._menuEl = null; }
    if(this._ctxEl){ this._ctxEl.remove(); this._ctxEl = null; }
  }
  _marqueeSelect(d){
    const x0 = Math.min(d.x0,d.x1), x1 = Math.max(d.x0,d.x1);
    const y0 = Math.min(d.y0,d.y1), y1 = Math.max(d.y0,d.y1);
    this.selSet = new Set(d.base);
    for(const n of this.nodes){
      const [w, h] = this.dims(n);
      if(n.x < x1 && n.x+w > x0 && n.y < y1 && n.y+h > y0) this.selSet.add(n);
    }
  }
  // ---- draw ----
  // Bounding box of a group, and the header band that drags it.
  // Shared by the renderer and the hit test so the strip you can grab is exactly the strip that is drawn.
  groupBox(g){
    const byId = new Map(this.nodes.map(n => [n.id, n]));
    const ms = g.members.map(id => byId.get(id)).filter(Boolean);
    if(!ms.length) return null;
    const PAD = 14, TOP = 22;
    // The room added by dragging the backdrop's edges is padding around the members rather than a free rectangle, so a group can be given as much space as it needs and still cannot be shrunk off the nodes it names.
    const p = g.pad || { l:0, t:0, r:0, b:0 };
    const x0 = Math.min(...ms.map(n => n.x)) - PAD - (p.l || 0);
    // The label sits in the header band, so the box has to be wide enough to hold it.
    // A narrow group with a long name otherwise wrote its label out over whatever was to its right, and two adjacent backdrops read as one run-on line even though the boxes themselves were clear of each other.
    const wide = LABEL_ADV*(g.label || '').length + 16;
    return { ms, x0,
      y0: Math.min(...ms.map(n => n.y)) - PAD - TOP - (p.t || 0),
      x1: Math.max(x0 + wide,
        Math.max(...ms.map(n => n.x)) + W + PAD + (p.r || 0)),
      y1: Math.max(...ms.map(n => n.y)) + H + PAD + (p.b || 0),
      head: TOP + PAD };
  }
  // Which edges of a group's backdrop a point is on, for resizing.
  // The band is a fixed width on screen rather than in graph units, so the grab is the same size however far the view is zoomed out.
  // The header keeps priority over the top edge, since dragging a group is the commoner move.
  hitGroupEdge(gx, gy){
    const t = Math.min(8/this.view.scale, 14);
    let best = null, bestZ = -Infinity;
    for(const g of (this.groups || [])){
      const b = this.groupBox(g);
      if(!b) continue;
      if(gx < b.x0 - t || gx > b.x1 + t || gy < b.y0 - t || gy > b.y1 + t) continue;
      const l = Math.abs(gx - b.x0) <= t, r = Math.abs(gx - b.x1) <= t;
      const tp = Math.abs(gy - b.y0) <= t, bt = Math.abs(gy - b.y1) <= t;
      if(!l && !r && !tp && !bt) continue;
      // the header band belongs to the drag, not to the top edge
      if(tp && !l && !r && gy > b.y0 + 2) continue;
      const z = g.z || 0;
      if(z >= bestZ){ bestZ = z; best = { group:g, box:b, l, r, t:tp, b:bt }; }
    }
    return best;
  }
  // Only the header band grabs, as a backdrop does elsewhere: the body has to stay clickable so marquee selection still works inside a group, and so dragging a node out of one is not intercepted.
  // The one drawn in front wins, and among groups at the same depth the smallest does, so a group nested inside another stays reachable.
  hitGroup(gx, gy){
    let best = null, bestZ = -Infinity, bestArea = Infinity;
    for(const g of (this.groups || [])){
      const b = this.groupBox(g);
      if(!b) continue;
      if(gx < b.x0 || gx > b.x1 || gy < b.y0 || gy > b.y0 + b.head) continue;
      const z = g.z || 0, area = (b.x1-b.x0)*(b.y1-b.y0);
      if(z > bestZ || (z === bestZ && area < bestArea)){
        bestZ = z; bestArea = area; best = { group:g, box:b };
      }
    }
    return best;
  }

  // Groups overlap on purpose once they are made by hand, and which one reads as the container is a choice rather than a consequence of the order they happened to be created in, so each carries a depth and the higher one paints last.
  _sortedGroups(){
    return [...this.groups].map((g, i) => [g, i])
      .sort((a, b) => ((a[0].z || 0) - (b[0].z || 0)) || (a[1] - b[1]))
      .map(p => p[0]);
  }
  // Backdrops are drawn before the wires so they sit behind everything, and skipped entirely when a group's members have all been deleted.
  _groups(c, lw){
    if(!this.groups || !this.groups.length) return;
    for(const g of this._sortedGroups()){
      const b = this.groupBox(g);
      if(!b) continue;
      const { x0, y0, x1, y1 } = b;
      const hue = g.hue ?? 210;
      const held = this.drag && this.drag.mode === 'group' && this.drag.group === g;
      c.fillStyle = `hsl(${hue} 45% 7%)`;
      c.fillRect(x0, y0, x1-x0, y1-y0);
      c.lineWidth = held ? 2*lw : lw;
      c.strokeStyle = `hsl(${hue} 40% ${held ? 55 : 26}%)`;
      c.strokeRect(x0 + lw/2, y0 + lw/2, x1-x0-lw, y1-y0-lw);
      c.fillStyle = `hsl(${hue} 45% ${held ? 80 : 62}%)`;
      c.font = `600 12px ui-monospace, Menlo, monospace`;
      c.textAlign = 'left'; c.textBaseline = 'alphabetic';
      c.fillText(g.label, x0 + 8, y0 + 15);
    }
  }

  draw(){
    if(this.errorNode === undefined) this.errorNode = null;
    if(this.errorNode !== null && !this._pulseTimer) this._pulseTimer = setInterval(() => this.draw(), 60);
    if(this.errorNode === null && this._pulseTimer){ clearInterval(this._pulseTimer); this._pulseTimer = null; }
    if(this.canvas.dataset.debug === '1')
      this.canvas.dataset.state = JSON.stringify(this.toJSON());
    const cv = this.canvas, dpr = devicePixelRatio;
    const w = cv.clientWidth, h = cv.clientHeight;
    if(cv.width!==w*dpr || cv.height!==h*dpr){ cv.width=w*dpr; cv.height=h*dpr; }
    // a view that is not a number draws nothing and no gesture can reach it
    if(!Number.isFinite(this.view.x) || !Number.isFinite(this.view.y) || !(this.view.scale > 0)){
      this.view = { x:0, y:0, scale:1 }; this.frameAll();
    }
    const c = this.ctx;
    c.setTransform(dpr,0,0,dpr,0,0);
    c.fillStyle = '#000'; c.fillRect(0,0,w,h);
    c.setTransform(dpr*this.view.scale, 0, 0, dpr*this.view.scale, dpr*this.view.x, dpr*this.view.y);
    const lw = 1/this.view.scale;
    this._groups(c, lw);
    for(const wr of this.wireList()){
      const hot = this.insertHit && this.insertHit.src===wr.src &&
                  this.insertHit.dst===wr.dst && this.insertHit.port===wr.port;
      c.lineWidth = hot ? 2*lw : lw;
      c.strokeStyle = hot ? '#ffffff' : '#5c5c5c';
      this._wire(c, ...this.outPort(wr.src), ...this.inPort(wr.dst, wr.port),
        this.isSide(wr.dst, wr.port));
    }
    if(this.drag && this.drag.mode==='wire'){
      c.lineWidth = lw; c.strokeStyle = '#aaa';
      const d = this.drag;
      if(d.fixed.kind==='out') this._wire(c, ...this.outPort(d.fixed.node), d.gx, d.gy, false);
      else this._wire(c, d.gx, d.gy, ...this.inPort(d.fixed.node, d.fixed.port),
        this.isSide(d.fixed.node, d.fixed.port));
    }
    c.font = '10px ui-monospace, Menlo, monospace';
    const slotOf = {};
    for(const d in this.viewSlots) slotOf[this.viewSlots[d]] = d;
    for(const n of this.nodes){
      if(n.type === 'note'){ this._drawNote(c, n, lw); continue; }
      if(n.type === 'pin'){ this._drawDot(c, n, lw); continue; }
      const def = NODE_DEFS[n.type];
      const off = n.on === false;
      c.fillStyle = '#000'; c.fillRect(n.x, n.y, W, H);
      c.lineWidth = lw;
      const flash = n._flash && performance.now() - n._flash < 300;
      const seld = n===this.sel || this.selSet.has(n) || flash;
      c.strokeStyle = seld ? '#ffffff' : off ? '#555' : def.color;
      c.strokeRect(n.x+0.5*lw, n.y+0.5*lw, W-lw, H-lw);
      // the node whose computation failed pulses red until a computation succeeds; the pulse timer below redraws while one is marked
      if(this.errorNode === n.id){
        const ph = (Math.sin(performance.now()/250) + 1)/2;
        c.strokeStyle = `rgba(255, 60, 60, ${(0.45 + 0.55*ph).toFixed(2)})`;
        c.lineWidth = 3*lw;
        c.strokeRect(n.x - lw, n.y - lw, W + 2*lw, H + 2*lw);
        c.lineWidth = lw;
      }
      c.fillStyle = off ? '#555' : def.color; c.fillRect(n.x, n.y, 3, H);
      c.fillStyle = seld ? '#fff' : off ? '#666' : '#c8c8c8';
      // A named node reads as "type name": the type still identifies what it does, the name says which one it is.
      // The auto name equals the type token on a fresh node, which would draw "scatter scatter"; while a name still reads as its type it is not shown, so a name appears the moment it says something the title does not.
      const showName = n.name && tagToken(n.name) !== tagToken(def.title);
      if(showName){
        c.fillText(def.title, n.x+10, n.y+H/2+3.5);
        const tw = c.measureText(def.title).width;
        c.fillStyle = seld ? '#fff' : off ? '#555' : '#8a8a8a';
        c.fillText(n.name, n.x+14+tw, n.y+H/2+3.5);
      } else c.fillText(def.title, n.x+10, n.y+H/2+3.5);
      if(off){ c.strokeStyle = '#555';
        c.beginPath(); c.moveTo(n.x+4, n.y+H-4); c.lineTo(n.x+W-4, n.y+4); c.stroke(); }
      if(slotOf[n.id]){                      // viewer binding badge, boxed when viewed
        const activeV = n.id === this.activeView;
        c.fillStyle = activeV ? '#fff' : '#777';
        c.fillText(slotOf[n.id], n.x+W-14, n.y+H/2+3.5);
        if(activeV){ c.strokeStyle = '#fff'; c.lineWidth = lw;
          c.strokeRect(n.x+W-18, n.y+H/2-7, 12, 14); }
      }
      this._syncPorts(n);
      for(let i=0;i<n.inputs.length;i++)
        this._port(c, ...this.inPort(n,i), this._isHover(n,i,'in'), lw, this.isSide(n,i));
      // a side port with a name says what goes on it, beside the diamond
      if(def.sideLabels){
        c.font = '9px ui-monospace, Menlo, monospace';
        c.fillStyle = seld ? '#bbb' : '#666';
        for(let i=0;i<n.inputs.length;i++){
          const lab = this.isSide(n,i) && def.sideLabels[i - def.side];
          if(!lab) continue;
          const [px, py] = this.inPort(n,i);
          c.fillText(lab, px + 8, py + 3);
        }
        c.font = `600 12px ui-monospace, Menlo, monospace`;
      }
      this._port(c, ...this.outPort(n), this._isHover(n,null,'out'), lw, false);
    }
    if(this.drag && this.drag.mode==='marquee'){
      const d = this.drag;
      c.lineWidth = lw; c.strokeStyle = '#888'; c.setLineDash([4*lw, 3*lw]);
      c.strokeRect(Math.min(d.x0,d.x1), Math.min(d.y0,d.y1), Math.abs(d.x1-d.x0), Math.abs(d.y1-d.y0));
      c.setLineDash([]);
    }
  }
  // A reroute handle: a filled circle with the wire meeting its top and bottom.
  // Any name sits beside it, so a junction can be labeled without the node itself taking a node-sized bite out of the layout.
  _drawDot(c, n, lw){
    const r = DOT/2, cx = n.x + r, cy = n.y + r;
    const off = n.on === false;
    const flash = n._flash && performance.now() - n._flash < 300;
    const seld = n === this.sel || this.selSet.has(n) || flash;
    // lw is a screen pixel in graph units, so far enough out it passes the radius: a negative radius throws
    c.beginPath(); c.arc(cx, cy, Math.max(0.5, r - lw), 0, Math.PI*2);
    c.fillStyle = '#000'; c.fill();
    c.lineWidth = lw;
    c.strokeStyle = seld ? '#ffffff' : off ? '#555' : NODE_DEFS.pin.color;
    c.stroke();
    c.beginPath(); c.arc(cx, cy, Math.max(1.5, r*0.32), 0, Math.PI*2);
    c.fillStyle = seld ? '#ffffff' : off ? '#555' : NODE_DEFS.pin.color;
    c.fill();
    if(n.name && tagToken(n.name) !== tagToken((NODE_DEFS[n.type] && NODE_DEFS[n.type].title) || n.type)){
      c.fillStyle = seld ? '#fff' : '#8a8a8a';
      c.fillText(n.name, n.x + DOT + 6, cy + 3.5);
    }
  }
  _drawNote(c, n, lw){
    const w = Math.max(100, n.params.width|0 || 220);
    const key = n.params.text + '|' + w;
    if(n._noteKey !== key){
      const lines = [];
      for(const para of String(n.params.text || '').split('\n')){
        let line = '';
        for(const word of para.split(/\s+/)){
          const t = line ? line + ' ' + word : word;
          if(c.measureText(t).width > w - 16 && line){ lines.push(line); line = word; }
          else line = t;
        }
        lines.push(line);
      }
      n._noteLines = lines; n._noteKey = key;
      n._noteW = w; n._noteH = Math.max(26, lines.length*13 + 12);
    }
    const seld = n === this.sel || this.selSet.has(n);
    c.fillStyle = '#000'; c.fillRect(n.x, n.y, n._noteW, n._noteH);
    c.lineWidth = lw;
    c.strokeStyle = seld ? '#ffffff' : '#575030';
    c.strokeRect(n.x+0.5*lw, n.y+0.5*lw, n._noteW-lw, n._noteH-lw);
    c.fillStyle = seld ? '#fff' : '#b0a878';
    n._noteLines.forEach((ln, i) => c.fillText(ln, n.x+8, n.y+16+i*13));
  }
  _isHover(n, i, kind){ const hv = this.hover;
    return hv && hv.node===n && hv.kind===kind && (kind==='out' || hv.port===i); }
  _wire(c, x1,y1, x2,y2, side){
    c.beginPath(); c.moveTo(x1,y1);
    if(side){ c.setLineDash([4,3]); c.bezierCurveTo(x1, y1+34, x2+34, y2, x2, y2); }
    else c.bezierCurveTo(x1, y1+30, x2, y2-30, x2, y2);
    c.stroke(); c.setLineDash([]);
  }
  _port(c, x, y, hot, lw, diamond){
    c.fillStyle = '#000';
    c.strokeStyle = hot ? '#fff' : '#777'; c.lineWidth = lw;
    if(diamond){
      c.beginPath(); c.moveTo(x, y-5); c.lineTo(x+5, y); c.lineTo(x, y+5); c.lineTo(x-5, y);
      c.closePath(); c.fill(); c.stroke();
    } else {
      c.fillRect(x-PORT/2, y-PORT/2, PORT, PORT);
      c.strokeRect(x-PORT/2, y-PORT/2, PORT, PORT);
    }
  }
}

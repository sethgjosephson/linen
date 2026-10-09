import * as THREE from 'three';
import { matches as keyMatches } from './keymap.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { hueRgb, hueCss, NEURON_TYPES } from './nodes.js';
import { sliceBounds, sliceContains, slicePlanes } from './slice.js';

// per-shape gizmo behavior: wireframe mesh + param sync/writeback
const _center = (mesh, p) => { p.center = [mesh.position.x, mesh.position.y, mesh.position.z]; };
const _syncRot = (mesh, p) => mesh.rotation.set(...(p.rotate || [0,0,0]).map(d => d*Math.PI/180));
const _applyRot = (mesh, p) => { p.rotate = [mesh.rotation.x, mesh.rotation.y, mesh.rotation.z].map(r => r*180/Math.PI); };
const _avg = s => Math.max(0.1, (Math.abs(s.x)+Math.abs(s.y)+Math.abs(s.z))/3);
const GEO_EDIT = {
  sphere: { modes:['translate','scale'],
    make: () => new THREE.SphereGeometry(1, 24, 16),
    sync(mesh, p){ mesh.position.set(...p.center); mesh.scale.setScalar(p.radius); },
    apply(mesh, p){ _center(mesh, p);
      p.radius = _avg(mesh.scale); mesh.scale.setScalar(p.radius); } },
  box: { modes:['translate','rotate','scale'],
    make: () => new THREE.BoxGeometry(1, 1, 1),
    sync(mesh, p){ mesh.position.set(...p.center); mesh.scale.set(...p.size); _syncRot(mesh, p); },
    apply(mesh, p){ _center(mesh, p);
      p.size = [Math.abs(mesh.scale.x), Math.abs(mesh.scale.y), Math.abs(mesh.scale.z)];
      _applyRot(mesh, p); } },
  ellipsoid: { modes:['translate','rotate','scale'],
    make: () => new THREE.SphereGeometry(1, 24, 16),
    sync(mesh, p){ mesh.position.set(...p.center); mesh.scale.set(...p.radii); _syncRot(mesh, p); },
    apply(mesh, p){ _center(mesh, p);
      p.radii = [Math.abs(mesh.scale.x), Math.abs(mesh.scale.y), Math.abs(mesh.scale.z)];
      _applyRot(mesh, p); } },
  cylinder: { modes:['translate','rotate','scale'],
    make: () => new THREE.CylinderGeometry(1, 1, 1, 24),
    sync(mesh, p){ mesh.position.set(...p.center);
      mesh.scale.set(p.radius, p.height, p.radius); _syncRot(mesh, p); },
    apply(mesh, p){ _center(mesh, p);
      p.radius = Math.max(0.1, (Math.abs(mesh.scale.x)+Math.abs(mesh.scale.z))/2);
      p.height = Math.max(0.1, Math.abs(mesh.scale.y));
      mesh.scale.set(p.radius, p.height, p.radius); _applyRot(mesh, p); } },
  torus: { modes:['translate','rotate','scale'],
    make: p => { const g = new THREE.TorusGeometry(1, p.thickness/Math.max(0.1, p.radius), 10, 40);
      g.rotateX(Math.PI/2); return g; },   // ring in XZ, axis Y (thickness edited in props)
    sync(mesh, p){ mesh.position.set(...p.center); mesh.scale.setScalar(p.radius); _syncRot(mesh, p); },
    apply(mesh, p){ _center(mesh, p);
      p.radius = _avg(mesh.scale); mesh.scale.setScalar(p.radius); _applyRot(mesh, p); } },
  // keyed by node type, so a renamed node type needs its key renamed here or its gizmo never opens
  noisefield: { modes:['translate','scale'],    // bounds box of the density field
    make: () => new THREE.BoxGeometry(1, 1, 1),
    sync(mesh, p){ mesh.position.set(...p.center); mesh.scale.set(...p.size); },
    apply(mesh, p){ _center(mesh, p);
      p.size = [Math.abs(mesh.scale.x), Math.abs(mesh.scale.y), Math.abs(mesh.scale.z)]; } },
  move: { modes:['translate','rotate','scale'],   // manipulates the node's T/R/S params
    make: () => new THREE.BoxGeometry(220, 220, 220),
    sync(mesh, p){
      mesh.position.set(p.pivot[0]+p.translate[0], p.pivot[1]+p.translate[1], p.pivot[2]+p.translate[2]);
      mesh.scale.set(...p.scale); _syncRot(mesh, p);
    },
    apply(mesh, p){
      p.translate = [mesh.position.x-p.pivot[0], mesh.position.y-p.pivot[1], mesh.position.z-p.pivot[2]];
      p.scale = [mesh.scale.x, mesh.scale.y, mesh.scale.z];
      _applyRot(mesh, p);
    } },
};

export class Viewer {
  constructor(container, hud){
    this.hud = hud;
    // Pathways get their own readout line: the neuron HUD is about whatever was last clicked, and a set of bundles outlives that.
    this.paths = new Map();
    this.popBoxes = []; this.boxesOn = false; this.labels = [];
    this.hoverTag = ''; this.focusTag = ''; this.onHoverPopulation = null;
    this.pathHud = document.getElementById('pathhud') || hud;
    this.pathSample = new Map();
    this.container = container;
    this._makeRenderer();
    // The population names live in a layer over the canvas, inside the same element, so a label's position is simply the projected point: a layer anywhere else has to be corrected by the offset between the two, and that offset depends on where the pane's own chrome ends up.
    // The container's own position comes from the stylesheet, which places it absolutely inside the pane; set inline here, the element collapses to the canvas's default height and the viewport becomes a 150 pixel strip.
    this.labelEl = document.createElement('div');
    this.labelEl.id = 'poplabels';
    container.appendChild(this.labelEl);
    // A lost WebGL context is a normal event on this machine, not a fault: the CUDA engine works a hundred and fifty million synapses on the same card the browser draws with, and Windows takes the browser's context away under that load.
    // The run has not died: it is in another process, and on a host-pumped run it is not even in this machine's browser, so the page says so and takes the restore if the platform offers one.
    this._bindCanvas();
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);
    this.camera = new THREE.PerspectiveCamera(50, 1, 10, 300000);   // world units are µm
    this.camera.position.set(2300, 2700, 3300);
    this._bindControls();
    this.net = null; this.points = null; this.glow = null;
    this.base = null; this.selected = -1; this.selLines = null;
    this.selLinesIn = null; this.onSelect = null;
    this.onPickSource = null; this.onPickPopulation = null; this.gizmo = null; this.geoMesh = null;
    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points.threshold = 35;
    new ResizeObserver(() => this._resize(container)).observe(container);
    this._resize(container);
    this._bindPointer();
    window.addEventListener('keydown', e => {
      if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) || !this.gizmo) return;
      if(keyMatches(e, 'gizmoMove')) this.gizmo.setMode('translate');
      else if(keyMatches(e, 'gizmoRotate') && this.geoEdit.modes.includes('rotate')) this.gizmo.setMode('rotate');
      else if(keyMatches(e, 'gizmoScale') && this.geoEdit.modes.includes('scale')) this.gizmo.setMode('scale');
      else if(keyMatches(e, 'closeGizmo')) this.closeGizmo();
    });
  }
  // The renderer and everything tied to its canvas, in one place, so the canvas can be replaced: a context the driver has taken away during a CUDA run is not always given back, and a new renderer re-uploads the scene on its first frame (three.js caches per renderer).
  _makeRenderer(){
    this.renderer = new THREE.WebGLRenderer({ antialias:true });
    // the slicer box cuts the connection lines with clipping planes on their materials (setSlice keeps the planes; off, they are far away)
    this.renderer.localClippingEnabled = true;
    this.renderer.setPixelRatio(devicePixelRatio);
    // before the label layer, which stays on top
    this.container.insertBefore(this.renderer.domElement, this.container.firstChild);
  }
  _bindCanvas(){
    const gl = this.renderer.domElement;
    gl.addEventListener('webglcontextlost', e => {
      e.preventDefault();                  // without this there is no restore
      this._ctxLost = true;
      if(this.onContextLost) this.onContextLost();
    }, false);
    gl.addEventListener('webglcontextrestored', () => {
      this._ctxLost = false;
      if(this.onContextRestored) this.onContextRestored();
    }, false);
  }
  _bindControls(){
    if(this.controls) this.controls.dispose();
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
  }
  _bindPointer(){
    this.renderer.domElement.addEventListener('pointerdown', e => this._dn = [e.clientX, e.clientY]);
    this.renderer.domElement.addEventListener('pointerup', e => {
      if(this._dn && Math.hypot(e.clientX-this._dn[0], e.clientY-this._dn[1]) < 4) this._pick(e);
    });
    this.renderer.domElement.addEventListener('dblclick', e => {
      const idx = this._raycast(e);
      if(idx >= 0 && this.net && this.net.src && this.onPickSource) this.onPickSource(this.net.src[idx]);
      else if(idx < 0) this.closeGizmo();
    });
    // Past a handful of populations their names cannot all be on screen at once (see _placeLabels), so pointing at a box is how you read one.
    this.renderer.domElement.addEventListener('pointermove', e => this._hoverRegion(e));
    this.renderer.domElement.addEventListener('pointerleave', () => this._hoverRegion(null));
  }
  // A fresh canvas and renderer for the same scene and camera.
  // Used when a lost context was never restored; the gizmo is closed because its controls were bound to the previous canvas.
  rebuildRenderer(){
    this.closeGizmo();
    const old = this.renderer;
    try { old.dispose(); } catch(e){}
    if(old.domElement.parentNode) old.domElement.parentNode.removeChild(old.domElement);
    this._makeRenderer();
    this._bindCanvas();
    this._bindControls();
    this._bindPointer();
    this._ctxLost = false;
    this._resize(this.container);
    this.renderer.render(this.scene, this.camera);
  }
  _resize(el){
    const w = el.clientWidth || 1, h = el.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.camera.aspect = w/h; this.camera.updateProjectionMatrix();
  }
  _clearDisplay(){
    if(this.points){ this.scene.remove(this.points);
      this.points.geometry.dispose(); this.points.material.dispose(); this.points = null; }
    if(this.previewMesh){ this.scene.remove(this.previewMesh);
      this.previewMesh.geometry.dispose(); this.previewMesh.material.dispose(); this.previewMesh = null; }
  }
  _buildCloud(pos, ntype, n){
    this._ntype = ntype;
    this.typeBase = new Float32Array(n*3);
    this.sign = new Int8Array(n);
    for(let i=0;i<n;i++){
      const c = NEURON_TYPES[ntype[i]].color;
      this.typeBase[i*3]=c[0]; this.typeBase[i*3+1]=c[1]; this.typeBase[i*3+2]=c[2];
      this.sign[i] = NEURON_TYPES[ntype[i]].sign > 0 ? 1 : -1;
    }
    // The cloud is colored by cell type by default, which is what the dynamics are about.
    // A scene is also an anatomy, and by type every excitatory cell in every area is the same color: population coloring hands that back, using the hue each tag already wears in the pair table, the pathway lines and the marked sets.
    this.base = (this.popBase && this.popBase.length === n*3 && this.colorBy === 'population')
      ? this.popBase.slice() : this.typeBase.slice();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(this.base.slice(), 3));
    // mv = display potential per neuron (spikes shown as +40); points below the vmFloor uniform are clipped in the vertex shader.
    // Default 40 keeps everything visible until real potentials arrive.
    g.setAttribute('mv', new THREE.BufferAttribute(new Float32Array(n).fill(40), 1));
    if(!this.vmFloorU) this.vmFloorU = { value:-100 };
    this._sliceUniforms();
    this.cloudPos = pos;                     // what a first slice box is sized from
    const m = new THREE.PointsMaterial({ size:3.5, sizeAttenuation:false, vertexColors:true });
    m.onBeforeCompile = sh => {
      sh.uniforms.vmFloor = this.vmFloorU;
      // the slicer box (slice.js): a cell outside it is sent off screen the way the potential filter sends one below the floor
      sh.uniforms.sliceOn = this.sliceU.on;
      sh.uniforms.sliceMin = this.sliceU.min;
      sh.uniforms.sliceMax = this.sliceU.max;
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>',
          '#include <common>\nattribute float mv;\nuniform float vmFloor;\nuniform float sliceOn;\nuniform vec3 sliceMin;\nuniform vec3 sliceMax;')
        .replace('#include <project_vertex>',
          '#include <project_vertex>\nif(mv < vmFloor){ gl_Position = vec4(0.0, 0.0, 2.0, 1.0); }' +
          '\nif(sliceOn > 0.5 && (position.x < sliceMin.x || position.x > sliceMax.x || position.y < sliceMin.y || position.y > sliceMax.y || position.z < sliceMin.z || position.z > sliceMax.z)){ gl_Position = vec4(0.0, 0.0, 2.0, 1.0); }');
    };
    this.points = new THREE.Points(g, m);
    this.scene.add(this.points);
  }
  _sliceUniforms(){
    if(!this.sliceU) this.sliceU = { on:{ value:0 }, min:{ value:new THREE.Vector3() }, max:{ value:new THREE.Vector3() } };
    return this.sliceU;
  }
  // The six faces of the box as clipping planes, one set shared by every connection line's material, so a line from a shown cell to a hidden one is cut where it leaves the box rather than drawn or dropped whole.
  _slicePlanes(){
    if(!this.slicePlanesObj) this.slicePlanesObj = slicePlanes(null).map(p => new THREE.Plane(new THREE.Vector3(p[0], p[1], p[2]), p[3]));
    return this.slicePlanesObj;
  }
  // The slicer box: which cells are drawn, nothing more (slice.js).
  // The uniforms are shared by every cloud this viewer builds, so rewiring keeps the cut; the outline is a box like the population boxes, in white.
  setSlice(s){
    const U = this._sliceUniforms();
    this.slice = s && s.on ? s : null;
    const planes = this._slicePlanes();
    slicePlanes(this.slice).forEach((p, k) => { planes[k].normal.set(p[0], p[1], p[2]); planes[k].constant = p[3]; });
    if(this.sliceBox){ this.scene.remove(this.sliceBox); this.sliceBox.geometry.dispose(); this.sliceBox.material.dispose(); this.sliceBox = null; }
    if(!this.slice){ U.on.value = 0; return; }
    const { min, max } = sliceBounds(s);
    U.on.value = 1; U.min.value.set(...min); U.max.value.set(...max);
    const g = new THREE.BoxGeometry(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    this.sliceBox = new THREE.LineSegments(new THREE.EdgesGeometry(g),
      new THREE.LineBasicMaterial({ color:0xffffff, transparent:true, opacity:0.45 }));
    g.dispose();
    this.sliceBox.position.set((min[0] + max[0])/2, (min[1] + max[1])/2, (min[2] + max[2])/2);
    this.scene.add(this.sliceBox);
  }
  // Handles on the slicer box: drag to move, the scale mode to resize.
  // The same gizmo the shapes use, so the mode keys and escape work the same.
  editSlice(s, onChange){
    this.closeGizmo();
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ wireframe:true, color:0xffffff, transparent:true, opacity:0.12 }));
    mesh.position.set(...s.center); mesh.scale.set(...s.size.map(x => Math.max(1, Math.abs(x))));
    this.scene.add(mesh);
    this.geoMesh = mesh; this.geoNode = null; this.geoEdit = { modes:['translate', 'scale'] };
    this.sliceEditing = true;
    const tc = new TransformControls(this.camera, this.renderer.domElement);
    tc.addEventListener('dragging-changed', ev => { this.controls.enabled = !ev.value; });
    tc.addEventListener('objectChange', () => {
      s.center = [mesh.position.x, mesh.position.y, mesh.position.z];
      s.size = [mesh.scale.x, mesh.scale.y, mesh.scale.z].map(x => Math.max(1, Math.abs(x)));
      this.setSlice(s);
      if(onChange) onChange(s);
    });
    tc.attach(mesh);
    this.scene.add(tc);
    this.gizmo = tc;
  }
  // A box around each population, in that population's color, with its name above it. boxes: [{ tag, hue, min, max, count, src }], built by main.js from the scatter tags because tags belong to the graph; src is the population's source node id, for the click on its label.
  setPopulationBoxes(boxes){
    this.popBoxes = boxes || [];
    this._rebuildBoxes();
  }
  // The region the pointer is over, or a row in the REGIONS panel.
  // It reads and the rest recede; nothing about the network changes.
  hoverPopulation(tag){
    tag = tag || '';
    if(tag === this.hoverTag) return;
    this.hoverTag = tag;
    this._emphasise();
    if(this.onHoverPopulation) this.onHoverPopulation(tag);
  }
  // The region the graph selection made, so selecting its scatter says which box in the cloud it is.
  // Kept across a rebuild.
  focusPopulation(tag){
    tag = tag || '';
    if(tag === this.focusTag) return;
    this.focusTag = tag;
    this._emphasise();
  }
  focusPopulationBySrc(src){
    const b = (this.popBoxes || []).find(x => x.src === src);
    this.focusPopulation(b ? b.tag : '');
  }
  // Whichever region is live is drawn at full strength and the others drop back.
  // Two dozen boxes are otherwise a mesh of lines with nothing picked out of it.
  _emphasise(){
    const live = this.hoverTag || this.focusTag;
    for(const l of this.labels){
      const on = !!l.tag && (l.tag === this.hoverTag || l.tag === this.focusTag);
      l.on = on;
      if(l.edges) l.edges.material.opacity = on ? 0.95 : live ? 0.12 : 0.35;
      l.el.classList.toggle('on', on);
    }
  }
  showPopulationBoxes(on){
    this.boxesOn = !!on;
    this._rebuildBoxes();
  }
  _rebuildBoxes(){
    if(this.boxGroup){
      this.scene.remove(this.boxGroup);
      this.boxGroup.traverse(o => { if(o.geometry) o.geometry.dispose(); if(o.material) o.material.dispose(); });
      this.boxGroup = null;
    }
    if(this.labelEl) this.labelEl.textContent = '';
    this.labels = [];
    if(!this.boxesOn || !this.popBoxes || !this.popBoxes.length) return;
    this.boxGroup = new THREE.Group();
    for(const b of this.popBoxes){
      const sx = Math.max(1, b.max[0] - b.min[0]), sy = Math.max(1, b.max[1] - b.min[1]),
            sz = Math.max(1, b.max[2] - b.min[2]);
      const g = new THREE.BoxGeometry(sx, sy, sz);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g),
        new THREE.LineBasicMaterial({ color:new THREE.Color(...hueRgb(b.hue)),
          transparent:true, opacity:0.35 }));
      g.dispose();
      edges.position.set(b.min[0] + sx/2, b.min[1] + sy/2, b.min[2] + sz/2);
      this.boxGroup.add(edges);
      // the label rides above the top face, in the document rather than in the scene, so it stays crisp and reads at any zoom
      if(this.labelEl){
        const el = document.createElement('div');
        el.className = 'poplabel';
        el.textContent = b.tag + (b.count ? ' · ' + b.count.toLocaleString() : '');
        // a solid block in the population's color with the text inside, so the name reads over a dense cloud of the same color
        el.style.background = hueCss(b.hue);
        el.title = 'select the ' + b.tag + ' node';
        el.onclick = e => { e.stopPropagation(); if(this.onPickPopulation) this.onPickPopulation(b.src, b.tag); };
        this.labelEl.appendChild(el);
        // the box itself travels with the label: the ray hits it for hover, and its lines are what brightens when the region is live
        this.labels.push({ el, at:new THREE.Vector3(b.min[0] + sx/2, b.max[1], b.min[2] + sz/2),
          tag:b.tag, src:b.src, edges,
          box3:new THREE.Box3(new THREE.Vector3(...b.min), new THREE.Vector3(...b.max)) });
      }
    }
    this.scene.add(this.boxGroup);
    this._emphasise();
  }
  // Which population volume the pointer is in, by ray against the boxes.
  // The nearest hit wins, which is the box you take yourself to be pointing at when territories sit inside one another.
  _hoverRegion(e){
    if(!this.boxesOn || !this.labels.length){ this.hoverPopulation(''); return; }
    let tag = '';
    if(e){
      const r = this.renderer.domElement.getBoundingClientRect();
      const m = new THREE.Vector2(((e.clientX - r.left)/r.width)*2 - 1,
                                  -((e.clientY - r.top)/r.height)*2 + 1);
      this.raycaster.setFromCamera(m, this.camera);
      const hit = new THREE.Vector3();
      let best = Infinity;
      for(const l of this.labels){
        if(!l.box3) continue;
        const inside = l.box3.containsPoint(this.camera.position);
        const p = inside ? null : this.raycaster.ray.intersectBox(l.box3, hit);
        if(!inside && !p) continue;
        const d = inside ? 0 : p.distanceTo(this.camera.position);
        if(d < best){ best = d; tag = l.tag; }
      }
    }
    this.hoverPopulation(tag);
  }
  // Screen positions for the labels, once per frame: a projected point is only right for the camera it was projected with.
  _placeLabels(){
    if(!this.labels || !this.labels.length) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    // the labels live in their own layer, which need not sit exactly over the canvas: the offset between the two is measured rather than assumed, since the viewer bar above them changes height when it wraps
    const cr = this.labelEl.getBoundingClientRect();
    const dx = r.left - cr.left, dy = r.top - cr.top;
    const shown = [];
    for(const l of this.labels){
      const v2 = l.at.clone().project(this.camera);
      const behind = v2.z > 1;
      l.el.style.display = behind ? 'none' : 'block';
      if(behind) continue;
      l.x = (v2.x + 1)/2*r.width + dx;
      l.y = (1 - v2.y)/2*r.height + dy;
      l.z = v2.z;
      l.dist = l.at.distanceTo(this.camera.position);
      shown.push(l);
    }
    // The live region first, since that name is drawn whatever else has to go, then nearest first: a label that moves out of the way is the one further off.
    // Two populations sharing a layer project to the same place and their names would otherwise sit on top of each other.
    shown.sort((a, b) => (b.on ? 1 : 0) - (a.on ? 1 : 0) || a.dist - b.dist);
    let dMin = Infinity, dMax = 0;
    for(const l of shown){ if(l.dist < dMin) dMin = l.dist; if(l.dist > dMax) dMax = l.dist; }
    const dRange = dMax - dMin;
    const placed = [];
    let named = 0;
    for(const l of shown){
      // the block's own size, so the stack step follows the text size and the padding; a fixed 13 px step would overlap the blocks
      const w = l.el.offsetWidth || 70, h = l.el.offsetHeight || 13;
      // One step out of the way, and no further.
      // Stacking every clash builds a wall of names on a column of two dozen populations, every name pushed under the last until they cover the tissue they are naming.
      // A name with nowhere to go is dropped instead.
      // Nothing is lost: the REGIONS panel lists every population, pointing at a box names it, and selecting its node names it.
      const clashAt = yy => placed.find(q => Math.abs(q.x - l.x) < (q.w + w)/2 && Math.abs(q.y - yy) < (q.h + h)/2 + 2);
      let y = l.y;
      const first = clashAt(y);
      if(first){
        const down = first.y + (first.h + h)/2 + 2;
        if(!clashAt(down) && down < dy + r.height - h) y = down;
        else if(!l.on){ l.el.style.display = 'none'; continue; }
      }
      // the block floats above the point, so a name near the top of the canvas is cut in half; the live one is held inside instead of lost
      if(l.on && y < dy + 1.7*h) y = dy + 1.7*h;
      // off the canvas, or pushed off it: not shown.
      // The layer is clipped as well, since a label that reached the node pane took its clicks.
      if(l.x < dx - w || l.x > dx + r.width + w || y < dy + 1.5*h || y > dy + r.height){ l.el.style.display = 'none'; continue; }
      placed.push({ x:l.x, y, w, h });
      named++;
      l.el.style.left = l.x + 'px';
      l.el.style.top = y + 'px';
      // further boxes recede rather than competing with the near ones, relative to the other labels: the projected depth is nearly 1 for everything in a scene this small against a 300 mm far plane, so it faded them all equally
      const f = dRange > 0 ? (l.dist - dMin)/dRange : 0;
      l.el.style.opacity = l.on ? '1' : (1 - 0.45*f).toFixed(2);
    }
    this.regionsNamed = named;              // of this.labels.length, for the panel
  }

  // The type colors again from the type table, for a computation that changed a row's color without changing the network (the cell type node's hue): the cloud is kept, only its colors are rewritten.
  refreshTypeColors(){
    const ntype = this._ntype; if(!ntype || !this.typeBase) return;
    const n = ntype.length; let changed = false;
    for(let i = 0; i < n; i++){
      const c = NEURON_TYPES[ntype[i]].color; if(!c) continue;
      if(this.typeBase[i*3] !== c[0] || this.typeBase[i*3+1] !== c[1] || this.typeBase[i*3+2] !== c[2]){
        this.typeBase[i*3] = c[0]; this.typeBase[i*3+1] = c[1]; this.typeBase[i*3+2] = c[2]; changed = true;
      }
    }
    if(changed) this.setColorBy(this.colorBy);
  }
  // rgb per neuron for the population coloring; main.js builds it from the scatter tags because tags and their hues belong to the graph
  setPopulationColors(rgb){
    this.popBase = rgb || null;
    if(this.colorBy === 'population') this.setColorBy('population');
  }
  setColorBy(mode){
    this.colorBy = mode === 'population' ? 'population' : 'type';
    // the population colors belong to the graph and arrive after the cloud; between a new network and their arrival they are the old network's, sized for it, and writing them into the new cloud throws
    const popOk = this.popBase && this.typeBase && this.popBase.length === this.typeBase.length;
    const want = this.colorBy === 'population' && popOk ? this.popBase : this.typeBase;
    if(!want || !this.points || want.length !== this.points.geometry.attributes.color.array.length) return;
    this.base = want.slice();
    const col = this.points.geometry.attributes.color;
    col.array.set(this.base);
    col.needsUpdate = true;
  }
  setPotentialFloor(v){
    if(!this.vmFloorU) this.vmFloorU = { value:-100 };
    this.vmFloorU.value = v;
  }
  // Which spikes are drawn: 0 all, 1 excitatory only, -1 inhibitory only.
  // Display only; a hidden spike still happened and still reaches the raster.
  // 'bright' lifts the cell toward white, which washes its color out at the moment it fires.
  // 'sat' rests every cell as a dim gray of its own luminance and flashes it in a boosted version of its color, so the hue is what fires.
  setSpikeStyle(s){ this.spikeStyle = s === 'sat' ? 'sat' : 'bright'; }
  setSpikeSign(s){ this.spikeSign = s === 1 || s === -1 ? s : 0; }
  _showSpike(i){ return !this.spikeSign || !this.sign || this.sign[i] === this.spikeSign; }
  setPotentials(v, fired){
    if(!this.points || this.mode !== 'net') return;
    const a = this.points.geometry.attributes.mv;
    if(!a) return;
    const arr = a.array, n = Math.min(arr.length, v.length);
    for(let i=0;i<n;i++) arr[i] = fired && fired[i] && this._showSpike(i) ? 40 : v[i];
    a.needsUpdate = true;
  }
  // Frame the camera on whatever is loaded.
  // The default camera looks at the origin from a fixed distance, which leaves any model that is not centered there small and off to one side; multi-structure scenes (an eye and a thalamus millimeters away from a column) are effectively invisible without this.
  fitToNet(pad = 1.35){
    if(!this.net || !this.net.count) return;
    this.fitToPoints(this.net.pos, this.net.count, pad);
  }
  fitToPoints(pos, count, pad = 1.35){
    if(!pos || !count) return;
    {
    let x0=Infinity, y0=Infinity, z0=Infinity, x1=-Infinity, y1=-Infinity, z1=-Infinity;
    for(let i=0;i<count;i++){
      const x=pos[i*3], y=pos[i*3+1], z=pos[i*3+2];
      if(x<x0)x0=x; if(x>x1)x1=x;
      if(y<y0)y0=y; if(y>y1)y1=y;
      if(z<z0)z0=z; if(z>z1)z1=z;
    }
    const cx=(x0+x1)/2, cy=(y0+y1)/2, cz=(z0+z1)/2;
    const radius = Math.max(1, 0.5*Math.hypot(x1-x0, y1-y0, z1-z0));
    const dist = pad * radius / Math.tan(this.camera.fov*Math.PI/360);
    const dir = new THREE.Vector3(0.55, 0.62, 0.76).normalize();
    this.camera.position.set(cx + dir.x*dist, cy + dir.y*dist, cz + dir.z*dist);
    this.camera.near = Math.max(1, dist/2000);
    this.camera.far = dist*20;
    this.camera.updateProjectionMatrix();
    this.controls.target.set(cx, cy, cz);
    this.controls.update();
    }
  }
  setNetwork(net){                           // simulated display
    // the selected cell survives rewiring by its stable identity (scatter node, local index), which is the same cell in the new network
    const keep = this.selectedIdentity();
    this.clearMarkSet();
    this._clearDisplay();
    this._select(-1);
    this.mode = 'net';
    this.net = net;
    if(!net){ this.glow = null; return; }
    this.glow = new Float32Array(net.count);
    this._buildCloud(net.pos, net.ntype, net.count);
    this.reselect(keep);
  }
  selectedIdentity(){
    const net = this.net;
    if(this.selected < 0 || !net || !net.src || !net.lidx) return null;
    return [net.src[this.selected], net.lidx[this.selected]];
  }
  reselect(id){
    const net = this.net;
    if(!id || !net || !net.src || !net.lidx) return;
    for(let i = 0; i < net.count; i++)
      if(net.src[i] === id[0] && net.lidx[i] === id[1]){ this._select(i); return; }
  }
  // static previews: the sim (if any) keeps running underneath
  showPoints(pts){
    this._clearDisplay(); this._select(-1);
    this.mode = 'points'; this.net = null; this.glow = null;
    this._buildCloud(pts.pos, pts.ntype, pts.count);
  }
  showGeo(geo){
    this._clearDisplay(); this._select(-1);
    this.mode = 'geo'; this.net = null; this.glow = null;
    let g = null;
    if(geo.shape === 'sphere') { g = new THREE.SphereGeometry(geo.radius, 24, 16); }
    else if(geo.shape === 'ellipsoid'){ g = new THREE.SphereGeometry(1, 24, 16);
      g.scale(...geo.radii); }
    else if(geo.shape === 'cylinder'){ g = new THREE.CylinderGeometry(geo.radius, geo.radius, geo.height, 24); }
    else if(geo.shape === 'torus'){ g = new THREE.TorusGeometry(geo.radius, geo.thickness, 10, 40);
      g.rotateX(Math.PI/2); }
    else if(geo.shape === 'spline'){ g = this._tubeGeo(geo); }
    else if(geo.shape === 'mesh'){
      // the triangles are already in world space
      g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(geo.tris, 3));
    }
    else { g = new THREE.BoxGeometry(...geo.size); }   // box and noise bounds
    const mesh = new THREE.Mesh(g,
      new THREE.MeshBasicMaterial({ wireframe:true, color:0xdb7a36, transparent:true, opacity:0.35 }));
    if(geo.shape !== 'spline' && geo.shape !== 'mesh'){
      mesh.position.set(...geo.center);
      if(geo.rotate) mesh.rotation.set(...geo.rotate.map(d => d*Math.PI/180));
    }
    this.scene.add(mesh); this.previewMesh = mesh;
  }
  onFired(fired){
    if(!this.glow) return;
    for(let i=0;i<fired.length;i++) if(fired[i] && this._showSpike(i)) this.glow[i] = 1;
  }
  frame(){
    const dd = this.renderer.domElement.dataset;
    if(dd.debug === '1'){
      if(this.geoMesh){
        const v = this.geoMesh.position.clone().project(this.camera);
        const r = this.renderer.domElement.getBoundingClientRect();
        dd.giz = JSON.stringify([ r.left + (v.x+1)/2*r.width, r.top + (1-v.y)/2*r.height ]);
      } else dd.giz = '';
      dd.axis = this.gizmo ? String(this.gizmo.axis) + '/' + this.gizmo.mode + '/' + this.gizmo.enabled : 'none';
    }
    if(this.points && this.glow){
      const col = this.points.geometry.attributes.color.array, base = this.base, glow = this.glow;
      if(this.spikeStyle === 'sat') for(let i=0;i<glow.length;i++){
        const g = glow[i], r = base[i*3], gr = base[i*3+1], b = base[i*3+2];
        const L = 0.299*r + 0.587*gr + 0.114*b, rest = 0.3*L;
        col[i*3]   = rest + (Math.min(1, Math.max(0, L + (r - L)*1.8))  - rest)*g;
        col[i*3+1] = rest + (Math.min(1, Math.max(0, L + (gr - L)*1.8)) - rest)*g;
        col[i*3+2] = rest + (Math.min(1, Math.max(0, L + (b - L)*1.8))  - rest)*g;
        glow[i] *= 0.86;
      }
      else for(let i=0;i<glow.length;i++){
        const g = glow[i], k = 0.26 + 1.0*g, add = 0.32*g;
        col[i*3]   = Math.min(1, base[i*3]*k   + add);
        col[i*3+1] = Math.min(1, base[i*3+1]*k + add);
        col[i*3+2] = Math.min(1, base[i*3+2]*k + add);
        glow[i] *= 0.86;
      }
      this.points.geometry.attributes.color.needsUpdate = true;
    }
    this.controls.update();
    this._placeLabels();
    this.renderer.render(this.scene, this.camera);
  }
  _pick(e){
    if(this._splineMarkers && this.gizmo){    // grab a control point before neuron picking
      const r = this.renderer.domElement.getBoundingClientRect();
      const m = new THREE.Vector2(((e.clientX-r.left)/r.width)*2-1, -((e.clientY-r.top)/r.height)*2+1);
      this.raycaster.setFromCamera(m, this.camera);
      const hits = this.raycaster.intersectObjects(this._splineMarkers);
      if(hits.length){ this.gizmo.attach(hits[0].object); return; }
    }
    if(this.mode !== 'net') return;           // previews are not inspectable
    this._select(this._raycast(e));
  }
  // screen-space picking: the neuron nearest the cursor within a few pixels, nearest to the camera among those, and never one the potential filter is hiding.
  // A world-space ray threshold picks points the user cannot even see; this matches what is actually under the cursor.
  _raycast(e){
    if(!this.points || !this.net) return -1;
    const r = this.renderer.domElement.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    this.camera.updateMatrixWorld();
    const mat = new THREE.Matrix4().multiplyMatrices(
      this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    const el = mat.elements;
    const pos = this.net.pos, n = this.net.count;
    const mv = this.points.geometry.attributes.mv ?
      this.points.geometry.attributes.mv.array : null;
    const floor = this.vmFloorU ? this.vmFloorU.value : -100;
    const slice = this.slice;                     // a cell the slicer hides is not under the cursor
    const PX2 = 9*9;
    let best = -1, bestScore = Infinity;
    for(let i=0;i<n;i++){
      if(mv && mv[i] < floor) continue;
      const x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
      if(slice && !sliceContains(slice, x, y, z)) continue;
      const w = el[3]*x + el[7]*y + el[11]*z + el[15];
      if(w <= 0) continue;                          // behind the camera
      const sx = ((el[0]*x + el[4]*y + el[8]*z + el[12])/w*0.5 + 0.5)*r.width;
      const sy = (0.5 - (el[1]*x + el[5]*y + el[9]*z + el[13])/w*0.5)*r.height;
      const dx = sx-mx, dy = sy-my;
      const d2 = dx*dx + dy*dy;
      if(d2 > PX2) continue;
      const depth = (el[2]*x + el[6]*y + el[10]*z + el[14])/w;
      const score = depth*1000 + d2*0.01;           // nearest first, pixels break ties
      if(score < bestScore){ bestScore = score; best = i; }
    }
    return best;
  }
  // composes downstream transform-node params into one matrix so gizmo previews sit where the rendered (transformed) points are; gizmo objects live inside this group, so their local coords stay in shape space
  _chainGroup(chain){
    const G = new THREE.Group();
    const M = new THREE.Matrix4();
    for(const t of chain || []){
      const m = new THREE.Matrix4()
        .makeTranslation(t.translate[0]+t.pivot[0], t.translate[1]+t.pivot[1], t.translate[2]+t.pivot[2])
        .multiply(new THREE.Matrix4().makeRotationFromEuler(
          new THREE.Euler(...t.rotate.map(d => d*Math.PI/180))))
        .multiply(new THREE.Matrix4().makeScale(...t.scale))
        .multiply(new THREE.Matrix4().makeTranslation(-t.pivot[0], -t.pivot[1], -t.pivot[2]));
      M.premultiply(m);
    }
    G.matrix.copy(M);
    G.matrixAutoUpdate = false;
    G.matrixWorldNeedsUpdate = true;
    this.scene.add(G);
    return G;
  }
  editGeometry(node, onChange, chain){
    this.closeGizmo();
    if(node.type === 'spline') return this._editSpline(node, onChange, chain);
    const p = node.params;
    const kind = node.type === 'stimulus' ? 'sphere' : node.type;   // raw stimulus region = sphere
    const E = GEO_EDIT[kind]; if(!E) return;
    this.geoNode = node; this.geoEdit = E;
    this._geoGroup = this._chainGroup(chain);
    const mesh = new THREE.Mesh(E.make(p),
      new THREE.MeshBasicMaterial({ wireframe:true,
        color: node.type === 'stimulus' ? 0xd8c93a :
               node.type === 'move' ? 0x9a6fd0 : 0xdb7a36,
        transparent:true, opacity:0.3 }));
    E.sync(mesh, p);
    this._geoGroup.add(mesh); this.geoMesh = mesh;
    const tc = new TransformControls(this.camera, this.renderer.domElement);
    tc.addEventListener('dragging-changed', ev => { this.controls.enabled = !ev.value; });
    tc.addEventListener('objectChange', () => { E.apply(mesh, p); onChange(node); });
    tc.attach(mesh);
    this.scene.add(tc);
    this.gizmo = tc;
  }
  _tubeGeo(p){
    return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(
      [p.p0, p.p1, p.p2, p.p3].map(q => new THREE.Vector3(...q)),
      false, 'catmullrom', 0.5), 64, p.radius, 8, false);
  }
  // spline editing: one marker per control point; click a marker to grab it, drag to reshape the curve live
  _editSpline(node, onChange, chain){
    const p = node.params;
    this.geoNode = node; this.geoEdit = { modes:['translate'] };
    this._geoGroup = this._chainGroup(chain);
    const tube = new THREE.Mesh(this._tubeGeo(p),
      new THREE.MeshBasicMaterial({ wireframe:true, color:0xdb7a36, transparent:true, opacity:0.3 }));
    this._geoGroup.add(tube); this.geoMesh = tube;
    this._splineMarkers = ['p0','p1','p2','p3'].map(k => {
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(Math.max(90, p.radius*0.7)),
        new THREE.MeshBasicMaterial({ wireframe:true, color:0xffffff, transparent:true, opacity:0.7 }));
      m.position.set(...p[k]); m.userData.key = k;
      this._geoGroup.add(m); return m;
    });
    const tc = new TransformControls(this.camera, this.renderer.domElement);
    tc.addEventListener('dragging-changed', ev => { this.controls.enabled = !ev.value; });
    tc.addEventListener('objectChange', () => {
      const m = tc.object; if(!m) return;
      p[m.userData.key] = [m.position.x, m.position.y, m.position.z];
      tube.geometry.dispose(); tube.geometry = this._tubeGeo(p);
      onChange(node);
    });
    tc.attach(this._splineMarkers[0]);
    this.scene.add(tc);
    this.gizmo = tc;
  }
  closeGizmo(){
    if(!this.gizmo) return;
    this.gizmo.detach(); this.scene.remove(this.gizmo); this.gizmo.dispose();
    if(this.sliceEditing){ this.scene.remove(this.geoMesh); this.sliceEditing = false; }   // the cut itself stays
    this.geoMesh.geometry.dispose(); this.geoMesh.material.dispose();
    if(this._splineMarkers){
      for(const m of this._splineMarkers){ m.geometry.dispose(); m.material.dispose(); }
      this._splineMarkers = null;
    }
    if(this._geoGroup){ this.scene.remove(this._geoGroup); this._geoGroup = null; }
    this.gizmo = null; this.geoMesh = null; this.geoNode = null; this.geoEdit = null;
    this.controls.enabled = true;
  }
  _clearLines(){
    for(const key of ['selLines','selLinesIn']){
      if(this[key]){ this.scene.remove(this[key]);
        this[key].geometry.dispose(); this[key].material.dispose(); this[key] = null; }
    }
  }
  // selection never touches synapse arrays on this thread: the sim worker owns them (at scale, the only copy) and answers 'query' messages
  _select(idx){
    this._clearLines();
    this.selected = idx;
    if(this.onSelect) this.onSelect(idx);
    if(idx < 0 || !this.net){ this.hud.textContent = ''; return; }
    const t = NEURON_TYPES[this.net.ntype[idx]];
    this.hud.textContent = `${this._nid(idx)} \u00b7 ${t.key} \u00b7 querying connections\u2026`;
  }
  _nid(idx){                                 // stable identity: scatter node - local index
    const net = this.net;
    if(!net.lidx || !net.src) return 'neuron ' + idx;
    return String(net.src[idx]).padStart(3, '0') + '-' + String(net.lidx[idx]).padStart(6, '0');
  }
  applyQuery(idx, out, inn, outTotal, inTotal){
    if(idx !== this.selected || !this.net) return;   // stale result
    const pos = this.net.pos;
    const seg = (a, b, arr) => arr.push(pos[a*3], pos[a*3+1], pos[a*3+2],
                                        pos[b*3], pos[b*3+1], pos[b*3+2]);
    const outPos = [], inPos = [];
    for(const j of out) seg(idx, j, outPos);
    for(const i of inn) seg(i, idx, inPos);
    this._clearLines();
    this.selLines = this._lines(outPos, 0xffffff, 0.4);
    this.selLinesIn = this._lines(inPos, 0xa870de, 0.6);
    const t = NEURON_TYPES[this.net.ntype[idx]];
    const capped = out.length < outTotal || inn.length < inTotal ? ' \u00b7 lines capped' : '';
    this.hud.textContent =
      `${this._nid(idx)} \u00b7 ${t.key} \u00b7 axon out ${outTotal} (white) \u00b7 dendrite in ${inTotal} (violet)${capped}`;
  }
  // A whole pathway rather than one neuron's synapses: every segment is a real synapse from the pair the table row names, sampled from a handful of presynaptic cells because a pathway in this block can hold millions and the point is to see where it goes, not to draw all of it.
  // Pathways are a set, not a selection: several can be on at once, each in its own color, and each stays until it is switched off.
  // They are kept apart from the single-neuron selection lines so that clicking a neuron does not wipe out the bundles someone has arranged.
  // Each segment runs from the source population's color to the target's, so a bundle says where it starts and where it lands without an arrowhead, and two pathways sharing a target are visibly related at the end they share.
  addPathway(key, pairs, rgbFrom, rgbTo, sample){
    if(!this.net) return;
    this.removePathway(key);
    // what the bundle is a sample of, so the HUD can say so in numbers
    this.pathSample.set(key, sample || null);
    const pos = this.net.pos;
    const n = pairs.length/2;
    if(!n){ this._pathHud(); return; }
    const vert = new Float32Array(n*6), col = new Float32Array(n*6);
    for(let k = 0; k < n; k++){
      const a = pairs[k*2], b = pairs[k*2+1], o = k*6;
      vert[o] = pos[a*3]; vert[o+1] = pos[a*3+1]; vert[o+2] = pos[a*3+2];
      vert[o+3] = pos[b*3]; vert[o+4] = pos[b*3+1]; vert[o+5] = pos[b*3+2];
      col[o] = rgbFrom[0]; col[o+1] = rgbFrom[1]; col[o+2] = rgbFrom[2];
      col[o+3] = rgbTo[0]; col[o+4] = rgbTo[1]; col[o+5] = rgbTo[2];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(vert, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const line = new THREE.LineSegments(g,
      new THREE.LineBasicMaterial({ vertexColors:true, transparent:true, opacity:0.55, clippingPlanes:this._slicePlanes() }));
    this.scene.add(line);
    this.paths.set(key, line);
    this._pathHud();
  }
  removePathway(key){
    const line = this.paths.get(key);
    if(!line) return;
    this.scene.remove(line); line.geometry.dispose(); line.material.dispose();
    this.paths.delete(key);
    this._pathHud();
  }
  // A set of neurons a node is about, marked in the view: the cells a probe reads, the cells a stimulus drives, the cells an input lands on, the conjunctive cells an analysis found.
  // Each of those is decided by masks, maps and measurements rather than by anything visible in the graph, so without it a node aimed at the wrong population looks exactly like one aimed at the right population.
  // The set is drawn as larger points over the running cloud rather than instead of it, so the rest of the network stays visible around the part being pointed at.
  showMarkSet(pos3, colors, label, size = 7){
    this.clearMarkSet();
    if(!pos3 || !pos3.length) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos3, 3));
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const m = new THREE.PointsMaterial({ size, sizeAttenuation:false,
      vertexColors:true, transparent:true, opacity:0.9, depthTest:false , clippingPlanes:this._slicePlanes() });
    this.probeMark = new THREE.Points(g, m);
    this.probeMark.renderOrder = 3;
    this.scene.add(this.probeMark);
    if(label) this.pathHud.textContent = label;
    this.probeLabel = label || '';
  }
  clearMarkSet(){
    if(this.probeMark){
      this.scene.remove(this.probeMark);
      this.probeMark.geometry.dispose(); this.probeMark.material.dispose();
      this.probeMark = null;
    }
    if(this.probeLabel && this.pathHud.textContent === this.probeLabel)
      this._pathHud();                      // give the line back to the pathways
    this.probeLabel = '';
  }
  clearPathways(){
    for(const key of [...this.paths.keys()]) this.removePathway(key);
    this._pathHud();
  }
  // The bundle is a sample: a dozen source cells of the pathway, each with up to a query's worth of its synapses.
  // The line says how many of the sources were drawn, since twelve lines out of twenty thousand cells otherwise read as a pathway that reaches almost nothing.
  _pathHud(){
    if(!this.paths.size){ this.pathHud.textContent = ''; return; }
    let segs = 0, drawn = 0, of = 0, per = 0;
    for(const [k, l] of this.paths){
      segs += l.geometry.attributes.position.count/2;
      const sm = this.pathSample.get(k);
      if(sm){ drawn += sm.sampled; of += sm.ofSources; per = Math.max(per, sm.perSource || 0); }
    }
    const n = this.paths.size;
    this.pathHud.textContent =
      `${n} pathway${n > 1 ? 's' : ''} · ${segs.toLocaleString()} synapses drawn, ` +
      `up to ${per} each from ${drawn.toLocaleString()} of ${of.toLocaleString()} source cells`;
  }
  _lines(arr, color, opacity){
    if(!arr.length) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(arr), 3));
    const l = new THREE.LineSegments(g,
      new THREE.LineBasicMaterial({ color, transparent:true, opacity, clippingPlanes:this._slicePlanes() }));
    this.scene.add(l);
    return l;
  }
}

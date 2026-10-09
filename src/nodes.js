import { buildExpr } from './expr.js';
import { h32, pairHash, pairGauss, rng } from './rand.js';
import { classicRow } from './form.js';
import { PROTOCOL } from './protocol.js';
import { parsePhases } from './curriculum.js';

// Neuron presets: the seven Izhikevich (2003) parameter sets and the fly rows. sign = excitatory(+1)/inhibitory(-1) for Dale's law (TC relay cells are glutamatergic; RZ is used here as a resonator interneuron).
// Every row runs as an Izhikevich 2007 row (f7; Dynamical Systems in Neuroscience, chapter 8): C dv/dt = k (v - vr)(v - vt) - u + I, du/dt = a (b (v - vr) - u), spike at vpeak (MODEL.md 1).
// The engines read f7 only.
// On rows 0 to 6, a, b, c, d are the Izhikevich 2003 parameters and their f7 rows are derived from them below (form.js).
export const NEURON_TYPES = [
  { key:'RS',  name:'RS regular spiking (exc)',   a:0.02, b:0.2,  c:-65, d:8,    sign: 1, color:[0.86,0.48,0.21],
    f7:null },
  { key:'IB',  name:'IB intrinsic burst (exc)',   a:0.02, b:0.2,  c:-55, d:4,    sign: 1, color:[0.89,0.79,0.25],
    f7:null },
  { key:'CH',  name:'CH chattering (exc)',        a:0.02, b:0.2,  c:-50, d:2,    sign: 1, color:[0.85,0.29,0.51],
    f7:null },
  { key:'FS',  name:'FS fast spiking (inh)',      a:0.10, b:0.2,  c:-65, d:2,    sign:-1, color:[0.22,0.72,0.85],
    f7:null },
  { key:'LTS', name:'LTS low threshold (inh)',    a:0.02, b:0.25, c:-65, d:2,    sign:-1, color:[0.38,0.82,0.42],
    f7:null },
  { key:'TC',  name:'TC thalamocortical (exc)',   a:0.02, b:0.25, c:-65, d:0.05, sign: 1, color:[0.63,0.45,0.88],
    f7:null },
  { key:'RZ',  name:'RZ resonator (inh)',         a:0.10, b:0.26, c:-65, d:2,    sign:-1, color:[0.90,0.30,0.30],
    f7:null },
  // Fly rows (for the MaleCNS import), written as 2007 rows; the top-level a, b, c, d are RS's and no engine reads them.
  // FL is a spiking central neuron: 20 pF and about 1 GOhm (a membrane time constant near 20 ms, Gouwens and Wilson 2009), rest near -58 mV (Wilson and Laurent 2005), so k (vt - vr) is 1 nS and the rheobase is a few picoamps.
  // FG is a graded relay, the lamina and medulla cells that do not spike, approximated as a cell that fires readily and in proportion: the threshold 10 mV over rest, a low peak, little adaptation.
  // Both take their sign from the transmitter column when the connections file node is in that mode; FG is excitatory on its own.
  { key:'FL',  name:'FL fly central (exc)', hidden:true,       a:0.02, b:0.2,  c:-65, d:8,    sign: 1, color:[0.95,0.80,0.45],
    f7:{ C:20,  k:0.06, vr:-58, vt:-40, vpeak:20, a:0.05, b:0,    c:-50, d:10 } },
  // Graded rows do not spike: release is (v - thr) / slope clamped to
  // [0, 1] every millisecond, delivered through the same synapses as a
  // spike of that amplitude (MODEL.md 1).
  { key:'FG',  name:'FG fly graded relay (exc)', hidden:true,  a:0.02, b:0.2,  c:-65, d:8,    sign: 1, color:[0.70,0.90,0.60],
    f7:{ C:20,  k:0.06, vr:-55, vt:-45, vpeak:-20, a:0.1, b:0,    c:-50, d:2, graded:{ thr:-55, slope:10 } } },
  // FS is the graded relay made slow: the same cell with k a fifth of FG's, so its membrane time constant near rest is about 170 ms rather than 30.
  // The fly's Mi9 (and the other slow arms of the motion detector) filter their input over roughly that long, which is where T4's direction selectivity comes from (the Mi9 arm arrives late).
  { key:'FS7', name:'FS7 fly graded slow (exc)', hidden:true,  a:0.02, b:0.2,  c:-65, d:8,    sign: 1, color:[0.55,0.75,0.95],
    f7:{ C:20,  k:0.012, vr:-55, vt:-45, vpeak:-20, a:0.05, b:0,   c:-50, d:1, graded:{ thr:-55, slope:10 } } },
];

// Rows 0 to 6 are the Izhikevich 2003 presets, carried as rows of the 2007 form (form.js: C 1, k 0.04, the rest at the lower root of 0.04 v^2 + (5 - b) v + 140, the threshold at -125 - vr), derived here from a, b, c, d so they cannot drift from the numbers they stand for.
// They are what a scene means by RS, FS and the rest, so every scene and every default weight, bias and drive keeps the behavior it had under the 2003 form; the capacitance of 1 pF is a unit choice, and under these rows a picoamp is the 2003 model's current unit.
for(let i = 0; i < 7; i++) NEURON_TYPES[i].f7 = classicRow(NEURON_TYPES[i]);
// The built-in rows, for the cell type node's presets; taken before any scene registers a row of its own.
export const CELL_PRESETS = [];
// The measured rows of Izhikevich 2007 chapter 8 (C, k, vr, vt, vpeak, a, b, c, d) for the same seven types, as rows of their own: FS as a linear reading of the book's cubic recovery nullcline, LTS and TC with a fixed vpeak, RZ built as a resonator since the book has none.
// These are the rows a scene reaches for when it wants a cell type with a measured rest; their currents are picoamps and a scene written on them sets its own weights.
NEURON_TYPES.push(
  { key:'RS07', name:'RS regular spiking, 2007 measured (exc)', measured:true, hidden:true, sign: 1, color:NEURON_TYPES[0].color,
    f7:{ C:100, k:0.7, vr:-60, vt:-40, vpeak:35, a:0.03, b:-2,   c:-50, d:100 } },
  { key:'IB07', name:'IB intrinsic burst, 2007 measured (exc)', measured:true, hidden:true, sign: 1, color:NEURON_TYPES[1].color,
    f7:{ C:150, k:1.2, vr:-75, vt:-45, vpeak:50, a:0.01, b:5,    c:-56, d:130 } },
  { key:'CH07', name:'CH chattering, 2007 measured (exc)', measured:true, hidden:true, sign: 1, color:NEURON_TYPES[2].color,
    f7:{ C:50,  k:1.5, vr:-60, vt:-40, vpeak:25, a:0.03, b:1,    c:-40, d:150 } },
  { key:'FS07', name:'FS fast spiking, 2007 measured (inh)', measured:true, hidden:true, sign:-1, color:NEURON_TYPES[3].color,
    f7:{ C:20,  k:1,   vr:-55, vt:-40, vpeak:25, a:0.2,  b:0.2,  c:-45, d:0 } },
  { key:'LTS07', name:'LTS low threshold, 2007 measured (inh)', measured:true, hidden:true, sign:-1, color:NEURON_TYPES[4].color,
    f7:{ C:100, k:1,   vr:-56, vt:-42, vpeak:40, a:0.03, b:8,    c:-53, d:20 } },
  { key:'TC07', name:'TC thalamocortical, 2007 measured (exc)', measured:true, hidden:true, sign: 1, color:NEURON_TYPES[5].color,
    f7:{ C:200, k:1.6, vr:-60, vt:-50, vpeak:35, a:0.01, b:15,   c:-60, d:10 } },
  { key:'RZ07', name:'RZ resonator, 2007 measured (inh)', measured:true, hidden:true, sign:-1, color:NEURON_TYPES[6].color,
    f7:{ C:100, k:0.7, vr:-60, vt:-40, vpeak:35, a:0.1,  b:5,    c:-60, d:10 } },
);
CELL_PRESETS.push(...NEURON_TYPES.filter(t => !t.custom));
// A scene's own cell types join the type table by name (the cell type node): a new name is appended, a known one is updated in place so its index, and every ntype pointing at it, stays valid.
// Indices above the built-in rows belong to these.
// The computing thread is the authority on where a row sits: a row that rides on a stream carries its index (`index`), and a wiring worker puts it there.
// Numbering rows in the order a worker meets them matches the computing thread's order only while a session has made one custom row.
// A gap a worker has not been told about is filled with a placeholder no cell of the stream points at.
export function registerCellType(row){
  const k = NEURON_TYPES.findIndex(t => t.key === row.key);
  const want = Number.isInteger(row.index) ? row.index : -1;
  if(k >= 0 && (want < 0 || want === k)){ Object.assign(NEURON_TYPES[k], row); return k; }
  if(want >= 0){
    if(want >= 255) throw new Error('cell type: the type table holds 255 rows');
    if(k >= 0) NEURON_TYPES[k] = { ...NEURON_TYPES[k], key:'_moved' + k, placeholder:true };
    while(NEURON_TYPES.length < want) NEURON_TYPES.push({ ...NEURON_TYPES[0], key:'_hole' + NEURON_TYPES.length, name:'(unused)', custom:true, placeholder:true, hidden:true });
    NEURON_TYPES[want] = row;
    return want;
  }
  if(NEURON_TYPES.length >= 255) throw new Error('cell type: the type table holds 255 rows');
  NEURON_TYPES.push(row);
  return NEURON_TYPES.length - 1;
}
// The type table as an engine takes it: the 2007 row of every type (f7), nothing else.
// Every init message builds its table here.
export function engineTypes(types = NEURON_TYPES){
  return types.map(t => ({ f7:t.f7 ? { ...t.f7, graded:t.f7.graded ? { ...t.f7.graded } : undefined } : undefined }));
}
// Which cells are graded: a byte per neuron, 1 for a graded row, and the release threshold and slope per neuron; null when none.
export function gradedArrays(ntype, types){
  const n = ntype.length;
  let any = false;
  for(let i = 0; i < n; i++){ const t = types[ntype[i]]; if(t && t.f7 && t.f7.graded){ any = true; break; } }
  if(!any) return null;
  const grd = new Uint8Array(n), thr = new Float32Array(n), slope = new Float32Array(n);
  for(let i = 0; i < n; i++){ const g = types[ntype[i]].f7.graded; if(g){ grd[i] = 1; thr[i] = g.thr; slope[i] = g.slope > 0 ? g.slope : 1; } }
  return { grd, thr, slope };
}
// Plasticity has no spike to time against on a graded cell, so every synapse it sends is frozen (rule 0) for the engine: a copy of the plasticity byte array with those entries zeroed, or the input when no cell is graded.
// Applied at init by every engine, so the wiring and the brain files do not change.
export function freezeGraded(pmask, preStart, grd, m){
  if(!grd) return pmask;
  const out = pmask ? Uint8Array.from(pmask) : new Uint8Array(m).fill(1);
  for(let i = 0; i < grd.length; i++) if(grd[i]) for(let s = preStart[i]; s < preStart[i+1]; s++) out[s] &= 0xe0;   // the rule bits only; the channel stays
  return out;
}
const MAX_DELAY = 16;   // sim steps (ms); the ring buffer is sized from it


export function need(x, kind, msg){ if(!x || x.kind!==kind) throw new Error(msg); return x; }

// seeded value noise (trilinear hash noise, 2 octaves, ~[0,1])
function hash3(x, y, z, seed){
  let h = (x*374761393 + y*668265263 + z*1440662683 + seed*974711) | 0;
  h = Math.imul(h ^ h>>>13, 1274126177);
  return ((h ^ h>>>16)>>>0)/4294967296;
}
function vnoise(x, y, z, seed){
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const u = (x-xi)*(x-xi)*(3-2*(x-xi)), v = (y-yi)*(y-yi)*(3-2*(y-yi)),
        w = (z-zi)*(z-zi)*(3-2*(z-zi));
  const n = (X,Y,Z) => hash3(xi+X, yi+Y, zi+Z, seed);
  const L = (a,b,t) => a+(b-a)*t;
  return L(L(L(n(0,0,0),n(1,0,0),u), L(n(0,1,0),n(1,1,0),u), v),
           L(L(n(0,0,1),n(1,0,1),u), L(n(0,1,1),n(1,1,1),u), v), w);
}
function fnoise(x, y, z, seed){
  return 0.65*vnoise(x, y, z, seed) + 0.35*vnoise(x*2.7, y*2.7, z*2.7, seed+101);
}
function fbm(x, y, z, seed, oct){          // 1-4 octaves, ~[0,1]
  let sum = 0, amp = 1, tot = 0, f = 1;
  const K = Math.max(1, Math.min(4, oct|0 || 2));
  for(let k=0;k<K;k++){
    sum += amp*vnoise(x*f, y*f, z*f, seed + k*101);
    tot += amp; amp *= 0.55; f *= 2.7;
  }
  return sum/tot;
}
// the noise volume's density test (local coords).
// Patterns model observed tissue statistics: blobs = multi-scale clumping (fBm), patches = periodic blob/barrel maps at one characteristic wavelength, stripes = meandering band systems (ocular-dominance-like: warped thresholded sine).
function noiseField(g, x, y, z){
  const fs = Math.max(10, g.scale);
  const cov = Math.max(0.05, Math.min(1, g.coverage));
  const pat = g.pattern|0;
  if(pat === 2){
    const d0 = g.direction || [1,0,0];
    const d = (d0[0]||d0[1]||d0[2]) ? d0 : [1,0,0];
    const L = Math.hypot(d[0], d[1], d[2]);
    const phase = 2*Math.PI*(x*d[0]+y*d[1]+z*d[2])/(L*fs);
    const wig = 5*(fnoise(x/fs, y/fs, z/fs, g.seed) - 0.5);
    return Math.sin(phase + wig) >= Math.cos(Math.PI*cov);
  }
  if(pat === 1)
    return vnoise(2*x/fs, 2*y/fs, 2*z/fs, g.seed) <= cov*0.62;
  return fbm(x/fs, y/fs, z/fs, g.seed, g.octaves) <= cov;
}
function gaussRand(r){                     // Box-Muller
  let u = 0; do { u = r(); } while(!u);
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r());
}

// pair-hash: a uniform in [0,1) determined only by the two neurons' stable identities and the seed.
// Connectivity built from this is additive: adding populations, raising density, or widening kernels never changes the outcome for existing pairs. h32, pairHash, pairGauss and rng live in rand.js, with every other seeded draw.
// Stable neuron identity is (src scatter node id, local index within that scatter); it survives merging, culling and reordering, and is the basis for pair-hash connectivity and learned-state persistence.
// The key of a spatial grid cell: the three cell coordinates joined, which cannot collide.
// An XOR of scaled coordinates collides for cells on opposite sides of the origin ((-1,1,0) and (1,-1,0), (-1,-1,-1) and (1,1,1), twelve pairs among the 27 neighbors of the origin cell), so a bucket would hold two cells' points and the neighbor scan visit it twice (src/wire.test.mjs).
const gridKey = (x, y, z) => x + ',' + y + ',' + z;
const gridKey2 = (x, y) => x + ',' + y;
function freshLidx(n){ return Uint32Array.from({ length:n }, (_, i) => i); }
export function clonePts(p){ return { kind:'points', count:p.count, pos:p.pos.slice(),
  ntype:p.ntype.slice(), bias:p.bias.slice(), ...(p.comp ? { comp:p.comp } : {}),
  ...(p.projs ? { projs:p.projs } : {}), ...(p.rules ? { rules:p.rules } : {}),
  ...(p.pools ? { pools:p.pools } : {}), ...(p.ids ? { ids:p.ids.slice() } : {}),
  ...(p.connectionTables ? { connectionTables:p.connectionTables } : {}), ...(p.receptors ? { receptors:p.receptors } : {}),
  ...(p.cellTypes ? { cellTypes:p.cellTypes } : {}), ...(p.hues ? { hues:p.hues } : {}),
  src:p.src ? p.src.slice() : new Int32Array(p.count).fill(-1),
  lidx:p.lidx ? p.lidx.slice() : freshLidx(p.count),
  tags:{ ...(p.tags || {}) } }; }
// Identity of repeat copies.
// Copy k of cell (src, lidx) is (src, lidx*64 + k), and when the copies are numbered its source becomes src*65 + (k+1).
// Both are mixed-radix encodings, so they stay unique when a repeat feeds another repeat (a ring of bulbs repeated into tiers), where an additive block would land tier1.bulb2 and tier2.bulb1 on one id.
// Neither depends on how many cells the stream carried.
// Limits: 64 copies per repeat, and a scatter's count times 64 to the nesting depth under 2^32.
export const LIDX_RADIX = 64, SRC_RADIX = 65;
function repeatCompute(pts, p){
  const N = Math.max(2, Math.min(64, p.copies | 0));
  const numbered = (p.pops | 0) === 0;
  const tag = String(p.tag || '').trim().replace(/\s+/g, '');
  // copy k's population name: with a tag on the node, tag1.pop; without, pop1, pop2 ... (a scatter named RSexc comes out as RSexc1 and RSexc2)
  const nameOf = (pop, k) => tag ? tag + (k+1) + (pop ? '.' + pop : '')
    : (pop || 'copy') + (k+1);
  const m = pts.count, n = m*N;
  const out = { kind:'points', count:n, pos:new Float32Array(n*3),
    ntype:new Uint8Array(n), bias:new Float32Array(n), src:new Int32Array(n),
    lidx:new Uint32Array(n), tags:{} };
  const src0 = pts.src || new Int32Array(m).fill(-1);
  const lidx0 = pts.lidx || freshLidx(m);
  const tags0 = pts.tags || {};
  const rot = p.rotate.some(v => v) ? rotMat(...p.rotate.map(d => d*Math.PI/180)) : null;
  const ax = (p.mirror | 0) - 1;                 // -1 none, else the axis reflected
  const step = v => {                            // one step of the transform, about the pivot
    let x = v[0]-p.pivot[0], y = v[1]-p.pivot[1], z = v[2]-p.pivot[2];
    x *= p.scale[0]; y *= p.scale[1]; z *= p.scale[2];
    if(rot){ const rx = rot[0]*x+rot[1]*y+rot[2]*z, ry = rot[3]*x+rot[4]*y+rot[5]*z,
                   rz = rot[6]*x+rot[7]*y+rot[8]*z; x=rx; y=ry; z=rz; }
    v[0] = x + p.pivot[0] + p.translate[0];
    v[1] = y + p.pivot[1] + p.translate[1];
    v[2] = z + p.pivot[2] + p.translate[2];
  };
  const v = [0, 0, 0];
  for(let k=0;k<N;k++){
    const o = k*m;
    out.ntype.set(pts.ntype, o); out.bias.set(pts.bias, o);
    const reid = s0 => numbered ? s0*SRC_RADIX + (k+1) : s0;
    for(let i=0;i<m;i++){
      v[0] = pts.pos[i*3]; v[1] = pts.pos[i*3+1]; v[2] = pts.pos[i*3+2];
      // odd copies are reflected first, then stepped like the rest
      if(ax >= 0 && (k & 1)) v[ax] = 2*p.pivot[ax] - v[ax];
      for(let s=0;s<k;s++) step(v);
      out.pos[(o+i)*3] = v[0]; out.pos[(o+i)*3+1] = v[1]; out.pos[(o+i)*3+2] = v[2];
      const s0 = src0[i];
      out.src[o+i] = s0 < 0 ? -1 : reid(s0);
      out.lidx[o+i] = lidx0[i]*LIDX_RADIX + k;
    }
    for(const id of Object.keys(tags0)){
      const sid = +id;
      if(numbered) out.tags[reid(sid)] = nameOf(tags0[id], k);
      else out.tags[sid] = tags0[id];
    }
  }
  // A projection made before the repeat travels with the populations it joins: copy k of a1 to b1 is a2 to b2, on the copied geometry, with its own seed.
  // A tag the stream does not carry (a target outside the repeat) is left as it is.
  // Shared copies keep the projection once, since the tags are the same in every copy.
  const projs0 = pts.projs || [];
  if(projs0.length){
    const own = Object.values(tags0);
    const rename = (t, k) => {
      const hit = own.find(x => sameTag(x, t));
      return hit !== undefined ? nameOf(hit, k) : t;
    };
    // A projection with an offset joins copy k to copy k plus offset, at the repeat it names (or the first numbered one it meets); the copies that would reach past the end are left out.
    // Once applied the offset is spent, so an outer repeat copies the pathway as it is.
    const applies = pr => (pr.offset|0) !== 0 && (!pr.across || sameTag(pr.across, tag));
    out.projs = numbered
      ? projs0.flatMap(pr => applies(pr)
          ? Array.from({ length:N }, (_, k) => k + (pr.offset|0))
              .map((j, k) => (j < 0 || j >= N) ? null
                : ({ ...pr, from:rename(pr.from, k), to:rename(pr.to, j), seed:(pr.seed|0) + k, offset:0, across:'' }))
              .filter(Boolean)
          : Array.from({ length:N }, (_, k) =>
              ({ ...pr, from:rename(pr.from, k), to:rename(pr.to, k), seed:(pr.seed|0) + k })))
      : projs0.slice();
  }
  if(pts.rules) out.rules = pts.rules;
  // What rides on the stream and does not name a population goes through as it is, a custom cell type's row included: the cells keep its index and the wiring workers need the row to read its sign.
  if(pts.cellTypes) out.cellTypes = pts.cellTypes;
  if(pts.receptors) out.receptors = pts.receptors;
  if(pts.hues) out.hues = pts.hues;
  // A pool on one of the stream's own populations is one pool per numbered copy, named like the copy; a pool on a class, on everything, or on a pattern is carried as it is and resolves when the network is wired.
  if(pts.pools){
    const own = Object.values(tags0);
    out.pools = numbered
      ? pts.pools.flatMap(pl => own.some(x => sameTag(x, pl.tag))
          ? Array.from({ length:N }, (_, k) => ({ ...pl, tag:nameOf(own.find(x => sameTag(x, pl.tag)), k) }))
          : [pl])
      : pts.pools.slice();
  }
  if(pts.connectionTables) out.connectionTables = pts.connectionTables;   // ids do not survive a repeat, so this lands nothing there
  return out;
}
function pickPts(pts, idx){
  const n = idx.length;
  const out = { kind:'points', count:n, pos:new Float32Array(n*3),
    ntype:new Uint8Array(n), bias:new Float32Array(n), src:new Int32Array(n),
    lidx:new Uint32Array(n) };
  for(let k=0;k<n;k++){ const i = idx[k];
    out.pos[k*3] = pts.pos[i*3]; out.pos[k*3+1] = pts.pos[i*3+1]; out.pos[k*3+2] = pts.pos[i*3+2];
    out.ntype[k] = pts.ntype[i]; out.bias[k] = pts.bias[i];
    out.src[k] = pts.src ? pts.src[i] : -1;
    out.lidx[k] = pts.lidx ? pts.lidx[i] : i;
  }
  out.tags = { ...(pts.tags || {}) };
  if(pts.projs) out.projs = pts.projs;
  if(pts.rules) out.rules = pts.rules;
  if(pts.pools) out.pools = pts.pools;
  if(pts.ids) out.ids = Array.from(idx, i => pts.ids[i]);
  if(pts.connectionTables) out.connectionTables = pts.connectionTables;
  if(pts.receptors) out.receptors = pts.receptors;
  if(pts.cellTypes) out.cellTypes = pts.cellTypes;
  if(pts.hues) out.hues = pts.hues;
  return out;
}

function rotMat(rx, ry, rz){                       // R = Rz * Ry * Rx
  const cx=Math.cos(rx), sx=Math.sin(rx), cy=Math.cos(ry), sy=Math.sin(ry),
        cz=Math.cos(rz), sz=Math.sin(rz);
  return [ cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
           sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx,
           -sy,   cy*sx,            cy*cx ];
}

// Euler angles back out of a rotation matrix, in the same Rz*Ry*Rx order rotMat builds it with, so a rotation composed here can be handed to a shape that stores its own angles.
function eulerOf(m){
  const sy = -m[6];
  if(Math.abs(sy) < 0.99999)
    return [Math.atan2(m[7], m[8]), Math.asin(sy), Math.atan2(m[3], m[0])];
  // gimbal lock: pitch is straight up or down and the other two angles are one degree of freedom between them, so roll is folded into yaw
  return [0, Math.asin(sy), Math.atan2(-m[1], m[4])];
}
function matMul(a, b){
  const o = new Array(9);
  for(let r=0;r<3;r++) for(let c=0;c<3;c++)
    o[r*3+c] = a[r*3]*b[c] + a[r*3+1]*b[3+c] + a[r*3+2]*b[6+c];
  return o;
}
// A transform applied to a region rather than to points.
// Masks are geometry, so a transform in a mask wire moves, turns and resizes the region a probe reads or an input drives, which is the same operation the gizmo does by hand and the only way to do it to a region that several nodes share.
export function transformGeo(g, p){
  const rot = p.rotate.some(v => v);
  const rad = p.rotate.map(d => d*Math.PI/180);
  const R = rot ? rotMat(...rad) : null;
  const piv = p.pivot || [0, 0, 0];
  const place = c => {
    let x = c[0] - piv[0], y = c[1] - piv[1], z = c[2] - piv[2];
    x *= p.scale[0]; y *= p.scale[1]; z *= p.scale[2];
    if(R){ const rx = R[0]*x + R[1]*y + R[2]*z, ry = R[3]*x + R[4]*y + R[5]*z,
      rz = R[6]*x + R[7]*y + R[8]*z; x = rx; y = ry; z = rz; }
    return [x + piv[0] + p.translate[0], y + piv[1] + p.translate[1], z + piv[2] + p.translate[2]];
  };
  const out = { ...g };
  if(g.center) out.center = place(g.center);
  for(const k of ['p0', 'p1', 'p2', 'p3']) if(g[k]) out[k] = place(g[k]);
  if(g.shape === 'mesh'){
    // every vertex moves; the bounds follow; the bucket grid is rebuilt on the next inside test
    const t = new Float32Array(g.tris.length);
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for(let i = 0; i < t.length; i += 3){
      const q = place([g.tris[i], g.tris[i+1], g.tris[i+2]]);
      for(let k = 0; k < 3; k++){ t[i+k] = q[k]; if(q[k] < min[k]) min[k] = q[k]; if(q[k] > max[k]) max[k] = q[k]; }
    }
    out.tris = t; out.min = min; out.max = max; delete out._grid;
    out.center = [(min[0]+max[0])/2, (min[1]+max[1])/2, (min[2]+max[2])/2];
    out.size = [max[0]-min[0], max[1]-min[1], max[2]-min[2]];
    return out;
  }
  const s = p.scale, mean = (Math.abs(s[0]) + Math.abs(s[1]) + Math.abs(s[2]))/3;
  if(g.size) out.size = [g.size[0]*Math.abs(s[0]), g.size[1]*Math.abs(s[1]), g.size[2]*Math.abs(s[2])];
  if(g.radii) out.radii = [g.radii[0]*Math.abs(s[0]), g.radii[1]*Math.abs(s[1]), g.radii[2]*Math.abs(s[2])];
  // A sphere, a cylinder's girth and a torus have one radius between them and three scale factors, so a non-uniform scale on those is the mean of the three: the shape stays the shape it is rather than silently becoming another one.
  if(g.radius !== undefined) out.radius = g.radius*mean;
  if(g.thickness !== undefined) out.thickness = g.thickness*mean;
  if(g.height !== undefined) out.height = g.height*Math.abs(s[1]);
  if(rot){
    const own = g.rotate ? rotMat(...g.rotate.map(d => d*Math.PI/180)) : null;
    if(g.rotate || g.shape === 'box' || g.shape === 'ellipsoid' ||
       g.shape === 'cylinder' || g.shape === 'torus' || g.shape === 'noisefield'){
      const m = own ? matMul(R, own) : R;
      out.rotate = eulerOf(m).map(r => r*180/Math.PI);
    }
    // a sphere has no orientation to turn, and a spline's control points have already moved, which is its rotation
  }
  return out;
}

// ---- mesh geometry (OBJ) -------------------------------------------------
// A triangle mesh from a file is a region like a sphere or a box: the scatter fills it, a mask reads it, a transform moves it.
// The file lives in the project folder and is read through whatever the page has for files, registered here so this module stays importable in workers and tests.
let fileReader = null;                    // path in the project -> Promise<Uint8Array>
export function setFileReader(fn){ fileReader = fn; }
const meshCache = new Map();               // path -> parsed objects, until reload
export function forgetMesh(path){ if(path === undefined) meshCache.clear(); else meshCache.delete(path); }

// Wavefront OBJ: v lines are vertices, f lines are faces (any polygon, fanned into triangles; indices 1-based, negative counts back from the end, v/vt/vn forms keep the first field), o and g lines name objects.
// Nothing else is read.
// Returns the named objects with a flat triangle array each.
export function parseObj(text){
  const verts = [];
  const objects = [];
  let cur = null;
  const open = name => { cur = { name, tris:[] }; objects.push(cur); };
  const lines = String(text).split(/\r?\n/);
  for(let li = 0; li < lines.length; li++){
    const line = lines[li];
    if(!line || line[0] === '#') continue;
    const parts = line.trim().split(/\s+/);
    const k = parts[0];
    if(k === 'v'){ verts.push(+parts[1] || 0, +parts[2] || 0, +parts[3] || 0); }
    else if(k === 'o' || k === 'g'){ const name = parts.slice(1).join(' ').trim(); if(!cur || cur.tris.length || cur.name !== name) open(name || ('object ' + (objects.length + 1))); }
    else if(k === 'f'){
      if(!cur) open('object 1');
      const idx = [];
      for(let i = 1; i < parts.length; i++){
        const f = parts[i].split('/')[0];
        let v = parseInt(f, 10); if(!Number.isFinite(v)) continue;
        if(v < 0) v = verts.length/3 + v + 1;
        idx.push(v - 1);
      }
      for(let i = 1; i + 1 < idx.length; i++)
        for(const q of [idx[0], idx[i], idx[i+1]]){
          const b = q*3;
          if(b < 0 || b + 2 >= verts.length) throw new Error('OBJ: face index ' + (q + 1) + ' past the ' + verts.length/3 + ' vertices (line ' + (li + 1) + ')');
          cur.tris.push(verts[b], verts[b+1], verts[b+2]);
        }
    }
  }
  return { objects: objects.filter(o => o.tris.length).map(o => ({ name:o.name, tris:Float32Array.from(o.tris) })),
    vertices: verts.length/3 };
}

// glTF 2.0 binary (.glb): a 12-byte header, a JSON chunk and a binary chunk.
// Meshes are read through their accessors (float positions, u8, u16 or u32 indices, triangle lists only), placed by the scene graph's node transforms (a matrix, or translation, rotation, scale), and named by the node or the mesh.
// Draco and meshopt compression are refused rather than decoded; an external .bin is refused, so export as one .glb.
// A .gltf with its buffer embedded as a data URI is accepted through parseGltfJson.
export function parseGlb(bytes){
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if(bytes.byteLength < 20 || dv.getUint32(0, true) !== 0x46546C67) throw new Error('GLB: not a glTF binary (no glTF magic)');
  const version = dv.getUint32(4, true);
  if(version !== 2) throw new Error('GLB: glTF version ' + version + '; version 2 is what is read');
  let at = 12, json = null, bin = null;
  while(at + 8 <= bytes.byteLength){
    const len = dv.getUint32(at, true), type = dv.getUint32(at + 4, true);
    const body = bytes.subarray(at + 8, at + 8 + len);
    if(type === 0x4E4F534A) json = JSON.parse(new TextDecoder().decode(body));
    else if(type === 0x004E4942) bin = body;
    at += 8 + len;
  }
  if(!json) throw new Error('GLB: no JSON chunk');
  return parseGltfJson(json, bin);
}
export function parseGltfJson(json, bin){
  const ext = (json.extensionsRequired || []).concat(json.extensionsUsed || []);
  for(const e of ext) if(/draco|meshopt/i.test(e))
    throw new Error('glTF: ' + e + ' compression is not read; export without compression');
  const buffers = (json.buffers || []).map((b, i) => {
    if(b.uri){
      const m = /^data:.*?;base64,(.*)$/.exec(b.uri);
      if(!m) throw new Error('glTF: buffer ' + i + ' is an external file (' + b.uri + '); export as one .glb');
      const s = atob(m[1]); const out = new Uint8Array(s.length);
      for(let k = 0; k < s.length; k++) out[k] = s.charCodeAt(k);
      return out;
    }
    if(!bin) throw new Error('glTF: buffer ' + i + ' has no data');
    return bin;
  });
  const accessor = idx => {
    const a = json.accessors[idx], bv = json.bufferViews[a.bufferView];
    const buf = buffers[bv.buffer];
    const off = buf.byteOffset + (bv.byteOffset || 0) + (a.byteOffset || 0);
    const comps = { SCALAR:1, VEC2:2, VEC3:3, VEC4:4, MAT4:16 }[a.type] || 1;
    const size = { 5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4 }[a.componentType];
    const stride = bv.byteStride || comps*size;
    const dv = new DataView(buf.buffer, off, a.count*stride - (stride - comps*size));
    const out = new Float64Array(a.count*comps);
    const rd = { 5120:(o) => dv.getInt8(o), 5121:(o) => dv.getUint8(o), 5122:(o) => dv.getInt16(o, true),
      5123:(o) => dv.getUint16(o, true), 5125:(o) => dv.getUint32(o, true), 5126:(o) => dv.getFloat32(o, true) }[a.componentType];
    if(!rd) throw new Error('glTF: accessor component type ' + a.componentType);
    for(let i = 0; i < a.count; i++) for(let c = 0; c < comps; c++) out[i*comps + c] = rd(i*stride + c*size);
    return { data:out, comps, count:a.count };
  };
  const I = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
  const mul = (a, b) => { const o = new Array(16);       // column-major, o = a * b
    for(let c = 0; c < 4; c++) for(let r = 0; r < 4; r++){ let s = 0;
      for(let k = 0; k < 4; k++) s += a[k*4 + r]*b[c*4 + k]; o[c*4 + r] = s; } return o; };
  const local = n => {
    if(n.matrix) return n.matrix.slice();
    const t = n.translation || [0,0,0], q = n.rotation || [0,0,0,1], s = n.scale || [1,1,1];
    const [x, y, z, w] = q;
    const R = [1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w), 0,
               2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w), 0,
               2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y), 0,
               0, 0, 0, 1];
    for(let c = 0; c < 3; c++) for(let r = 0; r < 3; r++) R[c*4 + r] *= s[c];
    R[12] = t[0]; R[13] = t[1]; R[14] = t[2];
    return R;
  };
  const objects = [];
  const visit = (ni, parent, seen) => {
    if(seen.has(ni)) return; seen.add(ni);
    const n = json.nodes[ni];
    const M = mul(parent, local(n));
    if(n.mesh !== undefined){
      const mesh = json.meshes[n.mesh];
      const tris = [];
      for(const prim of mesh.primitives || []){
        if(prim.mode !== undefined && prim.mode !== 4){
          if(prim.mode === 0 || prim.mode === 1 || prim.mode === 2 || prim.mode === 3) continue;   // points and lines
          throw new Error('glTF: mesh ' + (mesh.name || n.mesh) + ' uses triangle strips or fans; export as triangle lists');
        }
        if(prim.attributes.POSITION === undefined) continue;
        const P = accessor(prim.attributes.POSITION);
        const idx = prim.indices !== undefined ? accessor(prim.indices).data : Float64Array.from({ length:P.count }, (_, i) => i);
        for(let i = 0; i + 2 < idx.length; i += 3) for(const q of [idx[i], idx[i+1], idx[i+2]]){
          const px = P.data[q*3], py = P.data[q*3+1], pz = P.data[q*3+2];
          tris.push(M[0]*px + M[4]*py + M[8]*pz + M[12], M[1]*px + M[5]*py + M[9]*pz + M[13], M[2]*px + M[6]*py + M[10]*pz + M[14]);
        }
      }
      if(tris.length) objects.push({ name:n.name || mesh.name || ('node ' + ni), tris:Float32Array.from(tris) });
    }
    for(const c of n.children || []) visit(c, M, seen);
  };
  const scene = json.scenes ? json.scenes[json.scene || 0] : null;
  const roots = scene ? scene.nodes : (json.nodes || []).map((_, i) => i);
  const seen = new Set();
  for(const r of roots || []) visit(r, I, seen);
  return { objects, vertices: (json.accessors || []).length };
}
// ---- points from a file (CSV, PLY) ---------------------------------------
// Real positions in place of a scatter: a table with x, y, z columns (CSV, TSV, or whitespace; a header row or not) or a PLY vertex list (ascii or binary).
// Returns positions and, when a column is named for it, the file's own id per row, kept as text because a connectome's ids exceed 2^53.
const PLY_SIZES = { char:1, int8:1, uchar:1, uint8:1, short:2, int16:2, ushort:2, uint16:2,
  int:4, int32:4, uint:4, uint32:4, float:4, float32:4, double:8, float64:8 };
function plyRead(dv, off, type, little){
  switch(type){
    case 'char': case 'int8': return dv.getInt8(off);
    case 'uchar': case 'uint8': return dv.getUint8(off);
    case 'short': case 'int16': return dv.getInt16(off, little);
    case 'ushort': case 'uint16': return dv.getUint16(off, little);
    case 'int': case 'int32': return dv.getInt32(off, little);
    case 'uint': case 'uint32': return dv.getUint32(off, little);
    case 'float': case 'float32': return dv.getFloat32(off, little);
    case 'double': case 'float64': return dv.getFloat64(off, little);
  }
  throw new Error('PLY: property type ' + type);
}
export function parsePly(bytes, idName){
  // the header is ascii up to end_header; the body follows in the stated format
  let headEnd = -1;
  for(let i = 0; i + 10 < bytes.length; i++)
    if(bytes[i] === 0x65 && new TextDecoder().decode(bytes.subarray(i, i + 10)) === 'end_header'){ headEnd = i + 10; break; }
  if(headEnd < 0) throw new Error('PLY: no end_header');
  while(headEnd < bytes.length && (bytes[headEnd] === 0x0d || bytes[headEnd] === 0x0a)) headEnd++;
  const head = new TextDecoder().decode(bytes.subarray(0, headEnd)).split(/\r?\n/);
  if(head[0].trim() !== 'ply') throw new Error('PLY: not a PLY file');
  let format = 'ascii', little = true;
  const elements = [];
  for(const line of head){
    const p = line.trim().split(/\s+/);
    if(p[0] === 'format'){ format = p[1]; little = p[1] !== 'binary_big_endian'; }
    else if(p[0] === 'element') elements.push({ name:p[1], count:+p[2], props:[] });
    else if(p[0] === 'property' && elements.length){
      if(p[1] === 'list') elements[elements.length - 1].props.push({ list:true, countType:p[2], type:p[3], name:p[4] });
      else elements[elements.length - 1].props.push({ type:p[1], name:p[2] });
    }
  }
  const vert = elements.find(e => e.name === 'vertex');
  if(!vert) throw new Error('PLY: no vertex element');
  const ix = vert.props.findIndex(q => q.name === 'x'), iy = vert.props.findIndex(q => q.name === 'y'), iz = vert.props.findIndex(q => q.name === 'z');
  if(ix < 0 || iy < 0 || iz < 0) throw new Error('PLY: the vertex element has no x, y, z');
  const iid = idName ? vert.props.findIndex(q => q.name === idName) : -1;
  if(idName && iid < 0) throw new Error('PLY: no vertex property named ' + idName + '; it has ' + vert.props.map(q => q.name).join(', '));
  const n = vert.count, pos = new Float32Array(n*3), ids = iid >= 0 ? new Array(n) : null;
  if(elements[0] !== vert) throw new Error('PLY: the vertex element must come first');
  if(format === 'ascii'){
    const body = new TextDecoder().decode(bytes.subarray(headEnd)).split(/\r?\n/);
    for(let i = 0, li = 0; i < n; i++, li++){
      while(li < body.length && !body[li].trim()) li++;
      const f = (body[li] || '').trim().split(/\s+/);
      pos[i*3] = +f[ix]; pos[i*3+1] = +f[iy]; pos[i*3+2] = +f[iz];
      if(ids) ids[i] = f[iid];
    }
  } else {
    const dv = new DataView(bytes.buffer, bytes.byteOffset + headEnd, bytes.byteLength - headEnd);
    let off = 0;
    for(let i = 0; i < n; i++){
      for(let q = 0; q < vert.props.length; q++){
        const pr = vert.props[q];
        if(pr.list){ const c = plyRead(dv, off, pr.countType, little); off += PLY_SIZES[pr.countType] + c*PLY_SIZES[pr.type]; continue; }
        const v = plyRead(dv, off, pr.type, little);
        if(q === ix) pos[i*3] = v; else if(q === iy) pos[i*3+1] = v; else if(q === iz) pos[i*3+2] = v;
        if(q === iid) ids[i] = String(v);
        off += PLY_SIZES[pr.type];
      }
    }
  }
  return { count:n, pos, ids, columns:vert.props.map(q => q.name), fields:{} };
}
export function parseTable(text, columns, idName){
  const lines = String(text).split(/\r?\n/).filter(l => l.trim() && !l.startsWith('#'));
  if(!lines.length) throw new Error('points: the file is empty');
  const first = lines[0];
  const sep = first.includes('\t') ? /\t/ : first.includes(',') ? /,/ : first.includes(';') ? /;/ : /\s+/;
  const split = l => l.trim().split(sep).map(s => s.trim().replace(/^"|"$/g, ''));
  const row0 = split(first);
  const numeric = s => s !== '' && Number.isFinite(+s);
  const header = row0.some(s => !numeric(s)) ? row0 : null;
  const rows = header ? lines.slice(1) : lines;
  // which columns: named (by header) or numbered (1-based), or x y z from the header, or the first three numeric columns of the first row
  const want = String(columns || '').trim().split(/[\s,]+/).filter(Boolean);
  const col = name => {
    if(/^\d+$/.test(name)) return +name - 1;
    if(!header) throw new Error('points: column "' + name + '" needs a header row; the file has none');
    const k = header.findIndex(h => h.toLowerCase() === name.toLowerCase());
    if(k < 0) throw new Error('points: no column named ' + name + '; the header has ' + header.join(', '));
    return k;
  };
  let cx, cy, cz;
  if(want.length >= 3){ cx = col(want[0]); cy = col(want[1]); cz = col(want[2]); }
  else if(header && ['x', 'y', 'z'].every(a => header.some(h => h.toLowerCase() === a))){ cx = col('x'); cy = col('y'); cz = col('z'); }
  else {
    const nums = split(rows[0] || '').map((s, i) => numeric(s) ? i : -1).filter(i => i >= 0);
    if(nums.length < 3) throw new Error('points: no three numeric columns; name them in the columns setting');
    [cx, cy, cz] = nums;
  }
  const cid = idName ? col(idName) : -1;
  const n = rows.length, pos = new Float32Array(n*3), ids = cid >= 0 ? new Array(n) : null;
  // every other column as text, by header name, for a tag column or a filter
  const names = header || row0.map((_, i) => String(i + 1));
  const fields = {}; for(const h of names) fields[h] = new Array(n);
  for(let i = 0; i < n; i++){
    const f = split(rows[i]);
    pos[i*3] = +f[cx]; pos[i*3+1] = +f[cy]; pos[i*3+2] = +f[cz];
    if(!Number.isFinite(pos[i*3]) || !Number.isFinite(pos[i*3+1]) || !Number.isFinite(pos[i*3+2]))
      throw new Error('points: row ' + (i + 1 + (header ? 1 : 0)) + ' has no number in a position column');
    if(ids) ids[i] = f[cid];
    for(let k = 0; k < names.length; k++) fields[names[k]][i] = f[k] === undefined ? '' : f[k];
  }
  return { count:n, pos, ids, columns:names, fields };
}
export function parsePointsFile(path, bytes, columns, idName){
  const ext = String(path).toLowerCase().replace(/^.*\./, '');
  if(ext === 'ply') return parsePly(bytes, idName);
  if(ext === 'csv' || ext === 'tsv' || ext === 'txt') return parseTable(new TextDecoder().decode(bytes), columns, idName);
  throw new Error('points: ' + path + ' is not .csv, .tsv, .txt or .ply');
}
// Population ids for a points file tagged by column: far above any node id or repeat product, one block of ids per node, one id per distinct value.
const POINTS_SRC_BASE = 1000000000, POINTS_SRC_STRIDE = 1000;
const pointsCache = new Map();
export function forgetPoints(path){ if(path === undefined) pointsCache.clear(); else pointsCache.delete(path); }

// ---- synapses from a connections table ----------------------------------------
// A connections table: one row per presynaptic id, postsynaptic id and synapse count, the shape a connectome release takes (FlyWire's pre_root_id, post_root_id, syn_count, nt_type; neuPrint's bodyId_pre, bodyId_post, weight).
// Columns are named on the node or found by these names.
// Ids are text and are matched against the ids a points file carried onto its cells.
// An optional transmitter column gives the sign.
const WIRE_PRE = ['pre_root_id', 'pre_pt_root_id', 'pre', 'pre_id', 'source', 'src', 'bodyid_pre', 'body_pre', 'from'];
const WIRE_POST = ['post_root_id', 'post_pt_root_id', 'post', 'post_id', 'target', 'dst', 'bodyid_post', 'body_post', 'to'];
const WIRE_COUNT = ['syn_count', 'count', 'weight', 'n', 'synapses', 'size'];
const WIRE_SIGN = ['nt_type', 'neurotransmitter', 'nt', 'sign', 'type'];
export function parseConnectionsTable(text, cols = {}){
  // Ids are interned: the table holds an index per row into one list of distinct ids, so five million rows cost forty megabytes rather than a string apiece.
  // Lines are scanned by hand for the same reason: no array per row.
  const src = String(text);
  const nl = src.indexOf('\n');
  const first = (nl < 0 ? src : src.slice(0, nl)).replace(/\r$/, '');
  const sepCh = first.includes('\t') ? '\t' : first.includes(',') ? ',' : first.includes(';') ? ';' : ' ';
  const fieldsOf = line => (sepCh === ' ' ? line.trim().split(/\s+/) : line.split(sepCh)).map(f => f.trim().replace(/^"|"$/g, ''));
  const header = fieldsOf(first).map(h => h.toLowerCase());
  const find = (want, names, what, required) => {
    if(want){ const k = header.indexOf(String(want).toLowerCase());
      if(k < 0) throw new Error('connections file: no column named ' + want + '; the header has ' + header.join(', '));
      return k; }
    for(const nm of names){ const k = header.indexOf(nm); if(k >= 0) return k; }
    if(required) throw new Error('connections file: no ' + what + ' column found; the header has ' + header.join(', ') + '. Name it on the node.');
    return -1;
  };
  const cp = find(cols.pre, WIRE_PRE, 'presynaptic id', true);
  const cq = find(cols.post, WIRE_POST, 'postsynaptic id', true);
  const cc = find(cols.count, WIRE_COUNT, 'count', false);
  const cs = find(cols.sign, WIRE_SIGN, 'transmitter', false);
  const need = Math.max(cp, cq, cc, cs);
  // count the rows first so the arrays are sized once
  let n = 0;
  for(let i = nl + 1; i >= 1 && i < src.length;){
    const e = src.indexOf('\n', i); const end = e < 0 ? src.length : e;
    if(end > i && src[i] !== '#' && src.slice(i, Math.min(end, i + 8)).trim()) n++;
    if(e < 0) break; i = e + 1;
  }
  if(!n) throw new Error('connections file: the table is empty');
  if(n > 1e7) throw new Error('connections file: ' + n.toLocaleString() + ' rows; more than ten million is past what one node reads');
  const idIndex = new Map(), ids = [];
  const intern = s => { let k = idIndex.get(s); if(k === undefined){ k = ids.length; idIndex.set(s, k); ids.push(s); } return k; };
  const pre = new Int32Array(n), post = new Int32Array(n), count = new Float32Array(n);
  const sign = cs >= 0 ? new Int8Array(n) : null;
  // the transmitter per row, interned, for a receptor node to map onto a channel
  const nt = cs >= 0 ? new Uint8Array(n) : null, ntNames = [], ntIndex = new Map();
  let r = 0;
  const field = (line, k) => {                       // field k of a line, by scanning
    if(sepCh === ' ') return fieldsOf(line)[k];
    let at = 0;
    for(let q = 0; q < k; q++){ at = line.indexOf(sepCh, at); if(at < 0) return ''; at++; }
    const e = line.indexOf(sepCh, at);
    const f = e < 0 ? line.slice(at) : line.slice(at, e);
    return f.trim().replace(/^"|"$/g, '');
  };
  for(let i = nl + 1; i >= 1 && i < src.length;){
    const e = src.indexOf('\n', i); const end = e < 0 ? src.length : e;
    if(end > i && src[i] !== '#'){
      let line = src.slice(i, end); if(line.endsWith('\r')) line = line.slice(0, -1);
      if(line.trim()){
        pre[r] = intern(field(line, cp)); post[r] = intern(field(line, cq));
        count[r] = cc >= 0 ? (+field(line, cc) || 0) : 1;
        // GABA, glutamate and histamine inhibit: glutamate through GluCl in the fly, histamine at the photoreceptor synapse through HisCl (Hardie 1989)
        if(sign){ const t = field(line, cs).toLowerCase(); sign[r] = /gaba|glut|hist|inh|^-/.test(t) ? -1 : 1;
          let k = ntIndex.get(t); if(k === undefined){ k = ntNames.length; if(k > 255) throw new Error('connections file: more than 256 transmitter names'); ntIndex.set(t, k); ntNames.push(t); } nt[r] = k; }
        r++;
      }
    }
    if(e < 0) break; i = e + 1;
  }
  return { rows:r, pre, post, count, sign, nt, ntNames, ids, columns:header };
}
const connectionsCache = new Map();
export function forgetConnections(path){ if(path === undefined) connectionsCache.clear(); else connectionsCache.delete(path); }

// Explicit synapses from every connections table entry on the stream, appended to the wired CSR: ids resolve against the cells' ids, the weight is the node's (times the count, or the count times over), the delay is distance over the node's velocity, the sign is the presynaptic cell's or the table's.
// Rows whose ids are not on the stream are counted and skipped.
// Runs after the pair-hash sweep and the cluster boost, before the proxy compensation.
// The receptor declaration a stream carries: channels 2 and up, each a time constant, a sign and the transmitters that use it (MODEL.md 2).
// Channels 0 and 1 are the checkpoint's E and I. Returns null for none.
export function receptorChannels(pts){
  const r = pts && pts.receptors;
  if(!r || !r.channels || !r.channels.length) return null;
  return r.channels;
}
export function wireConnectionsFile(pts, entries, st, byName){
  const n = pts.count, pos = pts.pos;
  const idMap = new Map();
  if(pts.ids) for(let i = 0; i < n; i++) if(pts.ids[i] != null) idMap.set(String(pts.ids[i]), i);
  const eP = [], eQ = [], eW = [], eD = [], eM = [];
  const report = { rows:0, matched:0, unmatched:0, self:0, synapses:0, clampedDelay:0 };
  for(const en of entries){
    const t = en.table; if(!t) continue;
    const pm = en.frozen ? 0 : parseRuleRef(en.rule, byName);
    if(pm > 31) throw new Error('connections file: rule ids above 31 do not fit the plasticity byte beside a channel');
    const chans = receptorChannels(pts);
    const chanOfNt = {};
    if(chans && t.nt) t.ntNames.forEach((name, q) => { const k = chans.findIndex(c => c.transmitters.includes(name)); if(k >= 0) chanOfNt[q] = k; });
    // the table's distinct ids resolve once; rows index them
    const local = new Int32Array(t.ids.length);
    for(let k = 0; k < t.ids.length; k++){ const q = idMap.get(String(t.ids[k])); local[k] = q === undefined ? -1 : q; }
    for(let r = 0; r < t.rows; r++){
      report.rows++;
      const i = local[t.pre[r]], j = local[t.post[r]];
      if(i < 0 || j < 0){ report.unmatched++; continue; }
      if(i === j){ report.self++; continue; }
      report.matched++;
      // with a receptor node on the stream the transmitter names the channel and the channel's sign is the synapse's sign; a transmitter the node does not name is refused rather than guessed
      let chan = 0, sgn = (en.signMode | 0) === 1 && t.sign ? t.sign[r] : NEURON_TYPES[pts.ntype[i]].sign;
      if(chans){
        if(!t.nt) throw new Error('connections file: a receptor node needs the table to carry a transmitter column');
        const k = chanOfNt[t.nt[r]];
        if(k === undefined) throw new Error('connections file: transmitter "' + t.ntNames[t.nt[r]] + '" has no channel on the receptor node; add it there (the node names ' +
          chans.map(c => c.name + ': ' + c.transmitters.join(' ')).join('; ') + ')');
        chan = k + 2; sgn = chans[k].sign;
      }
      let c = t.count[r] > 0 ? t.count[r] : 1;
      if(en.cap > 0 && c > en.cap) c = en.cap;
      const cm = en.compress | 0;
      if(cm === 1) c = Math.sqrt(c); else if(cm === 2) c = Math.log2(1 + c); else if(cm === 3) c = 1;
      const reps = (en.mode | 0) === 1 ? Math.max(1, Math.round(c)) : 1;
      // the inhibitory factor is the E to I balance of the table's synapses: one weight per synapse count unit, and inhibitory ones times this
      const wv = (sgn > 0 ? 1 : -(en.wInh === undefined ? 1 : en.wInh)) * Math.abs(en.weight) * ((en.mode | 0) === 1 ? 1 : c);
      const dx = pos[i*3]-pos[j*3], dy = pos[i*3+1]-pos[j*3+1], dz = pos[i*3+2]-pos[j*3+2];
      // clamped to the ring like every other delay, and counted like them
      const dr = Math.max(1, Math.round(Math.hypot(dx, dy, dz)/Math.max(1e-6, en.velocity)));
      if(dr > MAX_DELAY) report.clampedDelay += reps;
      const d = Math.min(MAX_DELAY, dr);
      for(let k = 0; k < reps; k++){ eP.push(i); eQ.push(j); eW.push(wv); eD.push(d); eM.push(pm | (chan << 5)); report.synapses++; }
    }
  }
  if(!eP.length) return { ...st, connectionsReport:report };
  let { preStart, post:postA, w:wA, delay:delA, pmask:pmA, m, KE, KI } = st;
  const added = eP.length, m2 = m + added;
  const ps = new Int32Array(n+1), cur2 = new Int32Array(n);
  for(let i=0;i<n;i++) ps[i+1] = preStart[i+1]-preStart[i];
  for(let s=0;s<added;s++) ps[eP[s]+1]++;
  for(let i=0;i<n;i++) ps[i+1] += ps[i];
  const post2 = new Int32Array(m2), w2 = new Float32Array(m2), del2 = new Uint8Array(m2), pmc = new Uint8Array(m2);
  for(let i=0;i<n;i++)
    for(let s=preStart[i]; s<preStart[i+1]; s++){
      const at = ps[i]+cur2[i]++;
      post2[at]=postA[s]; w2[at]=wA[s]; del2[at]=delA[s]; pmc[at]=pmA[s];
    }
  for(let s=0;s<added;s++){ const i=eP[s], at=ps[i]+cur2[i]++;
    post2[at]=eQ[s]; w2[at]=eW[s]; del2[at]=eD[s]; pmc[at]=eM[s];
    if(KE){ if(eW[s] > 0) KE[eQ[s]]++; else KI[eQ[s]]++; } }
  return { ...st, preStart:ps, post:post2, w:w2, delay:del2, pmask:pmc, m:m2, KE, KI, connectionsReport:report };
}

// One entry for any file the node reads, by extension.
export function parseMeshFile(path, bytes){
  const ext = String(path).toLowerCase().replace(/^.*\./, '');
  if(ext === 'glb') return parseGlb(bytes);
  if(ext === 'gltf') return parseGltfJson(JSON.parse(new TextDecoder().decode(bytes)), null);
  if(ext === 'obj') return parseObj(new TextDecoder().decode(bytes));
  throw new Error('mesh: ' + path + ' is not .obj, .glb or .gltf');
}

// The geometry: triangles in world micrometers after the node's scale, axis swap and centring, with the bounds every other reader of a region wants.
export function meshGeo(objects, p){
  let total = 0; for(const o of objects) total += o.tris.length;
  const tris = new Float32Array(total);
  let at = 0; for(const o of objects){ tris.set(o.tris, at); at += o.tris.length; }
  const s = +p.scale > 0 ? +p.scale : 1;
  const swap = (p.up | 0) === 1;
  for(let i = 0; i < tris.length; i += 3){
    let x = tris[i]*s, y = tris[i+1]*s, z = tris[i+2]*s;
    if(swap){ const t = y; y = z; z = -t; }   // a Z-up file into this Y-up space
    tris[i] = x; tris[i+1] = y; tris[i+2] = z;
  }
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for(let i = 0; i < tris.length; i += 3) for(let k = 0; k < 3; k++){
    if(tris[i+k] < min[k]) min[k] = tris[i+k]; if(tris[i+k] > max[k]) max[k] = tris[i+k]; }
  if(!tris.length) throw new Error('mesh: no triangles');
  const c = [(min[0]+max[0])/2, (min[1]+max[1])/2, (min[2]+max[2])/2];
  if((p.center | 0) === 1){
    for(let i = 0; i < tris.length; i += 3){ tris[i] -= c[0]; tris[i+1] -= c[1]; tris[i+2] -= c[2]; }
    for(let k = 0; k < 3; k++){ min[k] -= c[k]; max[k] -= c[k]; c[k] = 0; }
  }
  return { kind:'geo', shape:'mesh', tris, center:c,
    size:[max[0]-min[0], max[1]-min[1], max[2]-min[2]], min, max,
    triCount: tris.length/9, objects: objects.map(o => o.name) };
}

// Point in mesh by ray parity: a ray along +x crosses the surface an odd number of times from inside.
// Triangles are bucketed over (y, z) once per geometry so a query tests only the few whose projection could hold the point.
// Ties on shared edges are measure zero for scattered points.
function meshGrid(g){
  if(g._grid) return g._grid;
  const T = g.tris.length/9;
  const N = Math.max(4, Math.min(256, Math.ceil(Math.sqrt(T))));
  const y0 = g.min[1], z0 = g.min[2];
  const dy = Math.max(1e-9, (g.max[1] - y0)/N), dz = Math.max(1e-9, (g.max[2] - z0)/N);
  const cells = new Array(N*N);
  const t = g.tris;
  for(let i = 0; i < T; i++){
    const b = i*9;
    const ya = Math.min(t[b+1], t[b+4], t[b+7]), yb = Math.max(t[b+1], t[b+4], t[b+7]);
    const za = Math.min(t[b+2], t[b+5], t[b+8]), zb = Math.max(t[b+2], t[b+5], t[b+8]);
    const iy0 = Math.max(0, Math.floor((ya - y0)/dy)), iy1 = Math.min(N-1, Math.floor((yb - y0)/dy));
    const iz0 = Math.max(0, Math.floor((za - z0)/dz)), iz1 = Math.min(N-1, Math.floor((zb - z0)/dz));
    for(let iy = iy0; iy <= iy1; iy++) for(let iz = iz0; iz <= iz1; iz++){
      const k = iy*N + iz; (cells[k] || (cells[k] = [])).push(i); }
  }
  return g._grid = { N, y0, z0, dy, dz, cells };
}
export function meshInside(g, x, y, z){
  if(x < g.min[0] || x > g.max[0] || y < g.min[1] || y > g.max[1] || z < g.min[2] || z > g.max[2]) return false;
  // A ray through a shared edge is counted by both triangles, or by neither, and a lattice of query points lands on the diagonals of every quad face exactly; a fixed nudge of a tenth of a nanometer takes the ray off them.
  y += 1.3e-4; z += 0.7e-4;
  const G = meshGrid(g);
  const iy = Math.min(G.N-1, Math.max(0, Math.floor((y - G.y0)/G.dy)));
  const iz = Math.min(G.N-1, Math.max(0, Math.floor((z - G.z0)/G.dz)));
  const list = G.cells[iy*G.N + iz];
  if(!list) return false;
  const t = g.tris;
  let crossings = 0;
  for(const i of list){
    const b = i*9;
    const y1 = t[b+1], z1 = t[b+2], y2 = t[b+4], z2 = t[b+5], y3 = t[b+7], z3 = t[b+8];
    // point in the (y, z) projection, by the sign of the three edge tests
    const d1 = (y - y2)*(z1 - z2) - (y1 - y2)*(z - z2);
    const d2 = (y - y3)*(z2 - z3) - (y2 - y3)*(z - z3);
    const d3 = (y - y1)*(z3 - z1) - (y3 - y1)*(z - z1);
    const neg = (d1 < 0) || (d2 < 0) || (d3 < 0), pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    if(neg && pos) continue;
    // x on the triangle's plane at (y, z)
    const x1 = t[b], x2 = t[b+3], x3 = t[b+6];
    const ax = x2 - x1, ay = y2 - y1, az = z2 - z1, bx = x3 - x1, by = y3 - y1, bz = z3 - z1;
    const nx = ay*bz - az*by, ny = az*bx - ax*bz, nz = ax*by - ay*bx;
    if(nx === 0) continue;                 // the ray lies in the plane: no crossing
    const xp = x1 - (ny*(y - y1) + nz*(z - z1))/nx;
    if(xp > x) crossings++;
  }
  return (crossings & 1) === 1;
}

// Catmull-Rom polyline through the spline's 4 control points (65 samples)
export function splinePoly(g){
  const P = [g.p0, g.p1, g.p2, g.p3];
  const N = 64, out = new Float32Array((N+1)*3);
  for(let i=0;i<=N;i++){
    const u = i/N*3, j = Math.min(2, Math.floor(u)), t = u-j, t2 = t*t, t3 = t2*t;
    const A = P[Math.max(0,j-1)], B = P[j], C = P[j+1], D = P[Math.min(3,j+2)];
    for(let k=0;k<3;k++)
      out[i*3+k] = 0.5*(2*B[k] + (C[k]-A[k])*t
        + (2*A[k]-5*B[k]+4*C[k]-D[k])*t2 + (3*B[k]-A[k]-3*C[k]+D[k])*t3);
  }
  return out;
}

// Volume of a source geometry in cubic micrometers: exact for the analytic shapes, the bounds times a seeded acceptance fraction for a noise field or a mesh.
export function geoVolume(g){
  if(g.shape === 'sphere') return 4/3*Math.PI*g.radius*g.radius*g.radius;
  if(g.shape === 'ellipsoid') return 4/3*Math.PI*g.radii[0]*g.radii[1]*g.radii[2];
  if(g.shape === 'cylinder') return Math.PI*g.radius*g.radius*g.height;
  if(g.shape === 'torus') return 2*Math.PI*Math.PI*g.radius*g.thickness*g.thickness;
  if(g.shape === 'spline'){
    const poly = splinePoly(g); let L = 0;
    for(let s = 0; s + 1 < poly.length/3; s++)
      L += Math.hypot(poly[(s+1)*3]-poly[s*3], poly[(s+1)*3+1]-poly[s*3+1], poly[(s+1)*3+2]-poly[s*3+2]);
    return L*Math.PI*g.radius*g.radius;
  }
  const box = g.size[0]*g.size[1]*g.size[2];
  if(g.shape === 'noisefield' || g.shape === 'mesh'){
    const r = rng(104729), N = 4000; let hit = 0;
    for(let i = 0; i < N; i++){
      const x = (r()-0.5)*g.size[0], y = (r()-0.5)*g.size[1], z = (r()-0.5)*g.size[2];
      if(g.shape === 'noisefield' ? noiseField(g, x, y, z) : meshInside(g, x + g.center[0], y + g.center[1], z + g.center[2])) hit++;
    }
    return box*hit/N;
  }
  return box;
}
function scatterCompute(ins, p, node){
  const g = need(ins[0], 'geo', 'scatter needs a source geometry');
  // at density: the count follows the volume, so a bigger shape is more cells at the same spacing; by count (and every stream computed without the setting): the authored number
  const authored = (p.fill|0) === 1 ? Math.max(1, Math.round((+p.density || 0) * geoVolume(g) / 1e9)) : Math.floor(p.count);
  const n = Math.max(1, Math.round(authored * simResolution));
  const pos = new Float32Array(n*3), r = rng(p.seed*7919+1);
  const rot = g.rotate && g.rotate.some(v => v) ?
    rotMat(...g.rotate.map(d => d*Math.PI/180)) : null;
  let poly = null, cum = null;                       // spline: length-uniform sampling
  if(g.shape === 'spline'){
    poly = splinePoly(g);
    const segs = poly.length/3 - 1;
    cum = new Float32Array(segs+1);
    for(let s=0;s<segs;s++){
      const dx = poly[(s+1)*3]-poly[s*3], dy = poly[(s+1)*3+1]-poly[s*3+1],
            dz = poly[(s+1)*3+2]-poly[s*3+2];
      cum[s+1] = cum[s] + Math.hypot(dx, dy, dz);
    }
  }
  const sample = () => {                             // one world-space candidate
    let x, y, z;
    if(g.shape === 'sphere'){
      do { x=r()*2-1; y=r()*2-1; z=r()*2-1; } while(x*x+y*y+z*z > 1);
      x*=g.radius; y*=g.radius; z*=g.radius;
    } else if(g.shape === 'ellipsoid'){
      do { x=r()*2-1; y=r()*2-1; z=r()*2-1; } while(x*x+y*y+z*z > 1);
      x*=g.radii[0]; y*=g.radii[1]; z*=g.radii[2];
    } else if(g.shape === 'cylinder'){
      const th = 2*Math.PI*r(), rr = Math.sqrt(r())*g.radius;
      x = rr*Math.cos(th); z = rr*Math.sin(th); y = (r()-0.5)*g.height;
    } else if(g.shape === 'torus'){
      let u, ph, th, ww;                             // reject to stay volume-uniform
      do { th = 2*Math.PI*r(); ph = 2*Math.PI*r(); u = Math.sqrt(r())*g.thickness;
        ww = g.radius + u*Math.cos(ph);
      } while(r()*(g.radius + g.thickness) > ww);
      x = ww*Math.cos(th); z = ww*Math.sin(th); y = u*Math.sin(ph);
    } else if(g.shape === 'noisefield'){
      let tries = 0;
      do {
        x=(r()-0.5)*g.size[0]; y=(r()-0.5)*g.size[1]; z=(r()-0.5)*g.size[2];
      } while(!noiseField(g, x, y, z) && ++tries < 200);
    } else if(g.shape === 'mesh'){
      // uniform in the bounds, kept when inside the surface; the vertices are already in world space, so the center is added back below
      let tries = 0;
      do {
        x=(r()-0.5)*g.size[0]; y=(r()-0.5)*g.size[1]; z=(r()-0.5)*g.size[2];
      } while(!meshInside(g, x + g.center[0], y + g.center[1], z + g.center[2]) && ++tries < 4000);
    } else if(g.shape === 'spline'){
      const L = cum[cum.length-1]*r();               // arc-length uniform along the tube
      let s = 0; while(s < cum.length-2 && cum[s+1] < L) s++;
      const f = (L-cum[s])/Math.max(1e-6, cum[s+1]-cum[s]);
      let ox, oy, oz;
      do { ox=r()*2-1; oy=r()*2-1; oz=r()*2-1; } while(ox*ox+oy*oy+oz*oz > 1);
      return [ poly[s*3]  +(poly[(s+1)*3]  -poly[s*3])  *f + ox*g.radius,
               poly[s*3+1]+(poly[(s+1)*3+1]-poly[s*3+1])*f + oy*g.radius,
               poly[s*3+2]+(poly[(s+1)*3+2]-poly[s*3+2])*f + oz*g.radius ];
    } else {
      x=(r()-0.5)*g.size[0]; y=(r()-0.5)*g.size[1]; z=(r()-0.5)*g.size[2];
    }
    if(rot){
      const rx = rot[0]*x+rot[1]*y+rot[2]*z, ry = rot[3]*x+rot[4]*y+rot[5]*z,
            rz = rot[6]*x+rot[7]*y+rot[8]*z;
      x=rx; y=ry; z=rz;
    }
    return [x+g.center[0], y+g.center[1], z+g.center[2]];
  };
  // sampling strategies: minicolumns snap points onto a jittered hex lattice of strands along an axis (Mountcastle, 30-60 µm pitch); min spacing enforces soma exclusion zones (retinal-mosaic-style regularity)
  const mini = (p.pattern|0) === 1;
  const A = p.axis|0, B = (A+1)%3, C = (A+2)%3;
  const pitch = Math.max(5, p.pitch || 40), sj = Math.max(0, p.jitter || 0);
  const spacing = Math.max(0, p.spacing || 0), sp2 = spacing*spacing;
  const cells = spacing > 0 ? new Map() : null;
  const ck2 = gridKey;
  const farEnough = (x,y,z) => {
    const cx = Math.floor(x/spacing), cy = Math.floor(y/spacing), cz = Math.floor(z/spacing);
    for(let dx=-1;dx<=1;dx++) for(let dy=-1;dy<=1;dy++) for(let dz=-1;dz<=1;dz++){
      const bk = cells.get(ck2(cx+dx, cy+dy, cz+dz)); if(!bk) continue;
      for(const q of bk){
        const ax = x-pos[q*3], ay = y-pos[q*3+1], az = z-pos[q*3+2];
        if(ax*ax+ay*ay+az*az < sp2) return false;
      }
    }
    return true;
  };
  for(let i=0;i<n;i++){
    let placed = false;
    for(let attempt=0; attempt<40 && !placed; attempt++){
      const q = sample();
      if(mini){
        const row = Math.round(q[C]/(pitch*0.866));
        const col = Math.round(q[B]/pitch - (row & 1 ? 0.5 : 0));
        q[B] = (col + (row & 1 ? 0.5 : 0))*pitch + gaussRand(r)*sj*0.5;
        q[C] = row*pitch*0.866 + gaussRand(r)*sj*0.5;
        if(!pointInGeo(g, q[0], q[1], q[2])) continue;   // snapped out of the volume
      }
      if(cells && !farEnough(q[0], q[1], q[2])) continue;
      pos[i*3] = q[0]; pos[i*3+1] = q[1]; pos[i*3+2] = q[2];
      if(cells){
        const k = ck2(Math.floor(q[0]/spacing), Math.floor(q[1]/spacing), Math.floor(q[2]/spacing));
        let bk = cells.get(k); if(!bk){ bk = []; cells.set(k, bk); } bk.push(i);
      }
      placed = true;
    }
    if(!placed) throw new Error(
      `scatter: cannot place ${n.toLocaleString()} points (spacing or pattern too tight for the volume)`);
  }
  const ntype = new Uint8Array(n); ntype.fill(p.type|0);
  const src = new Int32Array(n); src.fill(node ? node.id : -1);
  const hv = p.hue === undefined || p.hue === '' ? -1 : (p.hue|0);   // -1 and absent: automatic
  return { kind:'points', count:n, pos, ntype, bias:new Float32Array(n), src,
    lidx:freshLidx(n), tags:{ [node ? node.id : -1]: p.tag || '' },
    ...(hv >= 0 && p.tag ? { hues:{ [p.tag]: hv } } : {}) };
}

function mergeCompute(ins){
  const pts = ins.filter(x => x && x.kind==='points');
  if(!pts.length) throw new Error('merge has no point inputs');
  // A population arriving twice is the same cells twice: a stream merged with a transformed copy of itself keeps every identity, so the pair hash makes the same decisions for both copies and every tag names both.
  // A copy of a population is a scatter of its own, with its own tag.
  // The same cell arriving twice, same identity at the same position, is kept once: one population projected into two others by two project nodes passes through both of them and meets itself at the merge.
  // The same identity at a different position is the mirrored-structure case above and is still refused.
  const where = new Map();                       // "src:lidx" -> first position index
  const keep = pts.map(p => new Uint8Array(p.count).fill(1));
  let dropped = 0;
  pts.forEach((p, pi) => {
    const src = p.src || new Int32Array(p.count).fill(-1);
    const lidx = p.lidx || freshLidx(p.count);
    for(let i=0;i<p.count;i++){
      if(src[i] < 0) continue;
      const key = src[i] + ':' + lidx[i];
      const first = where.get(key);
      if(first === undefined){ where.set(key, [pi, i]); continue; }
      const [qi, j] = first, q = pts[qi];
      if(q.pos[j*3] === p.pos[i*3] && q.pos[j*3+1] === p.pos[i*3+1] && q.pos[j*3+2] === p.pos[i*3+2]){
        keep[pi][i] = 0; dropped++; continue;
      }
      throw new Error('merge receives population ' +
        ((p.tags || {})[src[i]] || ('#' + src[i])) + ' twice at different positions; a copy of a population is a scatter of its own');
    }
  });
  const n = pts.reduce((s,p)=>s+p.count, 0) - dropped;
  const out = { kind:'points', count:n, pos:new Float32Array(n*3),
    ntype:new Uint8Array(n), bias:new Float32Array(n), src:new Int32Array(n),
    lidx:new Uint32Array(n), tags:{} };
  // a file's own ids, where a stream carries them; null for the others
  const anyIds = pts.some(p => p.ids);
  if(anyIds) out.ids = new Array(n).fill(null);
  let o=0;
  pts.forEach((p, pi) => {
    const src = p.src || new Int32Array(p.count).fill(-1);
    const lidx = p.lidx || freshLidx(p.count);
    const k = keep[pi];
    for(let i=0;i<p.count;i++){
      if(!k[i]) continue;
      out.pos[o*3] = p.pos[i*3]; out.pos[o*3+1] = p.pos[i*3+1]; out.pos[o*3+2] = p.pos[i*3+2];
      out.ntype[o] = p.ntype[i]; out.bias[o] = p.bias[i]; out.src[o] = src[i]; out.lidx[o] = lidx[i];
      if(anyIds && p.ids) out.ids[o] = p.ids[i];
      o++;
    }
    Object.assign(out.tags, p.tags || {});
    if(p.projs) out.projs = [...(out.projs || []), ...p.projs];
    if(p.rules) out.rules = [...(out.rules || []), ...p.rules];
    if(p.pools) out.pools = [...(out.pools || []), ...p.pools];
    if(p.connectionTables) out.connectionTables = [...(out.connectionTables || []), ...p.connectionTables];
    if(p.receptors){ if(out.receptors && out.receptors !== p.receptors) throw new Error('merge: two receptor declarations meet; a scene has one receptor node'); out.receptors = p.receptors; }
    if(p.cellTypes){ const have = out.cellTypes || []; out.cellTypes = [...have.filter(r => !p.cellTypes.some(q => q.key === r.key)), ...p.cellTypes]; }
    if(p.hues) out.hues = { ...(out.hues || {}), ...p.hues };
  });
  return out;
}

// population-pair table: lines "pre post probMul wMul plastic", '#' comments.
// The optional fifth column (default 1) gates plasticity per pair class: 0 freezes those synapses against STDP, homeostasis, and scaling (per-pathway plasticity gating).
// Keys: a scatter's population tag, the class keys E and I, or *.
// Most specific match wins (tag=2 points per side, class=1, *=0); among equal specificity the first line wins.
// Multipliers modulate the distance kernel and base weight per ordered pair class, so pair-hash additivity holds: editing one class's row leaves all other classes byte-identical.
// A population's color, derived from its tag rather than assigned.
// A tag is already the stable name for a population, so deriving from it means v1e is the same color in every scene and every session without anyone having to keep a table of assignments, and a scene made today matches one made last month.
// The scatter node can override it when a scene wants a particular arrangement; nothing has to.
//
// Hue only.
// The palette rule here is a single saturation, so hue carries the identity and everything stays visually of a piece.
// Assigned by position in the sorted list of tags, not by hashing the name.
// A hash gives a stable color per name but no control over the gaps between them: measured over one scene's eleven populations, the closest pair landed three degrees apart, which is no color difference at all.
// The golden angle only spreads values when it steps through consecutive indices, and a hash is not consecutive.
//
// Alphabetical order puts v.L2/3e next to v.L4e, so the layers of one area come out a full golden angle apart: the names most easily confused get the colors least easily confused.
// The cost is that adding a population reshuffles the scene, which is the right trade for a scene that is being built once and read many times.
const GOLDEN_ANGLE = 137.507764;
export function tagHues(tags){
  const names = [...new Set((Array.isArray(tags) ? tags : Object.values(tags || {}))
    .map(t => String(t || '')).filter(Boolean))].sort();
  const out = new Map();
  names.forEach((t, i) => out.set(t, Math.round((i*GOLDEN_ANGLE) % 360)));
  return out;
}
// Hues locked per scene: a tag keeps the hue it was first given (saved with the scene), and a tag seen for the first time takes its place in the golden-angle order, moved on by golden angles while it sits within twelve degrees of a hue already in use.
// Colors then hold across reloads and edits instead of following the set of tags.
// `saved` is mutated with the new assignments.
export function lockHues(saved, tags, overrides){
  const base = tagHues(tags);
  // a hue set on a node (scatter, population, points file) wins outright
  if(overrides) for(const t in overrides) if(overrides[t] >= 0) saved[t] = Math.round(overrides[t]) % 360;
  const used = new Set(Object.values(saved).map(h => Math.round(h)));
  const out = new Map();
  const far = h => [...used].every(u => Math.min(Math.abs(u - h), 360 - Math.abs(u - h)) >= 12);
  for(const [t, h0] of base){
    if(saved[t] !== undefined){ out.set(t, saved[t]); continue; }
    let h = h0, tries = 0;
    while(!far(h) && tries++ < 64) h = Math.round((h + GOLDEN_ANGLE) % 360);
    saved[t] = h; used.add(h); out.set(t, h);
  }
  return out;
}
export const hueCss = h => `hsl(${h},75%,62%)`;
export function hueRgb(h){                       // 0..1 triple, for three.js
  const s = 0.75, l = 0.62;
  const c = (1 - Math.abs(2*l - 1))*s, x = c*(1 - Math.abs((h/60) % 2 - 1)), m = l - c/2;
  const seg = [[c,x,0],[x,c,0],[0,c,x],[0,x,c],[x,0,c],[c,0,x]][Math.floor(h/60) % 6];
  return [seg[0] + m, seg[1] + m, seg[2] + m];
}

// The table the engines index with the per-synapse rule byte.
// One flat Float32Array shared by all three, in one field order, so a rule means the same thing whichever engine is running and a parity test compares like with like.
// Row 0 is never read (0 means frozen and every loop skips it); row 1 is the checkpoint's own values; rows 2 up are the declared rules.
//
// Only amplitudes and bounds are here. tauS and tauY are absent because the traces they govern, Kpre and Kslow, are per neuron rather than per synapse: two rules wanting different windows on synapses onto the same cell would need a separate trace each, which is per-neuron state multiplied by the rule count rather than a wider table. iRho, rhoMode, calS, scale and sEta are absent for the same reason from the other side, being set points and normalizations belonging to the postsynaptic neuron.
// The short-term terms are per presynaptic neuron.
//
// The clearest result in the literature for per-pathway rules is that timing-dependent LTD at L4 to L2/3 has a considerably longer window than at L2/3 to L2/3 (Larsen et al. 2014), and a window is a time constant, so the amplitudes differ per pathway here and the windows do not.
export const RULE_FIELDS = ['aP', 'aM', 'wmax', 'wdep', 'trip', 'het', 'tin',
  'iEta', 'cons', 'consW', 'consP'];
export const RULE_STRIDE = RULE_FIELDS.length;
// Everything an engine reads from a computed network, in one place, so the page, the trainer and the battery cannot drift from each other or from ENGINE.md.
// Spread into the init and tune messages.
export function engineConfig(net){
  return { protocol:PROTOCOL, syn:net.syn, tauE:net.tauE, tauI:net.tauI, psc:net.psc|0,
    // conductance mode (syn 2): the reversal potentials of E, I and each receptor channel (a channel without one takes 0 for + and -70 for -)
    eRevE:Number.isFinite(+net.eRevE) ? +net.eRevE : 0, eRevI:Number.isFinite(+net.eRevI) ? +net.eRevI : -70,
    chanErev:(receptorChannels(net) || []).map(c => Number.isFinite(c.erev) ? c.erev : (c.sign > 0 ? 0 : -70)),
    protocols:net.protocols || [],
    plast:net.plast, aP:net.aP, aM:net.aM, tauS:net.tauS, tauM:net.tauM,
    wmax:net.wmax, wdep:net.wdep, iEta:net.iEta, iRho:net.iRho,
    rhoMode:net.rhoMode, calS:net.calS, scale:net.scale, sEta:net.sEta,
    trip:net.trip, tauY:net.tauY, het:net.het, tin:net.tin,
    cons:net.cons, consW:net.consW, consP:net.consP, commit:net.commit|0,
    tauCons:net.tauCons > 0 ? net.tauCons*60000 : 0,   // sim-minutes in the panel, ms in the engine
    consStep:net.consStep > 0 ? net.consStep*1000 : 0,  // sim-seconds in the panel, ms in the engine
    stp:net.stp, stpU:net.stpU, stpNorm:net.stpNorm|0, stpOrder:net.stpOrder|0,
    stpTauD:net.stpTauD, stpTauF:net.stpTauF, refrac:net.refrac,
    vmin:net.vmin === undefined ? -90 : +net.vmin,
    // receptor channels beyond E and I: their time constants, in order (channel 2 first); the wiring put each synapse's channel in the top three bits of its plasticity byte
    chanTau:(receptorChannels(net) || []).map(c => c.tau),
    // feedback inhibition: which pool each cell is in and each pool's kick per spike, wired by wirePools; absent when the graph has no pool node
    pool:net.pool, poolK:net.poolK,
    // the engine settings node's entries, passed to the engine as they were entered (ENGINE.md section 2); the stock engines ignore them
    custom:net.custom || {},
    ...buildRuleTable(net, net.rules), ruleStride:RULE_STRIDE };
}

// Feedback inhibition, resolved at wiring time.
// Each entry a pool node declared becomes one pool per population it names (a pattern with * names one per matching tag; E, I or blank name one pool of that class or of every cell).
// Every spike in a pool delivers gain * 1000 / N to every cell of the pool one millisecond later, on the inhibitory path, so what a cell receives per millisecond is gain times the pool's mean rate in Hz whatever its size.
// This is a lumped inhibitory interneuron with every cell of the population on both its input and its output: the mushroom body's APL neuron (Lin et al. 2014, Nat Neurosci 17:559) and the dentate's basket cell feedback.
// A cell belongs to at most one pool; a second claim is refused rather than resolved, since the two gains would have to be summed or one dropped and neither is what the scene said.
// Returns null when nothing declares a pool.
export function wirePools(pts){
  const list = pts.pools || [];
  if(!list.length) return null;
  const n = pts.count, tags = pts.tags || {}, src = pts.src, nt = pts.ntype;
  const all = [...new Set(Object.values(tags))];
  const pool = new Uint16Array(n);
  const K = [0], info = [];
  for(const pl of list){
    const tag = String(pl.tag || '').trim();
    const gain = Number.isFinite(+pl.gain) ? +pl.gain : 0;
    // the pools this entry makes, each as a membership test on a cell
    const groups = [];
    if(!tag) groups.push(['every cell', () => true]);
    else if(tag === 'E') groups.push(['E', i => NEURON_TYPES[nt[i]].sign > 0]);
    else if(tag === 'I') groups.push(['I', i => NEURON_TYPES[nt[i]].sign < 0]);
    else {
      for(const t of all){
        if(tagMatch(t, tag) === null) continue;
        const ids = new Set(Object.keys(tags).filter(id => sameTag(tags[id], t)).map(Number));
        groups.push([t, i => ids.has(src[i])]);
      }
      if(!groups.length) throw new Error('feedback inhibition: no population matches "' + tag + '"');
    }
    for(const [name, test] of groups){
      const g = K.length;
      if(g > 65535) throw new Error('feedback inhibition: more than 65535 pools');
      let count = 0;
      for(let i=0;i<n;i++){
        if(!test(i)) continue;
        if(pool[i]) throw new Error('feedback inhibition: population ' + name + ' is already in the pool ' +
          info[pool[i]-1].name + '; a cell belongs to one pool');
        pool[i] = g; count++;
      }
      if(!count) throw new Error('feedback inhibition: the pool ' + name + ' has no cells');
      K.push(gain*1000/count);
      info.push({ name, count, gain, node:pl.node });
    }
  }
  return { pool, poolK:Float32Array.from(K), poolInfo:info };
}

export function buildRuleTable(base, rules){
  const list = rules || [];
  const rows = 2 + list.length;
  // The engine contract carries 16 rows (ENGINE.md, set by the WebGPU uniform), so 14 named rules beside frozen and the checkpoint's own.
  if(rows > 16) throw new Error('plasticity: at most 14 named rules (the engines carry a 16-row rule table: frozen, the checkpoint, and 14 declared rules)');
  const t = new Float32Array(rows*RULE_STRIDE);
  const put = (row, src) => {
    RULE_FIELDS.forEach((f, k) => {
      // a rule that does not name a field takes the checkpoint's value, so a node set up to change one amplitude changes exactly that
      const v = src && src[f] !== undefined && src[f] !== null ? +src[f] : +base[f];
      t[row*RULE_STRIDE + k] = Number.isFinite(v) ? v : 0;
    });
    // consW 0 (the default on both nodes) means a tenth of the row's wmax, the published ratio.
    // The engines apply that default only to the value outside the table, which no page-computed network reaches, so the table has to carry the resolved number or a 0 clamps every reference to [0, 0] in all three engines.
    const o = row*RULE_STRIDE, cw = RULE_FIELDS.indexOf('consW');
    if(!(t[o + cw] > 0)) t[o + cw] = 0.1*t[o + RULE_FIELDS.indexOf('wmax')];
  };
  put(1, null);                                  // the checkpoint's own set
  list.forEach((r, i) => put(2 + i, r));
  return { ruleTable:t, ruleCount:rows };
}

// Rule ids are what the per-synapse byte carries: 0 frozen, 1 the checkpoint's own settings, and declared rules from 2 up.
// Names resolve here so a typo fails at compute time with the list of what exists, rather than silently falling through to the wildcard row the way an unknown population tag still does.
export function ruleIds(rules){
  const byName = new Map();
  (rules || []).forEach((r, i) => {
    if(byName.has(r.name))
      throw new Error('two plasticity nodes both declare a rule named "' + r.name
        + '". Rules are referred to by name, so the names have to be distinct.');
    byName.set(r.name, i + 2);
  });
  return byName;
}
// Absent means the default rule; 0, frozen, off or no freezes the pair; 1, default, on or yes is the default rule; anything else names a plasticity node's rule.
export function parseRuleRef(tok, byName){
  if(tok === undefined || tok === null || tok === '') return 1;
  const raw = String(tok).trim(), s = raw.toLowerCase();
  if(s === '0' || s === 'frozen' || s === 'off' || s === 'no') return 0;
  if(s === '1' || s === 'default' || s === 'on' || s === 'yes') return 1;
  const id = byName && byName.get(raw);
  if(id === undefined){
    const known = byName && byName.size ? [...byName.keys()].join(', ')
      : '(no plasticity nodes in this graph)';
    throw new Error('pair table: no plasticity rule named "' + raw
      + '". Declared rules: ' + known + '. Use frozen or default for the built-in cases.');
  }
  return id;
}
export function buildPairLUT(pts, tableText, rules){
  const n = pts.count, src = pts.src, tags = pts.tags || {};
  const groups = [];                        // distinct src ids -> group index
  const gOf = new Map();
  const gIdx = new Int32Array(n);
  for(let i=0;i<n;i++){
    let g = gOf.get(src[i]);
    if(g === undefined){ g = groups.length; gOf.set(src[i], g); groups.push(src[i]); }
    gIdx[i] = g;
  }
  const G = groups.length;
  const probMul = new Float32Array(G*G).fill(1);
  const wMul = new Float32Array(G*G).fill(1);
  const plMul = new Uint8Array(G*G).fill(1);
  const byName = ruleIds(rules);
  // A name that resolves to nothing falls back to the default rule and is reported, rather than failing the wiring.
  // Bypassing a plasticity node is the ordinary way to ask what a rule was doing, and every row naming it would otherwise take the whole scene down: bypass would be unusable on exactly the node it is most wanted on.
  // The pathway still says it learns, so it learns by the checkpoint's settings, and the panel colors the name red so this is visible rather than silent.
  const warn = [];
  const refOf = tok => {
    try { return parseRuleRef(tok, byName); }
    catch(e){ warn.push(String(tok).trim()); return 1; }
  };
  const rows = [];
  for(const line of String(tableText || '').split('\n')){
    const t = line.split('#')[0].trim();
    if(!t) continue;
    const f = t.split(/\s+/);
    if(f.length < 3) continue;
    rows.push({ pre:f[0], post:f[1], pm:+f[2], wm:f.length > 3 ? +f[3] : 1,
      pl:refOf(f[4]) });
  }
  // A plasticity node that names both ends scopes itself, and does it without touching connection density.
  // A winning table row carries probability and weight as well as the rule, so overriding plasticity with a row changes connection density; these resolve against the rule alone and never against pm or wm.
  const scoped = (rules || [])
    .map((r, i) => ({ pre:r.from, post:r.to, pl:i + 2 }))
    .filter(r => r.pre && r.post);
  if(rows.length || scoped.length){
    const groupTag = groups.map(id => tags[id] || '');
    const groupSign = groups.map((id, g) => {
      for(let i=0;i<n;i++) if(gIdx[i] === g)
        return NEURON_TYPES[pts.ntype[i]].sign > 0 ? 'E' : 'I';
      return 'E';
    });
    // a key ending in * matches every population it is a prefix of, which is how one row addresses every copy a repeat node made (RSexc* covers RSexc1, RSexc2, ...); it sits between an exact tag and a class key
    const score = (key, g) => key !== '' && sameTag(key, groupTag[g]) ? 2
      : key.length > 1 && key.endsWith('*') &&
        String(groupTag[g] || '').toLowerCase().startsWith(key.slice(0, -1).toLowerCase()) ? 1.5
      : key === groupSign[g] ? 1 : key === '*' ? 0 : -1;
    for(let a=0;a<G;a++) for(let b=0;b<G;b++){
      let best = -1, pm = 1, wm = 1, pl = 1;
      for(const r of rows){
        const sa = score(r.pre, a), sb = score(r.post, b);
        if(sa < 0 || sb < 0) continue;
        if(sa + sb > best){ best = sa + sb; pm = r.pm; wm = r.wm; pl = r.pl; }
      }
      // Scoped rules resolve after the table and win a tie, because dropping a node onto a pathway is the more deliberate act than a row in a generated table, and because a rule that loses silently to a row nobody can see is the failure this whole change exists to end.
      for(const r of scoped){
        const sa = score(r.pre, a), sb = score(r.post, b);
        if(sa < 0 || sb < 0) continue;
        if(sa + sb >= best){ best = sa + sb; pl = r.pl; }
      }
      probMul[a*G+b] = pm; wMul[a*G+b] = wm; plMul[a*G+b] = pl;
    }
  }
  return { gIdx, G, probMul, wMul, plMul,
    ruleWarn:[...new Set(warn)] };
}

// opts.range = [i0, i1) restricts the wiring to a slice of presynaptic neurons and returns the raw CSR for that slice instead of a finished net.
// Every decision in here depends only on the two neurons' stable identities and the seed, never on iteration order or on any other pair, so a slice produces exactly the synapses the whole-range wiring would produce for those same presynaptic neurons, in the same order.
// That is what makes splitting the sweep across workers safe rather than merely fast: the pair-hash property the graph editor already relies on for additive edits is the same property that makes this data-parallel.
export function wireConnect(pts, p, onProgress, opts){   // runs inside computeworker.js
  // A scene's own cell type rows were registered by the cell type compute on the thread that computed it; this wiring may run on a worker with its own copy of the type table, and a row it has never seen reads as no row at all (every sign, every weight below fails on it).
  // The stream carries the rows for that reason; register them before anything reads a type.
  for(const r of pts.cellTypes || []) registerCellType(r);
  const { count:n, pos } = pts;
  const RANGE = opts && opts.range;
  const I0 = RANGE ? RANGE[0] : 0, I1 = RANGE ? RANGE[1] : n;
  const src = pts.src, lidx = pts.lidx || freshLidx(n);
  const seed = p.seed | 0;
  const T = buildPairLUT(pts, p.table, pts.rules);
  // A projection writes the gate byte directly rather than going through the table, so it resolves its own rule name the same way a table row does.
  // Frozen still wins over a named rule: a projection that says frozen means it, and reading the name first would let a stale field silently unfreeze a tract.
  const prNames = ruleIds(pts.rules);
  const prWarn = [];
  const prRule = pr => { if(pr.frozen) return 0;
    try { return parseRuleRef(pr.rule, prNames); }
    catch(e){ prWarn.push(String(pr.rule).trim()); return 1; } };
  const R = p.radius, R2 = R*R, s2 = 2*p.sigma*p.sigma;
  const grid = new Map();
  const ck = gridKey;
  const cell = i => [Math.floor(pos[i*3]/R), Math.floor(pos[i*3+1]/R), Math.floor(pos[i*3+2]/R)];
  for(let i=0;i<n;i++){ const k = ck(...cell(i));
    let a = grid.get(k); if(!a){ a=[]; grid.set(k,a); } a.push(i); }
  const MAX_SYN = Math.max(1e6, p.__maxSyn || 5e7);
  const logn = (p.wdist|0) === 1, sg = p.wsigma || 1;
  // proxy downscaling (van Albada, Helias & Diesmann 2015; NEST microcircuit K-scaling): the incoming points represent `density`% of real density, so weights scale by 1/sqrt(density) to preserve input variance, and a per-neuron DC (from realized in-degrees and target rates) restores the mean.
  // Rates and regime are preserved; correlations shrink with size (a known limit).
  // Three independent things reduce realized in-degree and all three want the same correction, because the compensation depends on how far the in-degree falls short of full scale and not on the reason it does.
  //   density  advanced baseline, a declared fraction of real density
  //   __res    the resolution slider, which places proxy neurons
  //   kscale   a smaller cortical footprint at real density: with a kernel
  //            that spans the box, in-degree is proportional to the number
  //            of source neurons in range and therefore to the area placed,
  //            so a quarter of the surface realizes a quarter of the
  //            in-degree while every neuron, distance and delay stays real
  // Territory reduction is strictly better than the resolution slider for the same saving: both lose in-degree, only the slider also replaces real neurons with proxies.
  const declared = (p.density === undefined ? 100 : p.density)/100;   // advanced baseline
  const kscale = (p.kscale === undefined ? 100 : p.kscale)/100;       // footprint fraction
  const res = Math.max(0.01, Math.min(1,
    declared * kscale * (p.__res === undefined ? 1 : p.__res)));
  const wScale = 1/Math.sqrt(res);
  const KE = res < 1 ? new Int32Array(n) : null, KI = res < 1 ? new Int32Array(n) : null;

  // Two passes over the same deterministic decisions: the first counts each neuron's out-degree, the second fills exactly sized CSR arrays.
  // Counting first costs a second pass over the kernel and holds peak memory at the 13 bytes per synapse the result needs.
  const deg = new Int32Array(n);
  let preStart = null, cursor = null;
  let postA = null, wA = null, delA = null, pmA = null;
  let m = 0, counting = true;
  // Delays over MAX_DELAY are clamped, and the count is reported with the wiring rather than lost: a pair further apart than the ring buffer reaches gets the wrong delay, and wiring that does that silently is the failure mode this project refuses everywhere else.
  let clampedDelay = 0;
  const qd = ms => { const r = Math.round(ms);
    if(r > MAX_DELAY) clampedDelay++;
    return r < 1 ? 1 : r > MAX_DELAY ? MAX_DELAY : r; };

  const emit = (i, j, wv, dv, pv, tgtForK, wForK) => {
    if(counting){
      deg[i]++;
      if(++m > MAX_SYN) throw new Error(
        `> ${(MAX_SYN/1e6).toFixed(0)}M synapses: lower radius, probability, or resolution`);
      if(KE){ if(wForK > 0) KE[tgtForK]++; else KI[tgtForK]++; }
      return;
    }
    const at = preStart[i] + cursor[i]++;
    postA[at] = j; wA[at] = wv; delA[at] = dv; pmA[at] = pv;
  };

  // one sweep of every connection rule; run twice with `counting` toggled
  const sweep = () => {
    for(let i=I0;i<I1;i++){
      if(onProgress && (i & 8191) === 0)
        // pass 2 knows the total from pass 1, so the count travels with the fraction: on a big scene the wait is long enough that knowing how many synapses are coming is the difference between watching a bar and wondering whether it has hung
        onProgress((counting ? 0 : 0.5) + (i-I0)/Math.max(1, I1-I0)*0.5,
          counting ? 0 : m);
      const [cx,cy,cz] = cell(i);
      const sign = NEURON_TYPES[pts.ntype[i]].sign;
      const wbase = sign > 0 ? p.wExc : p.wInh;
      const tRow = T.gIdx[i]*T.G;
      for(let dx=-1;dx<=1;dx++) for(let dy=-1;dy<=1;dy++) for(let dz=-1;dz<=1;dz++){
        const bucket = grid.get(ck(cx+dx, cy+dy, cz+dz)); if(!bucket) continue;
        for(const j of bucket){
          if(j===i) continue;
          const ax=pos[i*3]-pos[j*3], ay=pos[i*3+1]-pos[j*3+1], az=pos[i*3+2]-pos[j*3+2];
          const d2 = ax*ax+ay*ay+az*az;
          if(d2 > R2) continue;
          // pair-hash: existence and weight depend only on the two identities and the seed, never on iteration order or unrelated pairs
          const tCell = tRow + T.gIdx[j];
          if(pairHash(src[i], lidx[i], src[j], lidx[j], seed)
              < p.prob * T.probMul[tCell] * Math.exp(-d2/s2)){
            if(counting){ emit(i, j, 0, 0, 0, j, wbase); continue; }
            // lognormal (Song et al 2005): mean-preserving so total drive matches wbase
            const wv = (logn
              ? wbase*Math.exp(sg*pairGauss(src[i], lidx[i], src[j], lidx[j], seed) - sg*sg/2)
              : wbase*(0.5 + pairHash(src[i], lidx[i], src[j], lidx[j], seed + 101)))
              *wScale*T.wMul[tCell];
            emit(i, j, wv,
              qd(Math.sqrt(d2)/p.velocity),
              T.plMul[tCell], j, wbase);
          }
        }
      }
    }
    // topographic projections (tract wiring between tagged populations, from project nodes upstream): each source neuron's lateral coordinates map onto the target population's frame, the Gaussian kernel applies in that mapped space, and the delay comes from tract length over conduction velocity instead of straight-line proximity.
    // Pair-hash keyed on its own seed stream, so projections stay additive with the local kernel; the connect table does not apply to them (they are explicit wiring).
    for(const pr0 of (pts.projs || [])) for(const pr of expandProjection(pr0, pts.tags)){
      const tags = pts.tags || {};
      const A = [], B = [];
      for(let i=0;i<n;i++){
        const t = tags[src[i]] || '';
        if(tagMatch(t, pr.from) !== null) A.push(i);
        if(tagMatch(t, pr.to) !== null) B.push(i);
      }
      if(!A.length) throw new Error(`projection: no neurons tagged "${pr.from}"`);
      if(!B.length) throw new Error(`projection: no neurons tagged "${pr.to}"`);
      const lat = ax => ax===0 ? [1,2] : ax===2 ? [0,1] : [0,2];
      const [a1,a2] = lat(pr.axisFrom|0), [b1,b2] = lat(pr.axisTo|0);
      const bb = (set, x1, x2) => {
        let u1=Infinity, U1=-Infinity, u2=Infinity, U2=-Infinity;
        for(const i of set){ const u=pos[i*3+x1], v=pos[i*3+x2];
          if(u<u1)u1=u; if(u>U1)U1=u; if(v<u2)u2=v; if(v>U2)U2=v; }
        return [u1, Math.max(1e-6, U1-u1), u2, Math.max(1e-6, U2-u2)];
      };
      const [am1, ar1, am2, ar2] = bb(A, a1, a2);
      const [bm1, br1, bm2, br2] = bb(B, b1, b2);
      const cen = set => { let x=0,y=0,z=0;
        for(const i of set){ x+=pos[i*3]; y+=pos[i*3+1]; z+=pos[i*3+2]; }
        return [x/set.length, y/set.length, z/set.length]; };
      const ca = cen(A), cb = cen(B);
      const straight = Math.hypot(ca[0]-cb[0], ca[1]-cb[1], ca[2]-cb[2]);
      const vel = Math.max(1, pr.velocity);
      const baseMs = (pr.tract > 0 ? pr.tract : straight) / vel;
      const pseed = seed + ((pr.seed|0)+1)*7919;
      // dispersed mode: no map at all, each source contacts a seeded random subset of the target population.
      // Primary sensory pathways preserve topography, but higher-order and association areas receive convergent mixed input, and binding needs any source to be able to meet any other on a shared target neuron rather than only the one its coordinates happen to land on.
      if(pr.dispersed){
        // each source reaches fanout cells of the target on average
        const pp = Math.min(1, (pr.fanout || 50) / B.length);
        for(const i of A){
          if(i < I0 || i >= I1) continue;      // this slice's sources only
          const sign = NEURON_TYPES[pts.ntype[i]].sign;
          const wbase = sign > 0 ? pr.weight : -pr.weight;
          for(const j of B){
            if(j === i) continue;
            if(pairHash(src[i], lidx[i], src[j], lidx[j], pseed) >= pp) continue;
            if(counting){ emit(i, j, 0, 0, 0, j, wbase); continue; }
            const wv = (logn
              ? wbase*Math.exp(sg*pairGauss(src[i], lidx[i], src[j], lidx[j], pseed) - sg*sg/2)
              : wbase*(0.5 + pairHash(src[i], lidx[i], src[j], lidx[j], pseed + 101)))
              *wScale;
            emit(i, j, wv, qd(baseMs),
              prRule(pr), j, wbase);
          }
        }
        continue;
      }
      // matched: the map is the population order, the kernel is proximity's.
      // Each source cell has a copy in the target, the cell at the same place in the target's order (two populations laid down by one repeat node are copies in the same order), and it reaches the target cells around that copy with the spread and probability on the distance from the copy.
      // A pattern in the lower bulb arrives in the upper bulb as itself, blurred by the spread.
      // Topographic maps by one axis and folds a volume onto a line; proximity reaches only the facing surfaces; a single contact per cell cannot fire anything (all three measured 2026-09-06).
      // Order is stable identity order, so the map is as additive as everything else in the wiring.
      if(pr.matched){
        const byOrder = set => set.slice().sort((a, b) => (lidx[a] - lidx[b]) || (src[a] - src[b]));
        const As = byOrder(A), Bs = byOrder(B);
        const ms2 = 2*pr.sigma*pr.sigma, mreach = 3*pr.sigma, mreach2 = mreach*mreach;
        const bgM = new Map();
        const bkM = (x, y, z) => gridKey(Math.floor(x/mreach), Math.floor(y/mreach), Math.floor(z/mreach));
        for(const j of B){ const k = bkM(pos[j*3], pos[j*3+1], pos[j*3+2]);
          let a = bgM.get(k); if(!a){ a=[]; bgM.set(k,a); } a.push(j); }
        for(let r = 0; r < As.length; r++){
          const i = As[r];
          if(i < I0 || i >= I1) continue;
          const sign = NEURON_TYPES[pts.ntype[i]].sign;
          const wbase = sign > 0 ? pr.weight : -pr.weight;
          const c = Bs[As.length > 1 ? Math.round(r*(Bs.length - 1)/(As.length - 1)) : 0];
          const x = pos[c*3], y = pos[c*3+1], z = pos[c*3+2];
          const gx = Math.floor(x/mreach), gy = Math.floor(y/mreach), gz = Math.floor(z/mreach);
          for(let dx=-1;dx<=1;dx++) for(let dy=-1;dy<=1;dy++) for(let dz=-1;dz<=1;dz++){
            const bucket = bgM.get(gridKey(gx+dx, gy+dy, gz+dz));
            if(!bucket) continue;
            for(const j of bucket){
              if(j === i) continue;
              const du = pos[j*3]-x, dv = pos[j*3+1]-y, dw = pos[j*3+2]-z;
              const d2 = du*du + dv*dv + dw*dw;
              if(d2 > mreach2) continue;
              if(pairHash(src[i], lidx[i], src[j], lidx[j], pseed) < pr.prob * Math.exp(-d2/ms2)){
                if(counting){ emit(i, j, 0, 0, 0, j, wbase); continue; }
                const wv = (logn
                  ? wbase*Math.exp(sg*pairGauss(src[i], lidx[i], src[j], lidx[j], pseed) - sg*sg/2)
                  : wbase*(0.5 + pairHash(src[i], lidx[i], src[j], lidx[j], pseed + 101)))
                  *wScale;
                emit(i, j, wv, qd(baseMs), prRule(pr), j, wbase);
              }
            }
          }
        }
        continue;
      }
      const ps2 = 2*pr.sigma*pr.sigma, reach = 3*pr.sigma, reach2 = reach*reach;
      // proximity: the kernel on the actual distance between the two cells, in three dimensions, with no map.
      // A source reaches the part of the target nearest to it, so a ring of sources around a core lands in sectors, which is the geometry the sector readout asks about.
      if(pr.proximity){
        const bg3 = new Map();
        const bk3 = (x, y, z) => gridKey(Math.floor(x/reach), Math.floor(y/reach), Math.floor(z/reach));
        for(const j of B){ const k = bk3(pos[j*3], pos[j*3+1], pos[j*3+2]);
          let a = bg3.get(k); if(!a){ a=[]; bg3.set(k,a); } a.push(j); }
        for(const i of A){
          if(i < I0 || i >= I1) continue;
          const x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
          const sign = NEURON_TYPES[pts.ntype[i]].sign;
          const wbase = sign > 0 ? pr.weight : -pr.weight;
          const gx = Math.floor(x/reach), gy = Math.floor(y/reach), gz = Math.floor(z/reach);
          for(let dx=-1;dx<=1;dx++) for(let dy=-1;dy<=1;dy++) for(let dz=-1;dz<=1;dz++){
            const bucket = bg3.get(gridKey(gx+dx, gy+dy, gz+dz));
            if(!bucket) continue;
            for(const j of bucket){
              if(j === i) continue;
              const du = pos[j*3]-x, dv = pos[j*3+1]-y, dw = pos[j*3+2]-z;
              const d2 = du*du + dv*dv + dw*dw;
              if(d2 > reach2) continue;
              if(pairHash(src[i], lidx[i], src[j], lidx[j], pseed) < pr.prob * Math.exp(-d2/ps2)){
                if(counting){ emit(i, j, 0, 0, 0, j, wbase); continue; }
                const wv = (logn
                  ? wbase*Math.exp(sg*pairGauss(src[i], lidx[i], src[j], lidx[j], pseed) - sg*sg/2)
                  : wbase*(0.5 + pairHash(src[i], lidx[i], src[j], lidx[j], pseed + 101)))
                  *wScale;
                emit(i, j, wv, qd(Math.sqrt(d2)/vel), prRule(pr), j, wbase);
              }
            }
          }
        }
        continue;
      }
      const bg = new Map();
      const bk = (x, y) => gridKey2(Math.floor(x/reach), Math.floor(y/reach));
      for(const j of B){ const k = bk(pos[j*3+b1], pos[j*3+b2]);
        let a = bg.get(k); if(!a){ a=[]; bg.set(k,a); } a.push(j); }
      for(const i of A){
        if(i < I0 || i >= I1) continue;        // this slice's sources only
        let u = (pos[i*3+a1]-am1)/ar1, v = (pos[i*3+a2]-am2)/ar2;
        if(pr.flipU) u = 1-u;
        if(pr.flipV) v = 1-v;
        const tx = bm1 + u*br1, ty = bm2 + v*br2;
        const sign = NEURON_TYPES[pts.ntype[i]].sign;
        const wbase = sign > 0 ? pr.weight : -pr.weight;
        const gx = Math.floor(tx/reach), gy = Math.floor(ty/reach);
        for(let dx=-1;dx<=1;dx++) for(let dy=-1;dy<=1;dy++){
          const bucket = bg.get(gridKey2(gx+dx, gy+dy)); if(!bucket) continue;
          for(const j of bucket){
            if(j === i) continue;
            const du = pos[j*3+b1]-tx, dv = pos[j*3+b2]-ty;
            const d2 = du*du + dv*dv;
            if(d2 > reach2) continue;
            if(pairHash(src[i], lidx[i], src[j], lidx[j], pseed)
                < pr.prob * Math.exp(-d2/ps2)){
              if(counting){ emit(i, j, 0, 0, 0, j, wbase); continue; }
              const wv = (logn
                ? wbase*Math.exp(sg*pairGauss(src[i], lidx[i], src[j], lidx[j], pseed) - sg*sg/2)
                : wbase*(0.5 + pairHash(src[i], lidx[i], src[j], lidx[j], pseed + 101)))
                *wScale;
              emit(i, j, wv, qd(baseMs + Math.sqrt(d2)/vel), prRule(pr), j, wbase);
            }
          }
        }
      }
    }
  };

  sweep();                                   // pass 1: count
  preStart = new Int32Array(n+1);
  for(let i=0;i<n;i++) preStart[i+1] = preStart[i] + deg[i];
  postA = new Int32Array(m); wA = new Float32Array(m);
  delA = new Uint8Array(m); pmA = new Uint8Array(m);
  cursor = new Int32Array(n);
  counting = false;                          // in-degrees were tallied in pass 1
  sweep();                                   // pass 2: fill

  // A slice returns its raw CSR and its in-degree tallies and stops here. preStart is zero for every neuron below the slice, so the arrays are already based at zero and concatenating slices in order reconstructs the whole CSR without touching a single offset.
  if(RANGE) return { partial:true, i0:I0, i1:I1, deg, preStart,
    post:postA, w:wA, delay:delA, pmask:pmA, synCount:m, KE, KI, clampedDelay };

  return finishConnect(pts, p, { preStart, post:postA, w:wA, delay:delA,
    pmask:pmA, m, KE, KI, T, lidx, R2, logn, sg, wScale, MAX_SYN, clampedDelay });
}

// The part of the wiring that is not per-presynaptic-neuron: triadic closure, which samples the finished network with one sequential stream, and the proxy mean compensation, which needs the in-degree tallies from every slice.
// Shared by the serial and the parallel paths so there is one implementation of both.
function finishConnect(pts, p, st, onProgress){
  const n = pts.count, pos = pts.pos;
  const lidx = st.lidx || pts.lidx || freshLidx(n);
  const T = st.T || buildPairLUT(pts, p.table, pts.rules);
  const R2 = st.R2 !== undefined ? st.R2 : p.radius*p.radius;
  const logn = st.logn !== undefined ? st.logn : (p.wdist|0) === 1;
  const sg = st.sg !== undefined ? st.sg : (p.wsigma || 1);
  const declared = (p.density === undefined ? 100 : p.density)/100;
  const kscale = (p.kscale === undefined ? 100 : p.kscale)/100;
  const res = Math.max(0.01, Math.min(1,
    declared * kscale * (p.__res === undefined ? 1 : p.__res)));
  const wScale = st.wScale !== undefined ? st.wScale : 1/Math.sqrt(res);
  const MAX_SYN = st.MAX_SYN || Math.max(1e6, p.__maxSyn || 5e7);
  let { preStart, post:postA, w:wA, delay:delA, pmask:pmA, m, KE, KI } = st;

  // cluster boost (Perin/Markram common-neighbor statistics): append extra synapses by triadic closure: pick a wired i->j, then one of j's targets k, and close the triangle i->k if k is within radius.
  // Separate RNG stream, so cluster=0 leaves the base network byte-identical.
  if(p.cluster > 0 && m > 0){
    const pre = new Int32Array(m);            // presynaptic owner per synapse
    for(let i=0;i<n;i++)
      for(let s=preStart[i]; s<preStart[i+1]; s++) pre[s] = i;
    const rc = rng(p.seed*31+7);
    const want = Math.max(0, Math.min(MAX_SYN - m, Math.round(p.cluster*m)));
    const eP = new Int32Array(want), eQ = new Int32Array(want),
          eW = new Float32Array(want), eD = new Uint8Array(want),
          eM = new Uint8Array(want);
    let added = 0, guard = want*30;
    while(added < want && guard-- > 0){
      if(onProgress && (added & 65535) === 0) onProgress(0.9 + 0.1*added/want, m + added);
      const s1 = (rc()*m)|0, i = pre[s1], j = postA[s1];
      const dj = preStart[j+1]-preStart[j]; if(!dj) continue;
      const k = postA[preStart[j] + ((rc()*dj)|0)];
      if(k === i) continue;
      // a pair the table sets to zero is a statement that these populations do not connect; triadic closure must not invent tracts the table forbids.
      // Without this check the boost closes retina to cortex shortcuts across the thalamic relays, since the projected tracts are wired paths like any other.
      if(T.probMul[T.gIdx[i]*T.G + T.gIdx[k]] === 0) continue;
      const ax = pos[i*3]-pos[k*3], ay = pos[i*3+1]-pos[k*3+1], az = pos[i*3+2]-pos[k*3+2];
      const d2 = ax*ax+ay*ay+az*az;
      if(d2 > R2) continue;
      const base = NEURON_TYPES[pts.ntype[i]].sign > 0 ? p.wExc : p.wInh;
      eP[added] = i; eQ[added] = k;
      eW[added] = (logn ? base*Math.exp(sg*gaussRand(rc) - sg*sg/2) : base*(0.5+rc()))
        *wScale*T.wMul[T.gIdx[i]*T.G + T.gIdx[k]];
      const dr = Math.max(1, Math.round(Math.sqrt(d2)/p.velocity));
      if(dr > MAX_DELAY) st.clampedDelay = (st.clampedDelay || 0) + 1;   // counted as the sweep's are
      eD[added] = Math.min(MAX_DELAY, dr);
      eM[added] = T.plMul[T.gIdx[i]*T.G + T.gIdx[k]];
      if(KE){ if(base > 0) KE[k]++; else KI[k]++; }
      added++;
    }
    if(added){
      const m2 = m + added;
      const ps = new Int32Array(n+1), cur2 = new Int32Array(n);
      for(let i=0;i<n;i++) ps[i+1] = preStart[i+1]-preStart[i];
      for(let s=0;s<added;s++) ps[eP[s]+1]++;
      for(let i=0;i<n;i++) ps[i+1] += ps[i];
      const post2 = new Int32Array(m2), w2 = new Float32Array(m2), del2 = new Uint8Array(m2),
            pmc = new Uint8Array(m2);
      for(let i=0;i<n;i++)
        for(let s=preStart[i]; s<preStart[i+1]; s++){
          const at = ps[i]+cur2[i]++;
          post2[at]=postA[s]; w2[at]=wA[s]; del2[at]=delA[s]; pmc[at]=pmA[s];
        }
      for(let s=0;s<added;s++){ const i=eP[s], at=ps[i]+cur2[i]++;
        post2[at]=eQ[s]; w2[at]=eW[s]; del2[at]=eD[s]; pmc[at]=eM[s]; }
      preStart = ps; postA = post2; wA = w2; delA = del2; pmA = pmc; m = m2;
    }
  }
  // explicit synapses from connections tables, on top of what the sweep made
  let connectionsReport = null;
  if(pts.connectionTables && pts.connectionTables.length){
    const w2 = wireConnectionsFile(pts, pts.connectionTables, { preStart, post:postA, w:wA, delay:delA, pmask:pmA, m, KE, KI }, ruleIds(pts.rules));
    ({ preStart, post:postA, w:wA, delay:delA, pmask:pmA, m } = w2);
    connectionsReport = w2.connectionsReport;
    if(m > MAX_SYN) throw new Error('connections file: ' + m.toLocaleString() + ' synapses is past the budget of ' + MAX_SYN.toLocaleString());
  }
  // proxy mean compensation: restore the full-density mean input per neuron.
  // The mean input is rate times in-degree times the charge one spike delivers, and what a weight means as a charge is the checkpoint's to say (kick: w; exp peak: w tau; exp charge: w), so the wiring carries the two terms with the charge left at one and the checkpoint's compute scales them (compensationBias).
  let comp;
  if(res < 1){
    const f = (1/res - 1/Math.sqrt(res))*0.001;   // Hz -> per-ms, missing-mean factor
    // the presynaptic rates: set on the node, else the rate of each class measured at full resolution (one rate for both classes put a scene whose interneurons fire three times faster than its pyramids 66 percent over its full-scale rate, 2026-09-24)
    const auto = p.__rateHint || { E:5, I:5 };
    const nuE = p.nuE > 0 ? p.nuE : (auto.E > 0 ? auto.E : 5), nuI = p.nuI > 0 ? p.nuI : (auto.I > 0 ? auto.I : 5);
    const e = new Float32Array(n), i2 = new Float32Array(n);
    for(let i=0;i<n;i++){ e[i] = f*nuE*KE[i]*p.wExc; i2[i] = f*nuI*KI[i]*p.wInh; }
    comp = { e, i:i2 };
  }
  const pools = wirePools(pts);
  return { kind:'net', count:n, pos:pts.pos, ntype:pts.ntype, bias:pts.bias, src:pts.src, ...(comp ? { comp } : {}),
    lidx, tags:pts.tags || {}, preStart, post:postA, w:wA, delay:delA,
    pmask:pmA, synCount:m,
    hues:pts.hues, receptors:pts.receptors, ...(pools || {}),
    // the declared rules ride along so the checkpoint can build the table the engines index with the per-synapse byte
    rules: pts.rules && pts.rules.length ? pts.rules : undefined,
    // names the wiring could not resolve to a rule, so the status line can say so without the wiring having failed
    ruleWarn: T.ruleWarn && T.ruleWarn.length ? T.ruleWarn : undefined,
    // how many delays the ring buffer could not hold, for the status line
    delayClamped: (st.clampedDelay || 0) + (connectionsReport ? connectionsReport.clampedDelay || 0 : 0), delayMax: MAX_DELAY,
    // what the connections tables added and could not place, for the status line
    connectionsReport: connectionsReport || undefined,
    // the engines seed their noise and take their initial state from this
    seed: p.seed | 0 };
}

// ---- parallel wiring -----------------------------------------------
// The sweep is the whole cost of the wiring and it is embarrassingly parallel over presynaptic neurons.
// Measured 2026-08-30 on the cortical column: 14,030 neurons and 10.06M synapses took 6.6s, of which the two sweeps were 4.8s, examining 213M candidate pairs per pass.
// The rest of the wiring (the cluster boost, which walks the finished network with one sequential random stream, and the proxy compensation, which needs every slice's in-degree tallies) stays on the assembling thread.
export function wireWorkerCount(){
  const hw = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
  // leave a core for the assembling thread and one for whatever else the page is doing, and never split so finely that the per-slice fixed cost (rebuilding the spatial grid and the pair table, both O(n)) dominates
  return Math.max(1, Math.min(12, hw - 2));
}

// Nested workers are not available in every embedding (the pane this was developed in refuses them outright, classic and module alike), so the pool is owned by the page and the compute worker stays the assembler.
// The page does no arithmetic: it hands out ranges and forwards each finished slice on with its buffers transferred, so nothing is copied and nothing is computed on the thread that has to stay responsive.
let poolOK = null;
export async function wirePoolAvailable(){
  if(poolOK !== null) return poolOK;
  poolOK = await new Promise(res => {
    let w;
    try { w = new Worker(new URL('./wireworker.js', import.meta.url), { type:'module' }); }
    catch(e){ res(false); return; }
    let settled = false;
    const done = v => { if(settled) return; settled = true;
      try { w.terminate(); } catch(e){} res(v); };
    w.onmessage = e => done(!!(e.data && e.data.pong));
    w.onerror = () => done(false);
    w.postMessage({ ping:true });
    setTimeout(() => done(false), 3000);
  });
  return poolOK;
}

// Run the sweep across a pool, returning the raw slices in presynaptic order.
// Assembly is the caller's business.
export async function wireSlices(pts, p, onProgress, pool){
  const n = pts.count;
  // Equal neuron counts, not equal work: a neuron's cost is the size of its neighborhood, which is not uniform in a layered scene.
  // Slices are small and plentiful relative to the pool so a worker that draws a cheap one comes straight back for another.
  const SLICES = pool.length * 4;
  const bounds = [];
  for(let k=0;k<SLICES;k++){
    const a = Math.floor(k*n/SLICES), b = Math.floor((k+1)*n/SLICES);
    if(b > a) bounds.push([a, b]);
  }
  const done = new Array(bounds.length).fill(null);
  let synSoFar = 0;
  let next = 0, finished = 0;
  await Promise.all(pool.map(w => new Promise((resolve, reject) => {
    const take = () => {
      if(next >= bounds.length){ resolve(); return; }
      const idx = next++, [a, b] = bounds[idx];
      w.onmessage = ev => {
        const d = ev.data;
        if(!d || !d.ok){ reject(new Error(d && d.message || 'wiring slice failed')); return; }
        done[idx] = d.r; finished++;
        // the pool reports the synapses its finished slices actually hold, which is the same number the serial path reports
        synSoFar += (d.r && d.r.synCount) || 0;
        if(onProgress) onProgress(finished/bounds.length, synSoFar);
        take();
      };
      w.onerror = ev => reject(new Error('wiring worker: ' + (ev.message || 'failed to run')));
      w.postMessage({ pts, params:p, i0:a, i1:b });
    };
    take();
  })));
  return done;
}

// Concatenate the slices and finish the wiring.
// Ranges are contiguous and in order, and a slice's CSR is based at zero because preStart is zero for every neuron below it, so this is concatenation: no offset is rewritten and no synapse moves.
export function assembleConnect(pts, p, parts, onProgress){
  // the same reason as in wireConnect: this runs on the compute worker, whose type table has not seen the scene's own rows
  for(const r of pts.cellTypes || []) registerCellType(r);
  const n = pts.count;
  let m = 0;
  for(const r of parts) m += r.synCount;
  // Each slice checked the budget against its own count only, so slices each under it could total over it and wire past the memory guard.
  // Checked on the total, before the arrays.
  const MAX_SYN = Math.max(1e6, p.__maxSyn || 5e7);
  if(m > MAX_SYN) throw new Error(
    `> ${(MAX_SYN/1e6).toFixed(0)}M synapses: lower radius, probability, or resolution`);
  const deg = new Int32Array(n);
  for(const r of parts) for(let i=r.i0;i<r.i1;i++) deg[i] = r.deg[i];
  const preStart = new Int32Array(n+1);
  for(let i=0;i<n;i++) preStart[i+1] = preStart[i] + deg[i];
  const post = new Int32Array(m), w = new Float32Array(m),
    delay = new Uint8Array(m), pmask = new Uint8Array(m);
  let at = 0;
  for(const r of parts){
    post.set(r.post, at); w.set(r.w, at);
    delay.set(r.delay, at); pmask.set(r.pmask, at);
    at += r.synCount;
  }
  // in-degree tallies are indexed by target, so every slice contributes to every neuron and these are summed rather than concatenated
  let KE = null, KI = null;
  if(parts.length && parts[0].KE){
    KE = new Int32Array(n); KI = new Int32Array(n);
    for(const r of parts) for(let i=0;i<n;i++){ KE[i] += r.KE[i]; KI[i] += r.KI[i]; }
  }
  let clampedDelay = 0;
  for(const r of parts) clampedDelay += r.clampedDelay || 0;
  return finishConnect(pts, p, { preStart, post, w, delay, pmask, m, KE, KI, clampedDelay },
    onProgress);
}

// global resolution (the top-bar slider): scales every scatter's count at compute time and feeds the connect proxy compensation, non-destructively: authored params always describe the full-scale network
let simResolution = 1;
export function setResolution(v){ simResolution = Math.max(0.01, Math.min(1, v)); }
function getResolution(){ return simResolution; }
// measured full-scale mean rate (Hz), captured by main while running at 100%; the proxy compensation targets it when nuE/nuI are left on auto
let rateHint = { E:5, I:5 };
// a rate per class (excitatory, inhibitory), or one number for both
export function setRateHint(hz){
  if(typeof hz === 'number'){ if(hz > 0) rateHint = { E:hz, I:hz }; }
  else if(hz && hz.E > 0 && hz.I > 0) rateHint = { E:hz.E, I:hz.I };
}

// the connect node computes in a dedicated worker so the UI (and the running sim) never block during wiring; progress streams back through onComputeProgress
let onComputeProgress = null;
export function setComputeProgress(fn){ onComputeProgress = fn; }
// Progress belongs to the newest wiring and to no other.
// There is one hook and two wirings can overlap: starting a training run leaves the interactive computation running while the trainer starts its own, and both would report into the same bar.
// An older wiring reports nothing once a newer one has begun; it is on its way to being discarded anyway.
let wireEpoch = 0;
const newWireEpoch = () => ++wireEpoch;
const reportProgress = (epoch, f, syn) => {
  if(epoch === wireEpoch && onComputeProgress) onComputeProgress(f, syn);
};
const computeWorkers = new Set();
function cancelComputes(){
  for(const e of computeWorkers){
    e.w.terminate();
    const err = new Error('superseded'); err.superseded = true;
    e.reject(err);
  }
  computeWorkers.clear();
}
// Synapse budget.
// This is a memory guard, not a design limit: the ceiling is what the machine can actually hold (13 B per synapse while wiring), so it is derived from reported device memory rather than a fixed number.
// Override either way with localStorage.setItem('neuron-playground-max-synapses', '800000000').
// A headless run (tools/linen.mjs, tune.mjs) has no browser storage and no device report, so it sets the budget from the machine's memory here.
let maxSynOverride = 0;
export function setMaxSyn(n){ maxSynOverride = n >= 1e6 ? Math.round(n) : 0; }
export function maxSynLimit(){
  if(maxSynOverride) return maxSynOverride;
  try {
    const v = +localStorage.getItem('neuron-playground-max-synapses');
    if(v >= 1e6) return v;
  } catch(e){}
  return deviceSynLimit();
}
// what this machine reports it can hold, before any override
export function deviceSynLimit(){
  // navigator.deviceMemory is privacy-clamped to 8, so >=8 means 8 GB or much more; leave room for the sim worker's own copy and the browser
  const gb = (typeof navigator !== 'undefined' && navigator.deviceMemory) || 4;
  return Math.max(5e7, Math.round(gb*0.35 * 1e9 / 13));
}
async function connectCompute(ins, p){
  const src = need(ins[0], 'points', 'connect needs points');
  const pts = { count:src.count, pos:src.pos, ntype:src.ntype, bias:src.bias,
    src:src.src, lidx:src.lidx, tags:src.tags, projs:src.projs, rules:src.rules,
    // the pools travel with the rules and the projections: this copy is the stream the wiring sees, and a field left off it is a mechanism that computes and is never wired
    pools:src.pools, ids:src.ids, connectionTables:src.connectionTables, receptors:src.receptors, cellTypes:src.cellTypes, hues:src.hues };
  const params = { ...p, __maxSyn:maxSynLimit(), __res:simResolution,
    __rateHint:rateHint };
  // Fan the sweep out when there is enough of it to be worth the fixed cost of starting workers and shipping the point cloud to each.
  const W = wireWorkerCount();
  const epoch = newWireEpoch();          // this wiring owns the bar until another starts
  let parts = null;
  if(W >= 2 && pts.count >= 2000 && await wirePoolAvailable()){
    const pool = [];
    try {
      for(let k=0;k<W;k++)
        pool.push(new Worker(new URL('./wireworker.js', import.meta.url), { type:'module' }));
      parts = await wireSlices(pts, params, (f, syn) => {
        reportProgress(epoch, f*0.9, syn);           // assembly is the last tenth
      }, pool);
    } catch(err){
      if(err && err.superseded) throw err;
      parts = null;                                // fall back to one thread
    } finally {
      for(const w of pool) w.terminate();
    }
  }
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL('./computeworker.js', import.meta.url), { type:'module' });
    const entry = { w, reject };
    computeWorkers.add(entry);
    w.onmessage = e => {
      const m = e.data;
      if(m.t === 'progress'){ reportProgress(epoch, m.f, m.syn); return; }
      computeWorkers.delete(entry); w.terminate();
      if(m.t === 'done') resolve(m.net);
      else reject(new Error(m.message));
    };
    w.onerror = err => { computeWorkers.delete(entry); w.terminate();
      reject(new Error(err.message || 'compute worker failed')); };
    if(parts){
      // hand the slices over without copying them
      const bufs = [];
      for(const r of parts){
        bufs.push(r.deg.buffer, r.preStart.buffer, r.post.buffer, r.w.buffer,
          r.delay.buffer, r.pmask.buffer);
        if(r.KE) bufs.push(r.KE.buffer, r.KI.buffer);
      }
      w.postMessage({ t:'assemble', pts, params, parts }, bufs);
    }
    else w.postMessage({ pts, params });
  });
}

function pointInGeo(g, x, y, z){
  if(g.shape === 'mesh') return meshInside(g, x, y, z);
  if(g.shape === 'spline'){                          // min distance to polyline
    const poly = g._poly || (g._poly = splinePoly(g));
    const r2 = g.radius*g.radius;
    for(let s=0;s<poly.length/3-1;s++){
      const ax = poly[s*3], ay = poly[s*3+1], az = poly[s*3+2];
      const bx = poly[(s+1)*3]-ax, by = poly[(s+1)*3+1]-ay, bz = poly[(s+1)*3+2]-az;
      const len2 = bx*bx+by*by+bz*bz;
      let t = len2 ? ((x-ax)*bx+(y-ay)*by+(z-az)*bz)/len2 : 0;
      t = Math.max(0, Math.min(1, t));
      const dx = x-ax-bx*t, dy = y-ay-by*t, dz = z-az-bz*t;
      if(dx*dx+dy*dy+dz*dz <= r2) return true;
    }
    return false;
  }
  let dx = x-g.center[0], dy = y-g.center[1], dz = z-g.center[2];
  if(g.shape === 'sphere') return dx*dx+dy*dy+dz*dz <= g.radius*g.radius;
  if(g.shape === 'noisefield'){
    if(Math.abs(dx) > g.size[0]/2 || Math.abs(dy) > g.size[1]/2 || Math.abs(dz) > g.size[2]/2)
      return false;
    // the same field the scatter fills, so a noise volume means one thing as a region and as a mask
    return noiseField(g, dx, dy, dz);
  }
  if(g.rotate && g.rotate.some(v => v)){
    const m = rotMat(...g.rotate.map(d => d*Math.PI/180));   // world = R * local
    const lx = m[0]*dx+m[3]*dy+m[6]*dz, ly = m[1]*dx+m[4]*dy+m[7]*dz,
          lz = m[2]*dx+m[5]*dy+m[8]*dz;                      // local = R^T * world
    dx=lx; dy=ly; dz=lz;
  }
  if(g.shape === 'ellipsoid'){
    const ex = dx/g.radii[0], ey = dy/g.radii[1], ez = dz/g.radii[2];
    return ex*ex+ey*ey+ez*ez <= 1;
  }
  if(g.shape === 'cylinder')
    return dx*dx+dz*dz <= g.radius*g.radius && Math.abs(dy) <= g.height/2;
  if(g.shape === 'torus'){
    const q = Math.hypot(dx, dz) - g.radius;
    return q*q + dy*dy <= g.thickness*g.thickness;
  }
  return Math.abs(dx) <= g.size[0]/2 && Math.abs(dy) <= g.size[1]/2 && Math.abs(dz) <= g.size[2]/2;
}

// Which neurons a stimulus drives is decided by a mask, a radius and a population tag together, so it is not readable from any one of them.
// The compute records the set on the network (display only; nothing downstream reads it) so the viewer can draw what this node is actually doing.
function stimulusCompute(ins, p, node){
  const src = ins[0];
  if(!src || (src.kind !== 'net' && src.kind !== 'points'))
    throw new Error('stimulus needs a network or points');
  const mask = ins[1] && ins[1].kind === 'geo' ? ins[1] : null;
  const R2 = p.radius*p.radius, [cx,cy,cz] = p.center, pos = src.pos;
  // optional population filter: a scatter tag, or the class keys E and I. External drive is population specific in cortex (the reference microcircuit gives each population its own external in-degree), which a purely geometric mask cannot express since E and I share a layer.
  const tag = (p.tag || '').trim();
  const tags = src.tags || {};
  const tagOk = !tag ? (() => true) : (i => {
    if(tag === 'E') return NEURON_TYPES[src.ntype[i]].sign > 0;
    if(tag === 'I') return NEURON_TYPES[src.ntype[i]].sign < 0;
    return tagMatch(tags[src.src[i]], tag) !== null;   // a tag, or a pattern with one *
  });
  const inside = i => {
    if(!tagOk(i)) return false;
    const x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    if(mask) return pointInGeo(mask, x, y, z);
    const dx=x-cx, dy=y-cy, dz=z-cz; return dx*dx+dy*dy+dz*dz <= R2;
  };
  // Per-neuron spread of the drive.
  // The background current stands for every input arriving from outside the simulated volume, and there is no reason that should be identical for every cell: synapse counts per neuron are heavy-tailed like most anatomical quantities, and cortical firing rates are lognormal in every region and state examined (Buzsaki and Mizuseki 2014).
  // A uniform current produces a uniform population, measured on 2026-08-28 as a 1.4x spread from median to 99th percentile with no silent cells, where cortex spans orders of magnitude.
  // The multiplier is lognormal with the given sigma and is normalized so the population mean current is unchanged, which keeps this a change in heterogeneity rather than in total drive.
  // It is keyed to stable neuron identity, so the same graph computes the same network.
  const spread = p.spread > 0 ? +p.spread : 0;
  const srcIds = src.src, lidxs = src.lidx;
  const spreadAt = spread > 0 && srcIds && lidxs
    ? (i => {
        const g = pairGauss(srcIds[i]|0, lidxs[i]|0, 0x5EED, p.seedSpread|0, 0x9E37);
        return Math.exp(spread*g - spread*spread/2);
      })
    : (() => 1);
  const mode = p.mode|0;
  if(src.kind === 'net' && mode > 0){
    // pulse/ramp/noise: evaluated per sim-ms in the worker; bias stays untouched
    const idx = [];
    for(let i=0;i<src.count;i++) if(inside(i)) idx.push(i);
    const proto = { idx:Uint32Array.from(idx), amp:p.current, mode,
      period:Math.max(1, p.period), width:Math.max(1, p.width),
      t0:Math.max(0, p.t0), duration:Math.max(0, p.duration) };
    if(spread > 0) proto.gain = Float32Array.from(idx, i => spreadAt(i));
    const drive = { id:node ? node.id : -1, idx:proto.idx, amp:p.current, mode,
      gain:proto.gain || null };
    return { ...src, protocols:[...(src.protocols || []), proto],
      drives:[...(src.drives || []), drive] };
  }
  // constant: added into bias. net -> net clones only the bias array, so synapses upstream are never rewired by a stimulus edit; points -> points kept for older graphs (constant only)
  const out = src.kind === 'net' ? { ...src, bias: src.bias.slice() } : clonePts(src);
  const hit = [], gains = [];
  for(let i=0;i<out.count;i++) if(inside(i)){
    const g = spreadAt(i);
    out.bias[i] += p.current*g;
    hit.push(i); gains.push(g);
  }
  if(src.kind === 'net')
    out.drives = [...(src.drives || []), { id:node ? node.id : -1,
      idx:Uint32Array.from(hit), amp:p.current, mode:0,
      gain: spread > 0 ? Float32Array.from(gains) : null }];
  return out;
}

// live input encoder: makes a deterministic channel -> neuron map from the masked region's geometry at compute time.
// The live source (test pattern, microphone, webcam) runs on the main thread and streams per-channel gains to the engine between frames (inputFrame); the computation itself never touches live data, so it stays pure and memo-safe.
function inputCompute(ins, p, node){
  const src = ins[0];
  if(!src || src.kind !== 'net') throw new Error('input needs a network (place after connect)');
  // side 1 is the region mask, side 2 the signal, each refused if the other kind lands on it so a wire on the wrong port says so instead of working by accident
  const mask = ins[1] || null, sig = ins[2] || null;
  if(mask && mask.kind !== 'geo')
    throw new Error('the mask port takes geometry; the signal goes on the signal port');
  if(sig && sig.kind !== 'signal')
    throw new Error('the signal port takes a curriculum, test signal, live or read node');
  if(!sig)
    throw new Error('input needs a signal: wire a curriculum, test signal, live or read node into its signal port');
  // Optional population, the same filter the probe and stimulus nodes take: a scatter tag, or the class keys E and I. Either sense can drive any population, and a mask alone cannot separate cells that share a volume.
  const tag = (p.tag || '').trim();
  const tags = src.tags || {};
  const tagOk = !tag ? (() => true) : (i => {
    if(tag === 'E') return NEURON_TYPES[src.ntype[i]].sign > 0;
    if(tag === 'I') return NEURON_TYPES[src.ntype[i]].sign < 0;
    return tagMatch(tags[src.src[i]], tag) !== null;   // a tag, or a pattern with one *
  });
  const pos = src.pos, targets = [];
  for(let i=0;i<src.count;i++){
    if(!tagOk(i)) continue;
    if(!mask || pointInGeo(mask, pos[i*3], pos[i*3+1], pos[i*3+2])) targets.push(i);
  }
  if(!targets.length) throw new Error(tag
    ? `input drives no neurons: no cell is tagged ${tag}` + (mask ? ' inside its mask' : '')
    : 'input mask selects no neurons');
  const mapMode = p.map|0;                    // 0 sheet, 1 bands, 2 distributed
  const grid2d = mapMode !== 1;               // the source's raster shape
  const axis = p.axis|0;                      // sheet: projection normal; bands: the axis
  const a1 = grid2d ? (axis===0 ? 1 : 0) : axis;
  const a2 = axis===2 ? 1 : 2;
  const cols = Math.max(2, p.cols|0), rows = grid2d ? Math.max(2, p.rows|0) : 1;
  const G = cols*rows;                        // raster channels
  const retina = (p.code|0) === 1;            // ON/OFF contrast doubles them
  const C = retina ? 2*G : G;
  const lidx = src.lidx, srcIds = src.src;
  const entC = [], entI = [], entW = [];      // channel/neuron/weight triplets
  if(mapMode === 2){
    // distributed random projection (olfactory bulb to piriform style, Stettler and Axel 2009): every channel scatters across the whole target population, deterministic per seed; no spatial map at all
    const seed = (p.seed|0) + 1;
    const per = Math.max(1, Math.round(targets.length * Math.max(1, p.fanin) / C));
    let wSum = 0;
    for(let c=0;c<C;c++)
      for(let j=0;j<per;j++){
        const h = h32((c+1)*2654435761 ^ (j+1)*334214467 ^ seed*974711) >>> 0;
        const wv = 0.4 + 0.6*((h32(h ^ 0x9e3779b9) >>> 0) / 4294967296);
        entC.push(c); entI.push(targets[h % targets.length]); entW.push(wv);
        wSum += wv;
      }
    const norm = entW.length / wSum;          // mean weight 1, heterogeneity kept
    for(let e=0;e<entW.length;e++) entW[e] *= norm;
  } else {
    // topographic (retinotopy / tonotopy): each neuron connects to the grid channels within a Gaussian arbor around its projected position (thalamocortical afferents innervate a patch, not a point).
    // In retina coding, neurons split into interleaved ON and OFF target mosaics.
    let m1=Infinity, M1=-Infinity, m2=Infinity, M2=-Infinity;
    for(const i of targets){
      const x1 = pos[i*3+a1], x2 = pos[i*3+a2];
      if(x1<m1) m1=x1; if(x1>M1) M1=x1;
      if(x2<m2) m2=x2; if(x2>M2) M2=x2;
    }
    const r1 = Math.max(1e-6, M1-m1), r2 = Math.max(1e-6, M2-m2);
    const arbor = Math.max(0, p.arbor), reach = Math.max(0, Math.ceil(2*arbor));
    const s2 = 2*Math.max(1e-4, arbor*arbor);
    for(let k=0;k<targets.length;k++){
      const i = targets[k];
      const u = (pos[i*3+a1]-m1)/r1*cols - 0.5;
      const v = grid2d ? (pos[i*3+a2]-m2)/r2*rows - 0.5 : 0;
      const pol = retina ? ((h32(((srcIds[i]|0)+1)*2246822519 ^ (lidx[i]+1)*3266489917) >>> 0) & 1) : 0;
      const c0 = Math.min(cols-1, Math.max(0, Math.round(u)));
      const r0 = Math.min(rows-1, Math.max(0, Math.round(v)));
      let base = entW.length, wSum = 0;
      for(let dr=-reach;dr<=reach;dr++){
        const r = r0+dr; if(r<0 || r>=rows) continue;
        if(!grid2d && dr !== 0) continue;
        for(let dc=-reach;dc<=reach;dc++){
          const c = c0+dc; if(c<0 || c>=cols) continue;
          const du = c-u, dv = grid2d ? r-v : 0;
          const wv = reach === 0 ? 1 : Math.exp(-(du*du + dv*dv)/s2);
          if(wv < 0.05) continue;
          entC.push(pol*G + r*cols + c); entI.push(i); entW.push(wv);
          wSum += wv;
        }
      }
      if(wSum === 0){ entC.push(pol*G + r0*cols + c0); entI.push(i); entW.push(1); wSum = 1; }
      for(let e=base;e<entW.length;e++) entW[e] /= wSum;   // unit drive per neuron
    }
  }
  const E = entC.length;
  const cnt = new Uint32Array(C);
  for(let e=0;e<E;e++) cnt[entC[e]]++;
  const chStart = new Uint32Array(C+1);
  for(let c=0;c<C;c++) chStart[c+1] = chStart[c] + cnt[c];
  const at = new Uint32Array(C);
  const chIdx = new Uint32Array(E), chW = new Float32Array(E);
  for(let e=0;e<E;e++){
    const o = chStart[entC[e]] + at[entC[e]]++;
    chIdx[o] = entI[e]; chW[o] = entW[e];
  }
  // the runtime keys its source on a small integer: 0 bar sweep, 1 microphone, 2 webcam, -1 read node (file playback), -2 curriculum
  const sigSrc = sig.src === 'curriculum' ? -2 : sig.src === 'footage' ? -1
    : sig.src === 'live' ? (sig.device === 'webcam' ? 2 : 1) : 0;
  // what the channels carry is the signal's to say; only a curriculum has a sense, and everything else is taken as an image
  const sense = sig.sense === 'sound' ? 'sound' : 'sight';
  const im = { id:node.id, source:sigSrc, cols, rows, sheet:grid2d, sense,
    tag, masked:!!mask,
    code: retina ? 1 : 0, transient:p.transient, jitter:p.jitter,
    lagMs:p.lagMs || 0,
    period:sig.period || 1500, amp:p.amp, chStart, chIdx, chW,
    signal: { ...sig } };
  return { ...src, inputMaps:[...(src.inputMaps || []), im] };
}

// probe: registers a population for live readout.
// Observation only, per the project rule that nothing is ever assigned to the tissue; the app counts the probe's spikes from the fired flags it already receives
function probeCompute(ins, p, node){
  const src = ins[0];
  if(!src || src.kind !== 'net') throw new Error('probe needs a network (place after connect)');
  const mask = ins[1] && ins[1].kind === 'geo' ? ins[1] : null;
  // Optional population filter, the same one the stimulus node takes: a scatter tag, or the class keys E and I. Geometry alone cannot name a population, since excitatory and inhibitory cells share a volume in every scene here, so without this a probe on a mixed population reads the two together and calls the average a rate.
  const tag = (p.tag || '').trim();
  const tags = src.tags || {};
  const tagOk = !tag ? (() => true) : (i => {
    if(tag === 'E') return NEURON_TYPES[src.ntype[i]].sign > 0;
    if(tag === 'I') return NEURON_TYPES[src.ntype[i]].sign < 0;
    return tagMatch(tags[src.src[i]], tag) !== null;   // a tag, or a pattern with one *
  });
  const pos = src.pos, idx = [];
  for(let i=0;i<src.count;i++){
    if(!tagOk(i)) continue;
    if(!mask || pointInGeo(mask, pos[i*3], pos[i*3+1], pos[i*3+2])) idx.push(i);
  }
  if(!idx.length) throw new Error(tag
    ? `probe selects no neurons: no cell is tagged ${tag}` +
      (mask ? ' inside its mask' : '')
    : 'probe mask selects no neurons');
  const pr = { id:node.id, label:p.label || ('probe ' + node.id), idx:Uint32Array.from(idx),
    audio:p.audio|0, record:p.record|0 };
  // decoder views: bin the probed neurons back through the same spatial maps the encoders use (observation only; the tissue is never touched)
  const axis = p.axis|0;
  const cell = (set, x1, x2, cols, rows) => {
    let m1=Infinity, M1=-Infinity, m2=Infinity, M2=-Infinity;
    for(const i of set){ const u=pos[i*3+x1], v=pos[i*3+x2];
      if(u<m1)m1=u; if(u>M1)M1=u; if(v<m2)m2=v; if(v>M2)M2=v; }
    const r1 = Math.max(1e-6, M1-m1), r2 = Math.max(1e-6, M2-m2);
    return set.map(i => {
      const c = Math.min(cols-1, Math.floor((pos[i*3+x1]-m1)/r1*cols));
      const r = rows > 1 ? Math.min(rows-1, Math.floor((pos[i*3+x2]-m2)/r2*rows)) : 0;
      return r*cols + c;
    });
  };
  if((p.view|0) === 1){
    const a1 = axis===0 ? 1 : 0, a2 = axis===2 ? 1 : 2;
    const cols = Math.max(2, p.cols|0), rows = Math.max(2, p.rows|0);
    pr.imgCols = cols; pr.imgRows = rows;
    pr.chImg = Uint32Array.from(cell(idx, a1, a2, cols, rows));
  }
  if((p.audio|0) === 2){
    const nb = Math.max(2, p.cols|0);
    pr.nBands = nb;
    pr.chBand = Uint32Array.from(cell(idx, axis, axis===2 ? 1 : 2, nb, 1));
  }
  return { ...src, probes:[...(src.probes || []), pr] };
}

// The rules this tool has, as configurations rather than as amplitudes: each one is a published rule with the terms it needs and nothing else on.
// The amplitudes are the ones the papers state where they state them, and otherwise the ones these scenes have been run at.
// Every value stays editable; a preset writes a starting point.
//
//   pair STDP          Song, Miller and Abbott 2000; soft bounds after
//                      Gutig et al. 2003, where the fixed point of a
//                      weight is A+/A- times the pre/post rate ratio
//   triplet STDP       Pfister and Gerstner 2006, the third factor that
//                      makes the rule rate-dependent and lets assemblies
//                      form; used by Pokorny et al. 2019 to associate two
//                      memory traces
//   inhibitory         Vogels et al. 2011: inhibition follows the
//   homeostasis        postsynaptic rate to a set point, which is what
//                      holds a recurrent network at a rate instead of
//                      letting it run away or fall silent
//   heterosynaptic     Zenke, Agnes and Gerstner 2015: synapses onto an
//   competition        active cell are pulled back toward their reference,
//                      so potentiating one costs the others
//   consolidation      a bistable well: weights near the consolidated value
//                      are held there, so what is learned survives what
//                      comes after it
//   frozen             every term off. A pathway can also be frozen from
//                      the pair table by writing 0 in its rule column; this
//                      is the same thing said on the node
const RULE_PRESETS = [
  null,                                       // custom: change nothing
  { set:{ name:'pair', aP:0.008, aM:0.001, wdep:1, trip:0, iEta:0, het:0, tin:0, cons:0 } },
  { set:{ name:'triplet', aP:0.005, aM:0.001, wdep:1, trip:0.005, iEta:0, het:0, tin:0, cons:0 } },
  { set:{ name:'inh', aP:0, aM:0, wdep:1, trip:0, iEta:0.02, het:0, tin:0, cons:0 } },
  { set:{ name:'compete', aP:0.008, aM:0.001, wdep:1, trip:0, iEta:0, het:0.002, tin:0, cons:0 } },
  { set:{ name:'consolidate', aP:0.008, aM:0.001, wdep:1, trip:0, iEta:0, het:0, tin:0,
          cons:0.5, consW:2, consP:10 } },
  { set:{ name:'frozen', aP:0, aM:0, wdep:1, trip:0, iEta:0, het:0, tin:0, cons:0 } },
];

// A tag names one population whatever its case: 'Input 2' and 'input 2' are the same cells, and a picker that offers the tag never has to care.
export function sameTag(a, b){
  return String(a || '').trim().toLowerCase() === String(b || '').trim().toLowerCase();
}
// A tag pattern: a tag, or a tag with one * standing for any run of characters, compared without case.
// Returns the text the * matched (the empty string for a plain tag) or null.
export function tagMatch(tag, pattern){
  const t = String(tag || '').trim().toLowerCase(), q = String(pattern || '').trim().toLowerCase();
  const k = q.indexOf('*');
  if(k < 0) return t === q ? '' : null;
  const head = q.slice(0, k), tail = q.slice(k + 1);
  if(t.length < head.length + tail.length || !t.startsWith(head) || !t.endsWith(tail)) return null;
  return t.slice(head.length, t.length - tail.length);
}
// A projection whose from and to both carry a * is one projection per matching pair: the text the * matched on the source side names the target (tier1.bulb*.RS to tier2.bulb*.RS joins bulb 1 to bulb 1).
// One side with a * takes every tag it matches at once.
export function expandProjection(pr, tags){
  const from = String(pr.from || ''), to = String(pr.to || '');
  if(!from.includes('*') || !to.includes('*')) return [pr];
  const all = [...new Set(Object.values(tags || {}))];
  const out = [], seen = new Set();
  for(const t of all){
    const m = tagMatch(t, from);
    if(m === null || seen.has(m)) continue;
    seen.add(m);
    const target = to.replace('*', m);
    if(all.some(u => sameTag(u, target))) out.push({ ...pr, from:t, to:target });
  }
  return out;
}
// Whether a curriculum node has another curriculum on its side port, in which case it takes that node's schedule and shows only its own stream.
export function followsCurriculum(n, nodes){
  const c = n && n.inputs && n.inputs[0];
  return !!(c && (nodes || []).some(x => x.id === c.id && x.type === 'curriculum'));
}

// The cell type node's two halves share this: index `idx` of the type table onto all, or a seeded fraction, of the stream or of one population in it.
// The draw is the retype node's (node id times 6151 plus 11), so a scene migrated from one draws the cells it drew.
function builtinRow(rowParam){
  const t = CELL_PRESETS[rowParam - 1];
  if(!t) throw new Error('cell type: no such row');
  return NEURON_TYPES.indexOf(t);
}
function applyCellType(pts, p, node, idx, customRow){
  const out = clonePts(pts);
  const tag = String(p.tag || '').trim(), tags = pts.tags || {};
  const hit = tag ? (i => tag === 'E' ? NEURON_TYPES[pts.ntype[i]].sign > 0 : tag === 'I' ? NEURON_TYPES[pts.ntype[i]].sign < 0
    : tagMatch(tags[pts.src[i]], tag) !== null) : (() => true);
  if(tag && ![...Array(out.count).keys()].some(hit)) throw new Error('cell type: no cell is tagged ' + tag + '; the stream has ' +
    [...new Set(Object.values(tags))].filter(Boolean).join(', '));
  const frac = p.frac === undefined ? 1 : +p.frac;
  if(frac >= 1){ for(let i = 0; i < out.count; i++) if(hit(i)) out.ntype[i] = idx; }
  else {
    const r = rng((node ? node.id : 1)*6151+11);
    for(let i = 0; i < out.count; i++){ const draw = r() < frac; if(draw && hit(i)) out.ntype[i] = idx; }
  }
  if(customRow) out.cellTypes = [...(pts.cellTypes || []).filter(r => r.key !== customRow.key), customRow];
  return out;
}
export const NODE_DEFS = {
  sphere: { title:'sphere', cat:'regions', color:'hsl(25,75%,55%)', inputs:0, params:[
      { k:'center', label:'center µm', t:'vec3', def:[0,0,0] },
      { k:'radius', label:'radius µm', t:'float', def:300, min:1, s:[50,3000] } ],
    compute:(ins,p)=>({ kind:'geo', shape:'sphere', center:[...p.center], radius:p.radius }) },
  box: { title:'box', cat:'regions', color:'hsl(25,75%,55%)', inputs:0, params:[
      { k:'center', label:'center µm', t:'vec3', def:[0,0,0] },
      { k:'size', label:'size µm', t:'vec3', def:[600,600,600] },
      { k:'rotate', label:'rotate deg', t:'vec3', def:[0,0,0] } ],
    compute:(ins,p)=>({ kind:'geo', shape:'box', center:[...p.center], size:[...p.size], rotate:[...p.rotate] }) },
  ellipsoid: { title:'ellipsoid', cat:'regions', color:'hsl(25,75%,55%)', inputs:0, params:[
      { k:'center', label:'center µm', t:'vec3', def:[0,0,0] },
      { k:'radii', label:'radii µm', t:'vec3', def:[400,200,300] },
      { k:'rotate', label:'rotate deg', t:'vec3', def:[0,0,0] } ],
    compute:(ins,p)=>({ kind:'geo', shape:'ellipsoid', center:[...p.center], radii:[...p.radii], rotate:[...p.rotate] }) },
  cylinder: { title:'cylinder', cat:'regions', color:'hsl(25,75%,55%)', inputs:0, params:[
      { k:'center', label:'center µm', t:'vec3', def:[0,0,0] },
      { k:'radius', label:'radius µm', t:'float', def:200, min:1, s:[20,1500] },
      { k:'height', label:'height µm', t:'float', def:600, min:1, s:[50,4000] },
      { k:'rotate', label:'rotate deg', t:'vec3', def:[0,0,0] } ],
    compute:(ins,p)=>({ kind:'geo', shape:'cylinder', center:[...p.center], radius:p.radius, height:p.height, rotate:[...p.rotate] }) },
  torus: { title:'torus', cat:'regions', color:'hsl(25,75%,55%)', inputs:0, params:[
      { k:'center', label:'center µm', t:'vec3', def:[0,0,0] },
      { k:'radius', label:'ring radius µm', t:'float', def:400, min:1, s:[50,2500] },
      { k:'thickness', label:'thickness µm', t:'float', def:100, min:1, s:[10,600] },
      { k:'rotate', label:'rotate deg', t:'vec3', def:[0,0,0] } ],
    compute:(ins,p)=>({ kind:'geo', shape:'torus', center:[...p.center], radius:p.radius, thickness:p.thickness, rotate:[...p.rotate] }) },
  // A triangle mesh from an OBJ or glTF file in the project folder, as a region.
  // One file, one node per named object, each feeding its own scatter.
  mesh: { title:'mesh', cat:'regions', color:'hsl(25,70%,60%)', inputs:0, params:[
      { k:'file', label:'mesh file (OBJ or GLB, in the project)', t:'str', def:'' },
      { k:'object', label:'object (blank: all)', t:'str', def:'' },
      { k:'scale', label:'file units to µm', t:'float', def:1, min:1e-6, s:[0.001,1000] },
      { k:'up', label:'up axis', t:'select', def:0, options:['as authored (Y up)', 'Z up in the file'] },
      { k:'center', label:'position', t:'select', def:0, options:['as authored', 'centered at the origin'] },
      { k:'reload', label:'reload the file', t:'action', action:'meshReload' } ],
    compute: async (ins, p) => {
      const path = String(p.file || '').trim().replace(/^\/+/, '');
      if(!path) throw new Error('mesh: name a mesh file in the project folder, for example meshes/brain.obj or meshes/brain.glb');
      if(!fileReader) throw new Error('mesh: no file access in this context');
      let parsed = meshCache.get(path);
      if(!parsed){
        let bytes = null;
        try { bytes = await fileReader(path); } catch(e){ bytes = null; }
        if(!bytes || !bytes.length) throw new Error('mesh: ' + path + ' is not in the project folder');
        parsed = parseMeshFile(path, bytes);
        if(!parsed.objects.length) throw new Error('mesh: ' + path + ' has no triangles');
        meshCache.set(path, parsed);
      }
      const want = String(p.object || '').trim().toLowerCase();
      const objs = want ? parsed.objects.filter(o => o.name.toLowerCase() === want) : parsed.objects;
      if(!objs.length) throw new Error('mesh: no object named "' + p.object + '" in ' + path +
        '; it holds ' + parsed.objects.map(o => o.name).join(', '));
      return meshGeo(objs, p);
    } },
  spline: { title:'spline', cat:'regions', color:'hsl(25,75%,55%)', inputs:0, params:[
      { k:'p0', label:'point 0 µm', t:'vec3', def:[-800,0,0] },
      { k:'p1', label:'point 1 µm', t:'vec3', def:[-300,0,400] },
      { k:'p2', label:'point 2 µm', t:'vec3', def:[300,0,-400] },
      { k:'p3', label:'point 3 µm', t:'vec3', def:[800,0,0] },
      { k:'radius', label:'radius µm', t:'float', def:150, min:1, s:[20,800] } ],
    compute:(ins,p)=>({ kind:'geo', shape:'spline', p0:[...p.p0], p1:[...p.p1], p2:[...p.p2], p3:[...p.p3], radius:p.radius }) },
  noisefield: { title:'noise field', cat:'regions', color:'hsl(25,70%,60%)', inputs:0, params:[
      { k:'center', label:'center µm', t:'vec3', def:[0,0,0] },
      { k:'size', label:'bounds µm', t:'vec3', def:[2000,2000,2000] },
      { k:'pattern', label:'pattern', t:'select', def:0, options:['blobs (fBm)','patches','stripes'] },
      { k:'scale', label:'feature µm', t:'float', def:400, min:10, s:[50,1500] },
      { k:'coverage', label:'coverage', t:'float', def:0.4, min:0.05, max:1 },
      { k:'octaves', label:'octaves', t:'int', def:2, min:1, max:4, cond:q => (q.pattern|0) === 0 },
      { k:'direction', label:'stripe dir', t:'vec3', def:[1,0,0], cond:q => (q.pattern|0) === 2 },
      { k:'seed', label:'seed', t:'int', def:1, s:[1,99] } ],
    compute:(ins,p)=>({ kind:'geo', shape:'noisefield', center:[...p.center], size:[...p.size],
      pattern:p.pattern|0, scale:p.scale, coverage:p.coverage, octaves:p.octaves|0,
      direction:[...p.direction], seed:p.seed|0 }) },
  scatter: { title:'neuron scatter', stem:'scatter', cat:'cells', color:'hsl(190,75%,55%)', inputs:1, params:[
      // A population is a density in a volume before it is a count: 90,000 per mm3 is mouse cortex (Herculano-Houzel et al. 2013), and the resolution slider is what makes a smaller build of it.
      // Count is the authored number for a scene that wants one.
      { k:'fill', label:'fill', t:'select', def:1, options:['by count', 'at density (per mm3)'] },
      { k:'density', label:'density per mm3', t:'float', def:90000, min:1, s:[1000,150000], cond:q => (q.fill|0) === 1 },
      { k:'count', label:'count', t:'int', def:2000, min:1, s:[100,50000], cond:q => (q.fill|0) !== 1 },
      { k:'type', label:'neuron type', t:'select', def:0, options:NEURON_TYPES.map(t=>t.name), hideOption:i => !!(NEURON_TYPES[i] && NEURON_TYPES[i].hidden) },
      // identity: this tag names the population, so a new node adopts its name as the tag and the tag follows a rename (tagname.js).
      // The probe and stimulus tags are filters where blank means every cell, and must never be filled in by a name.
      { k:'tag', label:'population tag', t:'str', def:'', identity:true },
      { k:'hue', label:'color hue (-1: automatic)', t:'int', def:-1, min:-1, max:360 },
      { k:'spacing', label:'min spacing µm', t:'float', def:0, min:0, s:[0,50] },
      { k:'pattern', label:'pattern', t:'select', def:0, options:['uniform','minicolumns'] },
      { k:'pitch', label:'column pitch µm', t:'float', def:40, min:5, s:[10,120], cond:q => (q.pattern|0) === 1 },
      { k:'jitter', label:'strand jitter µm', t:'float', def:8, min:0, s:[0,30], cond:q => (q.pattern|0) === 1 },
      { k:'axis', label:'column axis', t:'select', def:1, options:['X','Y','Z'], cond:q => (q.pattern|0) === 1 },
      { k:'seed', label:'seed', t:'int', def:1, s:[1,99] } ],
    compute: scatterCompute },
  // Real positions from a file as a population, in place of a scatter: a CSV or PLY in the project folder.
  // Row order is the stable identity; an id column, when named, rides along for a connections table to refer to.
  pointfile: { title:'points file', cat:'cells', color:'hsl(190,75%,62%)', inputs:0, params:[
      { k:'file', label:'file (CSV or PLY, in the project)', t:'str', def:'' },
      { k:'columns', label:'x y z columns (blank: auto)', t:'str', def:'' },
      { k:'id', label:'id column (blank: none)', t:'str', def:'' },
      { k:'scale', label:'file units to µm', t:'float', def:1, min:1e-9, s:[0.001,1000] },
      { k:'up', label:'up axis', t:'select', def:0, options:['as authored (Y up)', 'Z up in the file'] },
      { k:'type', label:'neuron type', t:'select', def:0, options:NEURON_TYPES.map(t=>t.name), hideOption:i => !!(NEURON_TYPES[i] && NEURON_TYPES[i].hidden) },
      { k:'tag', label:'population tag', t:'str', def:'', identity:true },
      // one file, many populations: every distinct value of this column is its own population, tagged tag.value, with its own identity the way a repeat numbers its copies (a connectome's superclass, type, neuromere, side or transmitter column)
      { k:'tagColumn', g:'populations', label:'population per value of column (blank: one)', t:'str', def:'' },
      { k:'hue', g:'populations', label:'color hue (-1: automatic)', t:'int', def:-1, min:-1, max:360, cond:q => !String(q.tagColumn || '').trim() },
      { k:'filter', g:'populations', label:'keep rows where (column=value; ...)', t:'str', def:'' },
      // one node per population, laid out in a row and merged back, so each can be viewed and edited on its own
      { k:'expand', g:'populations', label:'expand into one node per population', t:'action', action:'expandPops',
        cond:(q, n, nodes) => !!String(q.tagColumn || '').trim() &&
          !(nodes || []).some(m => m.type === 'population' && m.inputs && m.inputs[0] && n && m.inputs[0].id === n.id) },
      { k:'collapse', g:'populations', label:'collapse the populations back', t:'action', action:'collapsePops',
        cond:(q, n, nodes) => (nodes || []).some(m => m.type === 'population' && m.inputs && m.inputs[0] && n && m.inputs[0].id === n.id) },
      { k:'reload', g:'populations', label:'reload the file', t:'action', action:'pointsReload' } ],
    compute: async (ins, p, node) => {
      const path = String(p.file || '').trim().replace(/^\/+/, '');
      if(!path) throw new Error('points file: name a CSV or PLY in the project folder, for example points/somas.csv');
      if(!fileReader) throw new Error('points file: no file access in this context');
      const key = path + '|' + String(p.columns || '') + '|' + String(p.id || '');
      let parsed = pointsCache.get(key);
      if(!parsed){
        let bytes = null;
        try { bytes = await fileReader(path); } catch(e){ bytes = null; }
        if(!bytes || !bytes.length) throw new Error('points file: ' + path + ' is not in the project folder');
        parsed = parsePointsFile(path, bytes, p.columns, String(p.id || '').trim());
        pointsCache.set(key, parsed);
      }
      // the rows the filter keeps: column=value terms, all of which must hold
      const field = name => { const f = parsed.fields && parsed.fields[name] !== undefined ? parsed.fields[name]
        : parsed.fields && Object.keys(parsed.fields).find(k => k.toLowerCase() === String(name).toLowerCase());
        return typeof f === 'string' ? parsed.fields[f] : f; };
      let keep = null;
      const terms = String(p.filter || '').split(/[;,]/).map(x => x.trim()).filter(Boolean);
      if(terms.length){
        keep = new Uint8Array(parsed.count).fill(1);
        for(const term of terms){
          const m = /^([^=!]+)(!=|=)(.*)$/.exec(term);
          if(!m) throw new Error('points file: filter "' + term + '" is not column=value');
          const col = field(m[1].trim()); const want = m[3].trim().toLowerCase(), neq = m[2] === '!=';
          if(!col) throw new Error('points file: no column named ' + m[1].trim() + '; the file has ' + parsed.columns.join(', '));
          // a blank cell reads as none, the name the population column gives it
          for(let i = 0; i < parsed.count; i++) if(keep[i] && (((String(col[i]).trim().toLowerCase() || 'none') === want) === neq)) keep[i] = 0;
        }
      }
      const kept = []; for(let i = 0; i < parsed.count; i++) if(!keep || keep[i]) kept.push(i);
      if(!kept.length) throw new Error('points file: the filter keeps no rows');
      const n = Math.max(1, Math.round(kept.length * simResolution));
      const s = +p.scale > 0 ? +p.scale : 1, swap = (p.up | 0) === 1;
      const pos = new Float32Array(n*3);
      // below full resolution, every k-th row, so the identity of a row is the same at every resolution
      const step = kept.length / n;
      const rows = new Uint32Array(n);
      for(let i = 0; i < n; i++) rows[i] = kept[Math.min(kept.length - 1, Math.floor(i*step))];
      for(let i = 0; i < n; i++){
        const r = rows[i];
        let x = parsed.pos[r*3]*s, y = parsed.pos[r*3+1]*s, z = parsed.pos[r*3+2]*s;
        if(swap){ const t = y; y = z; z = -t; }
        pos[i*3] = x; pos[i*3+1] = y; pos[i*3+2] = z;
      }
      const ntype = new Uint8Array(n); ntype.fill(p.type|0);
      const src = new Int32Array(n);
      const base = node ? node.id : -1;
      const tags = {};
      const tcol = String(p.tagColumn || '').trim();
      if(!tcol){ src.fill(base); tags[base] = p.tag || ''; }
      else {
        // one population per distinct value: its own src id, so its cells have their own identity and its tag is one the whole graph can name.
        // The cells come out grouped by population in first-seen order, rows in order within each, which is the order a fan of population nodes and a merge produces, so the two are wired alike.
        const col = field(tcol);
        if(!col) throw new Error('points file: no column named ' + tcol + '; the file has ' + parsed.columns.join(', '));
        const popOf = new Map(), kOf = new Int32Array(n);
        for(let i = 0; i < n; i++){
          const v = String(col[rows[i]] || '').trim() || 'none';
          let k = popOf.get(v);
          if(k === undefined){
            k = popOf.size;
            if(k >= 1000) throw new Error('points file: column ' + tcol + ' has more than 1000 distinct values; the pair table indexes populations pairwise, so tag by a coarser column');
            popOf.set(v, k);
            const sid = POINTS_SRC_BASE + base*POINTS_SRC_STRIDE + k;
            tags[sid] = (p.tag ? p.tag + '.' : '') + v.replace(/\s+/g, '_');
          }
          kOf[i] = k;
        }
        const order = Array.from({ length:n }, (_, i) => i).sort((a, b) => kOf[a] - kOf[b] || a - b);
        const rows2 = new Uint32Array(n), pos2 = new Float32Array(n*3);
        order.forEach((i, j) => { rows2[j] = rows[i]; pos2[j*3] = pos[i*3]; pos2[j*3+1] = pos[i*3+1]; pos2[j*3+2] = pos[i*3+2];
          src[j] = POINTS_SRC_BASE + base*POINTS_SRC_STRIDE + kOf[i]; });
        rows.set(rows2); pos.set(pos2);
      }
      const out = { kind:'points', count:n, pos, ntype, bias:new Float32Array(n), src,
        lidx:Uint32Array.from(rows), tags };
      if(parsed.ids) out.ids = Array.from(rows, r => parsed.ids[r]);
      { const hv = p.hue === undefined || p.hue === '' ? -1 : (p.hue|0); if(!tcol && hv >= 0 && p.tag) out.hues = { [p.tag]: hv }; }
      if(tcol){ const col = field(tcol); const seen = new Set();
        for(let i = 0; i < n; i++) seen.add(String(col[rows[i]] || '').trim() || 'none');
        out.popValues = [...seen]; }             // for the expand action
      return out;
    } },
  // The cell type of all, or a fraction, of a stream or of one population in it: a row of the type table, or a row of the scene's own with the 2007 form's numbers, spiking or graded, excitatory or inhibitory.
  // A custom row joins the type table by name, so a later computation with other numbers updates it and every engine reads the current numbers at init; the numbers ride on the stream so a change rewires the network where the sign matters.
  celltype: { title:'cell type', cat:'celltypes', color:'hsl(300,70%,66%)', inputs:1, params:[
      { k:'row', label:'cell type', t:'select', def:1,
        options:['custom (the numbers below)', ...CELL_PRESETS.map(t => t.name)],
        hideOption:i => i > 0 && !!(CELL_PRESETS[i - 1] && CELL_PRESETS[i - 1].hidden) },
      // blank is every cell; a tag (or a pattern with one *, or E or I) takes one population of a stream that carries many, so a file with a region column can give each region its own cell type
      { k:'tag', label:'population (blank: all)', t:'tag', def:'' },
      { k:'frac', label:'fraction', t:'float', def:1, min:0, max:1 },
      { k:'name', label:'name', t:'str', def:'mycell', cond:q => (q.row|0) === 0 },
      // The published rows as a starting point: choosing one writes every number below, then the numbers are the scene's to edit.
      // The defaults are the RS row of Izhikevich 2007 chapter 8.
      { k:'preset', label:'copy the numbers of', t:'select', def:0, cond:q => (q.row|0) === 0,
        options:['custom', ...CELL_PRESETS.map(t => t.name)],
        apply:(q, i) => {
          const t = CELL_PRESETS[i - 1];
          if(!t) return;                     // custom writes nothing
          const f = t.f7;
          Object.assign(q, { C:f.C, k:f.k, vr:f.vr, vt:f.vt, vpeak:f.vpeak, a:f.a, b:f.b, c:f.c, d:f.d,
            sign:t.sign < 0 ? 1 : 0, kind:f.graded ? 1 : 0,
            thr:f.graded ? f.graded.thr : q.thr, slope:f.graded ? f.graded.slope : q.slope });
        } },
      { k:'sign', label:'class', t:'select', def:0, options:['excitatory', 'inhibitory'], cond:q => (q.row|0) === 0 },
      { k:'kind', label:'output', t:'select', def:0, options:['spiking', 'graded (release, no spikes)'], cond:q => (q.row|0) === 0 },
      { k:'C', g:'membrane', label:'C pF', t:'float', def:100, min:0.1, s:[1,300], cond:q => (q.row|0) === 0 },
      { k:'k', g:'membrane', label:'k nS/mV', t:'float', def:0.7, min:0.001, s:[0.01,2], cond:q => (q.row|0) === 0 },
      { k:'vr', g:'membrane', label:'rest mV', t:'float', def:-60, s:[-90,-40], cond:q => (q.row|0) === 0 },
      { k:'vt', g:'membrane', label:'threshold mV', t:'float', def:-40, s:[-70,-20], cond:q => (q.row|0) === 0 },
      { k:'vpeak', g:'membrane', label:'peak mV', t:'float', def:35, s:[-40,60], cond:q => (q.row|0) === 0 },
      { k:'a', g:'recovery', label:'a', t:'float', def:0.03, min:0, s:[0,1], cond:q => (q.row|0) === 0 },
      { k:'b', g:'recovery', label:'b', t:'float', def:-2, s:[-5,15], cond:q => (q.row|0) === 0 },
      { k:'c', g:'recovery', label:'c reset mV', t:'float', def:-50, s:[-80,-30], cond:q => (q.row|0) === 0 },
      { k:'d', g:'recovery', label:'d', t:'float', def:100, s:[0,200], cond:q => (q.row|0) === 0 },
      { k:'thr', g:'release', label:'release threshold mV', t:'float', def:-55, s:[-80,-30], cond:q => (q.row|0) === 0 && (q.kind|0) === 1 },
      { k:'slope', g:'release', label:'release slope mV', t:'float', def:10, min:0.1, s:[1,30], cond:q => (q.row|0) === 0 && (q.kind|0) === 1 },
      { k:'hue', label:'color hue', t:'int', def:300, min:0, max:360, cond:q => (q.row|0) === 0 } ],
    compute:(ins, p, node) => {
      const pts = need(ins[0], 'points', 'cell type needs points');
      if((p.row|0) > 0) return applyCellType(pts, p, node, builtinRow(p.row|0), null);
      const name = String(p.name || '').trim().toLowerCase().replace(/[^a-z0-9_]/g, '');
      if(!name) throw new Error('cell type: give the row a name');
      if(NEURON_TYPES.some(t => !t.custom && t.key.toLowerCase() === name)) throw new Error('cell type: ' + name + ' is a built-in row; pick another name');
      const hue = ((p.hue|0) % 360 + 360) % 360;
      const rgb = hueRgb(hue);
      const row = { key:name, name:name + (p.kind|0 ? ' (graded, ' : ' (') + ((p.sign|0) === 1 ? 'inh)' : 'exc)'), custom:true,
        sign:(p.sign|0) === 1 ? -1 : 1, color:[rgb[0], rgb[1], rgb[2]],
        f7:{ C:+p.C || 100, k:+p.k || 0.7, vr:+p.vr, vt:+p.vt, vpeak:+p.vpeak, a:+p.a, b:+p.b, c:+p.c, d:+p.d,
          ...((p.kind|0) === 1 ? { graded:{ thr:+p.thr, slope:+p.slope > 0 ? +p.slope : 1 } } : {}) } };
      if(!(row.f7.vt > row.f7.vr)) throw new Error('cell type: the threshold has to sit above the rest');
      const at = registerCellType(row);
      row.index = at;                      // where the computation put it, for the wiring workers
      return applyCellType(pts, p, node, at, row);
    } },
  // Receptor channels: named synaptic time constants beyond the checkpoint's E and I, each with a sign and the transmitters that use it.
  // A connections table's transmitter column then picks each synapse's channel, and a transmitter the node does not name refuses the wiring.
  // One node per scene, exponential synapses only (MODEL.md 2).
  receptor: { title:'receptors', cat:'wiring', color:'hsl(140,50%,52%)', inputs:1, params:[
      { k:'table', label:'channels (name tau sign [reversal mV] transmitters...)', t:'text',
        def:'ach 3 + acetylcholine\ngabaa 8 - gaba\nglucl 30 - glutamate histamine' } ],
    compute:(ins, p) => {
      const pts = need(ins[0], 'points', 'receptors need points');
      if(pts.receptors) throw new Error('receptors: the stream already carries a receptor node; a scene has one');
      const channels = [];
      for(const raw of String(p.table || '').split(/\r?\n/)){
        const line = raw.replace(/#.*$/, '').trim(); if(!line) continue;
        const f = line.split(/\s+/);
        if(f.length < 4) throw new Error('receptors: each line is name, tau ms, sign (+ or -), then the transmitters: "' + line + '"');
        const tau = +f[1]; if(!(tau > 0)) throw new Error('receptors: ' + f[0] + ' needs a time constant in ms above zero');
        if(f[2] !== '+' && f[2] !== '-') throw new Error('receptors: ' + f[0] + ' needs a sign, + or -');
        // an optional reversal potential in mV after the sign, read in conductance mode; a number where a transmitter name would be
        let at = 3, erev;
        if(f.length > 4 && /^-?[0-9.]+$/.test(f[3])){ erev = +f[3]; at = 4; }
        if(f.length <= at) throw new Error('receptors: ' + f[0] + ' names no transmitter');
        channels.push({ name:f[0].toLowerCase(), tau, sign:f[2] === '+' ? 1 : -1, erev, transmitters:f.slice(at).map(x => x.toLowerCase()) });
      }
      if(!channels.length) throw new Error('receptors: no channel declared');
      if(channels.length > 6) throw new Error('receptors: at most six channels beyond E and I');
      const seen = new Set();
      for(const c of channels) for(const t of c.transmitters){ if(seen.has(t)) throw new Error('receptors: transmitter ' + t + ' is on two channels'); seen.add(t); }
      return { ...pts, receptors:{ channels } };
    } },
  // One population out of a stream, by tag: the cells keep their identity, so a fan of these from a points file and a merge back is the same tissue as the file node alone, with each population a node of its own.
  population: { title:'population', cat:'cells', color:'hsl(190,60%,50%)', inputs:1, params:[
      { k:'tag', label:'population', t:'tag', def:'' },
      { k:'hue', label:'color hue (-1: automatic)', t:'int', def:-1, min:-1, max:360 } ],
    compute:(ins, p) => {
      const pts = need(ins[0], 'points', 'population needs points');
      const tag = String(p.tag || '').trim();
      if(!tag) throw new Error('population: pick a population tag');
      const tags = pts.tags || {};
      const idx = [];
      for(let i = 0; i < pts.count; i++){
        if(tag === 'E' ? NEURON_TYPES[pts.ntype[i]].sign > 0 : tag === 'I' ? NEURON_TYPES[pts.ntype[i]].sign < 0
          : tagMatch(tags[pts.src[i]], tag) !== null) idx.push(i);
      }
      if(!idx.length) throw new Error('population: no cell is tagged ' + tag + '; the stream has ' +
        [...new Set(Object.values(tags))].filter(Boolean).join(', '));
      const out = pickPts(pts, idx);
      // a hue set here is the color every tag this node keeps wears
      const hv = p.hue === undefined || p.hue === '' ? -1 : (p.hue|0);   // -1 and absent: automatic
      if(hv >= 0){ out.hues = { ...(out.hues || {}) }; for(const t of new Set(idx.map(i => tags[pts.src[i]]))) if(t) out.hues[t] = hv; }
      return out;
    } },
  gather: { title:'gather', cat:'routing', color:'hsl(0,0%,55%)', inputs:4, portsParam:'ports', params:[
      { k:'ports', label:'inputs', t:'int', def:4, min:2, s:[2,16] } ],
    compute:(ins)=>mergeCompute(ins) },
  move: { title:'move', cat:'arrange', color:'hsl(270,75%,65%)', inputs:1, params:[
      { k:'translate', label:'translate µm', t:'vec3', def:[0,0,0] },
      { k:'rotate', label:'rotate deg', t:'vec3', def:[0,0,0] },
      { k:'scale', label:'scale', t:'vec3', def:[1,1,1] },
      { k:'pivot', label:'pivot µm', t:'vec3', def:[0,0,0] } ],
    carriesGeo:true,                       // so it can sit in a mask wire
    compute:(ins,p)=>{
      // a mask is a geometry, and moving the region is the same operation as moving the points inside it
      if(ins[0] && ins[0].kind === 'geo') return transformGeo(ins[0], p);
      const pts = need(ins[0], 'points', 'transform needs points or a geometry');
      const out = clonePts(pts);
      const rot = p.rotate.some(v => v) ? rotMat(...p.rotate.map(d => d*Math.PI/180)) : null;
      for(let i=0;i<out.count;i++){
        let x = out.pos[i*3]-p.pivot[0], y = out.pos[i*3+1]-p.pivot[1], z = out.pos[i*3+2]-p.pivot[2];
        x *= p.scale[0]; y *= p.scale[1]; z *= p.scale[2];
        if(rot){ const rx = rot[0]*x+rot[1]*y+rot[2]*z, ry = rot[3]*x+rot[4]*y+rot[5]*z,
                       rz = rot[6]*x+rot[7]*y+rot[8]*z; x=rx; y=ry; z=rz; }
        out.pos[i*3]   = x + p.pivot[0] + p.translate[0];
        out.pos[i*3+1] = y + p.pivot[1] + p.translate[1];
        out.pos[i*3+2] = z + p.pivot[2] + p.translate[2];
      }
      return out; } },
  // Copies of a stream, each stepped on from the last by one transform, so a row of columns, a fan, a stack or a mirrored pair is one node.
  // Every copy keeps a distinct identity: the local index is re-blocked per copy, so the pair hash draws each copy's wiring on its own and the merge's rule that no cell arrives twice holds.
  // A tag on the node renames the copies' populations (tag1.RSexc, tag2.RSexc, ...) under sources of their own, so they can be probed, projected and tabled apart; blank keeps every copy in the original populations.
  repeat: { title:'repeat', cat:'arrange', color:'hsl(270,70%,70%)', inputs:1, params:[
      { k:'copies', label:'copies', t:'int', def:2, min:2, s:[2,64] },
      { k:'pops', label:'populations', t:'select', def:0,
        options:['numbered per copy', 'shared across copies'] },
      { k:'tag', label:'tag copies as', t:'str', def:'', cond:q => (q.pops|0) === 0 },
      { k:'translate', label:'step translate µm', t:'vec3', def:[0,0,0] },
      { k:'rotate', label:'step rotate deg', t:'vec3', def:[0,0,0] },
      { k:'scale', label:'step scale', t:'vec3', def:[1,1,1] },
      { k:'pivot', label:'pivot µm', t:'vec3', def:[0,0,0] },
      { k:'mirror', label:'mirror alternate copies', t:'select', def:0,
        options:['no', 'in x', 'in y', 'in z'] } ],
    compute:(ins, p) => repeatCompute(need(ins[0], 'points', 'repeat needs points'), p) },
  noisewarp: { title:'noise warp', cat:'arrange', color:'hsl(270,75%,65%)', inputs:1, params:[
      { k:'amplitude', label:'amplitude µm', t:'float', def:200, min:0, s:[0,800] },
      { k:'scale', label:'feature µm', t:'float', def:500, min:10, s:[50,1500] },
      { k:'seed', label:'seed', t:'int', def:1, s:[1,99] } ],
    compute:(ins,p)=>{
      const pts = need(ins[0], 'points', 'noise warp needs points');
      const out = clonePts(pts);
      const fs = Math.max(10, p.scale), a = 2*p.amplitude, s = (p.seed|0)*7;
      for(let i=0;i<out.count;i++){
        const x = out.pos[i*3]/fs, y = out.pos[i*3+1]/fs, z = out.pos[i*3+2]/fs;
        out.pos[i*3]   += a*(fnoise(x, y, z, s+1)-0.5);
        out.pos[i*3+1] += a*(fnoise(x, y, z, s+2)-0.5);
        out.pos[i*3+2] += a*(fnoise(x, y, z, s+3)-0.5);
      }
      return out; } },
  twist: { title:'twist', cat:'arrange', color:'hsl(270,75%,65%)', inputs:1, params:[
      { k:'axis', label:'axis', t:'select', def:1, options:['X','Y','Z'] },
      { k:'angle', label:'deg / mm', t:'float', def:90, s:[-360,360] },
      { k:'center', label:'center µm', t:'vec3', def:[0,0,0] } ],
    compute:(ins,p)=>{
      const pts = need(ins[0], 'points', 'twist needs points');
      const out = clonePts(pts);
      const a = p.axis|0, b = (a+1)%3, c2 = (a+2)%3, k = p.angle*Math.PI/180/1000;
      for(let i=0;i<out.count;i++){
        const q = [out.pos[i*3]-p.center[0], out.pos[i*3+1]-p.center[1], out.pos[i*3+2]-p.center[2]];
        const ang = q[a]*k, co = Math.cos(ang), si = Math.sin(ang);
        const u = q[b]*co - q[c2]*si, v = q[b]*si + q[c2]*co;
        q[b] = u; q[c2] = v;
        out.pos[i*3] = q[0]+p.center[0]; out.pos[i*3+1] = q[1]+p.center[1]; out.pos[i*3+2] = q[2]+p.center[2];
      }
      return out; } },
  gaussblur: { title:'gauss blur', cat:'arrange', color:'hsl(270,75%,65%)', inputs:1, params:[
      { k:'sigma', label:'sigma µm', t:'float', def:50, min:0, s:[0,300] },
      { k:'seed', label:'seed', t:'int', def:1, s:[1,99] } ],
    compute:(ins,p)=>{
      const pts = need(ins[0], 'points', 'gauss blur needs points');
      const out = clonePts(pts);
      const r = rng((p.seed|0)*13+5);
      for(let i=0;i<out.count*3;i++) out.pos[i] += gaussRand(r)*p.sigma;
      return out; } },
  cull: { title:'cull', cat:'arrange', color:'hsl(270,75%,65%)', inputs:2, side:1, params:[
      { k:'keep', label:'keep', t:'select', def:0, options:['outside mask','inside mask'] } ],
    compute:(ins,p)=>{
      const pts = need(ins[0], 'points', 'cull needs points');
      const mask = ins[1] && ins[1].kind === 'geo' ? ins[1] : null;
      if(!mask) throw new Error('cull needs a geometry mask (side input)');
      const keepInside = (p.keep|0) === 1;
      const idx = [];
      for(let i=0;i<pts.count;i++)
        if(pointInGeo(mask, pts.pos[i*3], pts.pos[i*3+1], pts.pos[i*3+2]) === keepInside) idx.push(i);
      if(!idx.length) throw new Error('cull removed every point');
      return pickPts(pts, idx); } },
  gradient: { title:'gradient', cat:'arrange', color:'hsl(270,75%,65%)', inputs:1, params:[
      { k:'axis', label:'axis', t:'select', def:1, options:['X','Y','Z'] },
      { k:'start', label:'keep at min', t:'float', def:1, min:0, max:1 },
      { k:'end', label:'keep at max', t:'float', def:0.2, min:0, max:1 },
      { k:'seed', label:'seed', t:'int', def:1, s:[1,99] } ],
    // laminar-style density profile: keep probability ramps along an axis
    compute:(ins,p)=>{
      const pts = need(ins[0], 'points', 'gradient needs points');
      const a = p.axis|0;
      let lo = Infinity, hi = -Infinity;
      for(let i=0;i<pts.count;i++){ const v = pts.pos[i*3+a]; if(v<lo) lo=v; if(v>hi) hi=v; }
      const span = Math.max(1e-6, hi-lo), r = rng((p.seed|0)*17+3);
      const idx = [];
      for(let i=0;i<pts.count;i++){
        const t = (pts.pos[i*3+a]-lo)/span;
        if(r() < p.start + (p.end-p.start)*t) idx.push(i);
      }
      if(!idx.length) throw new Error('gradient removed every point');
      return pickPts(pts, idx); } },
  // A formula applied to every point: read the attributes flowing past, write new ones.
  // Position, drive and neuron class are writable; the stable identity is not, because pair-hash connectivity is keyed on (src, lidx) and rewriting either would silently rewire every synapse the graph has learned.
  ops: { title:'operations', cat:'arrange', color:'hsl(270,70%,70%)', inputs:1, params:[
      { k:'expr', label:'formula', t:'text',
        def:'# one assignment per line, for example:\n# y = y + sin(x*0.01)*40\nbias = bias' },
      { k:'a', label:'a', t:'float', def:0, s:[-100,100] },
      { k:'b', label:'b', t:'float', def:0, s:[-100,100] },
      { k:'c', label:'c', t:'float', def:0, s:[-100,100] },
      { k:'d', label:'d', t:'float', def:0, s:[-100,100] },
      { k:'seed', label:'seed', t:'int', def:1, s:[1,99] },
      // the control for anything read off position: the cells keep their identity and wiring and swap positions by a seeded permutation, so a stimulus mapped by position loses retinotopy and a readout that survives the shuffle was never spatial
      { k:'shuffle', label:'positions', t:'select', def:0, options:['as they are', 'shuffled among the cells (control)'] } ],
    compute:(ins,p)=>{
      const pts = need(ins[0], 'points', 'operations needs points');
      const out = clonePts(pts);
      const n = out.count, seed = (p.seed|0)*31 + 7;
      if((p.shuffle|0) === 1){
        const r = rng(seed ^ 0x5bd1e995), src = out.pos.slice(), perm = Array.from({ length:n }, (_, i) => i);
        for(let i = n - 1; i > 0; i--){ const j = Math.floor(r()*(i + 1)); const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
        for(let i = 0; i < n; i++){ out.pos[i*3] = src[perm[i]*3]; out.pos[i*3+1] = src[perm[i]*3+1]; out.pos[i*3+2] = src[perm[i]*3+2]; }
      }
      // rand is seeded from the point's stable identity rather than from a running counter, so a formula gives the same answer for the same neuron whatever order the points arrive in and however often the network is rewired: one stream per point, seeded from its source and local index, so successive calls in a formula differ.
      let cur = null;
      const rnd = () => cur();
      const fns = { rand:rnd, noise:(x,y,z) => fnoise(x, y, z, seed) };
      const READ = new Set(['x','y','z','bias','ntype','i','n','lidx','src','a','b','c','d']);
      const WRITE = new Set(['x','y','z','bias','ntype']);
      let ex;
      try { ex = buildExpr(p.expr || '', READ, WRITE, fns); }
      catch(e){ throw new Error('operations: ' + e.message); }
      const env = ex.env;
      env.n = n; env.a = p.a; env.b = p.b; env.c = p.c; env.d = p.d;
      const maxType = NEURON_TYPES.length - 1;
      for(let k=0;k<n;k++){
        cur = rng(h32(seed ^ h32(Math.imul(out.src[k] + 1, 0x9e3779b1) ^ out.lidx[k])));
        env.x = out.pos[k*3]; env.y = out.pos[k*3+1]; env.z = out.pos[k*3+2];
        env.bias = out.bias[k]; env.ntype = out.ntype[k];
        env.i = k; env.lidx = out.lidx[k]; env.src = out.src[k];
        ex.run();
        // a formula that produces a non-number would otherwise poison the network quietly: NaN positions vanish from the viewer and NaN drive makes a neuron never fire again
        const nx = env.x, ny = env.y, nz = env.z, nb = env.bias;
        if(!isFinite(nx) || !isFinite(ny) || !isFinite(nz) || !isFinite(nb))
          throw new Error('operations: the formula produced a non-finite value at point ' + k +
            ' (x=' + nx + ' y=' + ny + ' z=' + nz + ' bias=' + nb + ')');
        out.pos[k*3] = nx; out.pos[k*3+1] = ny; out.pos[k*3+2] = nz;
        out.bias[k] = nb;
        let t = Math.round(env.ntype);
        out.ntype[k] = t < 0 ? 0 : t > maxType ? maxType : t;
      }
      return out; } },
  stimulus: { title:'stimulus', cat:'drive', color:'hsl(55,75%,55%)', inputs:2, side:1, params:[
      { k:'center', label:'center µm', t:'vec3', def:[0,0,0], cond:(q,n) => !(n && n.inputs && n.inputs[1]) },
      { k:'radius', label:'radius µm', t:'float', def:400, min:1, s:[50,3000], cond:(q,n) => !(n && n.inputs && n.inputs[1]) },
      { k:'current', label:'current', t:'float', def:10, s:[-50,50] },
      { k:'tag', label:'population', t:'tag', def:'' },
      { k:'mode', label:'mode', t:'select', def:0, options:['constant','pulse train','ramp','noise'] },
      { k:'period', label:'period / ramp ms', t:'float', def:100, min:1, s:[10,3000], cond:q => (q.mode|0) === 1 || (q.mode|0) === 2 },
      { k:'width', label:'pulse width ms', t:'float', def:10, min:1, s:[1,300], cond:q => (q.mode|0) === 1 },
      { k:'t0', label:'onset ms', t:'float', def:0, min:0, s:[0,5000], cond:q => (q.mode|0) > 0 },
      { k:'duration', label:'duration ms (0=on)', t:'float', def:0, min:0, s:[0,10000], cond:q => (q.mode|0) > 0 },
      { k:'spread', label:'per-neuron spread', t:'float', def:0, min:0, s:[0,1.5] },
      { k:'seedSpread', label:'spread seed', t:'int', def:1, s:[1,999], cond:q => q.spread > 0 } ],
    compute: stimulusCompute },        // net->net (or points->points); input 2: optional geometry mask; * params used only without mask
  footage: { title:'footage', cat:'drive', color:'hsl(55,65%,58%)', inputs:0, params:[
      { k:'path', label:'folder path', t:'str', def:'' },
      { k:'choose', label:'choose folder…', t:'action', action:'chooseFolder' },
      { k:'fps', label:'frames / s', t:'float', def:12, min:0.1, s:[1,60] },
      { k:'loop', label:'playback', t:'select', def:1, options:['once','loop'] } ],
    // the compute emits only the descriptor: file access is asynchronous and permissioned, so the io runtime resolves the path on the main thread
    compute:(ins, p) => ({ kind:'signal', src:'footage', path:p.path, fps:p.fps, loop:p.loop|0 }) },
  // A curriculum is one stream of one sense.
  // Two streams of the same lesson are two curriculum nodes, the second wired into the first's side port so it follows that schedule and only says what it carries.
  curriculum: { title:'curriculum', cat:'drive', color:'hsl(55,60%,50%)', inputs:1, side:0,
    sideLabels:['follows'], params:[
      { k:'sense', g:'stream', label:'sense', t:'select', def:0, options:['sight', 'sound'] },
      { k:'acuity', g:'stream', label:'acuity', t:'float', def:1, min:0, max:1,
        cond:q => (q.sense|0) === 0 },
      { k:'jitter', g:'stream', label:'position jitter', t:'float', def:0.18, min:0, max:1,
        cond:q => (q.sense|0) === 0 },
      { k:'scaleVar', g:'stream', label:'size variation', t:'float', def:0.25, min:0, max:1,
        cond:q => (q.sense|0) === 0 },
      { k:'rotVar', g:'stream', label:'rotation deg', t:'float', def:12, min:0, s:[0,45],
        cond:q => (q.sense|0) === 0 },
      { k:'voice', g:'stream', label:'recorded voice', t:'select', def:0, options:['off','mix'],
        cond:q => (q.sense|0) === 1 },
      // a following stream can run ahead of or behind the schedule it follows, which is how the two senses of one lesson are staggered
      { k:'offset', g:'stream', label:'offset from followed (ms)', t:'float', def:0, s:[-200,200],
        cond:(q, n, nodes) => followsCurriculum(n, nodes) },
      // Every entry of SETS in curriculum.js has to appear here, or a scene on a set past the list shows an empty dropdown and any other edit silently resets the lesson.
      { k:'set', g:'lesson', label:'lesson set', t:'select', def:0,
        options:['alphabet A-Z','letters, phonics order','vowels A E I O U',
          'digits 0-9','shapes','objects (spoken names)','written words',
          'picture book (image + word)','spots and tones (abstract pairs)'],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      { k:'order', g:'lesson', label:'order', t:'select', def:1, options:['sequential','shuffled'],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      { k:'onMs', g:'lesson', label:'presentation ms', t:'float', def:400, min:20, s:[100,2000],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      { k:'offMs', g:'lesson', label:'gap ms', t:'float', def:350, min:0, s:[0,2000],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      { k:'phases', g:'lesson', label:'training blocks', t:'str', def:'', cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      { k:'scramble', g:'lesson', label:'scramble the pairing', t:'select', def:0, options:['no','yes'],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      // Where each item lands on the input: the whole sheet (glyphs and spectra overlap heavily), its own tile of the sheet and block of the bands (distinct, sparse, small), or its own offset (full size, shifted per item, partly overlapping).
      // The tile and offset stand in for a separation stage before the input.
      { k:'place', g:'lesson', label:'placement', t:'select', def:0,
        options:['whole sheet', 'own tile per item', 'own offset per item'],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      // Whether sound uses the same tile order as sight.
      // With the same order, position carries identity across the senses and a core sector both reach reads one item from the other by wiring alone (cross decoding 0.54 with every plasticity term off, 2026-09-09).
      // Shuffled, by the seed, no item shares a tile, so a cross-modal number has to be learned.
      { k:'placeOrder', g:'lesson', label:'tile order', t:'select', def:0,
        options:['same for both senses', 'shuffled for sound (by seed)'],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) && (q.place|0) > 0 },
      { k:'seed', g:'lesson', label:'seed', t:'int', def:1, s:[1,999], cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      { k:'probeEvery', g:'probes', label:'unimodal probe every', t:'int', def:0, min:0, s:[0,20],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      { k:'condNames', g:'probes', label:'probe condition names', t:'str', def:'both, sight, sound',
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) && (q.probeEvery|0) > 0 },
      { k:'calCycles', g:'probes', label:'calibration passes', t:'int', def:0, min:0, s:[0,5],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) },
      { k:'reSweepEvery', g:'probes', label:'re-sweep every (presentations)', t:'int', def:0, min:0, s:[0,5000],
        cond:(q, n, nodes) => !followsCurriculum(n, nodes) } ],
    // a generated lesson stream: the same schedule drives every input node wired to it (item, timing, and per-presentation jitter derive from the simulated clock and the seed), so a visual node and an audio node see the same letter at the same moment: synchronized multimodal input without any shared state (Bahrick and Lickliter intersensory redundancy).
    // The compute emits only the descriptor; rendering happens in the io runtime.
    compute:(ins, p) => {
      const up = ins[0];
      if(up && !(up.kind === 'signal' && up.src === 'curriculum'))
        throw new Error('a curriculum can only follow another curriculum');
      // the schedule: this node's own, or the one it follows
      const sched = up ? up : {
        set:p.set|0, order:p.order|0, onMs:p.onMs, offMs:p.offMs,
        probeEvery:p.probeEvery|0, calCycles:p.calCycles|0, reSweepEvery:p.reSweepEvery|0,
        // condition 0, 1, 2 ... named, so an analysis can say which cells it means in words the experiment uses rather than in indices only this scenario understands
        condNames:String(p.condNames || '').split(',').map(x => x.trim()).filter(Boolean),
        // a training schedule in blocks of simulated minutes, empty for the usual stream where every training trial drives every channel
        phasesText:String(p.phases || ''), phases:parsePhases(p.phases),
        scramble:(p.scramble|0) === 1, seed:p.seed|0, place:p.place|0, placeOrder:p.placeOrder|0 };
      return { ...sched, kind:'signal', src:'curriculum',
        sense:(p.sense|0) === 1 ? 'sound' : 'sight',
        jitter:p.jitter, scaleVar:p.scaleVar, rotVar:p.rotVar, acuity:p.acuity,
        voice:p.voice|0,
        // positive runs behind the followed stream, negative ahead of it
        offset:up ? (+p.offset || 0) : 0 };
    } },
  // Live media from the machine.
  // What arrives is what the device sees or hears; the input node it feeds decides the channels and where they land.
  live: { title:'live', cat:'drive', color:'hsl(55,60%,55%)', inputs:0, params:[
      { k:'device', label:'device', t:'select', def:0, options:['microphone','webcam'] } ],
    compute:(ins, p) => ({ kind:'signal', src:'live', device:(p.device|0) === 1 ? 'webcam' : 'microphone' }) },
  // A generated test pattern that needs no permissions and no files.
  testsignal: { title:'test signal', cat:'drive', color:'hsl(55,55%,48%)', inputs:0, params:[
      { k:'pattern', label:'pattern', t:'select', def:0, options:['bar sweep'] },
      { k:'period', label:'sweep period ms', t:'float', def:1500, min:100, s:[200,6000] },
      // the sweep runs both ways so a direction-selective cell (the fly's T4 and T5) can be read as its rate one way against the other; alternate is one sweep forward, the next backward, and the probes then report a rate per direction and the index between them
      { k:'direction', label:'direction', t:'select', def:0, options:['forward', 'backward', 'alternate every sweep'] } ],
    compute:(ins, p) => ({ kind:'signal', src:'sweep', period:p.period, direction:Math.max(0, Math.min(2, p.direction|0)) }) },
  // The input node is the mapping: which cells a signal lands on and how its channels lie over them.
  // What the signal is comes from the node on its signal port (curriculum, test signal, live, read).
  input: { title:'input', cat:'drive', color:'hsl(55,70%,62%)', inputs:3, side:1,
    sideLabels:['mask', 'signal'], params:[
      { k:'tag', g:'mapping', label:'population', t:'tag', def:'' },
      { k:'map', g:'mapping', label:'mapping', t:'select', def:0,
        options:['sheet (topographic 2D)','bands (tonotopic 1D)','distributed (random)'] },
      { k:'code', g:'coding', label:'coding', t:'select', def:1,
        options:['raw luminance','retinal contrast (ON/OFF)'] },
      { k:'axis', g:'mapping', label:'axis', t:'select', def:1, options:['x','y','z'], cond:q => (q.map|0) < 2 },
      { k:'cols', g:'mapping', label:'channels across', t:'int', def:16, min:2, s:[2,64] },
      { k:'rows', g:'mapping', label:'channels down', t:'int', def:16, min:2, s:[2,64], cond:q => (q.map|0) !== 1 },
      { k:'arbor', g:'mapping', label:'arbor sigma (cells)', t:'float', def:0.8, min:0, s:[0,2], cond:q => (q.map|0) < 2 },
      { k:'fanin', g:'mapping', label:'channels / neuron', t:'float', def:8, min:1, s:[1,32], cond:q => (q.map|0) === 2 },
      { k:'seed', g:'mapping', label:'seed', t:'int', def:1, s:[1,99], cond:q => (q.map|0) === 2 },
      { k:'transient', g:'coding', label:'adaptation', t:'float', def:0.6, min:0, max:1 },
      { k:'lagMs', g:'delivery', label:'arrival lag ms', t:'float', def:0, min:0, s:[0,200] },
      { k:'jitter', g:'coding', label:'microsaccade (cells)', t:'float', def:1, min:0, s:[0,3], cond:q => (q.map|0) !== 1 },
      // negative inverts the signal: the cells are driven in the dark and released by light, which is what the fly's lamina does (its ON pathway is a sign inversion at the first synapse)
      { k:'amp', g:'delivery', label:'current at gain 1 (negative inverts)', t:'float', def:6, s:[-30,30] } ],
    compute: inputCompute },           // net->net; side 1: region mask, side 2: the signal
  // Everything the readout does, as graph state, so the measurement travels with the scene and two people running the same graph get the same numbers.
  // The node is optional; a graph without one measures with the defaults.
  // A plot of what the run measured, in the app rather than in a file.
  // Like the analysis node it passes the network through untouched and only declares what to show, so it can sit anywhere in the readout chain and adding one never changes what is simulated. view: every setting on this node is display only, so the run lock leaves it live, since a chart is for watching a run develop.
  chart: { title:'chart', cat:'readout', color:'hsl(175,60%,55%)', inputs:1, view:true, params:[
      { k:'plot', g:'plot', label:'plot', t:'select', def:0,
        options:['a measure over time', 'weight distribution', 'response matrix',
          'similarity matrix', 'PSTH (items by time)', 'weight block matrix', 'confusion matrix',
          // the live plots read the page's recording of what the probes see (record.js): no training run needed
          'rate over time (live)', 'spike raster (live)', 'interval histogram (live)', 'membrane trace (selected cell)',
          // the REC recording: every probe rate since REC was pressed, with the edits marked
          'recording (rates and edits)'] },
      { k:'pop', g:'plot', label:'population', t:'str', def:'' },
      { k:'window', g:'plot', label:'window (s)', t:'float', def:5, min:0.1, s:[1,60], cond:q => (q.plot|0) >= 7 && (q.plot|0) <= 8 },
      { k:'smooth', g:'plot', label:'moving average (ms)', t:'float', def:0, min:0, s:[0,500], cond:q => (q.plot|0) === 7 || (q.plot|0) === 11 },
      { k:'which', g:'plot', label:'condition', t:'select', def:0,
        options:['both senses', 'sight alone', 'sound alone', 'sight against sound'],
        cond:q => (q.plot|0) === 3 || (q.plot|0) === 4 },
      { k:'field', g:'plot', label:'measure', t:'str', def:'decode',
        cond:q => (q.plot|0) === 0 },
      { k:'bins', g:'plot', label:'bins', t:'int', def:40, min:4, s:[8,120],
        cond:q => (q.plot|0) === 1 },
      { k:'height', g:'plot', label:'height px', t:'int', def:130, min:60, s:[80,400] },
      // Which run to draw.
      // Empty is the run this page is doing or last did; otherwise a run tag from the project folder.
      // It is a parameter rather than a mode of the panel because it belongs to the document: a scene that draws a particular run says so when it is reopened.
      { k:'source', g:'plot', label:'run', t:'str', def:'' },
      // How the plot reads, on the panel and in the saved figure alike.
      // Blank text is the plot's own; a blank range end follows the data.
      { k:'title', g:'look', label:'title', t:'str', def:'' },
      { k:'xText', g:'look', label:'x axis title (blank: automatic)', t:'str', def:'' },
      { k:'yText', g:'look', label:'y axis title (blank: automatic)', t:'str', def:'' },
      { k:'yMin', g:'look', label:'range from (blank: automatic)', t:'str', def:'' },
      { k:'yMax', g:'look', label:'range to (blank: automatic)', t:'str', def:'' },
      { k:'grid', g:'look', label:'grid lines', t:'select', def:1, options:['off', 'on'] },
      { k:'legend', g:'look', label:'legend', t:'select', def:1, options:['off', 'on'] },
      { k:'cmap', g:'look', label:'color map', t:'select', def:0,
        options:['the theme\'s', 'viridis', 'gray (white to black)', 'teal (black to white)'],
        cond:q => (q.plot|0) >= 2 && (q.plot|0) <= 6 },
      // The figure that leaves the app: drawn again at the size, resolution and style set here (chartpaint.js), not copied from the panel, whose canvas is a few hundred pixels wide.
      // Sizes are physical because that is how a journal or a slide states them; the font and line sizes are in points on the page whatever the resolution.
      { k:'preview', g:'figure', label:'panel shows', t:'select', def:0, options:['the working plot', 'the figure as it will be saved'] },
      { k:'format', g:'figure', label:'format', t:'select', def:0, options:['PNG', 'SVG (vector)', 'both'] },
      { k:'sizePreset', g:'figure', label:'size', t:'select', def:0,
        options:['custom', 'one column (85 x 60 mm)', 'one and a half columns (114 x 80 mm)', 'two columns (180 x 100 mm)', 'slide, 16:9 (254 x 143 mm)', 'square (120 x 120 mm)'],
        apply:(q, i) => { const s = [null, [85, 60], [114, 80], [180, 100], [254, 143], [120, 120]][i]; if(s){ q.figW = s[0]; q.figH = s[1]; } } },
      { k:'figW', g:'figure', label:'width (mm)', t:'float', def:180, min:20, max:1000, s:[40,300] },
      { k:'figH', g:'figure', label:'height (mm)', t:'float', def:100, min:15, max:1000, s:[30,300] },
      { k:'dpi', g:'figure', label:'resolution (dots per inch)', t:'int', def:300, min:72, max:1200, s:[72,600], cond:q => (q.format|0) !== 1 },
      { k:'theme', g:'figure', label:'theme', t:'select', def:0, options:['light (print)', 'light, transparent background', 'dark'] },
      { k:'figFont', g:'figure', label:'font', t:'select', def:0, options:['sans serif (Arial, Helvetica)', 'serif (Times)', 'monospace'] },
      { k:'fontPt', g:'figure', label:'font size (pt)', t:'float', def:8, min:4, max:24, s:[5,14] },
      { k:'linePt', g:'figure', label:'line width (pt)', t:'float', def:1, min:0.1, max:6, s:[0.25,3] },
      { k:'caption', g:'figure', label:'provenance', t:'select', def:0, options:['a caption under the figure', 'in the file only'] },
      { k:'savePng', g:'figure', label:'save figure', t:'action', action:'chartPng' },
      { k:'saveCsv', g:'figure', label:'save data', t:'action', action:'chartCsv' } ],
    // A chart declares what to draw and nothing else, so it passes the network through untouched.
    // It belongs after the checkpoint: what it draws is what the checkpoint's run produced, and a node that reads a run's output sitting upstream of the node that produces it was the one place in the graph where the wire ran against the data.
    compute:(ins, p) => ins[0] },
  analysis: { title:'analysis', cat:'readout', color:'hsl(175,50%,42%)', inputs:1, params:[
      { k:'cap', label:'cells sampled per probe', t:'int', def:8192, min:64, s:[256,16384] },
      { k:'topQ', label:'response set: top fraction', t:'float', def:0.1, min:0.01, max:0.5,
        cond:q => (q.doBind|0) === 1 },
      { k:'setExpr', label:'cells of interest', t:'str',
        def:'both and not sight and not sound',
        cond:q => (q.doBind|0) === 1 },
      { k:'probeWith', label:'test recall with', t:'str', def:'sight, sound',
        cond:q => (q.doBind|0) === 1 || (q.doDecode|0) === 1 },
      { k:'margin', label:'conjunction rank margin', t:'float', def:0.15, min:0, max:0.9,
        cond:q => (q.doBind|0) === 1 },
      { k:'minCells', label:'timing needs at least (cells)', t:'int', def:4, min:2, s:[2,64],
        cond:q => (q.doTiming|0) === 1 },
      { k:'minTrials', label:'decode needs per item (trials)', t:'int', def:2, min:1, s:[1,20],
        cond:q => (q.doDecode|0) === 1 },
      { k:'nPerm', label:'cross-decode permutations', t:'int', def:40, min:0, s:[0,500],
        cond:q => (q.doDecode|0) === 1 },
      { k:'doDecode', label:'decode', t:'select', def:1, options:['off','on'] },
      // these three run inside the decode pass
      { k:'doBind', label:'conjunction and binding', t:'select', def:1, options:['off','on'],
        cond:q => (q.doDecode|0) === 1 },
      { k:'doTiming', label:'spike timing agreement', t:'select', def:1, options:['off','on'],
        cond:q => (q.doDecode|0) === 1 },
      { k:'doLike', label:'like-to-like connectivity', t:'select', def:1, options:['off','on'],
        cond:q => (q.doDecode|0) === 1 },
      { k:'exportTrials', label:'export the trial matrix', t:'select', def:0,
        options:['off','on (recording probes)'] } ],
    compute:(ins, p) => {
      const net = ins[0];
      if(!net || net.kind !== 'net') return net;
      return { ...net, analysis: { cap:Math.max(64, p.cap|0), nPerm:Math.max(0, p.nPerm|0),
        setExpr:String(p.setExpr || ''),
        probeWith:String(p.probeWith || '').split(',').map(x => x.trim()).filter(Boolean),
        topQ:Math.min(0.5, Math.max(0.01, +p.topQ)),
        margin:Math.min(0.9, Math.max(0, +p.margin)),
        minCells:Math.max(2, p.minCells|0), minTrials:Math.max(1, p.minTrials|0),
        doDecode:(p.doDecode|0) === 1, doBind:(p.doBind|0) === 1,
        exportTrials:(p.exportTrials|0) === 1,
        doTiming:(p.doTiming|0) === 1, doLike:(p.doLike|0) === 1 } };
    } },
  probe: { title:'probe', cat:'readout', color:'hsl(175,60%,48%)', inputs:2, side:1, params:[
      { k:'label', label:'label', t:'str', def:'' },
      { k:'tag', label:'population', t:'tag', def:'' },
      { k:'record', label:'record for analysis', t:'select', def:0, options:['off','on'] },
      // view: display settings, left live by the run lock
      { k:'view', label:'view', t:'select', def:0, options:['rate','image (map)'], view:true },
      { k:'audio', label:'audio', t:'select', def:0, options:['off','spikes (crackle)','bands (tones)'], view:true },
      { k:'axis', label:'axis', t:'select', def:1, options:['x','y','z'], view:true,
        cond:q => (q.view|0) === 1 || (q.audio|0) === 2 },
      { k:'cols', label:'cells across / bands', t:'int', def:24, min:2, s:[2,64], view:true,
        cond:q => (q.view|0) === 1 || (q.audio|0) === 2 },
      { k:'rows', label:'cells down', t:'int', def:24, min:2, s:[2,64], view:true, cond:q => (q.view|0) === 1 } ],
    compute: probeCompute },           // net->net; input 2: optional geometry mask; readout only
  // Two ports: the source population on the first, the target on the second, both passed through together like a merge with the projection on top.
  // With only the first port wired the populations are named by tag.
  project: { title:'project', cat:'wiring', color:'hsl(140,65%,50%)', inputs:2, params:[
      { k:'from', label:'from population', t:'tag', def:'' },
      { k:'to', label:'to population', t:'tag', def:'' },
      { k:'axisFrom', label:'source axis', t:'select', def:1, options:['x','y','z'] },
      { k:'axisTo', label:'target axis', t:'select', def:1, options:['x','y','z'] },
      { k:'dispersed', label:'mapping', t:'select', def:0,
        options:['topographic','dispersed (convergent)','proximity','matched (by order)'] },
      { k:'flipU', label:'mirror u', t:'select', def:0, options:['no','yes'], cond:q => !(q.dispersed|0) },
      { k:'flipV', label:'mirror v', t:'select', def:0, options:['no','yes'], cond:q => !(q.dispersed|0) },
      { k:'sigma', label:'spread µm (target)', t:'float', def:60, min:1, s:[10,400],
        cond:q => (q.dispersed|0) !== 1 },
      { k:'prob', label:'probability', t:'float', def:0.35, min:0, max:1,
        cond:q => (q.dispersed|0) !== 1 },
      // dispersed: how many target cells each source reaches, as a count, since a probability over the whole target population scales with its size and read as the topographic default made 3,500 afferents per cell
      { k:'fanout', label:'contacts per source', t:'int', def:50, min:1, s:[1,2000],
        cond:q => (q.dispersed|0) === 1 },
      // Inside a repeat: a projection from copy k to copy k plus this offset (bulb below to bulb above is offset 1 across the tier repeat).
      // The repeat named by across applies it; blank, the first numbered repeat the projection passes through does.
      { k:'offset', label:'to copy k plus', t:'int', def:0, min:-63, s:[-3,3] },
      { k:'across', label:'across repeat (tag)', t:'str', def:'', cond:q => (q.offset|0) !== 0 },
      { k:'weight', label:'weight', t:'float', def:0.5, min:0, s:[0,20] },
      { k:'velocity', label:'axon vel µm/ms', t:'float', def:5000, min:1, s:[100,20000] },
      { k:'tract', label:'tract length µm (0=straight)', t:'float', def:0, min:0, s:[0,20000] },
      { k:'frozen', label:'plasticity', t:'select', def:0, options:['plastic','frozen'] },
      { k:'rule', label:'learns by rule', t:'str', def:'',
        cond:q => (q.frozen|0) === 0 },
      { k:'seed', label:'seed', t:'int', def:1, s:[1,99] } ],
    // topographic tract between tagged populations, consumed by connect in the same wiring; place between merge and connect
    compute:(ins, p, node) => {
      const A = need(ins[0], 'points', 'project needs points (place before connect)');
      const B = ins[1] && ins[1].kind === 'points' ? ins[1] : null;
      let pts = A, from = p.from, to = p.to;
      // blank from or to means the one population on that port; a port carrying several has to be told which, loudly
      const only = (s, port, key) => {
        const u = [...new Set(Object.values(s.tags || {}))];
        if(u.length === 1) return u[0];
        throw new Error('project: port ' + port + ' carries ' + (u.length ? u.join(', ') : 'no tagged population') + '; choose one in ' + key);
      };
      if(B && B !== A){
        pts = mergeCompute([A, B]);
        if(!String(from || '').trim()) from = only(A, 'A', 'from');
        if(!String(to || '').trim()) to = only(B, 'B', 'to');
      } else if(B){
        // one population on both ports: a pathway from each copy to another copy of itself, which the offset below names
        if(!String(from || '').trim()) from = only(A, 'A', 'from');
        if(!String(to || '').trim()) to = from;
      }
      return { ...pts, projs:[...(pts.projs || []), {
        node:node ? node.id : -1,           // which node made it, for the pair table
        from, to, axisFrom:p.axisFrom|0, axisTo:p.axisTo|0,
        offset:p.offset|0, across:String(p.across || '').trim(),
        flipU:(p.flipU|0) === 1, flipV:(p.flipV|0) === 1,
        dispersed:(p.dispersed|0) === 1, proximity:(p.dispersed|0) === 2,
        matched:(p.dispersed|0) === 3,
        sigma:p.sigma, prob:p.prob, fanout:Math.max(1, p.fanout|0), weight:p.weight,
        velocity:p.velocity, tract:p.tract, frozen:(p.frozen|0) === 1,
        rule:String(p.rule || '').trim(),
        seed:p.seed|0 }] }; } },
  // A named plasticity rule.
  // The checkpoint's learning parameters are global, one rule for every synapse in the network, while the per-synapse gate can only say plastic or frozen per pathway, which is half a mechanism.
  // The biology is specific that the missing half is real: timing dependent LTD at layer 4 to layer 2/3 needs presynaptic NMDA receptors and has a longer window, while the same measurement at layer 2/3 to layer 2/3 in the same tissue needs postsynaptic ones (Larsen et al. 2014; Banerjee et al. 2014).
  // Rules also differ by the postsynaptic cell type reached from one axon (Buchanan et al. 2012), which is the long-term counterpart of the classic short-term result that two terminals of one axon release with an order of magnitude different probability depending on their target (Markram et al. 1998; Reyes et al. 1998).
  //
  // Scope is optional and is the whole point of the node's ergonomics.
  // With from and to left blank the rule is declared and nothing more, and the pair table refers to it by name.
  // With them filled in the node scopes itself, which is the case almost everyone wants: drop the node, pick two populations, done, without opening a table of 197 rows.
  // Both forms land in the same list and resolve by the same specificity, so there is one cascade rather than two systems that have to agree.

  plasticity: { title:'plasticity', cat:'wiring', color:'hsl(140,55%,58%)', inputs:1, params:[
      // Published rules, named, rather than a row of amplitudes to fill in from memory.
      // Choosing one writes the terms below and names the rule, which is the name the connect pair table's fifth column refers to, so a scene reads 'weavei weave inh' and the node beside it says inhibitory homeostasis in words.
      // Editing any term afterwards is ordinary: the preset is a starting point, not a mode, and the panel shows what it wrote.
      { k:'preset', g:'scope', label:'preset', t:'select', def:0,
        options:['custom', 'pair STDP', 'triplet STDP', 'inhibitory homeostasis',
                 'heterosynaptic competition', 'consolidation', 'frozen'],
        apply:(q, i) => {
          const R = RULE_PRESETS[i];
          if(!R) return;                     // custom writes nothing
          // Read the name before writing anything: a preset supplies the handle the pair table uses, but a name chosen for this scene is the scene's, and the pair table is already referring to it.
          const cur = String(q.name || '').trim();
          const owned = cur === '' || cur === 'rule' ||
            RULE_PRESETS.some(x => x && x.set.name === cur);
          Object.assign(q, R.set);
          if(!owned) q.name = cur;
        } },
      { k:'name', g:'scope', label:'rule name', t:'str', def:'rule' },
      { k:'from', g:'scope', label:'from population', t:'str', def:'' },
      { k:'to', g:'scope', label:'to population', t:'str', def:'' },
      { k:'aP', g:'stdp', label:'stdp A+', t:'float', def:0.008, min:0, s:[0,0.05] },
      { k:'aM', g:'stdp', label:'stdp A-', t:'float', def:0.001, min:0, s:[0,0.05] },
      { k:'wmax', g:'stdp', label:'weight limit', t:'float', def:30, min:0.1, s:[1,60] },
      { k:'wdep', g:'stdp', label:'weight bounds', t:'select', def:1,
        options:['hard (additive)','soft (weight dependent)'] },
      { k:'trip', g:'stdp', label:'triplet A3+', t:'float', def:0, min:0, s:[0,0.02] },
      { k:'iEta', g:'homeostasis', label:'homeo rate eta', t:'float', def:0.002, min:0, s:[0,0.02] },
      { k:'het', g:'more', label:'heterosynaptic beta', t:'float', def:0, min:0, s:[0,0.05] },
      { k:'tin', g:'more', label:'transmitter delta', t:'float', def:0, min:0, s:[0,0.001] },
      { k:'cons', g:'more', label:'consolidation', t:'float', def:0, min:0, s:[0,1] },
      { k:'consW', g:'more', label:'consolidated weight', t:'float', def:0, min:0, s:[0,20],
        cond:q => +q.cons > 0 },
      { k:'consP', g:'more', label:'well depth', t:'float', def:10, min:0, s:[0,50],
        cond:q => +q.cons > 0 } ],
    // Passes the stream through and appends itself, the same way project accumulates projs.
    // Place it anywhere before connect, since connect is what wires the per-synapse rule assignment.
    //
    // The parameters here are the ones that can differ per synapse: amplitudes and bounds.
    // Time constants are not among them, and that is a limitation rather than an oversight. tauS and tauY govern Kpre and Kslow, which are per-neuron traces, so two rules wanting different windows onto the same cell would each need their own copy of that state.
    // The set points and normalizations, iRho, rhoMode, calS, scale and sEta, belong to the postsynaptic neuron for the same reason, and the short-term terms belong to the presynaptic one.
    // All of those stay on the checkpoint and are global.
    // See buildRuleTable.
    compute:(ins, p) => { const pts = need(ins[0], 'points',
        'plasticity needs points (place before connect)');
      const name = String(p.name || '').trim();
      if(!name) throw new Error('plasticity: the rule needs a name, since the pair table refers to it by one');
      return { ...pts, rules:[...(pts.rules || []), {
        name, from:String(p.from || '').trim(), to:String(p.to || '').trim(),
        aP:+p.aP, aM:+p.aM, wmax:+p.wmax, wdep:p.wdep|0, trip:+p.trip,
        iEta:+p.iEta, het:+p.het, tin:+p.tin,
        cons:+p.cons, consW:+p.consW, consP:+p.consP }] }; } },
  // A lumped inhibitory pool per population: every spike in the population inhibits every cell of it one millisecond later, by gain times the population's mean rate in Hz.
  // The mushroom body's APL neuron and the dentate's basket cell feedback, which hold a stage sparse whatever arrives (Lin et al. 2014).
  // Pass-through like the plasticity node; the wiring resolves it (wirePools) and every engine steps it the same way.
  pool: { title:'feedback inhibition (pool)', cat:'wiring', color:'hsl(140,60%,40%)', inputs:1, params:[
      { k:'tag', label:'population', t:'tag', def:'' },
      { k:'gain', label:'inhibition per Hz', t:'float', def:-2, max:0, s:[-10,0] } ],
    compute:(ins, p, node) => { const pts = need(ins[0], 'points',
        'feedback inhibition needs points (place before connect)');
      return { ...pts, pools:[...(pts.pools || []), {
        tag:String(p.tag || '').trim(), gain:Math.min(0, +p.gain || 0),
        node:node ? node.id : -1 }] }; } },
  // Explicit synapses from a connections table, on top of the pair-hash wiring: the fly connectome's shape.
  // Pass-through like the plasticity node; connect appends what the table names among the cells that carry ids from a points file.
  connectionsfile: { title:'connections file', cat:'wiring', color:'hsl(140,65%,50%)', inputs:1, params:[
      { k:'file', label:'table (CSV, in the project)', t:'str', def:'' },
      { k:'pre', g:'columns', label:'presynaptic id (blank: auto)', t:'str', def:'' },
      { k:'post', g:'columns', label:'postsynaptic id (blank: auto)', t:'str', def:'' },
      { k:'count', g:'columns', label:'count (blank: auto, else 1)', t:'str', def:'' },
      { k:'signCol', g:'columns', label:'transmitter (blank: auto)', t:'str', def:'' },
      { k:'weight', g:'synapse', label:'weight per synapse', t:'float', def:0.5, min:0, s:[0,20] },
      { k:'wInh', g:'synapse', label:'inhibitory weight factor', t:'float', def:1, min:0, s:[0,3] },
      { k:'mode', g:'synapse', label:'a row with count c is', t:'select', def:0,
        options:['one synapse of weight × c', 'c synapses of the weight'] },
      // a pair table's counts run from one to thousands; what the weight multiplies is a choice, and the strongest pairs decide the dynamics
      { k:'compress', g:'synapse', label:'count taken as', t:'select', def:0,
        options:['c', 'square root of c', 'log2 of 1 + c', 'one (every pair alike)'] },
      { k:'cap', g:'synapse', label:'count ceiling (0: none)', t:'float', def:0, min:0, s:[0,200] },
      { k:'signMode', g:'synapse', label:'sign', t:'select', def:0,
        options:['by the presynaptic cell type', 'by the transmitter column (GABA, glutamate, histamine inhibit)'] },
      { k:'velocity', g:'synapse', label:'axon vel µm/ms', t:'float', def:200, min:1, s:[50,5000] },
      { k:'rule', g:'synapse', label:'plasticity rule (blank: default)', t:'str', def:'' },
      { k:'reload', g:'synapse', label:'reload the table', t:'action', action:'connectionsReload' } ],
    compute: async (ins, p, node) => {
      const pts = need(ins[0], 'points', 'connections file needs points (place before connect)');
      const path = String(p.file || '').trim().replace(/^\/+/, '');
      if(!path) throw new Error('connections file: name a connections table in the project folder, for example tables/connections.csv');
      if(!fileReader) throw new Error('connections file: no file access in this context');
      const cols = { pre:String(p.pre || '').trim(), post:String(p.post || '').trim(), count:String(p.count || '').trim(), sign:String(p.signCol || '').trim() };
      const key = path + '|' + JSON.stringify(cols);
      let table = connectionsCache.get(key);
      if(!table){
        let bytes = null;
        try { bytes = await fileReader(path); } catch(e){ bytes = null; }
        if(!bytes || !bytes.length) throw new Error('connections file: ' + path + ' is not in the project folder');
        table = parseConnectionsTable(new TextDecoder().decode(bytes), cols);
        connectionsCache.set(key, table);
      }
      const rule = String(p.rule || '').trim().toLowerCase();
      return { ...pts, connectionTables:[...(pts.connectionTables || []), {
        node: node ? node.id : -1, table, weight:+p.weight || 0, mode:p.mode|0, signMode:p.signMode|0,
        wInh:Number.isFinite(+p.wInh) ? Math.max(0, +p.wInh) : 1,
        compress:p.compress|0, cap:+p.cap > 0 ? +p.cap : 0,
        velocity:+p.velocity > 0 ? +p.velocity : 200,
        rule: rule === 'frozen' || rule === 'off' || rule === '0' ? '' : String(p.rule || '').trim(),
        frozen: rule === 'frozen' || rule === 'off' || rule === '0' }] };
    } },
  connect: { title:'connect', cat:'wiring', color:'hsl(140,75%,45%)', inputs:1, memo:true, params:[
      // Paired-recording numbers (MODEL.md 3): a Gaussian of width 100 um (Levy and Reyes 2012), 0.15 at zero distance for pyramid pairs (Holmgren 2003, Song 2005), a unitary EPSP of half a millivolt and an IPSP of one and a half, lognormal (Song 2005).
      // The radius is wide enough that the kernel, not the cutoff, ends the arbor.
      { k:'radius', label:'max dist µm', t:'float', def:500, min:1, s:[50,1500] },
      { k:'sigma', label:'sigma µm', t:'float', def:100, min:1, s:[20,400] },
      { k:'prob', label:'probability', t:'float', def:0.15, min:0, max:1 },
      { k:'wExc', label:'weight exc', t:'float', def:0.5, s:[0,20] },
      { k:'wInh', label:'weight inh', t:'float', def:-1.5, s:[-30,0] },
      { k:'wdist', label:'weight dist', t:'select', def:1, options:['uniform ±50%','lognormal'] },
      { k:'wsigma', label:'lognorm sigma', t:'float', def:1, min:0.1, s:[0.1,2], cond:q => (q.wdist|0) === 1 },
      { k:'cluster', label:'cluster boost', t:'float', def:0, min:0, max:1 },
      { k:'table', label:'pair table', t:'pairs', def:'' },
      { k:'velocity', label:'axon vel µm/ms', t:'float', def:200, min:1, s:[50,1000] },
      { k:'density', label:'density % of real', t:'float', def:100, min:1, max:100 },
      { k:'kscale', label:'in-degree % of full scale', t:'float', def:100, min:1, max:100 },
      { k:'nuE', label:'target exc Hz (0=auto)', t:'float', def:0, min:0, s:[0,30],
        cond:q => q.density < 100 || q.kscale < 100 },
      { k:'nuI', label:'target inh Hz (0=auto)', t:'float', def:0, min:0, s:[0,30],
        cond:q => q.density < 100 || q.kscale < 100 },
      { k:'seed', label:'seed', t:'int', def:1, s:[1,99] } ],
    compute: connectCompute },
  // A reroute handle: passes its input straight through so wires can be bent around a crowded graph instead of running diagonally across it.
  // It is invisible to computation by construction (its compute returns the input unchanged) and invisible to the memo signature by the bypass rule in sigOf, so dropping pins into a full-density graph to tidy it up never costs a rewiring of the network. passthrough is what lets it sit in a mask wire as well as a main one: the editor asks for this flag rather than naming the type.
  pin: { title:'pin', cat:'routing', color:'hsl(0,0%,55%)', inputs:1, passthrough:true, params:[],
    compute:(ins) => ins[0] },
  note: { title:'note', cat:'notes', color:'hsl(55,30%,45%)', inputs:0, params:[
      { k:'text', label:'text', t:'text', def:'note' },
      { k:'width', label:'width px', t:'int', def:220, min:100, max:500 },
      // the scene layout puts a note beside the group whose label starts with this; blank notes go in the margin column
      { k:'near', label:'beside group (blank: margin)', t:'str', def:'' } ],
    compute:()=>({ kind:'note' }) },        // annotation only; no ports, never computed
  // Settings for an engine of one's own that the tune message has no field for: one key and one value per line, sent to the engine as custom on init and tune (ENGINE.md section 2).
  // It sits on the network between connect and the checkpoint, so an edit tunes the running engine and never rewires.
  enginesettings: { title:'engine settings', cat:'run', color:'hsl(0,75%,55%)', inputs:1, params:[
      { k:'table', label:'settings (key value per line)', t:'text', def:'' } ],
    compute:(ins, p) => {
      const net = need(ins[0], 'net', 'engine settings take a connected network: place the node between connect and the checkpoint');
      const custom = { ...(net.custom || {}) };
      const mine = parseEngineSettings(p.table);
      for(const k of Object.keys(mine)){
        if(Object.prototype.hasOwnProperty.call(custom, k)) throw new Error('engine settings: ' + k + ' is already set by an engine settings node above this one');
        custom[k] = mine[k];
      }
      return { ...net, custom };
    } },
  // term: the hello term (TERMS in protocol.js) a setting belongs to; the panel grays the row when the running engine does not list it.
  checkpoint: { title:'checkpoint', cat:'run', color:'hsl(0,75%,55%)', inputs:1, params:[
      { k:'steps', g:'engine', label:'steps / frame', t:'int', def:2, min:1, max:20 },
      { k:'engine', g:'engine', label:'engine', t:'select', def:0,
        options:['auto','cpu (reference)','gpu (experimental)','remote (engine address)'] },
      // where the remote engine listens: host/host.mjs, or an engine of one's own that speaks the protocol (ENGINE.md, writing an engine)
      { k:'liveHost', g:'engine', label:'engine address', t:'str', def:'ws://localhost:8801',
        cond:q => (q.engine|0) === 3 },
      // conductance: the exp decay with a reversal potential per channel, so a synapse drives g (E - v): excitation saturates as the cell depolarizes and inhibition near rest shunts (divides) rather than subtracts (MODEL.md 2)
      { k:'syn', g:'engine', label:'synapse', t:'select', def:0, options:['kick (instant)','exp decay','conductance (exp with reversal)'] },
      { k:'tauE', g:'engine', term:['syn:1','syn:2'], label:'tau exc ms', t:'float', def:3, min:0.5, s:[0.5,20], cond:q => (q.syn|0) >= 1 },
      { k:'tauI', g:'engine', term:['syn:1','syn:2'], label:'tau inh ms', t:'float', def:8, min:0.5, s:[0.5,30], cond:q => (q.syn|0) >= 1 },
      { k:'eRevE', g:'engine', term:'syn:2', label:'E reversal mV', t:'float', def:0, s:[-20,20], cond:q => (q.syn|0) === 2 },
      { k:'eRevI', g:'engine', term:'syn:2', label:'I reversal mV', t:'float', def:-70, s:[-90,-40], cond:q => (q.syn|0) === 2 },
      // What w means in exp mode. peak: the current jumps by w and decays, so the charge per spike is w tau (NEST iaf_psc_exp, Brian). charge: the jump is scaled by (1 - e^(-1/tau)) so the discrete sum over steps is exactly w, the same charge kick mode delivers. exact: w is the peak, and the current over a step is the integral of the exponential across it, so the charge is w tau exactly at 1 ms.
      { k:'psc', g:'engine', term:['psc:1','psc:2'], label:'exp current scale', t:'select', def:0,
        options:['peak: w is the peak current (NEST, Brian)', 'charge: w is the total charge per spike', 'exact: w is the peak, integrated exactly over each step'],
        cond:q => (q.syn|0) === 1 },
      { k:'refrac', g:'engine', term:'refrac', label:'refractory ms', t:'int', def:2, min:0, s:[0,5] },
      // The potassium reversal potential as a floor on the membrane: no hyperpolarizing current takes v below it.
      // The Izhikevich quadratic grows with the square of the distance below rest and sends a membrane pushed to about -150 mV (classic rows) over threshold or to infinity, which a synchronous inhibitory volley can do (measured 2026-09-09: a feedback pool's onset kick took every core cell to NaN).
      // 0 disables the floor.
      { k:'vmin', g:'engine', term:'vmin', label:'membrane floor mV', t:'float', def:-90, s:[-120,-60] },
      // Training settings live on the checkpoint because a training run is this checkpoint left running with plasticity on: same network, same rule, longer clock, and a saved scene records how it is trained.
      { k:'plast', g:'timing', term:'plast', label:'plasticity', t:'select', def:0, options:['off','STDP + homeostasis'] },
      // A+ and A- set where the excitatory population relaxes to, not just how fast it learns.
      // Under the weight-dependent bound below the fixed point is A+/A-, so these default to 0.008 / 0.001, which is a mean weight of 8: the default excitatory weight on the connect node.
      // A depression stronger than potentiation under hard bounds has no fixed point and walks every excitatory weight to zero on any uncorrelated network.
      { k:'aP', g:'timing', term:'plast', label:'stdp A+', t:'float', def:0.008, min:0, s:[0,0.05], cond:q => (q.plast|0) === 1 },
      { k:'aM', g:'timing', term:'plast', label:'stdp A-', t:'float', def:0.001, min:0, s:[0,0.05], cond:q => (q.plast|0) === 1 },
      // The two window time constants, Bi and Poo 1998: potentiation decays with about 16.8 ms, depression with about 33.7.
      // Song, Miller and Abbott 2000 use 20 for both; set them equal to recover that.
      { k:'tauS', g:'timing', term:'plast', label:'stdp tau+ ms', t:'float', def:16.8, min:1, s:[5,100], cond:q => (q.plast|0) === 1 },
      { k:'tauM', g:'timing', term:'plast', label:'stdp tau- ms', t:'float', def:33.7, min:1, s:[5,100], cond:q => (q.plast|0) === 1 },
      // The clamp rewrites anything above this on its first plasticity event, so it has to clear the largest wired weight, not the mean.
      // Default 30 rather than 10: the default inhibitory weight is -7 and a lognormal draw carries it several times higher.
      { k:'trip', g:'timing', term:'trip', label:'triplet A3+', t:'float', def:0, min:0, s:[0,0.02],
        cond:q => (q.plast|0) === 1 },
      { k:'tauY', g:'timing', term:'trip', label:'triplet tau ms', t:'float', def:114, min:10, s:[50,300],
        cond:q => (q.plast|0) === 1 && q.trip > 0 },
      { k:'tin', g:'timing', term:'tin', label:'transmitter delta', t:'float', def:0, min:0, s:[0,0.001],
        cond:q => (q.plast|0) === 1 },
      // Absolute refractory period.
      // Default 2 ms, which is sodium channel inactivation in a cortical pyramidal cell; with nothing stopping a next-millisecond spike a saturated network reaches 979 Hz sustained.
      // It caps a cell at 1000/(refrac+1) Hz, so it bounds a runaway without touching a network that is already in a physiological range.
      { k:'wmax', g:'timing', term:'plast', label:'weight limit', t:'float', def:30, min:0.1, s:[1,60], cond:q => (q.plast|0) === 1 },
      // Soft by default.
      // Additive spike-timing plasticity has no interior fixed point: with hard bounds every excitatory weight walks to one end and the recurrent structure dissolves.
      // The weight-dependent form (van Rossum, Bi and Turrigiano 2000) settles at A+/A- instead.
      { k:'wdep', g:'timing', term:'plast', label:'weight bounds', t:'select', def:1,
        options:['hard (clamped)', 'soft (weight-dependent)'],
        cond:q => (q.plast|0) === 1 },
      { k:'iEta', g:'homeostasis', term:'plast', label:'homeo rate eta', t:'float', def:0.002, min:0, s:[0,0.02], cond:q => (q.plast|0) === 1 },
      { k:'rhoMode', g:'homeostasis', term:'rhoMode', label:'rate set points', t:'select', def:0,
        options:['fixed target','measured baseline (per neuron)'], cond:q => (q.plast|0) === 1 },
      { k:'calS', g:'homeostasis', term:'rhoMode', label:'calibration s', t:'float', def:10, min:2, s:[5,60],
        cond:q => (q.plast|0) === 1 && (q.rhoMode|0) === 1 },
      { k:'iRho', g:'homeostasis', term:'plast', label:'homeo target Hz', t:'float', def:5, min:0.1, s:[0.5,30],
        cond:q => (q.plast|0) === 1 && (q.rhoMode|0) === 0 },
      { k:'scale', g:'homeostasis', term:'scale', label:'synaptic scaling', t:'select', def:0,
        options:['off','toward target rate'], cond:q => (q.plast|0) === 1 },
      { k:'sEta', g:'homeostasis', term:'scale', label:'scaling rate /s', t:'float', def:0.001, min:0, s:[0,0.01],
        cond:q => (q.plast|0) === 1 && (q.scale|0) === 1 },
      { k:'het', g:'competition', term:'het', label:'heterosynaptic beta', t:'float', def:0, min:0, s:[0,0.05],
        cond:q => (q.plast|0) === 1 },
      { k:'cons', g:'competition', term:'cons', label:'consolidation', t:'float', def:0, min:0, s:[0,1],
        cond:q => (q.plast|0) === 1,
        info:'Zenke 2015 eq 16: the heterosynaptic reference weight follows the weight through a double well instead of staying frozen at its value when plasticity began. Without it competition can only decay synapses toward their initial state.' },
      { k:'consW', g:'competition', term:'cons', label:'consolidated weight', t:'float', def:0, min:0, s:[0,20],
        cond:q => (q.plast|0) === 1 && q.cons > 0 },
      { k:'consP', g:'competition', term:'cons', label:'well depth', t:'float', def:10, min:0, s:[0,50],
        cond:q => (q.plast|0) === 1 && q.cons > 0 },
      // These three govern consolidation network-wide, and a plasticity node can switch it on for one pathway while this node's own control reads zero, so they follow consolidation anywhere in the graph rather than this node's copy of it.
      { k:'tauCons', g:'competition', term:'cons', label:'commit after (sim-min)', t:'float', def:20,
        min:0.5, s:[1, 120],
        cond:(q, n, nodes) => (q.plast|0) === 1 && (q.cons > 0 ||
          (nodes || []).some(x => x.type === 'plasticity' && x.params.cons > 0)) },
      { k:'consStep', g:'competition', term:'cons', label:'commit check every (sim-s)', t:'float', def:1.2,
        min:0.05, s:[0.1, 10],
        cond:(q, n, nodes) => (q.plast|0) === 1 && (q.cons > 0 ||
          (nodes || []).some(x => x.type === 'plasticity' && x.params.cons > 0)) },
      { k:'commit', g:'competition', term:'commit', label:'freeze consolidated', t:'select', def:0,
        options:['off', 'on'],
        cond:(q, n, nodes) => (q.plast|0) === 1 && (q.cons > 0 ||
          (nodes || []).some(x => x.type === 'plasticity' && x.params.cons > 0)) },
      // The two published synapse classes of Markram, Wang and Tsodyks (1998, PNAS 95:5323): depressing (pyramid to pyramid, U 0.5, recovery 800 ms, no facilitation) and facilitating (pyramid to bitufted interneuron, U 0.03, recovery 130 ms, facilitation 530 ms).
      // The defaults are the depressing class; the term is off until the amount above zero.
      { k:'stpPreset', g:'short-term', term:'stp', label:'synapse class', t:'select', def:0,
        options:['custom', 'depressing (Markram 1998)', 'facilitating (Markram 1998)'],
        apply:(q, i) => {
          if(i === 1) Object.assign(q, { stpU:0.5, stpTauD:800, stpTauF:1 });
          else if(i === 2) Object.assign(q, { stpU:0.03, stpTauD:130, stpTauF:530 });
        } },
      { k:'stp', g:'short-term', term:'stp', label:'short-term plasticity', t:'float', def:0, min:0, s:[0,1],
        info:'Tsodyks-Markram short-term plasticity, per presynaptic neuron: resources deplete with each spike and recover with tau_d, release probability facilitates and decays with tau_f. Short-term depression makes what arrives sublinear in presynaptic rate, which is the curvature that puts a stable fixed point at intermediate rates (Zenke et al. 2015 equations 9 and 10).' },
      { k:'stpU', g:'short-term', term:'stp', label:'release probability U', t:'float', def:0.5, min:0.01, s:[0.01,1],
        cond:p => p.stp > 0 },
      // As published, a rested synapse releases u x = U of its weight on the first spike, so switching the term on rescales transmission and the weights are tuned with it in mind.
      // The scaled option multiplies every release by the inverse of the rested release so a rested synapse delivers exactly w and only the dynamics change.
      { k:'stpNorm', g:'short-term', term:'stpNorm', label:'stp delivery', t:'select', def:0,
        options:['as published: a rested synapse releases U', 'scaled: a rested synapse delivers w'],
        cond:p => p.stp > 0 },
      // Release before the facilitation jump is the published order (Tsodyks, Pawelzik and Markram 1998; Zenke et al. 2015 and Auryn).
      // Facilitation first releases U(2 - U) from rest; scenes tuned on it declare it (scene format 9).
      { k:'stpOrder', g:'short-term', term:'stpOrder', label:'stp order', t:'select', def:0,
        options:['release, then facilitation (published)', 'facilitation, then release'],
        cond:p => p.stp > 0 },
      { k:'stpTauD', g:'short-term', term:'stp', label:'depression tau (ms)', t:'float', def:800, min:1, s:[10,1000],
        cond:p => p.stp > 0 },
      { k:'stpTauF', g:'short-term', term:'stp', label:'facilitation tau (ms)', t:'float', def:1, min:1, s:[10,2000],
        cond:p => p.stp > 0 },
      // The sweep: variants of the graph, each a list of edits to nodes above this one, run one after another on a copy of the graph per run.
      // A tab here rather than a node, because a node may only reach what is above it and every node is above the checkpoint.
      { k:'variants', g:'sweep', label:'variants', t:'sweeps', def:[] },
      { k:'trHours', g:'run', label:'train sim-hours', t:'float', def:16, min:0.05, s:[0.25,48] },
      { k:'trCkptMin', g:'run', label:'metrics every (sim-min)', t:'float', def:2, min:0.2, s:[0.2,30] },
      { k:'trBrainMin', g:'run', label:'brain file every (sim-min)', t:'float', def:5, min:1, s:[1,60] },
      { k:'trKeep', g:'run', label:'brain files kept', t:'int', def:4, min:1, s:[1,20] },
      { k:'trReps', g:'run', label:'replicates', t:'int', def:1, min:1, s:[1,10] },
      { k:'trSweep', g:'run', label:'conditions', t:'select', def:0,
        options:['as configured', 'paired and scrambled'] },
      // one variant per line, each a list of edits applied to a copy of the graph before its run: selector.param=value; ... A selector is a node name, a name prefix ending in *, #type for every node of a type, or @tag for the scatter carrying that population tag.
      { k:'trEngine', g:'run', label:'train on', t:'select', def:0,
        options:['gpu', 'cpu (reference)', 'cuda (remote host)'] },
      // A host is a long-lived process that keeps the code it loaded, so a second one on another port is how a run gets current code while an older host is still busy
      { k:'trHost', g:'run', label:'engine host', t:'str', def:'ws://localhost:8801',
        cond:q => (q.trEngine | 0) === 2 },
      { k:'train', g:'run', label:'START TRAINING', t:'action', action:'train' },
      { k:'runs', g:'run', label:'BROWSE RUNS', t:'action', action:'runs' },
      // A brain is this checkpoint's weights, so saving and loading one are actions of this node.
      { k:'brainSave', g:'run', label:'SAVE BRAIN', t:'action', action:'brainSave' },
      { k:'brainLoad', g:'run', label:'LOAD BRAIN', t:'action', action:'brainLoad' },
      // the network leaves the app: as tables, and as a case Brian 2 runs with the engine's second implementation (tools/brian_ref.py)
      { k:'exportNet', g:'export', label:'save network (CSV)', t:'action', action:'exportNet' },
      { k:'exportSecs', g:'export', label:'Brian 2 run length (s)', t:'float', def:2, min:0.01, s:[0.1,60] },
      { k:'exportBrian', g:'export', label:'save for Brian 2', t:'action', action:'exportBrian' } ] ,
    compute:(ins,p)=>{ const net = need(ins[0],'net','output needs a connected network');
      return { ...net, bias:compensationBias(net, p), steps:p.steps, engine:p.engine|0, liveHost:String(p.liveHost || '').trim(),
        syn:p.syn|0, tauE:p.tauE, tauI:p.tauI,
        psc:p.psc|0,
        plast:p.plast|0, aP:p.aP, aM:p.aM, tauS:p.tauS, tauM:p.tauM, wmax:p.wmax, wdep:p.wdep|0,
        iEta:p.iEta, iRho:p.iRho, rhoMode:p.rhoMode|0, calS:p.calS,
        scale:p.scale|0, sEta:p.sEta,
        trip:p.trip || 0, tauY:p.tauY || 114, het:p.het || 0, tin:p.tin || 0,
        cons:p.cons || 0, consW:p.consW || 0, consP:p.consP || 10,
        // the commit switch and its two timings reach the engine through engineConfig, which reads them off the network
        commit:p.commit|0, tauCons:+p.tauCons || 20, consStep:+p.consStep || 1.2,
        // the node's own defaults, the Markram 1998 depressing class
        stp:p.stp || 0, stpU:p.stpU || 0.5, stpNorm:p.stpNorm|0, stpOrder:p.stpOrder|0,
        stpTauD:p.stpTauD || 800, stpTauF:p.stpTauF || 1,
        refrac:p.refrac|0,
        // the membrane floor and the reversal potentials have to reach the network, or every engine runs -90, 0 and -70 whatever the panel says; src/engineparams.test.mjs moves every engine-facing setting and requires the engine config to follow
        vmin:+p.vmin, eRevE:+p.eRevE, eRevI:+p.eRevI }; } },
};

// ---- nodes from outside the repository ----------------------------------------
// A node a project brings with it (src/plugins.js loads the module) joins NODE_DEFS here, so the Tab menu, the properties panel, the node reference and the setting hover find it as they find a built-in one.
// The categories the Tab menu and the node reference list, in their order; a node in any other category would be listed nowhere.
export const NODE_CATS = ['regions', 'cells', 'arrange', 'celltypes', 'wiring', 'drive', 'readout', 'run', 'routing', 'notes'];
// The setting kinds the properties panel draws a control for.
export const PARAM_TYPES = ['float', 'int', 'select', 'str', 'text', 'pairs', 'vec3', 'action', 'tag', 'sweeps'];
const BUILTIN_NODES = new Set(Object.keys(NODE_DEFS));
export const isBuiltinNode = key => BUILTIN_NODES.has(key);

// What is wrong with a node's documentation, as a list of sentences (empty when nothing is): a blurb, and prose for every setting and for nothing else.
// The built-in nodes are held to this by docs.test.mjs, and a plugin's node by registerNode.
export function nodeDocProblems(key, def, doc){
  const out = [];
  if(!doc || typeof doc !== 'object') return [key + ' has no documentation'];
  if(!(typeof doc.blurb === 'string' && doc.blurb.length > 20)) out.push(key + ' needs a blurb of more than 20 characters');
  // A blurb is one short sentence and a setting's note one short line; the reference table shows defaults and ranges beside them.
  const words = s => s.trim().split(/\s+/).filter(Boolean).length;
  if(typeof doc.blurb === 'string' && words(doc.blurb) > 20) out.push('blurb of ' + key + ' is ' + words(doc.blurb) + ' words; 20 at most');
  const keys = new Set((def.params || []).map(p => p.k));
  const prose = doc.params || {};
  for(const k of keys) if(!(typeof prose[k] === 'string' && prose[k].trim())) out.push('setting ' + key + '.' + k + ' has no prose');
  for(const k of Object.keys(prose)) if(!keys.has(k)) out.push('prose for ' + key + '.' + k + ', which is not a setting');
  for(const [k, t] of Object.entries(prose))
    if(typeof t === 'string' && words(t) > 25) out.push('setting ' + key + '.' + k + ' is ' + words(t) + ' words; 25 at most');
  return out;
}

// The documentation of a node: the built-in prose in docs.js (passed as docs), or the doc a plugin registered with its node.
export function nodeDoc(type, docs){
  return (docs && docs[type]) || (NODE_DEFS[type] && NODE_DEFS[type].doc) || null;
}

// Add a node from outside the repository: the same shape as a NODE_DEFS entry (title, cat, color, inputs, params, compute) plus doc, the prose docs.js keeps for a built-in node ({ io, blurb, params:{ key: text } }).
// The third argument names the module the node came from and a hash of its source; src/plugins.js passes it, and a scene that uses the node records the module (scenePlugins there).
// A key a built-in node has, or one another module registered, is refused; the same module registering again replaces its own node, which is how a changed module reloads.
export function registerNode(key, def, from = { module:'(registered directly)', hash:'' }){
  const where = 'registerNode ' + JSON.stringify(key) + ' from ' + from.module;
  if(typeof key !== 'string' || !/^[A-Za-z][A-Za-z0-9_]*$/.test(key))
    throw new Error(where + ': a node key is letters, digits and underscores, starting with a letter');
  if(BUILTIN_NODES.has(key))
    throw new Error(where + ': ' + key + ' is a built-in node; give the plugin node another key');
  const prev = NODE_DEFS[key];
  if(prev && prev.plugin && prev.plugin.module !== from.module)
    throw new Error(where + ': ' + key + ' is already registered by ' + prev.plugin.module + '; one of the two modules has to use another key');
  if(!def || typeof def !== 'object') throw new Error(where + ': the definition is an object');
  const bad = [];
  if(!(typeof def.title === 'string' && def.title.trim())) bad.push('a title');
  if(!NODE_CATS.includes(def.cat)) bad.push('cat, one of ' + NODE_CATS.join(', '));
  if(typeof def.color !== 'string') bad.push('a color (a CSS color string)');
  if(!(Number.isInteger(def.inputs) && def.inputs >= 0 && def.inputs <= 16)) bad.push('inputs, a whole number from 0 to 16');
  if(def.side !== undefined && !(Number.isInteger(def.side) && def.side >= 0 && def.side < def.inputs)) bad.push('side, the index of the first side port, below inputs');
  if(typeof def.compute !== 'function') bad.push('compute, a function (ins, params, node) returning a stream');
  if(!Array.isArray(def.params)) bad.push('params, an array (empty for none)');
  else {
    const seen = new Set();
    for(const p of def.params){
      const k = p && p.k;
      if(typeof k !== 'string' || !k){ bad.push('a k on every setting'); continue; }
      if(seen.has(k)) bad.push('setting ' + k + ' once only');
      seen.add(k);
      if(typeof p.label !== 'string') bad.push('a label on setting ' + k);
      if(!PARAM_TYPES.includes(p.t)) bad.push('setting ' + k + ' of a known kind (t: ' + PARAM_TYPES.join(', ') + ')');
      else if(p.t === 'select' && !Array.isArray(p.options)) bad.push('options on the select setting ' + k);
      else if(p.t !== 'action' && p.def === undefined) bad.push('a def on setting ' + k);
    }
  }
  if(bad.length) throw new Error(where + ': the definition needs ' + bad.join('; '));
  const docBad = nodeDocProblems(key, def, def.doc);
  if(docBad.length) throw new Error(where + ': ' + docBad.join('; '));
  NODE_DEFS[key] = { ...def, plugin:{ module:from.module, hash:String(from.hash || '') } };
  return NODE_DEFS[key];
}
// Put back what a key held before a module that failed half way registered it (src/plugins.js): the earlier version of the same module's node, or nothing.
export function restoreNode(key, def){
  if(BUILTIN_NODES.has(key)) return;
  if(def) NODE_DEFS[key] = def; else delete NODE_DEFS[key];
}

// The engine settings node's table: one key and one value per line, # starts a comment.
// A value that reads as a number is sent as a number, anything else as the text after the key.
export function parseEngineSettings(text){
  const out = {};
  for(const raw of String(text || '').split(/\r?\n/)){
    const line = raw.replace(/#.*$/, '').trim();
    if(!line) continue;
    const m = /^(\S+)\s+(.+)$/.exec(line);
    if(!m) throw new Error('engine settings: each line is a key and a value: "' + line + '"');
    const k = m[1], v = m[2].trim();
    if(!/^[A-Za-z_][\w.:-]*$/.test(k) || k === '__proto__')
      throw new Error('engine settings: a key is letters, digits and _ . : -, starting with a letter or _: "' + k + '"');
    if(Object.prototype.hasOwnProperty.call(out, k)) throw new Error('engine settings: ' + k + ' is set twice');
    out[k] = /^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(v) ? +v : v;
  }
  return out;
}

// The proxy compensation applied under the checkpoint's synapse convention (finishConnect carries it per unit charge).
// The charge one spike delivers per unit of weight, as the engine delivers it at its 1 ms step: a kick synapse delivers w; an exponential one in the peak convention (psc 0) holds the accumulator for each step and sums to w / (1 - e^(-1/tau)), 3.53 w at tauE 3; the charge convention (psc 1) delivers w; the exact convention (psc 2) delivers w tau; a conductance synapse (nS) delivers about w (E_rev - v_rest) / (1 - e^(-1/tau)) with the cell at rest, which is the mean input the compensation is restoring.
// MODEL.md 3.
export function chargePerWeight(p, sign, vrest){
  const syn = p.syn|0, tau = sign > 0 ? (p.tauE > 0 ? p.tauE : 3) : (p.tauI > 0 ? p.tauI : 8);
  if(syn === 0) return 1;
  const held = 1/(1 - Math.exp(-1/tau));
  if(syn === 2){
    const erev = sign > 0 ? (Number.isFinite(+p.eRevE) ? +p.eRevE : 0) : (Number.isFinite(+p.eRevI) ? +p.eRevI : -70);
    // g (E - v) with |g| the weight: the sign of the current is the weight's
    return held*Math.abs(erev - vrest);
  }
  const psc = p.psc|0;
  return psc === 1 ? 1 : psc === 2 ? tau : held;
}
export function compensationBias(net, p){
  if(!net.comp) return net.bias;
  const bias = net.bias.slice(), { e, i:ic } = net.comp, n = net.count;
  const cond = (p.syn|0) === 2;
  for(let k=0;k<n;k++){
    const vr = cond ? (NEURON_TYPES[net.ntype[k]] && NEURON_TYPES[net.ntype[k]].f7 ? NEURON_TYPES[net.ntype[k]].f7.vr : -65) : 0;
    bias[k] += e[k]*chargePerWeight(p, 1, vr) + ic[k]*chargePerWeight(p, -1, vr);
  }
  return bias;
}

// ---- memo LRU for heavy computations (revisiting a previously computed
// graph state returns the stored result instead of computing it again) ---- Sound because computation is deterministic: the result is a pure function of the upstream graph signature.
// Only states actually computed get stored, so cost is memory (bounded below), never extra compute time. navigator.deviceMemory is privacy-clamped to a max of 8, so >=8 means "8GB or (much) more"
const MEMO_BYTES = (typeof navigator !== 'undefined' && navigator.deviceMemory >= 8) ? 1.5e9 : 4e8;
// ---- what a stream is, for the wiring cache ----------------------------------
// The wiring is a pure function of the points stream and the connect params: pair existence and weight come from the two cells' stable identities and the seed, projections, rules, pools and connections tables ride on the stream.
// So the cache is keyed on the stream's content, not on the shape of the graph that made it: the same cells in the same order with the same entries reuse the wiring whatever nodes sit above the connect node (an expanded fan of populations and its collapse).
// Positions are hashed exactly, so a moved cell is a new tissue, as it should be.
function fnv(h, bytes){
  for(let i = 0; i < bytes.length; i++){ h ^= bytes[i]; h = Math.imul(h, 16777619) >>> 0; }
  return h;
}
const bytesOf = a => a ? new Uint8Array(a.buffer, a.byteOffset, a.byteLength) : new Uint8Array(0);
function hashText(h, s){ return fnv(h, new TextEncoder().encode(String(s))); }
function hashTable(t){                     // a connections table, hashed once and kept on it
  if(t._hash) return t._hash;
  let h = 2166136261 >>> 0;
  h = fnv(h, bytesOf(t.pre)); h = fnv(h, bytesOf(t.post)); h = fnv(h, bytesOf(t.count));
  if(t.sign) h = fnv(h, bytesOf(t.sign));
  h = hashText(h, t.rows + '|' + (t.ids || []).join('\u0001'));
  return t._hash = h.toString(16);
}
export function streamHash(pts){
  if(!pts) return 'none';
  if(pts.kind !== 'points') return pts.kind + ':' + hashText(2166136261 >>> 0, JSON.stringify(pts, (k, v) => k.startsWith('_') ? undefined : v)).toString(16);
  let h = 2166136261 >>> 0;
  h = hashText(h, 'n' + pts.count);
  h = fnv(h, bytesOf(pts.pos)); h = fnv(h, bytesOf(pts.ntype)); h = fnv(h, bytesOf(pts.bias));
  h = fnv(h, bytesOf(pts.src)); h = fnv(h, bytesOf(pts.lidx));
  h = hashText(h, JSON.stringify(pts.tags || {}));
  h = hashText(h, JSON.stringify(pts.projs || []));
  h = hashText(h, JSON.stringify(pts.rules || []));
  h = hashText(h, JSON.stringify(pts.pools || []));
  h = hashText(h, JSON.stringify(pts.receptors || null));
  // a cell type row without its color: the numbers change the tissue, the hue is display only
  h = hashText(h, JSON.stringify(pts.cellTypes ? pts.cellTypes.map(r => ({ ...r, color:undefined })) : null));
  // hues are display only and stay out of the hash: a color change never rewires the network
  if(pts.ids) h = hashText(h, pts.ids.join('\u0001'));
  for(const en of pts.connectionTables || []){
    h = hashText(h, JSON.stringify({ ...en, table:undefined }));
    if(en.table) h = hashText(h, hashTable(en.table));
  }
  return h.toString(16);
}

const memo = new Map();                    // signature -> { val, bytes }, Map order = LRU
let memoTotal = 0;
function memoGet(key){
  const e = memo.get(key); if(!e) return null;
  memo.delete(key); memo.set(key, e);      // refresh recency
  return e.val;
}
function memoPut(key, val){
  const bytes = (val.synCount || 0)*13 + (val.count || 0)*25;
  if(bytes > MEMO_BYTES) return;           // never retain a single over-budget entry;
                                           // such nets get their buffers transferred to the sim worker instead (see startSim)
  memo.set(key, { val, bytes }); memoTotal += bytes;
  for(const [k, e] of memo){
    if(memoTotal <= MEMO_BYTES || k === key) break;
    memo.delete(k); memoTotal -= e.bytes;
  }
}
function sigOf(node, byId){                // mirrors computeNode's bypass semantics
  // a dot only moves a wire on screen, so it must not change the signature
  if(node.on === false || node.type === 'pin')
    return node.inputs[0] ? sigOf(byId(node.inputs[0].id), byId) : '#';
  const ins = node.inputs.map(c => c ? sigOf(byId(c.id), byId) : '#');
  // node.id is part of the signature because stable neuron identity is (scatter node id, local index): two structurally identical graphs with different ids wire identity-different networks, and a memo hit across them would corrupt anything keyed on identity (brain files)
  return node.type + '#' + node.id + JSON.stringify(node.params) + '[' + ins.join(',') + ']';
}

// The memo key of a node whose definition asks for one (memo, the connect node's wiring): its params and the content of each input stream, per resolution, with the quantized rate hint only when compensation is active (resolution, density or kscale below full); else full-resolution entries would churn.
// A plugin node's key carries its module and the hash of the module's source, so an edited module never finds the result its earlier version computed.
// Upstream plugins need nothing here: what they changed is in the stream, and streamHash reads the stream.
export function memoKey(node, def, memoIns){
  // compensation is active by the test finishConnect applies, density and kscale both; leaving either out lets a memo hit at full resolution carry compensation from an older rate hint
  const comp = simResolution < 1 ||
    (node.params.density !== undefined && node.params.density < 100) ||
    (node.params.kscale !== undefined && node.params.kscale < 100);
  const who = def && def.plugin ? '@' + def.plugin.module + '#' + def.plugin.hash : '';
  return simResolution + '|' + (comp ? Math.round(rateHint.E) + ',' + Math.round(rateHint.I) : '') + '|' + node.type + who + '|' +
    JSON.stringify(node.params) + '|' + memoIns.map(streamHash).join('|');
}

const pendingComputes = new Map();            // signature -> in-flight compute promise
// A computation that throws names its node on the error (the innermost one, since an error from upstream already carries its own), so the page can mark the node that failed rather than only saying what failed.
async function computeOf(def, ins, node){
  try { return await def.compute(ins, node.params, node); }
  catch(e){ if(e && typeof e === 'object' && e.nodeId === undefined) e.nodeId = node ? node.id : -1; throw e; }
}
export async function computeNode(node, byId){
  if(node._cache) return node._cache;
  if(node.on === false)                    // bypassed: pass main input through
    return node.inputs[0] ? computeNode(byId(node.inputs[0].id), byId) : null;
  const def = NODE_DEFS[node.type];
  if(!def) throw Object.assign(new Error(node.type + ': no node of this type is defined; load the module that registers it'), { nodeId:node.id });
  if(def.memo){
    // The inputs are computed first (computing points is cheap; files parse once), and the key is their content plus this node's params (memoKey).
    // The graph's shape is not in the key: an expanded fan of populations wires to the same net as the file node alone, and finds it here.
    const memoIns = [];
    for(const c of node.inputs) memoIns.push(c ? await computeNode(byId(c.id), byId) : null);
    const key = memoKey(node, def, memoIns);
    const hit = memoGet(key);
    // A hit whose synapse buffers were transferred into a sim worker is not a cache entry, it is a receipt for arrays that live somewhere else: over-budget nets transfer rather than copy so exactly one copy exists, and the memo keeps the husk.
    // Every reader of such a hit fails on the detached buffer.
    // Treat it as a miss and compute again.
    const usable = h => !(h && h.kind === 'net' && h.synCount > 0 &&
      h.post && h.post.buffer && h.post.byteLength === 0);
    if(hit && usable(hit)) return node._cache = hit;
    let p = pendingComputes.get(key);
    if(!p){
      pendingComputes.clear(); cancelComputes(); // other in-flight computations are for stale graph states
      p = (async () => {
        const ins = memoIns;
        const val = await computeOf(def, ins, node);
        memoPut(key, val);
        return val;
      })();
      pendingComputes.set(key, p);
      p.finally(() => { if(pendingComputes.get(key) === p) pendingComputes.delete(key); })
        .catch(() => {});                  // superseded/failed computations may go unawaited
    }
    return node._cache = await p;
  }
  const ins = [];
  for(const c of node.inputs) ins.push(c ? await computeNode(byId(c.id), byId) : null);
  node._cache = await computeOf(def, ins, node);
  return node._cache;
}

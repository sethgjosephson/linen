// Brain files: a saved brain is the graph, the learned weights keyed by stable pair identity, and the curriculum position.
// Weights are keyed by the 128-bit pair identity (srcA, lidxA, srcB, lidxB as four 32-bit ints, never packed into one JS number) so they survive additive graph edits: on load, pairs that still exist get their saved weight, new synapses start at baseline, and saved pairs that do not exist in the wiring are dropped.
// That is the formation-vs- persistence rule applied to files.
//
// Binary layout (.npb, little endian):
//   u32 magic 'NPB1', u32 version
//   u32 graphLen, graph JSON (UTF-8)
//   u32 metaLen, meta JSON (UTF-8)
//   pad to 4 bytes
//   u32 synCount m
//   i32[m] srcA, u32[m] lidxA, i32[m] srcB, u32[m] lidxB, f32[m] w
//
// Weights-only files (.npb, magic 'NPB2') drop the pair identity and keep the weights in wiring order.
// The wiring is deterministic (same graph, same seeds, same network), so a checkpoint taken during a run that never edits its graph can restore by index and does not need to say which pair each weight belongs to.
// That is 4 bytes per synapse instead of 20, which is what makes a full-density column checkpointable at all: the pair-keyed form of a 321 million synapse network is 6.4 GB in one ArrayBuffer and browsers cap a single buffer near 2 GB, so writing one throws RangeError.
// Weights-only trades portability across graph edits for that factor of five, so training checkpoints use it and interactive saves, which are expected to be loaded into an edited graph, use NPB1.
//
// Layout: u32 magic 'NPB2', u32 version, u32 graphLen, graph JSON,
//   u32 metaLen, meta JSON, pad to 4, u32 synCount m, f32[m] w
//
// Plastic-only files ('NPB3') go one step further and keep only the weights that can change.
// A gated scenario freezes most of its synapses, and a frozen weight is a pure function of the graph and the seed: the wiring reproduces it exactly, which is the same determinism weights-only files already depend on to restore by index.
// Storing it is storing the same number the wiring is about to compute anyway.
//
// Alphabet school 2 at full density has 151.7M synapses of which 12.9M are plastic, so a checkpoint falls from 579 MB to 49, and the write from about thirteen wall-minutes to under one.
// At the authored cadence of a brain file every five simulated minutes, the old size meant a sixteen-hour run spent roughly forty hours writing files and could never finish; this is what makes an overnight run possible rather than merely smaller on disk.
//
// Layout: u32 magic 'NPB3', u32 version, u32 graphLen, graph JSON,
//   u32 metaLen, meta JSON, pad to 4, u32 synCount m, u32 plasticCount k,
//   f32[k] w, in wiring order over the synapses whose pmask bit is set. m is
//   kept so a load can refuse a wiring of the wrong size before reading on.

const BRAIN_MAGIC = 0x3142504e;   // 'NPB1' pair-keyed
const WEIGHT_MAGIC = 0x3242504e;  // 'NPB2' weights-only
const PLASTIC_MAGIC = 0x3342504e; // 'NPB3' plastic weights only

// Measured ceiling for a single ArrayBuffer in current browsers.
// Chrome allocates 2.00 GB and throws RangeError at 2.20 GB.
export const MAX_BUFFER = 2 * 1024 * 1024 * 1024;

export const pairFileBytes = m => m*20;
export const weightFileBytes = m => m*4;
export const plasticFileBytes = k => k*4;
// how many synapses a plastic-only file would carry; null when the network has no gate, where plastic-only would save nothing and NPB2 is the file
export function plasticCount(net){
  if(!net.pmask) return null;
  // A transferred ArrayBuffer is detached in the sender, and a typed array over a detached buffer has length 0 and reads undefined at every index, which would count as "no synapse is plastic".
  // Length is the tell, so it is checked rather than trusted.
  if(net.pmask.length !== net.synCount)
    throw new Error('plasticity mask has ' + net.pmask.length + ' entries for ' +
      net.synCount + ' synapses. A mask transferred to a worker is detached ' +
      'here; keep a copy before transferring it.');
  let k = 0;
  for(let s=0;s<net.synCount;s++) if(net.pmask[s] & 31) k++;   // the rule is the low five bits
  return k;
}

// The order the cells were wired in, as a hash of their (src, lidx) sequence: an index-keyed file restores weights by synapse index, and the same cells in another order are the same tissue with every index moved, so the loader refuses that rather than loading weights onto the wrong cells (the wiring cache does not depend on the graph's shape, so cell order is a thing a scene can change on its own).
export function cellOrder(net){
  if(!net || !net.src || !net.lidx) return null;
  let h = 2166136261 >>> 0;
  const hb = a => { const b = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    for(let i = 0; i < b.length; i++){ h ^= b[i]; h = Math.imul(h, 16777619) >>> 0; } };
  hb(net.src instanceof Int32Array ? net.src : Int32Array.from(net.src)); hb(net.lidx);
  return h.toString(16);
}
function header(magic, graphObj, net, m, simMs, curriculum, kind){
  const enc = new TextEncoder();
  const graph = enc.encode(JSON.stringify(graphObj));
  const meta = enc.encode(JSON.stringify({
    app:'linen', simMs: simMs || 0, kind,
    savedAt: new Date().toISOString(),
    neurons: net.count, synapses: m, curriculum: curriculum || null,
    order: cellOrder(net) }));
  return { graph, meta, head: 8 + 4 + graph.length + 4 + meta.length };
}

export function buildBrainFile(graphObj, net, w, simMs, curriculum){
  const m = net.synCount;
  // Fail with the reason rather than a bare RangeError from the allocator.
  if(pairFileBytes(m) > MAX_BUFFER) throw new Error(
    `pair-keyed brain file needs ${(pairFileBytes(m)/1e9).toFixed(2)} GB in one buffer ` +
    `for ${m.toLocaleString()} synapses, over the ${(MAX_BUFFER/1e9).toFixed(2)} GB limit: ` +
    `use buildWeightsFile (weights-only) for a network this size`);
  const { graph, meta, head } = header(BRAIN_MAGIC, graphObj, net, m, simMs, curriculum, 'pairs');
  const keyOff = (head + 3) & ~3;
  const buf = new ArrayBuffer(keyOff + 4 + m*20);
  const dv = new DataView(buf);
  let o = 0;
  dv.setUint32(o, BRAIN_MAGIC, true); o += 4;
  dv.setUint32(o, 1, true); o += 4;
  dv.setUint32(o, graph.length, true); o += 4;
  new Uint8Array(buf, o, graph.length).set(graph); o += graph.length;
  dv.setUint32(o, meta.length, true); o += 4;
  new Uint8Array(buf, o, meta.length).set(meta); o += meta.length;
  o = keyOff;
  dv.setUint32(o, m, true); o += 4;
  const srcA = new Int32Array(buf, o, m); o += m*4;
  const lidxA = new Uint32Array(buf, o, m); o += m*4;
  const srcB = new Int32Array(buf, o, m); o += m*4;
  const lidxB = new Uint32Array(buf, o, m); o += m*4;
  new Float32Array(buf, o, m).set(w.subarray ? w.subarray(0, m) : w);
  const { preStart, post, src, lidx } = net;
  for(let i=0;i<net.count;i++)
    for(let s=preStart[i];s<preStart[i+1];s++){
      srcA[s] = src[i]; lidxA[s] = lidx[i];
      const j = post[s];
      srcB[s] = src[j]; lidxB[s] = lidx[j];
    }
  return buf;
}

// Refuses the ungated case: with no pmask every synapse is plastic and this would be NPB2 with an extra field.
export function buildPlasticFile(graphObj, net, w, simMs, curriculum){
  const m = net.synCount, pm = net.pmask;
  if(!pm) throw new Error('plastic-only file needs a plasticity mask; use buildWeightsFile');
  if(pm.length !== m)
    throw new Error('plasticity mask has ' + pm.length + ' entries for ' + m +
      ' synapses. A mask transferred to a worker is detached here; keep a ' +
      'copy before transferring it.');
  // the rule is the low five bits; the top three are the receptor channel, which a frozen synapse can carry too, so the whole byte must not be tested
  let k = 0;
  for(let s=0;s<m;s++) if(pm[s] & 31) k++;
  // If nothing is plastic there is nothing for this format to carry, and the caller wanted buildWeightsFile.
  if(k === 0 && m > 0)
    throw new Error('plastic-only file would hold 0 of ' + m + ' weights; ' +
      'nothing is marked plastic, so use buildWeightsFile');
  const { graph, meta, head } = header(PLASTIC_MAGIC, graphObj, net, m, simMs,
    curriculum, 'plastic');
  const wOff = (head + 3) & ~3;
  const buf = new ArrayBuffer(wOff + 8 + k*4);
  const dv = new DataView(buf);
  let o = 0;
  dv.setUint32(o, PLASTIC_MAGIC, true); o += 4;
  dv.setUint32(o, 1, true); o += 4;
  dv.setUint32(o, graph.length, true); o += 4;
  new Uint8Array(buf, o, graph.length).set(graph); o += graph.length;
  dv.setUint32(o, meta.length, true); o += 4;
  new Uint8Array(buf, o, meta.length).set(meta); o += meta.length;
  o = wOff;
  dv.setUint32(o, m, true); o += 4;
  dv.setUint32(o, k, true); o += 4;
  const out = new Float32Array(buf, o, k);
  let j = 0;
  for(let s=0;s<m;s++) if(pm[s] & 31) out[j++] = w[s];
  return buf;
}

export function buildWeightsFile(graphObj, net, w, simMs, curriculum){
  const m = net.synCount;
  if(weightFileBytes(m) > MAX_BUFFER) throw new Error(
    `weights file needs ${(weightFileBytes(m)/1e9).toFixed(2)} GB in one buffer ` +
    `for ${m.toLocaleString()} synapses, over the ${(MAX_BUFFER/1e9).toFixed(2)} GB limit`);
  const { graph, meta, head } = header(WEIGHT_MAGIC, graphObj, net, m, simMs, curriculum, 'weights');
  const wOff = (head + 3) & ~3;
  const buf = new ArrayBuffer(wOff + 4 + m*4);
  const dv = new DataView(buf);
  let o = 0;
  dv.setUint32(o, WEIGHT_MAGIC, true); o += 4;
  dv.setUint32(o, 1, true); o += 4;
  dv.setUint32(o, graph.length, true); o += 4;
  new Uint8Array(buf, o, graph.length).set(graph); o += graph.length;
  dv.setUint32(o, meta.length, true); o += 4;
  new Uint8Array(buf, o, meta.length).set(meta); o += meta.length;
  o = wOff;
  dv.setUint32(o, m, true); o += 4;
  new Float32Array(buf, o, m).set(w.subarray ? w.subarray(0, m) : w);
  return buf;
}

export function parseBrainFile(buf){
  const dv = new DataView(buf);
  const magic = dv.getUint32(0, true);
  if(magic !== BRAIN_MAGIC && magic !== WEIGHT_MAGIC && magic !== PLASTIC_MAGIC)
    throw new Error('not a brain file');
  if(dv.getUint32(4, true) !== 1) throw new Error('unsupported brain file version');
  const dec = new TextDecoder();
  let o = 8;
  const gLen = dv.getUint32(o, true); o += 4;
  const graph = JSON.parse(dec.decode(new Uint8Array(buf, o, gLen))); o += gLen;
  const mLen = dv.getUint32(o, true); o += 4;
  const meta = JSON.parse(dec.decode(new Uint8Array(buf, o, mLen))); o += mLen;
  o = (o + 3) & ~3;
  const m = dv.getUint32(o, true); o += 4;
  if(magic === WEIGHT_MAGIC)
    return { kind:'weights', graph, meta, m, w:new Float32Array(buf, o, m) };
  if(magic === PLASTIC_MAGIC){
    const k = dv.getUint32(o, true); o += 4;
    return { kind:'plastic', graph, meta, m, k, w:new Float32Array(buf, o, k) };
  }
  const srcA = new Int32Array(buf, o, m); o += m*4;
  const lidxA = new Uint32Array(buf, o, m); o += m*4;
  const srcB = new Int32Array(buf, o, m); o += m*4;
  const lidxB = new Uint32Array(buf, o, m); o += m*4;
  const w = new Float32Array(buf, o, m);
  return { kind:'pairs', graph, meta, m, srcA, lidxA, srcB, lidxB, w };
}

// Same synapse count, different cell order: the same tissue with every index moved.
// A file that recorded its order is refused onto another; a file from before orders were recorded loads as it always did.
function orderCheck(net, brain, what){
  const saved = brain.meta && brain.meta.order;
  if(!saved) return;
  const now = cellOrder(net);
  if(now && saved !== now) throw new Error(
    what + ' file was saved with the cells in another order than this wiring has: it restores ' +
    'by synapse index, so its weights would land on the wrong cells');
}
// Apply saved weights onto a (re)wired network: matching pairs restore, everything else keeps its wired baseline.
// Returns a new net object with a private weight array (never mutates the input; its arrays may be shared with the connect memo cache).
// Older files carry a membrane form in their header (0 the 2003 form, 1 this one).
// It is not checked: a 2003 brain's weights are the same currents on the classic rows that replaced that form, and the cell-order hash above already refuses a file from another tissue.
export function applyBrainWeights(net, brain){
  // Weights-only files restore by index, so the wiring has to match exactly.
  // A mismatch means the graph, seeds, or resolution moved since the save, and there is no identity in the file to recover from that.
  if(brain.kind === 'weights'){
    if(brain.m !== net.synCount) throw new Error(
      `weights file holds ${brain.m.toLocaleString()} synapses but this wiring has ` +
      `${net.synCount.toLocaleString()}: weights-only files restore by index and need ` +
      `the graph, seeds, and resolution they were saved from`);
    orderCheck(net, brain, 'weights-only');
    return { net:{ ...net, w:brain.w.slice() }, matched:brain.m,
      saved:brain.m, wired:net.synCount };
  }
  // Plastic-only: the frozen weights come from the wiring, which produced them deterministically, and the file supplies the rest in the same order the gate walks.
  // A size mismatch means a different network.
  if(brain.kind === 'plastic'){
    if(brain.m !== net.synCount) throw new Error(
      `plastic-only file was saved from ${brain.m.toLocaleString()} synapses but this ` +
      `wiring has ${net.synCount.toLocaleString()}: it restores by index and needs the ` +
      `graph, seeds, and resolution it was saved from`);
    if(!net.pmask) throw new Error(
      'plastic-only file needs the plasticity gate it was saved with; this wiring has none');
    orderCheck(net, brain, 'plastic-only');
    const wp = net.w.slice();
    let j = 0;
    for(let s=0;s<net.synCount;s++) if(net.pmask[s] & 31){
      if(j >= brain.k) break;
      wp[s] = brain.w[j++];
    }
    if(j !== brain.k) throw new Error(
      `plastic-only file holds ${brain.k.toLocaleString()} weights but this wiring gates ` +
      `${j.toLocaleString()}: the plasticity gate moved since the save`);
    return { net:{ ...net, w:wp }, matched:j, saved:brain.k, wired:net.synCount };
  }
  const w = net.w.slice();
  const map = new Map();
  for(let s=0;s<brain.m;s++)
    map.set(brain.srcA[s]+':'+brain.lidxA[s]+':'+brain.srcB[s]+':'+brain.lidxB[s],
      brain.w[s]);
  let matched = 0;
  const { preStart, post, src, lidx } = net;
  for(let i=0;i<net.count;i++)
    for(let s=preStart[i];s<preStart[i+1];s++){
      const j = post[s];
      const v = map.get(src[i]+':'+lidx[i]+':'+src[j]+':'+lidx[j]);
      if(v !== undefined){ w[s] = v; matched++; }
    }
  return { net:{ ...net, w }, matched, saved:brain.m, wired:net.synCount };
}

// Wire codec for the engine protocol.
// The messages ENGINE.md defines are plain objects carrying typed arrays; postMessage moves them by structured clone, a socket cannot.
// This encodes one message as one binary frame:
//
//   u32 headerLen (LE) | JSON header | payloads, each 8-byte aligned
//
// The header is the message with every typed array replaced by a { __ta: { d: dtype, n: length } } descriptor; payloads follow in the order the descriptors appear when the header is walked depth-first, which JSON round-trips preserve.
// Works unchanged in the browser and in Node, and depends on nothing.

const CTORS = {
  Float32Array, Float64Array,
  Int8Array, Int16Array, Int32Array,
  Uint8Array, Uint8ClampedArray, Uint16Array, Uint32Array,
};

const align8 = n => (n + 7) & ~7;

export function encodeMsg(obj){
  const bufs = [];
  const walk = v => {
    if(v === null || typeof v !== 'object') return v;
    if(ArrayBuffer.isView(v) && !(v instanceof DataView)){
      bufs.push(v);
      return { __ta: { d: v.constructor.name, n: v.length } };
    }
    if(Array.isArray(v)) return v.map(walk);
    const out = {};
    for(const k of Object.keys(v)){
      const w = walk(v[k]);
      if(w !== undefined) out[k] = w;
    }
    return out;
  };
  const header = new TextEncoder().encode(JSON.stringify(walk(obj)));
  let total = align8(4 + header.length);
  for(const b of bufs) total = align8(total + b.byteLength);
  const out = new Uint8Array(total);
  new DataView(out.buffer).setUint32(0, header.length, true);
  out.set(header, 4);
  let off = align8(4 + header.length);
  for(const b of bufs){
    out.set(new Uint8Array(b.buffer, b.byteOffset, b.byteLength), off);
    off = align8(off + b.byteLength);
  }
  return out.buffer;
}

export function decodeMsg(ab){
  const dv = new DataView(ab);
  const hlen = dv.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 4, hlen)));
  let off = align8(4 + hlen);
  const walk = v => {
    if(v === null || typeof v !== 'object') return v;
    if(v.__ta && CTORS[v.__ta.d]){
      const C = CTORS[v.__ta.d];
      const view = new C(ab, off, v.__ta.n);
      off = align8(off + view.byteLength);
      return view;
    }
    if(Array.isArray(v)) return v.map(walk);
    const out = {};
    for(const k of Object.keys(v)) out[k] = walk(v[k]);
    return out;
  };
  return walk(header);
}

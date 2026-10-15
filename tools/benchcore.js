// What tools/enginebench.mjs and the page it opens (tools/enginebench.html) share: the file an init message travels in, and the timing loop every engine is measured by.
// Plain ES module with no Node imports, so the page loads it as it is.
import { checkHello, missingTerms } from '../src/protocol.js';

const TYPED = { Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Uint8Array, Uint16Array, Uint32Array };

// An init message as one file: a little-endian u32 with the header's length, the header as JSON with every typed array replaced by its type, offset and length, then the arrays, each on an 8-byte boundary.
export function packMessage(msg){
  const arrays = [];
  let at = 0;
  const walk = v => {
    if(ArrayBuffer.isView(v)){
      at = Math.ceil(at/8)*8;
      const ref = { $ta:v.constructor.name, o:at, n:v.length };
      arrays.push([at, v]); at += v.byteLength;
      return ref;
    }
    if(Array.isArray(v)) return v.map(walk);
    if(v && typeof v === 'object'){ const o = {}; for(const k of Object.keys(v)) o[k] = walk(v[k]); return o; }
    return v;
  };
  const head = new TextEncoder().encode(JSON.stringify(walk(msg)));
  const base = Math.ceil((4 + head.length)/8)*8;
  const out = new Uint8Array(base + at);
  new DataView(out.buffer).setUint32(0, head.length, true);
  out.set(head, 4);
  for(const [o, v] of arrays) out.set(new Uint8Array(v.buffer, v.byteOffset, v.byteLength), base + o);
  return out;
}
export function unpackMessage(buf){
  const len = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, len)));
  const base = Math.ceil((4 + len)/8)*8;
  const walk = v => {
    if(Array.isArray(v)) return v.map(walk);
    if(v && typeof v === 'object'){
      // each array on a buffer of its own, as an engine receives it from the page
      if(v.$ta){ const T = TYPED[v.$ta]; return new T(buf.slice(base + v.o, base + v.o + v.n*T.BYTES_PER_ELEMENT)); }
      const o = {}; for(const k of Object.keys(v)) o[k] = walk(v[k]); return o;
    }
    return v;
  };
  return walk(head);
}

// Messages from an engine, waited for by command: next('state') resolves with the next state frame, and an error frame rejects whatever is waiting.
export function makeInbox(){
  const queue = [], waiting = [];
  return {
    push(m){
      if(m.cmd === 'error'){ const w = waiting.splice(0); for(const x of w) x.reject(new Error(m.message)); if(!w.length) queue.push(m); return; }
      const k = waiting.findIndex(x => x.cmd === m.cmd);
      if(k >= 0) waiting.splice(k, 1)[0].resolve(m); else queue.push(m);
    },
    next(cmd){
      const err = queue.findIndex(m => m.cmd === 'error');
      if(err >= 0) return Promise.reject(new Error(queue.splice(err, 1)[0].message));
      const k = queue.findIndex(m => m.cmd === cmd);
      if(k >= 0) return Promise.resolve(queue.splice(k, 1)[0]);
      return new Promise((resolve, reject) => waiting.push({ cmd, resolve, reject }));
    },
  };
}

// One engine through hello, init, a warm-up and the timed stretch, ticking `tick` simulated milliseconds per message throughout.
// With ready set, the engine answers init with a ready frame (the WebGPU worker builds its buffers after init returns, and a tick before then is dropped).
// Deliveries are counted from the state frames: every spike of cell i is one delivery per outgoing synapse of i.
// The returned times are wall clock, from now(), in seconds.
export async function timeEngine({ post, next, init, ready = false, warmMs, ms, tick, now }){
  post({ cmd:'hello' });
  const hello = await next('hello');
  const bad = checkHello(hello), miss = bad ? [] : missingTerms(init, hello);
  if(bad || miss.length) throw new Error(bad || 'the ' + hello.engine + ' engine does not implement ' + miss.map(m => m[1]).join(', '));
  const n = init.count, ps = init.preStart;
  const t0 = now();
  post(init);
  if(ready) await next('ready');
  for(let t = 0; t < warmMs; t += tick){ post({ cmd:'tick', steps:tick }); await next('state'); }
  const initS = now() - t0;
  let spikes = 0, deliveries = 0, simMs = 0;
  const t1 = now();
  while(simMs < ms){
    post({ cmd:'tick', steps:tick });
    const st = await next('state');
    const f = st.fired;
    for(let i = 0; i < n; i++) if(f[i]){ spikes += f[i]; deliveries += f[i]*(ps[i + 1] - ps[i]); }
    simMs += st.steps || tick;
  }
  const wallS = now() - t1;
  return { engine:hello.engine, initS, wallS, simMs, spikes, deliveries,
    realtime:simMs/1000/wallS, rate:spikes/n/(simMs/1000), deliveriesPerS:deliveries/wallS };
}

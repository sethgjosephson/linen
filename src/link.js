// A link that restores a scene exactly.
// The scene file (nodes, groups, hues, slice, resolution, every seed) is deflated and written into the address bar's fragment, which never leaves the browser, so nothing is uploaded and no server holds it.
// Opening the link loads the scene the way the address bar's ?scenario= does: on screen, not written over the open file until something is edited by hand.
// Determinism does the rest: the same file and seeds compute the same network and run the same way.
//
// The fragment is `scene=<base64url of deflate-raw of the JSON>`.
// A typical scene of forty nodes is a few kilobytes; the largest scene in the project, 172 nodes, is under twenty.
// Browsers carry fragments far past that.

const enc = new TextEncoder(), dec = new TextDecoder();

export function toBase64Url(bytes){
  let s = '';
  for(let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
export function fromBase64Url(text){
  const s = atob(text.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((text.length + 3) % 4));
  const out = new Uint8Array(s.length);
  for(let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}
async function pump(bytes, stream){
  // the writer's promises are not awaited before reading (a large input would wait on backpressure forever) but they are settled before returning, so a corrupt input rejects the read and nothing is left unhandled
  const w = stream.writable.getWriter();
  const written = Promise.all([w.write(bytes), w.close()]).catch(() => {});
  const chunks = []; const r = stream.readable.getReader();
  for(;;){ const { value, done } = await r.read(); if(done) break; chunks.push(value); }
  await written;
  const n = chunks.reduce((a, c) => a + c.length, 0), out = new Uint8Array(n);
  let p = 0; for(const c of chunks){ out.set(c, p); p += c.length; }
  return out;
}
// the fragment (without the #) for a scene, given its JSON text
export async function encodeSceneFragment(json){
  const packed = await pump(enc.encode(json), new CompressionStream('deflate-raw'));
  return 'scene=' + toBase64Url(packed);
}
// the scene object from a fragment, or null when the fragment carries none
export async function decodeSceneFragment(fragment){
  const m = /(?:^|[#&])scene=([A-Za-z0-9_-]+)/.exec(fragment || '');
  if(!m) return null;
  const bytes = await pump(fromBase64Url(m[1]), new DecompressionStream('deflate-raw'));
  const d = JSON.parse(dec.decode(bytes));
  if(!d || !Array.isArray(d.nodes)) throw new Error('the link does not carry a scene');
  return d;
}

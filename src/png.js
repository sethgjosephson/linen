// PNG at the byte level, for two things a canvas cannot do.
// A canvas writes a PNG with no physical size and no text, so a figure saved at 300 dots per inch opens in a layout program at 72 and several times too large: withMeta adds the pHYs chunk (pixels per meter) and iTXt chunks (what the figure is, from which scene, when).
// And encodePng writes a PNG from raw pixels with no canvas at all, which is how a matrix gets into a vector figure.
// Pure, so it runs under node.
// PNG specification, W3C 2003.

const SIG = [137, 80, 78, 71, 13, 10, 26, 10];
let TABLE = null;
export function crc32(bytes, crc = 0xffffffff){
  if(!TABLE){ TABLE = new Uint32Array(256);
    for(let n = 0; n < 256; n++){ let c = n; for(let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; TABLE[n] = c >>> 0; } }
  for(let i = 0; i < bytes.length; i++) crc = TABLE[(crc ^ bytes[i]) & 255] ^ (crc >>> 8);
  return crc;
}
function adler32(bytes){
  let a = 1, b = 0;
  for(let i = 0; i < bytes.length; i++){ a = (a + bytes[i]) % 65521; b = (b + a) % 65521; }
  return ((b << 16) | a) >>> 0;
}
const u32 = v => [(v >>> 24) & 255, (v >>> 16) & 255, (v >>> 8) & 255, v & 255];
const ascii = s => Array.from(s, c => c.charCodeAt(0) & 255);
export function chunk(type, data){
  const body = new Uint8Array(4 + data.length);
  body.set(ascii(type), 0); body.set(data, 4);
  const out = new Uint8Array(12 + data.length);
  out.set(u32(data.length), 0); out.set(body, 4);
  out.set(u32((crc32(body) ^ 0xffffffff) >>> 0), 8 + data.length);
  return out;
}
function utf8(s){
  if(typeof TextEncoder !== 'undefined') return new TextEncoder().encode(s);
  return Uint8Array.from(unescape(encodeURIComponent(s)), c => c.charCodeAt(0));
}
export function textChunk(keyword, text){
  const k = ascii(String(keyword).slice(0, 79)), t = utf8(String(text));
  const data = new Uint8Array(k.length + 5 + t.length);
  data.set(k, 0);                       // keyword, then a zero
  // compression flag 0, method 0, empty language tag, empty translated keyword
  data.set(t, k.length + 5);
  return chunk('iTXt', data);
}
// the physical size chunk: pixels per meter on both axes
export function physChunk(dpi){
  const ppm = Math.round(dpi/0.0254);
  return chunk('pHYs', Uint8Array.from([...u32(ppm), ...u32(ppm), 1]));
}
// png: the bytes a canvas produced.
// Returns them with the size and the text chunks placed after the header, any earlier pHYs removed.
export function withMeta(png, o = {}){
  const src = png instanceof Uint8Array ? png : new Uint8Array(png);
  for(let i = 0; i < 8; i++) if(src[i] !== SIG[i]) throw new Error('not a PNG');
  const parts = [src.subarray(0, 8)];
  let p = 8, placed = false;
  while(p + 12 <= src.length){
    const len = ((src[p] << 24) | (src[p+1] << 16) | (src[p+2] << 8) | src[p+3]) >>> 0;
    const type = String.fromCharCode(src[p+4], src[p+5], src[p+6], src[p+7]);
    const whole = src.subarray(p, p + 12 + len);
    if(type !== 'pHYs' || !(o.dpi > 0)) parts.push(whole);
    if(type === 'IHDR' && !placed){
      placed = true;
      if(o.dpi > 0) parts.push(physChunk(o.dpi));
      for(const [k, v] of Object.entries(o.text || {})) if(v !== undefined && v !== null && v !== '') parts.push(textChunk(k, v));
    }
    p += 12 + len;
  }
  let n = 0; for(const a of parts) n += a.length;
  const out = new Uint8Array(n); let q = 0;
  for(const a of parts){ out.set(a, q); q += a.length; }
  return out;
}
export function readMeta(png){
  const src = png instanceof Uint8Array ? png : new Uint8Array(png);
  const out = { dpi:0, text:{}, width:0, height:0 };
  let p = 8;
  while(p + 12 <= src.length){
    const len = ((src[p] << 24) | (src[p+1] << 16) | (src[p+2] << 8) | src[p+3]) >>> 0;
    const type = String.fromCharCode(src[p+4], src[p+5], src[p+6], src[p+7]);
    const d = src.subarray(p + 8, p + 8 + len);
    const be = i => ((d[i] << 24) | (d[i+1] << 16) | (d[i+2] << 8) | d[i+3]) >>> 0;
    if(type === 'IHDR'){ out.width = be(0); out.height = be(4); }
    if(type === 'pHYs' && d[8] === 1) out.dpi = Math.round(be(0)*0.0254);
    if(type === 'iTXt'){ const z = d.indexOf(0);
      out.text[String.fromCharCode(...d.subarray(0, z))] = new TextDecoder().decode(d.subarray(z + 5)); }
    p += 12 + len;
  }
  return out;
}
// rgba: width*height*4 bytes.
// Eight bit RGBA, no filter, the pixel data in stored (uncompressed) deflate blocks: larger than a compressed file and valid everywhere, and the images this is for are a matrix at data resolution, a few hundred kilobytes at most.
export function encodePng(rgba, width, height){
  const stride = width*4, raw = new Uint8Array((stride + 1)*height);
  for(let y = 0; y < height; y++){ raw[y*(stride + 1)] = 0; raw.set(rgba.subarray(y*stride, (y + 1)*stride), y*(stride + 1) + 1); }
  const blocks = Math.max(1, Math.ceil(raw.length/65535));
  const z = new Uint8Array(2 + raw.length + blocks*5 + 4);
  z[0] = 0x78; z[1] = 0x01;
  let q = 2;
  for(let b = 0; b < blocks; b++){
    const from = b*65535, n = Math.min(65535, raw.length - from);
    z[q++] = b === blocks - 1 ? 1 : 0;
    z[q++] = n & 255; z[q++] = n >>> 8; z[q++] = ~n & 255; z[q++] = (~n >>> 8) & 255;
    z.set(raw.subarray(from, from + n), q); q += n;
  }
  z.set(u32(adler32(raw)), q);
  const ihdr = Uint8Array.from([...u32(width), ...u32(height), 8, 6, 0, 0, 0]);
  const parts = [Uint8Array.from(SIG), chunk('IHDR', ihdr), chunk('IDAT', z), chunk('IEND', new Uint8Array(0))];
  let n = 0; for(const a of parts) n += a.length;
  const out = new Uint8Array(n); let p = 0;
  for(const a of parts){ out.set(a, p); p += a.length; }
  return out;
}
export function base64(bytes){
  if(typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let s = '';
  for(let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

// Live input runtime.
// Encoder nodes define the structure (a deterministic channel-to-neuron map, computed from geometry); this file only feeds gains: main-thread sources sampled at display rate and streamed to the engine as inputFrame messages.
// Nothing here touches the compute path, so live media never breaks determinism of the wired network.
import { resolveDir } from './fsstore.js';
import { unif } from './rand.js';
import { scheduleAt, renderItem, renderAudio } from './curriculum.js';

// read-node playback: an image sequence from a granted folder, played at the node's fps.
// Frames repost only when the frame index changes; the engine holds the last gains in between.
const IMG_RE = /\.(png|jpe?g|webp|gif|bmp)$/i;
class ReadSource {
  constructor(onError){
    this.onError = onError;
    this.frames = null; this.err = false; this.pathKey = null;
    this.t0 = 0; this.lastFrame = -1;
    this.cv = document.createElement('canvas');
  }
  async load(sig){
    this.frames = null; this.err = false; this.lastFrame = -1;
    try {
      const dir = await resolveDir(sig.path);
      const names = [];
      for await (const [name, h] of dir.entries())
        if(h.kind === 'file' && IMG_RE.test(name)) names.push(name);
      names.sort();
      if(!names.length) throw new Error('no image files in "' + sig.path + '"');
      const frames = [];
      for(const name of names.slice(0, 2048))
        frames.push(await createImageBitmap(await (await dir.getFileHandle(name)).getFile()));
      this.frames = frames;
      this.t0 = null;   // anchored to the caller's clock on first sample
    } catch(e){
      this.err = true;
      if(this.onError) this.onError('read: ' + (e.message || e));
    }
  }
  sample(now, map){
    const sig = map.signal;
    if(!sig || !sig.path) return null;
    if(sig.path !== this.pathKey){ this.pathKey = sig.path; this.load(sig); return null; }
    if(!this.frames || this.err) return null;
    if(this.t0 === null) this.t0 = now;
    const n = this.frames.length;
    let f = Math.floor((now - this.t0)/1000 * Math.max(0.1, sig.fps || 12));
    f = sig.loop ? f % n : Math.min(f, n-1);
    if(f === this.lastFrame) return null;
    this.lastFrame = f;
    const { cols, rows, sheet } = map;
    const R = sheet ? rows : 1;
    if(this.cv.width !== cols || this.cv.height !== R){
      this.cv.width = cols; this.cv.height = R;
    }
    const c2 = this.cv.getContext('2d', { willReadFrequently:true });
    c2.drawImage(this.frames[f], 0, 0, cols, R);
    const px = c2.getImageData(0, 0, cols, R).data;
    const g = new Float32Array(cols*R);
    // flipped vertically so image top lands on the high end of the axis; not mirrored (authored files, unlike the webcam selfie view)
    for(let r=0;r<R;r++)
      for(let c=0;c<cols;c++){
        const o = (r*cols + c)*4;
        const v = (px[o] + px[o+1] + px[o+2]) / 765;
        if(v >= 0.03) g[(R-1-r)*cols + c] = v;
      }
    return g;
  }
  stop(){
    if(this.frames) for(const f of this.frames) if(f.close) f.close();
    this.frames = null;
  }
}

export class BarSweep {
  sample(now, map){
    const { cols, rows, sheet } = map;
    const period = Math.max(100, map.period || 1500);
    const g = new Float32Array(cols * (sheet ? rows : 1));
    let pos = (now % period) / period * cols;
    const dsel = map.signal ? map.signal.direction | 0 : 0;
    // backward is the same bar the other way; alternate flips every sweep
    const dir = dsel === 2 ? Math.floor(now / period) % 2 : (dsel === 1 ? 1 : 0);
    if(dir === 1) pos = cols - pos;
    this.dir = dir; this.alternating = dsel === 2;
    for(let c=0;c<cols;c++){
      const d = Math.min(Math.abs(c + 0.5 - pos), cols - Math.abs(c + 0.5 - pos));
      const v = Math.exp(-d*d / (2 * 1.2 * 1.2));
      if(v < 0.01) continue;
      if(sheet) for(let r=0;r<rows;r++) g[r*cols + c] = v;
      else g[c] = v;
    }
    return g;
  }
  stop(){}
}

class Microphone {
  constructor(onError){
    this.ready = false;
    navigator.mediaDevices.getUserMedia({ audio:true }).then(stream => {
      this.stream = stream;
      this.ctx = new AudioContext();
      const src = this.ctx.createMediaStreamSource(stream);
      this.an = this.ctx.createAnalyser();
      this.an.fftSize = 2048;
      this.an.smoothingTimeConstant = 0.6;
      src.connect(this.an);
      this.buf = new Uint8Array(this.an.frequencyBinCount);
      this.ready = true;
    }).catch(e => onError && onError('microphone: ' + e.message));
  }
  sample(now, map){
    if(!this.ready) return null;
    this.an.getByteFrequencyData(this.buf);
    const { cols, rows, sheet } = map;
    const g = new Float32Array(cols * (sheet ? rows : 1));
    const binHz = this.ctx.sampleRate / this.an.fftSize;
    // log-spaced bands, 80 Hz to 8 kHz, mapped across the columns (tonotopy)
    for(let c=0;c<cols;c++){
      const f0 = 80 * Math.pow(100, c/cols), f1 = 80 * Math.pow(100, (c+1)/cols);
      const b0 = Math.max(1, Math.floor(f0/binHz)), b1 = Math.max(b0+1, Math.ceil(f1/binHz));
      let s = 0;
      for(let b=b0;b<b1 && b<this.buf.length;b++) s += this.buf[b];
      // sqrt compression approximates cochlear loudness compression
      const v = Math.sqrt(Math.max(0, s / (b1-b0) / 255 - 0.06));
      if(v < 0.01) continue;
      if(sheet) for(let r=0;r<rows;r++) g[r*cols + c] = v;
      else g[c] = v;
    }
    return g;
  }
  stop(){
    if(this.stream) this.stream.getTracks().forEach(t => t.stop());
    if(this.ctx) this.ctx.close();
    this.ready = false;
  }
}

class Webcam {
  constructor(onError){
    this.ready = false;
    navigator.mediaDevices.getUserMedia({ video:{ width:160, height:120 } }).then(stream => {
      this.stream = stream;
      this.video = document.createElement('video');
      this.video.srcObject = stream;
      this.video.muted = true;
      this.video.play();
      this.cv = document.createElement('canvas');
      this.ready = true;
    }).catch(e => onError && onError('webcam: ' + e.message));
  }
  sample(now, map){
    if(!this.ready || this.video.readyState < 2) return null;
    const { cols, rows, sheet } = map;
    const R = sheet ? rows : 1;
    if(this.cv.width !== cols || this.cv.height !== R){
      this.cv.width = cols; this.cv.height = R;
    }
    const c2 = this.cv.getContext('2d', { willReadFrequently:true });
    c2.drawImage(this.video, 0, 0, cols, R);
    const px = c2.getImageData(0, 0, cols, R).data;
    const g = new Float32Array(cols * R);
    // mirrored horizontally (selfie view) and flipped vertically so the top of the camera image lands on the high end of the mapped axis
    for(let r=0;r<R;r++)
      for(let c=0;c<cols;c++){
        const o = (r*cols + c) * 4;
        const v = (px[o] + px[o+1] + px[o+2]) / 765;
        if(v >= 0.03) g[(R-1-r)*cols + (cols-1-c)] = v;
      }
    return g;
  }
  stop(){
    if(this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.ready = false;
  }
}

// encoding stage between the raw source raster and the engine gains. raw coding passes through; retinal coding half-wave rectifies a center- surround difference (ON block then OFF block, lateral inhibition in 1D for bands), and both apply slow adaptation so static input fades, the way retinal output does.
// Microsaccade jitter (small random raster shifts every few hundred ms) refreshes contrast the same way biology does.
export function encodeGains(raw, map, st, now, seed){
  const cols = map.cols, R = map.sheet ? map.rows : 1, G = cols*R;
  let g = raw;
  if(map.jitter > 0 && map.sheet){
    if(!st.nj || now > st.nj){
      // Seeded, from the same hash the engines draw noise with: an unseeded draw would give a training run of any scene with jitter on a different stimulus every time, which the determinism rule forbids and the reproducibility check does not see because it runs a scene with jitter off.
      // Keyed on the map, a per-map refresh count and the run seed rather than on the clock, so the offsets do not depend on how many steps a tick takes.
      st.jn = (st.jn | 0) + 1;
      const h = (Math.imul(map.id | 0, 0x9E3779B1) ^ Math.imul(st.jn, 0x85EBCA6B) ^ (seed | 0)) >>> 0;
      const u0 = unif(h), u1 = unif((h ^ 0x9e3779b9) >>> 0), u2 = unif((h ^ 0x7f4a7c15) >>> 0);
      st.nj = now + 200 + u0*250;
      st.jx = Math.round((u1*2 - 1)*map.jitter);
      st.jy = Math.round((u2*2 - 1)*map.jitter);
    }
    if(st.jx || st.jy){
      g = new Float32Array(G);
      for(let r=0;r<R;r++){
        const sr = Math.min(R-1, Math.max(0, r + st.jy));
        for(let c=0;c<cols;c++){
          const sc = Math.min(cols-1, Math.max(0, c + st.jx));
          g[r*cols + c] = raw[sr*cols + sc];
        }
      }
    }
  }
  let out;
  if(map.code === 1){
    out = new Float32Array(2*G);
    for(let r=0;r<R;r++)
      for(let c=0;c<cols;c++){
        let s = 0, n = 0;
        for(let dr=-1;dr<=1;dr++){
          const rr = r+dr; if(rr<0 || rr>=R) continue;
          if(!map.sheet && dr !== 0) continue;
          for(let dc=-1;dc<=1;dc++){
            const cc = c+dc; if(cc<0 || cc>=cols || (dr===0 && dc===0)) continue;
            s += g[rr*cols + cc]; n++;
          }
        }
        const d = g[r*cols + c] - (n ? s/n : 0);
        if(d > 0.005) out[r*cols + c] = 2*d;            // ON block
        else if(d < -0.005) out[G + r*cols + c] = -2*d; // OFF block
      }
  } else {
    out = g === raw ? raw.slice() : g;   // buffers transfer, never share
  }
  if(!st.ad || st.ad.length !== out.length) st.ad = new Float32Array(out.length);
  const a = 0.06;                        // adaptation tau roughly half a second
  const T = map.transient || 0;
  for(let i=0;i<out.length;i++){
    st.ad[i] += (out[i] - st.ad[i]) * a;
    out[i] = Math.max(0, out[i] - T*st.ad[i]);
  }
  return out;
}

// generated lessons: one schedule shared by every encoder wired to the same curriculum node, so vision and audio present the same item at the same moment.
// An encoder set to sight gets the glyph, one set to sound gets the spoken-name spectrum; during the gap between presentations nothing drives.
// The sense is a setting of its own rather than read off the mapping: a two dimensional encoder is not always sight, since a distributed map can carry sound and a one dimensional one can carry sight.
// The tile sound item k lands in when the tile order is shuffled: items ordered by a seeded key, each sent to the next tile in that order, which is a random cycle and so has no fixed point.
export function shuffledTile(k, n, seed){
  const order = Array.from({ length:n }, (_, i) => i)
    .sort((a, b) => unif((seed*9973 + a*7919 + 17) >>> 0) - unif((seed*9973 + b*7919 + 17) >>> 0) || a - b);
  const at = order.indexOf(k);
  return order[(at + 1) % n];
}

class CurriculumSource {
  constructor(){ this.lastKey = ''; this.item = ''; }
  sample(now, map){
    const sig = map.signal;
    if(!sig) return null;
    const st0 = scheduleAt(sig, now);
    // the scrambled partner applies to the non-sheet (auditory) encoder, so sight keeps its identity and sound stops predicting it.
    // Only on a presentation of both: a probe presents one sense, has no pairing to scramble, and is labeled with its item, so a sound probe playing the scrambled partner measured every scrambled run's sound code under labels unrelated to what it heard
    const hears = map.sense === 'sound';
    const scr = sig.scramble && hears && st0.altItem && (st0.cond | 0) === 0;
    const st = scr ? { ...st0, item:st0.altItem } : st0;
    this.item = st.item; this.on = st.on; this.cond = st.cond;
    // probe presentations drop one modality so cross-modal completion can be measured: cond 1 shows the glyph with no sound, cond 2 the spoken name with no glyph.
    // Which presentations are probes is a pure function of the presentation index, so every encoder agrees without shared state.
    const silent = st.cond === 1 ? hears : st.cond === 2 ? !hears : false;
    if(!st.on || silent){
      // blank gap or silenced modality: emit zeros once, then hold
      const key = (silent ? 'mute' : 'off') + st.k;
      if(key === this.lastKey) return null;
      this.lastKey = key;
      return new Float32Array(map.cols * (map.sheet ? map.rows : 1));
    }
    // placement: the index of the item this encoder renders (the scrambled partner's for a scrambled sound), the item count, and the mode
    const place = sig.place | 0;
    let pi = scr ? (st0.altIdx | 0) : (st.idx | 0);
    const nItems = Math.max(1, st.count | 0);
    // tile order: sound takes a seeded shuffle of the tiles in which no item keeps its own tile (a random cycle), so position cannot carry identity across the senses
    if(hears && (sig.placeOrder | 0) === 1 && nItems > 1) pi = shuffledTile(pi, nItems, sig.seed | 0);
    if(!hears){
      const key = 'v' + st.k;               // a glyph is static while shown
      if(key === this.lastKey) return null;
      this.lastKey = key;
      const cols = map.cols, rows = map.sheet ? map.rows : 1;
      const acuity = sig.acuity === undefined ? 1 : sig.acuity;
      if(place === 1 && rows > 1){
        // own tile: the glyph drawn small into tile pi of a t by t grid
        const t = Math.ceil(Math.sqrt(nItems));
        const tc = Math.max(1, Math.floor(cols/t)), tr = Math.max(1, Math.floor(rows/t));
        const small = renderItem(st, tc, tr, acuity);
        const g = new Float32Array(cols*rows);
        const ox = (pi % t)*tc, oy = Math.floor(pi/t)*tr;
        for(let y=0;y<tr;y++) for(let x=0;x<tc;x++){
          const X = ox + x, Y = oy + y;
          if(X < cols && Y < rows) g[Y*cols + X] = small[y*tc + x];
        }
        return g;
      }
      if(place === 2){
        // own offset: full size, shifted by an item-specific displacement
        const t = Math.ceil(Math.sqrt(nItems));
        const dx = ((pi % t) - (t-1)/2) * (0.6/Math.max(1, t-1)) * 1.0;
        const dy = (Math.floor(pi/t) - (t-1)/2) * (0.6/Math.max(1, t-1)) * 1.0;
        return renderItem({ ...st, dx:(st.dx || 0) + dx, dy:(st.dy || 0) + dy }, cols, rows, acuity);
      }
      // a sheet mapping renders the glyph over two dimensions; any other mapping takes the same glyph collapsed to one row of channels
      return renderItem(st, cols, rows, acuity);
    }
    // audio evolves within the item.
    // A two dimensional mapping gets one band per cell of the sheet, so sound fills the same channel count sight does.
    const bands = map.cols * (map.sheet ? map.rows : 1);
    if(place === 1){
      // own block: the spectrum rendered over the item's share of the bands
      const bl = Math.max(1, Math.floor(bands/nItems));
      const small = renderAudio(st, bl);
      const g = new Float32Array(bands);
      for(let i=0;i<bl;i++){ const j = pi*bl + i; if(j < bands) g[j] = small[i]; }
      return g;
    }
    if(place === 2){
      // own offset: the full spectrum rolled by an item-specific number of bands
      const full = renderAudio(st, bands);
      const shift = Math.floor(bands/Math.max(2, nItems))*pi;
      const g = new Float32Array(bands);
      for(let i=0;i<bands;i++) g[(i + shift) % bands] = full[i];
      return g;
    }
    return renderAudio(st, bands);
  }
  stop(){}
}

export class IORuntime {
  constructor(onError){
    this.onError = onError;
    this.worker = null; this.maps = []; this.sources = new Map(); this.last = 0;
    this.enc = new Map();                // per-map encoder state (adaptation, jitter)
    this.lastGains = new Map();          // last gains posted per map (input monitor)
    this.muted = new Set();              // map ids the user has silenced
    this.seed = 0;                       // the run seed; jitter draws key on it
  }
  // the direction of an alternating bar sweep this frame (0 forward, 1 backward), or null when no sweep alternates; the probes split by it
  get sweepDirection(){ return this.sweepDir === undefined ? null : this.sweepDir; }
  // (re)bind to the running worker and the computed input maps; sources are kept alive across tunes when their kind is unchanged
  attach(worker, maps, seed){
    this.sweepDir = undefined;
    this.worker = worker;
    this.maps = maps || [];
    if(seed !== undefined && seed !== null) this.seed = seed | 0;
    const live = new Set();
    for(const m of this.maps){
      live.add(m.id);
      const cur = this.sources.get(m.id);
      if(cur && cur.kind === m.source) continue;
      if(cur) cur.inst.stop();
      const inst = m.source === -2 ? new CurriculumSource()
        : m.source === -1 ? new ReadSource(this.onError)
        : m.source === 1 ? new Microphone(this.onError)
        : m.source === 2 ? new Webcam(this.onError) : new BarSweep();
      this.sources.set(m.id, { kind:m.source, inst });
    }
    for(const [id, s] of [...this.sources])
      if(!live.has(id)){ s.inst.stop(); this.sources.delete(id); this.enc.delete(id); }
  }
  detach(){ this.worker = null; }   // keep sources alive across a worker restart
  // muting a modality is how the cross-modal question gets asked: train with both, then present one alone and watch what the tissue still does
  toggleMute(id){
    if(this.muted.has(id)) this.muted.delete(id); else this.muted.add(id);
    this.mutedDirty = true;
    return this.muted.has(id);
  }
  // displayOnly computes the stimulus without sending it to the engine.
  // With the host driving the pump it renders the frames itself, so the page would otherwise have nothing to show in the input monitor: the schedule is a pure function of the clock, so recomputing it here shows exactly what the host is feeding in, and posting nothing keeps the page from stimulating a network it does not drive.
  frame(now, displayOnly){
    if(!this.worker || !this.maps.length) return;
    if(now - this.last < 33) return;
    this.last = now;
    for(let mi=0;mi<this.maps.length;mi++){
      const m = this.maps[mi], s = this.sources.get(m.id);
      let st = this.enc.get(m.id);
      if(!st){ st = {}; this.enc.set(m.id, st); }
      if(this.muted.has(m.id)){
        // post silence once, then leave the engine holding zeros
        if(st.wasMuted) continue;
        st.wasMuted = true;
        const z = new Float32Array((m.code === 1 ? 2 : 1) * m.cols * (m.sheet ? m.rows : 1));
        this.lastGains.set(m.id, z.slice());
        if(!displayOnly)
          this.worker.postMessage({ cmd:'inputFrame', mi, gains:z }, [z.buffer]);
        continue;
      }
      st.wasMuted = false;
      // Arrival lag: this map reads the lesson clock a little behind the others.
      // Senses do not reach cortex together, and a rule that depends on spike order needs an order to depend on: with both modalities arriving at once, potentiation and depression on the connections between them cancel.
      // A curriculum that follows another sits ahead of or behind it by its own offset, which is the stagger between two senses.
      const off = (m.signal && m.signal.offset) || 0;
      const mNow = now - (m.lagMs || 0) - off;
      let raw = s && s.inst.sample(mNow, m);
      if(raw) st.lastRaw = raw;
      // between source frames, adaptation and microsaccades keep evolving
      else if(st.lastRaw && (m.transient > 0 || (m.jitter > 0 && m.sheet)))
        raw = st.lastRaw;
      if(!raw) continue;
      const g = encodeGains(raw, m, st, mNow, this.seed);
      this.lastGains.set(m.id, g.slice());     // for the input monitor
      // an alternating sweep tells the probes which direction is running
      if(s && s.inst instanceof BarSweep && s.inst.alternating) this.sweepDir = s.inst.dir;
      if(!displayOnly)
        this.worker.postMessage({ cmd:'inputFrame', mi, gains:g }, [g.buffer]);
    }
  }
  stop(){
    for(const [, s] of this.sources) s.inst.stop();
    this.sources.clear();
    this.worker = null; this.maps = [];
  }
}

import { h32 as h32s } from './rand.js';
// Generated lesson streams for the curriculum node.
// One schedule function drives every input node wired to the same curriculum node: item, timing, and per-presentation jitter all derive from the simulated clock and the seed, so a visual encoder and an audio encoder independently agree on what is being presented at any moment.
// That synchrony is the point: redundant, temporally aligned multimodal input is learned faster than either stream alone (Bahrick and Lickliter, intersensory redundancy).
//
// Variety is the other point.
// A single repeating stimulus drives the same pathway on every cycle, which lets spike-timing plasticity ratchet one set of synapses without bound.
// Interleaved items, jittered position, size and rotation, and blank gaps between presentations decorrelate the input the way real developmental experience does.
//
// Set design, in short (sources are cited beside each choice below): objects are silhouettes because shape, not texture, carries object names (Landau, Smith and Jones 1988); the object nouns are ones real toddlers acquire early (Wordbank / MacArthur-Bates CDI norms); the letter order option follows systematic synthetic phonics (s a t i p n first, chosen because most early words are spellable from them); and the picture-book set presents image, spoken word and written word together, which is the two-factor structure of the Simple View of Reading.

// ---- item sets -------------------------------------------------------
// kind: how an item is rendered. glyph = a character; shape/object = a drawn silhouette; word = the written word; book = picture above word.
export const SETS = [
  { name:'alphabet A-Z', kind:'glyph',
    items:'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('') },
  { name:'letters, phonics order', kind:'glyph',
    items:'SATIPNCKEHRMDGOULFBQJZWVYX'.split('') },
  { name:'vowels A E I O U', kind:'glyph', items:['A','E','I','O','U'] },
  { name:'digits 0-9', kind:'glyph', items:'0123456789'.split('') },
  { name:'shapes', kind:'shape',
    items:['circle','square','triangle','cross','bar'] },
  { name:'objects (spoken names)', kind:'object',
    items:['ball','dog','cat','apple','banana','cup','shoe','book','tree','star','fish','house'] },
  { name:'written words', kind:'word',
    items:['sat','tap','pin','pat','tin','nap','sit','pit','nip','tan'] },
  { name:'picture book (image + word)', kind:'book',
    items:['ball','dog','cat','apple','banana','cup','shoe','book','tree','star','fish','house'] },
  // Abstract pairs.
  // One item is one place on the retina lighting up and one tone, with nothing else varying.
  // Glyphs are the wrong stimulus for asking whether a rule builds selectivity: a letter covers most of the sheet and any two letters overlap heavily, so the inputs are nearly the same pattern and there is little for competition to separate.
  // A spot and a tone are as separable as a stimulus gets, which makes this the control case: a rule that cannot learn these cannot learn anything.
  { name:'spots and tones (abstract pairs)', kind:'spot',
    items:['0','1','2','3','4','5','6','7','8','9','10','11'] },
];

// A training schedule in blocks: "20 A; 20 B; 60 AB" presents channel A alone for twenty simulated minutes, then channel B alone for twenty, then both together for sixty.
// The last block runs to the end of the run whatever its stated length, so a schedule cannot leave the network with nothing to do.
//
// This exists because association has an order to it.
// Pokorny et al. 2019 (Cerebral Cortex, STDP forms associations between memory traces in networks of spiking neurons) build each memory trace first, by presenting its item on its own until an assembly has formed, and only then present two items together to associate them.
// Presenting both channels from the first millisecond asks the rule to build two codes and a link between them at once out of a network that has neither.
// Blocks separate those questions: what the probes measure during the pairing block can be compared against what the same probes measured during the blocks before it, in the same network, in one run.
export function parsePhases(str){
  const out = [];
  for(const part of String(str || '').split(';')){
    const t = part.trim();
    if(!t) continue;
    const m = /^([0-9]*\.?[0-9]+)\s+(.+)$/.exec(t);
    if(!m) continue;
    const w = m[2].trim().toLowerCase();
    // named for the channels rather than for a modality: this schedule is read by whatever encoders are wired to the curriculum
    const cond = (w === 'a' || w === '1') ? 1 : (w === 'b' || w === '2') ? 2 : 0;
    out.push({ min:Math.max(0, +m[1]), cond, name:w });
  }
  return out;
}
// which block simulated time t falls in, t measured from the end of the opening calibration sweep so that a schedule means what it says however long the sweep took
export function phaseAt(list, tMs){
  if(!list || !list.length) return null;
  let acc = 0;
  for(let i = 0; i < list.length; i++){
    acc += list[i].min*60000;
    if(tMs < acc || i === list.length - 1) return { i, ...list[i] };
  }
  return null;
}

const h32 = x => h32s(x) >>> 0;   // the shared mixer, unsigned for the modulo draws below
const unit = (k, salt, seed) => h32(k*2654435761 ^ salt*40503 ^ seed*974711) / 4294967296;

// what is being presented at simulated time t, and how it is posed
export function scheduleAt(sig, tMs){
  const set = SETS[sig.set|0] || SETS[0];
  const N = set.items.length;
  const on = Math.max(20, sig.onMs), off = Math.max(0, sig.offMs);
  const period = on + off;
  const k = Math.floor(tMs / period);
  const phase = tMs - k*period;
  // The calibration sweep, before anything is allowed to learn.
  // Every item is presented in all three conditions, in order, for calCycles passes: bimodal for a full alphabet, then vision alone for a full alphabet, then sound alone.
  // This is what the conjunctive sets are built from, so it has to cover every item and it has to happen while plasticity is off.
  // The ordinary probe schedule cannot do it: with probeEvery 4 and 26 items the probes land on k = 3 mod 4, and gcd(4,26) is 2, so they only ever reach the odd-numbered letters and the other thirteen are never seen alone.
  const calCycles = Math.max(0, sig.calCycles | 0);
  const calK = calCycles*3*N;                           // presentations it owns
  // The same sweep again, every reSweepEvery presentations, with the network as training has left it.
  // The opening sweep says which cells were conjunctive before anything learned; these say what that set looks like now, so drift can be told from learning.
  // Plasticity is not touched: this measures the network mid-training rather than pausing it.
  const reEvery = Math.max(0, sig.reSweepEvery | 0);
  let sweepK = -1, isRe = false;
  if(calK > 0 && k < calK) sweepK = k;
  else if(calK > 0 && reEvery > calK){
    // since >= reEvery, not just >= 0: at k exactly calK the modulo is zero, which would start a repeat the instant the opening sweep ended and run the two together as one double-length window.
    // The first repeat belongs a full period after the opening one, not adjacent to it.
    const since = k - calK, pos = since % reEvery;
    if(since >= reEvery && pos < calK){ sweepK = pos; isRe = true; }
  }
  if(sweepK >= 0){
    const pass = Math.floor(sweepK / N) % 3;            // 0 both, 1 sight, 2 sound
    const ci = sweepK % N;
    return {
      k, idx:ci, altIdx:ci, item:set.items[ci], altItem:set.items[ci],
      setName:set.name, kind:set.kind, count:N,
      cond:pass, voice: sig.voice|0, calibrating:!isRe, resweep:isRe, sweep:true,
      on: phase < on, phase, onMs:on, offMs:off,
      u: Math.min(1, phase/on),
      // the sweep is the measurement the whole run is scored against, so it is posed identically every time: no jitter, no scaling, no rotation
      dx:0, dy:0, scale:1, rot:0,
    };
  }
  const idx = (sig.order|0) === 1
    ? h32(k*2246822519 ^ (sig.seed|0)*3266489917) % N   // shuffled draw
    : k % N;                                            // sequential recital
  const j = sig.jitter || 0, sv = sig.scaleVar || 0, rv = sig.rotVar || 0;
  // Unimodal probe trials.
  // Cross-modal completion cannot be measured while both modalities are always present, so a fraction of presentations show one alone.
  // Which one is a pure function of the presentation index, so the two encoders agree without talking to each other and a run replays identically.
  // 0 both, 1 visual only, 2 audio only.
  const pe = Math.max(0, sig.probeEvery | 0);
  const isProbe = !!(pe && k % pe === pe - 1);
  // The block schedule sets what a training trial is.
  // Probe trials are the measurement and are posed identically in every block, so that a change in what a probe reads is a change in the network rather than a change in the question: the same one-channel probes run through the familiarization blocks and through the pairing block that follows them.
  const ph = phaseAt(sig.phases, tMs - calK*period);
  // A scheduled run probes in all three conditions rather than two.
  // Every measure the trainer computes is gated on having bimodal trials in the window, so a familiarization block that never presents the two channels together measures nothing at all, and the blocks that were supposed to be the control for the pairing block come back empty.
  // Rotating the probe through both, A alone and B alone puts a paired presentation in one trial in fifteen during every block, which is a small amount of pairing in the blocks before the pairing block and exactly the same amount in each of them.
  // So the comparison across blocks is between a little pairing and a lot of it, on an identical measurement, rather than between a number and a blank.
  const cond = isProbe
    ? (ph ? [1, 2, 0][Math.floor(k/pe) % 3] : ((Math.floor(k/pe) % 2) ? 2 : 1))
    : (ph ? ph.cond : 0);
  // Scrambled partner.
  // The same twelve items and the same twelve tones in the same proportions, but which tone accompanies which place is redrawn every presentation, so the pairing carries no information while every other property of the stimulus stream is unchanged.
  // This is the control that separates an association from generic sharpening.
  let altIdx = idx;
  if(sig.scramble){
    let z = (k*2654435761 + 12345) >>> 0;
    z = (z ^ (z >>> 15)) >>> 0;
    altIdx = z % N;
    if(altIdx === idx) altIdx = (altIdx + 1) % N;
  }
  return {
    k, idx, altIdx, item:set.items[idx], altItem:set.items[altIdx],
    setName:set.name, kind:set.kind, count:N,
    cond, voice: sig.voice|0,
    probe:isProbe, phase:ph ? ph.i : -1, phaseName:ph ? ph.name : '',
    on: phase < on,
    phase, onMs:on, offMs:off,
    // fraction of the presentation elapsed: drives the phone sequence in audio and the appearance transient in vision
    u: Math.min(1, phase/on),
    dx: (unit(k, 1, sig.seed)*2 - 1) * j,
    dy: (unit(k, 2, sig.seed)*2 - 1) * j,
    scale: 1 + (unit(k, 3, sig.seed)*2 - 1) * sv,
    rot: (unit(k, 4, sig.seed)*2 - 1) * rv,
  };
}

// ---- visual: silhouettes, glyphs and words ---------------------------
// Object drawings are deliberately simple outlines.
// Shape is what carries an object's name for children learning nouns, so silhouettes are the right level of detail for naming; they say nothing about texture, color or viewpoint, and the docs state that limitation.
function drawObject(c, item, r){
  c.lineWidth = Math.max(3, r*0.22);
  c.lineJoin = 'round'; c.lineCap = 'round';
  const P = (pts, close) => {
    c.beginPath();
    pts.forEach(([x, y], i) => i ? c.lineTo(x*r, y*r) : c.moveTo(x*r, y*r));
    if(close) c.closePath();
    c.stroke();
  };
  switch(item){
    case 'ball':
      c.beginPath(); c.arc(0, 0, r*0.85, 0, 7); c.stroke();
      c.beginPath(); c.moveTo(-r*0.85, 0); c.quadraticCurveTo(0, -r*0.5, r*0.85, 0); c.stroke();
      break;
    case 'apple':
      c.beginPath();
      c.moveTo(0, -r*0.45);
      c.bezierCurveTo(-r*1.05, -r*1.05, -r*1.0, r*0.85, 0, r*0.9);
      c.bezierCurveTo(r*1.0, r*0.85, r*1.05, -r*1.05, 0, -r*0.45);
      c.stroke();
      c.beginPath(); c.moveTo(0, -r*0.5); c.lineTo(r*0.1, -r*1.0); c.stroke();
      break;
    case 'banana':
      c.beginPath();
      c.moveTo(-r*0.9, -r*0.35);
      c.quadraticCurveTo(-r*0.2, r*1.0, r*0.9, r*0.15);
      c.quadraticCurveTo(r*0.2, r*0.55, -r*0.65, -r*0.5);
      c.closePath(); c.stroke();
      break;
    case 'cup':
      P([[-0.6,-0.7],[-0.45,0.8],[0.45,0.8],[0.6,-0.7]], false);
      c.beginPath(); c.moveTo(-0.6*r, -0.7*r); c.lineTo(0.6*r, -0.7*r); c.stroke();
      c.beginPath(); c.arc(r*0.62, r*0.0, r*0.35, -1.2, 1.2); c.stroke();
      break;
    case 'dog':
      c.beginPath(); c.ellipse(0, r*0.15, r*0.85, r*0.45, 0, 0, 7); c.stroke();
      c.beginPath(); c.arc(-r*0.85, -r*0.35, r*0.34, 0, 7); c.stroke();
      P([[-1.15,-0.6],[-1.0,-1.05],[-0.75,-0.6]], false);          // ear
      P([[0.8,-0.15],[1.15,-0.8]], false);                          // tail
      P([[-0.5,0.6],[-0.5,1.0]], false); P([[0.4,0.6],[0.4,1.0]], false);
      break;
    case 'cat':
      c.beginPath(); c.ellipse(0, r*0.3, r*0.7, r*0.42, 0, 0, 7); c.stroke();
      c.beginPath(); c.arc(-r*0.55, -r*0.45, r*0.4, 0, 7); c.stroke();
      P([[-0.95,-0.75],[-0.85,-1.15],[-0.55,-0.85]], false);        // ears
      P([[-0.3,-0.85],[-0.2,-1.15],[-0.05,-0.8]], false);
      P([[0.65,0.2],[1.05,-0.5]], false);                           // tail
      break;
    case 'tree':
      c.beginPath(); c.arc(0, -r*0.35, r*0.75, 0, 7); c.stroke();
      P([[-0.16,0.35],[-0.16,1.0],[0.16,1.0],[0.16,0.35]], false);
      break;
    case 'house':
      P([[-0.8,0.9],[-0.8,-0.1],[0.8,-0.1],[0.8,0.9]], true);
      P([[-1.0,-0.1],[0,-1.0],[1.0,-0.1]], false);
      break;
    case 'star': {
      const pts = [];
      for(let i=0;i<10;i++){
        const a = -Math.PI/2 + i*Math.PI/5, rr = i%2 ? 0.42 : 1.0;
        pts.push([Math.cos(a)*rr, Math.sin(a)*rr]);
      }
      P(pts, true);
      break;
    }
    case 'fish':
      c.beginPath();
      c.moveTo(-r*0.35, 0);
      c.quadraticCurveTo(r*0.15, -r*0.75, r*0.85, 0);
      c.quadraticCurveTo(r*0.15, r*0.75, -r*0.35, 0);
      c.stroke();
      P([[-0.35,0],[-1.0,-0.55],[-1.0,0.55]], true);                // tail
      break;
    case 'shoe':
      P([[-0.95,0.6],[-0.95,-0.1],[-0.2,-0.1],[0.3,-0.6],[0.6,-0.6],[0.95,0.1],[0.95,0.6]], true);
      break;
    case 'book':
      P([[-0.9,-0.65],[-0.9,0.65],[0,0.5],[0.9,0.65],[0.9,-0.65],[0,-0.5]], true);
      P([[0,-0.5],[0,0.5]], false);
      break;
    case 'circle': c.beginPath(); c.arc(0, 0, r*0.9, 0, 7); c.stroke(); break;
    case 'square': c.strokeRect(-r*0.85, -r*0.85, r*1.7, r*1.7); break;
    case 'triangle': P([[0,-0.95],[0.95,0.8],[-0.95,0.8]], true); break;
    case 'cross': P([[-0.9,-0.9],[0.9,0.9]], false); P([[0.9,-0.9],[-0.9,0.9]], false); break;
    default: c.fillRect(-r*0.9, -r*0.26, r*1.8, r*0.52);            // bar
  }
}

// low-pass the rendered grid: early acuity is poor, and blurred-then-sharp training produces better configural processing than sharp throughout (Vogelsang et al. 2018). acuity 1 = sharp, 0 = heavily blurred.
function blurGrid(g, cols, rows, acuity){
  const a = Math.max(0, Math.min(1, acuity));
  if(a >= 0.999) return g;
  const passes = a > 0.66 ? 1 : a > 0.33 ? 2 : 3;
  let cur = g;
  for(let p=0;p<passes;p++){
    const out = new Float32Array(cur.length);
    for(let r=0;r<rows;r++)
      for(let x=0;x<cols;x++){
        let s = 0, n = 0;
        for(let dr=-1;dr<=1;dr++){
          const rr = r+dr; if(rr < 0 || rr >= rows) continue;
          for(let dx=-1;dx<=1;dx++){
            const xx = x+dx; if(xx < 0 || xx >= cols) continue;
            s += cur[rr*cols + xx]; n++;
          }
        }
        out[r*cols + x] = s/n;
      }
    cur = out;
  }
  // blur spreads energy; renormalize so drive does not fall with acuity
  let mx = 0, mo = 0;
  for(let i=0;i<g.length;i++){ if(g[i] > mo) mo = g[i]; if(cur[i] > mx) mx = cur[i]; }
  if(mx > 1e-6 && mo > 1e-6){ const f = mo/mx; for(let i=0;i<cur.length;i++) cur[i] *= f; }
  return cur;
}

let gcv = null;
// ---- glyph atlas: canvas-free rendering for the engine host ----------
// The canvas path below needs a DOM canvas and a font, which the engine host does not have.
// The trainer therefore rasterizes each item once in the browser at canonical pose, and rendering becomes sampling that atlas through the pose transform, a pure array operation that runs identically in the page and on the host.
// Both sides use the atlas once it is set, so a run's stimuli do not depend on which side rendered.
let ATLAS = null;
export function setGlyphAtlas(a){ ATLAS = a || null; }
const hasAtlas = () => !!ATLAS;

// canonical raster of one item: S x S luminance, identity pose (browser only; this is the single remaining use of the canvas per item)
function rasterCanonical(kind, item, S){
  if(!gcv) gcv = document.createElement('canvas');
  if(gcv.width !== S){ gcv.width = S; gcv.height = S; }
  const c = gcv.getContext('2d', { willReadFrequently:true });
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.fillStyle = '#000'; c.fillRect(0, 0, S, S);
  c.save();
  c.translate(S/2, S/2);
  c.fillStyle = '#fff'; c.strokeStyle = '#fff';
  if(kind === 'glyph'){
    c.font = 'bold ' + Math.round(S*0.72) + 'px sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(item, 0, 0);
  } else if(kind === 'word'){
    c.font = 'bold ' + Math.round(S*0.30) + 'px sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(item, 0, 0);
  } else if(kind === 'book'){
    c.save(); c.translate(0, -S*0.20); drawObject(c, item, S*0.26); c.restore();
    c.font = 'bold ' + Math.round(S*0.20) + 'px sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(item, 0, S*0.30);
  } else {
    drawObject(c, item, S*0.32);
  }
  c.restore();
  const px = c.getImageData(0, 0, S, S).data;
  const g = new Float32Array(S*S);
  for(let i=0;i<S*S;i++) g[i] = (px[i*4] + px[i*4+1] + px[i*4+2])/765;
  return g;
}

export function buildAtlas(items, kind){
  const S = 96, out = { S, kind, items:{} };
  for(const it of items) out.items[it] = rasterCanonical(kind, it, S);
  return out;
}

export function renderItem(state, cols, rows, acuity){
  // Spots never touch the canvas.
  // A spot is a linear-falloff disc, and computing it directly makes the stimulus identical in the browser and in the engine host, which has no canvas; it is also the same box average over the same 96 pixel raster the canvas path uses, evaluated analytically at pixel centers instead of rasterized.
  if((state.kind || 'glyph') === 'spot'){
    const S = 96, idx = +state.item || 0, GW = 4, GH = 3;
    const gx0 = ((idx % GW) + 0.5)/GW - 0.5, gy0 = (Math.floor(idx/GW) + 0.5)/GH - 0.5;
    const a = (state.rot || 0)*Math.PI/180, ca = Math.cos(a), sa = Math.sin(a);
    const lx = gx0*S*0.78, ly = gy0*S*0.72;
    const px0 = S/2 + (state.dx || 0)*S*0.5 + lx*ca - ly*sa;
    const py0 = S/2 + (state.dy || 0)*S*0.5 + lx*sa + ly*ca;
    const rad = S*0.085*(state.scale === undefined ? 1 : state.scale);
    const g = new Float32Array(cols*rows);
    const bx = S/cols, by = S/rows;
    for(let rr=0;rr<rows;rr++)
      for(let x=0;x<cols;x++){
        let sum = 0, n = 0;
        const y0 = Math.floor(rr*by), y1 = Math.max(y0+1, Math.floor((rr+1)*by));
        const x0 = Math.floor(x*bx), x1 = Math.max(x0+1, Math.floor((x+1)*bx));
        for(let yy=y0;yy<y1;yy++) for(let xx=x0;xx<x1;xx++){
          const d = Math.hypot(xx + 0.5 - px0, yy + 0.5 - py0);
          sum += Math.max(0, 1 - d/rad); n++;
        }
        const v = sum/Math.max(1, n);
        if(v >= 0.03) g[(rows-1-rr)*cols + x] = v;
      }
    return acuity === undefined ? g : blurGrid(g, cols, rows, acuity);
  }
  // atlas sampling: the canonical raster is looked up through the inverse pose transform (the canvas path transforms then draws; sampling after the fact inverts the same transform), then box-averaged into the encoder grid exactly as the canvas path does
  if(ATLAS && ATLAS.items[state.item]){
    const S = 96, img = ATLAS.items[state.item], A = ATLAS.S;
    const cx = S/2 + (state.dx || 0)*S*0.5, cy = S/2 + (state.dy || 0)*S*0.5;
    const a = -(state.rot || 0)*Math.PI/180, ca = Math.cos(a), sa = Math.sin(a);
    const inv = 1/(state.scale || 1), r = A/S;
    const samp = (x, y) => {
      const fx = Math.min(A - 1.001, Math.max(0, x)), fy = Math.min(A - 1.001, Math.max(0, y));
      const x0 = Math.floor(fx), y0 = Math.floor(fy), tx = fx - x0, ty = fy - y0;
      const o = y0*A + x0;
      return img[o]*(1-tx)*(1-ty) + img[o+1]*tx*(1-ty) +
        img[o+A]*(1-tx)*ty + img[o+A+1]*tx*ty;
    };
    const g = new Float32Array(cols*rows);
    const bx = S/cols, by = S/rows;
    for(let rr=0;rr<rows;rr++)
      for(let x=0;x<cols;x++){
        let sum = 0, n = 0;
        const y0 = Math.floor(rr*by), y1 = Math.max(y0+1, Math.floor((rr+1)*by));
        const x0 = Math.floor(x*bx), x1 = Math.max(x0+1, Math.floor((x+1)*bx));
        for(let yy=y0;yy<y1;yy++) for(let xx=x0;xx<x1;xx++){
          const ux = xx + 0.5 - cx, uy = yy + 0.5 - cy;
          const rx = (ux*ca - uy*sa)*inv + S/2, ry = (ux*sa + uy*ca)*inv + S/2;
          sum += samp(rx*r - 0.5, ry*r - 0.5); n++;
        }
        const v = sum/Math.max(1, n);
        if(v >= 0.03) g[(rows-1-rr)*cols + x] = v;
      }
    return acuity === undefined ? g : blurGrid(g, cols, rows, acuity);
  }
  if(!gcv) gcv = document.createElement('canvas');
  const S = 96;
  if(gcv.width !== S){ gcv.width = S; gcv.height = S; }
  const c = gcv.getContext('2d', { willReadFrequently:true });
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.fillStyle = '#000'; c.fillRect(0, 0, S, S);
  c.save();
  c.translate(S/2 + state.dx*S*0.5, S/2 + state.dy*S*0.5);
  c.rotate(state.rot*Math.PI/180);
  const s = state.scale;
  c.fillStyle = '#fff'; c.strokeStyle = '#fff';
  const kind = state.kind || 'glyph', item = state.item;
  if(kind === 'glyph'){
    c.font = 'bold ' + Math.round(S*0.72*s) + 'px sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(item, 0, 0);
  } else if(kind === 'word'){
    c.font = 'bold ' + Math.round(S*0.30*s) + 'px sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(item, 0, 0);
  } else if(kind === 'book'){
    // a picture-book page: the object above, its written word below
    c.save(); c.translate(0, -S*0.20); drawObject(c, item, S*0.26*s); c.restore();
    c.font = 'bold ' + Math.round(S*0.20*s) + 'px sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(item, 0, S*0.30);
  } else {
    drawObject(c, item, S*0.32*s);
  }
  c.restore();
  // downsample into the encoder grid (luminance, flipped so image top lands on the high end of the mapped axis, as the read node does)
  const px = c.getImageData(0, 0, S, S).data;
  const g = new Float32Array(cols*rows);
  const bx = S/cols, by = S/rows;
  for(let r=0;r<rows;r++)
    for(let x=0;x<cols;x++){
      let sum = 0, n = 0;
      const y0 = Math.floor(r*by), y1 = Math.max(y0+1, Math.floor((r+1)*by));
      const x0 = Math.floor(x*bx), x1 = Math.max(x0+1, Math.floor((x+1)*bx));
      for(let yy=y0;yy<y1;yy++) for(let xx=x0;xx<x1;xx++){
        const o = (yy*S + xx)*4;
        sum += (px[o] + px[o+1] + px[o+2])/765; n++;
      }
      const v = sum/Math.max(1, n);
      if(v >= 0.03) g[(rows-1-r)*cols + x] = v;
    }
  return acuity === undefined ? g : blurGrid(g, cols, rows, acuity);
}

// ---- audio: formant-like spectra -------------------------------------
// English letter names cluster into vowel families, so the alphabet set carries real confusability structure (B, C, D, E, G, P, T, V and Z all share the same vowel) that a learner has to resolve from the consonant onset rather than the steady vowel.
// Words are rendered as a sequence of phones across the presentation, which gives the audio channel the temporal structure that spoken words actually have.
// Formant values sit in typical adult ranges (Peterson and Barney 1952 style tables).
// This is synthetic formant structure, not recorded speech; recorded audio belongs on the read node.
const VOWEL = {
  'i':  [300, 2300], 'ey': [500, 2100], 'e':  [600, 1800],
  'ay': [700, 1300], 'o':  [500, 900],  'u':  [330, 900],
  'ar': [650, 1100], 'a':  [730, 1090], 'ih': [390, 1990],
  'uh': [520, 1190], 'aw': [570, 840],
};
const LETTER_VOWEL = {
  A:'ey', B:'i', C:'i', D:'i', E:'i', F:'e', G:'i', H:'ey', I:'ay', J:'ey',
  K:'ey', L:'e', M:'e', N:'e', O:'o', P:'i', Q:'u', R:'ar', S:'e', T:'i',
  U:'u', V:'i', W:'u', X:'e', Y:'ay', Z:'i',
};
// consonant onset: burst center frequency and noisiness
const ONSET = {
  B:[700,0.2], C:[3400,0.9], D:[2600,0.4], F:[5200,1.0], G:[2000,0.4],
  H:[3000,0.8], J:[2400,0.7], K:[3200,0.8], L:[1400,0.2], M:[900,0.15],
  N:[1600,0.2], P:[900,0.5], Q:[3200,0.8], R:[1300,0.3], S:[6000,1.0],
  T:[3800,0.7], V:[1200,0.5], W:[800,0.2], X:[5600,1.0], Y:[2200,0.3],
  Z:[4800,1.0],
};
// word pronunciations as phone sequences.
// Vowel phones name a VOWEL entry; consonants name an ONSET entry.
// Kept small and hand-checked rather than guessed by rule, since English spelling does not map cleanly to sound.
const WORD_PHONES = {
  ball:  ['B','aw','L'],      dog:   ['D','aw','G'],
  cat:   ['K','a','T'],       apple: ['a','P','uh','L'],
  banana:['B','uh','N','a','N','uh'], cup: ['K','uh','P'],
  shoe:  ['S','u'],           book:  ['B','u','K'],
  tree:  ['T','R','i'],       star:  ['S','T','ar'],
  fish:  ['F','ih','S'],      house: ['H','a','S'],
  sat:   ['S','a','T'],       tap:   ['T','a','P'],
  pin:   ['P','ih','N'],      pat:   ['P','a','T'],
  tin:   ['T','ih','N'],      nap:   ['N','a','P'],
  sit:   ['S','ih','T'],      pit:   ['P','ih','T'],
  nip:   ['N','ih','P'],      tan:   ['T','a','N'],
  circle:['S','er','K','L'],  square:['S','K','W','e','R'],
  triangle:['T','R','ay','a','N','G','L'], cross:['K','R','aw','S'],
  bar:   ['B','ar'],
};

// ---- recorded voice bank ---------------------------------------------
// Band-envelope spectrograms of recorded speech, produced offline from the teleprompter clips and loaded at run start (the trainer fetches it in the browser; the engine host reads it from disk).
// Envelopes use the same log band layout as spectrum() below, so a recorded clip and a synthetic rendition drive the cochlea through one code path.
// Null until loaded; renderAudio falls back to formants.
let VOICE = null;
export function setVoice(bank){
  if(!bank || !bank.items){ VOICE = null; return; }
  const dec = typeof atob === 'function'
    ? s => Uint8Array.from(atob(s), ch => ch.charCodeAt(0))
    : s => new Uint8Array(Buffer.from(s, 'base64'));
  const items = {};
  for(const key in bank.items){
    items[key] = bank.items[key].map(p => ({
      pass: p.pass, n: p.n, env: dec(p.d) }));
  }
  VOICE = { bands: bank.bands, items };
}
const hasVoice = () => !!VOICE;

// A recorded rendition of the current item, time-warped to the presentation window.
// Which pass speaks is a pure function of the presentation index, with index 0 reserved for the synthetic formants, so recordings and synthesis interleave deterministically and a run replays identically.
function voiceSpectrum(state, bands){
  const passes = VOICE.items[state.item];
  if(!passes || !passes.length) return null;
  const pick = h32(state.k*2718281 ^ 421) % (passes.length + 1);
  if(pick === 0) return null;                 // synthetic turn
  const p = passes[pick - 1];
  const fi = Math.min(p.n - 1, Math.floor(state.u * p.n));
  const row = fi * VOICE.bands, g = new Float32Array(bands);
  for(let b = 0; b < bands; b++){
    const x = (b + 0.5) * VOICE.bands / bands - 0.5;
    const i0 = Math.max(0, Math.min(VOICE.bands - 1, Math.floor(x)));
    const i1 = Math.min(VOICE.bands - 1, i0 + 1);
    const t = Math.min(1, Math.max(0, x - i0));
    g[b] = (p.env[row + i0]*(1 - t) + p.env[row + i1]*t) / 255;
  }
  return g;
}

function spectrum(bands, adds){
  const g = new Float32Array(bands);
  const bandOf = f => Math.log(Math.max(80, Math.min(8000, f))/80)/Math.log(100)*bands;
  for(const [f, amp, width] of adds){
    const c = bandOf(f);
    for(let b=0;b<bands;b++){
      const d = (b + 0.5 - c)/width;
      const v = amp*Math.exp(-0.5*d*d);
      if(v > 0.01) g[b] += v;
    }
  }
  let mx = 0;
  for(let b=0;b<bands;b++) if(g[b] > mx) mx = g[b];
  if(mx > 1) for(let b=0;b<bands;b++) g[b] /= mx;
  return g;
}

// band gains for the current lesson state, matched to the input encoder's log-spaced 80 Hz to 8 kHz band layout
export function renderAudio(state, bands){
  const item = state.item, kind = state.kind || 'glyph';
  if(state.voice && VOICE && kind !== 'spot'){
    const g = voiceSpectrum(state, bands);
    if(g) return g;
  }
  if(kind === 'spot'){
    // One narrow tone per item and nothing else: no harmonic, and a width that lands on about two bands.
    // A spectrum wide enough to cover most of the cochlea makes every item nearly the same input, which is the same problem glyphs have on the retina.
    const idx = +item || 0;
    const f = 220*Math.pow(2, idx/3);
    const env = state.u < 0.1 ? state.u/0.1 : (state.u > 0.9 ? (1-state.u)/0.1 : 1);
    return spectrum(bands, [[f, env, 0.18]]);
  }
  // single letters: consonant onset then the letter-name vowel
  if(kind === 'glyph' && LETTER_VOWEL[item]){
    const f = VOWEL[LETTER_VOWEL[item]], on = ONSET[item];
    const adds = [];
    if(on && state.u < 0.2){
      const env = 1 - state.u/0.2;
      adds.push([on[0], 0.9*env, 1.2 + 2.5*on[1]]);
      if(on[1] > 0.6) adds.push([on[0]*1.5, 0.5*env*on[1], 3]);
    }
    const venv = state.u < 0.15 ? state.u/0.15 : 1;
    const glide = (LETTER_VOWEL[item] === 'ey' || LETTER_VOWEL[item] === 'ay')
      ? Math.max(0, (state.u - 0.3)/0.7) : 0;
    adds.push([f[0] + (300 - f[0])*glide*0.6, 1.0*venv, 1.1]);
    adds.push([f[1] + (2300 - f[1])*glide*0.6, 0.8*venv, 1.4]);
    adds.push([2900, 0.25*venv, 1.6]);
    return spectrum(bands, adds);
  }
  // words and named objects: walk the phone sequence across the presentation, vowels held longer than consonants
  const phones = WORD_PHONES[item];
  if(phones && phones.length){
    const dur = phones.map(p => VOWEL[p] ? 2.2 : 1);
    const total = dur.reduce((a, b) => a + b, 0);
    let acc = 0, pi = 0, frac = 0;
    for(let i=0;i<phones.length;i++){
      const w = dur[i]/total;
      if(state.u < acc + w || i === phones.length-1){
        pi = i; frac = Math.min(1, Math.max(0, (state.u - acc)/w)); break;
      }
      acc += w;
    }
    const p = phones[pi];
    // brief taper at each phone boundary so transitions are audible events
    const env = Math.min(1, Math.min(frac, 1 - frac)*6 + 0.25);
    if(VOWEL[p]){
      const f = VOWEL[p];
      return spectrum(bands, [[f[0], 1.0*env, 1.1], [f[1], 0.8*env, 1.4],
        [2900, 0.25*env, 1.6]]);
    }
    const on = ONSET[p] || [2000, 0.5];
    return spectrum(bands, [[on[0], 0.95*env, 1.2 + 2.5*on[1]],
      [on[0]*1.5, 0.45*env*on[1], 3]]);
  }
  // anything without a pronunciation: a stable two-tone signature so the audio channel still carries item identity
  const sf = (item.charCodeAt(0)*37) % 9;
  return spectrum(bands, [[250 + sf*90, 1.0, 1.2], [1100 + sf*330, 0.7, 1.5]]);
}

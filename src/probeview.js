import { NEURON_TYPES } from './nodes.js';
// Live probe readouts, shared by the playground and the training runner.
// Rate sparklines, decoded image views, and audio playback are all pure observation: they read the spike flags the engine already reports.
// The image view bins spikes back through the region's spatial map, and audio plays spikes as crackle or tonotopic bands as tones.
// Browsers require one click on the speaker button before audio can start.
let probesEl = null, probesPanel = null;
// the panel carries a header and a body; the probes go in the body, the panel shows and hides as a whole
export function attachProbes(el){ probesPanel = el; probesEl = el.querySelector('.ovbody') || el; }
let probeState = [], audioCtx = null, audioOn = false, noiseBuf = null;
function stopProbeAudio(){
  for(const p of probeState) if(p.oscs){
    for(const o of p.oscs){ try { o.osc.stop(); } catch(e){} }
    p.oscs = null;
  }
}
function ensureAudio(){
  if(!audioCtx){
    audioCtx = new AudioContext();
    noiseBuf = audioCtx.createBuffer(1, 512, audioCtx.sampleRate);
    const d = noiseBuf.getChannelData(0);
    for(let i=0;i<d.length;i++) d[i] = (Math.random()*2-1) * (1 - i/d.length);
  }
  if(audioCtx.state === 'suspended') audioCtx.resume();
}
export function setupProbes(net){
  stopProbeAudio();
  const ps = (net && net.probes) || [];
  // A population of graded cells (MODEL.md 1) never spikes, so its readout is the mean release, from the potentials the engine streams when asked (probesWantV below); a mixed population reads as spikes.
  const gradedOf = i => { const t = NEURON_TYPES[net.ntype[i]]; return t && t.f7 && t.f7.graded ? t.f7.graded : null; };
  // A tune (a stimulus edit, a drive change) computes again without rewiring, and a readout starting from nothing each time would read as a rewiring.
  // A probe over the same cells as before keeps its trace.
  const prev = new Map((probeState || []).map(p => [p.label, p]));
  const sameCells = (a, b) => a && b && a.length === b.length && a.every((v, k) => v === b[k]);
  probeState = ps.map(p => {
    const graded = !!(net && net.ntype && p.idx.length && p.idx.every(i => gradedOf(i)));
    const old = prev.get(p.label);
    const kept = old && sameCells(old.idx, p.idx) && old.graded === graded ? old : null;
    const st = { label:p.label, idx:p.idx, ema:kept ? kept.ema : 0,
      hist:kept ? kept.hist : new Float32Array(80), hp:kept ? kept.hp : 0, audio:p.audio|0,
      chImg:p.chImg, imgCols:p.imgCols, imgRows:p.imgRows,
      chBand:p.chBand, nBands:p.nBands, graded };
    if(graded){ st.gthr = Float32Array.from(p.idx, i => gradedOf(i).thr); st.gslp = Float32Array.from(p.idx, i => gradedOf(i).slope || 1); }
    return st;
  });
  (probesPanel || probesEl).style.display = probeState.length ? 'block' : 'none';
  probesEl.innerHTML = '';
  if(probeState.some(p => p.audio)){
    const btn = document.createElement('button');
    btn.textContent = audioOn ? 'audio on' : 'audio off';
    btn.onclick = () => {
      audioOn = !audioOn;
      if(audioOn) ensureAudio();
      else if(audioCtx){
        for(const p of probeState) if(p.oscs)
          for(const o of p.oscs) o.g.gain.setTargetAtTime(0, audioCtx.currentTime, 0.02);
        audioCtx.suspend();
      }
      btn.textContent = audioOn ? 'audio on' : 'audio off';
    };
    probesEl.appendChild(btn);
  }
  for(const p of probeState){
    const row = document.createElement('div'); row.className = 'probe';
    const lab = document.createElement('span'); lab.textContent = p.label;
    const val = document.createElement('span'); val.className = 'pval';
    const cv = document.createElement('canvas'); cv.width = 90; cv.height = 22;
    row.append(lab, val, cv); probesEl.appendChild(row);
    // A rate read once is one sample of a fluctuating thing, so the line under it says what the last few seconds of readings were: their mean, their spread, and how long that is.
    const sp = document.createElement('div'); sp.className = 'pspread';
    probesEl.appendChild(sp);
    p.valEl = val; p.cv = cv; p.spEl = sp;
    // the readings themselves, not the smoothed line the sparkline draws: a spread taken from a smoothed trace understates what was seen
    p.raw = new Float32Array(p.hist.length);
    p.stamp = new Float64Array(p.hist.length);
    if(p.chImg){
      const img = document.createElement('canvas');
      img.width = 96; img.height = Math.round(96*p.imgRows/p.imgCols);
      img.className = 'pimg';
      probesEl.appendChild(img);
      p.imgCv = img;
      p.off = document.createElement('canvas');
      p.off.width = p.imgCols; p.off.height = p.imgRows;
      p.acc = new Float32Array(p.imgCols*p.imgRows);
      p.accMax = 0.001;
    }
    if(p.chBand) p.bandAcc = new Float32Array(p.nBands);
  }
}
// Display only: the trainer computes its checkpoint rates from its own accumulation, so throttling this changes nothing measured.
// It walks every probed neuron, which at full density is tens of thousands per call, so it does not run on every state message, nor while the tab is hidden and drawing nothing.
let lastDraw = 0;
// the wall clock, wherever this runs: the training page has no performance object in every context this module is loaded from
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
// whether any probe reads a graded population, which needs the potentials
export function probesWantV(){ return probeState.some(p => p.graded); }
// dir: the direction of an alternating bar sweep (0 or 1), or null.
// With one, every probe keeps a slow average per direction and shows both with the index (forward minus backward over their sum), which is how a direction-selective cell reads; the averages are slow (a few sweeps) so a reading needs several alternations to settle.
export function updateProbes(fired, steps, force, v, dir){
  if(!probeState.length) return;
  const now = performance.now();
  if(!force){
    if(typeof document !== 'undefined' && document.hidden) return;
    if(now - lastDraw < 100) return;          // ~10 Hz is plenty for a readout
    lastDraw = now;
  }
  return updateProbesNow(fired, steps, v, dir);
}
const byDir = (p, dir, value, unit) => {
  if(!p.dirEma) p.dirEma = [null, null];
  const a = 0.02;                          // slow, so a sweep's worth of frames counts
  p.dirEma[dir] = p.dirEma[dir] === null ? value : p.dirEma[dir]*(1 - a) + value*a;
  const f = p.dirEma[0], b = p.dirEma[1];
  if(f === null || b === null) return (dir ? 'back ' : 'fwd ') + value.toFixed(1) + unit + ' …';
  const di = f + b > 0 ? (f - b)/(f + b) : 0;
  return 'fwd ' + f.toFixed(1) + ' back ' + b.toFixed(1) + unit + ' · DI ' + (di >= 0 ? '+' : '') + di.toFixed(2);
};
// mean and spread of the readings kept in a probe's ring, with the wall clock they cover: the ring is written every update and holds the last 80
const spreadOf = p => {
  const n = Math.min(p.hp, p.hist.length);
  if(n < 4 || !p.raw) return '';
  let sum = 0;
  for(let k = 0; k < n; k++) sum += p.raw[k];
  const mu = sum/n;
  let v2 = 0;
  for(let k = 0; k < n; k++){ const d = p.raw[k] - mu; v2 += d*d; }
  const sd = Math.sqrt(v2/(n - 1));
  const unit = p.graded ? '%' : ' Hz';
  const dp = p.graded ? 0 : 1;
  let span = '';
  if(p.stamp){
    let lo = Infinity, hi = 0;
    for(let k = 0; k < n; k++){ const t = p.stamp[k]; if(t){ if(t < lo) lo = t; if(t > hi) hi = t; } }
    if(hi > lo) span = ' over ' + ((hi - lo)/1000).toFixed(1) + ' s';
  }
  return mu.toFixed(dp) + ' ± ' + sd.toFixed(dp) + unit + span;
};
function updateProbesNow(fired, steps, v, dir){
  const split = dir === 0 || dir === 1;
  for(const p of probeState){
    // fired is each cell's spike count over the tick (ENGINE.md): counted as a flag, the rate below could not pass 1000/steps Hz
    let c = 0; const idx = p.idx;
    for(let k=0;k<idx.length;k++) c += fired[idx[k]];
    if(p.graded){
      if(!v){ p.valEl.textContent = 'release …'; continue; }
      let r = 0;
      for(let k=0;k<idx.length;k++) r += Math.min(1, Math.max(0, (v[idx[k]] - p.gthr[k])/p.gslp[k]));
      const rel = 100*r/idx.length;
      p.ema = p.ema*0.85 + rel*0.15;
      if(p.stamp) p.stamp[p.hp % p.hist.length] = now();
      if(p.raw) p.raw[p.hp % p.hist.length] = rel;
      p.hist[p.hp++ % p.hist.length] = p.ema;
      p.valEl.textContent = split ? byDir(p, dir, rel, '%') : p.ema.toFixed(0) + '% release';
      if(p.spEl) p.spEl.textContent = spreadOf(p);
    } else {
    const hz = c/idx.length/steps*1000;
    p.ema = p.ema*0.85 + hz*0.15;
    if(p.stamp) p.stamp[p.hp % p.hist.length] = now();
    if(p.raw) p.raw[p.hp % p.hist.length] = hz;
    p.hist[p.hp++ % p.hist.length] = p.ema;
    p.valEl.textContent = split ? byDir(p, dir, hz, ' Hz') : p.ema.toFixed(1) + ' Hz';
    if(p.spEl) p.spEl.textContent = spreadOf(p);
    }
    const ctx = p.cv.getContext('2d'), W = p.cv.width, H = p.cv.height;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
    const max = p.graded ? 100 : Math.max(5, ...p.hist);
    ctx.strokeStyle = 'hsl(190,55%,60%)'; ctx.beginPath();
    for(let k=0;k<p.hist.length;k++){
      const v = p.hist[(p.hp + k) % p.hist.length];
      const x = k/(p.hist.length-1)*W, y = H - 1 - v/max*(H-2);
      if(k) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.stroke();
    if(p.chImg){
      const acc = p.acc;
      for(let k=0;k<acc.length;k++) acc[k] *= 0.82;
      for(let k=0;k<idx.length;k++) acc[p.chImg[k]] += fired[idx[k]];
      let fm = 0;
      for(let k=0;k<acc.length;k++) if(acc[k] > fm) fm = acc[k];
      p.accMax = Math.max(p.accMax*0.995, fm, 0.001);
      const o = p.off.getContext('2d');
      const im = o.createImageData(p.imgCols, p.imgRows);
      for(let k=0;k<acc.length;k++){
        const v = Math.min(255, Math.round(acc[k]/p.accMax*255));
        im.data[k*4] = v; im.data[k*4+1] = v; im.data[k*4+2] = v; im.data[k*4+3] = 255;
      }
      o.putImageData(im, 0, 0);
      const g = p.imgCv.getContext('2d');
      g.imageSmoothingEnabled = true;
      g.drawImage(p.off, 0, 0, p.imgCv.width, p.imgCv.height);
    }
    if(audioOn && audioCtx && p.audio === 1 && c > 0){
      const src = audioCtx.createBufferSource();
      src.buffer = noiseBuf;
      const g = audioCtx.createGain();
      g.gain.value = Math.min(0.35, 0.02 + c*0.01);
      src.connect(g); g.connect(audioCtx.destination);
      src.start();
    }
    if(audioOn && audioCtx && p.audio === 2 && p.chBand){
      if(!p.oscs){
        p.oscs = [];
        for(let b=0;b<p.nBands;b++){
          const osc = audioCtx.createOscillator();
          osc.frequency.value = 80 * Math.pow(100, b/p.nBands);   // matches the input encoder
          const g = audioCtx.createGain(); g.gain.value = 0;
          osc.connect(g); g.connect(audioCtx.destination);
          osc.start();
          p.oscs.push({ osc, g });
        }
        p.bandN = new Float32Array(p.nBands);
        for(let k=0;k<p.chBand.length;k++) p.bandN[p.chBand[k]]++;
      }
      const counts = new Float32Array(p.nBands);
      for(let k=0;k<idx.length;k++) counts[p.chBand[k]] += fired[idx[k]];
      for(let b=0;b<p.nBands;b++){
        const r = counts[b]/Math.max(1, p.bandN[b]);
        p.bandAcc[b] = p.bandAcc[b]*0.8 + r*0.2;
        p.oscs[b].g.gain.setTargetAtTime(
          Math.min(0.2, p.bandAcc[b]*2.5), audioCtx.currentTime, 0.05);
      }
    }
  }
}

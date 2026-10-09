// Live recording of what the engine reports, for figures that leave the app: a rate per frame for every probed population, a rolling spike raster over a sample of its cells, and an interval histogram per cell.
//
// Observation only, and pure: the page hands in the spike flags it already receives with each state message (a count per neuron over the frame's steps) and reads series back for the chart node.
// Nothing here reaches an engine or a computation.
// Time is the simulated clock in milliseconds, advanced by the frame's step count, so a figure's x axis is simulated time; a frame is the engine's tick (2 ms by default), which is the resolution of every spike time recorded here.
//
// Bounds, so an hour of watching costs a fixed amount: a rate series keeps the last SERIES frames (a minute at 2 ms), a raster the last EVENTS spikes over at most ROWS cells (a stride through the population when it is larger, the same cells throughout), and the interval histogram is BINS bins of BIN_MS up to a ceiling with an overflow bin.

export const SERIES = 30000, EVENTS = 200000, ROWS = 1000, BINS = 50, BIN_MS = 4;

export class Recorder {
  constructor(){ this.reset(); }
  reset(){ this.t = 0; this.frames = 0; this.pops = new Map(); this.stepMs = 0; }
  // [{ label, idx }] from the network's probes; called after every computation
  setPopulations(list){
    // a population over the same cells as before keeps its recording, so a tune (a stimulus edit) does not restart every plot; reset() is the clean slate, for a new engine
    const prev = this.pops;
    this.pops = new Map();
    for(const p of list || []){
      if(!p || !p.idx || !p.idx.length) continue;
      const old = prev.get(p.label);
      if(old && old.n === p.idx.length && old.idx.every((v, k) => v === p.idx[k])){ this.pops.set(p.label, old); continue; }
      const n = p.idx.length, stride = Math.max(1, Math.ceil(n/ROWS));
      const rows = []; for(let k = 0; k < n; k += stride) rows.push(k);
      this.pops.set(p.label, {
        label:p.label, idx:p.idx, n,
        // rate per frame, a ring
        hz:new Float32Array(SERIES), tf:new Float32Array(SERIES), hp:0, filled:0,
        // raster: (t, row) pairs in two rings over the sampled rows
        rows, rowOf:new Map(rows.map((k, r) => [p.idx[k], r])),
        et:new Float32Array(EVENTS), er:new Uint16Array(EVENTS), ep:0, efilled:0,
        // intervals: last spike time per cell (frame resolution) and the histogram
        last:new Float32Array(n).fill(-1), isi:new Uint32Array(BINS + 1), nIsi:0,
      });
    }
  }
  // one state message: fired[i] is the spike count of neuron i over `steps` ms
  onState(fired, steps){
    if(!fired) return;
    const t0 = this.t, t1 = t0 + steps;
    this.t = t1; this.frames++; this.stepMs = steps;
    for(const p of this.pops.values()){
      let count = 0;
      const idx = p.idx, last = p.last, isi = p.isi;
      for(let k = 0; k < p.n; k++){
        const c = fired[idx[k]];
        if(!c) continue;
        count += c;
        if(last[k] >= 0){
          const d = t1 - last[k];
          const b = Math.min(BINS, Math.floor(d/BIN_MS));
          isi[b]++; p.nIsi++;
        }
        last[k] = t1;
        const r = p.rowOf.get(idx[k]);
        if(r !== undefined){ p.et[p.ep] = t1; p.er[p.ep] = r; p.ep = (p.ep + 1) % EVENTS; if(p.efilled < EVENTS) p.efilled++; }
      }
      p.hz[p.hp] = count/(p.n*steps/1000); p.tf[p.hp] = t1;
      p.hp = (p.hp + 1) % SERIES; if(p.filled < SERIES) p.filled++;
    }
  }
  labels(){ return [...this.pops.keys()]; }
  // rate over the last windowMs, as [t_ms, hz] points oldest first, smoothed over `smoothMs` of frames so a 2 ms frame does not read as noise
  series(label, windowMs = 5000, smoothMs = 20){
    const p = this.pops.get(label); if(!p || !p.filled) return [];
    const from = this.t - windowMs, out = [];
    const k = Math.max(1, Math.round(smoothMs/Math.max(1, this.stepMs)));
    const start = (p.hp - p.filled + SERIES) % SERIES;
    let acc = 0, q = [];
    for(let j = 0; j < p.filled; j++){
      const i = (start + j) % SERIES;
      acc += p.hz[i]; q.push(p.hz[i]); if(q.length > k) acc -= q.shift();
      if(p.tf[i] >= from) out.push([p.tf[i], acc/q.length]);
    }
    return out;
  }
  // spikes over the last windowMs as [t_ms, row] pairs, oldest first, and how many rows the raster has and which cells they are
  raster(label, windowMs = 5000){
    const p = this.pops.get(label); if(!p || !p.efilled) return { events:[], rows:0, cells:[] };
    const from = this.t - windowMs, out = [];
    const start = (p.ep - p.efilled + EVENTS) % EVENTS;
    for(let j = 0; j < p.efilled; j++){
      const i = (start + j) % EVENTS;
      if(p.et[i] >= from) out.push([p.et[i], p.er[i]]);
    }
    return { events:out, rows:p.rows.length, cells:p.rows.map(k => p.idx[k]), stride:Math.max(1, Math.ceil(p.n/ROWS)) };
  }
  // the interval histogram: counts per bin, the bin edges in ms, and the last bin as everything at or past the ceiling
  intervals(label){
    const p = this.pops.get(label); if(!p) return { counts:[], edges:[], total:0 };
    return { counts:[...p.isi], edges:Array.from({ length:BINS + 2 }, (_, i) => i*BIN_MS), total:p.nIsi, ceiling:BINS*BIN_MS };
  }
  // the whole population's mean rate over the last windowMs, in Hz
  meanRate(label, windowMs = 1000){
    const s = this.series(label, windowMs, 1);
    return s.length ? s.reduce((a, b) => a + b[1], 0)/s.length : 0;
  }
}

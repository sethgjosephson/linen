// A recording of a live session: what the probes read, for as long as REC is on, and everything done to the tissue meanwhile, each at its simulated time.
// The chart node's live plots keep a minute (record.js); this keeps the whole session at a coarser bin, unbounded, and writes it out as one file with the edit log, so a figure can say "at 33 s the pulse onset went from 1 to 2 ms and this is what the rate did".
//
// Pure: fed the spike flags the page already receives, the parameter edits the page already sees, and the weight dumps the engine already gives.
// The clock is its own, advanced by the frame's step count, so a rewiring in the middle (a new engine, whose clock restarts) does not restart the record; it is marked instead.

export const BIN_MS = 20;

// summary of one weight dump: E and I by sign, for a before and an after
export function weightSummary(w){
  if(!w || !w.length) return null;
  const s = { count:w.length, e:{ n:0, sum:0, sq:0, max:0 }, i:{ n:0, sum:0, sq:0, max:0 } };
  for(let k = 0; k < w.length; k++){
    const v = w[k]; if(v === 0) continue;
    const a = v > 0 ? s.e : s.i, m = Math.abs(v);
    a.n++; a.sum += m; a.sq += m*m; if(m > a.max) a.max = m;
  }
  const fin = a => a.n ? { n:a.n, mean:+(a.sum/a.n).toFixed(5), sd:+Math.sqrt(Math.max(0, a.sq/a.n - (a.sum/a.n)**2)).toFixed(5), max:+a.max.toFixed(4) } : { n:0 };
  return { count:s.count, e:fin(s.e), i:fin(s.i) };
}

export class Recording {
  // scene: a name for the file; pops: [{ label, idx }] from the network's probes
  constructor(scene, pops){
    this.scene = scene || 'scene';
    this.started = new Date().toISOString();
    this.t = 0; this.frames = 0; this.binMs = BIN_MS;
    this.pops = new Map();
    this.events = [];                      // { t, kind:'edit'|'mark', ... } in time order
    this.weights = { before:null, after:null };
    this.intervals = null;
    this.setPopulations(pops);
  }
  setPopulations(pops){
    for(const p of pops || []){
      if(!p || !p.idx || !p.idx.length) continue;
      const old = this.pops.get(p.label);
      if(old){ old.idx = p.idx; old.n = p.idx.length; continue; }   // a rewiring: same name, its cells as they are now
      this.pops.set(p.label, { label:p.label, idx:p.idx, n:p.idx.length, hz:[], count:0, from:Math.floor(this.t/BIN_MS)*BIN_MS });
    }
  }
  // one state message: fired[i] spikes of neuron i over `steps` ms
  onState(fired, steps){
    if(!fired || !(steps > 0)) return;
    const t0 = this.t; this.t += steps; this.frames++;
    for(const p of this.pops.values()){
      let c = 0;
      for(let k = 0; k < p.n; k++){ const v = fired[p.idx[k]]; if(v) c += v; }
      p.count += c;
    }
    if(Math.floor(this.t/BIN_MS) > Math.floor(t0/BIN_MS)) this._close();
  }
  _close(){
    const binMs = this.t - this._binT0(), ms = binMs > 0 ? binMs : BIN_MS;
    for(const p of this.pops.values()){
      p.hz.push(p.n ? +(p.count/(p.n*ms/1000)).toFixed(3) : 0);
      p.count = 0;
    }
    this._lastBin = this.t;
  }
  _binT0(){ return this._lastBin === undefined ? 0 : this._lastBin; }
  // a setting changed: node is the node's name, key the setting
  edit(node, key, from, to){ this.events.push({ t:Math.round(this.t), kind:'edit', node, key, from, to }); }
  // something else done: pause, resume, lock, unlock, rewire
  mark(what, detail){ this.events.push({ t:Math.round(this.t), kind:'mark', what, ...(detail ? { detail } : {}) }); }
  seconds(){ return this.t/1000; }
  edits(){ return this.events.filter(e => e.kind === 'edit').length; }
  labels(){ return [...this.pops.keys()]; }
  // [t_ms, hz] points for a population, whole recording
  series(label){
    const p = this.pops.get(label); if(!p) return [];
    const out = [];
    for(let k = 0; k < p.hz.length; k++) out.push([p.from + (k + 1)*BIN_MS, p.hz[k]]);
    return out;
  }
  toJSON(){
    return { format:'linen-recording-1', scene:this.scene, started:this.started, binMs:BIN_MS, ms:Math.round(this.t),
      populations:[...this.pops.values()].map(p => ({ label:p.label, cells:p.n, from:p.from, hz:p.hz })),
      events:this.events, weights:this.weights, intervals:this.intervals };
  }
  // one table: a row per bin with a rate per population, and the events as rows of their own where they fell, so a plotting script needs one file
  csv(){
    const labels = this.labels();
    const rows = [['t_ms', ...labels.map(l => l + '_hz'), 'event'].join(',')];
    const evs = this.events.slice();
    let bins = 0; for(const p of this.pops.values()) bins = Math.max(bins, p.from/BIN_MS + p.hz.length);
    let e = 0;
    for(let k = 0; k < bins; k++){
      const t = (k + 1)*BIN_MS;
      while(e < evs.length && evs[e].t < t){ rows.push([evs[e].t, ...labels.map(() => ''), q(describe(evs[e]))].join(',')); e++; }
      rows.push([t, ...labels.map(l => { const p = this.pops.get(l); const j = k - p.from/BIN_MS; return j >= 0 && j < p.hz.length ? p.hz[j] : ''; }), ''].join(','));
    }
    while(e < evs.length){ rows.push([evs[e].t, ...labels.map(() => ''), q(describe(evs[e]))].join(',')); e++; }
    return rows.join('\n') + '\n';
  }
}
const q = s => '"' + String(s).replace(/"/g, '""') + '"';
export function describe(e){
  if(e.kind === 'edit') return e.node + ' ' + e.key + ': ' + fmt(e.from) + ' to ' + fmt(e.to);
  return e.what + (e.detail ? ' ' + e.detail : '');
}
const fmt = v => typeof v === 'object' ? JSON.stringify(v) : String(v);
export function changedKeys(before, after){
  const out = [];
  for(const k of new Set([...Object.keys(before || {}), ...Object.keys(after || {})]))
    if(JSON.stringify(before ? before[k] : undefined) !== JSON.stringify(after ? after[k] : undefined)) out.push(k);
  return out;
}

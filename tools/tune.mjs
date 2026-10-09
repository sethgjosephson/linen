// Headless scene tuning: build a scenario by name through the same fake editor the battery uses, compute it at a resolution, run it on the reference engine in process, and print the rate per population.
// Overrides edit node params before the computation: OVER="scatter:sheet:density=72000;connect:*:wExc=0.5" (type:tag-or-name:key=value, * for every node of that type).
//   node tools/tune.mjs "guided tour" [seconds] [resolution 0..1]
import { NODE_DEFS, NEURON_TYPES, computeNode, engineConfig, engineTypes, setResolution, setMaxSyn, wireConnect, assembleConnect } from '../src/nodes.js';
import os from 'node:os';
setMaxSyn(Math.floor(os.totalmem()*0.5/13));   // the budget from the machine, not the browser's default
// node has no web Worker: the wiring runs in this process
import { inProcessWorker } from './harness.mjs';
inProcessWorker(wireConnect, assembleConnect);
import { initState } from '../src/rand.js';
import { BarSweep, encodeGains } from '../src/io.js';
import { ALL_SCENARIOS } from '../src/experiments.js';
import { splitTerms, applyOverrides } from './overrides.mjs';

const [name = 'guided tour', secs = '3', res = '1'] = process.argv.slice(2);
function fakeEditor(){
  const nodes = [];
  return { nodes, nextId:1, groups:[],
    addNode(type, x, y){
      const def = NODE_DEFS[type], params = {};
      def.params.forEach(p => params[p.k] = structuredClone(p.def));
      const n = { id:this.nextId++, type, x, y, params, inputs:new Array(def.inputs).fill(null) };
      nodes.push(n); return n;
    } };
}
const sc = ALL_SCENARIOS.find(s => s.name.toLowerCase().includes(name.toLowerCase()));
if(!sc){ console.error('no scenario ' + name); process.exit(1); }
const ed = fakeEditor(); sc.build(ed);
try {
  for(const a of applyOverrides(ed.nodes, splitTerms(process.env.OVER)))
    console.log('override', a.term, '->', a.nodes, 'node(s)');
} catch(e){ console.error('tune: ' + e.message); process.exit(1); }
setResolution(+res);
const outs = ed.nodes.filter(n => n.type === 'checkpoint');
const out = process.env.OUT ? outs[+process.env.OUT] : outs[0];
const t0 = performance.now();
const net = await computeNode(out, id => ed.nodes.find(n => n.id === id));
console.log(sc.name + ': ' + net.count.toLocaleString() + ' cells, ' + net.synCount.toLocaleString() + ' synapses at resolution ' + res + ', computed in ' + Math.round(performance.now() - t0) + ' ms');
const n = net.count;
// in-degree by population
const tagOf = i => (net.tags && net.tags[net.src[i]]) || NEURON_TYPES[net.ntype[i]].key;
const pops = new Map();
for(let i = 0; i < n; i++){ const t = tagOf(i); if(!pops.has(t)) pops.set(t, { n:0, deg:0, spikes:0, bin:0, bins:[] }); pops.get(t).n++; }
for(let i = 0; i < n; i++) for(let k = net.preStart[i]; k < net.preStart[i+1]; k++) pops.get(tagOf(net.post[k])).deg++;
const replies = []; globalThis.postMessage = d => replies.push(d); globalThis.onmessage = null;
await import('../src/simworker.js');
const send = d => globalThis.onmessage({ data:d });
send({ cmd:'init', count:n, ntype:net.ntype, bias:net.bias, ...engineConfig(net), pmask:net.pmask,
  preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, types:engineTypes(),
  inputs:(net.inputMaps || []).map(m => ({ chStart:m.chStart, chIdx:m.chIdx, chW:m.chW, amp:m.amp })),
  seed:3, ...initState(3, net.ntype, engineTypes()) });
const err = replies.find(r => r.cmd === 'error'); if(err){ console.error('engine: ' + err.message); process.exit(1); }
const ms = Math.round(+secs*1000); const t1 = performance.now();
const perSec = []; let secSpikes = 0;
// REGIONS="cued:-600,0,-200,200;far:600,0,-300,200": a rate per bin for the cells inside each sphere
const REG = (process.env.REGIONS || '').split(';').filter(Boolean).map(t => { const [nm, v] = t.split(':'); const [x, y, z, r] = v.split(',').map(Number);
  const idx = []; for(let i = 0; i < n; i++){ const dx = net.pos[i*3]-x, dy = net.pos[i*3+1]-y, dz = net.pos[i*3+2]-z; if(dx*dx+dy*dy+dz*dz <= r*r) idx.push(i); }
  return { nm, idx, spikes:0, bins:[] }; });
// the battery's wave measure: mean distance of the firing cells from the pacemaker center per frame; an expanding front is a run of busy frames whose mean distance grows (WAVE="-800,-500")
// ISI=1: interspike interval histogram per population (doublets are the < 5 ms bin)
const ISI = !!process.env.ISI; const lastT = new Int32Array(n).fill(-1); const isiBins = new Map();
const WAVE = process.env.WAVE ? process.env.WAVE.split(',').map(Number) : null; const traj = [];
// INPUT=1: drive every input node's signal (the bar sweep) as the page would, one frame per 10 ms
const frameTotals = [];
const INPUT = !!process.env.INPUT, bar = new BarSweep(), ist = (net.inputMaps || []).map(() => ({}));
for(let t = 0; t < ms; t += 2){
  if(INPUT && t % 10 === 0) (net.inputMaps || []).forEach((m, mi) => { send({ cmd:'inputFrame', mi, gains:encodeGains(bar.sample(t, m), m, ist[mi], t, 3) }); });
  replies.length = 0; send({ cmd:'tick', steps:2 });
  for(const r of replies) if(r.cmd === 'state'){
    if(process.env.SPECTRUM) frameTotals.push(r.spikes);
    let wn = 0, ws = 0;
    for(const g of REG) for(const i of g.idx) if(r.fired[i]) g.spikes += r.fired[i];
    if(ISI) for(let i = 0; i < n; i++) if(r.fired[i]){ const tg = tagOf(i); if(!isiBins.has(tg)) isiBins.set(tg, [0, 0, 0, 0, 0]);
      if(lastT[i] >= 0){ const d = t - lastT[i]; isiBins.get(tg)[d < 5 ? 0 : d < 10 ? 1 : d < 50 ? 2 : d < 200 ? 3 : 4]++; } lastT[i] = t; }
    for(let i = 0; i < n; i++) if(r.fired[i]){ const pp = pops.get(tagOf(i)); pp.spikes += r.fired[i]; pp.bin += r.fired[i]; secSpikes += r.fired[i];
      if(WAVE){ const dx = net.pos[i*3]-WAVE[0], dz = net.pos[i*3+2]-WAVE[1]; ws += Math.sqrt(dx*dx+dz*dz); wn++; } }
    if(WAVE) traj.push([wn, wn ? ws/wn : 0]);
  }
  const BIN = +(process.env.BIN || 1000);
  if((t + 2) % BIN === 0){ perSec.push((secSpikes/(n*BIN/1000)).toFixed(1)); secSpikes = 0; for(const pp of pops.values()){ pp.bins.push(pp.bin/(pp.n*BIN/1000)); pp.bin = 0; } for(const g of REG){ g.bins.push((g.spikes/(Math.max(1, g.idx.length)*BIN/1000)).toFixed(1)); g.spikes = 0; } }
}
if(process.env.SERIES) for(const [t, p] of pops) console.log('  series ' + t.padEnd(10) + p.bins.map(v => v.toFixed(1).padStart(6)).join(''));
console.log('ran ' + ms + ' ms in ' + Math.round(performance.now() - t1) + ' ms; mean Hz per second: ' + perSec.join(' '));
// SPECTRUM=1: the peak of the power spectrum of the whole-net rate per frame (frames are stepMs)
const frameRate = [];
if(process.env.SPECTRUM){
  // rebuild the per-frame rate from the populations' series (every frame, no smoothing)
}
if(WAVE){
  const speeds = []; let cur = null; const busy = Math.max(300, n*0.02);
  for(let i = 0; i < traj.length; i++){
    if(traj[i][0] > busy){ if(!cur) cur = []; cur.push(traj[i][1]); }
    // a front: busy for FRAMES frames or more while the mean distance grows by GROW um (8 and 400 suit a sheet over a millimeter across; a smaller sheet is crossed sooner: GROW=200 FRAMES=5)
    else if(cur){
      if(cur.length >= +(process.env.FRAMES || 8) && cur[cur.length-1] > cur[0] + +(process.env.GROW || 400)){
        // GP=1: the speed over the growing part of the run (its minimum distance to the maximum after it) rather than end to end over the whole busy episode, whose tail is the front dying back
        if(process.env.GP){
          let lo = 0; for(let k = 1; k < cur.length; k++) if(cur[k] < cur[lo]) lo = k;
          let hi = lo; for(let k = lo + 1; k < cur.length; k++) if(cur[k] > cur[hi]) hi = k;
          if(hi > lo) speeds.push((cur[hi]-cur[lo])/((hi-lo)*2));
        } else speeds.push((cur[cur.length-1]-cur[0])/(cur.length*2));
      }
      cur = null; }
  }
  if(process.env.TRAJ){ const k = traj.findIndex(x => x[0] === Math.max(...traj.map(y => y[0])));
    console.log('frames around the busiest (count, mean distance um): ' + traj.slice(Math.max(0, k - 12), k + 24).map(x => x[0] + '@' + Math.round(x[1])).join(' ')); }
  console.log('busiest frame ' + Math.max(...traj.map(x => x[0])) + ' cells (front threshold ' + busy + ')');
  console.log('wave fronts ' + speeds.length + ', speed ' + (speeds.length ? (speeds.reduce((x, y) => x + y, 0)/speeds.length/10).toFixed(1) : '-') + ' cm/s (band 1 to 10)');
}
if(ISI) for(const [tg, h] of isiBins){ const tot = h.reduce((x, y) => x + y, 0); console.log('  ISI ' + tg.padEnd(10) + ' <5 ms ' + (100*h[0]/tot).toFixed(1) + '%  5-10 ' + (100*h[1]/tot).toFixed(1) + '%  10-50 ' + (100*h[2]/tot).toFixed(1) + '%  50-200 ' + (100*h[3]/tot).toFixed(1) + '%  >200 ' + (100*h[4]/tot).toFixed(1) + '%  (' + tot + ' intervals, 2 ms frames)'); }
if(process.env.SPECTRUM && frameTotals.length > 64){
  // a plain DFT of the mean-removed frame totals after the first 200 ms; frames are 2 ms, so the axis runs to 250 Hz
  const skip = Math.min(100, frameTotals.length >> 2), x = frameTotals.slice(skip); const N = x.length, mean = x.reduce((a, b) => a + b, 0)/N;
  const dt = 0.002, top = []; 
  for(let k = 1; k < N/2; k++){ let re = 0, im = 0; for(let j = 0; j < N; j++){ const ph = 2*Math.PI*k*j/N; re += (x[j] - mean)*Math.cos(ph); im -= (x[j] - mean)*Math.sin(ph); }
    top.push([k/(N*dt), re*re + im*im]); }
  top.sort((a, b) => b[1] - a[1]);
  const total = top.reduce((a, b) => a + b[1], 0);
  console.log('spectrum peak ' + top[0][0].toFixed(1) + ' Hz (' + (100*top[0][1]/total).toFixed(1) + '% of power), next ' + top.slice(1, 4).map(p => p[0].toFixed(1)).join(', ') + ' Hz');
}
for(const g of REG) console.log('  region ' + g.nm + ' (' + g.idx.length + ' cells) Hz per bin: ' + g.bins.join(' '));
// POPBINS=1 adds each population's busiest bin, so a response the sweep dilutes still shows
for(const [t, p] of pops) console.log('  ' + t.padEnd(12) + String(p.n).padStart(8) + ' cells  in-degree ' + (p.deg/p.n).toFixed(1).padStart(7) + '  ' + (p.spikes/(p.n*ms/1000)).toFixed(2).padStart(7) + ' Hz' + (process.env.POPBINS ? '  peak bin ' + Math.max(...p.bins).toFixed(1) + ' Hz, median ' + p.bins.slice().sort((x, y) => x - y)[p.bins.length >> 1].toFixed(1) : ''));

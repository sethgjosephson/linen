// A bench for the fly's visual pathway, headless: regions of the MaleCNS release with their real wiring, computed through the same nodes the scene uses (a points file per region, a cell type per region, merge, a connections file, connect at probability zero, stimulus nodes for the drives, input nodes on the lamina fed by the bar sweep) and run on the reference engine in process under the 2007 membrane form.
// Two modes: FIELD compares dark (the lamina's tonic drive alone) with full-field light (a current withdrawn from L1 and L2, the way the scene's inverted input does over a whole field); SWEEP drives the same inverted inputs with the moving bar the page uses, through the same encoder, and compares dark with the sweep.
// A grid point takes seconds rather than a minute per point in the page.
//
//   node tools/flybench.mjs                  environment:
//     MODE=field|sweep      (field)          REGIONS=L1,L2,L3,medulla_on,medulla_off
//     TYPES=FG,...          per region       SECONDS=3
//     INPUTS=L1:-,L2:-      region:sign per light input node
//     COLUMNS='cx cy cz'    column-derived positions; SIDE=R one eye; COLS=36 PERIOD=3000 the bar
//     RECEPTORS='name tau sign [erev] transmitters;...'   a receptor node on the stream
//     SYN=2 EREVE=0 EREVI=-70   conductance synapses with these reversals
//     GRID=[{...}]          rows of { lam, on, off, t4, light, w, wi, tauI, noise } (wi: inhibitory weight factor)
import fs from 'node:fs';
import path from 'node:path';
import { NODE_DEFS, NEURON_TYPES, setFileReader, wireConnect, engineConfig, engineTypes, gradedArrays } from '../src/nodes.js';
import { initState } from '../src/rand.js';
import { BarSweep, encodeGains } from '../src/io.js';

const PROJ = process.env.PROJECT || path.resolve('project-fly');
setFileReader(async p => fs.readFileSync(path.join(PROJ, p)));
const MODE = process.env.MODE || 'field';
const REGIONS = (process.env.REGIONS || 'L1,L2,L3,medulla_on,medulla_off').split(',');
const SECONDS = +(process.env.SECONDS || 3);
const TYPE_KEYS = (process.env.TYPES || 'FG').split(',');
// COLUMNS=cx cy cz reads the column-derived positions the points file
// carries (the optic lobe's hex columns on a plane, 10 micrometers per column, a depth per type) so the bar is resolved by column rather than by soma; SIDE=R keeps one eye, since the two eyes' preferred directions mirror each other; COLS and PERIOD shape the bar (raster width, ms)
const COLUMNS = process.env.COLUMNS || '';
const SIDE = process.env.SIDE || '';
const COLS = +(process.env.COLS || 12);
const PERIOD = +(process.env.PERIOD || 1500);
// AXIS is the input node's sheet normal: 2 sweeps the bar along x (the first hex axis under COLUMNS), 0 along y (the second)
const AXIS = +(process.env.AXIS || 2);
const keyIndex = k => { const i = NEURON_TYPES.findIndex(t => t.key === k); if(i < 0) throw new Error('no type ' + k); return i; };

const t0 = performance.now();
const pfs = [];
for(const [k, r] of REGIONS.entries())
  pfs.push(await NODE_DEFS.pointfile.compute([], { file:'points/neurons.csv', columns:COLUMNS, id:'id', scale:COLUMNS ? 1 : 0.008, up:0, type:0,
    tag:'fly', tagColumn:'region', filter:'region=' + r + (SIDE ? '; side=' + SIDE : '') }, { id:10 + k }));
let merged = NODE_DEFS.gather.compute(pfs, { ports:Math.max(2, pfs.length) });
// SHUFFLE=1 is the control for anything read off position: the cells keep
// their identity and their wiring and swap positions by a seeded permutation, so a direction index that survives it was never retinotopic
const SHUFFLE = +(process.env.SHUFFLE || 0) === 1;
if(SHUFFLE){
  const n = merged.count, pos = merged.pos.slice(), perm = Array.from({ length:n }, (_, i) => i);
  let st = 12345; const rnd = () => { st ^= st << 13; st >>>= 0; st ^= st >> 17; st ^= st << 5; st >>>= 0; return st/4294967296; };
  for(let i = n - 1; i > 0; i--){ const j = Math.floor(rnd()*(i + 1)); const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
  for(let i = 0; i < n; i++){ merged.pos[i*3] = pos[perm[i]*3]; merged.pos[i*3+1] = pos[perm[i]*3+1]; merged.pos[i*3+2] = pos[perm[i]*3+2]; }
}
// a cell type node per region, on the built-in row of the type (row is the type index plus one, zero being custom)
REGIONS.forEach((r, k) => { merged = NODE_DEFS.celltype.compute([merged],
  { row:keyIndex(TYPE_KEYS[k] || TYPE_KEYS[TYPE_KEYS.length - 1]) + 1, tag:'fly.' + r, frac:1 }, { id:40 + k }); });
// RECEPTORS='ach 3 + acetylcholine;gabaa 8 - gaba;glucl 30 - glutamate histamine'
// puts a receptor node on the stream (lines separated by ;), so the connections file gives each of its synapses the channel its transmitter names (MODEL.md 2)
const RECEPTORS = process.env.RECEPTORS || '';
if(RECEPTORS) merged = NODE_DEFS.receptor.compute([merged], { table:RECEPTORS.split(';').join(String.fromCharCode(10)) }, { id:60 });
console.log(MODE + ': ' + merged.count.toLocaleString() + ' cells (' + REGIONS.join(', ') + '), types ' + TYPE_KEYS.join(',') +
  ', in ' + Math.round(performance.now() - t0) + ' ms');
const CN = { radius:1, sigma:1, prob:0, wExc:1, wInh:-1, wdist:0, wsigma:1, cluster:0, velocity:300, density:100, nuE:0, nuI:0, seed:1, table:'' };

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const send = d => globalThis.onmessage({ data:d });

async function wire(weight, wInh){
  const t1 = performance.now();
  const wired = await NODE_DEFS.connectionsfile.compute([merged], { file:'wiring/connections.csv', pre:'', post:'', count:'', signCol:'',
    weight, wInh, mode:0, compress:1, cap:20, signMode:1, velocity:300, rule:'' }, { id:20 });
  const net = await wireConnect(wired, CN, null);
  console.log('  wiring w=' + weight + ' wInh=' + wInh + ': ' + net.synCount.toLocaleString() + ' synapses in ' + Math.round(performance.now() - t1) + ' ms');
  return net;
}
// drives: [tag, current, mode] with mode 0 constant, 3 noise; spread 0.2
function drive(net, list, id){
  let n = net;
  for(const [tag, current, mode] of list){
    if(!current) continue;
    n = NODE_DEFS.stimulus.compute([n, null], { center:[200, 260, 300], radius:5000, current, tag, mode:mode || 0, period:100, width:10, t0:0,
      duration:0, spread:0.2, seedSpread:1 }, { id:id++ });
  }
  return n;
}
// the light inputs: the bar sweep, raw luminance, one input node per region in INPUTS (region:sign, the scene's default L1:-,L2:- being the lamina withdrawn by light; a + region is driven by light, the way an injection at the medulla would drive Mi1 and Tm3)
const INPUTS = (process.env.INPUTS || 'L1:-,L2:-').split(',').map(x => { const [r, sg] = x.split(':'); return [r, sg === '-' ? -1 : 1]; });
function lightInputs(net, amp, id, direction){
  const sig = NODE_DEFS.testsignal.compute([], { pattern:0, period:PERIOD, direction:direction || 0 });
  let n = net;
  for(const [r, sg] of INPUTS){
    if(!REGIONS.includes(r)) continue;
    n = NODE_DEFS.input.compute([n, null, sig], { tag:'fly.' + r, map:0, code:0, axis:AXIS, cols:COLS, rows:COLS, arbor:0.8, fanin:8, seed:1,
      transient:0, lagMs:0, jitter:0, amp:sg*amp }, { id:id++ });
  }
  return n;
}
function run(net, tauI, sweep){
  const n = net.count;
  // SYN=2 runs conductance synapses (weights in nS, reversals EREVE/EREVI on
  // the checkpoint, per channel on the receptor table)
  net.tauI = tauI; net.tauE = 3; net.syn = +(process.env.SYN || 1); net.psc = 0; net.refrac = 2; net.plast = 0;
  net.eRevE = +(process.env.EREVE || 0); net.eRevI = +(process.env.EREVI || -70);
  const maps = (net.inputMaps || []).map(m => ({ chStart:m.chStart, chIdx:m.chIdx, chW:m.chW, amp:m.amp }));
  replies.length = 0;
  send({ cmd:'init', count:n, ntype:net.ntype, bias:net.bias, ...engineConfig(net), pmask:net.pmask,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, types:engineTypes(), inputs:maps,
    seed:3, ...initState(3, net.ntype, engineTypes(), 1) });
  const err = replies.find(r => r.cmd === 'error'); if(err) throw new Error(err.message);
  const counts = new Float64Array(n), relSum = new Float64Array(n);
  // graded cells never spike: their readout is the mean release, from v
  const G = gradedArrays(net.ntype, engineTypes(), 1);
  if(G) send({ cmd:'sendV', on:true });
  const ms = SECONDS*1000; let ticks = 0;
  const bar = new BarSweep(), st = (net.inputMaps || []).map(() => ({}));
  for(let t = 0; t < ms; t += 10){
    if(sweep) (net.inputMaps || []).forEach((m, mi) => {
      send({ cmd:'inputFrame', mi, gains:encodeGains(bar.sample(t, m), m, st[mi], t, 3) }); });
    replies.length = 0; send({ cmd:'tick', steps:10 });
    const stt = replies.find(r => r.cmd === 'state');
    for(let i = 0; i < n; i++) counts[i] += stt.fired[i];
    if(G && stt.v) for(let i = 0; i < n; i++) if(G.grd[i]) relSum[i] += Math.min(1, Math.max(0, (stt.v[i] - G.thr[i])/G.slope[i]));
    ticks++;
  }
  const byRegion = {}, tags = net.tags || {};
  for(let i = 0; i < n; i++){ const r = String(tags[net.src[i]] || '').replace(/^fly[.]/, ''); (byRegion[r] = byRegion[r] || []).push(G && G.grd[i] ? { rel:relSum[i]/ticks } : { c:counts[i] }); }
  // a spiking region reads in Hz; a graded region reads as percent release (marked %)
  const out = {};
  for(const r of REGIONS){ const a = byRegion[r] || []; if(!a.length){ out[r] = null; continue; }
    if(a.every(x => x.rel !== undefined)) out[r] = (100*a.reduce((x, y) => x + y.rel, 0)/a.length).toFixed(1) + '%';
    else out[r] = +(a.reduce((x, y) => x + (y.c || 0), 0)/a.length/SECONDS).toFixed(2); }
  return out;
}

// the grid: lamina tonic, ON tonic, OFF tonic, T4/T5 tonic, light (withdrawal in FIELD, input amplitude in SWEEP), weight, tauI, noise
const GRID = JSON.parse(process.env.GRID || 'null') || (MODE === 'sweep'
  ? [ { lam:6, on:2, off:1, t4:1, light:70, w:4, tauI:20, noise:1.5 },
      { lam:6, on:2, off:1, t4:1, light:120, w:4, tauI:20, noise:1.5 } ]
  : [ { lam:6, on:2, off:1, t4:1, light:8, w:4, tauI:20, noise:1.5 },
      { lam:8, on:1.6, off:1, t4:1, light:10, w:6, tauI:20, noise:1.5 } ]);
let lastW = null, net = null;
const cols = REGIONS.map(r => r.padEnd(12)).join(' | ');
console.log('condition'.padEnd(66) + '| ' + cols + '   (dark/' + (MODE === 'sweep' ? 'sweep' : 'light') + ')');
for(const g of GRID){
  const wk = g.w + '/' + (g.wi === undefined ? 1 : g.wi);
  if(wk !== lastW){ net = await wire(g.w, g.wi === undefined ? 1 : g.wi); lastW = wk; }
  const base = [['fly.L1', g.lam], ['fly.L2', g.lam], ['fly.L3', g.lam], ['fly.medulla_on', g.on], ['fly.medulla_off', g.off],
    ['fly.T4', g.t4], ['fly.T5', g.t4],
    ...['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'].map(r => ['fly.' + r, g.t4]),
    // the arms of the motion detector: Mi9 and Tm9 are dark-active, Mi4 light-active
    ['fly.Mi9', g.off], ['fly.Tm9', g.off], ['fly.Mi4', g.on],
    ['fly.Mi1', g.on], ['fly.Tm3', g.on], ['fly.Tm1', g.off], ['fly.Tm2', g.off], ['fly.Tm4', g.off],
    ['', g.noise, 3]].filter(([tag]) => !tag || REGIONS.includes(tag.slice(4)));
  let dark, lit, back = null;
  if(MODE === 'sweep'){
    const fwd = lightInputs(drive(net, base, 100), g.light, 200, 0);
    dark = run(fwd, g.tauI, false);
    lit = run(fwd, g.tauI, true);
    // the bar the other way: a direction-selective region reads as (forward - backward) / (forward + backward)
    back = run(lightInputs(drive(net, base, 100), g.light, 200, 1), g.tauI, true);
    const dsi = REGIONS.map(r => { const a = parseFloat(lit[r]), b = parseFloat(back[r]); return r + ' ' + (a + b > 0 ? ((a - b)/(a + b)).toFixed(2) : '-') + ' (' + a + '/' + b + ')'; });
    console.log('    direction index, forward against backward: ' + dsi.join(', '));
  } else {
    dark = run(drive(net, base, 100), g.tauI, false);
    lit = run(drive(net, [...base, ['fly.L1', -g.light], ['fly.L2', -g.light]], 100), g.tauI, false);
  }
  const f = r => (dark[r] + '/' + lit[r]).padEnd(12);
  console.log(JSON.stringify(g).padEnd(66) + '| ' + REGIONS.map(f).join(' | '));
  // JSON=1: one machine-readable line per grid row, for a sweep driver
  if(process.env.JSON === '1') console.log('@@ ' + JSON.stringify({ grid:g, mode:MODE, regions:REGIONS, types:TYPE_KEYS, receptors:RECEPTORS, axis:AXIS, side:SIDE,
    columns:COLUMNS, shuffle:SHUFFLE, inputs:INPUTS, seconds:SECONDS, dark, lit, back:(typeof back !== 'undefined' ? back : null) }));
}

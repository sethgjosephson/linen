// What does the one millisecond step cost?
//
// The engine steps at 1 ms and quantizes delays to whole milliseconds (ENGINE.md).
// Brian, NEST and NEURON let a user shrink dt, and a reviewer will ask what the fixed step costs.
// This measures it: one network, computed once, run on the reference engine at 1 ms and on Brian 2 at a list of smaller steps, with rate, ISI CV and population synchrony reported per population and the 1 ms run's error against the finest step.
//
// Both sides run the identical network: the same init message goes to the engine and, through src/briancase.js, to tools/brian_ref.py, which is a second implementation of the engine's step.
// At dt = 1 the two agree spike for spike, and that run is kept in the table as a control: if it does not agree, nothing below it means anything.
// Delays stay whole milliseconds at every step, so what changes down the table is the integration step alone.
//
//   node tools/steperr.mjs ["balanced random net"] [seconds] [resolution] [dt list]
//   OVER="scatter:exc:count=800" node tools/steperr.mjs "balanced random net" 4 1
//
// Settle time (the transient from the initial state) is SETTLE ms, dropped from every statistic.
// Needs Brian 2 in .venv-analysis; says DID NOT RUN without it.
// Writes runs/steperr-<scene>-<stamp>.json.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { NODE_DEFS, NEURON_TYPES, computeNode, setResolution, wireConnect, assembleConnect } from '../src/nodes.js';
import { initMessage } from '../src/document.js';
import { caseFromInit } from '../src/briancase.js';
import { ALL_SCENARIOS } from '../src/experiments.js';
import { cv, meanRate, populationSynchrony, mean } from '../src/spikestats.js';
import { inProcessWorker } from './harness.mjs';
import { splitTerms, applyOverrides } from './overrides.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url)), ROOT = path.join(HERE, '..');
const PY = [path.join(ROOT, '.venv-analysis', 'Scripts', 'python.exe'),
            path.join(ROOT, '.venv-analysis', 'bin', 'python')].find(p => fs.existsSync(p));
if(!PY){ console.log('steperr: Brian 2 not found in .venv-analysis, so the measurement DID NOT RUN.'); process.exit(0); }

const [name = 'balanced random net', secs = '4', res = '1', dtArg = '1,0.5,0.25,0.1'] = process.argv.slice(2);
const ms = Math.round(+secs*1000);
const SETTLE = +(process.env.SETTLE || 500);       // the transient from the initial state
const SYNC_BIN = +(process.env.SYNCBIN || 3);      // ms, the window population synchrony counts in
const MIN_CV = 4;                                  // spikes a CV needs to mean anything (src/spikestats.js)
const dts = dtArg.split(',').map(Number).filter(d => d > 0 && d <= 1);

// ---- the control on one cell: an independent integrator at 0.1 ms
//
// Everything below rests on one reading of what a smaller step means, which is the reading in tools/brian_ref.py.
// This checks that reading against a second one: tools/izh2003_cells.py, forward Euler at 0.1 ms on the published 2003 equations, written for the cell class comparison and not for this.
// One classic RS cell held at three currents, no synapses, no noise, no refractory period and no floor, so the integration is the only thing that differs.
//
//   node tools/steperr.mjs cell [seconds]
if(name === 'cell'){
  const { classicRow } = await import('../src/form.js');
  const CELLS = [{ a:0.02, b:0.2, c:-65, d:8, v0:-70, bias:5 },
                 { a:0.02, b:0.2, c:-65, d:8, v0:-70, bias:10 },
                 { a:0.02, b:0.2, c:-65, d:8, v0:-70, bias:20 }];
  const cms = Math.round(+(process.argv[3] || 2)*1000), settle = 200;
  const types = CELLS.map(c => ({ f7:classicRow(c) }));
  const nc = CELLS.length;
  const { PROTOCOL } = await import('../src/protocol.js');
  const init = { cmd:'init', protocol:PROTOCOL, count:nc, ntype:Uint8Array.from(CELLS, (c, i) => i),
    bias:Float32Array.from(CELLS, c => c.bias), syn:0, psc:0, tauE:3, tauI:8, protocols:[], inputs:[],
    plast:0, refrac:0, vmin:0, seed:1,
    v0:Float32Array.from(CELLS, c => c.v0),
    u0:Float32Array.from(CELLS, (c, i) => types[i].f7.b*(c.v0 - types[i].f7.vr)),
    preStart:new Int32Array(nc + 1), post:new Int32Array(0), w:new Float32Array(0),
    delay:new Uint8Array(0), types };
  const rep = []; globalThis.postMessage = d => rep.push(d); globalThis.onmessage = null;
  await import('../src/simworker.js');
  const put = d => globalThis.onmessage({ data:d });
  put(init);
  const e0 = rep.find(r => r.cmd === 'error'); if(e0){ console.error('engine: ' + e0.message); process.exit(1); }
  const eng = CELLS.map(() => []);
  for(let t = 0; t < cms; t++){
    rep.length = 0; put({ cmd:'tick', steps:1 });
    const st = rep.find(r => r.cmd === 'state');
    for(let i = 0; i < nc; i++) if(st.fired[i]) eng[i].push(t);
  }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stepcell-'));
  const run = (script, payload) => {
    const a = path.join(dir, 'in.json'), b = path.join(dir, 'out.json');
    fs.writeFileSync(a, JSON.stringify(payload));
    execFileSync(PY, [path.join(HERE, script), a, b], { stdio:['ignore', 'ignore', 'pipe'], timeout:900000 });
    return JSON.parse(fs.readFileSync(b, 'utf8'));
  };
  const rateOf = ts => ts.filter(t => t >= settle).length/((cms - settle)/1000);
  const rows = [['engine 1 ms', eng.map(rateOf)]];
  for(const dt of [1, 0.5, 0.25, 0.1, 0.05]){
    const br = run('brian_ref.py', { ...caseFromInit(init, { ms:cms }), dt });
    rows.push(['brian ' + dt + ' ms', br.times.map(rateOf)]);
  }
  const ref = run('izh2003_cells.py', { ms:cms, cells:CELLS.map(c => ({ ...c, steps:[] })) });
  rows.push(['published equations 0.1 ms', ref.times.map(rateOf)]);
  console.log('one classic RS cell at a constant current, ' + (cms/1000) + ' s, the first '
    + settle + ' ms dropped');
  console.log('integration'.padEnd(28) + CELLS.map(c => ('I = ' + c.bias).padStart(12)).join(''));
  for(const [label, r] of rows)
    console.log(label.padEnd(28) + r.map(x => (x.toFixed(1) + ' Hz').padStart(12)).join(''));
  fs.rmSync(dir, { recursive:true, force:true });
  process.exit(0);
}

// ---- the summary: every run in runs/, one line per population
//
// A single run is one draw of a chaotic network, so the error is reported over the replicates on disk: the mean across seeds and the range.
//
//   node tools/steperr.mjs summary ["balanced"]
if(name === 'summary'){
  const want = (process.argv[3] || '').toLowerCase();
  const dir = path.join(ROOT, 'runs');
  const files = fs.readdirSync(dir).filter(f => /^steperr-.*\.json$/.test(f));
  const scenes = new Map();
  for(const f of files){
    const r = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
    if(want && !r.scene.toLowerCase().includes(want)) continue;
    // runs with different settings are different tables; the seed is the replicate and does not separate them
    const settings = (r.over || '').split(';').map(x => x.trim()).filter(x => x && !/:seed=/.test(x)).join('; ');
    const key = r.scene + ' at resolution ' + r.resolution + (settings ? ' with ' + settings : '');
    if(!scenes.has(key)) scenes.set(key, []);
    scenes.get(key).push(r);
  }
  const span = xs => { const v = xs.filter(Number.isFinite); if(!v.length) return '-';
    const m = v.reduce((a, b) => a + b, 0)/v.length;
    return m.toFixed(1) + (v.length > 1 ? ' (' + Math.min(...v).toFixed(1) + ' to ' + Math.max(...v).toFixed(1) + ')' : ''); };
  for(const [key, rs] of scenes){
    const r0 = rs[0];
    console.log('');
    console.log(key + ': ' + r0.cells.toLocaleString() + ' cells, ' + r0.synapses.toLocaleString()
      + ' synapses, ' + (r0.ms/1000) + ' s, ' + rs.length + ' seed' + (rs.length === 1 ? '' : 's')
      + ', against ' + r0.steps[r0.steps.length - 1].dt + ' ms on Brian 2 ' + r0.brian2);
    const bad = rs.filter(r => !/identical/.test(r.control));
    if(bad.length) console.log('  ' + bad.length + ' run(s) failed the 1 ms control and are still counted');
    const tags = r0.rows.map(x => x.tag);
    console.log('  population          error in rate %         error in CV %       error in synchrony %');
    for(const tag of tags){
      const pick = k => rs.map(r => (r.rows.find(x => x.tag === tag) || {})[k]);
      console.log('  ' + tag.padEnd(14) + span(pick('rateErr')).padStart(22)
        + span(pick('cvErr')).padStart(24) + span(pick('syncErr')).padStart(24));
    }
  }
  process.exit(0);
}

inProcessWorker(wireConnect, assembleConnect);
function fakeEditor(){
  const nodes = [];
  return { nodes, nextId:1, groups:[], addNode(type, x, y){
    const def = NODE_DEFS[type], params = {};
    def.params.forEach(p => params[p.k] = structuredClone(p.def));
    const n = { id:this.nextId++, type, x, y, params, inputs:new Array(def.inputs).fill(null) };
    nodes.push(n); return n; } };
}
const sc = ALL_SCENARIOS.find(s => s.name.toLowerCase().includes(name.toLowerCase()));
if(!sc){ console.error('no scene ' + name); process.exit(1); }
const ed = fakeEditor(); sc.build(ed);
// settings come from nodes, here as elsewhere: type:tag-or-name:key=value
try {
  for(const a of applyOverrides(ed.nodes, splitTerms(process.env.OVER)))
    console.log('override ' + a.term + ' -> ' + a.nodes + ' node(s)');
} catch(e){ console.error('steperr: ' + e.message); process.exit(1); }
setResolution(+res);
const out = ed.nodes.find(n => n.type === 'checkpoint');
const net = await computeNode(out, id => ed.nodes.find(n => n.id === id));
const n = net.count;
console.log(sc.name + ' at resolution ' + res + ': ' + n.toLocaleString() + ' cells, '
  + net.synCount.toLocaleString() + ' synapses, ' + (ms/1000) + ' s, settle ' + SETTLE + ' ms');

const init = { ...initMessage(net), inputs:[] };
if(init.plast) console.log('note: plasticity is on in this scene and a step below 1 ms refuses it; '
  + 'turn it off on the checkpoint to measure the step');

// ---- the reference engine, one millisecond ticks so every spike has its own step
const replies = []; globalThis.postMessage = d => replies.push(d); globalThis.onmessage = null;
await import('../src/simworker.js');
const send = d => globalThis.onmessage({ data:d });
send({ ...init, w:init.w.slice() });
const err = replies.find(r => r.cmd === 'error');
if(err){ console.error('engine: ' + err.message); process.exit(1); }
const engineTimes = Array.from({ length:n }, () => []);
const t0 = performance.now();
for(let t = 0; t < ms; t++){
  replies.length = 0; send({ cmd:'tick', steps:1 });
  const s = replies.find(r => r.cmd === 'state');
  for(let i = 0; i < n; i++) if(s.fired[i]) engineTimes[i].push(t);
}
const engineWall = (performance.now() - t0)/1000;
console.log('the engine ran ' + ms + ' ms in ' + engineWall.toFixed(1) + ' s');

// ---- Brian 2 at each step
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'steperr-'));
const runs = [];
for(const dt of dts){
  const file = caseFromInit(init, { ms });
  file.dt = dt;
  const inFile = path.join(tmp, 'case-' + dt + '.json'), outFile = path.join(tmp, 'out-' + dt + '.json');
  const json = JSON.stringify(file);
  fs.writeFileSync(inFile, json);
  process.stdout.write('Brian 2 at ' + dt + ' ms (case ' + (json.length/1e6).toFixed(1) + ' MB) ... ');
  const w0 = performance.now();
  try {
    execFileSync(PY, [path.join(HERE, 'brian_ref.py'), inFile, outFile],
      { stdio:['ignore', 'ignore', 'pipe'], timeout:7200000, encoding:'utf8' });
  } catch(e){
    console.log('failed');
    console.error(String(e.stderr || e.message).split(/\r?\n/).filter(l => /Error|error:/.test(l)).slice(-4).join('\n'));
    process.exit(1);
  }
  const br = JSON.parse(fs.readFileSync(outFile, 'utf8'));
  console.log(Math.round((performance.now() - w0)/1000) + ' s');
  fs.rmSync(inFile);
  runs.push({ dt, times:br.times, wallS:br.wallS, brian2:br.versions.brian2 });
}

// ---- the control: Brian at 1 ms is the engine, spike for spike
const control = runs.find(r => r.dt === 1);
let controlLine = 'not run (1 ms was not in the list)';
if(control){
  let differ = 0;
  for(let i = 0; i < n; i++){
    const a = engineTimes[i], b = control.times[i] || [];
    if(a.length !== b.length || !a.every((t, k) => t === b[k])) differ++;
  }
  controlLine = differ === 0 ? 'every cell identical to the engine' : differ + ' of ' + n + ' cells differ from the engine';
  if(differ) console.log('CONTROL FAILED: ' + controlLine + '. The rest of this table is not a step measurement.');
}

// ---- statistics per population
const tagOf = i => (net.tags && net.tags[net.src[i]]) || NEURON_TYPES[net.ntype[i]].key;
const pops = new Map();
for(let i = 0; i < n; i++){
  const t = tagOf(i);
  if(!pops.has(t)) pops.set(t, []);
  pops.get(t).push(i);
}
// after the settle time, and shifted so every statistic is over the same window
const windowOf = times => {
  const w = [];
  for(const t of times){
    const keep = [];
    for(const x of t) if(x >= SETTLE) keep.push(x);
    w.push(keep);
  }
  return w;
};
function stats(times){
  const per = new Map();
  for(const [tag, idx] of pops){
    const trains = windowOf(idx.map(i => times[i] || []));
    const rates = trains.map(t => meanRate(t, SETTLE, ms));
    const cvs = trains.filter(t => t.length >= MIN_CV).map(cv);
    per.set(tag, { cells:idx.length, spikes:trains.reduce((a, t) => a + t.length, 0),
      rate:mean(rates), cv:cvs.length ? mean(cvs) : NaN, cvCells:cvs.length,
      sync:populationSynchrony(trains, SYNC_BIN, SETTLE, ms) });
  }
  return per;
}
const engineStats = stats(engineTimes);
const byDt = runs.map(r => ({ dt:r.dt, wallS:r.wallS, per:stats(r.times) }));
const finest = byDt[byDt.length - 1];

const pad = (s, w) => String(s).padStart(w);
const fmt = (v, d = 2) => Number.isFinite(v) ? v.toFixed(d) : '-';
console.log('\n' + sc.name + ': the engine at 1 ms against Brian 2 at ' + finest.dt + ' ms');
console.log('population        cells      rate 1ms   rate ' + finest.dt + 'ms    err        CV 1ms    CV ' + finest.dt + 'ms     err       sync 1ms  sync ' + finest.dt + 'ms    err');
const rows = [];
for(const [tag, e] of engineStats){
  const f = finest.per.get(tag);
  const rel = (a, b) => Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? (a - b)/Math.abs(b)*100 : NaN;
  const row = { tag, cells:e.cells, rate:e.rate, rateFine:f.rate, rateErr:rel(e.rate, f.rate),
    cv:e.cv, cvFine:f.cv, cvErr:rel(e.cv, f.cv), sync:e.sync, syncFine:f.sync, syncErr:rel(e.sync, f.sync) };
  rows.push(row);
  console.log(tag.padEnd(16) + pad(e.cells, 7) + pad(fmt(e.rate), 11) + ' Hz' + pad(fmt(f.rate), 9)
    + ' Hz' + pad(fmt(row.rateErr, 1), 8) + '%' + pad(fmt(e.cv), 10) + pad(fmt(f.cv), 10)
    + pad(fmt(row.cvErr, 1), 9) + '%' + pad(fmt(e.sync), 10) + pad(fmt(f.sync), 10)
    + pad(fmt(row.syncErr, 1), 9) + '%');
}
console.log('\nevery step, mean over populations:');
console.log('step        rate Hz      CV      synchrony    wall s');
const netRate = p => mean([...p.values()].map(x => x.rate).filter(Number.isFinite));
const netCv = p => mean([...p.values()].map(x => x.cv).filter(Number.isFinite));
const netSync = p => mean([...p.values()].map(x => x.sync).filter(Number.isFinite));
console.log('engine 1 ms' + pad(fmt(netRate(engineStats)), 9) + pad(fmt(netCv(engineStats)), 9)
  + pad(fmt(netSync(engineStats)), 10) + pad(engineWall.toFixed(1), 11));
for(const r of byDt)
  console.log(('brian ' + r.dt + ' ms').padEnd(11) + pad(fmt(netRate(r.per)), 9) + pad(fmt(netCv(r.per)), 9)
    + pad(fmt(netSync(r.per)), 10) + pad(r.wallS.toFixed(1), 11));
console.log('\ncontrol at 1 ms: ' + controlLine);

// the local clock, the one the commits use
const d = new Date(), p2 = x => String(x).padStart(2, '0');
const stamp = '' + d.getFullYear() + p2(d.getMonth() + 1) + p2(d.getDate())
  + 'T' + p2(d.getHours()) + p2(d.getMinutes());
const slug = sc.name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
const dir = path.join(ROOT, 'runs');
fs.mkdirSync(dir, { recursive:true });
const file = path.join(dir, 'steperr-' + slug + '-' + stamp + '.json');
fs.writeFileSync(file, JSON.stringify({
  scene:sc.name, resolution:+res, cells:n, synapses:net.synCount, ms, settleMs:SETTLE,
  syncBinMs:SYNC_BIN, over:process.env.OVER || '', control:controlLine,
  brian2:runs[0].brian2, node:process.version, engineWallS:+engineWall.toFixed(1),
  steps:[{ dt:1, engine:'reference (src/simworker.js)', per:Object.fromEntries(engineStats) },
    ...byDt.map(r => ({ dt:r.dt, engine:'brian2', wallS:r.wallS, per:Object.fromEntries(r.per) }))],
  rows,
}, null, 1));
console.log('wrote ' + path.relative(ROOT, file));
fs.rmSync(tmp, { recursive:true, force:true });

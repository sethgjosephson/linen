// A direction-selectivity sweep on the fly bench: the motion stage (the medulla arms, Mi9 slow or fast, T4 by subtype) under a column-resolved bar over one eye, both ways, with a receptor table, over a grid of the settings that decide it: the glutamate-gated chloride time constant, the Mi9 row, the drives on the arms, the connections file's weight, the bar axis.
// Every point also runs the shuffled-position control (SHUFFLE=1 in the bench), so an index has to vanish when retinotopy is broken before it counts.
// Each point is one child run of tools/flybench.mjs with the settings in its environment; the settings live on the nodes the bench builds.
//
//   node tools/flysweep.mjs [out.json]      (about a minute per point)
//
// Writes project-fly/runs/dsweep-<stamp>.json (every point, rates per region dark, forward and backward, control included) and prints the points sorted by the T4a index against its control.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const OUT = process.argv[2] || path.join('project-fly', 'runs', 'dsweep-' + new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '') + '.json');
// COND=1 runs the grid on conductance synapses: weights in nS, ACh toward
// 0 mV, the chloride channels toward -70, the modulatory channel toward 0
const COND = process.env.COND === '1';
fs.mkdirSync(path.dirname(OUT), { recursive:true });
const REGIONS = 'medulla,Mi1,Tm3,medulla_on,Mi9,Mi4,Tm9,T4a,T4b,T4c,T4d';
const BASE = { MODE:'sweep', SIDE:'R', COLUMNS:'cx cy cz', COLS:'36', PERIOD:'3000', REGIONS, INPUTS:'Mi1:+,Tm3:+,medulla_on:+,Mi9:-,Mi4:+', SECONDS:COND ? '10' : '6', JSON:'1',
  ...(COND ? { SYN:'2', EREVE:'0', EREVI:'-70' } : {}) };
const points = [];
// weights are currents in current mode and nanosiemens against a 1 nS leak in conductance mode, where 0.3 nS already saturates the stage
for(const glucl of (COND ? [30, 100] : [10, 30, 100]))
  for(const mi9 of ['FG', 'FS7'])
    for(const off of [2, 4])
      for(const w of (COND ? [0.05, 0.1, 0.2] : [6, 12]))
        for(const axis of [2, 0])
          points.push({ glucl, mi9, off, w, axis });
console.log(points.length + ' points, each with its shuffled control; writing ' + OUT);

const results = [];
const run = (p, shuffle) => {
  const env = { ...process.env, ...BASE, AXIS:String(p.axis), SHUFFLE: shuffle ? '1' : '0',
    TYPES:['FG', 'FG', 'FG', 'FG', p.mi9, 'FG', 'FG', 'FL', 'FL', 'FL', 'FL'].join(','),
    RECEPTORS:COND ? 'ach 3 + 0 ach unclear;gabaa 8 - -70 gaba;glucl ' + p.glucl + ' - -70 glut;hisc 10 - -70 hist;mod 50 + 0 da oa 5ht'
      : 'ach 3 + ach unclear;gabaa 8 - gaba;glucl ' + p.glucl + ' - glut;hisc 10 - hist;mod 50 + da oa 5ht',
    GRID:JSON.stringify([{ lam:0, on:1, off:p.off, t4:2, light:30, w:p.w, wi:1, tauI:8, noise:0.5 }]) };
  const r = spawnSync(process.execPath, ['tools/flybench.mjs'], { env, encoding:'utf-8', maxBuffer:64*1024*1024 });
  const line = (r.stdout || '').split('\n').find(l => l.startsWith('@@ '));
  if(!line) return { error:(r.stderr || r.stdout || '').slice(-400) };
  return JSON.parse(line.slice(3));
};
const idx = (res, r) => { if(!res || !res.lit || !res.back) return null; const a = parseFloat(res.lit[r]), b = parseFloat(res.back[r]); return a + b > 0 ? (a - b)/(a + b) : 0; };
const t0 = Date.now();
points.forEach((p, k) => {
  const real = run(p, false), ctrl = run(p, true);
  const row = { point:p, real, control:ctrl, index:{}, controlIndex:{} };
  for(const r of ['T4a', 'T4b', 'T4c', 'T4d']){ row.index[r] = idx(real, r); row.controlIndex[r] = idx(ctrl, r); }
  results.push(row);
  fs.writeFileSync(OUT, JSON.stringify({ base:BASE, started:new Date(t0).toISOString(), points:results }, null, 1));
  const f = x => x === null || x === undefined ? '  -  ' : (x >= 0 ? '+' : '') + x.toFixed(2);
  console.log(`${String(k + 1).padStart(2)}/${points.length} ${JSON.stringify(p).padEnd(52)} T4a ${f(row.index.T4a)} (ctrl ${f(row.controlIndex.T4a)})  T4b ${f(row.index.T4b)}  T4c ${f(row.index.T4c)}  T4d ${f(row.index.T4d)}  ${real.error ? 'ERROR ' + real.error.slice(0, 80) : ''}  ${Math.round((Date.now() - t0)/60000)} min`);
});
// the summary: the points where a subtype's index stands clear of its control
const clear = results.filter(r => !r.real.error).map(r => {
  let best = 0, bestR = '';
  for(const s of ['T4a', 'T4b', 'T4c', 'T4d']){ const d = Math.abs((r.index[s] || 0) - (r.controlIndex[s] || 0)); if(d > best){ best = d; bestR = s; } }
  return { p:r.point, sub:bestR, gap:best, index:r.index[bestR], control:r.controlIndex[bestR] };
}).sort((a, b) => b.gap - a.gap);
console.log('\nlargest gaps between a subtype index and its shuffled control:');
for(const c of clear.slice(0, 10)) console.log('  ' + JSON.stringify(c.p) + '  ' + c.sub + ' ' + (c.index || 0).toFixed(3) + ' vs control ' + (c.control || 0).toFixed(3));
console.log('wrote ' + OUT);

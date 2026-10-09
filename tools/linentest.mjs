// Does tools/linen.mjs run a scene and write a folder that says what it did?
//
// The headless runner is an entry point of its own: nothing else in the suite would notice if it stopped computing, stopped writing a file, or quietly ignored a setting.
// This runs it on a small scene twice, once by scene name and once by scene file, and checks the folder against itself.
//
//   node tools/linentest.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ok, report } from './harness.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url)), ROOT = path.join(HERE, '..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'linentest-'));
const cli = (...args) => spawnSync(process.execPath, [path.join(HERE, 'linen.mjs'), ...args],
  { cwd:ROOT, encoding:'utf8' });

const SCENE = 'balanced random net';
const dirA = path.join(tmp, 'a');
const a = cli('run', SCENE, '--seconds', '0.4', '--res', '0.02', '--cells', 'all',
  '--bin', '200', '--out', dirA, '--quiet');
ok('a run by scene name succeeds', a.status === 0, (a.stderr || '').trim().split('\n').pop());

const read = d => ({
  run:JSON.parse(fs.readFileSync(path.join(d, 'run.json'), 'utf8')),
  rates:fs.readFileSync(path.join(d, 'rates.csv'), 'utf8').trim().split('\n'),
  cells:fs.readFileSync(path.join(d, 'cells.csv'), 'utf8').trim().split('\n'),
  spikes:fs.readFileSync(path.join(d, 'spikes.txt'), 'utf8').split('\n').slice(0, -1),
  settings:JSON.parse(fs.readFileSync(path.join(d, 'settings.json'), 'utf8')),
});
const A = read(dirA);
ok('it wrote a cell for every line of spike times', A.spikes.length === A.cells.length - 1,
  A.spikes.length + ' lines against ' + (A.cells.length - 1) + ' cells');
ok('every cell is written when asked for all of them', A.run.cellsWritten === A.run.cells,
  A.run.cellsWritten + ' of ' + A.run.cells);
const written = A.spikes.reduce((n, l) => n + (l.trim() ? l.trim().split(/\s+/).length : 0), 0);
ok('the spike times are the spikes the run counted', written === A.run.spikes,
  written + ' in the file against ' + A.run.spikes + ' counted');
ok('the network fired at all', A.run.spikes > 0, A.run.spikes + ' spikes');
const heads = A.rates[0].split(',');
ok('a rate column for every population', heads.length === A.run.populations.length + 1,
  A.rates[0]);
ok('a rate row for every bin', A.rates.length - 1 === Math.round(A.run.seconds*1000/A.run.binMs),
  (A.rates.length - 1) + ' rows');
ok('the settings of every node are written', A.settings.nodes.length > 0
  && A.settings.nodes.every(n => n.params && typeof n.type === 'string'), A.settings.nodes.length + ' nodes');
ok('the run says which commit it ran', /^[0-9a-f]{7,}/.test(A.run.commit || ''), A.run.commit);

// The same graph again, this time from a file, which is the path a scene saved by the app takes. settings.json is the graph the run computed, so a scene file made from it must compute to the same network and fire the same.
const sceneFile = path.join(tmp, 'scene.json');
fs.writeFileSync(sceneFile, JSON.stringify({ v:1, format:10, nextId:10000,
  scenarioName:'', nodes:A.settings.nodes.map(n => ({ id:n.id, type:n.type, x:0, y:0, on:n.on,
    name:n.name, params:n.params, inputs:n.inputs.map(id => id === null ? null : { id }) })) }));
const dirB = path.join(tmp, 'b');
const b = cli('run', sceneFile, '--seconds', '0.4', '--res', '0.02', '--cells', 'all',
  '--bin', '200', '--out', dirB, '--quiet');
ok('a run from a scene file succeeds', b.status === 0, (b.stderr || '').trim().split('\n').pop());
if(b.status === 0){
  const B = read(dirB);
  ok('the scene file computes the same network', B.run.cells === A.run.cells && B.run.synapses === A.run.synapses,
    B.run.cells + ' cells and ' + B.run.synapses + ' synapses against ' + A.run.cells + ' and ' + A.run.synapses);
  ok('and fires the same', B.run.spikes === A.run.spikes,
    B.run.spikes + ' spikes against ' + A.run.spikes);
}

// A setting names a node, and a setting no node has is an error rather than a run that quietly ignored it.
const good = cli('run', SCENE, '--seconds', '0.2', '--res', '0.02',
  '--over', 'connect:*:seed=7', '--out', path.join(tmp, 'c'), '--quiet');
ok('a setting on a node is applied', good.status === 0
  && JSON.parse(fs.readFileSync(path.join(tmp, 'c', 'run.json'), 'utf8')).seed === 7,
  (good.stderr || '').trim().split('\n').pop());
const bad = cli('run', SCENE, '--seconds', '0.2', '--res', '0.02',
  '--over', 'connect:*:nosuchsetting=1', '--out', path.join(tmp, 'd'), '--quiet');
ok('a setting no node has is refused', bad.status !== 0 && /no setting called/.test(bad.stderr || ''),
  (bad.stderr || '').trim().split('\n').pop());
const nothing = cli('run', 'a scene by no such name', '--quiet');
ok('an unknown scene is refused, with the list', nothing.status !== 0
  && /no scene file and no scene called/.test(nothing.stderr || ''), String(nothing.status));

// A graded population never spikes, so a spike count reads it as silent whatever it does; the run reads it as release, the way a probe does.
// An input node fed by the test signal is driven on simulated time, so a bar on a population moves its rate, and an amplitude of zero on the same node does not.
const gnode = (id, type, params, inputs = []) => ({ id, type, x:0, y:0, on:true, name:type + ' ' + id, params, inputs:inputs.map(k => k === null ? null : { id:k }) });
const fgRow = 1 + 8, flRow = 1 + 7;        // the cell type node's row is the type table index plus one; FG and FL are rows 8 and 7
const gradedScene = path.join(tmp, 'graded.json');
fs.writeFileSync(gradedScene, JSON.stringify({ v:1, format:13, nextId:100, scenarioName:'graded and driven', nodes:[
  gnode(1, 'sphere', { center:[0, 0, 0], radius:150 }),
  gnode(2, 'scatter', { fill:0, count:120, tag:'g' }, [1]),
  gnode(3, 'scatter', { fill:0, count:120, tag:'s', seed:2 }, [1]),
  gnode(4, 'gather', { ports:2 }, [2, 3]),
  gnode(5, 'celltype', { row:fgRow, tag:'g', frac:1 }, [4]),
  gnode(6, 'celltype', { row:flRow, tag:'s', frac:1 }, [5]),
  gnode(7, 'connect', { prob:0 }, [6]),
  gnode(8, 'stimulus', { tag:'g', current:4, mode:0, radius:5000 }, [7, null]),
  gnode(10, 'testsignal', { pattern:0, period:300, direction:0 }),
  gnode(9, 'input', { tag:'s', amp:60, jitter:0, transient:0, cols:8, rows:8 }, [8, null, 10]),
  gnode(11, 'checkpoint', { plast:0 }, [9]),
] }));
const dirG = path.join(tmp, 'g'), dirG0 = path.join(tmp, 'g0');
const g = cli('run', gradedScene, '--seconds', '0.6', '--bin', '200', '--out', dirG, '--quiet');
ok('a scene with a graded population and a test signal runs', g.status === 0, (g.stderr || '').trim().split('\n').pop());
const g0 = cli('run', gradedScene, '--seconds', '0.6', '--bin', '200', '--out', dirG0, '--quiet', '--over', 'input:*:amp=0');
ok('and with the input at zero amplitude', g0.status === 0, (g0.stderr || '').trim().split('\n').pop());
if(g.status === 0 && g0.status === 0){
  const G = JSON.parse(fs.readFileSync(path.join(dirG, 'run.json'), 'utf8'));
  const G0 = JSON.parse(fs.readFileSync(path.join(dirG0, 'run.json'), 'utf8'));
  const pg = G.populations.find(p => p.tag === 'g'), ps = G.populations.find(p => p.tag === 's');
  ok('the graded population is marked graded and fires no spikes', pg && pg.graded === true && pg.meanHz === 0, JSON.stringify(pg));
  ok('its release under a drive above its release threshold is read', pg && pg.releasePct > 1, JSON.stringify(pg));
  ok('a spiking population carries no release', ps && ps.graded === undefined && ps.releasePct === undefined, JSON.stringify(ps));
  const rel = fs.existsSync(path.join(dirG, 'release.csv')) ? fs.readFileSync(path.join(dirG, 'release.csv'), 'utf8').trim().split('\n') : [];
  ok('release.csv has a column for the graded population and a row per bin', rel.length === 4 && rel[0].split(',').length === 2,
    rel.join(' / '));
  ok('the test signal input is driven and said to be', G.inputsDriven === true && G.drivenInputs.length === 1, JSON.stringify(G.drivenInputs));
  const s0 = G0.populations.find(p => p.tag === 's');
  ok('the bar moves the rate of the population it drives', ps && s0 && ps.meanHz > s0.meanHz + 1,
    (ps && ps.meanHz) + ' Hz driven against ' + (s0 && s0.meanHz) + ' Hz at zero amplitude');
}

fs.rmSync(tmp, { recursive:true, force:true });
report('linen');

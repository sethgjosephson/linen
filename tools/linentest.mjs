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

fs.rmSync(tmp, { recursive:true, force:true });
report('linen');

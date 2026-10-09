// An engine of one's own, end to end: tools/engine_template.py started on a free port, a Tab menu scene run on it by tools/linen.mjs --engine remote, and a term it lacks refused before anything runs.
// Then the conformance check, tools/enginecheck.mjs, against host/host.mjs (the reference over the socket, which passes every check) and against the template (which passes the protocol checks and differs on the measures, as a different model should).
// Needs Python with NumPy and the websockets package; without them it says the check DID NOT RUN.
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ok, report } from './harness.mjs';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const PY = process.platform === 'win32' ? 'python' : 'python3';
if(spawnSync(PY, ['-c', 'import numpy, websockets'], { encoding:'utf8' }).status !== 0){
  console.log('remotetest: Python with NumPy and websockets not found, so the check DID NOT RUN.');
  process.exit(0);
}
const PORT = 8893, URL = 'ws://localhost:' + PORT;
const tpl = spawn(PY, [path.join(ROOT, 'tools', 'engine_template.py'), String(PORT)], { stdio:['ignore', 'pipe', 'pipe'] });
let said = '';
tpl.stdout.on('data', d => { said += d; });
tpl.stderr.on('data', d => { said += d; });
for(let i = 0; i < 100 && !/listening/.test(said); i++) await new Promise(r => setTimeout(r, 100));
ok('the template engine starts', /listening on ws:\/\/localhost:8893/.test(said), said.trim());

const out = fs.mkdtempSync(path.join(os.tmpdir(), 'linen-remote-'));
const linen = args => spawnSync(process.execPath, [path.join(ROOT, 'tools', 'linen.mjs'), 'run', ...args], { cwd:ROOT, encoding:'utf8' });
try {
  const r = linen(['balanced random net', '--engine', 'remote', '--host', URL, '--seconds', '0.5', '--res', '0.1', '--quiet', '--out', out]);
  ok('linen runs a scene on the template engine', r.status === 0, (r.stderr || r.stdout).trim().split('\n').slice(-3).join(' | '));
  const run = fs.existsSync(path.join(out, 'run.json')) ? JSON.parse(fs.readFileSync(path.join(out, 'run.json'), 'utf8')) : {};
  ok('run.json names the remote engine', /remote \(lif-template at ws:\/\/localhost:8893\)/.test(run.engine || ''), run.engine);
  ok('and the folder holds rates and spikes', fs.existsSync(path.join(out, 'rates.csv')) && fs.existsSync(path.join(out, 'spikes.txt')));
  ok('the network fired', run.spikes > 0, 'spikes ' + run.spikes);
  const p = linen(['balanced random net', '--engine', 'remote', '--host', URL, '--seconds', '0.1', '--res', '0.1', '--quiet', '--over', 'checkpoint:*:plast=1', '--out', out + '-p']);
  ok('a term the engine does not name is refused before the run', p.status !== 0 && /lif-template engine does not implement plasticity/.test(p.stderr), p.stderr.trim());
  const q = linen(['balanced random net', '--engine', 'remote', '--host', 'ws://localhost:8894', '--seconds', '0.1', '--res', '0.1', '--quiet', '--out', out + '-q']);
  ok('an address with nothing listening is an error naming it', q.status !== 0 && /ws:\/\/localhost:8894/.test(q.stderr), q.stderr.trim());

  // the conformance check
  const check = url => spawnSync(process.execPath, [path.join(ROOT, 'tools', 'enginecheck.mjs'), url, '--seconds', '2'], { cwd:ROOT, encoding:'utf8' });
  const host = spawn(process.execPath, [path.join(ROOT, 'host', 'host.mjs'), '8895'], { stdio:['ignore', 'pipe', 'pipe'] });
  let hs = '';
  host.stdout.on('data', d => { hs += d; });
  for(let i = 0; i < 100 && !/listening/.test(hs); i++) await new Promise(r => setTimeout(r, 100));
  try {
    const h = check('ws://localhost:8895');
    ok('enginecheck passes every check against host/host.mjs', h.status === 0 && /every check passed/.test(h.stdout), h.stdout.trim().split('\n').slice(-4).join(' | '));
    ok('and names the reference engine with the host\'s run controller', /engine: reference at ws:\/\/localhost:8895/.test(h.stdout) && /optional: .*host/.test(h.stdout));
  } finally { host.kill(); }
  const t = check(URL);
  const lines = t.stdout.split('\n');
  ok('against the template the protocol checks pass', !lines.some(l => /^FAIL/.test(l)) && /ok    state replies carry fired/.test(t.stdout), lines.filter(l => /^FAIL/.test(l)).join(' | '));
  ok('and the rates are reported different, as expected of another model', t.status !== 0 && /rate Hz .*DIFFERENT/.test(t.stdout) && /^expected: /m.test(t.stdout), lines.slice(-4).join(' | '));
} finally {
  tpl.kill();
  fs.rmSync(out, { recursive:true, force:true });
}
report('remotetest');

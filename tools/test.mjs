// One command that runs every test and fails if any of them do.
//
//   node tools/test.mjs           the pure-JS suites, then the cuda suites, servetest.py, linentest.mjs, remotetest.mjs, brianref.mjs, published.mjs and exportcheck.mjs
//   node tools/test.mjs --quick   the pure-JS suites only; no GPU, no server
//
// The battery (validate.html) needs a browser and is not run here; it stays a manual gate.
// This is the automatable half, and --quick is what the pre-commit hook runs, because a GPU spawn and a 58 s server spawn on every commit would be turned off within a day.
//
// A suite counts as passed if it exits 0, so a suite need not use tools/harness.mjs to be counted.

import { readdirSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const quick = process.argv.includes('--quick');
const node = process.execPath;

// Every *.test.mjs under src/ and host/, discovered rather than listed so a new suite is covered the moment it is written.
// These are pure JS: no GPU, no built binary, no server, so they are the quick set and the pre-commit set.
const jsSuites = [];
for(const dir of ['src', 'host']){
  const d = join(ROOT, dir);
  if(!existsSync(d)) continue;
  for(const f of readdirSync(d).sort())
    if(f.endsWith('.test.mjs')) jsSuites.push(join(dir, f));
}

const results = [];
let failed = 0;
const t0 = Date.now();

function run(label, cmd, args){
  const start = Date.now();
  const r = spawnSync(cmd, args, { cwd: ROOT, encoding: 'utf8' });
  const ms = Date.now() - start;
  const bad = r.status !== 0;
  if(bad) failed++;
  results.push({ label, ok: !bad, out: (r.stdout || '') + (r.stderr || '') });
  process.stdout.write(`${bad ? 'FAIL' : 'ok  '} ${label.padEnd(30)} ${String(ms).padStart(6)} ms\n`);
  // On failure show the suite's own output, so the runner alone is enough to diagnose from rather than just a verdict.
  if(bad)
    for(const l of results[results.length - 1].out.trim().split('\n').slice(-12))
      process.stdout.write('       ' + l + '\n');
}

for(const s of jsSuites) run(s, node, [join(ROOT, s)]);

if(!quick){
  // The cuda suites spawn the CUDA child (a built engine.exe and a GPU); the server test spawns a server for 58 s.
  // Full run only.
  for(const f of ['ruletest.mjs', 'querytest.mjs'])
    if(existsSync(join(ROOT, 'cuda', f)))
      run(join('cuda', f), node, [join(ROOT, 'cuda', f)]);
  const py = process.platform === 'win32' ? 'python' : 'python3';
  run('tools/servetest.py', py, [join(ROOT, 'tools', 'servetest.py')]);
  // the headless runner end to end (tools/linentest.mjs): it computes a small scene twice and reads back what it wrote, a few seconds
  run('tools/linentest.mjs', node, [join(ROOT, 'tools', 'linentest.mjs')]);
  // an engine of one's own end to end (tools/remotetest.mjs): the Python template engine on a port, a scene run on it, a term it lacks refused; needs NumPy and websockets
  run('tools/remotetest.mjs', node, [join(ROOT, 'tools', 'remotetest.mjs')]);
  // the reference engine against Brian 2 (tools/brianref.mjs): about three minutes of Python in .venv-analysis.
  // Full run only.
  run('tools/brianref.mjs', node, [join(ROOT, 'tools', 'brianref.mjs')]);
  run('tools/published.mjs', node, [join(ROOT, 'tools', 'published.mjs')]);
  run('tools/exportcheck.mjs', node, [join(ROOT, 'tools', 'exportcheck.mjs')]);
} else {
  process.stdout.write('skip cuda/*.mjs, tools/servetest.py, tools/brianref.mjs, tools/published.mjs and tools/exportcheck.mjs (--quick)\n');
}

process.stdout.write(
  `\n${failed ? failed + ' FAILED' : 'all ' + results.length + ' passed'} ` +
  `in ${((Date.now() - t0)/1000).toFixed(1)} s` + (quick ? ' (quick)' : '') + '\n');
process.stdout.write('note: the statistics battery (validate.html) is a ' +
  'separate manual gate and was not run here.\n');
process.exit(failed ? 1 : 0);

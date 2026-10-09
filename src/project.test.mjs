// The project layout and the migration into it, against a real server.
// The migration moves the only copy of a night's training: a move that deletes before it verifies, or that runs twice and loses the second half, costs data that cannot be regenerated.

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, existsSync, mkdirSync, writeFileSync,
         readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:net';
import { probe } from './store.js';
import { PROJECT_FORMAT, PROJECT_FILE, readProject, writeProject, newProject,
         openProject, addScene, migrate, runDir, runFile, runBrainPath,
         runBrainsDir, ckptName, RUN_STATUS, RUN_METRICS, RUN_TRIALS,
         RUN_CALIBRATION } from './project.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
// A port nothing else holds, asked of the OS, so a server another project left on a fixed number cannot answer in place of the one spawned here.
const PORT = await new Promise((res, rej) => { const s = createServer(); s.on('error', rej);
  s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const BASE = 'http://localhost:' + PORT;
const PROJ = mkdtempSync(join(tmpdir(), 'npproj-'));

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond) pass++;
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};

// --- names, with no server needed -----------------------------------------
{
  ok('a checkpoint keeps its run tag, because the file travels',
     ckptName('20260901T2000', 3*60000) === 'ckpt-20260901T2000-00003min.npb',
     ckptName('20260901T2000', 3*60000));
  ok('checkpoint minutes are zero padded so they sort',
     ckptName('t', 7*60000) < ckptName('t', 70*60000));
  ok('a run brain path sits under the run', runBrainPath('t', 'a.npb') === 'runs/t/brains/a.npb');
  ok('a run file does not', runFile('t', 'run.json') === 'runs/t/run.json');
}

const srv = spawn('python', [join(ROOT, 'serve.py'), String(PORT), '--project', PROJ],
                  { cwd: ROOT, stdio: 'ignore' });
try {
  let up = false;
  for(let i = 0; i < 60 && !up; i++){
    if(srv.exitCode !== null) break;
    try { await fetch(BASE + '/project'); up = true; }
    catch(e){ await new Promise(r => setTimeout(r, 250)); }
  }
  if(!up){ console.log('FAIL: the server never came up on port ' + PORT); process.exit(1); }
  const st = await probe(BASE, fetch);

  // --- the project file ---------------------------------------------------
  ok('an empty folder has no project file yet', (await readProject(st)) === null);
  const p = await openProject(st, 'demo');
  ok('opening one creates it', p.format === PROJECT_FORMAT && p.name === 'demo');
  ok('and it is on disk', existsSync(join(PROJ, PROJECT_FILE)));
  ok('opening again returns the same one, not a fresh one',
     (await openProject(st, 'other')).created === p.created);

  addScene(p, 'scenes/main.json');
  addScene(p, 'scenes/main.json');          // twice on purpose
  await writeProject(st, p);
  const reread = await readProject(st);
  ok('a scene is listed once, not twice', reread.scenes.length === 1, reread.scenes.length);
  ok('and the first scene becomes the active one',
     reread.activeScene === 'scenes/main.json');
  ok('the label defaults to the bare name', reread.scenes[0].label === 'main',
     reread.scenes[0].label);

  // A project written by a later build must not be read as if it were this one: reading it would write it back downgraded.
  await st.write(PROJECT_FILE, JSON.stringify({ ...newProject('future'),
    format: PROJECT_FORMAT + 1 }));
  let threw = '';
  try { await readProject(st); } catch(e){ threw = e.message; }
  ok('a newer format is refused, not guessed at', /format/.test(threw), threw);
  ok('and the refusal says what to do', /[Uu]pdate/.test(threw), threw);

  await st.write(PROJECT_FILE, '{ not json');
  threw = '';
  try { await readProject(st); } catch(e){ threw = e.message; }
  ok('a corrupt project file names itself in the error',
     threw.includes(PROJECT_FILE), threw);
  await writeProject(st, p);                 // put it back

  // --- migration ----------------------------------------------------------
  // A folder in the earlier run layout, built by hand so the test does not depend on code that writes it.
  const old = (tag, name, body) => {
    mkdirSync(join(PROJ, 'runs', tag), { recursive:true });
    writeFileSync(join(PROJ, 'runs', tag, name), body);
  };
  old('20260830T0100', 'status.json', '{"phase":"done"}');
  old('20260830T0100', 'metrics-20260830T0100.json', '[{"simMin":1}]');
  old('20260830T0100', 'trials-20260830T0100.npt', Buffer.from([1,2,3,4]));
  old('20260830T0100', 'calibration-20260830T0100.json', '{"cal":1}');
  old('20260830T0100', 'ckpt-20260830T0100-00001min.npb', Buffer.alloc(64, 7));
  old('20260830T0100', 'ckpt-20260830T0100-00002min.npb', Buffer.alloc(64, 9));
  old('20260830T0100', 'notes-i-wrote.txt', 'do not touch this');
  old('20260831T0200', 'status.json', '{"phase":"error"}');

  // six in the first run (status, metrics, trials, calibration, two checkpoints) and the second run's status, so seven
  const r1 = await migrate(st, () => {});
  ok('it reports what it moved', r1.moved === 7 && r1.runs === 2,
     JSON.stringify(r1));

  const at = (...a) => join(PROJ, 'runs', ...a);
  ok('status becomes run.json', existsSync(at('20260830T0100', RUN_STATUS)));
  ok('metrics loses the redundant tag', existsSync(at('20260830T0100', RUN_METRICS)));
  ok('trials too', existsSync(at('20260830T0100', RUN_TRIALS)));
  ok('calibration too', existsSync(at('20260830T0100', RUN_CALIBRATION)));
  ok('checkpoints move into brains/',
     existsSync(at('20260830T0100', 'brains', 'ckpt-20260830T0100-00001min.npb')) &&
     existsSync(at('20260830T0100', 'brains', 'ckpt-20260830T0100-00002min.npb')));
  ok('and no longer sit loose in the run',
     !existsSync(at('20260830T0100', 'ckpt-20260830T0100-00001min.npb')));
  ok('a second run is migrated as well',
     existsSync(at('20260831T0200', RUN_STATUS)));

  ok('contents survive the move byte for byte',
     readFileSync(at('20260830T0100', 'brains', 'ckpt-20260830T0100-00002min.npb'))
       .every(b => b === 9));
  ok('and so does the text',
     readFileSync(at('20260830T0100', RUN_METRICS), 'utf8') === '[{"simMin":1}]');

  ok('an unrecognized file is left exactly where it was',
     existsSync(at('20260830T0100', 'notes-i-wrote.txt')) &&
     readFileSync(at('20260830T0100', 'notes-i-wrote.txt'), 'utf8')
       === 'do not touch this');

  const r2 = await migrate(st, () => {});
  ok('running it again moves nothing', r2.moved === 0, JSON.stringify(r2));
  ok('and did not disturb what it moved the first time',
     existsSync(at('20260830T0100', RUN_METRICS)) &&
     existsSync(at('20260830T0100', 'brains', 'ckpt-20260830T0100-00001min.npb')));

  // interrupted half way: the copy exists, the original still does too
  old('20260901T0300', 'status.json', '{"phase":"done"}');
  mkdirSync(at('20260901T0300', 'brains'), { recursive:true });
  writeFileSync(at('20260901T0300', 'ckpt-20260901T0300-00001min.npb'), Buffer.alloc(32, 3));
  writeFileSync(at('20260901T0300', 'brains', 'ckpt-20260901T0300-00001min.npb'),
                Buffer.alloc(32, 3));
  const r3 = await migrate(st, () => {});
  ok('a half-finished move is finished, not doubled', r3.moved === 2,
     JSON.stringify(r3));
  ok('the duplicate original is cleared',
     !existsSync(at('20260901T0300', 'ckpt-20260901T0300-00001min.npb')));
  ok('and the copy is still intact',
     readFileSync(at('20260901T0300', 'brains', 'ckpt-20260901T0300-00001min.npb'))
       .every(b => b === 3));

  const bare = await migrate(st, () => {});
  ok('a folder with nothing left to move reports nothing', bare.moved === 0);
} finally {
  srv.kill();
  rmSync(PROJ, { recursive:true, force:true });
}


// --- startupScene: the buffer never saves under a name that is not its own ---
{
  const { startupScene } = await import('./project.js');
  const M = 'scenes/main.json', X = 'scenes/exp1.json';
  let r = startupScene(M, M, true);
  ok('project and buffer agree: keep the buffer, no load', r.scene === M && r.load === false);
  r = startupScene(M, X, true);
  ok('project names main, buffer is exp1, main has a file: load main from disk',
     r.scene === M && r.load === true);
  r = startupScene(M, X, false);
  ok('project names main, buffer is exp1, no main file: stay in exp1 so it saves to itself',
     r.scene === X && r.load === false);
  r = startupScene(M, null, true);
  ok('untagged buffer with a file on disk: disk wins', r.scene === M && r.load === true);
  r = startupScene(M, null, false);
  ok('untagged buffer and no file: fresh project, buffer becomes main', r.scene === M && r.load === false);
  r = startupScene(null, X, false);
  ok('no project scene named: the buffer keeps its own', r.scene === X && r.load === false);
  r = startupScene(null, null, false);
  ok('nothing known: no scene, no load', r.scene === null && r.load === false);
}

console.log(fail ? fail + ' failed, ' + pass + ' passed'
                 : 'all passed (' + pass + ' checks)');
process.exit(fail ? 1 : 0);

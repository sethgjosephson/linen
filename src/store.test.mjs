// The storage layer, checked against a real serve.py rather than a stub: every bug this layer can have lives in the seam (a path the client encodes one way and the server decodes another, a 404 that should read as no file), so the assertions are made against the files on disk, not against what the API claims it did.
// The browser half needs OPFS, which node does not have; what is checked here is the part that decides which store you get, and the shared path normalizer both halves route through, which is what keeps the two layouts identical.

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, existsSync, readFileSync, mkdirSync,
         writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:net';
import { normPath, probe } from './store.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
// A port nothing else holds, asked of the OS, so a server another project left on a fixed number cannot answer in place of the one spawned here.
const PORT = await new Promise((res, rej) => { const s = createServer(); s.on('error', rej);
  s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const BASE = 'http://localhost:' + PORT;
const PROJ = mkdtempSync(join(tmpdir(), 'npstore-'));

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond) pass++;
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};

// --- the path normalizer, which both halves share -------------------------
{
  for(const [given, want] of [
    ['runs/t1/a.npb', 'runs/t1/a.npb'],
    ['/runs/t1/a.npb', 'runs/t1/a.npb'],
    ['runs//t1///a.npb', 'runs/t1/a.npb'],
    ['runs\\t1\\a.npb', 'runs/t1/a.npb'],      // a windows path still means one thing
    ['./runs/./t1/a.npb', 'runs/t1/a.npb'],
    ['runs/t1/../t2/a.npb', 'runs/t2/a.npb'],  // resolved, not refused
    ['brains/x.npb/', 'brains/x.npb'],
  ]) ok('normalizes ' + given, normPath(given) === want, 'got ' + normPath(given));

  for(const bad of ['../escape', 'runs/../../escape', '', '/', '..', './']){
    let threw = false;
    try { normPath(bad); } catch(e){ threw = true; }
    ok('refuses ' + JSON.stringify(bad), threw);
  }
}

// --- which store you get --------------------------------------------------
{
  const s = await probe('http://localhost:1', fetch);
  ok('with no server it falls back to browser storage', s.mode === 'browser');
  ok('and the fallback says why', /serve\.py/.test(s.note), s.note);
  ok('and the fallback has no root to name', s.root === null);

  // A server that answers /project but reports the caller cannot write: that is the deployed site, and it must not be treated as a project folder.
  const notWritable = async () => ({
    ok: true, status: 200,
    json: async () => ({ ok:true, root:'/somewhere', name:'somewhere',
                         writable:false })
  });
  const r = await probe('', notWritable);
  ok('a read-only project folder still falls back', r.mode === 'browser');
  ok('and says the folder is not writable from here',
     /writable/.test(r.note), r.note);
}

const srv = spawn('python',
  [join(ROOT, 'serve.py'), String(PORT), '--project', PROJ],
  { cwd: ROOT, stdio: 'ignore' });

try {
  let up = false;
  for(let i = 0; i < 60 && !up; i++){
    try { await fetch(BASE + '/project'); up = true; }
    catch(e){ await new Promise(r => setTimeout(r, 250)); }
  }
  if(!up){ console.log('FAIL: the server never came up on port ' + PORT); process.exit(1); }

  const s = await probe(BASE, fetch);
  ok('served by serve.py, it uses the project folder', s.mode === 'project');
  ok('and names the folder it was given',
     s.root.replace(/\\/g, '/') === PROJ.replace(/\\/g, '/'), s.root);
  ok('and the label is the path, not a euphemism', s.label === s.root);

  await s.write('runs/t1/metrics.json', '{"rate":3.5}');
  ok('a string writes as utf-8 on disk',
     readFileSync(join(PROJ, 'runs', 't1', 'metrics.json'), 'utf8')
       === '{"rate":3.5}');
  const back = await s.read('runs/t1/metrics.json');
  ok('and reads back as bytes',
     new TextDecoder().decode(back) === '{"rate":3.5}');

  const blob = new Uint8Array(4096);
  for(let i = 0; i < blob.length; i++) blob[i] = (i * 7 + 13) & 255;
  await s.write('runs/t1/ckpt.npb', blob);
  const got = await s.read('runs/t1/ckpt.npb');
  let same = got && got.length === blob.length;
  if(same) for(let i = 0; i < blob.length; i++)
    if(got[i] !== blob[i]){ same = false; break; }
  ok('binary survives the round trip byte for byte', same,
     got ? 'length ' + got.length : 'null');

  // a typed array that is a view into a larger buffer, which is what a subarray of a checkpoint is: writing its whole backing buffer instead of the view would silently store the wrong bytes
  const bigger = new Float32Array([1, 2, 3, 4, 5, 6]);
  await s.write('runs/t1/view.bin', bigger.subarray(2, 4));
  const view = await s.read('runs/t1/view.bin');
  ok('a subarray writes only its own bytes', view.length === 8, view.length);
  ok('and they are the right ones',
     new Float32Array(view.buffer, view.byteOffset, 2)[0] === 3);

  await s.write('runs/t1/deep/deeper/still/x.txt', 'here');
  ok('a deep path creates its parents',
     existsSync(join(PROJ, 'runs', 't1', 'deep', 'deeper', 'still', 'x.txt')));

  ok('a missing file reads as null',
     (await s.read('runs/t1/nope.json')) === null);

  const entries = await s.list('runs/t1');
  const names = entries.map(e => e.name);
  ok('a listing sees the files', names.includes('metrics.json')
     && names.includes('ckpt.npb'), names.join(','));
  ok('and the subdirectory, marked as one',
     entries.some(e => e.name === 'deep' && e.dir === true));
  ok('and reports real sizes',
     entries.find(e => e.name === 'ckpt.npb').size === 4096);
  ok('a listing of nothing is empty, not an error',
     (await s.list('runs/does-not-exist')).length === 0);
  ok('the root lists the top level',
     (await s.list('')).some(e => e.name === 'runs'));

  // names with characters a URL would otherwise eat
  await s.write('runs/t1/a b+c#d%e.json', 'awkward');
  ok('an awkward filename round trips',
     new TextDecoder().decode(await s.read('runs/t1/a b+c#d%e.json'))
       === 'awkward');
  ok('and lands under that exact name on disk',
     existsSync(join(PROJ, 'runs', 't1', 'a b+c#d%e.json')));

  const url = await s.url('runs/t1/ckpt.npb');
  ok('a url points at the server', url.startsWith(BASE + '/project/file/'), url);
  const direct = await fetch(url);
  ok('and fetching it returns the file',
     (await direct.arrayBuffer()).byteLength === 4096);

  ok('a delete reports it removed something',
     (await s.remove('runs/t1/metrics.json')) === true);
  ok('and the file is gone',
     !existsSync(join(PROJ, 'runs', 't1', 'metrics.json')));
  ok('deleting what is already gone is not an error',
     (await s.remove('runs/t1/metrics.json')) === false);

  // the guard, reached through this layer rather than by hand
  let threw = false;
  try { await s.write('../escaped.txt', 'nope'); } catch(e){ threw = true; }
  ok('a write above the project is refused before it is sent', threw);
  ok('and nothing escaped',
     !existsSync(join(dirname(PROJ), 'escaped.txt')));

  // A run written by the host process, read back through the store: the two write into the same folder and neither knows about the other, so this is the only place that pairing is checked.
  mkdirSync(join(PROJ, 'runs', 'hostrun'), { recursive:true });
  writeFileSync(join(PROJ, 'runs', 'hostrun', 'ckpt-000.npb'), Buffer.from([1,2,3]));
  ok('a file the host wrote is visible to the app',
     (await s.list('runs/hostrun')).some(e => e.name === 'ckpt-000.npb'));
  ok('and readable', (await s.read('runs/hostrun/ckpt-000.npb')).length === 3);
} finally {
  srv.kill();
  rmSync(PROJ, { recursive:true, force:true });
}

console.log(fail ? fail + ' failed, ' + pass + ' passed'
                 : 'all passed (' + pass + ' checks)');
process.exit(fail ? 1 : 0);

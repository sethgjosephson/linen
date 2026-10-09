// Every module parses.
// Several run only in the page (the trainer, the viewer, the battery), so no suite imports them, and node --check can pass a battery that the browser refuses (an apostrophe inside a quoted label).
// A SourceTextModule parses a module without linking its imports, so it reads files whose imports only the page can resolve.
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ok, report } from '../tools/harness.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const files = [];
for(const dir of ['src', 'host', 'tools', 'cuda'])
  for(const f of readdirSync(join(root, dir)))
    if(/\.(mjs|js)$/.test(f)) files.push(join(root, dir, f));

const script = `
const vm = require('vm'), fs = require('fs');
const bad = [];
for(const f of process.argv.slice(1)){
  try { new vm.SourceTextModule(fs.readFileSync(f, 'utf8')); }
  catch(e){ bad.push(f + ': ' + e.message); }
}
process.stdout.write(JSON.stringify(bad));`;
const r = spawnSync(process.execPath, ['--experimental-vm-modules', '--no-warnings', '-e', script, ...files], { encoding:'utf8' });
let bad = null;
try { bad = JSON.parse(r.stdout); } catch(_){ }
ok('the parse check ran', Array.isArray(bad), (r.stderr || '').slice(0, 300));
ok('every module parses', Array.isArray(bad) && !bad.length, (bad || []).join('; '));
ok('it read the page-only modules too', files.some(f => /train\.js$/.test(f)) && files.some(f => /validate\.js$/.test(f)),
  files.length + ' files');
report('parse');

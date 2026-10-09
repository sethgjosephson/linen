// Every node a tool or module calls by name exists.
// A node folded into another (retype into cell type, scene format 11) is migrated in saved scenes, but a script that computes it by name breaks on load.
import { readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NODE_DEFS } from './nodes.js';
import { ok, report } from '../tools/harness.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const files = [];
for(const dir of ['src', 'tools', 'host'])
  for(const f of readdirSync(join(root, dir)))
    if(/\.(mjs|js)$/.test(f)) files.push(join(dir, f));

const missing = [];
let uses = 0;
for(const f of files){
  const text = readFileSync(join(root, f), 'utf8');
  for(const m of text.matchAll(/NODE_DEFS\.([A-Za-z_$][\w$]*)|NODE_DEFS\[['"]([^'"]+)['"]\]/g)){
    const name = m[1] || m[2];
    uses++;
    if(!(name in NODE_DEFS)) missing.push(f + ': ' + name);
  }
}
ok('the scan finds the named node calls', uses > 20, uses + ' uses in ' + files.length + ' files');
ok('every node called by name exists', !missing.length, missing.join(', '));
report('nodedefs');

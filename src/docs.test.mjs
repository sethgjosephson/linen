// The documentation module, checked where it is cheapest to check: an unescaped apostrophe inside a single-quoted blurb makes the whole file fail to parse, which otherwise only the battery in a browser would catch.
// Run: node src/docs.test.mjs
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NODE_DEFS, nodeDoc, nodeDocProblems } from './nodes.js';
import { NODES as DOCS, PAGES } from './docs.js';
import { loadPlugin, importer } from './plugins.js';

assert.ok(DOCS && typeof DOCS === 'object', 'docs.js parses and exports its node prose');
assert.ok(Array.isArray(PAGES) && PAGES.length, 'and its pages');

const types = Object.keys(NODE_DEFS);
for(const t of types){
  const d = DOCS[t];
  assert.ok(d, `every node type is documented: ${t} is not`);
  assert.ok(typeof d.blurb === 'string' && d.blurb.length > 20,
    `every node has a blurb: ${t}`);
}
console.log('ok  every node type has documentation (' + types.length + ' types)');

// the same rule registerNode holds a plugin node to (nodeDocProblems), so a built-in node and a plugin node are documented alike
for(const t of types){
  const problems = nodeDocProblems(t, NODE_DEFS[t], nodeDoc(t, DOCS));
  assert.deepEqual(problems, [], problems.join('; '));
}
console.log('ok  parameters and their prose agree in both directions');

// the example node modules in the repository (examples/nodes) load, and their nodes are documented by the same rule
const exdir = join(dirname(fileURLToPath(import.meta.url)), '..', 'examples', 'nodes');
const mods = readdirSync(exdir).filter(f => /\.m?js$/.test(f));
assert.ok(mods.length, 'the repository holds an example node module');
for(const f of mods){
  const r = await loadPlugin(f, readFileSync(join(exdir, f), 'utf8'), importer());
  assert.equal(r.error, null, 'example node module ' + f + ' loads: ' + r.error);
  for(const k of r.keys){
    const problems = nodeDocProblems(k, NODE_DEFS[k], nodeDoc(k, DOCS));
    assert.deepEqual(problems, [], problems.join('; '));
    assert.ok(!DOCS[k], 'a plugin node carries its own prose, not an entry in docs.js: ' + k);
  }
}
console.log('ok  the example node modules are documented (' + mods.join(', ') + ')');
console.log('docs: all checks passed');

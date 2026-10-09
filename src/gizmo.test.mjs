// Every node the graph editor hands to the 3D manipulator on a double-click has an entry in the viewer's table, keyed by its current type name.
// The table is read from the source, since viewer.js imports three.js.
import { readFileSync } from 'node:fs';
import { NODE_DEFS } from './nodes.js';
import { ok, report } from '../tools/harness.mjs';

const src = readFileSync(new URL('./viewer.js', import.meta.url), 'utf8');
const start = src.indexOf('const GEO_EDIT = {');
const body = src.slice(start, src.indexOf('\n};', start));
const keys = new Set([...body.matchAll(/^  ([A-Za-z_$][\w$]*): \{ modes:/gm)].map(m => m[1]));
ok('the table is found', keys.size >= 5, [...keys].join(', '));

// what editor.js sends: every regions node, the move node, and a stimulus (drawn as a sphere).
// A spline has its own editor; a mesh region has none and a double-click on it opens only its properties.
const sent = Object.entries(NODE_DEFS).filter(([, d]) => d.cat === 'regions').map(([k]) => k)
  .concat(['move']).filter(k => k !== 'spline' && k !== 'mesh');
const missing = sent.filter(k => !keys.has(k));
ok('every node sent to the manipulator has an entry', !missing.length, 'missing: ' + missing.join(', '));
ok('every entry is a node type', [...keys].every(k => k in NODE_DEFS), [...keys].filter(k => !(k in NODE_DEFS)).join(', '));
report('gizmo');

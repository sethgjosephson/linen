// The statistics battery loads.
// It runs only in a browser, so nothing else in the quick suite reads it, and a syntax error in it (an apostrophe inside a quoted label) can pass node --check and every suite while validate.html cannot start.
// A real import parses it.
import { ok, report } from '../tools/harness.mjs';

let V = null, err = null;
try { V = await import('./validate.js'); } catch(e){ err = e; }
ok('validate.js imports', !!V, err ? err.name + ': ' + err.message : '');
ok('every group has a name, a source and a run',
  !!V && V.TESTS.length > 40 && V.TESTS.every(t => t.name && t.source && typeof t.run === 'function'),
  V ? V.TESTS.length + ' groups' : 'not loaded');
const names = V ? V.TESTS.map(t => t.name) : [];
ok('group names are unique, so ?only= finds one', new Set(names).size === names.length);
report('battery');

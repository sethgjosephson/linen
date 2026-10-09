// MODEL.md's citations must point at code that still exists.
//
// Every formula in MODEL.md names the file that implements it, with a line number as a hint.
// The line numbers drift as the engine files change, so an exact-line gate cannot hold; this gate checks the durable half instead: for each formula, a distinctive token must still be present in the file MODEL.md cites for it.
// A citation whose file no longer contains the formula, the rot that matters, fails here; a line number that slid by a few, which does not, does not.
//
// It also checks the reverse for the anchor set: the token must NOT have moved to a different source file, so a formula relocated across files is caught.

import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ok, report } from '../tools/harness.mjs';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const doc = readFileSync(join(ROOT, 'MODEL.md'), 'utf8');
const src = rel => existsSync(join(ROOT, rel)) ? readFileSync(join(ROOT, rel), 'utf8') : null;

// Every file MODEL.md cites must exist.
// This alone catches a file renamed or removed out from under the document.
const citedFiles = [...new Set([...doc.matchAll(/`?(src\/[\w./]+\.js)/g)].map(m => m[1]))];
ok('MODEL.md cites source files', citedFiles.length > 2, String(citedFiles.length));
for(const f of citedFiles)
  ok('cited file exists: ' + f, existsSync(join(ROOT, f)), 'MODEL.md points at a file that is gone');

// A formula, the file MODEL.md says implements it, and a token from that formula that must appear in the file.
const anchors = [
  ['Izhikevich half-step', 'src/simworker.js', '(vi - vri)*(vi - vti)'],
  ['absolute refractory', 'src/simworker.js', 'S.refCnt[i] > 0'],
  ['total input current', 'src/simworker.js', 'ring[row+i] + bias[i]'],
  ['kick synapse delivery', 'src/simworker.js', '((cur+delay[s])%ROWS)*n + post[s]'],
  ['exp synapse current', 'src/simworker.js', 'ge*itauE + gi*itauI'],
  ['psc convention', 'src/simworker.js', 'export const itauOf'],
  ['trace decay', 'src/simworker.js', 'Kpre[i] *= decS'],
  ['heterosynaptic gate', 'src/simworker.js', 'z/S.kRef'],
  ['soft bounds', 'src/simworker.js', 'function softBound'],
  ['committed-synapse criterion', 'src/simworker.js', 'RT[o+9]*0.9'],
  ['synaptic scaling error', 'src/simworker.js', 'Math.max(-2, Math.min(2'],
  ['measured set point', 'src/simworker.js', 'S.rhoI[i] = Math.min(30'],
  ['STP facilitation', 'src/simworker.js', 'R[i] + U*(1 - R[i])'],
  ['STP recovery', 'src/simworker.js', '1 - (1 - X[i])*dx'],
  ['per-ms noise', 'src/rand.js', 'export function noiseDraw'],
  ['connection probability', 'src/nodes.js', 'p.prob * T.probMul'],
  ['lognormal weights', 'src/nodes.js', 'wbase*Math.exp(sg*pairGauss'],
  ['proxy variance scale', 'src/nodes.js', '1/Math.sqrt(res)'],
  ['proxy mean compensation', 'src/nodes.js', '(1/res - 1/Math.sqrt(res))*0.001'],
  ['rule table field order', 'src/nodes.js', "RULE_FIELDS = ['aP'"],
];

// Where each token actually lives, so a formula moved to another file is caught.
const home = new Map();
for(const f of ['src/simworker.js', 'src/rand.js', 'src/nodes.js', 'src/gpuworker.js']){
  const s = src(f);
  if(s) home.set(f, s);
}

for(const [name, rel, tok] of anchors){
  const s = home.get(rel) || src(rel);
  const here = s ? s.includes(tok) : false;
  ok(name + ' is still in ' + rel, here,
     here ? '' : 'token "' + tok + '" not found where MODEL.md cites it');
  // and not only elsewhere
  if(!here){
    for(const [f, s2] of home)
      if(f !== rel && s2.includes(tok)){ ok('  (it moved to ' + f + ')', false); break; }
  }
}

report('modelcite');

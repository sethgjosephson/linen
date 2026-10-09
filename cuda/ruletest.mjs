// Does the CUDA child obey the per-synapse rule byte?
//
// The byte carries a rule index, and the danger is a graph, a pair table and a page of documentation all saying a pathway learned by its own rule while the engine quietly used the checkpoint's numbers.
// Two halves of one synapse array get two rules that differ in one amplitude and the same drive; the weights must come apart by the ratio of the amplitudes.
//
//   node cuda/ruletest.mjs [prefix] [seconds] [exePath]
//
// The same check runs for the reference and WebGPU engines in the validation battery, under "engines obey the per-synapse rule".
// This one exists because the child cannot be reached from the battery: it lives behind a WebSocket and a host process.

import fs from 'node:fs';
import { PROTOCOL } from '../src/protocol.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { CudaEngine } from '../host/cudaengine.mjs';
import { initState } from '../src/rand.js';
import { classicRow } from '../src/form.js';
import { buildRuleTable } from '../src/nodes.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PREFIX = process.argv[2] || 'paritynet';
const SECONDS = +(process.argv[3] || 6);
const EXE = process.argv[4] || path.join(HERE, 'engine.exe');

const buf = fs.readFileSync(path.join(HERE, PREFIX + '.bin'));
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
const n = dv.getUint32(0, true), m = dv.getUint32(4, true);
const syn = dv.getUint32(8, true);
const tauE = dv.getFloat32(12, true), tauI = dv.getFloat32(16, true);

let off = 68;
const take = (Type, count) => {
  const a = new Type(buf.buffer.slice(buf.byteOffset + off,
    buf.byteOffset + off + count*Type.BYTES_PER_ELEMENT));
  off += count*Type.BYTES_PER_ELEMENT;
  return a;
};
take(Float32Array, n); take(Float32Array, n);
take(Float32Array, n); take(Float32Array, n);
const bias = take(Float32Array, n);
const preStart = take(Int32Array, n+1), post = take(Int32Array, m),
  w = take(Float32Array, m), delay = take(Uint8Array, m);

const NE = Math.round(n*0.8);
const ntype = new Uint8Array(n);
for(let i=0;i<n;i++) ntype[i] = i < NE ? 0 : 1;
// RS and FS as classic rows: the 2003 presets carried as rows of the 2007 form (src/form.js), which is what every engine takes
const CLASSES = [{ f7:classicRow({ a:0.02, b:0.2, c:-65, d:8 }) }, { f7:classicRow({ a:0.1, b:0.2, c:-65, d:2 }) }];

// Two rules differing in one amplitude and nothing else.
// Everything the checkpoint would normally supply is here as the base, so rule 1 is that base exactly and rule 2 is it with a fivefold A+.
// Small amplitudes: at 0.004 and 0.02 over six seconds every excitatory weight reaches the wmax clamp and the ratio reads 1.00, so the test has to stay well inside the bounds to see anything.
const A_WEAK = +(process.argv[5] || 0.0002);
const A_STRONG = +(process.argv[6] || 0.001);
// aM is zero: depression is a common term shared by both halves, so with it on the ratio of the two halves is not the ratio of their amplitudes; with only potentiation running it is exactly that.
const base = { aP:A_WEAK, aM:0, wmax:60, wdep:0, trip:0, het:0, tin:0,
  iEta:0.002, cons:0, consW:0, consP:10 };
const { ruleTable, ruleCount } = buildRuleTable(base, [{ name:'strong', aP:A_STRONG }]);
const RATIO = A_STRONG/A_WEAK;

// the first half of the synapse array on rule 1, the second on rule 2
const pmask = new Uint8Array(m);
for(let s=0;s<m;s++) pmask[s] = s < (m >> 1) ? 1 : 2;

const w0 = w.slice();
const eng = new CudaEngine(EXE,
  msg => { if(/error|Error/.test(String(msg))) console.error(msg); });
let learned = null;
eng.on('message', msg => { if(msg.cmd === 'weights') learned = msg.w; });

eng.postMessage({ cmd:'init', protocol:PROTOCOL, count:n, ntype, bias,
  syn, tauE, tauI, protocols:[], plast:1,
  ...base, tauS:20, iRho:5, rhoMode:0, calS:10, scale:0, sEta:0.001,
  tauY:114, refrac:0, stp:0, stpU:0.2, stpTauD:200, stpTauF:600,
  ruleTable, ruleCount, pmask,
  inputs:[], preStart, post, w:w.slice(), delay, types:CLASSES,
  seed:1, ...initState(1, ntype, CLASSES) });

const MS = SECONDS*1000, CHUNK = 500;
for(let t=0;t<MS;t+=CHUNK) eng.postMessage({ cmd:'tick', steps:CHUNK });
eng.postMessage({ cmd:'getWeights' });

await new Promise(r => {
  const poll = () => learned ? r() : setTimeout(poll, 50);
  setTimeout(r, 60000);
  poll();
});

if(!learned){
  console.log('FAIL: the child returned no weights');
  process.exit(1);
}

const meanDelta = (lo, hi) => {
  let sum = 0, c = 0;
  for(let k=lo;k<hi;k++) if(w0[k] > 0){ sum += learned[k] - w0[k]; c++; }
  return c ? sum/c : 0;
};
const dA = meanDelta(0, m >> 1), dB = meanDelta(m >> 1, m);
const ratio = dB/(dA || 1e-9);
// how many excitatory weights ended up against the ceiling; a saturated run cannot answer the question and should say so rather than report a ratio
let pinned = 0, exc = 0;
for(let k=0;k<m;k++) if(w0[k] > 0){ exc++; if(learned[k] >= base.wmax*0.999) pinned++; }
const pinnedFrac = pinned/Math.max(1, exc);

console.log(`cuda rule test: ${m} synapses, ${SECONDS}s`);
console.log(`  rule 1 (aP ${A_WEAK}) mean dw = ${dA.toFixed(6)}`);
console.log(`  rule 2 (aP ${A_STRONG}) mean dw = ${dB.toFixed(6)}`);
console.log(`  ratio ${ratio.toFixed(2)}x against an amplitude ratio of ${RATIO}x`);
console.log(`  ${(pinnedFrac*100).toFixed(1)}% of excitatory weights at the clamp`);

if(pinnedFrac > 0.05){
  console.log('  INCONCLUSIVE: the run saturated, so the two rules cannot be told apart.');
  console.log('  Lower the amplitudes or shorten the run: node cuda/ruletest.mjs ' +
    PREFIX + ' ' + SECONDS + ' <exe> <weakA+> <strongA+>');
  process.exit(1);
}
const moved = Math.abs(dA) > 1e-6 && Math.abs(dB) > 1e-6;
const obeys = ratio > RATIO*0.7 && ratio < RATIO*1.3;
console.log(`  both halves moved         ${moved ? 'ok' : 'MISMATCH'}`);
console.log(`  the stronger rule leads   ${obeys ? 'ok' : 'MISMATCH'}`);
eng.postMessage({ cmd:'stop' });
process.exit(moved && obeys ? 0 : 1);

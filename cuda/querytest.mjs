// Checks that the CUDA child answers the two viewer messages, watch and query, against the reference engine's answers for the same network.
//
//   node cuda/querytest.mjs <prefix> [idx] [exePath]
import fs from 'node:fs';
import { PROTOCOL } from '../src/protocol.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { CudaEngine } from '../host/cudaengine.mjs';
import { initState } from '../src/rand.js';
import { classicRow } from '../src/form.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PREFIX = process.argv[2] || 'parity2';
const IDX = +(process.argv[3] || 1234);
const EXE = process.argv[4] || path.join(HERE, 'engine.exe');

const buf = fs.readFileSync(path.join(HERE, PREFIX + '.bin'));
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
const n = dv.getUint32(0, true), m = dv.getUint32(4, true);
const syn = dv.getUint32(8, true);
const tauE = dv.getFloat32(12, true), tauI = dv.getFloat32(16, true);
let off = 68;
const take = (T, c) => {
  const a = new T(buf.buffer.slice(buf.byteOffset + off,
    buf.byteOffset + off + c*T.BYTES_PER_ELEMENT));
  off += c*T.BYTES_PER_ELEMENT;
  return a;
};
const a = take(Float32Array, n), b = take(Float32Array, n),
  c = take(Float32Array, n), d = take(Float32Array, n), bias = take(Float32Array, n);
const preStart = take(Int32Array, n+1), post = take(Int32Array, m),
  w = take(Float32Array, m), delay = take(Uint8Array, m);

// what the reference engine would answer, computed directly from the CSR
const outTotalRef = preStart[IDX+1] - preStart[IDX];
let inTotalRef = 0;
for(let s = 0; s < m; s++) if(post[s] === IDX) inTotalRef++;

const NE = Math.round(n*0.8);
const ntype = new Uint8Array(n);
for(let i=0;i<n;i++) ntype[i] = i < NE ? 0 : 1;
// RS and FS as classic rows: the 2003 presets carried as rows of the 2007 form (src/form.js), which is what every engine takes
const CLASSES = [{ f7:classicRow({ a:0.02, b:0.2, c:-65, d:8 }) }, { f7:classicRow({ a:0.1, b:0.2, c:-65, d:2 }) }];

// the child's own errors are worth seeing when a reply does not arrive; the rest of its chatter is not
const eng = new CudaEngine(EXE,
  m => { if(/error|Error|CUDA error/.test(String(m))) console.log('  child: ' + m); });
let qr = null, sawTrace = null;
eng.on('message', msg => {
  if(msg.cmd === 'queryResult') qr = msg;
  if(msg.cmd === 'state' && msg.vtrace) sawTrace = msg.vtrace;
});
eng.postMessage({ cmd:'init', protocol:PROTOCOL, count:n, ntype, bias, syn, tauE, tauI,
  protocols:[], plast:0, aP:0.003, aM:0.001, tauS:20, wmax:60, wdep:1,
  iEta:0.01, iRho:2, rhoMode:0, calS:10, scale:0, sEta:0.001,
  inputs:[], preStart, post, w:w.slice(), delay, types:CLASSES,
  seed:1, ...initState(1, ntype, CLASSES) });
eng.postMessage({ cmd:'watch', idx:IDX });
// dir 'out' asks for the outgoing side alone; the child has the reverse index and pays nothing either way, but honors the flag so all three engines answer alike.
eng.postMessage({ cmd:'query', idx:IDX, cap:2000, dir:'out' });
await new Promise(r => setTimeout(r, 1200));
const qrOut = qr; qr = null;
eng.postMessage({ cmd:'query', idx:IDX, cap:2000 });
eng.postMessage({ cmd:'tick', steps:50 });
await new Promise(res => {
  const poll = () => (qr && sawTrace) ? res() : setTimeout(poll, 50);
  setTimeout(res, 8000);
  poll();
});
eng.terminate();

if(!qr){ console.log('FAIL: no queryResult from the child'); process.exit(1); }
const outOk = qr.outTotal === outTotalRef;
const inOk = qr.inTotal === inTotalRef;
let sliceOk = true;
for(let k = 0; k < qr.out.length; k++)
  if(qr.out[k] !== post[preStart[IDX] + k]) sliceOk = false;
let ownerOk = true;
for(const pre of qr.inn){
  let found = false;
  for(let s = preStart[pre]; s < preStart[pre+1]; s++) if(post[s] === IDX) found = true;
  if(!found) ownerOk = false;
}
const traceOk = !!sawTrace && sawTrace.length === 50
  && [...sawTrace].some(v => v !== 0);

console.log(`query for neuron ${IDX} of ${n}:`);
console.log(`  outTotal ${qr.outTotal} vs reference ${outTotalRef}  ${outOk ? 'ok' : 'MISMATCH'}`);
console.log(`  inTotal  ${qr.inTotal} vs reference ${inTotalRef}  ${inOk ? 'ok' : 'MISMATCH'}`);
console.log(`  returned ${qr.out.length} outgoing, ${qr.inn.length} incoming`);
console.log(`  outgoing targets match the CSR slice  ${sliceOk ? 'ok' : 'MISMATCH'}`);
console.log(`  every incoming owner projects to it   ${ownerOk ? 'ok' : 'MISMATCH'}`);
console.log(`  membrane trace ${sawTrace ? sawTrace.length : 0} samples, ` +
  `nonzero ${traceOk ? 'ok' : 'MISSING'}`);
let dirOk = false;
if(!qrOut) console.log('  out-only query  NO REPLY');
else {
  const sameOut = qrOut.outTotal === outTotalRef &&
    qrOut.out.length === Math.min(2000, outTotalRef);
  const noIn = qrOut.inTotal === 0 && qrOut.inn.length === 0;
  dirOk = sameOut && noIn;
  console.log(`  out-only: outgoing kept ${sameOut ? 'ok' : 'MISMATCH'}, ` +
    `incoming suppressed ${noIn ? 'ok' : 'STILL ' + qrOut.inn.length}`);
}

process.exit(outOk && inOk && sliceOk && ownerOk && traceOk && dirOk ? 0 : 1);

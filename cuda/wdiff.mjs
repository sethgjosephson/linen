// How two learned-weight files differ (the f32[m] files parity.mjs and dumpnet.mjs write): the share of synapses that differ at all, the largest and the mean absolute difference, and the correlation.
//
//   node cuda/wdiff.mjs a-w.bin b-w.bin [label]
//
// Delivery sums into the ring in whatever order the warps run, so two runs of the same binary differ in the last bits of most weights.
// The spread of a binary against its own second run is the control for a change: a change that moves the weights by no more than that has not changed the engine.
import fs from 'node:fs';
const load = p => { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const [A, B] = [process.argv[2], process.argv[3]].map(load);
if(A.length !== B.length){ console.error('wdiff: ' + A.length + ' against ' + B.length + ' weights'); process.exit(1); }
let n = 0, max = 0, sum = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, sumAbs = 0;
for(let i = 0; i < A.length; i++){
  const d = Math.abs(A[i] - B[i]);
  if(d > 0) n++;
  if(d > max) max = d;
  sum += d; sumAbs += Math.abs(A[i]);
  sa += A[i]; sb += B[i]; saa += A[i]*A[i]; sbb += B[i]*B[i]; sab += A[i]*B[i];
}
const m = A.length, r = (m*sab - sa*sb)/Math.sqrt((m*saa - sa*sa)*(m*sbb - sb*sb));
console.log((process.argv[4] ? process.argv[4] + ' ' : '') + 'synapses ' + m + ', differing ' + n + ' (' + (100*n/m).toFixed(1) + '%), max |dw| '
  + max.toExponential(2) + ', mean |dw| ' + (sum/m).toExponential(2) + ' (mean |w| ' + (sumAbs/m).toFixed(3) + '), r ' + r.toFixed(6));

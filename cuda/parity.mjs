// Drives the CUDA engine child over its stdio protocol on a network dumped by dumpnet.mjs, then compares the learned weights against the reference engine's own run of the same tissue.
//
//   node cuda/parity.mjs <prefix> [seconds] [exePath] [cons] [stp] [scale] [rhoMode]
//
// Reads <prefix>.bin (the tissue) and <prefix>-ref-w.bin (the reference engine's learned weights, written by dumpnet.mjs), runs the child for the same simulated duration, and reports the per-synapse correlation. cons > 0 turns consolidation on in the child only, which is how a new mechanism is checked for having no effect when it is switched off.
import fs from 'node:fs';
import { PROTOCOL } from '../src/protocol.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { CudaEngine } from '../host/cudaengine.mjs';
import { initState } from '../src/rand.js';
import { engineTypes } from '../src/nodes.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PREFIX = process.argv[2] || 'paritynet';
const SECONDS = +(process.argv[3] || 10);
const EXE = process.argv[4] || path.join(HERE, 'engine.exe');
const CONS = +(process.argv[5] || 0);
const STP = +(process.argv[6] || 0);
const SCALE = +(process.argv[7] || 0);
const RHOMODE = +(process.argv[8] || 0);
const PSC = +(process.argv[9] || 0);        // exp current scale, must match the dump
// must match the dump: the child is compared against a reference run that had these on, not against one that did not
const COMMIT = +(process.argv[10] || 0);
const TAUCONS = +(process.argv[11] || 0);

const buf = fs.readFileSync(path.join(HERE, PREFIX + '.bin'));
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
const n = dv.getUint32(0, true), m = dv.getUint32(4, true);
const syn = dv.getUint32(8, true);
const tauE = dv.getFloat32(12, true), tauI = dv.getFloat32(16, true);
const plast = dv.getUint32(20, true), wdep = dv.getUint32(24, true);
const R = { aP:dv.getFloat32(28, true), aM:dv.getFloat32(32, true),
  tauS:dv.getFloat32(36, true), wmax:dv.getFloat32(40, true),
  iEta:dv.getFloat32(44, true), iRho:dv.getFloat32(48, true),
  trip:dv.getFloat32(52, true), tauY:dv.getFloat32(56, true),
  het:dv.getFloat32(60, true), tin:dv.getFloat32(64, true) };

let off = 68;
const take = (Type, count) => {
  const a = new Type(buf.buffer.slice(buf.byteOffset + off,
    buf.byteOffset + off + count*Type.BYTES_PER_ELEMENT));
  off += count*Type.BYTES_PER_ELEMENT;
  return a;
};
const a = take(Float32Array, n), b = take(Float32Array, n),
  c = take(Float32Array, n), d = take(Float32Array, n),
  bias = take(Float32Array, n);
const preStart = take(Int32Array, n+1), post = take(Int32Array, m),
  w = take(Float32Array, m), delay = take(Uint8Array, m);
// feedback inhibition pools, when the dump carries them (dumpnet POOL)
let pool = null, poolK = null;
if(off + 4 <= buf.byteLength){
  const G1 = dv.getUint32(off, true); off += 4;
  const pa = take(Uint16Array, n), pk = take(Float32Array, G1);
  if(G1 >= 2){ pool = pa; poolK = pk; }
}

const NE = Math.round(n*0.8);
const ntype = new Uint8Array(n);
for(let i=0;i<n;i++) ntype[i] = i < NE ? 0 : 1;
// GRADED=1 makes the second class the graded fly relay (FG)
const GRADED = +(process.env.GRADED || 0) === 1;
// CHAN=<tau> puts every excitatory synapse on receptor channel 2 with that
// time constant (the top three bits of the plasticity byte), both sides
const CHAN = +(process.env.CHAN || 0);
// COND=1 runs the conductance synapse (mode 2) with the checkpoint reversals
const COND = +(process.env.COND || 0) === 1;
const chanBytes = (n, ntype, preStart, m) => { const pm = new Uint8Array(m).fill(1);
  if(CHAN > 0) for(let i = 0; i < n; i++) if(ntype[i] === 0) pm.fill(1 | (2 << 5), preStart[i], preStart[i+1]);
  return pm; };
// The rows are RS and FS from the type table: the classic rows (the 2003 presets as 2007 rows, src/form.js) by default.
// MEASURED=1 runs the measured chapter 8 rows instead (rows 10 and 13), whose currents are picoamps; ISCALE lifts the drive for them.
const MEASURED = +(process.env.MEASURED || 0) === 1;
const CLASSES = [engineTypes()[MEASURED ? 10 : 0], GRADED ? engineTypes().find((t, i) => i === 8) : engineTypes()[MEASURED ? 13 : 3]];

const eng = new CudaEngine(EXE, msg => console.error(msg));
let learned = null, learnedRho = null;
// Spike counts per neuron across the run, so the dynamics are compared independently of the weights: a weight correlation alone cannot tell a kernel defect from chaotic divergence in a strongly driven regime.
const childCounts = new Float64Array(n);
let childSpikes = 0;
eng.on('message', msg => {
  if(msg.cmd === 'weights'){ learned = msg.w; learnedRho = msg.rho; }
  else if(msg.cmd === 'state' && msg.fired){
    for(let i=0;i<n;i++) childCounts[i] += msg.fired[i];
    childSpikes += Number(msg.spikes || 0);
  }
});

eng.postMessage({ cmd:'init', protocol:PROTOCOL, count:n, ntype, bias,
  syn:COND ? 2 : syn, eRevE:0, eRevI:-70, psc:PSC, tauE, tauI, protocols:[], plast,
  aP:R.aP, aM:R.aM, tauS:R.tauS, wmax:R.wmax, wdep,
  iEta:R.iEta, iRho:R.iRho, rhoMode:RHOMODE, calS:10, scale:SCALE, sEta:0.001,
  trip:R.trip, tauY:R.tauY, het:R.het, tin:R.tin,
  cons:CONS, consW:0, consP:10,   // 0 = the scaled default, a tenth of wmax
  commit:COMMIT, tauCons:TAUCONS,
  stp:STP, stpOrder:+(process.env.STPORDER || 0), stpU:0.2, stpTauD:200, stpTauF:600,
  ...(pool ? { pool, poolK } : {}),
  inputs:[], preStart, post, w:w.slice(), delay, types:CLASSES,
  // the same initial state and seed dumpnet gave the reference (ESEED default 1)
  seed:1, ...initState(1, ntype, CLASSES),
  pmask:chanBytes(n, ntype, preStart, w.length), ...(CHAN > 0 ? { chanTau:[CHAN] } : {}) });

const MS = SECONDS*1000;
const CHUNK = 500;
const t0 = performance.now();
for(let t=0;t<MS;t+=CHUNK) eng.postMessage({ cmd:'tick', steps:CHUNK });
eng.postMessage({ cmd:'getWeights' });
await new Promise(res => {
  const poll = () => learned ? res() : setTimeout(poll, 50);
  poll();
});
const wall = (performance.now()-t0)/1000;
eng.terminate();

// Dynamics first: mean rate against the reference's, and the per-neuron count correlation.
// These do not depend on plasticity being on.
{
  const refJson = path.join(HERE, PREFIX + '-ref.json');
  if(fs.existsSync(refJson)){
    const ref = JSON.parse(fs.readFileSync(refJson, 'utf8'));
    const rc = ref.counts || [];
    const childHz = childSpikes / n / SECONDS;
    let mx = 0, my = 0;
    for(let i=0;i<n;i++){ mx += childCounts[i]; my += rc[i] || 0; }
    mx /= n; my /= n;
    let sxy = 0, sxx = 0, syy = 0;
    for(let i=0;i<n;i++){ const dx = childCounts[i] - mx, dy = (rc[i] || 0) - my;
      sxy += dx*dy; sxx += dx*dx; syy += dy*dy; }
    const cr = sxy / Math.sqrt(Math.max(1e-12, sxx*syy));
    console.log(`  spikes: child ${childHz.toFixed(3)} Hz vs reference ${(+ref.rateHz).toFixed(3)} Hz ` +
      `(ratio ${(childHz/ref.rateHz).toFixed(4)}), per-neuron count r = ${cr.toFixed(4)}`);
  }
}
const refPath = path.join(HERE, PREFIX + '-ref-w.bin');
if(!fs.existsSync(refPath)){
  console.log('no reference weights at ' + refPath + '; ran only');
  process.exit(0);
}
const rb = fs.readFileSync(refPath);
const ref = new Float32Array(rb.buffer.slice(rb.byteOffset, rb.byteOffset + rb.byteLength));
const k = Math.min(ref.length, learned.length);
let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, maxAbs = 0, exact = 0;
for(let i=0;i<k;i++){
  const x = ref[i], y = learned[i];
  sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y;
  const dd = Math.abs(x - y);
  if(dd > maxAbs) maxAbs = dd;
  if(dd === 0) exact++;
}
const cov = sxy/k - (sx/k)*(sy/k);
const r = cov/Math.sqrt(Math.max(1e-30,
  (sxx/k - (sx/k)**2)*(syy/k - (sy/k)**2)));
console.log(`parity ${PREFIX}: ${k} synapses, ${SECONDS}s, cons=${CONS}, stp=${STP}, scale=${SCALE}, commit=${COMMIT}`);
console.log(`  weight r = ${r.toFixed(6)}, max |diff| = ${maxAbs.toFixed(6)}, ` +
  `exact ${(100*exact/k).toFixed(1)}%`);
console.log(`  child wall ${wall.toFixed(1)}s`);
// The child's own weights, so two child runs can be differenced against each other.
// A term whose effect is smaller than the normal engine-to-engine divergence is invisible in the correlation above: synaptic scaling moves excitatory weights about 0.2% over ten seconds while the largest difference between engines is over 2, so the only way to see it is child against child.
const outW = path.join(HERE, PREFIX + '-cuda-w.bin');
fs.writeFileSync(outW, Buffer.from(learned.buffer, learned.byteOffset, learned.byteLength));
console.log(`  wrote ${path.basename(outW)}`);

// Measured set points, when the run asked for them.
// Compared against the reference engine's own targets rather than only checked for existing: a per-neuron target that is present but wrong is the failure worth catching.
if(learnedRho){
  const outR = path.join(HERE, PREFIX + '-cuda-rho.bin');
  fs.writeFileSync(outR, Buffer.from(learnedRho.buffer, learnedRho.byteOffset,
    learnedRho.byteLength));
  const rhoFile = path.join(HERE, PREFIX + '-ref-rho.bin');
  let band = 0, worst = 0, meanC = 0, meanR = 0;
  for(const v of learnedRho){ if(v >= 0.25 && v <= 30) band++; meanC += v; }
  meanC /= learnedRho.length;
  if(fs.existsSync(rhoFile)){
    const rb = fs.readFileSync(rhoFile);
    const ref = new Float32Array(rb.buffer, rb.byteOffset, rb.byteLength/4);
    const kk = Math.min(ref.length, learnedRho.length);
    for(let i=0;i<kk;i++){
      meanR += ref[i];
      const d = Math.abs(ref[i] - learnedRho[i]);
      if(d > worst) worst = d;
    }
    meanR /= kk;
    console.log(`  set points: child mean ${meanC.toFixed(3)} Hz vs reference ` +
      `${meanR.toFixed(3)} Hz, max |diff| ${worst.toFixed(3)}, ` +
      `${band}/${learnedRho.length} inside the 0.25-30 Hz band`);
  } else {
    console.log(`  set points: child mean ${meanC.toFixed(3)} Hz, ` +
      `${band}/${learnedRho.length} inside the 0.25-30 Hz band ` +
      `(no ${path.basename(rhoFile)} to compare against)`);
  }
}

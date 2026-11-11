// Does the CUDA child run neuromodulation as the reference does (MODEL.md 4): the modulator field raised by a time-course source or a releasing population, the Hebbian terms scaled by it or routed through an eligibility trace paid in at the sweeps, the synaptic gain and the current?
// The cases of src/mod.test.mjs on both engines through the same input map, the weight or the spike counts read back and compared to the reference's.
//
//   node cuda/modtest.mjs [exePath]
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ok, report } from '../tools/harness.mjs';
import { PROTOCOL } from '../src/protocol.js';
import { initState } from '../src/rand.js';
import { engineTypes, NEURON_TYPES, engineConfig } from '../src/nodes.js';
import { CudaEngine } from '../host/cudaengine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXE = process.argv[2] || path.join(HERE, 'engine.exe');
if(!fs.existsSync(EXE)){ console.error('modtest: no engine at ' + EXE); process.exit(1); }

const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('../src/simworker.js');
const handle = globalThis.onmessage;
const refSend = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };
async function child(){
  const eng = new CudaEngine(EXE, () => {});
  const inbox = [];
  eng.on('message', d => inbox.push(d));
  const next = async cmd => { for(let spin = 0; spin < 20000; spin++){ const k = inbox.findIndex(r => r.cmd === cmd); if(k >= 0) return inbox.splice(0, k + 1)[k]; const e = inbox.find(r => r.cmd === 'error'); if(e) throw new Error(e.message); await new Promise(r => setTimeout(r, 5)); } throw new Error('no ' + cmd); };
  eng.postMessage({ cmd:'hello' }); const hello = await next('hello');
  return { hello, send:(d, reply) => { eng.postMessage(d); return reply ? next(reply) : null; }, close:() => eng.terminate() };
}

const RS = NEURON_TYPES.findIndex(t => t.key === 'RS07');
const AP = 0.1, TAU = 20, W0 = 0.5;
function init(extra = {}, { w = W0, synapse = true } = {}){
  const ntype = Uint8Array.from([RS, RS]), bias = new Float32Array(2);
  const net = { count:2, ntype, bias,
    preStart:Int32Array.from(synapse ? [0, 1, 1] : [0, 0, 0]), post:Int32Array.from(synapse ? [1] : []),
    w:Float32Array.from(synapse ? [w] : []), delay:Uint8Array.from(synapse ? [1] : []), pmask:Uint8Array.from(synapse ? [1] : []),
    syn:0, tauE:3, tauI:8, psc:0, plast:1, aP:AP, aM:AP, tauS:TAU, tauM:TAU, wmax:30, wdep:0, stdpAt:0, iEta:0, iRho:5, refrac:2, vmin:-90,
    modTau:200, protocols:[], ...extra };
  return { cmd:'init', protocol:PROTOCOL, ...engineConfig(net), count:2, ntype, bias,
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, pmask:net.pmask,
    inputs:[{ chStart:Int32Array.from([0, 1, 2]), chIdx:Int32Array.from([0, 1]), chW:Float32Array.from([1, 1]), amp:20000 }],
    types:engineTypes(), seed:3, ...initState(3, ntype, NEURON_TYPES) };
}
const gains = (t, tPre, tPost) => Float32Array.from([t === tPre ? 1 : 0, t === tPost ? 1 : 0]);
function refRun(c, tPre, tPost, steps, opts){
  refSend(init(c, opts));
  const spikes = [0, 0];
  for(let t = 0; t < steps; t++){
    refSend({ cmd:'inputFrame', mi:0, gains:gains(t, tPre, tPost) });
    const st = refSend({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state');
    spikes[0] += st.fired[0]; spikes[1] += st.fired[1];
  }
  const wm = refSend({ cmd:'getWeights' }).find(x => x.cmd === 'weights');
  return { w:wm.w.length ? wm.w[0] : NaN, spikes };
}
async function childRun(c, tPre, tPost, steps, opts){
  const ch = await child();
  ch.send(init(c, opts));
  const spikes = [0, 0];
  // a few steps per tick, so the frame's cadence is not the reference's one step
  for(let t = 0; t < steps; t += 4){
    const k = Math.min(4, steps - t);
    for(let q = 0; q < k; q++) if(t + q === tPre || t + q === tPost) ch.send({ cmd:'inputFrame', mi:0, gains:gains(t + q, tPre, tPost) });
    // the input map holds across a tick, so a pairing step is run alone
    if((tPre >= t && tPre < t + k) || (tPost >= t && tPost < t + k)){
      for(let q = 0; q < k; q++){ ch.send({ cmd:'inputFrame', mi:0, gains:gains(t + q, tPre, tPost) }); const st = await ch.send({ cmd:'tick', steps:1 }, 'state'); spikes[0] += st.fired[0]; spikes[1] += st.fired[1]; }
    } else {
      ch.send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([0, 0]) });
      const st = await ch.send({ cmd:'tick', steps:k }, 'state'); spikes[0] += st.fired[0]; spikes[1] += st.fired[1];
    }
  }
  const wm = await ch.send({ cmd:'getWeights' }, 'weights');
  ch.close();
  return { w:wm.w.length ? wm.w[0] : NaN, spikes, terms:ch.hello.terms || [] };
}
const near = (a, b, tol = 2e-5) => Math.abs(a - b) < tol;
const src = (level, extra = {}) => ({ idx:Uint32Array.from([1]), amp:level, mode:0, period:100, width:10, t0:0, duration:0, ...extra });
const rel = { modGrp:Uint8Array.from([1, 0]), modTgt:Uint16Array.from([0, 1]), modK:Float32Array.from([0, 1000]) };
const cases = [
  ['the gain form with base 1 and no source is the plain rule', { modPlast:1, modBase:1 }, 2, 7, 40],
  ['the gain form scales the pairing with the field a constant source holds', { modPlast:1, modBase:0, modSrc:[src(20)] }, 2, 7, 40],
  ['the eligibility form pays a decayed trace in at the sweeps after a pulse', { modPlast:2, modBase:0, modStep:10, eligTau:1000, modSrc:[src(20, { mode:1, period:100000, width:10, t0:20 })] }, 2, 7, 40],
  ['the eligibility form with base 1 pays the pairing in over the sweeps', { modPlast:2, modBase:1, modStep:10, eligTau:1000 }, 2, 7, 200],
  ['a releasing cell raises the field at its target the step after its spike', { modPlast:1, modBase:0, ...rel }, 2, 7, 40],
];
let terms = null;
for(const [label, c, tPre, tPost, steps] of cases){
  const r = refRun(c, tPre, tPost, steps), ch = await childRun(c, tPre, tPost, steps); terms = ch.terms;
  ok(label + ' (the pairing moved the weight)', Math.abs(r.w - W0) > 1e-4 || label.includes('plain'), 'reference ' + r.w);
  ok(label + ' (child agrees)', near(ch.w, r.w), 'child ' + ch.w + ' reference ' + r.w);
}
// the synaptic gain and the current: spike counts, the reference's against the child's
{
  const pulse = { idx:Uint32Array.from([0]), amp:3000, mode:1, period:20, width:2, t0:0, duration:0 };
  const gained = { plast:0, protocols:[pulse], modDrive:1, modSrc:[src(2)] };
  // the child does not run a pulse-train stimulus; the pre cell is fired through the input map every 20 steps instead
  const refG = (() => { refSend(init({ plast:0, modDrive:1, modSrc:[src(2)] }, { w:1500 })); const sp = [0, 0];
    for(let t = 0; t < 400; t++){ refSend({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t % 20 === 0 ? 1 : 0, 0]) }); const st = refSend({ cmd:'tick', steps:1 }).find(x => x.cmd === 'state'); sp[0] += st.fired[0]; sp[1] += st.fired[1]; }
    return sp; })();
  const chG = await (async () => { const ch = await child(); ch.send(init({ plast:0, modDrive:1, modSrc:[src(2)] }, { w:1500 })); const sp = [0, 0];
    for(let t = 0; t < 400; t++){ ch.send({ cmd:'inputFrame', mi:0, gains:Float32Array.from([t % 20 === 0 ? 1 : 0, 0]) }); const st = await ch.send({ cmd:'tick', steps:1 }, 'state'); sp[0] += st.fired[0]; sp[1] += st.fired[1]; }
    ch.close(); return sp; })();
  ok('a synaptic gain per level makes the post cell fire on the reference', refG[0] >= 19 && refG[1] > 5, refG.join(' and ') + ' spikes');
  ok('the child fires the same counts', chG[0] === refG[0] && Math.abs(chG[1] - refG[1]) <= 1, 'child ' + chG.join(' and ') + ' against ' + refG.join(' and '));
  const refC = refRun({ plast:0, modCur:100, modSrc:[src(1)] }, -1, -1, 1000, { synapse:false }).spikes;
  const chC = await childRun({ plast:0, modCur:100, modSrc:[src(1)] }, -1, -1, 1000, { synapse:false });
  ok('a current per level drives a cell with no synapses on the reference', refC[1] > 5, refC[1] + ' spikes');
  ok('the child fires the same count', Math.abs(chC.spikes[1] - refC[1]) <= 1, 'child ' + chC.spikes[1] + ' against ' + refC[1]);
}
ok('the child names the neuromodulation terms in its hello', terms && terms.includes('mod') && terms.includes('elig'), String(terms));
report('cuda modtest');

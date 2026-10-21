// Under measured set points the global target has no say: every homeostatic term, the heterosynaptic gate included, uses the rate each cell measured for itself.
// So two runs that differ only in iRho learn identical weights once calibration has set the per-cell targets; at rhoMode 0 the same two runs differ, which is what shows the check can tell.
// Reference engine, in this process; the CUDA parity gate (cuda/parity.mjs on a rhoMode dump) and the battery's cross-engine group hold the other two to the reference.
// Run: node src/setpoint.test.mjs
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ok, report, inProcessWorker } from '../tools/harness.mjs';
import { computeNode, setResolution, setMaxSyn, wireConnect, assembleConnect } from './nodes.js';
import { initMessage, engineInputs } from './document.js';
import { loadScene } from '../tools/scene.mjs';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
inProcessWorker(wireConnect, assembleConnect);
setMaxSyn(Math.floor(os.totalmem()*0.5/13));
setResolution(0.05);

// the reference engine as the page drives it, one import, an init per run
const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('./simworker.js');
const handle = globalThis.onmessage;
const send = d => { replies.length = 0; handle({ data:d }); return replies.slice(); };

// the settings live on the checkpoint node; the test only varies them there
async function learnedWeights(rhoMode, iRho){
  const over = [`checkpoint:*:plast=1`, `checkpoint:*:het=0.3`, `checkpoint:*:trip=0.002`, `checkpoint:*:rhoMode=${rhoMode}`,
    `checkpoint:*:calS=1`, `checkpoint:*:iRho=${iRho}`, `checkpoint:*:scale=0`];
  const { ed } = await loadScene('balanced random net', { root:ROOT, overrides:over });
  const cp = ed.nodes.find(n => n.type === 'checkpoint');
  const net = await computeNode(cp, id => ed.nodes.find(n => n.id === id));
  // cloned: in this process the engine takes the arrays themselves, and the wiring memo hands every run the same net, so without a copy the second run would start from the first run's learned weights
  const init = structuredClone({ ...initMessage(net), inputs:engineInputs({ inputMaps:[] }) });
  send(init);
  let spikes = 0;
  for(let t = 0; t < 2500; t += 50){ const r = send({ cmd:'tick', steps:50 }); spikes += r.find(x => x.cmd === 'state').spikes; }
  const w = send({ cmd:'getWeights' }).find(x => x.cmd === 'weights').w;
  return { w:Float32Array.from(w), spikes, cells:net.count };
}
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const diff = (a, b) => { let n = 0; for(let i = 0; i < a.length; i++) if(a[i] !== b[i]) n++; return n; };

const m2 = await learnedWeights(1, 2), m20 = await learnedWeights(1, 20);
ok('the network fires and learns', m2.spikes > 0 && m2.cells > 500, m2.spikes + ' spikes, ' + m2.cells + ' cells');
ok('under measured set points, iRho 2 and iRho 20 learn identical weights', same(m2.w, m20.w), diff(m2.w, m20.w) + ' of ' + m2.w.length + ' weights differ');
const g2 = await learnedWeights(0, 2), g20 = await learnedWeights(0, 20);
ok('with the fixed target, the same two runs differ, so the check can tell', !same(g2.w, g20.w), diff(g2.w, g20.w) + ' weights differ');
report('setpoint');

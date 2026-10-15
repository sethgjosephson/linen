// Records the byte stream the CUDA engine child receives for one run of a scene (hello, init, warm-up and ticks), so the child can be replayed on its own, with the same input every time:
//
//   node cuda/record.mjs "balanced random net" frames.bin [seconds] [--over SPEC] [--res R]
//   LINEN_PROFILE=1 cuda/engine.exe < frames.bin > NUL      the time per kind of launch, to stderr
//
// The run that records is a live run on the engine, timed the way tools/enginebench.mjs times one, and its speed is printed.
// Settings are node settings, given as tools/linen.mjs takes them.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { computeNode, setResolution, setMaxSyn, wireConnect, assembleConnect } from '../src/nodes.js';
import { initMessage, engineInputs } from '../src/document.js';
import { inProcessWorker } from '../tools/harness.mjs';
import { loadScene } from '../tools/scene.mjs';
import { makeInbox, timeEngine } from '../tools/benchcore.js';
import { CudaEngine } from '../host/cudaengine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url)), ROOT = path.join(HERE, '..');
const argv = process.argv.slice(2);
const target = argv.shift(), out = argv.shift();
const fail = s => { console.error('record: ' + s); process.exit(1); };
if(!target || !out) fail('name a scene and the file to write');
let seconds = 3, res = 1;
const over = [];
if(argv[0] && !argv[0].startsWith('--')) seconds = +argv.shift();
for(let i = 0; i < argv.length; i++){
  const a = argv[i], next = () => { const v = argv[++i]; if(v === undefined) fail(a + ' needs a value'); return v; };
  if(a === '--over') over.push(next());
  else if(a === '--res') res = +next();
  else fail('unknown option ' + a);
}
const exe = path.join(HERE, 'engine.exe');
if(!fs.existsSync(exe)) fail('cuda/engine.exe is not built');

inProcessWorker(wireConnect, assembleConnect);
setMaxSyn(Math.floor(os.totalmem()*0.5/13));
let ed;
try { ({ ed } = await loadScene(target, { root:ROOT, overrides:over.flatMap(s => s.split(';')).map(s => s.trim()).filter(Boolean), say:console.log })); }
catch(e){ fail(e.message); }
setResolution(res);
const cp = ed.nodes.find(n => n.type === 'checkpoint');
if(!cp) fail('the scene has no checkpoint node');
const net = await computeNode(cp, id => ed.nodes.find(n => n.id === id));
console.log(net.count.toLocaleString() + ' cells, ' + net.synCount.toLocaleString() + ' synapses, plasticity ' + (cp.params.plast ? 'on' : 'off'));
if((net.inputMaps || []).length) fail('the scene has input maps, which this tool does not drive');
const init = { ...initMessage(net), inputs:engineInputs({ inputMaps:[] }) };

// every byte the adapter writes to the child goes to the file as well
const engine = new CudaEngine(exe, () => {});
const fd = fs.openSync(out, 'w');
const write = engine.proc.stdin.write.bind(engine.proc.stdin);
engine.proc.stdin.write = (chunk, ...rest) => { fs.writeSync(fd, Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)); return write(chunk, ...rest); };
const inbox = makeInbox();
engine.on('message', d => inbox.push(d));
let r;
try { r = await timeEngine({ post:d => engine.postMessage(d), next:inbox.next, init, warmMs:500, ms:Math.round(seconds*1000), tick:100, now:() => performance.now()/1000 }); }
catch(e){ fs.closeSync(fd); engine.terminate(); fail(e.message); }
fs.closeSync(fd);
console.log('wrote ' + out + ' (' + (fs.statSync(out).size/1e6).toFixed(0) + ' MB): 0.5 s of warm-up and ' + seconds + ' s at 100 ms a tick; the live run went '
  + r.realtime.toFixed(2) + 'x realtime at ' + r.rate.toFixed(2) + ' Hz mean');
engine.terminate();
process.exit(0);

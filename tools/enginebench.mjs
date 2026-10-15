// Engine speed on one network: a scene is computed once and the same init message is timed on the reference engine, the CUDA engine and the WebGPU engine, one after another, with the same loop (tools/benchcore.js).
// The network's size, its in-degree and its plasticity are settings on its nodes, given with --over as tools/linen.mjs takes them.
//
//   node tools/enginebench.mjs "balanced random net" --over "sphere:*:radius=600"
//   node tools/enginebench.mjs "balanced random net" --over "checkpoint:*:plast=1" --engines cuda,webgpu
//
//   --over SPEC      settings on nodes, type:tag-or-name:key=value (repeatable; ; separates several)
//   --res R          resolution, 0 to 1 (default 1)
//   --engines LIST   any of cpu, cuda, webgpu (default all three)
//   --seconds S      simulated seconds timed per engine (default 5)
//   --cpu-seconds S  the same for the reference engine, which is the slow one (default 1)
//   --warm MS        simulated milliseconds run before the timing starts (default 1000)
//   --tick MS        simulated milliseconds per tick message, the same for every engine (default 100)
//   --serve URL      a serve.py of this checkout, for the WebGPU page (default http://127.0.0.1:8174)
//   --project DIR    that server's project folder; the init message is written to bench/init.bin in it
//   --chrome PATH    the Chrome to run the page in, headless (default the usual Windows install)
//   --json FILE      append the result as one JSON line
//
// The WebGPU engine runs in headless Chrome on this machine's GPU (--enable-unsafe-webgpu), through tools/enginebench.html.
// Run nothing else on the GPU meanwhile: an open linen tab is a second engine on the same card, and contention reads as a slow engine.
// Every engine's mean rate is printed beside its speed, so a run where one engine simulated something else shows.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { computeNode, setResolution, setMaxSyn, wireConnect, assembleConnect } from '../src/nodes.js';
import { initMessage, engineInputs } from '../src/document.js';
import { inProcessWorker } from './harness.mjs';
import { loadScene } from './scene.mjs';
import { packMessage, makeInbox, timeEngine } from './benchcore.js';

const HERE = path.dirname(fileURLToPath(import.meta.url)), ROOT = path.join(HERE, '..');
const argv = process.argv.slice(2);
const target = argv.shift();
const opt = { over:[], res:1, engines:['cpu', 'cuda', 'webgpu'], seconds:5, cpuSeconds:1, warm:1000, tick:100,
  serve:'http://127.0.0.1:8174', project:'', chrome:'C:/Program Files/Google/Chrome/Application/chrome.exe', json:'' };
const fail = s => { console.error('enginebench: ' + s); process.exit(1); };
if(!target || target.startsWith('--')) fail('name a scene file or a scene first');
for(let i = 0; i < argv.length; i++){
  const a = argv[i], next = () => { const v = argv[++i]; if(v === undefined) fail(a + ' needs a value'); return v; };
  if(a === '--over') opt.over.push(next());
  else if(a === '--res') opt.res = +next();
  else if(a === '--engines') opt.engines = next().split(',').map(s => s.trim());
  else if(a === '--seconds') opt.seconds = +next();
  else if(a === '--cpu-seconds') opt.cpuSeconds = +next();
  else if(a === '--warm') opt.warm = +next();
  else if(a === '--tick') opt.tick = Math.max(1, Math.round(+next()));
  else if(a === '--serve') opt.serve = next().replace(/\/$/, '');
  else if(a === '--project') opt.project = next();
  else if(a === '--chrome') opt.chrome = next();
  else if(a === '--json') opt.json = next();
  else fail('unknown option ' + a);
}
for(const e of opt.engines) if(!['cpu', 'cuda', 'webgpu'].includes(e)) fail('--engines takes cpu, cuda and webgpu');
if(opt.engines.includes('webgpu') && !opt.project) fail('the WebGPU engine needs --project, the folder the server at ' + opt.serve + ' serves');

// ---- one network
inProcessWorker(wireConnect, assembleConnect);
setMaxSyn(Math.floor(os.totalmem()*0.5/13));
const overrides = opt.over.flatMap(s => s.split(';')).map(s => s.trim()).filter(Boolean);
let ed, sceneName, applied;
try { ({ ed, sceneName, applied } = await loadScene(target, { root:ROOT, overrides, say:console.log })); }
catch(e){ fail(e.message); }
setResolution(opt.res);
const cp = ed.nodes.find(n => n.type === 'checkpoint');
if(!cp) fail('the scene has no checkpoint node');
const c0 = Date.now();
const net = await computeNode(cp, id => ed.nodes.find(n => n.id === id));
const inDegree = net.synCount/net.count;
console.log(sceneName + ': ' + net.count.toLocaleString() + ' cells, ' + net.synCount.toLocaleString() + ' synapses (mean in-degree '
  + inDegree.toFixed(0) + '), plasticity ' + (cp.params.plast ? 'on' : 'off') + ', computed in ' + ((Date.now() - c0)/1000).toFixed(1) + ' s');
if((net.inputMaps || []).length) fail('the scene has input maps, which this bench does not drive; pick a scene driven by stimulus nodes');
const init = { ...initMessage(net), inputs:engineInputs({ inputMaps:[] }) };

// ---- the engines
const results = [];
const time = async (name, run) => {
  try {
    const r = await run();
    results.push({ name, ...r });
    console.log(name.padEnd(7) + ' ' + r.realtime.toFixed(2).padStart(8) + 'x realtime  ' + (r.deliveriesPerS/1e6).toFixed(0).padStart(7)
      + ' M deliveries/s  ' + r.rate.toFixed(2).padStart(6) + ' Hz mean  (' + (r.simMs/1000) + ' s simulated in ' + r.wallS.toFixed(2) + ' s; init and warm-up ' + r.initS.toFixed(1) + ' s)');
  } catch(e){
    results.push({ name, error:e.message });
    console.log(name.padEnd(7) + ' failed: ' + e.message);
  }
};
const nowS = () => performance.now()/1000;
const timing = seconds => ({ warmMs:opt.warm, ms:Math.round(seconds*1000), tick:opt.tick, now:nowS });

// WebGPU first, while this process holds the least: the init message goes to the page as a file in the server's project folder.
if(opt.engines.includes('webgpu')) await time('webgpu', async () => {
  const dir = path.join(path.resolve(opt.project), 'bench');
  fs.mkdirSync(dir, { recursive:true });
  fs.writeFileSync(path.join(dir, 'init.bin'), packMessage(init));
  try { return await runPage(); } finally { fs.rmSync(path.join(dir, 'init.bin'), { force:true }); }
});
if(opt.engines.includes('cuda')) await time('cuda', async () => {
  const exe = path.join(ROOT, 'cuda', 'engine.exe');
  if(!fs.existsSync(exe)) throw new Error('cuda/engine.exe is not built');
  const { CudaEngine } = await import('../host/cudaengine.mjs');
  const engine = new CudaEngine(exe, () => {});
  const inbox = makeInbox();
  engine.on('message', d => inbox.push(d));
  try { return await timeEngine({ post:d => engine.postMessage(d), next:inbox.next, init, ...timing(opt.seconds) }); }
  finally { engine.terminate(); }
});
if(opt.engines.includes('cpu')) await time('cpu', async () => {
  const inbox = makeInbox();
  globalThis.postMessage = d => inbox.push(d);
  globalThis.onmessage = null;                       // the worker assigns its handler to the global
  await import('../src/simworker.js');
  const handle = globalThis.onmessage;
  return timeEngine({ post:d => handle({ data:d }), next:inbox.next, init, ...timing(opt.cpuSeconds) });
});

if(opt.json){
  let commit = '';
  try { commit = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd:ROOT, encoding:'utf8' }).trim(); } catch(e){}
  fs.appendFileSync(opt.json, JSON.stringify({ scene:sceneName, overrides:applied.map(a => a.term), res:opt.res,
    cells:net.count, synapses:net.synCount, inDegree, plast:cp.params.plast, tick:opt.tick, commit,
    date:new Date().toISOString(), results }) + '\n');
}
process.exit(0);

// The page in headless Chrome, read over the DevTools protocol until it says it is done.
async function runPage(){
  const port = 9460 + Math.floor(Math.random()*30);
  const prof = fs.mkdtempSync(path.join(os.tmpdir(), 'linen-bench-'));
  const chrome = spawn(opt.chrome, ['--headless=new', '--remote-debugging-port=' + port, '--user-data-dir=' + prof,
    '--enable-unsafe-webgpu', '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
    '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio:'ignore' });
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  try {
    const url = opt.serve + '/tools/enginebench.html?dump=/project/file/bench/init.bin&tick=' + opt.tick
      + '&warm=' + opt.warm + '&ms=' + Math.round(opt.seconds*1000);
    let pg = null;
    for(let i = 0; i < 100 && !pg; i++){
      await sleep(200);
      try { pg = await (await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`, { method:'PUT' })).json(); } catch(e){}
    }
    if(!pg) throw new Error('headless Chrome did not start');
    const ws = new WebSocket(pg.webSocketDebuggerUrl);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error('no DevTools connection')); });
    let id = 0;
    const pending = new Map();
    ws.onmessage = e => { const m = JSON.parse(e.data); if(pending.has(m.id)){ pending.get(m.id)(m); pending.delete(m.id); } };
    const evaluate = expr => new Promise(r => { const k = ++id; pending.set(k, r); ws.send(JSON.stringify({ id:k, method:'Runtime.evaluate', params:{ expression:expr, returnByValue:true } })); });
    const limit = Date.now() + 30*60*1000;
    for(;;){
      await sleep(500);
      const m = await evaluate('JSON.stringify(window.bench || null)');
      const b = m.result && m.result.result && m.result.result.value ? JSON.parse(m.result.result.value) : null;
      if(b && b.done){
        ws.close();
        if(b.error) throw new Error(b.error);
        return b.result;
      }
      if(Date.now() > limit) throw new Error('the page did not finish in 30 minutes');
    }
  } finally {
    chrome.kill();
    await sleep(500);
    try { fs.rmSync(prof, { recursive:true, force:true }); } catch(e){}
  }
}

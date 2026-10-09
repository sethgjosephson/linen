// Native engine host: the reference engine behind a WebSocket, speaking the ENGINE.md protocol unchanged.
//
//   node host/host.mjs [port]     (default 8801)
//
// One engine per connection: connecting spawns a worker running src/simworker.js through the shim in engineworker.mjs, closing terminates it.
// Messages cross the socket in the framing src/frame.js defines, so the browser side needs only an adapter that looks like a Worker.
//
// The WebSocket layer is written out here rather than taken from a package so the host has no dependencies: an HTTP upgrade handshake and RFC 6455 frames, binary only, with ping and fragmentation handled.
import http from 'node:http';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { encodeMsg, decodeMsg } from '../src/frame.js';
import { IORuntime } from '../src/io.js';
import { buildWeightsFile, buildPlasticFile } from '../src/brain.js';
import { setVoice, setGlyphAtlas } from '../src/curriculum.js';
import { CudaEngine } from './cudaengine.mjs';

// --cuda [exe] runs connections on the CUDA engine child instead of the
// reference worker; the browser cannot tell the difference.
const CUDA = process.argv.includes('--cuda');
const CUDA_EXE = (() => {
  const i = process.argv.indexOf('--cuda');
  const nxt = i >= 0 ? process.argv[i+1] : null;
  // A following flag is the next option, not a path to an executable: "--cuda --project D:/x" would otherwise try to run "--project".
  return nxt && !/^\d+$/.test(nxt) && !nxt.startsWith('--') ? nxt
    : new URL('../cuda/engine.exe', import.meta.url).pathname.replace(/^\/(\w:)/, '$1');
})();

// --project PATH puts checkpoints in the folder the page writes into; serve.py always passes it.
// Without it checkpoints go to host/runs.
const PROJECT = (() => {
  const i = process.argv.indexOf('--project');
  const nxt = i >= 0 ? process.argv[i+1] : null;
  return nxt && !nxt.startsWith('--') ? nxt : null;
})();
const RUNS = PROJECT ? path.join(PROJECT, 'runs')
  : path.join(path.dirname(fileURLToPath(import.meta.url)), 'runs');

// Live runs by save directory, so a later connection can attach as a viewer of a run whose own client is gone.
const LIVE = new Map();


const PORT = +((process.argv.filter(x => /^\d+$/.test(x))[0]) || 8801);
// when this process started, against which the source files are checked each time a run is handed over (see the 'net' message)
const STARTED = Date.now();
// The files this process actually runs: its own module graph, followed through relative imports from this file, so an edit to the page's code does not count against a host that never loads it.
function hostSources(){
  const seen = new Set(), stack = [fileURLToPath(import.meta.url)];
  while(stack.length){
    const f = stack.pop();
    if(seen.has(f)) continue;
    seen.add(f);
    let src = '';
    try { src = fs.readFileSync(f, 'utf8'); } catch(e){ continue; }
    const re = /(?:from\s*|import\s*\(\s*)['"](\.[^'"]+)['"]/g;
    let m;
    while((m = re.exec(src))) stack.push(path.resolve(path.dirname(f), m[1]));
  }
  return [...seen];
}
function newerSource(since){
  for(const f of hostSources()){
    try { if(fs.statSync(f).mtimeMs > since) return path.relative(path.join(path.dirname(fileURLToPath(import.meta.url)), '..'), f).split(path.sep).join('/'); }
    catch(e){}
  }
  return null;
}
const MAGIC = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

// A run here can be a day long, so the host outliving a disconnection race matters more than it would in a request server.
// Writes to a peer that has just gone raise EPIPE or ECONNRESET asynchronously, on whichever stream noticed first, and an unhandled 'error' event ends the process, and with it the run that was wiring.
// Anything else is a real fault and still stops the host, because a host that swallows every error is a host whose results cannot be trusted.
process.on('uncaughtException', e => {
  if(e && (e.code === 'EPIPE' || e.code === 'ECONNRESET')){
    log('ignored ' + e.code + ' from a peer that went away');
    return;
  }
  log('fatal: ' + (e && e.stack || e));
  process.exit(1);
});
// The stimulus clock runs here when the pump is on, so the host needs its own copy; the curriculum only reads it when a run asks for voice mixing.
try {
  const vp = path.join(path.dirname(fileURLToPath(import.meta.url)),
    '..', 'speech', 'clips', 'voice.json');
  if(fs.existsSync(vp)){
    const bank = JSON.parse(fs.readFileSync(vp, 'utf8'));
    setVoice(bank);
    log('voice bank loaded: ' + Object.keys(bank.items).length + ' items');
  }
} catch(e){ log('voice bank unreadable: ' + e.message); }


function frame(payload){                    // server frame: binary, unmasked
  const n = payload.byteLength;
  let head;
  if(n < 126) head = Buffer.from([0x82, n]);
  else if(n < 65536){ head = Buffer.alloc(4); head[0] = 0x82; head[1] = 126; head.writeUInt16BE(n, 2); }
  else { head = Buffer.alloc(10); head[0] = 0x82; head[1] = 127; head.writeBigUInt64BE(BigInt(n), 2); }
  return Buffer.concat([head, Buffer.from(payload)]);
}

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/plain' });
  res.end('linen engine host; connect by WebSocket\n');
});

server.on('upgrade', (req, socket) => {
  const key = req.headers['sec-websocket-key'];
  if(!key){ socket.destroy(); return; }
  const accept = crypto.createHash('sha1').update(key + MAGIC).digest('base64');
  socket.write('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n' +
    'Connection: Upgrade\r\nSec-WebSocket-Accept: ' + accept + '\r\n\r\n');
  socket.setNoDelay(true);

  const engine = CUDA
    ? new CudaEngine(CUDA_EXE, (...a) => log(...a))
    : new Worker(new URL('./engineworker.mjs', import.meta.url));
  let msgs = 0, replies = 0;

  // Messages carrying __host are consumed here and never reach the engine; everything else is the ENGINE.md protocol passed through.
  // With the pump on, the host drives the tick loop and the stimulus clock itself, so a suspended browser tab does not stall a run: the page becomes a viewer that reconnects to reality whenever it wakes.
  // ENGINE.md section 7 documents the channel.
  const ctl = { pump:false, steps:50, simMs:0, io:null,
    ats:[], save:null, pendingSave:false, netShim:null, lastLog:0,
    pmask:null,
    detach:false, untilMs:0, gone:false, dir:'', watchers:new Set() };
  let watching = null;   // set when this connection is a viewer of another run
  const forward = d => {
    // a frozen client stops reading; simulate on and let its view lapse rather than buffering gigabytes of state frames
    if(!ctl.gone && socket.writableLength < (1 << 20)) socket.write(frame(encodeMsg(d)));
  };
  const tick = () => engine.postMessage({ cmd:'tick', steps:ctl.steps });
  const onCtl = m => {
    if(m.__host === 'clock'){ ctl.simMs = m.simMs || 0; if(ctl.io) ctl.io.last = -1e9; }
    else if(m.__host === 'maps'){
      ctl.io = new IORuntime(e => log('io error:', e));
      ctl.io.attach({ postMessage: d => engine.postMessage(d) }, m.maps, m.seed);
      ctl.io.last = -1e9;
    }
    else if(m.__host === 'at') ctl.ats.push({ simMs:m.simMs, msg:m.msg });
    else if(m.__host === 'assets'){
      // canonical stimulus rasters, browser-built since only it has a canvas; rendering here samples these (curriculum.js atlas path).
      // Module state: concurrent runs share the last atlas set, which is safe while they use the same lesson sets.
      if(m.atlas){
        setGlyphAtlas(m.atlas);
        log('stimulus atlas: ' + Object.keys(m.atlas.items).length +
          ' items (' + (m.atlas.kind || '?') + ')');
      }
    }
    else if(m.__host === 'net'){
      // A host keeps the code it loaded, so a run on stale code is refused at the moment it is asked for.
      const stale = newerSource(STARTED);
      if(stale){
        const msg = 'engine host is running code older than ' + stale +
          ' (host started ' + new Date(STARTED).toISOString() + '); stop this host and press START again, ' +
          'or point the checkpoint at another port';
        log(msg);
        try { socket.write(frame(encodeMsg({ cmd:'error', message:msg }))); } catch(_){ }
        // the connection ends here, before the detach and pump messages that follow a net hand-over can turn a refused run into a headless one pumping on the card
        ctl.refused = true;
        setTimeout(() => close('refused'), 300);
        return;
      }
      ctl.netShim = { count:m.count, synCount:m.synCount,
        graph:m.graph, curriculum:m.curriculum || null };
    }
    else if(m.__host === 'saveNow'){
      // While the pump is on the request is taken at the next state reply, so it joins the one tick chain.
      // While it is paused nothing is in flight and there is no chain to join, which is the case that matters most: pause, save, leave.
      if(!ctl.save || !ctl.netShim)
        log('save requested but this run has no save directory');
      else if(ctl.pendingSave) log('save already in progress; request ignored');
      else if(ctl.pump){ ctl.saveNow = true;
        log('save requested at sim ' + (ctl.simMs/60000).toFixed(1) + ' min'); }
      else {
        ctl.pendingSave = true; ctl.wreqAt = Date.now();
        engine.postMessage({ cmd:'getWeights' });
        log('save requested while paused, at sim ' + (ctl.simMs/60000).toFixed(1) + ' min');
      }
    }
    else if(m.__host === 'save'){
      ctl.save = { everyMs:m.everyMs, dir:m.dir, next:ctl.simMs + m.everyMs };
      ctl.dir = m.dir || 'run';
      LIVE.set(ctl.dir, ctl);
    }
    else if(m.__host === 'attach'){
      // this connection becomes a viewer of a live run: its own engine is never used, and the target run's state frames are copied to it
      let target = m.dir ? LIVE.get(m.dir) : null;
      if(!target) for(const c of LIVE.values()) target = c;   // latest registered
      if(!target){
        socket.write(frame(encodeMsg({ cmd:'error',
          message:'no live run to attach to' })));
        log('attach refused: no live run');
        return;
      }
      engine.terminate();
      const fn = d => {
        if(socket.writableLength < (1 << 20)){
          try { socket.write(frame(encodeMsg(d))); } catch(_){ }
        }
      };
      target.watchers.add(fn);
      watching = { ctl:target, fn };
      fn({ __attached:{ dir:target.dir, simMs:target.simMs,
        untilMs:target.untilMs } });
      log('viewer attached to ' + target.dir + ' at sim ' +
        (target.simMs/60000).toFixed(1) + ' min (' + target.watchers.size +
        ' watching)');
    }
    else if(ctl.refused && (m.__host === 'detach' || m.__host === 'pump')){
      log('ignored ' + m.__host + ' from a refused run');
    }
    else if(m.__host === 'detach'){
      // a detached run survives its client: on disconnect the host keeps pumping until the given clock and keeps writing brain files, and the page gets its results from host/runs rather than the socket.
      ctl.detach = true; ctl.untilMs = m.untilMs || 0;
      log('detach armed until sim ' + (ctl.untilMs/60000).toFixed(0) + ' min');
    }
    else if(m.__host === 'pump'){
      const was = ctl.pump;
      ctl.pump = !!m.on;
      if(m.steps) ctl.steps = m.steps;
      log('pump ' + (ctl.pump ? 'on, steps ' + ctl.steps : 'off') +
        ' at sim ' + (ctl.simMs/60000).toFixed(1) + ' min');
      if(ctl.pump && !was) tick();
    }
  };
  const saveBrain = w => {
    const n = ctl.netShim;
    if(!n || !ctl.save) return;
    try {
      // The host and the page write brains into one folder, so they have to agree on its shape; src/project.js is the definition and this is the one line outside the browser that has to match it.
      const dir = path.join(RUNS, ctl.save.dir || 'run', 'brains');
      fs.mkdirSync(dir, { recursive:true });
      const cur = n.curriculum ? { ...n.curriculum, simMs:ctl.simMs } : null;
      // Plastic-only whenever the run has a gate: a full weights file at density is 579 MB a save, the pump stops for the whole write, and the wiring reproduces the frozen weights exactly.
      const net = ctl.pmask ? { ...n, pmask:ctl.pmask } : n;
      const file = ctl.pmask
        ? buildPlasticFile(n.graph, net, w, ctl.simMs, cur)
        : buildWeightsFile(n.graph, net, w, ctl.simMs, cur);
      const name = 'ckpt-' + ctl.save.dir + '-' +
        String(Math.round(ctl.simMs/60000)).padStart(5, '0') + 'min.npb';
      fs.writeFileSync(path.join(dir, name), Buffer.from(file));
      // Readback and write are reported separately: they are not the same size, and shrinking the file only addresses the second.
      log('saved ' + name + ' (' + (file.byteLength/1048576).toFixed(1) + ' MB) at sim ' +
        (ctl.simMs/60000).toFixed(1) + ' min; readback ' +
        ((ctl.wbackMs || 0)/1000).toFixed(1) + ' s, build and write ' +
        ((Date.now() - ctl.wbackAt)/1000).toFixed(1) + ' s');
    } catch(e){ log('save failed: ' + e.message); }
  };
  engine.on('message', d => {
    replies++;
    // The engine answers hello for itself; the host adds its run controller to the optional queries (ENGINE.md section 7).
    if(d && d.cmd === 'hello' && Array.isArray(d.optional) && !d.optional.includes('host'))
      d = { ...d, optional:[...d.optional, 'host'] };
    if(ctl.pump && d.cmd === 'state'){
      ctl.simMs += d.steps;
      if(ctl.gone && ctl.simMs >= ctl.untilMs){
        ctl.pump = false;
        if(ctl.save && ctl.netShim){ ctl.pendingSave = true; engine.postMessage({ cmd:'getWeights' }); }
        log('detached run reached sim ' + (ctl.simMs/60000).toFixed(0) + ' min; final save and stop');
        LIVE.delete(ctl.dir);
        return;
      }
      for(let i = ctl.ats.length - 1; i >= 0; i--)
        if(ctl.simMs >= ctl.ats[i].simMs){
          log('scheduled message fired at sim ' + (ctl.simMs/1000).toFixed(0) + ' s');
          // a refusal here would reach the uncaught handler and take the host down; it goes to the page instead
          try { engine.postMessage(ctl.ats[i].msg); }
          catch(e){ log('engine refused a scheduled message:', e.message); forward({ cmd:'error', message:e.message }); }
          ctl.ats.splice(i, 1);
        }
      if(ctl.io) ctl.io.frame(ctl.simMs);
      d.__simMs = ctl.simMs;
      forward(d);
      for(const fn of ctl.watchers) fn(d);
      // Cadence or an explicit request from the client, taken at the same point so there is only ever one tick chain: asking for weights out of band while the pump is running would start a second.
      if(ctl.save && (ctl.simMs >= ctl.save.next || ctl.saveNow)){
        if(ctl.simMs >= ctl.save.next) ctl.save.next += ctl.save.everyMs;
        ctl.saveNow = false;
        ctl.pendingSave = true;
        ctl.wreqAt = Date.now();
        engine.postMessage({ cmd:'getWeights' });
      } else tick();
      if(ctl.simMs - ctl.lastLog >= 300000){
        ctl.lastLog = ctl.simMs;
        log('pumping, sim ' + (ctl.simMs/60000).toFixed(1) + ' min');
      }
      return;
    }
    if(d.cmd === 'weights' && ctl.pendingSave){
      ctl.pendingSave = false;
      ctl.wbackAt = Date.now();
      ctl.wbackMs = ctl.wbackAt - (ctl.wreqAt || ctl.wbackAt);
      saveBrain(d.w);
      forward(d);
      if(ctl.pump) tick();
      else if(ctl.gone){ log('detached run complete; engine down');
        LIVE.delete(ctl.dir); engine.terminate(); }
      return;
    }
    forward(d);
  });
  engine.on('error', e => {
    log('engine error:', e.message);
    try { socket.write(frame(encodeMsg({ cmd:'error', message:'host engine: ' + e.message }))); } catch(_){ }
  });
  log('client connected, engine spawned');

  let buf = Buffer.alloc(0);
  const parts = [];                          // fragmented message assembly
  const close = why => {
    log('client gone (' + why + '), ' + msgs + ' in / ' + replies + ' out');
    socket.destroy();
    if(watching){
      watching.ctl.watchers.delete(watching.fn);
      watching = null;
      log('viewer detached');
      return;
    }
    if(ctl.detach && ctl.simMs < ctl.untilMs){
      // Not conditional on the pump: detach is an instruction about the run and a pause is a viewing convenience of a client that has now gone, so a paused run resumes and carries on to the clock it was given.
      ctl.gone = true;
      if(!ctl.pump){
        ctl.pump = true;
        log('detached while paused: resuming, since the client that paused ' +
          'it is gone');
        tick();
      }
      log('detached: pumping on to sim ' + (ctl.untilMs/60000).toFixed(0) + ' min');
      return;
    }
    LIVE.delete(ctl.dir);
    engine.terminate();
  };
  socket.on('data', chunk => {
    buf = Buffer.concat([buf, chunk]);
    for(;;){
      if(buf.length < 2) return;
      const fin = (buf[0] & 0x80) !== 0, op = buf[0] & 0x0f;
      const masked = (buf[1] & 0x80) !== 0;
      let len = buf[1] & 0x7f, off = 2;
      if(len === 126){ if(buf.length < 4) return; len = buf.readUInt16BE(2); off = 4; }
      else if(len === 127){ if(buf.length < 10) return; len = Number(buf.readBigUInt64BE(2)); off = 10; }
      const need = off + (masked ? 4 : 0) + len;
      if(buf.length < need) return;
      let payload;
      if(masked){
        const mask = buf.subarray(off, off + 4);
        payload = Buffer.from(buf.subarray(off + 4, need));
        for(let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
      } else payload = Buffer.from(buf.subarray(off, need));
      buf = buf.subarray(need);
      if(op === 8){ close('close frame'); return; }
      if(op === 9){ socket.write(Buffer.concat([Buffer.from([0x8a, payload.length]), payload])); continue; }
      if(op === 10) continue;               // pong
      parts.push(payload);
      if(!fin) continue;
      const whole = parts.length === 1 ? parts[0] : Buffer.concat(parts);
      parts.length = 0;
      let m;
      try { m = decodeMsg(whole.buffer.slice(whole.byteOffset, whole.byteOffset + whole.byteLength)); }
      catch(e){ log('bad frame:', e.message); continue; }
      msgs++;
      // The adapter refuses what the engine cannot run by throwing, and the engine contract requires the refusal to reach the page, not only the log.
      try {
        if(m && m.__host) onCtl(m);
        else {
          // The plasticity mask rides along with the network on its way to the engine, and nothing else on the host reads it.
          // A plastic-only checkpoint cannot be written without it, so keep the reference here rather than making the client send another 151 MB it has already sent once.
          if(m && m.cmd === 'init' && m.pmask) ctl.pmask = m.pmask;
          engine.postMessage(m);
        }
      } catch(e){
        log('engine refused:', e.message);
        try { socket.write(frame(encodeMsg({ cmd:'error', message:e.message }))); } catch(_){ }
      }
    }
  });
  socket.on('error', () => close('socket error'));
  socket.on('end', () => close('end'));
});

server.listen(PORT, () => {
  log('engine host listening on ws://localhost:' + PORT +
    (CUDA ? ' (CUDA engine: ' + CUDA_EXE + ')' : ' (reference engine)'));
  // Said at startup rather than at the first save, which can be half an hour in: a host writing somewhere other than the project folder is worth knowing about before the run, not after it.
  log('checkpoints go to ' + RUNS + (PROJECT ? '' : ' (no --project given)'));
});

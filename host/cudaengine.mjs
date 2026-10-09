// Adapter that makes the CUDA engine child (cuda/engine.exe) look like the worker the host already drives: postMessage in, 'message' events out, terminate.
// The translation from ENGINE.md messages to the child's binary structs lives here, and so does the little policy the child refuses to own: expanding neuron types to per-neuron parameters, folding constant (mode 0) stimulus protocols into a static drive, folding noise (mode 3) protocols into per-neuron noise amplitudes, and rebuilding the external drive from input-map gains exactly the way src/simworker.js rebuildExt does.
// Anything the child does not support fails loudly here rather than silently simulating something else.
import { spawn } from 'node:child_process';
import { checkProtocol, helloReply, TERM_NAMES, OPTIONAL } from '../src/protocol.js';
import { gradedArrays, freezeGraded } from '../src/nodes.js';
import { EventEmitter } from 'node:events';

const CMD = { INIT:1, TUNE:2, EXT:3, NAMP:4, TICK:5, GETW:6,
  QUERY:7, WATCH:8, SENDV:9, SENDT:10, HELLO:11 };
// Every term but the time-varying stimulus protocols, which the child does not run (_protocols), and every optional query.
export const CUDA_TERMS = TERM_NAMES.filter(t => t !== 'protocols:1' && t !== 'protocols:2');

// Frame reassembly over the child's stdout: 8-byte header (cmd, length) then the body.
// Chunks accumulate by reference and are joined once, when a whole frame has arrived.
// Concatenating on every chunk is quadratic in frame size: a weights frame at full density is 607 MB arriving in about nine thousand pieces.
// Exported so the chunking can be tested without a child.
export function makeFrameReader(onFrame){
  let chunks = [], len = 0;
  return chunk => {
    chunks.push(chunk);
    len += chunk.length;
    for(;;){
      if(len < 8) return;
      // the header has to be contiguous before it can be read
      if(chunks[0].length < 8) chunks = [Buffer.concat(chunks, len)];
      const cmd = chunks[0].readUInt32LE(0);
      const flen = chunks[0].readUInt32LE(4);
      if(len < 8 + flen) return;
      const all = chunks.length === 1 ? chunks[0] : Buffer.concat(chunks, len);
      const body = all.subarray(8, 8 + flen);
      // copy the tail so the frame buffer can be released; it is a few bytes in the normal case, never another whole frame
      const rest = all.subarray(8 + flen);
      chunks = rest.length ? [Buffer.from(rest)] : [];
      len = rest.length;
      onFrame(cmd, body);
    }
  };
}

export class CudaEngine extends EventEmitter {
  constructor(exePath, log){
    super();
    this.log = log || (() => {});
    this.proc = spawn(exePath, [], { stdio:['pipe', 'pipe', 'pipe'] });
    this.proc.stderr.on('data', d => {
      const t = String(d).trim();
      this.log('cuda: ' + t);
      if(/CUDA error|out of memory|assert/i.test(t))
        this.emit('message', { cmd:'error', message:'CUDA engine: ' + t });
    });
    this.dead = false;
    this.proc.on('exit', code => {
      this.dead = true;
      this.log('cuda engine exited (' + code + ')');
      // Told to the client: otherwise the host pumps into a dead child and the page shows a run with a frozen clock, the failure legible only in a log file.
      if(!this.stopping) this.emit('message', { cmd:'error',
        message:'the CUDA engine exited (code ' + code + '). The run has ' +
          'stopped; see host/host.log for what it printed on the way out.' });
    });
    // The child can go at any moment: a failed allocation, a killed run, a client that disconnected and took its engine down.
    // On Windows its stdio are sockets, so the next write raises EPIPE on a stream with no listener, and an unhandled 'error' event ends the host process, and with it any run that is still wiring.
    // A dead child is a logged line, never an exit.
    this.proc.stdin.on('error', e => {
      this.dead = true;
      this.log('cuda stdin: ' + e.message);
    });
    this.proc.stdout.on('error', e => this.log('cuda stdout: ' + e.message));
    this.proc.on('error', e => {
      this.dead = true;
      this.log('cuda engine error: ' + e.message);
    });
    this.n = 0; this.m = 0;
    this.sendV = false; this.sendT = false;
    this.maps = [];                        // input CSRs + latest gains
    this.extBase = null;                   // constant mode-0 protocol drive
    this.proc.stdout.on('data', makeFrameReader((cmd, body) => {
        if(cmd === 100){
          const steps = body.readUInt32LE(0);
          const hasV = body.readUInt32LE(4);
          const spikes = Number(body.readBigUInt64LE(8));
          const fired = new Uint8Array(body.subarray(16, 16 + this.n));
          const msg = { cmd:'state', steps, spikes, fired };
          // the watched neuron's membrane trace, when the viewer has one selected
          if(hasV){
            const vt = new Float32Array(steps);
            const off = 16 + this.n;
            for(let i = 0; i < steps; i++) vt[i] = body.readFloatLE(off + i*4);
            msg.vtrace = vt;
          }
          // The whole membrane array rides at the end when the filter is on.
          // Nothing in the header announces it; the frame is longer by exactly n floats, which is what the length is checked for.
          const tail = 16 + this.n + (hasV ? steps*4 : 0);
          let off = tail;
          if(this.sendV && body.length >= off + this.n*4){
            msg.v = new Float32Array(body.buffer.slice(
              body.byteOffset + off, body.byteOffset + off + this.n*4));
            off += this.n*4;
          }
          // First-spike latency, one byte per neuron, 255 meaning silent.
          // The readout bins at the tick, so without this its finest grain is 50 ms against an STDP window of 20.
          if(this.sendT && body.length >= off + this.n)
            msg.lat = new Uint8Array(body.buffer.slice(
              body.byteOffset + off, body.byteOffset + off + this.n));
          this.emit('message', msg);
        } else if(cmd === 102){
          if(body.length >= 20){
            const idx = body.readUInt32LE(0);
            const outTotal = body.readUInt32LE(4), inTotal = body.readUInt32LE(8);
            const outN = body.readUInt32LE(12), innN = body.readUInt32LE(16);
            const out = new Uint32Array(outN), inn = new Uint32Array(innN);
            for(let i = 0; i < outN; i++) out[i] = body.readUInt32LE(20 + i*4);
            for(let i = 0; i < innN; i++)
              inn[i] = body.readUInt32LE(20 + outN*4 + i*4);
            this.emit('message', { cmd:'queryResult', idx, out, inn,
              outTotal, inTotal });
          }
        } else if(cmd === 103){
          // The child's own contract number goes into the reply, so a child built for another contract is refused by the hello check rather than run.
          clearTimeout(this.helloTimer);
          this.emit('message', { ...helloReply('cuda', CUDA_TERMS, OPTIONAL),
            protocol:body.length >= 4 ? body.readUInt32LE(0) : null });
        } else if(cmd === 101){
          // one copy rather than 151.7M readFloatLE calls; slicing the ArrayBuffer also sidesteps the alignment that a subarray at an arbitrary byte offset cannot promise a Float32Array
          const w = new Float32Array(body.buffer.slice(
            body.byteOffset, body.byteOffset + this.m*4));
          // Measured set points ride in the tail when the child has them, so the length says whether they are there.
          let rho = null;
          const extra = (body.length - this.m*4)/4;
          if(extra > 0){
            rho = new Float32Array(extra);
            for(let i = 0; i < extra; i++) rho[i] = body.readFloatLE((this.m + i)*4);
          }
          this.emit('message', { cmd:'weights', w, rho, t:0 });
        }
    }));
  }
  _send(cmd, payload){
    // writing to a child that has already gone is a no-op, not a crash
    if(this.dead || !this.proc.stdin.writable) return;
    const head = Buffer.alloc(8);
    head.writeUInt32LE(cmd, 0);
    head.writeUInt32LE(payload ? payload.byteLength : 0, 4);
    this.proc.stdin.write(head);
    if(payload) this.proc.stdin.write(Buffer.from(payload.buffer || payload,
      payload.byteOffset || 0, payload.byteLength));
  }
  // The frame is 120 bytes of scalars, the receptor channels and the conductance reversals to byte 184, then the per-rule table: a u32 row count and rowCount*11 floats, the eleven fields in the order every engine uses.
  // The child reads it behind a length guard, so a host that sends the shorter frame still works and the child then builds row 1 from the scalars itself.
  _plastStruct(msg){
    const RS = 11, RMAX = 16;
    const rows = (msg.ruleTable && msg.ruleCount >= 2)
      ? Math.min(RMAX, msg.ruleCount) : 0;
    // receptor channels at 120 (u32 count, six f32 taus), the rule table from 184
    const chan = Array.isArray(msg.chanTau) || ArrayBuffer.isView(msg.chanTau) ? Array.from(msg.chanTau).slice(0, 6) : [];
    const b = Buffer.alloc(rows ? 188 + rows*RS*4 : 184);
    b.writeUInt32LE(chan.length, 120);
    chan.forEach((t, x) => b.writeFloatLE(t > 0 ? +t : 3, 124 + 4*x));
    // conductance mode's reversals at 148 (E), 152 (I), 156 (six channels)
    b.writeFloatLE(Number.isFinite(+msg.eRevE) ? +msg.eRevE : 0, 148);
    b.writeFloatLE(Number.isFinite(+msg.eRevI) ? +msg.eRevI : -70, 152);
    const erev = Array.isArray(msg.chanErev) || ArrayBuffer.isView(msg.chanErev) ? Array.from(msg.chanErev).slice(0, 6) : [];
    erev.forEach((v, x) => b.writeFloatLE(+v || 0, 156 + 4*x));
    b.writeUInt32LE((msg.plast|0) === 1 ? 1 : 0, 0);
    b.writeUInt32LE((msg.wdep|0) === 1 ? 1 : 0, 4);
    b.writeFloatLE(msg.aP !== undefined ? +msg.aP : 0.008, 8);
    b.writeFloatLE(msg.aM !== undefined ? +msg.aM : 0.001, 12);
    b.writeFloatLE(msg.tauS > 0 ? msg.tauS : 16.8, 16);
    b.writeFloatLE(msg.wmax > 0 ? +msg.wmax : 30, 20);
    b.writeFloatLE(msg.iEta !== undefined ? +msg.iEta : 0.002, 24);
    // 5 Hz when absent, as the other two engines default it; a 0 would make the child's Vogels alpha 0 while its scaling falls back to 5
    b.writeFloatLE(msg.iRho || 5, 28);
    // the fast mechanism set (triplet, heterosynaptic, transmitter), ENGINE.md; zeros reproduce the pair rule exactly
    b.writeFloatLE(msg.trip > 0 ? +msg.trip : 0, 32);
    b.writeFloatLE(msg.tauY > 0 ? +msg.tauY : 114, 36);
    b.writeFloatLE(msg.het > 0 ? +msg.het : 0, 40);
    b.writeFloatLE(msg.tin > 0 ? +msg.tin : 0, 44);
    b.writeUInt32LE(msg.refrac > 0 ? Math.min(20, msg.refrac|0) : 0, 48);
    // consolidation of the reference weight, ENGINE.md; zero leaves the reference frozen
    b.writeFloatLE(msg.cons > 0 ? +msg.cons : 0, 52);
    // zero means the published ratio against this tissue's wmax, a tenth of maximum, rather than the published number for a wmax of 5
    b.writeFloatLE(msg.consW > 0 ? +msg.consW : 0.1*(msg.wmax > 0 ? msg.wmax : 10), 56);
    b.writeFloatLE(msg.consP > 0 ? +msg.consP : 10, 60);
    b.writeFloatLE(msg.stp > 0 ? 1 : 0, 64);
    b.writeFloatLE(msg.stpU > 0 ? +msg.stpU : 0.2, 68);
    b.writeFloatLE(msg.stpTauD > 0 ? +msg.stpTauD : 200, 72);
    b.writeFloatLE(msg.stpTauF > 0 ? +msg.stpTauF : 600, 76);
    b.writeFloatLE((msg.scale|0) === 1 ? 1 : 0, 80);
    b.writeFloatLE(msg.sEta > 0 ? +msg.sEta : 0.001, 84);
    b.writeFloatLE((msg.rhoMode|0) === 1 ? 1 : 0, 88);
    b.writeFloatLE(msg.calS > 0 ? +msg.calS : 10, 92);
    b.writeFloatLE(msg.tauM > 0 ? +msg.tauM : (msg.tauS > 0 ? +msg.tauS : 16.8), 96);
    // byte 100 carries two switches: 1 is stpNorm, 2 is stpOrder (facilitation before release)
    b.writeFloatLE(((msg.stpNorm|0) === 1 ? 1 : 0) + ((msg.stpOrder|0) === 1 ? 2 : 0), 100);
    b.writeFloatLE((msg.commit|0) === 1 ? 1 : 0, 104);
    b.writeFloatLE(msg.tauCons > 0 ? +msg.tauCons : 1200000, 108);
    b.writeFloatLE(msg.consStep > 0 ? +msg.consStep : 1200, 112);
    b.writeFloatLE(msg.vmin === undefined ? -90 : +msg.vmin, 116);
    if(rows){
      b.writeUInt32LE(rows, 184);
      for(let r = 0; r < rows; r++)
        for(let k = 0; k < RS; k++)
          b.writeFloatLE(msg.ruleTable[r*RS + k] || 0, 188 + (r*RS + k)*4);
    }
    return b;
  }
  _protocols(msg){
    // mode 0 folds into a constant base drive, mode 3 into per-neuron noise amplitude; the child has no time-varying protocol, so anything else must fail loudly
    const n = this.n;
    this.extBase = new Float32Array(n);
    const namp = new Float32Array(n);
    for(const p of (msg.protocols || [])){
      const mode = p.mode|0;
      const g = p.gain;              // per-neuron spread from the stimulus node
      if(mode === 0){
        p.idx.forEach((i, q) => { this.extBase[i] += p.amp*(g ? g[q] : 1); });
      } else if(mode === 3){
        p.idx.forEach((i, q) => { namp[i] += p.amp*(g ? g[q] : 1); });
      } else {
        throw new Error('cuda engine: unsupported protocol mode ' + mode +
          ' (only constant and noise are ported)');
      }
    }
    this._send(CMD.NAMP, namp);
    this._pushExt();
  }
  _pushExt(){
    const ext = Float32Array.from(this.extBase || new Float32Array(this.n));
    for(const im of this.maps){
      if(!im.gains) continue;
      const NC = Math.min(im.gains.length, im.chStart.length - 1);
      for(let c = 0; c < NC; c++){
        const g = im.gains[c]*im.amp;
        if(!g) continue;
        for(let k = im.chStart[c]; k < im.chStart[c+1]; k++)
          ext[im.chIdx[k]] += g*im.chW[k];
      }
    }
    this._send(CMD.EXT, ext);
  }
  postMessage(msg){
    if(msg.cmd === 'hello'){
      // A child built before HELLO existed ignores the command, so the silence is reported rather than waited on.
      clearTimeout(this.helloTimer);
      this.helloTimer = setTimeout(() => this.emit('message', { cmd:'error',
        message:'the CUDA engine child did not answer HELLO; cuda/engine.exe is from an older build and needs rebuilding' }), 5000);
      if(this.helloTimer.unref) this.helloTimer.unref();
      this._send(CMD.HELLO, null);
    }
    else if(msg.cmd === 'init'){
      // refused before the sizes are taken, so a refused init leaves no half-set engine for the next tick to run into
      { const bad = checkProtocol(msg, 'cuda engine'); if(bad) throw new Error(bad); }
      this.n = msg.count; this.m = msg.w.length;
      const n = this.n;
      const a = new Float32Array(n), b = new Float32Array(n),
        c = new Float32Array(n), d = new Float32Array(n);
      // the membrane (Izhikevich 2007, MODEL.md 1): the row's a, b, c, d in their arrays and its C, k, vr, vt, vpeak at the tail of the frame.
      if('form' in msg && (msg.form|0) !== 1)
        throw new Error('cuda engine: the 2003 membrane form was retired; the scene needs migrating (scene format 8)');
      const f7 = ['C', 'k', 'vr', 'vt', 'vpeak'].map(() => new Float32Array(n));
      for(let i = 0; i < n; i++){
        const t = msg.types[msg.ntype[i]] && msg.types[msg.ntype[i]].f7;
        if(!t) throw new Error('cuda engine: every neuron type needs a 2007 row (f7)');
        a[i] = t.a; b[i] = t.b; c[i] = t.c; d[i] = t.d;
        f7[0][i] = t.C; f7[1][i] = t.k; f7[2][i] = t.vr; f7[3][i] = t.vt; f7[4][i] = t.vpeak;
      }
      // graded rows: the flag, threshold and slope per cell ride at the very end of the frame, and their outgoing synapses are frozen for the child
      const G = gradedArrays(msg.ntype, msg.types);
      const pmask = freezeGraded(msg.pmask ? Uint8Array.from(msg.pmask) : new Uint8Array(this.m).fill(1),
        msg.preStart, G && G.grd, this.m);
      const head = Buffer.alloc(24);
      head.writeUInt32LE(n, 0); head.writeUInt32LE(this.m, 4);
      head.writeUInt32LE(Math.max(0, Math.min(2, msg.syn|0)), 8);   // 0 kick, 1 exp, 2 conductance
      head.writeFloatLE(msg.tauE > 0 ? msg.tauE : 3, 12);
      head.writeFloatLE(msg.tauI > 0 ? msg.tauI : 8, 16);
      head.writeUInt32LE(Math.max(0, Math.min(2, msg.psc|0)), 20);   // exp current scale: 0 peak, 1 charge, 2 exact
      const cat = (arrs) => {
        let total = head.length;
        for(const x of arrs) total += x.byteLength;
        const out = Buffer.alloc(total);
        head.copy(out, 0);
        let off = head.length;
        for(const x of arrs){
          Buffer.from(x.buffer, x.byteOffset, x.byteLength).copy(out, off);
          off += x.byteLength;
        }
        return out;
      };
      // Initial membrane state and the noise seed travel to the child, the same bytes every engine starts from (ENGINE.md section 2).
      if(!msg.v0 || !msg.u0 || msg.v0.length !== n || msg.u0.length !== n)
        throw new Error('cuda engine: init needs v0 and u0 (rand.js initState)');
      // Feedback inhibition rides at the tail: the pool count, the pool per cell, the kick per spike per pool.
      // A network without pools sends one pool (index 0, no pool) and zeros, so the frame length is fixed by n and the count and the child can check it.
      const hasPool = msg.pool && msg.pool.length === n && msg.poolK && msg.poolK.length >= 2;
      if(msg.pool && msg.pool.length && !hasPool)
        throw new Error('cuda engine: init pool arrays do not match the network');
      const payload = cat([a, b, c, d,
        Float32Array.from(msg.bias),
        Int32Array.from(msg.preStart), Int32Array.from(msg.post),
        Float32Array.from(msg.w), Uint8Array.from(msg.delay), pmask,
        Float32Array.from(msg.v0), Float32Array.from(msg.u0),
        Uint32Array.of((msg.seed >>> 0) || 1),
        Uint32Array.of(hasPool ? msg.poolK.length : 1),
        hasPool ? Uint16Array.from(msg.pool) : new Uint16Array(n),
        hasPool ? Float32Array.from(msg.poolK) : new Float32Array(1),
        ...f7,
        Uint32Array.of(G ? 1 : 0), ...(G ? [G.grd, G.thr, G.slope] : [])]);
      this._send(CMD.INIT, payload);
      this.maps = (msg.inputs || []).map(im => ({ chStart:im.chStart,
        chIdx:im.chIdx, chW:im.chW, amp:im.amp, gains:null }));
      this._send(CMD.TUNE, this._plastStruct(msg));
      this._protocols(msg);
    }
    else if(msg.cmd === 'tune'){
      // A term the reference has and the child lacks belongs in a guard here rather than being dropped in silence.
      //
      // The child's per-neuron targets differ from the reference's for about half the neurons by up to 1.1 Hz, which is not a counting bug: the reference engine's initial membrane jitter puts two runs of the reference on different engine seeds the same distance apart.
      // Reference against reference on two engine seeds gives 2083 of 4000 identical and a 1.200 Hz maximum, against 2033 and 1.100 for reference against child.
      // The port sits inside the engine's own initial-state spread, which is the equivalence ENGINE.md specifies.
      // Weights are bit-identical either way, because the weight trajectory washes the jitter out, which is exactly why no weight-based check could see any of this.
      this._send(CMD.TUNE, this._plastStruct(msg));
      if(msg.protocols !== undefined) this._protocols(msg);
    }
    else if(msg.cmd === 'inputFrame'){
      const im = this.maps[msg.mi];
      if(im){ im.gains = msg.gains; this._pushExt(); }
    }
    else if(msg.cmd === 'tick'){
      const b = Buffer.alloc(4);
      b.writeUInt32LE(msg.steps || 50, 0);
      this._send(CMD.TICK, b);
    }
    else if(msg.cmd === 'getWeights') this._send(CMD.GETW, null);
    else if(msg.cmd === 'watch'){
      const b = Buffer.alloc(4);
      b.writeInt32LE(msg.idx | 0, 0);
      this._send(CMD.WATCH, b);
    }
    else if(msg.cmd === 'query'){
      const b = Buffer.alloc(12);
      b.writeUInt32LE(msg.idx >>> 0, 0);
      b.writeUInt32LE((msg.cap || 2000) >>> 0, 4);
      b.writeUInt32LE(msg.dir === 'out' ? 1 : 0, 8);
      this._send(CMD.QUERY, b);
    }
    else if(msg.cmd === 'sendV'){
      const b = Buffer.alloc(4);
      b.writeUInt32LE(msg.on ? 1 : 0, 0);
      this.sendV = !!msg.on;
      this._send(CMD.SENDV, b);
    }
    else if(msg.cmd === 'sendT'){
      const b = Buffer.alloc(4);
      b.writeUInt32LE(msg.on ? 1 : 0, 0);
      this.sendT = !!msg.on;
      this._send(CMD.SENDT, b);
    }
    else this.log('cuda engine: ignoring unsupported cmd ' + msg.cmd);
  }
  terminate(){ clearTimeout(this.helloTimer); this.dead = true; this.stopping = true; try { this.proc.kill(); } catch(e){ } }
}

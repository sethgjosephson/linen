// One test harness: `ok`, a `report()` that prints one summary line in one format, and the exit code.
// Nothing forces adoption: the runner (tools/test.mjs) keys off the process exit code, so a suite that does not import this still counts.
//
//   import { ok, report } from '../tools/harness.mjs';   // from src/ or host/
//   ok('a thing is true', x === 1, 'got ' + x);
//   report('mysuite');   // prints, sets exitCode, returns the count

let pass = 0, fail = 0;

export function ok(name, cond, detail){
  if(cond){ pass++; return true; }
  fail++;
  console.log('FAIL: ' + name + (detail !== undefined ? '  ' + detail : ''));
  return false;
}

// Runs fn() and returns its thrown message, or null.
export function threw(fn){
  try { fn(); return null; } catch(e){ return e && e.message !== undefined ? e.message : String(e); }
}

// node has no web Worker.
// This stands in for one so a connect node computes in this process: the wiring pool's probe is answered with no, and the compute worker's two jobs run here.
// Pass the two functions from src/nodes.js (the harness imports nothing from the app).
//   inProcessWorker(wireConnect, assembleConnect);
export function inProcessWorker(wireConnect, assembleConnect){
  globalThis.Worker = class {
    constructor(url){ this.url = String(url); this.onmessage = null; this.onerror = null; }
    postMessage(m){
      if(m && m.ping){ setTimeout(() => this.onmessage && this.onmessage({ data:{ pong:false } }), 0); return; }
      if(this.url.includes('computeworker')) setTimeout(() => {
        try {
          const net = m.t === 'assemble' ? assembleConnect(m.pts, m.params, m.parts, () => {}) : wireConnect(m.pts, m.params, () => {});
          this.onmessage({ data:{ t:'done', net } });
        } catch(e){ this.onmessage({ data:{ t:'error', message:e.message } }); }
      }, 0);
    }
    terminate(){}
  };
}

export function counts(){ return { pass, fail }; }

// One summary line, one format everywhere, and the exit code.
// Named so the runner can find the number: "<name>: all passed (N)" or "<name>: M failed, N passed".
// Sets process.exitCode rather than calling exit, so a suite can report and then keep going if it wants; most call it last.
export function report(name){
  const line = fail
    ? `${name}: ${fail} failed, ${pass} passed`
    : `${name}: all passed (${pass})`;
  console.log(line);
  process.exitCode = fail ? 1 : 0;
  return { pass, fail };
}

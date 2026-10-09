// Browser-side adapter for the engine host (host/host.mjs).
// Presents the same surface as a Worker running an engine, so the trainer and viewer code cannot tell where the arithmetic happens; messages cross the socket in the framing src/frame.js defines and are otherwise exactly the ENGINE.md protocol.
import { encodeMsg, decodeMsg } from './frame.js';

// a listener that throws is a bug in the listener, not in the transport
const reportListener = e => console.error('engine listener failed:', e);

export class RemoteEngine {
  constructor(url){
    // A Worker takes any number of listeners, and the trainer takes the onmessage slot.
    // Without addEventListener as well, everything else that listens to the engine (neuron inspection, the membrane scope, the potential filter) is silently deaf on a remote run.
    this.onmessage = null;
    this._ls = new Set();
    this._q = [];                         // sends queued until the socket opens
    this._open = false;
    this._closed = false;
    const ws = this._ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      this._open = true;
      for(const b of this._q) ws.send(b);
      this._q.length = 0;
    };
    // a hello reply carries the address it came from, so the page can name the engine and where it runs
    ws.onmessage = ev => {
      const d = decodeMsg(ev.data);
      this._emit({ data: d && d.cmd === 'hello' ? { ...d, url } : d });
    };
    // a refused or dropped connection surfaces as an engine error, which is the failure mode the trainer already knows how to report
    const fail = why => () => {
      if(this._closed) return;
      this._closed = true;
      this._emit({ data: { cmd:'error', message: why + ' ' + url +
        ' (start host/host.mjs or a custom engine there)' } });
    };
    ws.onerror = fail('no engine answered at');
    ws.onclose = ev => { if(!ev.wasClean) fail('lost the connection to the engine at')(); };
  }
  // the Worker surface the rest of the app feature-tests for
  addEventListener(type, fn){ if(type === 'message' && fn) this._ls.add(fn); }
  removeEventListener(type, fn){ if(type === 'message') this._ls.delete(fn); }
  // One listener throwing must not stop the others from being told, or a fault in the viewer would silently stop the run's own handler.
  _emit(ev){
    if(this.onmessage) try { this.onmessage(ev); } catch(e){ reportListener(e); }
    for(const fn of this._ls) try { fn(ev); } catch(e){ reportListener(e); }
  }
  postMessage(msg /* , transfer ignored: the copy happens in the codec */){
    const b = encodeMsg(msg);
    if(this._open) this._ws.send(b); else this._q.push(b);
  }
  terminate(){
    this._closed = true;
    try { this._ws.close(); } catch(e){ }
  }
}

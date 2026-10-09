// Runs the reference engine inside a Node worker thread. simworker.js is written against the browser worker globals (onmessage, postMessage); this shim provides them over worker_threads and then imports the engine unchanged, which is the point: the host runs the same code the battery verifies, not a port of it.
import { parentPort } from 'node:worker_threads';

globalThis.postMessage = (data) => parentPort.postMessage(data);
globalThis.onmessage = null;

await import('../src/simworker.js');

parentPort.on('message', (data) => {
  if(typeof globalThis.onmessage === 'function')
    globalThis.onmessage({ data });
});

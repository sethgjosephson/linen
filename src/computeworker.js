// Runs the connect node's wiring off the main thread, streaming progress back.
//
// Two jobs, depending on what the page could arrange.
// If a pool of wiring workers is available, the page fans the sweep out across it and sends the finished slices here to be assembled: this worker owns the concatenation and the parts of the wiring that are not per-neuron (triadic closure, which walks the finished network with one sequential random stream, and the proxy compensation, which needs every slice's in-degree tallies).
// If the pool is unavailable, it wires serially here instead, which is the same code and the same result, only on one thread.
import { wireConnect, assembleConnect } from './nodes.js';

onmessage = e => {
  const m = e.data;
  try {
    if(m.t === 'assemble'){
      let last = 0;
      const net = assembleConnect(m.pts, m.params, m.parts, (f, syn) => {
        const now = performance.now();
        if(now - last > 100){ last = now; postMessage({ t:'progress', f, syn }); }
      });
      return send(net);
    }
    let last = 0;
    // the synapse count travels with the fraction; without forwarding it here the number is computed in this worker and thrown away at the boundary
    const net = wireConnect(m.pts, m.params, (f, syn) => {
      const now = performance.now();
      if(now - last > 100){ last = now; postMessage({ t:'progress', f, syn }); }
    });
    send(net);
  } catch(err){
    postMessage({ t:'error', message: err.message });
  }
};

function send(net){
  postMessage({ t:'done', net },
    [net.pos.buffer, net.ntype.buffer, net.bias.buffer, net.src.buffer,
     net.lidx.buffer, net.preStart.buffer, net.post.buffer, net.w.buffer,
     net.delay.buffer]);
}

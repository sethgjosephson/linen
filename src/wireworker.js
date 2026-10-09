// One slice of the synapse wiring.
// The parent hands over the point cloud and a presynaptic range; this returns the raw CSR for that range.
//
// Splitting by presynaptic neuron is exact rather than approximate: whether a pair exists, and what weight and delay it gets, is a pure function of the two neurons' stable identities and the seed (see pairHash in nodes.js).
// No decision reads another pair, and nothing depends on the order neurons are visited, so a slice produces precisely the synapses the serial wiring would produce for those same sources, in the same order.
// The battery's determinism check compares the two paths directly.
import { wireConnect } from './nodes.js';

self.onmessage = e => {
  // capability probe: the page checks once whether this file can run at all before committing to a pool, since some embeddings refuse extra workers
  if(e.data && e.data.ping){ postMessage({ pong:true }); return; }
  const { pts, params, i0, i1 } = e.data;
  try {
    const r = wireConnect(pts, params, null, { range:[i0, i1] });
    // transfer rather than copy: at full density a single slice's post array alone runs to hundreds of megabytes
    postMessage({ ok:true, r }, [
      r.deg.buffer, r.preStart.buffer, r.post.buffer, r.w.buffer,
      r.delay.buffer, r.pmask.buffer,
      ...(r.KE ? [r.KE.buffer, r.KI.buffer] : []),
    ]);
  } catch(err){
    postMessage({ ok:false, message: err.message || String(err) });
  }
};

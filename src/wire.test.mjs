// The wiring realizes the probability it is given.
// A population around the origin with a flat kernel has to come out at p N (N - 1) synapses within a few percent, and no pair may appear twice: a spatial grid whose cell keys collide for cells on opposite sides of the origin would have the neighbor scan visit a colliding bucket twice and the pair hash make the synapse twice.
import { NODE_DEFS, computeNode, setResolution, wireConnect, assembleConnect } from './nodes.js';
import { ok, report, inProcessWorker } from '../tools/harness.mjs';
inProcessWorker(wireConnect, assembleConnect);

function mkEditor(){
  return { nodes:[], nextId:1, groups:[], byId(id){ return this.nodes.find(n => n.id === id); },
    addNode(t, x, y){ const def = NODE_DEFS[t], params = {}; def.params.forEach(p => params[p.k] = structuredClone(p.def));
      const n = { id:this.nextId++, type:t, x, y, params, inputs:new Array(def.inputs).fill(null) }; this.nodes.push(n); return n; } };
}
setResolution(1);
const ed = mkEditor();
const sph = ed.addNode('sphere', 0, 0); sph.params.center = [0, 0, 0]; sph.params.radius = 300;
const sc = ed.addNode('scatter', 0, 100); sc.params.count = 3000; sc.params.fill = 0; sc.params.tag = 'a'; sc.params.seed = 3;
sc.inputs[0] = { id:sph.id };
const cn = ed.addNode('connect', 0, 200);
Object.assign(cn.params, { radius:1000, sigma:100000, prob:0.1, wExc:1, wInh:-1, wdist:0, cluster:0, seed:1, table:'' });
cn.inputs[0] = { id:sc.id };
const cp = ed.addNode('checkpoint', 0, 300); cp.inputs[0] = { id:cn.id };
const net = await computeNode(cp, id => ed.byId(id));
const n = net.count, expected = 0.1*n*(n - 1);
ok('the realized count is the expectation within three percent', Math.abs(net.synCount - expected)/expected < 0.03,
  net.synCount + ' against ' + expected.toFixed(0) + ' (' + ((net.synCount/expected - 1)*100).toFixed(1) + '%)');
let dup = 0;
for(let i = 0; i < n; i++){
  const seen = new Set();
  for(let k = net.preStart[i]; k < net.preStart[i + 1]; k++){ const j = net.post[k]; if(seen.has(j)) dup++; seen.add(j); }
}
ok('no pair appears twice', dup === 0, dup + ' duplicated pairs');

// The synapse budget holds on the parallel wiring's total, not only on each slice: two slices under it that are over it together are refused with the serial wiring's message, before anything is allocated for the whole network.
{
  const slice = (i0, i1, m) => ({ i0, i1, synCount:m, deg:new Int32Array(4), post:new Int32Array(0),
    w:new Float32Array(0), delay:new Uint8Array(0), pmask:new Uint8Array(0) });
  let msg = null;
  try { assembleConnect({ count:4 }, { __maxSyn:1e6 }, [slice(0, 2, 600000), slice(2, 4, 600000)]); }
  catch(e){ msg = e.message; }
  ok('slices over the budget together are refused', /> 1M synapses/.test(msg || ''), msg || 'accepted');
}
report('wiring');

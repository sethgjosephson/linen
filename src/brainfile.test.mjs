// Checkpoint files, and the way they can be silently empty.
// A training run transfers its synapse arrays to the engine, and a transferred ArrayBuffer is detached in the sender, so the plasticity mask left behind has length 0 and reads undefined at every index: counted as "no synapse is plastic" it selects the plastic format and writes a file with none of the weights in it.
// These pin it from both ends, the detached mask and the empty result.

import { plasticCount, buildPlasticFile, buildWeightsFile,
         parseBrainFile, applyBrainWeights } from './brain.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond) pass++;
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};
const threw = fn => { try { fn(); return null; } catch(e){ return e.message; } };

const GRAPH = { nodes:[{ id:1, type:'checkpoint', params:{} }], resolution:100 };
const makeNet = (m, maskFill) => ({
  count: 10, synCount: m,
  preStart: new Uint32Array(11), post: new Uint32Array(m),
  w: new Float32Array(m).fill(2.5), delay: new Uint8Array(m).fill(1),
  pmask: maskFill === null ? null : new Uint8Array(m).fill(maskFill),
  graph: GRAPH,
});

// --- the normal cases still work -----------------------------------------
{
  const net = makeNet(100, 1);
  ok('every synapse plastic counts them all', plasticCount(net) === 100);
  const net2 = makeNet(100, 0);
  ok('none plastic counts zero', plasticCount(net2) === 0);
  ok('no mask at all is null, not zero', plasticCount(makeNet(100, null)) === null);

  const mixed = makeNet(100, 0);
  for(let i = 0; i < 30; i++) mixed.pmask[i] = 1;
  ok('a partial mask counts the plastic ones', plasticCount(mixed) === 30);

  const buf = buildPlasticFile(GRAPH, mixed, mixed.w, 1000, null);
  const p = parseBrainFile(buf);
  ok('a plastic file records the full synapse count', p.m === 100, p.m);
  ok('and carries exactly the plastic weights', p.k === 30, p.k);
  ok('and the weights are the real values', p.w.length === 30 && p.w[0] === 2.5);
}

// --- the detached mask ----------------------------------------------------
{
  // Exactly what postMessage with a transfer list leaves behind: the view survives, its buffer does not.
  const net = makeNet(100, 1);
  structuredClone(net.pmask.buffer, { transfer: [net.pmask.buffer] });
  ok('a detached mask really does read as length 0', net.pmask.length === 0,
     String(net.pmask.length));
  ok('and every index reads undefined', net.pmask[0] === undefined);

  const msg = threw(() => plasticCount(net));
  ok('plasticCount refuses a detached mask instead of returning 0',
     msg !== null, 'returned ' + plasticCount0(net));
  ok('and the error says what happened', /detached/.test(msg || ''), msg);
  ok('and names both counts so the mismatch is visible',
     /0 entries for 100/.test(msg || ''), msg);

  const msg2 = threw(() => buildPlasticFile(GRAPH, net, net.w, 1000, null));
  ok('buildPlasticFile refuses it too', msg2 !== null);
  ok('with the same explanation', /detached/.test(msg2 || ''), msg2);
}

// --- an empty plastic file is never the right answer ----------------------
{
  const net = makeNet(100, 0);           // a real mask, nothing marked plastic
  const msg = threw(() => buildPlasticFile(GRAPH, net, net.w, 1000, null));
  ok('a plastic file holding no weights is refused', msg !== null);
  ok('and points at the format that would work',
     /buildWeightsFile/.test(msg || ''), msg);
  const p = parseBrainFile(buildWeightsFile(GRAPH, net, net.w, 1000, null));
  ok('and that file carries every weight', p.m === 100 && p.w.length === 100);
}

// --- receptor channels on frozen synapses ---------------------------------
{
  // The byte is the rule in the low five bits and the receptor channel in the top three, so a frozen synapse on a stream with a receptor node is nonzero.
  // The file has to carry the plastic ones only, the count the loader walks, and a reload has to put them back.
  const net = makeNet(4, 0);
  net.pmask.set([1 | (2 << 5), 0 | (2 << 5), 1, 0]);
  const w = Float32Array.from([7, 8, 9, 10]);
  const p = parseBrainFile(buildPlasticFile(GRAPH, net, w, 1000, null));
  ok('a frozen synapse with a channel is not written', p.k === plasticCount(net) && p.k === 2,
     'file ' + p.k + ', plasticCount ' + plasticCount(net));
  const back = threw(() => applyBrainWeights(net, p));
  ok('and the file reloads', back === null, back);
  if(back === null){
    const r = applyBrainWeights(net, p);
    ok('with the plastic weights restored and the frozen ones as wired',
       Array.from(r.net.w).join(',') === '7,2.5,9,2.5', Array.from(r.net.w).join(','));
  }
}

// --- the shape of the failure in one assertion ----------------------------
{
  // The failing arithmetic: a detached mask counts 0, and 0 is less than 90% of the synapses, which would choose the compact format.
  const synCount = 37723, pkFromDetachedMask = 0;
  ok('0 < 90% of the synapses, which would choose the empty format',
     pkFromDetachedMask < synCount*0.9);
  ok('and requiring more than zero rejects it',
     !(pkFromDetachedMask > 0 && pkFromDetachedMask < synCount*0.9));
}

function plasticCount0(net){            // the plain count, for the message
  let k = 0;
  for(let s = 0; s < net.synCount; s++) if(net.pmask[s]) k++;
  return k;
}

console.log(fail ? fail + ' failed, ' + pass + ' passed'
                 : 'all passed (' + pass + ' checks)');
process.exit(fail ? 1 : 0);

// Engine throughput benchmark.
// Wires a synthetic net (uniform sphere, Gaussian kernel sized for a target in-degree) and times sim throughput on the selected engine.
// URL params: n (neurons), k (target in-degree), engine (cpu | gpu), ms (simulated ms), steps (per tick message), amp (noise amplitude).
// Results print to the page and the console.
import { NODE_DEFS, NEURON_TYPES, wireConnect, engineTypes } from './nodes.js';
import { PROTOCOL } from './protocol.js';
import { initState } from './rand.js';

const Q = new URLSearchParams(location.search);
const N = parseInt(Q.get('n') || '50000', 10);
const K = parseFloat(Q.get('k') || '600');
const ENGINE = Q.get('engine') || 'cpu';
const MS = parseInt(Q.get('ms') || '3000', 10);
const STEPS = parseInt(Q.get('steps') || '50', 10);
const AMP = parseFloat(Q.get('amp') || '14');
const SIG = parseFloat(Q.get('sig') || '80');
const PLAST = Q.get('plast') === '1';

const out = document.getElementById('out');
const log = s => { out.textContent += s + '\n'; console.log(s); };

async function main(){
  log(`neurons ${N}, target in-degree ${K}, engine ${ENGINE}, ${MS} sim-ms, ` +
      `${STEPS} steps/tick${PLAST ? ', plasticity ON' : ''}`);
  const R = 400 * Math.cbrt(N / 50000);          // constant density across sizes
  const sigma = SIG;
  const rho = N / (4/3 * Math.PI * R*R*R);
  // expected in-degree under the kernel: prob * rho * (2 pi sigma^2)^(3/2)
  const prob = Math.min(0.9, K / (rho * Math.pow(2*Math.PI*sigma*sigma, 1.5)));
  const geo = NODE_DEFS.sphere.compute([], { center:[0,0,0], radius:R });
  const pts = NODE_DEFS.scatter.compute([geo],
    { count:N, type:0, seed:1, spacing:0, pattern:0, pitch:40, jitter:8, axis:1 }, { id:1 });
  for(let i=0;i<N;i++) if(i % 5 === 0) pts.ntype[i] = 3;   // 20% FS
  log(`wiring: radius ${R.toFixed(0)} um, kernel prob ${prob.toFixed(3)} ...`);
  let t0 = performance.now();
  const net = wireConnect(pts, { radius:sigma*3, sigma, prob, wExc:0.6, wInh:-2.4,
    wdist:0, wsigma:1, cluster:0, velocity:200, density:100, nuE:0, nuI:0, seed:1,
    __maxSyn:4e8 }, null);
  const wireS = (performance.now()-t0)/1000;
  log(`wired ${net.synCount.toLocaleString()} synapses in ${wireS.toFixed(1)} s ` +
      `(mean in-degree ${(net.synCount/N).toFixed(0)})`);

  const file = ENGINE === 'gpu' ? './gpuworker.js' : './simworker.js';
  const w = new Worker(new URL(file, import.meta.url), { type:'module' });
  const allIdx = Uint32Array.from({ length:N }, (_, i) => i);
  w.postMessage({ cmd:'init', protocol:PROTOCOL, count:N, ntype:pts.ntype, bias:net.bias,
    syn:0, tauE:3, tauI:8, plast:PLAST ? 1 : 0,
    aP:0.01, aM:0.012, tauS:20, wmax:10, iEta:0.002, iRho:5,
    protocols:[{ idx:allIdx, amp:AMP, mode:3, period:100, width:10, t0:0, duration:0 }],
    preStart:net.preStart, post:net.post, w:net.w, delay:net.delay,
    types:engineTypes(),
    seed:1, ...initState(1, pts.ntype, NEURON_TYPES) });

  let spikes = 0, t = 0, warm = true, tStart = 0;
  const WARM_MS = Math.min(500, MS);
  await new Promise((resolve, reject) => {
    w.onmessage = e => {
      if(e.data.cmd === 'error'){ reject(new Error(e.data.message)); return; }
      if(e.data.cmd !== 'state') return;
      t += e.data.steps;
      if(warm){
        if(t >= WARM_MS){ warm = false; t = 0; spikes = 0; tStart = performance.now(); }
      }
      else spikes += e.data.spikes;
      if(!warm && t >= MS){ resolve(); return; }
      w.postMessage({ cmd:'tick', steps:STEPS });
    };
    w.postMessage({ cmd:'tick', steps:STEPS });
  });
  const wallS = (performance.now()-tStart)/1000;
  w.terminate();
  const rate = spikes / N / (MS/1000);
  const deliveries = spikes * (net.synCount/N);
  log(`sim: ${MS} sim-ms in ${wallS.toFixed(2)} s wall = ` +
      `${(MS/1000/wallS).toFixed(2)}x realtime, ${(MS/wallS).toFixed(0)} sim-ms/s`);
  log(`activity: ${rate.toFixed(1)} Hz mean rate, ${spikes.toLocaleString()} spikes`);
  log(`throughput: ${(deliveries/wallS/1e6).toFixed(1)} M deliveries/s, ` +
      `${(N*MS/wallS/1e6).toFixed(1)} M neuron-updates/s`);
}
main().catch(e => log('ERROR: ' + e.message));

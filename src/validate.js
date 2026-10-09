// Validation battery: measures simulator statistics against published reference values.
// Run from validate.html.
// Each test states its source and tolerance band; bands are loose where the model class (point neurons, current-based synapses in every rate scene, since the conductance mode exists but no rate scene uses it) cannot match absolute numbers, and the report says so.
// The scenes that declare exponential synapses (syn 1, psc
// 1) use them because one-millisecond kicks make a
// neuron a same-millisecond coincidence detector and swing a column between silence and bursting; the balanced random net, sound localization, seizure and control and polychrony stay on kick synapses.
import { EQUIV, rateWithin, cvWithin, syncWithin } from './equivalence.js';
import { engineConfig, NODE_DEFS, NEURON_TYPES, wireConnect, computeNode, splinePoly,
  wirePoolAvailable, wireWorkerCount, wireSlices, assembleConnect,
  buildRuleTable, engineTypes, nodeDoc } from './nodes.js';
import { initState } from './rand.js';
import { classicRow, step2003 } from './form.js';
import { PROTOCOL, checkHello, TERM_NAMES } from './protocol.js';
import { mean as smean, std as sstd } from './spikestats.js';
import { corticalColumn, balancedNet, guidedTour, travelingWaves, SCENARIOS as TISSUE_SCENES } from './scenarios.js';
import { BarSweep, encodeGains } from './io.js';
import { ALL_SCENARIOS as SCENARIOS } from './experiments.js';
import { buildBrainFile, buildPlasticFile, parseBrainFile, applyBrainWeights,
  plasticCount } from './brain.js';
import { buildExpr } from './expr.js';
import { scheduleAt, renderItem, renderAudio, SETS } from './curriculum.js';
import { selectivityProfile, decodeAccuracy, completionScore,
  responseProfile, ensembleCorrelation, likeToLike, popLinks,
  decodeShape } from './analysis.js';


// the built-in rows, for the groups that make their own small networks; a run reads the table when it starts (simRun), since a scene's cell type node adds a row at compute time
const TYPES = engineTypes();
// ---- helpers ----------------------------------------------------------
function fakeEditor(){
  const nodes = [];
  return {
    nodes, nextId: 1,
    addNode(type, x, y){
      const def = NODE_DEFS[type], params = {};
      def.params.forEach(p => params[p.k] = structuredClone(p.def));
      const n = { id:this.nextId++, type, x, y, params, inputs:new Array(def.inputs).fill(null) };
      nodes.push(n); return n;
    },
  };
}
async function computeScenario(build, mutate){
  const ed = fakeEditor();
  build(ed);
  if(mutate) mutate(ed);
  const out = ed.nodes.find(n => n.type === 'checkpoint');
  return computeNode(out, id => ed.nodes.find(n => n.id === id));
}
function simRun(net, ms, { steps = 2, perNeuron = false, isiK = 0, onFrame = null,
    weights = false, gpu = false, partCap = 0, driver = null, potentials = false, protocolOverride,
    // the engine's noise and initial state, so one computed network can be run as several draws without wiring it again
    seed } = {}){
  const sd = seed === undefined ? net.seed : seed;
  return new Promise(resolve => {
    const w = new Worker(new URL((gpu ? './gpuworker.js' : './simworker.js') + '?p=' + PROTOCOL,
      import.meta.url), { type:'module' });
    // An engine that stops answering is indistinguishable on the page from a slow group.
    // A timed out run is reported as an error and fails its check, the same as any other engine error.
    const label = `${gpu ? 'gpu' : 'cpu'} run, ${net.count} neurons, ${ms} ms`;
    let done = false;
    const finish = r => { if(done) return; done = true;
      clearTimeout(guard); w.terminate(); resolve(r); };
    const guard = setTimeout(() => finish({ frames:[], counts:null, isis:[], steps,
      error:`engine did not finish within 120s (${label})` }), 120000);
    w.postMessage({ cmd:'init', count:net.count, ntype:net.ntype, bias:net.bias,
      ...engineConfig(net),
      ...(protocolOverride !== undefined ? { protocol:protocolOverride } : {}),
      pmask:net.pmask, __partCap:partCap || undefined,
      // a caller that built its own rule table wins over the one engineConfig derives; without either every engine synthesizes row 1
      ...(net.ruleTable ? { ruleTable:net.ruleTable, ruleCount:net.ruleCount } : {}),
      inputs:net.inputs || [],
      preStart:net.preStart, post:net.post, w:net.w, delay:net.delay, types:net.types || engineTypes(),
      seed:sd, ...initState(sd, net.ntype, net.types || engineTypes()) });
    const counts = perNeuron ? new Uint32Array(net.count) : null;
    const last = isiK ? new Int32Array(isiK).fill(-1) : null;
    const isis = [];
    const frames = [];
    let t = 0, parts = 0, lastV = null;
    if(potentials) w.postMessage({ cmd:'sendV', on:true });
    w.onmessage = e => {
      if(e.data.cmd === 'error'){
        finish({ frames, counts, isis, steps, error:e.data.message }); return;
      }
      if(e.data.cmd === 'ready'){ parts = e.data.parts || 0; return; }
      if(e.data.cmd === 'weights'){
        finish({ frames, counts, isis, steps, w:e.data.w, parts }); return;
      }
      if(e.data.cmd !== 'state') return;
      const fired = e.data.fired;
      if(e.data.v) lastV = e.data.v;
      frames.push(e.data.spikes);
      if(onFrame) onFrame(fired, t);
      // fired is a count per tick on every engine (ENGINE.md), not a flag
      if(counts) for(let i=0;i<fired.length;i++) counts[i] += fired[i];
      if(last) for(let i=0;i<isiK;i++) if(fired[i]){
        if(last[i] >= 0) isis.push(t - last[i]);
        last[i] = t;
      }
      t += steps;
      if(t >= ms){
        if(weights){ w.postMessage({ cmd:'getWeights' }); return; }
        finish({ frames, counts, isis, steps, parts, v:lastV }); return;
      }
      if(driver) driver(w, t);
      w.postMessage({ cmd:'tick', steps });
    };
    if(driver) driver(w, 0);
    w.postMessage({ cmd:'tick', steps });
  });
}
// Charge-to-peak conversion for the hand-built exp-mode nets below, the same factors scenarios.js applies: their weights were written as charges under the 1/tau scale, and psc 1 delivers exactly the charge only when the numbers carry the ratio (ENGINE.md section 3). tauE 3, tauI 8.
const EQ = 1/(3*(1 - Math.exp(-1/3)));
const IQ = 1/(8*(1 - Math.exp(-1/8)));
// spikestats.js owns the statistics (population std, ddof 0, as Elephant); the battery only adds the convention that an empty sample reads 0.
const mean = a => a.length ? smean(a) : 0;
const std = a => a.length ? sstd(a) : 0;
function mkPts(n, radius, seed, inhFrac = 0.2){
  const geo = NODE_DEFS.sphere.compute([], { center:[0,0,0], radius });
  const pts = NODE_DEFS.scatter.compute([geo],
    { count:n, type:0, seed, spacing:0, pattern:0, pitch:40, jitter:8, axis:1 }, { id:1 });
  if(inhFrac > 0)
    for(let i=0;i<n;i++) if(i % Math.round(1/inhFrac) === 0) pts.ntype[i] = 3;
  return pts;
}
const CN = { radius:200, sigma:120, prob:0.4, wExc:5, wInh:-10, wdist:0, wsigma:1,
  cluster:0, velocity:200, density:100, nuE:0, nuI:0, seed:1 };

// ---- tests ------------------------------------------------------------
export const TESTS = [

{ name:'determinism', source:'internal invariant (basis for the pair-hash regression)',
  async run(){
    const pts = mkPts(4000, 350, 7);
    const a = wireConnect(pts, { ...CN }, null);
    const b = wireConnect(pts, { ...CN }, null);
    let same = a.synCount === b.synCount;
    for(let s=0;s<a.synCount && same;s+=97)
      same = a.post[s] === b.post[s] && a.w[s] === b.w[s] && a.delay[s] === b.delay[s];
    const c = wireConnect(pts, { ...CN, seed:2 }, null);
    return [
      { label:'same seed, identical network', measured: same ? 'identical' : 'DIFFERS',
        expected:'identical', pass: same },
      { label:'different seed, different network', measured: c.synCount + ' vs ' + a.synCount,
        expected:'not identical', pass: c.synCount !== a.synCount },
    ];
  } },

{ name:'parallel wiring matches the serial wiring', source:'internal invariant. The sweep is split across a pool of workers by presynaptic neuron. That is only sound because every decision in it is a pure function of the stable identities of the two neurons and the seed, so a slice emits exactly the synapses the whole-range wiring would emit for those sources, in the same order. This gates that claim rather than trusting it: the two paths must agree on every array, not approximately and not statistically. Cluster boost is on here, because triadic closure walks the assembled network and would expose any mis-ordering in it',
  async run(){
    const pts = mkPts(4000, 350, 7);
    const P = { ...CN, cluster:0.2, wdist:1, wsigma:1 };
    const serial = wireConnect(structuredClone(pts), P, null);
    let parallel = null, why = '';
    try {
      if(!(await wirePoolAvailable())) why = 'wiring pool unavailable in this browser';
      else {
        const W = Math.min(4, wireWorkerCount());
        const pool = [];
        for(let k=0;k<W;k++)
          pool.push(new Worker(new URL('./wireworker.js', import.meta.url), { type:'module' }));
        try {
          const parts = await wireSlices(structuredClone(pts), P, null, pool);
          parallel = assembleConnect(structuredClone(pts), P, parts);
        } finally { for(const w of pool) w.terminate(); }
      }
    } catch(e){ why = e.message; }
    if(!parallel) return [
      { label:'parallel wiring ran', measured: why || 'no result',
        expected:'a pool of wiring workers', pass:false } ];
    const cmp = (a, b) => {
      if(a.length !== b.length) return 'length ' + a.length + ' vs ' + b.length;
      for(let k=0;k<a.length;k++) if(a[k] !== b[k]) return 'differs at ' + k;
      return 'identical';
    };
    const rows = [
      ['synapse count', serial.synCount === parallel.synCount
        ? 'identical' : serial.synCount + ' vs ' + parallel.synCount],
      ['preStart', cmp(serial.preStart, parallel.preStart)],
      ['post', cmp(serial.post, parallel.post)],
      ['weights', cmp(serial.w, parallel.w)],
      ['delays', cmp(serial.delay, parallel.delay)],
      ['plasticity mask', cmp(serial.pmask, parallel.pmask)],
      ['bias (proxy compensation)', cmp(serial.bias, parallel.bias)],
    ];
    return rows.map(([label, measured]) => ({ label, measured,
      expected:'byte-identical to the serial wiring', pass: measured === 'identical' }));
  } },

{ name:'pair-hash additive stability', source:'design rule: edits are additive, existing pairs never silently rewire',
  async run(){
    const mk = (id, cx, n, seed) => {
      const geo = NODE_DEFS.sphere.compute([], { center:[cx,0,0], radius:300 });
      return NODE_DEFS.scatter.compute([geo],
        { count:n, type: seed % 2 ? 0 : 3, seed, spacing:0, pattern:0, pitch:40, jitter:8, axis:1 },
        { id });
    };
    const A = mk(11, -150, 1500, 1), B = mk(22, 150, 1500, 2), C = mk(33, 0, 1000, 3);
    const P = { ...CN, prob:0.3, wdist:1 };
    const pairMap = net => {
      const map = new Map();
      for(let i=0;i<net.count;i++)
        for(let s=net.preStart[i]; s<net.preStart[i+1]; s++){
          const j = net.post[s];
          map.set(net.src[i]+':'+net.lidx[i]+'>'+net.src[j]+':'+net.lidx[j], net.w[s]);
        }
      return map;
    };
    const nAB = wireConnect(NODE_DEFS.gather.compute([A, B, null, null]), P, null);
    const nABC = wireConnect(NODE_DEFS.gather.compute([A, B, C, null]), P, null);
    const nBA = wireConnect(NODE_DEFS.gather.compute([B, A, null, null]), P, null);
    const nHi = wireConnect(NODE_DEFS.gather.compute([A, B, null, null]), { ...P, prob:0.5 }, null);
    const mAB = pairMap(nAB), mABC = pairMap(nABC), mBA = pairMap(nBA), mHi = pairMap(nHi);
    let preserved = true, weightsStable = true, orderFree = mAB.size === mBA.size, superset = true;
    for(const [k, w] of mAB){
      if(!mABC.has(k)) preserved = false;
      else if(mABC.get(k) !== w) weightsStable = false;
      if(!mBA.has(k) || mBA.get(k) !== w) orderFree = false;
      if(!mHi.has(k)) superset = false;
      if(!(preserved && weightsStable && orderFree && superset)) break;
    }
    return [
      { label:'adding a population preserves existing pairs', measured:
          `${mAB.size} pairs before, ${mABC.size} after adding 1000 neurons`,
        expected:'all original pairs present', pass: preserved },
      { label:'weights of surviving pairs unchanged', measured: weightsStable ? 'identical' : 'CHANGED',
        expected:'identical (lognormal draws are per-pair)', pass: weightsStable },
      { label:'merge order does not rewire', measured: `${mAB.size} vs ${mBA.size} pairs`,
        expected:'identical networks in identity space', pass: orderFree },
      { label:'raising probability is a superset', measured: `${mAB.size} into ${mHi.size}`,
        expected:'every low-prob pair exists at high prob', pass: superset },
    ];
  } },

{ name:'population-pair tables', source:'mechanism for Potjans-Diesmann style pair-specific connectivity; additive with pair-hash',
  async run(){
    const mk = (id, tag, type, seed) => {
      const geo = NODE_DEFS.sphere.compute([], { center:[0,0,0], radius:300 });
      return NODE_DEFS.scatter.compute([geo],
        { count:1500, type, tag, seed, spacing:0, pattern:0, pitch:40, jitter:8, axis:1 }, { id });
    };
    const merged = () => NODE_DEFS.gather.compute([mk(11,'A',0,1), mk(22,'B',3,2), null, null]);
    const P = { ...CN, prob:0.3 };
    const classCounts = net => {
      const c = { };
      for(let i=0;i<net.count;i++)
        for(let s=net.preStart[i]; s<net.preStart[i+1]; s++){
          const k = (net.src[i]===11?'A':'B') + '>' + (net.src[net.post[s]]===11?'A':'B');
          c[k] = (c[k]||0)+1;
        }
      return c;
    };
    const pairW = net => {
      const m = new Map();
      for(let i=0;i<net.count;i++)
        for(let s=net.preStart[i]; s<net.preStart[i+1]; s++)
          m.set(net.src[i]+':'+net.lidx[i]+'>'+net.src[net.post[s]]+':'+net.lidx[net.post[s]], net.w[s]);
      return m;
    };
    const base = wireConnect(merged(), P, null);
    const c0 = classCounts(base), w0 = pairW(base);
    const killAA = wireConnect(merged(), { ...P, table:'A A 0' }, null);
    const c1 = classCounts(killAA), w1 = pairW(killAA);
    let othersIdentical = true;
    for(const [k, w] of w1) if(!k.startsWith('11:') || !k.includes('>11:')){
      if(w0.get(k) !== w) { othersIdentical = false; break; }
    }
    const boostAB = wireConnect(merged(), { ...P, table:'A B 1 2.5' }, null);
    const w2 = pairW(boostAB);
    let scaled = true, checked = 0;
    for(const [k, w] of w0){
      const isAB = k.startsWith('11:') && k.includes('>22:');
      const expect = isAB ? w*2.5 : w;
      if(Math.abs(w2.get(k) - expect) > Math.abs(expect)*1e-5){ scaled = false; break; }
      checked++;
    }
    const half = wireConnect(merged(), { ...P, table:'E I 0.5' }, null);
    const c3 = classCounts(half);
    const spec = wireConnect(merged(), { ...P, table:'E I 0.5\nA B 1' }, null);
    const c4 = classCounts(spec);
    return [
      { label:'probMul 0 removes the class', measured: 'A>A: ' + (c1['A>A']||0) + ' (was ' + c0['A>A'] + ')',
        expected:'0', pass: !(c1['A>A']) },
      { label:'other classes byte-identical under table edit', measured: othersIdentical ? 'identical' : 'CHANGED',
        expected:'identical', pass: othersIdentical },
      { label:'wMul scales exactly one class', measured: checked + ' pairs checked',
        expected:'A>B x2.5, others unchanged', pass: scaled },
      { label:'class key halves the class', measured: 'A>B: ' + c3['A>B'] + ' vs ' + c0['A>B'],
        expected:'ratio 0.4-0.6', pass: c3['A>B'] > 0.4*c0['A>B'] && c3['A>B'] < 0.6*c0['A>B'] },
      { label:'tag row overrides class row', measured: 'A>B: ' + c4['A>B'] + ' vs ' + c0['A>B'],
        expected:'equal (specificity wins)', pass: c4['A>B'] === c0['A>B'] },
    ];
  } },

{ name:'lognormal weights', source:'Song et al. 2005 (PLOS Biol): lognormal EPSP amplitudes, sigma near 1',
  async run(){
    const pts = mkPts(4000, 350, 7, 0);           // excitatory only, keeps signs uniform
    const net = wireConnect(pts, { ...CN, wdist:1, wsigma:1, wExc:5 }, null);
    const logs = [];
    let m = 0;
    for(let s=0;s<net.synCount;s++){ m += net.w[s]; logs.push(Math.log(net.w[s]/5)); }
    m /= net.synCount;
    const mu = mean(logs), sg = std(logs);
    return [
      { label:'mean weight (mean-preserving)', measured: m.toFixed(3), expected:'5.00 ± 2%',
        pass: Math.abs(m-5)/5 < 0.02 },
      { label:'sigma of log-weights', measured: sg.toFixed(3), expected:'1.0 ± 0.05',
        pass: Math.abs(sg-1) < 0.05 },
      { label:'mu of log-weights (=-sigma^2/2)', measured: mu.toFixed(3), expected:'-0.5 ± 0.05',
        pass: Math.abs(mu+0.5) < 0.05 },
    ];
  } },

{ name:'cluster boost raises transitivity', source:'Perin, Berger, Markram 2011 (PNAS): above-chance clustering / common neighbors',
  async run(){
    const pts = mkPts(3000, 300, 7);
    const flat = wireConnect(pts, { ...CN, prob:0.25 }, null);
    const boosted = wireConnect(pts, { ...CN, prob:0.25, cluster:0.4 }, null);
    const trans = net => {                      // sampled directed transitivity
      const outSets = [];
      for(let i=0;i<net.count;i++){
        const s = new Set();
        for(let k=net.preStart[i]; k<net.preStart[i+1]; k++) s.add(net.post[k]);
        outSets.push(s);
      }
      let tri = 0, wedges = 0;
      for(let i=0;i<net.count;i+=3){
        const oi = [...outSets[i]];
        for(let a2=0;a2<Math.min(oi.length,20);a2++)
          for(let b2=a2+1;b2<Math.min(oi.length,20);b2++){
            wedges++;
            if(outSets[oi[a2]].has(oi[b2]) || outSets[oi[b2]].has(oi[a2])) tri++;
          }
      }
      return tri/Math.max(1,wedges);
    };
    const t0 = trans(flat), t1 = trans(boosted);
    return [
      { label:'transitivity, flat vs boosted', measured: t0.toFixed(3) + ' vs ' + t1.toFixed(3),
        expected:'boosted > 1.15 x flat', pass: t1 > 1.15*t0 },
      { label:'synapse count ratio at cluster 0.4', measured: (boosted.synCount/flat.synCount).toFixed(2),
        expected:'1.40 ± 0.05', pass: Math.abs(boosted.synCount/flat.synCount - 1.4) < 0.05 },
    ];
  } },

{ name:'spatial sampling statistics', source:'Waessle and Riemann 1978 (mosaic regularity); Mountcastle 1997 (minicolumns)',
  async run(){
    const geo = NODE_DEFS.sphere.compute([], { center:[0,0,0], radius:400 });
    const base = { count:2500, type:0, seed:1, spacing:0, pattern:0, pitch:50, jitter:6, axis:1 };
    const nn = pts => {                          // nearest-neighbor distances (sampled)
      const d = [];
      for(let i=0;i<600;i++){
        let best = 1e9;
        for(let j=0;j<pts.count;j++){
          if(j===i) continue;
          const dx=pts.pos[i*3]-pts.pos[j*3], dy=pts.pos[i*3+1]-pts.pos[j*3+1], dz=pts.pos[i*3+2]-pts.pos[j*3+2];
          const q = Math.sqrt(dx*dx+dy*dy+dz*dz); if(q<best) best=q;
        }
        d.push(best);
      }
      return d;
    };
    const uni = NODE_DEFS.scatter.compute([geo], base, { id:1 });
    const spaced = NODE_DEFS.scatter.compute([geo], { ...base, spacing:25 }, { id:1 });
    const dU = nn(uni), dS = nn(spaced);
    const riU = mean(dU)/std(dU), riS = mean(dS)/std(dS);   // regularity index
    const mc = NODE_DEFS.scatter.compute([geo], { ...base, pattern:1 }, { id:1 });
    let res = 0;
    for(let i=0;i<mc.count;i++){
      const row = Math.round(mc.pos[i*3]/(50*0.866));
      const col = Math.round(mc.pos[i*3+2]/50 - (row&1?0.5:0));
      res += Math.abs(mc.pos[i*3+2] - (col+(row&1?0.5:0))*50);
    }
    res /= mc.count;
    return [
      { label:'min spacing enforced', measured: Math.min(...dS).toFixed(1) + ' µm',
        expected:'>= 25 µm', pass: Math.min(...dS) >= 25 },
      { label:'regularity index, spaced vs uniform', measured: riS.toFixed(2) + ' vs ' + riU.toFixed(2),
        expected:'spaced > uniform (mosaics are more regular than random)', pass: riS > riU },
      { label:'minicolumn strand residual', measured: res.toFixed(1) + ' µm',
        expected:'< 5 µm at jitter 6', pass: res < 5 },
    ];
  } },

{ name:'Brunel regime separation', source:'Brunel 2000 (J Comput Neurosci): inhibition-dominated asynchronous irregular vs excitation-dominated synchronized',
  async run(){
    // g is the ratio to the scene's own unitary EPSP, whatever that is
    const runReg = async g => {
      const net = await computeScenario(balancedNet, ed => {
        const cn = ed.nodes.find(n => n.type==='connect'); cn.params.wInh = -g * cn.params.wExc;
      });
      const r = await simRun(net, 3000, { steps:1, isiK:300 });
      const f = r.frames.slice(500);
      const rate = mean(f)/net.count*1000;
      const fano = std(f)*std(f)/Math.max(0.001, mean(f));
      const cv = r.isis.length > 50 ? std(r.isis)/mean(r.isis) : 0;
      return { rate, fano, cv };
    };
    const ai = await runReg(5);                 // g = 5, inhibition-dominated
    const sr = await runReg(1);                 // g = 1, excitation-dominated
    return [
      { label:'AI state rate (g=5)', measured: ai.rate.toFixed(1) + ' Hz',
        expected:'1-30 Hz (sparse)', pass: ai.rate > 1 && ai.rate < 30 },
      { label:'AI state CV of ISI', measured: ai.cv.toFixed(2),
        expected:'0.45-1.5. Brunel’s CV of about 1 assumes LIF; adaptation on the RS07 row (d 100 pA on C 100 pF, Izhikevich 2007) regularizes ISIs below Poisson (Benda and Herz 2003), so sub-1 values are expected here',
        pass: ai.cv > 0.45 && ai.cv < 1.5 },
      { label:'synchrony (population Fano), SR vs AI', measured: sr.fano.toFixed(1) + ' vs ' + ai.fano.toFixed(1),
        expected:'SR > 3 x AI', pass: sr.fano > 3*ai.fano },
      { label:'rate, SR vs AI', measured: sr.rate.toFixed(1) + ' vs ' + ai.rate.toFixed(1) + ' Hz',
        expected:'SR > AI', pass: sr.rate > ai.rate },
    ];
  } },

{ name:'the engine contract has a version', source:'ENGINE.md section 2: every init carries protocol, the number of the contract the sender was written against, and an engine refuses any other number with a message naming both, so a page and an engine loaded from different builds fail loudly rather than run each other bytes.',
  run: async () => {
    const n = 8;
    const mk = () => ({ kind:'net', count:n, pos:new Float32Array(3*n), ntype:new Uint8Array(n), bias:new Float32Array(n).fill(6),
      src:new Int32Array(n).fill(1), lidx:Uint32Array.from({ length:n }, (_, i) => i),
      preStart:new Int32Array(n+1), post:new Int32Array(0), w:new Float32Array(0), delay:new Uint8Array(0), synCount:0,
      plast:0, refrac:2, seed:5, syn:0, tauE:3, tauI:8, protocols:[] });
    const good = await simRun(mk(), 300, { steps:1, perNeuron:true });
    const old = await simRun(mk(), 300, { steps:1, perNeuron:true, protocolOverride:0 });
    const none = await simRun(mk(), 300, { steps:1, perNeuron:true, protocolOverride:null });
    const out = [
      { label:'the current number runs', measured: good.error || ('ran, ' + [...good.counts].reduce((a, b) => a + b, 0) + ' spikes'), expected:'no error', pass: !good.error },
      { label:'another number is refused, naming both', measured: old.error || 'ran', expected:'an error naming contract 0 and contract ' + PROTOCOL, pass: /contract 0/.test(old.error || '') && new RegExp('contract ' + PROTOCOL).test(old.error || '') },
      { label:'an init without the field is refused', measured: none.error || 'ran', expected:'an error naming contract none', pass: /contract none/.test(none.error || '') },
    ];
    // hello (ENGINE.md section 2): each engine names its contract, every term, and the optional queries it answers
    const helloOf = gpu => new Promise(res => {
      const w = new Worker(new URL((gpu ? './gpuworker.js' : './simworker.js') + '?p=' + PROTOCOL, import.meta.url), { type:'module' });
      const t = setTimeout(() => { w.terminate(); res(null); }, 5000);
      w.onmessage = e => { if(e.data.cmd === 'hello'){ clearTimeout(t); w.terminate(); res(e.data); } };
      w.postMessage({ cmd:'hello' });
    });
    const hr = await helloOf(false);
    out.push({ label:'the reference engine answers hello with this contract and every term', measured: hr ? hr.engine + ', contract ' + hr.protocol + ', ' + hr.terms.length + ' terms, ' + hr.optional.join(' ') : 'no reply',
      expected:'contract ' + PROTOCOL + ', all ' + TERM_NAMES.length + ' terms', pass: !!hr && checkHello(hr) === null && TERM_NAMES.every(x => hr.terms.includes(x)) });
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const g = await simRun(mk(), 300, { steps:1, perNeuron:true, gpu:true, protocolOverride:0 });
      out.push({ label:'the gpu engine refuses another number too', measured: g.error || 'ran', expected:'an error naming contract 0', pass: /contract 0/.test(g.error || '') });
      const hg = await helloOf(true);
      out.push({ label:'the gpu engine answers hello with every term, and without first-spike latency', measured: hg ? hg.engine + ', contract ' + hg.protocol + ', ' + hg.terms.length + ' terms, ' + hg.optional.join(' ') : 'no reply',
        expected:'contract ' + PROTOCOL + ', all ' + TERM_NAMES.length + ' terms, no sendT', pass: !!hg && checkHello(hg) === null && TERM_NAMES.every(x => hg.terms.includes(x)) && !hg.optional.includes('sendT') });
    }
    return out;
  } },
{ name:'a cell type row of the scene survives the worker wiring', source:'internal invariant. The cell type node registers its row on the thread that computes the graph; the wiring slices and the assembly run on workers with their own copy of the type table, and a row they have never seen reads as no row at all. The stream rows are registered on every path that reads a type (wireConnect, assembleConnect).',
  run: async () => {
    const net = await computeScenario(ed => {
      const add = (t, x, y, p = {}) => { const n = ed.addNode(t, x, y); Object.assign(n.params, p); return n; };
      const wire = (d, port, src) => { d.inputs[port] = { id:src.id }; };
      const sph = add('sphere', 0, 0, { center:[0, 0, 0], radius:300 });
      const sc = add('scatter', 0, 60, { fill:0, count:2500, type:0, seed:1, tag:'cells' }); wire(sc, 0, sph);
      const ct = add('celltype', 0, 120, { row:0, name:'batteryrow', tag:'cells', preset:0 }); wire(ct, 0, sc);
      const m = add('gather', 0, 180); wire(m, 0, ct);
      const cn = add('connect', 0, 240, { radius:150, sigma:80, prob:0.3, wExc:5, wInh:-5, velocity:200, seed:1 }); wire(cn, 0, m);
      const out = add('checkpoint', 0, 300, {}); wire(out, 0, cn);
    });
    const row = NEURON_TYPES.findIndex(t => t.key === 'batteryrow');
    let onRow = 0; for(let i = 0; i < net.count; i++) if(net.ntype[i] === row) onRow++;
    return [
      { label:'the wiring ran with the row in the stream', measured: net.synCount + ' synapses', expected:'a wiring, not an error', pass: net.synCount > 0 },
      { label:'every cell is on the scene row', measured: onRow + ' of ' + net.count, expected: net.count + ' of ' + net.count, pass: row >= 0 && onRow === net.count },
    ];
  } },
{ name:'the type table (Izhikevich 2007 rows)', source:'Izhikevich 2007, Dynamical Systems in Neuroscience, chapter 8: C dv/dt = k (v - vr)(v - vt) - u + I with the rest, the threshold and the cap per type (MODEL.md 1). Rows 0 to 6 are the 2003 presets as rows of this form (form.js), rows 10 to 16 the measured chapter 8 rows, between them the fly rows. Every row must rest at its vr with no drive, fire above its rheobase, and a graded row must never spike.',
  run: async () => {
    // the built-in rows only: a scene row registered by an earlier group (the cell type group) is not this table
    const ROWS = NEURON_TYPES.filter(t => !t.custom), TYPES = engineTypes(ROWS);
    const n = TYPES.length;
    const mk = bias => ({ kind:'net', count:n, pos:new Float32Array(3*n),
      ntype:Uint8Array.from(TYPES.map((_, i) => i)), bias:Float32Array.from(bias),
      src:new Int32Array(n).fill(1), lidx:Uint32Array.from(TYPES.map((_, i) => i)),
      preStart:new Int32Array(n+1), post:new Int32Array(0),
      w:new Float32Array(0), delay:new Uint8Array(0), synCount:0,
      plast:0, refrac:2, seed:5, syn:0, tauE:3, tauI:8, protocols:[] });
    // the rheobase of a 2007 row: the peak of the v-nullcline with u at its fixed point
    const rheo = TYPES.map(t => { const f = t.f7, D = f.vt - f.vr; return (f.k*D + f.b)**2 / (4*f.k); });
    // A classic row starts where the 2003 form started, c plus up to 10 mV, which for LTS, TC and RZ is above the threshold: each fires once and settles, as the 2003 form did from that start.
    // So the rest is judged after the first 100 ms.
    const late = () => { const c = new Uint32Array(n); return { c, onFrame:(fired, t) => { if(t >= 100) for(let i = 0; i < n; i++) if(fired[i]) c[i]++; } }; };
    const L0 = late();
    const rest = await simRun(mk(new Array(n).fill(0)), 400, { steps:1, perNeuron:true, potentials:true, onFrame:L0.onFrame });
    const offs = rest.v ? TYPES.map((t, i) => Math.abs(rest.v[i] - t.f7.vr)) : [];
    const drive = rheo.map(r => Math.max(3*r, 2));
    const c1 = await simRun(mk(drive), 1000, { steps:1, perNeuron:true });
    const keys = ROWS.map(t => t.key);
    const row = r => keys.map((k, i) => k + ' ' + (r.counts ? r.counts[i] : '?')).join(', ');
    const out = [
      { label:'at zero drive each row rests at its vr', measured: rest.error || keys.map((k, i) => k + ' ' + (rest.v ? rest.v[i].toFixed(1) : '?') + '/' + TYPES[i].f7.vr.toFixed(1)).join(', '),
        expected:'within 1 mV after 400 ms, no spikes after the first 100 ms', pass: !rest.error && offs.length === n && offs.every(o => o < 1) && [...L0.c].every(c => c === 0) },
      { label:'at three times the rheobase each spiking row fires and no graded one does', measured: c1.error || row(c1),
        expected:'every spiking count above zero and under 400 Hz, graded rows at zero', pass: !c1.error && [...c1.counts].every((c, i) => TYPES[i].f7.graded ? c === 0 : (c > 0 && c < 400)) },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const g1 = await simRun(mk(drive), 1000, { steps:1, perNeuron:true, gpu:true });
      const LG = late();
      const gr = await simRun(mk(new Array(n).fill(0)), 400, { steps:1, perNeuron:true, potentials:true, gpu:true, onFrame:LG.onFrame });
      const goffs = gr.v ? TYPES.map((t, i) => Math.abs(gr.v[i] - t.f7.vr)) : [];
      const close = !g1.error && !c1.error && keys.every((k, i) => Math.abs(g1.counts[i] - c1.counts[i]) <= Math.max(3, 0.2*c1.counts[i]));
      out.push(
        { label:'gpu rests at vr too', measured: gr.error || keys.map((k, i) => k + ' ' + (gr.v ? gr.v[i].toFixed(1) : '?')).join(', '),
          expected:'within 1 mV, no spikes after the first 100 ms', pass: !gr.error && goffs.length === n && goffs.every(o => o < 1) && [...LG.c].every(c => c === 0) },
        { label:'gpu parity across the rows', measured: g1.error || ('gpu ' + row(g1)),
          expected:'each count within 20% (or 3 spikes) of the cpu count', pass: close });
    }
    return out;
  } },
{ name:'the 2003 form as a 2007 row', source:'form.js: with C = 1, k = 0.04 and the recovery variable measured from the rest, the Izhikevich 2003 equations are the 2007 equations term for term (the rest at the lower root of 0.04 v^2 + (5 - b) v + 140, the instantaneous threshold at -125 - vr). Rows 0 to 6 of the type table carry the 2003 presets that way, so every scene written for the retired 2003 form runs unchanged. Kept alive here against the 2003 arithmetic itself (form.js step2003: the engines two half steps, floor and reset, on the same initial bytes), and against the numbers measured 2026-09-13 on the 2003 branch of the engines: the recurrent network below read 21.33 Hz on that branch and 21.24 Hz on these rows. Judged at ENGINE.md 4 tolerances, since the two orderings of the same polynomial differ at rounding and a spike is a cutoff crossed inside a half step.',
  run: async () => {
    const P = 7;   // RS, IB, CH, FS, LTS, TC, RZ: rows 0 to 6 are the classic rows
    const keys = NEURON_TYPES.slice(0, P).map(t => t.key);
    // --- isolated cells: every preset at three drives, engine against the 2003 stepper ---
    const drives = [4, 8, 16], n1 = P*drives.length;
    const ntype = Uint8Array.from({ length:n1 }, (_, i) => i % P);
    const bias = Float32Array.from({ length:n1 }, (_, i) => drives[(i / P) | 0]);
    const iso = () => ({ kind:'net', count:n1, pos:new Float32Array(3*n1), ntype, bias,
      src:new Int32Array(n1).fill(1), lidx:Uint32Array.from({ length:n1 }, (_, i) => i),
      preStart:new Int32Array(n1 + 1), post:new Int32Array(0), w:new Float32Array(0), delay:new Uint8Array(0), synCount:0,
      plast:0, refrac:2, seed:5, syn:0, tauE:3, tauI:8, protocols:[] });
    const firsts = () => { const f = new Int32Array(n1).fill(-1); return { f, onFrame:(fired, t) => { for(let i = 0; i < n1; i++) if(fired[i] && f[i] < 0) f[i] = t; } }; };
    // the 2003 arithmetic on the same start: the engine's v0, and u0 plus b vr (the classic row measures u from the rest), stored to float32 each step as the engine stores its state, with the engine's 2 ms refractory hold
    const st0 = initState(5, ntype, TYPES);
    const ref = { counts:new Uint32Array(n1), first:new Int32Array(n1).fill(-1) };
    for(let i = 0; i < n1; i++){
      const t = NEURON_TYPES[ntype[i]], r = TYPES[ntype[i]].f7;
      let v = st0.v0[i], u = st0.u0[i] + t.b*r.vr, hold = 0;
      for(let ms = 0; ms < 1000; ms++){
        if(hold > 0){ hold--; u = Math.fround(u + t.a*(t.b*v - u)); continue; }
        let sp; [v, u, sp] = step2003(v, u, bias[i], t, -90);
        v = Math.fround(v); u = Math.fround(u);
        if(sp){ ref.counts[i]++; if(ref.first[i] < 0) ref.first[i] = ms; hold = 2; }
      }
    }
    const F1 = firsts();
    const a1 = await simRun(iso(), 1000, { steps:1, perNeuron:true, onFrame:F1.onFrame });
    const close = (x, y) => !x.error && [...x.counts].every((c, i) => Math.abs(c - y[i]) <= Math.max(3, 0.2*c));
    const rowOf = (counts) => keys.map((k, i) => k + ' ' + drives.map((_, j) => counts[j*P + i]).join('/')).join(', ');
    const firstAgree = (fa, fb, tol) => { let same = 0, seen = 0; for(let i = 0; i < n1; i++){ if(fa[i] < 0 && fb[i] < 0) continue; seen++; if(fa[i] >= 0 && fb[i] >= 0 && Math.abs(fa[i] - fb[i]) <= tol) same++; } return { same, seen }; };
    const fs = firstAgree(ref.first, F1.f, 1);
    // --- a recurrent E/I network: 320 RS, 80 FS, twenty random synapses per cell ---
    const nE = 320, nI = 80, n2 = nE + nI, K = 20;
    let z = 12345; const rnd = () => { z = (Math.imul(z, 1664525) + 1013904223) >>> 0; return z / 4294967296; };
    const pre = [], post = [], w = [], delay = [];
    for(let i = 0; i < n2; i++) for(let k = 0; k < K; k++){
      let j = (rnd()*n2) | 0; if(j === i) j = (j + 1) % n2;
      pre.push(i); post.push(j); w.push(i < nE ? 6 : -7); delay.push(1 + ((rnd()*4) | 0));
    }
    const order = [...pre.keys()].sort((x, y) => pre[x] - pre[y] || x - y);
    const preStart = new Int32Array(n2 + 1); for(const s of order) preStart[pre[s] + 1]++; for(let i = 0; i < n2; i++) preStart[i + 1] += preStart[i];
    const rec = () => ({ kind:'net', count:n2, pos:new Float32Array(3*n2),
      ntype:Uint8Array.from({ length:n2 }, (_, i) => i < nE ? 0 : 3), bias:new Float32Array(n2).fill(6),
      src:new Int32Array(n2).fill(1), lidx:Uint32Array.from({ length:n2 }, (_, i) => i),
      preStart, post:Int32Array.from(order, s => post[s]), w:Float32Array.from(order, s => w[s]), delay:Uint8Array.from(order, s => delay[s]),
      synCount:order.length, pmask:new Uint8Array(order.length).fill(0),
      plast:0, refrac:2, seed:5, syn:0, tauE:3, tauI:8, protocols:[] });
    const rateOf = r => r.error ? -1 : [...r.counts].reduce((a, b) => a + b, 0) / r.counts.length / 1.5;
    const b1 = await simRun(rec(), 1500, { steps:2, perNeuron:true });
    const r1 = rateOf(b1), REF = 21.33;   // the 2003 branch of the reference engine on this network, 2026-09-13
    const out = [
      { label:'seven presets at three drives, isolated: the first spike lands where the 2003 arithmetic puts it', measured: a1.error || (fs.same + ' of ' + fs.seen + ' cells within a millisecond'),
        expected:'every cell that spikes', pass: !a1.error && fs.seen > 15 && fs.same === fs.seen },
      { label:'and the counts over a second agree with the 2003 arithmetic', measured: a1.error || ('2003 ' + rowOf(ref.counts) + ' | classic row ' + rowOf(a1.counts)),
        expected:'each count within 20% or 3 spikes (phase slip in the bursting presets)', pass: close(a1, ref.counts) },
      { label:'a recurrent E/I network reads what the 2003 branch read', measured: b1.error || (r1.toFixed(2) + ' Hz against ' + REF + ' recorded'),
        expected:'within 10%', pass: r1 > 1 && Math.abs(r1 - REF) <= 0.1*REF },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const FG = firsts();
      const g1 = await simRun(iso(), 1000, { steps:1, perNeuron:true, gpu:true, onFrame:FG.onFrame });
      const gb = await simRun(rec(), 1500, { steps:2, perNeuron:true, gpu:true });
      const fsg = firstAgree(F1.f, FG.f, 1);
      out.push(
        { label:'gpu parity on the classic rows: first spikes within a millisecond', measured: g1.error || (fsg.same + ' of ' + fsg.seen + ' cells'),
          expected:'at least 90% of cells', pass: !g1.error && fsg.seen > 15 && fsg.same >= 0.9*fsg.seen },
        { label:'gpu parity on the classic rows, isolated counts', measured: g1.error || ('gpu ' + rowOf(g1.counts)),
          expected:'each count within 20% or 3 spikes of the cpu run', pass: !g1.error && close(g1, a1.counts) },
        { label:'gpu parity on the classic rows, the recurrent network', measured: gb.error || ('gpu ' + rateOf(gb).toFixed(2) + ' Hz vs cpu ' + r1.toFixed(2)),
          expected:'within 10%', pass: !gb.error && Math.abs(rateOf(gb) - r1) <= 0.1*r1 });
    }
    return out;
  } },
{ name:'conductance synapses', source:'conductance-based synapses (Brette et al. 2007, J Comput Neurosci, the COBA benchmark): a synapse is a conductance toward its reversal potential, I = g (E - v). Excitation toward 0 mV weakens as the cell depolarizes; a chloride conductance with its reversal near rest carries little current on its own and divides the response to excitation, a shunt. This is the interaction Groschner et al. 2022 (Nature 603:119) found under T4 direction selectivity. Gated on both page engines',
  run: async () => {
    // N spiking sources onto N targets; half the targets also get a shunting inhibitory synapse from a tonically active inhibitory source
    const N = 200, FL = TYPES.findIndex((t, i) => NEURON_TYPES[i].key === 'FL');
    const mk = (syn, gI, eRevI) => {
      const n = 3*N; const ntype = new Uint8Array(n).fill(FL); const bias = new Float32Array(n);
      for(let i = 0; i < N; i++){ bias[i] = 8 + 4*(i/N); bias[2*N + i] = 12; }
      // sources 0..N-1 -> targets N..2N-1 (one E synapse each); shunt sources 2N.. -> the second half of the targets
      const pre = [], post = [], w = [];
      for(let i = 0; i < N; i++){ pre.push(i); post.push(N + i); w.push(gI === undefined ? 4 : 4); }
      for(let i = 0; i < N; i++) if(i >= N/2){ pre.push(2*N + i); post.push(N + i); w.push(-gI); }
      const order = [...pre.keys()].sort((a, b) => pre[a] - pre[b]);
      const preStart = new Int32Array(n + 1); for(const s of order) preStart[pre[s] + 1]++; for(let i = 0; i < n; i++) preStart[i + 1] += preStart[i];
      return { kind:'net', count:n, pos:new Float32Array(3*n), ntype, bias, src:new Int32Array(n).fill(1), lidx:Uint32Array.from({ length:n }, (_, i) => i),
        preStart, post:Int32Array.from(order, s => post[s]), w:Float32Array.from(order, s => w[s]), delay:new Uint8Array(order.length).fill(1), synCount:order.length, pmask:new Uint8Array(order.length).fill(1),
        plast:0, refrac:2, seed:5, syn, tauE:3, tauI:8, psc:0, protocols:[], eRevE:0, eRevI };
    };
    const rate = (r, lo, hi) => { let s = 0; for(let i = lo; i < hi; i++) s += r.counts[i]; return s/(hi - lo); };
    // conductance mode: a shunt at the rest (-58 for FL) against the same synapse as a current.
    // A strong chloride shunt (60 nS peak, 8 ms) from sources at about 13 Hz, its reversal at the FL rest.
    const cond = await simRun(mk(2, 60, -58), 1000, { steps:1, perNeuron:true });
    const condFar = await simRun(mk(2, 60, -90), 1000, { steps:1, perNeuron:true });
    const plain = rate(cond, N, N + N/2), shunted = rate(cond, N + N/2, 2*N);
    const out = [
      { label:'a shunt at rest divides the response to excitation', measured: cond.error || ('targets ' + plain.toFixed(1) + ' Hz without, ' + shunted.toFixed(1) + ' Hz with the shunt; shunt sources ' + rate(cond, 2*N, 3*N).toFixed(1) + ' Hz'),
        expected:'lower with the shunt, above zero', pass: !cond.error && plain > 2 && shunted < 0.8*plain && shunted > 0 },
      { label:'a reversal below rest inhibits more than one at rest', measured: condFar.error || ('reversal -90: ' + rate(condFar, N + N/2, 2*N).toFixed(1) + ' Hz vs -58: ' + shunted.toFixed(1)),
        expected:'lower', pass: !condFar.error && rate(condFar, N + N/2, 2*N) < shunted },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const g = await simRun(mk(2, 60, -58), 1000, { steps:1, perNeuron:true, gpu:true });
      out.push({ label:'gpu parity in conductance mode', measured: g.error || ('gpu ' + rate(g, N, N + N/2).toFixed(1) + ' / ' + rate(g, N + N/2, 2*N).toFixed(1) + ' vs cpu ' + plain.toFixed(1) + ' / ' + shunted.toFixed(1) + ' Hz'),
        expected:'both target groups within 10%', pass: !g.error && Math.abs(rate(g, N, N + N/2) - plain) <= Math.max(0.5, 0.1*plain) && Math.abs(rate(g, N + N/2, 2*N) - shunted) <= Math.max(0.5, 0.1*shunted) });
    }
    return out;
  } },
{ name:'receptor channels', source:'synaptic receptors differ in kinetics: nicotinic acetylcholine and GABA-A currents decay in a few milliseconds, glutamate-gated chloride and GABA-B over tens (Hille 2001, Ion Channels of Excitable Membranes). A receptor node names channels beyond E and I with their own time constant and sign, and each synapse carries its channel in the top three bits of its plasticity byte (MODEL.md 2). Gated: a slow channel delivers more charge per spike than a fast one at the same peak, both page engines agree, and a channel needs exponential synapses',
  run: async () => {
    // N spiking sources, each onto its own target through one synapse on channel 2
    const N = 200;
    const mk = (tau, syn) => {
      const ntype = new Uint8Array(2*N); const bias = new Float32Array(2*N);
      for(let i = 0; i < N; i++) bias[i] = 6 + 2*(i/N);
      const preStart = new Int32Array(2*N + 1); for(let i = 0; i < N; i++) preStart[i+1] = i + 1; for(let i = N; i <= 2*N; i++) preStart[i] = N;
      const post = new Int32Array(N); for(let i = 0; i < N; i++) post[i] = N + i;
      return { kind:'net', count:2*N, pos:new Float32Array(6*N), ntype, bias, src:new Int32Array(2*N).fill(1), lidx:Uint32Array.from({ length:2*N }, (_, i) => i),
        preStart, post, w:new Float32Array(N).fill(4), delay:new Uint8Array(N).fill(1), synCount:N, pmask:new Uint8Array(N).fill(1 | (2 << 5)),
        receptors:{ channels:[{ name:'x', tau, sign:1, transmitters:['x'] }] },
        plast:0, refrac:2, seed:5, syn, tauE:3, tauI:8, psc:0, protocols:[] };
    };
    const rate = (r, lo, hi) => { let s = 0; for(let i = lo; i < hi; i++) s += r.counts[i]; return s/(hi - lo); };
    const fast = await simRun(mk(3, 1), 1000, { steps:1, perNeuron:true });
    const slow = await simRun(mk(30, 1), 1000, { steps:1, perNeuron:true });
    const kick = await simRun(mk(30, 0), 200, { steps:1, perNeuron:true });
    const out = [
      { label:'a slow channel drives its target harder than a fast one at the same peak', measured: (fast.error || slow.error) || ('targets ' + rate(fast, N, 2*N).toFixed(1) + ' Hz at tau 3 vs ' + rate(slow, N, 2*N).toFixed(1) + ' Hz at tau 30, sources ' + rate(fast, 0, N).toFixed(1) + ' Hz'),
        expected:'sources alike, targets higher at tau 30', pass: !fast.error && !slow.error && rate(fast, 0, N) > 1 && Math.abs(rate(fast, 0, N) - rate(slow, 0, N)) < 0.5 && rate(slow, N, 2*N) > 1.5*Math.max(0.5, rate(fast, N, 2*N)) },
      { label:'a channel needs exponential synapses', measured: kick.error || 'ran', expected:'refused with a message naming exp', pass: /exponential/.test(kick.error || '') },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const gf = await simRun(mk(3, 1), 1000, { steps:1, perNeuron:true, gpu:true });
      const gs = await simRun(mk(30, 1), 1000, { steps:1, perNeuron:true, gpu:true });
      out.push({ label:'gpu parity on both channels', measured: (gf.error || gs.error) || ('gpu ' + rate(gf, N, 2*N).toFixed(1) + ' / ' + rate(gs, N, 2*N).toFixed(1) + ' vs cpu ' + rate(fast, N, 2*N).toFixed(1) + ' / ' + rate(slow, N, 2*N).toFixed(1) + ' Hz'),
        expected:'target rates within 10% at tau 3 and tau 30', pass: !gf.error && !gs.error &&
          Math.abs(rate(gf, N, 2*N) - rate(fast, N, 2*N)) <= Math.max(0.5, 0.1*rate(fast, N, 2*N)) && Math.abs(rate(gs, N, 2*N) - rate(slow, N, 2*N)) <= Math.max(0.5, 0.1*rate(slow, N, 2*N)) });
    }
    return out;
  } },
{ name:'graded relay', source:'graded synaptic transmission, the non-spiking mode of the fly optic lobe and of vertebrate photoreceptors and bipolar cells (Juusola, French, Uusitalo and Weckstrom 1996, Trends Neurosci): release is a continuous function of the presynaptic potential. Here a graded row releases (v - thr) / slope clamped to [0, 1] every millisecond through its synapses, never spikes, and is frozen for plasticity (MODEL.md 1). Gated on both page engines',
  run: async () => {
    const FG = TYPES.findIndex((t, i) => NEURON_TYPES[i].key === 'FG'), FL = TYPES.findIndex((t, i) => NEURON_TYPES[i].key === 'FL');
    // N graded cells each onto its own spiking target
    const N = 200;
    const mk = (drive, plast) => {
      const ntype = new Uint8Array(2*N); const bias = new Float32Array(2*N);
      for(let i = 0; i < N; i++){ ntype[i] = FG; ntype[N+i] = FL; bias[i] = drive*(0.8 + 0.4*(i/N)); }
      const preStart = new Int32Array(2*N + 1); for(let i = 0; i < N; i++) preStart[i+1] = i + 1; for(let i = N; i <= 2*N; i++) preStart[i] = N;
      const post = new Int32Array(N); for(let i = 0; i < N; i++) post[i] = N + i;
      return { kind:'net', count:2*N, pos:new Float32Array(6*N), ntype, bias, src:new Int32Array(2*N).fill(1), lidx:Uint32Array.from({ length:2*N }, (_, i) => i),
        preStart, post, w:new Float32Array(N).fill(30), delay:new Uint8Array(N).fill(1), synCount:N, pmask:new Uint8Array(N).fill(1),
        plast, refrac:2, seed:5, syn:1, tauE:3, tauI:8, protocols:[], aP:0.01, aM:0.01 };
    };
    const rate = (r, lo, hi) => { let s = 0; for(let i = lo; i < hi; i++) s += r.counts[i]; return s/(hi - lo); };
    const off = await simRun(mk(0, 0), 1000, { steps:1, perNeuron:true });
    const low = await simRun(mk(2, 0), 1000, { steps:1, perNeuron:true });
    const high = await simRun(mk(6, 0), 1000, { steps:1, perNeuron:true });
    const pl = await simRun(mk(6, 1), 2000, { steps:1, perNeuron:true, weights:true });
    const out = [
      { label:'a graded cell never spikes', measured: 'graded spikes ' + rate(high, 0, N).toFixed(2) + ' per cell in 1 s', expected:'zero', pass: !high.error && rate(high, 0, N) === 0 },
      { label:'its release drives the target in proportion', measured: 'targets ' + rate(off, N, 2*N).toFixed(1) + ' / ' + rate(low, N, 2*N).toFixed(1) + ' / ' + rate(high, N, 2*N).toFixed(1) + ' Hz at drive 0 / 2 / 6 pA',
        expected:'near zero without drive (the initial jitter releases briefly), rising with it', pass: !low.error && !high.error && rate(off, N, 2*N) < 3 && rate(low, N, 2*N) > 5*Math.max(1, rate(off, N, 2*N)) && rate(high, N, 2*N) > rate(low, N, 2*N) },
      { label:'synapses from a graded cell stay frozen under plasticity', measured: pl.error || ('max |dw| ' + (pl.w ? Math.max(...Array.from(pl.w, x => Math.abs(x - 30))).toFixed(4) : '?')),
        expected:'every weight exactly 30 after 2 s of STDP', pass: !pl.error && pl.w && Array.from(pl.w).every(x => x === 30) },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const g = await simRun(mk(6, 1, 0), 1000, { steps:1, perNeuron:true, gpu:true });
      const gl = await simRun(mk(2, 1, 0), 1000, { steps:1, perNeuron:true, gpu:true });
      out.push({ label:'gpu parity', measured: g.error || ('gpu ' + rate(g, N, 2*N).toFixed(1) + ' / ' + rate(gl, N, 2*N).toFixed(1) + ' vs cpu ' + rate(high, N, 2*N).toFixed(1) + ' / ' + rate(low, N, 2*N).toFixed(1) + ' Hz'),
        expected:'target rates within 10% at both drives, no graded spikes', pass: !g.error && !gl.error && rate(g, 0, N) === 0 &&
          Math.abs(rate(g, N, 2*N) - rate(high, N, 2*N)) <= 0.1*rate(high, N, 2*N) && Math.abs(rate(gl, N, 2*N) - rate(low, N, 2*N)) <= Math.max(0.5, 0.1*rate(low, N, 2*N)) });
    }
    return out;
  } },
{ name:'cortical column layer signatures', topic:'column', source:'Potjans and Diesmann 2014 (Cereb Cortex): spontaneous per-population rates (qualitative at reduced density)',
  async run(){
    const net = await computeScenario(corticalColumn, ed => {
      // every pulse node, not the first: the thalamic drive is one node per cell class since the measured rows differ in capacitance
      for(const n of ed.nodes) if(n.type==='stimulus' && (n.params.mode|0) === 1) n.on = false;
    });
    // skip the onset transient and average a longer window: rates in the first seconds are several times the steady state, and a short window left the excitatory / inhibitory comparison inside run-to-run noise
    const SKIP = 3000, WIN = 9000;
    const cnt = new Float64Array(net.count);
    await simRun(net, SKIP + WIN, { steps:1, onFrame:(fired, t) => {
      if(t < SKIP) return;
      for(let i=0;i<fired.length;i++) if(fired[i]) cnt[i]++;
    } });
    const bySrc = new Map();
    for(let i=0;i<net.count;i++){
      const k = net.src[i];
      if(!bySrc.has(k)) bySrc.set(k, { n:0, s:0, exc: NEURON_TYPES[net.ntype[i]].sign > 0 });
      const e = bySrc.get(k); e.n++; e.s += cnt[i];
    }
    const pops = [...bySrc.values()].map(e => ({ ...e, hz: e.s/e.n/(WIN/1000) }));
    // populations ordered by size are identifiable: counts are unique per layer
    const byN = n => pops.find(p2 => p2.n === n) || { hz: 0 };
    const L23e = byN(3750), L4e = byN(4000), L5e = byN(880), L6e = byN(2620);
    const excHz = mean(pops.filter(p2 => p2.exc).map(p2 => p2.hz));
    const inhHz = mean(pops.filter(p2 => !p2.exc).map(p2 => p2.hz));
    return [
      { label:'L5e highest excitatory rate', measured:
          `L2/3e ${L23e.hz.toFixed(1)}, L4e ${L4e.hz.toFixed(1)}, L5e ${L5e.hz.toFixed(1)}, L6e ${L6e.hz.toFixed(1)} Hz`,
        expected:'L5e highest, every excitatory layer active. Measured in the steady state; the onset transient runs several times higher. The layer rates come from the population-pair tables (Potjans and Diesmann 2014) and the per-layer background drive',
        // above every other excitatory layer, L4e included, which is the closest
        pass: L5e.hz > Math.max(L23e.hz, L4e.hz, L6e.hz) &&
          L23e.hz > 0.05 && L4e.hz > 0.05 && L6e.hz > 0.05 },
      // A mean across populations cannot see a dead layer: 18 Hz in one population and zero in three others averages into the healthy band.
      { label:'no excitatory population is silent', measured:
          pops.filter(p2 => p2.exc).map(p2 => p2.hz.toFixed(1)).join(', ') + ' Hz',
        expected:'every excitatory population above 0.05 Hz',
        pass: pops.filter(p2 => p2.exc).every(p2 => p2.hz > 0.05) },
      { label:'inhibitory rates exceed excitatory', measured:
          `${inhHz.toFixed(2)} vs ${excHz.toFixed(2)} Hz over ${WIN/1000} s after a ${SKIP/1000} s settle`,
        expected:'inh mean > exc mean', pass: inhHz > excHz },
      { label:'network not silent, not epileptic', measured: (mean(pops.map(p2=>p2.hz))).toFixed(1) + ' Hz overall',
        expected:'0.5-60 Hz', pass: mean(pops.map(p2=>p2.hz)) > 0.5 && mean(pops.map(p2=>p2.hz)) < 60 },
    ];
  } },

{ name:'STDP window shape', source:'Bi and Poo 1998 (J Neurosci): potentiation for pre-before-post, depression for post-before-pre, ~20 ms decay',
  async run(){
    // two neurons, one synapse 0->1; spikes forced by strong pulses at a controlled offset, 20 pairings 200 ms apart, weight change read back
    const dw = async dt => {
      const net = { kind:'net', count:2, pos:new Float32Array([0,0,0, 100,0,0]),
        ntype:new Uint8Array([0,0]), bias:new Float32Array(2),
        src:new Int32Array([1,1]), lidx:new Uint32Array([0,1]),
        preStart:new Int32Array([0,1,1]), post:new Int32Array([1]),
        w:new Float32Array([1]), delay:new Uint8Array([1]), synCount:1,
        plast:1, aP:0.01, aM:0.012, tauS:16.8, tauM:33.7, wmax:10, iEta:0, iRho:5,
        protocols:[
          { idx:new Uint32Array([0]), amp:60, mode:1, period:200, width:3, t0:60, duration:0 },
          { idx:new Uint32Array([1]), amp:60, mode:1, period:200, width:3, t0:60+dt, duration:0 },
        ] };
      const r = await simRun(net, 4100, { steps:1, weights:true });
      return r.w[0] - 1;
    };
    const p10 = await dw(10), p40 = await dw(40), m10 = await dw(-10), m40 = await dw(-40);
    return [
      { label:'pre-before-post potentiates', measured: 'dw(+10ms) = ' + p10.toFixed(3),
        expected:'> 0', pass: p10 > 0.01 },
      { label:'post-before-pre depresses', measured: 'dw(-10ms) = ' + m10.toFixed(3),
        expected:'< 0', pass: m10 < -0.01 },
      { label:'window decays with delay (LTP)', measured: p10.toFixed(3) + ' vs ' + p40.toFixed(3),
        expected:'|dw(10)| > |dw(40)|', pass: p10 > p40 },
      { label:'window decays with delay (LTD)', measured: m10.toFixed(3) + ' vs ' + m40.toFixed(3),
        expected:'|dw(-10)| > |dw(-40)|', pass: Math.abs(m10) > Math.abs(m40) },
      // the two constants are separate tunables; with the Bi and Poo values depression decays about half as fast as potentiation
      { label:'depression window is wider than potentiation (tau- 33.7, tau+ 16.8)',
        measured: 'LTD ratio |dw(-40)|/|dw(-10)| = ' + Math.abs(m40/m10).toFixed(3) +
          ', LTP ratio dw(40)/dw(10) = ' + (p40/p10).toFixed(3),
        expected:'LTD ratio > LTP ratio', pass: Math.abs(m40/m10) > p40/p10 },
    ];
  } },

{ name:'fast plasticity mechanism set', source:'Pfister and Gerstner 2006 (J Neurosci) triplet rule fit to Sjostrom et al. 2001; Zenke, Agnes and Gerstner 2015 (Nat Commun) heterosynaptic and transmitter-induced terms',
  async run(){
    // the same forced two-neuron pairing harness as the window group, with pairing frequency and the new terms as the variables
    const pairDw = async (periodMs, extra, onFrame) => {
      const net = { kind:'net', count:2, pos:new Float32Array([0,0,0, 100,0,0]),
        ntype:new Uint8Array([0,0]), bias:new Float32Array(2),
        src:new Int32Array([1,1]), lidx:new Uint32Array([0,1]),
        preStart:new Int32Array([0,1,1]), post:new Int32Array([1]),
        w:new Float32Array([1]), delay:new Uint8Array([1]), synCount:1,
        plast:1, aP:0.01, aM:0.012, tauS:16.8, tauM:33.7, wmax:10, iEta:0, iRho:5,
        ...extra,
        protocols:[
          { idx:new Uint32Array([0]), amp:60, mode:1, period:periodMs, width:3, t0:60, duration:20*periodMs },
          { idx:new Uint32Array([1]), amp:60, mode:1, period:periodMs, width:3, t0:70, duration:20*periodMs },
        ] };
      const r = await simRun(net, 60 + 20*periodMs + 100, { steps:1, weights:true, onFrame });
      return r.w[0] - 1;
    };
    // twenty +10 ms pairings, slow against fast, pair rule against triplet; the slow run has its spike times recorded, for the pair rule's own arithmetic below
    const spk = [[], []];
    const pairSlow = await pairDw(500, {}, (fired, t) => { for(let i=0;i<2;i++) if(fired[i]) spk[i].push(t); });
    const pairFast = await pairDw(25, {});
    // the triplet term at zero is the pair rule: the same run with trip 0 stated must give the same change as the run that never mentions it
    const pairSlow0 = await pairDw(500, { trip:0, tauY:114 });
    // and the change is what the pair rule computes from the spikes that happened: every presynaptic spike before a postsynaptic one adds aP e^(-dt/tauS), every postsynaptic spike before a presynaptic one takes aM e^(-dt/tauM), on emission times, with hard bounds nowhere near.
    // Each 3 ms pulse fires the cell twice, 2 ms apart, so a pairing is four pairs at +8, +10, +10 and +12 ms, four times what one pair at
    // +10 ms gives, which is why this counts the spikes rather than assuming
    let pairExpect = 0;
    for(const tp of spk[1]) for(const tq of spk[0]){
      if(tq < tp) pairExpect += 0.01*Math.exp(-(tp - tq)/16.8);
      else if(tp < tq) pairExpect -= 0.012*Math.exp(-(tq - tp)/33.7);
    }
    const tripSlow = await pairDw(500, { trip:0.01, tauY:114 });
    const tripFast = await pairDw(25, { trip:0.01, tauY:114 });
    const gapPair = pairFast - pairSlow, gapTrip = tripFast - tripSlow;
    // transmitter-induced drip: presynaptic spikes alone, post silent, so the pair terms are inert and any change is the tin term exactly
    const tinNet = extra => ({ kind:'net', count:2,
      pos:new Float32Array([0,0,0, 100,0,0]),
      ntype:new Uint8Array([0,0]), bias:new Float32Array(2),
      src:new Int32Array([1,1]), lidx:new Uint32Array([0,1]),
      preStart:new Int32Array([0,1,1]), post:new Int32Array([1]),
      w:new Float32Array([1]), delay:new Uint8Array([1]), synCount:1,
      plast:1, aP:0.01, aM:0.012, tauS:16.8, tauM:33.7, wmax:10, iEta:0, iRho:5,
      ...extra,
      protocols:[{ idx:new Uint32Array([0]), amp:60, mode:1, period:100,
        width:3, t0:60, duration:2000 }] });
    const r0 = await simRun(tinNet({}), 2400, { steps:1, weights:true });
    const r1 = await simRun(tinNet({ tin:0.001 }), 2400, { steps:1, weights:true });
    const drip0 = r0.w[0] - 1, drip1 = r1.w[0] - 1;
    // heterosynaptic regression: potentiate the synapse first, then burst the postsynaptic cell without presynaptic drive; with het on the weight must fall back toward its starting value, with het off stay
    const hetRun = async het => {
      const net = { kind:'net', count:2, pos:new Float32Array([0,0,0, 100,0,0]),
        ntype:new Uint8Array([0,0]), bias:new Float32Array(2),
        src:new Int32Array([1,1]), lidx:new Uint32Array([0,1]),
        preStart:new Int32Array([0,1,1]), post:new Int32Array([1]),
        w:new Float32Array([3]), delay:new Uint8Array([1]), synCount:1,
        plast:1, aP:0, aM:0, tauS:20, wmax:10, iEta:0, iRho:5,
        trip:0, tauY:114, het, tin:0,
        protocols:[
          { idx:new Uint32Array([1]), amp:60, mode:1, period:40, width:3,
            t0:60, duration:4000 } ] };
      // The reference is captured at w = 3 when plasticity starts; a 2 s pairing phase (presynaptic and postsynaptic pulses, aP on) lifts the weight, then 4 s of postsynaptic bursts alone, where only het can act.
      net.aP = 0.02;
      net.protocols = [
        { idx:new Uint32Array([0]), amp:60, mode:1, period:50, width:3, t0:60, duration:2000 },
        { idx:new Uint32Array([1]), amp:60, mode:1, period:50, width:3, t0:70, duration:6000 },
      ];
      const r = await simRun(net, 6400, { steps:1, weights:true });
      return r.w[0];
    };
    const hetOff = await hetRun(0);
    const hetOn = await hetRun(0.02);
    return [
      { label:'triplet adds pairing-frequency dependence', measured:
          'rate gap pair ' + gapPair.toFixed(3) + ' vs triplet ' + gapTrip.toFixed(3),
        expected:'triplet gap clearly larger (Sjostrom frequency dependence)',
        pass: gapTrip > gapPair + 0.02 },
      { label:'triplet off reproduces the pair rule', measured:
          'dw slow ' + pairSlow.toFixed(4) + ' (with trip 0 stated ' + pairSlow0.toFixed(4) + '); ' +
          'the pair rule on the ' + spk[0].length + ' and ' + spk[1].length + ' spikes that happened ' + pairExpect.toFixed(4),
        expected:'identical with trip 0 stated, and equal to the pair rule computed from the spike times within half a percent',
        pass: Math.abs(pairSlow0 - pairSlow) < 1e-6 && pairExpect > 0 &&
          Math.abs(pairSlow - pairExpect) < 0.005*Math.abs(pairExpect) },
      { label:'transmitter term drips only when enabled', measured:
          'off ' + drip0.toFixed(4) + ', on ' + drip1.toFixed(4) +
          ' = ' + Math.round(drip1/0.001) + ' pre spikes x 0.001',
        expected:'off exactly 0; on an exact integer multiple of the delta',
        pass: Math.abs(drip0) < 1e-6 && drip1 > 0.01 &&
          Math.abs(drip1/0.001 - Math.round(drip1/0.001)) < 0.05 },
      { label:'heterosynaptic pull returns weights toward reference', measured:
          'final w het off ' + hetOff.toFixed(3) + ' vs on ' + hetOn.toFixed(3) + ' (start 3)',
        expected:'het on ends closer to the starting weight',
        pass: Math.abs(hetOn - 3) < Math.abs(hetOff - 3) - 0.05 },
    ];
  } },

{ name:'shape decoding is a safe alternative decoder', topic:'analysis', source:'internal invariant. decodeAccuracy scores by cosine similarity, which is already scale invariant, so a multiplicative gain cannot move it; and a per-trial additive baseline barely moves it either, because the offset is averaged into the centroids as well and the differences that separate items survive. Measured over synthetic codes on 2026-08-30: with an offset of up to 12 the raw decoder scored 0.91 against 0.94 without one, and in one arm the offset raised the score. So decodeShape is not a rescue for anything, and what is gated is the weaker property: it agrees with the raw decoder on clean trials and it sits at chance when the trials carry no item information, so a reading taken with it cannot be an artifact of the normalization',
  async run(){
    const nI = 5, nN = 120, per = 40, N = nI*per;
    let seed = 5150;
    const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed/4294967296; };
    const mk = (kind) => {
      const trials = new Float32Array(N*nN), labels = new Int32Array(N);
      let t = 0;
      for(let i=0;i<nI;i++) for(let k=0;k<per;k++){
        labels[t] = i;
        for(let j=0;j<nN;j++){
          const pref = (j % nI) === i ? 1 : 0.35;   // broad tuning
          trials[t*nN + j] = kind === 'noise'
            ? 0.5 + rnd()
            : pref*(0.7 + 0.6*rnd());
        }
        t++;
      }
      return { trials, labels };
    };
    const clean = mk('clean'), noise = mk('noise');
    const cRaw = decodeAccuracy(clean.trials, clean.labels, nI, nN);
    const cShape = decodeShape(clean.trials, clean.labels, nI, nN);
    const nShape = decodeShape(noise.trials, noise.labels, nI, nN);
    const nRaw = decodeAccuracy(noise.trials, noise.labels, nI, nN);
    return [
      { label:'agrees with the raw decoder on a clean code', measured:
          'shape ' + (100*cShape.acc).toFixed(0) + '% vs raw ' +
          (100*cRaw.acc).toFixed(0) + '%',
        expected:'within 15 points, so the normalization is not doing work of its own',
        pass: Math.abs(cShape.acc - cRaw.acc) < 0.15 },
      { label:'at chance when the trials carry no item information', measured:
          'shape ' + (100*nShape.acc).toFixed(0) + '%, chance ' +
          (100/nI).toFixed(0) + '%',
        expected:'within 12 points of chance',
        pass: Math.abs(nShape.acc - 1/nI) < 0.12 },
      { label:'the raw decoder is at chance on the same unstructured trials', measured:
          'raw ' + (100*nRaw.acc).toFixed(0) + '%',
        expected:'within 12 points of chance, so the fixture is unstructured',
        pass: Math.abs(nRaw.acc - 1/nI) < 0.12 },
    ];
  } },

{ name:'completion measure separates its failure modes', topic:'analysis', source:'internal invariant. Cross-modal transfer can fail two opposite ways, and a rank or a margin is blind to the difference. Either the two modalities share no representation, in which case the same-item cosine is near zero and there is no relation to learn, or they share one that carries no item identity, in which case the same-item cosine is high and the different-item cosine is just as high. Those call for opposite fixes, so the measure has to tell them apart before it is trusted on the block',
  async run(){
    const nI = 5, nN = 200;
    // three synthetic pairs of centroid sets, each a known failure or success
    const mk = kind => {
      let seed = 24680;
      const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed/4294967296; };
      const full = new Float64Array(nI*nN), part = new Float64Array(nI*nN);
      for(let i=0;i<nI;i++) for(let j=0;j<nN;j++){
        const own = (j % nI) === i ? 1 : 0;
        full[i*nN + j] = own ? 1 + 0.1*rnd() : 0.05*rnd();
        if(kind === 'bound')
          part[i*nN + j] = own ? 1 + 0.1*rnd() : 0.05*rnd();
        else if(kind === 'disjoint')
          // the partial modality drives a different, non-overlapping set
          part[i*nN + j] = ((j + 1) % nI) === i ? 1 + 0.1*rnd() : 0.05*rnd();
        else
          // Shared but identity-free: every item drives the same cells, and that set has to overlap all items equally.
          // Keying it to one item's preferred cells instead makes the same-item cosine high for that item and low for the rest, which averages to something that looks like the disjoint case.
          part[i*nN + j] = 1 + 0.1*rnd();
      }
      return completionScore(full, part, nI, nN);
    };
    const bound = mk('bound'), disjoint = mk('disjoint'), shared = mk('shared');
    return [
      { label:'a bound representation ranks first and has a high own cosine', measured:
          'rank1 ' + bound.rank1.toFixed(2) + ', own ' + bound.own.toFixed(3) +
          ', other ' + bound.other.toFixed(3),
        expected:'rank1 1.0 with own well above other',
        pass: bound.rank1 > 0.9 && bound.own > 0.7 && bound.own - bound.other > 0.3 },
      { label:'disjoint representations read as disjoint', measured:
          'own ' + disjoint.own.toFixed(3) + ', other ' + disjoint.other.toFixed(3),
        expected:'own low: nothing is shared, so there is no relation to learn',
        pass: disjoint.own < 0.35 },
      { label:'a shared but identity-free code is not read as disjoint', measured:
          'own ' + shared.own.toFixed(3) + ', other ' + shared.other.toFixed(3),
        expected:'own high and other equally high: the opposite failure, and it must not look like the one above',
        pass: shared.own > 0.4 && Math.abs(shared.own - shared.other) < 0.05 },
      { label:'the two failures are distinguishable', measured:
          'disjoint own ' + disjoint.own.toFixed(3) +
          ' vs shared own ' + shared.own.toFixed(3),
        expected:'the same-item cosine separates them, where rank1 alone does not',
        pass: shared.own - disjoint.own > 0.3 },
      { label:'centring removes the non-negative floor', measured:
          'bound ' + bound.ownC.toFixed(3) + ', disjoint ' +
          disjoint.ownC.toFixed(3) + ', shared ' + shared.ownC.toFixed(3),
        expected:'spike counts are non-negative so raw cosines sit high for any pair; on the item-specific residual a bound code is near 1, a disjoint one negative, and an identity-free one near zero',
        pass: bound.ownC > 0.9 && disjoint.ownC < -0.1
          && Math.abs(shared.ownC) < 0.1 },
    ];
  } },

{ name:'short-term plasticity', source:'Tsodyks and Markram; Zenke, Agnes and Gerstner 2015 (Nat Commun 6:6922) equations 9 and 10. Release depends on the recent history of the presynaptic cell: resources deplete with each spike and recover with tau_d, release probability facilitates and decays with tau_f. Short-term depression is what gives the effective input-output relation the curvature that puts a stable fixed point at intermediate firing rates, and the source reports that blocking any single component of its plasticity set prevented the network working as a memory',
  async run(){
    // The rule is deterministic given a spike train, so it is checked directly rather than through a network: a network test would confound the rule with the dynamics it is supposed to shape.
    const U = 0.2, tauD = 200, tauF = 600;
    // returns the per-spike release factor for a regular train at rate hz; order 0 as published: release u x, then deplete and facilitate; order 1 (stpOrder) facilitates first
    const train = (hz, nSpk, order = 0) => {
      let x = 1, r = U;
      const isi = 1000/hz, out = [];
      for(let k=0;k<nSpk;k++){
        if(k > 0){
          x = 1 - (1 - x)*Math.exp(-isi/tauD);
          r = U + (r - U)*Math.exp(-isi/tauF);
        }
        if(order) r += U*(1 - r);
        const rel = r*x;
        x -= rel;
        if(!order) r += U*(1 - r);
        out.push(rel);
      }
      return out;
    };
    const rest = train(0.01, 1)[0];          // effectively a rested synapse
    const restJump = train(0.01, 1, 1)[0];
    const lo = train(2, 12), hi = train(40, 12);
    const loSteady = lo[lo.length-1], hiSteady = hi[hi.length-1];
    // total transmitted per second: release factor times rate, which is the quantity that must saturate for the transfer function to bend
    const through = hz => { const t = train(hz, 60); return t[t.length-1]*hz; };
    const t5 = through(5), t20 = through(20), t80 = through(80);
    return [
      { label:'a rested synapse releases U, as published', measured:
          'factor ' + rest.toFixed(3) + ' at U ' + U,
        expected:U.toFixed(3) + ' (Tsodyks, Pawelzik and Markram 1998; stpNorm 1 scales this to 1)',
        pass: Math.abs(rest - U) < 0.02 },
      { label:'facilitation before release (stpOrder 1) releases U(2 - U) from rest', measured:
          'factor ' + restJump.toFixed(3) + ' at U ' + U,
        expected:(U*(2 - U)).toFixed(3),
        pass: Math.abs(restJump - U*(2 - U)) < 0.02 },
      { label:'high rates deplete more than low rates', measured:
          'steady factor ' + loSteady.toFixed(3) + ' at 2 Hz vs ' +
          hiSteady.toFixed(3) + ' at 40 Hz',
        expected:'depression grows with rate',
        pass: hiSteady < loSteady },
      { label:'throughput saturates rather than scaling with rate', measured:
          '5 Hz ' + t5.toFixed(2) + ', 20 Hz ' + t20.toFixed(2) +
          ', 80 Hz ' + t80.toFixed(2),
        expected:'sublinear in rate: doubling the rate less than doubles what arrives',
        pass: t20 < 4*t5 && t80 < 4*t20 },
      { label:'facilitation acts before depression dominates', measured:
          'first ' + hi[0].toFixed(3) + ', second ' + hi[1].toFixed(3) +
          ', steady ' + hiSteady.toFixed(3),
        expected:'the second spike is not simply weaker than the first at 40 Hz',
        pass: hi[1] > hiSteady },
    ];
  } },

{ name:'reference weight consolidates', source:'Zenke, Agnes and Gerstner 2015 (Nat Commun 6:6922) equation 16: the reference weight that heterosynaptic plasticity regresses toward is not fixed, it follows the synaptic weight through a double well whose lower fixed point is zero and whose upper one is w_P, with the midpoint unstable. The block froze that reference at the moment plasticity began, which made the heterosynaptic term a permanent decay toward the initial state; competition at 0.002, 0.0005 and 0.0002 then lowered rates and hurt decode at every dose without producing structure',
  async run(){
    // Two synapses on one neuron, driven so the pair rule pushes one up and leaves the other alone, run long enough for the slow reference to move.
    // The question is only whether the reference is bistable and follows, so this drives the weights directly rather than through a network: a network test would confound the rule with its dynamics.
    // This integrates equation 16 as src/simworker.js consolidate() writes it, clamp included, and checks the shape of the well it defines.
    // It is not an engine test: the engines' consolidation is held to the reference by the cross-engine parity group and to a second implementation by tools/brianref.mjs's consolidation case.
    // With w_P 0.5 and P 10, P w_P^2 / 4 is 0.625 and the equation has one stable fixed point and no double well.
    // The well exists when P w_P^2 / 4 exceeds 1; with w_P 1 and P 10 it is 2.5.
    const consW = 1, consP = 10, tauCons = 1200000, dt = 1200;
    const step = (wRef0, drive, ms) => {
      let r = wRef0;
      for(let t=0;t<ms;t+=dt){
        const nr = r + (dt/tauCons)*(drive - r - consP*r*(consW*0.5 - r)*(consW - r));
        r = nr < 0 ? 0 : (nr > consW ? consW : nr);
      }
      return r;
    };
    const H = 3600000*4;
    // a reference whose weight sits high climbs to the upper state
    const hi = step(0.9, 0.9, H);
    // one whose weight is low falls toward zero
    const lo = step(0.2, 0.05, H);
    // the midpoint is unstable: with the same drive, at the midpoint itself, a nudge either way leaves it and the two starts end in different wells
    const upFromMid = step(consW*0.5 + 0.01, consW*0.5, H);
    const downFromMid = step(consW*0.5 - 0.01, consW*0.5, H);
    const depth = consP*consW*consW/4;
    return [
      { label:'the constants define a double well', measured:
          'P w_P^2 / 4 = ' + depth.toFixed(2),
        expected:'above 1, where the equation has two stable states', pass: depth > 1 },
      { label:'a driven reference consolidates to the upper state', measured:
          'wRef ' + hi.toFixed(3),
        expected:'near w_P = 1', pass: hi > 0.8 },
      { label:'an undriven reference decays toward zero', measured:
          'wRef ' + lo.toFixed(3),
        expected:'below 0.1', pass: lo < 0.1 },
      { label:'the midpoint is unstable in both directions', measured:
          'from ' + (consW*0.5 + 0.01) + ' to ' + upFromMid.toFixed(3) + ', from ' +
          (consW*0.5 - 0.01) + ' to ' + downFromMid.toFixed(3) + ', both driven at the midpoint',
        expected:'the two starts leave w_P/2 in opposite directions',
        pass: upFromMid > 0.8 && downFromMid < 0.2 },
    ];
  } },

{ name:'like-to-like wiring is detectable', topic:'analysis', source:'Ding et al. 2025 (Nature), the MICrONS functional connectome: tuning similarity predicts fine-scale connectivity in mouse visual cortex beyond what axon and dendrite proximity explain. The block wires on distance and population pair alone, so before asking whether it has the rule the measure has to be shown to find the rule when it is present and to report nothing when it is absent, including when tuning is spatially clustered and proximity alone could manufacture the effect',
  async run(){
    // Three synthetic populations of tuned cells, identical except for how they are wired:
    //   random: wiring ignores tuning, so the measure must read zero
    //   like: wiring prefers similar tuning, so it must read positive
    //   clustered: wiring ignores tuning but tuning is laid out in space,
    //     so an unmatched measure would read positive from proximity
    //     alone and the distance-matched one must still read zero
    const nI = 12, nN = 300, perItem = 30, N = nI*perItem;
    const build = kind => {
      let seed = 987654321;
      const rand = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed/4294967296; };
      const inhib = j => kind === 'mixed' && (j % 5) === 0;
      const pref = new Int32Array(nN), pos = new Float32Array(nN*3);
      for(let j=0;j<nN;j++){
        // clustered: preference follows position, so nearby cells agree
        pref[j] = kind === 'clustered' ? Math.floor(j/(nN/nI)) % nI : j % nI;
        // laid out as a grid so distance bins are well populated
        pos[j*3] = (j % 20)*30; pos[j*3+1] = Math.floor(j/20)*30; pos[j*3+2] = 0;
      }
      const trials = new Float32Array(N*nN), labels = new Int32Array(N);
      let t = 0;
      for(let i=0;i<nI;i++) for(let k=0;k<perItem;k++){
        labels[t] = i;
        for(let j=0;j<nN;j++)
          trials[t*nN + j] = rand() < (pref[j] === i ? 0.8 : 0.05) ? 1 : 0;
        t++;
      }
      // wiring: distance dependent in every case, tuning dependent only in the 'like' case
      const out = Array.from({ length:nN }, () => []);
      for(let a=0;a<nN;a++) for(let b=0;b<nN;b++){
        if(a === b) continue;
        const dx = pos[a*3] - pos[b*3], dy = pos[a*3+1] - pos[b*3+1];
        const d = Math.sqrt(dx*dx + dy*dy);
        let p = 0.35*Math.exp(-d/120);
        if(kind === 'like') p *= (pref[a] === pref[b] ? 3 : 0.4);
        // 'mixed' is tuning-blind between excitatory cells, but a fifth of its cells are inhibitory and avoid similarly tuned targets.
        // Similarity then tracks the sign of the weight, which reads as a strong rule unless the measure looks at excitatory synapses only.
        if(inhib(a)) p *= (pref[a] === pref[b] ? 0.2 : 1.6);
        if(rand() < p) out[a].push(b);
      }
      const preStart = new Int32Array(nN + 1);
      for(let a=0;a<nN;a++) preStart[a+1] = preStart[a] + out[a].length;
      const post = new Int32Array(preStart[nN]), w = new Float32Array(preStart[nN]);
      let q = 0;
      for(let a=0;a<nN;a++) for(const b of out[a]){
        post[q] = b;
        // weights carry the same rule in the 'like' case only
        w[q] = kind === 'like' && pref[a] === pref[b] ? 6 : 2;
        if(inhib(a)) w[q] = -w[q];
        q++;
      }
      const idx = Int32Array.from({ length:nN }, (_, j) => j);
      const links = popLinks(idx, preStart, post);
      return likeToLike(trials, labels, nI, nN, pos, idx, links, { w, w0:w });
    };
    const rnd = build('random'), like = build('like'), clu = build('clustered'),
      mix = build('mixed');
    return [
      { label:'tuning-blind wiring reads as tuning-blind', measured:
          'delta ' + (rnd ? rnd.deltaCorr.toFixed(3) : 'null') +
          ' (connected ' + (rnd ? rnd.connCorr.toFixed(3) : '-') +
          ' vs unconnected ' + (rnd ? rnd.unconnCorr.toFixed(3) : '-') + ')',
        expected:'near zero: wiring ignored tuning',
        pass: !!rnd && Math.abs(rnd.deltaCorr) < 0.05 },
      { label:'like-to-like wiring is found', measured:
          'delta ' + (like ? like.deltaCorr.toFixed(3) : 'null'),
        expected:'clearly positive, above 0.1',
        pass: !!like && like.deltaCorr > 0.1 },
      { label:'spatially clustered tuning does not fake the rule', measured:
          'delta ' + (clu ? clu.deltaCorr.toFixed(3) : 'null'),
        expected:'near zero after distance matching, though nearby cells share tuning',
        pass: !!clu && Math.abs(clu.deltaCorr) < 0.05 },
      { label:'similarity against weight separates the two', measured:
          'like ' + (like ? like.deltaW.toFixed(3) : '-') +
          ' vs random ' + (rnd ? rnd.deltaW.toFixed(3) : '-'),
        expected:'positive where weights follow tuning, near zero where they do not',
        pass: !!like && !!rnd && like.deltaW > 0.15 && Math.abs(rnd.deltaW) < 0.05 },
      { label:'inhibitory synapses do not fake the rule', measured:
          'mixed delta ' + (mix ? mix.deltaCorr.toFixed(3) : 'null') +
          ', weight ' + (mix ? mix.deltaW.toFixed(3) : '-'),
        expected:'near zero on both: excitatory wiring here is tuning-blind, and only the inhibitory partners are anti-tuned',
        pass: !!mix && Math.abs(mix.deltaCorr) < 0.05 && Math.abs(mix.deltaW) < 0.08 },
    ];
  } },

{ name:'firing rate distribution is skewed', source:'Buzsaki and Mizuseki 2014 (Nat Rev Neurosci); Mizuseki and Buzsaki 2013 (Cell Rep): principal cell rates are lognormal in every cortical region and brain state examined, most cells firing rarely with a heavy tail carrying the rest',
  async run(){
    // A network of statistically identical neurons under a uniform drive produces a uniform population: measured on the block 2026-08-28 as a 1.4x spread from median to 99th percentile with no silent cells, where cortex spans orders of magnitude.
    // This group compares a uniform drive against a lognormal per-neuron drive of the same mean.
    const run = async spread => {
      const pts = mkPts(1200, 320, 11);
      const net = wireConnect(pts, { ...CN, radius:240, prob:0.35, wExc:4, wInh:-6 }, null);
      const idx = Uint32Array.from({ length:net.count }, (_, i) => i);
      const gain = new Float32Array(net.count);
      let sum = 0;
      for(let i=0;i<net.count;i++){
        // deterministic per-neuron lognormal, normalized to unit mean
        const u = ((Math.imul(i + 1, 2654435761) >>> 0) / 4294967296) || 1e-6;
        const v = ((Math.imul(i + 7919, 40503) >>> 0) / 4294967296) || 1e-6;
        const g = Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
        gain[i] = spread > 0 ? Math.exp(spread*g - spread*spread/2) : 1;
        sum += gain[i];
      }
      const norm = net.count/sum;
      let gm = 0;
      for(let i=0;i<net.count;i++){ gain[i] *= norm; gm += gain[i]; }
      Object.assign(net, { plast:0,
        protocols:[{ idx, amp:6, mode:0, period:100, width:10, t0:0, duration:0,
          gain: spread > 0 ? gain : undefined }] });
      const r = await simRun(net, 6000, { steps:2, perNeuron:true });
      const c = [...r.counts].sort((a, b) => a - b);
      const q = t => c[Math.min(c.length - 1, Math.floor(t*c.length))];
      return { p50:q(0.5), p99:q(0.99), silent:c.filter(x => x === 0).length/c.length,
        ratio: q(0.99)/Math.max(1, q(0.5)), meanGain:gm/net.count };
    };
    const flat = await run(0);
    const skew = await run(0.9);
    return [
      { label:'uniform drive gives a uniform population', measured:
          'p99/p50 = ' + flat.ratio.toFixed(2) + ', silent ' + (100*flat.silent).toFixed(0) + '%',
        expected:'documents the baseline: near 1, which is not the cortical shape',
        pass: flat.ratio < 3 },
      { label:'lognormal drive produces a heavy tail', measured:
          'p99/p50 = ' + skew.ratio.toFixed(2) + ' (p50 ' + skew.p50 + ', p99 ' + skew.p99 + ')',
        expected:'clearly skewed, above 3x, as cortical rate distributions are',
        pass: skew.ratio > 3 },
      { label:'the tail is not bought with silence', measured:
          'silent ' + (100*skew.silent).toFixed(0) + '%',
        expected:'below 60 percent: a skewed distribution, not a dead network',
        pass: skew.silent < 0.6 },
      { label:'spread does not change mean drive', measured:
          'mean gain ' + skew.meanGain.toFixed(6) + ' at spread 0.9',
        expected:'1 within 1e-4: heterogeneity only, not a drive change',
        pass: Math.abs(skew.meanGain - 1) < 1e-4 },
    ];
  } },

{ name:'layer transformation measures', topic:'analysis', source:'Transformation of primary sensory cortical representations from layer 4 to layer 2 (Nat Commun 2022): responsive fraction falls 0.16 to 0.09 from L4 to L2 while response probability among responders rises 0.13 to 0.22 and correlated ensembles multiply. Checked here against synthetic populations with a known answer, since a measure that cannot tell a sparse informative code from a merely quiet one cannot report the transformation',
  async run(){
    // Three synthetic populations, 12 items, 40 trials each.
    //  dense: every cell answers every item, so nothing is tuned
    //  tuned: each cell prefers one item and answers it reliably
    //  quiet: each cell prefers one item but answers it rarely
    // The measures must separate tuned from dense on responsive fraction and tuned from quiet on reliability.
    const nI = 12, nN = 120, perItem = 40, N = nI*perItem;
    const build = kind => {
      const trials = new Float32Array(N*nN), labels = new Int32Array(N);
      let t = 0;
      let seed = 12345;
      const rand = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed/4294967296; };
      for(let i=0;i<nI;i++) for(let k=0;k<perItem;k++){
        labels[t] = i;
        for(let j=0;j<nN;j++){
          const pref = j % nI;
          let p;
          if(kind === 'dense') p = 0.6;
          else if(kind === 'tuned') p = (pref === i) ? 0.85 : 0.05;
          else p = (pref === i) ? 0.12 : 0.01;      // quiet
          trials[t*nN + j] = rand() < p ? 1 : 0;
        }
        t++;
      }
      return { trials, labels };
    };
    const dense = build('dense'), tuned = build('tuned'), quiet = build('quiet');
    const rDense = responseProfile(dense.trials, dense.labels, nI, nN);
    const rTuned = responseProfile(tuned.trials, tuned.labels, nI, nN);
    const rQuiet = responseProfile(quiet.trials, quiet.labels, nI, nN);
    const cDense = ensembleCorrelation(dense.trials, dense.labels, nN);
    const cTuned = ensembleCorrelation(tuned.trials, tuned.labels, nN);
    return [
      { label:'a dense population reads as untuned', measured:
          'responsive ' + rDense.responsiveFrac.toFixed(2),
        expected:'near zero: no cell prefers anything',
        pass: rDense.responsiveFrac < 0.15 },
      { label:'a tuned population reads as responsive', measured:
          'responsive ' + rTuned.responsiveFrac.toFixed(2),
        expected:'high: every cell prefers one item',
        pass: rTuned.responsiveFrac > 0.8 },
      { label:'reliability separates tuned from quiet', measured:
          'tuned ' + rTuned.reliability.toFixed(2) + ' vs quiet ' + rQuiet.reliability.toFixed(2),
        expected:'both are tuned; only the first answers reliably',
        pass: rTuned.reliability > 0.6 && rQuiet.reliability < 0.35 },
      { label:'ensemble correlation rises with structure', measured:
          'tuned ' + cTuned.toFixed(3) + ' vs dense ' + cDense.toFixed(3),
        expected:'a tuned population is more correlated than an untuned one',
        pass: cTuned > cDense + 0.05 },
    ];
  } },

{ name:'refractory floor', source:'sodium channel inactivation imposes an absolute refractory period of one to two milliseconds; the Izhikevich reset provides relative refractoriness only, so under saturating drive nothing otherwise prevents a spike on the next millisecond',
  async run(){
    // one neuron, drive far above threshold, count spikes per second with the floor off and on: a cell is held for refrac steps after each spike, so the shortest interval is refrac + 1 ms and the rate is capped at 1000/(refrac + 1)
    const one = refrac => ({ kind:'net', count:1, pos:new Float32Array(3),
      // A drive that fires the cell on every step with no floor (25 fires the classic RS row at 47 Hz on the 2007 form, and the floor never engages).
      ntype:new Uint8Array([0]), bias:new Float32Array([1000]),
      src:new Int32Array([1]), lidx:new Uint32Array([0]),
      preStart:new Int32Array([0, 0]), post:new Int32Array(0),
      w:new Float32Array(0), delay:new Uint8Array(0), synCount:0,
      plast:0, refrac, protocols:[] });
    const rate = async refrac => {
      const r = await simRun(one(refrac), 3000, { steps:1, perNeuron:true });
      return r.counts[0]/3;
    };
    const free = await rate(0);
    const r2 = await rate(2);
    const r5 = await rate(5);
    return [
      { label:'unconstrained rate is unphysiological', measured: free.toFixed(0) + ' Hz',
        expected:'above 500 Hz, so the floors below have something to cap',
        pass: free > 500 },
      { label:'2 ms floor caps the rate', measured: r2.toFixed(0) + ' Hz',
        expected:'at the cap, 1000/(2 + 1) = 333 Hz', pass: Math.abs(r2 - 1000/3) < 2 },
      { label:'5 ms floor caps the rate', measured: r5.toFixed(0) + ' Hz',
        expected:'at the cap, 1000/(5 + 1) = 167 Hz', pass: Math.abs(r5 - 1000/6) < 2 },
      { label:'the floor only removes spikes', measured:
          free.toFixed(0) + ' -> ' + r2.toFixed(0) + ' -> ' + r5.toFixed(0) + ' Hz',
        expected:'monotonically lower with a longer floor',
        pass: r2 <= free + 1 && r5 <= r2 + 1 },
    ];
  } },

{ name:'feedback inhibition', source:'Lin et al. 2014 (Nat Neurosci): the APL neuron receives from and inhibits every Kenyon cell and holds the odor code sparse; Andersen, Eccles and Loyning 1963 (Nature): recurrent inhibition through basket cells',
  async run(){
    // Unconnected cells under a spread of steady drives with per-cell noise: the pool is the only coupling in the network, so what it does is exactly the difference between the runs.
    // The noise is a hash of (cell, t, seed), so a cell the pool does not touch computes the same bytes with and without a pool in the network.
    // Exp synapses, the mode most rate-reading scenes run; in kick mode a pool's whole output for a millisecond lands as one current step, and a synchronous pool at a high gain pushes the membrane so far below rest that one half step of the Izhikevich quadratic throws it back over threshold.
    const mk = (n, groups, gain, extra) => {
      const idx = Uint32Array.from({ length:n }, (_, i) => i);
      const pool = new Uint16Array(n);
      const K = [0];
      groups.forEach(([lo, hi], gi) => {
        for(let i=lo;i<hi;i++) pool[i] = gi + 1;
        K.push(gain*1000/(hi - lo));
      });
      const bias = new Float32Array(n);
      for(let i=0;i<n;i++) bias[i] = 5 + 2*((i % 7)/6);
      return { kind:'net', count:n, pos:new Float32Array(3*n),
        ntype:new Uint8Array(n), bias,
        src:new Int32Array(n).fill(1), lidx:idx.slice(),
        preStart:new Int32Array(n+1), post:new Int32Array(0),
        w:new Float32Array(0), delay:new Uint8Array(0), synCount:0,
        plast:0, refrac:2, seed:5, syn:1, psc:1, tauE:3, tauI:8,
        protocols:[{ idx, amp:4, mode:3, period:100, width:10, t0:0, duration:0 }],
        ...(groups.length ? { pool, poolK:Float32Array.from(K) } : {}), ...extra };
    };
    const MS = 2000;
    const run = (net, gpu) => simRun(net, MS, { steps:1, perNeuron:true, gpu });
    const meanHz = (r, lo, hi) => { let s = 0; for(let i=lo;i<hi;i++) s += r.counts[i]; return s/(hi-lo)/(MS/1000); };
    const same = (a, b, lo, hi) => { for(let i=lo;i<hi;i++) if(a.counts[i] !== b.counts[i]) return false; return true; };
    const within = (x, y, f) => Math.abs(x - y) <= f*Math.max(x, y, 0.1);
    const free = await run(mk(400, [], 0, {}));
    const on400 = await run(mk(400, [[0, 400]], -2, {}));
    const on200 = await run(mk(200, [[0, 200]], -2, {}));
    const half = await run(mk(400, [[0, 200]], -2, {}));
    const kick0 = await run(mk(400, [], 0, { syn:0 }));
    const kick1 = await run(mk(400, [[0, 400]], -1, { syn:0 }));
    const rFree = meanHz(free, 0, 400), r400 = meanHz(on400, 0, 400), r200 = meanHz(on200, 0, 200);
    const kFree = meanHz(kick0, 0, 400), kOn = meanHz(kick1, 0, 400);
    const out = [
      { label:'the pool lowers the rate', measured: rFree.toFixed(1) + ' -> ' + r400.toFixed(1) + ' Hz',
        expected:'lower with a pool at gain -2 than without', pass: rFree > 1 && r400 < 0.8*rFree },
      { label:'the same gain means the same inhibition whatever the size',
        measured: r200.toFixed(1) + ' Hz at 200 cells, ' + r400.toFixed(1) + ' Hz at 400',
        expected:'within 15 percent of each other', pass: within(r200, r400, 0.15) },
      { label:'a cell outside every pool is untouched',
        measured: same(half, free, 200, 400) ? 'identical spike counts' : 'counts differ',
        expected:'bit for bit what it does with no pool in the network', pass: same(half, free, 200, 400) },
      { label:'a pool sees only its own cells',
        measured: same(half, on200, 0, 200) ? 'identical spike counts' : 'counts differ',
        expected:'the pooled half computes what it does alone', pass: same(half, on200, 0, 200) },
      { label:'kick synapses: the pool lowers the rate',
        measured: kFree.toFixed(1) + ' -> ' + kOn.toFixed(1) + ' Hz',
        expected:'lower with a pool at gain -1 on the input current', pass: kOn < 0.8*kFree },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const g0 = await run(mk(400, [], 0, {}), true);
      const g1 = await run(mk(400, [[0, 400]], -2, {}), true);
      const gk = await run(mk(400, [[0, 400]], -1, { syn:0 }), true);
      const err = g0.error || g1.error || gk.error;
      if(err) out.push({ label:'gpu engine runs with a pool', measured:err, expected:'no engine error', pass:false });
      else {
        const gf = meanHz(g0, 0, 400), gp = meanHz(g1, 0, 400), gkp = meanHz(gk, 0, 400);
        out.push({ label:'gpu: the pool lowers the rate', measured: gf.toFixed(1) + ' -> ' + gp.toFixed(1) + ' Hz',
          expected:'lower with the pool', pass: gp < 0.8*gf });
        out.push({ label:'gpu agrees with the reference, pool on', measured: gp.toFixed(1) + ' against ' + r400.toFixed(1) + ' Hz',
          expected:'within 10 percent', pass: within(gp, r400, 0.1) });
        out.push({ label:'gpu agrees with the reference, kick synapses', measured: gkp.toFixed(1) + ' against ' + kOn.toFixed(1) + ' Hz',
          expected:'within 10 percent', pass: within(gkp, kOn, 0.1) });
      }
    } else out.push({ label:'WebGPU availability', measured:'unavailable, gpu checks skipped', expected:'runs where WebGPU exists', pass:false, skipped:true });
    return out;
  } },

{ name:'membrane floor', source:'the potassium reversal potential bounds hyperpolarization (Hille 2001); below rest the Izhikevich quadratic grows with the square of the distance, and a large enough inhibitory volley sends the membrane over threshold or to infinity',
  async run(){
    // The failure this guards against: a driven population in one feedback pool at a gain far past useful fires an onset volley, the pool's kick takes every membrane below the quadratic's range, and every cell goes to NaN and stays silent.
    // With the floor the same population settles at a low rate.
    // Both engines, both synapse modes.
    const N = 400;
    const mk = (vmin, syn) => {
      const idx = Uint32Array.from({ length:N }, (_, i) => i);
      return { kind:'net', count:N, pos:new Float32Array(3*N),
        ntype:new Uint8Array(N), bias:new Float32Array(N).fill(8),
        src:new Int32Array(N).fill(1), lidx:idx.slice(),
        preStart:new Int32Array(N+1), post:new Int32Array(0),
        w:new Float32Array(0), delay:new Uint8Array(0), synCount:0,
        plast:0, refrac:2, vmin, syn, psc:0, tauE:3, tauI:8, seed:1,
        pool:new Uint16Array(N).fill(1), poolK:Float32Array.from([0, -50*1000/N]),
        protocols:[{ idx, amp:3, mode:3, period:100, width:10, t0:0, duration:0 }] };
    };
    // spikes per cell in the second half of the run, after the volley
    const late = async (vmin, syn, gpu) => {
      let n = 0;
      const r = await simRun(mk(vmin, syn), 2000, { steps:1, gpu, onFrame:(fired, t) => {
        if(t >= 1000) for(let i=0;i<N;i++) n += fired[i]; } });
      return r.error ? -1 : n/N;
    };
    const offE = await late(0, 1, false), onE = await late(-90, 1, false);
    const offK = await late(0, 0, false), onK = await late(-90, 0, false);
    const out = [
      { label:'without the floor the volley kills the population (exp)', measured: offE.toFixed(2) + ' spikes per cell in the last second',
        expected:'none: every membrane diverged to NaN', pass: offE === 0 },
      { label:'with the floor the population settles (exp)', measured: onE.toFixed(2) + ' spikes per cell in the last second',
        expected:'firing goes on under the pool', pass: onE > 0.2 },
      { label:'without the floor the volley kills the population (kick)', measured: offK.toFixed(2),
        expected:'none', pass: offK === 0 },
      { label:'with the floor the population settles (kick)', measured: onK.toFixed(2),
        expected:'firing goes on', pass: onK > 0.2 },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const gOn = await late(-90, 1, true), gOff = await late(0, 1, true);
      out.push({ label:'gpu agrees with the reference', measured: gOn.toFixed(2) + ' with the floor, ' + gOff.toFixed(2) + ' without',
        expected:'settles with the floor, dies without, within 20 percent of the reference',
        pass: gOff === 0 && gOn > 0.2 && Math.abs(gOn - onE) <= 0.2*Math.max(gOn, onE) });
    } else out.push({ label:'WebGPU availability', measured:'unavailable, gpu check skipped', expected:'runs where WebGPU exists', pass:false, skipped:true });
    return out;
  } },

{ name:'homeostatic inhibitory plasticity', source:'Vogels et al. 2011 (Science): inhibitory plasticity drives excitatory rates toward a target',
  async run(){
    const pts = mkPts(1500, 350, 7);
    const net = wireConnect(pts, { ...CN, radius:250, prob:0.4, wExc:4, wInh:-6 }, null);
    Object.assign(net, { plast:1, aP:0, aM:0, tauS:20, wmax:30, iEta:0.005, iRho:5,
      protocols:[{ idx:Uint32Array.from({length:1500},(_,i)=>i), amp:11, mode:3,
        period:100, width:10, t0:0, duration:0 }] });
    const excIdx = [];
    for(let i=0;i<net.count;i++) if(NEURON_TYPES[net.ntype[i]].sign > 0) excIdx.push(i);
    let earlyE = 0, lateE = 0;
    const w0 = net.w.slice();
    const r = await simRun(net, 20000, { steps:2, weights:true, onFrame:(fired, t) => {
      let c = 0;
      for(const i of excIdx) if(fired[i]) c++;
      if(t < 2000) earlyE += c; else if(t >= 16000) lateE += c;
    } });
    const earlyHz = earlyE/excIdx.length/2, lateHz = lateE/excIdx.length/4;
    let dInh = 0, nInh = 0;
    for(let s=0;s<net.synCount;s++) if(w0[s] < 0){ dInh += Math.abs(r.w[s]) - Math.abs(w0[s]); nInh++; }
    return [
      { label:'starts above target', measured: earlyHz.toFixed(1) + ' Hz',
        expected:'> 8 Hz (target 5)', pass: earlyHz > 8 },
      { label:'converges toward target', measured: lateHz.toFixed(1) + ' Hz after 20 s',
        expected:'2-9 Hz, closer to 5 than the start', pass:
          lateHz > 2 && lateHz < 9 && Math.abs(lateHz-5) < Math.abs(earlyHz-5) },
      { label:'inhibitory weights strengthened', measured: 'mean |dW| = ' + (dInh/Math.max(1,nInh)).toFixed(3),
        expected:'> 0', pass: dInh/Math.max(1,nInh) > 0.05 },
    ];
  } },

{ name:'proxy downscaling invariance', source:'van Albada, Helias, Diesmann 2015 (PLOS Comput Biol): rate preservation under K-scaling with compensation',
  async run(){
    const full = await computeScenario(balancedNet);
    const rFull = await simRun(full, 3000, { perNeuron:true });
    const fullHz = mean(rFull.frames.slice(250))/full.count/0.002;
    // the compensation's targets are the presynaptic rates per class, which is what the page measures for it: on the measured rows the interneurons fire three times faster than the pyramids, and one mean rate for both put the small net 66 percent over (2026-09-24)
    const perClass = (net, r) => { let se = 0, si = 0, ne = 0, ni = 0;
      for(let i = 0; i < net.count; i++){ if(NEURON_TYPES[net.ntype[i]].sign > 0){ ne++; se += r.counts[i]; } else { ni++; si += r.counts[i]; } }
      return { E:se/Math.max(1, ne)/3, I:si/Math.max(1, ni)/3 }; };
    const fullCls = perClass(full, rFull);
    const small = await computeScenario(balancedNet, ed => {
      for(const sc of ed.nodes.filter(n => n.type==='scatter')){
        sc.params.count = Math.round(sc.params.count/5); sc.params.density = sc.params.density/5; }
      const cn = ed.nodes.find(n => n.type==='connect');
      cn.params.density = 20; cn.params.nuE = fullCls.E; cn.params.nuI = fullCls.I;
    });
    const rSmall = await simRun(small, 3000, {});
    const smallHz = mean(rSmall.frames.slice(250))/small.count/0.002;
    const un = await computeScenario(balancedNet, ed => {
      for(const sc of ed.nodes.filter(n => n.type==='scatter')){
        sc.params.count = Math.round(sc.params.count/5); sc.params.density = sc.params.density/5; }
    });
    const rUn = await simRun(un, 3000, {});
    const unHz = mean(rUn.frames.slice(250))/un.count/0.002;
    // The same scene on exponential synapses in the peak convention, where a weight is a peak current and the charge a spike delivers is w tau: the compensation has to restore the mean input in that charge, not in w.
    // The kick weights are charges, so they become peaks by tau.
    const exp = ed => {
      const cn = ed.nodes.find(n => n.type==='connect');
      cn.params.wExc = cn.params.wExc/3; cn.params.wInh = cn.params.wInh/8;
      const cp = ed.nodes.find(n => n.type==='checkpoint');
      cp.params.syn = 1; cp.params.psc = 0; cp.params.tauE = 3; cp.params.tauI = 8;
    };
    const fullX = await computeScenario(balancedNet, exp);
    const rFullX = await simRun(fullX, 3000, { perNeuron:true });
    const fullXHz = mean(rFullX.frames.slice(250))/fullX.count/0.002;
    const fullXCls = perClass(fullX, rFullX);
    const smallX = await computeScenario(balancedNet, ed => {
      exp(ed);
      for(const sc of ed.nodes.filter(n => n.type==='scatter')){
        sc.params.count = Math.round(sc.params.count/5); sc.params.density = sc.params.density/5; }
      const cn = ed.nodes.find(n => n.type==='connect');
      cn.params.density = 20; cn.params.nuE = fullXCls.E; cn.params.nuI = fullXCls.I;
    });
    const rSmallX = await simRun(smallX, 3000, {});
    const smallXHz = mean(rSmallX.frames.slice(250))/smallX.count/0.002;
    return [
      { label:'compensated rate vs full scale', measured: smallHz.toFixed(1) + ' vs ' + fullHz.toFixed(1) + ' Hz',
        expected:'within 30%', pass: Math.abs(smallHz-fullHz)/Math.max(0.1,fullHz) < 0.3 },
      { label:'compensation beats no compensation', measured:
          `err ${Math.abs(smallHz-fullHz).toFixed(1)} vs ${Math.abs(unHz-fullHz).toFixed(1)} Hz`,
        expected:'compensated error smaller', pass: Math.abs(smallHz-fullHz) < Math.abs(unHz-fullHz) },
      { label:'compensated rate vs full scale, exponential synapses in the peak convention',
        measured: smallXHz.toFixed(1) + ' vs ' + fullXHz.toFixed(1) + ' Hz',
        expected:'within 30%', pass: fullXHz > 0.5 && Math.abs(smallXHz-fullXHz)/Math.max(0.1,fullXHz) < 0.3 },
    ];
  } },

{ name:'the tissue scenes fire in their bands', source:'every Tab-menu scene at full density, run on the reference engine for two seconds on each of three noise seeds and held, by the mean of the three, to the rates the literature reports for that tissue: awake cortical excitatory cells 0.5 to 25 Hz and interneurons above them (de Kock and Sakmann 2009; Buzsaki and Mizuseki 2014; Gentet et al. 2010), thalamic relay cells 5 to 40 Hz in tonic mode (Steriade, McCormick and Sejnowski 1993), CA3 pyramids under 3 Hz at rest with recall as an event (Mizuseki and Buzsaki 2013), retinal ganglion cells 3 to 60 Hz under a moving bar (Meister and Berry 1999), a balanced net 1 to 30 Hz asynchronous (Brunel 2000). A scene that drifts out of its band fails here rather than misleading quietly.',
  run: async () => {
    // [tag or regex, low Hz, high Hz]; a band on a population the scene does not have is a failure too
    const BANDS = {
      'guided tour': [['sheet', 3, 30], ['sheeti', 3, 40], ['pacemaker', 15, 150]],
      'traveling waves': [['sheet', 3, 30], ['sheeti', 3, 40], ['pacemaker', 15, 150]],
      'mouse cortical column': [['L2/3e', 0.5, 25], ['L4e', 0.5, 25], ['L5e', 0.5, 30], ['L6e', 0.5, 25], ['L2/3i', 1, 40], ['L4i', 1, 40]],
      'visual pathway': [['rgc', 3, 60], ['lgn', 3, 40], ['L4e', 2, 30], ['L2/3e', 1, 30]],
      'thalamocortical loop': [['ce', 1, 25], ['ci', 2, 40], ['tc', 5, 40], ['rz', 2, 40]],
      'CA3 pattern completion': [['ca3', 0.1, 3], ['ca3i', 0.5, 10]],
      'balanced random net': [['exc', 1, 30], ['inh', 1, 40]],
      'sound localization': [['detectors', 0.5, 10], ['left', 3, 8], ['right', 3, 8]],
      'virtual patch rig': [['rs', 1, 100], ['ib', 1, 100], ['ch', 1, 200], ['fs', 1, 300], ['lts', 1, 200], ['tc', 1, 200], ['rz', 1, 300]],
      'seizure and control': [['exc', 1, 30], ['inh', 1, 40]],
      'gamma from inhibition': [['pyr', 5, 40], ['fs', 10, 80]],
      'working memory': [['mema', 2, 12], ['memb', 0, 3], ['inh', 0.3, 12]],
      'polychrony': [['srca', 3, 8], ['srcb', 3, 8], ['srcc', 3, 8], ['readx', 0.3, 6], ['ready', 0.3, 6]],
      'every node': [['exc', 1, 30], ['inh', 1, 40], ['tier1.ray1.arm', 0.5, 100], ['tier3.ray1.arm', 0.5, 100], ['tier1.ray1.fork1.bud', 0.5, 100], ['tier3.ray1.fork1.bud', 0.5, 100]],
    };
    const out = [];
    for(const sc of TISSUE_SCENES){
      const bands = BANDS[sc.name];
      if(!bands){ out.push({ label:sc.name + ' has a band', measured:'no band listed', expected:'a band per tissue scene', pass:false }); continue; }
      let net;
      try { net = await computeScenario(sc.build); }
      catch(e){ out.push({ label:sc.name + ' computes', measured:String(e.message || e), expected:'a network', pass:false }); continue; }
      const n = net.count, tagOf = i => (net.tags && net.tags[net.src[i]]) || '';
      net.inputs = (net.inputMaps || []).map(m => ({ chStart:m.chStart, chIdx:m.chIdx, chW:m.chW, amp:m.amp }));   // what the page's engineInputs hands the engine
      // CA3: spikes per 100 ms on the pyramids, for the recall event
      const isCa3 = /CA3/.test(sc.name); const bins = []; let bin = 0;
      const ms = 2000;
      // Three draws, not one.
      // A rate from a single seed can sit inside a band by luck or fall outside it the same way, and the bands are wide enough that a scene drifting out of one should be visible in every draw.
      // The network is computed once and only the engine's noise and initial state change, so this is three runs and not three wirings.
      const SEEDS = [net.seed, net.seed + 101, net.seed + 202];
      const perSeed = [];
      let failed = null;
      for(const sd of SEEDS){
        // the visual pathway is driven by its bar sweep, the way the page drives it; a fresh sweep per run, so every seed sees the same input
        const maps = net.inputMaps || [], bar = new BarSweep(), ist = maps.map(() => ({}));
        const driver = maps.length ? (w, t) => { if(t % 10 === 0) maps.forEach((m, mi) => w.postMessage({ cmd:'inputFrame', mi, gains:encodeGains(bar.sample(t, m), m, ist[mi], t, 3) })); } : null;
        // the recall event is a claim about one run, measured on the first
        const onFrame = isCa3 && !perSeed.length ? (fired, t) => { for(let i = 0; i < n; i++) if(fired[i] && tagOf(i) === 'ca3') bin += fired[i]; if(t > 0 && t % 100 === 0){ bins.push(bin); bin = 0; } } : null;
        const r = await simRun(net, ms, { steps:2, perNeuron:true, driver, onFrame, seed:sd });
        if(r.error){ failed = r.error; break; }
        const byTag = new Map();
        for(let i = 0; i < n; i++){ const t = tagOf(i); if(!byTag.has(t)) byTag.set(t, { n:0, c:0 }); const b = byTag.get(t); b.n++; b.c += r.counts[i]; }
        perSeed.push(byTag);
      }
      if(failed){ out.push({ label:sc.name + ' runs', measured:failed, expected:'a run', pass:false }); continue; }
      for(const [tag, lo, hi] of bands){
        const cells = (perSeed[0].get(tag) || {}).n;
        const hzs = perSeed.map(byTag => { const b = byTag.get(tag); return b ? b.c/(b.n*ms/1000) : NaN; });
        const mean = hzs.reduce((a, b) => a + b, 0)/hzs.length;
        out.push({ label:sc.name + ': ' + tag,
          measured: cells ? mean.toFixed(2) + ' Hz mean of ' + hzs.length + ' seeds ('
            + hzs.map(x => x.toFixed(2)).join(', ') + '; ' + cells + ' cells)' : 'no population ' + tag,
          expected: lo + ' to ' + hi + ' Hz', pass: !!cells && mean >= lo && mean <= hi });
      }
      if(isCa3){
        const nCa3 = (perSeed[0].get('ca3') || { n:1 }).n;
        const hzBins = bins.map(c => c/(nCa3*0.1));
        const sorted = hzBins.slice().sort((a, b) => a - b), median = sorted[sorted.length >> 1] || 0, peak = Math.max(...hzBins);
        out.push({ label:'CA3 pattern completion: recall is an event', measured:'peak 100 ms bin ' + peak.toFixed(1) + ' Hz against a median bin of ' + median.toFixed(2),
          expected:'peak above 10 Hz and above ten times the median', pass: peak > 10 && peak > 10*Math.max(0.05, median) });
      }
    }
    return out;
  } },
{ name:'the teaching scenes do what their notes say', source:'the claim each scene makes in its own notes, measured on the reference engine at full density (2026-09-15): the coincidence detectors peak in the middle at a 2 ms lead and in the right third at 6 ms (Jeffress 1948); every cell on the rig fires under the ramp and the fast spiking cell outruns the regular one (Connors and Gutnick 1990); the seizure scene runs away while inhibition is blocked and settles once inhibition returns (Brunel 2000 regimes); the gamma patch has its spectral peak between 30 and 80 Hz (Whittington et al. 2000).',
  run: async () => {
    const out = [];
    const rateIn = (net, r, x, rad, tag) => { let c = 0, n = 0; for(let i = 0; i < net.count; i++){ const dx = net.pos[i*3] - x, dy = net.pos[i*3+1], dz = net.pos[i*3+2]; if(dx*dx + dy*dy + dz*dz <= rad*rad && (net.tags[net.src[i]] || '') === tag){ n++; c += r.counts[i]; } } return n ? c/n : 0; };
    // sound localization: the place that fires follows the lead
    {
      const build = TISSUE_SCENES.find(x => x.name === 'sound localization').build;
      const at = async lead => { const net = await computeScenario(build, ed => { const clicks = ed.nodes.filter(n => n.type === 'stimulus'); clicks[1].params.t0 = lead; });
        const r = await simRun(net, 1000, { steps:2, perNeuron:true });
        return { left:rateIn(net, r, -260, 130, 'detectors'), mid:rateIn(net, r, 0, 130, 'detectors'), right:rateIn(net, r, 260, 130, 'detectors') }; };
      const a = await at(2), b = await at(6);
      out.push({ label:'sound localization: at a 2 ms lead the middle fires most', measured:'left ' + a.left.toFixed(1) + ', middle ' + a.mid.toFixed(1) + ', right ' + a.right.toFixed(1) + ' Hz', expected:'middle above both ends', pass:a.mid > a.left && a.mid > a.right });
      out.push({ label:'sound localization: at a 6 ms lead the peak is in the right third', measured:'left ' + b.left.toFixed(1) + ', middle ' + b.mid.toFixed(1) + ', right ' + b.right.toFixed(1) + ' Hz', expected:'right above the middle and the left', pass:b.right > b.mid && b.right > b.left });
    }
    // the rig: every cell answers the ramp, fast spiking fastest of the cortical rows
    {
      const build = TISSUE_SCENES.find(x => x.name === 'virtual patch rig').build;
      const net = await computeScenario(build);
      const r = await simRun(net, 3000, { steps:1, perNeuron:true });
      const hz = {}; for(let i = 0; i < net.count; i++) hz[net.tags[net.src[i]]] = r.counts[i]/3;
      out.push({ label:'patch rig: every cell fires under the ramp', measured:Object.entries(hz).map(([k, v]) => k + ' ' + v.toFixed(1)).join(', ') + ' Hz', expected:'each above 1 Hz over the three seconds', pass:Object.values(hz).every(v => v > 1) });
      out.push({ label:'patch rig: the fast spiking cell outruns the regular spiking one', measured:'fs ' + hz.fs.toFixed(1) + ' against rs ' + hz.rs.toFixed(1) + ' Hz', expected:'fs above rs', pass:hz.fs > hz.rs });
    }
    // the seizure: runaway during the block, recovery after
    {
      const build = TISSUE_SCENES.find(x => x.name === 'seizure and control').build;
      const net = await computeScenario(build);
      const bins = []; let bin = 0;
      const r = await simRun(net, 7500, { steps:2, onFrame:(fired, t) => { for(let i = 0; i < fired.length; i++) bin += fired[i]; if(t > 0 && t % 500 === 0){ bins.push(bin/(net.count*0.5)); bin = 0; } } });
      const avg = a => a.reduce((x, y) => x + y, 0)/Math.max(1, a.length);
      const before = avg(bins.slice(2, 6)), during = avg(bins.slice(6, 10)), after = avg(bins.slice(12, 14));   // 6 to 7 s, after the block ends at 5
      out.push({ label:'seizure: the rate runs away while inhibition is blocked', measured:before.toFixed(1) + ' Hz before, ' + during.toFixed(1) + ' during, ' + after.toFixed(1) + ' after', expected:'during above twice before', pass:!r.error && during > 2*before });
      out.push({ label:'seizure: the population settles once inhibition returns', measured:after.toFixed(1) + ' Hz against ' + before.toFixed(1) + ' before', expected:'after within 50 percent of before', pass:!r.error && Math.abs(after - before) < 0.5*Math.max(0.5, before) });
    }
    // polychrony: each reader answers the order that lines the spikes up at it, and not the other
    {
      const build = TISSUE_SCENES.find(x => x.name === 'polychrony').build;
      const net = await computeScenario(build);
      const tagOf = i => (net.tags && net.tags[net.src[i]]) || '';
      const who = new Uint8Array(net.count); let nX = 0, nY = 0;
      for(let i = 0; i < net.count; i++){ const t = tagOf(i); if(t === 'readx'){ who[i] = 1; nX++; } else if(t === 'ready'){ who[i] = 2; nY++; } }
      // spikes per cell in the 60 ms after each order's first pulse, over four cycles
      const c = { x1:0, x2:0, y1:0, y2:0 };
      const r = await simRun(net, 1600, { steps:1, onFrame:(fired, t) => {
        const ph = t % 400, w = ph < 60 ? 1 : (ph >= 200 && ph < 260) ? 2 : 0;
        if(!w) return;
        for(let i = 0; i < fired.length; i++){ const f = fired[i]; if(!f || !who[i]) continue;
          if(who[i] === 1){ if(w === 1) c.x1 += f; else c.x2 += f; } else { if(w === 1) c.y1 += f; else c.y2 += f; } } } });
      const per = (n, cells) => n/(cells*4);
      const x1 = per(c.x1, nX), x2 = per(c.x2, nX), y1 = per(c.y1, nY), y2 = per(c.y2, nY);
      out.push({ label:'polychrony: the reader beside a answers the order c, b, a', measured:r.error || (x1.toFixed(2) + ' spikes per cell per volley against ' + x2.toFixed(2) + ' for a, b, c'), expected:'above 0.2 and above five times the other order', pass:!r.error && x1 > 0.2 && x1 > 5*x2 });
      out.push({ label:'polychrony: the reader beside c answers the order a, b, c', measured:r.error || (y2.toFixed(2) + ' spikes per cell per volley against ' + y1.toFixed(2) + ' for c, b, a'), expected:'above 0.2 and above five times the other order', pass:!r.error && y2 > 0.2 && y2 > 5*y1 });
    }
    // working memory: the read pulse is answered by the cued population, and not when the facilitation is short
    {
      const build = TISSUE_SCENES.find(x => x.name === 'working memory').build;
      const measure = async mutate => {
        const net = await computeScenario(build, mutate);
        const tagOf = i => (net.tags && net.tags[net.src[i]]) || '';
        const who = new Uint8Array(net.count); let nA = 0, nB = 0;
        for(let i = 0; i < net.count; i++){ const t = tagOf(i); if(t === 'mema'){ who[i] = 1; nA++; } else if(t === 'memb'){ who[i] = 2; nB++; } }
        let readA = 0, readB = 0, gapA = 0;
        const r = await simRun(net, 3000, { steps:2, onFrame:(fired, t) => {
          const inRead = t >= 2100 && t < 2300, inGap = t >= 1100 && t < 2100;
          if(!inRead && !inGap) return;
          for(let i = 0; i < fired.length; i++){ const f = fired[i]; if(!f) continue;
            if(inRead){ if(who[i] === 1) readA += f; else if(who[i] === 2) readB += f; } else if(who[i] === 1) gapA += f; } } });
        return { err:r.error, a:readA/(nA*0.2), b:readB/(nB*0.2), gap:gapA/(nA*1.0) };   // the first cue is at 0.5 s, its read at 2.1
      };
      const m = await measure();
      out.push({ label:'working memory: the read pulse is answered by the cued population', measured:m.err || ('cued ' + m.a.toFixed(1) + ' Hz, other ' + m.b.toFixed(1) + ' Hz in the 200 ms of the read'), expected:'cued above 8 Hz and above twice the other', pass:!m.err && m.a > 8 && m.a > 2*m.b });
      out.push({ label:'working memory: nothing fires between the cue and the read', measured:m.err || (m.gap.toFixed(2) + ' Hz in the cued population from 1.1 to 2.1 s'), expected:'under 1 Hz: the item is in the synapses', pass:!m.err && m.gap < 1 });
      const c = await measure(ed => { ed.nodes.find(n => n.type === 'checkpoint').params.stpTauF = 50; });
      out.push({ label:'working memory: with a 50 ms facilitation the two answer alike', measured:c.err || ('cued ' + c.a.toFixed(1) + ' Hz, other ' + c.b.toFixed(1) + ' Hz'), expected:'within 40 percent of each other', pass:!c.err && Math.abs(c.a - c.b) < 0.4*Math.max(c.a, c.b, 1) });
    }
    // gamma: the spectral peak of the population rate
    {
      const build = TISSUE_SCENES.find(x => x.name === 'gamma from inhibition').build;
      const net = await computeScenario(build);
      const r = await simRun(net, 1500, { steps:1 });
      const x = r.frames.slice(300); const N = x.length, mean = x.reduce((a, b) => a + b, 0)/N;
      let best = [0, 0], total = 0;
      for(let k = 1; k < N/2; k++){ let re = 0, im = 0; for(let j = 0; j < N; j++){ const ph = 2*Math.PI*k*j/N; re += (x[j] - mean)*Math.cos(ph); im -= (x[j] - mean)*Math.sin(ph); }
        const p = re*re + im*im; total += p; if(p > best[1]) best = [k/(N*0.001), p]; }
      out.push({ label:'gamma: the population rate peaks in the gamma band', measured:best[0].toFixed(1) + ' Hz carrying ' + (100*best[1]/total).toFixed(0) + ' percent of the power', expected:'30 to 80 Hz, above 10 percent', pass:best[0] >= 30 && best[0] <= 80 && best[1] > 0.1*total });
    }
    return out;
  } },
{ name:'traveling wave front speed', source:'Chervin et al. 1988; Golomb and Amitai 1997: slice propagation 1-10 cm/s',
  async run(){
    // fronts: runs of busy frames over which the mean distance of the firing cells from the pacemaker grows.
    // The traveling waves scene is the 1.2 mm sheet the speed is measured on; the guided tour opens on a 0.7 mm sheet of the same tissue, which a front crosses sooner, so its fronts are shorter runs over a shorter distance.
    const fronts = async (build, cx, cz, frames, grow) => {
      const net = await computeScenario(build), pos = net.pos, traj = [];
      await simRun(net, 1400, { steps:2, onFrame: fired => {
        let n = 0, d = 0;
        for(let i=0;i<fired.length;i++) if(fired[i]){
          const dx = pos[i*3]-cx, dz = pos[i*3+2]-cz;
          d += Math.sqrt(dx*dx+dz*dz); n++;
        }
        traj.push([n, n ? d/n : 0]);
      } });
      const speeds = [];
      let cur = null;
      for(let i=0;i<traj.length;i++){
        if(traj[i][0] > 300){ if(!cur) cur = []; cur.push(traj[i][1]); }
        else if(cur){
          if(cur.length >= frames && cur[cur.length-1] > cur[0] + grow)
            speeds.push((cur[cur.length-1]-cur[0])/(cur.length*2));
          cur = null;
        }
      }
      return speeds;
    };
    const big = await fronts(travelingWaves, -400, -300, 8, 400);
    const v = big.length ? mean(big) : 0;
    const tour = await fronts(guidedTour, -220, -170, 5, 200);   // output 1 = the tour's sheet
    const vt = tour.length ? mean(tour) : 0;
    return [
      { label:'traveling waves: expanding wave fronts detected', measured: String(big.length),
        expected:'>= 2 in 1.4 s', pass: big.length >= 2 },
      { label:'traveling waves: front speed', measured: (v/10).toFixed(1) + ' cm/s',
        expected:'1-10 cm/s (slice band)', pass: v >= 10 && v <= 100 },
      // The tour's sheet is 700 um with a 120 um source in one quadrant, so a front crosses about 250 um: the speed printed here is over a run of a few frames and is not a measurement of propagation speed, which is why this line gates on fronts and the line above gates the speed.
      // On the measured rows the tour reads 24 fronts at 0.8 cm/s against the big sheet's 2.1, and on the classic rows both read 3.0 (2026-09-24).
      { label:'guided tour: its smaller sheet carries fronts too', measured: tour.length + ' fronts at ' + (vt/10).toFixed(1) + ' cm/s',
        expected:'>= 2 in 1.4 s (the speed is gated on the 1.2 mm sheet)', pass: tour.length >= 2 },
    ];
  } },

{ name:'framed input drive', source:'ENGINE.md inputFrame contract: channel gains inject constant current between frames; silent without drive',
  async run(){
    // isolated neurons (no synapses, no bias, no noise): activity can only come from the input map
    const pts = mkPts(500, 200, 3, 0);
    const net = wireConnect(pts, { ...CN, prob:0 }, null);
    const C = 4, per = Math.ceil(500/C);
    const chStart = new Uint32Array(C+1), chIdx = new Uint32Array(500),
          chW = new Float32Array(500).fill(1);
    for(let i=0;i<500;i++) chIdx[i] = i;
    for(let c=0;c<=C;c++) chStart[c] = Math.min(500, c*per);
    net.inputs = [{ chStart, chIdx, chW, amp:1 }];
    const run = gpu => simRun(net, 1200, { steps:2, gpu, driver:(w, t) => {
      if(t === 600) w.postMessage({ cmd:'inputFrame', mi:0,
        gains:Float32Array.from([8, 8, 0, 0]) });   // drive channels 0-1 only
    }, perNeuron:true });
    const seg = (r, a, b) => r.frames.slice(a/2, b/2).reduce((x,y)=>x+y,0);
    const c1 = await run(false);
    const cQuiet = seg(c1, 0, 600), cDrive = seg(c1, 700, 1200);
    let hit = 0, spared = 0;
    for(let i=0;i<500;i++){
      if(i < chStart[2] && c1.counts[i]) hit++;
      if(i >= chStart[2] && c1.counts[i]) spared++;
    }
    const checks = [
      { label:'silent without drive', measured: cQuiet + ' spikes in 600 ms',
        expected:'0', pass: cQuiet === 0 },
      { label:'fires under drive', measured: cDrive + ' spikes in 500 ms',
        expected:'> 100', pass: cDrive > 100 },
      { label:'only driven channels fire', measured:
          `${hit} driven neurons active, ${spared} undriven active`,
        expected:'driven > 200, undriven = 0', pass: hit > 200 && spared === 0 },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const g1 = await run(true);
      const gQuiet = seg(g1, 0, 600), gDrive = seg(g1, 700, 1200);
      checks.push({ label:'gpu engine parity', measured:
          `quiet ${gQuiet}, driven ${gDrive} vs cpu ${cDrive}`,
        expected:'quiet 0, driven within 30%', pass:
          gQuiet === 0 && Math.abs(gDrive-cDrive)/cDrive < 0.3 });
    }
    // encoder compute structure: ON/OFF mosaics, arbor normalization, distributed-projection determinism
    const net2 = wireConnect(mkPts(1200, 250, 9, 0), { ...CN, prob:0 }, null);
    const P = { source:0, map:0, code:1, axis:1, cols:8, rows:8, arbor:0.8,
      fanin:8, seed:1, transient:0.6, jitter:1, amp:6, period:1500 };
    // The signal is a node of its own on the input's third port (a curriculum, test signal, live or read node) and the input node is the mapping only; a computed test signal at its defaults stands in.
    const sig = NODE_DEFS.testsignal.compute([], Object.fromEntries(NODE_DEFS.testsignal.params.map(q => [q.k, structuredClone(q.def)])), { id:43 });
    const mk = q => NODE_DEFS.input.compute([net2, null, sig], { ...P, ...q }, { id:42 }).inputMaps[0];
    const ret = mk({}), G = 64;
    const onSet = new Set(), offSet = new Set();
    for(let c=0;c<G;c++) for(let k=ret.chStart[c];k<ret.chStart[c+1];k++) onSet.add(ret.chIdx[k]);
    for(let c=G;c<2*G;c++) for(let k=ret.chStart[c];k<ret.chStart[c+1];k++) offSet.add(ret.chIdx[k]);
    let overlap = 0; for(const i of onSet) if(offSet.has(i)) overlap++;
    const wByN = new Map();
    for(let c=0;c<2*G;c++) for(let k=ret.chStart[c];k<ret.chStart[c+1];k++)
      wByN.set(ret.chIdx[k], (wByN.get(ret.chIdx[k]) || 0) + ret.chW[k]);
    let badNorm = 0; for(const [, s] of wByN) if(Math.abs(s - 1) > 0.01) badNorm++;
    const d1 = mk({ map:2 }), d2 = mk({ map:2 }), d3 = mk({ map:2, seed:2 });
    const same = d1.chIdx.length === d2.chIdx.length &&
      d1.chIdx.every((x, k) => x === d2.chIdx[k] && d1.chW[k] === d2.chW[k]);
    const diff = d3.chIdx.some((x, k) => x !== d1.chIdx[k]);
    checks.push(
      { label:'retinal coding splits ON/OFF mosaics', measured:
          `${onSet.size} ON, ${offSet.size} OFF targets, ${overlap} overlap`,
        expected:'disjoint mosaics, both > 400 of 1200', pass:
          overlap === 0 && onSet.size > 400 && offSet.size > 400 },
      { label:'arbor weights give unit drive per neuron', measured:
          badNorm + ' neurons off unit weight sum',
        expected:'0', pass: badNorm === 0 },
      { label:'distributed projection deterministic per seed', measured:
          (same ? 'seed 1 reproducible' : 'seed 1 DIFFERS') + ', seed 2 ' + (diff ? 'differs' : 'IDENTICAL'),
        expected:'same seed identical, new seed differs', pass: same && diff },
    );
    return checks;
  } },

{ name:'curriculum lesson stream', topic:'curriculum', source:'design rule: interleaved varied items with gaps, and one schedule shared by every modality so paired input is synchronous (Bahrick and Lickliter intersensory redundancy)',
  async run(){
    const sig = { set:0, order:1, onMs:400, offMs:350, jitter:0.18,
      scaleVar:0.25, rotVar:12, seed:1 };
    // schedule is a pure function of simulated time: two encoders sampling independently must agree on the item, which is what makes the glyph and the spoken name arrive together
    let agree = true, onCount = 0, offCount = 0;
    const seen = new Map();
    for(let t=0; t<60000; t+=25){
      const a = scheduleAt(sig, t), b = scheduleAt(sig, t);
      if(a.item !== b.item || a.on !== b.on) agree = false;
      if(a.on){ onCount++; seen.set(a.item, (seen.get(a.item)||0)+1); }
      else offCount++;
    }
    const distinct = seen.size;
    // consecutive presentations differ (a repeating stimulus is what lets spike-timing plasticity ratchet one pathway)
    let reps = 0, prev = null, pres = 0;
    for(let k=0;k<200;k++){
      const st = scheduleAt(sig, k*(sig.onMs+sig.offMs) + 10);
      if(prev !== null && st.item === prev) reps++;
      prev = st.item; pres++;
    }
    // the same item is posed differently each time it appears
    const poses = [];
    for(let k=0;k<400 && poses.length<2;k++){
      const st = scheduleAt(sig, k*(sig.onMs+sig.offMs) + 10);
      if(st.item === 'A') poses.push(st);
    }
    const posed = poses.length === 2 &&
      (Math.abs(poses[0].dx-poses[1].dx) > 1e-6 || Math.abs(poses[0].scale-poses[1].scale) > 1e-6);
    // audio: the letter names of B and D share a vowel but differ at onset, so the streams must separate early and converge late
    const specDist = (x, y, u) => {
      const p = renderAudio({ ...scheduleAt(sig, 0), item:x, u }, 24);
      const q = renderAudio({ ...scheduleAt(sig, 0), item:y, u }, 24);
      let d = 0; for(let i=0;i<24;i++) d += Math.abs(p[i]-q[i]);
      return d;
    };
    const early = specDist('B','D',0.05), late = specDist('B','D',0.8);
    const far = specDist('B','O',0.8);
    // visual: a glyph fills part of the grid and different letters differ
    const gA = renderItem(scheduleAt(sig, 0).item === 'A' ? scheduleAt(sig, 0)
      : { ...scheduleAt(sig, 0), item:'A' }, 20, 20);
    const gO = renderItem({ ...scheduleAt(sig, 0), item:'O' }, 20, 20);
    let lit = 0, diff = 0;
    for(let i=0;i<gA.length;i++){ if(gA[i] > 0.05) lit++; diff += Math.abs(gA[i]-gO[i]); }
    return [
      { label:'schedule is a pure function of time (modalities agree)',
        measured: agree ? 'identical across independent samples' : 'DIVERGED',
        expected:'identical', pass: agree },
      { label:'presentations are interleaved with gaps', measured:
          `${onCount} on / ${offCount} off samples, ${distinct} distinct items in 60 s`,
        expected:'both phases present, > 10 distinct items', pass:
          onCount > 100 && offCount > 100 && distinct > 10 },
      { label:'consecutive items rarely repeat', measured:
          `${reps} repeats in ${pres} presentations`,
        expected:'< 15%', pass: reps/pres < 0.15 },
      { label:'the same item is re-posed each appearance', measured:
          posed ? 'position and size vary' : 'IDENTICAL POSE',
        expected:'varies', pass: posed },
      { label:'letter names share vowels but differ at onset', measured:
          `B vs D: ${early.toFixed(2)} early, ${late.toFixed(2)} late; B vs O late ${far.toFixed(2)}`,
        expected:'same-vowel pair converges late, different-vowel pair does not', pass:
          early > late && far > late },
      { label:'glyphs render distinctly into the encoder grid', measured:
          `${lit} of ${gA.length} cells lit, A vs O distance ${diff.toFixed(1)}`,
        expected:'20-70% lit, distance > 5', pass:
          lit > gA.length*0.05 && lit < gA.length*0.75 && diff > 5 },
      ...(() => {
        // spoken words are a phone sequence, so the spectrum has to move across the presentation the way a real word does
        const objSig = { ...sig, set:5 };
        const at = u => renderAudio({ ...scheduleAt(objSig, 0), item:'banana',
          kind:'object', u }, 24);
        const d = (a, b) => { let s2 = 0; for(let i=0;i<24;i++) s2 += Math.abs(a[i]-b[i]); return s2; };
        const a1 = at(0.08), a2 = at(0.45), a3 = at(0.85);
        const moves = d(a1, a2) + d(a2, a3);
        // a picture-book page carries the drawing above and the word below
        const bookSig = { ...sig, set:7, jitter:0, scaleVar:0, rotVar:0 };
        const bk = renderItem({ ...scheduleAt(bookSig, 0), item:'apple', kind:'book',
          dx:0, dy:0, scale:1, rot:0 }, 24, 24, 1);
        let top = 0, bot = 0;
        for(let r=0;r<24;r++) for(let c=0;c<24;c++)
          (r >= 12 ? (top += bk[r*24+c]) : (bot += bk[r*24+c]));
        // acuity: blurring lowers the spatial gradient of the same item
        const grad = g => { let s2 = 0;
          for(let r=0;r<24;r++) for(let c=1;c<24;c++) s2 += Math.abs(g[r*24+c]-g[r*24+c-1]);
          return s2; };
        const st0 = { ...scheduleAt(bookSig, 0), item:'star', kind:'shape',
          dx:0, dy:0, scale:1, rot:0 };
        const sharp = grad(renderItem(st0, 24, 24, 1));
        const blur = grad(renderItem(st0, 24, 24, 0.15));
        return [
          { label:'spoken words move through a phone sequence', measured:
              `spectral change across the presentation ${moves.toFixed(2)}`,
            expected:'> 2 (a static tone would not move)', pass: moves > 2 },
          { label:'picture book page holds image and word', measured:
              `picture half ${top.toFixed(1)}, word half ${bot.toFixed(1)}`,
            expected:'both halves carry energy', pass: top > 1 && bot > 1 },
          { label:'acuity ramp blurs early input', measured:
              `gradient ${sharp.toFixed(1)} sharp vs ${blur.toFixed(1)} blurred`,
            expected:'blurred is lower', pass: blur < sharp*0.8 },
        ];
      })(),
      ...(() => {
        // unimodal probes: every fourth presentation drops one modality, alternating which, and the choice is a pure function of the presentation index so both encoders agree without shared state
        const pSig = { ...sig, probeEvery:4 };
        const conds = [];
        for(let k=0;k<40;k++)
          conds.push(scheduleAt(pSig, k*(sig.onMs+sig.offMs) + 10).cond);
        const probes = conds.filter(c => c !== 0).length;
        const vis = conds.filter(c => c === 1).length;
        const aud = conds.filter(c => c === 2).length;
        const order = conds.filter(c => c !== 0).every((c, i) => c === (i % 2 ? 2 : 1));
        let stable = true;
        for(let k=0;k<40;k++){
          const base = k*(sig.onMs+sig.offMs);
          const a = scheduleAt(pSig, base + 10).cond;
          const b = scheduleAt(pSig, base + sig.onMs - 10).cond;
          if(a !== b) stable = false;
        }
        const off = [];
        for(let k=0;k<40;k++) off.push(scheduleAt(sig, k*(sig.onMs+sig.offMs) + 10).cond);
        return [
          { label:'unimodal probes fire at the configured rate', measured:
              `${probes} probes in 40 presentations (${vis} visual, ${aud} audio)`,
            expected:'10 probes, split evenly, alternating', pass:
              probes === 10 && vis === 5 && aud === 5 && order },
          { label:'probe condition is constant within a presentation', measured:
              stable ? 'same condition at onset and offset' : 'CHANGED MID-PRESENTATION',
            expected:'constant', pass: stable },
          { label:'probes are off by default', measured:
              `${off.filter(c => c !== 0).length} probes with probeEvery unset`,
            expected:'0', pass: off.every(c => c === 0) },
        ];
      })(),
    ];
  } },

{ name:'readout measures', topic:'analysis', source:'observer-side analysis (analysis.js): lifetime sparseness (Rolls and Tovee 1995), split-half nearest-centroid decoding, and cross-modal completion. Checked against synthetic activity with a known answer, since a measure that cannot tell structure from noise cannot report learning',
  async run(){
    const nI = 8, nN = 64, reps = 6;
    // a network that has learned: each item drives its own tenth of the population. a network that has not: identical activity every time.
    const mk = (structured, seed) => {
      let z = seed;
      const rnd = () => (z = (z*1664525 + 1013904223) >>> 0) / 4294967296;
      const trials = new Float32Array(nI*reps*nN), labels = new Int32Array(nI*reps);
      for(let i=0;i<nI;i++) for(let r=0;r<reps;r++){
        const t = i*reps + r; labels[t] = i;
        for(let j=0;j<nN;j++){
          const own = structured && Math.floor(j*nI/nN) === i;
          trials[t*nN + j] = (own ? 6 : 0) + rnd()*2;
        }
      }
      return { trials, labels };
    };
    const good = mk(true, 7), flat = mk(false, 7);
    const dGood = decodeAccuracy(good.trials, good.labels, nI, nN);
    const dFlat = decodeAccuracy(flat.trials, flat.labels, nI, nN);
    // centroids from the structured set, then a partial version of each item built from a different noise draw of the same tuning
    const cent = new Float32Array(nI*nN), part = new Float32Array(nI*nN);
    for(let i=0;i<nI;i++) for(let j=0;j<nN;j++){
      const own = Math.floor(j*nI/nN) === i;
      cent[i*nN + j] = own ? 6 : 0.5;
      part[i*nN + j] = own ? 3 : 0.4;
    }
    const comp = completionScore(part, cent, nI, nN);
    const shuf = new Float32Array(nI*nN);
    for(let i=0;i<nI;i++) shuf.set(part.subarray(((i+1)%nI)*nN, ((i+1)%nI+1)*nN), i*nN);
    const compShuf = completionScore(shuf, cent, nI, nN);
    const selGood = selectivityProfile(cent, nI, nN);
    const cflat = new Float32Array(nI*nN).fill(3);
    const selFlat = selectivityProfile(cflat, nI, nN);
    // a silent population must not read as selective
    const cdead = new Float32Array(nI*nN);
    for(let i=0;i<nI;i++) cdead[i*nN + i] = 1;      // 8 neurons alive of 64
    const selDead = selectivityProfile(cdead, nI, nN);
    // one trial per item cannot be split in half, so it must decline
    const thinT = new Float32Array(nI*nN), thinL = new Int32Array(nI);
    for(let i=0;i<nI;i++){
      thinT.set(good.trials.subarray(i*reps*nN, (i*reps + 1)*nN), i*nN);
      thinL[i] = i;
    }
    const dThin = decodeAccuracy(thinT, thinL, nI, nN);
    return [
      { label:'decoding recovers item identity from structured activity',
        measured:`${(dGood.acc*100).toFixed(0)}% correct, chance ${(dGood.chance*100).toFixed(0)}%`,
        expected:'near 100%', pass: dGood.acc > 0.9 },
      { label:'decoding stays at chance when activity carries no item',
        measured:`${(dFlat.acc*100).toFixed(0)}% correct, chance ${(dFlat.chance*100).toFixed(0)}%`,
        expected:'within 15 points of chance', pass: Math.abs(dFlat.acc - dFlat.chance) < 0.15 },
      { label:'decoding declines rather than fits when trials are too few',
        measured:`${dThin.n} scored trials from one repeat per item`,
        expected:'0 scored', pass: dThin.n === 0 && dThin.acc === 0 },
      { label:'completion ranks the matching item first', measured:
          `rank1 ${comp.rank1.toFixed(2)}, margin ${comp.margin.toFixed(2)}, chance ${comp.chance.toFixed(2)}`,
        expected:'rank1 = 1, margin > 0', pass: comp.rank1 > 0.99 && comp.margin > 0 },
      { label:'completion falls to chance when the pairing is broken', measured:
          `rank1 ${compShuf.rank1.toFixed(2)} after shuffling partial patterns`,
        expected:'at or below chance', pass: compShuf.rank1 <= compShuf.chance + 0.01 },
      { label:'selectivity separates tuned from untuned populations', measured:
          `tuned ${selGood.mean.toFixed(2)}, uniform ${selFlat.mean.toFixed(2)}`,
        expected:'tuned 0.80 exactly (one item at 6 against seven at 0.5), uniform 0', pass:
          selGood.mean > 0.78 && selGood.mean < 0.82 && selFlat.mean < 0.05 },
      { label:'silent neurons are excluded rather than counted as selective',
        measured:`${selDead.active} of ${nN} neurons active, mean over active ${selDead.mean.toFixed(2)}`,
        expected:'active count reports the silence', pass:
          selDead.active === nI && selDead.mean > 0.9 },
    ];
  } },

{ name:'plasticity builds input-specific structure', source:'Song, Miller and Abbott 2000 (Nat Neurosci) and Gutig, Aharonov, Rotter and Sompolinsky 2003 (J Neurosci): additive STDP with hard bounds is competitive and splits weights bimodally, weight-dependent depression has a stable interior fixed point and does not. Positive control: two disjoint input groups presented alternately onto a shared target population',
  async run(){
    // Deliberately tiny and hand built, so the only thing under test is the learning rule.
    // 60 + 60 input neurons, 40 excitatory targets, 10 inhibitory.
    // Every input contacts every target at the same starting weight, so any difference between the two groups at the end was made by the rule rather than by the wiring.
    const NA = 60, NB = 60, NE = 40, NI = 10, count = NA+NB+NE+NI;
    const E0 = NA+NB, I0 = E0+NE;
    const build = (w0) => {
      const pre = [], post = [], w = [], del = [];
      const add = (a, b, wv, d) => { pre.push(a); post.push(b); w.push(wv); del.push(d); };
      // Sparse and jittered, which is the whole point.
      // All-to-all wiring at one weight makes the two input groups exact clones of each other, their group means cannot diverge whatever the rule does, and the test reports zero for a reason that has nothing to do with plasticity.
      // Competition needs something to select between.
      let z = 12345;
      const rnd = () => (z = (z*1664525 + 1013904223) >>> 0) / 4294967296;
      for(let i=0;i<E0;i++) for(let e=0;e<NE;e++){
        if(rnd() > 0.5) continue;
        add(i, E0+e, w0*EQ*(0.7 + 0.6*rnd()), 1 + Math.floor(rnd()*3));
      }
      for(let e=0;e<NE;e++) for(let k=0;k<NI;k++) add(E0+e, I0+k, 4*EQ, 1);
      for(let k=0;k<NI;k++) for(let e=0;e<NE;e++) add(I0+k, E0+e, -6*IQ, 1);
      const m = pre.length;
      const preStart = new Int32Array(count+1);
      for(let sIdx=0;sIdx<m;sIdx++) preStart[pre[sIdx]+1]++;
      for(let i=0;i<count;i++) preStart[i+1] += preStart[i];
      const ntype = new Uint8Array(count);
      for(let k=0;k<NI;k++) ntype[I0+k] = 3;
      const chStart = new Uint32Array([0, NA, NA+NB]);
      const chIdx = new Uint32Array(NA+NB), chW = new Float32Array(NA+NB).fill(1);
      for(let i=0;i<NA+NB;i++) chIdx[i] = i;
      return { kind:'net', count, ntype, bias:new Float32Array(count),
        syn:1, psc:1, tauE:3, tauI:8, protocols:[],
        src:new Int32Array(count).fill(1), lidx:Uint32Array.from({length:count}, (_, i) => i),
        preStart, post:Int32Array.from(post), w:Float32Array.from(w),
        delay:Uint8Array.from(del), synCount:m,
        inputs:[{ chStart, chIdx, chW, amp:1 }] };
    };
    // Presentation clock: 250 ms on, 250 ms off, alternating which group.
    const ON = 250, PER = 500, MS = 200000;
    const groupAt = t => Math.floor(t/PER) % 2;
    const onAt = t => (t % PER) < ON;
    const regime = async (name, P, mult = 1) => {
      const net = Object.assign(build(P.w0), P.rule);
      // A positive control asks whether the rule can build structure, not how long it takes.
      // Scaling both learning rates together compresses the timescale without changing the fixed point, which is the ratio.
      net.aP *= mult; net.aM *= mult;
      const w0 = net.w.slice();
      // spikes per target, split by which group was on show, in an early and a late window: selectivity that grows within one run is the claim, and it needs no second run to compare against
      const early = [new Float32Array(NE), new Float32Array(NE)];
      const late = [new Float32Array(NE), new Float32Array(NE)];
      let inSpikes = 0;
      const r = await simRun(net, MS, { steps:25, weights:true,
        driver:(wk, t) => {
          if(t % ON) return;
          const g = onAt(t) ? (groupAt(t) ? [0, 12] : [12, 0]) : [0, 0];
          wk.postMessage({ cmd:'inputFrame', mi:0, gains:Float32Array.from(g) });
        },
        onFrame:(fired, t) => {
          if(!onAt(t)) return;
          const g = groupAt(t);
          const bin = t < MS*0.2 ? early : (t > MS*0.8 ? late : null);
          for(let i=0;i<E0;i++) if(fired[i]) inSpikes++;
          if(!bin) return;
          for(let e=0;e<NE;e++) if(fired[E0+e]) bin[g][e]++;
        } });
      // per target, mean weight from each input group
      const groupW = (wArr) => {
        const a = new Float64Array(NE), b = new Float64Array(NE);
        for(let i=0;i<E0;i++){
          const acc = i < NA ? a : b;
          for(let sIdx=net.preStart[i]; sIdx<net.preStart[i+1]; sIdx++){
            const e = net.post[sIdx] - E0;
            if(e >= 0 && e < NE) acc[e] += wArr[sIdx]/(i < NA ? NA : NB);
          }
        }
        return { a, b };
      };
      const diff = (wArr) => {
        const { a, b } = groupW(wArr);
        let sum = 0, n = 0;
        for(let e=0;e<NE;e++){
          const tot = a[e] + b[e];
          if(tot <= 1e-9) continue;
          sum += Math.abs(a[e] - b[e])/tot; n++;
        }
        return n ? sum/n : 0;
      };
      const selOf = bin => {
        let sum = 0, n = 0;
        for(let e=0;e<NE;e++){
          const tot = bin[0][e] + bin[1][e];
          if(tot < 4) continue;
          sum += Math.abs(bin[0][e] - bin[1][e])/tot; n++;
        }
        return { sel: n ? sum/n : 0, n };
      };
      // bimodality of the plastic excitatory weights: the competitive regime is expected to push them to the bounds, the other to a common value
      const wm = net.wmax;
      let atBound = 0, plastic = 0;
      for(let i=0;i<E0;i++)
        for(let sIdx=net.preStart[i]; sIdx<net.preStart[i+1]; sIdx++){
          plastic++;
          const v = r.w[sIdx];
          if(v <= wm*0.05 || v >= wm*0.95) atBound++;
        }
      let moved = 0;
      for(let sIdx=0;sIdx<net.synCount;sIdx++)
        if(Math.abs(r.w[sIdx]-w0[sIdx]) > Math.abs(w0[sIdx])*0.01) moved++;
      const e = selOf(early), l = selOf(late);
      let wSum = 0, w0Sum = 0, nP = 0;
      for(let i=0;i<E0;i++)
        for(let sIdx=net.preStart[i]; sIdx<net.preStart[i+1]; sIdx++){
          wSum += r.w[sIdx]; w0Sum += w0[sIdx]; nP++;
        }
      const tSpk = l.n ? 'scored' : 'none';
      return { name, inSpikes, moved:moved/net.synCount,
        wMean:wSum/Math.max(1,nP), wMean0:w0Sum/Math.max(1,nP), tSpk,
        dStart:diff(w0), dEnd:diff(r.w),
        selEarly:e.sel, selLate:l.sel, scored:l.n,
        bound:plastic ? atBound/plastic : 0 };
    };
    // the soft-bound configuration; quantities in weight units carry the conversion (aP, wmax, and aM where depression is additive), ratios and rates do not
    const SOFT = { w0:3,
      rule:{ plast:1, aP:0.00015*EQ, aM:0.001, tauS:20, tauM:20, wmax:10*EQ, wdep:1,
        iEta:0.003*IQ, iRho:3, rhoMode:1, calS:10, scale:0, sEta:0.001 } };
    // the hard-bound configuration
    const HARD = { w0:9,
      rule:{ plast:1, aP:0.0003*EQ, aM:0.00036*EQ, tauS:20, tauM:20, wmax:30*EQ, wdep:0,
        iEta:0.001*IQ, iRho:5, rhoMode:1, calS:20, scale:1, sEta:0.0003 } };
    // Weight-dependent depression at the additive rule's ratio and bounds, so the two differ only in whether depression scales with the weight.
    // Without this the comparison confounds the rule with the A+/A- ratio, which sets where the mean weight settles. aM is dimensionless under weight-dependent depression (the update is aM w K), so it carries no unit factor; the additive rule's aM is a weight and does.
    const SOFTR = { w0:9,
      rule:{ plast:1, aP:0.0003*EQ, aM:0.00036, tauS:20, tauM:20, wmax:30*EQ, wdep:1,
        iEta:0.001*IQ, iRho:5, rhoMode:1, calS:20, scale:1, sEta:0.0003 } };
    const M = 30;
    const soft = await regime('weight-dependent', SOFT);
    const hard = await regime('additive hard-bound', HARD);
    const softF = await regime('weight-dependent fast', SOFT, M);
    const hardF = await regime('additive hard-bound fast', HARD, M);
    const softRF = await regime('weight-dependent, additive ratio', SOFTR, M);
    const fmt = r => `mean w ${r.wMean0.toFixed(2)} to ${r.wMean.toFixed(2)}, ` +
      `${(r.moved*100).toFixed(0)}% moved, d ${r.dStart.toFixed(3)} to ${r.dEnd.toFixed(3)}, ` +
      `sel ${r.selEarly.toFixed(3)} to ${r.selLate.toFixed(3)}, ` +
      `${(r.bound*100).toFixed(0)}% at a bound`;
    return [
      { label:'inputs drive the targets at all', measured:
          `${soft.inSpikes} input spikes, ${soft.scored} of ${NE} targets scored`,
        expected:'both non-zero, else the test measures nothing', pass:
          soft.inSpikes > 1000 && soft.scored > 0 },
      { label:`additive hard-bound rule differentiates the two inputs at ${M}x rate`,
        measured: fmt(hardF), expected:'weight difference grows', pass:
          hardF.dEnd > hardF.dStart + 0.02 },
      { label:`weight-dependent rule at ${M}x rate, same protocol`, measured: fmt(softF),
        expected:'reported for comparison, not gated: the interior fixed point predicts little differentiation',
        advisory:true, pass: softF.dEnd > softF.dStart + 0.02 },
      { label:'competitive regime drives weights to the bounds', measured:
          `additive ${(hardF.bound*100).toFixed(0)}% vs weight-dependent ${(softF.bound*100).toFixed(0)}%`,
        expected:'additive higher (Song 2000 bimodality)', pass:
          hardF.bound > softF.bound },
      { label:'the difference is the weight dependence, not the A+/A- ratio',
        measured:`weight-dependent at the additive ratio and bounds: ${fmt(softRF)}`,
        expected:'still fails to differentiate, isolating the rule from the ratio', pass:
          softRF.dEnd < softRF.dStart + 0.02 && hardF.dEnd > softRF.dEnd },
      { label:'at the first alphabet school\'s rates, over 200 simulated seconds',
        measured:`additive ${fmt(hard)}; weight-dependent ${fmt(soft)}`,
        expected:'recorded, not gated: says how far 200 s gets at those rates',
        advisory:true, pass:true },
    ];
  } },

{ name:'topographic projection', source:'retinotopic order is preserved through tracts (visual pathway organization); delays follow tract length over conduction velocity',
  async run(){
    const mkSheet = (cx, tag, seedv, id) => {
      const geo = NODE_DEFS.box.compute([], { center:[cx,0,0], size:[600,20,600], rotate:[0,0,0] });
      return NODE_DEFS.scatter.compute([geo], { count:1500, type:0, seed:seedv,
        spacing:0, pattern:0, pitch:40, jitter:8, axis:1, tag }, { id });
    };
    const merged = NODE_DEFS.gather.compute([mkSheet(-2000,'a',3,1), mkSheet(0,'b',4,2)]);
    const spec = { from:'a', to:'b', axisFrom:1, axisTo:1, flipU:false, flipV:false,
      sigma:40, prob:0.6, weight:5, velocity:4000, tract:8000, seed:1 };
    const P = projs => wireConnect({ ...merged, projs }, { ...CN, prob:0 }, null);
    const net = P([spec]), net2 = P([spec]);
    let same = net.synCount === net2.synCount;
    for(let s=0;s<net.synCount && same;s+=53)
      same = net.post[s] === net2.post[s] && net.w[s] === net2.w[s];
    const topoErr = (nt, mapx) => {
      let e = 0, c = 0;
      for(let i=0;i<nt.count;i++){
        const s0 = nt.preStart[i], s1 = nt.preStart[i+1];
        if(s1 === s0) continue;
        let tx = 0;
        for(let s=s0;s<s1;s++) tx += nt.pos[nt.post[s]*3];
        e += Math.abs(tx/(s1-s0) - mapx(nt.pos[i*3])); c++;
      }
      return e/Math.max(1,c);
    };
    const errStraight = topoErr(net, x => x + 2000);
    const errMirror = topoErr(P([{ ...spec, flipU:true }]), x => -(x + 2000));
    let delayOk = true, delaySample = 0;
    for(let s=0;s<net.synCount;s++){
      if(net.delay[s] !== 2) delayOk = false;
      delaySample = net.delay[s];
    }
    // additivity: the local kernel's synapses are untouched by adding a tract
    const CL = { ...CN, radius:150, prob:0.3 };
    const loc = wireConnect(merged, CL, null);
    const both = wireConnect({ ...merged, projs:[spec] }, CL, null);
    const key = (nt, i, s) => `${nt.src[i]}:${nt.lidx[i]}:${nt.src[nt.post[s]]}:${nt.lidx[nt.post[s]]}`;
    const inBoth = new Map();
    for(let i=0;i<both.count;i++)
      for(let s=both.preStart[i];s<both.preStart[i+1];s++)
        inBoth.set(key(both, i, s), both.w[s]);
    let localKept = 0, localLost = 0;
    for(let i=0;i<loc.count;i++)
      for(let s=loc.preStart[i];s<loc.preStart[i+1];s++){
        if(inBoth.get(key(loc, i, s)) === loc.w[s]) localKept++;
        else localLost++;
      }
    return [
      { label:'deterministic', measured: same ? 'identical rewiring' : 'DIFFERS',
        expected:'identical', pass: same && net.synCount > 10000 },
      { label:'retinotopic order preserved', measured:
          `mean mapped error ${errStraight.toFixed(1)} µm at sigma 40`,
        expected:'< 40 µm', pass: errStraight < 40 },
      { label:'mirrored projection (chiasm)', measured:
          `mean error ${errMirror.toFixed(1)} µm against the reflected map`,
        expected:'< 40 µm', pass: errMirror < 40 },
      { label:'delay = tract / velocity', measured:
          `${delaySample} ms at 8000 µm / 4000 µm per ms`,
        expected:'2 ms on every tract synapse', pass: delayOk },
      { label:'additive with the local kernel', measured:
          `${localKept.toLocaleString()} local pairs kept, ${localLost} changed`,
        expected:'all kept byte-identical', pass: localLost === 0 && localKept > 1000 },
    ];
  } },

{ name:'brain file round trip', source:'design rule: pair-keyed weights survive additive edits; new tissue at baseline, dead pairs drop, no weight resurrection',
  async run(){
    const pts = mkPts(3000, 320, 5);
    const a = wireConnect(pts, { ...CN }, null);
    const learned = a.w.slice();
    for(let s=0;s<a.synCount;s++) learned[s] *= 1.5;   // stand-in for training
    const file = parseBrainFile(buildBrainFile({ nodes:[] }, a, learned, 777, null));
    const exact = applyBrainWeights(a, file);
    let same = exact.matched === a.synCount;
    for(let s=0;s<a.synCount && same;s++) same = exact.net.w[s] === learned[s];
    // Plastic-only files, the format overnight training writes.
    // Its premise is that a frozen weight is reproduced by the wiring rather than recorded, so the round trip has to restore the gated weights exactly and leave the frozen ones on the wired value, not on what was learned.
    const gated = { ...a, pmask:new Uint8Array(a.synCount) };
    for(let s=0;s<a.synCount;s++) gated.pmask[s] = (s % 7 === 0) ? 1 : 0;
    const pk = plasticCount(gated);
    const pBuf = buildPlasticFile({ nodes:[] }, gated, learned, 777, null);
    const pBytes = pBuf.byteLength, wBytes = a.synCount*4;
    const pFile = parseBrainFile(pBuf);
    const pRes = applyBrainWeights(gated, pFile);
    let pOk = pFile.k === pk && pRes.matched === pk;
    for(let s=0;s<a.synCount && pOk; s++)
      pOk = gated.pmask[s] ? pRes.net.w[s] === learned[s] : pRes.net.w[s] === a.w[s];
    // superset rewiring (higher probability): saved pairs restore, new at baseline
    const b = wireConnect(pts, { ...CN, prob:0.55 }, null);
    const sup = applyBrainWeights(b, file);
    // map pair -> saved weight to classify b's synapses
    const oldPairs = new Set();
    for(let i=0;i<a.count;i++)
      for(let s=a.preStart[i];s<a.preStart[i+1];s++)
        oldPairs.add(a.src[i]+':'+a.lidx[i]+':'+a.src[a.post[s]]+':'+a.lidx[a.post[s]]);
    let restoredOk = true, baselineOk = true;
    for(let i=0;i<b.count;i++)
      for(let s=b.preStart[i];s<b.preStart[i+1];s++){
        const isOld = oldPairs.has(b.src[i]+':'+b.lidx[i]+':'+b.src[b.post[s]]+':'+b.lidx[b.post[s]]);
        if(isOld){ if(Math.abs(sup.net.w[s]) <= Math.abs(b.w[s])) restoredOk = false; }
        else if(sup.net.w[s] !== b.w[s]) baselineOk = false;
      }
    // subset rewiring: saved pairs that no longer exist are dropped
    const sub = applyBrainWeights(a, parseBrainFile(buildBrainFile({ nodes:[] }, b, b.w, 0, null)));
    return [
      { label:'plastic-only file restores the gate and rewires the rest', measured:
          `${pFile.k} of ${a.synCount} weights stored, ${pRes.matched} restored, ` +
          `frozen left on the wired value: ${pOk}`,
        expected:'gated weights exact, frozen untouched', pass: pOk && pk > 0 },
      { label:'and is smaller by the gated fraction', measured:
          `${(pBytes/1024).toFixed(0)} kB against ${(wBytes/1024).toFixed(0)} kB of weights`,
        expected:'about a seventh here, the gated share', pass: pBytes < wBytes*0.3 },
      { label:'identical tissue restores exactly', measured:
          `${exact.matched.toLocaleString()} of ${a.synCount.toLocaleString()} matched, weights ${same ? 'byte-identical' : 'DIFFER'}`,
        expected:'all matched, byte-identical', pass: exact.matched === a.synCount && same },
      { label:'surviving pairs restore after additive edit', measured:
          `${sup.matched.toLocaleString()} matched into ${b.synCount.toLocaleString()} wired`,
        expected:'every saved pair matched, restored weights applied', pass:
          sup.matched === a.synCount && restoredOk },
      { label:'new tissue stays at baseline', measured: baselineOk ? 'untouched' : 'MODIFIED',
        expected:'wired weights untouched', pass: baselineOk },
      { label:'dead pairs drop, no resurrection', measured:
          `${sub.matched.toLocaleString()} of ${sub.saved.toLocaleString()} saved pairs survive`,
        expected:'matched < saved, matched = current tissue', pass:
          sub.matched < sub.saved && sub.matched === a.synCount },
    ];
  } },

{ name:'plasticity gating and synaptic scaling', source:'per-pathway gating (frozen circuits stay frozen); homeostatic scaling after Turrigiano 2008 / van Rossum 2000 (multiplicative, toward a target rate)',
  async run(){
    const mkTag = (cx, tag, seedv, id) => {
      const geo = NODE_DEFS.box.compute([], { center:[cx,0,0], size:[400,400,400], rotate:[0,0,0] });
      return NODE_DEFS.scatter.compute([geo], { count:800, type:0, seed:seedv,
        spacing:0, pattern:0, pitch:40, jitter:8, axis:1, tag }, { id });
    };
    const merged = NODE_DEFS.gather.compute([mkTag(-150,'A',5,1), mkTag(150,'B',6,2)]);
    const net = wireConnect(merged, { ...CN, radius:500, sigma:300, prob:0.15,
      table:'A B 1 1 0' }, null);   // A->B frozen, everything else plastic
    const isAB = new Uint8Array(net.synCount);
    for(let i=0;i<net.count;i++)
      for(let s=net.preStart[i];s<net.preStart[i+1];s++)
        isAB[s] = (merged.tags[net.src[i]] === 'A' && merged.tags[net.src[net.post[s]]] === 'B') ? 1 : 0;
    const w0 = net.w.slice();
    const allIdx = Uint32Array.from({ length:net.count }, (_, i) => i);
    const drive = amp => [{ idx:allIdx, amp, mode:3, period:100, width:10, t0:0, duration:0 }];
    // strong STDP; scaling off: frozen class must stay byte-identical
    const r1 = await simRun({ ...net, syn:0, plast:1, aP:0.01, aM:0.012, tauS:20,
      wmax:30, iEta:0.004, iRho:5, scale:0, protocols:drive(9) }, 4000,
      { steps:2, weights:true });
    let frozenSame = true, plasticMoved = 0;
    for(let s=0;s<net.synCount;s++){
      if(isAB[s]){ if(r1.w[s] !== w0[s]) frozenSame = false; }
      else if(r1.w[s] !== w0[s]) plasticMoved++;
    }
    // pure scaling (STDP amplitudes zero), quiet net below target: plastic excitatory inputs grow, frozen class still untouched
    const run2 = gpu => simRun({ ...net, syn:0, plast:1, aP:0, aM:0, tauS:20,
      wmax:30, iEta:0, iRho:5, scale:1, sEta:0.002, protocols:drive(4) }, 10000,
      { steps:2, weights:true, gpu });
    const meanExc = (w, mask) => {
      let s = 0, c = 0;
      for(let k=0;k<net.synCount;k++)
        if(w[k] > 0 && isAB[k] === mask){ s += w[k]; c++; }
      return s/Math.max(1,c);
    };
    const r2 = await run2(false);
    const grew = meanExc(r2.w, 0)/meanExc(w0, 0);
    let frozen2 = true;
    for(let s=0;s<net.synCount && frozen2;s++)
      if(isAB[s] && r2.w[s] !== w0[s]) frozen2 = false;
    const checks = [
      { label:'frozen pathway is byte-identical under STDP', measured:
          frozenSame ? 'unchanged' : 'CHANGED',
        expected:'unchanged', pass: frozenSame && plasticMoved > 1000 },
      { label:'plastic synapses still learn', measured:
          plasticMoved.toLocaleString() + ' moved', expected:'> 1,000', pass: plasticMoved > 1000 },
      { label:'scaling grows quiet excitatory inputs', measured:
          'x' + grew.toFixed(4) + ' over 10 s below target',
        expected:'> x1.005', pass: grew > 1.005 },
      { label:'scaling respects the gate', measured: frozen2 ? 'frozen class untouched' : 'LEAKED',
        expected:'untouched', pass: frozen2 },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const rg = await run2(true);
      if(rg.error) checks.push({ label:'gpu scaling parity', measured:rg.error,
        expected:'no engine error', pass:false });
      else {
        const grewG = meanExc(rg.w, 0)/meanExc(w0, 0);
        checks.push({ label:'gpu scaling parity', measured:
            `gpu x${grewG.toFixed(4)} vs cpu x${grew.toFixed(4)}`,
          expected:'both grow, within 1% of each other', pass:
            grewG > 1.005 && Math.abs(grewG-grew) < 0.01 });
      }
    }
    // measured set points: two drive groups at different natural rates keep their identities; a uniform target would pump the sparse one up.
    // Uses the stable E/I net from the Vogels group, split by drive amplitude.
    const vnet = wireConnect(mkPts(1500, 350, 7), { ...CN, radius:250, prob:0.4,
      wExc:4, wInh:-6 }, null);
    const aIdx = [], bIdx = [];
    for(let i=0;i<vnet.count;i++) (i < 750 ? aIdx : bIdx).push(i);
    const cal = { ...vnet, syn:0, plast:1, aP:0, aM:0, tauS:20, wmax:30,
      iEta:0.002, iRho:5, rhoMode:1, calS:5, scale:1, sEta:0.004,
      protocols:[
        { idx:Uint32Array.from(aIdx), amp:11, mode:3, period:100, width:10, t0:0, duration:0 },
        { idx:Uint32Array.from(bIdx), amp:6, mode:3, period:100, width:10, t0:0, duration:0 }] };
    let aCal = 0, bCal = 0, aEnd = 0, bEnd = 0;
    await simRun(cal, 25000, { steps:2, onFrame:(fired, t) => {
      let ca = 0, cb = 0;
      for(const i of aIdx) if(fired[i]) ca++;
      for(const i of bIdx) if(fired[i]) cb++;
      if(t < 5000){ aCal += ca; bCal += cb; }
      else if(t >= 20000){ aEnd += ca; bEnd += cb; }
    } });
    const hzA0 = aCal/aIdx.length/5, hzB0 = bCal/bIdx.length/5;
    const hzA1 = aEnd/aIdx.length/5, hzB1 = bEnd/bIdx.length/5;
    checks.push({ label:'measured set points preserve heterogeneity', measured:
        `A ${hzA0.toFixed(1)} to ${hzA1.toFixed(1)} Hz, B ${hzB0.toFixed(1)} to ${hzB1.toFixed(1)} Hz`,
      expected:'distinct baselines, each preserved within 60%', pass:
        hzA0 > hzB0 + 1 &&
        Math.abs(hzA1-hzA0) < Math.max(0.5, hzA0*0.6) &&
        Math.abs(hzB1-hzB0) < Math.max(0.5, hzB0*0.6) });
    return checks;
  } },

{ name:'the reference engine is reproducible', source:'project rule: the same graph and seeds must always compute the same network, which is worth nothing if simulating it twice gives two answers. An unseeded initial membrane jitter hides well: the weight trajectory washes the initial condition out, so learned weights come back bit-identical run to run and every weight-based check passes. What it moves is anything read from the early transient, such as measured per-neuron set points',
  async run(){
    const net = wireConnect(mkPts(1200, 260, 5), { ...CN, radius:220, prob:0.35 }, null);
    Object.assign(net, { syn:0, plast:0,
      protocols:[{ idx:Uint32Array.from({length:1200},(_,i)=>i), amp:9, mode:0,
        period:100, width:10, t0:0, duration:0 }] });
    const run = () => simRun(net, 1500, { steps:5, perNeuron:true });
    const a = await run(), b = await run();
    let same = 0, worst = 0, total = 0;
    const k = Math.min(a.counts.length, b.counts.length);
    for(let i=0;i<k;i++){
      const d = Math.abs(a.counts[i] - b.counts[i]);
      if(d === 0) same++;
      if(d > worst) worst = d;
      total += a.counts[i];
    }
    const spikesA = a.frames.reduce((x,y)=>x+y, 0), spikesB = b.frames.reduce((x,y)=>x+y, 0);
    // and again with noise driving it.
    // A noisy network is the harder case and the one every scenario runs, so a check that only covers constant drive would miss half of it.
    const nn = wireConnect(mkPts(1200, 260, 5), { ...CN, radius:220, prob:0.35 }, null);
    Object.assign(nn, { syn:0, plast:0,
      protocols:[{ idx:Uint32Array.from({length:1200},(_,i)=>i), amp:11, mode:3,
        period:100, width:10, t0:0, duration:0 }] });
    const nA = await simRun(nn, 1500, { steps:5, perNeuron:true });
    const nB = await simRun(nn, 1500, { steps:5, perNeuron:true });
    let nSame = 0;
    const nk = Math.min(nA.counts.length, nB.counts.length);
    for(let i=0;i<nk;i++) if(nA.counts[i] === nB.counts[i]) nSame++;
    const nSpikes = nA.frames.reduce((x,y)=>x+y, 0);
    return [
      { label:'two identical runs spike identically', measured:
          `${same}/${k} neurons match exactly, worst neuron differs by ${worst}`,
        expected:'every neuron identical', pass: k > 0 && same === k },
      { label:'and agree on the total', measured:
          `${spikesA} vs ${spikesB} spikes`,
        expected:'the same number', pass: spikesA === spikesB && spikesA > 0 },
      { label:'a noise-driven run repeats too', measured:
          `${nSame}/${nk} neurons match exactly across two runs, ${nSpikes} spikes`,
        expected:'every neuron identical', pass: nk > 0 && nSame === nk && nSpikes > 0 },
      { label:'the run was active enough to be a real test', measured:
          `${total} spikes across ${k} neurons`,
        expected:'a network that fired', pass: total > k },
    ];
  } },

{ name:'operations formulas are safe and deterministic', source:'internal invariant. The operations node evaluates a formula per point, and graphs are meant to be shared, so the formula must be data rather than a program: it cannot be JavaScript and cannot reach the page. Determinism is the other half, since the project rule is that the same graph and seeds always compute the same network, which rules out a clock or unseeded randomness. Identity is the third: pair-hash connectivity is keyed on (src, lidx), so a formula that could rewrite either would silently rewire every synapse including learned ones',
  async run(){
    const READ = new Set(['x','y','z','bias','ntype','i','n','lidx','src','a','b','c','d']);
    const WRITE = new Set(['x','y','z','bias','ntype']);
    const fns = { rand:() => 0.5, noise:() => 0.5 };
    const ev = (src, env) => {
      const e = buildExpr(src, READ, WRITE, fns);
      Object.assign(e.env, env); e.run(); return e.env;
    };
    // the escapes a shared scene file would try
    const unsafe = ['y = window', 'y = fetch(1)', 'y = eval(1)', 'y = this',
      'y = constructor', 'y = globalThis', 'y = Math'];
    let blocked = 0;
    for(const u of unsafe){ try { buildExpr(u, READ, WRITE, fns); } catch(e){ blocked++; } }
    // identity is readable but never writable
    let idBlocked = 0;
    for(const u of ['lidx = 1', 'src = 2', 'i = 3', 'n = 4']){
      try { buildExpr(u, READ, WRITE, fns); } catch(e){ idBlocked++; }
    }
    const readsId = (() => { try { ev('x = lidx + src', { lidx:2, src:3 }); return true; }
      catch(e){ return false; } })();
    // arithmetic the person writing a formula will assume
    const prec = ev('x = 2 + 3 * 4 ** 2', {}).x;
    const order = ev('y = y + 10; x = y * 2', { x:0, y:5 });
    const tern = ev('ntype = z > 0 ? 1 : 0', { z:7 }).ntype;
    const clampF = ev('bias = clamp(abs(a)*2, 0, 5)', { a:-4 }).bias;
    // same formula, same input, same answer, twice
    const r1 = ev('x = sin(x)*b + noise(x,y,z)', { x:1.25, y:2, z:3, b:7 }).x;
    const r2 = ev('x = sin(x)*b + noise(x,y,z)', { x:1.25, y:2, z:3, b:7 }).x;
    return [
      { label:'a formula cannot reach the page', measured:
          `${blocked}/${unsafe.length} escapes refused`,
        expected:'every one refused at parse time', pass: blocked === unsafe.length },
      { label:'stable identity is read-only', measured:
          `${idBlocked}/4 assignments refused, identity readable: ${readsId}`,
        expected:'lidx, src, i and n refuse assignment but can be read', pass:
          idBlocked === 4 && readsId },
      { label:'operator precedence follows arithmetic', measured:
          `2 + 3 * 4 ** 2 = ${prec}`, expected:'50', pass: prec === 50 },
      { label:'assignments run in order', measured:
          `y then x: y=${order.y}, x=${order.x}`,
        expected:'the second line sees the first, y=15 x=30', pass:
          order.y === 15 && order.x === 30 },
      { label:'conditionals and the function table work', measured:
          `z>0?1:0 = ${tern}, clamp(abs(-4)*2,0,5) = ${clampF}`,
        expected:'1 and 5', pass: tern === 1 && clampF === 5 },
      { label:'the same formula gives the same answer', measured:
          `${r1} then ${r2}`, expected:'identical', pass: r1 === r2 && isFinite(r1) },
    ];
  } },

{ name:'documentation integrity', source:'internal invariant: docs.html generates its parameter tables from NODE_DEFS, so a docs module that does not parse takes the whole page down silently',
  async run(){
    // A prose string with an unescaped apostrophe kills the docs page, and nothing imports docs.js except the page itself, so nothing else fails first.
    let mod = null, err = null;
    try { mod = await import('./docs.js'); } catch(e){ err = e.message; }
    const defs = Object.keys(NODE_DEFS);
    const documented = mod ? Object.keys(mod.NODES) : [];
    // a node a project module registered carries its own prose (registerNode checks it)
    const docOf = t => mod ? nodeDoc(t, mod.NODES) : null;
    const missing = defs.filter(t => !docOf(t));
    // docsview renders NODES[type].blurb; an entry under any other key renders as an empty description
    const missingBlurb = mod ? documented.filter(t => !mod.NODES[t].blurb) : [];
    // every parameter the app exposes should have prose behind it
    const undocParams = [];
    if(mod) for(const [type, def] of Object.entries(NODE_DEFS)){
      const doc = docOf(type);
      if(!doc) continue;
      for(const prm of (def.params || []))
        if(!(doc.params && doc.params[prm.k])) undocParams.push(`${type}.${prm.k}`);
    }
    // Documented is not the same as visible.
    // The reference page and its nav both iterate a fixed category list, so a node whose cat is missing from it renders nowhere while still passing every check above.
    const PAGE_CATS = ['regions', 'cells', 'arrange', 'celltypes', 'wiring', 'drive', 'readout', 'run', 'routing', 'notes'];
    const unrendered = Object.entries(NODE_DEFS)
      .filter(([, def]) => !PAGE_CATS.includes(def.cat || 'wiring'))
      .map(([t, def]) => `${t} (cat ${def.cat})`);
    // Scenarios are documented by hand rather than generated from the list, so nothing catches a new one until someone opens the page.
    const scenDoc = mod ? ((mod.PAGES.find(pg => pg.id === 'scenarios') || {}).html || '') : '';
    const missingScen = SCENARIOS.map(sc => sc.name)
      .filter(n => !scenDoc.toLowerCase().includes(n.toLowerCase()));
    // build each scenario into a throwaway editor and count its note nodes
    const noteCounts = SCENARIOS.map(sc => {
      const ns = [];
      let nid = 1;
      const fake = { nodes:ns, nextId:1, suggestResolution:0,
        addNode(t, x, y){
          const def = NODE_DEFS[t], params = {};
          def.params.forEach(pd => params[pd.k] = structuredClone(pd.def));
          const n = { id:nid++, type:t, x, y, params, inputs:new Array(def.inputs).fill(null) };
          ns.push(n); return n;
        } };
      try { sc.build(fake); } catch(e){ return { name:sc.name, notes:-1 }; }
      return { name:sc.name, notes: ns.filter(n => n.type === 'note').length };
    });
    // The param control types props.js renders; anything else silently produces no control in the properties panel. tag is the population picker, sweeps the sweep node's variant tabs.
    const KNOWN = ['float', 'int', 'select', 'str', 'text', 'pairs', 'vec3', 'action', 'tag', 'sweeps'];
    const badType = [];
    const staleParams = [];
    if(mod) for(const [type, def] of Object.entries(NODE_DEFS)){
      const doc = docOf(type);
      if(!doc || !doc.params) continue;
      const keys = new Set((def.params || []).map(p => p.k));
      for(const k of Object.keys(doc.params))
        if(!keys.has(k)) staleParams.push(`${type}.${k}`);
    }
    for(const [type, def] of Object.entries(NODE_DEFS))
      for(const prm of (def.params || [])){
        if(!KNOWN.includes(prm.t)) badType.push(`${type}.${prm.k} t=${prm.t}`);
        else if(prm.t === 'select' && !Array.isArray(prm.options))
          badType.push(`${type}.${prm.k} select without options`);
      }
    return [
      { label:'docs module parses', measured: err || 'ok',
        expected:'imports without a syntax error', pass: !err },
      { label:'every node type is documented', measured:
          missing.length ? missing.join(', ') : 'all ' + defs.length + ' types',
        expected:'no undocumented node types', pass: !!mod && missing.length === 0 },
      { label:'every node has a blurb', measured:
          missingBlurb.length ? missingBlurb.join(', ') : 'all ' + documented.length,
        expected:'no entry without a blurb key', pass: !!mod && missingBlurb.length === 0 },
      { label:'every parameter has prose', measured:
          undocParams.length ? undocParams.join(', ') : 'all documented',
        expected:'no undocumented parameters', pass: !!mod && undocParams.length === 0 },
      // The other direction: the check above looks for parameters without prose, this one for prose without a parameter.
      { label:'no prose for parameters that were removed', measured:
          staleParams.length ? staleParams.join(', ') : 'none',
        expected:'every documented parameter still exists on its node',
        pass: !!mod && staleParams.length === 0 },
      { label:'parameter control types are renderable', measured:
          badType.length ? badType.join(', ') : 'all known',
        expected:'every param uses a type the properties panel handles',
        pass: badType.length === 0 },
      { label:'every node renders on the reference page', measured:
          unrendered.length ? unrendered.join(', ') : 'all ' + defs.length + ' types',
        expected:'no node sits in a category the page does not list',
        pass: unrendered.length === 0 },
      { label:'every scenario is documented', measured:
          missingScen.length ? missingScen.join(', ') : 'all ' + SCENARIOS.length + ' scenarios',
        expected:'no scenario missing from the scenarios page',
        pass: !!mod && missingScen.length === 0 },
      // A loaded scene should explain itself in the graph, not only in the docs page that nobody has open while they are looking at it.
      { label:'every scenario carries notes in its graph', measured:
          noteCounts.map(n => `${n.name} ${n.notes}`).join(', '),
        expected:'at least two note nodes per scenario',
        pass: noteCounts.every(n => n.notes >= 2) },
    ];
  } },

{ name:'depression form and weight stability', source:'van Rossum, Bi and Turrigiano 2000 (J Neurosci); Gutig, Aharonov, Rotter and Sompolinsky 2003 (J Neurosci): additive STDP with hard bounds has no interior fixed point, weight-dependent depression does, at w = A+ / A-',
  async run(){
    // Same excitatory-inhibitory net and drive as the Vogels group, which is known to fire well above target, so the only difference between the two conditions is the depression rule.
    // Amplitudes are identical in both; only the form changes.
    const net = wireConnect(mkPts(700, 250, 7), { ...CN, radius:220, prob:0.3,
      wExc:4, wInh:-6 }, null);
    const w0 = net.w.slice();
    const drive = [{ idx:Uint32Array.from({length:net.count}, (_, i) => i), amp:11,
      mode:3, period:100, width:10, t0:0, duration:0 }];
    const meanExc = w => {
      let sum = 0, c = 0;
      for(let k=0;k<net.synCount;k++) if(w0[k] > 0){ sum += w[k]; c++; }
      return sum/Math.max(1, c);
    };
    const base = meanExc(w0);
    // A- above A+ is the configuration that keeps additive potentiation in check (Song et al. 2000).
    // The weight-dependent fixed point A+ / A- then sits below the wired mean, so both rules have somewhere to move and they should disagree about where they stop.
    // Amplitudes far above anything a scenario uses: the question is where each rule ends up, not how fast, and the battery cannot spend the minutes that scenario-scale amplitudes would need to get there.
    const aP = 0.08, aM = 0.16, fp = aP/aM;
    const cfg = (wdep, gpu) => simRun({ ...net, syn:0, plast:1, aP, aM, tauS:20,
      wmax:10, iEta:0, iRho:5, scale:0, wdep, protocols:drive }, 12000,
      { steps:2, weights:true, gpu });
    const add = await cfg(0, false);
    const dep = await cfg(1, false);
    const mAdd = meanExc(add.w), mDep = meanExc(dep.w);
    // Directional, not convergent.
    // Additive drift is linear in time so reaching a bound from this wired mean takes minutes of simulation, which the battery cannot spend; what distinguishes the two rules is that one is pulled toward an interior fixed point and the other is not, and that is visible long before either arrives.
    const checks = [
      { label:'both rules ran', measured:
          `additive ${base.toFixed(3)} to ${mAdd.toFixed(3)}, weight-dependent to ${mDep.toFixed(3)}`,
        expected:'weights moved under both (a silent net would prove nothing)',
        pass: Math.abs(mAdd - base) > 0.01 && Math.abs(mDep - base) > 0.01 },
      { label:'weight-dependent is pulled toward A+ / A-', measured:
          `ends ${Math.abs(mDep-fp).toFixed(3)} from the fixed point of ${fp.toFixed(2)}, ` +
          `additive ends ${Math.abs(mAdd-fp).toFixed(3)} away`,
        expected:'weight-dependent closer to the fixed point than additive',
        pass: Math.abs(mDep - fp) < Math.abs(mAdd - fp) },
      { label:'weight-dependent moves further in the same window', measured:
          `${(base-mDep).toFixed(3)} against ${(base-mAdd).toFixed(3)} Hz of drift`,
        expected:'the rule with a fixed point converges faster than linear drift',
        pass: Math.abs(base-mDep) > Math.abs(base-mAdd) },
      { label:'neither rule pinned a bound', measured:
          `additive ${mAdd.toFixed(3)}, weight-dependent ${mDep.toFixed(3)}, bounds 1e-4 to 10`,
        expected:'means stay interior over this window',
        pass: mAdd > 0.01 && mAdd < 9.9 && mDep > 0.01 && mDep < 9.9 },
    ];
    if(typeof navigator !== 'undefined' && navigator.gpu){
      const g = await cfg(1, true);
      if(g.error) checks.push({ label:'gpu depression parity', measured:g.error,
        expected:'no engine error', pass:false });
      else {
        const mG = meanExc(g.w);
        checks.push({ label:'gpu depression parity', measured:
            `gpu ${mG.toFixed(3)} vs cpu ${mDep.toFixed(3)}`,
          expected:'within 25% of each other',
          pass: Math.abs(mG - mDep)/Math.max(1e-6, mDep) < 0.25 });
      }
    }
    return checks;
  } },

{ name:'spike statistics match the reference implementation', topic:'analysis', source:'Elephant 1.2.1 (INCF Electrophysiology Analysis Toolkit) and Neo 0.14.5, the implementations the field uses. Every one of these measures has small choices inside it, population or sample variance, whether adjacent interval pairs are counted once or twice, what a train too short for the quantity should return, and a measure that differs from every published one by sqrt(N/(N-1)) is worse than no measure because it looks right. src/spikestats.test.mjs runs both implementations over the same trains and compares them value by value; this group checks the properties that hold by construction, which is what can be gated in a browser',
  async run(){
    const M = await import('./spikestats.js');
    // seeded, because a test that draws fresh randomness disagrees with itself and the disagreement gets read as a tolerance problem
    let x = 12345 >>> 0;
    const rnd = () => { x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
      return x/4294967296; };
    const T = 20000;
    const poisson = rateHz => { const out = []; let t = 0; const mu = 1000/rateHz;
      for(;;){ t += -Math.log(1 - rnd())*mu; if(t >= T) break; out.push(t); } return out; };
    const regular = rateHz => { const out = [], st = 1000/rateHz;
      for(let t = st; t < T; t += st) out.push(t); return out; };
    const p1 = poisson(20), p2 = poisson(20), p3 = poisson(20), reg = regular(20);
    const rows = [
      { label:'Poisson CV is 1', measured:M.cv(p1).toFixed(3),
        expected:'1 within 0.15 (exponential intervals)', pass:Math.abs(M.cv(p1) - 1) < 0.15 },
      { label:'Poisson LV is 1', measured:M.lv(p1).toFixed(3),
        expected:'1 within 0.2 (Shinomoto 2003)', pass:Math.abs(M.lv(p1) - 1) < 0.2 },
      { label:'a regular train has CV 0', measured:M.cv(reg).toExponential(1),
        expected:'0', pass:M.cv(reg) < 1e-9 },
      { label:'mean rate is spikes over seconds', measured:M.meanRate(reg, 0, T).toFixed(3) + ' Hz',
        expected:'20 Hz', pass:Math.abs(M.meanRate(reg, 0, T) - 20) < 0.1 },
    ];
    // the Fano factor of repeated Poisson observations is near 1, and it is not the population synchrony number audit.html reports
    const ff = M.fanoFactor([p1, p2, p3]);
    const sync = M.populationSynchrony([p1, p2, p3], 50, 0, T);
    rows.push({ label:'the Fano factor is not the synchrony measure',
      measured:'fano ' + ff.toFixed(3) + ', synchrony ' + sync.toFixed(3),
      expected:'two different numbers, and they have different names now',
      pass:Number.isFinite(ff) && Number.isFinite(sync) && ff !== sync });
    // The Fano factor must be taken over repeats of one stimulus.
    // Pooling items measures tuning instead, and does it silently: on counts that are Poisson by construction, pooling six items each driving a different cell read 3.81 where the answer is 1.
    const A = await import('./analysis.js');
    let y = 999 >>> 0;
    const rnd2 = () => { y ^= y << 13; y >>>= 0; y ^= y >> 17; y ^= y << 5; y >>>= 0;
      return y/4294967296; };
    const pois = lam => { const L = Math.exp(-lam); let n = 0, p = 1;
      do { n++; p *= rnd2(); } while(p > L); return n - 1; };
    // Built in the flat (trials, labels) layout the readout passes, not the {label, vec} objects the recorder's export uses, so the shape is part of what this asserts.
    const nI = 6, nN = 40, reps = 24;
    const nT = reps*nI;
    const tuned = new Float32Array(nT*nN), tunedLab = new Int32Array(nT);
    for(let rep = 0; rep < reps; rep++)
      for(let i = 0; i < nI; i++){
        const t = rep*nI + i;
        tunedLab[t] = i;
        for(let k = 0; k < nN; k++)
          tuned[t*nN + k] = pois((k % nI) === i ? 8 : 1);
      }
    const grouped = A.fanoPerCell(tuned, tunedLab, nI, nN);
    rows.push({ label:'Fano over repeats of one item is 1 for Poisson counts',
      measured:grouped.mean.toFixed(3) + ' over ' + grouped.cells + ' cell-items',
      expected:'near 1, allowing the (reps-1)/reps bias of the population variance',
      pass:grouped.mean > 0.8 && grouped.mean < 1.2 });
    // the same data pooled across items
    const pooled = A.fanoPerCell(tuned, new Int32Array(nT), 1, nN);
    rows.push({ label:'pooling items inflates it, which is why it is grouped',
      measured:'pooled ' + pooled.mean.toFixed(2) + ' against grouped ' + grouped.mean.toFixed(2),
      expected:'pooled well above grouped, since tuning becomes variance',
      pass:pooled.mean > grouped.mean*2 });
    // The readout's own call, run exactly as train.js makes it, so a change to the argument order fails here rather than several checkpoints into a training run.
    let threw = null;
    try { A.fanoPerCell(tuned, tunedLab, nI, nN); } catch(e){ threw = e.message; }
    rows.push({ label:'the readout call signature works',
      measured:threw ? 'threw: ' + threw : 'returns a number',
      expected:'(trials, labels, nItems, nNeurons), the order every measure here takes',
      pass:threw === null });

    // every spike lands in exactly one bin
    const h = M.timeHistogram([p1, p2, p3], 50, 0, T);
    let binned = 0; for(const v of h) binned += v;
    const spikes = p1.length + p2.length + p3.length;
    rows.push({ label:'the histogram loses no spikes',
      measured:binned + ' binned of ' + spikes, expected:'all of them',
      pass:binned === spikes && h.length === T/50 });
    return rows;
  } },

{ name:'engines obey the per-synapse rule', source:'internal invariant. The rule byte is assigned by the wiring and the table is made and sent, and the graph, the pair table and the documentation all say a pathway learns by its own rule; this asks whether a synapse given a different amplitude moves differently',
  async run(){
    const rows = [];
    // One network, two identical halves of the synapse array.
    // Every synapse is plastic and driven the same way, so the only thing that can separate the two halves is the rule each was given.
    const pts = mkPts(600, 220, 5);
    const net = wireConnect(pts, { ...CN, radius:200, prob:0.3, wExc:4, wInh:-6 }, null);
    const m = net.synCount;
    const w0 = net.w.slice();
    // rule 1 for the first half, rule 2 for the second
    const pmask = new Uint8Array(m);
    for(let s = 0; s < m; s++) pmask[s] = s < (m >> 1) ? 1 : 2;
    const base = { aP:0.004, aM:0.0005, wmax:30, wdep:0, trip:0, het:0, tin:0,
      iEta:0.002, cons:0, consW:0, consP:10 };
    // the second rule potentiates five times as hard and nothing else differs
    const { ruleTable, ruleCount } = buildRuleTable(base, [{ name:'strong', aP:0.02 }]);
    const drive = [{ idx:Uint32Array.from({ length:net.count }, (_, i) => i), amp:11,
      mode:3, period:100, width:10, t0:0, duration:0 }];
    // the drive belongs in protocols; inputs is the curriculum path and an engine given a drive there simply never fires, which reads as a plasticity result of exactly zero
    const cfg = { ...net, ...base, plast:1, tauS:20, iRho:5, rhoMode:0, calS:10,
      scale:0, sEta:0.001, tauY:114, refrac:0, stp:0,
      pmask, ruleTable, ruleCount, protocols:drive };
    const meanExcHalf = (w, lo, hi) => {
      let sum = 0, c = 0;
      for(let k = lo; k < hi; k++) if(w0[k] > 0){ sum += w[k] - w0[k]; c++; }
      return c ? sum/c : 0;
    };
    for(const gpu of [false, true]){
      if(gpu && (typeof navigator === 'undefined' || !navigator.gpu)){
        rows.push({ label:'gpu', measured:'WebGPU unavailable, arm skipped',
          expected:'runs where WebGPU exists', pass:false, skipped:true });
        continue;
      }
      const r = await simRun(cfg, 2000, { steps:1, weights:true, gpu });
      if(r.error || !r.w){
        rows.push({ label:(gpu ? 'gpu' : 'cpu') + ' run',
          measured:r.error || 'the engine returned no weights',
          expected:'the engine answers with weights', pass:false });
        continue;
      }
      const dA = meanExcHalf(r.w, 0, m >> 1);
      const dB = meanExcHalf(r.w, m >> 1, m);
      const eng = gpu ? 'gpu' : 'cpu';
      // both must move, or the run proves nothing about either rule
      rows.push({ label:`${eng}: both rules moved weights`,
        measured:`rule 1 ${dA.toFixed(4)}, rule 2 ${dB.toFixed(4)}`,
        expected:'neither half is inert', pass:Math.abs(dA) > 1e-5 && Math.abs(dB) > 1e-5 });
      // and the stronger rule must move further, which is the claim
      rows.push({ label:`${eng}: the stronger rule potentiates more`,
        measured:`rule 2 moved ${(dB/(dA || 1e-9)).toFixed(2)}x rule 1`,
        expected:'near fivefold, since that is the ratio of the two amplitudes',
        pass:dB > dA*3.5 && dB < dA*6.5 });
    }
    return rows;
  } },

{ name:'cross-engine equivalence (gpu)', source:'ENGINE.md contract: engines are statistically equivalent, never bitwise (independent RNG streams and initial state jitter)',
  async run(){
    if(typeof navigator === 'undefined' || !navigator.gpu)
      return [{ label:'WebGPU availability', measured:'unavailable in this browser, group skipped',
        expected:'runs where WebGPU exists', pass:false, skipped:true }];
    const pts = mkPts(3000, 300, 11);
    const net = wireConnect(pts, { ...CN, wInh:-12 }, null);
    const allIdx = Uint32Array.from({ length: net.count }, (_, i) => i);
    const noise = [{ idx:allIdx, amp:8, mode:3, period:100, width:10, t0:0, duration:0 }];
    const base = { ...net, syn:0, protocols:noise, plast:0 };
    const MS = 2000, SEC = MS/1000;
    const stats = r => {
      const rate = r.frames.reduce((a,b)=>a+b,0) / net.count / SEC;
      const cv = r.isis.length > 30 ? std(r.isis)/mean(r.isis) : 0;
      const fano = mean(r.frames) > 0 ? std(r.frames)**2/mean(r.frames) : 0;
      return { rate, cv, fano };
    };
    const opts = { steps:2, isiK:200 };
    const cpu = stats(await simRun(base, MS, opts));
    const g = await simRun(base, MS, { ...opts, gpu:true });
    if(g.error) return [{ label:'gpu engine runs', measured:g.error,
      expected:'no engine error', pass:false }];
    const gpu = stats(g);
    const gp = await simRun(base, MS, { ...opts, gpu:true, partCap: 100000 });
    if(gp.error) return [{ label:'partitioned gpu engine runs', measured:gp.error,
      expected:'no engine error', pass:false }];
    const gpuP = stats(gp);
    // Short-term depression, on the same network as the parity checks above so only the term differs.
    // The second check is the one that matters: an engine that ignores the term entirely still passes a cpu-to-gpu comparison if both ignore it, so the rate also has to move. stpNorm 1: with the published order a rested synapse releases U, and unscaled this network falls to about 0.5 Hz (2026-09-14), too quiet for a rate comparison; scaled, it stays active and the scale is exercised.
    const sBase = { ...base, stp:1, stpNorm:1, stpU:0.2, stpTauD:200, stpTauF:600 };
    const sCpu = stats(await simRun(sBase, MS, opts));
    const sG = await simRun(sBase, MS, { ...opts, gpu:true });
    if(sG.error) return [{ label:'gpu engine runs with short-term plasticity',
      measured:sG.error, expected:'no engine error', pass:false }];
    const sGpu = stats(sG);
    // the other release order (stpOrder 1, facilitation before release) on both engines: parity, and the order has to move the rate on each
    const oBase = { ...sBase, stpOrder:1 };
    const oCpu = stats(await simRun(oBase, MS, opts));
    const oG = await simRun(oBase, MS, { ...opts, gpu:true });
    if(oG.error) return [{ label:'gpu engine runs with facilitation before release',
      measured:oG.error, expected:'no engine error', pass:false }];
    const oGpu = stats(oG);
    const ge = stats(await simRun({ ...base, syn:1, psc:1, tauE:3, tauI:8 }, MS, { ...opts, gpu:true }));
    const ce = stats(await simRun({ ...base, syn:1, psc:1, tauE:3, tauI:8 }, MS, opts));
    // the exact convention (psc 2): parity on both engines, and it has to move the rate against the peak convention on the same weights, since an engine that ignores the mode passes parity with one that does too
    const gx = stats(await simRun({ ...base, syn:1, psc:2, tauE:3, tauI:8 }, MS, { ...opts, gpu:true }));
    const cx = stats(await simRun({ ...base, syn:1, psc:2, tauE:3, tauI:8 }, MS, opts));
    const cp = stats(await simRun({ ...base, syn:1, psc:0, tauE:3, tauI:8 }, MS, opts));
    // plasticity parity: the forced-pairing protocol is deterministic (noise free, pulse driven), so the two engines should land near-identical dw
    const dw = async (dt, gpu) => {
      const net2 = { count:2, ntype:new Uint8Array([0,0]), bias:new Float32Array(2),
        pos:new Float32Array(6), src:[1,1], lidx:new Uint32Array([0,1]),
        preStart:new Int32Array([0,1,1]), post:new Int32Array([1]),
        w:new Float32Array([1]), delay:new Uint8Array([1]), synCount:1,
        syn:0, plast:1, aP:0.01, aM:0.012, tauS:16.8, tauM:33.7, wmax:10, iEta:0, iRho:5,
        protocols:[
          { idx:new Uint32Array([0]), amp:60, mode:1, period:200, width:3, t0:60, duration:0 },
          { idx:new Uint32Array([1]), amp:60, mode:1, period:200, width:3, t0:60+dt, duration:0 },
        ] };
      const r = await simRun(net2, 4100, { steps:50, weights:true, gpu });
      return r.error !== undefined ? NaN : r.w[0] - 1;
    };
    const cLtp = await dw(10, false), gLtp = await dw(10, true);
    const cLtd = await dw(-10, false), gLtd = await dw(-10, true);
    // Triplet and transmitter terms on the same deterministic two-neuron pairing harness the mechanism group uses.
    // Noise free and pulse driven, so the two engines see identical input and any difference is the engine rather than the network.
    const pairDw = async (periodMs, extra, gpu) => {
      const net2 = { count:2, ntype:new Uint8Array([0,0]), bias:new Float32Array(2),
        preStart:new Int32Array([0,1,1]), post:new Int32Array([1]),
        w:new Float32Array([1]), delay:new Uint8Array([1]), synCount:1,
        syn:0, plast:1, aP:0.01, aM:0.012, tauS:16.8, tauM:33.7, wmax:10, iEta:0, iRho:5,
        ...extra,
        protocols:[
          { idx:new Uint32Array([0]), amp:60, mode:1, period:periodMs, width:3,
            t0:60, duration:20*periodMs },
          { idx:new Uint32Array([1]), amp:60, mode:1, period:periodMs, width:3,
            t0:70, duration:20*periodMs },
        ] };
      // steps:50, not the mechanism group's steps:1.
      // Batching moves when the host sees state, not the dynamics, and on the gpu a one millisecond tick is a submit and a readback each time: at steps:1 this harness took longer than the rest of the battery put together.
      const r = await simRun(net2, 60 + 20*periodMs + 100,
        { steps:50, weights:true, gpu });
      return r.error !== undefined ? NaN : r.w[0] - 1;
    };
    const TR = { trip:0.01, tauY:114 };
    const cTrip = await pairDw(25, TR, false), gTrip = await pairDw(25, TR, true);
    const gPair = await pairDw(25, {}, true);
    // Presynaptic spikes only, postsynaptic cell silent, so the pair terms are inert and the whole change is the drip.
    // The reference expects an exact integer multiple of the delta and so should this engine.
    const tinDw = async (tin, gpu) => {
      const net2 = { count:2, ntype:new Uint8Array([0,0]), bias:new Float32Array(2),
        preStart:new Int32Array([0,1,1]), post:new Int32Array([1]),
        w:new Float32Array([1]), delay:new Uint8Array([1]), synCount:1,
        syn:0, plast:1, aP:0.01, aM:0.012, tauS:16.8, tauM:33.7, wmax:10, iEta:0, iRho:5, tin,
        protocols:[{ idx:new Uint32Array([0]), amp:60, mode:1, period:100,
          width:3, t0:60, duration:2000 }] };
      const r = await simRun(net2, 2400, { steps:50, weights:true, gpu });
      return r.error !== undefined ? NaN : r.w[0] - 1;
    };
    const cTin = await tinDw(0.001, false), gTin = await tinDw(0.001, true);
    const gTin0 = await tinDw(0, true);
    // Heterosynaptic regression, on the mechanism group's harness: pair to lift the weight, then burst the postsynaptic cell alone, where the pull is the only thing that can act.
    // Consolidation rides on the same run because it moves the reference the pull targets, so it is visible in the final weight rather than only in state nothing reads back.
    const hetRun = async (het, cons, gpu, consW = 0) => {
      const net2 = { count:2, ntype:new Uint8Array([0,0]), bias:new Float32Array(2),
        preStart:new Int32Array([0,1,1]), post:new Int32Array([1]),
        w:new Float32Array([3]), delay:new Uint8Array([1]), synCount:1,
        syn:0, plast:1, aP:0.02, aM:0, tauS:20, wmax:10, iEta:0, iRho:5,
        trip:0, tauY:114, het, tin:0, cons, consW, consP:10,
        protocols:[
          { idx:new Uint32Array([0]), amp:60, mode:1, period:50, width:3, t0:60, duration:2000 },
          { idx:new Uint32Array([1]), amp:60, mode:1, period:50, width:3, t0:70, duration:6000 },
        ] };
      const r = await simRun(net2, 6400, { steps:50, weights:true, gpu });
      return r.error !== undefined ? NaN : r.w[0];
    };
    const cHet = await hetRun(0.02, 0, false), gHet = await hetRun(0.02, 0, true);
    const gHet0 = await hetRun(0, 0, true);
    const cCons = await hetRun(0.02, 0.001, false), gCons = await hetRun(0.02, 0.001, true);
    // consW 0 is the default and means a tenth of wmax, here 1.
    const cConsW = await hetRun(0.02, 0.001, false, 1);
    // The same competition carried by a declared rule over a checkpoint with het and cons off.
    // An engine that allocates the reference weights only for the checkpoint's own terms pulls toward a zero placeholder here; it has to end where the checkpoint's own het does, on both engines.
    const hetRuleRun = async (field, value, gpu) => {
      const base = { aP:0.02, aM:0, wmax:10, wdep:0, trip:0, het:0, tin:0, iEta:0, cons:0, consW:0, consP:10 };
      const { ruleTable, ruleCount } = buildRuleTable(base, [{ name:'compete', het:0.02, [field]:value }]);
      const net2 = { count:2, ntype:new Uint8Array([0,0]), bias:new Float32Array(2),
        preStart:new Int32Array([0,1,1]), post:new Int32Array([1]),
        w:new Float32Array([3]), delay:new Uint8Array([1]), synCount:1,
        syn:0, plast:1, ...base, tauS:20, iRho:5, tauY:114,
        pmask:new Uint8Array([2]), ruleTable, ruleCount,
        protocols:[
          { idx:new Uint32Array([0]), amp:60, mode:1, period:50, width:3, t0:60, duration:2000 },
          { idx:new Uint32Array([1]), amp:60, mode:1, period:50, width:3, t0:70, duration:6000 },
        ] };
      const r = await simRun(net2, 6400, { steps:50, weights:true, gpu });
      return r.error !== undefined ? NaN : r.w[0];
    };
    const cHetRule = await hetRuleRun('het', 0.02, false), gHetRule = await hetRuleRun('het', 0.02, true);
    const cConsRule = await hetRuleRun('cons', 0.001, false), gConsRule = await hetRuleRun('cons', 0.001, true);
    const vn = wireConnect(mkPts(1500, 350, 7), { ...CN, radius:250, prob:0.4,
      wExc:4, wInh:-6 }, null);
    Object.assign(vn, { syn:0, plast:1, aP:0, aM:0, tauS:20, wmax:30, iEta:0.005, iRho:5,
      protocols:[{ idx:Uint32Array.from({length:1500},(_,i)=>i), amp:11, mode:3,
        period:100, width:10, t0:0, duration:0 }] });
    const vExc = [];
    for(let i=0;i<vn.count;i++) if(NEURON_TYPES[vn.ntype[i]].sign > 0) vExc.push(i);
    // The same run on both engines: a 0.3 Hz drift over a 16 s chaotic run is inside what a change of float32 term order moves.
    // The reference decides convergence; the gpu is held to it.
    const vw0 = vn.w.slice();
    const homeo = async gpu => {
      let early = 0, late = 0;
      const r = await simRun(vn, 16000, { steps:50, weights:true, gpu,
        onFrame:(fired, t) => {
          let c = 0;
          for(const i of vExc) c += fired[i];
          if(t < 2000) early += c; else if(t >= 12000) late += c;
        } });
      let dInh = 0, nInh = 0;
      if(r.w) for(let s=0;s<vn.synCount;s++)
        if(vw0[s] < 0){ dInh += Math.abs(r.w[s]) - Math.abs(vw0[s]); nInh++; }
      return { error:r.error, earlyHz:early/vExc.length/2, lateHz:late/vExc.length/4, dInh:dInh/Math.max(1, nInh) };
    };
    const vc = await homeo(false), vg = await homeo(true);
    // Retuned mid-run, on the same network.
    // A tune at 3 s: the Vogels target from 5 to 12 Hz; and, with the Vogels rule off, synaptic scaling switched on with a target above the rate, whose first pass has to cover the same window on both engines.
    // The late rate and the weight change after one pass have to agree between the engines.
    const retune = async (gpu, from, change, ms, lateFrom) => {
      let late = 0;
      const cfg0 = { ...vn, ...from };
      const r = await simRun(cfg0, ms, { steps:50, weights:true, gpu,
        driver:(w, t) => { if(t === 3000) w.postMessage({ cmd:'tune', bias:cfg0.bias, ...engineConfig({ ...cfg0, ...change }) }); },
        onFrame:(fired, t) => { if(t >= lateFrom) for(const i of vExc) late += fired[i]; } });
      let dE = 0, nE = 0, dI = 0, nI = 0;
      if(r.w) for(let s=0;s<vn.synCount;s++){
        if(vw0[s] > 0){ dE += r.w[s]/vw0[s]; nE++; }
        else if(vw0[s] < 0){ dI += Math.abs(r.w[s]) - Math.abs(vw0[s]); nI++; }
      }
      return { error:r.error, lateHz:late/vExc.length/((ms - lateFrom)/1000), wRatio:dE/Math.max(1, nE), dInh:dI/Math.max(1, nI) };
    };
    const rhoC = await retune(false, {}, { iRho:12 }, 12000, 8000);
    const rhoG = await retune(true, {}, { iRho:12 }, 12000, 8000);
    const scOff = { iEta:0, scale:0, sEta:0.05, iRho:30 };
    const scC = await retune(false, scOff, { scale:1 }, 4500, 3000);
    const scG = await retune(true, scOff, { scale:1 }, 4500, 3000);
    const vEarlyHz = vg.earlyHz, vLateHz = vg.lateHz, vDInh = vg.dInh, vNInh = 1;
    // Spike counts per tick.
    // Unconnected cells on a fixed bias, 30 to 170 Hz, run with 50 steps per tick, where a 0/1 flag would read at most 20 Hz.
    // No noise and no synapses, so each cell's count is the same on both engines up to float order.
    const nc = 12;
    const cnet = { kind:'net', count:nc, pos:new Float32Array(3*nc), ntype:new Uint8Array(nc),
      bias:Float32Array.from({ length:nc }, (_, i) => 40 + 20*i),
      src:new Int32Array(nc).fill(1), lidx:Uint32Array.from({ length:nc }, (_, i) => i),
      preStart:new Int32Array(nc+1), post:new Int32Array(0), w:new Float32Array(0),
      delay:new Uint8Array(0), synCount:0, plast:0, refrac:2, seed:5, syn:0, tauE:3, tauI:8, protocols:[] };
    const cc = await simRun(cnet, 1000, { steps:50, perNeuron:true });
    const cg = await simRun(cnet, 1000, { steps:50, perNeuron:true, gpu:true });
    const cMax = cc.counts ? Math.max(...cc.counts) : 0, gMax = cg.counts ? Math.max(...cg.counts) : 0;
    const cSum = cc.counts ? cc.counts.reduce((a, b) => a + b, 0) : 0;
    const gSum = cg.counts ? cg.counts.reduce((a, b) => a + b, 0) : 0;
    const worst = cc.counts && cg.counts ? Math.max(...[...cc.counts].map((x, i) => Math.abs(x - cg.counts[i])/Math.max(1, x))) : 1;
    // A pulse train with a per-cell gain (a stimulus's spread), on the same cells with no bias.
    const gnet = { ...cnet, bias:new Float32Array(nc), protocols:[{ idx:cnet.lidx, amp:60, mode:1, period:100,
      width:50, t0:0, duration:0, gain:Float32Array.from({ length:nc }, (_, i) => 0.5 + i/(nc - 1)) }] };
    const pc = await simRun(gnet, 1000, { steps:10, perNeuron:true });
    const pg = await simRun(gnet, 1000, { steps:10, perNeuron:true, gpu:true });
    const pWorst = pc.counts && pg.counts ? Math.max(...[...pc.counts].map((x, i) => Math.abs(x - pg.counts[i])/Math.max(1, x))) : 1;
    const pSpread = pc.counts ? pc.counts[nc - 1] - pc.counts[0] : 0;
    const countRows = [
      { label:'fired is a count per tick on both engines', measured: cg.error || cc.error ||
          `fastest cell ${cMax} Hz cpu, ${gMax} Hz gpu at 50 steps per tick; per-tick counts sum to ` +
          `${cSum} (cpu) and ${gSum} (gpu) against spike totals ${cc.frames.reduce((a, b) => a + b, 0)} and ${cg.frames.reduce((a, b) => a + b, 0)}`,
        expected:'above 20 Hz on both, and the counts summing to the spike total on each', pass:
          !cc.error && !cg.error && cMax > 40 && gMax > 40 &&
          cSum === cc.frames.reduce((a, b) => a + b, 0) && gSum === cg.frames.reduce((a, b) => a + b, 0) },
      { label:'per-cell count parity', measured:`largest per-cell difference ${(100*worst).toFixed(1)}%`,
        expected:'within 3% on every cell', pass: worst <= 0.03 },
      { label:'a pulse with a per-cell gain', measured: pc.error || pg.error ||
          `counts ${pc.counts[0]} to ${pc.counts[nc - 1]} (cpu), ${pg.counts[0]} to ${pg.counts[nc - 1]} (gpu) over gains 0.5 to 1.5; ` +
          `largest per-cell difference ${(100*pWorst).toFixed(1)}%`,
        expected:'the count rising with the gain, and within 3% on every cell', pass:
          !pc.error && !pg.error && pSpread > 5 && pWorst <= 0.03 },
    ];
    const retuneRows = [
      { label:'a retuned Vogels target reaches both engines', measured: rhoC.error || rhoG.error ||
          `late rate after iRho 5 to 12 at 3 s: gpu ${rhoG.lateHz.toFixed(2)}, cpu ${rhoC.lateHz.toFixed(2)} Hz; ` +
          `unretuned cpu ${vc.lateHz.toFixed(2)}; mean inhibitory change gpu ${rhoG.dInh.toFixed(3)}, cpu ${rhoC.dInh.toFixed(3)}`,
        expected:'the cpu rate above the unretuned one, the gpu within 15% of the cpu', pass:
          !rhoC.error && !rhoG.error && rhoC.lateHz > vc.lateHz + 0.3 &&
          Math.abs(rhoG.lateHz - rhoC.lateHz) < 0.15*rhoC.lateHz },
      { label:'scaling switched on mid-run takes one ordinary step', measured: scC.error || scG.error ||
          `mean excitatory weight ratio after the first pass: gpu ${scG.wRatio.toFixed(4)}, cpu ${scC.wRatio.toFixed(4)} (sEta 0.05, one pass at most 1.10)`,
        expected:'both above 1 and at most 1 + 2 sEta, the gpu change within 30% of the cpu', pass:
          !scC.error && !scG.error && scC.wRatio > 1.001 && scC.wRatio <= 1.1 + 1e-6 && scG.wRatio <= 1.1 + 1e-6 &&
          Math.abs((scG.wRatio - 1) - (scC.wRatio - 1)) < 0.3*(scC.wRatio - 1) },
    ];
    return [
      ...countRows,
      ...retuneRows,
      // the three bands live in equivalence.js, where tools/enginecheck.mjs reads them for an engine of one's own
      { label:'population rate parity (kick)', measured:
          `gpu ${gpu.rate.toFixed(1)} vs cpu ${cpu.rate.toFixed(1)} Hz`,
        expected:'within ' + EQUIV.rate*100 + '%', pass: rateWithin(gpu.rate, cpu.rate) },
      { label:'CV of ISI parity', measured:
          `gpu ${gpu.cv.toFixed(2)} vs cpu ${cpu.cv.toFixed(2)}`,
        expected:'within ' + EQUIV.cv + ' absolute', pass: cvWithin(gpu.cv, cpu.cv) },
      { label:'population Fano parity', measured:
          `gpu ${gpu.fano.toFixed(1)} vs cpu ${cpu.fano.toFixed(1)}`,
        expected:'within a factor of ' + EQUIV.sync, pass: syncWithin(gpu.fano, cpu.fano) },
      { label:'partitioned synapse store matches', measured:
          `${gp.parts} partitions, ${gpuP.rate.toFixed(1)} vs ${gpu.rate.toFixed(1)} Hz`,
        expected:'>= 2 partitions forced, rate within 15%', pass:
          gp.parts >= 2 && Math.abs(gpuP.rate-gpu.rate)/Math.max(0.1,gpu.rate) < 0.15 },
      { label:'population rate parity (exp synapses)', measured:
          `gpu ${ge.rate.toFixed(1)} vs cpu ${ce.rate.toFixed(1)} Hz`,
        expected:'within 25%', pass: ce.rate > 0.5 &&
          Math.abs(ge.rate-ce.rate)/ce.rate < 0.25 },
      { label:'population rate parity (exp synapses, exact integral)', measured:
          `gpu ${gx.rate.toFixed(1)} vs cpu ${cx.rate.toFixed(1)} Hz`,
        expected:'within 25%', pass: cx.rate > 0.5 &&
          Math.abs(gx.rate-cx.rate)/cx.rate < 0.25 },
      { label:'the exact integral delivers less charge than the held current', measured:
          `exact ${cx.rate.toFixed(1)} vs peak ${cp.rate.toFixed(1)} Hz on the same weights`,
        expected:'exact below peak, by more than 2%', pass: cp.rate > 0.5 && cx.rate < cp.rate*0.98 },
      { label:'STDP window parity', measured:
          `LTP gpu ${gLtp.toFixed(3)} vs cpu ${cLtp.toFixed(3)}, ` +
          `LTD gpu ${gLtd.toFixed(3)} vs cpu ${cLtd.toFixed(3)}`,
        expected:'same signs, within 15%', pass:
          gLtp > 0 && gLtd < 0 &&
          Math.abs(gLtp-cLtp) < 0.15*Math.abs(cLtp) &&
          Math.abs(gLtd-cLtd) < 0.15*Math.abs(cLtd) },
      { label:'short-term plasticity parity, facilitation before release (stpOrder 1)', measured:
          `gpu ${oGpu.rate.toFixed(1)} vs cpu ${oCpu.rate.toFixed(1)} Hz`,
        expected:'within 20%', pass: oCpu.rate > 0.5 &&
          Math.abs(oGpu.rate-oCpu.rate)/oCpu.rate < 0.2 },
      { label:'the release order moves the rate the same way on both engines', measured:
          `cpu ${sCpu.rate.toFixed(1)} to ${oCpu.rate.toFixed(1)} Hz, gpu ${sGpu.rate.toFixed(1)} to ${oGpu.rate.toFixed(1)} Hz`,
        expected:'the cpu rate changes by more than 5%, the gpu rate in the same direction', pass:
          Math.abs(oCpu.rate-sCpu.rate)/Math.max(0.1, sCpu.rate) > 0.05 &&
          Math.sign(oGpu.rate-sGpu.rate) === Math.sign(oCpu.rate-sCpu.rate) },
      { label:'short-term plasticity parity', measured:
          `gpu ${sGpu.rate.toFixed(1)} vs cpu ${sCpu.rate.toFixed(1)} Hz`,
        expected:'within 20%', pass: sCpu.rate > 0.5 &&
          Math.abs(sGpu.rate-sCpu.rate)/sCpu.rate < 0.2 },
      { label:'short-term plasticity is not ignored on gpu', measured:
          `stp on ${sGpu.rate.toFixed(1)} Hz vs off ${gpu.rate.toFixed(1)} Hz`,
        expected:'rate differs by more than 20%', pass:
          Math.abs(sGpu.rate-gpu.rate)/Math.max(0.1, gpu.rate) > 0.2 },
      { label:'triplet parity', measured:
          `gpu ${gTrip.toFixed(4)} vs cpu ${cTrip.toFixed(4)} dw at 25 ms pairing`,
        expected:'same sign, within 15%', pass:
          cTrip > 0 && gTrip > 0 && Math.abs(gTrip-cTrip) < 0.15*Math.abs(cTrip) },
      { label:'triplet is not ignored on gpu', measured:
          `gpu trip on ${gTrip.toFixed(4)} vs off ${gPair.toFixed(4)}`,
        expected:'triplet potentiates more at high pairing frequency', pass:
          gTrip > gPair + 0.01 },
      { label:'transmitter parity', measured:
          `gpu ${gTin.toFixed(4)} vs cpu ${cTin.toFixed(4)}, gpu off ${gTin0.toFixed(6)}`,
        expected:'within 5%, and exactly zero with the term off', pass:
          cTin > 0.01 && Math.abs(gTin-cTin) < 0.05*cTin && Math.abs(gTin0) < 1e-6 },
      { label:'heterosynaptic parity', measured:
          `gpu ${gHet.toFixed(3)} vs cpu ${cHet.toFixed(3)} final w (start 3)`,
        expected:'within 10%', pass:
          isFinite(cHet) && isFinite(gHet) && Math.abs(gHet-cHet) < 0.1*Math.abs(cHet) },
      { label:'heterosynaptic is not ignored on gpu', measured:
          `gpu het on ${gHet.toFixed(3)} vs off ${gHet0.toFixed(3)} (start 3)`,
        expected:'het on ends closer to the starting weight', pass:
          Math.abs(gHet-3) < Math.abs(gHet0-3) - 0.05 },
      { label:'consolidation parity', measured:
          `gpu ${gCons.toFixed(3)} vs cpu ${cCons.toFixed(3)} final w with cons on`,
        expected:'within 10%, and the reference moved off the frozen snapshot', pass:
          isFinite(cCons) && isFinite(gCons) &&
          Math.abs(gCons-cCons) < 0.1*Math.abs(cCons) &&
          Math.abs(gCons-gHet) > 1e-4 },
      { label:'het in a declared rule only', measured:
          `gpu ${gHetRule.toFixed(3)} vs cpu ${cHetRule.toFixed(3)}, checkpoint het ${cHet.toFixed(3)} (start 3)`,
        expected:'both within 10% of each other and of the checkpoint het run', pass:
          isFinite(cHetRule) && isFinite(gHetRule) && Math.abs(gHetRule - cHetRule) < 0.1*Math.abs(cHetRule) &&
          Math.abs(cHetRule - cHet) < 0.1*Math.abs(cHet) },
      { label:'consolidation in a declared rule only', measured:
          `gpu ${gConsRule.toFixed(3)} vs cpu ${cConsRule.toFixed(3)}, checkpoint cons ${cCons.toFixed(3)}`,
        expected:'both within 10% of each other and of the checkpoint cons run', pass:
          isFinite(cConsRule) && isFinite(gConsRule) && Math.abs(gConsRule - cConsRule) < 0.1*Math.abs(cConsRule) &&
          Math.abs(cConsRule - cCons) < 0.1*Math.abs(cCons) },
      { label:'consolidated weight 0 means a tenth of wmax', measured:
          `final w ${cCons.toFixed(4)} with consW 0, ${cConsW.toFixed(4)} with consW 1 (wmax 10)`,
        expected:'identical', pass:
          isFinite(cCons) && cCons === cConsW },
      { label:'homeostasis on gpu tracks the reference', measured:
          vg.error || `gpu ${vEarlyHz.toFixed(1)} to ${vLateHz.toFixed(1)} Hz, cpu ${vc.earlyHz.toFixed(1)} to ${vc.lateHz.toFixed(1)}; mean |dW inh| gpu ${vDInh.toFixed(3)} cpu ${vc.dInh.toFixed(3)}`,
        expected:'late rates within 15% (or 0.5 Hz), inhibition strengthened on both, within 25% of each other', pass:
          !vg.error && !vc.error && Math.abs(vLateHz - vc.lateHz) <= Math.max(0.5, 0.15*vc.lateHz) &&
          vDInh > 0.05 && vc.dInh > 0.05 && Math.abs(vDInh - vc.dInh) <= 0.25*Math.max(vDInh, vc.dInh) },
    ];
  } },
];

// Groups carrying a topic are scenario or measure specific: they check that a scene or an observer-side measure behaves, not that an engine or a computation is correct, so they run when somebody is working on that topic.
// Everything untagged is a core invariant and always runs.
// The point is a smaller surface to read and maintain, not a faster run.
export const TOPICS = [...new Set(TESTS.map(t => t.topic).filter(Boolean))];

export function selectTests(topics){
  const on = new Set(topics || []);
  if(on.has('all')) return TESTS.slice();
  return TESTS.filter(t => !t.topic || on.has(t.topic));
}

export async function runAll(onTest, tests){
  const results = [];
  for(const t of (tests || TESTS)){
    const t0 = performance.now();
    let checks;
    try { checks = await t.run(); }
    catch(e){ checks = [{ label:'test crashed', measured:e.message, expected:'no crash', pass:false }]; }
    // A check whose work could not run here (no WebGPU) is skipped, not passed, so a browser without WebGPU does not show the cross-engine group green with no gpu check run.
    // A group whose every check was skipped did not run and is not counted as passed; one with some skipped passes on the checks that ran.
    const skipped = checks.length > 0 && checks.every(c => c.skipped);
    const res = { name:t.name, source:t.source, checks,
      ms: Math.round(performance.now()-t0), skipped,
      pass: !skipped && checks.every(c => c.pass || c.advisory || c.skipped) };
    results.push(res);
    if(onTest) onTest(res);
  }
  return results;
}

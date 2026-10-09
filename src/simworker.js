// Izhikevich network simulation. dt = 1ms (two 0.5ms substeps for v).
// Synapse modes: kick (instant charge, one signed ring), exp (separate E/I rings feeding per-neuron decaying accumulators, scaled by the psc convention: w is the peak current, or under psc 1 the total charge), and conductance (the exp decay, each channel driving g (E - v); MODEL.md 2).
// Stimulus protocols (pulse train / ramp) are evaluated here per sim-ms so editing them never rebuilds the network.
import { gradedArrays, freezeGraded } from './nodes.js';
import { checkProtocol, helloReply, TERM_NAMES, OPTIONAL } from './protocol.js';
import { noiseDraw } from './rand.js';

let S = null;
// the stream switches are kept across an init: the trainer asks for latency before it sends the network, so a request that arrives with no network has to survive until one does
const streams = { sendV:false, sendT:false };

// the per-step scale on an exponential accumulator under each psc convention (dec = e^(-1/tau)): 1 the peak, 1 - dec the charge, and tau (1 - dec) the exact integral over the millisecond
export const itauOf = (psc, dec, tau) => psc === 2 ? tau*(1 - dec) : psc === 1 ? 1 - dec : 1;
function configure(m){
  // a tune without a bias keeps the one in force, as the other engines do
  if(m.bias) S.bias = m.bias;
  S.exp = (m.syn|0) >= 1;
  // conductance mode (MODEL.md 2): every channel drives g (E - v)
  S.cond = (m.syn|0) === 2;
  S.eRevE = Number.isFinite(+m.eRevE) ? +m.eRevE : 0;
  S.eRevI = Number.isFinite(+m.eRevI) ? +m.eRevI : -70;
  S.chanErev = Float32Array.from(Array.isArray(m.chanErev) || ArrayBuffer.isView(m.chanErev) ? m.chanErev : (S.chanErev || []));
  const tE = m.tauE > 0 ? m.tauE : 3, tI = m.tauI > 0 ? m.tauI : 8;
  S.decE = Math.exp(-1/tE); S.decI = Math.exp(-1/tI);
  // psc 0: w is the peak of the exponential current, the jump is w and the charge per spike is w tau (NEST iaf_psc_exp, Brian). psc 1: w is the total charge, the jump is scaled by (1 - e^(-1/tau)) so the discrete sum over steps is exactly w, what kick mode delivers. psc 2: w is the peak as in 0, and the current applied over a step is the integral of the exponential across that step, tau (1 - e^(-1/tau)), so the charge a spike delivers is w tau exactly rather than 1.18 w tau at tauE 3 (the held current over-delivers by 1/(tau (1 - e^(-1/tau))); MODEL.md, what the one millisecond step costs).
  S.psc = Math.max(0, Math.min(2, m.psc|0));
  S.itauE = itauOf(S.psc, S.decE, tE); S.itauI = itauOf(S.psc, S.decI, tI);
  if(S.exp && !S.ring2){
    S.ring2 = new Float32Array(S.ROWS*S.n);
    S.gE = new Float32Array(S.n); S.gI = new Float32Array(S.n);
  }
  // Receptor channels (MODEL.md 2): channels 2 and up carry their own time constant; a synapse's channel is the top three bits of its plasticity byte, zero meaning by sign into E or I as always.
  // They need the exponential synapse, since a kick has no time constant to give.
  const extra = Array.isArray(m.chanTau) || ArrayBuffer.isView(m.chanTau) ? Array.from(m.chanTau) : (S.chanTau || []);
  if(extra.length && !S.exp) throw new Error('receptor channels need exponential synapses: set the checkpoint synapse mode to exp');
  if(extra.length > 6) throw new Error('receptor channels: at most six beyond E and I');
  S.chanTau = extra;
  S.KX = extra.length;
  S.chanDec = Float32Array.from(extra, t => Math.exp(-1/(t > 0 ? t : 3)));   // chanDec: decX is short-term plasticity's
  S.chanItau = Float32Array.from(S.chanDec, (d, k) => itauOf(S.psc, d, extra[k] > 0 ? extra[k] : 3));
  if(S.KX && (!S.gX || S.gX.length !== S.KX*S.n)){
    S.gX = new Float32Array(S.KX*S.n); S.ringX = new Float32Array(S.KX*S.ROWS*S.n);
  }
  S.protos = []; S.nprotos = [];             // DC (pulse/ramp) vs noise-amplitude protocols
  for(const p of (m.protocols || []))
    (p.mode === 3 ? S.nprotos : S.protos).push({ ...p, _s: -1 });
  S.stim = null; S.namp = null;
  // framed input maps (live encoders): per map, a channel -> neuron CSR; inputFrame messages set per-channel gains between frames
  if(m.inputs !== undefined){
    S.inputs = (m.inputs || []).map(im => ({ ...im,
      gains: new Float32Array(im.chStart.length - 1) }));
    S.ext = null; S.extDirty = true;
  }
  // plasticity: pair STDP on excitatory synapses (Bi and Poo 1998 window; online trace form after Song, Miller, Abbott 2000) and homeostatic inhibitory plasticity (Vogels et al. 2011).
  // Interactions use spike emission times; axonal delays affect current delivery only.
  S.plast = (m.plast|0) === 1;
  // an explicit zero amplitude is a zero, not a request for the default
  S.aP = m.aP !== undefined ? +m.aP : 0.008; S.aM = m.aM !== undefined ? +m.aM : 0.001;
  // two window time constants: tauS for the presynaptic (potentiation) trace, tauM for the postsynaptic (depression) trace; Bi and Poo 1998 fit 16.8 and 33.7, Song et al. 2000 use 20 and 20
  const tauS = m.tauS > 0 ? m.tauS : 16.8;
  const tauM = m.tauM > 0 ? m.tauM : tauS;
  S.decS = Math.exp(-1/tauS); S.decM = Math.exp(-1/tauM);
  S.wmax = m.wmax > 0 ? +m.wmax : 30;
  S.wdep = (m.wdep|0) === 1;                 // weight-dependent excitatory depression
  S.iEta = m.iEta !== undefined ? +m.iEta : 0.002;
  S.iRho = m.iRho || 5;
  // Vogels alpha = rho (tau+ + tau-): the fixed point of the inhibitory rule with distinct trace constants (ENGINE.md section 3); equal constants give the published 2 rho tau
  S.iAlpha = S.iRho*0.001*(tauS + tauM);
  // homeostatic synaptic scaling (Turrigiano 2008; van Rossum 2000 multiplicative form): once per simulated second, each neuron's plastic excitatory inputs scale a small step toward the target rate
  S.scaleOn = (m.scale|0) === 1;
  S.sEta = m.sEta > 0 ? m.sEta : 0.001;
  // rate set points: fixed (every neuron targets iRho) or measured (run the first calS seconds with the rules off, record each neuron's own baseline rate, and lock those as personal set points; Hengen & Turrigiano: firing-rate set points are individual, and a uniform target drives heterogeneous circuits into synchronous bursting)
  S.rhoMode = (m.rhoMode|0) === 1 ? 1 : 0;
  S.calMs = (m.calS > 0 ? m.calS : 10) * 1000;
  S.tauSv = tauS; S.tauMv = tauM;
  // Beyond the pair rule: the fast mechanism set of Zenke, Agnes and Gerstner 2015 (Nat Commun 6:6922), each grounded separately, each off by default so the committed pair rule is byte-identical without them. trip is the triplet LTP amplitude (Pfister and Gerstner 2006): potentiation at a postsynaptic spike becomes (aP + trip*Kslow) * Kpre, with Kslow the slow postsynaptic trace read before the current spike, which gives LTP the postsynaptic-rate dependence pair STDP cannot fit (Sjostrom, Turrigiano and Nelson 2001; Wang et al. 2005). tauY is the slow trace time constant in ms; 114 is the Pfister-Gerstner visual cortex fit. het is heterosynaptic regression of every incoming plastic excitatory weight toward its reference, gated by the cubed slow trace so only bursting engages it (Lynch et al. 1977; Chistiakova et al. 2014; implementation form of Zenke et al. 2015).
  // The reference is the weight vector at the moment het is switched on; after a resume that is the resumed state, a baseline that moves at checkpoint timescale. tin is transmitter-induced potentiation per presynaptic spike, the small non-Hebbian term that keeps low-rate synapses from silent collapse; the least measured of the set and stated so in the source.
  // Absolute refractory floor in ms, default 0.
  // The Izhikevich reset provides relative refractoriness only, so under saturating drive nothing stops a next-millisecond spike; sodium channel inactivation enforces one to two silent ms in real neurons.
  S.refrac = m.refrac > 0 ? Math.min(20, m.refrac|0) : 0;
  // Membrane floor: the potassium reversal potential, below which no hyperpolarizing current takes v.
  // Applied after each half step in every engine.
  // Below rest the Izhikevich quadratic grows with the square of the distance, and it sends a membrane pushed to about -150 mV (classic rows) over threshold or to infinity; a synchronous inhibitory volley (a feedback pool's onset kick) can do that to a whole population, which then sits at NaN, silent, for the run.
  // Default -90; a value at or above 0 disables the floor.
  S.vmin = m.vmin === undefined ? -90 : (+m.vmin < 0 ? +m.vmin : -Infinity);
  if(S.refrac > 0 && (!S.refCnt || S.refCnt.length !== S.n))
    S.refCnt = new Uint8Array(S.n);
  // The per-rule table, indexed by the byte the wiring put on each synapse.
  // Row 0 is never read (0 means frozen and the loops skip it), row 1 is the checkpoint's own values, rows 2 up are declared rules.
  // Field order is part of the protocol and stated in ENGINE.md, because the workers are self-contained and cannot import it.
  //
  // When no table is sent, one is built here from the scalars above rather than leaving the loops to branch on whether a table exists, so a graph with no plasticity node reads row 1 instead of running different code that happens to agree.
  S.RS = 11;                              // aP aM wmax wdep trip het tin iEta cons consW consP
  if(m.ruleTable && m.ruleCount >= 2){
    S.ruleTable = m.ruleTable instanceof Float32Array
      ? m.ruleTable : new Float32Array(m.ruleTable);
    S.ruleCount = m.ruleCount;
  } else {
    S.ruleTable = new Float32Array(2*S.RS);
    S.ruleCount = 2;
  }
  S.trip = m.trip > 0 ? +m.trip : 0;
  const tauY = m.tauY > 0 ? +m.tauY : 114;
  S.decY = Math.exp(-1/tauY);
  S.het = m.het > 0 ? +m.het : 0;
  S.tin = m.tin > 0 ? +m.tin : 0;
  // the heterosynaptic gate is the slow trace relative to its value at the homeostatic target rate, cubed: competition engages when the neuron fires above its set point, and the gate is dimensionless so het keeps one scale across targets and trace time constants
  S.kRef = Math.max(1e-6, tauY*(m.iRho || S.iRho || 5)/1000);
  // Consolidation of the reference weight (Zenke, Agnes and Gerstner 2015, Nat Commun 6:6922, equation 16).
  // Without it wRef is a frozen snapshot of the weights at the moment plasticity switched on, so the heterosynaptic term is a permanent decay toward the initial state and opposes every learned change forever.
  //
  // In the published rule the reference follows the weight on a slow timescale through a double well, so a synapse driven above the midpoint consolidates to the upper state and one left below it decays to zero.
  // That bistability is what lets learned structure survive the competition that is meant to shape it.
  // Short-term plasticity (Tsodyks and Markram; Zenke et al. 2015 equations 9 and 10).
  // State is per presynaptic neuron rather than per synapse, so it costs two floats per cell: x is available resources, u is release probability.
  // Short-term depression gives the effective input-output relation the curvature that puts a stable fixed point at intermediate rates, which is what the source says lets an assembly sit at an elevated rate without running away.
  //
  // stpOrder 0, as published (Tsodyks, Pawelzik and Markram 1998 eqs 2.1 to 2.3; Zenke et al. 2015 eqs 9 and 10 and their Auryn code): at a spike the released fraction is u times x, then x depletes and u jumps, u += U(1 - u), so a rested synapse releases U of its weight. stpOrder 1 jumps u before releasing, so a rested synapse releases U(2 - U); scenes tuned on that order declare it. stpNorm 1 multiplies every release by the inverse of the rested release, so a rested synapse delivers exactly w.
  S.stp = m.stp > 0 ? 1 : 0;
  S.stpU = m.stpU > 0 ? +m.stpU : 0.2;
  S.stpOrder = (m.stpOrder|0) === 1 ? 1 : 0;
  S.stpScale = (m.stpNorm|0) === 1 ? 1/(S.stpOrder ? S.stpU*(2 - S.stpU) : S.stpU) : 1;
  S.stpTauD = m.stpTauD > 0 ? +m.stpTauD : 200;   // ms, depression recovery
  S.stpTauF = m.stpTauF > 0 ? +m.stpTauF : 600;   // ms, facilitation decay
  if(S.stp && (!S.stpX || S.stpX.length !== S.n)){
    S.stpX = new Float32Array(S.n).fill(1);
    S.stpR = new Float32Array(S.n).fill(S.stpU);
  }
  S.cons = m.cons > 0 ? +m.cons : 0;
  // The published upper fixed point is 0.5 against a wmax of 5, so the consolidated state sits at a tenth of maximum.
  // Importing 0.5 literally into tissue whose wmax is 60 puts it near the median weight, which drags every strong synapse down roughly twentyfold instead of consolidating it, so the default is the published ratio rather than the published number.
  // Any tissue whose weights are not distributed like the reference model's still needs this calibrated against its own distribution.
  S.consW = m.consW > 0 ? +m.consW : 0.1*S.wmax;   // upper stable fixed point
  S.consP = m.consP > 0 ? +m.consP : 10;    // depth of the well
  // Commit: once a synapse's reference weight has climbed into the upper well, stop updating that synapse at all.
  // Consolidation on its own is a pull toward the reference whose strength is the heterosynaptic amplitude, so a synapse that has committed still moves under every subsequent pairing.
  // Commit takes what has been learned out of the plastic pool, so later learning has to use what is left.
  //
  // There is no new per-synapse state. wRef is already the bistable variable consolidate() drives to one of two fixed points, so "has committed" is "its reference reached the upper one", and the threshold is the upper fixed point of the rule that synapse belongs to.
  S.commit = (m.commit|0) === 1 ? 1 : 0;
  // The published rule fixes the consolidation timescale at twenty minutes; it is a parameter here because the answer depends on the curriculum: a synapse commits when its reference has been driven past the midpoint of the well and held there, so the same timescale means a different number of presentations for every lesson stream.
  S.tauCons = m.tauCons > 0 ? +m.tauCons : 1200000;   // 20 min, as published
  S.consStep = m.consStep > 0 ? +m.consStep : 1200;   // updated every 1.2 s
  if(S.consAcc === undefined) S.consAcc = 0;
  // Any rule wanting het or consolidation needs the reference weights, so the table decides this rather than the checkpoint's own two fields.
  // Row 1 is filled from the scalars here and not where they are first read, because the consolidation fields are assigned above and nowhere earlier; filled sooner it carries zeros for cons, consW and consP and this engine runs without consolidation while the GPU runs it.
  //
  // A tune that changes aP alone still moves the default rule, because this runs on every configure.
  // A declared rule keeps whatever it named and inherited the checkpoint's value for the rest at wiring time, which is the same rule the pair table follows.
  if(!(m.ruleTable && m.ruleCount >= 2)){
    const R = S.ruleTable, o = S.RS;
    R[o+0] = S.aP; R[o+1] = S.aM; R[o+2] = S.wmax; R[o+3] = S.wdep ? 1 : 0;
    R[o+4] = S.trip; R[o+5] = S.het; R[o+6] = S.tin; R[o+7] = S.iEta;
    R[o+8] = S.cons; R[o+9] = S.consW; R[o+10] = S.consP;
  }
  // any rule may switch these on, so the state they need follows the table
  let anyTrip = 0, anyHet = 0, anyCons = 0;
  for(let r = 1; r < S.ruleCount; r++){
    anyTrip = Math.max(anyTrip, S.ruleTable[r*S.RS + 4]);
    anyHet = Math.max(anyHet, S.ruleTable[r*S.RS + 5]);
    anyCons = Math.max(anyCons, S.ruleTable[r*S.RS + 8]);
  }
  S.anyHet = anyHet > 0; S.anyCons = anyCons > 0;
  if((anyTrip > 0 || anyHet > 0) && (!S.Kslow || S.Kslow.length !== S.n))
    S.Kslow = new Float32Array(S.n);
  if((S.anyHet || S.anyCons) && !S.wRef && S.w) S.wRef = S.w.slice();
  if(S.plast && !S.revSyn){                  // reverse index (NeMo-style RCM)
    const n = S.n, m2 = S.post.length;
    const inStart = new Int32Array(n+1);
    for(let s=0;s<m2;s++) inStart[S.post[s]+1]++;
    for(let i=0;i<n;i++) inStart[i+1] += inStart[i];
    const cur = new Int32Array(n),
          revSyn = new Int32Array(m2), revPre = new Int32Array(m2);
    for(let i=0;i<n;i++)
      for(let s=S.preStart[i]; s<S.preStart[i+1]; s++){
        const at = inStart[S.post[s]] + cur[S.post[s]]++;
        revSyn[at] = s; revPre[at] = i;
      }
    S.inStart = inStart; S.revSyn = revSyn; S.revPre = revPre;
    S.Kpre = new Float32Array(n); S.Kpost = new Float32Array(n);
    S.spkAcc = new Float32Array(n);          // spikes this scaling window
  }
  if(S.plast && S.rhoMode === 1 && !S.rhoI){
    S.calibrating = true; S.calStart = S.t;
    S.spkAcc.fill(0);
  } else if(!S.plast) S.calibrating = false;
  // Scaling's window opens when scaling turns on, and a pass follows every 1000 ms from there, as in the CUDA engine.
  // Otherwise the count runs on while scaling is off and the first pass after switching it on mid-run reads several seconds of spikes as one second's rate.
  // A run with scaling on from the start keeps its passes on whole seconds.
  const scaling = S.plast && S.scaleOn;
  if(scaling && !S.wasScaling && !S.calibrating){
    S.nextScale = S.t + 1000;
    if(S.spkAcc) S.spkAcc.fill(0);
  }
  S.wasScaling = scaling;
}

function finishCalibration(){
  const n = S.n, secs = S.calMs/1000;
  S.rhoI = new Float32Array(n);
  S.alphaI = new Float32Array(n);
  for(let i=0;i<n;i++){
    S.rhoI[i] = Math.min(30, Math.max(0.25, S.spkAcc[i]/secs));
    S.alphaI[i] = S.rhoI[i]*0.001*(S.tauSv + S.tauMv);
    S.spkAcc[i] = 0;
  }
  S.calibrating = false;
  S.nextScale = S.t + 1000;
}

function applyScaling(){
  const { n, inStart, revSyn, w, pmask, spkAcc, iRho, sEta, rhoI, ruleTable:RT, RS, wRef } = S;
  // a committed synapse has left the plastic pool, and scaling is part of that pool, so it keeps its weight under scaling as it does under STDP (MODEL.md 4, terms in combination)
  const commit = S.commit && wRef;
  for(let i=0;i<n;i++){
    const rho = rhoI ? rhoI[i] : iRho;
    const err = Math.max(-2, Math.min(2, (rho - spkAcc[i]) / rho));
    if(err){
      const f = 1 + sEta*err;
      for(let k=inStart[i], ke=inStart[i+1]; k<ke; k++){
        const s = revSyn[k];
        const rid = pmask ? (pmask[s] & 31) : 1;   // the rule is the low five bits; the channel the top three
        if(!rid) continue;
        if(commit && RT[rid*RS + 9] > 0 && wRef[s] >= RT[rid*RS + 9]*0.9) continue;
        // the synapse's own rule sets its ceiling, as in the other engines
        if(w[s] > 0) w[s] = Math.min(RT[rid*RS + 2], w[s]*f);
      }
    }
    spkAcc[i] = 0;
  }
}

// STDP bookkeeping when neuron i spikes: as presynaptic partner its outgoing synapses see the postsynaptic traces (depression for excitatory, the Vogels alpha term for inhibitory); as postsynaptic partner its incoming synapses see the presynaptic traces (potentiation / Vogels on-post term).
// Excitatory weights clamp to [1e-4, wmax], inhibitory to [-wmax, -1e-4]. wdep selects soft rather than hard weight bounds, on both signs.
//
// Excitatory: additive depression with hard bounds has no interior fixed point, so with aM above aP and uncorrelated firing every weight drifts to a bound and a column loses its recurrent excitation entirely (measured: mean excitatory weight 0.15 to 0.016 in 120 simulated seconds, with the network still looking healthy because the background current carries it).
// Depression proportional to the current weight (van Rossum, Bi and Turrigiano 2000; the mu = 1 end of Gutig et al. 2003) puts the fixed point at aP*<Kpre> = aM*w*<Kpost>, which is stable and unimodal.
//
// Inhibitory: the Vogels rule is a rate controller and does have a fixed point, at a postsynaptic rate of rho, but only if that rate is reachable.
// When it is not, the controller integrates without limit and every inhibitory weight walks into the clamp, and the population it governs falls silent for the rest of the run.
// Soft bounds scale each update by the room left in the direction it is heading, so an unreachable target saturates gracefully instead of welding the network shut.
function softBound(wv, delta, wmax){
  const a = wv < 0 ? -wv : wv;              // distance from zero
  const room = delta < 0 === wv < 0         // heading away from zero?
    ? (wmax - a)/wmax : a/wmax;
  return delta * (room > 0 ? room : 0);
}
function plastOnSpike(i){
  const { preStart, post, w, inStart, revSyn, revPre, pmask, alphaI,
          Kpre, Kpost, iAlpha, Kslow, wRef, ruleTable:RT, RS } = S;
  if(S.calibrating){                         // measure baselines, change nothing
    Kpre[i] += 1; Kpost[i] += 1;
    if(Kslow) Kslow[i] += 1;
    if(S.spkAcc) S.spkAcc[i] += 1;
    return;
  }
  // the slow trace is read before this spike is added: the triplet term potentiates by the postsynaptic history, not by the present spike
  const z = Kslow ? Kslow[i] : 0;
  // The heterosynaptic gate is dimensionless and shared, since it is built from this neuron's trace and its set point; only the amplitude het multiplying it is per rule, so the cube is computed once here.
  // The set point in the gate is the global one even under measured set points: the WebGPU incoming pass has no storage slot left for a per-neuron value, and the engines have to agree (MODEL.md 4, terms in combination).
  const g = z/S.kRef, gate = g*g*g;
  // The rule byte is read per synapse and the row is read from it.
  // When no plasticity node exists every plastic synapse carries 1 and every read lands on the same row: one global rule, arrived at by the same path rather than by a separate branch.
  const committed = S.commit && wRef
    ? (s, o) => RT[o+9] > 0 && wRef[s] >= RT[o+9]*0.9
    : null;
  for(let s=preStart[i], e=preStart[i+1]; s<e; s++){
    const rid = pmask ? (pmask[s] & 31) : 1;   // the rule is the low five bits; the channel the top three
    if(!rid) continue;                       // 0 is frozen
    const o = rid*RS;
    if(committed && committed(s, o)) continue;
    const wmax = RT[o+2], wdep = RT[o+3] !== 0;
    const wj = w[s];
    if(wj > 0) w[s] = Math.min(wmax, Math.max(1e-4,
      wj - RT[o+1]*Kpost[post[s]]*(wdep ? wj : 1) + RT[o+6]));
    else {
      const al = alphaI ? alphaI[post[s]] : iAlpha;
      let d = -RT[o+7]*(Kpost[post[s]] - al);
      if(wdep) d = softBound(wj, d, wmax);
      w[s] = Math.max(-wmax, Math.min(-1e-4, wj + d));
    }
  }
  for(let k=inStart[i], ke=inStart[i+1]; k<ke; k++){
    const s = revSyn[k];
    const rid = pmask ? (pmask[s] & 31) : 1;   // the rule is the low five bits; the channel the top three
    if(!rid) continue;
    const o = rid*RS;
    if(committed && committed(s, o)) continue;
    const wmax = RT[o+2], wdep = RT[o+3] !== 0, het = RT[o+5];
    const wp = w[s], kp = Kpre[revPre[k]];
    if(wp > 0){
      let nw = wp + (RT[o+0] + RT[o+4]*z)*kp;
      if(het > 0) nw -= Math.min(1, het*gate)*(wp - wRef[s]);
      w[s] = Math.min(wmax, Math.max(1e-4, nw));
    }
    else {
      let d = -RT[o+7]*kp;
      if(wdep) d = softBound(wp, d, wmax);
      w[s] = Math.max(-wmax, wp + d);
    }
  }
  Kpre[i] += 1; Kpost[i] += 1;
  if(Kslow) Kslow[i] += 1;
  if(S.spkAcc) S.spkAcc[i] += 1;
}

// Tsodyks-Markram release at a presynaptic spike, normalized so a rested synapse delivers its full weight.
// Returns the scale factor for every outgoing synapse of neuron i.
function stpRelease(i){
  const X = S.stpX, R = S.stpR, U = S.stpU;
  if(S.stpOrder) R[i] = R[i] + U*(1 - R[i]);   // stpOrder 1: facilitation before release
  const rel = R[i]*X[i];
  X[i] = X[i] - rel;                       // depletion
  if(!S.stpOrder) R[i] = R[i] + U*(1 - R[i]);  // as published: facilitation after release
  return rel*S.stpScale;
}

// Per-millisecond recovery of both state variables, exact exponential rather than Euler so the time constants hold at any step size.
function stpStep(){
  const X = S.stpX, R = S.stpR, U = S.stpU, n = S.n;
  const dx = Math.exp(-1/S.stpTauD), dr = Math.exp(-1/S.stpTauF);
  for(let i=0;i<n;i++){
    X[i] = 1 - (1 - X[i])*dx;
    R[i] = U + (R[i] - U)*dr;
  }
}

// Zenke et al. 2015 equation 16: the reference weight follows the weight through a double well whose lower fixed point is zero and whose upper one is consW, with the midpoint at consW/2 unstable.
// Excitatory plastic synapses only; inhibitory weights are governed by the Vogels rule and have no reference.
// Run every consStep ms rather than every step, which is what the published implementation does for the same reason.
function consolidate(){
  const { w, wRef, pmask, tauCons, consStep, ruleTable:RT, RS } = S;
  if(!w || !wRef) return;
  // cons gates rather than scales, here and in both other engines.
  // It reads as a 0 to 1 amount, which is misleading, but scaling the rate in one engine alone would make the reference disagree with the GPU and CUDA engines for every value between 0 and 1, which is the failure ENGINE.md exists to prevent.
  // The rate is the timescale beside it.
  const rate = consStep/tauCons;
  for(let s=0, m2=w.length; s<m2; s++){
    const rid = pmask ? (pmask[s] & 31) : 1;   // the rule is the low five bits; the channel the top three
    if(!rid) continue;
    const o = rid*RS, consW = RT[o+9];
    // a rule that does not consolidate leaves its references where they are, which is what keeps het pulling toward the weights as they were
    if(RT[o+8] <= 0) continue;
    const wj = w[s];
    if(wj <= 0) continue;
    const r = wRef[s], half = consW*0.5;
    // Clamped to the two fixed points of the well it lives in.
    // This is a forward Euler step over a cubic, so a short timescale makes it unstable: at a twenty second constant with the published well depth the reference overshoots, the cubic grows with the overshoot, and the weights reach NaN inside ten simulated seconds (measured 2026-09-03).
    // A reference outside [0, consW] has no meaning in the model anyway.
    const nr = r + rate*(wj - r - RT[o+10]*r*(half - r)*(consW - r));
    wRef[s] = nr < 0 ? 0 : (nr > consW ? consW : nr);
  }
}

function scaleOf(p, t){
  const dt = t - p.t0;
  if(dt < 0) return 0;
  if(p.duration > 0 && dt >= p.duration) return 0;
  if(p.mode === 1) return (dt % p.period) < p.width ? 1 : 0;   // pulse train
  if(p.mode === 2) return Math.min(1, dt / p.period);          // ramp up over period
  return 1;
}

function updateAdd(P, arr){                  // per-neuron sum of amp*scale(t)
  if(!P.length) return null;
  let changed = !arr;
  for(const p of P){
    const s = scaleOf(p, S.t);
    if(s !== p._s){ p._s = s; changed = true; }
  }
  if(!changed) return arr;
  if(!arr) arr = new Float32Array(S.n);
  arr.fill(0);
  for(const p of P){
    if(!p._s) continue;
    const k = p.amp * p._s, idx = p.idx, g = p.gain;
    if(g) for(let q=0;q<idx.length;q++) arr[idx[q]] += k*g[q];
    else for(let q=0;q<idx.length;q++) arr[idx[q]] += k;
  }
  return arr;
}
function updateStim(){
  S.stim = updateAdd(S.protos, S.stim);
  S.namp = updateAdd(S.nprotos, S.namp);
}

// rebuild the per-neuron external drive from the input maps' current gains; runs only when an inputFrame arrived (roughly display rate, not per ms)
function rebuildExt(){
  S.extDirty = false;
  if(!S.inputs || !S.inputs.length){ S.ext = null; return; }
  if(!S.ext) S.ext = new Float32Array(S.n);
  S.ext.fill(0);
  for(const im of S.inputs){
    const { chStart, chIdx, chW, gains, amp } = im;
    const NC = Math.min(gains.length, chStart.length - 1);   // stale frames after a retune
    for(let c=0;c<NC;c++){
      const g = gains[c] * amp;
      if(!g) continue;
      for(let k=chStart[c];k<chStart[c+1];k++) S.ext[chIdx[k]] += g * chW[k];
    }
  }
}

onmessage = (e) => {
  const m = e.data;
  // the reference implements every term and answers every optional query (ENGINE.md section 2)
  if(m.cmd === 'hello'){ postMessage(helloReply('reference', TERM_NAMES, OPTIONAL)); return; }
  if(m.cmd === 'init'){
    { const bad = checkProtocol(m, 'simworker'); if(bad){ postMessage({ cmd:'error', message:bad }); return; } }
    const n = m.count, T = m.types;
    const a = new Float32Array(n), b = new Float32Array(n),
          c = new Float32Array(n), d = new Float32Array(n);
    // The membrane (MODEL.md 1): Izhikevich 2007 on every type's f7 row, with the capacitance, the slope, the rest, the threshold and the peak as arrays of their own.
    // The 2003 presets are rows of this form (form.js), and an init that names the 2003 form is refused rather than run as something else.
    if('form' in m && (m.form|0) !== 1){
      postMessage({ cmd:'error', message:'simworker: the 2003 membrane form was retired; the scene needs migrating (scene format 8)' }); return; }
    const C7 = new Float32Array(n), k7 = new Float32Array(n), vr7 = new Float32Array(n),
          vt7 = new Float32Array(n), vp7 = new Float32Array(n);
    for(let i=0;i<n;i++){ const t = T[m.ntype[i]] && T[m.ntype[i]].f7;
      if(!t){ postMessage({ cmd:'error', message:'simworker: every neuron type needs a 2007 row (f7)' }); return; }
      a[i]=t.a; b[i]=t.b; c[i]=t.c; d[i]=t.d;
      C7[i]=t.C; k7[i]=t.k; vr7[i]=t.vr; vt7[i]=t.vt; vp7[i]=t.vpeak; }
    // The initial membrane state arrives from the host (initState in rand.js), the same bytes every engine starts from.
    // An init without it is refused rather than seeded here, so no engine can drift back to a generator of its own.
    if(!m.v0 || !m.u0 || m.v0.length !== n || m.u0.length !== n){
      postMessage({ cmd:'error', message:'simworker: init needs v0 and u0 (rand.js initState)' });
      return;
    }
    const v = Float32Array.from(m.v0), u = Float32Array.from(m.u0);
    const ROWS = 18;
    // graded rows (MODEL.md 1): a byte per cell, the release threshold and slope; their outgoing synapses are frozen for plasticity
    let G = null;
    try { G = gradedArrays(m.ntype, T); }
    catch(e){ postMessage({ cmd:'error', message:'simworker: ' + e.message }); return; }
    S = { n, a, b, c, d, v, u, C7, k7, vr7, vt7, vp7,
      grd:G ? G.grd : null, gthr:G ? G.thr : null, gslp:G ? G.slope : null,
      seed: ((m.seed|0) || 1) >>> 0,   // noise draws and the initial jitter
      preStart:m.preStart, post:m.post, w:m.w, delay:m.delay,
      pmask:freezeGraded(m.pmask || null, m.preStart, G && G.grd, m.w.length),
      ROWS, ring:new Float32Array(ROWS*n), ring2:null, cur:0, t:0,
      watch:-1, watchV:0, sendV:streams.sendV, sendT:streams.sendT };
    // Feedback inhibition (wirePools in nodes.js): the pool each cell is in, each pool's kick per spike, and the pool spike counts of the last and the current millisecond.
    // Index 0 is no pool and delivers 0.
    if(m.pool && m.pool.length){
      if(m.pool.length !== n || !m.poolK || m.poolK.length < 2){
        postMessage({ cmd:'error', message:'simworker: init pool arrays do not match the network' });
        return;
      }
      S.pool = m.pool instanceof Uint16Array ? m.pool : Uint16Array.from(m.pool);
      S.poolK = Float32Array.from(m.poolK);
      S.poolCnt = new Uint32Array(S.poolK.length);
      S.poolNext = new Uint32Array(S.poolK.length);
      S.poolIn = new Float32Array(S.poolK.length);
    }
    try { configure(m); } catch(e){ postMessage({ cmd:'error', message:'simworker: ' + e.message }); S = null; return; }
  }
  else if(m.cmd === 'watch'){ if(S) S.watch = m.idx; }
  else if(m.cmd === 'sendV'){ streams.sendV = !!m.on; if(S) S.sendV = streams.sendV; }
  // first-spike step within each tick (lat, 255 for silent) with each state reply, as the CUDA engine streams it
  else if(m.cmd === 'sendT'){ streams.sendT = !!m.on; if(S) S.sendT = streams.sendT; }
  else if(m.cmd === 'inputFrame'){ if(S && S.inputs && S.inputs[m.mi]){
    S.inputs[m.mi].gains = m.gains; S.extDirty = true; } }
  else if(m.cmd === 'tune'){ if(S){ try { configure(m); } catch(e){ postMessage({ cmd:'error', message:'simworker: ' + e.message }); } } }
  else if(m.cmd === 'getWeights'){          // learned-state readback (checkpoints, analysis)
    if(!S) return;
    const wc = S.w.slice();
    // measured set points travel with the weights when they exist: without them an unreachable homeostatic target can only be inferred, which leaves a collapse undiagnosed for a whole run
    const rc = S.rhoI ? S.rhoI.slice() : null;
    postMessage({ cmd:'weights', w:wc, t:S.t, rho:rc },
      rc ? [wc.buffer, rc.buffer] : [wc.buffer]);
  }
  else if(m.cmd === 'query'){               // connection lookup for the viewer
    if(!S) return;
    // dir 'out' skips the reverse scan below, which is the whole cost of a query here: it walks every synapse in the network looking for the ones that land on this neuron.
    // The pathway highlight only ever reads the outgoing list, and at 151.7M synapses the scan it does not need is nineteen tick slices of work competing with the simulation.
    S.q = { idx:m.idx, cap:m.cap || 2000, inn:[], pos:0, inTotal:0,   // 2000 as on the other engines
      outOnly: m.dir === 'out' };
    if(S.q.outOnly) S.q.pos = S.post.length;
    scanQuery(40e6);                        // small nets answer immediately;
  }                                         // big ones continue in tick slices
  else if(m.cmd === 'tick'){
    if(!S) return;
    if(S.extDirty) rebuildExt();
    const fired = new Uint8Array(S.n);
    const lat = S.sendT ? new Uint8Array(S.n).fill(255) : null;
    const vtrace = new Float32Array(m.steps);
    let spikes = 0;
    for(let s=0;s<m.steps;s++){
      updateStim();
      spikes += (S.exp ? stepExp : step)(fired, lat, s);
      vtrace[s] = S.watchV;
      S.t++;
      if(S.plast && S.calibrating && S.t >= S.calStart + S.calMs) finishCalibration();
      else if(S.plast && !S.calibrating && S.scaleOn && S.spkAcc && S.t >= S.nextScale){
        applyScaling();
        S.nextScale += 1000;
      }
    }
    if(S.q) scanQuery(6e6);
    const msg = { cmd:'state', fired, spikes, steps:m.steps, vtrace };
    const move = [fired.buffer, vtrace.buffer];
    if(S.sendV){ msg.v = S.v.slice(); move.push(msg.v.buffer); }
    if(lat){ msg.lat = lat; move.push(lat.buffer); }
    postMessage(msg, move);
  }
};

// chunked reverse-connectivity scan: never stalls the sim, caps the lines it returns but always reports exact totals
function scanQuery(budget){
  const q = S.q, { preStart, post, n } = S;
  const end = q.outOnly ? q.pos : Math.min(post.length, q.pos + budget);
  for(let s=q.pos; s<end; s++)
    if(post[s] === q.idx){
      q.inTotal++;
      if(q.inn.length < q.cap) q.inn.push(s);
    }
  q.pos = end;
  if(q.pos < post.length) return;
  const inn = new Uint32Array(q.inn.length);
  for(let k=0;k<q.inn.length;k++){          // synapse index -> presynaptic owner
    let lo = 0, hi = n; const s = q.inn[k];
    while(lo < hi-1){ const mid = (lo+hi)>>1; if(preStart[mid] <= s) lo = mid; else hi = mid; }
    inn[k] = lo;
  }
  const o0 = preStart[q.idx], o1 = preStart[q.idx+1];
  const outN = Math.min(q.cap, o1-o0);
  const out = new Uint32Array(outN);
  for(let k=0;k<outN;k++) out[k] = post[o0+k];
  postMessage({ cmd:'queryResult', idx:q.idx, out, inn, outTotal:o1-o0, inTotal:q.inTotal },
    [out.buffer, inn.buffer]);
  S.q = null;
}

// Per-millisecond noise is noiseDraw in rand.js: counter based, a pure function of (neuron, t, seed), shared with the WebGPU and CUDA copies.

// Feedback inhibition: what each pool delivers this millisecond is its kick per spike times the spikes it fired last millisecond, on the inhibitory path (the input current in kick mode, the I conductance in exp mode).
// The counts of this millisecond become last millisecond's at the end of the step.
// The same two tables, the same parity, in the WebGPU and CUDA kernels.
function poolDrive(){
  const { poolK, poolCnt, poolIn } = S;
  for(let g=0;g<poolK.length;g++) poolIn[g] = poolK[g]*poolCnt[g];
  return poolIn;
}
function poolSwap(){
  const t = S.poolCnt; S.poolCnt = S.poolNext; S.poolNext = t; t.fill(0);
}

function step(fired, lat, sIdx){
  const { n, v, u, a, b, c, d, ring, cur, ROWS, bias, stim, namp, ext,
          preStart, post, w, delay, pool, vmin, C7, k7, vr7, vt7, vp7, grd, gthr, gslp } = S;
  const poolIn = pool ? poolDrive() : null;
  const row = cur*n;
  let count = 0, watchSpiked = false;
  for(let i=0;i<n;i++){
    const I = ring[row+i] + bias[i] + (stim ? stim[i] : 0) + (ext ? ext[i] : 0)
      + (namp ? namp[i]*noiseDraw(i, S.t, S.seed) : 0) + (pool ? poolIn[pool[i]] : 0);
    let vi = v[i], ui = u[i];
    if(S.refCnt && S.refCnt[i] > 0){
      // absolute refractoriness: the membrane is held at reset and the threshold cannot be reached, whatever the drive
      S.refCnt[i]--;
      ui += a[i]*(b[i]*(vi - vr7[i]) - ui);
      u[i] = ui;
      continue;
    }
      // Izhikevich 2007: C dv = k (v - vr)(v - vt) - u + I, the same two half steps and the same floor; u is measured from the rest
      const Ci = C7[i], ki = k7[i], vri = vr7[i], vti = vt7[i];
      if(grd && grd[i]){
        // a graded cell's membrane is passive: the leak is the row's slope at rest, k (vt - vr), and the potential is capped at vpeak
        const gl = ki*(vti - vri);
        vi += 0.5*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
        vi += 0.5*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
        if(vi > vp7[i]) vi = vp7[i];
      } else {
      vi += 0.5*(ki*(vi - vri)*(vi - vti) - ui + I)/Ci;
      if(vi < vmin) vi = vmin;
      vi += 0.5*(ki*(vi - vri)*(vi - vti) - ui + I)/Ci;
      if(vi < vmin) vi = vmin;
      }
      ui += a[i]*(b[i]*(vi - vri) - ui);
    if(grd && grd[i]){
      // a graded cell: no spike, no reset, no refractory period; every millisecond it delivers a spike of amplitude r through its synapses
      const r = Math.min(1, Math.max(0, (vi - gthr[i])/gslp[i]));
      if(r > 0) for(let s=preStart[i], e1=preStart[i+1]; s<e1; s++)
        ring[((cur+delay[s])%ROWS)*n + post[s]] += w[s]*r;
      v[i] = vi; u[i] = ui;
      continue;
    }
    if(vi >= vp7[i]){
      vi = c[i]; ui += d[i]; count++;
      // count, not a flag: a flag capped every rate this readout could report at 1000/steps Hz. lat is the first spike of the tick in milliseconds, 255 for silent, which is the only grain STDP works on.
      if(fired[i] < 255) fired[i]++;
      if(lat && lat[i] === 255) lat[i] = sIdx < 254 ? sIdx : 254;
      if(S.refCnt) S.refCnt[i] = S.refrac;
      if(pool) S.poolNext[pool[i]]++;
      if(i===S.watch) watchSpiked = true;
      const sf = S.stp ? stpRelease(i) : 1;
      for(let s=preStart[i], e1=preStart[i+1]; s<e1; s++)
        ring[((cur+delay[s])%ROWS)*n + post[s]] += w[s]*sf;
      if(S.plast) plastOnSpike(i);
    }
    v[i] = vi; u[i] = ui;
  }
  if(S.plast){
    const { Kpre, Kpost, decS, decM } = S;
    for(let i=0;i<n;i++){ Kpre[i] *= decS; Kpost[i] *= decM; }
    if(S.Kslow){ const Ks = S.Kslow, dY = S.decY;
      for(let i=0;i<n;i++) Ks[i] *= dY; }
    if(S.anyCons && ++S.consAcc >= S.consStep){ S.consAcc = 0; consolidate(); }
  }
  if(S.stp) stpStep();
  ring.fill(0, row, row+n);
  if(pool) poolSwap();
  S.cur = (cur+1)%ROWS;
  S.watchV = S.watch>=0 ? (watchSpiked ? vp7[S.watch] : v[S.watch]) : 0;
  return count;
}

function stepExp(fired, lat, sIdx){
  const { n, v, u, a, b, c, d, ring, ring2, gE, gI, decE, decI, itauE, itauI,
          cur, ROWS, bias, stim, namp, ext, preStart, post, w, delay, pool, vmin,
          C7, k7, vr7, vt7, vp7, grd, gthr, gslp, KX, gX, ringX, chanDec, chanItau, pmask, cond, eRevE, eRevI, chanErev } = S;
  const poolIn = pool ? poolDrive() : null;
  const row = cur*n;
  let count = 0, watchSpiked = false;
  // a synapse's ring: by the channel in its plasticity byte, else by sign
  const ringOf = (s, wt) => { const ch = pmask ? pmask[s] >> 5 : 0; return ch >= 2 ? null : (wt > 0 ? ring : ring2); };
  for(let i=0;i<n;i++){
    const ge = gE[i]*decE + ring[row+i];
    const gi = gI[i]*decI + ring2[row+i] + (pool ? poolIn[pool[i]] : 0);
    const v0 = v[i];
    // current-based: g times the scale; conductance-based: |g| (E - v)
    let I = (cond ? Math.abs(ge)*(eRevE - v0) + Math.abs(gi)*(eRevI - v0) : ge*itauE + gi*itauI)
      + bias[i] + (stim ? stim[i] : 0) + (ext ? ext[i] : 0)
      + (namp ? namp[i]*noiseDraw(i, S.t, S.seed) : 0);
    if(KX) for(let x=0;x<KX;x++){
      const at = x*n + i, g = gX[at]*chanDec[x] + ringX[(x*ROWS + cur)*n + i];
      gX[at] = g; I += cond ? Math.abs(g)*(chanErev[x] - v0) : g*chanItau[x];
    }
    let vi = v[i], ui = u[i];
    if(S.refCnt && S.refCnt[i] > 0){
      S.refCnt[i]--;
      gE[i] = ge; gI[i] = gi;
      ui += a[i]*(b[i]*(vi - vr7[i]) - ui);
      u[i] = ui;
      continue;
    }
      const Ci = C7[i], ki = k7[i], vri = vr7[i], vti = vt7[i];
      if(grd && grd[i]){
        // a graded cell's membrane is passive: the leak is the row's slope at rest, k (vt - vr), and the potential is capped at vpeak
        const gl = ki*(vti - vri);
        vi += 0.5*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
        vi += 0.5*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
        if(vi > vp7[i]) vi = vp7[i];
      } else {
      vi += 0.5*(ki*(vi - vri)*(vi - vti) - ui + I)/Ci;
      if(vi < vmin) vi = vmin;
      vi += 0.5*(ki*(vi - vri)*(vi - vti) - ui + I)/Ci;
      if(vi < vmin) vi = vmin;
      }
      ui += a[i]*(b[i]*(vi - vri) - ui);
    if(grd && grd[i]){
      const r = Math.min(1, Math.max(0, (vi - gthr[i])/gslp[i]));
      if(r > 0) for(let s=preStart[i], e1=preStart[i+1]; s<e1; s++){
        const wt = w[s]*r, slot = (cur+delay[s])%ROWS, at = slot*n + post[s];
        const rg = ringOf(s, wt);
        if(rg) rg[at] += wt; else ringX[(((pmask[s] >> 5) - 2)*ROWS + slot)*n + post[s]] += wt;
      }
      v[i] = vi; u[i] = ui; gE[i] = ge; gI[i] = gi;
      continue;
    }
    if(vi >= vp7[i]){
      vi = c[i]; ui += d[i]; count++;
      // count, not a flag: a flag capped every rate this readout could report at 1000/steps Hz. lat is the first spike of the tick in milliseconds, 255 for silent, which is the only grain STDP works on.
      if(fired[i] < 255) fired[i]++;
      if(lat && lat[i] === 255) lat[i] = sIdx < 254 ? sIdx : 254;
      if(S.refCnt) S.refCnt[i] = S.refrac;
      if(pool) S.poolNext[pool[i]]++;
      if(i===S.watch) watchSpiked = true;
      const sf = S.stp ? stpRelease(i) : 1;
      for(let s=preStart[i], e1=preStart[i+1]; s<e1; s++){
        const wt = w[s]*sf, slot = (cur+delay[s])%ROWS, at = slot*n + post[s];
        const rg = ringOf(s, wt);
        if(rg) rg[at] += wt; else ringX[(((pmask[s] >> 5) - 2)*ROWS + slot)*n + post[s]] += wt;
      }
      if(S.plast) plastOnSpike(i);
    }
    v[i] = vi; u[i] = ui; gE[i] = ge; gI[i] = gi;
  }
  if(S.plast){
    const { Kpre, Kpost, decS, decM } = S;
    for(let i=0;i<n;i++){ Kpre[i] *= decS; Kpost[i] *= decM; }
    if(S.Kslow){ const Ks = S.Kslow, dY = S.decY;
      for(let i=0;i<n;i++) Ks[i] *= dY; }
    if(S.anyCons && ++S.consAcc >= S.consStep){ S.consAcc = 0; consolidate(); }
  }
  if(S.stp) stpStep();
  ring.fill(0, row, row+n); ring2.fill(0, row, row+n);
  if(KX) for(let x=0;x<KX;x++) ringX.fill(0, (x*ROWS + cur)*n, (x*ROWS + cur)*n + n);
  if(pool) poolSwap();
  S.cur = (cur+1)%ROWS;
  S.watchV = S.watch>=0 ? (watchSpiked ? vp7[S.watch] : v[S.watch]) : 0;
  return count;
}

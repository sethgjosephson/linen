// Izhikevich network simulation. dt = 1ms (two 0.5ms substeps for v).
// Synapse modes: kick (instant charge, one signed ring), exp (separate E/I rings feeding per-neuron decaying accumulators, scaled by the psc convention: w is the peak current, or under psc 1 the total charge), and conductance (the exp decay, each channel driving g (E - v); MODEL.md 2).
// Stimulus protocols (pulse train / ramp) are evaluated here per sim-ms so editing them never rebuilds the network.
import { gradedArrays, freezeGraded } from './nodes.js';
import { checkProtocol, helloReply, TERM_NAMES, OPTIONAL } from './protocol.js';
import { noiseDraw, poissonDraw, NOISE_SHARED } from './rand.js';
import { foldNoise } from './noisefold.js';

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
  // the magnesium block per channel, mM; 0 is no gate (MODEL.md 2)
  S.chanMg = Float32Array.from({ length:extra.length }, (_, k) => { const mg = m.chanMg ? +m.chanMg[k] : (S.chanMg ? S.chanMg[k] : 0); return mg > 0 ? mg : 0; });
  S.chanDec = Float32Array.from(extra, t => Math.exp(-1/(t > 0 ? t : 3)));   // chanDec: decX is short-term plasticity's
  S.chanItau = Float32Array.from(S.chanDec, (d, k) => itauOf(S.psc, d, extra[k] > 0 ? extra[k] : 3));
  if(S.KX && (!S.gX || S.gX.length !== S.KX*S.n)){
    S.gX = new Float32Array(S.KX*S.n); S.ringX = new Float32Array(S.KX*S.ROWS*S.n);
  }
  S.protos = []; S.nprotos = [];             // DC (pulse/ramp) vs noise-amplitude protocols
  for(const p of (m.protocols || []))
    (p.mode === 3 ? S.nprotos : S.protos).push({ ...p, _s: -1 });
  S.stim = null; S.namp = null;
  S.noise = foldNoise(S.nprotos, S.n);         // a noise stimulus's memory and shared part per cell (src/noisefold.js), null for the white draw; conflicting stimuli are refused here
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
  // The global factor on the inhibitory rule (Zenke et al. 2015, their GlobalPFConnection; MODEL.md 4): with iGlobal 1 every inhibitory update under a rule is the rule's eta times (the population rate over iRho minus one), the rate the low-passed spike count of the cells the rule inhibits over iTauG seconds, and the presynaptic term is the trace plus one.
  // The factor is read as the step before left it, in every engine.
  S.iGlobal = (m.iGlobal|0) === 1 ? 1 : 0;
  const tauG = m.iTauG > 0 ? +m.iTauG*1000 : 10000;
  S.decG = Math.exp(-1/tauG); S.tauGs = tauG/1000;
  if(S.gN) S.gInv = Math.fround(1/(S.gN*S.iRho));
  // the reversal the conductance adaptation draws toward (a row's adaptCond; MODEL.md 1)
  S.eRevAd = Number.isFinite(+m.eRevI) ? +m.eRevI : -70;
  // homeostatic synaptic scaling (Turrigiano 2008; van Rossum 2000 multiplicative form): once per simulated second, each neuron's plastic excitatory inputs scale a small step toward the target rate
  S.scaleOn = (m.scale|0) === 1;
  S.sEta = m.sEta > 0 ? m.sEta : 0.001;
  // rate set points: fixed (every neuron targets iRho) or measured (run the first calS seconds with the rules off, record each neuron's own baseline rate, and lock those as personal set points; Hengen & Turrigiano: firing-rate set points are individual, and a uniform target drives heterogeneous circuits into synchronous bursting)
  S.rhoMode = (m.rhoMode|0) === 1 ? 1 : 0;
  S.calMs = (m.calS > 0 ? m.calS : 10) * 1000;
  S.tauSv = tauS; S.tauMv = tauM;
  // The sliding threshold (MODEL.md 4, the P11Connection of Zenke, Agnes and Gerstner 2015): excitatory depression is scaled by the square of the cell's slow rate estimate over the target rate.
  // Khom is the slow trace (one per spike, decaying with slideTau), so the estimate in Hz is Khom times 1000 over the time constant in ms, and slideK folds the target in: the multiplier is (Khom slideK)^2.
  // The trace runs whenever plasticity does, so a tune that switches the threshold on reads the rate the cell has shown; slideK is zero when it is off.
  const slideTau = m.slideTau > 0 ? +m.slideTau : 600000;
  S.decH = Math.exp(-1/slideTau);
  S.slideK = (m.slide|0) === 1 && m.slideRate > 0 ? 1000/(slideTau*m.slideRate) : 0;
  S.khomInit = (m.slideInit > 0 ? +m.slideInit : 0)*slideTau/1000;
  // Where the pair rule reads its traces (MODEL.md 4).
  // 0: at the soma, both passes when the cell fires.
  // 1: at the synapse, where a presynaptic spike counts when it arrives, d steps after it was fired: potentiation at a post spike reads the pre trace as it was d steps ago, e^(d/tau+) (Kpre - the spikes of the last d - 1 steps), and depression waits d steps and reads Kpost then.
  // The dendritic part of the delay is 0, the NEST convention, so a post spike reaches every synapse of its cell at once.
  S.stdpAt = (m.stdpAt|0) === 1 ? 1 : 0;
  // The membrane step (MODEL.md 1): 0 two forward Euler half steps; 1 two exponential half steps, each the exact solution of the membrane linearized at the start of the half step, so a cell whose conductances hold over the step is integrated exactly however short its effective time constant.
  // It applies to rows whose membrane is linear over the spiking range (linRow: vt more than 100 mV above vpeak, the leaky integrate-and-fire rows); under 1 a quadratic row keeps the Euler half steps, since the linearization cannot follow its upswing (a fast-spiking 2007 row fired 34 times where a 1 microsecond clock fires 259).
  // 2 follows the equations: the linear rows as under 1, and a quadratic row solved exactly over each half step (quadMs).
  S.integ = (m.integ|0) === 2 ? 2 : (m.integ|0) === 1 ? 1 : 0;
  // The third factor (MODEL.md 4): a gain on the Hebbian terms of the pair rule (potentiation, depression, the triplet and transmitter terms), 1 unless the sender says otherwise.
  // The homeostatic terms (the inhibitory rule, scaling, the heterosynaptic pull, consolidation) run whatever it is: a neuromodulatory signal gates what is learned, not what keeps the tissue in its range.
  S.gain = m.plastGain === undefined ? 1 : Math.max(0, +m.plastGain || 0);
  // Neuromodulation (MODEL.md 4): one modulator level per cell, M, decaying with modTau and fed by time-course sources (mods, evaluated per step like the stimulus protocols) and by the spikes of releasing populations (modGrp, modTgt, modK, resolved at wiring like the pools).
  // M(t) = M(t-1) e^(-1/modTau) + the sources' increment at t + K times the releasing group's spikes at t-1; everything at step t reads M(t-1), the field as the step before left it.
  // The level a constant source holds is its amplitude, so the increment per step is amp (1 - dec); a release K is per spike and scaled the same way.
  // What reads the field: modPlast 1 multiplies the Hebbian terms at a synapse by modBase + M of the postsynaptic cell; modPlast 2 writes them to an eligibility trace per synapse instead, paid into the weight by modBase + M every modStep ms (Izhikevich 2007, Cereb Cortex 17:2443, his sweep every 10 ms); modDrive scales the synaptic current by 1 + modDrive M; modCur adds modCur M to the current.
  const modTau = m.modTau > 0 ? +m.modTau : 200;
  S.decMod = Math.fround(Math.exp(-1/modTau));
  S.modPlast = (m.modPlast|0) === 2 ? 2 : (m.modPlast|0) === 1 ? 1 : 0;
  S.modBase = m.modBase === undefined ? 1 : (+m.modBase || 0);
  S.modStep = m.modStep > 0 ? Math.max(1, Math.round(+m.modStep)) : 10;
  S.decElig = Math.fround(Math.exp(-S.modStep/(m.eligTau > 0 ? +m.eligTau : 1000)));
  // what a sweep pays: the trace times (1 - its decay per sweep), so over its life a trace pays its whole amount into the weight at a base plus level of 1, and the level means in the eligibility form what it means in the gain form
  S.eligPay = Math.fround(1 - S.decElig);
  S.modDrive = +m.modDrive || 0; S.modCur = +m.modCur || 0;
  S.modSrc = (m.mods || []).map(p => ({ ...p, amp:Math.fround((+p.amp || 0)*(1 - S.decMod)), gain:null, _s:-1 }));
  // the release arrays are fixed at init like the pools; a tune without them keeps those in force
  if(m.modGrp && m.modGrp.length === S.n && m.modTgt && m.modK){
    S.modGrp = m.modGrp instanceof Uint8Array ? m.modGrp : Uint8Array.from(m.modGrp);
    S.modTgt = m.modTgt instanceof Uint16Array ? m.modTgt : Uint16Array.from(m.modTgt);
    S.modKraw = Float32Array.from(m.modK);
  }
  S.modG = S.modKraw ? S.modKraw.length : 1;
  S.modK = S.modKraw ? Float32Array.from(S.modKraw, k => Math.fround(k*(1 - S.decMod))) : null;
  S.modOn = S.modSrc.length > 0 || S.modG > 1 || S.modPlast > 0 || S.modDrive !== 0 || S.modCur !== 0;
  if(S.modOn && !S.mod){
    S.mod = new Float32Array(S.n); S.modNew = new Float32Array(S.n);
    S.modAcc = 0;
  }
  if(S.modOn && (!S.modCnt || S.modCnt.length !== S.modG)){ S.modCnt = new Uint32Array(S.modG); S.modNext = new Uint32Array(S.modG); }
  S.modStim = null;
  // The weight dependence of soft depression (MODEL.md 4): depression scales with w^mu, 1 the proportional form (van Rossum, Bi and Turrigiano 2000), 0 the additive form, between the interpolation of Gutig et al. 2003 under which strong synapses are depressed less than in proportion and a competitive, stable weight distribution exists.
  S.wmu = m.wmu === undefined ? 1 : Math.max(0, Math.min(1, +m.wmu));
  S.ekS = Float64Array.from({ length:17 }, (_, k) => Math.exp(-k/tauS));   // what a spike k steps ago contributes to Kpre
  S.exdS = Float64Array.from({ length:17 }, (_, d) => Math.exp(d/tauS));   // the trace d steps ago from the trace now, once those spikes are taken out
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
  // Where the consolidation reference starts (MODEL.md 4): 0 the wired weight brought inside the well, 1 the lower fixed point (zero, where Zenke, Agnes and Gerstner 2015 start it), 2 the consolidated weight.
  S.refInit = m.refInit === undefined ? 0 : (m.refInit|0);
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
  S.stpOrder = (m.stpOrder|0) === 1 ? 1 : 0;
  S.stpScope = (m.stpScope|0) & 3;           // the release on every synapse (0), excitatory only (1) or inhibitory only (2)
  // Per-rule short-term plasticity: a class per distinct set of settings (stpClassTable in nodes.js), state per presynaptic cell and class at [c*n + i], and each rule row mapped to its class.
  // A network without the table is one class, the checkpoint's.
  const tbl = m.stpTable && m.stpClasses >= 1 ? m.stpTable
    : [m.stp > 0 ? 1 : 0, m.stpU > 0 ? +m.stpU : 0.2, m.stpTauD > 0 ? +m.stpTauD : 200, m.stpTauF > 0 ? +m.stpTauF : 600];
  const C = Math.max(1, (tbl.length/4)|0);
  S.stpC = C;
  S.stpCls = m.stpCls ? Uint8Array.from(m.stpCls) : new Uint8Array(16);
  // plain arrays, not Float32Array: U, the decays, the scale and the release are doubles, as the single-class engine always computed them (the state arrays are float32)
  S.stpOn = new Uint8Array(C); S.stpUc = new Array(C); S.stpScaleC = new Array(C);
  S.stpDx = new Array(C); S.stpDr = new Array(C);
  for(let c = 0; c < C; c++){
    const U = tbl[c*4 + 1] > 0 ? +tbl[c*4 + 1] : 0.2;
    S.stpOn[c] = tbl[c*4] > 0 ? 1 : 0; S.stpUc[c] = U;
    S.stpScaleC[c] = (m.stpNorm|0) === 1 ? 1/(S.stpOrder ? U*(2 - U) : U) : 1;
    S.stpDx[c] = Math.exp(-1/(tbl[c*4 + 2] > 0 ? +tbl[c*4 + 2] : 200));   // depression recovery
    S.stpDr[c] = Math.exp(-1/(tbl[c*4 + 3] > 0 ? +tbl[c*4 + 3] : 600));   // facilitation decay
  }
  S.stp = S.stpOn.some(x => x) ? 1 : 0;
  S.stpRel = new Array(C).fill(1);
  if(S.stp && (!S.stpX || S.stpX.length !== S.n*C)){
    S.stpX = new Float32Array(S.n*C).fill(1);
    S.stpR = new Float32Array(S.n*C);
    for(let c = 0; c < C; c++) S.stpR.fill(S.stpUc[c], c*S.n, (c + 1)*S.n);
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
  if((S.anyHet || S.anyCons) && !S.wRef && S.w){ S.wRef = S.w.slice(); refStart(); }
  // the eligibility trace per synapse, allocated when the eligibility form first switches on and kept after, as the reference weights are
  if(S.plast && S.modPlast === 2 && !S.elig && S.w) S.elig = new Float32Array(S.w.length);
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
    S.Khom = new Float32Array(n).fill(S.khomInit);   // the slow rate trace of the sliding threshold, started where slideInit says
    S.spkAcc = new Float32Array(n);          // spikes this scaling window
    // the global factor's population: every cell with an inhibitory synapse under a rule (the authors regulate the whole excitatory population); its low-passed spike count starts at the target, so the factor starts at zero as theirs does
    { const gm = new Uint8Array(n); let gn = 0;
      for(let s=0;s<m2;s++) if(S.w[s] < 0 && (S.pmask ? (S.pmask[s] & 31) : 1) && !gm[S.post[s]]){ gm[S.post[s]] = 1; gn++; }
      S.gMask = gn ? gm : null; S.gN = gn; S.gPf = Math.fround(S.iRho*gn); S.gRatio = 0; S.gCnt = 0; S.gInv = gn ? Math.fround(1/(gn*S.iRho)) : 0; }
    // each cell's spikes of the last 16 steps, bit k for k steps ago (bit 0 is this step, set as the cell fires and shifted out at the end of the step), and the delays its synapses carry, bit d for a delay of d: together they say when one of its spikes arrives somewhere
    S.spkHist = new Uint32Array(n);
    S.dmask = new Uint32Array(n);
    for(let i=0;i<n;i++)
      for(let s=S.preStart[i]; s<S.preStart[i+1]; s++) S.dmask[i] |= 1 << S.delay[s];
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
// The outgoing half for one synapse: depression against the postsynaptic trace, or the Vogels term on an inhibitory one.
// At the soma it runs when the cell fires; at the synapse it runs when the spike arrives, with Kpost as it is then.
function plastOutSyn(s, committed){
  const { post, w, pmask, alphaI, Kpost, Khom, iAlpha, ruleTable:RT, RS } = S;
  const rid = pmask ? (pmask[s] & 31) : 1;   // the rule is the low five bits; the channel the top three
  if(!rid) return;                           // 0 is frozen
  const o = rid*RS;
  if(committed && committed(s, o)) return;
  const wmax = RT[o+2], wdep = RT[o+3] !== 0, mu = S.wmu;
  const wj = w[s], j = post[s];
  // the sliding threshold: depression scaled by the square of the cell's slow rate estimate over the target, 1 when off
  const sl = S.slideK > 0 ? (Khom[j]*S.slideK)*(Khom[j]*S.slideK) : 1;
  // the third factor on the Hebbian terms: the gain, times modBase + M of the postsynaptic cell under modPlast 1 (MODEL.md 4)
  const g = S.modPlast === 1 ? S.gain*(S.modBase + S.mod[j]) : S.gain;
  if(wj > 0){
    // under the eligibility form (modPlast 2) the Hebbian terms accumulate on the synapse's trace, paid into the weight at the sweep
    if(S.modPlast === 2) S.elig[s] += -g*RT[o+1]*sl*Kpost[j]*(wdep ? (mu === 1 ? wj : Math.pow(wj, mu)) : 1) + g*RT[o+6];
    else w[s] = Math.min(wmax, Math.max(1e-4,
      wj - g*RT[o+1]*sl*Kpost[j]*(wdep ? (mu === 1 ? wj : Math.pow(wj, mu)) : 1) + g*RT[o+6]));
  }
  else {
    const al = alphaI ? alphaI[post[s]] : iAlpha;
    // under the global factor the amplitude is the rule's eta times the population ratio and the presynaptic term is the trace plus one, the authors' xi
    let d = S.iGlobal ? -RT[o+7]*S.gRatio*(Kpost[post[s]] + 1) : -RT[o+7]*(Kpost[post[s]] - al);
    if(wdep) d = softBound(wj, d, wmax);
    w[s] = Math.max(-wmax, Math.min(-1e-4, wj + d));
  }
}
// which synapses of each rule are committed and so skipped by both halves, or null when nothing commits
function committedTest(){
  const { wRef, ruleTable:RT } = S;
  return S.commit && wRef ? (s, o) => RT[o+9] > 0 && wRef[s] >= RT[o+9]*0.9 : null;
}
// Reading at the synapse, the pre trace a post spike at this step sees through a synapse of delay d: the spikes that have arrived are those d or more steps old, so the spikes of the last d - 1 steps are taken out of Kpre and the rest is brought forward by e^(d/tau+), which puts a spike exactly d steps old at 1, the peak of the window (dt 0 from its arrival).
// Kpre holds no spike of this step here, since at the synapse the traces are bumped at the end of the step.
function kpreAtSynapse(pre, d){
  const h = S.spkHist[pre], ek = S.ekS;
  let sub = 0;
  for(let k=1;k<d;k++) if((h >> k) & 1) sub += ek[k];
  const kp = S.exdS[d]*(S.Kpre[pre] - sub);
  return kp > 0 ? kp : 0;
}
// At the synapse, the arrivals of this step: every cell with a spike d steps ago and synapses of delay d, each such synapse depressed now against Kpost as it stands at the start of the step, before any cell of this step fires.
// A presynaptic arrival and a postsynaptic spike in the same step are read as pre before post: the arrival does not see the spike, and the spike's potentiation sees the arrival at the peak of its window.
function plastArrivals(){
  const { n, preStart, delay, spkHist, dmask } = S;
  const committed = committedTest();
  for(let i=0;i<n;i++){
    const h = spkHist[i];
    if(!(h & dmask[i])) continue;
    for(let s=preStart[i], e=preStart[i+1]; s<e; s++)
      if((h >> delay[s]) & 1) plastOutSyn(s, committed);
  }
}
function plastOnSpike(i){
  const { preStart, inStart, revSyn, revPre, delay, w, pmask, alphaI,
          Kpre, Kpost, Kslow, wRef, ruleTable:RT, RS } = S;
  // the third factor on the Hebbian terms: the gain, times modBase + M of this cell under modPlast 1 (MODEL.md 4)
  const atSyn = S.stdpAt === 1, gn = S.modPlast === 1 ? S.gain*(S.modBase + S.mod[i]) : S.gain;
  if(S.calibrating){                         // measure baselines, change nothing
    if(atSyn) return;                        // the traces are bumped at the end of the step
    Kpre[i] += 1; Kpost[i] += 1; S.Khom[i] += 1;
    if(Kslow) Kslow[i] += 1;
    if(S.spkAcc) S.spkAcc[i] += 1;
    return;
  }
  // the slow trace is read before this spike is added: the triplet term potentiates by the postsynaptic history, not by the present spike
  const z = Kslow ? Kslow[i] : 0;
  // The heterosynaptic term of Zenke, Agnes and Gerstner 2015 (their P10Connection): at a postsynaptic spike every incoming excitatory weight moves toward its reference by beta times the cube of the cell's fast postsynaptic trace, read before this spike is added.
  // A lone spike at a low rate leaves the cube near zero and a burst raises it steeply, which is the paper's burst detector; there is no normalization and no ceiling, and the weight limits bound the step.
  const kq = Kpost[i], hetCube = kq*kq*kq;
  // The rule byte is read per synapse and the row is read from it.
  // When no plasticity node exists every plastic synapse carries 1 and every read lands on the same row: one global rule, arrived at by the same path rather than by a separate branch.
  const committed = committedTest();
  // at the soma the outgoing half runs now; at the synapse it runs when each spike arrives (plastArrivals)
  if(!atSyn) for(let s=preStart[i], e=preStart[i+1]; s<e; s++) plastOutSyn(s, committed);
  for(let k=inStart[i], ke=inStart[i+1]; k<ke; k++){
    const s = revSyn[k];
    const rid = pmask ? (pmask[s] & 31) : 1;   // the rule is the low five bits; the channel the top three
    if(!rid) continue;
    const o = rid*RS;
    if(committed && committed(s, o)) continue;
    const wmax = RT[o+2], wdep = RT[o+3] !== 0, het = RT[o+5];
    const wp = w[s], kp = atSyn ? kpreAtSynapse(revPre[k], delay[s]) : Kpre[revPre[k]];
    if(wp > 0){
      // under the eligibility form the Hebbian potentiation goes to the trace; the heterosynaptic pull is homeostatic and moves the weight now
      let nw = wp;
      if(S.modPlast === 2) S.elig[s] += gn*(RT[o+0] + RT[o+4]*z)*kp;
      else nw = wp + gn*(RT[o+0] + RT[o+4]*z)*kp;
      if(het > 0) nw -= het*hetCube*(wp - wRef[s]);
      w[s] = Math.min(wmax, Math.max(1e-4, nw));
    }
    else {
      let d = S.iGlobal ? -RT[o+7]*S.gRatio*kp : -RT[o+7]*kp;
      if(wdep) d = softBound(wp, d, wmax);
      w[s] = Math.max(-wmax, wp + d);
    }
  }
  if(atSyn) return;                          // the traces are bumped at the end of the step
  Kpre[i] += 1; Kpost[i] += 1; S.Khom[i] += 1;
  if(Kslow) Kslow[i] += 1;
  if(S.spkAcc) S.spkAcc[i] += 1;
}
// The end of a step for the traces: at the synapse, the spikes of this step are added here (so every read within the step saw the traces as of the end of the step before), then every trace decays and the spike history moves on.
// Written as the two float32 stores the soma path makes (the bump, then the decay), so the two readings keep the same rounding.
// The global factor's population rate, low-passed, and the ratio the next step's inhibitory updates read: the rate over the target minus one (MODEL.md 4).
function globalFactor(){
  if(!S.gMask) return;
  // kept in float32 like the other engines' lanes: the count low-passed, then the ratio of the rate it stands for to the target, minus one
  S.gPf = Math.fround(S.gPf*S.decG + S.gCnt/S.tauGs);
  S.gRatio = Math.fround(S.gPf*S.gInv - 1);
  S.gCnt = 0;
}
function traceStep(){
  const { n, Kpre, Kpost, Khom, decS, decM, decH, spkHist } = S;
  const atSyn = S.stdpAt === 1;
  for(let i=0;i<n;i++){
    const h = spkHist ? spkHist[i] : 0;
    if(atSyn && (h & 1)){
      Kpre[i] += 1; Kpost[i] += 1; S.Khom[i] += 1;
      if(S.Kslow) S.Kslow[i] += 1;
      if(S.spkAcc) S.spkAcc[i] += 1;
    }
    Kpre[i] *= decS; Kpost[i] *= decM; Khom[i] *= decH;
    if(spkHist) spkHist[i] = (h << 1) & 0x1FFFF;
  }
  if(S.Kslow){ const Ks = S.Kslow, dY = S.decY;
    for(let i=0;i<n;i++) Ks[i] *= dY; }
}

// Tsodyks-Markram release at a presynaptic spike, normalized so a rested synapse delivers its full weight.
// Returns the scale factor for every outgoing synapse of neuron i.
// With several classes each one's release is written to S.stpRel and the class 0 release is returned; a class that is off releases 1.
function stpRelease(i){
  const X = S.stpX, R = S.stpR, n = S.n;
  let first = 1;
  for(let c = 0; c < S.stpC; c++){
    let f = 1;
    if(S.stpOn[c]){
      const U = S.stpUc[c], k = c*n + i;
      if(S.stpOrder) R[k] = R[k] + U*(1 - R[k]);   // stpOrder 1: facilitation before release
      const rel = R[k]*X[k];
      X[k] = X[k] - rel;                       // depletion
      if(!S.stpOrder) R[k] = R[k] + U*(1 - R[k]);  // as published: facilitation after release
      f = rel*S.stpScaleC[c];
    }
    S.stpRel[c] = f;
    if(c === 0) first = f;
  }
  return first;
}
// The release factor of synapse s at the spike just released, by its rule's class and the scope's sign.
const stpFactor = (s, w) => {
  const f = S.stpRel[S.stpCls[S.pmask ? (S.pmask[s] & 31) : 1]];
  return w > 0 ? (S.stpScope === 2 ? 1 : f) : (S.stpScope === 1 ? 1 : f);
};

// Per-millisecond recovery of both state variables, exact exponential rather than Euler so the time constants hold at any step size.
function stpStep(){
  const X = S.stpX, R = S.stpR, n = S.n;
  for(let c = 0; c < S.stpC; c++){
    if(!S.stpOn[c]) continue;
    const U = S.stpUc[c], dx = S.stpDx[c], dr = S.stpDr[c], o = c*n;
    for(let i=o, e=o+n;i<e;i++){
      X[i] = 1 - (1 - X[i])*dx;
      R[i] = U + (R[i] - U)*dr;
    }
  }
}

// Where the reference starts, for the excitatory synapses of a consolidating rule (refInit; MODEL.md 4).
// 0: the wired weight, brought inside the well: a synapse wired above consW starts at consW, where the exact equation settles from above (the forward Euler step from a reference far above consW overshoots to zero instead, and a strong pathway was then pulled to zero by the heterosynaptic term).
// 1: the lower fixed point, zero, which is where Zenke, Agnes and Gerstner 2015 start it (their weight_a), so every synapse is pulled toward nothing until it has stayed above the midpoint long enough to consolidate.
// 2: the consolidated weight.
// A rule without consolidation keeps the weight itself as the reference.
function refStart(){
  const { w, wRef, pmask, ruleTable:RT, RS, refInit } = S;
  for(let s=0, m2=w.length; s<m2; s++){
    const rid = pmask ? (pmask[s] & 31) : 1;
    if(!rid || !(w[s] > 0)) continue;
    const o = rid*RS;
    if(!(RT[o+8] > 0 && RT[o+9] > 0)) continue;
    if(refInit === 1) wRef[s] = 0;
    else if(refInit === 2) wRef[s] = RT[o+9];
    else if(wRef[s] > RT[o+9]) wRef[s] = RT[o+9];
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
    // Bounded by the equation's own invariant, as Zenke's code leaves it free: above both the weight and consW every term pulls the reference down, so a strong synapse settles above consW, where the weight's pull balances the well's.
    // The step cannot leave [0, max(r, w, consW)] in the exact equation and is held there.
    // This is a forward Euler step over a cubic, so a short timescale makes it unstable: at a twenty second constant with the published well depth the reference overshoots, the cubic grows with the overshoot, and the weights reach NaN inside ten simulated seconds (measured 2026-09-03).
    const nr = r + rate*(wj - r - RT[o+10]*r*(half - r)*(consW - r));
    const top = Math.max(r, wj, consW);
    wRef[s] = nr < 0 ? 0 : (nr > top ? top : nr);
  }
}

// The modulator field at the end of a step (MODEL.md 4): M(t) = M(t-1) dec + the sources' increment + K times the releasing group's spikes of the step before.
// Written into a second array and swapped, so every read within the step saw M(t-1) whatever the cell order; the step's own spike counts become the next step's.
function modUpdate(){
  const { n, mod, modNew, modStim, modTgt, modK, modCnt, decMod } = S;
  for(let i=0;i<n;i++){
    let v = mod[i]*decMod + (modStim ? modStim[i] : 0);
    if(modTgt){ const g = modTgt[i]; if(g) v += modK[g]*modCnt[g]; }
    modNew[i] = v;
  }
  S.mod = modNew; S.modNew = mod;
  S.modCnt = S.modNext; S.modNext = modCnt; modCnt.fill(0);
}
// The eligibility sweep (modPlast 2; Izhikevich 2007): every modStep ms each excitatory plastic synapse's trace pays (1 - e^(-modStep/eligTau)) of itself into its weight times modBase + M of its postsynaptic cell, then decays by e^(-modStep/eligTau).
// Izhikevich pays the whole trace at every sweep; the factor here makes the sum over a trace's life its amount, so a level means in this form what it means in the gain form.
// The field read is M(t), the step's own, since the sweep follows the field update in every engine.
function eligSweep(){
  const { w, elig, post, pmask, mod, modBase, decElig, eligPay, ruleTable:RT, RS } = S;
  const committed = committedTest();
  for(let s=0, m2=w.length; s<m2; s++){
    const rid = pmask ? (pmask[s] & 31) : 1;
    if(!rid) continue;
    const c = elig[s];
    if(c === 0) continue;
    const o = rid*RS;
    if(committed && committed(s, o)){ elig[s] = c*decElig; continue; }
    const wj = w[s];
    if(wj > 0) w[s] = Math.min(RT[o+2], Math.max(1e-4, wj + c*eligPay*(modBase + mod[post[s]])));
    elig[s] = c*decElig;
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
  if(S.modOn) S.modStim = updateAdd(S.modSrc, S.modStim);
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
    // The rows' dynamic threshold, conductance adaptation and Poisson output (MODEL.md 1), per cell, allocated only when a row asks: the threshold state starts at zero and is not saved in a brain.
    const RF = { thrJ:null, thrDec:null, thr:null, adC:null, poi:null };
    { let anyThr = false, anyAd = false, anyPoi = false;
      for(let i=0;i<n;i++){ const t = T[m.ntype[i]].f7; if(t.thrJump > 0) anyThr = true; if(t.adaptCond) anyAd = true; if(t.poisson) anyPoi = true; }
      if(anyThr){ RF.thrJ = new Float32Array(n); RF.thrDec = new Float32Array(n); RF.thr = new Float32Array(n); }
      if(anyAd) RF.adC = new Uint8Array(n);
      if(anyPoi) RF.poi = new Uint8Array(n);
      for(let i=0;i<n;i++){ const t = T[m.ntype[i]].f7;
        if(anyThr){ RF.thrJ[i] = t.thrJump > 0 ? +t.thrJump : 0; RF.thrDec[i] = Math.exp(-1/(t.thrTau > 0 ? +t.thrTau : 5)); }
        if(anyAd) RF.adC[i] = t.adaptCond ? 1 : 0;
        if(anyPoi) RF.poi[i] = t.poisson ? 1 : 0; } }
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
      nz:new Float32Array(n), nzs:new Float32Array(n),   // the two noise filter states per cell, zero at the start (MODEL.md 2)
      grd:G ? G.grd : null, gthr:G ? G.thr : null, gslp:G ? G.slope : null,
      thrJ:RF.thrJ, thrDec:RF.thrDec, thr:RF.thr, adC:RF.adC, poi:RF.poi,
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
  // The consolidation reference from a saved state.
  // It may arrive before plasticity has allocated one, so it allocates the reference itself and the capture at switch-on, which would overwrite it with the weights, is skipped.
  else if(m.cmd === 'setRef'){ if(S && S.w && m.wRef && m.wRef.length === S.w.length){ if(S.wRef) S.wRef.set(m.wRef); else S.wRef = Float32Array.from(m.wRef); } }
  else if(m.cmd === 'getWeights'){          // learned-state readback (checkpoints, analysis)
    if(!S) return;
    const wc = S.w.slice();
    // measured set points travel with the weights when they exist: without them an unreachable homeostatic target can only be inferred, which leaves a collapse undiagnosed for a whole run
    const rc = S.rhoI ? S.rhoI.slice() : null;
    // and the consolidation reference, which a resumed run needs as much as the weights: without it the run restarts every reference from refInit and forgets which synapses had consolidated
    const fc = S.wRef ? S.wRef.slice() : null;
    postMessage({ cmd:'weights', w:wc, t:S.t, rho:rc, ref:fc },
      [wc.buffer, ...(rc ? [rc.buffer] : []), ...(fc ? [fc.buffer] : [])]);
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
// A noise stimulus with a time constant filters the cell's draw as an Ornstein-Uhlenbeck process of the same variance, and one with a shared fraction mixes in the stimulus's own draw, filtered the same way (MODEL.md 2).
// The states are kept in f32, as the other engines keep them; the arithmetic between is this engine's.
function noiseAt(i){
  let d = noiseDraw(i, S.t, S.seed);
  const F = S.noise;
  if(!F) return d;
  const a = F.a[i];
  if(a > 0){ S.nz[i] = a*S.nz[i] + F.k[i]*d; d = S.nz[i]; }
  const c = F.c[i];
  if(c > 0){
    let s = noiseDraw(NOISE_SHARED + F.id[i], S.t, S.seed);
    if(a > 0){ S.nzs[i] = a*S.nzs[i] + F.k[i]*s; s = S.nzs[i]; }
    d = Math.sqrt(1 - c)*d + Math.sqrt(c)*s;
  }
  return d;
}

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

// One exponential half step of 0.5 ms (MODEL.md 1): with F(v) = k (v - vr)(v - vt) - adaptation + I + G (v0 - v), the step is 0.5 F(v)/C times (e^x - 1)/x at x = 0.5 F'(v)/C, which is exact when F is linear in v over the step.
// I is the current at v0, G the synaptic conductance (0 for current synapses), ac whether the adaptation u is a conductance toward the inhibitory reversal.
const linRow = (vti, vpk) => vti - vpk > 100;
function expHalf(vi, ui, I, G, v0, ki, vri, vti, Ci, ac, vmin){ return expSpan(vi, ui, I, G, v0, ki, vri, vti, Ci, ac, vmin, 0.5); }
// the same over h ms
function expSpan(vi, ui, I, G, v0, ki, vri, vti, Ci, ac, vmin, h){
  const F = ki*(vi - vri)*(vi - vti) - (ac ? ui*(vi - S.eRevAd) : ui) + I + G*(v0 - vi);
  const x = h*(ki*(2*vi - vri - vti) - (ac ? ui : 0) - G)/Ci;
  const phi = Math.abs(x) < 1e-6 ? 1 + 0.5*x : Math.expm1(x)/x;
  vi += h*F/Ci*phi;
  return vi < vmin ? vmin : vi;
}
// The exponential step's spike (MODEL.md 1): the crossing is put inside the millisecond by interpolating the distance to the threshold between the opening of the step and its end (the threshold relaxes during the step, so it is taken at both ends), the cell resets there, and the rest of the step is integrated from the reset with the jump in u already applied and the threshold already relaxing.
// Without it every interval is rounded up to the next millisecond, a quarter of a percent per millisecond of interval, two percent at 40 Hz.
function spanReset(i, vs, vi, vth, ui, I, G, v0, ac, vmin){
  const th = S.thr ? S.thr[i] : 0, thS = th > 0 && S.thrDec[i] > 0 ? th/S.thrDec[i] : th;
  const dS = vs - (vth - th + thS), dE = vi - vth, fr = dS < 0 && dE > dS ? Math.min(1, Math.max(0, dS/(dS - dE))) : 1, rest = 1 - fr;
  if(S.thr) S.thr[i] = S.thrJ[i]*Math.pow(S.thrDec[i], rest);
  return expSpan(S.c[i], ui, I, G, v0, S.k7[i], S.vr7[i], S.vt7[i], S.C7[i], ac, vmin, rest);
}

// The equations' membrane step (integ 2, MODEL.md 1) on a quadratic row.
// With u, the input and the synaptic conductance held, C v' = k (v - vr)(v - vt) - adaptation + I + G (v0 - v) is v' = A w^2 + q in w = v - m, a Riccati equation, whose solution is w(t) = (w0 + q X)/(1 - A w0 X) with X = tan(s t)/s, s^2 = A q (tanh(s t)/s when A q is below zero).
// The time from w0 to wp (w0 below it), Infinity if the membrane never gets there.
function quadTime(A, q, w0, wp){
  if(w0 >= wp) return 0;
  const y = wp - w0, x = q + A*w0*wp, z = A*q;
  if(x > 0){
    const R = y/x, e = z*R*R;
    if(Math.abs(e) < 1e-4) return R*(1 - e/3 + e*e/5);
    if(z > 0){ const s = Math.sqrt(z); return Math.atan(s*R)/s; }
    const s = Math.sqrt(-z), r = s*R;
    return r < 1 ? 0.5*Math.log((1 + r)/(1 - r))/s : Infinity;
  }
  // past the pole of the tangent: no fixed point, so the membrane gets there on the next branch
  if(z > 0){ const s = Math.sqrt(z); return (0.5*Math.PI + Math.atan(-x/(s*y)))/s; }
  return Infinity;
}
// w after h from w0, when the membrane does not reach the threshold within h, and the integral of w over the span left in qI.
// With phi = Cs - A w0 Sn, Cs = cos(s h) and Sn = sin(s h)/s (cosh and sinh when A q is below zero, both scaled by e^(-s h), which ls puts back), w = (w0 Cs + q Sn)/phi and the integral is -(1/A) ln phi.
// This form has no pole: phi stays positive until the membrane reaches the threshold.
let qI = 0;
function quadAt(A, q, w0, h){
  const z = A*q, e = z*h*h;
  let Cs, Sn, ls = 0;
  if(Math.abs(e) < 1e-4){ Cs = 1 - e/2 + e*e/24; Sn = h*(1 - e/6 + e*e/120); }
  else if(z > 0){ const s = Math.sqrt(z); Cs = Math.cos(s*h); Sn = Math.sin(s*h)/s; }
  else { const s = Math.sqrt(-z), E = Math.exp(-2*s*h); Cs = 0.5*(1 + E); Sn = 0.5*(1 - E)/s; ls = s*h; }
  const phi = Cs - A*w0*Sn;
  qI = -(ls + Math.log(Math.abs(phi)))/A;
  return (w0*Cs + q*Sn)/phi;
}
// One millisecond on a quadratic row: two half steps of 0.5 ms, each solved exactly with u held.
// After each half step u relaxes toward b (v - vr) at the rate a, solved exactly with v at its exact mean over the half step.
// A crossing inside a half step puts the reset there (v to c, u up by d, the threshold jumped and relaxing over the rest of the millisecond), and the rest of the half step runs from c.
// A cell fires at most once a millisecond: a second crossing inside it holds the potential at the threshold, and the cell fires at the opening of the next.
// Returns v; u and whether it fired are left in qU and qSp.
let qU = 0, qSp = 0;
function quadMs(i, vi, ui, I, G, v0, ac, vth){
  const Ci = S.C7[i], ki = S.k7[i], vri = S.vr7[i], vti = S.vt7[i], ai = S.a[i], bi = S.b[i], vmin = S.vmin, eA = S.eRevAd;
  const A = ki/Ci, hd = 0.5*(vti - vri), sm = vri + vti, ea = Math.exp(-0.5*ai);
  qSp = 0;
  for(let hs = 0; hs < 2; hs++){
    // the vertex m and the offset q of the parabola, with the adaptation conductance and the synaptic conductance folded into the slope
    let gs = (ac ? ui : 0) + G, m = 0.5*sm + gs/(2*ki);
    let q = (-ki*hd*hd + I + (ac ? ui*eA : -ui) + G*v0 - 0.5*gs*sm - gs*gs/(4*ki))/Ci;
    // iv: the integral of v over the half step
    let iv;
    const tc = quadTime(A, q, vi - m, vth - m);
    if(tc > 0.5){ vi = m + quadAt(A, q, vi - m, 0.5); iv = qI + 0.5*m; }
    else {
      quadAt(A, q, vi - m, tc); iv = qI + tc*m;
      const left = 0.5 - tc;
      if(qSp){ vi = vth; iv += left*vth; }
      else {
        qSp = 1;
        vi = S.c[i]; ui += S.d[i];
        if(S.thr){ S.thr[i] = S.thrJ[i]*Math.pow(S.thrDec[i], 0.5*(1 - hs) + left); vth = S.vp7[i] + S.thrJ[i]; }
        gs = (ac ? ui : 0) + G; m = 0.5*sm + gs/(2*ki);
        q = (-ki*hd*hd + I + (ac ? ui*eA : -ui) + G*v0 - 0.5*gs*sm - gs*gs/(4*ki))/Ci;
        const t2 = quadTime(A, q, vi - m, vth - m);
        if(t2 <= left){ quadAt(A, q, vi - m, t2); iv += qI + t2*m + (left - t2)*vth; vi = vth; }
        else { vi = m + quadAt(A, q, vi - m, left); iv += qI + left*m; }
      }
    }
    if(vi < vmin) vi = vmin;
    const tg = bi*(2*iv - vri);
    ui = tg + (ui - tg)*ea;
  }
  qU = ui;
  return vi;
}

function step(fired, lat, sIdx){
  const { n, v, u, a, b, c, d, ring, cur, ROWS, bias, stim, namp, ext,
          preStart, post, w, delay, pool, vmin, C7, k7, vr7, vt7, vp7, grd, gthr, gslp, thr, thrDec, thrJ, poi, adC } = S;
  const poolIn = pool ? poolDrive() : null;
  const row = cur*n;
  let count = 0, watchSpiked = false;
  // at the synapse, the spikes arriving this step depress their synapses before any cell fires
  if(S.plast && S.stdpAt === 1 && !S.calibrating) plastArrivals();
  const M = S.modOn ? S.mod : null, mDrive = S.modDrive, mCur = S.modCur, modGrp = S.modGrp;
  for(let i=0;i<n;i++){
    // the modulator field scales the synaptic current by 1 + modDrive M and adds modCur M, M as the step before left it (MODEL.md 4); without a field the terms add in the order they always did
    const I = M
      ? (ring[row+i] + (pool ? poolIn[pool[i]] : 0))*(1 + mDrive*M[i]) + mCur*M[i] + bias[i] + (stim ? stim[i] : 0) + (ext ? ext[i] : 0) + (namp ? namp[i]*noiseAt(i) : 0)
      : ring[row+i] + bias[i] + (stim ? stim[i] : 0) + (ext ? ext[i] : 0)
      + (namp ? namp[i]*noiseAt(i) : 0) + (pool ? poolIn[pool[i]] : 0);
    // the dynamic threshold relaxes before the test, whatever the cell does this step (Auryn's order)
    if(thr) thr[i] *= thrDec[i];
    let vi = v[i], ui = u[i], qc = 0;
    if(S.refCnt && S.refCnt[i] > 0){
      // absolute refractoriness: the membrane is held at reset and the threshold cannot be reached, whatever the drive
      S.refCnt[i]--;
      ui += a[i]*(b[i]*(vi - vr7[i]) - ui);
      u[i] = ui;
      continue;
    }
      // Izhikevich 2007: C dv = k (v - vr)(v - vt) - u + I, the same two half steps and the same floor; u is measured from the rest
      const Ci = C7[i], ki = k7[i], vri = vr7[i], vti = vt7[i];
      if(S.integ && !(grd && grd[i]) && !(poi && poi[i]) && linRow(vti, vp7[i])){
        vi = expHalf(vi, ui, I, 0, vi, ki, vri, vti, Ci, adC && adC[i], vmin); vi = expHalf(vi, ui, I, 0, v[i], ki, vri, vti, Ci, adC && adC[i], vmin);
      } else if(grd && grd[i]){
        // a graded cell's membrane is passive: the leak is the row's slope at rest, k (vt - vr), and the potential is capped at vpeak
        const gl = ki*(vti - vri);
        vi += 0.5*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
        vi += 0.5*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
        if(vi > vp7[i]) vi = vp7[i];
      } else if(poi && poi[i]){
        // a Poisson cell (MODEL.md 1): no membrane; it fires this step with probability I over 1000, the current read as a rate in hertz, on the counter-based draw, by being put at its peak
        if(I > 0 && poissonDraw(i, S.t, S.seed) < I*0.001) vi = vp7[i] + (thr ? thr[i] : 0);
      } else if(S.integ === 2){
        // the equations' step (integ 2) on a quadratic row: exact over each half step, the reset inside it
        vi = quadMs(i, vi, ui, I, 0, vi, adC && adC[i], vp7[i] + (thr ? thr[i] : 0)); ui = qU; qc = 1;
      } else {
      // with the row's adaptation a conductance (adaptCond), u in nanosiemens draws current toward the inhibitory reversal
      vi += 0.5*(ki*(vi - vri)*(vi - vti) - (adC && adC[i] ? ui*(vi - S.eRevAd) : ui) + I)/Ci;
      if(vi < vmin) vi = vmin;
      vi += 0.5*(ki*(vi - vri)*(vi - vti) - (adC && adC[i] ? ui*(vi - S.eRevAd) : ui) + I)/Ci;
      if(vi < vmin) vi = vmin;
      }
      if(!qc) ui += a[i]*(b[i]*(vi - vri) - ui);
    if(grd && grd[i]){
      // a graded cell: no spike, no reset, no refractory period; every millisecond it delivers a spike of amplitude r through its synapses
      const r = Math.min(1, Math.max(0, (vi - gthr[i])/gslp[i]));
      if(r > 0) for(let s=preStart[i], e1=preStart[i+1]; s<e1; s++)
        ring[((cur+delay[s])%ROWS)*n + post[s]] += w[s]*r;
      v[i] = vi; u[i] = ui;
      continue;
    }
    if(qc ? qSp : vi >= vp7[i] + (thr ? thr[i] : 0)){
      if(qc) count++;
      else if(S.integ && !(poi && poi[i]) && linRow(vt7[i], vp7[i])){ ui += d[i]; vi = spanReset(i, v[i], vi, vp7[i] + (thr ? thr[i] : 0), ui, I, 0, vi, adC && adC[i], vmin); count++; }
      else { vi = c[i]; ui += d[i]; count++;
      if(thr) thr[i] = thrJ[i]; }
      if(S.gMask && S.gMask[i]) S.gCnt++;
      // count, not a flag: a flag capped every rate this readout could report at 1000/steps Hz. lat is the first spike of the tick in milliseconds, 255 for silent, which is the only grain STDP works on.
      if(fired[i] < 255) fired[i]++;
      if(lat && lat[i] === 255) lat[i] = sIdx < 254 ? sIdx : 254;
      if(S.refCnt) S.refCnt[i] = S.refrac;
      if(pool) S.poolNext[pool[i]]++;
      if(i===S.watch) watchSpiked = true;
      const sf = S.stp ? stpRelease(i) : 1;
      // the release applies to the synapses in scope: every one, or one sign of weight (stpScope)
      const sfE = S.stpScope === 2 ? 1 : sf, sfI = S.stpScope === 1 ? 1 : sf;
      if(S.stpC > 1 && S.stp){
        for(let s=preStart[i], e1=preStart[i+1]; s<e1; s++)
          ring[((cur+delay[s])%ROWS)*n + post[s]] += w[s]*stpFactor(s, w[s]);
      } else for(let s=preStart[i], e1=preStart[i+1]; s<e1; s++)
        ring[((cur+delay[s])%ROWS)*n + post[s]] += w[s]*(w[s] > 0 ? sfE : sfI);
      if(modGrp && modGrp[i]) S.modNext[modGrp[i]]++;
      if(S.plast){ if(S.spkHist) S.spkHist[i] |= 1; plastOnSpike(i); }
    }
    v[i] = vi; u[i] = ui;
  }
  if(S.plast){
    globalFactor();
    traceStep();
    if(S.anyCons && ++S.consAcc >= S.consStep){ S.consAcc = 0; consolidate(); }
  }
  if(M){ modUpdate(); if(S.plast && S.modPlast === 2 && S.elig && ++S.modAcc >= S.modStep){ S.modAcc = 0; eligSweep(); } }
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
          C7, k7, vr7, vt7, vp7, grd, gthr, gslp, thr, thrDec, thrJ, poi, adC, KX, gX, ringX, chanDec, chanItau, chanMg, pmask, cond, eRevE, eRevI, chanErev } = S;
  const poolIn = pool ? poolDrive() : null;
  const row = cur*n;
  let count = 0, watchSpiked = false;
  // at the synapse, the spikes arriving this step depress their synapses before any cell fires
  if(S.plast && S.stdpAt === 1 && !S.calibrating) plastArrivals();
  // a synapse's ring: by the channel in its plasticity byte, else by sign
  const ringOf = (s, wt) => { const ch = pmask ? pmask[s] >> 5 : 0; return ch >= 2 ? null : (wt > 0 ? ring : ring2); };
  const M = S.modOn ? S.mod : null, mDrive = S.modDrive, mCur = S.modCur, modGrp = S.modGrp;
  for(let i=0;i<n;i++){
    const ge = gE[i]*decE + ring[row+i];
    const gi = gI[i]*decI + ring2[row+i] + (pool ? poolIn[pool[i]] : 0);
    const v0 = v[i];
    // current-based: g times the scale; conductance-based: |g| (E - v)
    let I = (cond ? Math.abs(ge)*(eRevE - v0) + Math.abs(gi)*(eRevI - v0) : ge*itauE + gi*itauI);
    // without a modulator field the drive terms add here, term by term, as they always did
    if(!M){ I += bias[i]; I += (stim ? stim[i] : 0); I += (ext ? ext[i] : 0); I += (namp ? namp[i]*noiseAt(i) : 0); }
    // the total synaptic conductance, for the exponential step: the current at v is I + G (v0 - v)
    let G = cond ? Math.abs(ge) + Math.abs(gi) : 0;
    if(KX) for(let x=0;x<KX;x++){
      const at = x*n + i, g = gX[at]*chanDec[x] + ringX[(x*ROWS + cur)*n + i];
      gX[at] = g;
      // the magnesium block (Jahr and Stevens 1990): B(v) = 1/(1 + [Mg]/3.57 exp(-0.062 v)), the fraction of the channel's current the voltage lets through
      const bI = chanMg[x] > 0 ? 1/(1 + (chanMg[x]/3.57)*Math.exp(-0.062*v0)) : 1;
      I += bI*(cond ? Math.abs(g)*(chanErev[x] - v0) : g*chanItau[x]);
      if(cond) G += bI*Math.abs(g);
    }
    // the modulator field scales the synaptic current and conductance by 1 + modDrive M and adds modCur M, M as the step before left it (MODEL.md 4); the drive terms follow
    if(M){ const f = 1 + mDrive*M[i]; I = I*f + mCur*M[i]; G *= f; I += bias[i]; I += (stim ? stim[i] : 0); I += (ext ? ext[i] : 0); I += (namp ? namp[i]*noiseAt(i) : 0); }
    // the dynamic threshold relaxes before the test, whatever the cell does this step (Auryn's order)
    if(thr) thr[i] *= thrDec[i];
    let vi = v[i], ui = u[i], qc = 0;
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
      } else if(poi && poi[i]){
        // a Poisson cell (MODEL.md 1): no membrane; it fires this step with probability I over 1000, the current read as a rate in hertz, on the counter-based draw, by being put at its peak
        if(I > 0 && poissonDraw(i, S.t, S.seed) < I*0.001) vi = vp7[i] + (thr ? thr[i] : 0);
      } else if(S.integ && linRow(vti, vp7[i])){
        vi = expHalf(vi, ui, I, G, v0, ki, vri, vti, Ci, adC && adC[i], vmin); vi = expHalf(vi, ui, I, G, v0, ki, vri, vti, Ci, adC && adC[i], vmin);
      } else if(S.integ === 2){
        vi = quadMs(i, vi, ui, I, G, v0, adC && adC[i], vp7[i] + (thr ? thr[i] : 0)); ui = qU; qc = 1;
      } else {
      // with the row's adaptation a conductance (adaptCond), u in nanosiemens draws current toward the inhibitory reversal
      vi += 0.5*(ki*(vi - vri)*(vi - vti) - (adC && adC[i] ? ui*(vi - S.eRevAd) : ui) + I)/Ci;
      if(vi < vmin) vi = vmin;
      vi += 0.5*(ki*(vi - vri)*(vi - vti) - (adC && adC[i] ? ui*(vi - S.eRevAd) : ui) + I)/Ci;
      if(vi < vmin) vi = vmin;
      }
      if(!qc) ui += a[i]*(b[i]*(vi - vri) - ui);
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
    if(qc ? qSp : vi >= vp7[i] + (thr ? thr[i] : 0)){
      if(qc) count++;
      else if(S.integ && !(poi && poi[i]) && linRow(vt7[i], vp7[i])){ ui += d[i]; vi = spanReset(i, v0, vi, vp7[i] + (thr ? thr[i] : 0), ui, I, G, v0, adC && adC[i], vmin); count++; }
      else { vi = c[i]; ui += d[i]; count++;
      if(thr) thr[i] = thrJ[i]; }
      if(S.gMask && S.gMask[i]) S.gCnt++;
      // count, not a flag: a flag capped every rate this readout could report at 1000/steps Hz. lat is the first spike of the tick in milliseconds, 255 for silent, which is the only grain STDP works on.
      if(fired[i] < 255) fired[i]++;
      if(lat && lat[i] === 255) lat[i] = sIdx < 254 ? sIdx : 254;
      if(S.refCnt) S.refCnt[i] = S.refrac;
      if(pool) S.poolNext[pool[i]]++;
      if(i===S.watch) watchSpiked = true;
      const sf = S.stp ? stpRelease(i) : 1;
      // the release applies to the synapses in scope: every one, or one sign of weight (stpScope)
      const sfE = S.stpScope === 2 ? 1 : sf, sfI = S.stpScope === 1 ? 1 : sf;
      for(let s=preStart[i], e1=preStart[i+1]; s<e1; s++){
        const wt = w[s]*(S.stpC > 1 && S.stp ? stpFactor(s, w[s]) : (w[s] > 0 ? sfE : sfI)), slot = (cur+delay[s])%ROWS, at = slot*n + post[s];
        const rg = ringOf(s, wt);
        if(rg) rg[at] += wt; else ringX[(((pmask[s] >> 5) - 2)*ROWS + slot)*n + post[s]] += wt;
      }
      if(modGrp && modGrp[i]) S.modNext[modGrp[i]]++;
      if(S.plast){ if(S.spkHist) S.spkHist[i] |= 1; plastOnSpike(i); }
    }
    v[i] = vi; u[i] = ui; gE[i] = ge; gI[i] = gi;
  }
  if(S.plast){
    globalFactor();
    traceStep();
    if(S.anyCons && ++S.consAcc >= S.consStep){ S.consAcc = 0; consolidate(); }
  }
  if(M){ modUpdate(); if(S.plast && S.modPlast === 2 && S.elig && ++S.modAcc >= S.modStep){ S.modAcc = 0; eligSweep(); } }
  if(S.stp) stpStep();
  ring.fill(0, row, row+n); ring2.fill(0, row, row+n);
  if(KX) for(let x=0;x<KX;x++) ringX.fill(0, (x*ROWS + cur)*n, (x*ROWS + cur)*n + n);
  if(pool) poolSwap();
  S.cur = (cur+1)%ROWS;
  S.watchV = S.watch>=0 ? (watchSpiked ? vp7[S.watch] : v[S.watch]) : 0;
  return count;
}

// The Brian 2 case: everything tools/brian_ref.py needs to run a network on Brian 2's spike queues, built from the engine's own init message so the export and the reference comparison (tools/brianref.mjs) cannot drift apart.
// What the engine derives from init at configure time (decay factors, per-protocol factors, plasticity constants) is derived the same way here, so Brian receives the engine's own doubles and float32s.
//
// A case is plain JSON: arrays of numbers, no typed arrays.
// For a large network the per-protocol factor arrays are one float per cell each, and the synapse arrays are one entry per synapse, so a ten million synapse scene is a file of a few hundred megabytes; the export says so.
import { RULE_FIELDS, gradedArrays } from './nodes.js';
import { foldNoise } from './noisefold.js';

const RS = RULE_FIELDS.length;
// the floor the engine resolves from what init sends (simworker.js configure)
export const floorOf = vmin => vmin === undefined ? -90 : (vmin < 0 ? vmin : null);

// init: the engine init message (count, ntype, bias, v0, u0, types, the synapse arrays, every field engineConfig carries). extra: { ms, frames, noiseProbe }.
export function caseFromInit(init, extra = {}){
  const types = init.types;
  const n = init.count !== undefined ? init.count : init.ntype.length;
  const row = t => { const f = types[t].f7;
    // the nine membrane numbers, then the dynamic threshold's jump and decay, the conductance adaptation flag and the Poisson flag (MODEL.md 1), as the engines keep them
    return [f.C, f.k, f.vr, f.vt, f.vpeak, f.a, f.b, f.c, f.d, f.thrJump > 0 ? +f.thrJump : 0, Math.exp(-1/(f.thrTau > 0 ? +f.thrTau : 5)), f.adaptCond ? 1 : 0, f.poisson ? 1 : 0].map(Math.fround); };
  const tE = init.tauE > 0 ? init.tauE : 3, tI = init.tauI > 0 ? init.tauI : 8;
  const decE = Math.exp(-1/tE), decI = Math.exp(-1/tI);
  const psc = Math.max(0, Math.min(2, init.psc|0));
  // the per-step scale under each convention; psc 2 follows the step in brian_ref.py, which recomputes it from tau and dt
  const itau = (dec, tau) => psc === 2 ? tau*(1 - dec) : psc === 1 ? 1 - dec : 1;
  const chanDec = Float32Array.from(init.chanTau || [], t => Math.exp(-1/(t > 0 ? t : 3)));
  // each protocol as the per-neuron factor updateAdd multiplies amp times scale by: the gain at its index, 1 without a gain, 0 for a cell it omits
  const asList = list => list.map(p => {
    const f = new Float64Array(n);
    Array.from(p.idx).forEach((i, q) => { f[i] = p.gain ? p.gain[q] : 1; });
    return { t0:p.t0, duration:p.duration, mode:p.mode, period:p.period, width:p.width, amp:p.amp, f:[...f] };
  });
  const protos = init.protocols || [];
  const out = { n, ms:extra.ms, rows:[...init.ntype].map(row), bias:[...init.bias],
    v0:[...init.v0], u0:[...init.u0], refrac:init.refrac, vmin:floorOf(init.vmin),
    syn:init.syn|0, integ:init.integ|0, psc, decE, decI, itauE:itau(decE, tE), itauI:itau(decI, tI),
    eRevE:Number.isFinite(+init.eRevE) ? +init.eRevE : 0,
    eRevI:Number.isFinite(+init.eRevI) ? +init.eRevI : -70,
    chanDec:[...chanDec], chanItau:[...Float32Array.from(chanDec, (d, k) => itau(d, (init.chanTau || [])[k] > 0 ? init.chanTau[k] : 3))],
    chanErev:[...Float32Array.from(init.chanErev || [])],
    chanMg:[...Float32Array.from(init.chanMg || [])],
    // a noise stimulus's memory and shared part per cell, the same fold every engine's host makes (src/noisefold.js); absent for the white draw
    ...(F => F ? { noise:{ a:[...F.a], k:[...F.k], c:[...F.c], id:[...F.id] } } : {})(foldNoise(init.protocols, init.count)),
    stim:asList(protos.filter(p => p.mode !== 3)), noiseProtocols:asList(protos.filter(p => p.mode === 3)),
    seed:((init.seed|0) || 1) >>> 0 };
  if(init.pool) Object.assign(out, { pool:[...init.pool], poolK:[...Float32Array.from(init.poolK)] });
  if(init.inputs && init.inputs.length){
    out.inputs = init.inputs.map(m => ({ chStart:[...m.chStart], chIdx:[...m.chIdx], chW:[...m.chW], amp:m.amp }));
    out.frames = (extra.frames || []).map(f => ({ t:f.t, mi:f.mi, gains:[...f.gains] }));
  }
  const G = gradedArrays(init.ntype, types);
  if(G) Object.assign(out, { grd:[...G.grd], gthr:[...G.thr], gslp:[...G.slope] });

  const r = init;
  if(r.plast){
    const tauS = r.tauS > 0 ? r.tauS : 16.8, tauM = r.tauM > 0 ? r.tauM : tauS;
    const wmax = r.wmax > 0 ? +r.wmax : 30;
    const aP = r.aP !== undefined ? +r.aP : 0.008, aM = r.aM !== undefined ? +r.aM : 0.001;
    const iEta = r.iEta !== undefined ? +r.iEta : 0.002, iRho = r.iRho || 5;
    const trip = r.trip > 0 ? +r.trip : 0, tauY = r.tauY > 0 ? +r.tauY : 114;
    const het = r.het > 0 ? +r.het : 0, tin = r.tin > 0 ? +r.tin : 0;
    const cons = r.cons > 0 ? +r.cons : 0, consW = r.consW > 0 ? +r.consW : 0.1*wmax;
    const consP = r.consP > 0 ? +r.consP : 10;
    let RT = r.ruleTable ? Float32Array.from(r.ruleTable) : null;
    if(!RT){
      RT = new Float32Array(2*RS);
      RT.set([aP, aM, wmax, (r.wdep|0) === 1 ? 1 : 0, trip, het, tin, iEta, cons, consW, consP], RS);
    }
    const rows = [];
    for(let k = 0; k < RT.length/RS; k++) rows.push([...RT.subarray(k*RS, (k + 1)*RS)]);
    let anyTrip = 0, anyHet = 0, anyCons = 0;
    for(let k = 1; k < rows.length; k++){
      anyTrip = Math.max(anyTrip, rows[k][4]); anyHet = Math.max(anyHet, rows[k][5]);
      anyCons = Math.max(anyCons, rows[k][8]);
    }
    const hasRef = anyHet > 0 || anyCons > 0;
    const tauCons = r.tauCons > 0 ? +r.tauCons : 1200000, consStep = r.consStep > 0 ? +r.consStep : 1200;
    const calMs = (r.calS > 0 ? r.calS : 10)*1000;
    // the sliding threshold (MODEL.md 4), derived as simworker.js configure derives it: the multiplier on depression is (Khom slideK)^2 when on
    const slideTau = r.slideTau > 0 ? +r.slideTau : 600000;
    const slideK = (r.slide|0) === 1 && r.slideRate > 0 ? 1000/(slideTau*r.slideRate) : 0;
    // where the pair rule is timed (stdpAt 1: at the synapse), with the engine's own tables: what a spike k steps ago contributes to the pre trace, and the factor that brings the trace d steps back
    out.plast = { rules:rows, refInit:r.refInit|0, iAlpha:iRho*0.001*(tauS + tauM),
      stdpAt:(r.stdpAt|0) === 1 ? 1 : 0,
      gain:r.plastGain === undefined ? 1 : Math.max(0, +r.plastGain || 0),
      wmu:r.wmu === undefined ? 1 : Math.max(0, Math.min(1, +r.wmu)),
      ek:Array.from({ length:17 }, (_, k) => Math.exp(-k/tauS)), exd:Array.from({ length:17 }, (_, d) => Math.exp(d/tauS)),
      decS:Math.exp(-1/tauS), decM:Math.exp(-1/tauM), decY:Math.exp(-1/tauY),
      hasSlow:(anyTrip > 0 || anyHet > 0) ? 1 : 0, commitOn:((r.commit|0) === 1 && hasRef) ? 1 : 0,
      sEta:r.sEta > 0 ? r.sEta : 0.001, iRho, scaleOn:(r.scale|0) === 1 ? 1 : 0,
      rhoMode:(r.rhoMode|0) === 1 ? 1 : 0, calMs, calSecs:calMs/1000, tauSum:tauS + tauM,
      consOn:anyCons > 0 ? 1 : 0, consStep, consRate:consStep/tauCons,
      slideOn:slideK > 0 ? 1 : 0, slideK, decH:Math.exp(-1/slideTau), khomInit:(r.slideInit > 0 ? +r.slideInit : 0)*slideTau/1000,
      // the global factor on the inhibitory rule (simworker.js S.iGlobal, S.decG, S.tauGs)
      iGlobalOn:(r.iGlobal|0) === 1 ? 1 : 0, decG:Math.exp(-1/(r.iTauG > 0 ? +r.iTauG*1000 : 10000)), tauGs:(r.iTauG > 0 ? +r.iTauG*1000 : 10000)/1000 };
  }
  if(r.stp){
    const U = r.stpU > 0 ? +r.stpU : 0.2;
    const order = (r.stpOrder|0) === 1 ? 1 : 0;
    out.stp = { U, order, scope:(r.stpScope|0) & 3, scale:(r.stpNorm|0) === 1 ? 1/(order ? U*(2 - U) : U) : 1,
      dx:Math.exp(-1/(r.stpTauD > 0 ? +r.stpTauD : 200)),
      dr:Math.exp(-1/(r.stpTauF > 0 ? +r.stpTauF : 600)), R0:Math.fround(U) };
    // per-rule short-term plasticity: every class with its own state, and the class of each rule row (stpClassTable in nodes.js)
    if(r.stpTable && (r.stpClasses|0) > 1){
      out.stp.classes = Array.from({ length:r.stpClasses|0 }, (_, c) => {
        const Uc = r.stpTable[c*4 + 1] > 0 ? +r.stpTable[c*4 + 1] : 0.2;
        return { on:r.stpTable[c*4] > 0 ? 1 : 0, U:Uc, scale:(r.stpNorm|0) === 1 ? 1/(order ? Uc*(2 - Uc) : Uc) : 1,
          dx:Math.exp(-1/(r.stpTable[c*4 + 2] > 0 ? +r.stpTable[c*4 + 2] : 200)),
          dr:Math.exp(-1/(r.stpTable[c*4 + 3] > 0 ? +r.stpTable[c*4 + 3] : 600)), R0:Math.fround(Uc) };
      });
      out.stp.cls = Array.from(r.stpCls || new Uint8Array(16));
    }
  }
  // neuromodulation (MODEL.md 4), folded as simworker.js configure folds it: the field's decay, the sources' increment per cell (constant sources, which Brian needs no schedule for), the release arrays with the release per spike folded by the decay, and what reads the field
  {
    const modTau = r.modTau > 0 ? +r.modTau : 200, decMod = Math.fround(Math.exp(-1/modTau));
    const srcs = r.mods || [], G = r.modK && r.modK.length >= 2 ? r.modK.length : 1;
    const modPlast = (r.modPlast|0) === 2 ? 2 : (r.modPlast|0) === 1 ? 1 : 0;
    const drive = +r.modDrive || 0, cur = +r.modCur || 0;
    if(srcs.length || G > 1 || modPlast || drive || cur){
      if(srcs.some(p => (p.mode|0) !== 0)) throw new Error('the Brian case carries constant modulator sources only');
      const inc = new Float32Array(n);
      for(const p of srcs){ const k = Math.fround((+p.amp || 0)*(1 - decMod)); for(const i of p.idx) inc[i] += k; }
      const step = r.modStep > 0 ? Math.max(1, Math.round(+r.modStep)) : 10;
      const decElig = Math.fround(Math.exp(-step/(r.eligTau > 0 ? +r.eligTau : 1000)));
      out.mod = { decMod, inc:[...inc], grp:G > 1 ? [...r.modGrp] : null, tgt:G > 1 ? [...r.modTgt] : null,
        Kf:[...Float32Array.from(G > 1 ? r.modK : [0], k => Math.fround(k*(1 - decMod)))],
        plast:modPlast, base:r.modBase === undefined ? 1 : (+r.modBase || 0), drive, cur, step,
        eligPay:Math.fround(1 - decElig), decElig };
    }
  }
  if(init.preStart){
    Object.assign(out, { pre:[...init.preStart], post:[...init.post], w:[...init.w], delay:[...init.delay],
      chan:init.pmask ? [...init.pmask].map(b => b >> 5) : null,
      rule:init.pmask ? [...init.pmask].map(b => b & 31) : null });
  }
  if(extra.noiseProbe && out.noiseProtocols.length) out.noiseProbe = extra.noiseProbe;
  return out;
}

// WebGPU simulation engine.
// Same message protocol and dynamics contract as the CPU reference engine (simworker.js); see ENGINE.md.
import { gradedArrays, freezeGraded } from './nodes.js';
import { checkProtocol, helloReply, TERM_NAMES, OPTIONAL } from './protocol.js';
import { NOISE_SHARED } from './rand.js';
import { foldNoise } from './noisefold.js';
// Differences inside the contract: spike deliveries accumulate in 1/4096 fixed point through i32 atomics (integer addition is order independent, so delivery sums are exactly reproducible run to run), and the per-ms noise draw comes from a counter based hash of (neuron, t, seed) instead of Math.random.
// Plasticity (pair STDP + Vogels homeostasis, same rules as the CPU engine) runs as three extra dispatches per step when enabled: outgoing updates (thread per spike over its CSR slice), incoming updates (thread per spike over the reverse index, filtered per partition), trace bump, with trace decay folded into the next update pass.
// One contract-level deviation: on the CPU, two neurons spiking in the same millisecond see each other's traces in neuron-index order; here all same-step updates read the pre-increment traces.
// Both conventions are arbitrary at 1 ms resolution and the battery only constrains statistics.
// Synapse storage is partitioned by presynaptic neuron range from day one (2 GB per-buffer device limit; delivery-list layout after NeMo, Fidjeland 2009).
// Each partition packs post and delay into one u32 (delay in the top 5 bits) plus an f32 weight array.

const SCALE = 4096;            // fixed point step for ring accumulation
const ROWS = 18;               // ring rows, matches the CPU engine
const WG = 256;                // update pass workgroup size
const DWG = 64;                // delivery pass workgroup size
const SLOT = 512;              // dynamic-offset alignment for step uniforms (a multiple of 256), equal to USIZE
const MAX_BATCH = 64;          // steps encoded per submission

const STEP_STRUCT = `
struct Step { t:u32, cur:u32, watch:i32, seed:u32, n:u32, syn:u32, slot:u32, plast:u32,
              decE:f32, decI:f32, itauE:f32, itauI:f32,
              aP:f32, aM:f32, decS:f32, wmax:f32, iEta:f32, iAlpha:f32,
              wdep:f32, refrac:f32,
              stpOn:u32, stpU:f32, decX:f32, decR:f32,
              tin:f32, trip:f32, decY:f32, het:f32,
              refInit:f32, consRate:f32, consW:f32, consP:f32,
              decM:f32, stpScale:f32, commit:f32, poolG:u32,
              vmin:f32, stpOrder:u32, kx:u32, slideK:f32,
              dx0:f32, dx1:f32, dx2:f32, dx3:f32, dx4:f32, dx5:f32,
              ix0:f32, ix1:f32, ix2:f32, ix3:f32, ix4:f32, ix5:f32,
              eE:f32, eI:f32, ex0:f32, ex1:f32, ex2:f32, ex3:f32, ex4:f32, ex5:f32,
              mg0:f32, mg1:f32, mg2:f32, mg3:f32, mg4:f32, mg5:f32, stpScope:u32, stdpAt:u32,
              plastGain:f32, wmu:f32, decH:f32, integ:u32,
              ek:array<vec4<f32>, 5>, exd:array<vec4<f32>, 5>,
              decG:f32, tauGs:f32, gInv:f32, iGlobal:u32,
              decMod:f32, modBase:f32, modDrive:f32, modCur:f32, modPlast:u32, modG:u32, eligPay:f32, decElig:f32, modOn:u32 };
struct Spikes { count:atomic<u32>, total:atomic<u32>, list:array<u32> };
// the cells whose spikes arrive somewhere this step, each with its spike history: filled by the decay pass at the end of the step before, walked by the arrival pass, then cleared
struct Arrivals { count:atomic<u32>, pad:u32, list:array<vec2<u32>> };
`;
const USIZE = 512;  // Step struct bytes (a multiple of 16, as uniforms require; equal to SLOT; the struct is 500 bytes)
const SST = 16;     // state floats per neuron: v, u, gE, gI, six receptor channels, two noise states, the dynamic threshold, the global factor's population flag, the modulator level, padding
// the lanes per cell of the two trace buffers: Kpost (the fast postsynaptic trace, the sliding threshold's rate trace, the modulator level, the cell's Vogels alpha) and ktr (Kpre, the slow trace, a copy of Kpost, the spike history, the modulator level, three spare)
const KL = 4, TL = 8;

const SHADER_UPDATE = STEP_STRUCT + `
@group(0) @binding(0) var<uniform> U: Step;
// Per neuron, 16 floats: a, b, c, d, then the 2007 row's C, k, vr, vt, vpeak, then the graded release threshold, slope and flag, then the dynamic threshold's jump and decay, the conductance adaptation flag and the Poisson flag (MODEL.md 1).
// (the u32 after vmin in the step struct is the short-term plasticity order, stpOrder)
@group(0) @binding(1) var<storage, read> abcd: array<f32>;        // 16n strided
@group(0) @binding(2) var<storage, read_write> st: array<f32>;    // v,u,gE,gI, six receptor conductances, two noise states, the threshold state, the population flag, the modulator level, padding: 16 per neuron
@group(0) @binding(3) var<storage, read> drive: array<f32>;       // eight lanes per cell: dc, namp, the noise fold's four, the modulator sources' increment, the cell's release group as a target
@group(0) @binding(4) var<storage, read_write> ring: array<atomic<i32>>;  // E rows then I rows
@group(0) @binding(5) var<storage, read_write> spikes: Spikes;
@group(0) @binding(6) var<storage, read_write> fired: array<u32>;
@group(0) @binding(7) var<storage, read_write> vtrace: array<f32>;
// Milliseconds of absolute refractoriness left, one per neuron, in the low byte of each word.
// A separate buffer rather than a fifth field of st, because widening that stride touches the state readback and every index in this shader; this takes the update pass to eight storage buffers, which is the guaranteed minimum.
// Feedback inhibition rides in the same buffer for that reason: the pool a cell is in sits in bits 8 to 23 of its word, the modulator group it releases into in the top byte, and past the n words come two halves of per-pool spike counts (the step's parity says which half is read and which is written) and each pool's kick per spike as float bits.
// After those, one word for the global factor's count, then the modulator groups' two count halves and each group's release per spike as float bits (MODEL.md 4).
// Atomic because the counts are.
@group(0) @binding(8) var<storage, read_write> refcnt: array<atomic<u32>>;

// pcg and unif mirror rand.js (noiseDraw) bit for bit in f32; the CUDA kernels carry the same copy.
// Change all three together.
fn pcg(x: u32) -> u32 {
  var s = x * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}
fn unif(x: u32) -> f32 { return f32(pcg(x)) * (1.0 / 4294967295.0); }
// the uniform draw a Poisson row fires on (rand.js poissonDraw): its own salt, so a Poisson cell under a noise stimulus is not correlated with its noise
fn poissonDraw(i: u32, t: u32, seed: u32) -> f32 { return unif(i ^ (t * 2654435761u) ^ seed ^ 0x7f4a7c15u); }
// the magnesium block (Jahr and Stevens 1990): the fraction of a channel's current the voltage lets through, 1 for a channel with no concentration
fn mgGate(mg: f32, v: f32) -> f32 { return select(1.0, 1.0/(1.0 + (mg/3.57)*exp(-0.062*v)), mg > 0.0); }
// The exponential membrane step over h ms (MODEL.md 1), as the reference engine's expSpan: h F(v)/C times (e^x - 1)/x at x = h F'(v)/C, with F(v) = k (v - vr)(v - vt) - adaptation + I + G (v0 - v).
fn expSpan(v: f32, u: f32, I: f32, G: f32, v0: f32, k: f32, vr: f32, vt: f32, C: f32, ac: bool, eI: f32, vmin: f32, h: f32) -> f32 {
  let F = k*(v - vr)*(v - vt) - select(u, u*(v - eI), ac) + I + G*(v0 - v);
  let x = h*(k*(2.0*v - vr - vt) - select(0.0, u, ac) - G)/C;
  let phi = select((exp(x) - 1.0)/x, 1.0 + x*(0.5 + x/6.0), abs(x) < 1e-3);
  return max(v + h*F/C*phi, vmin);
}

// The equations' membrane step (integ 2, MODEL.md 1), as the reference engine's quadTime, quadAt and quadMs.
fn quadTime(A: f32, q: f32, w0: f32, wp: f32) -> f32 {
  if(w0 >= wp){ return 0.0; }
  let y = wp - w0; let x = q + A*w0*wp; let z = A*q;
  if(x > 0.0){
    let R = y/x; let e = z*R*R;
    if(abs(e) < 1e-4){ return R*(1.0 - e/3.0 + e*e/5.0); }
    if(z > 0.0){ let s = sqrt(z); return atan(s*R)/s; }
    let s = sqrt(-z); let r = s*R;
    if(r < 1.0){ return 0.5*log((1.0 + r)/(1.0 - r))/s; }
    return 1e30;
  }
  if(z > 0.0){ let s = sqrt(z); return (1.5707964 + atan(-x/(s*y)))/s; }
  return 1e30;
}
// (w after h, the integral of w over the span)
fn quadAt(A: f32, q: f32, w0: f32, h: f32) -> vec2<f32> {
  let z = A*q; let e = z*h*h;
  var Cs: f32; var Sn: f32; var ls = 0.0;
  if(abs(e) < 1e-4){ Cs = 1.0 - e/2.0 + e*e/24.0; Sn = h*(1.0 - e/6.0 + e*e/120.0); }
  else if(z > 0.0){ let s = sqrt(z); Cs = cos(s*h); Sn = sin(s*h)/s; }
  else { let s = sqrt(-z); let E = exp(-2.0*s*h); Cs = 0.5*(1.0 + E); Sn = 0.5*(1.0 - E)/s; ls = s*h; }
  let phi = Cs - A*w0*Sn;
  return vec2<f32>((w0*Cs + q*Sn)/phi, -(ls + log(abs(phi)))/A);
}
// One millisecond on a quadratic row: (v, u, 1 if it fired, the threshold state after the reset)
fn quadMs(vIn: f32, uIn: f32, I: f32, G: f32, v0: f32, k: f32, vr: f32, vt: f32, C: f32, ra: f32, rb: f32, rc: f32, rd: f32, ac: bool, eA: f32, vmin: f32, vpk: f32, thr: f32, thrJ: f32, thrDec: f32) -> vec4<f32> {
  var v = vIn; var u = uIn; var sp = 0.0; var thN = thr; var vth = vpk + thr;
  let A = k/C; let hd = 0.5*(vt - vr); let sm = vr + vt; let ea = exp(-0.5*ra);
  for(var hs = 0u; hs < 2u; hs++){
    var gs = select(0.0, u, ac) + G; var m = 0.5*sm + gs/(2.0*k);
    var q = (-k*hd*hd + I + select(-u, u*eA, ac) + G*v0 - 0.5*gs*sm - gs*gs/(4.0*k))/C;
    var iv: f32;
    let tc = quadTime(A, q, v - m, vth - m);
    if(tc > 0.5){ let r = quadAt(A, q, v - m, 0.5); v = m + r.x; iv = r.y + 0.5*m; }
    else {
      iv = quadAt(A, q, v - m, tc).y + tc*m;
      let left = 0.5 - tc;
      if(sp > 0.5){ v = vth; iv += left*vth; }
      else {
        sp = 1.0; v = rc; u += rd;
        thN = select(0.0, thrJ*pow(thrDec, 0.5*f32(1u - hs) + left), thrJ > 0.0); vth = vpk + thrJ;
        gs = select(0.0, u, ac) + G; m = 0.5*sm + gs/(2.0*k);
        q = (-k*hd*hd + I + select(-u, u*eA, ac) + G*v0 - 0.5*gs*sm - gs*gs/(4.0*k))/C;
        let t2 = quadTime(A, q, v - m, vth - m);
        if(t2 <= left){ iv += quadAt(A, q, v - m, t2).y + t2*m + (left - t2)*vth; v = vth; }
        else { let r = quadAt(A, q, v - m, left); v = m + r.x; iv += r.y + left*m; }
      }
    }
    v = max(v, vmin);
    let tg = rb*(2.0*iv - vr);
    u = tg + (u - tg)*ea;
  }
  return vec4<f32>(v, u, sp, thN);
}

@compute @workgroup_size(${WG})
fn update(@builtin(global_invocation_id) gid: vec3<u32>){
  let i = gid.x;
  if(i >= U.n){ return; }
  let n = U.n;
  let rowE = U.cur*n + i;
  let rowI = ${ROWS}u*n + rowE;
  let eIn = f32(atomicLoad(&ring[rowE])) * (1.0/${SCALE}.0);
  let iIn = f32(atomicLoad(&ring[rowI])) * (1.0/${SCALE}.0);
  atomicStore(&ring[rowE], 0);
  atomicStore(&ring[rowI], 0);
  var I = drive[i*8u];
  let namp = drive[i*8u+1u];
  if(namp != 0.0){
    let h = i ^ (U.t * 2654435761u) ^ U.seed;
    var d = unif(h) + unif(h ^ 0x9e3779b9u) - 1.0;
    // a stimulus with a time constant filters the draw per cell, an Ornstein-Uhlenbeck process of the same variance; a shared fraction mixes in the stimulus's own draw, filtered the same way (MODEL.md 2)
    let na = drive[i*8u+2u];
    if(na > 0.0){ let z = na*st[i*16u+10u] + drive[i*8u+3u]*d; st[i*16u+10u] = z; d = z; }
    let nc = drive[i*8u+4u];
    if(nc > 0.0){
      let hs = (${NOISE_SHARED}u + u32(drive[i*8u+5u])) ^ (U.t * 2654435761u) ^ U.seed;
      var sdraw = unif(hs) + unif(hs ^ 0x9e3779b9u) - 1.0;
      if(na > 0.0){ let zs = na*st[i*16u+11u] + drive[i*8u+3u]*sdraw; st[i*16u+11u] = zs; sdraw = zs; }
      d = sqrt(1.0 - nc)*d + sqrt(nc)*sdraw;
    }
    I += namp * d;
  }
  // Feedback inhibition: what the cell's pool delivers this millisecond is its kick per spike times the spikes it fired last millisecond, read from the half of the count table the previous step wrote.
  let rcw = atomicLoad(&refcnt[i]);
  let g = (rcw >> 8u) & 0xFFFFu;
  let mg = rcw >> 24u;
  let rc = rcw & 0xFFu;
  var pin = 0.0;
  if(g != 0u){
    let G1 = U.poolG;
    let cnt = atomicLoad(&refcnt[n + ((U.t & 1u) ^ 1u)*G1 + g]);
    pin = bitcast<f32>(atomicLoad(&refcnt[n + 2u*G1 + g])) * f32(cnt);
  }
  // The modulator field (MODEL.md 4): the level as the step before left it is what this step reads, for the synaptic gain and the current here and for the learning passes through the lanes the decay pass copies.
  // The new level is the old one decayed, plus the sources' increment, plus the group's release per spike times the group's spikes of the step before, from the half of its count table the step before wrote.
  let mprev = st[i*16u+14u];
  var mf = 1.0; var mc = 0.0;
  if(U.modOn != 0u){
    mf = 1.0 + U.modDrive*mprev; mc = U.modCur*mprev;
    var mnew = mprev*U.decMod + drive[i*8u+6u];
    let tg = u32(drive[i*8u+7u]);
    if(tg != 0u){
      let MB = n + 3u*U.poolG + 1u;
      let cnt = atomicLoad(&refcnt[MB + ((U.t & 1u) ^ 1u)*U.modG + tg]);
      mnew = mnew + bitcast<f32>(atomicLoad(&refcnt[MB + 2u*U.modG + tg])) * f32(cnt);
    }
    st[i*16u+14u] = mnew;
  }
  var ge = 0.0; var gi = 0.0;
  var gx0 = 0.0;
  var gx1 = 0.0;
  var gx2 = 0.0;
  var gx3 = 0.0;
  var gx4 = 0.0;
  var gx5 = 0.0;
  let v0 = st[i*16u];
  let cnd = U.syn == 2u;
  // the synaptic current is summed apart from the drive, so the modulator field can scale it (mf) and add its current (mc); without a field both are the identity
  var Is = 0.0;
  if(U.syn >= 1u){
    ge = st[i*16u+2u]*U.decE + eIn;
    gi = st[i*16u+3u]*U.decI + iIn + pin;
    // current-based: g times the scale; conductance-based: |g| (E - v)
    Is += select(ge*U.itauE + gi*U.itauI, abs(ge)*(U.eE - v0) + abs(gi)*(U.eI - v0), cnd);
    // Receptor channels 2 and up (MODEL.md 2): their own decay, their own ring rows.
    // Each channel's current passes through its magnesium block, mgGate (MODEL.md 2); a gate of 0 mM lets everything through.
    if(U.kx > 0u){ let rx = (2u*${ROWS}u)*n + rowE; gx0 = st[i*16u+4u]*U.dx0 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); Is += mgGate(U.mg0, v0)*select(gx0*U.ix0, abs(gx0)*(U.ex0 - v0), cnd); }
    if(U.kx > 1u){ let rx = (3u*${ROWS}u)*n + rowE; gx1 = st[i*16u+5u]*U.dx1 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); Is += mgGate(U.mg1, v0)*select(gx1*U.ix1, abs(gx1)*(U.ex1 - v0), cnd); }
    if(U.kx > 2u){ let rx = (4u*${ROWS}u)*n + rowE; gx2 = st[i*16u+6u]*U.dx2 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); Is += mgGate(U.mg2, v0)*select(gx2*U.ix2, abs(gx2)*(U.ex2 - v0), cnd); }
    if(U.kx > 3u){ let rx = (5u*${ROWS}u)*n + rowE; gx3 = st[i*16u+7u]*U.dx3 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); Is += mgGate(U.mg3, v0)*select(gx3*U.ix3, abs(gx3)*(U.ex3 - v0), cnd); }
    if(U.kx > 4u){ let rx = (6u*${ROWS}u)*n + rowE; gx4 = st[i*16u+8u]*U.dx4 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); Is += mgGate(U.mg4, v0)*select(gx4*U.ix4, abs(gx4)*(U.ex4 - v0), cnd); }
    if(U.kx > 5u){ let rx = (7u*${ROWS}u)*n + rowE; gx5 = st[i*16u+9u]*U.dx5 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); Is += mgGate(U.mg5, v0)*select(gx5*U.ix5, abs(gx5)*(U.ex5 - v0), cnd); }
  } else {
    Is += eIn + iIn + pin;
  }
  I += Is*mf + mc;
  // the total synaptic conductance, for the exponential step: the current at v is I + G (v0 - v); the field scales it with the current
  let G = mf*select(0.0, abs(ge) + abs(gi) + mgGate(U.mg0, v0)*abs(gx0) + mgGate(U.mg1, v0)*abs(gx1) + mgGate(U.mg2, v0)*abs(gx2) + mgGate(U.mg3, v0)*abs(gx3) + mgGate(U.mg4, v0)*abs(gx4) + mgGate(U.mg5, v0)*abs(gx5), cnd);
  var v = st[i*16u]; var u = st[i*16u+1u];
  // the dynamic threshold (MODEL.md 1) relaxes before anything else this step, Auryn's order; its state is slot 12, its jump and decay slots 12 and 13 of the row
  let thr = st[i*16u+12u]*abcd[i*16u+13u]; st[i*16u+12u] = thr;
  let A = abcd[i*16u]; let Bq = abcd[i*16u+1u];
  let vr = abcd[i*16u+6u];
  // the recovery variable is measured from the rest (Izhikevich 2007)
  let uRef = vr;
  // Absolute refractoriness, matching the reference engine exactly: the membrane is held at reset and cannot reach threshold whatever the drive, the recovery variable keeps advancing, and the synaptic conductances keep decaying and accumulating.
  if(rc > 0u){
    atomicStore(&refcnt[i], rcw - 1u);
    u += A*(Bq*(v - uRef) - u);
    st[i*16u+1u] = u; st[i*16u+2u] = ge; st[i*16u+3u] = gi; st[i*16u+4u] = gx0; st[i*16u+5u] = gx1; st[i*16u+6u] = gx2; st[i*16u+7u] = gx3; st[i*16u+8u] = gx4; st[i*16u+9u] = gx5;
    if(i32(i) == U.watch){ vtrace[U.slot] = v; }
    return;
  }
  // the membrane floor after each half step, as the reference does
  var qc = false; var qsp = false; var qth = 0.0;
    // Izhikevich 2007: C dv = k (v - vr)(v - vt) - u + I
    let C = abcd[i*16u+4u]; let k = abcd[i*16u+5u]; let vt = abcd[i*16u+7u];
    if(abcd[i*16u+11u] > 0.5){
      // a graded cell's membrane is passive: the leak is the row's slope at rest, k (vt - vr), and the potential is capped at vpeak
      let gl = k*(vt - vr);
      v += 0.5*(-gl*(v - vr) - u + I)/C; v = max(v, U.vmin);
      v += 0.5*(-gl*(v - vr) - u + I)/C; v = max(v, U.vmin);
      v = min(v, abcd[i*16u+8u]);
    } else if(abcd[i*16u+15u] > 0.5){
      // a Poisson cell (MODEL.md 1): no membrane; it fires this step with probability I over 1000, the current read as a rate in hertz, by being put at its peak
      if(I > 0.0 && poissonDraw(i, U.t, U.seed) < I*0.001){ v = abcd[i*16u+8u] + thr; }
    } else if(U.integ >= 1u && vt - abcd[i*16u+8u] > 100.0){
      // the exponential membrane step (integ 1), on a row linear over the spiking range as the reference's linRow
      let ac = abcd[i*16u+14u] > 0.5;
      v = expSpan(v, u, I, G, v0, k, vr, vt, C, ac, U.eI, U.vmin, 0.5);
      v = expSpan(v, u, I, G, v0, k, vr, vt, C, ac, U.eI, U.vmin, 0.5);
    } else if(U.integ == 2u){
      // the equations' step (integ 2) on a quadratic row, as the reference's quadMs
      let r = quadMs(v, u, I, G, v0, k, vr, vt, C, A, Bq, abcd[i*16u+2u], abcd[i*16u+3u], abcd[i*16u+14u] > 0.5, U.eI, U.vmin, abcd[i*16u+8u], thr, abcd[i*16u+12u], abcd[i*16u+13u]);
      v = r.x; u = r.y; qsp = r.z > 0.5; qth = r.w; qc = true;
    } else {
      // with the row's adaptation a conductance (slot 14), u in nanosiemens draws current toward the inhibitory reversal
      let ac = abcd[i*16u+14u] > 0.5;
      v += 0.5*(k*(v - vr)*(v - vt) - select(u, u*(v - U.eI), ac) + I)/C;
      v = max(v, U.vmin);
      v += 0.5*(k*(v - vr)*(v - vt) - select(u, u*(v - U.eI), ac) + I)/C;
      v = max(v, U.vmin);
    }
    if(!qc){ u += A*(Bq*(v - vr) - u); }
  // A graded cell (MODEL.md 1): no spike, no reset; every millisecond it enters the event list with its release as the amplitude byte (0 to 254; a spike carries 255), and the delivery pass scales its synapses.
  if(abcd[i*16u+11u] > 0.5){
    let r = clamp((v - abcd[i*16u+9u])/abcd[i*16u+10u], 0.0, 1.0);
    let ab = u32(r*254.0 + 0.5);
    if(ab > 0u){ let k = atomicAdd(&spikes.count, 1u); spikes.list[k] = i | (ab << 24u); }
    st[i*16u] = v; st[i*16u+1u] = u; st[i*16u+2u] = ge; st[i*16u+3u] = gi; st[i*16u+4u] = gx0; st[i*16u+5u] = gx1; st[i*16u+6u] = gx2; st[i*16u+7u] = gx3; st[i*16u+8u] = gx4; st[i*16u+9u] = gx5;
    if(i32(i) == U.watch){ vtrace[U.slot] = v; }
    return;
  }
  let vpk = abcd[i*16u+8u];
  var spiked = false;
  if(select(v >= vpk + thr, qsp, qc)){
    spiked = true;
    if(qc){
      // the equations' step reset the cell at its crossing
      st[i*16u+12u] = qth;
    } else if(U.integ >= 1u && abcd[i*16u+15u] <= 0.5 && vt - vpk > 100.0){
      // the exponential step's reset inside the millisecond, as the reference engine's spanReset: the distance to the threshold interpolated between the step's opening and its end, the threshold at both ends, the rest of the step from the reset
      let thS = select(thr, thr/abcd[i*16u+13u], thr > 0.0 && abcd[i*16u+13u] > 0.0);
      let dS = st[i*16u] - (vpk + thS); let dE = v - (vpk + thr); let fr = select(1.0, clamp(dS/(dS - dE), 0.0, 1.0), dS < 0.0 && dE > dS); let rest = 1.0 - fr;
      u += abcd[i*16u+3u];
      v = expSpan(abcd[i*16u+2u], u, I, G, v0, k, vr, vt, C, abcd[i*16u+14u] > 0.5, U.eI, U.vmin, rest);
      st[i*16u+12u] = abcd[i*16u+12u]*pow(abcd[i*16u+13u], rest);
    } else {
      v = abcd[i*16u+2u]; u += abcd[i*16u+3u];
      st[i*16u+12u] = abcd[i*16u+12u];
    }
    // the global factor's population count for this step, the word after the pool tables (slot 13 flags the inhibited cells)
    if(st[i*16u+13u] > 0.5){ atomicAdd(&refcnt[n + 3u*U.poolG], 1u); }
    atomicStore(&refcnt[i], (mg << 24u) | (g << 8u) | u32(U.refrac));
    if(g != 0u){ atomicAdd(&refcnt[n + (U.t & 1u)*U.poolG + g], 1u); }
    // a releasing cell's spike, counted for its modulator group, read by the group's targets next step
    if(mg != 0u){ atomicAdd(&refcnt[n + 3u*U.poolG + 1u + (U.t & 1u)*U.modG + mg], 1u); }
    // a count per tick, as the reference and CUDA engines report it: this thread is the only writer of fired[i], so no atomic is needed.
    // A 0/1 flag here would cap every rate read from it, the trainer's probes and synaptic scaling's included, at 1000/steps Hz.
    fired[i] = fired[i] + 1u;
    atomicAdd(&spikes.total, 1u);
    let k = atomicAdd(&spikes.count, 1u);
    spikes.list[k] = i | (255u << 24u);      // index in 24 bits, amplitude byte 255 for a spike
  }
  st[i*16u] = v; st[i*16u+1u] = u; st[i*16u+2u] = ge; st[i*16u+3u] = gi; st[i*16u+4u] = gx0; st[i*16u+5u] = gx1; st[i*16u+6u] = gx2; st[i*16u+7u] = gx3; st[i*16u+8u] = gx4; st[i*16u+9u] = gx5;
  if(i32(i) == U.watch){
    vtrace[U.slot] = select(v, vpk, spiked);
  }
}
`;

// Zero the half of the pool count table this step read, so the next step, whose parity flips, counts into a clean table.
// Dispatched after update.
const SHADER_POOL = STEP_STRUCT + `
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<storage, read_write> refcnt: array<atomic<u32>>;
@compute @workgroup_size(${WG})
fn poolClear(@builtin(global_invocation_id) gid: vec3<u32>){
  let g = gid.x;
  if(g < U.poolG){ atomicStore(&refcnt[U.n + ((U.t & 1u) ^ 1u)*U.poolG + g], 0u); }
  // the modulator groups' half this step read, after the pool tables and the global factor's word
  if(g < U.modG){ atomicStore(&refcnt[U.n + 3u*U.poolG + 1u + ((U.t & 1u) ^ 1u)*U.modG + g], 0u); }
}
`;

const SHADER_CLEAR = STEP_STRUCT + `
@group(0) @binding(0) var<storage, read_write> spikes: Spikes;
@compute @workgroup_size(1)
fn clearCount(){ atomicStore(&spikes.count, 0u); }
`;

const SHADER_IND = STEP_STRUCT + `
@group(0) @binding(0) var<storage, read_write> spikes: Spikes;
@group(0) @binding(1) var<storage, read_write> ind: array<u32>;
@group(0) @binding(2) var<storage, read_write> arrivals: Arrivals;
@compute @workgroup_size(1)
fn writeIndirect(){
  ind[0] = (atomicLoad(&spikes.count) + ${DWG - 1}u) / ${DWG}u;
  ind[1] = 1u; ind[2] = 1u;
  ind[3] = (atomicLoad(&arrivals.count) + ${DWG - 1}u) / ${DWG}u;
  ind[4] = 1u; ind[5] = 1u;
}
`;
// the arrival list is consumed once per step and cleared after its pass
const SHADER_CLEAR_ARR = STEP_STRUCT + `
@group(0) @binding(0) var<storage, read_write> arrivals: Arrivals;
@compute @workgroup_size(1)
fn clearArrivals(){ atomicStore(&arrivals.count, 0u); }
`;

// Short-term plasticity per rule, as classes (stpClassTable in nodes.js): per class U, the two decays and the release scale, its on flag, and the class of each of the 16 rule rows.
// A uniform, since the delivery pass is at the eight storage buffers WebGPU guarantees.
const STP_STRUCT = `
struct StpClasses { par: array<vec4<f32>, 16>, on: array<vec4<u32>, 4>, cls: array<vec4<u32>, 4>, count: vec4<u32> };
`;
const STP_CLASS_BYTES = 400;

const SHADER_DELIVER = STEP_STRUCT + STP_STRUCT + `
struct Part { n0:u32, n1:u32, base:u32, pad:u32 };
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<uniform> P: Part;
@group(0) @binding(2) var<storage, read_write> spikes: Spikes;
@group(0) @binding(3) var<storage, read_write> ring: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read> preStart: array<u32>;
@group(0) @binding(5) var<storage, read> postd: array<u32>;   // post | delay<<27
@group(0) @binding(6) var<storage, read> wgt: array<f32>;
// Tsodyks and Markram short-term plasticity state, per presynaptic neuron rather than per synapse (same as the reference engine: the release factor belongs to the axon, not the contact), so this is two small arrays and not a second copy of the weights.
@group(0) @binding(7) var<storage, read_write> stpX: array<f32>;
@group(0) @binding(8) var<storage, read_write> stpR: array<f32>;
// the plasticity bytes, packed four to a word: the top three bits of each are the synapse's receptor channel (0: by sign into E or I)
@group(0) @binding(9) var<storage, read> rid: array<u32>;
@group(0) @binding(10) var<uniform> SC: StpClasses;

@compute @workgroup_size(${DWG})
fn deliver(@builtin(global_invocation_id) gid: vec3<u32>){
  let k = gid.x;
  if(k >= atomicLoad(&spikes.count)){ return; }
  let e = spikes.list[k];
  let i = e & 0xffffffu;
  let ab = e >> 24u;                          // 255 a spike, else a graded release
  if(i < P.n0 || i >= P.n1){ return; }
  let n = U.n;
  let s0 = preStart[i] - P.base;
  let s1 = preStart[i+1u] - P.base;
  // The release consumes state, so it has to happen exactly once per spiking neuron per step.
  // It does: partitions tile the presynaptic range, so the filter above admits i in exactly one of them, and within that partition one thread owns the whole CSR slice.
  // Deliberately here and not in the update pass, which is already at the eight storage buffers WebGPU guarantees.
  // Order matches the reference, which releases at spike time before delivering that neuron's synapses.
  var sf = 1.0;
  if(ab != 255u){
    sf = f32(ab)/254.0;                      // a graded release: no short-term plasticity
  } else if(U.stpOn == 1u && SC.count.x <= 1u){
    if(U.stpOrder == 1u){ stpR[i] = stpR[i] + U.stpU*(1.0 - stpR[i]); }   // stpOrder 1: facilitation before release
    let rel = stpR[i]*stpX[i];
    stpX[i] = stpX[i] - rel;                 // depletion
    if(U.stpOrder == 0u){ stpR[i] = stpR[i] + U.stpU*(1.0 - stpR[i]); }   // as published: facilitation after release
    sf = rel*U.stpScale;
  }
  // With more than one class, each class releases from its own state at [c*n + i], and a synapse takes its rule's class.
  let C = SC.count.x;
  var rel: array<f32, 16>;
  let multi = ab == 255u && U.stpOn == 1u && C > 1u;
  if(multi){
    for(var c = 0u; c < C; c++){
      var f = 1.0;
      if(SC.on[c >> 2u][c & 3u] != 0u){
        let k = c*n + i;
        let Uc = SC.par[c].x;
        if(U.stpOrder == 1u){ stpR[k] = stpR[k] + Uc*(1.0 - stpR[k]); }
        let r = stpR[k]*stpX[k];
        stpX[k] = stpX[k] - r;
        if(U.stpOrder == 0u){ stpR[k] = stpR[k] + Uc*(1.0 - stpR[k]); }
        f = r*SC.par[c].w;
      }
      rel[c] = f;
    }
  }
  // stpScope confines a spike's release to one sign of weight; a graded release is not short-term plasticity and applies to every synapse
  let spike = ab == 255u;
  var sfE = select(sf, 1.0, spike && U.stpScope == 2u);
  var sfI = select(sf, 1.0, spike && U.stpScope == 1u);
  for(var s = s0; s < s1; s++){
    let pd = postd[s];
    let j = pd & 0x7ffffffu;
    let d = pd >> 27u;
    let row = (U.cur + d) % ${ROWS}u;
    let byte = (rid[s >> 2u] >> ((s & 3u) * 8u)) & 0xffu;
    if(multi){
      let r5 = byte & 31u;
      let f = rel[SC.cls[r5 >> 2u][r5 & 3u]];
      sfE = select(f, 1.0, U.stpScope == 2u);
      sfI = select(f, 1.0, U.stpScope == 1u);
    }
    // Route on the stored weight, not the scaled one: the release factor is never negative but can be zero, and a zero must not be read as inhibitory.
    let w0 = wgt[s];
    let fx = i32(round(w0 * select(sfI, sfE, w0 > 0.0) * ${SCALE}.0));
    let ch = byte >> 5u;
    if(ch >= 2u){ atomicAdd(&ring[ch*${ROWS}u*n + row*n + j], fx); }
    else if(w0 > 0.0){ atomicAdd(&ring[row*n + j], fx); }
    else { atomicAdd(&ring[${ROWS}u*n + row*n + j], fx); }
  }
}
`;

// STDP + homeostasis, same rules as simworker.plastOnSpike.
// Outgoing: each spiking neuron owns its CSR slice, so weight writes never race.
// Incoming: each synapse is visited only when its unique postsynaptic neuron spikes.
const SHADER_PLAST_OUT = STEP_STRUCT + `
struct Part { n0:u32, n1:u32, base:u32, sEnd:u32 };
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<uniform> P: Part;
@group(0) @binding(2) var<storage, read_write> spikes: Spikes;
@group(0) @binding(3) var<storage, read> preStart: array<u32>;
@group(0) @binding(4) var<storage, read> postd: array<u32>;
@group(0) @binding(5) var<storage, read_write> wgt: array<f32>;
@group(0) @binding(6) var<storage, read> Kpost: array<f32>;
@group(0) @binding(7) var<storage, read> rid: array<u32>;   // 1 rule byte per local synapse, four to a word
// the eligibility trace per synapse (modPlast 2, MODEL.md 4), partitioned like the weights; a placeholder when the form is off, never read since the pass guards on U.modPlast
@group(0) @binding(8) var<storage, read_write> elig: array<f32>;
// The consolidated reference, read here as well as in the incoming pass: a committed synapse has to be skipped by both halves of the update or it is frozen against potentiation and still open to depression.
@group(0) @binding(9) var<storage, read> wRef: array<f32>;
@group(0) @binding(10) var<uniform> RT: Rules;

// The per-rule table, indexed by the byte the wiring put on each synapse.
// A uniform rather than a storage buffer on purpose: the incoming pass is at the eight storage buffers WebGPU guarantees, and uniforms are a separate budget.
// Twelve floats a rule, the eleven fields plus one of padding, so each element is 48 bytes and satisfies the sixteen byte stride a uniform array requires.
// Sixteen rules is the ceiling here, well under the 255 the byte allows, and buildRuleTable is where that limit is stated.
struct Rule { aP:f32, aM:f32, wmax:f32, wdep:f32,
  trip:f32, het:f32, tin:f32, iEta:f32,
  cons:f32, consW:f32, consP:f32, pad:f32 };
struct Rules { r: array<Rule, 16> };

// Soft weight bounds, the same rule the reference engine applies.
// Each update is scaled by the room left in the direction it is heading, so a homeostatic target the network cannot reach saturates instead of walking every inhibitory weight into the clamp.
fn softBound(wv: f32, delta: f32, wmax: f32, on: f32) -> f32 {
  let a = abs(wv);
  let away = (delta < 0.0) == (wv < 0.0);
  let room = select(a/wmax, (wmax - a)/wmax, away);
  return delta * mix(1.0, max(0.0, room), on);
}
@compute @workgroup_size(${DWG})
fn plastOut(@builtin(global_invocation_id) gid: vec3<u32>){
  let k = gid.x;
  if(k >= atomicLoad(&spikes.count)){ return; }
  let e = spikes.list[k];
  if((e >> 24u) != 255u){ return; }            // a graded release is not a spike
  let i = e & 0xffffffu;
  if(i < P.n0 || i >= P.n1){ return; }
  let s0 = preStart[i] - P.base;
  let s1 = preStart[i+1u] - P.base;
  for(var s = s0; s < s1; s++){
    let ri = (rid[s >> 2u] >> ((s & 3u) * 8u)) & 0x1fu;   // the rule: the low five bits
    if(ri == 0u){ continue; }                    // 0 is frozen
    let R = RT.r[ri];
    // committed: its reference has reached the upper well and it is out of the plastic pool, keeping its weight and still transmitting
    if(U.commit > 0.5 && R.consW > 0.0 && wRef[s] >= R.consW*0.9){ continue; }
    let wj = wgt[s];
    let j = postd[s] & 0x7ffffffu;
    // the sliding threshold: depression scaled by the square of the cell's slow rate estimate (the second lane of Kpost) over the target, 1 when off
    let kh = Kpost[4u*j + 1u]*U.slideK;
    let sl = select(1.0, kh*kh, U.slideK > 0.0);
    // the third factor: the gain, times modBase + the modulator level of the postsynaptic cell (the third lane) under modPlast 1 (MODEL.md 4)
    let g = U.plastGain*select(1.0, U.modBase + Kpost[4u*j + 2u], U.modPlast == 1u);
    if(wj > 0.0){
      let dep = g*R.aM*sl*Kpost[4u*j]*mix(1.0, select(pow(wj, U.wmu), wj, U.wmu == 1.0), R.wdep);
      // under the eligibility form the Hebbian terms accumulate on the trace, paid into the weight at the sweep
      if(U.modPlast == 2u){ elig[s] = elig[s] - dep + g*R.tin; }
      else { wgt[s] = min(R.wmax, max(1e-4, wj - dep + g*R.tin)); }
    }
    else {
      // under the global factor (U.iGlobal) the amplitude is the rule's eta times the population ratio and the presynaptic term is the trace plus one, the authors' xi; the cell's Vogels alpha is the fourth lane
      let d = softBound(wj, select(-R.iEta*(Kpost[4u*j] - Kpost[4u*j + 3u]), -R.iEta*Kpost[4u*U.n + 1u]*(Kpost[4u*j] + 1.0), U.iGlobal == 1u), R.wmax, R.wdep);
      wgt[s] = max(-R.wmax, min(-1e-4, wj + d));
    }
  }
}
`;

// The pair rule timed at the synapse (stdpAt 1): the outgoing half runs when a spike arrives, d steps after it was fired, against Kpost as it stands at the start of that step.
// The decay pass lists each cell whose history meets the delays its synapses carry, with the history; here every synapse whose delay bit is set in that history is depressed.
// The same bindings as the pass at the soma, the list in place of the spikes.
const SHADER_PLAST_ARRIVE = STEP_STRUCT + `
struct Part { n0:u32, n1:u32, base:u32, sEnd:u32 };
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<uniform> P: Part;
@group(0) @binding(2) var<storage, read_write> arrivals: Arrivals;
@group(0) @binding(3) var<storage, read> preStart: array<u32>;
@group(0) @binding(4) var<storage, read> postd: array<u32>;
@group(0) @binding(5) var<storage, read_write> wgt: array<f32>;
@group(0) @binding(6) var<storage, read> Kpost: array<f32>;
@group(0) @binding(7) var<storage, read> rid: array<u32>;
@group(0) @binding(8) var<storage, read_write> elig: array<f32>;
@group(0) @binding(9) var<storage, read> wRef: array<f32>;
@group(0) @binding(10) var<uniform> RT: Rules;
struct Rule { aP:f32, aM:f32, wmax:f32, wdep:f32,
  trip:f32, het:f32, tin:f32, iEta:f32,
  cons:f32, consW:f32, consP:f32, pad:f32 };
struct Rules { r: array<Rule, 16> };
fn softBound(wv: f32, delta: f32, wmax: f32, on: f32) -> f32 {
  let a = abs(wv);
  let away = (delta < 0.0) == (wv < 0.0);
  let room = select(a/wmax, (wmax - a)/wmax, away);
  return delta * mix(1.0, max(0.0, room), on);
}
@compute @workgroup_size(${DWG})
fn plastArrive(@builtin(global_invocation_id) gid: vec3<u32>){
  let k = gid.x;
  if(k >= atomicLoad(&arrivals.count)){ return; }
  let e = arrivals.list[k];
  let i = e.x;
  let h = e.y;
  if(i < P.n0 || i >= P.n1){ return; }
  let s0 = preStart[i] - P.base;
  let s1 = preStart[i+1u] - P.base;
  for(var s = s0; s < s1; s++){
    let pd = postd[s];
    if(((h >> (pd >> 27u)) & 1u) == 0u){ continue; }   // no spike of this cell arrives through this synapse now
    let ri = (rid[s >> 2u] >> ((s & 3u) * 8u)) & 0x1fu;
    if(ri == 0u){ continue; }
    let R = RT.r[ri];
    if(U.commit > 0.5 && R.consW > 0.0 && wRef[s] >= R.consW*0.9){ continue; }
    let wj = wgt[s];
    let j = pd & 0x7ffffffu;
    // the sliding threshold: depression scaled by the square of the cell's slow rate estimate (the second lane of Kpost) over the target, 1 when off
    let kh = Kpost[4u*j + 1u]*U.slideK;
    let sl = select(1.0, kh*kh, U.slideK > 0.0);
    // the third factor: the gain, times modBase + the modulator level of the postsynaptic cell (the third lane) under modPlast 1 (MODEL.md 4)
    let g = U.plastGain*select(1.0, U.modBase + Kpost[4u*j + 2u], U.modPlast == 1u);
    if(wj > 0.0){
      let dep = g*R.aM*sl*Kpost[4u*j]*mix(1.0, select(pow(wj, U.wmu), wj, U.wmu == 1.0), R.wdep);
      // under the eligibility form the Hebbian terms accumulate on the trace, paid into the weight at the sweep
      if(U.modPlast == 2u){ elig[s] = elig[s] - dep + g*R.tin; }
      else { wgt[s] = min(R.wmax, max(1e-4, wj - dep + g*R.tin)); }
    }
    else {
      // under the global factor (U.iGlobal) the amplitude is the rule's eta times the population ratio and the presynaptic term is the trace plus one, the authors' xi; the cell's Vogels alpha is the fourth lane
      let d = softBound(wj, select(-R.iEta*(Kpost[4u*j] - Kpost[4u*j + 3u]), -R.iEta*Kpost[4u*U.n + 1u]*(Kpost[4u*j] + 1.0), U.iGlobal == 1u), R.wmax, R.wdep);
      wgt[s] = max(-R.wmax, min(-1e-4, wj + d));
    }
  }
}
`;

const SHADER_PLAST_IN = STEP_STRUCT + `
struct Part { n0:u32, n1:u32, base:u32, sEnd:u32 };
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<uniform> P: Part;
@group(0) @binding(2) var<storage, read_write> spikes: Spikes;
@group(0) @binding(3) var<storage, read> inStart: array<u32>;
// the reverse index: for each entry the synapse, and the presynaptic cell with the synapse's delay in the top five bits, as one pair; interleaved to keep this pass at the eight storage buffers WebGPU guarantees once the eligibility trace is bound
@group(0) @binding(4) var<storage, read> rev: array<vec2<u32>>;
// the eligibility trace per synapse (modPlast 2, MODEL.md 4), partitioned like the weights; a placeholder when the form is off
@group(0) @binding(5) var<storage, read_write> elig: array<f32>;
@group(0) @binding(6) var<storage, read_write> wgt: array<f32>;
// Kpre, the slow trace, the fast postsynaptic trace, the spike history and the modulator level interleaved, eight floats per neuron: [i*8] is the fast presynaptic trace, [i*8+1] the slow postsynaptic one for the triplet term, [i*8+2] a copy of Kpost as the decay pass left it (the heterosynaptic term's trace), [i*8+3] the history as an integer in a float (bit k for a spike k steps ago), [i*8+4] the modulator level as the step before left it.
// They share a binding because this pass is at the eight storage buffers WebGPU guarantees.
// Kpost is not in here: it is the hot read in the outgoing pass, so it stays its own array, four lanes per cell: the fast postsynaptic trace, the slow rate trace the sliding threshold reads, the modulator level and the cell's Vogels alpha.
// The slow trace is read before the bump pass adds this spike, so potentiation is scaled by the postsynaptic history and not by the present spike, matching the reference.
@group(0) @binding(7) var<storage, read> ktr: array<f32>;
@group(0) @binding(8) var<storage, read> rid: array<u32>;
@group(0) @binding(10) var<uniform> RT: Rules;
// the per-rule table, laid out as in the outgoing pass
struct Rule { aP:f32, aM:f32, wmax:f32, wdep:f32,
  trip:f32, het:f32, tin:f32, iEta:f32,
  cons:f32, consW:f32, consP:f32, pad:f32 };
struct Rules { r: array<Rule, 16> };

// Reference weight the heterosynaptic term regresses toward, one per synapse and partitioned exactly like the weights.
// This is the binding the trace interleave freed; without it this pass would need a ninth storage buffer and eight is the guaranteed maximum.
@group(0) @binding(9) var<storage, read> wRef: array<f32>;

// soft weight bounds, as in the outgoing pass
fn softBound(wv: f32, delta: f32, wmax: f32, on: f32) -> f32 {
  let a = abs(wv);
  let away = (delta < 0.0) == (wv < 0.0);
  let room = select(a/wmax, (wmax - a)/wmax, away);
  return delta * mix(1.0, max(0.0, room), on);
}
@compute @workgroup_size(${DWG})
fn plastIn(@builtin(global_invocation_id) gid: vec3<u32>){
  let k = gid.x;
  if(k >= atomicLoad(&spikes.count)){ return; }
  let e = spikes.list[k];
  if((e >> 24u) != 255u){ return; }            // a graded release is not a spike
  let i = e & 0xffffffu;
  let z = ktr[i*8u + 1u];
  // The heterosynaptic term (Zenke, Agnes and Gerstner 2015): beta times the cube of the cell's fast postsynaptic trace as it stood before this spike, which the third lane carries from the decay pass; no normalization and no ceiling, as in the reference engine.
  let kq = ktr[i*8u + 2u];
  let hetCube = kq*kq*kq;
  // the third factor: the gain, times modBase + this cell's modulator level (the fifth lane) under modPlast 1 (MODEL.md 4)
  let g = U.plastGain*select(1.0, U.modBase + ktr[i*8u + 4u], U.modPlast == 1u);
  for(var r = inStart[i]; r < inStart[i+1u]; r++){
    let s = rev[r].x;
    if(s < P.base || s >= P.sEnd){ continue; }
    let sl = s - P.base;
    let ri = (rid[sl >> 2u] >> ((sl & 3u) * 8u)) & 0x1fu;
    if(ri == 0u){ continue; }
    let R = RT.r[ri];
    if(U.commit > 0.5 && R.consW > 0.0 && wRef[sl] >= R.consW*0.9){ continue; }
    // the presynaptic cell and the synapse's delay, packed in the pair's second word
    let p = rev[r].y & 0x7ffffffu;
    var kp = ktr[p*8u];
    if(U.stdpAt == 1u){
      // timed at the synapse: the pre trace as it was d steps ago, the spikes of the last d - 1 steps taken out and the rest brought forward by e^(d/tau+)
      let d = rev[r].y >> 27u;
      let hp = u32(ktr[p*8u + 3u]);
      var sub = 0.0;
      for(var q = 1u; q < d; q++){ if(((hp >> q) & 1u) != 0u){ sub = sub + U.ek[q >> 2u][q & 3u]; } }
      kp = max(0.0, U.exd[d >> 2u][d & 3u]*(kp - sub));
    }
    let wp = wgt[sl];
    if(wp > 0.0){
      // One clamp on the combined result, and the pull reads the weight from before this step's potentiation, both as the reference does.
      // Under the eligibility form the Hebbian potentiation goes to the trace; the heterosynaptic pull is homeostatic and moves the weight now.
      var nw = wp;
      if(U.modPlast == 2u){ elig[sl] = elig[sl] + g*(R.aP + R.trip*z)*kp; }
      else { nw = wp + g*(R.aP + R.trip*z)*kp; }
      if(R.het > 0.0){ nw = nw - R.het*hetCube*(wp - wRef[sl]); }
      wgt[sl] = min(R.wmax, max(1e-4, nw));
    }
    else {
      let d = softBound(wp, select(-R.iEta*kp, -R.iEta*ktr[8u*U.n]*kp, U.iGlobal == 1u), R.wmax, R.wdep);
      wgt[sl] = max(-R.wmax, wp + d);
    }
  }
}
`;

// Consolidation, Zenke, Agnes and Gerstner 2015 equation 16: the reference weight follows the weight through a double well whose lower fixed point is zero and whose upper one is consW, with the midpoint unstable.
// Excitatory plastic synapses only; inhibitory weights are governed by the Vogels rule and have no reference.
// One thread per synapse in the partition, on the published cadence rather than every step, matching consolidate() in simworker.js.
const SHADER_CONS = STEP_STRUCT + `
struct Part { n0:u32, n1:u32, base:u32, sEnd:u32 };
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<uniform> P: Part;
@group(0) @binding(2) var<storage, read> wgt: array<f32>;
@group(0) @binding(3) var<storage, read_write> wRef: array<f32>;
@group(0) @binding(4) var<storage, read> rid: array<u32>;
@group(0) @binding(5) var<uniform> RT: Rules;
// the per-rule table, laid out as in the outgoing pass
struct Rule { aP:f32, aM:f32, wmax:f32, wdep:f32,
  trip:f32, het:f32, tin:f32, iEta:f32,
  cons:f32, consW:f32, consP:f32, pad:f32 };
struct Rules { r: array<Rule, 16> };

@compute @workgroup_size(${WG})
fn consolidate(@builtin(global_invocation_id) gid: vec3<u32>){
  let s = gid.x;
  if(s >= P.sEnd - P.base){ return; }
  let ri = (rid[s >> 2u] >> ((s & 3u) * 8u)) & 0x1fu;   // the rule: the low five bits
  if(ri == 0u){ return; }
  let R = RT.r[ri];
  if(R.cons <= 0.0){ return; }        // a rule that does not consolidate
  let wj = wgt[s];
  if(wj <= 0.0){ return; }
  let r = wRef[s];
  // held in [0, max(r, w, consW)], which the exact equation cannot leave, as in the reference: the step is forward Euler over a cubic and a short timescale otherwise diverges
  let nr = r + U.consRate*(wj - r - R.consP*r*(R.consW*0.5 - r)*(R.consW - r));
  wRef[s] = clamp(nr, 0.0, max(max(r, wj), R.consW));
}
// The reference starts inside its well: a weight above consW under a consolidating rule takes consW, as refIntoWell() in simworker.js.
@compute @workgroup_size(${WG})
fn initRef(@builtin(global_invocation_id) gid: vec3<u32>){
  let s = gid.x;
  if(s >= P.sEnd - P.base){ return; }
  let ri = (rid[s >> 2u] >> ((s & 3u) * 8u)) & 0x1fu;
  if(ri == 0u){ return; }
  let R = RT.r[ri];
  if(R.cons <= 0.0 || R.consW <= 0.0){ return; }
  if(wgt[s] <= 0.0){ return; }
  let mode = u32(U.refInit + 0.5);
  if(mode == 1u){ wRef[s] = 0.0; }
  else if(mode == 2u){ wRef[s] = R.consW; }
  else if(wRef[s] > R.consW){ wRef[s] = R.consW; }
}
`;

// The eligibility sweep (modPlast 2, MODEL.md 4), as eligSweep() in simworker.js: every modStep ms each excitatory plastic synapse's trace pays (1 - e^(-modStep/eligTau)) of itself into its weight times modBase + the modulator level of its postsynaptic cell, then decays.
// One thread per synapse in the partition, after the decay pass of the step, so the level read (Kpost's third lane) is the step's own, as in the reference.
const SHADER_ELIG = STEP_STRUCT + `
struct Part { n0:u32, n1:u32, base:u32, sEnd:u32 };
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<uniform> P: Part;
@group(0) @binding(2) var<storage, read_write> wgt: array<f32>;
@group(0) @binding(3) var<storage, read_write> elig: array<f32>;
@group(0) @binding(4) var<storage, read> postd: array<u32>;
@group(0) @binding(5) var<storage, read> rid: array<u32>;
@group(0) @binding(6) var<storage, read> Kpost: array<f32>;
@group(0) @binding(7) var<storage, read> wRef: array<f32>;
@group(0) @binding(8) var<uniform> RT: Rules;
struct Rule { aP:f32, aM:f32, wmax:f32, wdep:f32,
  trip:f32, het:f32, tin:f32, iEta:f32,
  cons:f32, consW:f32, consP:f32, pad:f32 };
struct Rules { r: array<Rule, 16> };

@compute @workgroup_size(${WG})
fn eligSweep(@builtin(global_invocation_id) gid: vec3<u32>){
  let s = gid.x;
  if(s >= P.sEnd - P.base){ return; }
  let ri = (rid[s >> 2u] >> ((s & 3u) * 8u)) & 0x1fu;
  if(ri == 0u){ return; }
  let c = elig[s];
  if(c == 0.0){ return; }
  let R = RT.r[ri];
  // a committed synapse keeps its weight; its trace still decays
  if(!(U.commit > 0.5 && R.consW > 0.0 && wRef[s] >= R.consW*0.9)){
    let wj = wgt[s];
    if(wj > 0.0){ wgt[s] = min(R.wmax, max(1e-4, wj + c*U.eligPay*(U.modBase + Kpost[4u*(postd[s] & 0x7ffffffu) + 2u]))); }
  }
  elig[s] = c*U.decElig;
}
`;

// Homeostatic synaptic scaling: each neuron's plastic excitatory inputs scale a small step toward its target rate, with the factors computed on the CPU from the fired readback once per simulated second.
const SHADER_SCALE = STEP_STRUCT + `
struct Part { n0:u32, n1:u32, base:u32, sEnd:u32 };
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<uniform> P: Part;
@group(0) @binding(2) var<storage, read_write> wgt: array<f32>;
@group(0) @binding(3) var<storage, read> postd: array<u32>;
@group(0) @binding(4) var<storage, read> rid: array<u32>;
@group(0) @binding(5) var<storage, read> factor: array<f32>;
@group(0) @binding(6) var<uniform> RT: Rules;
// the reference weight, so a committed synapse is left where it is: it has left the plastic pool, and scaling is part of that pool (MODEL.md 4, terms in combination); a placeholder when the terms are off, never read because the test guards on U.commit
@group(0) @binding(7) var<storage, read> wRef: array<f32>;
// the per-rule table, laid out as in the outgoing pass
struct Rule { aP:f32, aM:f32, wmax:f32, wdep:f32,
  trip:f32, het:f32, tin:f32, iEta:f32,
  cons:f32, consW:f32, consP:f32, pad:f32 };
struct Rules { r: array<Rule, 16> };

@compute @workgroup_size(${WG})
fn scalePass(@builtin(global_invocation_id) gid: vec3<u32>){
  let s = gid.x;
  if(s >= P.sEnd - P.base){ return; }
  let ri = (rid[s >> 2u] >> ((s & 3u) * 8u)) & 0x1fu;   // the rule: the low five bits
  if(ri == 0u){ return; }
  let w = wgt[s];
  if(w <= 0.0){ return; }
  if(U.commit > 0.5 && RT.r[ri].consW > 0.0 && wRef[s] >= RT.r[ri].consW*0.9){ return; }
  wgt[s] = min(RT.r[ri].wmax, w * factor[postd[s] & 0x7ffffffu]);
}
`;

const SHADER_BUMP = STEP_STRUCT + `
@group(0) @binding(0) var<storage, read_write> spikes: Spikes;
@group(0) @binding(1) var<storage, read_write> ktr: array<f32>;
@group(0) @binding(2) var<storage, read_write> Kpost: array<f32>;
@compute @workgroup_size(${DWG})
fn bump(@builtin(global_invocation_id) gid: vec3<u32>){
  let k = gid.x;
  if(k >= atomicLoad(&spikes.count)){ return; }
  let e = spikes.list[k];
  if((e >> 24u) != 255u){ return; }            // a graded release is not a spike
  let i = e & 0xffffffu;
  ktr[i*8u] = ktr[i*8u] + 1.0;
  Kpost[4u*i] = Kpost[4u*i] + 1.0;
  Kpost[4u*i + 1u] = Kpost[4u*i + 1u] + 1.0;
  ktr[i*8u + 1u] = ktr[i*8u + 1u] + 1.0;
  ktr[i*8u + 3u] = ktr[i*8u + 3u] + 1.0;    // this step's spike, bit 0 of the history (clear since the shift)
}
`;

// Each cell's Vogels alpha into the fourth lane of Kpost, where the outgoing passes read it (their alpha binding carries the eligibility trace).
// Dispatched before the first step after alpha is written (a tune of the target, the end of a set point calibration), so the next pairing reads the new alpha as it did when the passes bound alpha itself; the decay pass copies it every step after.
const SHADER_ALPHA = STEP_STRUCT + `
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<storage, read_write> Kpost: array<f32>;
@group(0) @binding(2) var<storage, read> alphaI: array<f32>;
@compute @workgroup_size(${WG})
fn copyAlpha(@builtin(global_invocation_id) gid: vec3<u32>){
  let i = gid.x;
  if(i >= U.n){ return; }
  Kpost[4u*i + 3u] = alphaI[i];
}
`;

// Per-millisecond recovery of both short-term plasticity variables.
// Exact exponentials rather than Euler, matching the reference, so the time constants hold whatever the step size.
// One thread per cell and class, the class from the index (the state is at [c*n + i]); a class that is off is left alone.
const SHADER_STP = STEP_STRUCT + STP_STRUCT + `
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<storage, read_write> stpX: array<f32>;
@group(0) @binding(2) var<storage, read_write> stpR: array<f32>;
@group(0) @binding(3) var<uniform> SC: StpClasses;
@compute @workgroup_size(${WG})
fn stpRecover(@builtin(global_invocation_id) gid: vec3<u32>){
  let k = gid.x;
  let c = k / U.n;
  if(c >= max(SC.count.x, 1u)){ return; }
  if(SC.on[c >> 2u][c & 3u] == 0u){ return; }
  let p = SC.par[c];
  stpX[k] = 1.0 - (1.0 - stpX[k])*p.y;
  stpR[k] = p.x + (stpR[k] - p.x)*p.z;
}
`;

// The third lane of the interleave carries a copy of the fast postsynaptic trace, the heterosynaptic term's trace, copied here every step: the incoming pass reads the interleave and has no binding to spare.
// The fourth lane is the spike history, moved on here; a cell whose history meets the delays its synapses carry (dmask, bit d for a delay of d) has a spike arriving next step and is listed for the arrival pass when the rule is timed at the synapse.
// The fifth lane, and Kpost's third, take the modulator level from the state buffer (the update pass wrote this step's), so the learning passes of the next step read the level as the step before left it, as the reference does; Kpost's fourth lane takes the cell's Vogels alpha, so the outgoing passes bind the eligibility trace in alpha's place.
const SHADER_DECAY = STEP_STRUCT + `
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<storage, read_write> ktr: array<f32>;
@group(0) @binding(2) var<storage, read_write> Kpost: array<f32>;
@group(0) @binding(3) var<storage, read> alphaI: array<f32>;
@group(0) @binding(4) var<storage, read> dmask: array<u32>;
@group(0) @binding(5) var<storage, read_write> arrivals: Arrivals;
@group(0) @binding(6) var<storage, read_write> refcnt: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> st: array<f32>;
@compute @workgroup_size(${WG})
fn decay(@builtin(global_invocation_id) gid: vec3<u32>){
  let i = gid.x;
  if(i >= U.n){ return; }
  // The global factor (MODEL.md 4): one thread folds this step's count of the inhibited population's spikes into the low-passed count (Kpost's extra lane) and writes the ratio the next step's inhibitory updates read, after the Kpost lanes for the outgoing passes and after the ktr lanes for the incoming one.
  if(i == 0u && U.gInv > 0.0){
    let cnt = atomicExchange(&refcnt[U.n + 3u*U.poolG], 0u);
    let pf = Kpost[4u*U.n]*U.decG + f32(cnt)/U.tauGs;
    let ratio = pf*U.gInv - 1.0;
    Kpost[4u*U.n] = pf; Kpost[4u*U.n + 1u] = ratio; ktr[8u*U.n] = ratio;
  }
  ktr[i*8u] = ktr[i*8u] * U.decS;
  Kpost[4u*i] = Kpost[4u*i] * U.decM;
  Kpost[4u*i + 1u] = Kpost[4u*i + 1u] * U.decH;
  ktr[i*8u + 1u] = ktr[i*8u + 1u] * U.decY;
  ktr[i*8u + 2u] = Kpost[4u*i];
  let mlev = st[i*16u + 14u];
  Kpost[4u*i + 2u] = mlev; ktr[i*8u + 4u] = mlev;
  Kpost[4u*i + 3u] = alphaI[i];
  let h = (u32(ktr[i*8u + 3u]) << 1u) & 0x1ffffu;
  ktr[i*8u + 3u] = f32(h);
  if(U.stdpAt == 1u && (h & dmask[i]) != 0u){
    let k = atomicAdd(&arrivals.count, 1u);
    arrivals.list[k] = vec2<u32>(i, h);
  }
}
`;

let S = null;

function scaleOf(p, t){
  const dt = t - p.t0;
  if(dt < 0) return 0;
  if(p.duration > 0 && dt >= p.duration) return 0;
  if(p.mode === 1) return (dt % p.period) < p.width ? 1 : 0;
  if(p.mode === 2) return Math.min(1, dt / p.period);
  return 1;
}

// recompute the packed drive array (dc = bias + DC protocols, namp = noise amplitude) only when a protocol scale actually changes; returns true when the GPU copy needs re-upload
function updateDrive(){
  let changed = false;
  for(const p of S.allProtos){
    const s = scaleOf(p, S.t);
    if(s !== p._s){ p._s = s; changed = true; }
  }
  for(const p of S.modSrc){
    const s = scaleOf(p, S.t);
    if(s !== p._s){ p._s = s; changed = true; }
  }
  if(!changed && S.driveValid && !S.extDirty) return false;
  // eight lanes per cell: the DC drive, the noise amplitude, then the noise stimulus's decay, its complement, its shared fraction and its index (src/noisefold.js), the same floats the other engines fold, then the modulator sources' increment this step and the group the cell receives a released modulator from
  const d = S.driveHost, n = S.n, bias = S.bias, F = S.noise, MT = S.modTgt;
  for(let i=0;i<n;i++){
    d[i*8] = bias[i]; d[i*8+1] = 0;
    d[i*8+2] = F ? F.a[i] : 0; d[i*8+3] = F ? F.k[i] : 0; d[i*8+4] = F ? F.c[i] : 0; d[i*8+5] = F && F.id[i] >= 0 ? F.id[i] : 0;
    d[i*8+6] = 0; d[i*8+7] = MT ? MT[i] : 0;
  }
  for(const p of S.allProtos){
    if(!p._s) continue;
    // gain: the stimulus's per-cell multiplier (spread), as the other two engines read it
    const k = p.amp * p._s, off = p.mode === 3 ? 1 : 0, idx = p.idx, g = p.gain;
    for(let q=0;q<idx.length;q++) d[idx[q]*8 + off] += g ? k*g[q] : k;
  }
  // the modulator sources' increment per step, the level times (1 - the field's decay) folded at configure, as the reference engine adds it
  for(const p of S.modSrc){
    if(!p._s) continue;
    const k = p.amp * p._s, idx = p.idx;
    for(let q=0;q<idx.length;q++) d[idx[q]*8 + 6] += k;
  }
  // framed input maps fold into the DC half of the drive array
  for(const im of (S.inputs || [])){
    const { chStart, chIdx, chW, gains, amp } = im;
    const NC = Math.min(gains.length, chStart.length - 1);   // stale frames after a retune
    for(let c=0;c<NC;c++){
      const g = gains[c] * amp;
      if(!g) continue;
      for(let k=chStart[c];k<chStart[c+1];k++) d[chIdx[k]*8] += g * chW[k];
    }
  }
  S.extDirty = false;
  S.driveValid = true;
  return true;
}

function configure(m){
  if(m.bias) S.bias = m.bias;
  S.syn = Math.max(0, Math.min(2, m.syn|0));   // 0 kick, 1 exp, 2 conductance (exp with reversals)
  S.eRevE = Number.isFinite(+m.eRevE) ? +m.eRevE : 0;
  S.eRevI = Number.isFinite(+m.eRevI) ? +m.eRevI : -70;
  if(Array.isArray(m.chanErev) || ArrayBuffer.isView(m.chanErev)) S.chanErev = Array.from(m.chanErev);
  if(!S.chanErev) S.chanErev = [];
  // the magnesium block per channel, mM; 0 is no gate (MODEL.md 2)
  if(Array.isArray(m.chanMg) || ArrayBuffer.isView(m.chanMg)) S.chanMg = Array.from(m.chanMg, x => x > 0 ? +x : 0);
  if(!S.chanMg) S.chanMg = [];
  const tE = m.tauE > 0 ? m.tauE : 3, tI = m.tauI > 0 ? m.tauI : 8;
  S.decE = Math.exp(-1/tE); S.decI = Math.exp(-1/tI);
  // psc 0: peak current (NEST, Brian); psc 1: total charge; psc 2: the exact integral over the step, tau (1 - dec), matching the reference engine's configure (itauOf there)
  S.psc = Math.max(0, Math.min(2, m.psc|0));
  const itauOf = (dec, tau) => S.psc === 2 ? tau*(1 - dec) : S.psc === 1 ? 1 - dec : 1;
  S.itauE = itauOf(S.decE, tE); S.itauI = itauOf(S.decI, tI);
  S.allProtos = (m.protocols || []).map(p => ({ ...p, _s:-1 }));
  S.noise = foldNoise(S.allProtos, S.n);   // a noise stimulus's memory and shared part per cell, or null for the white draw (refuses conflicting stimuli)
  S.driveValid = false;
  if(m.inputs !== undefined){
    S.inputs = (m.inputs || []).map(im => ({ ...im,
      gains: new Float32Array(im.chStart.length - 1) }));
    S.extDirty = true;
  }
  S.plast = (m.plast|0) === 1;
  // Every term the reference engine has is implemented here.
  // The guard stays: a term this engine cannot run must fail loudly rather than silently diverge (ENGINE.md), and the list is empty only because nothing is missing, not because the rule was dropped.
  // Add a term to it the moment one is added to the reference without a port.
  const missing = [];
  if(missing.length)
    throw new Error('gpu engine: ' + missing.join(', ') +
      ' not implemented in the WebGPU worker; use engine=cpu or engine=remote');
  S.refrac = m.refrac > 0 ? Math.min(20, m.refrac|0) : 0;
  // membrane floor (the reference's S.vmin); at or above 0 disables it, which in f32 is the most negative finite value rather than -Infinity
  S.vmin = m.vmin === undefined ? -90 : (+m.vmin < 0 ? +m.vmin : -3.4e38);
  // the membrane step (MODEL.md 1): 1 the exponential half steps and the reset inside the millisecond
  S.integ = (m.integ|0) === 2 ? 2 : (m.integ|0) === 1 ? 1 : 0;
  // receptor channels (MODEL.md 2): fixed at init, since the ring is sized by them
  const extra = Array.isArray(m.chanTau) || ArrayBuffer.isView(m.chanTau) ? Array.from(m.chanTau) : null;
  if(extra){
    if(extra.length > 6) throw new Error('gpu engine: at most six receptor channels beyond E and I');
    if(extra.length && S.syn < 1) throw new Error('receptor channels need exponential synapses: set the checkpoint synapse mode to exp');
    if(S.KX !== undefined && extra.length !== S.KX) throw new Error('gpu engine: the receptor channels changed; the network needs rewiring');
    S.KX = extra.length; S.chanTau = extra;
  }
  if(S.KX === undefined){ S.KX = 0; S.chanTau = []; }
  // chanDec, not decX: decX is the short-term plasticity depression decay
  S.chanDec = Float32Array.from(S.chanTau, t => Math.exp(-1/(t > 0 ? t : 3)));
  S.chanItau = Float32Array.from(S.chanDec, (d, k) => S.psc === 2 ? (S.chanTau[k] > 0 ? S.chanTau[k] : 3)*(1 - d) : S.psc === 1 ? 1 - d : 1);
  // Short-term plasticity.
  // The reference and CUDA engines seed the state (X 1, R at the then-current U) the first time the term is on and never again, so a re-enable resumes where the state was left; the device buffers are allocated at init and seeded on the first tick with the term on.
  S.stp = m.stp > 0 ? 1 : 0;
  S.stpU = m.stpU > 0 ? +m.stpU : 0.2;
  S.stpOrder = (m.stpOrder|0) === 1 ? 1 : 0;
  S.stpScope = (m.stpScope|0) & 3;           // the release on every synapse (0), excitatory only (1) or inhibitory only (2)
  // where the pair rule is timed (MODEL.md 4): 0 at the soma, 1 at the synapse, with the tables the reference engine uses
  S.stdpAt = (m.stdpAt|0) === 1 ? 1 : 0;
  S.plastGain = m.plastGain === undefined ? 1 : Math.max(0, +m.plastGain || 0);   // the gain on the Hebbian terms (MODEL.md 4)
  // Neuromodulation (MODEL.md 4), the same fields and folds as the reference engine's configure: the field's decay, what reads it, the time-course sources with their increment folded, and the release arrays, fixed at init
  const modTau = m.modTau > 0 ? +m.modTau : 200;
  S.decMod = Math.fround(Math.exp(-1/modTau));
  S.modPlast = (m.modPlast|0) === 2 ? 2 : (m.modPlast|0) === 1 ? 1 : 0;
  S.modBase = m.modBase === undefined ? 1 : (+m.modBase || 0);
  S.modStep = m.modStep > 0 ? Math.max(1, Math.round(+m.modStep)) : 10;
  S.decElig = Math.fround(Math.exp(-S.modStep/(m.eligTau > 0 ? +m.eligTau : 1000)));
  S.eligPay = Math.fround(1 - S.decElig);
  S.modDrive = +m.modDrive || 0; S.modCur = +m.modCur || 0;
  S.modSrc = (m.mods || []).map(p => ({ ...p, amp:Math.fround((+p.amp || 0)*(1 - S.decMod)), gain:null, _s:-1 }));
  if(m.modGrp && m.modGrp.length === S.n && m.modTgt && m.modK){
    S.modGrp = m.modGrp instanceof Uint8Array ? m.modGrp : Uint8Array.from(m.modGrp);
    S.modTgt = m.modTgt instanceof Uint16Array ? m.modTgt : Uint16Array.from(m.modTgt);
    S.modKraw = Float32Array.from(m.modK);
  }
  if(S.modKraw && S.modG && S.modKraw.length !== S.modG) throw new Error('gpu engine: the release groups changed; the network needs rewiring');
  S.modG = S.modKraw ? S.modKraw.length : 1;
  S.modOn = S.modSrc.length > 0 || S.modG > 1 || S.modPlast > 0 || S.modDrive !== 0 || S.modCur !== 0;
  if(S.modAcc === undefined) S.modAcc = 0;
  // the release per spike folded by the field's decay, as the reference folds it, written into the refractory buffer's tail where the update pass reads it
  if(S.modKraw && S.bufRefcnt){
    const kb = new Uint32Array(Float32Array.from(S.modKraw, k => Math.fround(k*(1 - S.decMod))).buffer);
    S.device.queue.writeBuffer(S.bufRefcnt, (S.n + 3*S.poolG + 1 + 2*S.modG)*4, kb);
  }
  if(S.plast && S.modPlast === 2 && S.parts && !S.eligReady) S.needElig = true;
  S.wmu = m.wmu === undefined ? 1 : Math.max(0, Math.min(1, +m.wmu));             // the weight exponent of soft depression (MODEL.md 4)
  S.stpScale = (m.stpNorm|0) === 1 ? 1/(S.stpOrder ? S.stpU*(2 - S.stpU) : S.stpU) : 1;
  S.decX = Math.exp(-1/(m.stpTauD > 0 ? +m.stpTauD : 200));
  S.decR = Math.exp(-1/(m.stpTauF > 0 ? +m.stpTauF : 600));
  // The class table (stpClassTable in nodes.js), as the uniform the delivery and recovery passes read; a network without it is one class, the checkpoint's.
  {
    const tbl = m.stpTable && m.stpClasses >= 1 ? m.stpTable
      : [m.stp > 0 ? 1 : 0, m.stpU > 0 ? +m.stpU : 0.2, m.stpTauD > 0 ? +m.stpTauD : 200, m.stpTauF > 0 ? +m.stpTauF : 600];
    const C = Math.max(1, Math.min(16, (tbl.length/4)|0));
    const buf = new ArrayBuffer(STP_CLASS_BYTES), f = new Float32Array(buf), u = new Uint32Array(buf);
    let any = 0;
    for(let c = 0; c < C; c++){
      const U = tbl[c*4 + 1] > 0 ? +tbl[c*4 + 1] : 0.2;
      f[c*4] = U;
      f[c*4 + 1] = Math.exp(-1/(tbl[c*4 + 2] > 0 ? +tbl[c*4 + 2] : 200));
      f[c*4 + 2] = Math.exp(-1/(tbl[c*4 + 3] > 0 ? +tbl[c*4 + 3] : 600));
      f[c*4 + 3] = (m.stpNorm|0) === 1 ? 1/(S.stpOrder ? U*(2 - U) : U) : 1;
      u[64 + c] = tbl[c*4] > 0 ? 1 : 0; any |= u[64 + c];
    }
    const cls = m.stpCls || [];
    for(let r = 0; r < 16; r++) u[80 + r] = cls[r] | 0;
    u[96] = C;
    // the state is per class: a change in the number of classes starts it again, as the reference engine reallocates it
    if(S.stpC !== undefined && S.stpC !== C) S.stpSeed = true;
    S.stpC = C; S.stpClassBuf = buf; S.stpClassDirty = true;
    S.stp = any ? 1 : 0;
  }
  // Triplet potentiation (Pfister and Gerstner 2006) and the transmitter drip.
  // Both ride on state the engine already keeps: tin is a constant in the outgoing pass, trip needs one more per-neuron trace.
  S.trip = m.trip > 0 ? +m.trip : 0;
  S.tin = m.tin > 0 ? +m.tin : 0;
  const tauY = m.tauY > 0 ? +m.tauY : 114;
  S.decY = Math.exp(-1/tauY); S.tauY = tauY;
  // Heterosynaptic competition and consolidation.
  // The gate is dimensionless, the slow trace against its value at the target rate, so het keeps one scale across targets and trace constants.
  // Both read the same per-synapse reference weight.
  S.het = m.het > 0 ? +m.het : 0;
  S.refInit = m.refInit === undefined ? 0 : (m.refInit|0);
  // the sliding threshold (MODEL.md 4), the same arithmetic as the reference engine: the multiplier on depression is (Khom slideK)^2, Khom the slow rate trace in the second lane of the Kpost buffer
  const slideTau = m.slideTau > 0 ? +m.slideTau : 600000;
  S.decH = Math.exp(-1/slideTau);
  S.slideK = (m.slide|0) === 1 && m.slideRate > 0 ? 1000/(slideTau*m.slideRate) : 0;
  S.khomInit = (m.slideInit > 0 ? +m.slideInit : 0)*slideTau/1000;
  S.cons = m.cons > 0 ? +m.cons : 0;
  S.consW = m.consW > 0 ? +m.consW : 0.1*(m.wmax > 0 ? m.wmax : 10);
  S.consP = m.consP > 0 ? +m.consP : 10;
  S.tauCons = m.tauCons > 0 ? +m.tauCons : 1200000;
  S.commit = (m.commit|0) === 1 ? 1 : 0;
  S.consStep = m.consStep > 0 ? +m.consStep : 1200;
  if(S.consAcc === undefined) S.consAcc = 0;
  // the reference is the weight vector at the moment either term switches on, the same capture rule as the reference engine
  let anyHet = S.het > 0, anyCons = S.cons > 0;
  if(m.ruleTable && m.ruleCount >= 2)
    for(let r = 1; r < m.ruleCount; r++){
      if(m.ruleTable[r*11 + 5] > 0) anyHet = true;
      if(m.ruleTable[r*11 + 8] > 0) anyCons = true;
    }
  S.anyHet = anyHet; S.anyCons = anyCons;
  if((anyHet || anyCons) && S.parts && !S.wRefReady) S.needWRef = true;
  // an explicit zero amplitude is a zero, not a request for the default
  S.aP = m.aP !== undefined ? +m.aP : 0.008; S.aM = m.aM !== undefined ? +m.aM : 0.001;
  // the per-rule table, if the caller sent one; writeRules below fills row 1 from the scalars when it did not
  if(m.ruleTable && m.ruleCount >= 2){
    S.ruleTable = m.ruleTable instanceof Float32Array
      ? m.ruleTable : new Float32Array(m.ruleTable);
    S.ruleCount = m.ruleCount;
  }
  const tauS = m.tauS > 0 ? m.tauS : 16.8;
  const tauM = m.tauM > 0 ? m.tauM : tauS;
  S.decS = Math.exp(-1/tauS); S.decM = Math.exp(-1/tauM);
  S.wmax = m.wmax > 0 ? +m.wmax : 30;
  S.wdep = (m.wdep|0) === 1 ? 1 : 0;         // weight-dependent excitatory depression
  S.iEta = m.iEta !== undefined ? +m.iEta : 0.002;
  S.iRho = m.iRho || 5;
  S.iAlpha = S.iRho*0.001*(tauS + tauM);   // rho (tau+ + tau-)
  // the global factor on the inhibitory rule (MODEL.md 4): the switch, the decay of the low-passed count per step and its window in seconds; the population's inverse times the target is set at init and follows a tuned target
  S.iGlobal = (m.iGlobal|0) === 1 ? 1 : 0;
  const tauG = m.iTauG > 0 ? +m.iTauG*1000 : 10000;
  S.decG = Math.exp(-1/tauG); S.tauGs = tauG/1000;
  if(S.gN) S.gInv = Math.fround(1/(S.gN*S.iRho));
  S.scaleOn = (m.scale|0) === 1;
  S.sEta = m.sEta > 0 ? m.sEta : 0.001;
  // per-neuron set points (measured-baseline mode): first calS seconds run with the rules off while rates record, then lock personal targets
  S.rhoMode = (m.rhoMode|0) === 1 ? 1 : 0;
  S.calMs = (m.calS > 0 ? m.calS : 10) * 1000;
  S.tauSv = tauS; S.tauMv = tauM;
  S.ekS = Float32Array.from({ length:17 }, (_, k) => Math.exp(-k/tauS));
  S.exdS = Float32Array.from({ length:17 }, (_, d) => Math.exp(d/tauS));
  if(S.plast && S.rhoMode === 1 && !S.rhoI){
    S.calibrating = true; S.calStart = S.t;
    if(S.spkAcc) S.spkAcc.fill(0);
  } else if(!S.plast) S.calibrating = false;
  // A tune of iRho, tauS or tauM moves the Vogels target, and the shader reads it per cell from bufAlpha, which was filled once when plasticity was set up; the reference and CUDA engines use the new target at once.
  // Measured set points own bufAlpha, so only the fixed target is written.
  if(S.bufAlpha && !S.rhoI){ S.device.queue.writeBuffer(S.bufAlpha, 0, new Float32Array(S.n).fill(S.iAlpha)); S.alphaDirty = true; }
  // Scaling measures its window from lastScale and steps by the window's length, so a window left from before scaling or plasticity was on made the first pass one oversized step.
  // It starts afresh when scaling turns on, as the other engines' one-second pass does.
  const scaling = S.plast && S.scaleOn;
  if(scaling && !S.wasScaling && !S.calibrating){
    S.lastScale = S.t; S.nextScale = S.t + 1000;
    if(S.spkAcc) S.spkAcc.fill(0);
  }
  S.wasScaling = scaling;
}

// The reference weight vector, captured from the weights at the moment heterosynaptic competition or consolidation switches on, exactly as the reference engine captures it.
// Partitioned with the weights because a storage buffer binding is capped well below a large network's weight array; on a 157M synapse scene this is another 600 MB, which is why it is allocated on the edge rather than always.
function ensureWRef(){
  if(!S.parts || S.wRefReady) return;
  // any rule row, not only the checkpoint's, as the other two engines allocate: a declared rule with het or cons over a checkpoint with both off would otherwise leave the reference unallocated, so het would pull toward the zero placeholder and consolidation would never run
  if(!(S.anyHet || S.anyCons)) return;
  const { device } = S;
  for(const p of S.parts){
    const bytes = (p.s1 - p.s0)*4;
    p.wRef = device.createBuffer({ size: Math.max(16, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    // a reference restored from a saved state replaces the capture, and is not clamped again: it is already where the run that saved it left it
    if(S.pendingRef){ if(bytes) device.queue.writeBuffer(p.wRef, 0, S.pendingRef, p.s0, p.s1 - p.s0); continue; }
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(p.wgt, 0, p.wRef, 0, bytes);
    device.queue.submit([enc.finish()]);
  }
  S.wRefReady = true;
  S.needWRef = false;
  // the copy is clamped into the well by the initRef pass at the start of the next step, before any plasticity pass reads it
  S.wRefInitPending = !S.pendingRef;
  S.pendingRef = null;
}

// The eligibility trace per synapse (modPlast 2, MODEL.md 4): zero at allocation, partitioned with the weights, allocated on the edge where the form first switches on, as the reference engine allocates it, and kept after.
function ensureElig(){
  if(!S.parts || S.eligReady) return;
  if(!(S.plast && S.modPlast === 2)) return;
  const { device } = S;
  for(const p of S.parts){
    const bytes = (p.s1 - p.s0)*4;
    p.elig = device.createBuffer({ size: Math.max(16, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  S.eligReady = true;
  S.needElig = false;
}

// The consolidation reference from a saved state: written now if the reference exists, otherwise held until ensureWRef allocates it.
function setRef(r){
  if(!S || !S.w || !r || r.length !== S.w.length) return;
  S.pendingRef = r instanceof Float32Array ? r : Float32Array.from(r);
  if(!S.wRefReady) return;
  for(const p of S.parts) if(p.s1 > p.s0) S.device.queue.writeBuffer(p.wRef, 0, S.pendingRef, p.s0, p.s1 - p.s0);
  S.pendingRef = null; S.wRefInitPending = false;
}

// The reference reassembled from its partitions, as readWeights does the weights.
async function readRef(){
  const { device } = S;
  const r = new Float32Array(S.w.length);
  for(const p of S.parts){
    const bytes = (p.s1 - p.s0)*4;
    if(!bytes) continue;
    const st = device.createBuffer({ size:bytes,
      usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(p.wRef, 0, st, 0, bytes);
    device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    r.set(new Float32Array(st.getMappedRange()).subarray(0, p.s1 - p.s0), p.s0);
    st.unmap(); st.destroy();
  }
  return r;
}

// The incoming and consolidation bind groups both name the reference weight buffer, so they are rebuilt whenever it appears.
// With the terms off the binding still has to point somewhere, and it must not point at the weight buffer: that buffer is already bound writable in the same group, and binding one buffer as both writable and read-only storage is a validation error.
// WebGPU reports those asynchronously, so the whole pass silently becomes a no-op rather than throwing, which reads as an enormous speedup.
// A small placeholder is bound instead; the shader never reads it because it guards on the per-rule R.het and R.cons.
// The rule table as a uniform, sixteen rows of twelve floats: the eleven fields plus one of padding, so each row is 48 bytes and meets the sixteen byte stride a uniform array requires.
// Uniforms are a separate budget from the eight storage buffers the incoming pass is already at, which is the only reason a per-synapse rule fits on this engine without freeing a slot.
//
// With no table sent, row 1 is filled from the scalars, so a synapse carrying 1 computes what it computed before rules existed, by the same path rather than by a different one.
const RULE_MAX = 16, RULE_VEC = 12;
function writeRules(){
  const { device } = S;
  if(!device) return;
  if(!S.bufRules) S.bufRules = device.createBuffer({ size:RULE_MAX*RULE_VEC*4,
    usage:GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const t = new Float32Array(RULE_MAX*RULE_VEC);
  const src = S.ruleTable, rows = Math.min(RULE_MAX, S.ruleCount || 0);
  if(src && rows >= 2){
    for(let r = 1; r < rows; r++)
      for(let k = 0; k < 11; k++) t[r*RULE_VEC + k] = src[r*11 + k];
  } else {
    const o = RULE_VEC;                       // row 1
    t[o+0] = S.aP; t[o+1] = S.aM; t[o+2] = S.wmax; t[o+3] = S.wdep ? 1 : 0;
    t[o+4] = S.trip; t[o+5] = S.het; t[o+6] = S.tin; t[o+7] = S.iEta;
    t[o+8] = S.cons || 0; t[o+9] = S.consW || 0; t[o+10] = S.consP || 10;
  }
  device.queue.writeBuffer(S.bufRules, 0, t);
}

// The scaling pass names the reference weight as well, so it is rebuilt with the others when the reference appears.
function rebuildScaleGroups(){
  if(!S.parts || !S.bglScale || !S.bufFactor) return;
  const { device } = S;
  if(!S.bufWRefNull) S.bufWRefNull = device.createBuffer({ size:16,
    usage: GPUBufferUsage.STORAGE });
  const e = (binding, buffer, extra) => ({ binding, resource:{ buffer, ...extra } });
  S.bgScale = S.parts.map(p => device.createBindGroup({ layout:S.bglScale, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, p.uni),
    e(2, p.wgt), e(3, p.postd), e(4, p.pm), e(5, S.bufFactor),
    e(6, S.bufRules), e(7, p.wRef || S.bufWRefNull) ] }));
}
function rebuildWRefGroups(){
  if(!S.parts || !S.bglIn) return;
  rebuildScaleGroups();
  const { device } = S;
  if(!S.bufWRefNull) S.bufWRefNull = device.createBuffer({ size:16,
    usage: GPUBufferUsage.STORAGE });
  const e = (binding, buffer, extra) => ({ binding, resource:{ buffer, ...extra } });
  writeRules();                       // creates the uniform on first call
  // The eligibility trace binds like the reference: a placeholder until the form is on, never read since every pass guards on U.modPlast.
  if(!S.bufEligNull) S.bufEligNull = device.createBuffer({ size:16, usage: GPUBufferUsage.STORAGE });
  // The outgoing pass reads the reference too now, so it is built here with the others rather than once at setup: either term can be switched on by a tune long after setup ran, which reallocates these buffers.
  if(S.bglOut) S.bgPlastOut = S.parts.map(p => device.createBindGroup({ layout:S.bglOut, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, p.uni), e(2, S.bufSpikes),
    e(3, S.bufPre), e(4, p.postd), e(5, p.wgt), e(6, S.bufKpost), e(7, p.pm),
    e(8, p.elig || S.bufEligNull), e(9, p.wRef || S.bufWRefNull), e(10, S.bufRules) ] }));
  // the arrival pass binds the same, with the arrival list in place of the spikes
  if(S.bglOut) S.bgPlastArrive = S.parts.map(p => device.createBindGroup({ layout:S.bglOut, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, p.uni), e(2, S.bufArr),
    e(3, S.bufPre), e(4, p.postd), e(5, p.wgt), e(6, S.bufKpost), e(7, p.pm),
    e(8, p.elig || S.bufEligNull), e(9, p.wRef || S.bufWRefNull), e(10, S.bufRules) ] }));
  S.bgPlastIn = S.parts.map(p => device.createBindGroup({ layout:S.bglIn, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, p.uni), e(2, S.bufSpikes),
    e(3, S.bufInStart), e(4, S.bufRev), e(5, p.elig || S.bufEligNull),
    e(6, p.wgt), e(7, S.bufKtr), e(8, p.pm),
    e(9, p.wRef || S.bufWRefNull), e(10, S.bufRules) ] }));
  if(S.bglElig) S.bgElig = S.parts.map(p => device.createBindGroup({ layout:S.bglElig, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, p.uni),
    e(2, p.wgt), e(3, p.elig || S.bufEligNull), e(4, p.postd), e(5, p.pm),
    e(6, S.bufKpost), e(7, p.wRef || S.bufWRefNull), e(8, S.bufRules) ] }));
  S.bgCons = S.parts.map(p => device.createBindGroup({ layout:S.bglCons, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, p.uni),
    e(2, p.wgt), e(3, p.wRef || S.bufWRefNull), e(4, p.pm),
    e(5, S.bufRules) ] }));
}

// reverse index + plasticity pipelines, built on first enable (like the CPU engine's lazy reverse index). +8 B per synapse of GPU memory while on.
function ensurePlast(){
  if(S.revReady) return;
  const { device, n, preStart, post } = S;
  const m2 = post.length;
  const inStart = new Uint32Array(n+1);
  for(let s=0;s<m2;s++) inStart[post[s]+1]++;
  for(let i=0;i<n;i++) inStart[i+1] += inStart[i];
  const cur = new Uint32Array(n),
        rev = new Uint32Array(2*m2);
  // each entry a pair: the synapse, then the presynaptic cell with the delay in its top five bits as in postd, so the incoming pass at the synapse reads it with the presynaptic cell and binds nothing more
  const dmask = new Uint32Array(n);
  for(let i=0;i<n;i++)
    for(let s=preStart[i]; s<preStart[i+1]; s++){
      const at = inStart[post[s]] + cur[post[s]]++;
      rev[2*at] = s; rev[2*at + 1] = i | (S.delay[s] << 27);
      dmask[i] |= 1 << S.delay[s];
    }
  const ST = GPUBufferUsage.STORAGE;
  const up = (usage, arr) => { const b = device.createBuffer({
      size: Math.max(16, arr.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, arr); return b; };
  S.bufInStart = up(ST, inStart);
  S.bufRev = up(ST, rev);
  S.bufDmask = up(ST, dmask);

  const C = GPUShaderStage.COMPUTE;
  const sb = (binding, type) => ({ binding, visibility:C, buffer:{ type } });
  const dynU = { binding:0, visibility:C,
    buffer:{ type:'uniform', hasDynamicOffset:true, minBindingSize:USIZE } };
  const bglOut = device.createBindGroupLayout({ entries:[ dynU,
    { binding:1, visibility:C, buffer:{ type:'uniform' } },
    sb(2,'storage'), sb(3,'read-only-storage'), sb(4,'read-only-storage'),
    sb(5,'storage'), sb(6,'read-only-storage'), sb(7,'read-only-storage'),
    sb(8,'storage'), sb(9,'read-only-storage'),
    { binding:10, visibility:C, buffer:{ type:'uniform' } } ] });
  // eight storage buffers, the WebGPU guaranteed maximum: spikes, the two reverse-index arrays, the eligibility trace, the weights, the interleaved traces, the mask, and the reference weight.
  // Nothing else fits here without freeing a slot.
  const bglIn = device.createBindGroupLayout({ entries:[ dynU,
    { binding:1, visibility:C, buffer:{ type:'uniform' } },
    sb(2,'storage'), sb(3,'read-only-storage'), sb(4,'read-only-storage'),
    sb(5,'storage'), sb(6,'storage'), sb(7,'read-only-storage'),
    sb(8,'read-only-storage'), sb(9,'read-only-storage'),
    { binding:10, visibility:C, buffer:{ type:'uniform' } } ] });
  const bglElig = device.createBindGroupLayout({ entries:[ dynU,
    { binding:1, visibility:C, buffer:{ type:'uniform' } },
    sb(2,'storage'), sb(3,'storage'), sb(4,'read-only-storage'), sb(5,'read-only-storage'),
    sb(6,'read-only-storage'), sb(7,'read-only-storage'),
    { binding:8, visibility:C, buffer:{ type:'uniform' } } ] });
  const bglScale = device.createBindGroupLayout({ entries:[ dynU,
    { binding:1, visibility:C, buffer:{ type:'uniform' } },
    sb(2,'storage'), sb(3,'read-only-storage'), sb(4,'read-only-storage'),
    sb(5,'read-only-storage'),
    { binding:6, visibility:C, buffer:{ type:'uniform' } },
    sb(7,'read-only-storage') ] });
  const bglCons = device.createBindGroupLayout({ entries:[ dynU,
    { binding:1, visibility:C, buffer:{ type:'uniform' } },
    sb(2,'read-only-storage'), sb(3,'storage'), sb(4,'read-only-storage'),
    { binding:5, visibility:C, buffer:{ type:'uniform' } } ] });
  const bglBump = device.createBindGroupLayout({ entries:[
    sb(0,'storage'), sb(1,'storage'), sb(2,'storage') ] });
  const bglDecay = device.createBindGroupLayout({ entries:[ dynU,
    sb(1,'storage'), sb(2,'storage'), sb(3,'read-only-storage'), sb(4,'read-only-storage'), sb(5,'storage'), sb(6,'storage'), sb(7,'read-only-storage') ] });
  const bglClearArr = device.createBindGroupLayout({ entries:[ sb(0,'storage') ] });
  const bglAlpha = device.createBindGroupLayout({ entries:[ dynU, sb(1,'storage'), sb(2,'read-only-storage') ] });
  const mkPipe = (code, entry, bgl) => device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts:[bgl] }),
    compute:{ module: device.createShaderModule({ code }), entryPoint: entry } });
  S.pPlastOut = mkPipe(SHADER_PLAST_OUT, 'plastOut', bglOut);
  S.pPlastArrive = mkPipe(SHADER_PLAST_ARRIVE, 'plastArrive', bglOut);
  S.pPlastIn = mkPipe(SHADER_PLAST_IN, 'plastIn', bglIn);
  S.pClearArr = mkPipe(SHADER_CLEAR_ARR, 'clearArrivals', bglClearArr);
  S.bgClearArr = device.createBindGroup({ layout:bglClearArr, entries:[ { binding:0, resource:{ buffer:S.bufArr } } ] });
  S.pBump = mkPipe(SHADER_BUMP, 'bump', bglBump);
  S.pDecay = mkPipe(SHADER_DECAY, 'decay', bglDecay);
  S.pScale = mkPipe(SHADER_SCALE, 'scalePass', bglScale);
  S.pCons = mkPipe(SHADER_CONS, 'consolidate', bglCons);
  S.pInitRef = mkPipe(SHADER_CONS, 'initRef', bglCons);
  S.pElig = mkPipe(SHADER_ELIG, 'eligSweep', bglElig);
  S.bglElig = bglElig;
  S.pAlpha = mkPipe(SHADER_ALPHA, 'copyAlpha', bglAlpha);

  const e = (binding, buffer, extra) => ({ binding, resource:{ buffer, ...extra } });
  writeRules();
  S.bufAlpha = device.createBuffer({ size:S.n*4,
    usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(S.bufAlpha, 0, new Float32Array(S.n).fill(S.iAlpha));
  S.bgAlpha = device.createBindGroup({ layout:bglAlpha, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, S.bufKpost), e(2, S.bufAlpha) ] });
  S.alphaDirty = true;
  S.bglOut = bglOut;
  // Reference weights: a second copy of every weight, so they are allocated only when heterosynaptic competition or consolidation is actually on.
  // With them off, p.wRef aliases p.wgt, which keeps the bind group static and is never read, because the shader guards on the per-rule R.het.
  // Kept on S because either term can be switched on by a tune long after this ran, which needs these two groups rebuilt against the new buffers.
  S.bglIn = bglIn; S.bglCons = bglCons;
  ensureWRef();
  ensureElig();
  rebuildWRefGroups();
  S.bufFactor = device.createBuffer({ size:S.n*4,
    usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  S.bglScale = bglScale;
  rebuildScaleGroups();
  S.spkAcc = new Float32Array(S.n);
  S.factorHost = new Float32Array(S.n);
  S.nextScale = S.t + 1000; S.lastScale = S.t;
  S.bgBump = device.createBindGroup({ layout:bglBump, entries:[
    e(0, S.bufSpikes), e(1, S.bufKtr), e(2, S.bufKpost) ] });
  S.bgDecay = device.createBindGroup({ layout:bglDecay, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, S.bufKtr), e(2, S.bufKpost), e(3, S.bufAlpha), e(4, S.bufDmask), e(5, S.bufArr), e(6, S.bufRefcnt), e(7, S.bufState) ] });
  S.revReady = true;
}

async function init(m){
  if(!navigator.gpu) throw new Error('WebGPU unavailable in this worker');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference:'high-performance' });
  if(!adapter) throw new Error('no WebGPU adapter');
  const lim = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits:{
    maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
    maxBufferSize: lim.maxBufferSize } });
  device.lost.then(info => {
    if(info.reason !== 'destroyed')
      postMessage({ cmd:'error', message:'GPU device lost: ' + info.message });
  });

  const n = m.count, T = m.types;
  // twelve floats per neuron: the f7 row's a, b, c, d, then C, k, vr, vt, vpeak, then the graded threshold, slope and flag
  { const bad = checkProtocol(m, 'gpu engine'); if(bad) throw new Error(bad); }
  if('form' in m && (m.form|0) !== 1) throw new Error('gpu engine: the 2003 membrane form was retired; the scene needs migrating (scene format 8)');
  const abcd = new Float32Array(16*n);
  for(let i=0;i<n;i++){ const t = T[m.ntype[i]] && T[m.ntype[i]].f7;
    if(!t) throw new Error('gpu engine: every neuron type needs a 2007 row (f7)');
    abcd[i*16]=t.a; abcd[i*16+1]=t.b; abcd[i*16+2]=t.c; abcd[i*16+3]=t.d;
    { abcd[i*16+4]=t.C; abcd[i*16+5]=t.k; abcd[i*16+6]=t.vr; abcd[i*16+7]=t.vt; abcd[i*16+8]=t.vpeak; }
    // the dynamic threshold's jump and decay, the conductance adaptation flag and the Poisson flag (MODEL.md 1)
    abcd[i*16+12] = t.thrJump > 0 ? +t.thrJump : 0; abcd[i*16+13] = Math.exp(-1/(t.thrTau > 0 ? +t.thrTau : 5));
    abcd[i*16+14] = t.adaptCond ? 1 : 0; abcd[i*16+15] = t.poisson ? 1 : 0; }
  // graded rows: threshold, slope and a flag in the last three slots; the event list packs the index in 24 bits, which caps a graded net
  const G = gradedArrays(m.ntype, T);
  if(G){
    if(n > 0xffffff) throw new Error('gpu engine: graded rows need the neuron index in 24 bits (16.7 M cells at most)');
    for(let i=0;i<n;i++) if(G.grd[i]){ abcd[i*16+9] = G.thr[i]; abcd[i*16+10] = G.slope[i]; abcd[i*16+11] = 1; }
  }
  m.pmask = freezeGraded(m.pmask || null, m.preStart, G && G.grd, m.w.length);
  // The initial membrane state arrives from the host (initState in rand.js), the same bytes the reference engine and the CUDA child start from.
  // Refused when absent rather than seeded here, so no engine draws its own.
  if(!m.v0 || !m.u0 || m.v0.length !== n || m.u0.length !== n)
    throw new Error('gpu engine: init needs v0 and u0 (rand.js initState)');
  const st = new Float32Array(SST*n);
  for(let i=0;i<n;i++){ st[i*SST] = m.v0[i]; st[i*SST+1] = m.u0[i]; }
  // the global factor's population: every cell with an inhibitory synapse under a rule, flagged in slot 13 (MODEL.md 4); the count lands on the state object once it exists
  let gNmask = 0;
  { const gm = new Uint8Array(n); let gN = 0;
    for(let s=0;s<m.w.length;s++) if(m.w[s] < 0 && (m.pmask ? (m.pmask[s] & 31) : 1) && !gm[m.post[s]]){ gm[m.post[s]] = 1; gN++; }
    for(let i=0;i<n;i++) st[i*SST+13] = gm[i];
    gNmask = gN; }

  // CPU copies kept for query and getWeights
  S = { n, device, t:0, cur:0, watch:-1, seed:(m.seed>>>0) || 1, gN:gNmask, gInv:gNmask ? Math.fround(1/(gNmask*(m.iRho || 5))) : 0,
        preStart:m.preStart, post:m.post, w:m.w, delay:m.delay,
        bias:m.bias, driveHost:new Float32Array(8*n), driveValid:false,
        zeroN:new Uint32Array(n), zero1:new Uint32Array(1) };
  configure(m);

  const B = (usage, size) => device.createBuffer({ size, usage });
  const up = (usage, arr) => { const b = device.createBuffer({
      size: Math.max(16, arr.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, arr); return b; };
  const ST = GPUBufferUsage.STORAGE;

  S.bufAbcd = up(ST, abcd);
  S.bufState = up(ST | GPUBufferUsage.COPY_SRC, st);
  S.bufDrive = B(ST | GPUBufferUsage.COPY_DST, 8*n*4);
  S.bufRing = B(ST | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, (2 + S.KX)*ROWS*n*4);   // E, I, then a row set per receptor channel
  S.bufSpikes = B(ST | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, (2+n)*4);
  S.bufFired = B(ST | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, n*4);
  S.bufVtrace = B(ST | GPUBufferUsage.COPY_SRC, MAX_BATCH*4);
  // Feedback inhibition shares the refractory buffer (see the binding): the pool per cell above the refractory byte, two halves of per-pool spike counts, then each pool's kick per spike as float bits.
  S.poolG = (m.pool && m.pool.length === n && m.poolK && m.poolK.length >= 2) ? m.poolK.length : 1;
  if(m.pool && m.pool.length && S.poolG === 1)
    throw new Error('gpu engine: init pool arrays do not match the network');
  // one word past the pool tables: the global factor's count of the inhibited population's spikes this step; then the modulator groups' two count halves and their release per spike (MODEL.md 4)
  const rcInit = new Uint32Array(n + 3*S.poolG + 1 + 3*S.modG);
  if(S.poolG > 1){
    for(let i=0;i<n;i++) rcInit[i] = (m.pool[i] & 0xFFFF) << 8;
    const kb = new Uint32Array(Float32Array.from(m.poolK).buffer);
    for(let g=0;g<S.poolG;g++) rcInit[n + 2*S.poolG + g] = kb[g];
  }
  if(S.modG > 1){
    for(let i=0;i<n;i++) rcInit[i] |= (S.modGrp[i] & 0xFF) << 24;
    const kb = new Uint32Array(Float32Array.from(S.modKraw, k => Math.fround(k*(1 - S.decMod))).buffer);
    for(let g=0;g<S.modG;g++) rcInit[n + 3*S.poolG + 1 + 2*S.modG + g] = kb[g];
  }
  S.bufRefcnt = up(ST | GPUBufferUsage.COPY_DST, rcInit);
  // Always allocated, even with the term off, so the delivery bind group is built once and never rebuilt on a tune.
  // Two floats per neuron is nothing beside the synapse arrays.
  // Sized for every class the table can hold (16), so a tune that adds a class needs no new buffer: 128 bytes a cell.
  S.bufStpX = B(ST | GPUBufferUsage.COPY_DST, n*16*4);
  S.bufStpR = B(ST | GPUBufferUsage.COPY_DST, n*16*4);
  S.bufStpC = B(GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, STP_CLASS_BYTES);
  S.stpSeed = true;
  S.bufInd = B(ST | GPUBufferUsage.INDIRECT, 24);   // two dispatch sizes: the spikes, then the arrivals
  S.bufKtr = B(ST | GPUBufferUsage.COPY_DST, n*TL*4 + 16);  // Kpre, Kslow, a copy of Kpost, the spike history and the modulator level interleaved (TL lanes); the global factor's ratio after them
  S.bufKpost = B(ST | GPUBufferUsage.COPY_DST, n*KL*4 + 8);   // KL lanes per cell: the fast postsynaptic trace, the sliding threshold's slow rate trace, the modulator level, the cell's Vogels alpha; the global factor's low-passed count and ratio after them
  device.queue.writeBuffer(S.bufKtr, 0, new Float32Array(TL*n));
  S.bufArr = B(ST | GPUBufferUsage.COPY_DST, 8 + n*8);   // the arrival list: a count, then (cell, history) pairs
  { const k0 = new Float32Array(KL*n + 2); for(let i = 0; i < n; i++){ k0[KL*i + 1] = S.khomInit || 0; k0[KL*i + 3] = S.iAlpha; }
    // the low-passed count starts at the target times the population, so the factor starts at zero as the authors' does
    k0[KL*n] = Math.fround((m.iRho || 5)*S.gN); device.queue.writeBuffer(S.bufKpost, 0, k0); }
  S.bufStep = device.createBuffer({ size: MAX_BATCH*SLOT,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  S.bufPre = up(ST, m.preStart instanceof Uint32Array ? m.preStart : new Uint32Array(m.preStart));

  // partitions: contiguous presynaptic ranges whose CSR slice stays under the byte cap; __partCap (bytes) exists so tests can force several
  const cap = Math.min(m.__partCap || 2**30, lim.maxStorageBufferBindingSize);
  S.parts = [];
  let n0 = 0;
  while(n0 < n){
    let n1 = n0;
    const base = m.preStart[n0];
    while(n1 < n && (m.preStart[n1+1] - base)*4 <= cap) n1++;
    if(n1 === n0) throw new Error('gpu engine: one neuron exceeds the partition cap');
    const s0 = base, s1 = m.preStart[n1];
    const postd = new Uint32Array(s1 - s0);
    for(let s=s0;s<s1;s++) postd[s-s0] = m.post[s] | (m.delay[s] << 27);
    // The rule byte per local synapse, four to a word.
    // Carrying an index rather than a bit costs eight times the buffer, 152 MB at 151.7M synapses.
    // Absent means every synapse takes rule 1, the checkpoint's own set, which is what a graph with no plasticity node wants.
    const pmBytes = new Uint32Array(Math.max(1, Math.ceil((s1-s0)/4)));
    if(m.pmask){
      for(let s=s0;s<s1;s++){
        const l = s - s0;
        pmBytes[l>>2] |= (m.pmask[s] & 0xff) << ((l & 3)*8);
      }
    } else pmBytes.fill(0x01010101);
    S.parts.push({
      uni: up(GPUBufferUsage.UNIFORM, new Uint32Array([n0, n1, s0, s1])),
      postd: up(ST, postd),
      wgt: up(ST | GPUBufferUsage.COPY_SRC, m.w.subarray(s0, s1).slice()),
      pm: up(ST, pmBytes),
      s0, s1,
    });
    n0 = n1;
  }

  // explicit layouts: binding 0 of update and deliver is a dynamic-offset uniform (one 256 B slot per encoded step)
  const C = GPUShaderStage.COMPUTE;
  const sb = (binding, type) => ({ binding, visibility:C, buffer:{ type } });
  const bglUpdate = device.createBindGroupLayout({ entries:[
    { binding:0, visibility:C, buffer:{ type:'uniform', hasDynamicOffset:true, minBindingSize:USIZE } },
    sb(1,'read-only-storage'), sb(2,'storage'), sb(3,'read-only-storage'),
    sb(4,'storage'), sb(5,'storage'), sb(6,'storage'), sb(7,'storage'),
    sb(8,'storage') ] });
  const bglClear = device.createBindGroupLayout({ entries:[ sb(0,'storage') ] });
  const bglPool = device.createBindGroupLayout({ entries:[
    { binding:0, visibility:C, buffer:{ type:'uniform', hasDynamicOffset:true, minBindingSize:USIZE } },
    sb(1,'storage') ] });
  const bglInd = device.createBindGroupLayout({ entries:[ sb(0,'storage'), sb(1,'storage'), sb(2,'storage') ] });
  const bglDeliver = device.createBindGroupLayout({ entries:[
    { binding:0, visibility:C, buffer:{ type:'uniform', hasDynamicOffset:true, minBindingSize:USIZE } },
    { binding:1, visibility:C, buffer:{ type:'uniform' } },
    sb(2,'storage'), sb(3,'storage'),
    sb(4,'read-only-storage'), sb(5,'read-only-storage'), sb(6,'read-only-storage'),
    sb(7,'storage'), sb(8,'storage'), sb(9,'read-only-storage'),
    { binding:10, visibility:C, buffer:{ type:'uniform' } } ] });
  const bglStp = device.createBindGroupLayout({ entries:[
    { binding:0, visibility:C, buffer:{ type:'uniform', hasDynamicOffset:true, minBindingSize:USIZE } },
    sb(1,'storage'), sb(2,'storage'),
    { binding:3, visibility:C, buffer:{ type:'uniform' } } ] });

  const mkPipe = (code, entry, bgl) => device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts:[bgl] }),
    compute:{ module: device.createShaderModule({ code }), entryPoint: entry } });
  S.pUpdate = mkPipe(SHADER_UPDATE, 'update', bglUpdate);
  S.pClear = mkPipe(SHADER_CLEAR, 'clearCount', bglClear);
  S.pPool = mkPipe(SHADER_POOL, 'poolClear', bglPool);
  S.pInd = mkPipe(SHADER_IND, 'writeIndirect', bglInd);
  S.pDeliver = mkPipe(SHADER_DELIVER, 'deliver', bglDeliver);
  S.pStp = mkPipe(SHADER_STP, 'stpRecover', bglStp);

  const e = (binding, buffer, extra) => ({ binding, resource:{ buffer, ...extra } });
  S.bgUpdate = device.createBindGroup({ layout:bglUpdate, entries:[
    e(0, S.bufStep, { size:USIZE }),
    e(1, S.bufAbcd), e(2, S.bufState), e(3, S.bufDrive), e(4, S.bufRing),
    e(5, S.bufSpikes), e(6, S.bufFired), e(7, S.bufVtrace),
    e(8, S.bufRefcnt) ] });
  S.bgClear = device.createBindGroup({ layout:bglClear, entries:[ e(0, S.bufSpikes) ] });
  S.bgPool = device.createBindGroup({ layout:bglPool, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, S.bufRefcnt) ] });
  S.bgInd = device.createBindGroup({ layout:bglInd, entries:[
    e(0, S.bufSpikes), e(1, S.bufInd), e(2, S.bufArr) ] });
  S.bgDeliver = S.parts.map(p => device.createBindGroup({ layout:bglDeliver, entries:[
    e(0, S.bufStep, { size:USIZE }),
    e(1, p.uni), e(2, S.bufSpikes), e(3, S.bufRing),
    e(4, S.bufPre), e(5, p.postd), e(6, p.wgt),
    e(7, S.bufStpX), e(8, S.bufStpR), e(9, p.pm), e(10, S.bufStpC) ] }));
  S.bgStp = device.createBindGroup({ layout:bglStp, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, S.bufStpX), e(2, S.bufStpR), e(3, S.bufStpC) ] });

  // readback staging (reused; ticks are strictly serialized)
  S.stFired = B(GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, n*4);
  S.stMeta = B(GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 8);
  S.stVtrace = B(GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, MAX_BATCH*4);
  S.stepHost = new ArrayBuffer(USIZE);
  S.stepU32 = new Uint32Array(S.stepHost); S.stepI32 = new Int32Array(S.stepHost);
  S.stepF32 = new Float32Array(S.stepHost);
  if(S.plast) ensurePlast();
  // WebGPU surfaces validation failures asynchronously: without this a bad bind group turns every dispatch into a no-op and the engine keeps answering, reporting state for a simulation that never ran.
  // That is the silent-divergence failure this contract exists to prevent, so it is raised as an engine error like any other.
  device.addEventListener('uncapturederror', ev => {
    postMessage({ cmd:'error',
      message:'gpu engine: device error, the simulation did not run: ' +
        (ev.error && ev.error.message ? ev.error.message : String(ev.error)) });
  });
  postMessage({ cmd:'ready', parts:S.parts.length });
}

function encodeSteps(count, slotBase){
  const { device } = S;
  // queue.writeBuffer is ordered against submits, so reusing slots across segments within one tick is safe
  for(let k=0;k<count;k++){
    S.stepU32[0] = S.t + k; S.stepU32[1] = (S.cur + k) % ROWS;
    S.stepI32[2] = S.watch; S.stepU32[3] = S.seed;
    S.stepU32[4] = S.n; S.stepU32[5] = S.syn; S.stepU32[6] = slotBase + k;
    S.stepU32[7] = S.plast ? 1 : 0;
    S.stepF32[8] = S.decE; S.stepF32[9] = S.decI;
    S.stepF32[10] = S.itauE; S.stepF32[11] = S.itauI;
    S.stepF32[12] = S.aP; S.stepF32[13] = S.aM;
    S.stepF32[14] = S.decS; S.stepF32[15] = S.wmax;
    S.stepF32[16] = S.iEta; S.stepF32[17] = S.iAlpha;
    S.stepF32[18] = S.wdep; S.stepF32[19] = S.refrac || 0;
    S.stepU32[20] = S.stp ? 1 : 0;
    S.stepF32[21] = S.stpU; S.stepF32[22] = S.decX; S.stepF32[23] = S.decR;
    S.stepF32[24] = S.tin; S.stepF32[25] = S.trip; S.stepF32[26] = S.decY;
    S.stepF32[27] = S.het; S.stepF32[28] = S.refInit || 0;
    S.stepF32[29] = (S.consStep || 1200)/(S.tauCons || 1200000);
    S.stepF32[34] = S.commit ? 1 : 0;
    S.stepF32[30] = S.consW || 1; S.stepF32[31] = S.consP || 10;
    S.stepF32[32] = S.decM; S.stepF32[33] = S.stpScale;
    S.stepU32[35] = S.poolG || 1;
    S.stepF32[36] = S.vmin;
    S.stepU32[37] = S.stpOrder ? 1 : 0;
    S.stepU32[38] = S.KX || 0;
    S.stepF32[39] = S.slideK || 0;
    for(let x = 0; x < 6; x++){ S.stepF32[40 + x] = S.chanDec && x < S.chanDec.length ? S.chanDec[x] : 0; S.stepF32[46 + x] = S.chanItau && x < S.chanItau.length ? S.chanItau[x] : 0; }
    S.stepF32[52] = S.eRevE; S.stepF32[53] = S.eRevI;
    for(let x = 0; x < 6; x++) S.stepF32[54 + x] = S.chanErev && x < S.chanErev.length ? S.chanErev[x] : 0;
    for(let x = 0; x < 6; x++) S.stepF32[60 + x] = S.chanMg && x < S.chanMg.length ? S.chanMg[x] : 0;
    S.stepU32[66] = S.stp ? (S.stpScope || 0) : 0;
    S.stepU32[67] = S.plast && S.stdpAt ? 1 : 0;
    S.stepF32[68] = S.plastGain === undefined ? 1 : S.plastGain;
    S.stepF32[69] = S.wmu === undefined ? 1 : S.wmu;
    S.stepF32[70] = S.decH === undefined ? 1 : S.decH;
    S.stepU32[71] = S.integ|0;
    for(let q = 0; q < 17; q++){ S.stepF32[72 + q] = S.ekS ? S.ekS[q] : 0; S.stepF32[92 + q] = S.exdS ? S.exdS[q] : 0; }
    S.stepF32[112] = S.decG === undefined ? 1 : S.decG; S.stepF32[113] = S.tauGs || 1; S.stepF32[114] = S.gInv || 0; S.stepU32[115] = S.iGlobal ? 1 : 0;
    // neuromodulation (MODEL.md 4): the field's decay, what reads it, the group count, the sweep's payment and decay, and whether the field runs at all
    S.stepF32[116] = S.decMod; S.stepF32[117] = S.modBase; S.stepF32[118] = S.modDrive; S.stepF32[119] = S.modCur;
    S.stepU32[120] = S.plast ? S.modPlast : 0; S.stepU32[121] = S.modG || 1;
    S.stepF32[122] = S.eligPay; S.stepF32[123] = S.decElig; S.stepU32[124] = S.modOn ? 1 : 0;
    device.queue.writeBuffer(S.bufStep, k*SLOT, S.stepHost);
  }
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  const wgN = Math.ceil(S.n / WG);
  // alpha was written since the last step: into its Kpost lane before any plasticity pass reads it
  if(S.alphaDirty && S.bgAlpha){
    S.alphaDirty = false;
    pass.setPipeline(S.pAlpha); pass.setBindGroup(0, S.bgAlpha, [0]);
    pass.dispatchWorkgroups(wgN);
  }
  // the reference weights were just copied from the weights: clamp them into their wells before any plasticity pass of this step reads them
  if(S.wRefInitPending && S.wRefReady && S.bgCons){
    S.wRefInitPending = false;
    pass.setPipeline(S.pInitRef);
    for(let pi=0; pi<S.parts.length; pi++){
      pass.setBindGroup(0, S.bgCons[pi], [0]);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil((S.parts[pi].s1 - S.parts[pi].s0)/WG)));
    }
  }
  for(let k=0;k<count;k++){
    const off = [k*SLOT];
    pass.setPipeline(S.pClear); pass.setBindGroup(0, S.bgClear);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(S.pUpdate); pass.setBindGroup(0, S.bgUpdate, off);
    pass.dispatchWorkgroups(wgN);
    if(S.poolG > 1 || S.modG > 1){
      pass.setPipeline(S.pPool); pass.setBindGroup(0, S.bgPool, off);
      pass.dispatchWorkgroups(Math.ceil(Math.max(S.poolG, S.modG) / WG));
    }
    pass.setPipeline(S.pInd); pass.setBindGroup(0, S.bgInd);
    pass.dispatchWorkgroups(1);
    // at the synapse, the spikes arriving this step depress their synapses before delivery and before the step's own passes, as the reference does at the start of its step; the list is cleared once walked
    if(S.plast && S.revReady){
      if(S.stdpAt && !S.calibrating){
        S.weightsDirty = true;
        pass.setPipeline(S.pPlastArrive);
        for(const bg of S.bgPlastArrive){
          pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroupsIndirect(S.bufInd, 12);
        }
      }
      pass.setPipeline(S.pClearArr); pass.setBindGroup(0, S.bgClearArr);
      pass.dispatchWorkgroups(1);
    }
    pass.setPipeline(S.pDeliver);
    for(const bg of S.bgDeliver){
      pass.setBindGroup(0, bg, off);
      pass.dispatchWorkgroupsIndirect(S.bufInd, 0);
    }
    if(S.plast){
      if(!S.calibrating){                    // measure first, change nothing
        S.weightsDirty = true;
        // at the soma the outgoing half runs at the spike; at the synapse it ran above, at the arrivals
        if(!S.stdpAt){
          pass.setPipeline(S.pPlastOut);
          for(const bg of S.bgPlastOut){
            pass.setBindGroup(0, bg, off);
            pass.dispatchWorkgroupsIndirect(S.bufInd, 0);
          }
        }
        pass.setPipeline(S.pPlastIn);
        for(const bg of S.bgPlastIn){
          pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroupsIndirect(S.bufInd, 0);
        }
      }
      pass.setPipeline(S.pBump); pass.setBindGroup(0, S.bgBump);
      pass.dispatchWorkgroupsIndirect(S.bufInd, 0);
      pass.setPipeline(S.pDecay); pass.setBindGroup(0, S.bgDecay, off);
      pass.dispatchWorkgroups(wgN);
      // the eligibility sweep on its cadence, after the decay pass copied the step's level, as the reference sweeps after its field update
      if(S.modPlast === 2 && S.eligReady && S.bgElig){
        if(++S.modAcc >= S.modStep){
          S.modAcc = 0; S.weightsDirty = true;
          pass.setPipeline(S.pElig);
          for(let pi=0; pi<S.parts.length; pi++){
            pass.setBindGroup(0, S.bgElig[pi], off);
            pass.dispatchWorkgroups(Math.max(1, Math.ceil((S.parts[pi].s1 - S.parts[pi].s0)/WG)));
          }
        }
      }
    }
    // consolidation keeps its cadence through the set point calibration, as in the reference and CUDA engines; it moves the reference, not a weight
    if(S.plast && S.anyCons && S.wRefReady){
      S.consAcc = (S.consAcc || 0) + 1;
      if(S.consAcc >= (S.consStep || 1200)){
        S.consAcc = 0;
        pass.setPipeline(S.pCons);
        for(let pi=0; pi<S.parts.length; pi++){
          pass.setBindGroup(0, S.bgCons[pi], off);
          pass.dispatchWorkgroups(Math.max(1,
            Math.ceil((S.parts[pi].s1 - S.parts[pi].s0)/WG)));
        }
      }
    }
    if(S.stp){
      pass.setPipeline(S.pStp); pass.setBindGroup(0, S.bgStp, off);
      pass.dispatchWorkgroups(Math.ceil(S.n*(S.stpC || 1)/WG));
    }
  }
  pass.end();
  device.queue.submit([enc.finish()]);
  S.t += count; S.cur = (S.cur + count) % ROWS;
}

async function tick(steps){
  const { device, n } = S;
  // a tune can switch either term on after the plasticity pipelines were built, which is when the reference weights first have to exist
  if((S.needWRef || S.needElig) && S.revReady){
    ensureWRef();
    ensureElig();
    rebuildWRefGroups();
  }
  // Seed the short-term plasticity state on the edge where the term turns on, which is where the reference allocates it: resources full, release probability at U.
  if(S.stpClassDirty && S.bufStpC){ S.stpClassDirty = false; device.queue.writeBuffer(S.bufStpC, 0, S.stpClassBuf); }
  if(S.stp && S.stpSeed){
    S.stpSeed = false;
    const C = S.stpC || 1, cf = new Float32Array(S.stpClassBuf);
    device.queue.writeBuffer(S.bufStpX, 0, new Float32Array(n*C).fill(1));
    const r0 = new Float32Array(n*C);
    for(let c = 0; c < C; c++) r0.fill(cf[c*4], c*n, (c + 1)*n);
    device.queue.writeBuffer(S.bufStpR, 0, r0);
  }
  device.queue.writeBuffer(S.bufFired, 0, S.zeroN);   // clear batch accumulator
  device.queue.writeBuffer(S.bufSpikes, 4, S.zero1);  // clear total
  let done = 0, slotBase = 0;
  while(done < steps){
    if(updateDrive()) device.queue.writeBuffer(S.bufDrive, 0, S.driveHost);
    // encode until the next protocol scale change, the slot cap, or the end
    const maxSeg = Math.min(MAX_BATCH - slotBase, steps - done);
    let seg = 1;
    while(seg < maxSeg){
      let changes = false;
      for(const p of S.allProtos)
        if(scaleOf(p, S.t + seg) !== p._s){ changes = true; break; }
      // the modulator sources change the drive array as the stimulus protocols do
      if(!changes) for(const p of S.modSrc)
        if(scaleOf(p, S.t + seg) !== p._s){ changes = true; break; }
      if(changes) break;
      seg++;
    }
    encodeSteps(seg, slotBase);
    done += seg; slotBase += seg;
    if(slotBase >= MAX_BATCH) slotBase = 0;   // vtrace wraps past the cap
  }
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(S.bufFired, 0, S.stFired, 0, n*4);
  enc.copyBufferToBuffer(S.bufSpikes, 0, S.stMeta, 0, 8);
  enc.copyBufferToBuffer(S.bufVtrace, 0, S.stVtrace, 0, Math.min(steps, MAX_BATCH)*4);
  if(S.sendV){
    if(!S.stState) S.stState = device.createBuffer({ size:n*SST*4,
      usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(S.bufState, 0, S.stState, 0, n*SST*4);
  }
  device.queue.submit([enc.finish()]);
  const maps = [S.stFired.mapAsync(GPUMapMode.READ),
    S.stMeta.mapAsync(GPUMapMode.READ), S.stVtrace.mapAsync(GPUMapMode.READ)];
  if(S.sendV) maps.push(S.stState.mapAsync(GPUMapMode.READ));
  await Promise.all(maps);
  const firedU32 = new Uint32Array(S.stFired.getMappedRange());
  const fired = new Uint8Array(n);
  for(let i=0;i<n;i++) fired[i] = Math.min(255, firedU32[i]);   // capped at 255 as on the other engines
  // synaptic scaling: rate estimate from the fired readback, one factor upload and one pass per simulated second (tick-boundary timing; the CPU engine scales on the exact step, which is inside the statistical contract)
  if(S.plast && S.revReady && (S.scaleOn || S.calibrating)){
    for(let i=0;i<n;i++) S.spkAcc[i] += firedU32[i];   // the uncapped count
    if(S.calibrating){
      if(S.t >= S.calStart + S.calMs){
        const csecs = Math.max(1, (S.t - S.calStart)/1000);
        S.rhoI = new Float32Array(n);
        const al = new Float32Array(n);
        for(let i=0;i<n;i++){
          S.rhoI[i] = Math.min(30, Math.max(0.25, S.spkAcc[i]/csecs));
          al[i] = S.rhoI[i]*0.001*(S.tauSv + S.tauMv);
          S.spkAcc[i] = 0;
        }
        device.queue.writeBuffer(S.bufAlpha, 0, al);
        S.alphaDirty = true;
        S.calibrating = false;
        S.lastScale = S.t; S.nextScale = S.t + 1000;
      }
    }
    else if(S.scaleOn && S.t >= S.nextScale){
      const secs = Math.max(0.2, (S.t - S.lastScale)/1000);
      for(let i=0;i<n;i++){
        const rho = S.rhoI ? S.rhoI[i] : S.iRho;
        const err = Math.max(-2, Math.min(2, (rho - S.spkAcc[i]/secs)/rho));
        S.factorHost[i] = 1 + S.sEta*err*secs;
        S.spkAcc[i] = 0;
      }
      S.lastScale = S.t; S.nextScale = S.t + 1000;
      device.queue.writeBuffer(S.bufFactor, 0, S.factorHost);
      const enc2 = device.createCommandEncoder();
      const pass2 = enc2.beginComputePass();
      pass2.setPipeline(S.pScale);
      for(let pi=0;pi<S.parts.length;pi++){
        pass2.setBindGroup(0, S.bgScale[pi], [0]);
        pass2.dispatchWorkgroups(Math.max(1, Math.ceil((S.parts[pi].s1 - S.parts[pi].s0)/WG)));
      }
      pass2.end();
      device.queue.submit([enc2.finish()]);
      S.weightsDirty = true;
    }
  }
  const meta = new Uint32Array(S.stMeta.getMappedRange());
  const spikes = meta[1];
  const vtrace = new Float32Array(Math.min(steps, MAX_BATCH));
  vtrace.set(new Float32Array(S.stVtrace.getMappedRange()).subarray(0, vtrace.length));
  let v = null;
  if(S.sendV){
    const sv = new Float32Array(S.stState.getMappedRange());
    v = new Float32Array(n);
    for(let i=0;i<n;i++) v[i] = sv[i*SST];   // state is strided v,u,gE,gI, channels
    S.stState.unmap();
  }
  S.stFired.unmap(); S.stMeta.unmap(); S.stVtrace.unmap();
  if(v) postMessage({ cmd:'state', fired, spikes, steps, vtrace, v },
    [fired.buffer, vtrace.buffer, v.buffer]);
  else postMessage({ cmd:'state', fired, spikes, steps, vtrace },
    [fired.buffer, vtrace.buffer]);
}

function query(idx, cap, outOnly){
  const { preStart, post, n } = S;
  cap = cap || 2000;
  // Same reason as the reference engine: the incoming list costs a walk of every synapse, and the caller that asks for a pathway never reads it.
  const inn = []; let inTotal = 0;
  if(!outOnly) for(let s=0;s<post.length;s++)
    if(post[s] === idx){
      inTotal++;
      if(inn.length < cap) inn.push(s);
    }
  const innIdx = new Uint32Array(inn.length);
  for(let k=0;k<inn.length;k++){
    let lo = 0, hi = n; const s = inn[k];
    while(lo < hi-1){ const mid = (lo+hi)>>1; if(preStart[mid] <= s) lo = mid; else hi = mid; }
    innIdx[k] = lo;
  }
  const o0 = preStart[idx], o1 = preStart[idx+1];
  const outN = Math.min(cap, o1-o0);
  const out = new Uint32Array(outN);
  for(let k=0;k<outN;k++) out[k] = post[o0+k];
  postMessage({ cmd:'queryResult', idx, out, inn:innIdx, outTotal:o1-o0, inTotal },
    [out.buffer, innIdx.buffer]);
}

// live weights live on the GPU once plasticity has ever run; reassemble the partition slices into one array for checkpoints and analysis
async function readWeights(){
  const { device } = S;
  const w = new Float32Array(S.w.length);
  for(const p of S.parts){
    const bytes = (p.s1 - p.s0)*4;
    if(!bytes) continue;
    const st = device.createBuffer({ size:bytes,
      usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(p.wgt, 0, st, 0, bytes);
    device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    w.set(new Float32Array(st.getMappedRange()).subarray(0, p.s1 - p.s0), p.s0);
    st.unmap(); st.destroy();
  }
  return w;
}

let chain = Promise.resolve();   // init and ticks strictly serialized
onmessage = (e) => {
  const m = e.data;
  chain = chain.then(async () => {
    // every term, and every optional query but first-spike latency (ENGINE.md section 2); answered without the device, which init asks for
    if(m.cmd === 'hello') postMessage(helloReply('webgpu', TERM_NAMES, OPTIONAL.filter(q => q !== 'sendT')));
    else if(m.cmd === 'init') await init(m);
    else if(m.cmd === 'watch'){ if(S) S.watch = m.idx; }
    else if(m.cmd === 'sendV'){ if(S) S.sendV = !!m.on; }
    // First-spike latency within a tick.
    // The CUDA child carries it; this engine does not yet, and ENGINE.md's rule is that an engine refuses a term it lacks rather than accepting it in silence, because a timing measure that quietly receives nothing reads as an absence of timing structure in the network.
    else if(m.cmd === 'sendT'){
      if(m.on) postMessage({ cmd:'unsupported', of:'sendT',
        message:'this engine does not report first-spike latency, so the '+
          'timing measures are unavailable on this run' });
    }
    else if(m.cmd === 'inputFrame'){ if(S && S.inputs && S.inputs[m.mi]){
      S.inputs[m.mi].gains = m.gains; S.extDirty = true; } }
    else if(m.cmd === 'tune'){ if(S){

      configure(m); writeRules();
      if(S.plast) ensurePlast();
    } }
    else if(m.cmd === 'getWeights'){ if(S){
      const wc = S.weightsDirty ? await readWeights() : S.w.slice();
      // measured set points travel with the weights, matching the reference engine: without them an unreachable homeostatic target can only be inferred after the fact
      const rc = S.rhoI ? S.rhoI.slice() : null;
      // and the consolidation reference, as the reference engine sends it
      const fc = S.wRefReady ? await readRef() : null;
      postMessage({ cmd:'weights', w:wc, t:S.t, rho:rc, ref:fc },
        [wc.buffer, ...(rc ? [rc.buffer] : []), ...(fc ? [fc.buffer] : [])]);
    } }
    else if(m.cmd === 'setRef') setRef(m.wRef);
    else if(m.cmd === 'query'){ if(S) query(m.idx, m.cap, m.dir === 'out'); }
    // a readback of the ring, the state and the step uniform, for the battery and for debugging an engine change; never used by the page
    else if(m.cmd === 'debug'){ if(S){
      const rb = async (buf, bytes) => { const st = S.device.createBuffer({ size:bytes, usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = S.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, st, 0, bytes); S.device.queue.submit([enc.finish()]);
        await st.mapAsync(GPUMapMode.READ); const out = st.getMappedRange().slice(0); st.unmap(); st.destroy(); return out; };
      const ring = new Int32Array(await rb(S.bufRing, (2 + S.KX)*ROWS*S.n*4));
      const state = new Float32Array(await rb(S.bufState, S.n*SST*4));
      postMessage({ cmd:'debug', kx:S.KX, t:S.t, cur:S.cur, step:Array.from(S.stepU32), ring:Array.from(ring), state:Array.from(state) });
    } }
    else if(m.cmd === 'tick'){ if(S) await tick(m.steps); }
  }).catch(err => {
    postMessage({ cmd:'error', message: err && err.message || String(err) });
  });
};

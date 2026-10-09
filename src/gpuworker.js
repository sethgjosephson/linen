// WebGPU simulation engine.
// Same message protocol and dynamics contract as the CPU reference engine (simworker.js); see ENGINE.md.
import { gradedArrays, freezeGraded } from './nodes.js';
import { checkProtocol, helloReply, TERM_NAMES, OPTIONAL } from './protocol.js';
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
const SLOT = 256;              // dynamic-offset alignment for step uniforms
const MAX_BATCH = 64;          // steps encoded per submission

const STEP_STRUCT = `
struct Step { t:u32, cur:u32, watch:i32, seed:u32, n:u32, syn:u32, slot:u32, plast:u32,
              decE:f32, decI:f32, itauE:f32, itauI:f32,
              aP:f32, aM:f32, decS:f32, wmax:f32, iEta:f32, iAlpha:f32,
              wdep:f32, refrac:f32,
              stpOn:u32, stpU:f32, decX:f32, decR:f32,
              tin:f32, trip:f32, decY:f32, het:f32,
              kRef:f32, consRate:f32, consW:f32, consP:f32,
              decM:f32, stpScale:f32, commit:f32, poolG:u32,
              vmin:f32, stpOrder:u32, kx:u32, pad4:f32,
              dx0:f32, dx1:f32, dx2:f32, dx3:f32, dx4:f32, dx5:f32,
              ix0:f32, ix1:f32, ix2:f32, ix3:f32, ix4:f32, ix5:f32,
              eE:f32, eI:f32, ex0:f32, ex1:f32, ex2:f32, ex3:f32, ex4:f32, ex5:f32 };
struct Spikes { count:atomic<u32>, total:atomic<u32>, list:array<u32> };
`;
const USIZE = 256;  // Step struct bytes (a multiple of 16, as uniforms require; equal to SLOT)
const SST = 12;     // state floats per neuron: v, u, gE, gI, then six receptor channels, padding

const SHADER_UPDATE = STEP_STRUCT + `
@group(0) @binding(0) var<uniform> U: Step;
// Per neuron, 12 floats: a, b, c, d, then the 2007 row's C, k, vr, vt, vpeak, then the graded release threshold, slope and flag (MODEL.md 1).
// (the u32 after vmin in the step struct is the short-term plasticity order, stpOrder)
@group(0) @binding(1) var<storage, read> abcd: array<f32>;        // 12n strided
@group(0) @binding(2) var<storage, read_write> st: array<f32>;    // v,u,gE,gI, six receptor conductances, padding: 12 per neuron
@group(0) @binding(3) var<storage, read> drive: array<f32>;       // dc, namp strided 2n
@group(0) @binding(4) var<storage, read_write> ring: array<atomic<i32>>;  // E rows then I rows
@group(0) @binding(5) var<storage, read_write> spikes: Spikes;
@group(0) @binding(6) var<storage, read_write> fired: array<u32>;
@group(0) @binding(7) var<storage, read_write> vtrace: array<f32>;
// Milliseconds of absolute refractoriness left, one per neuron, in the low byte of each word.
// A separate buffer rather than a fifth field of st, because widening that stride touches the state readback and every index in this shader; this takes the update pass to eight storage buffers, which is the guaranteed minimum.
// Feedback inhibition rides in the same buffer for that reason: the pool a cell is in sits in the top 24 bits of its word, and past the n words come two halves of per-pool spike counts (the step's parity says which half is read and which is written) and each pool's kick per spike as float bits.
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
  var I = drive[i*2u];
  let namp = drive[i*2u+1u];
  if(namp != 0.0){
    let h = i ^ (U.t * 2654435761u) ^ U.seed;
    I += namp * (unif(h) + unif(h ^ 0x9e3779b9u) - 1.0);
  }
  // Feedback inhibition: what the cell's pool delivers this millisecond is its kick per spike times the spikes it fired last millisecond, read from the half of the count table the previous step wrote.
  let rcw = atomicLoad(&refcnt[i]);
  let g = rcw >> 8u;
  let rc = rcw & 0xFFu;
  var pin = 0.0;
  if(g != 0u){
    let G1 = U.poolG;
    let cnt = atomicLoad(&refcnt[n + ((U.t & 1u) ^ 1u)*G1 + g]);
    pin = bitcast<f32>(atomicLoad(&refcnt[n + 2u*G1 + g])) * f32(cnt);
  }
  var ge = 0.0; var gi = 0.0;
  var gx0 = 0.0;
  var gx1 = 0.0;
  var gx2 = 0.0;
  var gx3 = 0.0;
  var gx4 = 0.0;
  var gx5 = 0.0;
  let v0 = st[i*12u];
  let cnd = U.syn == 2u;
  if(U.syn >= 1u){
    ge = st[i*12u+2u]*U.decE + eIn;
    gi = st[i*12u+3u]*U.decI + iIn + pin;
    // current-based: g times the scale; conductance-based: |g| (E - v)
    I += select(ge*U.itauE + gi*U.itauI, abs(ge)*(U.eE - v0) + abs(gi)*(U.eI - v0), cnd);
    // receptor channels 2 and up (MODEL.md 2): their own decay, their own ring rows
    if(U.kx > 0u){ let rx = (2u*${ROWS}u)*n + rowE; gx0 = st[i*12u+4u]*U.dx0 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); I += select(gx0*U.ix0, abs(gx0)*(U.ex0 - v0), cnd); }
    if(U.kx > 1u){ let rx = (3u*${ROWS}u)*n + rowE; gx1 = st[i*12u+5u]*U.dx1 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); I += select(gx1*U.ix1, abs(gx1)*(U.ex1 - v0), cnd); }
    if(U.kx > 2u){ let rx = (4u*${ROWS}u)*n + rowE; gx2 = st[i*12u+6u]*U.dx2 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); I += select(gx2*U.ix2, abs(gx2)*(U.ex2 - v0), cnd); }
    if(U.kx > 3u){ let rx = (5u*${ROWS}u)*n + rowE; gx3 = st[i*12u+7u]*U.dx3 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); I += select(gx3*U.ix3, abs(gx3)*(U.ex3 - v0), cnd); }
    if(U.kx > 4u){ let rx = (6u*${ROWS}u)*n + rowE; gx4 = st[i*12u+8u]*U.dx4 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); I += select(gx4*U.ix4, abs(gx4)*(U.ex4 - v0), cnd); }
    if(U.kx > 5u){ let rx = (7u*${ROWS}u)*n + rowE; gx5 = st[i*12u+9u]*U.dx5 + f32(atomicLoad(&ring[rx]))*(1.0/${SCALE}.0); atomicStore(&ring[rx], 0); I += select(gx5*U.ix5, abs(gx5)*(U.ex5 - v0), cnd); }
  } else {
    I += eIn + iIn + pin;
  }
  var v = st[i*12u]; var u = st[i*12u+1u];
  let A = abcd[i*12u]; let Bq = abcd[i*12u+1u];
  let vr = abcd[i*12u+6u];
  // the recovery variable is measured from the rest (Izhikevich 2007)
  let uRef = vr;
  // Absolute refractoriness, matching the reference engine exactly: the membrane is held at reset and cannot reach threshold whatever the drive, the recovery variable keeps advancing, and the synaptic conductances keep decaying and accumulating.
  if(rc > 0u){
    atomicStore(&refcnt[i], rcw - 1u);
    u += A*(Bq*(v - uRef) - u);
    st[i*12u+1u] = u; st[i*12u+2u] = ge; st[i*12u+3u] = gi; st[i*12u+4u] = gx0; st[i*12u+5u] = gx1; st[i*12u+6u] = gx2; st[i*12u+7u] = gx3; st[i*12u+8u] = gx4; st[i*12u+9u] = gx5;
    if(i32(i) == U.watch){ vtrace[U.slot] = v; }
    return;
  }
  // the membrane floor after each half step, as the reference does
    // Izhikevich 2007: C dv = k (v - vr)(v - vt) - u + I
    let C = abcd[i*12u+4u]; let k = abcd[i*12u+5u]; let vt = abcd[i*12u+7u];
    if(abcd[i*12u+11u] > 0.5){
      // a graded cell's membrane is passive: the leak is the row's slope at rest, k (vt - vr), and the potential is capped at vpeak
      let gl = k*(vt - vr);
      v += 0.5*(-gl*(v - vr) - u + I)/C; v = max(v, U.vmin);
      v += 0.5*(-gl*(v - vr) - u + I)/C; v = max(v, U.vmin);
      v = min(v, abcd[i*12u+8u]);
    } else {
      v += 0.5*(k*(v - vr)*(v - vt) - u + I)/C;
      v = max(v, U.vmin);
      v += 0.5*(k*(v - vr)*(v - vt) - u + I)/C;
      v = max(v, U.vmin);
    }
    u += A*(Bq*(v - vr) - u);
  // A graded cell (MODEL.md 1): no spike, no reset; every millisecond it enters the event list with its release as the amplitude byte (0 to 254; a spike carries 255), and the delivery pass scales its synapses.
  if(abcd[i*12u+11u] > 0.5){
    let r = clamp((v - abcd[i*12u+9u])/abcd[i*12u+10u], 0.0, 1.0);
    let ab = u32(r*254.0 + 0.5);
    if(ab > 0u){ let k = atomicAdd(&spikes.count, 1u); spikes.list[k] = i | (ab << 24u); }
    st[i*12u] = v; st[i*12u+1u] = u; st[i*12u+2u] = ge; st[i*12u+3u] = gi; st[i*12u+4u] = gx0; st[i*12u+5u] = gx1; st[i*12u+6u] = gx2; st[i*12u+7u] = gx3; st[i*12u+8u] = gx4; st[i*12u+9u] = gx5;
    if(i32(i) == U.watch){ vtrace[U.slot] = v; }
    return;
  }
  let vpk = abcd[i*12u+8u];
  var spiked = false;
  if(v >= vpk){
    v = abcd[i*12u+2u]; u += abcd[i*12u+3u];
    spiked = true;
    atomicStore(&refcnt[i], (g << 8u) | u32(U.refrac));
    if(g != 0u){ atomicAdd(&refcnt[n + (U.t & 1u)*U.poolG + g], 1u); }
    // a count per tick, as the reference and CUDA engines report it: this thread is the only writer of fired[i], so no atomic is needed.
    // A 0/1 flag here would cap every rate read from it, the trainer's probes and synaptic scaling's included, at 1000/steps Hz.
    fired[i] = fired[i] + 1u;
    atomicAdd(&spikes.total, 1u);
    let k = atomicAdd(&spikes.count, 1u);
    spikes.list[k] = i | (255u << 24u);      // index in 24 bits, amplitude byte 255 for a spike
  }
  st[i*12u] = v; st[i*12u+1u] = u; st[i*12u+2u] = ge; st[i*12u+3u] = gi; st[i*12u+4u] = gx0; st[i*12u+5u] = gx1; st[i*12u+6u] = gx2; st[i*12u+7u] = gx3; st[i*12u+8u] = gx4; st[i*12u+9u] = gx5;
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
  if(g >= U.poolG){ return; }
  atomicStore(&refcnt[U.n + ((U.t & 1u) ^ 1u)*U.poolG + g], 0u);
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
@compute @workgroup_size(1)
fn writeIndirect(){
  ind[0] = (atomicLoad(&spikes.count) + ${DWG - 1}u) / ${DWG}u;
  ind[1] = 1u; ind[2] = 1u;
}
`;

const SHADER_DELIVER = STEP_STRUCT + `
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
  } else if(U.stpOn == 1u){
    if(U.stpOrder == 1u){ stpR[i] = stpR[i] + U.stpU*(1.0 - stpR[i]); }   // stpOrder 1: facilitation before release
    let rel = stpR[i]*stpX[i];
    stpX[i] = stpX[i] - rel;                 // depletion
    if(U.stpOrder == 0u){ stpR[i] = stpR[i] + U.stpU*(1.0 - stpR[i]); }   // as published: facilitation after release
    sf = rel*U.stpScale;
  }
  for(var s = s0; s < s1; s++){
    let pd = postd[s];
    let j = pd & 0x7ffffffu;
    let d = pd >> 27u;
    let row = (U.cur + d) % ${ROWS}u;
    // Route on the stored weight, not the scaled one: the release factor is never negative but can be zero, and a zero must not be read as inhibitory.
    let w0 = wgt[s];
    let fx = i32(round(w0 * sf * ${SCALE}.0));
    let ch = ((rid[s >> 2u] >> ((s & 3u) * 8u)) & 0xffu) >> 5u;
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
@group(0) @binding(8) var<storage, read> alphaI: array<f32>;  // per-neuron Vogels alpha
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
    if(wj > 0.0){ wgt[s] = min(R.wmax,
      max(1e-4, wj - R.aM*Kpost[j]*mix(1.0, wj, R.wdep) + R.tin)); }
    else {
      let d = softBound(wj, -R.iEta*(Kpost[j] - alphaI[j]), R.wmax, R.wdep);
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
@group(0) @binding(4) var<storage, read> revSyn: array<u32>;
@group(0) @binding(5) var<storage, read> revPre: array<u32>;
@group(0) @binding(6) var<storage, read_write> wgt: array<f32>;
// Kpre and the slow trace interleaved, two floats per neuron: [i*2] is the fast presynaptic trace, [i*2+1] the slow postsynaptic one for the triplet term.
// They share a binding because this pass is at the eight storage buffers WebGPU guarantees and the heterosynaptic reference weight needs one.
// Kpost is not in here: it is the hot read in the outgoing pass, which has a slot free, so it stays a plain array.
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
  let z = ktr[i*2u + 1u];
  // The gate is the slow trace against its value at the target rate, cubed, so competition engages when the neuron fires above its set point.
  // The pull fraction saturates at one: a synapse can return fully to its reference in one step but never overshoot it.
  // The gate is built from this neuron's trace and its set point, so it is shared; only the amplitude multiplying it is per rule.
  let g = z/U.kRef;
  let gate = g*g*g;
  for(var r = inStart[i]; r < inStart[i+1u]; r++){
    let s = revSyn[r];
    if(s < P.base || s >= P.sEnd){ continue; }
    let sl = s - P.base;
    let ri = (rid[sl >> 2u] >> ((sl & 3u) * 8u)) & 0x1fu;
    if(ri == 0u){ continue; }
    let R = RT.r[ri];
    if(U.commit > 0.5 && R.consW > 0.0 && wRef[sl] >= R.consW*0.9){ continue; }
    let kp = ktr[revPre[r]*2u];
    let wp = wgt[sl];
    if(wp > 0.0){
      // one clamp on the combined result, and the pull reads the weight from before this step's potentiation, both as the reference does
      var nw = wp + (R.aP + R.trip*z)*kp;
      if(R.het > 0.0){ nw = nw - min(1.0, R.het*gate)*(wp - wRef[sl]); }
      wgt[sl] = min(R.wmax, max(1e-4, nw));
    }
    else {
      let d = softBound(wp, -R.iEta*kp, R.wmax, R.wdep);
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
  // clamped to the two fixed points, as in the reference: the step is forward Euler over a cubic and a short timescale otherwise diverges
  let nr = r + U.consRate*(wj - r - R.consP*r*(R.consW*0.5 - r)*(R.consW - r));
  wRef[s] = clamp(nr, 0.0, R.consW);
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
  ktr[i*2u] = ktr[i*2u] + 1.0;
  Kpost[i] = Kpost[i] + 1.0;
  ktr[i*2u + 1u] = ktr[i*2u + 1u] + 1.0;
}
`;

// Per-millisecond recovery of both short-term plasticity variables.
// Exact exponentials rather than Euler, matching the reference, so the time constants hold whatever the step size.
const SHADER_STP = STEP_STRUCT + `
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<storage, read_write> stpX: array<f32>;
@group(0) @binding(2) var<storage, read_write> stpR: array<f32>;
@compute @workgroup_size(${WG})
fn stpRecover(@builtin(global_invocation_id) gid: vec3<u32>){
  let i = gid.x;
  if(i >= U.n){ return; }
  stpX[i] = 1.0 - (1.0 - stpX[i])*U.decX;
  stpR[i] = U.stpU + (stpR[i] - U.stpU)*U.decR;
}
`;

const SHADER_DECAY = STEP_STRUCT + `
@group(0) @binding(0) var<uniform> U: Step;
@group(0) @binding(1) var<storage, read_write> ktr: array<f32>;
@group(0) @binding(2) var<storage, read_write> Kpost: array<f32>;
@compute @workgroup_size(${WG})
fn decay(@builtin(global_invocation_id) gid: vec3<u32>){
  let i = gid.x;
  if(i >= U.n){ return; }
  ktr[i*2u] = ktr[i*2u] * U.decS;
  Kpost[i] = Kpost[i] * U.decM;
  ktr[i*2u + 1u] = ktr[i*2u + 1u] * U.decY;
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
  if(!changed && S.driveValid && !S.extDirty) return false;
  const d = S.driveHost, n = S.n, bias = S.bias;
  for(let i=0;i<n;i++){ d[i*2] = bias[i]; d[i*2+1] = 0; }
  for(const p of S.allProtos){
    if(!p._s) continue;
    // gain: the stimulus's per-cell multiplier (spread), as the other two engines read it
    const k = p.amp * p._s, off = p.mode === 3 ? 1 : 0, idx = p.idx, g = p.gain;
    for(let q=0;q<idx.length;q++) d[idx[q]*2 + off] += g ? k*g[q] : k;
  }
  // framed input maps fold into the DC half of the drive array
  for(const im of (S.inputs || [])){
    const { chStart, chIdx, chW, gains, amp } = im;
    const NC = Math.min(gains.length, chStart.length - 1);   // stale frames after a retune
    for(let c=0;c<NC;c++){
      const g = gains[c] * amp;
      if(!g) continue;
      for(let k=chStart[c];k<chStart[c+1];k++) d[chIdx[k]*2] += g * chW[k];
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
  const tE = m.tauE > 0 ? m.tauE : 3, tI = m.tauI > 0 ? m.tauI : 8;
  S.decE = Math.exp(-1/tE); S.decI = Math.exp(-1/tI);
  // psc 0: peak current (NEST, Brian); psc 1: total charge; psc 2: the exact integral over the step, tau (1 - dec), matching the reference engine's configure (itauOf there)
  S.psc = Math.max(0, Math.min(2, m.psc|0));
  const itauOf = (dec, tau) => S.psc === 2 ? tau*(1 - dec) : S.psc === 1 ? 1 - dec : 1;
  S.itauE = itauOf(S.decE, tE); S.itauI = itauOf(S.decI, tI);
  S.allProtos = (m.protocols || []).map(p => ({ ...p, _s:-1 }));
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
  S.stpScale = (m.stpNorm|0) === 1 ? 1/(S.stpOrder ? S.stpU*(2 - S.stpU) : S.stpU) : 1;
  S.decX = Math.exp(-1/(m.stpTauD > 0 ? +m.stpTauD : 200));
  S.decR = Math.exp(-1/(m.stpTauF > 0 ? +m.stpTauF : 600));
  // Triplet potentiation (Pfister and Gerstner 2006) and the transmitter drip.
  // Both ride on state the engine already keeps: tin is a constant in the outgoing pass, trip needs one more per-neuron trace.
  S.trip = m.trip > 0 ? +m.trip : 0;
  S.tin = m.tin > 0 ? +m.tin : 0;
  const tauY = m.tauY > 0 ? +m.tauY : 114;
  S.decY = Math.exp(-1/tauY);
  // Heterosynaptic competition and consolidation.
  // The gate is dimensionless, the slow trace against its value at the target rate, so het keeps one scale across targets and trace constants.
  // Both read the same per-synapse reference weight.
  S.het = m.het > 0 ? +m.het : 0;
  S.kRef = Math.max(1e-6, tauY*(m.iRho || 5)/1000);
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
  S.scaleOn = (m.scale|0) === 1;
  S.sEta = m.sEta > 0 ? m.sEta : 0.001;
  // per-neuron set points (measured-baseline mode): first calS seconds run with the rules off while rates record, then lock personal targets
  S.rhoMode = (m.rhoMode|0) === 1 ? 1 : 0;
  S.calMs = (m.calS > 0 ? m.calS : 10) * 1000;
  S.tauSv = tauS; S.tauMv = tauM;
  if(S.plast && S.rhoMode === 1 && !S.rhoI){
    S.calibrating = true; S.calStart = S.t;
    if(S.spkAcc) S.spkAcc.fill(0);
  } else if(!S.plast) S.calibrating = false;
  // A tune of iRho, tauS or tauM moves the Vogels target, and the shader reads it per cell from bufAlpha, which was filled once when plasticity was set up; the reference and CUDA engines use the new target at once.
  // Measured set points own bufAlpha, so only the fixed target is written.
  if(S.bufAlpha && !S.rhoI) S.device.queue.writeBuffer(S.bufAlpha, 0, new Float32Array(S.n).fill(S.iAlpha));
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
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(p.wgt, 0, p.wRef, 0, bytes);
    device.queue.submit([enc.finish()]);
  }
  S.wRefReady = true;
  S.needWRef = false;
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
  // The outgoing pass reads the reference too now, so it is built here with the others rather than once at setup: either term can be switched on by a tune long after setup ran, which reallocates these buffers.
  if(S.bglOut) S.bgPlastOut = S.parts.map(p => device.createBindGroup({ layout:S.bglOut, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, p.uni), e(2, S.bufSpikes),
    e(3, S.bufPre), e(4, p.postd), e(5, p.wgt), e(6, S.bufKpost), e(7, p.pm),
    e(8, S.bufAlpha), e(9, p.wRef || S.bufWRefNull), e(10, S.bufRules) ] }));
  S.bgPlastIn = S.parts.map(p => device.createBindGroup({ layout:S.bglIn, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, p.uni), e(2, S.bufSpikes),
    e(3, S.bufInStart), e(4, S.bufRevSyn), e(5, S.bufRevPre),
    e(6, p.wgt), e(7, S.bufKtr), e(8, p.pm),
    e(9, p.wRef || S.bufWRefNull), e(10, S.bufRules) ] }));
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
        revSyn = new Uint32Array(m2), revPre = new Uint32Array(m2);
  for(let i=0;i<n;i++)
    for(let s=preStart[i]; s<preStart[i+1]; s++){
      const at = inStart[post[s]] + cur[post[s]]++;
      revSyn[at] = s; revPre[at] = i;
    }
  const ST = GPUBufferUsage.STORAGE;
  const up = (usage, arr) => { const b = device.createBuffer({
      size: Math.max(16, arr.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, arr); return b; };
  S.bufInStart = up(ST, inStart);
  S.bufRevSyn = up(ST, revSyn);
  S.bufRevPre = up(ST, revPre);

  const C = GPUShaderStage.COMPUTE;
  const sb = (binding, type) => ({ binding, visibility:C, buffer:{ type } });
  const dynU = { binding:0, visibility:C,
    buffer:{ type:'uniform', hasDynamicOffset:true, minBindingSize:USIZE } };
  const bglOut = device.createBindGroupLayout({ entries:[ dynU,
    { binding:1, visibility:C, buffer:{ type:'uniform' } },
    sb(2,'storage'), sb(3,'read-only-storage'), sb(4,'read-only-storage'),
    sb(5,'storage'), sb(6,'read-only-storage'), sb(7,'read-only-storage'),
    sb(8,'read-only-storage'), sb(9,'read-only-storage'),
    { binding:10, visibility:C, buffer:{ type:'uniform' } } ] });
  // eight storage buffers, the WebGPU guaranteed maximum: spikes, the three reverse-index arrays, the weights, the interleaved traces, the mask, and the reference weight.
  // Nothing else fits here without freeing a slot.
  const bglIn = device.createBindGroupLayout({ entries:[ dynU,
    { binding:1, visibility:C, buffer:{ type:'uniform' } },
    sb(2,'storage'), sb(3,'read-only-storage'), sb(4,'read-only-storage'),
    sb(5,'read-only-storage'), sb(6,'storage'), sb(7,'read-only-storage'),
    sb(8,'read-only-storage'), sb(9,'read-only-storage'),
    { binding:10, visibility:C, buffer:{ type:'uniform' } } ] });
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
    sb(1,'storage'), sb(2,'storage') ] });
  const mkPipe = (code, entry, bgl) => device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts:[bgl] }),
    compute:{ module: device.createShaderModule({ code }), entryPoint: entry } });
  S.pPlastOut = mkPipe(SHADER_PLAST_OUT, 'plastOut', bglOut);
  S.pPlastIn = mkPipe(SHADER_PLAST_IN, 'plastIn', bglIn);
  S.pBump = mkPipe(SHADER_BUMP, 'bump', bglBump);
  S.pDecay = mkPipe(SHADER_DECAY, 'decay', bglDecay);
  S.pScale = mkPipe(SHADER_SCALE, 'scalePass', bglScale);
  S.pCons = mkPipe(SHADER_CONS, 'consolidate', bglCons);

  const e = (binding, buffer, extra) => ({ binding, resource:{ buffer, ...extra } });
  writeRules();
  S.bufAlpha = device.createBuffer({ size:S.n*4,
    usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(S.bufAlpha, 0, new Float32Array(S.n).fill(S.iAlpha));
  S.bglOut = bglOut;
  // Reference weights: a second copy of every weight, so they are allocated only when heterosynaptic competition or consolidation is actually on.
  // With them off, p.wRef aliases p.wgt, which keeps the bind group static and is never read, because the shader guards on the per-rule R.het.
  // Kept on S because either term can be switched on by a tune long after this ran, which needs these two groups rebuilt against the new buffers.
  S.bglIn = bglIn; S.bglCons = bglCons;
  ensureWRef();
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
    e(0, S.bufStep, { size:USIZE }), e(1, S.bufKtr), e(2, S.bufKpost) ] });
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
  const abcd = new Float32Array(12*n);
  for(let i=0;i<n;i++){ const t = T[m.ntype[i]] && T[m.ntype[i]].f7;
    if(!t) throw new Error('gpu engine: every neuron type needs a 2007 row (f7)');
    abcd[i*12]=t.a; abcd[i*12+1]=t.b; abcd[i*12+2]=t.c; abcd[i*12+3]=t.d;
    { abcd[i*12+4]=t.C; abcd[i*12+5]=t.k; abcd[i*12+6]=t.vr; abcd[i*12+7]=t.vt; abcd[i*12+8]=t.vpeak; } }
  // graded rows: threshold, slope and a flag in the last three slots; the event list packs the index in 24 bits, which caps a graded net
  const G = gradedArrays(m.ntype, T);
  if(G){
    if(n > 0xffffff) throw new Error('gpu engine: graded rows need the neuron index in 24 bits (16.7 M cells at most)');
    for(let i=0;i<n;i++) if(G.grd[i]){ abcd[i*12+9] = G.thr[i]; abcd[i*12+10] = G.slope[i]; abcd[i*12+11] = 1; }
  }
  m.pmask = freezeGraded(m.pmask || null, m.preStart, G && G.grd, m.w.length);
  // The initial membrane state arrives from the host (initState in rand.js), the same bytes the reference engine and the CUDA child start from.
  // Refused when absent rather than seeded here, so no engine draws its own.
  if(!m.v0 || !m.u0 || m.v0.length !== n || m.u0.length !== n)
    throw new Error('gpu engine: init needs v0 and u0 (rand.js initState)');
  const st = new Float32Array(SST*n);
  for(let i=0;i<n;i++){ st[i*SST] = m.v0[i]; st[i*SST+1] = m.u0[i]; }

  // CPU copies kept for query and getWeights
  S = { n, device, t:0, cur:0, watch:-1, seed:(m.seed>>>0) || 1,
        preStart:m.preStart, post:m.post, w:m.w, delay:m.delay,
        bias:m.bias, driveHost:new Float32Array(2*n), driveValid:false,
        zeroN:new Uint32Array(n), zero1:new Uint32Array(1) };
  configure(m);

  const B = (usage, size) => device.createBuffer({ size, usage });
  const up = (usage, arr) => { const b = device.createBuffer({
      size: Math.max(16, arr.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, arr); return b; };
  const ST = GPUBufferUsage.STORAGE;

  S.bufAbcd = up(ST, abcd);
  S.bufState = up(ST | GPUBufferUsage.COPY_SRC, st);
  S.bufDrive = B(ST | GPUBufferUsage.COPY_DST, 2*n*4);
  S.bufRing = B(ST | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, (2 + S.KX)*ROWS*n*4);   // E, I, then a row set per receptor channel
  S.bufSpikes = B(ST | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, (2+n)*4);
  S.bufFired = B(ST | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, n*4);
  S.bufVtrace = B(ST | GPUBufferUsage.COPY_SRC, MAX_BATCH*4);
  // Feedback inhibition shares the refractory buffer (see the binding): the pool per cell above the refractory byte, two halves of per-pool spike counts, then each pool's kick per spike as float bits.
  S.poolG = (m.pool && m.pool.length === n && m.poolK && m.poolK.length >= 2) ? m.poolK.length : 1;
  if(m.pool && m.pool.length && S.poolG === 1)
    throw new Error('gpu engine: init pool arrays do not match the network');
  const rcInit = new Uint32Array(n + 3*S.poolG);
  if(S.poolG > 1){
    for(let i=0;i<n;i++) rcInit[i] = (m.pool[i] & 0xFFFFFF) << 8;
    const kb = new Uint32Array(Float32Array.from(m.poolK).buffer);
    for(let g=0;g<S.poolG;g++) rcInit[n + 2*S.poolG + g] = kb[g];
  }
  S.bufRefcnt = up(ST | GPUBufferUsage.COPY_DST, rcInit);
  // Always allocated, even with the term off, so the delivery bind group is built once and never rebuilt on a tune.
  // Two floats per neuron is nothing beside the synapse arrays.
  S.bufStpX = B(ST | GPUBufferUsage.COPY_DST, n*4);
  S.bufStpR = B(ST | GPUBufferUsage.COPY_DST, n*4);
  S.stpSeed = true;
  S.bufInd = B(ST | GPUBufferUsage.INDIRECT, 12);
  S.bufKtr = B(ST | GPUBufferUsage.COPY_DST, n*8);   // Kpre, Kslow interleaved
  S.bufKpost = B(ST | GPUBufferUsage.COPY_DST, n*4);
  device.queue.writeBuffer(S.bufKtr, 0, new Float32Array(2*n));
  device.queue.writeBuffer(S.bufKpost, 0, new Float32Array(n));
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
  const bglInd = device.createBindGroupLayout({ entries:[ sb(0,'storage'), sb(1,'storage') ] });
  const bglDeliver = device.createBindGroupLayout({ entries:[
    { binding:0, visibility:C, buffer:{ type:'uniform', hasDynamicOffset:true, minBindingSize:USIZE } },
    { binding:1, visibility:C, buffer:{ type:'uniform' } },
    sb(2,'storage'), sb(3,'storage'),
    sb(4,'read-only-storage'), sb(5,'read-only-storage'), sb(6,'read-only-storage'),
    sb(7,'storage'), sb(8,'storage'), sb(9,'read-only-storage') ] });
  const bglStp = device.createBindGroupLayout({ entries:[
    { binding:0, visibility:C, buffer:{ type:'uniform', hasDynamicOffset:true, minBindingSize:USIZE } },
    sb(1,'storage'), sb(2,'storage') ] });

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
    e(0, S.bufSpikes), e(1, S.bufInd) ] });
  S.bgDeliver = S.parts.map(p => device.createBindGroup({ layout:bglDeliver, entries:[
    e(0, S.bufStep, { size:USIZE }),
    e(1, p.uni), e(2, S.bufSpikes), e(3, S.bufRing),
    e(4, S.bufPre), e(5, p.postd), e(6, p.wgt),
    e(7, S.bufStpX), e(8, S.bufStpR), e(9, p.pm) ] }));
  S.bgStp = device.createBindGroup({ layout:bglStp, entries:[
    e(0, S.bufStep, { size:USIZE }), e(1, S.bufStpX), e(2, S.bufStpR) ] });

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
    S.stepF32[27] = S.het; S.stepF32[28] = S.kRef || 1;
    S.stepF32[29] = (S.consStep || 1200)/(S.tauCons || 1200000);
    S.stepF32[34] = S.commit ? 1 : 0;
    S.stepF32[30] = S.consW || 1; S.stepF32[31] = S.consP || 10;
    S.stepF32[32] = S.decM; S.stepF32[33] = S.stpScale;
    S.stepU32[35] = S.poolG || 1;
    S.stepF32[36] = S.vmin;
    S.stepU32[37] = S.stpOrder ? 1 : 0;
    S.stepU32[38] = S.KX || 0;
    for(let x = 0; x < 6; x++){ S.stepF32[40 + x] = S.chanDec && x < S.chanDec.length ? S.chanDec[x] : 0; S.stepF32[46 + x] = S.chanItau && x < S.chanItau.length ? S.chanItau[x] : 0; }
    S.stepF32[52] = S.eRevE; S.stepF32[53] = S.eRevI;
    for(let x = 0; x < 6; x++) S.stepF32[54 + x] = S.chanErev && x < S.chanErev.length ? S.chanErev[x] : 0;
    device.queue.writeBuffer(S.bufStep, k*SLOT, S.stepHost);
  }
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  const wgN = Math.ceil(S.n / WG);
  for(let k=0;k<count;k++){
    const off = [k*SLOT];
    pass.setPipeline(S.pClear); pass.setBindGroup(0, S.bgClear);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(S.pUpdate); pass.setBindGroup(0, S.bgUpdate, off);
    pass.dispatchWorkgroups(wgN);
    if(S.poolG > 1){
      pass.setPipeline(S.pPool); pass.setBindGroup(0, S.bgPool, off);
      pass.dispatchWorkgroups(Math.ceil(S.poolG / WG));
    }
    pass.setPipeline(S.pInd); pass.setBindGroup(0, S.bgInd);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(S.pDeliver);
    for(const bg of S.bgDeliver){
      pass.setBindGroup(0, bg, off);
      pass.dispatchWorkgroupsIndirect(S.bufInd, 0);
    }
    if(S.plast){
      if(!S.calibrating){                    // measure first, change nothing
        S.weightsDirty = true;
        pass.setPipeline(S.pPlastOut);
        for(const bg of S.bgPlastOut){
          pass.setBindGroup(0, bg, off);
          pass.dispatchWorkgroupsIndirect(S.bufInd, 0);
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
      pass.dispatchWorkgroups(wgN);
    }
  }
  pass.end();
  device.queue.submit([enc.finish()]);
  S.t += count; S.cur = (S.cur + count) % ROWS;
}

async function tick(steps){
  const { device, n } = S;
  // a tune can switch either term on after the plasticity pipelines were built, which is when the reference weights first have to exist
  if(S.needWRef && S.revReady){
    ensureWRef();
    rebuildWRefGroups();
  }
  // Seed the short-term plasticity state on the edge where the term turns on, which is where the reference allocates it: resources full, release probability at U.
  if(S.stp && S.stpSeed){
    S.stpSeed = false;
    device.queue.writeBuffer(S.bufStpX, 0, new Float32Array(n).fill(1));
    device.queue.writeBuffer(S.bufStpR, 0, new Float32Array(n).fill(S.stpU));
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
      postMessage({ cmd:'weights', w:wc, t:S.t, rho:rc },
        rc ? [wc.buffer, rc.buffer] : [wc.buffer]);
    } }
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

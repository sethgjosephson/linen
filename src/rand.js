// Seeded randomness, in one place.
// Every draw in the project that has to be reproducible comes from here: the pair hash that decides synapse existence and weight, the sequential generator the scatter and modifiers use, the counter-based noise the engines draw, and the initial membrane state every engine starts from.
// The WebGPU shader and the CUDA kernels carry their own copies of pcg and the noise construction because they cannot import; those copies are named in ENGINE.md section 9 and must stay identical to these.

// 32-bit avalanche mixer, two rounds of the lowbias32 constant.
// Signed result; callers wanting [0, 2^32) apply >>> 0.
export function h32(x){
  x = Math.imul(x ^ x>>>16, 0x45d9f3b);
  x = Math.imul(x ^ x>>>16, 0x45d9f3b);
  return x ^ x>>>16;
}

// Uniform in [0, 1) from two stable neuron identities and a seed.
// Synapse existence and weight are pure functions of this, never of iteration order or of any other pair, which is what makes rewiring additive and lets the wiring be split across workers exactly (nodes.js).
export function pairHash(aSrc, aIdx, bSrc, bIdx, seed){
  let h = seed | 0;
  h = h32(h ^ Math.imul(aSrc, 0x9e3779b1));
  h = h32(h ^ Math.imul(aIdx, 0x85ebca6b));
  h = h32(h ^ Math.imul(bSrc, 0xc2b2ae35));
  h = h32(h ^ Math.imul(bIdx, 0x27d4eb2f));
  return (h >>> 0) / 4294967296;
}

// Standard normal from two pair hashes at offset seeds (Box-Muller).
export function pairGauss(aSrc, aIdx, bSrc, bIdx, seed){
  const u = 1 - pairHash(aSrc, aIdx, bSrc, bIdx, seed + 101);
  const v = pairHash(aSrc, aIdx, bSrc, bIdx, seed + 202);
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}

// Sequential generator for a seed (mulberry32).
// For the places where a stream is the right tool: scatter positions, the modifiers, the cluster boost, which by construction references realized topology.
export function rng(seed){ let t = (seed>>>0)+0x9e3779b9;
  return function(){ t += 0x6D2B79F5; let r = Math.imul(t ^ t>>>15, 1|t);
    r ^= r + Math.imul(r ^ r>>>7, 61|r); return ((r ^ r>>>14)>>>0)/4294967296; }; }

// Per-millisecond engine noise, counter based: the draw for a neuron is a pure function of (neuron, t, seed), so it does not depend on how many draws came before it and cannot drift between engines.
// PCG output hash, and the sum of two uniforms minus one, a triangular distribution on
// [-1, 1] with variance 1/6. The WGSL and CUDA copies compute the same
// bits in f32.
export function pcg(x){
  let st = (Math.imul(x >>> 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul((st >>> ((st >>> 28) + 4)) ^ st, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}
export const unif = x => pcg(x) / 4294967295;
export function noiseDraw(i, t, seed){
  const h = (i ^ Math.imul(t, 2654435761) ^ seed) >>> 0;
  return unif(h) + unif((h ^ 0x9e3779b9) >>> 0) - 1;
}

// Computed here on the host from one xorshift stream keyed by the seed and sent in the init message, so the reference engine, the WebGPU worker and the CUDA child start from the same bytes rather than each seeding its own generator (ENGINE.md sections 2 and 4). c, vr and b are rounded to f32 first, as the engines hold them.
// The initial membrane state, the same bytes on every engine: v at the row's rest plus up to 10 mV (under half the rest-to-threshold gap, so a row with a 10 mV gap, the fly's graded relays, never starts a cell on the unstable point), and u = b (v - vr), the recovery variable measured from the rest as the 2007 form has it.
// A classic row (a 2003 preset carried as a 2007 row, form.js) starts where the 2003 form started, at c plus up to 10 mV, so a migrated scene begins from the bytes it always did.
export function initState(seed, ntype, types){
  const n = ntype.length;
  const v0 = new Float32Array(n), u0 = new Float32Array(n);
  let st = ((seed|0) || 1) * 2654435761 >>> 0;
  const urand = () => {
    st ^= st << 13; st >>>= 0; st ^= st >> 17; st ^= st << 5; st >>>= 0;
    return st / 4294967296;
  };
  for(let i=0;i<n;i++){
    const t = types[ntype[i]] && types[ntype[i]].f7;
    if(!t) throw new Error('initState: every neuron type needs a 2007 row (f7)');
    if(t.classic){
      v0[i] = Math.fround(t.c) + urand()*10;
      u0[i] = Math.fround(t.b) * (v0[i] - Math.fround(t.vr));
      continue;
    }
    v0[i] = Math.fround(t.vr) + urand()*Math.min(10, 0.5*(t.vt - t.vr));
    u0[i] = Math.fround(t.b) * (v0[i] - Math.fround(t.vr));
  }
  return { v0, u0 };
}

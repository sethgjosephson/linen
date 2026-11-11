// The reference engine's step and plasticity as CUDA kernels, used by the engine child (engine.cu), which cuda/parity.mjs drives.
// Arithmetic and semantics mirror src/simworker.js exactly; the plasticity decomposition (outgoing pass, incoming pass, trace pass as sequential kernels over disjoint weight slices) is the one the WebGPU engine proved against the cross-engine battery.
#pragma once
#include <cuda_runtime.h>

#define ROWS 18

// Short-term plasticity per rule, as classes (stpClassTable in src/nodes.js): per class U, the depression and facilitation decays and the release scale, its on flag, and the class of each rule row.
// The state is per presynaptic cell and class at [c*n + i]; with one class the kernels take the single-class path they always had.
#define STP_CMAX 16
__constant__ float4 cStpP[STP_CMAX];
__constant__ int cStpOn[STP_CMAX];
__constant__ unsigned char cStpCls[32];
__constant__ int cStpC;
// Every class's release at a spike of cell i into rc, matching stpRelease() in src/simworker.js; a class that is off releases 1.
__device__ inline void stpReleaseC(int i, int n, float* stpX, float* stpR, int stpOrder, float* rc){
  for(int c = 0; c < cStpC; c++){
    float f = 1.0f;
    if(cStpOn[c]){
      const size_t k = (size_t)c*n + i;
      const float U = cStpP[c].x;
      if(stpOrder) stpR[k] = stpR[k] + U*(1.0f - stpR[k]);
      float rel = stpR[k]*stpX[k];
      stpX[k] = stpX[k] - rel;
      if(!stpOrder) stpR[k] = stpR[k] + U*(1.0f - stpR[k]);
      f = rel*cStpP[c].w;
    }
    rc[c] = f;
  }
}
// The release a synapse takes: its rule's class from the entry's per-class amplitudes, or the entry's one amplitude.
__device__ __forceinline__ float stpAmpOf(const float* dlvAmpC, unsigned int k, const unsigned char* pm, int s, float a){
  return dlvAmpC ? dlvAmpC[(size_t)k*STP_CMAX + cStpCls[pm[s] & 31]] : a;
}

// Per-millisecond noise, counter based: the draw for a neuron is a pure function of (neuron, t, seed), the same PCG hash and pair-of-uniforms construction as rand.js noiseDraw and the WebGPU shader, computed in f32 so all three engines draw the same bits.
// Triangular on [-1, 1].
__device__ inline unsigned int pcgH(unsigned int x){
  unsigned int s = x*747796405u + 2891336453u;
  unsigned int word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}
__device__ inline float unifH(unsigned int x){ return pcgH(x) * (1.0f/4294967295.0f); }
__device__ inline float noiseDraw(unsigned int i, unsigned int t, unsigned int seed){
  unsigned int h = i ^ (t*2654435761u) ^ seed;
  return unifH(h) + unifH(h ^ 0x9e3779b9u) - 1.0f;
}
// the uniform draw a Poisson row fires on (rand.js poissonDraw): its own salt, so a Poisson cell under a noise stimulus is not correlated with its noise
__device__ inline float poissonDraw(unsigned int i, unsigned int t, unsigned int seed){
  return unifH(i ^ (t*2654435761u) ^ seed ^ 0x7f4a7c15u);
}
// The cell's draw with its stimulus's memory and shared part (MODEL.md 2): an Ornstein-Uhlenbeck filter per cell when the decay nA is above zero, and a fraction nC of a draw keyed on the stimulus (NOISE_SHARED + its id), filtered the same way; nz and nzs are the two states.
#define NOISE_SHARED 0x40000000u
__device__ inline float noiseAt(int i, unsigned int t, unsigned int seed,
    const float* nA, const float* nK, const float* nC, const int* nId, float* nz, float* nzs){
  float d = noiseDraw((unsigned int)i, t, seed);
  if(!nA) return d;
  const float a = nA[i];
  if(a > 0.0f){ float z = a*nz[i] + nK[i]*d; nz[i] = z; d = z; }
  const float c = nC[i];
  if(c > 0.0f){
    float s = noiseDraw(NOISE_SHARED + (unsigned int)nId[i], t, seed);
    if(a > 0.0f){ float zs = a*nzs[i] + nK[i]*s; nzs[i] = zs; s = zs; }
    d = sqrtf(1.0f - c)*d + sqrtf(c)*s;
  }
  return d;
}

// One exponential half step of 0.5 ms (MODEL.md 1), as the reference engine's expHalf: 0.5 F(v)/C times (e^x - 1)/x at x = 0.5 F'(v)/C, with F(v) = k (v - vr)(v - vt) - adaptation + I + G (v0 - v), exact when F is linear in v over the step.
__device__ __forceinline__ float expSpanD(float vi, float ui, float I, float G, float v0, float ki, float vri, float vti, float Ci, bool ac, float eI, float vmin, float h){
  const float F = ki*(vi - vri)*(vi - vti) - (ac ? ui*(vi - eI) : ui) + I + G*(v0 - vi);
  const float x = h*(ki*(2.0f*vi - vri - vti) - (ac ? ui : 0.0f) - G)/Ci;
  const float phi = fabsf(x) < 1e-4f ? 1.0f + 0.5f*x : expm1f(x)/x;
  vi += h*F/Ci*phi;
  return vi < vmin ? vmin : vi;
}
__device__ __forceinline__ float expHalfD(float vi, float ui, float I, float G, float v0, float ki, float vri, float vti, float Ci, bool ac, float eI, float vmin){
  return expSpanD(vi, ui, I, G, v0, ki, vri, vti, Ci, ac, eI, vmin, 0.5f);
}
// The exponential step's spike, as the reference engine's spanReset: the distance to the threshold interpolated inside the millisecond (the threshold at both ends of the step), the reset there, the rest of the step integrated from it with the jump in u applied and the threshold relaxing.
__device__ __forceinline__ float spanResetD(int i, float vs, float vi, float vth, float ui, float I, float G, float v0, float c, float ki, float vri, float vti, float Ci, bool ac, float eI, float vmin,
    const float* thrJ, const float* thrDec, float* thr){
  const float th = thr ? thr[i] : 0.0f, thS = th > 0.0f && thrDec[i] > 0.0f ? th/thrDec[i] : th;
  const float dS = vs - (vth - th + thS), dE = vi - vth, fr = dS < 0.0f && dE > dS ? fminf(1.0f, fmaxf(0.0f, dS/(dS - dE))) : 1.0f, rest = 1.0f - fr;
  if(thr) thr[i] = thrJ[i]*powf(thrDec[i], rest);
  return expSpanD(c, ui, I, G, v0, ki, vri, vti, Ci, ac, eI, vmin, rest);
}

// The equations' membrane step (integ 2, MODEL.md 1), as the reference engine's quadTime, quadAt and quadMs.
__device__ __forceinline__ float quadTimeD(float A, float q, float w0, float wp){
  if(w0 >= wp) return 0.0f;
  const float y = wp - w0, x = q + A*w0*wp, z = A*q;
  if(x > 0.0f){
    const float R = y/x, e = z*R*R;
    if(fabsf(e) < 1e-4f) return R*(1.0f - e/3.0f + e*e/5.0f);
    if(z > 0.0f){ const float s = sqrtf(z); return atanf(s*R)/s; }
    const float s = sqrtf(-z), r = s*R;
    return r < 1.0f ? 0.5f*logf((1.0f + r)/(1.0f - r))/s : 1e30f;
  }
  if(z > 0.0f){ const float s = sqrtf(z); return (1.5707964f + atanf(-x/(s*y)))/s; }
  return 1e30f;
}
// w after h, with the integral of w over the span in iw
__device__ __forceinline__ float quadAtD(float A, float q, float w0, float h, float& iw){
  const float z = A*q, e = z*h*h;
  float Cs, Sn, ls = 0.0f;
  if(fabsf(e) < 1e-4f){ Cs = 1.0f - e/2.0f + e*e/24.0f; Sn = h*(1.0f - e/6.0f + e*e/120.0f); }
  else if(z > 0.0f){ const float s = sqrtf(z); Cs = cosf(s*h); Sn = sinf(s*h)/s; }
  else { const float s = sqrtf(-z), E = expf(-2.0f*s*h); Cs = 0.5f*(1.0f + E); Sn = 0.5f*(1.0f - E)/s; ls = s*h; }
  const float phi = Cs - A*w0*Sn;
  iw = -(ls + logf(fabsf(phi)))/A;
  return (w0*Cs + q*Sn)/phi;
}
// One millisecond on a quadratic row; u is updated in place, sp says whether the cell fired, and the threshold state is written at the reset.
__device__ float quadMsD(int i, float v, float& u, float I, float G, float v0, float k, float vr, float vt, float C, float ra, float rb, float rc, float rd, bool ac, float eA, float vmin, float vpk, float th,
    const float* thrJ, const float* thrDec, float* thr, bool& sp){
  float vth = vpk + th, iw;
  const float A = k/C, hd = 0.5f*(vt - vr), sm = vr + vt, ea = expf(-0.5f*ra);
  sp = false;
  for(int hs = 0; hs < 2; hs++){
    float gs = (ac ? u : 0.0f) + G, m = 0.5f*sm + gs/(2.0f*k);
    float q = (-k*hd*hd + I + (ac ? u*eA : -u) + G*v0 - 0.5f*gs*sm - gs*gs/(4.0f*k))/C;
    float iv;
    const float tc = quadTimeD(A, q, v - m, vth - m);
    if(tc > 0.5f){ v = m + quadAtD(A, q, v - m, 0.5f, iw); iv = iw + 0.5f*m; }
    else {
      quadAtD(A, q, v - m, tc, iw); iv = iw + tc*m;
      const float left = 0.5f - tc;
      if(sp){ v = vth; iv += left*vth; }
      else {
        sp = true; v = rc; u += rd;
        if(thr){ thr[i] = thrJ[i]*powf(thrDec[i], 0.5f*(float)(1 - hs) + left); vth = vpk + thrJ[i]; }
        gs = (ac ? u : 0.0f) + G; m = 0.5f*sm + gs/(2.0f*k);
        q = (-k*hd*hd + I + (ac ? u*eA : -u) + G*v0 - 0.5f*gs*sm - gs*gs/(4.0f*k))/C;
        const float t2 = quadTimeD(A, q, v - m, vth - m);
        if(t2 <= left){ quadAtD(A, q, v - m, t2, iw); iv += iw + t2*m + (left - t2)*vth; v = vth; }
        else { v = m + quadAtD(A, q, v - m, left, iw); iv += iw + left*m; }
      }
    }
    if(v < vmin) v = vmin;
    const float tg = rb*(2.0f*iv - vr);
    u = tg + (u - tg)*ea;
  }
  return v;
}

__global__ void stepKick(int n, int cur,
    const float* a, const float* b, const float* c, const float* d,
    const float* bias, const float* ext, const float* namp,
    const float* nA, const float* nK, const float* nC, const int* nId, float* nz, float* nzs,
    unsigned int t, unsigned int seed, float* v, float* u, float* ring,
    const int* preStart, const int* post, const float* w,
    const unsigned char* delay, unsigned char* fired,
    unsigned int* counts, unsigned long long* total,
    unsigned char* refCnt, unsigned int refrac,
    float* stpX, float* stpR, float stpU, float stpScale, int stpOrder,
    unsigned char* firstMs, int stepIdx,
    const unsigned short* pool, const float* poolK, unsigned int* poolCnt,
    unsigned int poolG, unsigned int par, float vmin, int integ,
    const float* C7, const float* k7, const float* vr7, const float* vt7, const float* vp7,
    const unsigned char* grd, const float* gthr, const float* gslp,
    const float* thrJ, const float* thrDec, float* thr, const float* adC, const float* poi, float eI,
    const unsigned char* gMask, unsigned int* gCnt,
    unsigned int* spk, unsigned int* nSpk,
    unsigned int* dlv, float* dlvAmp, unsigned int* nDlv, float* dlvAmpC,
    const float* modPrev, float* modNew, const float* modStim, const unsigned short* modTgt, const float* modKf,
    const unsigned int* modCntR, unsigned int* modCntW, const unsigned char* modGrp, float decMod, float modDrive, float modCur, int modOn){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  // The modulator field (MODEL.md 4), as modUpdate() in simworker.js: the level as the step before left it (modPrev) is what this step reads, for the synaptic gain and the current here and for the learning passes; the new level, decayed plus the sources' increment plus the group's release per spike times the group's spikes of the step before, is written for the next step.
  float mpv = 0.0f, mf = 1.0f, mc = 0.0f;
  if(modOn){
    mpv = modPrev[i]; mf = 1.0f + modDrive*mpv; mc = modCur*mpv;
    float mn = mpv*decMod + (modStim ? modStim[i] : 0.0f);
    if(modTgt){ unsigned short g = modTgt[i]; if(g) mn += modKf[g]*(float)modCntR[g]; }
    modNew[i] = mn;
  }
  // Izhikevich 2007 (MODEL.md 1): u is measured from the rest, the membrane is k (v - vr)(v - vt) over C, the spike at the row's vpeak
  const float uRef = vr7[i];
  // the dynamic threshold (MODEL.md 1) relaxes before anything else this step, Auryn's order
  float th = 0.0f; if(thr){ th = thr[i]*thrDec[i]; thr[i] = th; }
  if(refrac > 0 && refCnt[i] > 0){
    refCnt[i]--;
    float ui2 = u[i] + a[i]*(b[i]*(v[i] - uRef) - u[i]);
    u[i] = ui2; fired[i] = 0;
    return;
  }
  float I;
  if(modOn){
    // the field scales the synaptic current (the ring and the pool) by 1 + modDrive M and adds modCur M
    float Is = ring[cur*n + i];
    if(pool){ unsigned short g = pool[i];
      if(g) Is += poolK[g]*(float)poolCnt[(par ^ 1u)*poolG + g]; }
    I = Is*mf + mc + bias[i] + (ext ? ext[i] : 0.0f);
  } else {
    I = ring[cur*n + i] + bias[i] + (ext ? ext[i] : 0.0f);
    // feedback inhibition: the pool's kick per spike times the spikes it fired last millisecond, from the half of the table the last step wrote
    if(pool){ unsigned short g = pool[i];
      if(g) I += poolK[g]*(float)poolCnt[(par ^ 1u)*poolG + g]; }
  }
  if(namp && namp[i] != 0.0f) I += namp[i]*noiseAt(i, t, seed, nA, nK, nC, nId, nz, nzs);
  float vi = v[i], ui = u[i];
  bool qc = false, qsp = false;
    const float Ci = C7[i], ki = k7[i], vri = vr7[i], vti = vt7[i];
    if(grd && grd[i]){
      // a graded cell's membrane is passive: the leak is the row's slope at rest, k (vt - vr), and the potential is capped at vpeak
      const float gl = ki*(vti - vri);
      vi += 0.5f*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
      vi += 0.5f*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
      if(vi > vp7[i]) vi = vp7[i];
    } else if(poi && poi[i] > 0.5f){
      // a Poisson cell (MODEL.md 1): no membrane; it fires this step with probability I over 1000, the current read as a rate in hertz, by being put at its peak
      if(I > 0.0f && poissonDraw((unsigned int)i, t, seed) < I*0.001f) vi = vp7[i] + th;
    } else if(integ && vti - vp7[i] > 100.0f){
    // the exponential membrane step (integ 1), on a row linear over the spiking range as the reference's linRow: current synapses add no conductance to the slope
    const bool ac = adC && adC[i] > 0.5f; const float vs = vi;
    vi = expHalfD(vi, ui, I, 0.0f, vs, ki, vri, vti, Ci, ac, eI, vmin); vi = expHalfD(vi, ui, I, 0.0f, vs, ki, vri, vti, Ci, ac, eI, vmin);
    } else if(integ == 2){
    // the equations' step (integ 2) on a quadratic row, as the reference's quadMs
    vi = quadMsD(i, vi, ui, I, 0.0f, vi, ki, vri, vti, Ci, a[i], b[i], c[i], d[i], adC && adC[i] > 0.5f, eI, vmin, vp7[i], th, thrJ, thrDec, thr, qsp); qc = true;
    } else {
    // with the row's adaptation a conductance (adC), u in nanosiemens draws current toward the inhibitory reversal
    const bool ac = adC && adC[i] > 0.5f;
    vi += 0.5f*(ki*(vi - vri)*(vi - vti) - (ac ? ui*(vi - eI) : ui) + I)/Ci;
    if(vi < vmin) vi = vmin;
    vi += 0.5f*(ki*(vi - vri)*(vi - vti) - (ac ? ui*(vi - eI) : ui) + I)/Ci;
    if(vi < vmin) vi = vmin;
    }
    if(!qc) ui += a[i]*(b[i]*(vi - vri) - ui);
    const float vpk = vp7[i];
  // a graded cell (MODEL.md 1): no spike, no reset, no refractory period; each millisecond its synapses carry a spike of amplitude r, delivered by the delivery pass from the list
  if(grd && grd[i]){
    float r = fminf(1.0f, fmaxf(0.0f, (vi - gthr[i])/gslp[i]));
    if(r > 0.0f){ unsigned int k = atomicAdd(nDlv, 1u); dlv[k] = (unsigned int)i; dlvAmp[k] = -r; }   // negative: a graded release, outside short-term plasticity's scope
    v[i] = vi; u[i] = ui; fired[i] = 0;
    return;
  }
  unsigned char f = 0;
  if(qc ? qsp : vi >= vpk + th){
    if(qc) f = 1;
    else if(integ && !(poi && poi[i] > 0.5f) && vti - vpk > 100.0f){ ui += d[i]; vi = spanResetD(i, v[i], vi, vpk + th, ui, I, 0.0f, vi, c[i], ki, vri, vti, Ci, adC && adC[i] > 0.5f, eI, vmin, thrJ, thrDec, thr); f = 1; }
    else { vi = c[i]; ui += d[i]; f = 1;
    if(thr) thr[i] = thrJ[i]; }
    // the global factor's population count for this step (MODEL.md 4)
    if(gMask && gMask[i]) atomicAdd(gCnt, 1u);
    spk[atomicAdd(nSpk, 1u)] = (unsigned int)i;   // the list of this step's spikes, which the learning passes walk
    if(refrac > 0) refCnt[i] = (unsigned char)refrac;
    if(pool && pool[i]) atomicAdd(&poolCnt[par*poolG + pool[i]], 1u);
    // a releasing cell's spike, counted for its modulator group, read by the group's targets next step (MODEL.md 4)
    if(modGrp && modGrp[i]) atomicAdd(&modCntW[modGrp[i]], 1u);
    counts[i]++;
    // the step in the tick this cell first fired, 255 if it did not: the readout bins at the tick, so without this the finest grain any measurement has is the whole tick, coarser at 50 steps than the STDP window.
    if(firstMs && firstMs[i] == 255)
      firstMs[i] = (unsigned char)(stepIdx < 254 ? stepIdx : 254);
    atomicAdd(total, 1ull);
    // Tsodyks-Markram release: as published (stpOrder 0) the probability times the resources is released, then the probability jumps; stpOrder 1 jumps first. stpScale is 1, or the inverse of the rested release so a rested synapse delivers its full weight.
    // Matches stpRelease() in src/simworker.js
    float sf = 1.0f;
    float rc[STP_CMAX];
    if(stpX){
      if(cStpC <= 1){
        if(stpOrder) stpR[i] = stpR[i] + stpU*(1.0f - stpR[i]);
        float rel = stpR[i]*stpX[i];
        stpX[i] = stpX[i] - rel;
        if(!stpOrder) stpR[i] = stpR[i] + stpU*(1.0f - stpR[i]);
        sf = rel*stpScale;
      } else { stpReleaseC(i, n, stpX, stpR, stpOrder, rc); sf = rc[0]; }
    }
    unsigned int k = atomicAdd(nDlv, 1u); dlv[k] = (unsigned int)i; dlvAmp[k] = sf;   // delivered by deliverKick from the list
    if(stpX && cStpC > 1 && dlvAmpC) for(int c = 0; c < cStpC; c++) dlvAmpC[(size_t)k*STP_CMAX + c] = rc[c];
  }
  v[i] = vi; u[i] = ui; fired[i] = f;
}

// Delivery, a warp per entry of the step's list (a spiking cell with its release, or a graded cell with its amplitude): the 32 lanes stride over that cell's synapses, each adding its weight times the amplitude into the ring row its delay lands on.
// Every delay is at least one step, so no entry writes the row the step just read, and the pass runs after the membrane step and before the learning passes, as the per-neuron form delivered inside the step.
// Sums into the ring are atomic and land in whatever order the warps run, as before.
// The scope (stpScope) confines a spike's release factor to one sign of weight (1 excitatory, 2 inhibitory); a graded release (a negative amplitude in the list) is not short-term plasticity and applies to every synapse.
__device__ __forceinline__ void ampBySign(float a, int stpScope, float& aE, float& aI){
  if(a < 0.0f){ aE = aI = -a; return; }
  aE = stpScope == 2 ? 1.0f : a;
  aI = stpScope == 1 ? 1.0f : a;
}
// With more than one short-term class (dlvAmpC given) a spike's synapse takes its own rule's class; a graded entry (negative) is one amplitude for every synapse.
__global__ void deliverKick(const unsigned int* dlv, const float* dlvAmp, const unsigned int* nDlv,
    int n, int cur, float* ring,
    const int* preStart, const int* post, const float* w, const unsigned char* delay, int stpScope,
    const float* dlvAmpC, const unsigned char* pmS){
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warps = (gridDim.x*blockDim.x) >> 5;
  const unsigned int count = *nDlv;
  for(unsigned int k = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; k < count; k += warps){
    const int i = (int)dlv[k];
    float aE, aI; ampBySign(dlvAmp[k], stpScope, aE, aI);
    const float* ac = dlvAmp[k] >= 0.0f ? dlvAmpC : nullptr;
    for(int s = preStart[i] + (int)lane, e = preStart[i+1]; s < e; s += 32){
      if(ac) ampBySign(stpAmpOf(ac, k, pmS, s, 0.0f), stpScope, aE, aI);
      atomicAdd(&ring[((cur + delay[s]) % ROWS)*n + post[s]], w[s]*(w[s] > 0.0f ? aE : aI));
    }
  }
}

__global__ void stepExp(int n, int cur, float decE, float decI,
    float itauE, float itauI,
    const float* a, const float* b, const float* c, const float* d,
    const float* bias, const float* ext, const float* namp,
    const float* nA, const float* nK, const float* nC, const int* nId, float* nz, float* nzs,
    unsigned int t, unsigned int seed, float* v, float* u, float* ring, float* ring2,
    float* gE, float* gI,
    const int* preStart, const int* post, const float* w,
    const unsigned char* delay, unsigned char* fired,
    unsigned int* counts, unsigned long long* total,
    unsigned char* refCnt, unsigned int refrac,
    float* stpX, float* stpR, float stpU, float stpScale, int stpOrder,
    unsigned char* firstMs, int stepIdx,
    const unsigned short* pool, const float* poolK, unsigned int* poolCnt,
    unsigned int poolG, unsigned int par, float vmin, int integ,
    const float* C7, const float* k7, const float* vr7, const float* vt7, const float* vp7,
    const unsigned char* grd, const float* gthr, const float* gslp,
    int KX, float* gX, float* ringX, const float* decX, const float* itauX, const unsigned char* pmask,
    int cond, float eE, float eI, const float* erevX, const float* mgX,
    const float* thrJ, const float* thrDec, float* thr, const float* adC, const float* poi,
    const unsigned char* gMask, unsigned int* gCnt,
    unsigned int* spk, unsigned int* nSpk,
    unsigned int* dlv, float* dlvAmp, unsigned int* nDlv, float* dlvAmpC,
    const float* modPrev, float* modNew, const float* modStim, const unsigned short* modTgt, const float* modKf,
    const unsigned int* modCntR, unsigned int* modCntW, const unsigned char* modGrp, float decMod, float modDrive, float modCur, int modOn){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  // The modulator field (MODEL.md 4), as modUpdate() in simworker.js: the level as the step before left it (modPrev) is what this step reads, for the synaptic gain and the current here and for the learning passes; the new level, decayed plus the sources' increment plus the group's release per spike times the group's spikes of the step before, is written for the next step.
  float mpv = 0.0f, mf = 1.0f, mc = 0.0f;
  if(modOn){
    mpv = modPrev[i]; mf = 1.0f + modDrive*mpv; mc = modCur*mpv;
    float mn = mpv*decMod + (modStim ? modStim[i] : 0.0f);
    if(modTgt){ unsigned short g = modTgt[i]; if(g) mn += modKf[g]*(float)modCntR[g]; }
    modNew[i] = mn;
  }
  // the dynamic threshold (MODEL.md 1) relaxes before anything else this step, Auryn's order
  float th = 0.0f; if(thr){ th = thr[i]*thrDec[i]; thr[i] = th; }
  const float uRef = vr7[i];               // u is measured from the rest (2007)
  const float v0 = v[i];
  float ge = gE[i]*decE + ring[cur*n + i];
  float gi = gI[i]*decI + ring2[cur*n + i];
  // receptor channels 2 and up (MODEL.md 2): their own decay, their own ring;
  // conductance mode drives |g| (E - v) with the step's opening v, and a channel with a magnesium concentration passes the fraction B(v) = 1/(1 + [Mg]/3.57 exp(-0.062 v)) (Jahr and Stevens 1990)
  float Ix = 0.0f, Gx = 0.0f;
  for(int x = 0; x < KX; x++){
    float g = gX[x*n + i]*decX[x] + ringX[((size_t)x*ROWS + cur)*n + i];
    gX[x*n + i] = g;
    float bI = mgX[x] > 0.0f ? 1.0f/(1.0f + (mgX[x]/3.57f)*expf(-0.062f*v0)) : 1.0f;
    Ix += bI*(cond ? fabsf(g)*(erevX[x] - v0) : g*itauX[x]);
    if(cond) Gx += bI*fabsf(g);
  }
  // feedback inhibition, on the inhibitory conductance like any I delivery
  if(pool){ unsigned short g = pool[i];
    if(g) gi += poolK[g]*(float)poolCnt[(par ^ 1u)*poolG + g]; }
  if(refrac > 0 && refCnt[i] > 0){
    // absolute refractoriness: conductances keep integrating, the membrane is held at reset, no threshold this millisecond
    refCnt[i]--;
    float ui2 = u[i] + a[i]*(b[i]*(v[i] - uRef) - u[i]);
    u[i] = ui2; gE[i] = ge; gI[i] = gi; fired[i] = 0;
    return;
  }
  // the field scales the synaptic current, the channels included, by 1 + modDrive M and adds modCur M; the synaptic conductance the exponential step reads scales with it
  float I;
  if(modOn) I = ((cond ? fabsf(ge)*(eE - v0) + fabsf(gi)*(eI - v0) : ge*itauE + gi*itauI) + Ix)*mf + mc + bias[i] + (ext ? ext[i] : 0.0f);
  else I = (cond ? fabsf(ge)*(eE - v0) + fabsf(gi)*(eI - v0) : ge*itauE + gi*itauI) + Ix + bias[i] + (ext ? ext[i] : 0.0f);
  const float Gs = (cond ? fabsf(ge) + fabsf(gi) + Gx : 0.0f)*mf;
  if(namp && namp[i] != 0.0f) I += namp[i]*noiseAt(i, t, seed, nA, nK, nC, nId, nz, nzs);
  float vi = v[i], ui = u[i];
  bool qc = false, qsp = false;
    const float Ci = C7[i], ki = k7[i], vri = vr7[i], vti = vt7[i];
    if(grd && grd[i]){
      // a graded cell's membrane is passive: the leak is the row's slope at rest, k (vt - vr), and the potential is capped at vpeak
      const float gl = ki*(vti - vri);
      vi += 0.5f*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
      vi += 0.5f*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
      if(vi > vp7[i]) vi = vp7[i];
    } else if(poi && poi[i] > 0.5f){
      // a Poisson cell (MODEL.md 1): no membrane; it fires this step with probability I over 1000, the current read as a rate in hertz, by being put at its peak
      if(I > 0.0f && poissonDraw((unsigned int)i, t, seed) < I*0.001f) vi = vp7[i] + th;
    } else if(integ && vti - vp7[i] > 100.0f){
    // the exponential membrane step (integ 1), on a row linear over the spiking range: the synaptic conductance enters the slope and the current is taken at the voltage it acts on
    const bool ac = adC && adC[i] > 0.5f; const float G = Gs;
    vi = expHalfD(vi, ui, I, G, v0, ki, vri, vti, Ci, ac, eI, vmin); vi = expHalfD(vi, ui, I, G, v0, ki, vri, vti, Ci, ac, eI, vmin);
    } else if(integ == 2){
    vi = quadMsD(i, vi, ui, I, Gs, v0, ki, vri, vti, Ci, a[i], b[i], c[i], d[i], adC && adC[i] > 0.5f, eI, vmin, vp7[i], th, thrJ, thrDec, thr, qsp); qc = true;
    } else {
    // with the row's adaptation a conductance (adC), u in nanosiemens draws current toward the inhibitory reversal
    const bool ac = adC && adC[i] > 0.5f;
    vi += 0.5f*(ki*(vi - vri)*(vi - vti) - (ac ? ui*(vi - eI) : ui) + I)/Ci;
    if(vi < vmin) vi = vmin;
    vi += 0.5f*(ki*(vi - vri)*(vi - vti) - (ac ? ui*(vi - eI) : ui) + I)/Ci;
    if(vi < vmin) vi = vmin;
    }
    if(!qc) ui += a[i]*(b[i]*(vi - vri) - ui);
    const float vpk = vp7[i];
  if(grd && grd[i]){
    float r = fminf(1.0f, fmaxf(0.0f, (vi - gthr[i])/gslp[i]));
    if(r > 0.0f){ unsigned int k = atomicAdd(nDlv, 1u); dlv[k] = (unsigned int)i; dlvAmp[k] = -r; }   // delivered by deliverExp from the list; negative marks a graded release, outside short-term plasticity's scope
    v[i] = vi; u[i] = ui; gE[i] = ge; gI[i] = gi; fired[i] = 0;
    return;
  }
  unsigned char f = 0;
  if(qc ? qsp : vi >= vpk + th){
    if(qc) f = 1;
    else if(integ && !(poi && poi[i] > 0.5f) && vti - vpk > 100.0f){ ui += d[i]; vi = spanResetD(i, v0, vi, vpk + th, ui, I, Gs, v0, c[i], ki, vri, vti, Ci, adC && adC[i] > 0.5f, eI, vmin, thrJ, thrDec, thr); f = 1; }
    else { vi = c[i]; ui += d[i]; f = 1;
    if(thr) thr[i] = thrJ[i]; }
    // the global factor's population count for this step (MODEL.md 4)
    if(gMask && gMask[i]) atomicAdd(gCnt, 1u);
    spk[atomicAdd(nSpk, 1u)] = (unsigned int)i;   // the list of this step's spikes, which the learning passes walk
    if(refrac > 0) refCnt[i] = (unsigned char)refrac;
    if(pool && pool[i]) atomicAdd(&poolCnt[par*poolG + pool[i]], 1u);
    if(modGrp && modGrp[i]) atomicAdd(&modCntW[modGrp[i]], 1u);
    counts[i]++;
    // the step in the tick this cell first fired, 255 if it did not (see stepKick)
    if(firstMs && firstMs[i] == 255)
      firstMs[i] = (unsigned char)(stepIdx < 254 ? stepIdx : 254);
    atomicAdd(total, 1ull);
    float sf = 1.0f;
    float rc[STP_CMAX];
    if(stpX){
      if(cStpC <= 1){
        if(stpOrder) stpR[i] = stpR[i] + stpU*(1.0f - stpR[i]);   // stpOrder 1: facilitation before release
        float rel = stpR[i]*stpX[i];
        stpX[i] = stpX[i] - rel;
        if(!stpOrder) stpR[i] = stpR[i] + stpU*(1.0f - stpR[i]);  // as published: facilitation after release
        sf = rel*stpScale;
      } else { stpReleaseC(i, n, stpX, stpR, stpOrder, rc); sf = rc[0]; }
    }
    unsigned int k = atomicAdd(nDlv, 1u); dlv[k] = (unsigned int)i; dlvAmp[k] = sf;   // delivered by deliverExp from the list
    if(stpX && cStpC > 1 && dlvAmpC) for(int c = 0; c < cStpC; c++) dlvAmpC[(size_t)k*STP_CMAX + c] = rc[c];
  }
  v[i] = vi; u[i] = ui; gE[i] = ge; gI[i] = gi; fired[i] = f;
}

// The exponential and conductance form of the delivery pass (see deliverKick): excitatory and inhibitory weights into their own rings, a receptor channel into its own.
__global__ void deliverExp(const unsigned int* dlv, const float* dlvAmp, const unsigned int* nDlv,
    int n, int cur, float* ring, float* ring2, float* ringX,
    const int* preStart, const int* post, const float* w, const unsigned char* delay, const unsigned char* pmask, int stpScope,
    const float* dlvAmpC, const unsigned char* pmS){
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warps = (gridDim.x*blockDim.x) >> 5;
  const unsigned int count = *nDlv;
  for(unsigned int k = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; k < count; k += warps){
    const int i = (int)dlv[k];
    float aE, aI; ampBySign(dlvAmp[k], stpScope, aE, aI);
    const float* ac = dlvAmp[k] >= 0.0f ? dlvAmpC : nullptr;
    for(int s = preStart[i] + (int)lane, e = preStart[i+1]; s < e; s += 32){
      if(ac) ampBySign(stpAmpOf(ac, k, pmS, s, 0.0f), stpScope, aE, aI);
      float wt = w[s]*(w[s] > 0.0f ? aE : aI);
      int slot = (cur + delay[s]) % ROWS, at = slot*n + post[s];
      int ch = pmask ? (pmask[s] >> 5) : 0;
      if(ch >= 2) atomicAdd(&ringX[((size_t)(ch - 2)*ROWS + slot)*n + post[s]], wt);
      else atomicAdd(wt > 0 ? &ring[at] : &ring2[at], wt);
    }
  }
}

__device__ inline float softBound(float wv, float delta, float wmax){
  float aa = wv < 0 ? -wv : wv;
  float room = ((delta < 0) == (wv < 0)) ? (wmax - aa)/wmax : aa/wmax;
  return delta * (room > 0 ? room : 0);
}


// The per-rule table, the same eleven fields in the same order the reference engine and the WebGPU shaders use, because the byte on each synapse means the same thing on all three.
// Held in constant memory: every thread in a warp reads the same row whenever a pathway is contiguous in the CSR, which is the broadcast case constant memory is for.
#define RULE_STRIDE 11
#define RULE_MAX 16
__constant__ float cRules[RULE_MAX*RULE_STRIDE];
// the pair rule timed at the synapse (stdpAt 1): what a spike k steps ago contributes to Kpre, e^(-k/tau+), and the factor that brings the trace d steps back, e^(d/tau+)
__constant__ float cEk[17];
__constant__ float cExd[17];
#define R_aP 0
#define R_aM 1
#define R_wmax 2
#define R_wdep 3
#define R_trip 4
#define R_het 5
#define R_tin 6
#define R_iEta 7
#define R_cons 8
#define R_consW 9
#define R_consP 10

// alphaI is the per-neuron Vogels target term from measured set points; when it is null every neuron uses the scalar computed from the fixed iRho, which is what the reference does with rhoMode off.
// A committed synapse is out of the plastic pool: its consolidated reference has reached the upper well, so it keeps its weight and keeps transmitting and neither half of the update touches it.
// The reference is read here as well as in plastIn, since freezing one half only would leave a synapse closed to potentiation and open to depression.
__device__ __forceinline__ bool isCommitted(int commit, const float* wRef,
    const float* R, int s){
  return commit && wRef && R[R_consW] > 0.0f && wRef[s] >= R[R_consW]*0.9f;
}
// The two weight-update halves run over the list of the neurons that fired this step (spk, nSpk, written by the step kernel), a warp per spike: the 32 lanes stride over that neuron's synapses.
// One thread per neuron walking every synapse of a spiking cell alone left the rest of its warp idle, which at a few hertz was most of the card for the whole pass.
// Each synapse is still updated by exactly one lane with the per-synapse arithmetic below, so the weights come out the same bits as the per-neuron form did.
#define WARP 32u
// gain is the third factor on the Hebbian terms (MODEL.md 4), times modBase + the modulator level of the postsynaptic cell under modPlast 1; under modPlast 2 the Hebbian terms accumulate on the eligibility trace instead of moving the weight.
// The inhibitory rule runs ungated.
__device__ __forceinline__ void plastOutSyn(int s, const int* post, float* w,
    const unsigned char* pmask, const float* Kpost,
    float iAlpha, const float* alphaI, const float* wRef, int commit, float gain, float mu, const float* Khom, float slideK,
    const float* gState, int iGlobal, const float* modPrev, int modPlast, float modBase, float* elig){
  unsigned int rid = pmask ? (pmask[s] & 31u) : 1u;   // the rule is the low five bits; the channel the top three
  if(rid == 0u) return;                    // 0 is frozen
  const float* R = cRules + rid*RULE_STRIDE;
  if(isCommitted(commit, wRef, R, s)) return;
  const float wmax = R[R_wmax];
  const int wdep = R[R_wdep] != 0.0f;
  float wj = w[s];
  if(wj > 0){
    // the sliding threshold: depression scaled by the square of the cell's slow rate estimate over the target, 1 when off (MODEL.md 4)
    float sl = 1.0f;
    if(slideK > 0.0f && Khom){ float kh = Khom[post[s]]*slideK; sl = kh*kh; }
    const float g = modPlast == 1 ? gain*(modBase + modPrev[post[s]]) : gain;
    const float dep = g*R[R_aM]*sl*Kpost[post[s]]*(wdep ? (mu == 1.0f ? wj : powf(wj, mu)) : 1.0f);
    if(modPlast == 2){ elig[s] += -dep + g*R[R_tin]; return; }
    float nw = wj - dep + g*R[R_tin];
    if(nw > wmax) nw = wmax;
    w[s] = nw > 1e-4f ? nw : 1e-4f;
  } else {
    float al = alphaI ? alphaI[post[s]] : iAlpha;
    // under the global factor the amplitude is the rule's eta times the population ratio and the presynaptic term is the trace plus one, the authors' xi (MODEL.md 4)
    float dd = iGlobal ? -R[R_iEta]*gState[1]*(Kpost[post[s]] + 1.0f) : -R[R_iEta]*(Kpost[post[s]] - al);
    if(wdep) dd = softBound(wj, dd, wmax);
    float nw = wj + dd;
    if(nw > -1e-4f) nw = -1e-4f;
    if(nw < -wmax) nw = -wmax;
    w[s] = nw;
  }
}
__global__ void plastOut(const unsigned int* spk, const unsigned int* nSpk,
    const int* preStart, const int* post, float* w,
    const unsigned char* pmask, const float* Kpost,
    float iAlpha, const float* alphaI, const float* wRef, int commit, float gain, float mu, const float* Khom, float slideK,
    const float* gState, int iGlobal, const float* modPrev, int modPlast, float modBase, float* elig){
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warps = (gridDim.x*blockDim.x) >> 5;
  const unsigned int count = *nSpk;
  for(unsigned int k = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; k < count; k += warps){
    const int i = (int)spk[k];
    for(int s = preStart[i] + (int)lane, e = preStart[i+1]; s < e; s += (int)WARP)
      plastOutSyn(s, post, w, pmask, Kpost, iAlpha, alphaI, wRef, commit, gain, mu, Khom, slideK, gState, iGlobal, modPrev, modPlast, modBase, elig);
  }
}

// The pair rule timed at the synapse (stdpAt 1): the outgoing half runs when a spike arrives, d steps after it was fired, against Kpost as it stands before the step's spikes.
// The trace step lists each cell whose history meets the delays its synapses carry (arr, nArr); a warp per listed cell walks its slice and updates every synapse whose delay bit is set in the cell's history.
__global__ void plastArrive(const unsigned int* arr, const unsigned int* nArr, const unsigned int* hist,
    const int* preStart, const int* post, float* w, const unsigned char* delay,
    const unsigned char* pmask, const float* Kpost,
    float iAlpha, const float* alphaI, const float* wRef, int commit, float gain, float mu, const float* Khom, float slideK,
    const float* gState, int iGlobal, const float* modPrev, int modPlast, float modBase, float* elig){
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warps = (gridDim.x*blockDim.x) >> 5;
  const unsigned int count = *nArr;
  for(unsigned int k = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; k < count; k += warps){
    const int i = (int)arr[k];
    const unsigned int h = hist[i];
    for(int s = preStart[i] + (int)lane, e = preStart[i+1]; s < e; s += (int)WARP)
      if((h >> delay[s]) & 1u) plastOutSyn(s, post, w, pmask, Kpost, iAlpha, alphaI, wRef, commit, gain, mu, Khom, slideK, gState, iGlobal, modPrev, modPlast, modBase, elig);
  }
}

// The triplet term reads the slow postsynaptic trace and the heterosynaptic term the fast one, both before this millisecond's spike is added, matching the reference: the triplet potentiates by postsynaptic history (Pfister and Gerstner 2006), and beta times the cubed fast trace pulls every incoming weight toward its reference (Zenke, Agnes and Gerstner 2015).
// The pre trace a post spike sees through a synapse of delay d when the rule is timed at the synapse: the spikes of the last d - 1 steps (the cell's history, bit k for k steps ago) taken out of Kpre and the rest brought forward by e^(d/tau+), as simworker.js kpreAtSynapse.
__device__ __forceinline__ float kpreAtSynapse(float kpre, unsigned int h, unsigned int d){
  float sub = 0.0f;
  for(unsigned int k = 1u; k < d; k++) if((h >> k) & 1u) sub += cEk[k];
  float kp = cExd[d]*(kpre - sub);
  return kp > 0.0f ? kp : 0.0f;
}
// gain arrives folded by the caller: the third factor, times modBase + the postsynaptic cell's modulator level under modPlast 1; under modPlast 2 the Hebbian potentiation goes to the eligibility trace and the heterosynaptic pull, homeostatic, moves the weight now.
__device__ __forceinline__ void plastInSyn(int s, unsigned int preD, unsigned int mask, float* w,
    const float* Kpre, const unsigned int* hist,
    const float* wRef, float z, float gate, int commit, float gain, const float* gState, int iGlobal, int modPlast, float* elig){
  unsigned int rid = mask & 31u;           // the rule is the low five bits; the channel the top three
  if(rid == 0u) return;
  const float* R = cRules + rid*RULE_STRIDE;
  if(isCommitted(commit, wRef, R, s)) return;
  const float wmax = R[R_wmax], het = R[R_het];
  const int wdep = R[R_wdep] != 0.0f;
  // the presynaptic cell and the synapse's delay, packed in the reverse list
  const unsigned int pre = preD & 0x7ffffffu;
  float wp = w[s], kp = Kpre[pre];
  if(hist) kp = kpreAtSynapse(kp, hist[pre], preD >> 27);
  if(wp > 0){
    float nw = wp;
    if(modPlast == 2) elig[s] += gain*(R[R_aP] + R[R_trip]*z)*kp;
    else nw = wp + gain*(R[R_aP] + R[R_trip]*z)*kp;
    if(het > 0 && wRef) nw -= het*gate*(wp - wRef[s]);
    if(nw > wmax) nw = wmax;
    w[s] = nw > 1e-4f ? nw : 1e-4f;
  } else {
    float dd = iGlobal ? -R[R_iEta]*gState[1]*kp : -R[R_iEta]*kp;
    if(wdep) dd = softBound(wp, dd, wmax);
    float nw = wp + dd;
    w[s] = nw > -wmax ? nw : -wmax;
  }
}
// revMask is the rule byte per entry of the reverse list (pmask in that order), read in step with revSyn: one scattered read fewer per weight.
// Kpost is the fast postsynaptic trace, whose cube the heterosynaptic term multiplies. hist is the spike history per cell when the rule is timed at the synapse, else null.
__global__ void plastIn(const unsigned int* spk, const unsigned int* nSpk,
    const int* inStart, const int* revSyn, const int* revPre, float* w,
    const unsigned char* revMask, const float* Kpre, const unsigned int* hist,
    const float* Kslow, const float* wRef, const float* Kpost, int commit, float gain, const float* gState, int iGlobal,
    const float* modPrev, int modPlast, float modBase, float* elig){
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warps = (gridDim.x*blockDim.x) >> 5;
  const unsigned int count = *nSpk;
  for(unsigned int k = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; k < count; k += warps){
    const int i = (int)spk[k];
    float z = Kslow ? Kslow[i] : 0.0f;
    const float kq = Kpost[i];
    const float gate = kq*kq*kq;
    // the third factor for this cell's incoming synapses: the gain, times modBase + its modulator level under modPlast 1 (MODEL.md 4)
    const float g = modPlast == 1 ? gain*(modBase + modPrev[i]) : gain;
    for(int kk = inStart[i] + (int)lane, ke = inStart[i+1]; kk < ke; kk += (int)WARP)
      plastInSyn(revSyn[kk], (unsigned int)revPre[kk], revMask ? revMask[kk] : 1u, w, Kpre, hist, wRef, z, gate, commit, g, gState, iGlobal, modPlast, elig);
  }
}

// The eligibility sweep (modPlast 2, MODEL.md 4), as eligSweep() in simworker.js: every modStep ms each excitatory plastic synapse's trace pays (1 - e^(-modStep/eligTau)) of itself into its weight times modBase + the modulator level of its postsynaptic cell (the step's own, after the field update), then decays; a committed synapse keeps its weight and its trace still decays.
__global__ void eligSweepK(unsigned int m, float* w, float* elig, const int* post, const unsigned char* pmask,
    const float* mod, float modBase, float eligPay, float decElig, const float* wRef, int commit){
  unsigned int s = blockIdx.x*blockDim.x + threadIdx.x;
  if(s >= m) return;
  unsigned int rid = pmask ? (pmask[s] & 31u) : 1u;
  if(rid == 0u) return;
  float c = elig[s];
  if(c == 0.0f) return;
  const float* R = cRules + rid*RULE_STRIDE;
  if(!isCommitted(commit, wRef, R, s)){
    float wj = w[s];
    if(wj > 0){
      float nw = wj + c*eligPay*(modBase + mod[post[s]]);
      if(nw > R[R_wmax]) nw = R[R_wmax];
      w[s] = nw > 1e-4f ? nw : 1e-4f;
    }
  }
  elig[s] = c*decElig;
}

// hist and dmask are each cell's spike history (bit k for a spike k steps ago) and the delays its synapses carry (bit d); the history moves on here, and under the rule at the synapse (atSyn) a cell whose history meets its mask has a spike arriving next step and is listed for the arrival pass.
// The global factor (MODEL.md 4): the count of the inhibited population's spikes since the last fold, low-passed into the count that stands for its rate, and the ratio of that rate to the target minus one, which the next step's inhibitory updates read; the same float32 stores as the other engines.
__global__ void gfacK(unsigned int* gCnt, float* gState, float decG, float tauGs, float gInv){
  float pf = gState[0]*decG + (float)(*gCnt)/tauGs;
  gState[0] = pf; gState[1] = pf*gInv - 1.0f; *gCnt = 0u;
}
__global__ void traceStep(int n, const unsigned char* fired,
    float* Kpre, float* Kpost, float decS, float decM, float* Kslow, float decY, float* Khom, float decH,
    unsigned int* spkAcc, unsigned int* hist, const unsigned int* dmask, int atSyn,
    unsigned int* arr, unsigned int* nArr){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  // the scaling accumulator rides along here rather than in its own pass: this kernel already visits every neuron every step with fired in hand
  if(spkAcc && fired[i]) spkAcc[i] += 1u;
  if(hist){
    unsigned int h = ((hist[i] | (fired[i] ? 1u : 0u)) << 1) & 0x1ffffu;
    hist[i] = h;
    if(atSyn && (h & dmask[i])){ unsigned int k = atomicAdd(nArr, 1u); arr[k] = (unsigned int)i; }
  }
  float kp = Kpre[i] + (fired[i] ? 1.0f : 0.0f);
  float ko = Kpost[i] + (fired[i] ? 1.0f : 0.0f);
  Kpre[i] = kp*decS; Kpost[i] = ko*decM;   // tau+ and tau- (Bi and Poo)
  if(Kslow){
    float kz = Kslow[i] + (fired[i] ? 1.0f : 0.0f);
    Kslow[i] = kz*decY;
  }
  // the sliding threshold's slow rate trace, the same two float32 stores the reference engine makes
  if(Khom){
    float kh = Khom[i] + (fired[i] ? 1.0f : 0.0f);
    Khom[i] = kh*decH;
  }
}
// One value into every entry: the slow rate trace starts where slideInit says.
__global__ void fillK(int n, float* a, float v){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) a[i] = v;
}

// Zenke, Agnes and Gerstner 2015 equation 16: the heterosynaptic reference weight follows the synaptic weight through a double well, lower fixed point at zero and upper at consW with the midpoint unstable.
// The reference starts inside its well: a weight above consW under a consolidating rule takes consW, matching refIntoWell() in src/simworker.js.
__global__ void initRefK(unsigned int m, float* wRef, const float* w, const unsigned char* pmask, int mode){
  unsigned int s = blockIdx.x*blockDim.x + threadIdx.x;
  if(s >= m) return;
  unsigned int rid = pmask ? (pmask[s] & 31u) : 1u;
  if(rid == 0u || !(w[s] > 0.0f)) return;
  const float* R = cRules + rid*RULE_STRIDE;
  if(R[R_cons] <= 0.0f || R[R_consW] <= 0.0f) return;
  if(mode == 1) wRef[s] = 0.0f;
  else if(mode == 2) wRef[s] = R[R_consW];
  else if(wRef[s] > R[R_consW]) wRef[s] = R[R_consW];
}
// Excitatory plastic synapses only, matching src/simworker.js consolidate().
__global__ void consolidateK(unsigned int m, const float* w, float* wRef,
    const unsigned char* pmask, float rate){
  unsigned int s = blockIdx.x*blockDim.x + threadIdx.x;
  if(s >= m) return;
  unsigned int rid = pmask ? (pmask[s] & 31u) : 1u;   // the rule is the low five bits; the channel the top three
  if(rid == 0u) return;
  const float* R = cRules + rid*RULE_STRIDE;
  if(R[R_cons] <= 0.0f) return;
  float wj = w[s];
  if(wj <= 0) return;
  const float consW = R[R_consW];
  float r = wRef[s];
  // held in [0, max(r, w, consW)], which the exact equation cannot leave, as in the reference: the step is forward Euler over a cubic and a short timescale otherwise diverges
  float nr = r + rate*(wj - r - R[R_consP]*r*(consW*0.5f - r)*(consW - r));
  float top = r > wj ? r : wj;
  if(consW > top) top = consW;
  wRef[s] = nr < 0.0f ? 0.0f : (nr > top ? top : nr);
}

// Multiplicative synaptic scaling (Turrigiano 2008; van Rossum, Bi and Turrigiano 2000), matching applyScaling() in src/simworker.js: once per simulated second each neuron scales its incoming excitatory weights by its rate error against the target, and the accumulator resets.
// Runs over the reverse index, which this engine already builds for plastIn, so the only new state is the per-neuron spike counter.
// A committed synapse has left the plastic pool, and scaling is part of that pool: it is left where it is (MODEL.md 4, terms in combination), the same test the two weight-update halves make.
// The revMask argument is the rule byte per entry of the reverse list, as plastIn reads it.
__global__ void scaleK(int n, const int* inStart, const int* revSyn,
    float* w, const unsigned char* revMask, unsigned int* spkAcc,
    float rho0, float sEta, const float* rhoI, const float* wRef, int commit){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  float rho = rhoI ? rhoI[i] : rho0;
  float err = (rho - (float)spkAcc[i]) / rho;
  if(err > 2.0f) err = 2.0f; else if(err < -2.0f) err = -2.0f;
  if(err != 0.0f){
    float f = 1.0f + sEta*err;
    for(int k = inStart[i], ke = inStart[i+1]; k < ke; k++){
      int s = revSyn[k];
      unsigned int rid = revMask ? (revMask[k] & 31u) : 1u;   // the rule is the low five bits; the channel the top three
      if(rid == 0u) continue;
      if(isCommitted(commit, wRef, &cRules[rid*RULE_STRIDE], s)) continue;
      const float wmax = cRules[rid*RULE_STRIDE + R_wmax];
      float wj = w[s];
      if(wj > 0){ float nw = wj*f; w[s] = nw < wmax ? nw : wmax; }
    }
  }
  spkAcc[i] = 0;
}

// Measured per-neuron set points (rhoMode).
// The first calS seconds run with the plasticity rules off while each neuron's own rate is recorded; this then locks that rate as its personal homeostatic target, clamped to the band the reference uses so a silent or a runaway cell cannot set an unreachable one.
// Matches finishCalibration() in src/simworker.js.
__global__ void finishCalK(int n, const unsigned int* spkAcc, float* rhoI,
    float* alphaI, float secs, float tauSum){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  float r = (float)spkAcc[i]/secs;
  if(r < 0.25f) r = 0.25f;
  if(r > 30.0f) r = 30.0f;
  rhoI[i] = r;
  alphaI[i] = r*0.001f*tauSum;   // rho (tau+ + tau-)
}

// Per-millisecond recovery of the short-term plasticity state, exact exponential, matching stpStep() in src/simworker.js.
// A thread per cell and class over n*C, the class from the index; a class that is off is left alone.
__global__ void stpStepK(int n, int C, float* stpX, float* stpR){
  size_t k = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if(k >= (size_t)n*C) return;
  const int c = (int)(k / n);
  if(!cStpOn[c]) return;
  const float4 p = cStpP[c];
  stpX[k] = 1.0f - (1.0f - stpX[k])*p.y;
  stpR[k] = p.x + (stpR[k] - p.x)*p.z;
}

// Records one neuron's membrane potential into a per-step slot, so the viewer's scope can show a live trace from a run the child is driving.
// One thread; the copy back happens once per tick rather than per step.
__global__ void watchK(const float* v, float* dst, int slot, int idx){
  if(threadIdx.x || blockIdx.x) return;
  dst[slot] = idx >= 0 ? v[idx] : 0.0f;
}

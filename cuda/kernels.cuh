// The reference engine's step and plasticity as CUDA kernels, used by the engine child (engine.cu), which cuda/parity.mjs drives.
// Arithmetic and semantics mirror src/simworker.js exactly; the plasticity decomposition (outgoing pass, incoming pass, trace pass as sequential kernels over disjoint weight slices) is the one the WebGPU engine proved against the cross-engine battery.
#pragma once
#include <cuda_runtime.h>

#define ROWS 18

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

__global__ void stepKick(int n, int cur,
    const float* a, const float* b, const float* c, const float* d,
    const float* bias, const float* ext, const float* namp,
    unsigned int t, unsigned int seed, float* v, float* u, float* ring,
    const int* preStart, const int* post, const float* w,
    const unsigned char* delay, unsigned char* fired,
    unsigned int* counts, unsigned long long* total,
    unsigned char* refCnt, unsigned int refrac,
    float* stpX, float* stpR, float stpU, float stpScale, int stpOrder,
    unsigned char* firstMs, int stepIdx,
    const unsigned short* pool, const float* poolK, unsigned int* poolCnt,
    unsigned int poolG, unsigned int par, float vmin,
    const float* C7, const float* k7, const float* vr7, const float* vt7, const float* vp7,
    const unsigned char* grd, const float* gthr, const float* gslp){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  // Izhikevich 2007 (MODEL.md 1): u is measured from the rest, the membrane is k (v - vr)(v - vt) over C, the spike at the row's vpeak
  const float uRef = vr7[i];
  if(refrac > 0 && refCnt[i] > 0){
    refCnt[i]--;
    float ui2 = u[i] + a[i]*(b[i]*(v[i] - uRef) - u[i]);
    u[i] = ui2; fired[i] = 0;
    return;
  }
  float I = ring[cur*n + i] + bias[i] + (ext ? ext[i] : 0.0f);
  // feedback inhibition: the pool's kick per spike times the spikes it fired last millisecond, from the half of the table the last step wrote
  if(pool){ unsigned short g = pool[i];
    if(g) I += poolK[g]*(float)poolCnt[(par ^ 1u)*poolG + g]; }
  if(namp && namp[i] != 0.0f) I += namp[i]*noiseDraw((unsigned int)i, t, seed);
  float vi = v[i], ui = u[i];
    const float Ci = C7[i], ki = k7[i], vri = vr7[i], vti = vt7[i];
    if(grd && grd[i]){
      // a graded cell's membrane is passive: the leak is the row's slope at rest, k (vt - vr), and the potential is capped at vpeak
      const float gl = ki*(vti - vri);
      vi += 0.5f*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
      vi += 0.5f*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
      if(vi > vp7[i]) vi = vp7[i];
    } else {
    vi += 0.5f*(ki*(vi - vri)*(vi - vti) - ui + I)/Ci;
    if(vi < vmin) vi = vmin;
    vi += 0.5f*(ki*(vi - vri)*(vi - vti) - ui + I)/Ci;
    if(vi < vmin) vi = vmin;
    }
    ui += a[i]*(b[i]*(vi - vri) - ui);
    const float vpk = vp7[i];
  // a graded cell (MODEL.md 1): no spike, no reset, no refractory period; each millisecond its synapses carry a spike of amplitude r
  if(grd && grd[i]){
    float r = fminf(1.0f, fmaxf(0.0f, (vi - gthr[i])/gslp[i]));
    if(r > 0.0f) for(int s = preStart[i], e = preStart[i+1]; s < e; s++)
      atomicAdd(&ring[((cur + delay[s]) % ROWS)*n + post[s]], w[s]*r);
    v[i] = vi; u[i] = ui; fired[i] = 0;
    return;
  }
  unsigned char f = 0;
  if(vi >= vpk){
    vi = c[i]; ui += d[i]; f = 1;
    if(refrac > 0) refCnt[i] = (unsigned char)refrac;
    if(pool && pool[i]) atomicAdd(&poolCnt[par*poolG + pool[i]], 1u);
    counts[i]++;
    // the step in the tick this cell first fired, 255 if it did not: the readout bins at the tick, so without this the finest grain any measurement has is the whole tick, coarser at 50 steps than the STDP window.
    if(firstMs && firstMs[i] == 255)
      firstMs[i] = (unsigned char)(stepIdx < 254 ? stepIdx : 254);
    atomicAdd(total, 1ull);
    // Tsodyks-Markram release: as published (stpOrder 0) the probability times the resources is released, then the probability jumps; stpOrder 1 jumps first. stpScale is 1, or the inverse of the rested release so a rested synapse delivers its full weight.
    // Matches stpRelease() in src/simworker.js
    float sf = 1.0f;
    if(stpX){
      if(stpOrder) stpR[i] = stpR[i] + stpU*(1.0f - stpR[i]);
      float rel = stpR[i]*stpX[i];
      stpX[i] = stpX[i] - rel;
      if(!stpOrder) stpR[i] = stpR[i] + stpU*(1.0f - stpR[i]);
      sf = rel*stpScale;
    }
    for(int s = preStart[i], e = preStart[i+1]; s < e; s++)
      atomicAdd(&ring[((cur + delay[s]) % ROWS)*n + post[s]], w[s]*sf);
  }
  v[i] = vi; u[i] = ui; fired[i] = f;
}

__global__ void stepExp(int n, int cur, float decE, float decI,
    float itauE, float itauI,
    const float* a, const float* b, const float* c, const float* d,
    const float* bias, const float* ext, const float* namp,
    unsigned int t, unsigned int seed, float* v, float* u, float* ring, float* ring2,
    float* gE, float* gI,
    const int* preStart, const int* post, const float* w,
    const unsigned char* delay, unsigned char* fired,
    unsigned int* counts, unsigned long long* total,
    unsigned char* refCnt, unsigned int refrac,
    float* stpX, float* stpR, float stpU, float stpScale, int stpOrder,
    unsigned char* firstMs, int stepIdx,
    const unsigned short* pool, const float* poolK, unsigned int* poolCnt,
    unsigned int poolG, unsigned int par, float vmin,
    const float* C7, const float* k7, const float* vr7, const float* vt7, const float* vp7,
    const unsigned char* grd, const float* gthr, const float* gslp,
    int KX, float* gX, float* ringX, const float* decX, const float* itauX, const unsigned char* pmask,
    int cond, float eE, float eI, const float* erevX){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  const float uRef = vr7[i];               // u is measured from the rest (2007)
  const float v0 = v[i];
  float ge = gE[i]*decE + ring[cur*n + i];
  float gi = gI[i]*decI + ring2[cur*n + i];
  // receptor channels 2 and up (MODEL.md 2): their own decay, their own ring;
  // conductance mode drives |g| (E - v) with the step's opening v
  float Ix = 0.0f;
  for(int x = 0; x < KX; x++){
    float g = gX[x*n + i]*decX[x] + ringX[((size_t)x*ROWS + cur)*n + i];
    gX[x*n + i] = g; Ix += cond ? fabsf(g)*(erevX[x] - v0) : g*itauX[x];
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
  float I = (cond ? fabsf(ge)*(eE - v0) + fabsf(gi)*(eI - v0) : ge*itauE + gi*itauI) + Ix + bias[i] + (ext ? ext[i] : 0.0f);
  if(namp && namp[i] != 0.0f) I += namp[i]*noiseDraw((unsigned int)i, t, seed);
  float vi = v[i], ui = u[i];
    const float Ci = C7[i], ki = k7[i], vri = vr7[i], vti = vt7[i];
    if(grd && grd[i]){
      // a graded cell's membrane is passive: the leak is the row's slope at rest, k (vt - vr), and the potential is capped at vpeak
      const float gl = ki*(vti - vri);
      vi += 0.5f*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
      vi += 0.5f*(-gl*(vi - vri) - ui + I)/Ci; if(vi < vmin) vi = vmin;
      if(vi > vp7[i]) vi = vp7[i];
    } else {
    vi += 0.5f*(ki*(vi - vri)*(vi - vti) - ui + I)/Ci;
    if(vi < vmin) vi = vmin;
    vi += 0.5f*(ki*(vi - vri)*(vi - vti) - ui + I)/Ci;
    if(vi < vmin) vi = vmin;
    }
    ui += a[i]*(b[i]*(vi - vri) - ui);
    const float vpk = vp7[i];
  if(grd && grd[i]){
    float r = fminf(1.0f, fmaxf(0.0f, (vi - gthr[i])/gslp[i]));
    if(r > 0.0f) for(int s = preStart[i], e = preStart[i+1]; s < e; s++){
      float wt = w[s]*r;
      int slot = (cur + delay[s]) % ROWS, at = slot*n + post[s];
      int ch = pmask ? (pmask[s] >> 5) : 0;
      if(ch >= 2) atomicAdd(&ringX[((size_t)(ch - 2)*ROWS + slot)*n + post[s]], wt);
      else atomicAdd(wt > 0 ? &ring[at] : &ring2[at], wt);
    }
    v[i] = vi; u[i] = ui; gE[i] = ge; gI[i] = gi; fired[i] = 0;
    return;
  }
  unsigned char f = 0;
  if(vi >= vpk){
    vi = c[i]; ui += d[i]; f = 1;
    if(refrac > 0) refCnt[i] = (unsigned char)refrac;
    if(pool && pool[i]) atomicAdd(&poolCnt[par*poolG + pool[i]], 1u);
    counts[i]++;
    // the step in the tick this cell first fired, 255 if it did not (see stepKick)
    if(firstMs && firstMs[i] == 255)
      firstMs[i] = (unsigned char)(stepIdx < 254 ? stepIdx : 254);
    atomicAdd(total, 1ull);
    float sf = 1.0f;
    if(stpX){
      if(stpOrder) stpR[i] = stpR[i] + stpU*(1.0f - stpR[i]);   // stpOrder 1: facilitation before release
      float rel = stpR[i]*stpX[i];
      stpX[i] = stpX[i] - rel;
      if(!stpOrder) stpR[i] = stpR[i] + stpU*(1.0f - stpR[i]);  // as published: facilitation after release
      sf = rel*stpScale;
    }
    for(int s = preStart[i], e = preStart[i+1]; s < e; s++){
      float wt = w[s]*sf;
      int slot = (cur + delay[s]) % ROWS, at = slot*n + post[s];
      int ch = pmask ? (pmask[s] >> 5) : 0;
      if(ch >= 2) atomicAdd(&ringX[((size_t)(ch - 2)*ROWS + slot)*n + post[s]], wt);
      else atomicAdd(wt > 0 ? &ring[at] : &ring2[at], wt);
    }
  }
  v[i] = vi; u[i] = ui; gE[i] = ge; gI[i] = gi; fired[i] = f;
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
__global__ void plastOut(int n, const unsigned char* fired,
    const int* preStart, const int* post, float* w,
    const unsigned char* pmask, const float* Kpost,
    float iAlpha, const float* alphaI, const float* wRef, int commit){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n || !fired[i]) return;
  for(int s = preStart[i], e = preStart[i+1]; s < e; s++){
    unsigned int rid = pmask ? (pmask[s] & 31u) : 1u;   // the rule is the low five bits; the channel the top three
    if(rid == 0u) continue;                  // 0 is frozen
    const float* R = cRules + rid*RULE_STRIDE;
    if(isCommitted(commit, wRef, R, s)) continue;
    const float wmax = R[R_wmax];
    const int wdep = R[R_wdep] != 0.0f;
    float wj = w[s];
    if(wj > 0){
      float nw = wj - R[R_aM]*Kpost[post[s]]*(wdep ? wj : 1.0f) + R[R_tin];
      if(nw > wmax) nw = wmax;
      w[s] = nw > 1e-4f ? nw : 1e-4f;
    } else {
      float al = alphaI ? alphaI[post[s]] : iAlpha;
      float dd = -R[R_iEta]*(Kpost[post[s]] - al);
      if(wdep) dd = softBound(wj, dd, wmax);
      float nw = wj + dd;
      if(nw > -1e-4f) nw = -1e-4f;
      if(nw < -wmax) nw = -wmax;
      w[s] = nw;
    }
  }
}

// The triplet and heterosynaptic terms read the slow postsynaptic trace before this millisecond's spike is added, matching the reference: the triplet potentiates by postsynaptic history (Pfister and Gerstner 2006), and the cubed trace gates heterosynaptic regression toward the reference weight so only bursting engages it (Zenke et al. 2015).
__global__ void plastIn(int n, const unsigned char* fired,
    const int* inStart, const int* revSyn, const int* revPre, float* w,
    const unsigned char* pmask, const float* Kpre,
    const float* Kslow, const float* wRef, float kRef, int commit){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n || !fired[i]) return;
  float z = Kslow ? Kslow[i] : 0.0f;
  // the gate is this neuron's trace against its set point, so it is shared; only the amplitude multiplying it is per rule
  float g = z/kRef;
  float gate = g*g*g;
  for(int k = inStart[i], ke = inStart[i+1]; k < ke; k++){
    int s = revSyn[k];
    unsigned int rid = pmask ? (pmask[s] & 31u) : 1u;   // the rule is the low five bits; the channel the top three
    if(rid == 0u) continue;
    const float* R = cRules + rid*RULE_STRIDE;
    if(isCommitted(commit, wRef, R, s)) continue;
    const float wmax = R[R_wmax], het = R[R_het];
    const int wdep = R[R_wdep] != 0.0f;
    float wp = w[s], kp = Kpre[revPre[k]];
    if(wp > 0){
      float nw = wp + (R[R_aP] + R[R_trip]*z)*kp;
      if(het > 0 && wRef){
        float hetF = het*gate;
        if(hetF > 1.0f) hetF = 1.0f;
        nw -= hetF*(wp - wRef[s]);
      }
      if(nw > wmax) nw = wmax;
      w[s] = nw > 1e-4f ? nw : 1e-4f;
    } else {
      float dd = -R[R_iEta]*kp;
      if(wdep) dd = softBound(wp, dd, wmax);
      float nw = wp + dd;
      w[s] = nw > -wmax ? nw : -wmax;
    }
  }
}

__global__ void traceStep(int n, const unsigned char* fired,
    float* Kpre, float* Kpost, float decS, float decM, float* Kslow, float decY,
    unsigned int* spkAcc){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  // the scaling accumulator rides along here rather than in its own pass: this kernel already visits every neuron every step with fired in hand
  if(spkAcc && fired[i]) spkAcc[i] += 1u;
  float kp = Kpre[i] + (fired[i] ? 1.0f : 0.0f);
  float ko = Kpost[i] + (fired[i] ? 1.0f : 0.0f);
  Kpre[i] = kp*decS; Kpost[i] = ko*decM;   // tau+ and tau- (Bi and Poo)
  if(Kslow){
    float kz = Kslow[i] + (fired[i] ? 1.0f : 0.0f);
    Kslow[i] = kz*decY;
  }
}

// Zenke, Agnes and Gerstner 2015 equation 16: the heterosynaptic reference weight follows the synaptic weight through a double well, lower fixed point at zero and upper at consW with the midpoint unstable.
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
  // clamped to the two fixed points, as in the reference: the step is forward Euler over a cubic and a short timescale otherwise diverges
  float nr = r + rate*(wj - r - R[R_consP]*r*(consW*0.5f - r)*(consW - r));
  wRef[s] = nr < 0.0f ? 0.0f : (nr > consW ? consW : nr);
}

// Multiplicative synaptic scaling (Turrigiano 2008; van Rossum, Bi and Turrigiano 2000), matching applyScaling() in src/simworker.js: once per simulated second each neuron scales its incoming excitatory weights by its rate error against the target, and the accumulator resets.
// Runs over the reverse index, which this engine already builds for plastIn, so the only new state is the per-neuron spike counter.
// A committed synapse has left the plastic pool, and scaling is part of that pool: it is left where it is (MODEL.md 4, terms in combination), the same test the two weight-update halves make.
__global__ void scaleK(int n, const int* inStart, const int* revSyn,
    float* w, const unsigned char* pmask, unsigned int* spkAcc,
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
      unsigned int rid = pmask ? (pmask[s] & 31u) : 1u;   // the rule is the low five bits; the channel the top three
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
__global__ void stpStepK(int n, float* stpX, float* stpR,
    float stpU, float decD, float decF){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  stpX[i] = 1.0f - (1.0f - stpX[i])*decD;
  stpR[i] = stpU + (stpR[i] - stpU)*decF;
}

// Records one neuron's membrane potential into a per-step slot, so the viewer's scope can show a live trace from a run the child is driving.
// One thread; the copy back happens once per tick rather than per step.
__global__ void watchK(const float* v, float* dst, int slot, int idx){
  if(threadIdx.x || blockIdx.x) return;
  dst[slot] = idx >= 0 ? v[idx] : 0.0f;
}

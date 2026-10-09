// CUDA engine child: pure compute behind the host. host/host.mjs spawns this process and speaks a fixed binary command protocol over stdio, one frame per command: u32 cmd, u32 payloadBytes, payload.
// The host keeps every protocol smart (curriculum, stimulus encoding, warm-up schedule, checkpoint writing) and translates the ENGINE.md messages the browser sends into these structs, so this process never parses JSON and never owns policy.
// Replies use the same framing.
//
// commands in            replies out
//   1 INIT                 -
//   2 TUNE                 -
//   3 EXT   f32[n]         -
//   4 NAMP  f32[n]         -
//   5 TICK  u32 steps      100 STATE: u32 steps, u32 hasV, u64 spikes, u8 fired[n]
//   9 SENDV u32 on   10 SENDT u32 on
//   6 GETW                 101 WEIGHTS: f32[m]
//   7 QUERY u32 idx, cap    102 QUERYRESULT: u32 idx, outTotal, inTotal,
//                                outN, innN, u32 out[outN], u32 inn[innN]
//   8 WATCH i32 idx         (folded into STATE as vtrace)
//   11 HELLO                103 HELLO: u32 the contract number this child was built for
//
// STATE carries the watched neuron's membrane trace when one is selected: u32 steps, u32 hasV, u64 spikes, u8 fired[n], f32 vtrace[steps] if hasV, then f32 v[n] when SENDV is on, told apart by the frame length.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <io.h>
#include <fcntl.h>
#include "kernels.cuh"

// The engine contract this child implements, PROTOCOL in src/protocol.js; src/protocol.test.mjs holds the two equal.
#define ENGINE_PROTOCOL 3

#define CK(x) do { cudaError_t e = (x); if(e != cudaSuccess){ \
  fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
  exit(1); } } while(0)

static bool readAll(void* p, size_t nBytes){
  size_t got = 0;
  while(got < nBytes){
    int r = _read(0, (char*)p + got, (unsigned int)((nBytes - got) > 1u<<20 ? 1u<<20 : (nBytes - got)));
    if(r <= 0) return false;
    got += r;
  }
  return true;
}
static void writeAll(const void* p, size_t nBytes){
  size_t put = 0;
  while(put < nBytes){
    int r = _write(1, (const char*)p + put, (unsigned int)((nBytes - put) > 1u<<20 ? 1u<<20 : (nBytes - put)));
    if(r <= 0) exit(0);
    put += r;
  }
}
static void reply(unsigned int cmd, const void* p, unsigned int nBytes){
  unsigned int head[2] = { cmd, nBytes };
  writeAll(head, 8);
  if(nBytes) writeAll(p, nBytes);
}

struct Net {
  unsigned int n = 0, m = 0, syn = 0;
  float tauE = 3, tauI = 8;
  unsigned int plast = 0, wdep = 0;
  float aP = 0, aM = 0, tauS = 16.8f, wmax = 10, iEta = 0, iRho = 0;
  float tauM = 16.8f, stpNorm = 0;          // depression window; stp delivery scale
  int stpOrder = 0;                         // stp: 0 release then facilitation, 1 the reverse
  unsigned int psc = 0;                     // exp current scale: 0 peak, 1 charge, 2 exact integral over the step
  float trip = 0, tauY = 114, het = 0, tin = 0;
  unsigned int refrac = 0;
  float vmin = -90.0f;                      // membrane floor; >= 0 disables
  float *Kslow = nullptr, *wRef = nullptr;
  float cons = 0, consW = 0.5f, consP = 10.0f;
  int commit = 0;
  // any rule wanting het or consolidation, which is what the reference weights and the consolidation cadence follow rather than the two checkpoint fields alone
  unsigned int anyHet = 0, anyCons = 0, ruleCount = 0;
  int watch = -1;
  unsigned int sendV = 0;               // stream every membrane potential
  unsigned int sendT = 0;               // stream first-spike latency per tick
  unsigned char *firstMs = nullptr;     // device, one byte per neuron
  float *vtrace = nullptr;              // device, one entry per step of a tick
  unsigned int vtraceCap = 0;
  float *stpX = nullptr, *stpR = nullptr;
  float stp = 0, stpU = 0.2f, stpTauD = 200.0f, stpTauF = 600.0f;
  float tauCons = 1200000.0f, consStep = 1200.0f;
  float consAcc = 0;
  // synaptic scaling: the cadence follows the consolidation pattern above rather than an absolute clock, which is equivalent because the accumulator resets each pass
  float scaleOn = 0, sEta = 0.001f, scaleAcc = 0;
  unsigned int *spkAcc = nullptr;
  // measured per-neuron set points: the first calS seconds run with the rules off while rates record, then each neuron locks its own target
  float rhoMode = 0, calS = 10, calAcc = 0;
  unsigned int calibrating = 0;
  float *rhoI = nullptr, *alphaI = nullptr;
  unsigned char *refCnt = nullptr;
  // feedback inhibition (ENGINE.md section 14): pool per cell, kick per spike per pool, two halves of per-pool spike counts; poolG counts the no-pool index 0, so 1 means the network has none
  unsigned short *pool = nullptr;
  // the membrane's per-cell arrays (Izhikevich 2007, MODEL.md 1): C, k, vr, vt, vpeak.
  // The 2003 presets arrive as rows of this form, so these are always given.
  float *C7 = nullptr, *k7 = nullptr, *vr7 = nullptr, *vt7 = nullptr, *vp7 = nullptr;
  // graded rows (MODEL.md 1): a flag byte per cell, release threshold and slope
  unsigned int graded = 0;
  unsigned char *grd = nullptr; float *gthr = nullptr, *gslp = nullptr;
  // receptor channels beyond E and I (MODEL.md 2): count, decay and scale per channel on the device, a conductance per cell per channel, a ring per channel
  int KX = 0; float *gX = nullptr, *ringX = nullptr, *decX = nullptr, *itauX = nullptr;
  float chanTau[6] = {0,0,0,0,0,0};
  // conductance mode (syn 2): reversal potentials of E, I and each channel
  float eE = 0.0f, eI = -70.0f, chanErev[6] = {0,0,0,0,0,0}; float *erevX = nullptr;
  float *poolK = nullptr;
  unsigned int *poolCnt = nullptr, poolG = 0;
  float *a, *b, *c, *d, *bias, *ext, *namp, *v, *u;
  float *ring, *ring2, *gE, *gI, *Kpre, *Kpost, *w;
  int *pre, *post, *inStart, *revSyn, *revPre;
  unsigned char *delay, *pmask, *fired;
  unsigned int *counts;
  unsigned int seed = 1, t = 0;      // noise seed from the host; simulated ms since init
  unsigned long long *total;
  int cur = 0;
};
static Net N;

int main(){
  _setmode(0, _O_BINARY);
  _setmode(1, _O_BINARY);
  std::vector<unsigned char> hostFired;
  std::vector<char> payload;
  for(;;){
    unsigned int head[2];
    if(!readAll(head, 8)) break;
    unsigned int cmd = head[0], len = head[1];
    payload.resize(len);
    if(len && !readAll(payload.data(), len)) break;
    const char* p = payload.data();

    if(cmd == 1){                          // INIT
      memcpy(&N.n, p, 4); memcpy(&N.m, p+4, 4); memcpy(&N.syn, p+8, 4);
      memcpy(&N.tauE, p+12, 4); memcpy(&N.tauI, p+16, 4);
      memcpy(&N.psc, p+20, 4);
      p += 24;
      unsigned int n = N.n, m = N.m;
      // The frame length is fixed by n and m: a host on another protocol version sends a different length, and the child exits rather than starting from state it invented.
      // The pool tail (u32 count, u16 per cell, f32 per pool) follows the seed and its length is read from the frame, so the check is in two parts.
      size_t base = 24 + (size_t)n*4*5 + (size_t)(n+1)*4 + (size_t)m*4*2
        + (size_t)m*2 + (size_t)n*4*2 + 4;
      if((size_t)len < base + 4){
        fprintf(stderr, "engine.cu: INIT payload is %u bytes, expected at least %zu; "
          "the host and this engine.exe disagree on the protocol (rebuild)\n", len, base + 4);
        exit(1);
      }
      auto up = [&](void** dst, size_t nBytes){
        CK(cudaMalloc(dst, nBytes));
        CK(cudaMemcpy(*dst, p, nBytes, cudaMemcpyHostToDevice));
        p += nBytes;
      };
      up((void**)&N.a, n*4); up((void**)&N.b, n*4);
      up((void**)&N.c, n*4); up((void**)&N.d, n*4); up((void**)&N.bias, n*4);
      up((void**)&N.pre, (n+1)*4); up((void**)&N.post, (size_t)m*4);
      up((void**)&N.w, (size_t)m*4); up((void**)&N.delay, m);
      up((void**)&N.pmask, m);
      // reverse index built here, same construction as the reference
      std::vector<int> preH(n+1), postH(m);
      CK(cudaMemcpy(preH.data(), N.pre, (n+1)*4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(postH.data(), N.post, (size_t)m*4, cudaMemcpyDeviceToHost));
      std::vector<int> inStart(n+1, 0), revSyn(m), revPre(m), fill(n, 0);
      for(unsigned int s = 0; s < m; s++) inStart[postH[s]+1]++;
      for(unsigned int i = 0; i < n; i++) inStart[i+1] += inStart[i];
      for(unsigned int i = 0; i < n; i++)
        for(int s = preH[i]; s < preH[i+1]; s++){
          int at = inStart[postH[s]] + fill[postH[s]]++;
          revSyn[at] = s; revPre[at] = (int)i;
        }
      CK(cudaMalloc((void**)&N.inStart, (n+1)*4));
      CK(cudaMalloc((void**)&N.revSyn, (size_t)m*4));
      CK(cudaMalloc((void**)&N.revPre, (size_t)m*4));
      CK(cudaMemcpy(N.inStart, inStart.data(), (n+1)*4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(N.revSyn, revSyn.data(), (size_t)m*4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(N.revPre, revPre.data(), (size_t)m*4, cudaMemcpyHostToDevice));
      // Initial membrane state and the noise seed come from the host, the same bytes every engine starts from (ENGINE.md section 2).
      up((void**)&N.v, n*4); up((void**)&N.u, n*4);
      memcpy(&N.seed, p, 4); p += 4;
      memcpy(&N.poolG, p, 4); p += 4;
      // After the pools: five f32[n] arrays (C, k, vr, vt, vpeak), then a u32 graded flag and under 1 a byte and two floats per cell (MODEL.md 1, ENGINE.md 3).
      size_t withPools = base + 4 + (size_t)n*2 + (size_t)N.poolG*4;
      size_t membrane = (size_t)n*4*5;
      if((size_t)len < withPools + membrane + 4){
        fprintf(stderr, "engine.cu: INIT payload is %u bytes, expected at least %zu for %u pools; "
          "the host and this engine.exe disagree on the protocol (rebuild)\n",
          len, withPools + membrane + 4, N.poolG);
        exit(1);
      }
      memcpy(&N.graded, p + (size_t)n*2 + (size_t)N.poolG*4 + membrane, 4);
      size_t expect = withPools + membrane + 4 + (N.graded ? (size_t)n*9 : 0);
      if((size_t)len != expect){
        fprintf(stderr, "engine.cu: INIT payload is %u bytes, expected %zu for %u pools; "
          "the host and this engine.exe disagree on the protocol (rebuild)\n",
          len, expect, N.poolG);
        exit(1);
      }
      up((void**)&N.pool, (size_t)n*2); up((void**)&N.poolK, (size_t)N.poolG*4);
      up((void**)&N.C7, n*4); up((void**)&N.k7, n*4); up((void**)&N.vr7, n*4);
      up((void**)&N.vt7, n*4); up((void**)&N.vp7, n*4);
      p += 4;                                // the graded word, read above
      if(N.graded){ up((void**)&N.grd, n); up((void**)&N.gthr, n*4); up((void**)&N.gslp, n*4); }
      CK(cudaMalloc((void**)&N.poolCnt, (size_t)N.poolG*8));
      CK(cudaMemset(N.poolCnt, 0, (size_t)N.poolG*8));
      N.t = 0;
      CK(cudaMalloc((void**)&N.ring, (size_t)ROWS*n*4));
      CK(cudaMalloc((void**)&N.ring2, (size_t)ROWS*n*4));
      CK(cudaMalloc((void**)&N.gE, n*4)); CK(cudaMalloc((void**)&N.gI, n*4));
      CK(cudaMalloc((void**)&N.Kpre, n*4)); CK(cudaMalloc((void**)&N.Kpost, n*4));
      CK(cudaMalloc((void**)&N.Kslow, n*4));
      CK(cudaMalloc((void**)&N.refCnt, n));
      CK(cudaMemset(N.refCnt, 0, n));
      CK(cudaMalloc((void**)&N.ext, n*4)); CK(cudaMalloc((void**)&N.namp, n*4));
      CK(cudaMalloc((void**)&N.fired, n)); CK(cudaMalloc((void**)&N.counts, n*4));
      CK(cudaMalloc((void**)&N.total, 8));
      CK(cudaMemset(N.ring, 0, (size_t)ROWS*n*4));
      CK(cudaMemset(N.ring2, 0, (size_t)ROWS*n*4));
      CK(cudaMemset(N.gE, 0, n*4)); CK(cudaMemset(N.gI, 0, n*4));
      CK(cudaMemset(N.Kpre, 0, n*4)); CK(cudaMemset(N.Kpost, 0, n*4));
      CK(cudaMemset(N.Kslow, 0, n*4));
      CK(cudaMemset(N.ext, 0, n*4)); CK(cudaMemset(N.namp, 0, n*4));
      CK(cudaMemset(N.counts, 0, n*4)); CK(cudaMemset(N.total, 0, 8));
      hostFired.resize(n);
      // Allocated here as well as in SENDT: the client asks for latency before init, when SENDT sees n == 0 and allocates nothing.
      if(N.firstMs){ cudaFree(N.firstMs); N.firstMs = nullptr; }
      if(N.sendT && n) CK(cudaMalloc(&N.firstMs, n));
      fprintf(stderr, "engine.cu: init %u neurons, %u synapses, %s\n",
        n, m, N.syn ? "exp" : "kick");
    }
    else if(cmd == 2){                     // TUNE
      memcpy(&N.plast, p, 4); memcpy(&N.wdep, p+4, 4);
      memcpy(&N.aP, p+8, 4); memcpy(&N.aM, p+12, 4);
      memcpy(&N.tauS, p+16, 4); memcpy(&N.wmax, p+20, 4);
      memcpy(&N.iEta, p+24, 4); memcpy(&N.iRho, p+28, 4);
      if(len >= 48){
        memcpy(&N.trip, p+32, 4); memcpy(&N.tauY, p+36, 4);
        memcpy(&N.het, p+40, 4); memcpy(&N.tin, p+44, 4);
      } else { N.trip = 0; N.het = 0; N.tin = 0; }
      if(len >= 52) memcpy(&N.refrac, p+48, 4); else N.refrac = 0;
      if(N.refrac > 20) N.refrac = 20;
      // consolidation of the reference weight (Zenke 2015 eq 16)
      if(len >= 64){
        memcpy(&N.cons, p+52, 4); memcpy(&N.consW, p+56, 4);
        memcpy(&N.consP, p+60, 4);
      } else { N.cons = 0; }
      // zero means the published ratio against this tissue's wmax, a tenth of maximum, rather than the published number for a wmax of 5
      if(N.consW <= 0) N.consW = 0.1f*N.wmax;
      if(N.consP <= 0) N.consP = 10.0f;
      // short-term plasticity (Tsodyks-Markram; Zenke 2015 eqs 9-10)
      if(len >= 80){
        memcpy(&N.stp, p+64, 4); memcpy(&N.stpU, p+68, 4);
        memcpy(&N.stpTauD, p+72, 4); memcpy(&N.stpTauF, p+76, 4);
      } else { N.stp = 0; }
      if(N.stpU <= 0) N.stpU = 0.2f;
      if(N.stpTauD <= 0) N.stpTauD = 200.0f;
      if(N.stpTauF <= 0) N.stpTauF = 600.0f;
      if(len >= 88){
        memcpy(&N.scaleOn, p+80, 4); memcpy(&N.sEta, p+84, 4);
      } else { N.scaleOn = 0; }
      if(N.sEta <= 0) N.sEta = 0.001f;
      if(len >= 96){
        memcpy(&N.rhoMode, p+88, 4); memcpy(&N.calS, p+92, 4);
      } else { N.rhoMode = 0; }
      if(N.calS <= 0) N.calS = 10;
      // the depression window and the short-term plasticity delivery scale; absent means a symmetric window and the published release
      if(len >= 104){
        memcpy(&N.tauM, p+96, 4); memcpy(&N.stpNorm, p+100, 4);
      } else { N.tauM = N.tauS; N.stpNorm = 0; }
      if(N.tauM <= 0) N.tauM = N.tauS;
      // byte 100 carries two switches: 1 is stpNorm, 2 is stpOrder (facilitation before release)
      { int code = (int)(N.stpNorm + 0.5f); N.stpOrder = (code >> 1) & 1; N.stpNorm = (float)(code & 1); }
      if(len >= 116){
        float cm = 0;
        memcpy(&cm, p+104, 4); N.commit = cm > 0.5f ? 1 : 0;
        memcpy(&N.tauCons, p+108, 4); memcpy(&N.consStep, p+112, 4);
      } else { N.commit = 0; }
      if(N.tauCons <= 0) N.tauCons = 1200000.0f;
      if(N.consStep <= 0) N.consStep = 1200.0f;
      // the membrane floor; absent means the default, at or above zero means none (the most negative float, never reached)
      if(len >= 120) memcpy(&N.vmin, p+116, 4); else N.vmin = -90.0f;
      // receptor channels: u32 count at 120, six taus from 124.
      // Allocated on the first tune that names them, sized by the count. conductance mode's reversals: eE at 148, eI at 152, six channel reversals from 156; the rule table sits at 184
      if(len >= 180){ memcpy(&N.eE, p+148, 4); memcpy(&N.eI, p+152, 4); memcpy(N.chanErev, p+156, 24); }
      if(!N.erevX) CK(cudaMalloc((void**)&N.erevX, 24));
      CK(cudaMemcpy(N.erevX, N.chanErev, 24, cudaMemcpyHostToDevice));
      { unsigned int kx = 0; if(len >= 148){ memcpy(&kx, p+120, 4); memcpy(N.chanTau, p+124, 24); }
        if(kx > 6) kx = 6;
        if(kx && !N.syn){ fprintf(stderr, "engine.cu: receptor channels need exponential synapses\n"); exit(1); }
        if((int)kx != N.KX){
          if(N.gX){ cudaFree(N.gX); cudaFree(N.ringX); cudaFree(N.decX); cudaFree(N.itauX); N.gX = N.ringX = N.decX = N.itauX = nullptr; }
          N.KX = (int)kx;
          if(N.KX){
            CK(cudaMalloc((void**)&N.gX, (size_t)N.KX*N.n*4)); CK(cudaMemset(N.gX, 0, (size_t)N.KX*N.n*4));
            CK(cudaMalloc((void**)&N.ringX, (size_t)N.KX*ROWS*N.n*4)); CK(cudaMemset(N.ringX, 0, (size_t)N.KX*ROWS*N.n*4));
            CK(cudaMalloc((void**)&N.decX, 24)); CK(cudaMalloc((void**)&N.itauX, 24));
          }
        }
        if(N.KX){
          float dx[6], ix[6];
          for(int x = 0; x < N.KX; x++){ float tau = N.chanTau[x] > 0 ? N.chanTau[x] : 3.0f; dx[x] = expf(-1.0f/tau); ix[x] = N.psc == 2 ? tau*(1.0f - dx[x]) : N.psc == 1 ? (1.0f - dx[x]) : 1.0f; }
          CK(cudaMemcpy(N.decX, dx, 24, cudaMemcpyHostToDevice)); CK(cudaMemcpy(N.itauX, ix, 24, cudaMemcpyHostToDevice));
        }
      }
      if(N.vmin >= 0) N.vmin = -3.4e38f;
      // The per-rule table, appended to the frame from byte 120: a u32 row count then rowCount*11 floats, the same fields in the same order the reference engine and the WebGPU shaders use.
      // Absent, or a count below two, means row 1 is built here from the scalars, so a synapse carrying 1 computes what it computed before rules existed.
      {
        float rules[RULE_MAX*RULE_STRIDE];
        memset(rules, 0, sizeof(rules));
        unsigned int rc = 0;
        if(len >= 188) memcpy(&rc, p+184, 4);
        if(rc > RULE_MAX) rc = RULE_MAX;
        bool sent = rc >= 2 && len >= 188 + rc*RULE_STRIDE*4;
        if(sent){
          memcpy(rules, p + 188, (size_t)rc*RULE_STRIDE*4);
          N.ruleCount = rc;
        } else {
          float* R = rules + RULE_STRIDE;             // row 1
          R[0] = N.aP; R[1] = N.aM; R[2] = N.wmax; R[3] = N.wdep ? 1.0f : 0.0f;
          R[4] = N.trip; R[5] = N.het; R[6] = N.tin; R[7] = N.iEta;
          R[8] = N.cons; R[9] = N.consW; R[10] = N.consP;
          N.ruleCount = 2;
        }
        N.anyHet = 0; N.anyCons = 0;
        for(unsigned int r = 1; r < N.ruleCount; r++){
          if(rules[r*RULE_STRIDE + R_het] > 0) N.anyHet = 1;
          if(rules[r*RULE_STRIDE + R_cons] > 0) N.anyCons = 1;
        }
        CK(cudaMemcpyToSymbol(cRules, rules, sizeof(rules)));
        fprintf(stderr, "engine.cu: rules=%u anyHet=%u anyCons=%u aP[1]=%g\n",
          N.ruleCount, N.anyHet, N.anyCons, rules[RULE_STRIDE + R_aP]);
      }
      if((N.scaleOn > 0 || N.rhoMode > 0) && !N.spkAcc && N.n){
        CK(cudaMalloc((void**)&N.spkAcc, (size_t)N.n*4));
        CK(cudaMemset(N.spkAcc, 0, (size_t)N.n*4));
      }
      // begin a measurement window on the edge where the mode turns on with plasticity, exactly where the reference begins one
      if(N.plast && N.rhoMode > 0 && !N.rhoI && N.n){
        CK(cudaMalloc((void**)&N.rhoI, (size_t)N.n*4));
        CK(cudaMalloc((void**)&N.alphaI, (size_t)N.n*4));
        CK(cudaMemset(N.spkAcc, 0, (size_t)N.n*4));
        N.calibrating = 1; N.calAcc = 0;
      } else if(!N.plast) N.calibrating = 0;
      if(N.stp > 0 && !N.stpX && N.n){
        CK(cudaMalloc((void**)&N.stpX, (size_t)N.n*4));
        CK(cudaMalloc((void**)&N.stpR, (size_t)N.n*4));
        std::vector<float> ix(N.n, 1.0f), ir(N.n, N.stpU);
        CK(cudaMemcpy(N.stpX, ix.data(), (size_t)N.n*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(N.stpR, ir.data(), (size_t)N.n*4, cudaMemcpyHostToDevice));
      }
      if((N.anyHet > 0 || N.anyCons > 0) && !N.wRef && N.w){
        // heterosynaptic reference = the weights when het switches on, the same capture rule as the reference engine
        CK(cudaMalloc((void**)&N.wRef, (size_t)N.m*4));
        CK(cudaMemcpy(N.wRef, N.w, (size_t)N.m*4, cudaMemcpyDeviceToDevice));
      }
      fprintf(stderr, "engine.cu: tune plast=%u wdep=%u aP=%g aM=%g wmax=%g trip=%g het=%g tin=%g cons=%g consW=%g\n",
        N.plast, N.wdep, N.aP, N.aM, N.wmax, N.trip, N.het, N.tin,
        N.cons, N.consW);
      fprintf(stderr, "engine.cu: tune stp=%g scale=%g sEta=%g\n",
        N.stp, N.scaleOn, N.sEta);
    }
    else if(cmd == 3){                     // EXT
      CK(cudaMemcpy(N.ext, p, N.n*4, cudaMemcpyHostToDevice));
    }
    else if(cmd == 4){                     // NAMP
      CK(cudaMemcpy(N.namp, p, N.n*4, cudaMemcpyHostToDevice));
    }
    else if(cmd == 7){                     // QUERY
      // Outgoing synapses are a slice of the forward CSR; incoming ones come from the reverse index the plasticity passes already build, so this is two small copies rather than a scan of every synapse.
      // A third word, optional, says the caller wants outgoing only; it exists for the JavaScript engines, where the incoming list means a scan of every synapse, and this engine honors it so all three answer the same request the same way.
      unsigned int idx = 0, cap = 2000, outOnly = 0;
      memcpy(&idx, p, 4); memcpy(&cap, p+4, 4);
      if(len >= 12) memcpy(&outOnly, p+8, 4);
      if(idx >= N.n || !N.pre || !N.post){ reply(102, nullptr, 0); }
      else {
        int p0 = 0, p1 = 0, i0 = 0, i1 = 0;
        CK(cudaMemcpy(&p0, N.pre + idx, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&p1, N.pre + idx + 1, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&i0, N.inStart + idx, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&i1, N.inStart + idx + 1, 4, cudaMemcpyDeviceToHost));
        unsigned int outTotal = (unsigned int)(p1 - p0);
        unsigned int inTotal = outOnly ? 0u : (unsigned int)(i1 - i0);
        unsigned int outN = outTotal < cap ? outTotal : cap;
        unsigned int innN = inTotal < cap ? inTotal : cap;
        std::vector<int> outv(outN), innv(innN);
        if(outN) CK(cudaMemcpy(outv.data(), N.post + p0, (size_t)outN*4,
          cudaMemcpyDeviceToHost));
        if(innN) CK(cudaMemcpy(innv.data(), N.revPre + i0, (size_t)innN*4,
          cudaMemcpyDeviceToHost));
        std::vector<unsigned char> out(20 + (size_t)(outN + innN)*4);
        unsigned int head[5] = { idx, outTotal, inTotal, outN, innN };
        memcpy(out.data(), head, 20);
        if(outN) memcpy(out.data() + 20, outv.data(), (size_t)outN*4);
        if(innN) memcpy(out.data() + 20 + (size_t)outN*4, innv.data(),
          (size_t)innN*4);
        reply(102, out.data(), (unsigned int)out.size());
      }
    }
    else if(cmd == 8){                     // WATCH
      memcpy(&N.watch, p, 4);
      if(N.watch >= (int)N.n) N.watch = -1;
    }
    else if(cmd == 9){                     // SENDV
      memcpy(&N.sendV, p, 4);
    }
    else if(cmd == 10){                    // SENDT, first-spike latency
      memcpy(&N.sendT, p, 4);
      if(N.sendT && !N.firstMs && N.n) CK(cudaMalloc(&N.firstMs, N.n));
    }
    else if(cmd == 5){                     // TICK
      unsigned int steps; memcpy(&steps, p, 4);
      unsigned int n = N.n;
      int block = 256, grid = (n + block - 1)/block;
      float decE = expf(-1.0f/N.tauE), decI = expf(-1.0f/N.tauI);
      // psc 0: w is the peak current, the jump is w (NEST, Brian); psc 1: the jump is scaled so the summed charge is exactly w; psc 2: the current over a step is the exact integral of the exponential, so the charge per spike is w tau exactly
      float itauE = N.psc == 2 ? N.tauE*(1.0f - decE) : N.psc == 1 ? (1.0f - decE) : 1.0f;
      float itauI = N.psc == 2 ? N.tauI*(1.0f - decI) : N.psc == 1 ? (1.0f - decI) : 1.0f;
      float decS = expf(-1.0f/N.tauS), decM = expf(-1.0f/N.tauM);
      float iAlpha = N.iRho*0.001f*(N.tauS + N.tauM);   // rho (tau+ + tau-)
      float stpScale = N.stpNorm > 0 ? 1.0f/(N.stpOrder ? N.stpU*(2.0f - N.stpU) : N.stpU) : 1.0f;
      if(N.watch >= 0 && steps > N.vtraceCap){
        if(N.vtrace) CK(cudaFree(N.vtrace));
        CK(cudaMalloc((void**)&N.vtrace, (size_t)steps*4));
        N.vtraceCap = steps;
      }
      unsigned long long before = 0;
      CK(cudaMemcpy(&before, N.total, 8, cudaMemcpyDeviceToHost));
      std::vector<unsigned int> c0(n);
      CK(cudaMemcpy(c0.data(), N.counts, n*4, cudaMemcpyDeviceToHost));
      // 255 is "did not fire in this tick"; the kernels only ever write a smaller value, and only the first time
      if(N.sendT && N.firstMs) CK(cudaMemset(N.firstMs, 255, n));
      for(unsigned int t = 0; t < steps; t++){
        int cur = N.cur;
        unsigned int par = N.t & 1u;         // which pool count half this step writes
        if(N.syn)
          stepExp<<<grid, block>>>(n, cur, decE, decI, itauE, itauI,
            N.a, N.b, N.c, N.d, N.bias, N.ext, N.namp, N.t, N.seed,
            N.v, N.u, N.ring, N.ring2, N.gE, N.gI,
            N.pre, N.post, N.w, N.delay, N.fired, N.counts, N.total,
            N.refCnt, N.refrac,
            N.stp > 0 ? N.stpX : nullptr, N.stpR, N.stpU, stpScale, N.stpOrder,
            N.sendT ? N.firstMs : nullptr, (int)t,
            N.poolG > 1 ? N.pool : nullptr, N.poolK, N.poolCnt, N.poolG, par, N.vmin,
            N.C7, N.k7, N.vr7, N.vt7, N.vp7, N.grd, N.gthr, N.gslp,
            N.KX, N.gX, N.ringX, N.decX, N.itauX, N.KX ? N.pmask : nullptr,
            N.syn == 2 ? 1 : 0, N.eE, N.eI, N.erevX);
        else
          stepKick<<<grid, block>>>(n, cur,
            N.a, N.b, N.c, N.d, N.bias, N.ext, N.namp, N.t, N.seed,
            N.v, N.u, N.ring,
            N.pre, N.post, N.w, N.delay, N.fired, N.counts, N.total,
            N.refCnt, N.refrac,
            N.stp > 0 ? N.stpX : nullptr, N.stpR, N.stpU, stpScale, N.stpOrder,
            N.sendT ? N.firstMs : nullptr, (int)t,
            N.poolG > 1 ? N.pool : nullptr, N.poolK, N.poolCnt, N.poolG, par, N.vmin,
            N.C7, N.k7, N.vr7, N.vt7, N.vp7, N.grd, N.gthr, N.gslp);
        // the half this step read is zeroed for the step after next
        if(N.poolG > 1)
          CK(cudaMemsetAsync(N.poolCnt + (size_t)(par ^ 1u)*N.poolG, 0, (size_t)N.poolG*4));
        N.t++;                              // simulated ms, the noise counter
        if(N.stp > 0 && N.stpX)
          stpStepK<<<grid, block>>>(n, N.stpX, N.stpR, N.stpU,
            expf(-1.0f/N.stpTauD), expf(-1.0f/N.stpTauF));
        if(N.watch >= 0 && N.vtrace && t < N.vtraceCap)
          watchK<<<1, 1>>>(N.v, N.vtrace, (int)t, N.watch);
        if(N.plast){
          // while calibrating, measure and change nothing: the traces and the spike accumulator still advance, the weights do not
          if(!N.calibrating){
          plastOut<<<grid, block>>>(n, N.fired, N.pre, N.post, N.w, N.pmask,
            N.Kpost, iAlpha, N.alphaI, N.wRef, N.commit);
          plastIn<<<grid, block>>>(n, N.fired, N.inStart, N.revSyn, N.revPre,
            N.w, N.pmask, N.Kpre, N.Kslow, N.wRef,
            N.tauY*(N.iRho > 0 ? N.iRho : 5.0f)*0.001f, N.commit);
          }
          traceStep<<<grid, block>>>(n, N.fired, N.Kpre, N.Kpost, decS, decM,
            N.Kslow, expf(-1.0f/N.tauY),
            (N.scaleOn > 0 || N.calibrating) ? N.spkAcc : nullptr);
          if(N.calibrating){
            N.calAcc += 1.0f;
            if(N.calAcc >= N.calS*1000.0f){
              finishCalK<<<grid, block>>>(n, N.spkAcc, N.rhoI, N.alphaI,
                N.calS, N.tauS + N.tauM);
              CK(cudaMemsetAsync(N.spkAcc, 0, (size_t)n*4));
              N.calibrating = 0;
            }
          }
          // multiplicative scaling toward the target rate, once per simulated second, matching the reference cadence
          if(N.scaleOn > 0 && N.spkAcc && !N.calibrating){
            N.scaleAcc += 1.0f;
            if(N.scaleAcc >= 1000.0f){
              N.scaleAcc = 0;
              scaleK<<<grid, block>>>(n, N.inStart, N.revSyn, N.w, N.pmask,
                N.spkAcc, N.iRho > 0 ? N.iRho : 5.0f, N.sEta, N.rhoI, N.wRef, N.commit);
            }
          }
          // the reference weight follows the weight through a double well, on the published cadence rather than every step
          if(N.anyCons > 0 && N.wRef){
            N.consAcc += 1.0f;
            if(N.consAcc >= N.consStep){
              N.consAcc = 0;
              int mb = 256, mg = (int)((N.m + mb - 1)/mb);
              consolidateK<<<mg, mb>>>(N.m, N.w, N.wRef, N.pmask,
                N.consStep/N.tauCons);
            }
          }
        }
        CK(cudaMemsetAsync(N.ring + (size_t)cur*n, 0, n*4));
        if(N.syn) CK(cudaMemsetAsync(N.ring2 + (size_t)cur*n, 0, n*4));
        for(int x = 0; x < N.KX; x++) CK(cudaMemsetAsync(N.ringX + ((size_t)x*ROWS + cur)*n, 0, n*4));
        N.cur = (N.cur + 1) % ROWS;
      }
      std::vector<unsigned int> c1(n);
      CK(cudaMemcpy(c1.data(), N.counts, n*4, cudaMemcpyDeviceToHost));
      unsigned long long after = 0;
      CK(cudaMemcpy(&after, N.total, 8, cudaMemcpyDeviceToHost));
      // fired carries the count of spikes in the tick, not a flag: a flag would cap every rate this readout can report at 1000/steps Hz.
      // Non-zero still means fired.
      for(unsigned int i = 0; i < n; i++){
        unsigned int dc = c1[i] - c0[i];
        hostFired[i] = (unsigned char)(dc > 255u ? 255u : dc);
      }
      unsigned long long spikes = after - before;
      const unsigned int hasV = (N.watch >= 0 && N.vtrace) ? 1u : 0u;
      const size_t traceBytes = hasV ? (size_t)steps*4 : 0;
      // The whole membrane array, when the viewer's potential filter is on.
      // No header field says so: the frame is longer by exactly n floats and the reader knows n, which is the same way GETW carries the measured set points.
      const size_t vBytes = N.sendV ? (size_t)n*4 : 0;
      const size_t tBytes = (N.sendT && N.firstMs) ? (size_t)n : 0;
      std::vector<char> out(16 + n + traceBytes + vBytes + tBytes);
      memcpy(out.data(), &steps, 4);
      memcpy(out.data()+4, &hasV, 4);
      memcpy(out.data()+8, &spikes, 8);
      memcpy(out.data()+16, hostFired.data(), n);
      if(hasV) CK(cudaMemcpy(out.data() + 16 + n, N.vtrace,
        traceBytes, cudaMemcpyDeviceToHost));
      if(vBytes) CK(cudaMemcpy(out.data() + 16 + n + traceBytes, N.v,
        vBytes, cudaMemcpyDeviceToHost));
      if(tBytes) CK(cudaMemcpy(out.data() + 16 + n + traceBytes + vBytes,
        N.firstMs, tBytes, cudaMemcpyDeviceToHost));
      reply(100, out.data(), (unsigned int)out.size());
    }
    else if(cmd == 11){                    // HELLO: the contract this build implements, so a stale child is caught before init
      const unsigned int proto = ENGINE_PROTOCOL;
      reply(103, &proto, 4);
    }
    else if(cmd == 6){                     // GETW
      const unsigned int extra = N.rhoI ? N.n : 0;
      std::vector<float> wf((size_t)N.m + extra);
      CK(cudaMemcpy(wf.data(), N.w, (size_t)N.m*4, cudaMemcpyDeviceToHost));
      // measured set points ride in the tail, so a reader expecting only m floats is unaffected and one that knows about them can tell by length.
      // Without them an unreachable homeostatic target could only be inferred after the fact, which is why the other engines send them.
      if(extra)
        CK(cudaMemcpy(wf.data() + N.m, N.rhoI, (size_t)extra*4, cudaMemcpyDeviceToHost));
      reply(101, wf.data(), ((size_t)N.m + extra)*4);
    }
  }
  return 0;
}

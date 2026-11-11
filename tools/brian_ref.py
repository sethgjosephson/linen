"""A second implementation of the reference engine's step, run on Brian 2.

The reference engine (src/simworker.js) defines what linen computes. This
transcribes its step into Brian 2 code and runs a network written by
tools/brianref.mjs on Brian's spike queues, synaptic delays, synapse
indexing and scheduling, so the two sets of spike trains and learned weights
can be compared exactly. Any difference is then a difference between the
engine's code and this reading of it (when a delivery arrives, which ring a
synapse feeds, how a conductance decays, the order of reset and recovery,
the refractory count, the floor, the noise draw, a stimulus protocol, an
input frame, a pool, a graded release, a plasticity term), not rounding.
Because the arithmetic is transcribed from the engine, agreement shows the
engine does what its own rules say, not that the rules match the literature.

Matching the engine to the bit takes five things:

- The step is the engine's, not Brian's integrator: `step` and `stepExp` in
  simworker.js, the membrane in MODEL.md section 1, the synapses in section
  2, plasticity in the sections after.
- Numbers are stored as float32, as the engine keeps them (parameters, bias,
  v, u, conductances, rings, stimulus and external drive, pool drive,
  weights, traces, short-term state), with double arithmetic inside a step.
  `fround` rounds each stored value the way a Float32Array write does.
- Every sum is taken two terms at a time in the engine's order, and every
  constant the engine derives with Math.exp arrives computed by the engine's
  own Math.exp, since a float sum depends on its order and two libraries' exp
  can differ in the last bit.
- A spike at step t with delay d reaches the target's input at step t + d.
  Brian delivers in each step's synapse slot, after the neurons have updated,
  so that is a Brian delay of d - 1 ms. Under plasticity or short-term
  plasticity the engine delivers the weight as it was at emission, so the
  emitting pathway stashes it and a second pathway delivers the stash; that
  is exact while every presynaptic interval covers the longest delay, which
  brianref.mjs checks. A graded cell releases every millisecond, so it keeps
  a short history of its release and each of its synapses reads the entry
  its delay names.
- The engine applies plasticity one spiking neuron at a time in index order.
  When a synapse's pre and post spike in the same millisecond the lower index
  goes first, so masks on `i > j` reproduce which trace already holds this
  step's spike and which weight is delivered.

The case may carry `dt` (milliseconds, 1 by default). Below one millisecond
the same network is integrated on a finer clock: the membrane half steps and
the recovery follow dt, the synaptic decay factors are raised to dt, a delay
of d milliseconds still delivers d milliseconds after the spike, and the
per-millisecond draws (noise, the stimulus protocols) hold across the
substeps of their millisecond. Delays stay whole milliseconds, as the engine
quantizes them, so the difference between a run at 1 ms and one at 0.1 ms is
the integration step alone. Plasticity, short-term plasticity, graded rows,
input frames and pools all carry per-millisecond bookkeeping and are refused
below 1 ms.

Case file (JSON): n, ms, rows (per neuron: C, k, vr, vt, vpeak, a, b, c, d),
bias, v0, u0, refrac (ms), vmin (mV, or null for no floor), syn (0 kick,
1 exponential, 2 conductance) with decE, decI, itauE, itauI, eRevE, eRevI,
chanDec, chanItau and chanErev (one per receptor channel beyond E and I);
stim and noiseProtocols (the DC and noise-amplitude protocols in the order
the engine keeps them: t0, duration, mode, period, width, amp and f, the
per-neuron factor) and seed; optionally pool and poolK, inputs (framed input
maps: chStart, chIdx, chW, amp) with frames (t, mi, gains), and grd, gthr,
gslp for graded rows; optionally the synapses as the engine takes them: pre
(CSR row starts, n + 1), post, w, delay (ms, 1 to 17), chan and rule (the
two fields of each synapse's plasticity byte); optionally plast (the rule
table and the constants configure derives) and stp (short-term plasticity
constants). noiseProbe, a list of [i, t, seed], asks for those noise draws.
Writes spike times per neuron, and the final weights and measured set points
under plast or stp.

    .venv-analysis/Scripts/python.exe tools/brian_ref.py case.json out.json
"""
import json
import sys
import time

import numpy as np
import brian2
from brian2 import (NeuronGroup, Synapses, SpikeMonitor, Function, Network,
                    network_operation, defaultclock, ms, prefs)

prefs.codegen.target = 'numpy'
prefs.logging.console_log_level = 'ERROR'

# a Float32Array write: round to float32, keep computing in doubles
fround = Function(
    lambda x: np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64),
    arg_units=[1], return_unit=1)

# rand.js pcg and noiseDraw, in unsigned 64-bit integers masked to 32 bits; every product stays below 2^63, so nothing wraps before the mask
MASK = np.uint64(0xFFFFFFFF)


def pcg(x):
    x = np.asarray(x, dtype=np.uint64) & MASK
    st = (x*np.uint64(747796405) + np.uint64(2891336453)) & MASK
    word = (((st >> ((st >> np.uint64(28)) + np.uint64(4))) ^ st)*np.uint64(277803737)) & MASK
    return ((word >> np.uint64(22)) ^ word) & MASK


# the draw a noise stimulus shares among its cells is keyed on the stimulus's index above this base (src/rand.js NOISE_SHARED)
NOISE_SHARED = 0x40000000


def noise_draw(i, t, seed):
    i = np.asarray(i, dtype=np.uint64)
    t = np.asarray(t, dtype=np.uint64)
    h = (i ^ ((t*np.uint64(2654435761)) & MASK) ^ np.uint64(seed)) & MASK
    a = pcg(h).astype(np.float64)/4294967295.0
    b = pcg(h ^ np.uint64(0x9e3779b9)).astype(np.float64)/4294967295.0
    return (a + b) - 1.0


# rand.js poissonDraw: the uniform a Poisson row fires on, under its own salt
def poisson_draw(i, t, seed):
    i = np.asarray(i, dtype=np.uint64)
    t = np.asarray(t, dtype=np.uint64)
    h = (i ^ ((t*np.uint64(2654435761)) & MASK) ^ np.uint64(seed) ^ np.uint64(0x7f4a7c15)) & MASK
    return pcg(h).astype(np.float64)/4294967295.0


def poisson_function(seed):
    def draw(i, tms):
        steps = np.floor(np.asarray(tms, dtype=np.float64) + 1e-6).astype(np.uint64)
        return poisson_draw(i, steps, seed)
    return Function(draw, arg_units=[1, 1], return_unit=1,
                    arg_types=['integer', 'float'], return_type='float')


def noise_function(seed):
    def draw(i, tms):
        # the millisecond a step falls in: the engine draws once per millisecond, and a substep holds the draw of the millisecond it is in
        steps = np.floor(np.asarray(tms, dtype=np.float64) + 1e-6).astype(np.uint64)
        return noise_draw(i, steps, seed)
    return Function(draw, arg_units=[1, 1], return_unit=1,
                    arg_types=['integer', 'float'], return_type='float')


# simworker.js scaleOf: 0 before t0 and from t0 + duration, then a pulse train, a ramp over the period, or 1
def _pscale(tms, t0, dur, mode, per, wid):
    dt = np.floor(np.asarray(tms, dtype=np.float64) + 1e-6) - np.asarray(t0, dtype=np.float64)
    dur = np.asarray(dur, dtype=np.float64)
    mode = np.asarray(mode, dtype=np.float64)
    per = np.asarray(per, dtype=np.float64)
    wid = np.asarray(wid, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        pulse = np.where(np.fmod(dt, per) < wid, 1.0, 0.0)
        ramp = np.minimum(1.0, dt/per)
    s = np.where(mode == 1, pulse, np.where(mode == 2, ramp, 1.0))
    s = np.where((dur > 0) & (dt >= dur), 0.0, s)
    return np.where(dt < 0, 0.0, s)


pscale = Function(_pscale, arg_units=[1]*6, return_unit=1, arg_types=['float']*6, return_type='float')


# Gates on the step index k (the step that runs at t = k ms).
# The engine does its periodic work after step k, when its clock has become k + 1.
def _k(x):
    return np.rint(np.asarray(x, dtype=np.float64)).astype(np.int64)


every = Function(lambda tms, period: np.asarray((_k(tms) + 1) % _k(period) == 0, dtype=np.float64),
                 arg_units=[1, 1], return_unit=1, arg_types=['float', 'float'], return_type='float')
upto = Function(lambda tms, last: np.asarray(_k(tms) + 1 <= _k(last), dtype=np.float64),
                arg_units=[1, 1], return_unit=1, arg_types=['float', 'float'], return_type='float')
atstep = Function(lambda tms, s: np.asarray(_k(tms) + 1 == _k(s), dtype=np.float64),
                  arg_units=[1, 1], return_unit=1, arg_types=['float', 'float'], return_type='float')
# every `period` ms counted from `start`, after it: scaling's passes, whose window opens when scaling turns on or when the set point calibration ends (simworker.js nextScale, as in the CUDA and WebGPU engines)
every_from = Function(lambda tms, period, start: np.asarray(
    ((_k(tms) + 1) > _k(start)) & (((_k(tms) + 1) - _k(start)) % _k(period) == 0), dtype=np.float64),
    arg_units=[1, 1, 1], return_unit=1, arg_types=['float', 'float', 'float'], return_type='float')

# the rule table's fields in protocol order (nodes.js RULE_FIELDS), as per-synapse constants
RULE_VARS = ['raP', 'raM', 'rwmax', 'rwdep', 'rtrip', 'rhet', 'rtin', 'rieta', 'rcons', 'rconsw', 'rconsp']


def fold(prefix, count):
    """updateAdd: each protocol adds amp times its scale times the cell's factor,
    stored float32, protocol by protocol."""
    L = ['%sa0 = 0' % prefix]
    for k in range(count):
        L += ['%sk%d = %samp%d*pscale(t/ms, %st0%d, %sdur%d, %smode%d, %sper%d, %swid%d)'
              % (prefix, k, prefix, k, prefix, k, prefix, k, prefix, k, prefix, k, prefix, k),
              '%ss%d = %sk%d*%sf%d' % (prefix, k, prefix, k, prefix, k),
              '%sa%d = fround(%sa%d + %ss%d)' % (prefix, k + 1, prefix, k, prefix, k)]
    return L, '%sa%d' % (prefix, count)


# The equations' membrane step (integ 2, MODEL.md 1), the reference engine's quadTime, quadAt and quadMs without branches.
# Every case is computed and the one that holds is picked by its mask, so each case's arguments are kept finite where it does not hold (a NaN would survive the mask).
def quad_time(p, A, q, w0, wp):
    """The time from w0 to wp on w' = A w^2 + q, 1e30 if never, as p_t."""
    return ['%s_y = %s - %s' % (p, wp, w0),
            '%s_ys = %s_y*int(%s_y > 0) + int(%s_y <= 0)' % (p, p, p, p),
            '%s_x = %s + %s*%s*%s' % (p, q, A, w0, wp),
            '%s_xp = int(%s_x > 0)' % (p, p),
            '%s_xs = %s_x*%s_xp + 1 - %s_xp' % (p, p, p, p),
            '%s_z = %s*%s' % (p, A, q),
            '%s_zp = int(%s_z > 0)' % (p, p),
            '%s_s = sqrt(abs(%s_z)) + 1e-300' % (p, p),
            '%s_R = clip(%s_ys/%s_xs, 0, 1e30)' % (p, p, p),
            '%s_e = %s_z*%s_R*%s_R' % (p, p, p, p),
            '%s_sm = int(abs(%s_e) < 1e-4)' % (p, p),
            '%s_ser = %s_R*(1 - %s_e/3 + %s_e*%s_e/5)' % (p, p, p, p, p),
            '%s_ta = arctan(%s_s*%s_R)/%s_s' % (p, p, p, p),
            '%s_r = clip(%s_s*%s_R, 0, 1 - 1e-15)' % (p, p, p),
            '%s_th = 0.5*log((1 + %s_r)/(1 - %s_r))/%s_s' % (p, p, p, p),
            '%s_thv = %s_th*int(%s_s*%s_R < 1) + 1e30*int(%s_s*%s_R >= 1)' % (p, p, p, p, p, p),
            '%s_nb = (1.5707963267948966 + arctan(-%s_x/(%s_s*%s_ys)))/%s_s' % (p, p, p, p, p),
            '%s_tp = %s_sm*%s_ser + (1 - %s_sm)*(%s_zp*%s_ta + (1 - %s_zp)*%s_thv)' % (p, p, p, p, p, p, p, p),
            '%s_tn = %s_zp*%s_nb + (1 - %s_zp)*1e30' % (p, p, p, p),
            '%s_t = int(%s < %s)*(%s_xp*%s_tp + (1 - %s_xp)*%s_tn)' % (p, w0, wp, p, p, p, p)]


def quad_at(p, A, q, w0, h):
    """w after h (h within [0, 0.5 dt]) as p_w, and the integral of w over the span as p_iw."""
    return ['%s_z = %s*%s' % (p, A, q),
            '%s_e = %s_z*%s*%s' % (p, p, h, h),
            '%s_sm = int(abs(%s_e) < 1e-4)' % (p, p),
            '%s_zp = int(%s_z > 0)' % (p, p),
            '%s_s = sqrt(abs(%s_z)) + 1e-300' % (p, p),
            '%s_c1 = 1 - %s_e/2 + %s_e*%s_e/24' % (p, p, p, p),
            '%s_s1 = %s*(1 - %s_e/6 + %s_e*%s_e/120)' % (p, h, p, p, p),
            '%s_c2 = cos(%s_s*%s)' % (p, p, h),
            '%s_s2 = sin(%s_s*%s)/%s_s' % (p, p, h, p),
            '%s_E = exp(-2*%s_s*%s)' % (p, p, h),
            '%s_c3 = 0.5*(1 + %s_E)' % (p, p),
            '%s_s3 = 0.5*(1 - %s_E)/%s_s' % (p, p, p),
            '%s_Cs = %s_sm*%s_c1 + (1 - %s_sm)*(%s_zp*%s_c2 + (1 - %s_zp)*%s_c3)' % (p, p, p, p, p, p, p, p),
            '%s_Sn = %s_sm*%s_s1 + (1 - %s_sm)*(%s_zp*%s_s2 + (1 - %s_zp)*%s_s3)' % (p, p, p, p, p, p, p, p),
            '%s_ls = (1 - %s_sm)*(1 - %s_zp)*%s_s*%s' % (p, p, p, p, h),
            '%s_phi = %s_Cs - %s*%s*%s_Sn' % (p, p, A, w0, p),
            '%s_ph = %s_phi + int(%s_phi == 0)*1e-300' % (p, p, p),
            '%s_w = (%s*%s_Cs + %s*%s_Sn)/%s_ph' % (p, w0, p, q, p, p),
            '%s_iw = -(%s_ls + log(abs(%s_ph)))/%s' % (p, p, p, A)]


def quad_coef(p, u, I, Gs):
    """The vertex p_m and the offset p_q of the parabola with u held, the adaptation conductance and the synaptic conductance folded into the slope."""
    return ['%s_g = adC*%s + %s' % (p, u, Gs),
            '%s_m = 0.5*(vr + vt) + %s_g/(2*kk)' % (p, p),
            '%s_q = (-kk*(0.5*(vt - vr))*(0.5*(vt - vr)) + %s + adC*%s*eRevAd - (1 - adC)*%s + %s*v - 0.5*%s_g*(vr + vt) - %s_g*%s_g/(4*kk))/Cm' % (p, I, u, u, Gs, p, p, p)]


def quad_ms(I, Gs, dt):
    """One millisecond on a quadratic row from v and u: vqe, uqe, whether it fired (spq), the threshold after (thq)."""
    h = repr(0.5*dt)
    L = ['qA = kk/Cm', 'qea = exp(-%s*aa)' % h, 'qv0 = vpeak + thr']
    vin, uin, vth, thin, fired = 'v', 'u', 'qv0', 'thr', None
    for hs in (1, 2):
        P = 'h%d' % hs
        L += quad_coef(P, uin, I, Gs)
        L += quad_time(P + 'a', 'qA', P + '_q', '(%s - %s_m)' % (vin, P), '(%s - %s_m)' % (vth, P))
        L += ['%s_c = int(%sa_t <= %s)' % (P, P, h)]
        # no crossing: the whole half step
        L += quad_at(P + 'b', 'qA', P + '_q', '(%s - %s_m)' % (vin, P), h)
        L += ['%s_vA = %s_m + %sb_w' % (P, P, P), '%s_iA = %sb_iw + %s*%s_m' % (P, P, h, P)]
        # up to the crossing
        L += ['%s_tc = clip(%sa_t, 0, %s)' % (P, P, h), '%s_lf = %s - %s_tc' % (P, h, P)]
        L += quad_at(P + 'c', 'qA', P + '_q', '(%s - %s_m)' % (vin, P), P + '_tc')
        L += ['%s_iB = %sc_iw + %s_tc*%s_m' % (P, P, P, P)]
        # the reset at the crossing and the rest of the half step from c, held at the threshold if it crosses again
        L += ['%s_uR = %s + dd' % (P, uin), '%s_vth = vpeak + thrJ' % P,
              '%s_thn = thrJ*thrDec**((%s*%d + %s_lf)/%s)' % (P, h, 2 - hs, P, repr(float(dt)))]
        L += quad_coef(P + 'r', P + '_uR', I, Gs)
        L += quad_time(P + 'd', 'qA', P + 'r_q', '(cc - %sr_m)' % P, '(%s_vth - %sr_m)' % (P, P))
        L += ['%s_cr = int(%sd_t <= %s_lf)' % (P, P, P), '%s_t2 = clip(%sd_t, 0, %s_lf)' % (P, P, P)]
        L += quad_at(P + 'e', 'qA', P + 'r_q', '(cc - %sr_m)' % P, P + '_t2')
        L += quad_at(P + 'f', 'qA', P + 'r_q', '(cc - %sr_m)' % P, P + '_lf')
        L += ['%s_vR = %s_cr*%s_vth + (1 - %s_cr)*(%sr_m + %sf_w)' % (P, P, P, P, P, P),
              '%s_iR = %s_iB + %s_cr*(%se_iw + %s_t2*%sr_m + (%s_lf - %s_t2)*%s_vth) + (1 - %s_cr)*(%sf_iw + %s_lf*%sr_m)' % ((P,)*13)]
        # a crossing after the cell has already fired this millisecond holds it at the threshold
        L += ['%s_iH = %s_iB + %s_lf*%s' % (P, P, P, vth)]
        if fired is None:
            L += ['%s_v1 = %s_c*%s_vR + (1 - %s_c)*%s_vA' % (P, P, P, P, P),
                  '%s_iv = %s_c*%s_iR + (1 - %s_c)*%s_iA' % (P, P, P, P, P),
                  '%s_u1 = %s_c*%s_uR + (1 - %s_c)*%s' % (P, P, P, P, uin),
                  '%s_vt1 = %s_c*%s_vth + (1 - %s_c)*%s' % (P, P, P, P, vth),
                  '%s_th1 = %s_c*%s_thn + (1 - %s_c)*%s' % (P, P, P, P, thin),
                  '%s_f = %s_c' % (P, P)]
        else:
            # cr: a first crossing (reset), hd: a second one (held)
            L += ['%s_k1 = %s_c*(1 - %s)' % (P, P, fired), '%s_k2 = %s_c*%s' % (P, P, fired),
                  '%s_v1 = %s_k1*%s_vR + %s_k2*%s + (1 - %s_c)*%s_vA' % (P, P, P, P, vth, P, P),
                  '%s_iv = %s_k1*%s_iR + %s_k2*%s_iH + (1 - %s_c)*%s_iA' % (P, P, P, P, P, P, P),
                  '%s_u1 = %s_k1*%s_uR + (1 - %s_k1)*%s' % (P, P, P, P, uin),
                  '%s_vt1 = %s_k1*%s_vth + (1 - %s_k1)*%s' % (P, P, P, P, vth),
                  '%s_th1 = %s_k1*%s_thn + (1 - %s_k1)*%s' % (P, P, P, P, thin),
                  '%s_f = %s + %s_k1' % (P, fired, P)]
        L += ['%s_vf = clip(%s_v1, vmin, 1e300)' % (P, P),
              '%s_tg = bb*(%s_iv/%s - vr)' % (P, P, h),
              '%s_uf = %s_tg + (%s_u1 - %s_tg)*qea' % (P, P, P, P)]
        vin, uin, vth, thin, fired = P + '_vf', P + '_uf', P + '_vt1', P + '_th1', P + '_f'
    L += ['vqe = h2_vf', 'uqe = h2_uf', 'spq = h2_f', 'thq = h2_th1']
    return L



def step_code(syn, K, nstim=0, nnoise=0, ext=False, pool=False, graded=0, stp=False, dt=1.0, chan_mg=(), colored=False, integ=0, mod=False):
    """The engine's step, one assignment per quantity, sums two terms at a time.
    `graded` is the longest delay of a graded synapse, or 0. `dt` is the step in
    milliseconds: at 1 every line below is the engine's own, and under it the
    membrane, the recovery and the refractory count follow the smaller step.
    `mod`: a modulator field runs, so the synaptic sum (the channels included) is
    formed before the drive and scaled by 1 + modDrive M with modCur M added, as
    the engine forms it under a field (MODEL.md 4)."""
    L = []
    half = '0.5' if dt == 1.0 else repr(0.5*dt)
    # the dynamic threshold (MODEL.md 1) relaxes before anything else this step, in float32 as the engines store it
    L.append('thr = fround(thr*thrDec)')
    per = '' if dt == 1.0 else repr(dt) + '*'
    if mod:
        L += ['mf = 1 + modDrive*mlev', 'mc = modCur*mlev']
    syn_name = 'Isyn'
    if syn == 0:
        L.append('Isyn = ring + Igr' if graded else 'Isyn = ring')
        if mod and pool:
            L.append('Isyp = Isyn + pin')
            syn_name = 'Isyp'
    else:
        L.append('ge = gE*decE + ringE')
        L.append('gi0 = gI*decI + ringI')
        L.append('gi = gi0 + pin' if pool else 'gi = gi0')
        if syn == 2:
            L += ['s1 = abs(ge)*(eRevE - v)', 's2 = abs(gi)*(eRevI - v)']
        else:
            L += ['s1 = ge*itauE', 's2 = gi*itauI']
        L.append('Isyn = s1 + s2')
    def channel_lines(prev, name):
        # the receptor channels' currents, each through its magnesium block where it has one (Jahr and Stevens 1990), added to prev one at a time as name0, name1, ...
        out = []
        for k in range(K):
            out.append('gx%d = gX%d*chanDec%d + ringX%d' % (k, k, k, k))
            if syn == 2:
                out.append('c%d = abs(gx%d)*(chanErev%d - v)' % (k, k, k))
            else:
                out.append('c%d = gx%d*chanItau%d' % (k, k, k))
            if k < len(chan_mg) and float(chan_mg[k]) > 0:
                out.append('b%d = 1/(1 + (chanMg%d/3.57)*exp(-0.062*v))' % (k, k))
                out.append('cb%d = b%d*c%d' % (k, k, k))
                out.append('%s%d = %s + cb%d' % (name, k + 1, prev, k))
            else:
                out.append('%s%d = %s + c%d' % (name, k + 1, prev, k))
            prev = '%s%d' % (name, k + 1)
        return out, prev
    if mod:
        lines, synK = channel_lines(syn_name, 'Ic')
        L += lines + ['Ism0 = %s*mf' % synK, 'Ism = Ism0 + mc', 'Ib = Ism + bias']
    else:
        # the engine's order: synaptic, bias, stimulus, external, noise, pool
        L.append('Ib = Isyn + bias')
    cur = 'Ib'
    if nstim:
        lines, stim = fold('prq', nstim)
        L += lines + ['Is = %s + %s' % (cur, stim)]
        cur = 'Is'
    if ext:
        L.append('Ie = %s + ext' % cur)
        cur = 'Ie'
    if nnoise:
        lines, amp = fold('prz', nnoise)
        if colored:
            # a stimulus with a time constant filters the cell's draw (an Ornstein-Uhlenbeck process, the states nfou and nfous kept in f32 as the engines keep them); a shared fraction mixes in the stimulus's draw, filtered the same way (MODEL.md 2)
            L += lines + ['nfd0 = noise(i, t/ms)',
                          'nfou = fround(nfa*nfou + nfk*nfd0)*int(nfa > 0) + nfou*int(nfa <= 0)',
                          'nfd1 = nfou*int(nfa > 0) + nfd0*int(nfa <= 0)',
                          'nfs0 = noise(nfid, t/ms)',
                          'nfous = fround(nfa*nfous + nfk*nfs0)*int(nfa > 0) + nfous*int(nfa <= 0)',
                          'nfs1 = nfous*int(nfa > 0) + nfs0*int(nfa <= 0)',
                          'nfd2 = (sqrt(1 - nfc)*nfd1 + sqrt(nfc)*nfs1)*int(nfc > 0) + nfd1*int(nfc <= 0)',
                          'nz = %s*nfd2' % amp, 'In = %s + nz' % cur]
        else:
            L += lines + ['nz = %s*noise(i, t/ms)' % amp, 'In = %s + nz' % cur]
        cur = 'In'
    if pool and syn == 0 and not mod:
        L.append('Ip = %s + pin' % cur)
        cur = 'Ip'
    L.append('I0 = %s' % cur)
    if mod:
        I = 'I0'
    else:
        lines, I = channel_lines('I0', 'I')
        L += lines
    L.append('act = int(refc <= 0)' if dt == 1.0 else 'act = int(refc <= 1e-9)')
    if integ:
        # the exponential membrane step (MODEL.md 1): each half step is h F(v)/C times (e^x - 1)/x at x = h F'(v)/C, F carrying the synaptic conductance G through G (v - v1)
        if syn == 2:
            g = ['abs(ge)', 'abs(gi)'] + ['%sabs(gx%d)' % ('b%d*' % k if k < len(chan_mg) and float(chan_mg[k]) > 0 else '', k) for k in range(K)]
            L.append('Gs0 = ' + g[0])
            for q in range(1, len(g)):
                L.append('Gs%d = Gs%d + %s' % (q, q - 1, g[q]))
            Gs = 'Gs%d' % (len(g) - 1)
        else:
            L.append('Gs0 = 0'); Gs = 'Gs0'
        if mod:
            # the field scales the synaptic conductance with the current
            L.append('Gsm = %s*mf' % Gs); Gs = 'Gsm'
        for a, b0, n in (('v', 'v', 1), ('v1', 'v', 2)):
            L += ['uq%d = adC*u*(%s - eRevAd) + (1 - adC)*u' % (n - 1, a),
                  'F%d = kk*(%s - vr)*(%s - vt) - uq%d + %s + %s*(v - %s)' % (n, a, a, n - 1, I, Gs, a),
                  'x%d = %s*(kk*(2*%s - vr - vt) - adC*u - %s)/Cm' % (n, half, a, Gs),
                  'xs%d = x%d + int(abs(x%d) < 1e-9)*1e-9' % (n, n, n),
                  'ph%d = (exp(xs%d) - 1)/xs%d' % (n, n, n),
                  'v%s = clip(%s + %s*F%d/Cm*ph%d, vmin, 1e300)' % ('1' if n == 1 else '2x', a, half, n, n)]
    eul = [
        # with the row's adaptation a conductance (adC), u in nanosiemens draws current toward the inhibitory reversal
        'uq0 = adC*u*(v - eRevAd) + (1 - adC)*u',
        'v1 = clip(v + %s*(kk*(v - vr)*(v - vt) - uq0 + %s)/Cm, vmin, 1e300)' % (half, I),
        'uq1 = adC*u*(v1 - eRevAd) + (1 - adC)*u',
        'v2 = clip(v1 + %s*(kk*(v1 - vr)*(v1 - vt) - uq1 + %s)/Cm, vmin, 1e300)' % (half, I)]
    if integ == 2:
        # the exponential step on a row linear over the spiking range, the equations' step on the others
        L += quad_ms(I, Gs, dt)
        L += ['lin = int(vt - vpeak > 100)', 'v2 = lin*v2x + (1 - lin)*vqe']
        # qm: the cells the equations' step ran (quadratic, not Poisson, not graded), whose reset it has already made
        L.append('qm = (1 - lin)*(1 - poi)' + ('*(1 - gd)' if graded else ''))
    elif integ:
        # the exponential step on a row linear over the spiking range (vt more than 100 mV above vpeak), the Euler half steps on the others
        L += [x.replace('v1', 've1').replace('v2', 've2').replace('uq0', 'ue0').replace('uq1', 'ue1') for x in eul]
        L += ['lin = int(vt - vpeak > 100)', 'v2 = lin*v2x + (1 - lin)*ve2']
    else:
        L += eul
    L += [
        # a Poisson cell (MODEL.md 1): no membrane; it fires this step with probability I over 1000, the current read as a rate in hertz, by being put at its peak
        'pdr = poidraw(i, t/ms)',
        'psp = poi*int(%s > 0)*int(pdr < %s*0.001)' % (I, I),
        'vpo = psp*(vpeak + thr) + (1 - psp)*v',
        'vq2 = poi*vpo + (1 - poi)*v2',
    ]
    if graded:
        # a graded cell's membrane is passive, capped at vpeak, and never spikes
        L += ['gl = kk*(vt - vr)',
              'gw1 = clip(v + %s*(-gl*(v - vr) - u + %s)/Cm, vmin, 1e300)' % (half, I),
              'gw2 = clip(gw1 + %s*(-gl*(gw1 - vr) - u + %s)/Cm, vmin, 1e300)' % (half, I),
              'gw3 = clip(gw2, -1e300, vpeak)',
              'vq = act*vq2 + (1 - act)*v',
              'vn = gd*gw3 + (1 - gd)*vq',
              'un = u + %saa*(bb*(vn - vr) - u)' % per,
              'sp = act*int(vn >= vpeak + thr)*(1 - gd)']
    else:
        L += ['vn = act*vq2 + (1 - act)*v',
              'un = u + %saa*(bb*(vn - vr) - u)' % per,
              'sp = act*int(vn >= vpeak + thr)']
    if integ == 2:
        # the equations' step has stepped u and decided the spike itself
        L += ['qa = qm*act', 'un = qa*uqe + (1 - qa)*un', 'sp = qa*spq + (1 - qm)*sp']
    if integ:
        # the exponential step's reset inside the step: the distance to the threshold interpolated between the opening and the end (the threshold at both ends), the rest of the step integrated from the reset with the jump in u applied, the threshold relaxing over it
        L += ['thS = thr/thrDec*int(thrDec > 0) + thr*int(thrDec <= 0)',
              'dS = v - vpeak - thS', 'dE = vn - vpeak - thr',
              'fq = dS/(dS - dE + int(dE <= dS))',
              'fr = clip(fq, 0, 1)*int(dS < 0)*int(dE > dS) + 1 - int(dS < 0)*int(dE > dS)',
              'rst = %s*(1 - fr)' % repr(float(dt)),
              'ur = un + dd',
              'uqr = adC*ur*(cc - eRevAd) + (1 - adC)*ur',
              'Fr = kk*(cc - vr)*(cc - vt) - uqr + %s + %s*(v - cc)' % (I, Gs),
              'xr = rst*(kk*(2*cc - vr - vt) - adC*ur - %s)/Cm' % Gs,
              'xrs = xr + int(abs(xr) < 1e-9)*1e-9',
              'phr = (exp(xrs) - 1)/xrs',
              'vre = clip(cc + rst*Fr/Cm*phr, vmin, 1e300)',
              'pl = (1 - poi)*lin',
              'vrs = (1 - pl)*cc + pl*vre',
              'thr = sp*((1 - pl)*thrJ + pl*thrJ*thrDec**(rst/%s)) + (1 - sp)*thr' % repr(float(dt))]
        if integ == 2:
            L[-1] = 'thr = sp*(qm*thq + (1 - qm)*((1 - pl)*thrJ + pl*thrJ*thrDec**(rst/%s))) + (1 - sp)*thr' % repr(float(dt))
            L += ['vrs = qm*vn + (1 - qm)*vrs']
        L += ['v = fround(sp*vrs + (1 - sp)*vn)']
    else:
        # the threshold jumps at the spike (the row's thrJ), as Auryn sets it rather than adds to it
        L.append('thr = sp*thrJ + (1 - sp)*thr')
        L += ['v = fround(sp*cc + (1 - sp)*vn)']
    L += ['u = fround(un + sp*dd*(1 - qm))' if integ == 2 else 'u = fround(un + sp*dd)',
          'refc = sp*refrac + (1 - sp)*(refc - %sint(refc > 0))' % per,
          'spiked = sp']
    if graded:
        # release this millisecond, pushed onto the history the synapses read
        L += ['rq1 = vn - gthr', 'rq2 = rq1/gslp', 'rgr = clip(rq2, 0, 1)']
        L += ['rh%d = rh%d' % (d, d - 1) for d in range(graded, 1, -1)]
        L.append('rh1 = gd*rgr')
    if stp and stp.get('classes'):
        # per-rule short-term plasticity: each class that is on releases from its own state, the release a double as in the single-class path
        for c, cl in enumerate(stp['classes']):
            if not cl['on']:
                continue
            x, r, f = 'stpX%d' % c, 'stpR%d' % c, 'sfv%d' % c
            if stp.get('order', 0):
                L += ['sr1 = 1 - %s' % r, 'sr2 = stpU%d*sr1' % c, 'sR = fround(%s + sr2)' % r,
                      'rel = sR*%s' % x, 'sX = fround(%s - rel)' % x]
            else:
                L += ['rel = %s*%s' % (r, x), 'sX = fround(%s - rel)' % x,
                      'sr1 = 1 - %s' % r, 'sr2 = stpU%d*sr1' % c, 'sR = fround(%s + sr2)' % r]
            L += ['%s = sp*sR + (1 - sp)*%s' % (r, r), '%s = sp*sX + (1 - sp)*%s' % (x, x),
                  'sr3 = rel*stpScale%d' % c, '%s = sp*sr3 + (1 - sp)*%s' % (f, f)]
    elif stp:
        if stp.get('order', 0):
            # stpRelease at the spike, stpOrder 1: facilitation, then release
            L += ['sr1 = 1 - stpR', 'sr2 = stpU*sr1', 'sR = fround(stpR + sr2)',
                  'rel = sR*stpX', 'sX = fround(stpX - rel)']
        else:
            # as published: release from the pre-spike probability, then depletion and facilitation
            L += ['rel = stpR*stpX', 'sX = fround(stpX - rel)',
                  'sr1 = 1 - stpR', 'sr2 = stpU*sr1', 'sR = fround(stpR + sr2)']
        L += ['stpR = sp*sR + (1 - sp)*stpR', 'stpX = sp*sX + (1 - sp)*stpX',
              'sr3 = rel*stpScale', 'sfv = sp*sr3 + (1 - sp)*sfv']
    if syn == 0:
        L.append('ring = 0')
    else:
        L += ['gE = fround(ge)', 'gI = fround(gi)', 'ringE = 0', 'ringI = 0']
        for k in range(K):
            L += ['gX%d = fround(gx%d)' % (k, k), 'ringX%d = 0' % k]
    return '\n'.join(L)


def neuron_model(syn, K, nstim=0, nnoise=0, ext=False, pool=False, graded=0, dynamic=False, colored=False, stdp_at=False, stp_classes=0):
    names = ['v', 'u', 'refc', 'spiked', 'thr', 'gratio', 'mlev']
    if colored:
        names += ['nfou', 'nfous']
    names += ['ring'] if syn == 0 else ['gE', 'gI', 'ringE', 'ringI']
    for k in range(K):
        names += ['gX%d' % k, 'ringX%d' % k]
    if ext:
        names.append('ext')
    if pool:
        names.append('pin')
    if graded:
        names += ['Igr'] + ['rh%d' % d for d in range(1, graded + 1)]
    if dynamic:
        names += ['Kpre', 'Kpost', 'Kslow', 'Khom', 'spkAcc', 'rhoI', 'alphaI', 'stpX', 'stpR', 'sfv']
        for c in range(stp_classes):
            names += ['stpX%d' % c, 'stpR%d' % c, 'sfv%d' % c]
    if stdp_at:
        # the cell's spikes of the last 16 steps, h1 for the step before this one (simworker.js spkHist, bit k)
        names += ['h%d' % k for k in range(1, 17)]
    const = ['bias', 'Cm', 'kk', 'vr', 'vt', 'vpeak', 'aa', 'bb', 'cc', 'dd', 'thrJ', 'thrDec', 'adC', 'poi', 'minc']
    # prq and prz: short prefixes such as zs2 are Brian unit names
    const += ['prqf%d' % k for k in range(nstim)] + ['przf%d' % k for k in range(nnoise)]
    if graded:
        const += ['gd', 'gthr', 'gslp']
    if colored:
        const += ['nfa', 'nfk', 'nfc']
    # the stimulus a cell takes its shared draw from, as the integer the hash is keyed on
    ints = ['nfid'] if colored else []
    return '\n'.join(['%s : 1' % x for x in names] + ['%s : 1 (constant)' % x for x in const] + ['%s : integer (constant)' % x for x in ints])


# ---- plasticity (simworker.js plastOnSpike, consolidate, applyScaling, finishCalibration)

def softbound(W, D, OUT, p):
    # softBound: scale by the room left in the direction of travel, under wdep
    return ['%sa = abs(%s)' % (p, W),
            '%saway = 1 - abs(int(%s < 0) - int(%s < 0))' % (p, D, W),
            '%sr1 = (rwmax - %sa)/rwmax' % (p, p),
            '%sr2 = %sa/rwmax' % (p, p),
            '%sroom = %saway*%sr1 + (1 - %saway)*%sr2' % (p, p, p, p, p),
            '%src = clip(%sroom, 0, 1e300)' % (p, p),
            '%sds = %s*%src' % (p, D, p),
            '%s = rwdep*%sds + (1 - rwdep)*%s' % (OUT, p, D)]


def potentiate(W, KP, Z, OUT, p):
    # at a postsynaptic spike: excitatory (aP + trip z) Kpre, the heterosynaptic pull toward wRef by beta times the cubed fast postsynaptic trace (Zenke et al. 2015); inhibitory -iEta Kpre
    # pgn is the third factor (pgain, times modBase + the postsynaptic level under the gain form, MODEL.md 4); under the eligibility form (meOn) the Hebbian term goes to the trace (p + 'eli', added by the caller under its own mask) and the weight keeps its value here
    L = ['%sex = int(%s > 0)' % (p, W),
         '%st1 = rtrip*%s' % (p, Z), '%st2 = raP + %st1' % (p, p), '%st2g = pgn*%st2' % (p, p), '%st3 = %st2g*%s' % (p, p, KP),
         '%seli = meOn*%st3' % (p, p),
         '%snw = %s + %st3*(1 - meOn)' % (p, W, p),
         '%sgg = Kpost_post*Kpost_post' % p, '%sgate = %sgg*Kpost_post' % (p, p),
         '%shm = rhet*%sgate' % (p, p),
         '%sdw = %s - wRef' % (p, W), '%shh = %shm*%sdw' % (p, p, p), '%snw2 = %snw - %shh' % (p, p, p),
         '%swe = clip(%snw2, 0.0001, rwmax)' % (p, p),
         # under the global factor (iGlobalOn) the amplitude is eta times the population ratio, the step before's, held on every cell
         '%skg = gratio_post*%s' % (p, KP), '%skq = iGlobalOn*%skg + (1 - iGlobalOn)*%s' % (p, p, KP),
         '%sd0 = -rieta*%skq' % (p, p)]
    L += softbound(W, '%sd0' % p, '%sd1' % p, p + 's')
    L += ['%swi = clip(%s + %sd1, -rwmax, 1e300)' % (p, W, p),
          '%s = fround(%sex*%swe + (1 - %sex)*%swi)' % (OUT, p, p, p, p)]
    return L


def depress(W, KQ, KH, OUT, p):
    # at a presynaptic spike: excitatory w - aM sl Kpost (w under wdep, else 1) + tin, with sl the sliding threshold's (Khom slideK)^2 when it is on and 1 when off; inhibitory -iEta (Kpost - alpha)
    L = ['%sex = int(%s > 0)' % (p, W),
         '%sxp = abs(%s)**pmu' % (p, W),   # the excitatory branch alone reads this; abs keeps the unused inhibitory evaluation finite
         '%sx = rwdep*%sxp + (1 - rwdep)' % (p, p),
         '%skh = %s*slideK' % (p, KH), '%ssl = slideOn*%skh*%skh + (1 - slideOn)' % (p, p, p),
         '%st0 = pgn*raM' % p, '%st1 = %st0*%s*%ssl' % (p, p, KQ, p), '%st2 = %st1*%sx' % (p, p, p),
         # the trace's increment under the eligibility form, tin - depression as the engine adds it; the weight then keeps its value
         '%stg = pgn*rtin' % p, '%sedl = %stg - %st2' % (p, p, p), '%seli = meOn*%sedl' % (p, p),
         '%st3 = %s - %st2*(1 - meOn)' % (p, W, p), '%st4 = %st3 + %stg*(1 - meOn)' % (p, p, p),
         '%swe = clip(%st4, 0.0001, rwmax)' % (p, p),
         '%sq = %s - alq' % (p, KQ),
         # under the global factor the presynaptic term is the trace plus one, the authors' xi, times the ratio
         '%sq1 = %s + 1' % (p, KQ), '%sqg = gratio_post*%sq1' % (p, p), '%sqq = iGlobalOn*%sqg + (1 - iGlobalOn)*%sq' % (p, p, p),
         '%sd0 = -rieta*%sqq' % (p, p)]
    L += softbound(W, '%sd0' % p, '%sd1' % p, p + 's')
    L += ['%swi = clip(%s + %sd1, -rwmax, -0.0001)' % (p, W, p),
          '%s = fround(%sex*%swe + (1 - %sex)*%swi)' % (OUT, p, p, p, p)]
    return L


# a synapse changes only if its rule is not frozen, the rules are not calibrating, and it has not committed
MASKS = ['clb = rhoMode*upto(t/ms, calMs)',
         'cmt = commitOn*int(rconsw > 0)*int(wRef >= rconsw*0.9)',
         'ple = pl*(1 - clb)*(1 - cmt)',
         # the third factor: the gain, times modBase + the postsynaptic cell's modulator level under the gain form (mgOn), the gain alone otherwise (simworker.js plastOutSyn, plastOnSpike)
         'pgm = modBase + mlev_post', 'pgn = pgain*(1 - mgOn) + pgain*pgm*mgOn']

# At a presynaptic spike.
# When the post spiked in the same millisecond with the lower index, the engine potentiated first (old Kpre), so this delivers the potentiated weight and depresses with Kpost + 1.
PRE_PLAST = (MASKS + ['cp = ple*spiked_post*int(i > j)']
             + potentiate('w', 'Kpre_pre', 'Kslow_post', 'wpot', 'pz')
             + ['wa = cp*wpot + (1 - cp)*w',
                'sfs = 1 + (sfv_pre - 1)*(1 - stpScopeE*int(wa < 0))*(1 - stpScopeI*int(wa > 0))',
                'wsent = wa*sfs',
                'kqa = fround(Kpost_post + 1)',
                'kq = cp*kqa + (1 - cp)*Kpost_post',
                'kha = fround(Khom_post + 1)',
                'kh = cp*kha + (1 - cp)*Khom_post',
                'alq = rhoMode*alphaI_post + (1 - rhoMode)*iAlpha']
             + depress('wa', 'kq', 'kh', 'wdn', 'dz')
             + ['w = ple*wdn + (1 - ple)*wa',
                # the eligibility trace takes what the weight did not, under the same masks: the same-millisecond potentiation first, then the depression, two float32 stores as the engine makes them
                'elig = fround(fround(elig + cp*pzeli) + ple*dzeli)'])

# At a postsynaptic spike.
# A same-millisecond pre with the lower index went first, so Kpre already holds its spike; one with the higher index was handled in the presynaptic pathway.
POST_PLAST = (MASKS + ['cq = spiked_pre*int(i > j)',
                       'lower = spiked_pre*int(i < j)',
                       'kpa = fround(Kpre_pre + 1)',
                       'kpv = lower*kpa + (1 - lower)*Kpre_pre']
              + potentiate('w', 'kpv', 'Kslow_post', 'wpt', 'pq')
              + ['ap = ple*(1 - cq)', 'w = ap*wpt + (1 - ap)*w', 'elig = fround(elig + ap*pqeli)'])

# The pair rule timed at the synapse (stdpAt 1, simworker.js kpreAtSynapse and plastArrivals).
# A presynaptic spike counts when it arrives, d steps after it was fired.
# Potentiation at a postsynaptic spike reads the pre trace as it was d steps ago: the spikes of the last d - 1 steps are taken out of Kpre (h_k is the pre cell's spike k steps ago and ekd_k is e^(-k/tau+) for k below d, else 0, one product and one sum per step as the engine adds them) and the rest is brought forward by exd, e^(d/tau+).
# Kpre holds no spike of this step in either implementation here, so there is no same-millisecond correction to make.
# Depression runs in a third pathway, at the arrival, against Kpost as it stands before any cell of that step fires; it runs first in the step (order -3), as the engine's arrival pass does.
KPS = (['ha%d = h%d_pre*ekd%d' % (k, k, k) for k in range(1, 17)]
       + ['sb1 = ha1'] + ['sb%d = sb%d + ha%d' % (k, k - 1, k) for k in range(2, 17)]
       + ['kp0 = Kpre_pre - sb16', 'kp1 = exd*kp0', 'kps = clip(kp1, 0, 1e300)'])
PRE_PLAST_SYN = (MASKS + ['cp = ple*spiked_post*int(i > j)'] + KPS
                 + potentiate('w', 'kps', 'Kslow_post', 'wpot', 'pz')
                 + ['wa = cp*wpot + (1 - cp)*w',
                    'sfs = 1 + (sfv_pre - 1)*(1 - stpScopeE*int(wa < 0))*(1 - stpScopeI*int(wa > 0))',
                    'wsent = wa*sfs',
                    'w = wa', 'elig = fround(elig + cp*pzeli)'])
POST_PLAST_SYN = (MASKS + ['cq = spiked_pre*int(i > j)'] + KPS
                  + potentiate('w', 'kps', 'Kslow_post', 'wpt', 'pq')
                  + ['ap = ple*(1 - cq)', 'w = ap*wpt + (1 - ap)*w', 'elig = fround(elig + ap*pqeli)'])
ARRIVE_PLAST = (MASKS + ['alq = rhoMode*alphaI_post + (1 - rhoMode)*iAlpha']
                + depress('w', 'Kpost_post', 'Khom_post', 'wdn', 'dz')
                + ['w = ple*wdn + (1 - ple)*w', 'elig = fround(elig + ple*dzeli)'])
# The eligibility sweep (simworker.js eligSweep, MODEL.md 4): every modStep ms, at the end of the step after the field update, each excitatory plastic synapse's trace pays (1 - e^(-modStep/eligTau)) of itself times modBase + the postsynaptic level into the weight and decays; a committed synapse keeps its weight.
ELIG_SWEEP = '\n'.join([
    'gw = every(t/ms, modStep)',
    'ec = elig',
    'cmt = commitOn*int(rconsw > 0)*int(wRef >= rconsw*0.9)',
    'ebm = modBase + mlev_post', 'ep1 = ec*eligPay', 'ep2 = ep1*ebm', 'enw = w + ep2',
    'ewe = clip(enw, 0.0001, rwmax)',
    'ego = gw*pl*(1 - cmt)*int(w > 0)*int(ec != 0)',
    'w = ego*fround(ewe) + (1 - ego)*w',
    'edc = ec*decElig',
    'elig = gw*fround(edc) + (1 - gw)*ec'])
# the history moves on with the trace decay: h1 takes this step's spike
SHIFT = '\n'.join(['h%d = h%d' % (k, k - 1) for k in range(16, 1, -1)] + ['h1 = spiked'])

RESET_PLAST = '\n'.join(['Kpre = fround(Kpre + 1)', 'Kpost = fround(Kpost + 1)', 'Khom = fround(Khom + 1)',
                         'Kslow = hasSlow*fround(Kslow + 1)', 'spkAcc = fround(spkAcc + 1)'])
DECAY = '\n'.join(['Kpre = fround(Kpre*decS)', 'Kpost = fround(Kpost*decM)', 'Kslow = fround(Kslow*decY)', 'Khom = fround(Khom*decH)'])
STP_RECOVER = '\n'.join(['xr1 = 1 - stpX', 'xr2 = xr1*stpDx', 'stpX = fround(1 - xr2)',
                         'rr1 = stpR - stpU', 'rr2 = rr1*stpDr', 'stpR = fround(stpU + rr2)'])
CONSOLIDATE = '\n'.join([
    'gc = consOn*every(t/ms, consStep)',
    'cmk = gc*pl*int(rcons > 0)*int(w > 0)',
    'hlf = rconsw*0.5',
    'q1 = w - wRef', 'q2 = rconsp*wRef', 'q3 = hlf - wRef', 'q4 = q2*q3',
    'q5 = rconsw - wRef', 'q6 = q4*q5', 'q7 = q1 - q6', 'q8 = consRate*q7',
    'nr = wRef + q8',
    'top1 = wRef + int(w > wRef)*(w - wRef)', 'top2 = top1 + int(rconsw > top1)*(rconsw - top1)',
    'nrc = clip(nr, 0, top2)',
    'wRef = cmk*fround(nrc) + (1 - cmk)*wRef'])
SCALE = '\n'.join([
    'gs = scaleOn*every_from(t/ms, 1000.0, rhoMode*calMs)',
    'rho = rhoMode*rhoI_post + (1 - rhoMode)*iRho',
    # a set point of 0 exists only before calibration ends, when scaling is off; the guard keeps that unused branch finite and changes nothing else
    'rhoq = rho + int(rho <= 0)',
    'er1 = rhoq - spkAcc_post', 'er2 = er1/rhoq', 'err = clip(er2, -2, 2)',
    'fsc = 1 + sEta*err',
    'cmt = commitOn*int(rconsw > 0)*int(wRef >= rconsw*0.9)',
    'ga = gs*pl*(1 - cmt)*int(w > 0)*int(err != 0)',
    'wsc = w*fsc', 'wcap = clip(wsc, -1e300, rwmax)',
    'w = ga*fround(wcap) + (1 - ga)*w'])
CALIBRATE = '\n'.join([
    'gf = rhoMode*atstep(t/ms, calMs)',
    'rn1 = spkAcc/calSecs', 'rn2 = clip(rn1, 0.25, 30)',
    'rhoI = gf*fround(rn2) + (1 - gf)*rhoI',
    'an1 = rhoI*0.001', 'an2 = an1*tauSum',
    'alphaI = gf*fround(an2) + (1 - gf)*alphaI',
    'gsr = scaleOn*every_from(t/ms, 1000.0, rhoMode*calMs)',
    'rs = int(gf + gsr > 0.5)',
    'spkAcc = (1 - rs)*spkAcc'])


def ext_rebuild(maps, gains, n):
    """rebuildExt: map by map, channel by channel, entry by entry, float32."""
    ext = np.zeros(n, dtype=np.float64)
    for m, g in zip(maps, gains):
        starts, idx, wts, amp = m['chStart'], m['chIdx'], m['chW'], float(m['amp'])
        for c in range(min(len(g), len(starts) - 1)):
            gv = float(g[c])*amp
            if not gv:
                continue
            for k in range(starts[c], starts[c + 1]):
                i = idx[k]
                ext[i] = float(np.float32(ext[i] + gv*float(wts[k])))
    return ext


def run_case(case):
    n = int(case['n'])
    syn = int(case.get('syn', 0))
    chan_dec = case.get('chanDec') or []
    K = len(chan_dec)
    stim = case.get('stim') or []
    noise = case.get('noiseProtocols') or []
    pool = case.get('pool') is not None
    maps = case.get('inputs')
    dt = float(case.get('dt', 1.0))
    if not (0 < dt <= 1):
        raise ValueError('dt is in milliseconds and must be above 0 and at most 1')
    ext = maps is not None
    plast = case.get('plast')
    stp = case.get('stp')
    dynamic = plast is not None or stp is not None
    stdp_at = plast is not None and int(plast.get('stdpAt') or 0) == 1
    grd = np.asarray(case.get('grd') or np.zeros(n), dtype=np.float64)
    post = np.asarray(case.get('post') or [], dtype=np.int64)
    starts = np.asarray(case.get('pre') or np.zeros(n + 1), dtype=np.int64)
    pre = np.repeat(np.arange(n), np.diff(starts)) if len(post) else np.zeros(0, dtype=np.int64)
    delay = np.asarray(case.get('delay') or [], dtype=np.float64)
    from_graded = grd[pre] > 0 if len(post) else np.zeros(0, dtype=bool)
    graded = int(delay[from_graded].max()) if from_graded.any() else 0
    if graded and (syn != 0 or dynamic):
        raise ValueError('graded rows are compared on kick synapses without plasticity only')
    # A graded release is assigned to its target, not summed, and it is added to the spike ring as a separately rounded value, where the engine accumulates both into one float32 slot.
    # So a target reproduces the engine only with one graded input and no spiking one; brianref.mjs builds its graded case under the same condition.
    if graded:
        tg = post[from_graded]
        if len(np.unique(tg)) != len(tg):
            raise ValueError('a cell receives two or more graded inputs, which this reference '
                             'cannot sum the way the engine does')
        if np.isin(tg, post[~from_graded]).any():
            raise ValueError('a cell receives graded and spiking input both, which this reference '
                             'cannot add the way the engine does')
    # everything refused here keeps a count or a schedule in whole milliseconds (trace decay, consolidation, scaling, short-term recovery, a graded cell's release history, an input frame, a pool's spike count)
    if dt != 1.0 and (dynamic or graded or ext or pool):
        raise ValueError('a step below one millisecond runs a plain network only: no plasticity, '
                         'no short-term plasticity, no graded rows, no input frames, no pools')
    # A kick synapse has no time constant: its weight is a current applied for one step, so the charge a spike delivers is the weight times the step and the mode means something different at every dt.
    # Nothing about that is an integration error, so it is refused rather than measured.
    if dt != 1.0 and syn == 0 and len(post):
        raise ValueError('a kick synapse injects its weight for one step, so its meaning follows the '
                         'step; compare a step below one millisecond on exponential or conductance '
                         'synapses')

    vmin = case.get('vmin')
    ns = {'fround': fround, 'every': every, 'every_from': every_from, 'upto': upto, 'atstep': atstep, 'pscale': pscale,
          'vmin': -1e300 if vmin is None else float(vmin),
          'refrac': float(case.get('refrac', 0))}
    for prefix, protos in (('prq', stim), ('prz', noise)):
        for k, p in enumerate(protos):
            for field, name in (('amp', 'amp'), ('t0', 't0'), ('duration', 'dur'), ('mode', 'mode'),
                                ('period', 'per'), ('width', 'wid')):
                ns['%s%s%d' % (prefix, name, k)] = float(p[field])
    if syn:
        for key in ('decE', 'decI', 'itauE', 'itauI', 'eRevE', 'eRevI'):
            ns[key] = float(case[key])
        # the decay factors are per millisecond (exp(-1/tau)); a step of dt decays by exp(-dt/tau), which is the same factor raised to dt.
        # The current conversion (itau) is a scale between conductance and current and does not follow the step.
        if dt != 1.0:
            ns['decE'] = ns['decE']**dt
            ns['decI'] = ns['decI']**dt
        # psc 2 is the exact integral of the exponential over the step, so it does follow the step: tau (1 - e^(-dt/tau)) / dt, which tends to 1
        exact = int(case.get('psc', 0)) == 2
        def exact_itau(dec_ms):
            tau = -1.0/np.log(dec_ms)
            return tau*(1.0 - dec_ms**dt)/dt
        if exact and dt != 1.0:
            ns['itauE'] = exact_itau(float(case['decE']))
            ns['itauI'] = exact_itau(float(case['decI']))
        for k in range(K):
            ns['chanDec%d' % k] = float(chan_dec[k])**dt
            ns['chanItau%d' % k] = exact_itau(float(chan_dec[k])) if exact and dt != 1.0 else float(case['chanItau'][k])
            ns['chanErev%d' % k] = float(case['chanErev'][k]) if k < len(case.get('chanErev') or []) else float('nan')
            ns['chanMg%d' % k] = float(case['chanMg'][k]) if k < len(case.get('chanMg') or []) else 0.0
    if noise:
        ns['noise'] = noise_function(int(case['seed']))
    # poidraw, not poisson: Brian has a poisson() of its own
    ns['poidraw'] = poisson_function(int(case['seed']))
    # the reversal the conductance adaptation draws toward, the engine's inhibitory reversal (simworker.js S.eRevAd)
    ns['eRevAd'] = float(case.get('eRevI', -70))
    # neuromodulation (MODEL.md 4): what reads the field, 0 and the plain rule without a case entry
    mod = case.get('mod')
    ns.update({'modDrive': 0.0, 'modCur': 0.0, 'modBase': 1.0, 'mgOn': 0.0, 'meOn': 0.0, 'eligPay': 0.0, 'decElig': 1.0, 'modStep': 10.0})
    if mod is not None:
        ns['modDrive'] = float(mod.get('drive', 0.0)); ns['modCur'] = float(mod.get('cur', 0.0))
        ns['modBase'] = float(mod.get('base', 1.0))
        ns['mgOn'] = 1.0 if int(mod.get('plast', 0)) == 1 and plast is not None else 0.0
        ns['meOn'] = 1.0 if int(mod.get('plast', 0)) == 2 and plast is not None else 0.0
        ns['eligPay'] = float(mod.get('eligPay', 0.0)); ns['decElig'] = float(mod.get('decElig', 1.0))
        ns['modStep'] = float(mod.get('step', 10))
    if plast is not None:
        for key in ('iAlpha', 'decS', 'decM', 'decY', 'hasSlow', 'commitOn', 'sEta', 'iRho',
                    'scaleOn', 'rhoMode', 'calMs', 'calSecs', 'tauSum', 'consOn', 'consStep', 'consRate'):
            ns[key] = float(plast[key])
        # the third factor: the gain on the Hebbian terms, 1 unless the case says (simworker.js S.gain)
        ns['pgain'] = float(plast.get('gain', 1.0))
        # the weight exponent of soft depression (simworker.js S.wmu), 1 unless the case says
        ns['pmu'] = float(plast.get('wmu', 1.0))
        # the sliding threshold (simworker.js S.slideK, S.decH): off in a case that does not name it
        ns['slideOn'] = float(plast.get('slideOn', 0.0))
        ns['slideK'] = float(plast.get('slideK', 0.0))
        ns['decH'] = float(plast.get('decH', 1.0))
        ns['iGlobalOn'] = float(plast.get('iGlobalOn', 0.0))
    else:
        for key in ('rhoMode', 'commitOn', 'iGlobalOn'):
            ns[key] = 0.0
    # the scope: 1 keeps inhibitory synapses out of the release, 2 excitatory ones; named in every case, since the plastic delivery path reads them whether or not the term is on
    ns['stpScopeE'] = 0.0
    ns['stpScopeI'] = 0.0
    stp_classes = stp.get('classes') or [] if stp is not None else []
    if stp is not None:
        ns['stpScopeE'] = 1.0 if int(stp.get('scope') or 0) == 1 else 0.0
        ns['stpScopeI'] = 1.0 if int(stp.get('scope') or 0) == 2 else 0.0
        ns.update({'stpU': float(stp['U']), 'stpScale': float(stp['scale']),
                   'stpDx': float(stp['dx']), 'stpDr': float(stp['dr'])})
        for c, cl in enumerate(stp_classes):
            ns.update({'stpU%d' % c: float(cl['U']), 'stpScale%d' % c: float(cl['scale']),
                       'stpDx%d' % c: float(cl['dx']), 'stpDr%d' % c: float(cl['dr'])})

    defaultclock.dt = dt*ms
    group_args = {'threshold': 'spiked > 0.5', 'namespace': ns}
    if plast is not None:
        group_args['reset'] = RESET_PLAST
    G = NeuronGroup(n, neuron_model(syn, K, len(stim), len(noise), ext, pool, graded, dynamic, bool(case.get('noise')), stdp_at, len(stp_classes)), **group_args)
    G.run_regularly(step_code(syn, K, len(stim), len(noise), ext, pool, graded, stp, dt, case.get('chanMg') or [], bool(case.get('noise')), int(case.get('integ', 0)), mod is not None), when='groups')
    objects = [G]
    # after the step, in the engine's order: trace decay, consolidation, short-term recovery, then (at the clock tick) scaling and calibration
    if plast is not None:
        objects.append(G.run_regularly(DECAY + ('\n' + SHIFT if stdp_at else ''), when='end', order=0))
        objects.append(G.run_regularly(CALIBRATE, when='end', order=4))
    if stp is not None and stp_classes:
        rec = []
        for c, cl in enumerate(stp_classes):
            if cl['on']:
                rec.append(STP_RECOVER.replace('stpX', 'stpX%d' % c).replace('stpR', 'stpR%d' % c)
                           .replace('stpU', 'stpU%d' % c).replace('stpDx', 'stpDx%d' % c).replace('stpDr', 'stpDr%d' % c))
        if rec:
            objects.append(G.run_regularly('\n'.join(rec), when='end', order=2))
    elif stp is not None:
        objects.append(G.run_regularly(STP_RECOVER, when='end', order=2))
    rows = np.asarray(case['rows'], dtype=np.float64)
    for name, col in zip(['Cm', 'kk', 'vr', 'vt', 'vpeak', 'aa', 'bb', 'cc', 'dd', 'thrJ', 'thrDec', 'adC', 'poi'], rows.T):
        setattr(G, name, col)
    G.thr = 0.0
    G.gratio = 0.0
    G.mlev = 0.0
    G.minc = np.asarray(mod['inc'], dtype=np.float64) if mod is not None else 0.0
    G.bias = case['bias']
    G.v = case['v0']
    if plast is not None:
        G.Khom = float(plast.get('khomInit', 0.0))
    G.u = case['u0']
    if case.get('noise'):
        # a noise stimulus's memory and shared part per cell, folded by the case builder; the shared draw is keyed on NOISE_SHARED plus the stimulus's index
        nf = case['noise']
        G.nfa = np.asarray(nf['a'], dtype=np.float64)
        G.nfk = np.asarray(nf['k'], dtype=np.float64)
        G.nfc = np.asarray(nf['c'], dtype=np.float64)
        G.nfid = NOISE_SHARED + np.maximum(0, np.asarray(nf['id'], dtype=np.int64))
        G.nfou = 0.0
        G.nfous = 0.0
    for prefix, protos in (('prq', stim), ('prz', noise)):
        for k, p in enumerate(protos):
            setattr(G, '%sf%d' % (prefix, k), p['f'])
    if graded:
        G.gd = grd
        G.gthr = case['gthr']
        G.gslp = np.where(grd > 0, np.asarray(case['gslp'], dtype=np.float64), 1.0)
    if dynamic:
        G.stpX = 1.0
        G.stpR = float(stp['R0']) if stp is not None else 0.0
        G.sfv = 1.0
        for c, cl in enumerate(stp_classes):
            setattr(G, 'stpX%d' % c, 1.0)
            setattr(G, 'stpR%d' % c, float(cl['R0']))
            setattr(G, 'sfv%d' % c, 1.0)

    if plast is not None and len(post):
        # The global factor on the inhibitory rule (simworker.js globalFactor): the population is every cell with an inhibitory synapse under a rule; its spike count this step is folded into the low-passed count, kept in float32 as the engines keep it, and the ratio of the rate it stands for to the target, minus one, is held on every cell for the next step's inhibitory updates.
        wg = np.asarray(case['w'], dtype=np.float64)
        rg = np.asarray(case.get('rule') or np.ones(len(post)), dtype=np.int64)
        gmask = np.zeros(n, dtype=bool)
        gmask[post[(wg < 0) & (rg != 0)]] = True
        gN = int(gmask.sum())
        gstate = {'pf': np.float32(float(plast['iRho'])*gN)}
        gInv = float(np.float32(1.0/(gN*float(plast['iRho'])))) if gN else 0.0
        decG, tauGs = float(plast.get('decG', 1.0)), float(plast.get('tauGs', 1.0))
        if gN:
            @network_operation(when='end', order=6)
            def global_factor():
                cnt = float(np.count_nonzero(np.asarray(G.spiked[:])[gmask] > 0.5))
                pf = np.float32(float(gstate['pf'])*decG + cnt/tauGs)
                gstate['pf'] = pf
                G.gratio[:] = float(np.float32(float(pf)*gInv - 1.0))
            objects.append(global_factor)
    if mod is not None:
        # The modulator field at the end of the step (simworker.js modUpdate): M(t) = M(t-1) dec + the sources' increment + the group's release per spike times the group's spikes of the step before, stored in float32.
        # Everything in the step read M(t-1); the sweep (order 8) reads the new level.
        decMod = float(mod['decMod'])
        mgrp = np.asarray(mod['grp'], dtype=np.int64) if mod.get('grp') else None
        mtgt = np.asarray(mod['tgt'], dtype=np.int64) if mod.get('tgt') else None
        mKf = np.asarray(mod['Kf'], dtype=np.float64)
        mstate = {'cnt': np.zeros(len(mKf))}
        minc = np.asarray(mod['inc'], dtype=np.float64)

        @network_operation(when='end', order=7)
        def mod_update():
            m = np.asarray(G.mlev[:], dtype=np.float64)
            v = m*decMod + minc
            if mtgt is not None:
                v = v + mKf[mtgt]*mstate['cnt'][mtgt]
            G.mlev[:] = v.astype(np.float32).astype(np.float64)
            if mgrp is not None:
                fired = np.asarray(G.spiked[:]) > 0.5
                mstate['cnt'] = np.bincount(mgrp[fired], minlength=len(mKf)).astype(np.float64)
        objects.append(mod_update)
    if pool:
        # the pool drive for the next step: poolK times this step's spike count per pool
        pool_idx = np.asarray(case['pool'], dtype=np.int64)
        poolK = np.asarray(case['poolK'], dtype=np.float64)

        @network_operation(when='end')
        def pool_count():
            fired = np.asarray(G.spiked[:]) > 0.5
            cnt = np.bincount(pool_idx[fired], minlength=len(poolK)).astype(np.float64)
            G.pin[:] = (poolK*cnt).astype(np.float32).astype(np.float64)[pool_idx]
        objects.append(pool_count)

    if ext:
        # an input frame sent before tick t sets the gains; the drive is rebuilt then
        frames_at = {}
        for f in case.get('frames') or []:
            frames_at.setdefault(int(f['t']), []).append((int(f['mi']), f['gains']))
        gains = [np.zeros(len(m['chStart']) - 1) for m in maps]

        @network_operation(when='start')
        def apply_frames(t):
            k = int(round(float(t/ms)))
            if k in frames_at:
                for mi, g in frames_at[k]:
                    gains[mi] = list(g)
                G.ext[:] = ext_rebuild(maps, gains, n)
        objects.append(apply_frames)

    groups = []
    if len(post):
        w = np.asarray(case['w'], dtype=np.float64)
        chan = np.asarray(case.get('chan') or np.zeros(len(post)), dtype=np.int64)
        rule = np.asarray(case.get('rule') or np.ones(len(post)), dtype=np.int64)
        spiking = ~from_graded
        # the ring each synapse feeds (simworker.js stepExp ringOf): a channel of its own, else by the sign of its weight
        if syn == 0:
            targets = [('ring', spiking)]
        else:
            targets = [('ringE', spiking & (chan < 2) & (w > 0)), ('ringI', spiking & (chan < 2) & ~(w > 0))]
            targets += [('ringX%d' % k, spiking & (chan == k + 2)) for k in range(K)]
        rules = np.asarray(plast['rules'], dtype=np.float64) if plast is not None else None
        for ring, sel in targets:
            idx = np.flatnonzero(sel)
            if not len(idx):
                continue
            deliver = '%s_post = fround(%s_post + %s)' % (ring, ring, 'wsent' if dynamic else 'w')
            if not dynamic:
                S = Synapses(G, G, 'w : 1 (constant)', on_pre=deliver, namespace=ns)
            else:
                model = '\n'.join(['w : 1', 'wsent : 1', 'wRef : 1', 'elig : 1', 'pl : 1 (constant)'] + (['stc : 1 (constant)'] if stp_classes else [])
                                  + ['%s : 1 (constant)' % r for r in RULE_VARS]
                                  + (['exd : 1 (constant)'] + ['ekd%d : 1 (constant)' % k for k in range(1, 17)] if stdp_at else []))
                # the release factor in scope (stpScope): 1 on a synapse the scope leaves out
                if plast is None:
                    emit = 'sfs = 1 + (sfv_pre - 1)*(1 - stpScopeE*int(w < 0))*(1 - stpScopeI*int(w > 0))\nwsent = w*sfs'
                else:
                    emit = '\n'.join(PRE_PLAST_SYN if stdp_at else PRE_PLAST)
                if stp_classes:
                    # the synapse's release is its rule's class (stc), chosen from the classes' factors
                    sel = ' + '.join('sfv%d_pre*int(stc == %d)' % (c, c) for c in range(len(stp_classes)))
                    emit = 'sfsel = ' + sel + '\n' + emit.replace('sfv_pre', 'sfsel')
                pre_paths = {'pre': emit, 'deliver': deliver}
                if stdp_at:
                    pre_paths['arrive'] = '\n'.join(ARRIVE_PLAST)
                S = Synapses(G, G, model, on_pre=pre_paths,
                             on_post='\n'.join(POST_PLAST_SYN if stdp_at else POST_PLAST) if plast is not None else None,
                             namespace=ns)
            # in the engine's CSR order, so arrivals in one step are summed in the order the engine sums them
            S.connect(i=pre[idx], j=post[idx])
            S.w = w[idx]
            if not dynamic:
                # a spike at t reaches its target's input at t + d: Brian delivers in the synapse slot of the step after the delay, so the delay it is given is one step short of d
                S.delay = (delay[idx] - dt)*ms
            else:
                S.wRef = w[idx]
                S.pl = (rule[idx] != 0).astype(np.float64)
                if stp_classes:
                    S.stc = np.asarray(stp['cls'], dtype=np.float64)[rule[idx]]
                if rules is not None:
                    # a frozen synapse (rule 0) never uses its row, but the masks multiply the unused branch by 0 and 0 times NaN is NaN, so row 0's zero wmax (a division) is replaced by a harmless one
                    safe = rules.copy()
                    safe[0] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 10]
                    table = safe[rule[idx]]
                    for x, r in enumerate(RULE_VARS):
                        setattr(S, r, table[:, x])
                    # the reference starts inside its well: above consW under a consolidating rule it is consW (refIntoWell in simworker.js)
                    wr = np.array(w[idx], dtype=np.float64)
                    cons_rule = (table[:, 8] > 0) & (table[:, 9] > 0) & (wr > 0)
                    ref_init = int(plast.get('refInit', 0) or 0)
                    if ref_init == 1:
                        wr[cons_rule] = 0.0
                    elif ref_init == 2:
                        wr[cons_rule] = table[cons_rule, 9]
                    else:
                        into = cons_rule & (wr > table[:, 9])
                        wr[into] = table[into, 9]
                    S.wRef = wr
                S.pre.delay = 0*ms
                S.deliver.delay = (delay[idx] - 1)*ms
                S.pre.order = -2
                S.deliver.order = -1
                if stdp_at:
                    # the arrival, d steps after the spike, before anything else of that step
                    S.arrive.delay = delay[idx]*ms
                    S.arrive.order = -3
                    dl = delay[idx].astype(np.int64)
                    S.exd = np.asarray(plast['exd'], dtype=np.float64)[dl]
                    ek = np.asarray(plast['ek'], dtype=np.float64)
                    for k in range(1, 17):
                        setattr(S, 'ekd%d' % k, np.where(k < dl, ek[k], 0.0))
                if plast is not None:
                    S.post.order = 1
                    if plast['consOn']:
                        objects.append(S.run_regularly(CONSOLIDATE, when='end', order=1))
                    if plast['scaleOn']:
                        objects.append(S.run_regularly(SCALE, when='end', order=3))
                    if ns['meOn'] > 0.5:
                        objects.append(S.run_regularly(ELIG_SWEEP, when='end', order=8))
            objects.append(S)
            groups.append((idx, S))
        if graded:
            # graded release: after the step, each target's input for the next step is its synapse's weight times the release its delay names
            gidx = np.flatnonzero(from_graded)
            GS = Synapses(G, G, 'w : 1 (constant)\ndl : 1 (constant)', namespace=ns)
            GS.connect(i=pre[gidx], j=post[gidx])
            GS.w = w[gidx]
            GS.dl = delay[gidx]
            pick = ' + '.join('int(dl == %d)*rh%d_pre' % (d, d) for d in range(1, graded + 1))
            objects.append(GS.run_regularly('Igr_post = fround(w*(%s))' % pick, when='end', order=5))
            objects.append(GS)

    mon = SpikeMonitor(G)
    objects.append(mon)
    net = Network(*objects)
    t0 = time.time()
    net.run(float(case['ms'])*ms)
    trains = mon.spike_trains()
    # whole milliseconds at the engine's step, four decimals under it
    times = ([np.round(trains[i]/ms).astype(int).tolist() for i in range(n)] if dt == 1.0
             else [np.round(np.asarray(trains[i]/ms, dtype=np.float64), 4).tolist() for i in range(n)])
    out = {'times': times, 'dt': dt, 'wallS': round(time.time() - t0, 3)}
    if dynamic and len(post):
        wf = np.zeros(len(post))
        for idx, S in groups:
            wf[idx] = np.asarray(S.w[:], dtype=np.float64)
        out['w'] = wf.tolist()
        if plast is not None and plast['rhoMode']:
            out['rho'] = np.asarray(G.rhoI[:], dtype=np.float64).tolist()
    if case.get('noiseProbe'):
        out['noiseProbe'] = [float(noise_draw(i, t, s)) for i, t, s in case['noiseProbe']]
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        case = json.load(f)
    out = run_case(case)
    out['versions'] = {'brian2': brian2.__version__, 'numpy': np.__version__}
    with open(dst, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()

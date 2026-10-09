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


def noise_draw(i, t, seed):
    i = np.asarray(i, dtype=np.uint64)
    t = np.asarray(t, dtype=np.uint64)
    h = (i ^ ((t*np.uint64(2654435761)) & MASK) ^ np.uint64(seed)) & MASK
    a = pcg(h).astype(np.float64)/4294967295.0
    b = pcg(h ^ np.uint64(0x9e3779b9)).astype(np.float64)/4294967295.0
    return (a + b) - 1.0


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


def step_code(syn, K, nstim=0, nnoise=0, ext=False, pool=False, graded=0, stp=False, dt=1.0):
    """The engine's step, one assignment per quantity, sums two terms at a time.
    `graded` is the longest delay of a graded synapse, or 0. `dt` is the step in
    milliseconds: at 1 every line below is the engine's own, and under it the
    membrane, the recovery and the refractory count follow the smaller step."""
    L = []
    half = '0.5' if dt == 1.0 else repr(0.5*dt)
    per = '' if dt == 1.0 else repr(dt) + '*'
    if syn == 0:
        L.append('Isyn = ring + Igr' if graded else 'Isyn = ring')
    else:
        L.append('ge = gE*decE + ringE')
        L.append('gi0 = gI*decI + ringI')
        L.append('gi = gi0 + pin' if pool else 'gi = gi0')
        if syn == 2:
            L += ['s1 = abs(ge)*(eRevE - v)', 's2 = abs(gi)*(eRevI - v)']
        else:
            L += ['s1 = ge*itauE', 's2 = gi*itauI']
        L.append('Isyn = s1 + s2')
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
        L += lines + ['nz = %s*noise(i, t/ms)' % amp, 'In = %s + nz' % cur]
        cur = 'In'
    if pool and syn == 0:
        L.append('Ip = %s + pin' % cur)
        cur = 'Ip'
    L.append('I0 = %s' % cur)
    for k in range(K):
        L.append('gx%d = gX%d*chanDec%d + ringX%d' % (k, k, k, k))
        if syn == 2:
            L.append('c%d = abs(gx%d)*(chanErev%d - v)' % (k, k, k))
        else:
            L.append('c%d = gx%d*chanItau%d' % (k, k, k))
        L.append('I%d = I%d + c%d' % (k + 1, k, k))
    I = 'I%d' % K
    L += [
        'act = int(refc <= 0)' if dt == 1.0 else 'act = int(refc <= 1e-9)',
        'v1 = clip(v + %s*(kk*(v - vr)*(v - vt) - u + %s)/Cm, vmin, 1e300)' % (half, I),
        'v2 = clip(v1 + %s*(kk*(v1 - vr)*(v1 - vt) - u + %s)/Cm, vmin, 1e300)' % (half, I),
    ]
    if graded:
        # a graded cell's membrane is passive, capped at vpeak, and never spikes
        L += ['gl = kk*(vt - vr)',
              'gw1 = clip(v + %s*(-gl*(v - vr) - u + %s)/Cm, vmin, 1e300)' % (half, I),
              'gw2 = clip(gw1 + %s*(-gl*(gw1 - vr) - u + %s)/Cm, vmin, 1e300)' % (half, I),
              'gw3 = clip(gw2, -1e300, vpeak)',
              'vq = act*v2 + (1 - act)*v',
              'vn = gd*gw3 + (1 - gd)*vq',
              'un = u + %saa*(bb*(vn - vr) - u)' % per,
              'sp = act*int(vn >= vpeak)*(1 - gd)']
    else:
        L += ['vn = act*v2 + (1 - act)*v',
              'un = u + %saa*(bb*(vn - vr) - u)' % per,
              'sp = act*int(vn >= vpeak)']
    L += ['v = fround(sp*cc + (1 - sp)*vn)',
          'u = fround(un + sp*dd)',
          'refc = sp*refrac + (1 - sp)*(refc - %sint(refc > 0))' % per,
          'spiked = sp']
    if graded:
        # release this millisecond, pushed onto the history the synapses read
        L += ['rq1 = vn - gthr', 'rq2 = rq1/gslp', 'rgr = clip(rq2, 0, 1)']
        L += ['rh%d = rh%d' % (d, d - 1) for d in range(graded, 1, -1)]
        L.append('rh1 = gd*rgr')
    if stp:
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


def neuron_model(syn, K, nstim=0, nnoise=0, ext=False, pool=False, graded=0, dynamic=False):
    names = ['v', 'u', 'refc', 'spiked']
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
        names += ['Kpre', 'Kpost', 'Kslow', 'spkAcc', 'rhoI', 'alphaI', 'stpX', 'stpR', 'sfv']
    const = ['bias', 'Cm', 'kk', 'vr', 'vt', 'vpeak', 'aa', 'bb', 'cc', 'dd']
    # prq and prz: short prefixes such as zs2 are Brian unit names
    const += ['prqf%d' % k for k in range(nstim)] + ['przf%d' % k for k in range(nnoise)]
    if graded:
        const += ['gd', 'gthr', 'gslp']
    return '\n'.join(['%s : 1' % x for x in names] + ['%s : 1 (constant)' % x for x in const])


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
    # at a postsynaptic spike: excitatory (aP + trip z) Kpre, the heterosynaptic pull toward wRef gated by the cubed slow trace; inhibitory -iEta Kpre
    L = ['%sex = int(%s > 0)' % (p, W),
         '%st1 = rtrip*%s' % (p, Z), '%st2 = raP + %st1' % (p, p), '%st3 = %st2*%s' % (p, p, KP),
         '%snw = %s + %st3' % (p, W, p),
         '%sg = %s/kRef' % (p, Z), '%sgg = %sg*%sg' % (p, p, p), '%sgate = %sgg*%sg' % (p, p, p),
         '%shg = rhet*%sgate' % (p, p), '%shm = clip(%shg, -1e300, 1)' % (p, p),
         '%sdw = %s - wRef' % (p, W), '%shh = %shm*%sdw' % (p, p, p), '%snw2 = %snw - %shh' % (p, p, p),
         '%swe = clip(%snw2, 0.0001, rwmax)' % (p, p),
         '%sd0 = -rieta*%s' % (p, KP)]
    L += softbound(W, '%sd0' % p, '%sd1' % p, p + 's')
    L += ['%swi = clip(%s + %sd1, -rwmax, 1e300)' % (p, W, p),
          '%s = fround(%sex*%swe + (1 - %sex)*%swi)' % (OUT, p, p, p, p)]
    return L


def depress(W, KQ, OUT, p):
    # at a presynaptic spike: excitatory w - aM Kpost (w under wdep, else 1) + tin; inhibitory -iEta (Kpost - alpha)
    L = ['%sex = int(%s > 0)' % (p, W),
         '%sx = rwdep*%s + (1 - rwdep)' % (p, W),
         '%st1 = raM*%s' % (p, KQ), '%st2 = %st1*%sx' % (p, p, p), '%st3 = %s - %st2' % (p, W, p),
         '%st4 = %st3 + rtin' % (p, p),
         '%swe = clip(%st4, 0.0001, rwmax)' % (p, p),
         '%sq = %s - alq' % (p, KQ), '%sd0 = -rieta*%sq' % (p, p)]
    L += softbound(W, '%sd0' % p, '%sd1' % p, p + 's')
    L += ['%swi = clip(%s + %sd1, -rwmax, -0.0001)' % (p, W, p),
          '%s = fround(%sex*%swe + (1 - %sex)*%swi)' % (OUT, p, p, p, p)]
    return L


# a synapse changes only if its rule is not frozen, the rules are not calibrating, and it has not committed
MASKS = ['clb = rhoMode*upto(t/ms, calMs)',
         'cmt = commitOn*int(rconsw > 0)*int(wRef >= rconsw*0.9)',
         'ple = pl*(1 - clb)*(1 - cmt)']

# At a presynaptic spike.
# When the post spiked in the same millisecond with the lower index, the engine potentiated first (old Kpre), so this delivers the potentiated weight and depresses with Kpost + 1.
PRE_PLAST = (MASKS + ['cp = ple*spiked_post*int(i > j)']
             + potentiate('w', 'Kpre_pre', 'Kslow_post', 'wpot', 'pz')
             + ['wa = cp*wpot + (1 - cp)*w',
                'wsent = wa*sfv_pre',
                'kqa = fround(Kpost_post + 1)',
                'kq = cp*kqa + (1 - cp)*Kpost_post',
                'alq = rhoMode*alphaI_post + (1 - rhoMode)*iAlpha']
             + depress('wa', 'kq', 'wdn', 'dz')
             + ['w = ple*wdn + (1 - ple)*wa'])

# At a postsynaptic spike.
# A same-millisecond pre with the lower index went first, so Kpre already holds its spike; one with the higher index was handled in the presynaptic pathway.
POST_PLAST = (MASKS + ['cq = spiked_pre*int(i > j)',
                       'lower = spiked_pre*int(i < j)',
                       'kpa = fround(Kpre_pre + 1)',
                       'kpv = lower*kpa + (1 - lower)*Kpre_pre']
              + potentiate('w', 'kpv', 'Kslow_post', 'wpt', 'pq')
              + ['ap = ple*(1 - cq)', 'w = ap*wpt + (1 - ap)*w'])

RESET_PLAST = '\n'.join(['Kpre = fround(Kpre + 1)', 'Kpost = fround(Kpost + 1)',
                         'Kslow = hasSlow*fround(Kslow + 1)', 'spkAcc = fround(spkAcc + 1)'])
DECAY = '\n'.join(['Kpre = fround(Kpre*decS)', 'Kpost = fround(Kpost*decM)', 'Kslow = fround(Kslow*decY)'])
STP_RECOVER = '\n'.join(['xr1 = 1 - stpX', 'xr2 = xr1*stpDx', 'stpX = fround(1 - xr2)',
                         'rr1 = stpR - stpU', 'rr2 = rr1*stpDr', 'stpR = fround(stpU + rr2)'])
CONSOLIDATE = '\n'.join([
    'gc = consOn*every(t/ms, consStep)',
    'cmk = gc*pl*int(rcons > 0)*int(w > 0)',
    'hlf = rconsw*0.5',
    'q1 = w - wRef', 'q2 = rconsp*wRef', 'q3 = hlf - wRef', 'q4 = q2*q3',
    'q5 = rconsw - wRef', 'q6 = q4*q5', 'q7 = q1 - q6', 'q8 = consRate*q7',
    'nr = wRef + q8', 'nrc = clip(nr, 0, rconsw)',
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
    if noise:
        ns['noise'] = noise_function(int(case['seed']))
    if plast is not None:
        for key in ('kRef', 'iAlpha', 'decS', 'decM', 'decY', 'hasSlow', 'commitOn', 'sEta', 'iRho',
                    'scaleOn', 'rhoMode', 'calMs', 'calSecs', 'tauSum', 'consOn', 'consStep', 'consRate'):
            ns[key] = float(plast[key])
    else:
        for key in ('rhoMode', 'commitOn'):
            ns[key] = 0.0
    if stp is not None:
        ns.update({'stpU': float(stp['U']), 'stpScale': float(stp['scale']),
                   'stpDx': float(stp['dx']), 'stpDr': float(stp['dr'])})

    defaultclock.dt = dt*ms
    group_args = {'threshold': 'spiked > 0.5', 'namespace': ns}
    if plast is not None:
        group_args['reset'] = RESET_PLAST
    G = NeuronGroup(n, neuron_model(syn, K, len(stim), len(noise), ext, pool, graded, dynamic), **group_args)
    G.run_regularly(step_code(syn, K, len(stim), len(noise), ext, pool, graded, stp, dt), when='groups')
    objects = [G]
    # after the step, in the engine's order: trace decay, consolidation, short-term recovery, then (at the clock tick) scaling and calibration
    if plast is not None:
        objects.append(G.run_regularly(DECAY, when='end', order=0))
        objects.append(G.run_regularly(CALIBRATE, when='end', order=4))
    if stp is not None:
        objects.append(G.run_regularly(STP_RECOVER, when='end', order=2))
    rows = np.asarray(case['rows'], dtype=np.float64)
    for name, col in zip(['Cm', 'kk', 'vr', 'vt', 'vpeak', 'aa', 'bb', 'cc', 'dd'], rows.T):
        setattr(G, name, col)
    G.bias = case['bias']
    G.v = case['v0']
    G.u = case['u0']
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
                model = '\n'.join(['w : 1', 'wsent : 1', 'wRef : 1', 'pl : 1 (constant)']
                                  + ['%s : 1 (constant)' % r for r in RULE_VARS])
                emit = '\n'.join(PRE_PLAST) if plast is not None else 'wsent = w*sfv_pre'
                S = Synapses(G, G, model, on_pre={'pre': emit, 'deliver': deliver},
                             on_post='\n'.join(POST_PLAST) if plast is not None else None,
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
                if rules is not None:
                    # a frozen synapse (rule 0) never uses its row, but the masks multiply the unused branch by 0 and 0 times NaN is NaN, so row 0's zero wmax (a division) is replaced by a harmless one
                    safe = rules.copy()
                    safe[0] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 10]
                    table = safe[rule[idx]]
                    for x, r in enumerate(RULE_VARS):
                        setattr(S, r, table[:, x])
                S.pre.delay = 0*ms
                S.deliver.delay = (delay[idx] - 1)*ms
                S.pre.order = -2
                S.deliver.order = -1
                if plast is not None:
                    S.post.order = 1
                    if plast['consOn']:
                        objects.append(S.run_regularly(CONSOLIDATE, when='end', order=1))
                    if plast['scaleOn']:
                        objects.append(S.run_regularly(SCALE, when='end', order=3))
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

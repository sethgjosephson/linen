"""The Potjans and Diesmann 2014 microcircuit on Brian 2, as they published it.

The reference numbers for the replication (RESULTS.md) come from
their model run on this machine rather than from reading their figure: the
paper's text gives the excitatory rates exactly and the inhibitory ones only
as a figure. Everything below is from Table 5 of the paper and the NEST
reference implementation's network_params.py (the probability matrix and
K_bg were verified against it).

    .venv-analysis/Scripts/python.exe tools/pd2014_brian.py --seconds 12 --seed 1 --out runs/pd/brian-s1.json
    .venv-analysis/Scripts/python.exe tools/pd2014_brian.py --scale 0.05 --seconds 2   (a smoke test: N scaled, no compensation)

Leaky integrate-and-fire, exponential postsynaptic currents, a fixed total number
of synapses per population pair (their K formula, multapses allowed):
    C dv/dt = -g_L (v - E_L) + I_syn + I_bg,   g_L = C / tau_m
    dI_syn/dt = -I_syn / tau_syn
    tau_m 10 ms, C 250 pF, E_L -65 mV, threshold -50 mV, reset -65 mV,
    refractory 2 ms, tau_syn 0.5 ms, w 87.8 +- 8.8 pA, g -4, the L4e to
    L2/3e weight doubled, delays 1.5 +- 0.75 ms (E) and 0.8 +- 0.4 ms (I),
    background 8 Hz Poisson over K_bg synapses of strength w per cell,
    initial potentials -58 +- 10 mV (the reference implementation), dt 0.1 ms.
Measures, on the spikes after the first two seconds: mean rate per
population, the mean over cells of the ISI coefficient of variation (cells
with at least four spikes), and synchrony as the variance over mean of the
population spike count in 3 ms bins (their Figure 6 definition).
"""
import argparse, json, math, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scale', type=float, default=1.0, help='population sizes times this (no compensation; for smoke tests only)')
ap.add_argument('--seconds', type=float, default=12.0)
ap.add_argument('--settle', type=float, default=2.0, help='seconds dropped from every measure')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--dt', type=float, default=0.1, help='ms')
ap.add_argument('--out', default='')
args = ap.parse_args()

from brian2 import (NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network, prefs, defaultclock, seed,
                    ms, mV, pA, nS, pF, Hz, second)
prefs.codegen.target = 'numpy'
defaultclock.dt = args.dt*ms
seed(args.seed)
np.random.seed(args.seed)

POPS = ['L2/3e', 'L2/3i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']
N_FULL = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948]
K_BG = [1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100]
# rows = target, columns = source (src/scenarios.js PD_CONN, verified against it)
P = np.array([
  [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0,    0.0076, 0.0],
  [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0,    0.0042, 0.0],
  [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0],
  [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0,    0.1057, 0.0],
  [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
  [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0],
  [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
  [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
N = [max(1, int(round(n*args.scale))) for n in N_FULL]
W, W_SD, G = 87.8, 8.8, -4.0
D_E, D_E_SD, D_I, D_I_SD = 1.5, 0.75, 0.8, 0.4
NU_BG = 8.0

eqs = '''
dv/dt = (-(v - E_L)/tau_m) + I_syn/C : volt (unless refractory)
dI_syn/dt = -I_syn/tau_syn : amp
'''
G_all = NeuronGroup(sum(N), eqs, threshold='v > -50*mV', reset='v = -65*mV', refractory=2*ms, method='exact',
                    namespace={'E_L':-65*mV, 'tau_m':10*ms, 'C':250*pF, 'tau_syn':0.5*ms})
G_all.v = (-58 + 10*np.random.randn(sum(N)))*mV
G_all.I_syn = 0*pA
starts = np.concatenate([[0], np.cumsum(N)])
pops = [G_all[starts[k]:starts[k+1]] for k in range(8)]

t0 = time.time()
syn = []
nsyn = 0
for t in range(8):
    for s in range(8):
        p = P[t][s]
        if p <= 0: continue
        S = Synapses(pops[s], pops[t], 'w : amp', on_pre='I_syn_post += w', name='S_%d_%d' % (s, t))
        # the paper draws a fixed total number of synapses per population pair, K = log(1 - p) / log(1 - 1 / (N_pre N_post)), as random pairs with multapses and autapses allowed (NEST fixed_total_number), which is a few percent above the Bernoulli expectation p N_pre N_post
        K = int(round(math.log(1 - p)/math.log(1 - 1.0/(N[s]*N[t]))))
        if K <= 0: continue
        S.connect(i=np.random.randint(0, N[s], K), j=np.random.randint(0, N[t], K))
        n = len(S)
        if n == 0: continue
        nsyn += n
        exc = s % 2 == 0
        wv = np.random.normal(W, W_SD, n)
        wv = np.clip(wv, 0, None)
        if s == 2 and t == 0: wv = wv*2                # L4e to L2/3e
        if not exc: wv = wv*G
        S.w = wv*pA
        d = np.random.normal(D_E if exc else D_I, D_E_SD if exc else D_I_SD, n)
        d = np.clip(d, args.dt, None)
        S.delay = d*ms
        syn.append(S)
print('synapses: %s, built in %.0f s' % (format(nsyn, ','), time.time() - t0), flush=True)

bg = [PoissonInput(pops[k], 'I_syn', K_BG[k], NU_BG*Hz, W*pA) for k in range(8)]
mons = [SpikeMonitor(pops[k], name='M%d' % k) for k in range(8)]
net = Network(G_all, *syn, *bg, *mons)
t0 = time.time()
net.run(args.seconds*second, report='text', report_period=60*second)
wall = time.time() - t0
print('ran %.1f s of biology in %.0f s' % (args.seconds, wall), flush=True)

settle = args.settle*1000.0; stop = args.seconds*1000.0; span = (stop - settle)/1000.0
out = { 'model':'Potjans and Diesmann 2014 on Brian 2', 'scale':args.scale, 'cells':int(sum(N)), 'synapses':int(nsyn),
        'seconds':args.seconds, 'settle':args.settle, 'seed':args.seed, 'dt':args.dt, 'wall':round(wall, 1),
        'brian2':__import__('brian2').__version__, 'populations':{} }
for k in range(8):
    tm = np.asarray(mons[k].t/ms); im = np.asarray(mons[k].i)
    keep = tm >= settle
    tm, im = tm[keep], im[keep]
    rate = len(tm)/max(1, N[k])/span
    cvs = []
    if len(tm):
        order = np.argsort(im, kind='stable'); im2 = im[order]; tm2 = tm[order]
        bounds = np.flatnonzero(np.diff(im2)) + 1
        for cell in np.split(tm2, bounds):
            if len(cell) >= 4:
                d = np.diff(np.sort(cell))
                if d.mean() > 0: cvs.append(d.std()/d.mean())
    edges = np.arange(settle, stop + 3.0, 3.0)
    counts = np.histogram(tm, edges)[0]
    sync = float(counts.var()/counts.mean()) if counts.mean() > 0 else float('nan')
    out['populations'][POPS[k]] = { 'cells':int(N[k]), 'rate':round(float(rate), 3),
                                    'cv':round(float(np.mean(cvs)), 3) if cvs else None, 'cvCells':len(cvs),
                                    'sync':round(sync, 3) }
    print('  %-6s %6d cells  %7.2f Hz  CV %s  sync %.2f' % (POPS[k], N[k], rate, ('%.2f' % np.mean(cvs)) if cvs else '-', sync))
if args.out:
    import os
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print('wrote', args.out)

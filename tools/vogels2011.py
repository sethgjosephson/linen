"""The inhibitory plasticity program of Vogels et al. (2011), transcribed.

Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C. and Gerstner, W. (2011).
Inhibitory plasticity balances excitation and inhibition in sensory pathways
and memory networks. Science 334:1569. The MATLAB program the authors printed
in the supplementary material (inhib_plasticity.m, ModelDB 143751): one
conductance-based integrate-and-fire cell, 800 excitatory and 200 inhibitory
input spike trains in eight signal groups, the inhibitory synapses plastic.

    on an inhibitory presynaptic spike:  w += post - alpha,  w >= 0
    on a postsynaptic spike:             w += pre (every inhibitory synapse)

with pre and post traces that add eta per spike and decay with 20 ms, and
alpha = 0.25 eta. Transcribed step for step at the program's dt of 0.1 ms,
101 runs of 1.5 s as its loop runs them, including the membrane starting
each run from V(1). The per-input loop is vectorized: within one step the
inputs do not read each other's updates. Random draws are NumPy's, not
MATLAB's. tools/published.mjs reads the per-run output rate.

    .venv-analysis/Scripts/python.exe tools/vogels2011.py out.json [seed]
"""
import json
import math
import sys

import numpy as np


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 2011)

    eta = 0.001
    alpha = 0.25*eta
    tauPlasticity = 20.0
    duration = 1500
    dt = 0.1
    NRuns = 100
    tRef = 5.0
    gBarEx = 0.014
    gBarIn = 0.035
    VRest = -60.0
    Vth = -50.0
    taumem = 20.0
    EAMPA = 0.0
    EGABA = -80.0
    tauEx = 5.0
    tauIn = 10.0
    noisetau = 50.0
    Backgroundrate = 5*dt/1000
    ApproximateNormalizingFactor = 0.03
    Maxrate = 500*dt/1000
    NSigs = 8
    NCells = 1000
    ExFrac = 0.8
    ExGroupsize = (NCells*ExFrac)/NSigs
    InGroupsize = round((NCells*(1 - ExFrac))/NSigs)
    expGEx = math.exp(-dt/tauEx)
    expGIn = math.exp(-dt/tauIn)
    expPlasticity = math.exp(-dt/tauPlasticity)
    expnoise = math.exp(-dt/noisetau)
    steps = int(round(duration/dt))           # length(Timevector), Timevector=(0.1:dt:duration)

    # InputGroups, 1-based as printed
    InputGroup = np.zeros(NCells + 1, dtype=np.int64)
    temptype = 0
    for i in range(1, NCells + 1):
        if i <= NCells*ExFrac:
            if (i - 1) % ExGroupsize == 0:
                temptype += 1
            InputGroup[i] = temptype
        else:
            if i % InGroupsize == 0:
                temptype -= 1
            InputGroup[i] = -temptype
    InputGroup[1000] = InputGroup[999]
    group = np.abs(InputGroup[1:]) - 1        # 0-based signal index per train
    exc = InputGroup[1:] > 0
    inh = ~exc

    Synapse = np.ones(NCells)
    for i in range(800):
        Synapse[i] = 0.3 + (1.1/(1 + (InputGroup[i + 1] - 5)**4)) + rng.random()*0.1
    Synapse[800:] = 0.1

    gEx = 0.0
    gIn = 0.0
    gLeak = 1.0
    V1 = VRest                                # V(1), what every run starts from
    post = 0.0
    pre = np.zeros(NCells)
    FilteredWhiteNoise = np.zeros(NSigs)
    InputSpikeRefr = np.zeros(NCells)
    tolos = 0.0
    tRunning = 0.0
    rates, winh = [], []
    runcount = 0
    going = True
    while going:
        runcount += 1
        OutputSpikeCount = 0
        V = V1
        for t in range(2, steps + 1):
            tRunning += dt
            gEx *= expGEx
            gIn *= expGIn
            pre *= expPlasticity                # only 801:NCells in the program; exc entries stay 0
            post *= expPlasticity
            re = rng.random(NSigs) - 0.5
            FilteredWhiteNoise = re - (re - FilteredWhiteNoise)*expnoise
            Input = Backgroundrate + np.maximum(0, Maxrate*FilteredWhiteNoise)/ApproximateNormalizingFactor
            fire = (rng.random(NCells) < Input[group]) & (InputSpikeRefr <= 0)
            fe = fire & exc
            fi = fire & inh
            gEx += gBarEx*Synapse[fe].sum()
            gIn += gBarIn*Synapse[fi].sum()
            pre[fi] += eta
            Synapse[fi] = np.maximum(0.0, Synapse[fi] + post - alpha)
            InputSpikeRefr[~fire] -= dt
            InputSpikeRefr[fire] = tRef
            if (tRunning - tolos) < tRef:
                V = VRest
            else:
                gTot = gLeak + gEx + gIn
                tauEff = taumem/gTot
                VInf = (gLeak*VRest + gEx*EAMPA + gIn*EGABA)/gTot
                V = VInf + (V - VInf)*math.exp(-dt/tauEff)
            if V > Vth:
                tolos = tRunning
                if t == 2:
                    V1 = 0.0                    # V(t-1)=0 lands on V(1)
                V = VRest
                OutputSpikeCount += 1
                post += eta
                Synapse[800:] += pre[800:]
        rates.append(OutputSpikeCount/duration*1000)
        winh.append(float(Synapse[800:].mean()))
        if runcount > NRuns:
            going = False
    with open(out, 'w') as f:
        json.dump({'rates': rates, 'winh': winh, 'alpha': alpha, 'eta': eta, 'tau': tauPlasticity}, f)


if __name__ == '__main__':
    main()

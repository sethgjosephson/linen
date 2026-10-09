"""The network program printed in Izhikevich (2003), transcribed line for line.

Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions
on Neural Networks 14:1569. The paper prints a MATLAB program for 1000
randomly coupled neurons (800 excitatory, 200 inhibitory) and reports what it
produces: Poisson-like firing around 8 Hz with occasional synchronized
episodes in the alpha (10 Hz) and gamma (40 Hz) ranges. This is that program
in NumPy, in double precision, with its random draws replaced by numbers
read from files, so tools/published.mjs can give the reference engine exactly
the same neurons, weights and thalamic input:

    re, ri   uniform draws per excitatory and inhibitory neuron (float64)
    S        the weight matrix, S[i, j] from neuron j to neuron i (float32)
    input    the thalamic input per millisecond per neuron (float32)

The program's loop is kept as printed: at each millisecond the neurons whose
v reached 30 are reset, their weights are added to the input, and v takes two
half steps and u one step. A spike found at iteration t crossed during
iteration t - 1, so it is recorded at t - 1, the step the engine records it at.

    .venv-analysis/Scripts/python.exe tools/izh2003.py case.json out.json
"""
import json
import sys
import time

import numpy as np


def main():
    with open(sys.argv[1]) as f:
        case = json.load(f)
    Ne, Ni, T = int(case['Ne']), int(case['Ni']), int(case['ms'])
    N = Ne + Ni
    re = np.fromfile(case['re'], dtype=np.float64)
    ri = np.fromfile(case['ri'], dtype=np.float64)
    a = np.concatenate([0.02*np.ones(Ne), 0.02 + 0.08*ri])
    b = np.concatenate([0.2*np.ones(Ne), 0.25 - 0.05*ri])
    c = np.concatenate([-65 + 15*re**2, -65*np.ones(Ni)])
    d = np.concatenate([8 - 6*re**2, 2*np.ones(Ni)])
    S = np.fromfile(case['S'], dtype=np.float32).astype(np.float64).reshape(N, N)
    thalamic = np.fromfile(case['input'], dtype=np.float32).reshape(T, N)

    v = -65*np.ones(N)                                   # initial values of v
    u = b*v                                              # initial values of u
    times = [[] for _ in range(N)]
    t0 = time.time()
    for t in range(T):
        I = thalamic[t].astype(np.float64)               # thalamic input
        fired = np.flatnonzero(v >= 30)                  # indices of spikes
        for i in fired:
            times[i].append(t - 1)
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]
        I = I + S[:, fired].sum(axis=1)
        v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)      # step 0.5 ms
        v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)      # for numerical
        u = u + a*(b*v - u)                              # stability
    for i in np.flatnonzero(v >= 30):                    # the last step's crossings
        times[i].append(T - 1)
    with open(sys.argv[2], 'w') as f:
        json.dump({'times': times, 'wallS': round(time.time() - t0, 3)}, f)


if __name__ == '__main__':
    main()

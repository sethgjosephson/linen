"""The cell classes of Izhikevich (2003), Fig. 2, at the figure's resolution.

Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions
on Neural Networks 14:1569. Fig. 2 shows each class's response to "a step of
dc-current I = 10" at a "time resolution" of 0.1 ms. The equations

    v' = 0.04 v^2 + 5 v + 140 - u + I,   u' = a (b v - u),
    if v >= 30 mV: v <- c, u <- u + d

are stepped here by forward Euler at 0.1 ms, v first and u from the updated
v, then the spike test, which is how the author's own figure code steps them
(figure1.m on izhikevich.org, at 0.2 to 0.25 ms). tools/published.mjs runs the
same cells on the reference engine at its 1 ms and compares the two.

Case file (JSON): ms, and cells, each with a, b, c, d, v0 (u starts at b v0),
bias (a holding current from t = 0) and steps (pieces of added current, t0 to
t1 in ms). Writes each cell's spike times in ms and its potential at the end
of every whole millisecond.

    .venv-analysis/Scripts/python.exe tools/izh2003_cells.py case.json out.json
"""
import json
import sys

import numpy as np


def main():
    with open(sys.argv[1]) as f:
        case = json.load(f)
    cells, ms, dt = case['cells'], int(case['ms']), 0.1
    n = len(cells)
    a = np.array([c['a'] for c in cells], dtype=np.float64)
    b = np.array([c['b'] for c in cells], dtype=np.float64)
    c = np.array([c['c'] for c in cells], dtype=np.float64)
    d = np.array([c['d'] for c in cells], dtype=np.float64)
    bias = np.array([c['bias'] for c in cells], dtype=np.float64)
    v = np.array([c['v0'] for c in cells], dtype=np.float64)
    u = b*v
    piece_cell = np.array([i for i, cl in enumerate(cells) for _ in cl['steps']], dtype=np.int64)
    piece_t0 = np.array([p['t0'] for cl in cells for p in cl['steps']], dtype=np.float64)
    piece_t1 = np.array([p['t1'] for cl in cells for p in cl['steps']], dtype=np.float64)
    piece_amp = np.array([p['amp'] for cl in cells for p in cl['steps']], dtype=np.float64)

    times = [[] for _ in range(n)]
    vms = np.zeros((ms, n))
    per_ms = int(round(1/dt))
    for k in range(ms*per_ms):
        t = k*dt
        I = bias.copy()
        if len(piece_cell):
            on = (piece_t0 <= t + 1e-9) & (t + 1e-9 < piece_t1)
            np.add.at(I, piece_cell[on], piece_amp[on])
        v = v + dt*(0.04*v*v + 5*v + 140 - u + I)
        u = u + dt*a*(b*v - u)
        fired = np.flatnonzero(v >= 30)
        for i in fired:
            times[i].append(round(t + dt, 6))
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]
        if (k + 1) % per_ms == 0:
            vms[(k + 1)//per_ms - 1] = v
    with open(sys.argv[2], 'w') as f:
        json.dump({'times': times, 'v': vms.T.tolist()}, f)


if __name__ == '__main__':
    main()

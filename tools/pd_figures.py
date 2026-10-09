"""Figures for RESULTS.md from the replication's run folders: a raster of the
eight populations (the paper's Figure 6A layout, a sample of cells per
population, one second), and the per-population rates of the scene against
the paper's model on Brian 2, with the range across seeds.

    .venv-analysis/Scripts/python.exe tools/pd_figures.py runs/pd figures/pd
"""
import csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else 'runs/pd'
out = sys.argv[2] if len(sys.argv) > 2 else 'figures/pd'
os.makedirs(out, exist_ok=True)
POPS = ['L2/3e', 'L2/3i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']
plt.rcParams.update({ 'font.size':9, 'font.family':'DejaVu Sans', 'axes.spines.top':False, 'axes.spines.right':False })

# ---- the raster: pd-cpu-s1, 300 cells per excitatory population and 100 per inhibitory, 1 s
folder = os.path.join(src, 'pd-cpu-s1')
pop_of = {}
with open(os.path.join(folder, 'cells.csv'), newline='') as f:
    for r in csv.DictReader(f): pop_of[int(r['row'])] = r['population']
per = { p:[] for p in POPS }
with open(os.path.join(folder, 'spikes.txt')) as f:
    for row, line in enumerate(f):
        p = pop_of.get(row)
        if p in per: per[p].append(np.array([float(x) for x in line.split()]) if line.strip() else np.zeros(0))
T0, T1 = 5000.0, 6000.0
fig, ax = plt.subplots(figsize=(6.5, 5.5))
y = 0; ticks = []; labels = []
rng = np.random.default_rng(1)
for p in POPS:
    n = 300 if p.endswith('e') else 100
    idx = rng.choice(len(per[p]), size=min(n, len(per[p])), replace=False)
    start = y
    for i in idx:
        t = per[p][i]; t = t[(t >= T0) & (t < T1)]
        ax.plot((t - T0)/1000.0, np.full(len(t), y), '|', color='#1f4e79' if p.endswith('e') else '#b03a2e', markersize=2.5, markeredgewidth=0.6)
        y += 1
    ticks.append((start + y)/2); labels.append(p); y += 8
ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel('time (s), from 5 s'); ax.set_ylabel('population')
ax.set_title('PD microcircuit on the reference engine, seed 1: a sample of cells', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(out, 'raster.png'), dpi=150); plt.close(fig)

# ---- rates: scene (cpu) against the paper's model, mean and range over seeds
def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(src, pattern)))]
scene = load('pd-cpu-s[0-9]/measures.json'); brian = load('brian-s[0-9].json'); col = load('col-cpu-s[0-9]/measures.json')
def stat(runs, p, key='rate'):
    v = [r['populations'][p][key] for r in runs if p in r['populations'] and r['populations'][p][key] is not None]
    return (np.mean(v), np.min(v), np.max(v)) if v else (np.nan, np.nan, np.nan)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
x = np.arange(len(POPS)); w = 0.27
for k, (runs, label, color) in enumerate([(brian, 'the paper\'s model (Brian 2)', '#777777'), (scene, 'the scene, reference engine', '#1f4e79'), (col, 'the column, in vivo tuning', '#2e7d32')]):
    m = [stat(runs, p) for p in POPS]
    ax.bar(x + (k - 1)*w, [a for a, _, _ in m], w, color=color, label=label,
           yerr=[[a - lo for a, lo, _ in m], [hi - a for a, _, hi in m]], error_kw={ 'elinewidth':0.8, 'capsize':2 })
ax.set_xticks(x); ax.set_xticklabels(POPS); ax.set_ylabel('rate (Hz)'); ax.legend(frameon=False, fontsize=8)
ax.set_title('spontaneous rate per population, mean and range over three seeds', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(out, 'rates.png'), dpi=150); plt.close(fig)
print('wrote', os.path.join(out, 'raster.png'), 'and', os.path.join(out, 'rates.png'))

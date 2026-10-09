"""Per-population measures of a headless run folder (tools/linen.mjs), in the
terms Potjans and Diesmann 2014 report: mean rate, the mean over cells of the
ISI coefficient of variation, and synchrony as the variance over mean of the
population spike count in 3 ms bins. The first --settle seconds are dropped.
Writes measures.json into the folder and prints the table.

    .venv-analysis/Scripts/python.exe tools/pd_measure.py runs/pd/pd-cpu-s1 [--settle 2]
"""
import argparse, csv, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('folder')
ap.add_argument('--settle', type=float, default=2.0)
ap.add_argument('--bin', type=float, default=3.0, help='ms, the synchrony bin')
args = ap.parse_args()

run = json.load(open(os.path.join(args.folder, 'run.json')))
stop = float(run['seconds'])*1000.0
settle = args.settle*1000.0
span = (stop - settle)/1000.0
pop_of = {}
with open(os.path.join(args.folder, 'cells.csv'), newline='') as f:
    for r in csv.DictReader(f):
        pop_of[int(r['row'])] = r['population']
trains = {}
with open(os.path.join(args.folder, 'spikes.txt')) as f:
    for row, line in enumerate(f):
        t = np.array([float(x) for x in line.split()], dtype=np.float64) if line.strip() else np.zeros(0)
        trains.setdefault(pop_of.get(row, '?'), []).append(t[t >= settle])
out = { 'folder':args.folder, 'scene':run['scene'], 'engine':run['engine'], 'seed':run['seed'], 'cells':run['cells'],
        'synapses':run['synapses'], 'seconds':run['seconds'], 'settle':args.settle, 'commit':run.get('commit'), 'populations':{} }
edges = np.arange(settle, stop + args.bin, args.bin)
print('%s on %s, seed %s: %s cells, %s synapses, %.0f s after a %.0f s settle' % (run['scene'], run['engine'], run['seed'],
      format(run['cells'], ','), format(run['synapses'], ','), span, args.settle))
print('  population   cells      Hz     CV   sync')
for pop, per in trains.items():
    n = len(per)
    rate = sum(len(t) for t in per)/n/span
    cvs = []
    for t in per:
        if len(t) >= 4:
            d = np.diff(t)
            if d.mean() > 0: cvs.append(d.std()/d.mean())
    allt = np.concatenate(per) if per else np.zeros(0)
    counts = np.histogram(allt, edges)[0]
    sync = float(counts.var()/counts.mean()) if counts.mean() > 0 else float('nan')
    out['populations'][pop] = { 'cells':n, 'rate':round(float(rate), 3), 'cv':round(float(np.mean(cvs)), 3) if cvs else None,
                                'cvCells':len(cvs), 'sync':round(sync, 3) }
    print('  %-10s %6d %7.2f %6s %6.2f' % (pop, n, rate, ('%.2f' % np.mean(cvs)) if cvs else '-', sync))
json.dump(out, open(os.path.join(args.folder, 'measures.json'), 'w'), indent=1)

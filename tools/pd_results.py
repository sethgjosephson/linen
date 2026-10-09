"""The replication tables: every population's rate, ISI CV and synchrony from
the overnight run folders, as the mean over seeds with the range, one table
per engine and one for the paper's model on Brian 2, in Markdown.

    .venv-analysis/Scripts/python.exe tools/pd_results.py runs/pd > runs/pd/tables.md
"""
import glob, json, os, sys

folder = sys.argv[1] if len(sys.argv) > 1 else 'runs/pd'
POPS = ['L2/3e', 'L2/3i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

def span(vals, digits=2):
    v = [x for x in vals if x is not None]
    if not v: return '-'
    m = sum(v)/len(v)
    if len(v) == 1: return ('%.' + str(digits) + 'f') % m
    return ('%.' + str(digits) + 'f (%.' + str(digits) + 'f to %.' + str(digits) + 'f)') % (m, min(v), max(v))

def table(title, runs, note):
    if not runs: return
    print('### ' + title + '\n')
    print(note + ' ' + str(len(runs)) + ' seed' + ('' if len(runs) == 1 else 's') + ': '
          + ', '.join(str(r.get('seed')) for r in runs) + '. '
          + '%s cells, %s synapses.' % (format(runs[0]['cells'], ','), format(runs[0]['synapses'], ',')) + '\n')
    print('| population | cells | rate Hz | ISI CV | synchrony |')
    print('|---|---|---|---|---|')
    for p in POPS:
        rows = [r['populations'].get(p) for r in runs]
        rows = [x for x in rows if x]
        if not rows: continue
        print('| %s | %s | %s | %s | %s |' % (p, format(rows[0]['cells'], ','),
              span([x['rate'] for x in rows]), span([x['cv'] for x in rows]), span([x['sync'] for x in rows])))
    print()

def load(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, pattern))):
        try: out.append(json.load(open(f)))
        except Exception as e: print('could not read ' + f + ': ' + str(e), file=sys.stderr)
    return out

brian = load('brian-s[0-9]*.json')
table('Potjans and Diesmann 2014, their model on Brian 2', brian,
      'Leaky integrate and fire at 0.1 ms, every parameter from the paper, the first 2 s dropped;')
for eng, label in [('cpu', 'the reference engine'), ('cuda', 'the CUDA engine')]:
    runs = load('pd-%s-s*/measures.json' % eng)
    table('The microcircuit scene on ' + label, runs, 'The PD microcircuit scene at 1 ms, the first 2 s dropped;')
table('The scene on a rheobase-matched row (k 6.667 nS/mV, a 2.5 ms time constant at rest), the CUDA engine',
      load('pd-rheobase-cuda-s*/measures.json'), 'The PD microcircuit scene with both cell type rows at k 6.667, at 1 ms, the first 2 s dropped;')
table('Potjans and Diesmann 2014, their model on Brian 2 at a 1 ms step', load('brian-dt1-s[0-9]*.json'),
      'The same transcription with dt 1 ms (delays clipped to the step), the first 2 s dropped;')
for eng, label in [('cpu', 'the reference engine'), ('cuda', 'the CUDA engine')]:
    runs = load('col-%s-s*/measures.json' % eng)
    table('The mouse cortical column (in vivo tuning) on ' + label, runs,
          'The Tab menu column with its thalamic pulse off, at 1 ms, the first 2 s dropped;')

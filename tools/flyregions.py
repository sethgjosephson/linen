"""The fly points file's derived columns, reproducibly.

Reads the MaleCNS points file the connectome conversion wrote (id, x, y, z,
superclass, type, side, nt) and the body annotations feather, and writes
back the columns the fly scenes and the bench read through node settings:

  region  cell types sorted into optic lobe layers and central classes
          (L1, L2, L3, lamina, photoreceptor, medulla, Mi1, Tm3, Mi9, Mi4,
          Tm9, Tm1, Tm2, Tm4, medulla_on, medulla_off, lobula, T4a to T4d,
          T5a to T5d, lobula_plate, visual_projection, visual_centrifugal,
          central, descending, ascending, motor, other). medulla_on and
          medulla_off are the Mi1/Tm3 and Tm1/Tm2/Tm4 cells with a lamina
          input in the connections table; the release's lamina is partial.
  cx cy cz  positions from the optic lobe hex column assignments
          (assignedOlHex1, assignedOlHex2) on a plane, 10 micrometers per
          column, a depth per type; cells without a column keep their soma
          position scaled down, so a points file node reading these columns
          resolves a stimulus by column (the bench's COLUMNS setting)
  hex     1 for a cell with a column assignment

Commas inside fields are replaced by slashes, since the points parser
splits on commas without honoring quotes.

  python tools/flyregions.py [project-fly]
"""
import re, sys, os
import pandas as pd, numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else 'project-fly'
points = os.path.join(root, 'points', 'neurons.csv')
connections = os.path.join(root, 'wiring', 'connections.csv')
annot = os.path.join(root, 'raw', 'body-annotations.feather')

n = pd.read_csv(points, dtype=str, keep_default_na=False)
for c in n.columns: n[c] = n[c].str.replace(',', '/')

def region(r):
    s, t = r['superclass'], r['type']
    if s in ('ol_intrinsic', 'ol_sensory'):
        if t in ('L1', 'L2', 'L3'): return t
        if t in ('Mi1', 'Tm3', 'Mi9', 'Mi4', 'Tm9', 'Tm1', 'Tm2', 'Tm4'): return t
        if t[:3] in ('T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'): return t[:3]
        if re.match(r'^(L[45]$|Lawf|C2$|C3$|T1$|Lai$|Lat)', t): return 'lamina'
        if re.match(r'^(R[1-8]|R7|R8)', t): return 'photoreceptor'
        if re.match(r'^(Mi|Tm|TmY|Dm|Pm|Sm|Cm|MeVP|MeLo|Mt|MeTu|ME_|l-LNv|aMe|dCal|LpMe)', t): return 'medulla'
        if re.match(r'^(T4|T5|LPi|Y\d|Y_|LLPC|LPC|LPT|LPLC|H[12S]|VS|HS|CH|DCH|VCH|Am1|LOP_|LOLP)', t): return 'lobula_plate'
        if re.match(r'^(T2|T3|Li|LC|LT|LoVP|LoVC|Tlp|OA-AL2|MC|OLVC|LO_|CT1)', t): return 'lobula'
        return 'optic_other'
    return {'visual_projection': 'visual_projection', 'visual_projection_tbc': 'visual_projection',
            'visual_centrifugal': 'visual_centrifugal', 'cb_intrinsic': 'central', 'descending_neuron': 'descending',
            'ascending_neuron': 'ascending', 'cb_motor': 'motor', 'vnc_motor': 'motor'}.get(s, 'other')
n['region'] = n.apply(region, axis=1)

# the ON and OFF medulla groups: only the cells with a lamina input in the table
w = pd.read_csv(connections, dtype={0: str, 1: str})
typ = n.set_index('id')['type']
w['pt'] = w.iloc[:, 0].map(typ); w['qt'] = w.iloc[:, 1].map(typ)
on = set(w[(w.pt == 'L1') & (w.qt.isin(['Mi1', 'Tm3']))].iloc[:, 1])
off = set(w[(w.pt.isin(['L2', 'L3'])) & (w.qt.isin(['Tm1', 'Tm2', 'Tm4']))].iloc[:, 1])
n.loc[n.id.isin(on) & n.region.isin(['Mi1', 'Tm3']), 'region'] = 'medulla_on'
n.loc[n.id.isin(off) & n.region.isin(['Tm1', 'Tm2', 'Tm4']), 'region'] = 'medulla_off'

# hex column positions on a plane
try:
    import pyarrow.feather as f
    d = f.read_table(annot).to_pandas()
    d['bodyId'] = d['bodyId'].astype(str)
    h = d.set_index('bodyId')[['assignedOlHex1', 'assignedOlHex2']]
    h1 = n['id'].map(h['assignedOlHex1']); h2 = n['id'].map(h['assignedOlHex2'])
except Exception as e:
    print('no annotations feather (' + str(e) + '); hex columns left as soma positions')
    h1 = pd.Series([np.nan] * len(n)); h2 = h1
has = h1.notna()
types = sorted(n['type'].unique()); depth = {t: i % 40 for i, t in enumerate(types)}
x = n['x'].astype(float); y = n['y'].astype(float); z = n['z'].astype(float)
cx = np.where(has, (h1.astype(float).fillna(0) + h2.astype(float).fillna(0)/2)*10, x/1000)
cy = np.where(has, h2.astype(float).fillna(0)*0.8660254*10, y/1000)
cz = np.where(has, n['type'].map(depth).astype(float)*2, z/1000)
n['cx'] = np.round(cx, 3); n['cy'] = np.round(cy, 3); n['cz'] = np.round(cz, 3); n['hex'] = np.where(has, '1', '0')
n.to_csv(points, index=False)
print('wrote', points, len(n), 'rows;', int(has.sum()), 'with a column')
print(n['region'].value_counts().to_dict())

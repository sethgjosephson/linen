// Reading a sweep: the file names the runs it made, and a chart of it is one line per variant and condition with the spread across the replicates.
// The grouping is the part that can be wrong quietly, since a mean taken across two conditions is a number about nothing.
import { ok, report } from '../tools/harness.mjs';
import { listSweepTags, readSweep } from './runlist.js';

// a store like src/store.js: list a folder, read a file
function fakeStore(files){
  const enc = new TextEncoder();
  return {
    list: async dir => Object.keys(files).filter(p => p.startsWith(dir + '/'))
      .map(p => p.slice(dir.length + 1))
      .map(rest => rest.includes('/') ? { name:rest.split('/')[0], dir:true } : { name:rest, dir:false })
      .filter((e, i, a) => a.findIndex(x => x.name === e.name) === i),
    read: async p => { if(!(p in files)) throw new Error('no such file ' + p);
      return enc.encode(JSON.stringify(files[p])); },
  };
}
const rows = simMin => [{ simMin, readout:{ pops:{ core:{ decode:0.4 } } } }];
const files = {
  'runs/sweep-20260101T0101.json': { format:'np-sweep-2', stamp:'20260101T0101', replicates:2,
    variants:['as is', 'triplet x2'], conditions:['paired', 'scrambled'], hours:1, engine:'cpu',
    rows:[
      { variant:'as is', cond:'paired', rep:1, tag:'run1' },
      { variant:'as is', cond:'paired', rep:2, tag:'run2' },
      { variant:'as is', cond:'scrambled', rep:1, tag:'run3' },
      { variant:'triplet x2', cond:'paired', rep:1, tag:'run4' },
      { variant:'triplet x2', cond:'paired', rep:2, tag:'run5', failed:'engine gone' },
      { variant:'triplet x2', cond:'paired', rep:3, tag:'run6' },   // no metrics file
    ] },
  'runs/sweep-20251231T2359.json': { format:'np-sweep-2', stamp:'20251231T2359', rows:[] },
  'runs/notasweep.json': { nothing:true },
  'runs/run1/metrics.json': rows(5),
  'runs/run2/metrics.json': rows(5),
  'runs/run3/metrics.json': rows(5),
  'runs/run4/metrics.json': rows(5),
};
const ST = fakeStore(files);

const tags = await listSweepTags(ST);
ok('the sweeps of a folder are listed newest first, and nothing else is',
  JSON.stringify(tags) === JSON.stringify(['sweep-20260101T0101', 'sweep-20251231T2359']), JSON.stringify(tags));

const sw = await readSweep(ST, 'sweep-20260101T0101');
const labels = sw.groups.map(g => g.label);
ok('a group per variant and condition, in the order the sweep ran them',
  JSON.stringify(labels) === JSON.stringify(['as is · paired', 'as is · scrambled', 'triplet x2 · paired']),
  JSON.stringify(labels));
ok('a group holds the histories of its runs', sw.groups[0].histories.length === 2
  && sw.groups[0].histories[0][0].simMin === 5, JSON.stringify(sw.groups[0].tags));
ok('a failed run is not counted as a replicate',
  !sw.groups[2].tags.includes('run5'), JSON.stringify(sw.groups[2].tags));
ok('a run whose metrics are missing leaves the group smaller rather than empty',
  sw.groups[2].tags.length === 2 && sw.groups[2].histories.length === 1,
  sw.groups[2].tags.length + ' tags, ' + sw.groups[2].histories.length + ' histories');
ok('the sweep carries what it was', sw.stamp === '20260101T0101' && sw.replicates === 2 && sw.engine === 'cpu');

const plain = await readSweep(fakeStore({ 'runs/sweep-x.json': { rows:[{ cond:'as configured', rep:1, tag:'r' }] },
  'runs/r/metrics.json': rows(2) }), 'sweep-x');
ok('a sweep of one condition and no variants is one group, named for what it is',
  plain.groups.length === 1 && plain.groups[0].label === 'all runs', JSON.stringify(plain.groups.map(g => g.label)));

const empty = await readSweep(ST, 'sweep-20251231T2359');
ok('a sweep that recorded nothing has no groups rather than an empty one', empty.groups.length === 0);

report('runlist');

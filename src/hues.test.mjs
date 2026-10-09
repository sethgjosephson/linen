// Population colors.
// The point of deriving them from the tag list rather than hashing each name is control over the gaps: a hash gives a stable color per name and no promise at all about how close two of them land.
// Measured on one scene's eleven populations, hashing put the closest pair three degrees apart, which is not a color difference.

import { tagHues, hueCss, hueRgb, lockHues } from './nodes.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond) pass++;
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};
const closest = hues => {
  const h = [...hues].sort((a, b) => a - b);
  let min = 360;
  for(let i = 1; i < h.length; i++) min = Math.min(min, h[i] - h[i-1]);
  // wrap-around: the last and the first are neighbors on a circle
  if(h.length > 1) min = Math.min(min, 360 - h[h.length-1] + h[0]);
  return min;
};

{
  const lab = ['v1e','v1i','a1e','a1i','asse','assi','retina','cochlea'];
  const h = tagHues(lab);
  ok('every tag gets a hue', h.size === 8);
  ok('eight populations stay well apart', closest([...h.values()]) >= 20,
    closest([...h.values()]) + ' degrees');
}
{
  const block = ['v.L2/3e','v.L2/3i','v.L4e','v.L4i','x.L2/3e','x.L4e','a.L4e',
    'lgn','mgb','retina','coch'];
  const h = tagHues(block);
  ok('eleven populations stay apart', closest([...h.values()]) >= 15,
    closest([...h.values()]) + ' degrees');
  // the names most easily confused should get the colors least easily confused, which is what sorting before spacing buys
  const gap = Math.abs(h.get('v.L2/3e') - h.get('v.L2/3i'));
  ok('adjacent names are far apart in hue', Math.min(gap, 360-gap) > 90,
    Math.min(gap, 360-gap) + ' degrees between v.L2/3e and v.L2/3i');
}
{
  const a = tagHues(['b','a','c']), b = tagHues(['c','b','a']);
  ok('order of the input does not matter',
    a.get('a') === b.get('a') && a.get('b') === b.get('b') && a.get('c') === b.get('c'));
  ok('duplicates collapse', tagHues(['a','a','b']).size === 2);
  ok('blanks are dropped', tagHues(['a','','b']).size === 2);
  ok('an object of tags works too', tagHues({ 1:'a', 2:'b' }).size === 2);
}
{
  const rgb = hueRgb(0);
  ok('rgb is a 0..1 triple', rgb.length === 3 && rgb.every(v => v >= 0 && v <= 1),
    JSON.stringify(rgb));
  ok('css is the app palette saturation', hueCss(120) === 'hsl(120,75%,62%)', hueCss(120));
}

// locked hues: a saved tag keeps its hue whatever else arrives; a new tag takes a free one, at least twelve degrees from every hue in use
{
  const saved = {};
  const first = lockHues(saved, ['v.a', 'v.b', 'v.c']);
  const second = lockHues(saved, ['v.c', 'v.b', 'v.a', 'v.d', 'v.e']);
  ok('saved tags keep their hue when the set grows', ['v.a', 'v.b', 'v.c'].every(t => first.get(t) === second.get(t)));
  ok('new tags are assigned and saved', second.has('v.d') && second.has('v.e') && saved['v.d'] === second.get('v.d'));
  const all = [...second.values()];
  const minGap = Math.min(...all.flatMap((h, i) => all.slice(i + 1).map(k => Math.min(Math.abs(h - k), 360 - Math.abs(h - k)))));
  ok('every locked hue sits at least twelve degrees from the others', minGap >= 12, 'min gap ' + minGap);
  ok('a tag alone keeps its saved hue on a smaller set', lockHues(saved, ['v.e']).get('v.e') === second.get('v.e'));
}

console.log(fail ? (fail + ' failed, ' + pass + ' passed') : 'all passed (' + pass + ' checks)');
if(fail) process.exit(1);

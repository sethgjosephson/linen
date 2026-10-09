// The sector histogram: conjunctive cells counted by angle about the population's own center, against every cell, so a ring of six senses can be read for where binding cells gather.
import { sectorHistogram } from './analysis.js';
import { ok, report } from '../tools/harness.mjs';

// twelve cells on a ring of radius 100 about (50, 0, 50), one per 30 degrees
const idx = [], pos = [];
for(let k = 0; k < 12; k++){
  const a = -Math.PI + (k + 0.5) * (2*Math.PI/12);
  pos.push(50 + 100*Math.cos(a), 0, 50 + 100*Math.sin(a)); idx.push(k);
}
const P = Float32Array.from(pos);
{
  const h = sectorHistogram([[0, 1], [6]], idx, P, 12);
  ok('one cell per bin', h.all.every(v => v === 1), h.all.join(','));
  ok('the sets fall in their bins', h.conj[0] === 1 && h.conj[1] === 1 && h.conj[6] === 1 && h.conj.reduce((a, b) => a + b, 0) === 3, h.conj.join(','));
  ok('a cell in two sets counts once', sectorHistogram([[0], [0]], idx, P, 12).cells === 1);
  ok('fractions follow', h.frac[0] === 1 && h.frac[2] === 0);
}
ok('an empty population is null', sectorHistogram([[0]], [], P, 12) === null);
report('sector');

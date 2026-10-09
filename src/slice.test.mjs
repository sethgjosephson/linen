// The slicer box is display only: a box in viewer units that says which cells are drawn.
// These are the pieces the viewer and the scene share.
import { ok, report } from '../tools/harness.mjs';
import { sliceNormalize, sliceDefault, sliceBounds, sliceContains, slicePlanes } from './slice.js';

const pos = Float32Array.from([0,0,0, 100,20,10, 200,40,20, 400,60,30]);
const d = sliceDefault(pos);
ok('the default box is on and covers the lower half along x', d.on && d.center[0] === 100 && d.size[0] === 200, JSON.stringify(d));
ok('and the whole extent on y and z, with a margin', d.size[1] === 62 && d.size[2] === 32);
ok('cells in the lower half are inside, the upper half outside',
  sliceContains(d, 100, 20, 10) && sliceContains(d, 200, 40, 20) && !sliceContains(d, 400, 60, 30));
ok('an empty viewer gets a millimetre cube at the origin', sliceDefault(null).size.join() === '1000,1000,1000');
const b = sliceBounds({ center:[10, 20, 30], size:[4, -6, 0] });
ok('bounds come from center and size, size never under one and never negative',
  b.min.join() === '8,17,29.5' && b.max.join() === '12,23,30.5', JSON.stringify(b));
ok('a scene without a slice has none', sliceNormalize(undefined) === null && sliceNormalize(null) === null);
const n = sliceNormalize({ on:1, center:[1, 2, 3], size:[0, 5, 'x'] });
ok('a saved slice is read back into numbers, a bad size falls back',
  n.on === true && n.center.join() === '1,2,3' && n.size.join() === '1000,1000,1000', JSON.stringify(n));
ok('a good size is kept, absolute and at least one', sliceNormalize({ size:[-3, 0.2, 7] }).size.join() === '3,1,7');
const pl = slicePlanes({ on:true, center:[10, 20, 30], size:[4, 6, 8] });
const keep = (p, q) => pl.every(([a, b, c, k]) => a*p + b*q[0] + c*q[1] + k >= 0);
ok('the six planes keep the inside and cut the outside',
  pl.length === 6 && keep(10, [20, 30]) && keep(12, [23, 34]) && !keep(12.5, [20, 30]) && !keep(10, [20, 35]), JSON.stringify(pl));
ok('a slice that is off cuts nothing', slicePlanes({ on:false, center:[0,0,0], size:[1,1,1] }).every(p => p[3] === 1e12) && slicePlanes(null).length === 6);
report('slice');

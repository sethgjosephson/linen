// Feedback inhibition pools: the wiring resolves pool nodes into one pool per population, normalizes the gain by the pool's size, refuses a cell claimed twice, and carries pools through merge, repeat and pick the way projections and rules travel.
// The engines are checked against each other in the battery (validate.html, feedback inhibition) and cuda/parity.mjs.

import { ok, threw, report } from '../tools/harness.mjs';
import { wirePools, NODE_DEFS } from './nodes.js';

// three populations with stable tags: 4 RS cells of a, 2 RS of b, 2 FS of c
const pts = () => ({ kind:'points', count:8,
  pos:new Float32Array(24),
  ntype:Uint8Array.from([0,0,0,0, 0,0, 3,3]),
  bias:new Float32Array(8),
  src:Int32Array.from([1,1,1,1, 2,2, 3,3]),
  lidx:Uint32Array.from([0,1,2,3, 0,1, 0,1]),
  tags:{ 1:'tier1.a', 2:'tier2.a', 3:'c' } });

ok('no pool node, no arrays', wirePools(pts()) === null);

let r = wirePools({ ...pts(), pools:[{ tag:'tier1.a', gain:-2, node:9 }] });
ok('pool per cell', [...r.pool].join(',') === '1,1,1,1,0,0,0,0', [...r.pool].join(','));
ok('kick per spike is gain * 1000 / N', Math.abs(r.poolK[1] - (-2*1000/4)) < 1e-6, r.poolK[1]);
ok('index 0 delivers nothing', r.poolK[0] === 0);
ok('the pool is named for the status line', r.poolInfo[0].name === 'tier1.a' && r.poolInfo[0].count === 4);

r = wirePools({ ...pts(), pools:[{ tag:'tier*.a', gain:-1, node:9 }] });
ok('a pattern is one pool per match', [...r.pool].join(',') === '1,1,1,1,2,2,0,0', [...r.pool].join(','));
ok('each pool normalized by its own size', Math.abs(r.poolK[1] - (-250)) < 1e-6 && Math.abs(r.poolK[2] - (-500)) < 1e-6,
  [...r.poolK].join(','));

// the class keys and blank
r = wirePools({ ...pts(), pools:[{ tag:'I', gain:-1, node:9 }] });
ok('I names the inhibitory cells', [...r.pool].join(',') === '0,0,0,0,0,0,1,1', [...r.pool].join(','));
r = wirePools({ ...pts(), pools:[{ tag:'', gain:-1, node:9 }] });
ok('blank names every cell', [...r.pool].every(g => g === 1) && r.poolInfo[0].count === 8);
r = wirePools({ ...pts(), pools:[{ tag:'E', gain:-1, node:9 }, { tag:'c', gain:-3, node:10 }] });
ok('two nodes, two pools', [...r.pool].join(',') === '1,1,1,1,1,1,2,2' && Math.abs(r.poolK[2] - (-1500)) < 1e-6);

// refusals: a cell in two pools, a tag nothing matches, a class with no cells
let msg = threw(() => wirePools({ ...pts(), pools:[{ tag:'tier1.a', gain:-1 }, { tag:'E', gain:-1 }] }));
ok('a cell claimed twice is refused', msg && msg.includes('already in the pool'), msg);
msg = threw(() => wirePools({ ...pts(), pools:[{ tag:'nothing', gain:-1 }] }));
ok('a tag nothing matches is refused', msg && msg.includes('no population matches'), msg);
ok('tag comparison ignores case', wirePools({ ...pts(), pools:[{ tag:'TIER1.A', gain:-1 }] }).poolInfo[0].count === 4);

// the node's compute appends to the stream, and a bypass-free stream through merge and pick keeps the entry
const computed = NODE_DEFS.pool.compute([pts()], { tag:'c', gain:-4 }, { id:7 });
ok('the node appends a pool entry', computed.pools.length === 1 && computed.pools[0].tag === 'c' && computed.pools[0].gain === -4 && computed.pools[0].node === 7);
ok('a positive gain is clamped to zero (inhibition only)', NODE_DEFS.pool.compute([pts()], { tag:'c', gain:3 }, { id:7 }).pools[0].gain === 0);
const merged = NODE_DEFS.gather.compute([computed, NODE_DEFS.pool.compute([pts()], { tag:'tier2.a', gain:-1 }, { id:8 })], { ports:2 });
ok('merge keeps the same cells once and both pools', merged.count === 8 && merged.pools.length === 2, merged.count + ' cells, ' + (merged.pools || []).length + ' pools');

// a numbered repeat renames a pool on one of the stream's own populations per copy, and carries a class, blank or pattern pool as it is
const one = { kind:'points', count:2, pos:new Float32Array(6), ntype:new Uint8Array(2),
  bias:new Float32Array(2), src:Int32Array.from([1,1]), lidx:Uint32Array.from([0,1]), tags:{ 1:'bulb' },
  pools:[{ tag:'bulb', gain:-2, node:5 }, { tag:'E', gain:-1, node:6 }] };
const rep = NODE_DEFS.repeat.compute([one], { copies:3, pops:0, tag:'tier', translate:[100,0,0], rotate:[0,0,0], scale:[1,1,1], pivot:[0,0,0], mirror:0 });
const names = (rep.pools || []).map(p => p.tag);
ok('a repeat makes one pool per numbered copy', names.filter(t => /^tier\d\.bulb$/i.test(t)).length === 3, names.join(' '));
ok('a class pool is carried once', names.filter(t => t === 'E').length === 1, names.join(' '));
// wire the per-copy pools alone: the class pool would claim the same cells a second time, which the wiring refuses
const wired = wirePools({ ...rep, pools:rep.pools.filter(p => p.tag !== 'E') });
ok('the copies are wired as separate pools of the copy size', wired && wired.poolInfo.length === 3 &&
  wired.poolInfo.every(p => p.count === 2), JSON.stringify(wired && wired.poolInfo));
msg = threw(() => wirePools(rep));
ok('a class pool over cells already pooled per copy is refused', msg && msg.includes('already in the pool'), msg);

// the whole wiring, serially, the way the compute worker runs it: the net that comes out carries the arrays, since a stream field left off the copy the wiring sees is a mechanism that computes and is never wired
import { wireConnect } from './nodes.js';
const geo = NODE_DEFS.sphere.compute([], { center:[0,0,0], radius:150 });
const sc = NODE_DEFS.scatter.compute([geo], { count:300, type:0, seed:3, spacing:0, pattern:0, pitch:40, jitter:8, axis:1, tag:'blob' }, { id:1 });
const withPool = NODE_DEFS.pool.compute([sc], { tag:'blob', gain:-2 }, { id:2 });
const net = wireConnect(withPool, { radius:100, sigma:60, prob:0.3, wExc:5, wInh:-5, wdist:0, wsigma:1,
  cluster:0, velocity:200, density:100, nuE:0, nuI:0, seed:1, table:'' }, null);
ok('the wired net carries the pool per cell', net.pool && net.pool.length === 300 && [...net.pool].every(g => g === 1), net.pool && net.pool.length);
ok('the wired net carries the kick per pool', net.poolK && net.poolK.length === 2 && Math.abs(net.poolK[1] + 2000/300) < 1e-6, net.poolK && [...net.poolK]);
const netNo = wireConnect(sc, { radius:100, sigma:60, prob:0.3, wExc:5, wInh:-5, wdist:0, wsigma:1,
  cluster:0, velocity:200, density:100, nuE:0, nuI:0, seed:1, table:'' }, null);
ok('a wiring with no pool node carries no arrays', netNo.pool === undefined && netNo.poolK === undefined);

report('pool');

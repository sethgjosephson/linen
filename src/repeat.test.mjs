// The repeat node: copies of a stream, each an honest population.
import { NODE_DEFS, LIDX_RADIX, SRC_RADIX } from './nodes.js';
import { ok, report } from '../tools/harness.mjs';

// a stream of two scatters, three cells each, as merge would hand it on
const pts = { kind:'points', count:6,
  pos:Float32Array.from([0,0,0, 10,0,0, 20,0,0, 0,5,0, 10,5,0, 20,5,0]),
  ntype:Uint8Array.from([0,0,0, 3,3,3]), bias:new Float32Array(6),
  src:Int32Array.from([7,7,7, 9,9,9]), lidx:Uint32Array.from([0,1,2, 0,1,2]),
  tags:{ 7:'RSexc', 9:'FSinh' } };
const P = { copies:3, pops:1, tag:'', translate:[0,100,0], rotate:[0,0,0], scale:[1,1,1],
  pivot:[0,0,0], mirror:0 };
const compute = p => NODE_DEFS.repeat.compute([pts], { ...P, ...p });

{
  const o = compute({});
  ok('three copies of six cells', o.count === 18);
  ok('copy 0 is the stream as it arrived', o.pos[3] === 10 && o.pos[4] === 0);
  ok('copy 2 is stepped twice', o.pos[(12+1)*3+1] === 200, String(o.pos[(12+1)*3+1]));
  const ids = new Set();
  for(let i=0;i<o.count;i++) ids.add(o.src[i] + ':' + o.lidx[i]);
  ok('every cell has an identity of its own', ids.size === 18);
  ok('copy 0 keeps its source and index times the radix', o.src[1] === 7 && o.lidx[1] === 1*LIDX_RADIX);
  ok('copy 1 is re-indexed', o.lidx[6+1] === 1*LIDX_RADIX + 1);
  ok('shared: one population per scatter', o.tags[7] === 'RSexc' && Object.keys(o.tags).length === 2);
}
{
  const o = compute({ pops:0 });
  ok('numbered, no tag: the population takes the copy number',
     o.tags[7*SRC_RADIX + 1] === 'RSexc1' && o.tags[9*SRC_RADIX + 2] === 'FSinh2', JSON.stringify(o.tags));
}
{
  const o = compute({ pops:0, tag:'col' });
  ok('tagged copies live under sources of their own', o.src[6] === 7*SRC_RADIX + 2);
  ok('and are named copy first, population second',
     o.tags[7*SRC_RADIX + 1] === 'col1.RSexc' && o.tags[9*SRC_RADIX + 3] === 'col3.FSinh',
     JSON.stringify(o.tags));
  ok('the first copy is renamed too', o.src[0] === 7*SRC_RADIX + 1);
}
{
  const o = compute({ copies:2, mirror:2, translate:[0,0,0], pivot:[0,2.5,0] });
  ok('a mirrored copy is reflected about the pivot', o.pos[(6+3)*3+1] === 0 && o.pos[(6+0)*3+1] === 5,
     o.pos[(6+3)*3+1] + ' ' + o.pos[(6+0)*3+1]);
}
{
  const o = compute({ copies:2, rotate:[0,90,0], translate:[0,0,0] });
  ok('a rotation steps about the pivot', Math.abs(o.pos[(6+1)*3+2] + 10) < 1e-3 || Math.abs(o.pos[(6+1)*3+2] - 10) < 1e-3,
     String(o.pos[(6+1)*3+2]));
}
// --- a repeat of a repeat keeps every identity and every name -------------
// A ring of six repeated into two tiers must not put tier1.bulb2 and tier2.bulb1 on one source id, which additive blocks would do.
{
  const ring = compute({ copies:6, pops:0, tag:'bulb', translate:[0,0,0], rotate:[0,60,0] });
  const tiers = NODE_DEFS.repeat.compute([ring], { ...P, copies:2, pops:0, tag:'tier',
    translate:[0,760,0], rotate:[0,30,0], scale:[0.62,0.62,0.62] });
  ok('twelve copies of six cells', tiers.count === 72);
  const ids = new Set();
  for(let i=0;i<tiers.count;i++) ids.add(tiers.src[i] + ':' + tiers.lidx[i]);
  ok('every cell of every tier has its own identity', ids.size === 72, String(ids.size));
  const names = Object.values(tiers.tags);
  ok('twenty-four populations, all named', new Set(names).size === 24, names.length + ' ' + new Set(names).size);
  ok('names read tier first, bulb second', names.includes('tier1.bulb2.RSexc') && names.includes('tier2.bulb1.FSinh'),
     names.slice(0, 4).join(' '));
}

report('repeat');

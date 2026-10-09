// The wiring cache is keyed on what arrives at connect, not on the graph's shape: the same cells in the same order with the same entries hash the same whatever nodes made them.
// A points file tagged by column emits its cells grouped by population in first-seen order, which is the order a fan of population nodes and a merge produces, so expand and collapse hash alike.
// Brain files carry the cell order they were saved on and refuse another.
import { ok, threw, report } from '../tools/harness.mjs';
import { streamHash, NODE_DEFS, setFileReader } from './nodes.js';
import { buildWeightsFile, parseBrainFile, applyBrainWeights, cellOrder } from './brain.js';

const csv = 'id,x,y,z,superclass\na1,0,0,0,A\nb1,10,0,0,B\na2,20,0,0,A\nc1,30,0,0,C\nb2,40,0,0,B\n';
setFileReader(async path => path === 'p.csv' ? new TextEncoder().encode(csv) : null);
const P = { file:'p.csv', columns:'', id:'id', scale:1, up:0, type:0, tag:'t', tagColumn:'superclass', filter:'' };
const file = await NODE_DEFS.pointfile.compute([], P, { id:3 });
ok('the file node groups its cells by population, first seen first', file.ids.join(',') === 'a1,a2,b1,b2,c1', file.ids.join(','));
ok('identity is the file row', [...file.lidx].join(',') === '0,2,1,4,3');
const tags = [...new Set(Object.values(file.tags))];
const fan = tags.map(t => NODE_DEFS.population.compute([file], { tag:t }));
const merged = NODE_DEFS.gather.compute(fan, { ports:tags.length });
ok('the fan merges back in the same order', merged.ids.join(',') === file.ids.join(',') && [...merged.src].join() === [...file.src].join());
ok('and hashes the same as the file node alone', streamHash(merged) === streamHash(file));
const file2 = await NODE_DEFS.pointfile.compute([], P, { id:3 });
ok('the hash is stable across computations', streamHash(file2) === streamHash(file));
const other = await NODE_DEFS.pointfile.compute([], { ...P, scale:2 }, { id:3 });
ok('moved cells hash differently', streamHash(other) !== streamHash(file));
const reordered = NODE_DEFS.gather.compute([fan[1], fan[0], fan[2]], { ports:3 });
ok('another order hashes differently', streamHash(reordered) !== streamHash(file));
const withPool = NODE_DEFS.pool.compute([file], { tag:'t.A', gain:-2 }, { id:9 });
ok('an entry on the stream changes the hash', streamHash(withPool) !== streamHash(file));
ok('a hash is a short hex string', /^[0-9a-f]{1,8}$/.test(streamHash(file)), streamHash(file));

// brain files: the cell order rides in the header and is checked on load
const net = { count:3, src:Int32Array.from([1,1,1]), lidx:Uint32Array.from([0,1,2]), synCount:2,
  w:Float32Array.from([1, 2]), preStart:Int32Array.from([0,1,2,2]), post:Int32Array.from([1,2]) };
const bytes = buildWeightsFile({ nodes:[] }, net, net.w, 0, null);
const brain = parseBrainFile(bytes.buffer ? bytes.buffer : bytes);
ok('the header records the order', brain.meta.order === cellOrder(net), brain.meta.order);
ok('the same order loads', applyBrainWeights(net, brain).matched === 2);
const shuffled = { ...net, src:Int32Array.from([1,1,1]), lidx:Uint32Array.from([2,1,0]) };
ok('another order is refused', /another order/.test(threw(() => applyBrainWeights(shuffled, brain)) || ''));
const old = { ...brain, meta:{ ...brain.meta, order:undefined } };
ok('a file from before orders were recorded still loads', applyBrainWeights(shuffled, old).matched === 2);
// The header records no membrane form, and a file from either form loads, since a 2003 brain's weights are the same currents on the classic rows that replaced that form.
ok('the header records no membrane form', brain.meta.form === undefined, JSON.stringify(brain.meta.form));
ok('a file that recorded the 2003 form loads', applyBrainWeights(net, { ...brain, meta:{ ...brain.meta, form:0 } }).matched === 2);
ok('a file that recorded the 2007 form loads', applyBrainWeights(net, { ...brain, meta:{ ...brain.meta, form:1 } }).matched === 2);
ok('a hue on a population node is not in the hash: no rewiring for a color', streamHash(NODE_DEFS.population.compute([file], { tag:tags[0], hue:40 })) === streamHash(NODE_DEFS.population.compute([file], { tag:tags[0] })));
report('memo');

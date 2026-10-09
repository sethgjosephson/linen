// The points file node: tables and PLY vertex lists become a population with row order as identity, ids carried when a column is named, and the carriers (merge, pick) keeping them.
import { ok, threw, report } from '../tools/harness.mjs';
import { parseTable, parsePly, parsePointsFile, NODE_DEFS, setFileReader } from './nodes.js';

const csv = 'id,x,y,z,type\n720575940000001,10,20,30,KC\n720575940000002,11,21,31,KC\n720575940000003,12,22,32,APL\n';
let t = parseTable(csv, '', 'id');
ok('csv with a header: x y z found by name', t.count === 3 && t.pos[3] === 11 && t.pos[8] === 32, [...t.pos].join(','));
ok('the id column is kept as text past 2^53', t.ids[0] === '720575940000001' && typeof t.ids[0] === 'string');
ok('a missing id column is refused with the header', /the header has id, x, y, z, type/.test(threw(() => parseTable(csv, '', 'root')) || ''));
t = parseTable('1\t2\t3\n4\t5\t6\n', '', '');
ok('tsv without a header: the first three numeric columns', t.count === 2 && t.pos[5] === 6 && t.ids === null);
t = parseTable('a b c d\n9 1 2 3\n8 4 5 6\n', 'b c d', '');
ok('columns by name pick the right ones', t.pos[0] === 1 && t.pos[2] === 3 && t.pos[3] === 4);
t = parseTable('9 1 2 3\n8 4 5 6\n', '2 3 4', '');
ok('columns by number', t.pos[0] === 1 && t.pos[5] === 6);
ok('a non-number in a position column is refused, by file line', /row 3/.test(threw(() => parseTable('x,y,z\n1,2,3\n1,two,3\n', '', '')) || ''));

// PLY ascii and binary (little endian, with an extra property and an id)
const plyA = 'ply\nformat ascii 1.0\ncomment made here\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty int nid\nend_header\n1 2 3 255 7\n4 5 6 0 8\n';
let p = parsePly(new TextEncoder().encode(plyA), 'nid');
ok('ply ascii', p.count === 2 && p.pos[4] === 5 && p.ids[1] === '8', [...p.pos].join(',') + ' ' + p.ids);
const head = new TextEncoder().encode('ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty int nid\nend_header\n');
const body = new Uint8Array(2*(12 + 1 + 4)); const dv = new DataView(body.buffer);
let o = 0; for(const [x, y, z, r, id] of [[1, 2, 3, 255, 7], [4, 5, 6, 0, 8]]){
  dv.setFloat32(o, x, true); dv.setFloat32(o + 4, y, true); dv.setFloat32(o + 8, z, true); dv.setUint8(o + 12, r); dv.setInt32(o + 13, id, true); o += 17; }
const plyB = new Uint8Array(head.length + body.length); plyB.set(head, 0); plyB.set(body, head.length);
p = parsePly(plyB, 'nid');
ok('ply binary little endian', p.count === 2 && p.pos[3] === 4 && p.pos[5] === 6 && p.ids[0] === '7', [...p.pos].join(',') + ' ' + p.ids);
ok('a ply without z is refused', /no x, y, z/.test(threw(() => parsePly(new TextEncoder().encode('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n'), '')) || ''));
ok('the file entry picks the reader by extension', parsePointsFile('a.ply', plyB, '', '').count === 2 && /not \.csv/.test(threw(() => parsePointsFile('a.xyz', plyB, '', '')) || ''));

// the node: scale, up axis, type, tag, identity by row, ids carried
setFileReader(async path => path === 'points/s.csv' ? new TextEncoder().encode(csv) : null);
const pts = await NODE_DEFS.pointfile.compute([], { file:'points/s.csv', columns:'', id:'id', scale:2, up:1, type:3, tag:'kc' }, { id:5 });
ok('the node builds a population', pts.kind === 'points' && pts.count === 3 && pts.tags[5] === 'kc' && pts.ntype[0] === 3);
ok('scale and the up axis apply', pts.pos[0] === 20 && pts.pos[1] === 60 && pts.pos[2] === -40, [...pts.pos.slice(0, 3)].join(','));
ok('row order is the identity', pts.src[2] === 5 && pts.lidx[2] === 2);
ok('ids ride on the stream', pts.ids && pts.ids[2] === '720575940000003');
const merged = NODE_DEFS.gather.compute([pts, NODE_DEFS.scatter.compute([NODE_DEFS.sphere.compute([], { center:[0,0,0], radius:50 })], { count:4, type:0, seed:1, spacing:0, pattern:0, pitch:40, jitter:8, axis:1, tag:'s' }, { id:6 })], { ports:2 });
ok('merge carries ids, null for cells without one', merged.ids && merged.ids.length === 7 && merged.ids[1] === '720575940000002' && merged.ids[5] === null);
let msg = null; try { await NODE_DEFS.pointfile.compute([], { file:'points/none.csv', columns:'', id:'', scale:1, up:0, type:0, tag:'' }, { id:5 }); } catch(e){ msg = e.message; }
ok('a missing file is refused', /not in the project folder/.test(msg || ''), msg);

// a population per value of a column, and a row filter
const pop = await NODE_DEFS.pointfile.compute([], { file:'points/s.csv', columns:'', id:'id', scale:1, up:0, type:0, tag:'fly', tagColumn:'type', filter:'' }, { id:5 });
const srcs = [...new Set(pop.src)];
ok('one population per distinct value', srcs.length === 2 && Object.values(pop.tags).sort().join(',') === 'fly.APL,fly.KC', JSON.stringify(pop.tags));
ok('each population has its own identity, far from node ids', srcs.every(s => s >= 1e9) && pop.src[0] === pop.src[1] && pop.src[2] !== pop.src[0]);
const flt = await NODE_DEFS.pointfile.compute([], { file:'points/s.csv', columns:'', id:'id', scale:1, up:0, type:0, tag:'kc', tagColumn:'', filter:'type=KC' }, { id:5 });
ok('a filter keeps the rows it names', flt.count === 2 && flt.ids.join(',') === '720575940000001,720575940000002');
const flt2 = await NODE_DEFS.pointfile.compute([], { file:'points/s.csv', columns:'', id:'id', scale:1, up:0, type:0, tag:'x', tagColumn:'', filter:'type!=KC' }, { id:5 });
ok('!= keeps the others, and row identity is the file row', flt2.count === 1 && flt2.lidx[0] === 2);
msg = null; try { await NODE_DEFS.pointfile.compute([], { file:'points/s.csv', columns:'', id:'id', scale:1, up:0, type:0, tag:'x', tagColumn:'nope', filter:'' }, { id:5 }); } catch(e){ msg = e.message; }
ok('an unknown tag column is refused with the list', /the file has id, x, y, z, type/.test(msg || ''), msg);

report('pointfile');

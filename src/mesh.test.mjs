// The mesh geometry: an OBJ parses into named objects, the inside test by ray parity agrees with the closed forms it is checked against, a scatter fills it, and a transform moves it with its bounds.
import { ok, threw, report } from '../tools/harness.mjs';
import { parseObj, parseGlb, parseGltfJson, parseMeshFile, meshGeo, meshInside, transformGeo, NODE_DEFS, setFileReader } from './nodes.js';

// a unit cube (12 triangles) and a second object, a tetrahedron, one file
const cube = (ox, oy, oz, s, base) => {
  const v = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    .map(q => `v ${ox+q[0]*s} ${oy+q[1]*s} ${oz+q[2]*s}`).join('\n');
  const f = [[1,2,3,4],[5,8,7,6],[1,5,6,2],[2,6,7,3],[3,7,8,4],[4,8,5,1]]
    .map(q => 'f ' + q.map(i => i + base).join(' ')).join('\n');
  return v + '\n' + f + '\n';
};
const text = '# test\no cube\n' + cube(0, 0, 0, 100, 0) +
  'o tetra\nv 300 0 0\nv 400 0 0\nv 300 100 0\nv 300 0 100\nf 9/1/1 10/2/2 11/3/3\nf 9 10 12\nf 10 11 12\nf 9 12 11\n';
const parsed = parseObj(text);
ok('two named objects', parsed.objects.map(o => o.name).join(',') === 'cube,tetra', parsed.objects.map(o => o.name).join(','));
ok('quads fan into two triangles each', parsed.objects[0].tris.length === 12*9, parsed.objects[0].tris.length);
ok('v/vt/vn faces keep the vertex index', parsed.objects[1].tris.length === 4*9);
ok('a negative index counts from the end', parseObj('v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n').objects[0].tris.length === 9);
ok('an index past the vertices is refused', /past the/.test(threw(() => parseObj('v 0 0 0\nf 1 2 3\n')) || ''));

// inside by ray parity on the cube alone
const g = meshGeo([parsed.objects[0]], { scale:1, up:0, center:0 });
ok('bounds', g.min.join(',') === '0,0,0' && g.max.join(',') === '100,100,100', g.min + ' ' + g.max);
ok('center of the bounds', g.center.join(',') === '50,50,50');
let agree = 0, N = 0;
let z = 12345; const rnd = () => (z = (z*1664525 + 1013904223) >>> 0) / 4294967296;
for(let i = 0; i < 4000; i++){
  const x = rnd()*160 - 30, y = rnd()*160 - 30, zz = rnd()*160 - 30;
  const truth = x > 0 && x < 100 && y > 0 && y < 100 && zz > 0 && zz < 100;
  if(meshInside(g, x, y, zz) === truth) agree++; N++;
}
ok('inside agrees with the box on every random point', agree === N, agree + ' of ' + N);
ok('outside the bounds is outside', !meshInside(g, 500, 50, 50));

// scale, up axis and centring
const g2 = meshGeo([parsed.objects[0]], { scale:2, up:0, center:1 });
ok('scale doubles the extent', g2.size.join(',') === '200,200,200', g2.size.join(','));
ok('centered puts the middle at the origin', g2.center.join(',') === '0,0,0' && meshInside(g2, 0, 0, 0) && !meshInside(g2, 150, 0, 0));
const g3 = meshGeo([{ name:'t', tris:Float32Array.from([0,0,0, 1,0,0, 0,0,1]) }], { scale:1, up:1, center:0 });
ok('a Z-up file turns so its Z becomes Y', g3.tris[7] === 1 && Math.abs(g3.tris[8]) === 0 && g3.tris[3] === 1, [...g3.tris].join(','));

// a scatter fills the cube: every point inside, the count as asked
const pts = NODE_DEFS.scatter.compute([g], { count:500, type:0, seed:3, spacing:0, pattern:0, pitch:40, jitter:8, axis:1, tag:'m' }, { id:1 });
let inside = 0;
for(let i = 0; i < pts.count; i++){
  const x = pts.pos[i*3], y = pts.pos[i*3+1], zz = pts.pos[i*3+2];
  if(x >= 0 && x <= 100 && y >= 0 && y <= 100 && zz >= 0 && zz <= 100) inside++;
}
ok('the scatter fills the mesh', pts.count === 500 && inside === 500, inside + ' of ' + pts.count);

// a transform moves the vertices and the bounds together
const moved = transformGeo(g, { translate:[1000, 0, 0], rotate:[0, 0, 0], scale:[1, 1, 1], pivot:[0, 0, 0] });
ok('a transform moves the mesh', moved.min[0] === 1000 && moved.max[0] === 1100 && meshInside(moved, 1050, 50, 50) && !meshInside(moved, 50, 50, 50));
ok('the original is untouched', g.min[0] === 0 && meshInside(g, 50, 50, 50));

// the node reads through the registered reader and names the objects it has
setFileReader(async path => path === 'meshes/t.obj' ? new TextEncoder().encode(text) : null);
const geo = await NODE_DEFS.mesh.compute([], { file:'meshes/t.obj', object:'tetra', scale:1, up:0, center:0 });
ok('the node picks an object by name', geo.shape === 'mesh' && geo.triCount === 4 && geo.objects.join() === 'tetra');
let msg = null; try { await NODE_DEFS.mesh.compute([], { file:'meshes/t.obj', object:'nope', scale:1, up:0, center:0 }); } catch(e){ msg = e.message; }
ok('a missing object is refused with the list', /cube, tetra/.test(msg || ''), msg);
msg = null; try { await NODE_DEFS.mesh.compute([], { file:'meshes/none.obj', object:'', scale:1, up:0, center:0 }); } catch(e){ msg = e.message; }
ok('a missing file is refused', /not in the project folder/.test(msg || ''), msg);

// glTF: a unit cube as a triangle list with u16 indices, placed by a node translation, and a child node scaling a copy of it, in one .glb built here
const glb = (() => {
  const P = new Float32Array([0,0,0, 1,0,0, 1,1,0, 0,1,0, 0,0,1, 1,0,1, 1,1,1, 0,1,1]);
  const Q = [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]];
  const I = new Uint16Array(Q.flatMap(q => [q[0],q[1],q[2], q[0],q[2],q[3]]));
  const bin = new Uint8Array(P.byteLength + I.byteLength);
  bin.set(new Uint8Array(P.buffer), 0); bin.set(new Uint8Array(I.buffer), P.byteLength);
  const json = { asset:{ version:'2.0' }, scene:0, scenes:[{ nodes:[0, 1] }],
    nodes:[{ name:'box', mesh:0, translation:[10, 0, 0] }, { name:'parent', translation:[0, 100, 0], children:[2] },
      { name:'big', mesh:0, scale:[2, 2, 2] }],
    meshes:[{ name:'cube', primitives:[{ attributes:{ POSITION:0 }, indices:1, mode:4 }] }],
    accessors:[{ bufferView:0, componentType:5126, count:8, type:'VEC3' }, { bufferView:1, componentType:5123, count:36, type:'SCALAR' }],
    bufferViews:[{ buffer:0, byteOffset:0, byteLength:P.byteLength }, { buffer:0, byteOffset:P.byteLength, byteLength:I.byteLength }],
    buffers:[{ byteLength:bin.byteLength }] };
  const js = new TextEncoder().encode(JSON.stringify(json));
  const pad = n => (4 - n % 4) % 4;
  const total = 12 + 8 + js.length + pad(js.length) + 8 + bin.length + pad(bin.length);
  const out = new Uint8Array(total); const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546C67, true); dv.setUint32(4, 2, true); dv.setUint32(8, total, true);
  let at = 12;
  dv.setUint32(at, js.length + pad(js.length), true); dv.setUint32(at + 4, 0x4E4F534A, true); out.set(js, at + 8);
  for(let k = 0; k < pad(js.length); k++) out[at + 8 + js.length + k] = 0x20;
  at += 8 + js.length + pad(js.length);
  dv.setUint32(at, bin.length + pad(bin.length), true); dv.setUint32(at + 4, 0x004E4942, true); out.set(bin, at + 8);
  return { out, json, bin };
})();
const gl = parseGlb(glb.out);
ok('glb: two placed meshes', gl.objects.map(o => o.name).join(',') === 'box,big', gl.objects.map(o => o.name).join(','));
ok('glb: the cube is 12 triangles', gl.objects[0].tris.length === 12*9);
const gb = meshGeo([gl.objects[0]], { scale:1, up:0, center:0 });
ok('glb: the node translation placed it', gb.min.join(',') === '10,0,0' && gb.max.join(',') === '11,1,1', gb.min + ' ' + gb.max);
const gbig = meshGeo([gl.objects[1]], { scale:1, up:0, center:0 });
ok('glb: the parent translation and the child scale compose', gbig.min.join(',') === '0,100,0' && gbig.max.join(',') === '2,102,2', gbig.min + ' ' + gbig.max);
ok('glb: inside works on the placed cube', meshInside(gb, 10.5, 0.5, 0.5) && !meshInside(gb, 0.5, 0.5, 0.5));
const b64 = btoa(String.fromCharCode(...glb.bin));
const gj = parseGltfJson({ ...glb.json, buffers:[{ byteLength:glb.bin.byteLength, uri:'data:application/octet-stream;base64,' + b64 }] }, null);
ok('gltf with an embedded buffer reads the same', gj.objects.length === 2 && gj.objects[0].tris.length === 12*9);
ok('an external buffer is refused', /export as one \.glb/.test(threw(() => parseGltfJson({ ...glb.json, buffers:[{ byteLength:1, uri:'scene.bin' }] }, null)) || ''));
ok('draco is refused', /compression/.test(threw(() => parseGltfJson({ ...glb.json, extensionsRequired:['KHR_draco_mesh_compression'] }, glb.bin)) || ''));
ok('the file entry picks the reader by extension', parseMeshFile('a/b.glb', glb.out).objects.length === 2 && /not \.obj/.test(threw(() => parseMeshFile('x.stl', glb.out)) || ''));

report('mesh');

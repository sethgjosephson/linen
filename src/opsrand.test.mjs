// The operations node's rand() follows the cell, not its place in the stream, and a formula's successive calls draw different numbers.
import { NODE_DEFS } from './nodes.js';
import { ok, report } from '../tools/harness.mjs';

const geo = NODE_DEFS.sphere.compute([], { center:[0,0,0], radius:200 });
const pts = NODE_DEFS.scatter.compute([geo], { count:60, type:0, seed:3, spacing:0, pattern:0, pitch:40, jitter:8, axis:1 }, { id:1 });
const n = pts.count;

// the same points in reverse order: every per-point array reversed
const rev = { ...pts };
for(const [k, v] of Object.entries(pts)){
  if(!ArrayBuffer.isView(v)) continue;
  const w = v.length / n;
  if(w !== 1 && w !== 3) continue;
  const out = new v.constructor(v.length);
  for(let i = 0; i < n; i++) for(let c = 0; c < w; c++) out[(n - 1 - i)*w + c] = v[i*w + c];
  rev[k] = out;
}
const P = { expr:'x = rand()*1000; y = rand()*1000', a:0, b:0, c:0, d:0, seed:7, shuffle:0 };
const A = NODE_DEFS.ops.compute([pts], P), B = NODE_DEFS.ops.compute([rev], P);
const byId = o => { const m = new Map(); for(let i = 0; i < n; i++) m.set(o.src[i] + ':' + o.lidx[i], [o.pos[i*3], o.pos[i*3+1]]); return m; };
const ma = byId(A), mb = byId(B);
let same = 0;
for(const [k, v] of ma){ const u = mb.get(k); if(u && u[0] === v[0] && u[1] === v[1]) same++; }
ok('rand() gives each cell the same values whatever order the points arrive in', same === n, same + ' of ' + n);
let differ = 0;
for(let i = 0; i < n; i++) if(A.pos[i*3] !== A.pos[i*3+1]) differ++;
ok('two calls in one formula draw different numbers', differ === n, differ + ' of ' + n);
const vals = Array.from({ length:n }, (_, i) => A.pos[i*3]);
ok('and the values spread over the range', Math.min(...vals) < 200 && Math.max(...vals) > 800,
  Math.min(...vals).toFixed(0) + ' to ' + Math.max(...vals).toFixed(0));
report('opsrand');

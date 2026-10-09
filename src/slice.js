// The slicer box: a region of the viewer, not of the network.
// Cells outside it are not drawn and cannot be picked; nothing about the tissue changes, so a fly brain can be looked at one hemisphere at a time without cutting the file in half.
// It is part of the scene, saved with it like the resolution, because which part of a tissue is in view is something a scene says.
// Positions are micrometers, the viewer's units.
export function sliceNormalize(s){
  if(!s || typeof s !== 'object') return null;
  const v3 = (a, d) => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite) ? a.map(Number) : d;
  const size = v3(s.size, [1000, 1000, 1000]).map(x => Math.max(1, Math.abs(x)));
  return { on: !!s.on, center: v3(s.center, [0, 0, 0]), size };
}
// The first box a scene gets: the cells' bounding box cut in half along x, so turning the slice on visibly does something.
// An empty viewer gets a millimeter cube at the origin.
export function sliceDefault(pos){
  const n = pos ? (pos.length / 3) | 0 : 0;
  if(!n) return { on:true, center:[0, 0, 0], size:[1000, 1000, 1000] };
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for(let i = 0; i < n; i++) for(let k = 0; k < 3; k++){
    const v = pos[i*3 + k]; if(v < lo[k]) lo[k] = v; if(v > hi[k]) hi[k] = v;
  }
  const ext = hi.map((h, k) => Math.max(1, h - lo[k]));
  return { on:true,
    center:[lo[0] + ext[0]/4, lo[1] + ext[1]/2, lo[2] + ext[2]/2],
    size:[ext[0]/2, ext[1] + 2, ext[2] + 2] };
}
export function sliceBounds(s){
  const min = [0, 0, 0], max = [0, 0, 0];
  for(let k = 0; k < 3; k++){ const h = Math.max(1, Math.abs(s.size[k]))/2;
    min[k] = s.center[k] - h; max[k] = s.center[k] + h; }
  return { min, max };
}
// The box as six half-spaces, [nx, ny, nz, c] each, keeping the points with n . p + c >= 0 (the way three.js clipping planes read them).
// A slice that is off gives planes so far out that nothing is cut, so a material can carry them at all times.
export function slicePlanes(s){
  if(!s || !s.on) return [[1,0,0,1e12],[-1,0,0,1e12],[0,1,0,1e12],[0,-1,0,1e12],[0,0,1,1e12],[0,0,-1,1e12]];
  const { min, max } = sliceBounds(s);
  return [[1,0,0,-min[0]],[-1,0,0,max[0]],[0,1,0,-min[1]],[0,-1,0,max[1]],[0,0,1,-min[2]],[0,0,-1,max[2]]];
}
export function sliceContains(s, x, y, z){
  const { min, max } = sliceBounds(s);
  return x >= min[0] && x <= max[0] && y >= min[1] && y <= max[1] && z >= min[2] && z <= max[2];
}

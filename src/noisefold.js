// A noise stimulus's time constant and shared fraction, folded per cell the way its amplitude is (ENGINE.md section 9, MODEL.md 2).
// One function for every engine's host (the reference worker, the WebGPU worker, the CUDA adapter) and for the Brian case builder, so the four carry the same floats: a cell's decay a = e^(-1/tau), its complement sqrt(1 - a^2), the shared fraction and the index of the stimulus whose shared draw it takes.
// Two noise stimuli with different settings on one cell are refused; with the same settings their amplitudes add, as they always did, and the cell takes the first's shared draw.
// Returns null when no noise stimulus has a time constant or a shared fraction, so an engine runs the white draw with nothing new in its path.
export function foldNoise(protocols, n){
  const noise = (protocols || []).filter(p => p.mode === 3);
  if(!noise.some(p => p.tau > 0 || p.common > 0)) return null;
  const a = new Float32Array(n), k = new Float32Array(n), c = new Float32Array(n), id = new Int32Array(n).fill(-1);
  noise.forEach((p, q) => {
    const tau = p.tau > 0 ? +p.tau : 0, com = p.common > 0 ? Math.min(1, +p.common) : 0;
    const pa = tau > 0 ? Math.exp(-1/tau) : 0, pk = tau > 0 ? Math.sqrt(1 - pa*pa) : 0;
    for(const i of p.idx){
      if(id[i] >= 0){
        if(a[i] !== Math.fround(pa) || c[i] !== Math.fround(com))
          throw new Error('two noise stimuli with different time constants or shared fractions cover cell ' + i + '; give them the same settings or disjoint cells');
        continue;
      }
      a[i] = pa; k[i] = pk; c[i] = com; id[i] = q;
    }
  });
  return { a, k, c, id };
}

// An example node module: jitter moves every cell by a Gaussian offset.
// Copy this file into a project's nodes folder and the jitter node is in the Tab menu, under arrange.
//
// A node module imports nothing.
// Its default export is called with the plugin interface: registerNode, the stream helpers need and clonePts, and the seeded draws h32, pairHash, pairGauss and rng.
//
// The offset of a cell is drawn from its identity (the node that made it, src, and its index there, lidx) and the seed, through pairGauss.
// So a cell keeps its offset when another population is added, when the graph above it is rearranged and at every rewiring, which is how every other draw in a computation behaves.
export default function(linen){
  linen.registerNode('jitter', {
    title:'jitter', cat:'arrange', color:'hsl(270,60%,72%)', inputs:1,
    params:[
      { k:'sigma', label:'sigma µm', t:'float', def:20, min:0, s:[0,200] },
      { k:'seed', label:'seed', t:'int', def:1, s:[1,99] } ],
    compute(ins, p){
      const pts = linen.need(ins[0], 'points', 'jitter needs points: wire a neuron scatter or another points node into it');
      const out = linen.clonePts(pts);
      const sigma = Math.max(0, +p.sigma || 0);
      // the seed is spread so a jitter seed and a connect seed of the same value draw unrelated numbers
      const seed = Math.imul((p.seed | 0) || 1, 7919) + 0x6a17;
      for(let i = 0; i < out.count; i++)
        for(let axis = 0; axis < 3; axis++)
          out.pos[i*3 + axis] += sigma * linen.pairGauss(out.src[i], out.lidx[i], axis, 0, seed);
      return out;
    },
    doc:{
      io:'points in, points out',
      blurb:'Moves every cell by a Gaussian offset on each axis, drawn from its identity and the seed.',
      params:{
        sigma:'Standard deviation of the offset on each axis.',
        seed:'Seed of the offsets.' } },
  });
}

// The bands two engines running the same model must meet (ENGINE.md section 4).
// The battery's cross-engine equivalence group and tools/enginecheck.mjs both read them here, so a candidate engine is held to what the WebGPU engine is held to.
//
//   rate   population rate within 20% of the reference, which must be above 0.5 Hz for the ratio to mean anything
//   cv     ISI CV within 0.15 absolute
//   sync   population synchrony (variance over mean of the population count in 2 ms windows) within a factor of 2
export const EQUIV = { rate:0.2, minRate:0.5, cv:0.15, sync:2, syncBinMs:2 };
export const rateWithin = (a, ref) => ref > EQUIV.minRate && Math.abs(a - ref)/ref < EQUIV.rate;
export const cvWithin = (a, ref) => Math.abs(a - ref) < EQUIV.cv;
export const syncWithin = (a, ref) => a > 0 && ref > 0 && a/ref < EQUIV.sync && ref/a < EQUIV.sync;

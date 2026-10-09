// The scale arithmetic behind the chart node, checked without a browser: where an axis starts, what a tick is worth, which bin a value falls in.
// An axis that starts at the data minimum rather than a round number makes a flat trace look like a trend, and an off-by-one in binning moves mass between bins in a way that looks like a distribution.

import { niceStep, niceBounds, bin } from './chart.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond) pass++;
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};

// --- tick steps are 1, 2, 5 and their decades ------------------------------
{
  for(const [range, ticks, want] of [
    [1, 4, 0.2], [10, 4, 2], [100, 4, 20], [0.1, 4, 0.02],
    [3, 4, 1], [7, 4, 2], [45, 5, 10],
  ]) ok(`step for a range of ${range} over ${ticks} ticks`,
    Math.abs(niceStep(range, ticks) - want) < 1e-12,
    'got ' + niceStep(range, ticks) + ' wanted ' + want);
  ok('a zero range does not divide by zero', niceStep(0, 4) === 1);
  ok('a negative range is handled', niceStep(-5, 4) === 1);
}

// --- bounds land on round numbers and always contain the data --------------
{
  const cases = [[0, 1], [0.13, 0.87], [-3, 12], [5, 5], [0, 0.004]];
  for(const [lo, hi] of cases){
    const b = niceBounds(lo, hi);
    ok(`bounds contain [${lo}, ${hi}]`, b.lo <= lo && b.hi >= hi,
      JSON.stringify(b));
    ok(`bounds for [${lo}, ${hi}] sit on the step`,
      Math.abs(b.lo/b.step - Math.round(b.lo/b.step)) < 1e-9,
      JSON.stringify(b));
  }
  // a flat trace must not fill the plot: it gets padding, so it reads as flat
  const flat = niceBounds(0.5, 0.5);
  ok('a flat series is padded rather than degenerate', flat.hi > flat.lo,
    JSON.stringify(flat));
  const bad = niceBounds(NaN, NaN);
  ok('non-finite bounds fall back rather than throwing',
    Number.isFinite(bad.lo) && Number.isFinite(bad.hi));
}

// --- binning conserves mass ------------------------------------------------
{
  const vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const b = bin(vals, 5);
  let total = 0; for(const c of b.counts) total += c;
  ok('every value lands in exactly one bin', total === vals.length,
    total + ' of ' + vals.length);
  ok('bins span the data', b.lo === 0 && b.hi === 10, JSON.stringify(b));
  ok('the maximum is not lost off the end', b.counts[b.counts.length - 1] >= 1,
    'last bin ' + b.counts[b.counts.length - 1]);

  const g = bin(vals, 5, 0, 10);
  let gtotal = 0; for(const c of g.counts) gtotal += c;
  ok('explicit bounds conserve mass too', gtotal === vals.length);

  // values outside explicit bounds are clamped in rather than dropped, so a histogram of a bounded quantity never silently loses its tails
  const clamped = bin([-5, 0, 5, 15], 4, 0, 10);
  let ctotal = 0; for(const c of clamped.counts) ctotal += c;
  ok('out of range values are clamped, not dropped', ctotal === 4, String(ctotal));

  ok('non-finite values are skipped', (() => {
    const r = bin([1, NaN, 2, Infinity, 3], 3);
    let t = 0; for(const c of r.counts) t += c;
    return t === 3;
  })());
  ok('an empty input gives empty bins', bin([], 5).counts.length === 5 ||
    bin([], 5).counts.length === 0);
  ok('all-identical values do not divide by zero', (() => {
    const r = bin([7, 7, 7], 4);
    let t = 0; for(const c of r.counts) t += c;
    return t === 3 && Number.isFinite(r.lo) && Number.isFinite(r.hi);
  })());
}

console.log(fail ? (fail + ' failed, ' + pass + ' passed') : 'all passed (' + pass + ' checks)');
if(fail) process.exit(1);

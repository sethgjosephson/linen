// The standard spike train statistics, written to match Elephant's definitions rather than to seem reasonable.
//
// audit.html reports a number it calls "population Fano", computed as the variance over the mean of the population spike count per time window.
// That is a synchrony measure and a useful one, but it is not the Fano factor: the standard quantity is the variance over the mean of a single unit's spike count across trials (Elephant `statistics.fanofactor`, and every paper that reports one).
// So these are defined against the reference implementation, and tools/elephant_check.py runs both over the same trains to prove it.
//
// Times are milliseconds throughout, which is what the engines emit.
// Elephant works in whatever unit the Neo SpikeTrain carries, and the comparison harness passes ms, so the two agree without a conversion anywhere.

// Interspike intervals of one sorted spike train.
export function isi(times){
  const out = new Float64Array(Math.max(0, times.length - 1));
  for(let i = 1; i < times.length; i++) out[i-1] = times[i] - times[i-1];
  return out;
}

export function mean(a){
  if(!a.length) return NaN;
  let s = 0;
  for(let i = 0; i < a.length; i++) s += a[i];
  return s/a.length;
}

// Population standard deviation, ddof 0, which is numpy's default and so Elephant's.
// Using the sample form here would put every CV out by sqrt(N/(N-1)), small but systematic and exactly the kind of difference that looks like a result.
export function std(a){
  if(!a.length) return NaN;
  const m = mean(a);
  let v = 0;
  for(let i = 0; i < a.length; i++){ const d = a[i] - m; v += d*d; }
  return Math.sqrt(v/a.length);
}

// Coefficient of variation of the interspike intervals.
// 1 for a Poisson process, 0 for a perfectly regular one.
//
// One interval is enough, and it always gives exactly 0, because the standard deviation of a single number is zero.
// Elephant returns the 0, so this does too: the point of this file is that a value here can be compared against a published one, and a measure that disagrees with the reference on short trains cannot be.
// The guard belongs in whatever reports the number, where a spike count is available to report alongside it, not in the definition.
export function cv(times){
  const d = isi(times);
  if(d.length < 1) return NaN;
  const m = mean(d);
  return m === 0 ? NaN : std(d)/m;
}
// How many spikes a measure needs before its value means anything, as distinct from being defined.
// Callers that report to a person should use these; the functions above deliberately do not enforce them.
const MIN_SPIKES = { cv:4, cv2:4, lv:4, meanRate:1 };

// CV2, the local coefficient of variation (Holt et al. 1996): the mean over
// adjacent interval pairs of 2|d2-d1|/(d2+d1). Unlike CV it is insensitive to
// slow rate drift, which is why it is reported for non-stationary recordings where CV would mostly measure the drift.
export function cv2(times){
  const d = isi(times);
  if(d.length < 2) return NaN;
  let s = 0, n = 0;
  for(let i = 1; i < d.length; i++){
    const a = d[i-1], b = d[i];
    if(a + b === 0) continue;
    s += 2*Math.abs(b - a)/(a + b); n++;
  }
  return n ? s/n : NaN;
}

// LV, local variation (Shinomoto et al. 2003): the mean over adjacent pairs of 3(d1-d2)^2/(d1+d2)^2.
// Also drift insensitive, and the one of the two that is 1 for a Poisson process, which is why both are reported.
export function lv(times){
  const d = isi(times);
  if(d.length < 2) return NaN;
  let s = 0, n = 0;
  for(let i = 1; i < d.length; i++){
    const a = d[i-1], b = d[i];
    if(a + b === 0) continue;
    const r = (a - b)/(a + b);
    s += 3*r*r; n++;
  }
  return n ? s/n : NaN;
}

// Mean firing rate in Hz over a window given in milliseconds.
export function meanRate(times, tStartMs, tStopMs){
  const dur = (tStopMs - tStartMs)/1000;
  return dur > 0 ? times.length/dur : NaN;
}

// The Fano factor, as the field defines it: over a set of spike trains treated as repeated observations of the same unit, the variance of the spike counts over their mean.
// 1 for a Poisson process.
//
// This is not what audit.html calls Fano; that quantity is populationSynchrony below.
export function fanoFactor(trains){
  const counts = trains.map(t => t.length);
  if(counts.length < 2) return NaN;
  const m = mean(counts);
  if(m === 0) return NaN;
  let v = 0;
  for(const c of counts){ const d = c - m; v += d*d; }
  return (v/counts.length)/m;
}

// Population synchrony: the variance over the mean of the summed population count in successive windows.
// This is the quantity audit.html reports as Fano, under a name that says what it is.
export function populationSynchrony(trains, binMs, tStartMs, tStopMs){
  const bins = Math.floor((tStopMs - tStartMs)/binMs);
  if(bins < 2) return NaN;
  const counts = new Float64Array(bins);
  for(const t of trains)
    for(let i = 0; i < t.length; i++){
      const b = Math.floor((t[i] - tStartMs)/binMs);
      if(b >= 0 && b < bins) counts[b]++;
    }
  const m = mean(counts);
  if(m === 0) return NaN;
  let v = 0;
  for(let i = 0; i < bins; i++){ const d = counts[i] - m; v += d*d; }
  return (v/bins)/m;
}

// Spike counts per bin, summed over trains: the time histogram Elephant calls `time_histogram` and everyone else calls a PSTH once it is aligned to a stimulus and divided by trials and bin width.
export function timeHistogram(trains, binMs, tStartMs, tStopMs){
  const bins = Math.max(0, Math.floor((tStopMs - tStartMs)/binMs));
  const counts = new Float64Array(bins);
  for(const t of trains)
    for(let i = 0; i < t.length; i++){
      const b = Math.floor((t[i] - tStartMs)/binMs);
      if(b >= 0 && b < bins) counts[b]++;
    }
  return counts;
}

// The same, as a rate in Hz per train, which is the form a PSTH is plotted in and the form that can be compared against a measured firing rate.
export function psth(trains, binMs, tStartMs, tStopMs){
  const counts = timeHistogram(trains, binMs, tStartMs, tStopMs);
  const scale = 1000/(binMs*Math.max(1, trains.length));
  const out = new Float64Array(counts.length);
  for(let i = 0; i < counts.length; i++) out[i] = counts[i]*scale;
  return out;
}

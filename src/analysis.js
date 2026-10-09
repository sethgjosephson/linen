import { rng } from './rand.js';
import { evaluate as evaluateSet } from './setexpr.js';
// Observer-side analysis.
// Nothing here touches the tissue: it reads the spike flags the engine already reports and asks what they encode.
// That separation is the project rule (nothing is assigned, everything is observed), and it is also what makes the numbers meaningful, since a decoder that lived inside the network could shape what it measures.
//
// Three questions, in increasing order of ambition:
//   1. selectivity  do individual neurons respond to few items or many?
//   2. decoding     can an observer tell items apart from population
//                   activity, above chance?
//   3. completion   after training with two modalities, does one modality
//                   alone recreate the pattern both together produced?

// Lifetime sparseness of one neuron's responses across items (Rolls and Tovee 1995; Vinje and Gallant 2000).
// 0 = responds equally to everything, 1 = responds to a single item.
// Normalized so it does not depend on how many items were shown.
export function sparseness(resp){
  const n = resp.length;
  if(n < 2) return 0;
  let sum = 0, sumSq = 0;
  for(let i=0;i<n;i++){ sum += resp[i]; sumSq += resp[i]*resp[i]; }
  if(sum <= 0 || sumSq <= 0) return 0;
  const mean = sum/n, meanSq = sumSq/n;
  return (1 - (mean*mean)/meanSq) / (1 - 1/n);
}


// Trial-to-trial variability, per cell, averaged over the cells that fired enough for it to mean anything.
//
// This is the Fano factor as the field defines it and as Elephant computes it: for one unit, the variance of its spike count across repeated presentations over the mean of that count, which is 1 for a Poisson process. audit.html reports a different number under the same name, the variance over the mean of the whole population's count per time window, which measures synchrony across cells rather than variability across trials.
// Both are worth having; sharing a name they are worth nothing, because a reader comparing either against a published figure is comparing the wrong quantity. src/spikestats.js holds both under names that separate them, and src/spikestats.test.mjs checks this one against Elephant.
//
// Trials of one condition and one item, since a cell answering two conditions or two stimuli differently is the signal rather than the noise, and pooling either reports that difference as unreliability.
// The argument order is the one every other measure in this file takes, (trials, labels, nItems, nNeurons), with trials a flat Float32Array of N*nNeurons counts.
export function fanoPerCell(trials, labels, nItems, nNeurons,
                            minMeanCount = 1, minReps = 3){
  // Within one item, not across items.
  // The definition is repeated presentations of the same stimulus, and mixing items measures tuning instead: on a synthetic set where each cell answered one of six items at eight spikes and the rest at one, pooling them gave 3.81 for counts that were Poisson by construction and should have given 1.
  // The variance was real, it was just the stimulus changing rather than the response varying.
  if(!labels || typeof labels.length !== 'number' || ArrayBuffer.isView(trials) === false)
    throw new TypeError('fanoPerCell(trials, labels, nItems, nNeurons): ' +
      'trials must be a flat typed array of nTrials*nNeurons counts and ' +
      'labels a parallel array of item ids, one per trial');
  if(trials.length !== labels.length*nNeurons)
    throw new TypeError('fanoPerCell: trials holds ' + trials.length +
      ' values, but ' + labels.length + ' trials of ' + nNeurons +
      ' neurons needs ' + labels.length*nNeurons);
  const byItem = new Map();
  for(let t = 0; t < labels.length; t++){
    if(!byItem.has(labels[t])) byItem.set(labels[t], []);
    byItem.get(labels[t]).push(t);          // row indices, not copies
  }
  let sum = 0, used = 0, items = 0;
  for(const reps of byItem.values()){
    if(reps.length < minReps) continue;     // a ratio over two points says nothing
    items++;
    for(let k = 0; k < nNeurons; k++){
      let m = 0;
      for(const r of reps) m += trials[r*nNeurons + k];
      m /= reps.length;
      if(m < minMeanCount) continue;        // too quiet for a ratio to mean anything
      let v = 0;
      for(const r of reps){ const d = trials[r*nNeurons + k] - m; v += d*d; }
      sum += (v/reps.length)/m; used++;
    }
  }
  return { mean: used ? sum/used : NaN, cells: used, items };
}
// A note on reading the number, because it is biased and the bias is not a bug.
// The variance above is the population form, dividing by the number of repeats rather than by one less, which is what numpy defaults to and so what Elephant's fanofactor uses; matching it is the point.
// That form underestimates by (reps-1)/reps, so with four repeats per item a genuinely Poisson response reads about 0.75 rather than 1, and with twenty it reads about 0.95.
// Measured on synthetic Poisson counts at four repeats: 0.715.
// Compare Fano values only across windows with the same number of repeats.


// The trial matrix as a self-describing binary, so a run's data can be opened by something other than this app.
//
// Every measure in ANALYSIS.md is computed from this one array: decoding, the response matrix, sparseness, the Fano factor, tuning curves, the confusion matrix, PCA.
// Reporting only the scalars derived from it means every question nobody thought to ask in advance is unanswerable, and means nobody can check the arithmetic. tools/np_trials.py reads this into numpy and Neo.
//
// Layout: "NPT1", u32 header length, that many bytes of UTF-8 JSON, then the arrays back to back in the order the header lists them.
// Float32 and Int32 throughout, little endian, which is what both this and numpy already are.
export function packTrialFile(header, arrays){
  const enc = new TextEncoder();
  const head = enc.encode(JSON.stringify(header));
  // pad the header so every array starts 4-byte aligned, which numpy needs and which costs three bytes at most
  const pad = (4 - ((8 + head.length) % 4)) % 4;
  let bytes = 8 + head.length + pad;
  for(const a of arrays) bytes += a.byteLength;
  const buf = new ArrayBuffer(bytes);
  const u8 = new Uint8Array(buf), dv = new DataView(buf);
  u8[0] = 78; u8[1] = 80; u8[2] = 84; u8[3] = 49;      // NPT1
  dv.setUint32(4, head.length + pad, true);
  u8.set(head, 8);
  let off = 8 + head.length + pad;
  for(const a of arrays){
    u8.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), off);
    off += a.byteLength;
  }
  return buf;
}

// Per-neuron selectivity over an item x neuron mean-response matrix, plus the population mean.
// Neurons that never fire are excluded: they are not selective, they are silent, and averaging them in would inflate the score exactly when the network is doing least. minTotal guards the measure against its own arithmetic: a neuron whose summed mean response across every item is a fraction of one spike scores near-perfect sparseness on counting noise alone, so a population that is merely quiet reads as exquisitely selective.
// That artifact shows as selectivity rising exactly where decode falls.
// Cells below the threshold are excluded rather than scored, and activeFrac reports what fraction survived, so a high mean over few cells is visible as such.
export function selectivityProfile(byItem, nItems, nNeurons, minTotal = 1){
  const per = new Float32Array(nNeurons);
  const resp = new Float32Array(nItems);
  let active = 0, sum = 0;
  for(let j=0;j<nNeurons;j++){
    let tot = 0;
    for(let i=0;i<nItems;i++){ resp[i] = byItem[i*nNeurons + j]; tot += resp[i]; }
    if(tot < minTotal){ per[j] = 0; continue; }
    per[j] = sparseness(resp);
    active++; sum += per[j];
  }
  return { per, active, mean: active ? sum/active : 0 };
}

// Nearest-centroid decoding with a split-half protocol: centroids are built from half the trials and scored on the other half, so the number is a generalization estimate rather than a memorized fit.
// Chance is 1/nItems, and the caller should always report it alongside.
export function decodeAccuracy(trials, labels, nItems, nNeurons){
  const N = labels.length;
  // n always means trials scored, in this branch and the one below, so a metrics row cannot read a declined measure as a confident one
  if(N < nItems*2) return { acc:0, chance: nItems ? 1/nItems : 0, n:0 };
  const trainIdx = [], testIdx = [];
  const seen = new Map();
  for(let t=0;t<N;t++){
    const c = (seen.get(labels[t]) || 0);
    seen.set(labels[t], c+1);
    (c % 2 ? testIdx : trainIdx).push(t);      // alternate per item
  }
  const cent = new Float64Array(nItems*nNeurons), cnt = new Float64Array(nItems);
  for(const t of trainIdx){
    const L = labels[t];
    for(let j=0;j<nNeurons;j++) cent[L*nNeurons + j] += trials[t*nNeurons + j];
    cnt[L]++;
  }
  for(let i=0;i<nItems;i++)
    if(cnt[i]) for(let j=0;j<nNeurons;j++) cent[i*nNeurons + j] /= cnt[i];
  // Pearson similarity, the cosine of the centered vectors: robust to the overall activity level drifting between the two halves, which homeostasis guarantees will happen, and to one centroid being broader than the rest.
  // Uncentered, a centroid with many active cells is the nearest to everything (measured 2026-09-02 on the binding bench: the reverse direction of the cross-modal test sat at 0.01 to 0.02 against 0.083 chance while the forward one read 0.18), which is why RSA correlates centered patterns (Kriegeskorte, Mur and Bandettini 2008).
  const norm = new Float64Array(nItems), cmean = new Float64Array(nItems);
  for(let i=0;i<nItems;i++){
    let m = 0;
    for(let j=0;j<nNeurons;j++) m += cent[i*nNeurons + j];
    cmean[i] = m/nNeurons;
    let s = 0;
    for(let j=0;j<nNeurons;j++) s += (cent[i*nNeurons + j] - cmean[i])**2;
    norm[i] = Math.sqrt(s);
  }
  // rows the true item, columns the decoded one: the confusion matrix says which items are mistaken for which, which the accuracy cannot
  const conf = new Float32Array(nItems*nItems);
  let right = 0, scored = 0;
  for(const t of testIdx){
    let best = -1, bestSim = -Infinity, tn = 0, tm = 0;
    for(let j=0;j<nNeurons;j++) tm += trials[t*nNeurons + j];
    tm /= nNeurons;
    for(let j=0;j<nNeurons;j++) tn += (trials[t*nNeurons + j] - tm)**2;
    tn = Math.sqrt(tn);
    if(tn <= 0) continue;
    for(let i=0;i<nItems;i++){
      if(!cnt[i] || norm[i] <= 0) continue;
      let d = 0;
      for(let j=0;j<nNeurons;j++) d += (trials[t*nNeurons + j] - tm)*(cent[i*nNeurons + j] - cmean[i]);
      const sim = d/(tn*norm[i]);
      if(sim > bestSim){ bestSim = sim; best = i; }
    }
    scored++;
    if(best === labels[t]) right++;
    if(best >= 0) conf[labels[t]*nItems + best]++;
  }
  return { acc: scored ? right/scored : 0, chance: nItems ? 1/nItems : 0, n:scored, confusion:conf };
}

// Nearest-centroid decoding on per-trial normalized patterns.
// The decoder above compares raw summed spike counts, so a population whose cells are broadly tuned and whose item information lives in the relative rates across cells, rather than in how hard the population fired overall, scores far below what it actually carries: the overall level varies trial to trial with the network's state and dominates the comparison.
// Measured 2026-08-29, the association territory turns a 65 percent code into a 35 to 42 percent one and that does not move for tissue quantity, sampling breadth, sampling density or recurrence plasticity, which is the pattern a readout artifact would produce.
//
// Each trial is z-scored across its own cells before the comparison, so only the shape of the pattern counts.
// Same split-half protocol as decodeAccuracy, so the two numbers are directly comparable.
export function decodeShape(trials, labels, nItems, nNeurons){
  const N = labels.length;
  if(N < nItems*2) return { acc:0, chance: nItems ? 1/nItems : 0, n:0 };
  const norm = new Float32Array(N*nNeurons);
  for(let t=0;t<N;t++){
    const off = t*nNeurons;
    let m = 0;
    for(let j=0;j<nNeurons;j++) m += trials[off + j];
    m /= nNeurons;
    let v = 0;
    for(let j=0;j<nNeurons;j++){ const d = trials[off + j] - m; v += d*d; }
    const sd = Math.sqrt(v/nNeurons) || 1e-9;
    for(let j=0;j<nNeurons;j++) norm[off + j] = (trials[off + j] - m)/sd;
  }
  return decodeAccuracy(norm, labels, nItems, nNeurons);
}

// Cross-modal completion. centroidsFull are the patterns produced when both modalities were present; centroidsPartial are the patterns produced by one modality alone.
// If the tissue has bound them, an item's partial pattern should resemble its own full pattern more than any other item's.
// Reported as the fraction of items whose own match ranks first, plus the similarity margin (own minus best competitor), which says how decisively.
export function completionScore(centFull, centPart, nItems, nNeurons){
  const cos = (a, ao, b, bo) => {
    let d = 0, na = 0, nb = 0;
    for(let j=0;j<nNeurons;j++){
      const x = a[ao + j], y = b[bo + j];
      d += x*y; na += x*x; nb += y*y;
    }
    return (na > 0 && nb > 0) ? d/Math.sqrt(na*nb) : 0;
  };
  // Cosine on raw spike counts is bounded well above zero because the vectors are non-negative and share a large mean firing component: two unrelated centroids routinely read 0.99.
  // The centered pair below subtracts each neuron's across-item mean first, which is the component a decoder actually uses, so ownC and otherC say whether the item-specific residual favors the matching item.
  const centre = (src) => {
    const out = new Float64Array(nItems*nNeurons);
    for(let j=0;j<nNeurons;j++){
      let m = 0;
      for(let i=0;i<nItems;i++) m += src[i*nNeurons + j];
      m /= nItems;
      for(let i=0;i<nItems;i++) out[i*nNeurons + j] = src[i*nNeurons + j] - m;
    }
    return out;
  };
  const cFull = centre(centFull), cPart = centre(centPart);

  // The absolute cosines are reported alongside the margin, because the margin alone cannot separate the two ways transfer fails.
  // If ownSum is near zero the two modalities share no representation at all and completion is impossible.
  // If ownSum is high but otherSum is just as high, they do share one, but it carries no item identity: every item lands on the same cells and the site has learned that something is present rather than which thing.
  // Those call for opposite fixes.
  let hits = 0, marginSum = 0, scored = 0, ownSum = 0, otherSum = 0;
  let ownCSum = 0, otherCSum = 0, hitsC = 0;
  const perItem = [];
  for(let i=0;i<nItems;i++){
    let own = cos(centPart, i*nNeurons, centFull, i*nNeurons);
    let bestOther = -Infinity, otherMean = 0, otherN = 0;
    for(let k=0;k<nItems;k++){
      if(k === i) continue;
      const s = cos(centPart, i*nNeurons, centFull, k*nNeurons);
      if(s > bestOther) bestOther = s;
      otherMean += s; otherN++;
    }
    if(!isFinite(bestOther)) continue;
    scored++;
    if(own > bestOther) hits++;
    marginSum += own - bestOther;
    ownSum += own;
    otherSum += otherN ? otherMean/otherN : 0;
    // the same comparison on the item-specific residual
    const ownC = cos(cPart, i*nNeurons, cFull, i*nNeurons);
    let bestOtherC = -Infinity, otherCMean = 0, otherCN = 0;
    for(let k=0;k<nItems;k++){
      if(k === i) continue;
      const sc = cos(cPart, i*nNeurons, cFull, k*nNeurons);
      if(sc > bestOtherC) bestOtherC = sc;
      otherCMean += sc; otherCN++;
    }
    if(ownC > bestOtherC) hitsC++;
    ownCSum += ownC;
    otherCSum += otherCN ? otherCMean/otherCN : 0;
    perItem.push({ i, own:+ownC.toFixed(3), best:+bestOtherC.toFixed(3) });
  }
  return { rank1: scored ? hits/scored : 0, margin: scored ? marginSum/scored : 0,
    own: scored ? ownSum/scored : 0, other: scored ? otherSum/scored : 0,
    rank1C: scored ? hitsC/scored : 0,
    ownC: scored ? ownCSum/scored : 0, otherC: scored ? otherCSum/scored : 0,
    // Per item, the centered same-item cosine and the best competing one.
    // The means above cannot distinguish an association that is right for every item but weak from one that is strong for a couple of items and wrong for the rest, and those call for different work: the first is a signal-to-noise problem, the second is items whose representations are confusable with each other.
    perItem, chance: nItems ? 1/nItems : 0, n:scored };
}

// Trial-resolved transfer. completionScore above aggregates each item's partial-condition trials into one centroid and scores items: twelve binary verdicts whose statistic moves in steps of one twelfth.
// Scoring the individual trials against the other modality's centroids gives hundreds of measurements on the same chance floor, a smooth percentage and a standard error, and recomputes for any saved brain.
// Leave-one-out is not needed since the centroids come from the other condition entirely.
export function transferAccuracy(cent, cnt, packed, nItems, nNeurons){
  const { trials, labels, n } = packed;
  let hits = 0, scored = 0, marginSum = 0;
  // Rows are the item presented, columns the item the decoder chose.
  // An accuracy below chance says only that the response resembles other items more than its own; whether it resembles one particular other item (a consistent mis-association) or is merely pushed off its own (a repulsion) is a question about this matrix.
  const confusion = new Float64Array(nItems*nItems);
  for(let t=0;t<n;t++){
    const off = t*nNeurons, lab = labels[t];
    if(lab < 0 || lab >= nItems || !cnt[lab]) continue;
    let own = -Infinity, best = -Infinity, bestK = -1, second = -Infinity;
    // Pearson, as in decodeAccuracy: both vectors centered across cells
    let tm = 0;
    for(let j=0;j<nNeurons;j++) tm += trials[off + j];
    tm /= nNeurons;
    for(let k=0;k<nItems;k++){
      if(!cnt[k]) continue;
      let cm = 0;
      for(let j=0;j<nNeurons;j++) cm += cent[k*nNeurons + j];
      cm /= nNeurons;
      let d = 0, na = 0, nb = 0;
      for(let j=0;j<nNeurons;j++){
        const x = trials[off + j] - tm, y = cent[k*nNeurons + j] - cm;
        d += x*y; na += x*x; nb += y*y;
      }
      // A trial with no activity, or a centroid with none, has no direction to compare: scoring it 0 against every item would make the first item the winner by tie-break, and a population with nothing to say about one sense (V1 under sound alone) would read exactly the fraction of trials labeled item 0.
      if(na <= 0) break;
      if(nb <= 0) continue;
      const sc = d/Math.sqrt(na*nb);
      if(k === lab) own = sc;
      if(sc > best){ second = best; best = sc; bestK = k; }
      else if(sc > second) second = sc;
    }
    if(!isFinite(own) || bestK < 0) continue;
    scored++;
    confusion[lab*nItems + bestK]++;
    if(bestK === lab) hits++;
    marginSum += own - (bestK === lab ? second : best);
  }
  const acc = scored ? hits/scored : 0;
  return { acc, n: scored,
    se: scored ? Math.sqrt(acc*(1 - acc)/scored) : 0,
    margin: scored ? marginSum/scored : 0, confusion };
}

// ---- layer transformation measures ---------------------------------------
// Cortex does not merely sparsen from layer 4 to layer 2: the responsive fraction falls (0.16 to 0.09 in barrel cortex) while response probability among the cells that do respond rises (0.13 to 0.22), correlated ensembles multiply, and decodability improves.
// The block sparsens without any of the rest, so these three quantities are what distinguish a sparse informative code from a merely quiet one.
// Reported per population at every checkpoint.
//
// responsive: a cell counts as responsive to an item when its mean response to that item exceeds its own mean across items by a margin, so the measure is about tuning rather than about firing a lot.
export function responseProfile(trials, labels, nItems, nNeurons, margin = 0.5){
  const N = labels.length;
  if(N < nItems*2 || !nNeurons) return null;
  const sum = new Float64Array(nItems*nNeurons);
  const cnt = new Float64Array(nItems);
  const hit = new Float64Array(nItems*nNeurons);   // trials with any spike
  for(let t=0;t<N;t++){
    const lab = labels[t]; if(lab < 0 || lab >= nItems) continue;
    cnt[lab]++;
    const off = t*nNeurons, base = lab*nNeurons;
    for(let j=0;j<nNeurons;j++){
      const v = trials[off + j];
      sum[base + j] += v;
      if(v > 0) hit[base + j]++;
    }
  }
  let responsive = 0, relSum = 0, relN = 0;
  for(let j=0;j<nNeurons;j++){
    let tot = 0, best = -Infinity, bestItem = -1;
    for(let i=0;i<nItems;i++){
      const m = cnt[i] ? sum[i*nNeurons + j]/cnt[i] : 0;
      tot += m;
      if(m > best){ best = m; bestItem = i; }
    }
    const mean = tot/nItems;
    // responsive means preferring some item clearly above its own average
    if(best > mean*(1 + margin) && best > 0){
      responsive++;
      if(cnt[bestItem]) { relSum += hit[bestItem*nNeurons + j]/cnt[bestItem]; relN++; }
    }
  }
  return { responsiveFrac: responsive/nNeurons,
    reliability: relN ? relSum/relN : 0, responsive };
}

// Decoding restricted to the cells that actually carry the response.
// The experimental literature decodes from identified responsive neurons; a nearest-centroid read over the whole population puts the signal in a handful of dimensions and the noise in hundreds, so a genuinely sparse code scores worse than a dense one purely by dilution.
// Measured 2026-08-28: with responsive fraction at 0.026 the item lives in about thirteen cells of the 497 sampled.
// Reported beside the population decode so the two can disagree visibly.
export function decodeTopAccuracy(trials, labels, nItems, nNeurons, keep = 0.15){
  const N = labels.length;
  if(N < nItems*2 || !nNeurons) return { acc:0, n:0, used:0 };
  // rank cells by how much their response varies across items, which is what a decoder can use, rather than by raw activity
  const sum = new Float64Array(nItems*nNeurons), cnt = new Float64Array(nItems);
  for(let t=0;t<N;t++){
    const lab = labels[t]; if(lab < 0 || lab >= nItems) continue;
    cnt[lab]++;
    const off = t*nNeurons, base = lab*nNeurons;
    for(let j=0;j<nNeurons;j++) sum[base + j] += trials[off + j];
  }
  const score = new Float64Array(nNeurons);
  for(let j=0;j<nNeurons;j++){
    let mean = 0;
    for(let i=0;i<nItems;i++) mean += cnt[i] ? sum[i*nNeurons + j]/cnt[i] : 0;
    mean /= nItems;
    let v = 0;
    for(let i=0;i<nItems;i++){
      const m = cnt[i] ? sum[i*nNeurons + j]/cnt[i] : 0;
      v += (m - mean)*(m - mean);
    }
    score[j] = v;
  }
  const order = Array.from({ length:nNeurons }, (_, j) => j)
    .sort((a, b) => score[b] - score[a])
    .slice(0, Math.max(4, Math.round(nNeurons*keep)))
    .filter(j => score[j] > 0);
  const K = order.length;
  if(K < 4) return { acc:0, n:0, used:K };
  const sub = new Float32Array(N*K);
  for(let t=0;t<N;t++) for(let a=0;a<K;a++) sub[t*K + a] = trials[t*nNeurons + order[a]];
  const d = decodeAccuracy(sub, labels, nItems, K);
  return { acc:d.acc, n:d.n, used:K };
}

// Mean pairwise correlation of trial responses among the most active cells.
// Ensembles are reported in the literature as counts of correlated subnetworks; a mean correlation over the responsive population is the cheap scalar version of the same thing, and it rises L4 to L2 in cortex.
export function ensembleCorrelation(trials, labels, nNeurons, cap = 64){
  const N = labels.length;
  if(N < 8 || !nNeurons) return 0;
  const act = new Float64Array(nNeurons);
  for(let t=0;t<N;t++){ const off = t*nNeurons;
    for(let j=0;j<nNeurons;j++) act[j] += trials[off + j]; }
  const order = Array.from({ length:nNeurons }, (_, j) => j)
    .sort((a, b) => act[b] - act[a]).slice(0, Math.min(cap, nNeurons))
    .filter(j => act[j] > 0);
  const K = order.length;
  if(K < 4) return 0;
  const mu = new Float64Array(K), sd = new Float64Array(K);
  for(let a=0;a<K;a++){
    let m = 0; for(let t=0;t<N;t++) m += trials[t*nNeurons + order[a]];
    m /= N; mu[a] = m;
    let v = 0; for(let t=0;t<N;t++){ const d = trials[t*nNeurons + order[a]] - m; v += d*d; }
    sd[a] = Math.sqrt(v/N) || 1e-9;
  }
  // Ensembles live in the tail of the pairwise correlation distribution, not in its mean: a population of cells each tuned to a different item has strongly correlated same-preference pairs and uncorrelated cross-preference ones, so the mean washes to zero while the structure is real.
  // Reported as the mean of the strongest decile of pairs.
  const cors = [];
  for(let a=0;a<K;a++) for(let b=a+1;b<K;b++){
    let c = 0;
    for(let t=0;t<N;t++)
      c += (trials[t*nNeurons + order[a]] - mu[a])*(trials[t*nNeurons + order[b]] - mu[b]);
    cors.push((c/N)/(sd[a]*sd[b]));
  }
  if(!cors.length) return 0;
  cors.sort((x, y) => y - x);
  const take = Math.max(1, Math.round(cors.length*0.1));
  let sum = 0;
  for(let i=0;i<take;i++) sum += cors[i];
  return sum/take;
}

// ---- like-to-like wiring -------------------------------------------------
// The MICrONS functional connectome (Ding et al., Nature 2025) reports that tuning similarity predicts fine-scale connectivity in mouse visual cortex beyond what axon and dendrite proximity explain.
// The block wires on distance and population pair alone, so it has no term that could produce the rule.
//
// Two statistics, both matched for distance because the block's wiring is distance dependent and tuning may be spatially clustered, which would manufacture the effect from proximity alone:
//   deltaCorr: mean signal correlation of connected pairs minus that of
//     unconnected pairs in the same distance bin. This is the MICrONS
//     statistic and it reads zero for wiring that ignores tuning.
//   deltaW: within connected pairs only, the correlation between tuning
//     similarity and synaptic weight. Wiring here is fixed but weights
//     learn, so a functional like-to-like rule could appear in the weights
//     without appearing in the connectivity.
//
// Connectivity arrives as a precomputed within-population link list rather than as the network, because the trainer transfers the synapse arrays to the engine at init (they are gigabytes at full density) and reading them afterwards yields detached buffers. popLinks below builds the list while the arrays are still readable; a and b are population-local indices and slot is the global synapse index, so current weights can be looked up in whatever array the engine last returned.
// Extracts the synapses internal to one watched population, as local index pairs plus the global synapse slot.
// Must be called while preStart and post are still readable, which in the trainer means before the init message transfers them to the engine.
export function popLinks(idx, preStart, post){
  if(!idx || !preStart || !post || !preStart.length) return null;
  const g2l = new Map();
  for(let a=0;a<idx.length;a++) g2l.set(idx[a], a);
  const A = [], B = [], S = [];
  for(let a=0;a<idx.length;a++){
    const g = idx[a];
    for(let k=preStart[g]; k<preStart[g+1]; k++){
      const b = g2l.get(post[k]);
      if(b === undefined || b === a) continue;
      A.push(a); B.push(b); S.push(k);
    }
  }
  return { a:Int32Array.from(A), b:Int32Array.from(B), slot:Int32Array.from(S) };
}

export function likeToLike(trials, labels, nItems, nNeurons, gpos, gidx,
    links, opts = {}){
  const N = labels.length;
  const cap = opts.cap || 700, binUm = opts.binUm || 30;
  if(N < nItems*2 || !nNeurons || !gidx || !gpos || !links || !links.a
    || links.a.length < 20) return null;
  const w = opts.w, w0 = opts.w0;
  // Excitatory synapses only.
  // MICrONS measures the rule on excitatory connectivity, and mixing signs corrupts both statistics: an inhibitory synapse carries a negative weight, so if inhibitory partners are anti-tuned then similarity tracks the sign of w rather than anything learned.
  const exc = w0 && links.slot
    ? k => w0[links.slot[k]] > 0
    : () => true;
  // per-neuron tuning vector: mean response per item, centered and normalized so a correlation is a dot product
  const sum = new Float64Array(nItems*nNeurons), cnt = new Float64Array(nItems);
  for(let t=0;t<N;t++){
    const lab = labels[t]; if(lab < 0 || lab >= nItems) continue;
    cnt[lab]++;
    const off = t*nNeurons, base = lab*nNeurons;
    for(let j=0;j<nNeurons;j++) sum[base + j] += trials[off + j];
  }
  // keep only cells with some tuning to correlate; a silent cell has no preference and would drag every comparison toward zero equally
  const keep = [];
  for(let j=0;j<nNeurons;j++){
    let mean = 0;
    for(let i=0;i<nItems;i++) mean += cnt[i] ? sum[i*nNeurons + j]/cnt[i] : 0;
    mean /= nItems;
    let v = 0;
    for(let i=0;i<nItems;i++){
      const m = cnt[i] ? sum[i*nNeurons + j]/cnt[i] : 0;
      v += (m - mean)*(m - mean);
    }
    if(v > 0) keep.push(j);
  }
  if(keep.length < 40) return null;
  // subsample so the all-pairs sweep stays cheap at checkpoint cadence
  const K = Math.min(cap, keep.length), step = keep.length/K;
  const loc = new Int32Array(K);
  for(let a=0;a<K;a++) loc[a] = keep[Math.floor(a*step)];
  const tv = new Float64Array(K*nItems);
  for(let a=0;a<K;a++){
    const j = loc[a];
    let mean = 0;
    for(let i=0;i<nItems;i++) mean += cnt[i] ? sum[i*nNeurons + j]/cnt[i] : 0;
    mean /= nItems;
    let norm = 0;
    for(let i=0;i<nItems;i++){
      const d = (cnt[i] ? sum[i*nNeurons + j]/cnt[i] : 0) - mean;
      tv[a*nItems + i] = d; norm += d*d;
    }
    norm = Math.sqrt(norm) || 1e-9;
    for(let i=0;i<nItems;i++) tv[a*nItems + i] /= norm;
  }
  const sim = (a, b) => {
    let c = 0;
    for(let i=0;i<nItems;i++) c += tv[a*nItems + i]*tv[b*nItems + i];
    return c;
  };
  const gid = new Int32Array(K);
  for(let a=0;a<K;a++) gid[a] = gidx[loc[a]];
  const dist = (a, b) => {
    const p = gid[a]*3, q = gid[b]*3;
    const dx = gpos[p] - gpos[q], dy = gpos[p+1] - gpos[q+1],
      dz = gpos[p+2] - gpos[q+2];
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
  };
  // population-local index -> index within this subsample, so links that touch a cell the subsample dropped are skipped rather than misread
  const l2s = new Int32Array(nNeurons).fill(-1);
  for(let a=0;a<K;a++) l2s[loc[a]] = a;
  const linked = new Set();
  let wSum = 0, wN = 0, sSum = 0, swSum = 0, ssSum = 0, wwSum = 0;
  for(let k=0;k<links.a.length;k++){
    const a = l2s[links.a[k]], b = l2s[links.b[k]];
    if(a < 0 || b < 0 || a === b || !exc(k)) continue;
    linked.add(a < b ? a*K + b : b*K + a);
    if(w && links.slot){
      const s = sim(a, b), wv = w[links.slot[k]];
      if(wv === undefined) continue;
      wSum += wv; sSum += s; swSum += s*wv; ssSum += s*s; wwSum += wv*wv; wN++;
    }
  }
  if(linked.size < 20) return null;
  // distance-matched comparison
  const nBins = 40;
  const cS = new Float64Array(nBins), cN = new Float64Array(nBins);
  const uS = new Float64Array(nBins), uN = new Float64Array(nBins);
  for(let a=0;a<K;a++) for(let b=a+1;b<K;b++){
    const bin = Math.min(nBins - 1, Math.floor(dist(a, b)/binUm));
    const s = sim(a, b);
    if(linked.has(a*K + b)){ cS[bin] += s; cN[bin]++; }
    else { uS[bin] += s; uN[bin]++; }
  }
  let dSum = 0, dW = 0, conn = 0, unconn = 0, cTot = 0, uTot = 0;
  for(let i=0;i<nBins;i++){
    if(cN[i] < 10 || uN[i] < 10) continue;
    dSum += (cS[i]/cN[i] - uS[i]/uN[i])*cN[i]; dW += cN[i];
    conn += cS[i]; cTot += cN[i]; unconn += uS[i]; uTot += uN[i];
  }
  // similarity against weight among connected pairs
  let rW = 0;
  if(wN > 8){
    const cov = swSum/wN - (sSum/wN)*(wSum/wN);
    const vs = Math.sqrt(Math.max(0, ssSum/wN - (sSum/wN)**2));
    const vw = Math.sqrt(Math.max(0, wwSum/wN - (wSum/wN)**2));
    rW = vs > 1e-9 && vw > 1e-9 ? cov/(vs*vw) : 0;
  }
  return { deltaCorr: dW ? dSum/dW : 0,
    connCorr: cTot ? conn/cTot : 0, unconnCorr: uTot ? unconn/uTot : 0,
    deltaW: rW, pairs: linked.size, cells: K, matched: dW };
}

// ---- completion trajectory ----------------------------------------------
// Every transfer measure above classifies a whole presentation: it sums the spikes and compares that to a centroid.
// A slow association escapes that: if recurrence pulls activity toward an item's learned state over the course of a presentation, the pattern at the end of a trial is closer to that state than the pattern at the start, and a sum over the whole trial averages the approach away.
//
// For each unimodal trial, take the first and second half separately, and measure how much closer the late half sits to the item's own joint template than the early half did. drift is that change for the correct item; driftOther is the same change averaged over the other items.
// A network completing toward the right state has drift above driftOther; one that merely responds has both near zero, and one whose activity decays has both negative.
export function completionTrajectory(centFull, trials, nItems, nNeurons){
  const cos = (a, ao, b, bo) => {
    let d = 0, na = 0, nb = 0;
    for(let j=0;j<nNeurons;j++){
      const x = a[ao + j], y = b[bo + j];
      d += x*y; na += x*x; nb += y*y;
    }
    return (na > 0 && nb > 0) ? d/Math.sqrt(na*nb) : 0;
  };
  // center the templates on each neuron's across-item mean, as elsewhere: raw cosines on non-negative counts sit near one for any pair
  const cf = new Float64Array(nItems*nNeurons);
  for(let j=0;j<nNeurons;j++){
    let m = 0;
    for(let i=0;i<nItems;i++) m += centFull[i*nNeurons + j];
    m /= nItems;
    for(let i=0;i<nItems;i++) cf[i*nNeurons + j] = centFull[i*nNeurons + j] - m;
  }
  let dSum = 0, oSum = 0, n = 0;
  for(const t of trials){
    if(!t.early || !t.late) continue;
    const lab = t.label;
    if(lab < 0 || lab >= nItems) continue;
    const own = cos(t.late, 0, cf, lab*nNeurons) - cos(t.early, 0, cf, lab*nNeurons);
    let other = 0, oN = 0;
    for(let k=0;k<nItems;k++){
      if(k === lab) continue;
      other += cos(t.late, 0, cf, k*nNeurons) - cos(t.early, 0, cf, k*nNeurons);
      oN++;
    }
    dSum += own; oSum += oN ? other/oN : 0; n++;
  }
  if(!n) return null;
  const drift = dSum/n, driftOther = oSum/n;
  // standard error of the difference, for reading it against zero
  let vs = 0;
  for(const t of trials){
    if(!t.early || !t.late) continue;
    const lab = t.label;
    if(lab < 0 || lab >= nItems) continue;
    const own = cos(t.late, 0, cf, lab*nNeurons) - cos(t.early, 0, cf, lab*nNeurons);
    let other = 0, oN = 0;
    for(let k=0;k<nItems;k++){
      if(k === lab) continue;
      other += cos(t.late, 0, cf, k*nNeurons) - cos(t.early, 0, cf, k*nNeurons);
      oN++;
    }
    const d = own - (oN ? other/oN : 0);
    vs += (d - (drift - driftOther))**2;
  }
  return { drift, driftOther, gap: drift - driftOther,
    se: n > 1 ? Math.sqrt(vs/(n*(n-1))) : 0, n };
}

// Accumulates presentation-aligned responses: one vector per presentation per condition.
// The caller feeds it spike flags with the item and condition currently on show; it flushes a trial whenever the presentation changes.
// Ticks of PSTH kept from presentation onset; a presentation longer than this is measured on its first PSTH_T ticks.
const PSTH_T = 400;

export class ItemRecorder {
  constructor(idx, items, tickMs = 50){
    this.idx = idx;                       // neurons watched (probe indices)
    this.tickMs = tickMs;                 // simulated ms per sample, for onset latency
    this.items = items;                   // item name -> label index
    this.nNeurons = idx.length;
    this.acc = new Float32Array(idx.length);
    // Per-sample flags for the presentation in progress, so it can be split into halves once its true length is known.
    // A summed count over a whole presentation cannot see a network that completes gradually: if recurrence pulls activity toward an item's learned state over hundreds of milliseconds, the early and late halves of one trial differ and their sum hides exactly that difference.
    this.buf = [];
    this.samples = 0;
    this.curKey = null; this.curLabel = -1; this.curCond = 0;
    this.trials = [];                 // { cond, label, vec, early, late }
    // population spike count per tick from onset, per condition and item
    this.psth = new Map(); this.psthT = 0;
  }
  // cond: 0 both modalities, 1 visual only, 2 audio only.
  // Optional lat is the first-spike step within the tick, one byte per neuron, 255 for silent.
  // Counts say which cells a stimulus recruits; latency says when, which is the only scale STDP works on and the one a tick-level flag threw away.
  sample(fired, key, label, cond, lat){
    if(key !== this.curKey){
      this.flush();
      this.curKey = key; this.curLabel = label; this.curCond = cond;
    }
    if(label < 0) return;
    const idx = this.idx, acc = this.acc;
    const row = new Uint8Array(idx.length);
    for(let k=0;k<idx.length;k++){
      const c = fired[idx[k]];
      if(c){ acc[k] += c; row[k] = c > 255 ? 255 : c; }
    }
    if(lat){
      // Each cell's first spike from the presentation's onset: the ticks before this one plus the step within it, taken once.
      if(!this.latOn) this.latOn = new Float32Array(idx.length).fill(-1);
      const base = this.samples*this.tickMs;
      for(let k=0;k<idx.length;k++){
        if(this.latOn[k] >= 0) continue;
        const t = lat[idx[k]];
        if(t < 255) this.latOn[k] = base + t;
      }
    }
    // population PSTH at tick resolution, per condition and item
    if(this.samples < PSTH_T){
      let pc = 0;
      for(let k=0;k<idx.length;k++) pc += row[k];
      const pk = cond*4096 + label;
      let ps = this.psth.get(pk);
      if(!ps){ ps = { sum:new Float32Array(PSTH_T), cnt:new Float32Array(PSTH_T) }; this.psth.set(pk, ps); }
      ps.sum[this.samples] += pc; ps.cnt[this.samples]++;
      if(this.samples + 1 > this.psthT) this.psthT = this.samples + 1;
    }
    this.buf.push(row);
    this.samples++;
  }
  flush(){
    if(this.samples > 0 && this.curLabel >= 0){
      const v = new Float32Array(this.acc);
      const m = this.nNeurons, half = this.buf.length >> 1;
      const early = new Float32Array(m), late = new Float32Array(m);
      for(let t=0;t<this.buf.length;t++){
        const dst = t < half ? early : late, r = this.buf[t];
        for(let k=0;k<m;k++) if(r[k]) dst[k]++;
      }
      const tr = { cond:this.curCond, label:this.curLabel, vec:v, early, late };
      if(this.latOn){
        tr.lat = this.latOn.slice();       // -1 where the cell never fired
        this.latOn.fill(-1);
      }
      this.trials.push(tr);
    }
    this.acc.fill(0); this.samples = 0; this.buf.length = 0;
  }
  // trials of one condition packed for the routines above
  pack(cond){
    const sel = this.trials.filter(t => t.cond === cond);
    const n = sel.length, m = this.nNeurons;
    const trials = new Float32Array(n*m), labels = new Int32Array(n);
    sel.forEach((t, i) => { trials.set(t.vec, i*m); labels[i] = t.label; });
    return { trials, labels, n };
  }
  centroids(cond, nItems){
    const m = this.nNeurons;
    const cent = new Float32Array(nItems*m), cnt = new Float32Array(nItems);
    for(const t of this.trials){
      if(t.cond !== cond) continue;
      for(let j=0;j<m;j++) cent[t.label*m + j] += t.vec[j];
      cnt[t.label]++;
    }
    for(let i=0;i<nItems;i++)
      if(cnt[i]) for(let j=0;j<m;j++) cent[i*m + j] /= cnt[i];
    return { cent, cnt };
  }
  // Split-half centroids for one condition, alternating trials per item.
  // The cross-modal cosine is only interpretable against this: a centered centroid built from a handful of trials is mostly noise, and two noise vectors correlate at zero whether or not the codes behind them are related.
  // Correlating one half of a modality against its own other half gives the ceiling that any cross-modal number has to be read against.
  centroidsSplit(cond, nItems){
    const m = this.nNeurons;
    const a = new Float32Array(nItems*m), b = new Float32Array(nItems*m);
    const ca = new Float32Array(nItems), cb = new Float32Array(nItems);
    const seen = new Map();
    for(const t of this.trials){
      if(t.cond !== cond) continue;
      const k = (seen.get(t.label) || 0);
      seen.set(t.label, k + 1);
      const even = (k % 2) === 0;
      const dst = even ? a : b, cnt = even ? ca : cb;
      for(let j=0;j<m;j++) dst[t.label*m + j] += t.vec[j];
      cnt[t.label]++;
    }
    for(let i=0;i<nItems;i++){
      if(ca[i]) for(let j=0;j<m;j++) a[i*m + j] /= ca[i];
      if(cb[i]) for(let j=0;j<m;j++) b[i*m + j] /= cb[i];
    }
    return { a, b, ca, cb };
  }
  // Mean population spike count per tick since presentation onset, one row per item, for one condition: the PSTH at the resolution the run samples.
  psthMatrix(cond, nItems){
    const T = this.psthT;
    if(!T) return null;
    const values = new Float32Array(nItems*T);
    let any = false;
    for(let i=0;i<nItems;i++){
      const ps = this.psth.get(cond*4096 + i);
      if(!ps) continue;
      any = true;
      for(let t=0;t<T;t++) values[i*T + t] = ps.cnt[t] ? ps.sum[t]/ps.cnt[t] : 0;
    }
    return any ? { rows:nItems, cols:T, values } : null;
  }
  reset(){ this.trials.length = 0; this.acc.fill(0); this.samples = 0; this.curKey = null;
    this.psth.clear(); this.psthT = 0; }
}

// ---- the conjunction, and why it is the thing to measure ----
//
// A letter's identity does not live in the cells vision wakes, nor in the ones sound wakes.
// Those are sensory.
// If it lives anywhere here it is in the cells that neither modality drives on its own and the pair does together: the conjunction.
// Binding is that set becoming partially reachable from one modality later, which is what recognizing a letter from its sound alone would have to mean in this tissue.
// The cross-modal cosine cannot see it: sight and sound are driven by disjoint afferents, so there is no reason for their patterns to match, and a pathway that made them match would be producing vision, not recognizing a letter.
//
// Picks, per item, the cells in the top q of the joint response that are in the top q of neither single modality.
// Ranks rather than thresholds, because a sixteen-hour run moves every absolute scale underneath the measurement and a rank is immune to that.
// Membership is by tie-averaged rank, not by taking the first k of a sort.
// Spike counts are small integers and most cells share one, so a plain top-k hands out places within a tied block by sort order: the joint set would fill with an arbitrary slice of the tie, the unimodal sets would take a different arbitrary slice, and the difference between them would be an artifact of sorting.
// With ties averaged, a block of equal values sits at one shared rank and either clears the cut together or not at all.
// A top-q that is entirely tied selects nothing, which is the safe failure: the metric then reports absent rather than zero.
export function conjunctiveSets(centBoth, centVis, centAud, nItems, nN,
    q = 0.1, margin = 0.15){
  const hi = 1 - q;
  const sets = [];
  for(let i=0;i<nItems;i++){
    const rb = rankRow(centBoth, i, nN);
    const rv = rankRow(centVis, i, nN);
    const ra = rankRow(centAud, i, nN);
    const c = [];
    for(let j=0;j<nN;j++){
      if(!(rb[j] >= hi && rv[j] < hi && ra[j] < hi)) continue;
      // Being in the top of one condition and not of another can happen to a cell that behaves identically in both, simply because the cut falls between two nearly equal values, and how often that happens rises with the firing rate.
      // Measured on a real run: the retina, where a conjunction is structurally impossible, gave 0.2 cells per item at calibration with the network quiet and 61 at sim 143 with the mean rate up from 1.6 to 2.2 Hz.
      // So the cell must also sit clearly higher with both senses than with either alone, which a cell that is merely noisy does not.
      if(rb[j] - Math.max(rv[j], ra[j]) < margin) continue;
      c.push(j);
    }
    sets.push(c);
  }
  return sets;
}

// Percentile rank of every cell within one probe pattern, ties averaged.
// Ties are not a detail here: most cells sit at zero counts, and breaking those ties by sort order would hand a silent set whatever rank the sort happened to give it.
function rankRow(cent, i, nN){
  const base = i*nN;
  const ord = Array.from({ length:nN }, (_, j) => j)
    .sort((a, b) => cent[base+a] - cent[base+b]);
  const r = new Float32Array(nN);
  let s = 0;
  while(s < nN){
    let e = s;
    while(e + 1 < nN && cent[base+ord[e+1]] === cent[base+ord[s]]) e++;
    const avg = (s + e)/2 / Math.max(1, nN - 1);
    for(let t=s;t<=e;t++) r[ord[t]] = avg;
    s = e + 1;
  }
  return r;
}

// Does one modality alone reach an item's conjunctive cells, and its own item's rather than another's?
//
// Everything is in rank units, so a set of cells sitting at chance scores 0.5 whatever the mean rate has drifted to.
// The reported number is own minus other, both measured at the same moment, on sets built the same way, under the same drive, so anything that moves the whole population cancels rather than being mistaken for learning.
// Zero is chance.
// Above zero means this modality wakes this item's conjunction specifically, which is the claim and the only one this can support.
export function conjunctiveRecall(sets, centProbe, cntProbe, nItems, nN){
  let own = 0, other = 0, used = 0;
  for(let i=0;i<nItems;i++){
    if(!sets[i] || !sets[i].length) continue;
    if(cntProbe && !cntProbe[i]) continue;      // this item was never probed alone
    const r = rankRow(centProbe, i, nN);
    const mean = set => { let s = 0; for(const j of set) s += r[j]; return s/set.length; };
    let o = 0, n = 0;
    for(let j=0;j<nItems;j++){
      if(j === i || !sets[j] || !sets[j].length) continue;
      o += mean(sets[j]); n++;
    }
    if(!n) continue;
    own += mean(sets[i]); other += o/n; used++;
  }
  if(!used) return null;
  own /= used; other /= used;
  return { own:+own.toFixed(4), other:+other.toFixed(4),
    bind:+(own - other).toFixed(4), items:used };
}

// ---- binding in time, not just in membership ----
//
// Counts say which cells a stimulus recruits.
// They cannot say whether one sense has learned to drive a cell, because a cell recruited by chance and a cell recruited through a trained synapse look identical in a count.
// STDP works on the order of a few milliseconds, so if V alone has come to drive an item's conjunctive cells through synapses that STDP built, it should drive them with the timing the pair drives them with: the same cells early, the same cells late.
//
// So: per item, take the conjunctive cells' mean first-spike latency under both senses, and under one sense alone, and correlate the two across cells.
// A correlation near zero means one sense wakes those cells in an order unrelated to the order the pair wakes them, which is what an untrained pathway does.
// A rising correlation means the temporal structure is being reproduced from one sense, which is what a trained one does.
//
// Reported alongside the count-based score deliberately.
// The two can disagree, and which one moves says something: membership without timing is a cell being recruited some other way, timing without membership would be a set drawn too tightly. minCells: an item whose set holds fewer cells than this is skipped, and so is one whose cells did not fire in both conditions that many times.
// A correlation over two or three points is noise.
export function latencyAgreement(sets, trialsBoth, trialsProbe, nItems, nN, minCells = 4){
  const meanLat = (trials, item) => {
    const sum = new Float64Array(nN), cnt = new Float64Array(nN);
    for(const t of trials){
      if(t.label !== item || !t.lat) continue;
      for(let j=0;j<nN;j++) if(t.lat[j] >= 0){ sum[j] += t.lat[j]; cnt[j]++; }
    }
    const out = new Float64Array(nN);
    for(let j=0;j<nN;j++) out[j] = cnt[j] > 0 ? sum[j]/cnt[j] : -1;
    return out;
  };
  let rSum = 0, used = 0, nCells = 0;
  for(let i=0;i<nItems;i++){
    const set = sets[i];
    if(!set || set.length < minCells) continue;
    const a = meanLat(trialsBoth, i), b = meanLat(trialsProbe, i);
    // only cells that fired in both conditions can be compared at all
    const xs = [], ys = [];
    for(const j of set) if(a[j] >= 0 && b[j] >= 0){ xs.push(a[j]); ys.push(b[j]); }
    if(xs.length < minCells) continue;
    const n = xs.length;
    let mx = 0, my = 0;
    for(let k=0;k<n;k++){ mx += xs[k]; my += ys[k]; }
    mx /= n; my /= n;
    let sxy = 0, sxx = 0, syy = 0;
    for(let k=0;k<n;k++){
      const dx = xs[k] - mx, dy = ys[k] - my;
      sxy += dx*dy; sxx += dx*dx; syy += dy*dy;
    }
    if(sxx <= 0 || syy <= 0) continue;     // no spread: nothing to correlate
    rSum += sxy/Math.sqrt(sxx*syy); used++; nCells += n;
  }
  if(!used) return null;
  return { r:+(rSum/used).toFixed(4), items:used,
    cells:+(nCells/used).toFixed(1) };
}

// The general form of conjunctiveSets.
// Instead of three fixed conditions and a fixed question, take the response set for each named condition and combine them however the expression says. conjunctiveSets is the special case `both and not sight and not sound`, and every other question an experiment might ask is sayable in the same terms.
//
// cents: name -> Float32Array(nItems*nN) of mean responses per condition.
// Returns one index array per item, as conjunctiveSets does, so every measure downstream is unchanged.
export function setsFromExpr(ast, cents, nItems, nN, q = 0.1, margin = 0){
  const hi = 1 - q;
  const names = Object.keys(cents);
  const sets = [];
  for(let i=0;i<nItems;i++){
    const memb = {}, ranks = {};
    for(const nm of names){
      const r = rankRow(cents[nm], i, nN);
      ranks[nm] = r;
      const m = new Uint8Array(nN);
      for(let j=0;j<nN;j++) m[j] = r[j] >= hi ? 1 : 0;
      memb[nm] = m;
    }
    const keep = evaluateSet(ast, memb, nN);
    const out = [];
    for(let j=0;j<nN;j++){
      if(!keep[j]) continue;
      // The margin, when asked for, means the same thing it did: a cell must rank clearly higher in the conditions the expression includes than in those it excludes.
      // Without an exclusion there is nothing to compare against and the margin does not apply.
      if(margin > 0){
        let inMin = Infinity, outMax = -Infinity;
        for(const nm of names){
          const v = ranks[nm][j];
          if(memb[nm][j]) inMin = Math.min(inMin, v);
          else outMax = Math.max(outMax, v);
        }
        if(outMax > -Infinity && inMin - outMax < margin) continue;
      }
      out.push(j);
    }
    sets.push(out);
  }
  return sets;
}

// ---- the field's cross-modal measures (ANALYSIS.md section 12) ----

// Single-neuron multisensory integration in the form of Meredith and Stein 1983 and Stein and Stanford 2008 (Nat Rev Neurosci 9:255).
// Per cell and item: the crossmodal enhancement index CRE = (CM - SMmax)/SMmax, where CM is the mean response to both senses and SMmax the larger of the two unimodal means, and the additivity class of CM against SMv + SMa within a tolerance.
// Summarized over the cells given per item (the conjunctive sets by default; every cell when sets is null).
// Cells with no unimodal response have no defined index and are counted rather than scored.
export function multisensoryIndex(centBoth, centVis, centAud, sets, nItems, nN, tol = 0.1){
  let creSum = 0, n = 0, sup = 0, add = 0, subAdd = 0, undef = 0;
  const perItem = [];
  for(let i=0;i<nItems;i++){
    const cells = sets ? sets[i] : null;
    if(sets && (!cells || !cells.length)) continue;
    const list = cells || Array.from({ length:nN }, (_, j) => j);
    let s = 0, k = 0;
    for(const j of list){
      const cm = centBoth[i*nN + j], sv = centVis[i*nN + j], sa = centAud[i*nN + j];
      const smax = Math.max(sv, sa), sum = sv + sa;
      if(!(smax > 0)){ undef++; continue; }
      s += (cm - smax)/smax; k++;
      if(cm > sum*(1 + tol)) sup++; else if(cm < sum*(1 - tol)) subAdd++; else add++;
    }
    if(k){ creSum += s; n += k; perItem.push({ item:i, cre:+(s/k).toFixed(3), cells:k }); }
  }
  const cl = sup + add + subAdd;
  return { cre: n ? +(creSum/n).toFixed(4) : null, cells:n, perItem,
    superadditive: cl ? +(sup/cl).toFixed(3) : null,
    additive: cl ? +(add/cl).toFixed(3) : null,
    subadditive: cl ? +(subAdd/cl).toFixed(3) : null,
    undefined: undef };
}

// Centroids from packed trials under an arbitrary label vector, so the permutation null below rebuilds them exactly the way the statistic does.
function centroidsOf(packed, labels, nItems, nN){
  const cent = new Float32Array(nItems*nN), cnt = new Float32Array(nItems);
  const { trials, n } = packed;
  for(let t=0;t<n;t++){
    const l = labels[t];
    if(l < 0 || l >= nItems) continue;
    const off = t*nN;
    for(let j=0;j<nN;j++) cent[l*nN + j] += trials[off + j];
    cnt[l]++;
  }
  for(let i=0;i<nItems;i++) if(cnt[i]) for(let j=0;j<nN;j++) cent[i*nN + j] /= cnt[i];
  return { cent, cnt };
}

// A decoder that is wrong can be wrong in two ways, and they mean different things: it may send an item consistently to one other item, which is an association to the wrong partner, or it may scatter, which is the item's own response being pushed away from itself and nothing more. topOffMass is the share of the wrong answers that go to each item's single favorite wrong item, against the share expected if the wrong answers were spread evenly.
export function confusionShape(confusion, nItems, minRowErrors = 3){
  if(!confusion) return null;
  let off = 0, diag = 0, top = 0, kept = 0, keptOff = 0;
  const partner = new Int32Array(nItems).fill(-1);
  for(let i=0;i<nItems;i++){
    let bestK = -1, best = 0, rowOff = 0;
    for(let k=0;k<nItems;k++){
      const v = confusion[i*nItems + k];
      if(k === i){ diag += v; continue; }
      rowOff += v;
      if(v > best){ best = v; bestK = k; }
    }
    off += rowOff;
    if(bestK >= 0 && best > 0) partner[i] = bestK;
    // A row with one or two wrong answers has its favorite wrong item whatever it does, so it says nothing about concentration and would read as the ceiling.
    // Only rows with enough errors to spread are counted: measured 2026-09-03, a bench window with 26 trials read the ceiling on rows holding a single error each.
    if(rowOff >= minRowErrors){ kept++; keptOff += rowOff; top += best; }
  }
  if(off <= 0) return null;
  // even spread would put 1/(nItems-1) of a row's wrong answers on its favorite, so 1 is scatter and higher is a consistent wrong partner
  const even = 1/Math.max(1, nItems - 1);
  const out = { off, diag, rows:kept, minRowErrors,
    partners:Array.from(partner) };
  if(kept){
    out.topOffMass = +(top/keptOff).toFixed(4);
    out.topOffRatio = +((top/keptOff)/even).toFixed(3);
  }
  return out;
}

// Multivariate cross-classification (Kaplan, Man and Greening 2015, Front Hum Neurosci 9:151): train on one modality, test on the other, in both directions, with the within-modality split-half accuracies beside them and a permutation null built by shuffling the training labels and re-running the same classifier. p is (permutations at or above the observed accuracy + 1)/(nPerm + 1).
// Nearest-centroid throughout, so the numbers sit on the same footing as the rest of the readout.
// Within does not imply across, which is the point of reporting both.
export function crossDecode(packA, packB, nItems, nN, nPerm = 40, seed = 1){
  const dir = (train, test) => {
    if(!train.n || !test.n) return null;
    const c = centroidsOf(train, train.labels, nItems, nN);
    const obs = transferAccuracy(c.cent, c.cnt, test, nItems, nN);
    const r = rng(seed);
    const perm = Int32Array.from(train.labels);
    let ge = 0, sum = 0;
    for(let p=0;p<nPerm;p++){
      for(let k=perm.length-1;k>0;k--){
        const q = Math.floor(r()*(k+1)); const t = perm[k]; perm[k] = perm[q]; perm[q] = t; }
      const cp = centroidsOf(train, perm, nItems, nN);
      const acc = transferAccuracy(cp.cent, cp.cnt, test, nItems, nN).acc;
      sum += acc; if(acc >= obs.acc) ge++;
    }
    return { acc:+obs.acc.toFixed(4), n:obs.n, se:+obs.se.toFixed(4),
      nullMean: nPerm ? +(sum/nPerm).toFixed(4) : null,
      p: nPerm ? +((ge + 1)/(nPerm + 1)).toFixed(4) : null,
      shape: confusionShape(obs.confusion, nItems) };
  };
  const within = pack => {
    if(!pack.n) return null;
    const d = decodeAccuracy(pack.trials, pack.labels, nItems, nN);
    return d.n ? { acc:+d.acc.toFixed(4), n:d.n } : null;   // too few trials to split
  };
  return { chance:+(1/Math.max(1, nItems)).toFixed(4), nPerm,
    ab:dir(packA, packB), ba:dir(packB, packA),
    withinA:within(packA), withinB:within(packB) };
}

// Representational similarity (Kriegeskorte, Mur and Bandettini 2008; Nili et al. 2014).
// An RDM over the items the mask admits, dissimilarity 1 - Pearson correlation between two items' mean response vectors, each centered across cells.
export function rdm(cent, nItems, nN, mask){
  const items = [];
  for(let i=0;i<nItems;i++) if(!mask || mask[i]) items.push(i);
  const k = items.length, d = new Float32Array(k*k);
  const mean = i => { let m = 0; for(let j=0;j<nN;j++) m += cent[i*nN + j]; return m/nN; };
  const corr = (a, b) => {
    const ma = mean(a), mb = mean(b);
    let xy = 0, xx = 0, yy = 0;
    for(let j=0;j<nN;j++){ const x = cent[a*nN + j] - ma, y = cent[b*nN + j] - mb;
      xy += x*y; xx += x*x; yy += y*y; }
    return (xx > 0 && yy > 0) ? xy/Math.sqrt(xx*yy) : 0;
  };
  for(let p=0;p<k;p++) for(let q=0;q<k;q++) d[p*k + q] = p === q ? 0 : 1 - corr(items[p], items[q]);
  return { items, k, d };
}

function spearman(a, b){
  const rank = v => {
    const o = v.map((x, i) => [x, i]).sort((u, w) => u[0] - w[0]);
    const r = new Float64Array(v.length);
    let s = 0;
    while(s < o.length){
      let e = s;
      while(e + 1 < o.length && o[e+1][0] === o[s][0]) e++;
      const avg = (s + e)/2;
      for(let t=s;t<=e;t++) r[o[t][1]] = avg;
      s = e + 1;
    }
    return r;
  };
  const ra = rank(a), rb = rank(b), n = a.length;
  let ma = 0, mb = 0;
  for(let i=0;i<n;i++){ ma += ra[i]; mb += rb[i]; }
  ma /= n; mb /= n;
  let xy = 0, xx = 0, yy = 0;
  for(let i=0;i<n;i++){ const x = ra[i] - ma, y = rb[i] - mb; xy += x*y; xx += x*x; yy += y*y; }
  return (xx > 0 && yy > 0) ? xy/Math.sqrt(xx*yy) : 0;
}

// Second-order similarity: Spearman correlation of the upper triangles of two RDMs over the same items.
// High means the two modalities arrange the items the same way, whether or not the same cells carry them.
export function rdmCorrelation(A, B){
  if(!A || !B || A.k !== B.k || A.k < 3) return null;
  const a = [], b = [];
  for(let p=0;p<A.k;p++) for(let q=p+1;q<A.k;q++){ a.push(A.d[p*A.k + q]); b.push(B.d[p*B.k + q]); }
  return { rho:+spearman(a, b).toFixed(4), pairs:a.length, items:A.k };
}

// Cross-modal RDM: item i under modality A against item j under modality B, centered correlation.
// Diagonal dominance, the mean on-diagonal similarity minus the mean off-diagonal, measures binding as shared geometry: above zero an item seen sits nearer to itself heard than to other items heard.
export function crossRdm(centA, centB, nItems, nN, mask){
  const items = [];
  for(let i=0;i<nItems;i++) if(!mask || mask[i]) items.push(i);
  const k = items.length;
  if(k < 2) return null;
  const mean = (cent, i) => { let m = 0; for(let j=0;j<nN;j++) m += cent[i*nN + j]; return m/nN; };
  const sim = (i, j) => {
    const ma = mean(centA, i), mb = mean(centB, j);
    let xy = 0, xx = 0, yy = 0;
    for(let t=0;t<nN;t++){ const x = centA[i*nN + t] - ma, y = centB[j*nN + t] - mb;
      xy += x*y; xx += x*x; yy += y*y; }
    return (xx > 0 && yy > 0) ? xy/Math.sqrt(xx*yy) : 0;
  };
  const d = new Float32Array(k*k);
  let on = 0, off = 0, nOff = 0;
  for(let p=0;p<k;p++) for(let q=0;q<k;q++){
    const s = sim(items[p], items[q]);
    d[p*k + q] = 1 - s;
    if(p === q) on += s; else { off += s; nOff++; }
  }
  return { items, k, d, onDiag:+(on/k).toFixed(4), offDiag:+(off/nOff).toFixed(4),
    dominance:+((on/k) - (off/nOff)).toFixed(4) };
}

// Population sparseness per item (Rolls and Tovee 1995), in the Vinje and Gallant 2000 normalization the lifetime measure above already uses: 0 when every cell responds equally to the item, 1 when a single cell does.
// The complement of lifetime sparseness: how few cells one item drives.
export function populationSparseness(cent, cnt, nItems, nN){
  let sum = 0, n = 0;
  for(let i=0;i<nItems;i++){
    if(cnt && !(cnt[i] > 0)) continue;
    let s1 = 0, s2 = 0;
    for(let j=0;j<nN;j++){ const r = Math.max(0, cent[i*nN + j]); s1 += r; s2 += r*r; }
    if(!(s2 > 0)) continue;
    const a = (s1/nN)*(s1/nN)/(s2/nN);
    sum += (1 - a)/(1 - 1/nN); n++;
  }
  return { mean: n ? +(sum/n).toFixed(4) : null, items:n };
}

// Assembly block structure (Litwin-Kumar and Doiron 2014; Zenke, Agnes and Gerstner 2015): mean recurrent weight within each item's conjunctive set against between sets, over the links internal to the watched population (popLinks).
// A cell in more than one set is assigned to the first.
// This is the scalar behind the sorted weight matrix; ratio above one is the block structure those papers show.
export function blockStructure(sets, links, w, nN){
  const nS = sets.length;
  const owner = new Int32Array(nN).fill(-1);
  sets.forEach((cells, i) => { for(const j of (cells || [])) if(owner[j] < 0) owner[j] = i; });
  let within = 0, nW = 0, between = 0, nB = 0;
  const sum = new Float64Array(nS*nS), cnt = new Float64Array(nS*nS);
  for(let k=0;k<links.a.length;k++){
    const oa = owner[links.a[k]], ob = owner[links.b[k]];
    if(oa < 0 || ob < 0) continue;
    const v = w[links.slot[k]];
    if(!(v > 0)) continue;              // excitatory recurrent weights, as the papers plot them
    sum[oa*nS + ob] += v; cnt[oa*nS + ob]++;
    if(oa === ob){ within += v; nW++; } else { between += v; nB++; }
  }
  if(!nW || !nB) return null;
  // the sorted weight matrix itself, set to set, for the chart node
  const matrix = new Float32Array(nS*nS);
  for(let i=0;i<nS*nS;i++) matrix[i] = cnt[i] ? sum[i]/cnt[i] : 0;
  return { within:+(within/nW).toFixed(4), between:+(between/nB).toFixed(4),
    ratio:+((within/nW)/(between/nB)).toFixed(4), nWithin:nW, nBetween:nB,
    matrix, sets:nS };
}

// ---------------------------------------------------------------------------
// Association in the weights rather than in a decoder.
//
// A cross-classification asks whether one channel's response resembles the other's, which needs the population to still be readable: if training drives every cell to respond to everything, the measure goes quiet whether or not anything was learned.
// This asks the connections instead.
// Two sets of cells, one per channel, give four ordered pathways, and the question is whether the two that run between the codes grew more than the two that run within them.
// The within pairs are a control that sits inside every run, so no separate baseline network is needed.
//
// Cells in both sets are excluded from every pathway: a synapse between two of them belongs to no direction, and counting it in both would put the overlap on both sides of the comparison.
// Frozen synapses are excluded too (they cannot answer, and they would dilute all four with zeros).

// Per-synapse pathway label: 0 A>A, 1 A>B, 2 B>A, 3 B>B, 255 not compared.
// Must be called while the synapse arrays are still here, which in a training run means before the init message transfers them to the engine.
export function pairClasses(net, setA, setB){
  if(!net || !net.preStart || !net.post || !net.preStart.length) return null;
  const cls = i => (setA[i] && !setB[i]) ? 0 : (setB[i] && !setA[i]) ? 1 : -1;
  const out = new Uint8Array(net.synCount).fill(255);
  const counts = [0, 0, 0, 0];
  for(let i=0;i<net.count;i++){
    const ci = cls(i);
    if(ci < 0) continue;
    for(let s=net.preStart[i], e=net.preStart[i+1]; s<e; s++){
      if(net.pmask && net.pmask[s] === 0) continue;
      const cj = cls(net.post[s]);
      if(cj < 0) continue;
      const k = ci*2 + cj;
      out[s] = k; counts[k]++;
    }
  }
  return { cls:out, counts };
}

// Mean weight and mean change per pathway, and the one number the comparison is about: how much more the between-code synapses moved than the within-code ones.
// Positive is an association written into the weights, whatever a decoder then makes of it.
export function setWeightStats(pairClass, w, w0){
  if(!pairClass || !w || !w0) return null;
  const sum = [0, 0, 0, 0], d = [0, 0, 0, 0], cnt = [0, 0, 0, 0];
  for(let s=0;s<pairClass.length;s++){
    const k = pairClass[s];
    if(k === 255 || !(w0[s] > 0)) continue;      // excitatory pathways only
    sum[k] += w[s]; d[k] += w[s] - w0[s]; cnt[k]++;
  }
  const key = ['aa', 'ab', 'ba', 'bb'], out = {};
  for(let k=0;k<4;k++) if(cnt[k]) out[key[k]] = {
    w:+(sum[k]/cnt[k]).toFixed(4), dw:+(d[k]/cnt[k]).toFixed(4), n:cnt[k] };
  const mean = ks => { let s = 0, n = 0;
    for(const k of ks) if(out[k]){ s += out[k].dw*out[k].n; n += out[k].n; }
    return n ? s/n : null; };
  const bet = mean(['ab', 'ba']), wit = mean(['aa', 'bb']);
  if(bet !== null && wit !== null) out.betweenMinusWithin = +(bet - wit).toFixed(4);
  return out;
}


// ---- cross-modal learning, measured where the input does not reach ---------
//
// The decoders above compare whole response patterns, and a pattern is set mostly by which cells each input drives directly, so they measure the geometry of the two input codes rather than learning.
// Learning can only show where the input does not reach: in the other code's cells when one sense is presented alone, and in the synapses running between the two codes.
// The three functions below measure those two things, and each has an exact or a measured null.
//
// Item cell sets.
// Per item, the cells of one drive set (and not the other) that the item drives when its own sense is presented alone: the cells whose mean response to the item exceeds their own mean across items, the top fraction topQ of the set's cells by that excess.
// Taken once, at the opening calibration sweep with plasticity off, so learning cannot move a cell into or out of the set it is scored against.
export function itemCellSets(cent, cnt, nItems, nN, inSet, topQ = 0.1){
  const cand = [];
  for(let j=0;j<nN;j++) if(inSet[j]) cand.push(j);
  const take = Math.max(1, Math.round(cand.length*topQ));
  const mean = new Float64Array(nN);
  let items = 0;
  for(let i=0;i<nItems;i++) if(cnt[i] > 0){
    items++;
    for(const j of cand) mean[j] += cent[i*nN + j];
  }
  if(!items || !cand.length) return null;
  for(const j of cand) mean[j] /= items;
  const sets = [];
  for(let i=0;i<nItems;i++){
    if(!(cnt[i] > 0)){ sets.push([]); continue; }
    const scored = cand.map(j => [j, cent[i*nN + j] - mean[j]]).filter(x => x[1] > 0);
    scored.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    sets.push(scored.slice(0, take).map(x => x[0]));
  }
  return sets;
}

// Evoked transfer.
// On trials presenting one sense alone, the mean response per cell in the other sense's set for the same item, against the mean in the other items' sets.
// The input for the presented sense reaches none of these cells (they are the other drive set's, and cells in both sets are left out), so whatever arrives there came through the recurrent synapses.
// Positive is an item-specific association.
// The same number taken at the calibration sweep, before any learning, is its baseline, which carries whatever the recurrent wiring does with the input geometry at wiring time; the change from it is the learned part.
export function evokedTransfer(packed, targetSets, nItems, nN){
  if(!packed || !packed.n || !targetSets) return null;
  const sets = targetSets.map(s => s || []);
  const own = new Float64Array(nItems), other = new Float64Array(nItems), trials = new Float64Array(nItems);
  const meanIn = (off, set) => { if(!set.length) return null; let s = 0; for(const j of set) s += packed.trials[off + j]; return s/set.length; };
  for(let t=0;t<packed.n;t++){
    const lab = packed.labels[t];
    if(lab < 0 || lab >= nItems || !sets[lab].length) continue;
    const off = t*nN;
    const o = meanIn(off, sets[lab]);
    let so = 0, no = 0;
    for(let k=0;k<nItems;k++){ if(k === lab) continue; const v = meanIn(off, sets[k]); if(v !== null){ so += v; no++; } }
    if(!no) continue;
    own[lab] += o; other[lab] += so/no; trials[lab]++;
  }
  const d = [];
  let sumO = 0, sumX = 0;
  for(let i=0;i<nItems;i++) if(trials[i]){
    const o = own[i]/trials[i], x = other[i]/trials[i];
    d.push(o - x); sumO += o; sumX += x;
  }
  if(!d.length) return null;
  const n = d.length, m = d.reduce((a, b) => a + b, 0)/n;
  const sd = n > 1 ? Math.sqrt(d.reduce((a, b) => a + (b - m)**2, 0)/(n - 1)) : 0;
  return { own:+(sumO/n).toFixed(4), other:+(sumX/n).toFixed(4), d:+m.toFixed(4),
    se:+(n > 1 ? sd/Math.sqrt(n) : 0).toFixed(4), items:n };
}

// The synapses between the two codes, with their two ends as cell indices, captured before the init message transfers the synapse arrays to the engine. pairClass is pairClasses' labels: 1 A>B, 2 B>A.
export function betweenSynapses(net, pairClass){
  if(!net || !pairClass || !net.preStart || !net.post) return null;
  const s = [], pre = [], post = [], dir = [];
  for(let i=0;i<net.count;i++)
    for(let q=net.preStart[i], e=net.preStart[i+1]; q<e; q++){
      const k = pairClass[q];
      if(k === 1 || k === 2){ s.push(q); pre.push(i); post.push(net.post[q]); dir.push(k); }
    }
  return { s:Int32Array.from(s), pre:Int32Array.from(pre), post:Int32Array.from(post), dir:Uint8Array.from(dir) };
}

// Item-specific coupling.
// For every synapse running from one code to the other whose presynaptic cell is in item k's set on its side and whose postsynaptic cell is in item m's set on the other, the weight change since the wiring, counted as own when k is m and other when it is not.
// Own minus other, per direction.
// With nothing learned every change is zero and so is this, exactly: the wiring's geometry cannot enter a difference of changes.
// A pairing that is learned makes own exceed other; a scrambled pairing, redrawn every presentation, gives it nothing consistent to write. setsA and setsB hold network indices.
// Excitatory synapses only.
export function itemCoupling(between, setsA, setsB, w, w0){
  if(!between || !setsA || !setsB || !w || !w0) return null;
  const member = sets => {
    const m = new Map();
    sets.forEach((set, k) => { for(const c of set || []){ if(!m.has(c)) m.set(c, []); m.get(c).push(k); } });
    return m;
  };
  const mA = member(setsA), mB = member(setsB);
  const acc = { 1:{ own:0, nOwn:0, other:0, nOther:0 }, 2:{ own:0, nOwn:0, other:0, nOther:0 } };
  for(let q=0;q<between.s.length;q++){
    const s = between.s[q];
    if(!(w0[s] > 0)) continue;
    const dir = between.dir[q];
    const preItems = (dir === 1 ? mA : mB).get(between.pre[q]);
    const postItems = (dir === 1 ? mB : mA).get(between.post[q]);
    if(!preItems || !postItems) continue;
    const dw = w[s] - w0[s], a = acc[dir];
    for(const k of preItems) for(const m of postItems){
      if(k === m){ a.own += dw; a.nOwn++; } else { a.other += dw; a.nOther++; }
    }
  }
  const side = a => (a.nOwn && a.nOther) ? {
    own:+(a.own/a.nOwn).toFixed(5), other:+(a.other/a.nOther).toFixed(5),
    d:+(a.own/a.nOwn - a.other/a.nOther).toFixed(5), nOwn:a.nOwn, nOther:a.nOther } : null;
  return { ab:side(acc[1]), ba:side(acc[2]) };
}

// Where a population's conjunctive cells sit around the vertical axis, against where all its cells sit: counts per angular bin of the cells any item's set holds, and of every sampled cell, with the angle taken about y from the population's own center.
// For a structure whose senses enter at known angles (the mandelbulb, with sight and sound bulbs alternating around a ring), this is the readout of whether binding cells gather at the boundaries between a sight sector and a sound sector or spread evenly.
// Angles are in degrees from the +x axis toward +z.
export function sectorHistogram(sets, idx, pos, bins = 12){
  const n = idx.length;
  if(!n) return null;
  let cx = 0, cz = 0;
  for(let k = 0; k < n; k++){ cx += pos[idx[k]*3]; cz += pos[idx[k]*3 + 2]; }
  cx /= n; cz /= n;
  const binOf = k => {
    const a = Math.atan2(pos[idx[k]*3 + 2] - cz, pos[idx[k]*3] - cx);
    return Math.min(bins - 1, Math.floor(((a + Math.PI) / (2*Math.PI)) * bins));
  };
  const all = new Array(bins).fill(0), conj = new Array(bins).fill(0);
  for(let k = 0; k < n; k++) all[binOf(k)]++;
  const seen = new Set();
  for(const set of sets || []) for(const k of set || []) seen.add(k);
  for(const k of seen) conj[binOf(k)]++;
  // the fraction conjunctive per bin, so a crowded bin does not read as a dense one
  const frac = all.map((a, b) => a ? +(conj[b]/a).toFixed(4) : 0);
  return { bins, all, conj, frac, cells:seen.size };
}

# analysis: what the field measures, and what this app can measure

This is a survey of what computational and systems neuroscience papers
report, set against what this app measures.

Ordering is by how often the measure appears in the literature, not by
how hard it is to build. Status is what this app can do.

## 1. Spike raster over time

A scatter of neuron index against time, usually with a stimulus bar
underneath. It is the single most reproduced figure in spiking network
work: every network paper surveyed opens with one, and reproduction
guides treat it as the primary artifact
(Pauli et al. 2018, Front Neuroinform, reproducing polychronization).

Status: the viewer's raster shows 220 cells sampled across the whole
network, in merge order, at tick resolution. A per-population raster
aligned to presentations is still missing; the PSTH (section 2) is the
first step toward it.

## 2. Peri-stimulus time histogram (PSTH)

Spike count in time bins, aligned to stimulus onset, averaged over
trials, per population or per cell. Answers "what is the response",
which is prior to every selectivity question.

Status: computed and drawn. The recorder accumulates
the population spike count per tick from presentation onset, per item
and condition, and the chart node's PSTH plot draws it as items by time
at the run's tick resolution.

## 3. Firing rate distribution, ISI distribution, CV of ISI

Rate histograms are normally log-scaled because cortical rates are
approximately lognormal. CV of the interspike interval distribution,
sigma(ISI)/mu(ISI), is the standard irregularity measure, with CV near 1
meaning Poisson-like. Fano factor, var(N)/mean(N) over trials, is the
standard variability measure, also 1 for a Poisson process.

Status: partly built. `src/spikestats.js` holds CV, CV2, LV,
mean rate, the Fano factor and the time histogram, written against
Elephant's definitions and checked value for value against it by
`src/spikestats.test.mjs` (47 checks against elephant 1.2.1, neo 0.14.5).
Every probe now reports the Fano factor per checkpoint.

What that exercise found is worth keeping: audit.html reports a number it
calls Fano which is the variance over the mean of the population count
per time window. That is synchrony across cells, not variability across
trials, and the two are not interchangeable. Measured on the same Poisson
trains: 0.285 against 1.097. They have separate names now
(`fanoFactor`, `populationSynchrony`).

Still missing: ISI CV per population during a run, since the recorder
keeps per-trial counts and not spike times.

## 4. Response matrix: cells against stimuli

A heatmap of mean response, cells on one axis and items on the other,
usually sorted so structure is visible. The tuning curve is one row of
it. This is how selectivity is shown before it is reduced to an index.

Status: exported. The analysis node's "export the trial
matrix" writes `trials-<run>.npt` at each checkpoint: the per-trial,
per-cell counts every measure here derives from, with the item and
condition of each trial. `tools/np_trials.py` reads it into numpy and
into Neo, so a run can be opened by the rest of the Python stack rather
than only by this app. One file, overwritten each checkpoint, because a
recording probe of 8192 cells over a couple of hundred trials is about
six megabytes and ninety of those is not a reasonable thing to leave on
someone's disk.

Not yet plotted in the app, which is the remaining half.

## 5. Sparseness and selectivity indices

Two different quantities that are easy to conflate. Population
sparseness is how few cells respond to one stimulus; lifetime
sparseness is how few stimuli one cell responds to
(Willmore and Tolhurst 2001; Rolls and Tovee 1995). Both run 0 to 1.
Sparse selective coding is the expected end state of the learning this
project is trying to produce, so these are the measures that would say
whether it happened.

Status: present. `selectivityProfile` in src/analysis.js is a lifetime sparseness measure
over the item by neuron response matrix, guarded by `minTotal` against
the artifact where a merely quiet population reads as exquisitely
selective. It reports `selectivity` and `activeFrac`, and every
checkpoint carries both, along with `responsiveFrac`.

What is missing from the app is population sparseness, the
other half of the pair: how few cells respond to one stimulus, as against
how few stimuli one cell responds to. `activeFrac` is not that, since it
counts cells active over the whole item set rather than per item.
`tools/np_trials.py` computes both from an exported trial matrix, in the
Rolls and Tovee 1995 form, which is the shorter route to having the
number than adding it to the run.

The measures exist, the run emits them, and nothing in the app shows
them.

## 6. Weight distribution, and its change over time

A histogram of synaptic weights, typically shown before and after
learning. Additive STDP with hard bounds drives it bimodal at the
bounds; weight-dependent depression gives an interior fixed point
(van Rossum, Bi and Turrigiano 2000; Gutig et al. 2003). The battery
already tests this behavior; the app cannot show it.

Status: the chart node's weight distribution reads the live weights back
from the engine, plastic and frozen split, at any moment of a run.

## 7. Sorted weight matrix (assembly block structure)

The mean recurrent weight between groups of cells, with cells reordered
by assembly membership, so a formed assembly appears as a bright block
on the diagonal. This is the central figure of the assembly-formation
literature and is how Zenke, Agnes and Gerstner 2015 show that their
orchestrated plasticity set works at all.

Status: drawn as the weight block matrix: the mean
excitatory recurrent weight from each item's conjunctive set to each
other, with the within against between ratio reported as `block` per
checkpoint (section 12.4).

## 8. Decoding accuracy, and the confusion matrix

Accuracy of a classifier reading population activity, reported against a
chance line, often as a function of time or of training. The confusion
matrix behind it says which items are confused with which, which a
scalar cannot.

Status: the confusion matrix is drawn by the chart node (rows true item, columns decoded, split-half counts), and every accuracy
plots over checkpoints with its chance line, nested measures by dotted
path.

## 9. Cross-correlogram

Spike-time correlation between a pair or population of cells, at
millisecond resolution. Used for functional connectivity, synchrony, and
for showing that a delay structure exists.

Status: missing. The latency measure added for timing agreement is
related but is a first-spike statistic, not a correlogram.

## 10. Dimensionality and population trajectories

PCA of the population response, variance explained, participation ratio,
and trajectories through the leading components. Assembly detection by
PCA is a standard method (Peyrache et al. 2010; Lopes-dos-Santos et al.
2011). Later variants include demixed PCA (Kobak et al. 2016) and tensor
component analysis (Williams et al. 2018).

Status: missing. Worth noting as a direction rather than a gap: this is
a heavier class of analysis and is not required before the items above.

## 11. Power spectrum and LFP

Spectral content of a population signal or synthetic LFP, for
oscillation work. NetPyNE computes LFP by a line-source approximation.

Status: missing, and low priority for this project's questions.

## 12. Cross-modal learning: what the literature measures, and what this app measures

Four bodies of work measure it, from four directions, and a
report that a scientist will accept carries one measure from each.

### 12.1 Single-neuron multisensory integration (Stein and Meredith)

The oldest and most standard descriptors, from superior colliculus
recordings, describe how one neuron's response to the combined stimulus
compares with its responses to each modality alone (Meredith and Stein
1983; Stein and Stanford 2008, Nat Rev Neurosci 9:255; Stanford and
Stein 2009, "Challenges in quantifying multisensory integration", on the
criteria and their limits).

- Crossmodal enhancement index: CRE = (CM - SMmax) / SMmax x 100, where
  CM is the mean response to the combined stimulus and SMmax the mean
  response to the more effective single modality. Positive is
  enhancement, negative is depression.
- Additivity: CM against SMv + SMa. Above the sum is superadditive, equal
  is additive, below is subadditive. Reported as a class, not a number,
  because the sum is the null model.
- Inverse effectiveness: enhancement is largest when the unisensory
  responses are weakest. Tested by varying stimulus strength; the app's
  acuity and amplitude controls are the levers.

What the app has: `conjunctiveSets` selects cells the pair drives that
neither sense alone drives, which is a hard threshold on the same three
quantities. It does not report CRE or the additivity class per cell.
Both are one line each from the centroids the recorder already holds.

### 12.2 Cross-classification: a modality-invariant representation (MVPA)

The standard population-level test for a shared representation is to
train a decoder on one modality and test it on the other. Kaplan, Man
and Greening 2015 (Front Hum Neurosci 9:151, "Multivariate
cross-classification") set the method out: cross-classification
accuracy above chance is direct evidence that the representations are
alike across contexts, within-modality accuracy does not imply it, and
significance comes from a permutation test that shuffles item labels in
both modalities so the null is built the same way the statistic is. The
same logic appears as "cross-decoding" in the fMRI multisensory
literature and as modality-invariant regions in the semantic-circuit
work (Cerebral Cortex 31:4825).

What the app has: `transferAccuracy` builds centroids from one condition
and scores trials from another, which is exactly this, in
nearest-centroid form. The similarity is Pearson, the cosine of the two
vectors after each is centered across cells, as RSA uses (Kriegeskorte,
Mur and Bandettini 2008); an uncentered cosine makes the broadest
centroid the nearest to every trial, and on the binding bench
(2026-09-02) that held one direction of the test at 0.01 to 0.02
against 0.083 chance while the other read 0.18. Both directions are reported, with a label-shuffle permutation p per
window and the within-modality accuracies beside them. A thirty-minute
window holds about forty probe trials per direction, so its p floors near
0.02; the readout therefore also reports the test pooled over every
window from a start minute (`crossDecodePooled`, 90 sim-minutes by
default, `poolfrom` on the address bar): correct over total in each
direction against a binomial at chance. A confidence interval from
repeated splits is still not reported.

### 12.2b Where the wrong answers went

Cross-classification accuracy is one number and it hides the shape of the
errors, which two scenes here needed. A decoder
can be wrong in two ways that mean different things. It can send an item
consistently to one particular other item, which is an association to the
wrong partner. Or its errors can scatter, which says the item's own
response has been pushed off itself and nothing more.

`confusionShape` reports the share of the wrong answers that go to each
item's single favorite wrong item, against the share an even scatter
would give: 1 is scatter, and the ceiling is the item count minus one.
The weave reads the ceiling, 11 of 11, from its first checkpoint window
onward: twelve items, three answers. Rows holding fewer than three errors
are left out of the ratio, because a row with one error has a favorite
whatever it does; the number of rows that qualified is reported beside
it.

Two things this does not do. Concentration does not move the score:
a decoder whose answers are independent of the true label scores
sum over items of p(true = i) p(answer = i), which for balanced items is
chance no matter how few answers it uses, so a collapsed decoder scoring
above or below chance is still telling you something real. And a
concentrated confusion is not by itself a fault in the measure; it is a
statement about the representation, that the pattern one channel evokes in
the other's subspace hardly depends on which item was shown.

It rides along in `crossDecode` and in the pooled statistic, so every
cross-modal number this project reports carries the shape of its errors
beside it.

### 12.3 Representational similarity (RSA)

Where cross-classification asks whether the same cells carry an item in
both modalities, representational similarity asks whether the two
modalities arrange the items the same way, which can hold even when
different cells are involved (Kriegeskorte, Mur and Bandettini 2008;
Nili et al. 2014, "A toolbox for representational similarity analysis";
NeuroRA for the multi-modal case). Compute an item-by-item dissimilarity
matrix per modality from the population responses (1 - correlation is
the usual distance), then correlate the two matrices' upper triangles
(Spearman). A shared geometry gives a high second-order correlation. The
cross-modal dissimilarity matrix, item i seen against item j heard,
shows binding as a dominant diagonal.

What the app has: nothing of this. The trial matrix export
(`tools/np_trials.py`) makes it a few lines of numpy, and it is a few
lines of JavaScript in `analysis.js` too, since the centroids exist.

### 12.4 Pattern completion and assembly structure

The assembly-formation literature this project's plasticity set comes
from measures learning two ways. Functionally, by cued recall: drive a
subset of an assembly and measure the overlap of the evoked population
state with the stored pattern, the Hopfield overlap m = (1/N) sum of
xi_i s_i, normalized so 1 is full recall and 0 is chance (Zenke, Agnes
and Gerstner 2015 show delay activity from a partial cue; the
attractor literature reports overlap after recall dynamics).
Structurally, by the weight matrix: mean recurrent weight within an
assembly against between assemblies, which is the block structure of
the sorted weight matrix and the central figure of Litwin-Kumar and
Doiron 2014 and Zenke et al. 2015 (ANALYSIS.md section 7).

What the app has: `completionScore` and `completionTrajectory` are the
functional form, with the cue being one modality and the stored pattern
the bimodal centroid; they report rank and margin rather than the
normalized overlap, which is why the field's number is not directly
comparable. `likeToLike` is a scalar summary of the structural form;
the within-set against between-set weight means for the conjunctive
sets are not computed, though the synapse slots (`popLinks`) and the
weight readback both exist.

### 12.5 Controls the literature expects

- A scrambled pairing, where sound and sight are paired at random, so
  whatever survives was not learned from the pairing. The curriculum
  node has this (`scramble`); Bahrick and Lickliter's intersensory
  redundancy work is the developmental source.
- Unimodal against multimodal decoding, and decoding with one modality
  removed or degraded, which is how the one directly comparable spiking
  model reports itself: Rathi and Roy 2021 (IEEE TETCI 5(1):143,
  online 2018), two
  unimodal STDP ensembles on images and spoken digits joined by
  cross-modal connections, evaluated by classification accuracy for the
  multimodal network against each unimodal one and under noise on one
  modality.
- A chance line and a null distribution on every accuracy, and more than
  one checkpoint before any statement about a trend.

### 12.6 The standard report, and what it takes

One table per checkpoint, per watched population:

| Measure | Source | Status | Work |
|---|---|---|---|
| CRE and additivity class per conjunctive cell, with the population distribution | Stein and Meredith | `multisensoryIndex`, reported as `msi` per checkpoint | done |
| Within-modality decode, both modalities, with chance | MVPA | `crossDecode.withinA` / `withinB` beside `chance` | done |
| Cross-decode, both directions, permutation p | Kaplan et al. 2015 | `crossDecode.ab` / `ba` with `p` from `nPerm` label shuffles (analysis node parameter) | done; a CI over repeated splits is not reported |
| RDM per modality and their Spearman correlation; cross-modal RDM diagonal dominance | Kriegeskorte 2008; Nili 2014 | `rsa.visAud`, `visBoth`, `audBoth`, `rsa.cross.dominance` | done; the matrices themselves are not yet drawn |
| Cued completion as normalized overlap with the bimodal template, scrambled control beside it | Hopfield; Zenke 2015 | `completionScore.ownC` is the centered cosine, which is the normalized overlap of two centered patterns; reported as `ownC` / `otherC` | done under that name |
| Within-set against between-set mean weight on the conjunctive sets | Litwin-Kumar and Doiron 2014 | `blockStructure`, reported as `block` (within, between, ratio) | done |
| Population and lifetime sparseness | Rolls and Tovee 1995; Willmore and Tolhurst 2001 | `populationSparseness` as `popSparseness`, lifetime as `selectivity` | done |

Beside them the readout reports `silentFrac`, the fraction of trials in
each condition that leave the population with no spikes at all. Those
trials carry no pattern, so the decoders skip them and every accuracy is
a fraction of the trials that did respond. It is reported per condition
because the two senses do not behave alike: on the binding bench the
association area answers every sound-alone probe and falls silent on
about a fifth of the sight-alone ones (2026-09-03), which is a fact
about the network rather than a detail of the decoder.

All of the above are checked on synthetic shared and unshared codes by
`src/xmodal.test.mjs`, and the trainer reports them in the metrics row of
every checkpoint where both unimodal probe conditions have trials. The
figures (the RDMs, the sorted weight matrix, the confusion matrix) remain
the chart node's job, which is section 11's first item.

Every one of these reduces from the same two objects the recorder
already holds and throws away: the per-trial response matrix
(trials x cells, labeled by item and condition) and the readback
weights. Keeping those two, and letting the chart node draw a matrix,
is the whole prerequisite (section 11's conclusion, restated).

### 12.7 Modeling precedent: who has built one, and how close

Three groups have shown the thing, each sharing part of this app's setup and dropping another part. None has
the whole of it, which is where the open work is.

**Pokorny, Ison, Rao, Legenstein, Papadimitriou and Maass 2019 (Cerebral
Cortex 30(3):952, "STDP Forms Associations between Memory Traces in
Networks of Spiking Neurons") is the nearest neighbor.** 432 excitatory and 108 inhibitory spiking
neurons on a 3D grid, inhibitory connections falling off with distance,
triplet STDP (Pfister and Gerstner 2006), data-constrained short-term
plasticity, unsupervised throughout, and the outcome read from emergent
structure rather than a trained readout. Two assemblies form independently
during a 250 s encoding phase from three sparse input patterns; a combined
pattern is then presented 20 times over 36 s, after which each assembly has
expanded and recruited neurons from the other. They set it against human
medial temporal lobe recordings.

That is this app's architecture with one difference: their two traces are
different sparse subsets of the same 200 input channels, not two sensory
pathways with their own areas converging on an association area. So it
demonstrates the mechanism the binding bench depends on without being
cross-modal in the sense of section 12.

Two of their choices bear on open bench items. They constrain the dynamics
with "very strong and local inhibition" that does not learn, which is what
the bench arrived at independently with the inhibitory ceiling of 4 (B6).
And their association phase is 20 presentations over 36 s after the
assemblies already exist, where the bench pairs from the first
presentation; that difference is B8.

**Rathi and Roy 2021 is the only cross-modal spiking model with
unsupervised STDP.** Cited in 12.5 for how it reports itself. Two unimodal
LIF ensembles, image and audio, joined by cross-modal connections trained
with a power-law weight-dependent STDP rule. It has no spatial structure
and no conduction delays, and its success criterion is labeled
classification accuracy, which is the measure this project's second
principle refuses to build. Its internals beyond the abstract are
unverified here: the paper is paywalled and no preprint was found.

**Cuppini, Magosso, Rowland, Stein and Ursino 2012 (Biological Cybernetics
106:691-713, "Hebbian mechanisms help explain development of multisensory
integration in the superior colliculus") is the biologically grounded
version, and is not spiking.** A firing-rate model of the SC, extended from
an adult version to a developmental one, in which multisensory integration
appears only after exposure to correlated cross-modal signals, through
Hebbian LTP and LTD. No 3D connectivity and no delays.

**The empirical target for the CRE of 12.1 is published.** Wang, Yu, Xu,
Stein and Rowland 2020 (Front Integr Neurosci 14:18, "Experience Creates
the Multisensory Transform in the Superior Colliculus") compared 44
normally reared visual-auditory SC neurons against 25 from animals dark
reared to adulthood. Multisensory enhancement was 95 +/- 51 percent in the
normally reared animals and 6 +/- 28 percent in the dark reared ones, and
they ruled out unisensory response magnitude, temporal alignment and input
imbalance as explanations. Normal animals show superadditive enhancement
early in the response then linear amplification; dark reared animals show
statistical facilitation then suppression, which is competition rather than
cooperation.

That is the same quantity `multisensoryIndex` reports as `msi`, so the
bench now has both a number to reach and a number that means no cross-modal
experience was had. The dark-reared figure is the more useful of the two: a
bench run that reports an enhancement near 6 percent has not failed to
measure anything, it has reproduced the untrained animal.


### 12.8 Sources for section 12

Meredith, M. A., Stein, B. E. (1983), J Neurophysiol 56:640. Stein, B. E.,
Stanford, T. R. (2008), Nat Rev Neurosci 9:255. Stanford, T. R., Stein,
B. E. (2009), Exp Brain Res 198:113. Kaplan, J. T., Man, K., Greening,
S. G. (2015), Front Hum Neurosci 9:151. Kriegeskorte, N., Mur, M.,
Bandettini, P. (2008), Front Syst Neurosci 2:4. Nili, H. et al. (2014),
PLoS Comput Biol 10:e1003553. Litwin-Kumar, A., Doiron, B. (2014), Nat
Commun 5:5319. Zenke, F., Agnes, E. J., Gerstner, W. (2015), Nat Commun
6:6922.

For 12.7. Pokorny, C., Ison, M. J., Rao, A., Legenstein,
R., Papadimitriou, C., Maass, W. (2019), Cereb Cortex 30(3):952, doi
10.1093/cercor/bhz140. Rathi, N., Roy, K. (2021), IEEE Trans Emerg Top
Comput Intell 5(1):143, online 2018. Cuppini, C., Magosso, E., Rowland,
B., Stein, B., Ursino, M. (2012), Biol Cybern 106:691. Wang, Z., Yu, L.,
Xu, J., Stein, B. E., Rowland, B. A. (2020), Front Integr Neurosci 14:18.
Pfister, J.-P., Gerstner, W. (2006), J Neurosci 26:9673.

Verified against the published records, except Rathi and Roy, whose
internals are known here only from the abstract.

## 13. What people from Brian and NEST will look for

Brian and NEST produce spike trains and state monitors and leave the
analysis to the user, who reaches for Elephant (the NEST ecosystem's
toolkit), Brian's own `SpikeMonitor` and `PopulationRateMonitor` plots,
or numpy. The figures they will expect from any spiking model, in the
order they appear in papers, are the first eleven sections of this
file: raster, PSTH and population rate, ISI and CV and Fano, rate
distributions, tuning and response matrices, weight distributions, the
sorted weight matrix, decoding with a confusion matrix, correlograms,
dimensionality. Sections 1, 2, 6, 7 and 9 are the ones the app cannot
draw at all today, and all five are drawings of arrays the app already
computes or could keep. So the sequence is the one section 11 gives:
the chart node learns to draw a time series, a histogram and a heat map;
the analysis node emits arrays as well as scalars; the recorder keeps
spike times and the response matrix; then each figure is a small
change. The cross-modal measures in section 12 ride on the same two
objects.

## What this implies for the tool

The pattern in the list is that the app computes scalars and discards
the distributions and matrices they were reduced from. Four of the ten
entries above are cases where the data is already in memory during a run
and is thrown away: the response matrix, the per-trial windows behind a
PSTH, the weight distribution, and the confusion matrix.

So the first move is not more measures. It is that the analysis node
should be able to emit an array or a matrix, not only a number, and that
something in the app should be able to draw one. Concretely, in order:

1. A chart or plot node that can render a time series, a histogram, and
   a matrix as a heatmap. Without it every measure below is still a file.
2. Keep and expose the response matrix, which unlocks 4, 5, and 8 at
   once, since sparseness, selectivity, and the confusion matrix are all
   functions of it.
3. Raster and PSTH, which need a spike-time buffer the recorder does not
   currently keep.
4. Weight distribution and the sorted weight matrix, which need a weight
   readback path that already exists for checkpoints.

Sources are listed inline above. The general surveys used in compiling
this: Pauli et al. 2018 (Front Neuroinform) on what a reproducible
spiking network report contains, and the spiking variability methods
literature for the CV, Fano, and local variation measures.

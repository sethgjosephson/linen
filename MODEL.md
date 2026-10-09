# The model, formula by formula

Every equation the simulation actually evaluates, what each symbol is, and
where it came from.

This is the *what*. The companion documents cover the rest:

- `README.md` is the overview: the interface, the node graph, the scenes,
  the module map and the sources.
- `ENGINE.md` is the message protocol and the frozen contract between the
  page and the three engines, with the derivations that belong to the
  protocol (the Vogels target term, short-term plasticity across engines,
  committed synapses, feedback inhibition).
- `RESULTS.md` is the replication of the Potjans and Diesmann 2014
  microcircuit on these engines, set beside the same model on Brian 2.
- `ANALYSIS.md` is the survey behind the measures in section 5.
- `EXTENDING.md` says how to add a node, a mechanism or a measure, what is
  frozen and which gates a change has to pass.
- The limitations page of the in-app docs (`limits` in `src/docs.js`) lists
  what the model leaves out and what each choice costs, each item naming the
  section of this document that established it.

The reference engine `src/simworker.js` is the definition of correct
behavior; where this document and that file disagree, the file is right.
Every formula below names the file it was read from and the function, kernel
or named constant that holds it. The section "Where each term lives" lists,
for every term, its location in the reference engine, in the WebGPU worker
(`src/gpuworker.js`, whose WGSL shaders are strings inside that file) and in
the CUDA child (`cuda/kernels.cuh` with `cuda/engine.cu`).
`src/modelcite.test.mjs` checks that each formula's code is still present in
the file cited for it, and runs in the test suite.

## Notation and units

| Symbol | Meaning |
|---|---|
| `i`, `j` | neuron indices, `i` presynaptic and `j` postsynaptic unless stated |
| `s` | synapse index in the CSR store |
| `t` | simulated time, integer milliseconds |
| `v`, `u` | membrane potential (mV) and recovery variable |
| `I` | total input current into a neuron this millisecond |
| `w` | synaptic weight, signed: positive excitatory, negative inhibitory |
| `d` | Euclidean distance between two neurons, micrometers |

Time advances in fixed 1 ms steps. Distances are micrometers everywhere.
Rates are stated in Hz and converted to per-millisecond by a factor of 0.001
at the point of use.

One convention is load bearing and easy to miss: **inhibitory weights are
negative numbers**, so "strengthening inhibition" means moving `w` further
below zero, and the inhibitory branches below read as sign-flipped versions
of the excitatory ones for that reason.

### The units of a setting

Every setting on every node, by the unit it carries. A label in the panel
states the unit wherever there is one; the rest are dimensionless.

| Unit | Settings |
|---|---|
| micrometer | every region's center, size, radius, height and thickness; a spline's control points; the noise field's center, size and feature scale; the scatter node's minimum spacing, minicolumn pitch and jitter; move and repeat translate and pivot; the noise warp's amplitude and feature scale; gauss blur sigma; a stimulus center and radius; connect radius and kernel sigma; a projection's target spread and tract length |
| µm/ms | axon velocity, on the connect, project and connections file nodes (100 µm/ms is 0.1 m/s, the unmyelinated intracortical range) |
| per cubic millimeter | the scatter node's density |
| cells | the input node's arbor sigma and microsaccade, which are counted in channels rather than in distance |
| millisecond | the checkpoint's synaptic and trace time constants (tauE, tauI, tauS, tauM, tauY, the short-term depression and facilitation pair) and its refractory period; a stimulus period, width, start and duration; the test signal period; a curriculum lesson's on and off; the chart's moving average; the input node's lag |
| second | the checkpoint's calibration window, its consolidation check interval and the length of a Brian 2 export; the chart's window; the synaptic scaling rate is per second |
| hertz | the homeostatic target rate, the connect node's target rates for its automatic in-degree, and the pool node's inhibition per hertz |
| millivolt | every voltage on a cell type row (rest, threshold, peak, reset) and a graded row's release threshold and slope; the checkpoint's reversal potentials and its membrane floor |
| picofarad | a cell type row's capacitance |
| nS/mV | a cell type row's k, the slope factor of its membrane nullcline |
| nS | a cell type row's b, the recovery variable's coupling to voltage |
| per millisecond | a cell type row's a, the recovery variable's rate |
| picoamp | every current: a stimulus current, the input node's current at gain 1 and a cell type row's d |
| the synapse's own unit | every weight (connect wExc and wInh, a projection's weight, a connections file's weight per synapse) and the plasticity quantities that are in weight units (A+, the triplet term, the transmitter delta, the weight limit, the consolidated weight and the inhibitory learning rate). What that unit is depends on the checkpoint's synapse settings (section 2): the peak current in picoamps under `psc` 0 and 2, the total charge per spike in picoamp milliseconds under `psc` 1, a current held for one millisecond under kick synapses, and a conductance in nanosiemens under conductance synapses (`syn` 2) |
| degree | the rotation on a region, a move or a repeat, and the curriculum's rotation variance; the twist node's angle is degrees per millimeter |
| percent | the connect node's density and in-degree scale, both fractions of full scale |
| point, millimeter, dpi | the chart node's figure settings: font and line width in points, width and height in millimeters, and the export resolution in dots per inch |
| none | probabilities, fractions and gains (a connection probability, a cell type fraction, a noise field's coverage, the cluster boost, the lognormal sigma, a release probability, the weight-dependence and soft-bound exponents, A- under soft bounds, the heterosynaptic beta); counts (a scatter count, a repeat's copies, a gather's ports, contacts per source, channels across); seeds; and every choice a select offers |

Currents and weights are picoamps on the measured rows. On the classic rows
(the Izhikevich 2003 presets, carried as rows of the 2007 form with C = 1 pF)
a picoamp for one millisecond is one millivolt, which is that model's own
current unit, and the same rate takes roughly a twelfth of the current. A
scene states which rows it is on; a scene on the measured rows says so in its
notes.

---

## 1. Neuron model

Izhikevich (2007), *Dynamical Systems in Neuroscience*, chapter 8: the
dimensional form of the model of Izhikevich (2003), with the integration
scheme published with the 2003 model, two half-steps for `v` and one full
step for `u`.

    C v ← C v + 0.5 (k (v − v_r)(v − v_t) − u + I)      twice
    u ← u + a (b (v − v_r) − u)                        using the updated v

    if v ≥ v_peak:   v ← c,   u ← u + d                spike and reset

`src/simworker.js`, `step` and `stepExp` (the line with
`(vi - vri)*(vi - vti)`). The two half-steps are not an optimization: the
single-step form is numerically unstable at 1 ms for this quadratic. Every
type is one row of this form, and the three engines carry no other membrane.

| Symbol | Meaning | Source |
|---|---|---|
| `C` | membrane capacitance, pF | Izhikevich 2007 per neuron type |
| `k` | steepness of the instantaneous I-V curve, nS/mV | same |
| `v_r` | resting potential, mV | same |
| `v_t` | instantaneous threshold, mV | same |
| `v_peak` | spike cutoff, mV, per type | same |
| `a` | recovery time scale, 1/ms | same |
| `b` | sensitivity of `u` to `v − v_r`, nS | same |
| `c` | post-spike reset of `v`, mV | same |
| `d` | post-spike increment of `u`, pA | same |
| `I` | current, pA | ring buffer, bias, protocols, noise (section 2) |

Every current in a scene (weights, bias, noise amplitude, drive) is in the
units of the row it lands on: picoamps on a measured row, and on a classic
row, below, the unit of the 2003 model.

### The 2003 presets as rows of this form

The 2003 model, `dv/dt = 0.04 v² + 5 v + 140 − u + I` with
`du/dt = a (b v − u)`, is this form with `C = 1`, `k = 0.04`, the recovery
variable measured from the rest (`U = u − b v_r`), the rest at the lower root
of `0.04 v² + (5 − b) v + 140 = 0` and the instantaneous threshold at
`v_t = −125 − v_r`: term for term, for every `v`. `src/form.js` derives the
row from `(a, b, c, d)` and `src/form.test.mjs` checks it against the
arithmetic above at every step. Rows 0 to 6 of the type table (RS, IB, CH,
FS, LTS, TC, RZ) are the Izhikevich 2003 figure 2 presets carried this way:
RS `0.02/0.2/−65/8` at rest −70 mV and threshold −55; IB `0.02/0.2/−55/4`;
CH `0.02/0.2/−50/2`; FS `0.1/0.2/−65/2`; LTS `0.02/0.25/−65/2` and TC
`0.02/0.25/−65/0.05` at rest −64.4 and threshold −60.6; RZ `0.1/0.26/−65/2`,
whose rest and threshold coincide at −62.5, the resonator at its
saddle-node. These rows are what a scene means by RS, FS and the rest.
Their capacitance of 1 pF is a unit choice, not a measurement, and under
them a picoamp is the 2003 model's current unit; the defaults on every node
(weights, bias, noise, drive, the plasticity amplitudes and the weight
limit) are numbers in that unit. No engine carries the 2003 form itself,
and an init that names it (a `form` field) is refused.

### The measured rows

Rows per type from Izhikevich 2007 chapter 8 (C, k, v_r, v_t, v_peak, a,
b, c, d), rows 10 to 16 of the type table, keyed RS07 to RZ07: RS `100/0.7/−60/−40/35/0.03/−2/−50/100`, IB
`150/1.2/−75/−45/50/0.01/5/−56/130`, CH `50/1.5/−60/−40/25/0.03/1/−40/150`,
FS `20/1/−55/−40/25/0.2/0.2/−45/0` (the book's FS has a cubic recovery
nullcline; this row is its linear reading), LTS
`100/1/−56/−42/40/0.03/8/−53/20` (the book's v_peak depends on u; fixed
here), TC `200/1.6/−60/−50/35/0.01/15/−60/10` (tonic mode; the book's
v_peak depends on u). RZ has no 2007 row in the book; its row
`100/0.7/−60/−40/35/0.1/5/−60/10` is a resonator built the same way the
2003 preset is, a fast recovery variable with a strong positive coupling.
Three fly rows sit between the two sets, rows 7 to 9: FL, a spiking central neuron, `20/0.06/−58/−40/20/0.05/0/−50/10`,
from a capacitance near 20 pF and an input resistance near 1 GΩ (a
membrane time constant of 20 ms; Gouwens and Wilson 2009, J Neurosci) and
a rest near −58 mV (Wilson and Laurent 2005, J Neurosci), which puts
`k (v_t − v_r)` at 1 nS and the rheobase at 4.9 pA; and FG, a graded relay
for the lamina and medulla cells that do not spike, approximated as a cell
that fires readily and in proportion, `20/0.06/−55/−45/−20/0.1/0/−50/2`
(threshold 10 mV over rest, a low peak, little adaptation), a project
approximation rather than a measured cell; and FS7, the same relay made
slow, `20/0.012/−55/−45/−20/0.05/0/−50/1`, a membrane time constant near
170 ms for the slow arms of the fly's motion detector (Mi9; Arenz et al.
2017, Curr Biol, on the temporal filtering of T4's inputs).
### Graded rows: a non-spiking relay

A row may carry `graded: { thr, slope }` (FG and FS7 do). Such a cell
never spikes and is never reset or refractory. Its
membrane is passive rather than the quadratic above, which has no stable
state above `v_t` without a reset: the leak is the row's slope at rest,
`g = k (v_t − v_r)`, and the potential is capped at the row's `v_peak`,

    C v ← C v + 0.5 (−g (v − v_r) − u + I)          twice, then v ← min(v, v_peak)

so `v` sits at `v_r + I / g` in the linear range (for FG, 0.6 nS: a picoamp
is about 1.7 mV) and saturates at the cap. Every millisecond it releases

    r = clamp((v − thr) / slope, 0, 1)

and each of its synapses delivers `w r` at `t + delay`, exactly as a spike
of amplitude `r` would through the same ring buffer and the same
exponential conductance. Short-term plasticity does not act on a graded
release, and every synapse a graded cell sends is frozen for plasticity
(rule 0 at init in every engine), since spike-timing rules have no spike to
time against. This is graded synaptic transmission, the mode of the fly
optic lobe and of vertebrate photoreceptors and bipolar cells (Juusola,
French, Uusitalo and Weckstrom 1996, Trends Neurosci 19:292); the linear
clamped release is the project's approximation of the sigmoid release
curve those cells show. FG and FS7 release from `thr = v_r` (−55 mV) with
`slope` 10 mV, so release is full at their `v_t`. In the WebGPU
engine the event list carries the neuron index in 24 bits and an
amplitude byte (255 for a spike, 0 to 254 for a release), which caps a
net with graded cells at 16.7 million neurons and rounds a release to the
nearest 1/254, delivering nothing below 1/508 (the other engines deliver
the exact release); the CUDA child takes a
graded flag, threshold and slope per cell at the end of the INIT frame.

### Absolute refractory period

A project addition, not in Izhikevich 2003, standing in for sodium channel
inactivation. For `refrac` milliseconds after a spike the membrane is held at
reset and the threshold cannot be reached whatever the drive, while `u`
continues to advance and synaptic conductances keep integrating.

    if refCnt > 0:   refCnt ← refCnt − 1
                     u ← u + a (b (v − v_r) − u)
                     v unchanged, no spike possible

`src/simworker.js`, `step` and `stepExp` (the `refCnt` branch). The
checkpoint's default `refrac` = 2 ms (an engine given no `refrac` uses 0;
every engine clamps it to 20), which caps a neuron at 1000/(refrac+1) = 333
Hz. Without it the reset alone permits a spike on the very next millisecond.

### Membrane floor

    v ← max(v, v_min)   after each half step

`v_min` = −90 mV by default, the potassium equilibrium potential of a
mammalian neuron (Hille 2001), the checkpoint's `vmin`; 0 disables it. A
project addition: below rest the quadratic grows with the square of the
distance, so the explicit half step overshoots: on a classic row a membrane
pushed to about −150 mV in
one half step is sent over threshold or to infinity, which a synchronous
inhibitory volley (a feedback pool's onset kick) can do.
Applied in all three engines after each half step, so a run that never
reaches the floor is unchanged.

### Initial state

    v = v_r + U(0, min(10, (v_t − v_r)/2)),   u = b (v − v_r)

and on a classic row, where the 2003 form started,

    v = c + U(0, 10),   u = b (v − v_r)

`src/rand.js` `initState`. The jitter stays under half the rest-to-threshold
gap so a row with a 10 mV gap (the fly's graded relays) never starts a cell
on the unstable point. Computed once on the host and sent in the init
message so all three engines start from identical bytes.

---

## 2. Input current

### Conductance mode

The checkpoint's third synapse mode keeps the exponential decay and makes
every channel a conductance toward a reversal potential:

    I_syn = Σ_c |g_c| (E_c − v)

with `E` and `I` reversals on the checkpoint (`eRevE`, 0 mV; `eRevI`, −70
mV) and one per receptor channel (an optional number after the sign on the
receptor node; without it + is 0 and − is −70). `v` is the membrane at the
start of the step. Weights are conductances in nS (a conductance times
millivolts is picoamps; on a classic row the same identity holds in the
2003 unit); the current scale does not apply. Excitation toward
0 mV weakens as the cell depolarizes, and a chloride conductance whose
reversal sits at the rest carries no current of its own and divides the
response to excitation: a shunt (Brette et al. 2007, J Comput Neurosci
23:349, the COBA benchmark). That division is the interaction Groschner,
Malis, Zuidinga and Borst 2022 (Nature 603:119) found under T4's
direction selectivity, a multiplication carried by a shunting chloride
conductance and release from a tonic one on a passive dendrite.
Implemented in all three engines.

### Receptor channels

The exponential synapse above has two conductances per cell, E with
`tauE` and I with `tauI`, chosen by the sign of the weight. A receptor
node declares up to six more channels, each a time constant, a sign and
the transmitters that use it, and the connections file node gives every synapse
whose transmitter the node names that channel and that sign; a
transmitter the node does not name refuses the wiring. A synapse's channel
is the top three bits of its plasticity byte (the rule id is the low five,
which could name 30 rules; the engine contract's rule table holds 16 rows, so
at most 14 named rules), zero meaning by sign into E or I as before.
Each channel `c` is one more term in the current,

    g_c ← g_c e^(−1/tau_c) + (deliveries)        I += g_c · itau_c

with `itau_c` following the `psc` convention like E and I. This is the
kinetic difference between receptors: nicotinic acetylcholine and GABA-A
currents decay in a few milliseconds, glutamate-gated chloride and GABA-B
over tens (Hille 2001, Ion Channels of Excitable Membranes, 3rd ed.).
Channels need exponential synapses (a kick has no time constant), are
fixed at init in the WebGPU engine (the ring is sized by them), and are
frozen with the rest of a graded cell's synapses. Implemented in all
three engines.

    I_i = (synaptic input) + bias_i + stim_i + ext_i + amp_i · ξ(i, t, seed)

`src/simworker.js`, `step` and `stepExp` (the line that sums `ring`,
`bias`, `stim`, `ext` and the noise draw). `stim_i` is evaluated per
millisecond from the stimulus protocols by `scaleOf` and `updateAdd`
(constant, pulse train, ramp, and a noise-amplitude protocol that sets
`amp_i`).

| Term | Meaning |
|---|---|
| `bias_i` | constant background current per neuron, including the proxy compensation of section 3 |
| `stim_i` | current from stimulus nodes for this millisecond |
| `ext_i` | current from external protocols and probes |
| `amp_i · ξ` | per-neuron noise |

### Noise

    ξ = U₁ + U₂ − 1

with `U₁`, `U₂` independent uniforms on [0,1) drawn from a counter hash of
`(i, t, seed)`. This is triangular on [−1, 1] with variance 1/6.

`src/rand.js`, `noiseDraw`, built on `pcg` and `unif`. The counter hash
rather than a stream generator is what lets all three engines draw *the
same* noise rather than merely reproducible noise. Nothing in any engine may
use unseeded randomness.

### Synapses

Two models, selected by `syn` on the checkpoint.

**Kick (current delta).** The weight is added directly to the target's input
on the arrival millisecond.

    I_j(t + delay) += w · f_stp

`src/simworker.js`, `step` (the delivery loop after the reset, writing
`ring[((cur+delay[s])%ROWS)*n + post[s]]`).

**Exponential.** Arrivals accumulate into a conductance that decays each
millisecond.

    g ← g e^(−1/τ) + (arrivals this ms)
    I = g_E k_E + g_I k_I + ...

`src/simworker.js`, `stepExp`. Excitatory and inhibitory arrivals go to
separate accumulators with separate time constants (`tauE` default 3 ms,
`tauI` default 8 ms).

The scale `k` selects a convention, and getting this wrong is invisible:

| `psc` | `k` | Meaning |
|---|---|---|
| 0 (default) | `1` | `w` is the **peak** of the exponential current; charge per spike is `w τ`. NEST `iaf_psc_exp`, Brian |
| 1 | `1 − e^(−1/τ)` | `w` is the **total charge** delivered per spike |
| 2 | `τ (1 − e^(−1/τ))` | `w` is the peak, and the current over a step is the **exact integral** of the exponential across it, so the charge per spike is `w τ` exactly. Mode 0 holds the post-decay current for the step and delivers `1/(τ (1 − e^(−1/τ)))` times that: 1.18 at τE 3, 1.06 at τI 8 (what the one millisecond step costs, below) |

`src/simworker.js`, `itauOf`, applied in `configure` (`S.itauE`,
`S.itauI`, and `S.chanItau` for the receptor channels). A scene declares
the convention it was tuned under on its checkpoint.

### Feedback inhibition

A lumped inhibitory interneuron per population, with every cell of the
population on both its input and its output. Pool `g` counts the spikes its
cells fire in millisecond `t` and every cell of the pool receives, in
millisecond `t + 1`, on the inhibitory path (the input current in kick mode,
`g_I` in exp mode):

    P_g(t + 1) = K_g · C_g(t),      K_g = gain_g · 1000 / N_g

so what a cell receives per millisecond is `gain` times the pool's mean rate
in Hz, whatever the pool's size. `gain` is negative and in weight units.
`C_g(0) = 0`; a cell outside every pool receives 0.

| Symbol | Meaning | Source |
|---|---|---|
| `N_g` | cells in pool `g` | the population the pool node names |
| `gain_g` | inhibition per Hz of the pool's mean rate | the pool node, default −2 |
| the 1 ms latency | fixed | the feedback interneuron's own spike and synapse |

`src/nodes.js` `wirePools` (the pools, one per matching population tag),
`src/simworker.js` `poolDrive` / `poolSwap`, the same two count tables and
parity in the WebGPU update shader and `cuda/kernels.cuh`. The circuit is
the APL neuron of the fly mushroom body, which receives from and inhibits
every Kenyon cell and holds the odor code sparse (Lin et al. 2014, Nat
Neurosci 17:559), and the basket cell recurrent inhibition of the
hippocampus (Andersen, Eccles and Loyning 1963, Nature 198:540). The gain
normalization by `N_g` is this project's convention so that one gain means
one inhibition per Hz at any population size.

### Delays

    delay(i,j) = clamp( round( d(i,j) / velocity ), 1, 16 )   ms

Conduction velocity in µm/ms, so a real distance produces a real delay. The
upper clamp is 16 ms (`MAX_DELAY` in `src/nodes.js`; the ring buffer holds
18 rows). The delay is computed once at wiring time, by `qd` inside
`wireConnect` for the sweep and the projections, and by the same rounding
and clamp in `finishConnect` (the cluster boost) and `wireConnectionsFile` (connections
tables); the engines read the stored byte. For a projection `d` is replaced
by the tract length and the mapped distance (section 3, Projections). The
wiring counts clamped synapses, from the sweep, projections, the cluster boost
and connections tables alike, and the status line reports the count, because
silently truncating long-range delays changes the dynamics.

---

## 3. Connectivity, evaluated once at wiring time

### Pair hash

For the local sweep and the projections, whether a synapse exists, and what
weight it gets, are pure functions of the two stable identities and the
seed:

    pairHash( src_i, lidx_i, src_j, lidx_j, seed )

`src/rand.js`, `pairHash`: a 32-bit avalanche mix (`h32`) of the seed with
the four identity words, divided by 2³², a uniform on [0, 1). `pairGauss`
draws a standard normal from two such uniforms at seed offsets 101 and 202
by Box-Muller. Never of iteration order, and never of any other pair. This
is what makes graph edits additive: adding a population does not rewire the
ones already there. It is also what makes the parallel wiring exact rather
than approximate, since each worker can compute its own slice
independently.

Stable identity is `(source scatter node id, local index)`.

Two kinds of synapse are outside this claim. The cluster boost (below) walks
the finished network with one sequential generator, so each of its synapses
depends on every other pair and on the draw order. A connections table names its
pairs outright and draws nothing. Both are stated where they are described.

### Connection probability

    P(i → j) = p₀ · probMul(pop_i, pop_j) · exp( −d² / (2σ²) )     for d ≤ R
             = 0                                                   otherwise

`src/nodes.js`, `wireConnect` (the `sweep` closure, comparing the pair
hash against `p.prob * T.probMul[tCell] * Math.exp(-d2/s2)`). A Gaussian
distance kernel bounded by a maximum radius, following NEST's spatial
networks. The Gaussian shape is what paired
recordings and reconstructions report for local cortical connectivity:
Hellwig (2000, Biol Cybern 82:111) from axon and dendrite overlap of layer
2/3 pyramids over 0 to 500 µm; Levy and Reyes (2012, J Neurosci 32:5609)
from paired recordings in mouse auditory cortex, Gaussian fits with widths
of 85 to 114 µm for every pair class between pyramids, fast spiking and
non fast spiking cells; Perin, Berger and Markram (2011, PNAS 108:5419)
for layer 5 pyramids, falling with intersomatic distance. The cutoff `R`
is a computational bound, not a biological one: at the connect node's
defaults (σ 120, R 250) the kernel is still at 11 percent of `p₀` where it
is cut, and Boucsein et al. (2011, Front Neurosci 5:32) report that
horizontal inputs from 200 to 1500 µm away make up a large fraction of a
cortical cell's input. A scene that wants the surround raises `R`.

| Symbol | Meaning |
|---|---|
| `p₀` | `prob`, the connection probability at zero distance |
| `σ` | `sigma`, the kernel width in µm (the code stores `s2 = 2σ²`) |
| `R` | `radius`, the hard cutoff; pairs beyond it are never considered |
| `probMul` | per population-pair multiplier from the connect node's table |

The table is resolved once per wiring by `buildPairLUT` in `src/nodes.js`
into three matrices over the distinct source ids, `probMul`, `wMul` and
`plMul` (the probability multiplier, the weight multiplier and the rule id
of each population pair). A row matches a pair by the better of an exact
tag, a tag prefix with `*`, the class key `E` or `I`, and `*`; a plasticity
node that names both ends sets `plMul` for its pair and leaves the other two
alone.

### Weights

    w = w̄ · shape · s · wMul(pop_i, pop_j)

with `w̄ = wExc` if the presynaptic neuron's type has sign +1, else `wInh`.

Two shapes:

    lognormal:   shape = exp( σ_g g − σ_g² / 2 )      g ~ N(0,1) from the pair hash
    uniform:     shape = 0.5 + U                       U ~ U(0,1) from the pair hash

`src/nodes.js`, `wireConnect` (the `sweep` closure; `g` is `pairGauss`
at the pair's own seed, `U` is `pairHash` at `seed + 101`). The lognormal is
mean-preserving, `E[exp(σ_g g)] = exp(σ_g²/2)`, so total drive matches `w̄`
regardless of `σ_g`. Heavy-tailed weights follow Song et al. 2005. `s` is
the proxy scale `1/√r` of the downscaling below (1 at full resolution).

**Sign is a property of the presynaptic neuron type**, read once per
presynaptic neuron as `NEURON_TYPES[ntype[i]].sign` in `wireConnect`, so
every synapse leaving a neuron carries the same sign; a connections file node in its
transmitter mode takes the sign from the connection table's transmitter
column instead (GABA, glutamate and histamine inhibitory, the fly's
convention), still one sign per presynaptic cell. That is Dale's law in its
strong form. It is an approximation: the sign of a real synapse is set
postsynaptically by the receptor and the target's ionic gradients, and there
are documented cases of one axon exciting one target and inhibiting another.

### Cluster boost

`src/nodes.js`, `finishConnect` (the `p.cluster > 0` block), after the
sweep of every presynaptic slice has been assembled. With `cluster` = `c`
and `m` synapses in the sweep, the wiring appends up to

    want = clamp( round( c m ),  0,  MAX_SYN − m )

synapses by triadic closure over the finished network, drawn from one
sequential generator `rc = rng(seed · 31 + 7)` (`src/rand.js`, `rng`,
mulberry32). Each attempt, of at most `30 · want`:

    s₁ ~ U{0 … m−1}          a synapse i → j of the network so far (sweep and projections)
    k  ~ U{ targets of j }   one of j's postsynaptic cells, uniform over its CSR slice
    accept i → k   if   k ≠ i,  probMul(pop_i, pop_k) ≠ 0,  d(i,k) ≤ R

    w(i → k) = w̄ · shape · s · wMul(pop_i, pop_k)
        lognormal:  shape = exp( σ_g g − σ_g²/2 )     g ~ N(0,1) from gaussRand(rc)
        uniform:    shape = 0.5 + rc()
    delay(i → k) = clamp( round( d(i,k) / velocity ), 1, 16 )
    rule(i → k)  = plMul(pop_i, pop_k)

with `w̄`, `s`, `σ_g`, `R` and `velocity` the connect node's, as for the
sweep, and `gaussRand` the Box-Muller normal in `src/nodes.js` fed from
`rc`. A pair the table sets to zero probability is never closed. A clamped
delay is counted with the sweep's. The new synapse's target `k` is counted
in the realized in-degree `K_E,k` or `K_I,k` of the downscaling below. No
check is made that `i → k` is not already a synapse: a closed triangle that
repeats an existing pair adds a second synapse between the same two cells.
Every quantity here comes from `rc`, so these synapses depend on the whole
sweep and on the order of the draws; they are the exception to the purity
claim of the pair hash, and `cluster` = 0 leaves the sweep byte for byte.
The statistic this reproduces is the excess of common neighbors and
reciprocal triplets over a distance-only model in layer 5 pyramids (Perin,
Berger and Markram 2011, PNAS 108:5419).

### Projections

A project node adds a tract between two tagged populations `A` (its `from`
pattern) and `B` (its `to`), evaluated in `wireConnect` after the sweep
(the `pts.projs` loop). A tag pattern is a tag, or a tag with one `*`
standing for any run of characters (`tagMatch`); a projection whose `from`
and `to` both carry `*` is expanded by `expandProjection` into one
projection per matched text, joining copy to copy. The connect table's
`probMul` and `wMul` do not apply to a projection. Every projection mode
shares:

    pseed    = seed + (seed_pr + 1) · 7919          the projection's own hash seed
    v        = max(1, velocity_pr)                  µm/ms
    straight = | centroid(A) − centroid(B) |        µm
    base     = ( tract > 0 ? tract : straight ) / v   ms, the tract delay
    w̄        = sign_i · weight_pr                    sign from the presynaptic type
    w        = w̄ · shape · s                          shape as for the sweep, from pseed
    rule     = 0 if frozen, else the rule the node names (default 1)

| Symbol | Field | Meaning |
|---|---|---|
| `seed_pr` | `seed` | the node's seed, default 1 |
| `weight_pr` | `weight` | weight magnitude, default 0.5; the sign is the presynaptic cell's |
| `velocity_pr` | `velocity` | conduction velocity, default 5000 µm/ms |
| `tract` | `tract` | tract length in µm; 0 means the centroid distance |
| `σ_p` | `sigma` | target spread in µm, default 60 (topographic, proximity, matched) |
| `p_p` | `prob` | probability at zero mapped distance, default 0.35 (same three modes) |
| `F` | `fanout` | contacts per source, default 50 (dispersed) |

`shape` is the connect node's lognormal or uniform draw with the connect
node's `σ_g`, keyed on `pseed`, and `s` is the proxy scale `1/√r`. Pairs
with `j = i` are skipped. Each synapse's delay is `qd(·)`, the rounding and
clamp of section 2.

**Topographic** (`mapping` 0). The two axes other than the node's source
axis are the source's lateral plane, and likewise for the target. Each
source cell's lateral position is normalized in `A`'s bounding box over
that plane, optionally mirrored, and placed in `B`'s box:

    u = (x_i,a1 − min_A,a1) / range_A,a1       v likewise on a2
    u ← 1 − u if mirror u,   v ← 1 − v if mirror v
    (t_x, t_y) = ( min_B,b1 + u · range_B,b1 ,  min_B,b2 + v · range_B,b2 )

    P(i → j) = p_p · exp( −d² / (2σ_p²) ),   d² = (x_j,b1 − t_x)² + (x_j,b2 − t_y)²,   d ≤ 3σ_p
    delay    = qd( base + d / v )

where the pair hash at `pseed` decides against `P`, and `d` is the distance
in `B`'s lateral plane from the mapped point, not between the two cells.

**Dispersed** (`mapping` 1). No map. Every source reaches a seeded subset of
the whole target population:

    P(i → j) = min(1, F / |B|)       for every j in B
    delay    = qd( base )

so each source makes `F` contacts on average whatever the target's size.

**Proximity** (`mapping` 2). The kernel on the actual distance between the
two cells, in three dimensions:

    P(i → j) = p_p · exp( −d² / (2σ_p²) ),   d = |x_i − x_j|,   d ≤ 3σ_p
    delay    = qd( d / v )

The tract length is not used.

**Matched** (`mapping` 3). The map is the population order. Both
populations are sorted by stable identity (`lidx`, then `src`); the source
at rank `r` of `|A|` has a copy `c` in `B`, the cell at the proportional
rank, and reaches the target cells around that copy:

    c        = B_sorted[ round( r · (|B| − 1) / (|A| − 1) ) ]      (B_sorted[0] when |A| = 1)
    P(i → j) = p_p · exp( −d² / (2σ_p²) ),   d = |x_j − x_c|,   d ≤ 3σ_p
    delay    = qd( base )

Two populations laid down by one repeat node are copies in the same order,
so a pattern in one arrives in the other as itself, blurred by `σ_p`. The
delay is the tract delay for every synapse of the projection.

### Connections tables

A connections file node appends explicit synapses from a connections table (one row
per presynaptic id, postsynaptic id, synapse count and optional
transmitter), evaluated in `wireConnectionsFile` in `src/nodes.js` after the sweep
and the cluster boost. Ids resolve against the ids the points file carried
onto its cells; a row whose ids are not on the stream, or whose two ids are
the same cell, is counted and skipped. For a matched row with count `c₀`:

    c  = c₀ if c₀ > 0 else 1
    c  ← min(c, cap)                 if cap > 0
    c  ← √c | log₂(1 + c) | 1        under "count taken as" 1 | 2 | 3   (0 leaves c)

    sign  = the transmitter column's sign under "sign" 1 (a name containing gaba,
            glut, hist or inh, or starting with a hyphen, is −1, else +1),
            otherwise the presynaptic cell type's sign;
            with a receptor node on the stream, the channel's sign
    w     = ( sign > 0 ? 1 : −f_I ) · |weight| · c      mode 0, one synapse
    w     = ( sign > 0 ? 1 : −f_I ) · |weight|          mode 1, max(1, round(c)) synapses
    delay = clamp( round( |x_i − x_j| / max(10⁻⁶, velocity) ), 1, 16 )
    rule  = 0 if frozen, else the rule the node names (default 1), in the low five
            bits; the receptor channel (k + 2) in the top three

| Symbol | Field | Meaning |
|---|---|---|
| `weight` | `weight per synapse` | default 0.5, in the synapse's unit |
| `f_I` | `inhibitory weight factor` | default 1; multiplies an inhibitory row's weight |
| `cap` | `count ceiling` | 0 means none |
| `velocity` | `axon vel µm/ms` | default 200 |

These weights are not multiplied by the proxy scale `1/√r`; the target of
each is counted in the realized in-degree. A rule name the node gives that
no plasticity node declares is an error at wiring time (the sweep's table
falls back to the default rule with a warning instead), and a rule id above
31 is refused since it has no room beside the channel bits. The transmitter
signs follow the fly: GABA and glutamate through GABA-A and GluCl, histamine
at the photoreceptor synapse through HisCl (Hardie 1989, Nature 339:704).

### Proxy downscaling

van Albada, Helias and Diesmann 2015, and the NEST microcircuit K-scaling.
When the built network realizes only a fraction `r` of full in-degree, two
corrections restore the first two moments of the input.

    weights:   w ← w / √r

    bias:      bias_i += (1/r − 1/√r) · 0.001 · ( ν_E K_E,i w_Exc q_E + ν_I K_I,i w_Inh q_I )

where `q` is the charge one spike delivers per unit of weight under the
checkpoint's synapse convention, as the engine delivers it at its 1 ms step:
1 for a kick synapse; for an exponential synapse `1/(1 − e^(−1/τ))` in the
peak convention (`psc` 0, the held accumulator summed over steps: 3.53 at
τE 3, 8.51 at τI 8), 1 in the charge convention (`psc` 1) and `τ` in the
exact convention (`psc` 2); and `|E_rev − v_rest| / (1 − e^(−1/τ))` for a
conductance synapse, with the cell at its row's rest.
The wiring carries the two terms with `q` at one (`finishConnect`) and the
checkpoint's compute applies `q` (`compensationBias`, `chargePerWeight`), since
the convention is the checkpoint's.

`src/nodes.js`: the constant `wScale` in `wireConnect` (the variance term,
applied to every sweep, projection and cluster-boost weight) and
`finishConnect` (the mean term, `net.comp`).

| Symbol | Meaning |
|---|---|
| `r` | realized fraction of full in-degree, the product of three independent reductions |
| `K_E,i`, `K_I,i` | *realized* excitatory and inhibitory in-degrees of neuron `i`, counting every synapse onto `i` from the sweep, the projections, the cluster boost and the connections tables; the mean term multiplies each count by the connect node's `wExc` or `wInh`, whatever weight the synapse itself carries |
| `ν_E`, `ν_I` | assumed presynaptic rates in Hz: set on the connect node, else the rate of each class measured while running at full resolution |
| `0.001` | Hz to per-millisecond |

The variance correction is the `1/√r` on weights; the mean correction is the
DC term. `r` is the product of `density` (a declared fraction of real
density), `kscale` (a smaller cortical footprint at real density) and the
resolution slider. All three reduce in-degree and all three want the same
correction, because the compensation depends on how far in-degree falls short
and not on why.

What this preserves is rates and spectra. What it does **not** preserve is
correlations, and spike-timing plasticity is a function of correlations.
That is why training below 100 percent resolution is refused rather than
merely warned about, and why the override is the project's one labeled
non-biological intervention.

---

## 4. Plasticity

All rules are trace-based and online. No rule looks up a spike history.

### Traces

Each neuron carries three exponentially decaying traces, incremented by 1 at
its own spike and decayed once per millisecond:

    K_pre  ← K_pre  · e^(−1/τ₊)          fast, presynaptic role
    K_post ← K_post · e^(−1/τ₋)          fast, postsynaptic role
    K_slow ← K_slow · e^(−1/τ_y)         slow, for the triplet and het terms

`src/simworker.js`: the decay is the loop after the neuron loop in `step`
and `stepExp` (`Kpre[i] *= decS`), the increments are the last lines of
`plastOnSpike`. Increments happen at the end of the spike handler, so a
spike never sees its own contribution. The slow trace `z` is read *before*
the increment for the same reason: the triplet term potentiates by the
postsynaptic history, not by the present spike. Two cells that spike in the
same millisecond differ between engines: the reference handles spikes in
neuron-index order, so the later one reads the earlier one's incremented
trace, while the WebGPU and CUDA engines run every update of a step before
any increment, so both read the pre-increment traces.

| Symbol | Default | Source |
|---|---|---|
| `τ₊` (`tauS`) | 16.8 ms | Bi and Poo 1998 report ≈ 17 ms |
| `τ₋` (`tauM`) | 33.7 ms | Bi and Poo 1998 report ≈ 34 ms |
| `τ_y` (`tauY`) | 114 ms | Pfister and Gerstner 2006 |

Setting `τ₊ = τ₋ = 20` recovers the symmetric window of Song, Miller and
Abbott 2000, which the plastic scenes declare.

### Excitatory pair STDP, with triplet and heterosynaptic terms

On a **presynaptic** spike, for each outgoing synapse (depression):

    w ← clamp( w − A₋ K_post[j] · (w if wdep else 1) + δ ,  1e-4,  wmax )

On a **postsynaptic** spike, for each incoming synapse (potentiation):

    w ← clamp( w + (A₊ + A₃ z) K_pre[i] − min(1, β g³) (w − w̃) ,  1e-4,  wmax )

`src/simworker.js`, `plastOnSpike`: the outgoing loop over `preStart`
(depression) and the incoming loop over `inStart` (potentiation).

| Symbol | Field | Default | Meaning and source |
|---|---|---|---|
| `A₊` | `aP` | 0.008 | potentiation amplitude. Song, Miller and Abbott 2000 |
| `A₋` | `aM` | 0.001 | depression amplitude. same |
| `wdep` | `wdep` | 1 (soft) | depression proportional to `w`. van Rossum, Bi and Turrigiano 2000; the μ = 1 end of Gütig et al. 2003 |
| `A₃` | `trip` | 0 | triplet amplitude, multiplying the slow postsynaptic trace. Pfister and Gerstner 2006 |
| `δ` | `tin` | 0 | transmitter-induced potentiation, a constant per presynaptic spike. Zenke et al. 2015 |
| `β` | `het` | 0 | heterosynaptic regression amplitude. Zenke et al. 2015 |
| `w̃` | | | the reference weight, see consolidation below |
| `wmax` | `wmax` | 30 | upper bound |

The heterosynaptic gate is

    g = z / k_ref ,      k_ref = τ_y · ρ / 1000

`src/simworker.js`, `plastOnSpike` (`gate`), with `k_ref` set in
`configure` (`S.kRef`). The cube `g³` and the saturation at 1 are the
project's stated adaptation of Zenke's `−β (w − w̃) z³`: normalizing by the
trace at the target rate makes `β` dimensionless and comparable across
scenes, and the saturation stops a burst from erasing a synapse in one step.
`ρ` here is the global `iRho` in every engine, also under measured set
points: the WebGPU incoming pass has no storage slot for a per-neuron
value, and the engines have to agree. A neuron whose measured set point
differs from `iRho` has its competition engage at the global rate, not its
own (Terms in combination, below).

**Why weight-dependent depression is the default.** Additive depression with
hard bounds has no interior fixed point, so with `A₋` above `A₊` and
uncorrelated firing every weight drifts to a bound. Depression
proportional to `w` puts the fixed point at `A₊⟨K_pre⟩ = A₋ w ⟨K_post⟩`,
which is stable and unimodal.

### Inhibitory plasticity

Vogels et al. 2011, a rate controller on the postsynaptic cell.

    on presynaptic spike:   Δw = −η ( K_post[j] − α )
    on postsynaptic spike:  Δw = −η K_pre[i]

    presynaptic:   w ← clamp( w + Δw′,  −wmax,  −1e-4 )
    postsynaptic:  w ← max( w + Δw′,  −wmax )

    Δw′ = softBound(w, Δw) when wdep is on, else Δw

The postsynaptic change is never positive (η is at least 0 on the node and
`K_pre` is a trace), so that half needs no upper clamp to keep `w` below
zero.

`src/simworker.js`, `plastOnSpike`: the `else` branch of the outgoing loop
(on presynaptic) and of the incoming loop (on postsynaptic). The target
enters through

    α = ρ · 0.001 · (τ₊ + τ₋)

`src/simworker.js`, `configure` (`S.iAlpha`). This is the generalization of Vogels' `α = 2ρτ` to the
asymmetric window; with `τ₊ = τ₋ = τ` it reduces to it. The derivation is in
ENGINE.md section 3.

| Symbol | Field | Default | Meaning |
|---|---|---|---|
| `η` | `iEta` | 0.002 | learning rate |
| `ρ` | `iRho` | 5 Hz | target postsynaptic firing rate |

Weights are negative, so `Δw < 0` when the target fires above `ρ` means
stronger inhibition.

### Soft bounds

    a = |w|
    room = (wmax − a)/wmax   if the update moves away from zero
         = a/wmax            if it moves toward zero
    softBound(w, Δ, wmax) = Δ · max(room, 0)

`src/simworker.js`, `softBound`. Applied to both signs when `wdep` is on:
the excitatory depression reads `wdep` as the factor `w` in its own term,
and the two inhibitory updates pass their `Δw` through `softBound`.

The Vogels controller has a fixed point only if the target rate is
reachable; when it is not, it integrates without limit, and every
inhibitory weight reaches `wmax`. Soft bounds make an unreachable target
saturate instead.

### Consolidation

Zenke et al. 2015 equation 16. The reference weight `w̃` follows `w` through
a double well whose lower fixed point is 0 and whose upper one is `w_P`, with
an unstable midpoint at `w_P/2`.

    w̃ ← w̃ + (Δt / τ_cons) · ( w − w̃ − P w̃ (w_P/2 − w̃)(w_P − w̃) )

`src/simworker.js`, `consolidate`, called from `step` and `stepExp` on
the `consAcc` counter. Evaluated every `Δt` = 1200 ms rather than every
step, as the published implementation does and for the same reason.
Excitatory plastic synapses only; inhibitory weights are governed by the
Vogels rule and have no reference.

| Symbol | Field | Default | Meaning |
|---|---|---|---|
| `w_P` | `consW` | 0 | the consolidated (upper) weight; 0 means a tenth of the rule's `wmax` |
| `P` | `consP` | 10 | well depth |
| `cons` | `cons` | 0 | per-rule gate; a rule with `cons ≤ 0` leaves its references where they are, which is what keeps `het` pulling toward the weights as they were |

### Committed synapses

Once a synapse has consolidated it leaves the plastic pool: it keeps its
weight, keeps transmitting, and both halves of the weight update and the
synaptic scaling pass skip it, in all three engines.
There is no new per-synapse state. The reference `w̃` is already the
bistable variable consolidation drives to one of two fixed points, so
"committed" means its reference has reached the upper one, at nine tenths of
the consolidated weight of the rule the synapse belongs to:

    committed(s)   if and only if   w̃_s ≥ 0.9 · w_P

`src/simworker.js`, `plastOnSpike` (the `committed` test, applied in both
loops) and `applyScaling` (the same test). ENGINE.md section 8b. In all
three engines.

| Symbol | Field | Default | Meaning |
|---|---|---|---|
| `commit` | `commit` | 0 | the switch, a control on the checkpoint |
| `τ_cons` | `tauCons` | 1,200,000 ms (20 min) | timescale of the drive into the well, "commit after (sim-min)". Zenke et al. 2015 fix it at twenty minutes; it is exposed because how many presentations it takes to commit depends on the curriculum |
| `Δt` | `consStep` | 1200 ms | cadence of the consolidation step, "commit check every (sim-s)" |

The reference is clamped to the two fixed points of its own well, `[0,
w_P]`, in all three engines. The consolidation step is forward Euler over a
cubic, so a short timescale diverges without the clamp: at a twenty second
constant with the published well depth the reference overshoots, the cubic
grows with the overshoot, and weights reach NaN inside ten simulated
seconds.

None of the three is a rule-table field. Commitment is decided per synapse
from that synapse's own rule, but the switch and the two timings are
checkpoint-level, for the same reason the trace time constants are.

### Synaptic scaling

Turrigiano 2008, multiplicative, as in van Rossum et al. 2000. Once per
second, on incoming plastic excitatory synapses:

    err = clamp( (ρ_i − n_i) / ρ_i ,  −2,  +2 )
    w   ← min( wmax_rule ,  w · (1 + η_s · err) )

`src/simworker.js`, `applyScaling`, called from the tick loop when
`S.t` reaches `nextScale`, which advances by 1000 ms from the moment
scaling switched on (or calibration ended).

| Symbol | Field | Default | Meaning |
|---|---|---|---|
| `n_i` | | | spikes counted for neuron `i` in the last second |
| `ρ_i` | | | that neuron's target rate |
| `η_s` | `sEta` | 0.001 | scaling rate per second |

The ceiling is the *synapse's own rule* `wmax`, not the checkpoint's, in
all three engines. Frozen and committed synapses are skipped. The WebGPU engine
computes the factor on the host from the spike counts of whole ticks, so
its pass falls on the first tick boundary at or after the second and its
factor is `1 + η_s · err · T` with `T` the elapsed seconds since the last
pass (the error uses the rate `n_i / T`); the other two engines scale on the
exact millisecond with `T` = 1.

### Measured set points

Hengen and Turrigiano: individual neurons defend individual rates. For the
first `calS` seconds every rule that moves a weight is off and spikes are
counted (consolidation of the reference weight keeps its cadence), then

    ρ_i = clamp( spikes_i / calS ,  0.25,  30 )   Hz
    α_i = ρ_i · 0.001 · (τ₊ + τ₋)

`src/simworker.js`, `finishCalibration`; the calibration starts in
`configure` when plasticity and `rhoMode` are both on and no set points
exist yet. With `rhoMode` off, the single global `ρ` is used instead. In all
three engines: `finishCalibration` in the reference, the host side of
`tick` in the WebGPU worker (which then writes `α_i` per cell into the
buffer the plasticity passes read), and `finishCalK` in the CUDA child. The
measured `ρ_i` reach the Vogels term and synaptic scaling; the
heterosynaptic gate keeps the global `ρ` (above).

### Short-term plasticity

Tsodyks and Markram 1997, with facilitation as in Tsodyks, Pawelzik and
Markram 1998, in the form of Zenke et al. 2015 equations 9 and 10 and the
Auryn implementation.

On a presynaptic spike, release, then depletion, then facilitation
(`stpOrder` 0, the default):

    rel = R X
    X ← X − rel
    R ← R + U (1 − R)
    delivered weight = w · rel · s

With `stpOrder` 1 the facilitation step comes first, so a rested synapse
releases U(2 − U). The 1998 paper writes the probability as U1, decaying to
zero between spikes and read after each spike's increment; R = U + (1 − U) U1
before a spike, so the two give the same release on every spike.
`tools/published.mjs` checks the engine against the paper's form.

Per millisecond, exact exponentials rather than Euler so the time constants
hold at any step size:

    X ← 1 − (1 − X) e^(−1/τ_D)
    R ← U + (R − U) e^(−1/τ_F)

`src/simworker.js`, `stpRelease` (release, called from the spike branch
of `step` and `stepExp` before the deliveries) and `stpStep` (recovery,
once per millisecond after the neuron loop).

| Symbol | Field | Default | Meaning |
|---|---|---|---|
| `U` | `stpU` | 0.5 | release probability increment |
| `X` | | 1 | available resources, depleted by release |
| `R` | | `U` | release probability, facilitated by use |
| `τ_D` | `stpTauD` | 800 ms | recovery of resources |
| `τ_F` | `stpTauF` | 1 ms | decay of facilitation |
| `s` | `stpNorm` | `stpNorm` 0, so `s` = 1 | `s` = 1 by default; `stpNorm` 1 sets `s` to the inverse of the rested release, `1/U` (or `1/(U(2 − U))` with `stpOrder` 1), so a rested synapse delivers exactly `w` |
| | `stpOrder` | 0 | 0 releases before the facilitation step, as published; 1 facilitates first |

The defaults are the checkpoint node's, the depressing class of Markram,
Wang and Tsodyks (1998). An engine given an init without these fields
falls back to U 0.2, τ_D 200 and τ_F 600 (ENGINE.md 8); a computed network
always carries them.

The published order is release, then facilitation (ENGINE.md 8). A scene
tuned on the other order declares `stpOrder` 1 on its checkpoint.

### The rule table

Plasticity is per synapse, not global. Each synapse carries one byte, the
rule id: 0 is frozen, 1 is the checkpoint's own settings, 2 and above are
rules declared by plasticity nodes. The table is 11 fields per rule in this
order, identical in all three engines:

    aP, aM, wmax, wdep, trip, het, tin, iEta, cons, consW, consP

`src/nodes.js`, `RULE_FIELDS` (the order) and `buildRuleTable` (the
table). Row 0 is never read. A rule that does not name a field takes the
checkpoint's value, and a `consW` of 0 on any row becomes a tenth of that
row's `wmax` before the table is sent.

Deliberately **not** per rule, because they are per-neuron traces rather than
per-synapse quantities: `tauS`, `tauM`, `tauY`, `iRho`, `rhoMode`, `calS`,
`scale`, `sEta`, all short-term plasticity terms, and the commit switch with
its two timings (`commit`, `tauCons`, `consStep`).

### Terms in combination

The rules above come from five papers about five models. Each is correct
on its own; switched on together, some change each other's meaning. Every
statement here is read off the update equations as implemented.

**Two rate controllers on one cell.** The Vogels term (`iEta`, `iRho`) and
synaptic scaling (`scale`, `sEta`) both defend the postsynaptic rate, and
both read the same target, the global `ρ` or the measured `ρ_i`. With both
on, two integrators act on one error: the faster one does the work and the
slower drifts in the same direction, and the excitatory weights become a
control variable, rescaled by a factor that has nothing to do with what was
presented. Scaling is multiplicative, so ratios within a cell survive and
absolute values do not. `iEta` is a rule-table field and `scale` is not, so
a population whose rule sets `iEta` to 0 is held by scaling alone while its
neighbors are held by both.

**Soft bounds change the Vogels rule.** Vogels et al. 2011 is additive with
hard bounds. With `wdep` on, both inhibitory updates pass through
`softBound`, so `Δw` is multiplied by `|w|/wmax` toward zero and by
`(wmax − |w|)/wmax` away from it. The fixed point, where the postsynaptic
rate equals `ρ`, is unchanged when it is reachable, because the factor
multiplies `Δw` and `Δw = 0` there; an unreachable target makes the weights
approach `−wmax` asymptotically instead of reaching it. The rate of
convergence is scaled by the room, so `iEta` means something different at
different weights, and a scene tuned with `wdep` 0 needs its `iEta` retuned
under `wdep` 1.

**Heterosynaptic competition against consolidation and commit.** With
`cons` off, `w̃` is the weight at the moment either term first switched on
(the engines copy the weights into the reference then), so the
heterosynaptic term is a pull toward that initial network that opposes every
learned change. With `cons` on, a synapse commits once `w̃ ≥ 0.9 w_P`, which
needs `w` held above `w_P/2` for about `τ_cons`, while at each postsynaptic
spike the potentiation `(A₊ + A₃ z) K_pre` is opposed by the pull
`min(1, β g³)(w − w̃)`. The weight settles where the two balance, so the
reachable excess over the reference is `(A₊ + A₃ z) K_pre / min(1, β g³)`,
and commit from an initial weight `w₀ < w_P/2` requires

    w₀ + (A₊ + A₃ z) K_pre / min(1, β g³)  >  w_P / 2

on the presentations that should commit, held for `τ_cons`. Strong
competition (`β g³` near or above 1) and commitment cannot both hold for
one synapse; a scene chooses.

**Scaling and committed synapses.** A committed synapse is skipped by both
halves of the STDP update and by the scaling pass, in all three engines.
Scaling therefore carries the cell's whole rate error on the synapses that
have not committed: the more of a cell's inputs commit, the larger the
multiplicative step the rest take for the same error.

**The heterosynaptic gate and measured set points.** The gate is `g =
z/k_ref` with `k_ref = τ_y ρ / 1000`, so `g = 1` when the neuron fires at
`ρ`. Under `rhoMode` 1 the Vogels term and scaling use each neuron's own
`ρ_i`, but `k_ref` is computed once from the global `iRho` in every engine.
A neuron whose measured set point is a fifth of `iRho` has its competition
engage at five times its own rate; one whose set point is four times `iRho`
has it engage at a quarter of its own rate, which is always, so its
synapses are regressed toward the reference on every spike.

**Transmitter-induced potentiation needs a counterweight.** `δ` (`tin`)
adds a constant to every outgoing synapse on a presynaptic spike, whatever
the postsynaptic cell is doing. Onto a silent postsynaptic cell `K_post` is
zero, so the depression term is zero and the synapse grows by `δ` per
presynaptic spike until `wmax`. Zenke et al. 2015 balance it with the
heterosynaptic term; `tin` with neither `het` nor scaling saturates every
synapse from an active source onto a quiet target.

**Triplet amplitude and multiplicative depression.** Pfister and Gerstner
2006 fitted `A₃` for an additive rule. Here the depression is multiplicative
under `wdep` 1, so the fixed point is `(A₊ + A₃ z)⟨K_pre⟩ = A₋ w ⟨K_post⟩`,
stable and shifting upward with the postsynaptic rate through `z`. That is
the qualitative behavior the triplet term is for, and the published
numbers do not carry over: `A₃` has to be tuned against `A₋` in this form.

**The consolidation well against the bound.** `w̃` is clamped to `[0, w_P]`
and commit needs `w̃ ≥ 0.9 w_P`, while `w` never exceeds the rule's `wmax`.
For a weight held at `w`, the reference moves up through the midpoint only
if `w > w_P/2` (the well's push changes sign there), and its stationary
value solves `w − w̃ = P w̃ (w_P/2 − w̃)(w_P − w̃)`, which sits at or above
`0.9 w_P` only if `w ≥ 0.9 w_P − 0.036 P w_P³`. Both bounds are on `w`, so
a rule whose `wmax` is below `w_P/2` never commits. Nothing checks `consW`
against `wmax`; the default `consW` of 0 resolves to a tenth of `wmax` and
satisfies both.

**Short-term plasticity is not a free switch.** With `stpNorm` 0 a rested
synapse delivers `w · U` (`stpOrder` 0) or `w · U(2 − U)` (`stpOrder` 1),
which at the checkpoint's default `U` of 0.5 is half or three quarters of
`w`. Turning `stp` on at the weights a scene was tuned without it cuts
every delivery by that factor, and turning it off on a scene tuned with it
raises them by its inverse. `stpNorm` 1 sets the rested delivery to `w`,
so switching the term changes the dynamics and not the drive.

**Rate targets and the refractory cap.** The absolute refractory period
caps a cell at `1000/(refrac + 1)` Hz, and `refrac` is clamped to 20 ms in
every engine, so the lowest cap is 47.6 Hz. A target above the cap is
unreachable: the Vogels term integrates toward its bound (softly under
`wdep`) and scaling multiplies the excitatory weights up to `wmax`. A
measured set point is clamped to 30 Hz and cannot exceed the cap; the
global `iRho` is not clamped, and nothing compares it to the cap.

**Pairs that do not interact.** Pair STDP and the Vogels rule act on
disjoint synapses, by sign. Short-term plasticity scales the delivered
amount by `rel` and never touches `w`, and STDP updates `w` and never
touches `X` or `R`. Consolidation and commit act on one variable, `w̃`, and
commit reads what consolidation wrote. The measured `ρ_i` reaches the
Vogels term and scaling alike. Delays are whole
milliseconds and the STDP windows are tens of milliseconds, so the
quantization sits below both.

---

## Where each term lives

For every term above, the function that evaluates it in each engine. In
`src/gpuworker.js` the names are the WGSL entry points and functions inside
the shader strings (`SHADER_UPDATE`, `SHADER_DELIVER` and the rest) unless
marked host, which means JavaScript in the same file. CUDA names are
kernels in `cuda/kernels.cuh` unless marked `engine.cu`, whose `main` loop
handles the INIT, TUNE and TICK frames. Where an engine's arithmetic or
timing differs from the reference beyond f32 against f64 and summation
order (the documented residual, ENGINE.md section 4), the cell says so.

| Term | Reference (`src/simworker.js`) | WebGPU (`src/gpuworker.js`) | CUDA (`cuda/kernels.cuh`, `cuda/engine.cu`) |
|---|---|---|---|
| Membrane half steps, spike and reset | `step`, `stepExp` | `update` | `stepKick`, `stepExp` |
| Graded release | `step`, `stepExp` (the `grd` branch) | `update` (amplitude byte in the event list, rounded to 1/254), `deliver` (scales the synapses by it) | `stepKick`, `stepExp` (the `grd` branch) |
| Absolute refractory period | `step`, `stepExp` (`refCnt`) | `update` (low byte of `refcnt`) | `stepKick`, `stepExp` (`refCnt`) |
| Membrane floor | `step`, `stepExp` (`vmin`) | `update` (`U.vmin`) | `stepKick`, `stepExp` (`vmin`) |
| Initial state | computed on the host, `initState` in `src/rand.js`; read from the init message in the `init` branch of `onmessage` | the same `v0`, `u0`, in host `init` | the same, in the INIT frame of `engine.cu` |
| Noise draw | `noiseDraw` in `src/rand.js`, called from `step`, `stepExp` | `pcg`, `unif`, inline in `update` | `pcgH`, `unifH`, `noiseDraw` |
| Total input current, bias, stimulus, external drive | `step`, `stepExp`; protocols in `scaleOf`, `updateAdd`, `updateStim`; input maps in `rebuildExt` | `update`; protocols and input maps folded into `drive` by host `scaleOf`, `updateDrive` | `stepKick`, `stepExp`; the child takes a constant drive (EXT) and a noise amplitude (NAMP) only: `_protocols` in `host/cudaengine.mjs` folds constant and noise protocols and refuses a pulse train or a ramp |
| Kick delivery | `step` (f64 adds into `ring`) | `deliver` (i32 fixed point at 1/4096, atomic) | `stepKick` (f32 `atomicAdd`) |
| Exponential synapse | `stepExp` (`ring`, `ring2`, `gE`, `gI`) | `update` (`ge`, `gi`), `deliver` (E and I ring rows) | `stepExp` (`ring`, `ring2`, `gE`, `gI`) |
| `psc` convention | `itauOf`, in `configure` | host `configure` (its local `itauOf`) | `engine.cu` TICK (`itauE`, `itauI`) and TUNE (channels) |
| Conductance mode | `stepExp` (`cond`) | `update` (`cnd`) | `stepExp` (`cond`) |
| Receptor channels | `configure` (`chanDec`, `chanItau`), `stepExp` (`gX`, `ringX`) | `update` (`gx0` to `gx5`, `U.kx`), `deliver` (channel from the rule byte); fixed at init | `stepExp` (`KX`, `gX`, `ringX`), `engine.cu` TUNE |
| Feedback inhibition | `poolDrive`, `poolSwap`, from `step`, `stepExp` | `update` (`pin`, the count halves in `refcnt`), `poolClear` | `stepKick`, `stepExp` (`poolCnt`, `par`), the memset in `engine.cu` TICK |
| Delays | wiring only: `qd` in `wireConnect`, `finishConnect`, `wireConnectionsFile` in `src/nodes.js`; read as `delay[s]` in `step`, `stepExp` | `deliver` (top five bits of `postd`) | `stepKick`, `stepExp` (`delay[s]`) |
| Traces | decay in `step`, `stepExp`; increments in `plastOnSpike` | `bump` (increments), `decay` | `traceStep` (increment, then decay) |
| Pair STDP depression, transmitter term | `plastOnSpike`, outgoing loop | `plastOut` | `plastOut` |
| Potentiation, triplet, heterosynaptic terms | `plastOnSpike`, incoming loop | `plastIn` | `plastIn` |
| Heterosynaptic gate `k_ref` | `configure` (`S.kRef`) | host `configure` (`S.kRef`), `U.kRef` in `plastIn` | the `kRef` argument of `plastIn`, computed in `engine.cu` TICK |
| Inhibitory plasticity | `plastOnSpike`, both loops; `α` in `configure` (`S.iAlpha`) | `plastOut`, `plastIn`; `α` per cell in the `alphaI` buffer, written by host `configure` | `plastOut`, `plastIn`; `iAlpha` in `engine.cu` TICK |
| Soft bounds | `softBound` | `softBound` (in both plasticity shaders) | `softBound` |
| Consolidation | `consolidate` | `consolidate` (`SHADER_CONS`); cadence in host `encodeSteps` | `consolidateK`; cadence in `engine.cu` TICK |
| Committed synapses | `plastOnSpike` (`committed`), `applyScaling` | `plastOut`, `plastIn`, `scalePass` (`U.commit`, `wRef`) | `isCommitted`, in `plastOut`, `plastIn`, `scaleK` |
| Synaptic scaling | `applyScaling` | factor per cell on the host in `tick`, applied by `scalePass` | `scaleK`, cadence in `engine.cu` TICK |
| Measured set points | `finishCalibration` | host `tick` (the `calibrating` branch) | `finishCalK` |
| Short-term plasticity, release | `stpRelease` | `deliver` | `stepKick`, `stepExp` |
| Short-term plasticity, recovery | `stpStep` | `stpRecover` | `stpStepK` |
| Rule table | `configure` (`S.ruleTable`, row 1 from the scalars) | host `writeRules` (the `Rules` uniform, 16 rows) | `cRules` (constant memory), filled in `engine.cu` TUNE |

Every term is in all three engines. The differences the table marks are
the fixed-point kick delivery on WebGPU, the 1/254 rounding of a graded
release there, the same-millisecond trace order (Traces, section 4), the
tick-boundary cadence of WebGPU scaling (section 4), and the CUDA child's
stimulus protocols, which are constant drive and noise only.

---

## 5. Measures

In `src/spikestats.js`, checked against Elephant 1.2.1 in
`src/spikestats.test.mjs`. Minimum spike counts are enforced (`cv` 4, `cv2`
4, `lv` 4) because these are ratios that will happily return a number from
two intervals.

Let `d₁ … d_n` be the interspike intervals of one train.

**Coefficient of variation.** 1 for a Poisson process, but sensitive to rate
drift.

    CV = std(d) / mean(d)

Population standard deviation, dividing by `n` rather than `n − 1`, which is
what numpy defaults to and therefore what Elephant computes.

**CV2** (Holt et al. 1996). Drift insensitive.

    CV2 = mean over adjacent pairs of   2 |d_{k+1} − d_k| / (d_{k+1} + d_k)

**LV**, local variation (Shinomoto et al. 2003). Also drift insensitive, and
the one of the two that is 1 for a Poisson process, which is why both are
reported.

    LV = mean over adjacent pairs of   3 (d_k − d_{k+1})² / (d_k + d_{k+1})²

**Fano factor**, as the field defines it: over repeated presentations of the
same stimulus to the same unit,

    F = Var(counts) / Mean(counts)

Population variance, matching Elephant's `fanofactor`. That form
underestimates by `(reps−1)/reps`, so with four repeats a genuinely Poisson
response reads about 0.75 rather than 1. Compare Fano values only across
windows with the same number of repeats.

**Population synchrony.** A different quantity from the Fano factor, and
named apart from it for that reason.

    S = Var(population count per bin) / Mean(population count per bin)

Fano is variability of one cell across trials. Synchrony is covariation
across cells within a window. A reader comparing either against a published
figure while holding the other is comparing the wrong thing.

**Mean rate.** `N / duration`, duration in seconds.

---

## 6. Provenance

| Term | Source |
|---|---|
| Neuron dynamics | Izhikevich 2007, chapter 8 |
| Classic presets (rows 0 to 6) | Izhikevich 2003 |
| Absolute refractory period | project addition, sodium inactivation |
| Triangular per-ms noise | project choice, ENGINE.md section 3 |
| Exponential synapse convention | NEST `iaf_psc_exp`, Brian |
| Gaussian distance kernel | NEST spatial networks |
| Lognormal weights | Song et al. 2005 |
| Cluster boost, transitivity | Perin et al. 2011 |
| Projection modes (topographic, dispersed, proximity, matched) | project construction, section 3 |
| Connections tables | the connections table of a connectome release (FlyWire and neuPrint column names); the inhibitory transmitters of the fly, histamine through HisCl: Hardie 1989 |
| Graded synaptic transmission | Juusola, French, Uusitalo and Weckstrom 1996 |
| Conductance synapses, shunting | Brette et al. 2007; Groschner et al. 2022 |
| Receptor channel kinetics, membrane floor | Hille 2001 |
| Feedback inhibition | Lin et al. 2014; Andersen, Eccles and Loyning 1963 |
| Fly rows | Gouwens and Wilson 2009; Wilson and Laurent 2005; Arenz et al. 2017 |
| Proxy downscaling | van Albada, Helias and Diesmann 2015 |
| Pair STDP, trace form | Song, Miller and Abbott 2000 |
| Weight-dependent depression | van Rossum, Bi and Turrigiano 2000; Gütig et al. 2003 (μ = 1) |
| Window time constants | Bi and Poo 1998 |
| Triplet term | Pfister and Gerstner 2006 |
| Inhibitory plasticity | Vogels et al. 2011 |
| Heterosynaptic, consolidation, transmitter-induced | Zenke et al. 2015 |
| Synaptic scaling | Turrigiano 2008; van Rossum et al. 2000 |
| Individual set points | Hengen and Turrigiano |
| Short-term plasticity | Tsodyks and Markram 1997; Tsodyks, Pawelzik and Markram 1998 (facilitation, release order) |
| CV2 | Holt et al. 1996 |
| LV | Shinomoto et al. 2003 |

## What the one millisecond step costs

Measured with `tools/steperr.mjs` on an Intel Core Ultra 9 285K, node 24.19
and Brian 2 2.10.1. The mouse cortical column at resolution 0.15 (2,106 cells,
225,987 synapses, exponential synapses, no plasticity), four simulated seconds
with the first 500 ms dropped, three seeds. One network, computed once: the same
init message goes to the reference engine at its 1 ms step and, through
`src/briancase.js`, to `tools/brian_ref.py` at 0.5, 0.25, 0.1 and 0.05 ms.
Delays stay whole milliseconds at every step, so the integration step is the
only difference between the runs. Brian at 1 ms reproduced the engine's spike
trains exactly, cell for cell, in every seed of both measurements below, which
is what makes the rest of each table a measurement of the step.

The step costs what it costs relative to the membrane it is integrating, so
both row sets are measured. A pyramid on the measured rows of Izhikevich 2007
chapter 8 is 100 pF against 14 nS near rest, which charges over about 7 ms;
the same cell as a 2003 preset carried in the same form is 1 pF against 0.6
nS, which charges over 1.7. The classic rows are what `tools/published.mjs`
reproduces and what the virtual patch rig runs.

**On the measured rows** (2026-09-24), mean over the three seeds with the range
across them, against 0.05 ms:

| Population | error in rate | error in CV | error in synchrony |
|---|---|---|---|
| L2/3e | +13.8% (11.0 to 16.1) | −3.5% | +1.7% |
| L2/3i | −5.7% (−6.5 to −4.7) | −12.5% | +6.8% |
| L4e | −0.5% (−2.1 to +1.2) | −3.2% | −4.3% |
| L4i | −8.0% (−8.2 to −7.6) | −14.5% | +12.2% |
| L5e | +1.9% (+0.2 to +3.1) | −10.2% | −5.1% |
| L5i | −10.9% (−12.2 to −8.7) | −15.3% | +11.5% |
| L6e | +2.0% | −5.1% | 0.0% |
| L6i | −8.5% (−9.7 to −7.3) | −8.6% | +17.6% |

Excitatory populations a little fast or level, inhibitory ones five to eleven
percent slow, every ISI CV a few to fifteen percent low, and synchrony within
twenty percent. The mean rate over the populations reads 13.8 Hz at 1 ms
against 14.9 at 0.05, about seven percent low, and it converges by 0.1 ms.

**The same scene with the synaptic charge exact** (2026-09-24): the checkpoint
on `psc` 2, the weights scaled by `1 − e^(−1/τ)` so the continuous model is the
one the scene's own `psc` 1 describes, three seeds against 0.05 ms:

| Population | error in rate | error in CV | error in synchrony |
|---|---|---|---|
| L2/3e | +8.8% (7.6 to 10.9) | −5.9% | −7.6% |
| L2/3i | −9.2% (−10.1 to −8.6) | −9.7% | +3.2% |
| L4e | +3.5% (1.6 to 4.6) | −2.5% | −5.1% |
| L4i | −8.3% (−8.9 to −7.7) | −13.5% | +18.6% |
| L5e | −0.9% (−1.7 to 0.0) | −15.2% | −10.0% |
| L5i | −9.6% (−10.9 to −8.8) | −17.2% | +9.4% |
| L6e | +6.3% (5.6 to 6.7) | −3.5% | −1.2% |
| L6i | −7.5% (−9.1 to −6.5) | −11.1% | +6.1% |

The mean rate reads 13.5 Hz at 1 ms against 14.5 at 0.05, the same seven
percent. Removing the synaptic asymmetry (1.176 against 1.064, the paragraph
below) moved L2/3e from +13.8 to +8.8 and L6e from +2.0 to +6.3, and left the
inhibitory populations where they were. So the sign pattern of the measured-row
table is not the synaptic term after all: it is the membrane half step, which
costs a cell more the faster it fires, and the inhibitory populations fire at
10 to 30 Hz against 1 to 6 for the excitatory ones. What the exact integral
buys is a convention, the charge NEST's exact integration delivers at any step;
what it does not buy is a smaller step error, which only a smaller step would.

**On the classic rows** (2026-09-21), the same scene on the 2003 presets, mean
over three seeds:

| Population | rate at 1 ms | rate at 0.05 ms | rate error | CV error | synchrony error |
|---|---|---|---|---|---|
| L2/3e | 1.80 Hz | 0.99 Hz | +83% | −21% | +54% |
| L2/3i | 3.72 Hz | 3.79 Hz | −2% | −5% | +16% |
| L4e | 1.67 Hz | 1.24 Hz | +35% | −5% | +21% |
| L4i | 4.30 Hz | 4.36 Hz | −1% | +18% | +13% |
| L5e | 7.61 Hz | 7.19 Hz | +6% | −18% | +11% |
| L5i | 6.41 Hz | 7.41 Hz | −13% | +9% | −3% |
| L6e | 1.78 Hz | 1.28 Hz | +39% | −6% | +23% |
| L6i | 5.13 Hz | 5.93 Hz | −13% | +8% | −1% |

Spread across those three seeds was a few points: L2/3e's rate error ran 80 to
89 percent, L4e's 30 to 39, L6e's 36 to 41, and the values settled by 0.25 ms
(the mean rate over the populations read 3.98, 3.98 and 4.01 Hz at 0.25, 0.1 and
0.05 ms), so 0.05 ms is a converged reference rather than another point on a
slope. That mean over populations moved from 4.07 Hz at 1 ms to 4.01 at 0.05,
which was the point of the classic-row table: a whole-network rate check could
not see an error that was pulling the populations apart, excitatory ones too
fast and inhibitory ones too slowly, worst where the rate was lowest.

So the fixed step costs most of what it cost on rows whose membrane charges
faster than the step resolves, and most of that cost is gone on rows that carry
a measured capacitance. What is left is the membrane half step on the populations
that fire fastest, which the exact-charge table above isolates.

Two sources, of opposite sign on a single cell:

- The synaptic current is held at its post-decay value for the whole step,
  so the charge one spike delivers is `dt/(tau (1 − e^(−dt/tau)))` times the
  exact integral of the same exponential: 1.176 at `tauE` 3 ms and 1.064 at
  `tauI` 8 ms with a step of 1 ms, against 1.017 and 1.006 at 0.1 ms.
  Excitation is over-delivered nearly three times as much as inhibition.
- The membrane is integrated by two half steps, which on its own makes a
  cell fire slower, and more so the faster it fires. One classic RS row at a
  constant current, no synapses, 4 s: 10.0, 19.5 and 36.8 Hz at 1 ms against
  10.5, 22.4 and 43.2 Hz at 0.1 ms for currents of 5, 10 and 20. The 0.1 ms
  figures agree with `tools/izh2003_cells.py`, forward Euler at 0.1 ms on the
  published equations and a separate implementation (10.5, 22.1, 42.9 Hz).

Both push the same way in a network of excitatory and inhibitory cells: the
inhibitory populations, which fire fastest, lose the most to the membrane term,
and excitation gains the most from the synaptic term. On the classic rows the
membrane term is the larger of the two, so rows with a seven millisecond
membrane carry much less of the error; the synaptic term
does not depend on the cell at all, and it is what the measured-row table above
is showing. Running the classic-row column with both time constants at 30 ms,
where the synaptic factor is 1.017 on each side instead of 1.176 against 1.064,
left the rate errors at +21 to +31 percent on the excitatory populations and −9
to −19 on the inhibitory, and took the synchrony error from +54 percent to
within 7.

On the classic rows the error grew with the network: the same column at
resolution 0.4 (5,612 cells, 1,606,842 synapses, an in-degree of 286 against
107 at resolution 0.15) put L2/3e at +161 percent instead of +83 and L6e at +62
instead of +39, and its synchrony error reached +342 percent on L2/3e, because
at 1 ms that run had gone into a synchronized regime and at 0.1 ms it had not.

On the measured rows it does not. The same comparison at resolution 0.4
(2026-09-28, 5,612 cells, 1,493,080 synapses, three seeds, Brian at 1 ms again
identical to the engine cell for cell) gives rate errors of -7.4 to +7.1
percent in every population (L2/3e +7.1, L4e -4.7, L5e -4.4, L6e -7.2, the
inhibitory populations -3.3 to -7.4), ISI CV errors of -1 to -11 percent and
synchrony errors within 17 percent, the same size as at resolution 0.15 or
smaller. The excitatory populations that were most sensitive to the step on the
classic rows are the least sensitive here. Neither table reaches full density,
which is out of reach of this comparison, but on the measured rows nothing
between the two resolutions suggests the error grows toward it.

The measurement runs on exponential or conductance synapses. A kick
synapse's weight is a current applied for one step, so its charge follows
the step and the mode has no meaning at another one; `brian_ref.py` refuses
it below 1 ms.

## What is not modeled

- No dendrites. Every neuron is a point; synaptic input sums linearly with no
  cable filtering and no compartmental nonlinearity.
- Conductance synapses are optional (`syn` 2, section 2). In kick and exp
  modes currents are injected, so there is no
  driving-force dependence and no shunting inhibition in those modes.
- No astrocytes, no neuromodulation as a signal, no gap junctions.
- No axonal delay-dependent plasticity: STDP uses emission times, and delays
  affect delivery only.
- Synapse formation is distance-dependent at wiring time only. Existing
  synapses do not form or prune during a run.
- The step is fixed at one millisecond and delays are whole milliseconds
  clamped to 16 (section 2, Delays). A path longer than 1.6 mm at 100 um/ms
  cannot carry its own delay, and timing differences below a millisecond are
  not represented at all. The wiring counts the clamped synapses and the status
  line reports the count. What the fixed step costs is measured in the
  section above.
- A kick synapse's weight is a current applied for one step, so its charge is
  the weight times the step and the mode has no meaning apart from the
  millisecond. Exponential and conductance synapses carry a time constant and
  do.

# ENGINE.md: simulation engine protocol

The contract between the page and the three simulation engines:
`src/simworker.js` (the CPU reference), `src/gpuworker.js` (WebGPU) and
`cuda/engine.cu` with `cuda/kernels.cuh` (the native child behind
`host/host.mjs`). The reference engine defines correct behavior; every
other engine must satisfy this contract and the statistics battery
(`validate.html`). A change to the protocol updates this file, all three
engines and the battery together.

Engine selection is the checkpoint node's engine parameter. `auto`
chooses between the CPU and WebGPU engines by measurement: the WebGPU
engine pays a submit and a readback per tick (2.71 ms, measured
2026-08-30 on a 10.06M synapse network, against 0.2425 ms per step on the
GPU and 0.4774 ms per step on the CPU), so the round trip, not the
arithmetic, decides. Both per-step costs scale with synapse count, which
makes the boundary one product, `steps * synCount >= 1.16e8`
(`GPU_BREAKEVEN` in `src/document.js`). The fit is conservative: the GPU
scales better than linearly as occupancy improves, so some mid-size
networks stay on the CPU that the GPU would have won. `auto` never
selects an engine for a network using a term that engine does not
implement; every term below is implemented in all three. The CUDA engine
is selected only as the training engine, from the training dialog.
`remote` runs the live view on an engine at the checkpoint's engine
address (`liveHost`) over the WebSocket transport of section 7: `host/host.mjs`
or an engine of one's own (section 15).

## 1. Network interchange format

A wired network (produced by the connect node's compute, consumed by any engine)
is a plain object of typed arrays:

- `count` (int): neuron count n.
- `pos` (Float32Array, 3n): micrometer positions. Connectivity, weights
  and delays are derived from them at wiring time (the connect node's
  compute), so the synapse arrays below already carry everything the
  integrator needs; the engine does not read positions. They travel with
  the network for the viewer and for inputs computed by position (noise
  fields, image encoders).
- `ntype` (Uint8/Int array, n): index into the neuron type table.
- `bias` (Float32Array, n): constant input current per neuron.
- `src` (array, n) and `lidx` (Uint32Array, n): stable identity,
  (source scatter node id, local index). Engines treat these as opaque.
- `pool` (Uint16Array, n, optional) and `poolK` (Float32Array, optional):
  feedback inhibition (section 14). `pool[i]` is the pool cell i belongs
  to, 0 for none; `poolK[g]` is what every cell of pool g receives one
  millisecond after each spike in it, `poolK[0]` is 0. Wired by connect
  from the pool nodes in the stream (`wirePools` in `src/nodes.js`).
  Absent means no pools.
- `pmask` (Uint8Array, m, optional): the per-synapse plasticity byte,
  wired by connect (pair-table fifth column, project node rule field).
  The low five bits are the rule id: 0 freezes the synapse against STDP,
  homeostasis, scaling and consolidation; 1 selects the checkpoint's own
  amplitudes; 2 and above select a rule declared by a plasticity node, in
  the order those nodes appear in the stream. The top three bits are the
  receptor channel (section 3), zero meaning by weight sign into E or I.
  Every reader masks the rule id with `& 31`. Absent means every synapse
  takes rule 1 on the default channel. The array holds an index, not a
  mask bit; the name is kept because the brain file format stores the
  array under it.
- Synapses in CSR order by presynaptic neuron:
  - `preStart` (Int32/Uint32Array, n+1): synapse range of neuron i is
    [preStart[i], preStart[i+1]).
  - `post` (Uint32Array, m): postsynaptic neuron per synapse.
  - `w` (Float32Array, m): signed weight. Positive excitatory,
    negative inhibitory (Dale's law holds per neuron but engines must
    not assume it).
  - `delay` (Uint8Array, m): conduction delay in whole ms, 1 to 16.

13 bytes per synapse resident (post 4, w 4, delay 1, reverse index
amortized 8 more while plasticity is on).

## 2. Message protocol

Requests (main thread to engine):

- `hello`: `{ cmd }`. Sent first, before `init`. The engine answers with a
  `hello` reply (below) saying what it is and what it implements, and
  needs no network to do so. The sender refuses an engine whose reply
  names another contract number (`checkHello` in `src/protocol.js`),
  refuses to run a network that uses a term the reply does not name
  (`missingTerms`), and sends no optional query the reply does not name.
  The page and `tools/linen.mjs` send the init only once the reply has
  been checked, so a network an engine cannot run never reaches it; the
  page stops an engine that does not answer within ten seconds, and a
  tune made while it waits is carried by the init. The init is still
  checked by every engine on its own, so a sender that skips hello is
  refused the same way on a mismatch.
- `init`: `{ cmd, protocol, count, ntype, types, bias, preStart, post, w,
  delay, seed, v0, u0, ...tunables }`. `protocol` is the number of this
  contract the sender was written against (`PROTOCOL` in
  `src/protocol.js`, carried by `engineConfig`); every engine compares it
  with its own and refuses any other number, or none, with a message
  naming both, so a page and an engine loaded from different builds fail
  rather than run each other's bytes. The engine worker URLs carry the
  same number (`simworker.js?p=3`), so a page never loads an engine
  script of another contract by name. Any change to a message or frame
  shape bumps the number. `types` is the resolved type table: an `f7`
  row per type with the Izhikevich 2007 membrane's C, k, vr, vt, vpeak,
  a, b, c, d, and `graded` where the row releases instead of spiking;
  `engineTypes()` in `src/nodes.js` builds it. The table holds one
  membrane form; an init that carries a `form` field with any value but 1
  is refused by every engine. `v0` and `u0` (Float32Array, n) are the
  initial membrane state, computed once by the host with `initState` in
  `src/rand.js` (v = vr + U(0, min(10, (vt − vr)/2)), u = b (v − vr); on a
  classic row v = c + U(0, 10); from an xorshift stream keyed by `seed`)
  and sent, so every engine starts from the same bytes. An engine refuses
  an init without them rather than seeding its own. `seed` also keys the
  per-millisecond noise (section 9). An init also applies every tunable
  below.
- `tune`: `{ cmd, ...tunables }`. Re-applies tunables to the running
  state without touching v, u, ring buffers, or learned weights.
  Tunables: `bias` (Float32Array), `syn` (0 kick, 1 exp, 2 conductance),
  `tauE`, `tauI`, `psc` (exp mode: 0 peak current, 1 total charge, 2 exact
  integral; section 3), `tauM` (depression trace time constant; section
  3), `stpNorm` (short-term plasticity delivery scale; section 8),
  `stpOrder` (0 release then facilitation, 1 the reverse; section 8),
  `protocols` (list of `{ idx, amp, mode, period, width, t0, duration,
  gain }`; mode 0 constant, 1 pulse train, 2 ramp, 3 noise amplitude;
  `gain`, optional, is a Float32Array aligned with `idx` that multiplies
  `amp` per cell, the stimulus's spread; the CUDA engine runs constant and
  noise protocols only and refuses a pulse or ramp), `ruleTable`
  (Float32Array, optional) and `ruleCount` (int): the per-rule amplitudes
  the synapse byte indexes. Row stride is 11 floats in this order, and
  the order is part of the protocol because the workers are
  self-contained and restate it rather than importing it: `aP`, `aM`,
  `wmax`, `wdep`, `trip`, `het`, `tin`, `iEta`, `cons`, `consW`, `consP`.
  Row 0 is never read; row 1 is the checkpoint's own set; rows 2 up are
  declared rules. An engine given no table builds row 1 from the scalars
  below, so a network without a plasticity node computes by the same
  path. The ceiling is 16 rows (`RULE_ROWS` in `src/protocol.js`), set by
  the WebGPU uniform; `checkProtocol` refuses a larger table.

  Not per rule: `tauS` and `tauY`, because the traces they govern (Kpre,
  Kslow) are per neuron and two rules wanting different windows onto one
  cell would each need their own copy of that state; and `iRho`,
  `rhoMode`, `calS`, `scale`, `sEta` and the short-term terms, which
  belong to a neuron rather than to a synapse. Those are global. A
  per-pathway window (Larsen et al. 2014 report a longer timing-dependent
  LTD window at L4 to L2/3 than at L2/3 to L2/3) is therefore not
  expressible.

  `plast` (0/1), `aP`, `aM`, `tauS`, `wmax`, `iEta`, `iRho`,
  `cons` (consolidation of the heterosynaptic reference weight; any value
  above zero turns it on, the rate is the timescale `tauCons`),
  `consW` (upper stable fixed point of the well; 0, the default, means a
  tenth of the row's `wmax`, resolved when the rule table is built) and
  `consP` (well depth, default 10),
  `scale` (0/1) and `sEta` (homeostatic synaptic scaling, Turrigiano
  2008 / van Rossum 2000: once per simulated second, counted from when
  scaling turns on or the set point calibration ends, each neuron's
  plastic excitatory inputs multiply a step of at most 2 x sEta toward
  the iRho target rate; the WebGPU engine estimates rates from its
  uncapped per-tick counts and scales at tick boundaries, inside the
  statistical contract), `rhoMode` (0/1) and `calS` (measured per-neuron
  set points, MODEL.md section 4), `refrac`, `vmin`, `eRevE`, `eRevI`,
  `chanTau`, `chanErev` (section 3), `commit`, `tauCons`, `consStep`
  (section 8b), `stp`, `stpU`, `stpTauD`, `stpTauF` (section 8), `trip`,
  `tauY`, `het`, `tin` (section 11).

  `custom` (object, on every init and tune): the entries of the engine
  settings nodes between connect and the checkpoint, keys to values as
  they were typed, a value that reads as a number sent as a number
  (`parseEngineSettings` in `src/nodes.js`), `{}` when there are none.
  It is for settings of an engine of one's own that no tunable above
  carries. The page passes it untouched and the remote host forwards it
  with the rest of the message. An engine that reads it names `custom`
  in its hello reply; the reference, WebGPU and CUDA engines do not and
  ignore it, and the page and `tools/linen.mjs` say how many keys the
  running engine ignores ("engine settings: 3 keys ignored by the
  reference engine").
- `tick`: `{ cmd, steps }`. Advance `steps` ms.
- `watch`: `{ cmd, idx }`. Select one neuron for a per-ms voltage
  trace (-1 disables).
- `sendV`: `{ cmd, on }`. While on, every state reply carries the full
  membrane potential array (viewer potential filter). Off by default;
  engines pay the readback cost only while it is on.
- `sendT`: `{ cmd, on }`. While on, every state reply carries `lat`,
  each neuron's first-spike step within the tick (255 for silent). Off
  by default. It may arrive before init and holds across one. The
  reference and CUDA engines stream it; the WebGPU engine answers
  `unsupported`.
- `inputFrame`: `{ cmd, mi, gains (Float32Array) }`. Sets the current
  per-channel gains of input map `mi`. Input maps arrive as an
  `inputs` field on init or tune: a list of
  `{ chStart (Uint32Array C+1), chIdx (Uint32Array), chW
  (Float32Array), amp }`, a channel-to-neuron CSR built by encoder
  nodes. Each neuron receives `sum(gain_c * amp * chW)` as constant
  current until the next frame; gains default to zero. Frames arrive
  at display rate (tens of ms), never per sim-ms.
- `query`: `{ cmd, idx, cap, dir }`. Connection lookup for the viewer;
  `cap` defaults to 2000 on every engine. The reference engine answers
  incrementally so large networks never stall the sim; the WebGPU engine
  answers in one scan. `dir` is optional and the only accepted value is
  `'out'`: the reply then carries `inn` empty and `inTotal` 0, on every
  engine. Finding incoming synapses means walking the whole synapse
  array on the two JavaScript engines; the CUDA child reads them from
  the reverse index its plasticity passes build.
- `getWeights`: `{ cmd }`. Read back the current weight array
  (checkpoints, analysis).

Replies (engine to main thread):

- `hello`: `{ cmd, protocol, engine, terms, optional }`. `protocol` is the
  engine's contract number; `engine` a short name (`reference`, `webgpu`,
  `cuda`, or the name a custom engine gives itself); `terms` the terms
  beyond the baseline the engine implements and `optional` the optional
  queries it answers, both lists of names from `TERMS` and `OPTIONAL` in
  `src/protocol.js`, with `custom` among the terms of an engine that
  reads the custom block. The baseline every engine implements is the init's
  network (cell count, type table, bias, CSR synapses with delays of 1 to
  16 ms, seed, `v0`, `u0`), kick delivery, `tune` of the bias, `tick` and
  the `state` reply with `fired`, `spikes` and `steps`. A term is named
  by the tune field that switches it on, with the value where the field
  selects a mode: `syn:1` exponential synapses, `syn:2` conductance,
  `psc:1` and `psc:2` the two other exp conventions, `refrac`, `vmin`,
  `protocols:0` to `protocols:3` the four stimulus modes, `inputs` input
  maps, `graded` graded rows, `plast` pair STDP with the inhibitory rule
  (`aP`, `aM`, `tauS`, `tauM`, `wmax`, `wdep`, `iEta`, `iRho`), `rhoMode`,
  `scale`, `trip`, `het`, `tin`, `cons`, `commit`, `ruleTable` (named
  plasticity rules), `stp`, `stpNorm`, `stpOrder`, `chanTau` (receptor
  channels) and `pool` (feedback inhibition). The optional queries are
  `watch`, `sendV`, `sendT`, `query` and `getWeights`; `host/host.mjs`
  adds `host` (the run controller, section 7) to its engine's reply, and
  `src/remoteworker.js` adds `url`, the address the reply came from. The
  reference engine names every term and every query; the WebGPU engine
  every term and every query but `sendT`; the CUDA engine every term but
  `protocols:1` and `protocols:2`, and every query. The checkpoint
  panel grays a setting whose term the running engine does not name.
- `state`: `{ cmd, fired (Uint8Array n), spikes, steps, vtrace
  (Float32Array steps) }` after each tick, where `fired[i]` is neuron i's
  spike count over the tick, capped at 255, on every engine; plus
  `v (Float32Array n)`, the post-step membrane potentials, while sendV is
  on; plus `lat (Uint8Array n)` while sendT is on. The WebGPU engine's
  `vtrace` holds at most 64 steps, the most recent, where the other
  engines return the whole tick.
- `unsupported`: `{ cmd, of }`. The engine cannot provide what `of`
  names (a stream switch); nothing else changes.
- `queryResult`: `{ cmd, idx, out, inn, outTotal, inTotal }` where
  out/inn are capped lists and the totals are exact.
- `weights`: `{ cmd, w (Float32Array m), t, rho (Float32Array n, optional)
  }`. `t` is the engine's step count on the two JavaScript engines and 0
  from the CUDA host; nothing reads it. `rho` carries the measured
  per-neuron homeostatic set points when `rhoMode` is 1 and calibration
  has finished, and is absent otherwise.
- `error`: `{ cmd, message }`. A refused init or tune, a device error
  (section 10) or a dropped remote connection (section 7).

Large arrays move by transfer, never copy.

## 3. Dynamics contract

- Izhikevich 2007 membrane, dt = 1 ms, two 0.5 ms substeps for v, one
  1 ms step for u: C dv = k (v − vr)(v − vt) − u + I, du = a (b (v − vr)
  − u), spike at the type's vpeak (MODEL.md section 1). One membrane in
  every engine; the 2003 presets are rows of it (C 1, k 0.04, the rest at
  the lower root of 0.04 v² + (5 − b) v + 140, vt = −125 − vr), derived
  term for term in `src/form.js` and held by the battery against the
  2003 arithmetic. In the WebGPU worker the per-neuron parameter buffer
  is 12 floats wide (a, b, c, d, C, k, vr, vt, vpeak, release threshold,
  slope, graded flag) and the state buffer is 12 floats per neuron (v,
  u, gE, gI, six channel conductances, padding); the u32 after `vmin` in
  its Step struct is `stpOrder`. In the CUDA child the five membrane
  arrays ride at the tail of the INIT frame (section 13), and the child
  checks the frame length.
- Graded rows (MODEL.md section 1): a type row with `graded: { thr,
  slope }` never spikes; each ms it delivers `w r` through its synapses
  with `r = clamp((v − thr)/slope, 0, 1)`, no short-term plasticity, and
  its outgoing synapses are frozen (rule 0) at init by every engine
  (`freezeGraded` in `src/nodes.js`). `gradedArrays` in `src/nodes.js`
  builds the per-cell flag, threshold and slope. The WebGPU event list
  packs `index | amplitude << 24` (255 a spike, 0 to 254 a release,
  index in 24 bits); the delivery pass scales by the amplitude and the
  plasticity passes skip non-spike entries. So WebGPU rounds a release
  to the nearest 1/254 and delivers nothing below 1/508, where the other
  engines deliver the exact release. The CUDA INIT frame ends with u32
  graded, then under 1 u8 grd[n], f32 thr[n], f32 slope[n]. `fired`
  stays 0 for a graded cell, so its rate reads 0; its release is a
  function of `v`.
- Conductance mode (MODEL.md section 2): `syn` 2 is the exponential
  synapse with `I = Σ |g_c| (E_c − v)`; `eRevE`, `eRevI` and `chanErev`
  (one per receptor channel) travel with init and tune. The WebGPU Step
  struct carries the eight reversals (256 bytes; the dynamic-offset slot
  is 256); the CUDA tune frame carries eE at 148, eI at 152 and six
  channel reversals from 156 (section 13).
- Receptor channels (MODEL.md section 2): `chanTau` on init and tune,
  the time constants of channels 2 and up (at most six). Each synapse's
  channel is the top three bits of its plasticity byte (section 1), zero
  meaning by weight sign into E or I. The CPU reference keeps `gX` and
  `ringX` per extra channel; the WebGPU ring has a row set per channel
  after E and I, the Step struct carries the six decays and scales, and
  the delivery pass binds the packed plasticity bytes to route each
  synapse. The CUDA tune frame carries u32 count at 120 and six f32 taus
  from 124. Channels need exponential synapses and every engine refuses
  them under kick.
- Input current per neuron per ms: ring-buffer delivery + bias +
  protocol DC + noise `amp * (U(0,1) + U(0,1) - 1)` per ms.
- Spike at v >= the row's vpeak (graded rows never spike): v = c,
  u += d, deliveries scheduled at t + delay per synapse.
- `vmin`: membrane floor in mV, default -90 (the potassium reversal
  potential), applied after each half step: `v = max(v, vmin)`. 0
  disables it. Without it a hyperpolarizing step past about -150 mV
  sends the quadratic to infinity and the cell to NaN. Implemented in
  all three engines; MODEL.md section 1.
- `refrac`: absolute refractory period in whole ms; 2 on the checkpoint
  node, 0 in an init that omits it, clamped to 20, on every engine. For
  that many steps after a spike the membrane is held at reset and cannot
  reach threshold whatever the drive, while u keeps advancing and the
  exponential conductances keep decaying and accumulating. It caps a
  neuron at 1000/(refrac+1) Hz. The Izhikevich reset alone gives
  relative refractoriness only: nothing in it prevents a
  next-millisecond spike. Implemented in all three engines.
- Synapse modes: kick (delivered charge added to input directly) or
  exp (separate E and I first-order accumulators, time constants tauE
  and tauI). What w means in exp mode is `psc` (`itauOf` in
  `src/simworker.js`, the same factor in the other two engines). `psc`
  0, the default: the current jumps by w at delivery and decays, so the
  charge per spike is w tau; this is the NEST `iaf_psc_exp` and Brian
  convention, where w is a current. `psc` 1: the jump is scaled by
  (1 - e^(-1/tau)) so the charge summed over 1 ms steps is exactly w,
  the charge kick mode delivers, so the two modes are comparable and w
  is a charge. `psc` 2: w is the peak as in 0, and the current applied
  over a step is the integral of the exponential across that step,
  tau (1 - e^(-1/tau)), so the charge per spike is w tau exactly; the
  held current of mode 0 delivers 1/(tau (1 - e^(-1/tau))) times that,
  1.18 at tauE 3 and 1.06 at tauI 8 (MODEL.md, what the one millisecond
  step costs). The Brian reference at a step below one millisecond
  scales it as tau (1 - e^(-dt/tau)) / dt, so the mode names the same
  continuous model at every step. A scene declares the convention it was
  tuned under on its checkpoint.
- Plasticity (when on): pair STDP on excitatory synapses in the online
  trace form of Song, Miller and Abbott 2000 with two window time
  constants: `tauS` for the presynaptic trace, which sets the
  potentiation window (pre before post), and `tauM` for the postsynaptic
  trace, which sets the depression window (post before pre). The node's
  defaults are the Bi and Poo 1998 fit, 16.8 and 33.7 ms (an init that
  omits `tauM` gets `tauS` on every engine); Song et al. use 20 for
  both, and setting the two equal recovers that. The battery's window
  group checks that depression decays more slowly than potentiation at
  the defaults. Plus homeostatic inhibitory plasticity (Vogels et al.
  2011). Its target term is alpha = rho (tauS + tauM) with rho in kHz,
  which is the published 2 rho tau when the constants are equal. The
  generalization follows from the rule's fixed point: per unit time the
  pre-spike term contributes rho_pre (rho_post tauM - alpha) and the
  post-spike term rho_post rho_pre tauS, and the sum vanishes at
  alpha = rho_post (tauS + tauM). Both rules read the same two traces,
  so the inhibitory window is asymmetric whenever the excitatory one is.
  Interactions use spike emission times; delays affect delivery only.
  Excitatory weights clamp to [1e-4, wmax], inhibitory to
  [-wmax, -1e-4].
- `wdep` selects soft weight bounds on both signs. Excitatory
  depression scales with the current weight (van Rossum, Bi and
  Turrigiano 2000), giving an interior fixed point at A+ / A- instead
  of a walk to a bound. Inhibitory updates scale by the room left in
  the direction they are heading, so a homeostatic target the network
  cannot reach saturates instead of pinning every inhibitory weight to
  the clamp. The Vogels rule is a rate controller and has a fixed point
  only when its target rate is achievable; when it is not, an additive
  form integrates without limit.

## 4. Determinism and equivalence

- Wiring is bit-deterministic (same graph + seeds, same network); the
  battery gates this.
- The running sim is deterministic per seed. The initial state is
  computed by the host and sent (section 2), and the per-millisecond
  noise is a counter hash of (neuron, t, seed) that all three engines
  compute in f32 from the same code (section 9). Nothing in any engine
  draws from an unseeded or engine-private generator.
- What separates the engines is summation order and precision: the
  reference sums deliveries in neuron-index order in f64, the WebGPU
  engine accumulates fixed-point integers atomically (order independent,
  quantized to 1/4096), the CUDA child accumulates f32 atomically (order
  dependent). WebGPU also rounds a graded release to 1/254 (section 3),
  and the WebGPU and CUDA engines read pre-increment traces for
  same-millisecond spike pairs where the reference resolves them in
  neuron-index order (MODEL.md section 4, Traces); both conventions are
  arbitrary at 1 ms resolution. Engines are therefore equivalent when
  they match statistically, and the equivalence group gates that.
- The battery's cross-engine equivalence group runs one wired network on
  the reference and WebGPU engines (and again with the partition cap
  forced low, so several synapse partitions are exercised) and compares
  population rate, ISI CV and population synchrony, with short-term
  plasticity off and on in both release orders; the term must move the
  rate on each engine. `cuda/parity.mjs` is the corresponding gate for
  the CUDA child (section 13).

## 5. Where the wiring runs

The connect node's wiring runs on the CPU, split by presynaptic neuron across a
pool of `src/wireworker.js` instances; the split is exact because pair
existence and weight are pure functions of the two stable identities and
the seed (MODEL.md section 3). The synapse count is capped by the SYN
budget in the top bar (`maxSynLimit` in `src/nodes.js`; the machine's
own limit unless overridden), and wiring past it is refused with a
message naming the cap. No wiring runs on the GPU.

## 6. GPU layout requirements

- The synapse store is partitioned into contiguous presynaptic ranges
  whose CSR slice stays under the device's `maxStorageBufferBindingSize`
  (delivery-list layout after NeMo, Fidjeland 2009). No code path may
  assume one buffer; a single neuron whose slice exceeds the cap is an
  init error. Each partition packs `post` and `delay` into one u32
  (delay in the top five bits).
- Delivery accumulates in i32 fixed point at 1/4096 (atomic, order
  independent); plasticity, traces and the membrane are f32.
- The interchange format in section 1 stays the format every engine
  receives; the GPU engine repacks on init.
- Each compute pass is held to the eight storage buffers per stage that
  WebGPU guarantees; a new per-synapse or per-neuron array in a pass at
  that limit needs a slot freed first (section 11 lists the interleaves
  that free them).

## 7. WebSocket transport (remote engine host)

The protocol above also crosses a WebSocket, unchanged. `host/host.mjs`
runs the reference engine (`src/simworker.js`, imported as-is through a
`worker_threads` shim, `host/engineworker.mjs`) and speaks to the page
through the framing `src/frame.js` defines:

- one message per binary WebSocket frame:
  `u32 headerLen (LE) | JSON header | payloads, each 8-byte aligned`
- the header is the message object with every typed array replaced by a
  `{ __ta: { d: dtype, n: length } }` descriptor; payloads follow in the
  order the descriptors appear in a depth-first walk of the header,
  which JSON round-trips preserve
- one engine per connection; connecting spawns it, disconnecting
  terminates it; there is no session resume, a reconnect is a fresh init

The browser side is `src/remoteworker.js`, an adapter with a Worker's
surface (postMessage, onmessage, terminate), selected in the trainer
with `engine=remote` and `host=ws://...` (default `ws://localhost:8801`),
and for the live view by the checkpoint's engine `remote` with its
engine address (`liveHost`, the same default). The live view and the
trainer's host are separate settings: a training run's host also runs
the run controller below, which an engine of one's own need not. A
refused or dropped connection is delivered as an engine `error`
message; the live view stops and says so in the status line, and never
falls back to another engine. Transfer lists are ignored: the codec
copies, which is the correct semantics at a socket.

The tick pump and the stimulus clock run in the page by default;
`pump=host` moves both into the host through the controller below.

### Host run controller

Messages whose top-level object carries a `__host` key are consumed by
the host's run controller and never reach the engine; everything else
passes through. The controller commands:

- `{ __host:'pump', on, steps }`: the host drives the tick loop itself,
  sending the next tick as each state reply arrives. State frames are
  forwarded to the client only while its socket drains; a frozen tab
  lapses instead of stalling the run, and forwarded state frames carry
  `__simMs`, the host's clock, which a waking client adopts.
- `{ __host:'maps', maps, seed }`: the computed input maps; the host runs
  the stimulus clock (`IORuntime` over the curriculum schedule) itself.
- `{ __host:'clock', simMs }`: sets the host's simulated clock (resume).
- `{ __host:'at', simMs, msg }`: send `msg` to the engine when the clock
  reaches `simMs`; used for the warm-up plasticity switch.
- `{ __host:'net', count, synCount, graph, curriculum }`: the identity a
  checkpoint file needs. A host refuses a run when a file in its own
  import graph is newer than its start, with an error naming the file,
  and closes the connection.
- `{ __host:'save', everyMs, dir }`: write a weights-only brain file to
  `host/runs/<dir>/brains/` every `everyMs` of simulated time, on the
  host's disk. Naming a save directory also registers the run as live,
  which is what attach below finds.
- `{ __host:'saveNow' }`: write a brain file at once (at the next state
  reply while the pump is on, immediately while it is paused).
- `{ __host:'detach', untilMs }`: the run survives its client. On
  disconnect the host keeps pumping until the clock reaches `untilMs`
  and keeps writing brain files; results land in `host/runs`.
- `{ __host:'attach', dir }`: this connection becomes a viewer of the
  live run named by its save directory (empty picks the latest). The
  connection's own engine is discarded, the host answers with
  `{ __attached:{ dir, simMs, untilMs } }`, and every subsequent state
  frame of the watched run is copied to the viewer while its socket
  drains. A viewer sends nothing that steers the run; the trainer's
  `attach=1` mode uses this, wiring the same network locally for the
  scene (determinism makes it identical) and drawing the streamed
  spikes.
- `{ __host:'assets', atlas }`: canonical stimulus rasters built in the
  browser, since only it has a canvas and a font; host-side rendering
  samples these through the pose transform (the atlas path in
  `src/curriculum.js`). The atlas is host module state: runs sharing a
  host share the last atlas set.

Host-side stimulus rendering is canvas-free: the spot set is computed
analytically and glyphs, words and drawn objects render through the
shipped atlas. A canvas-only stimulus kind with no atlas fails at the
host with an error naming it.

## 8. Short-term plasticity across engines

Tsodyks and Markram release dynamics (Tsodyks, Pawelzik and Markram
1998; the form of Zenke, Agnes and Gerstner 2015 equations 9 and 10),
`stp` with `stpU`, `stpTauD` and `stpTauF`. State is two floats per
presynaptic neuron, not per synapse: the release factor belongs to the
axon and scales every outgoing contact of a spiking neuron equally.

    dx/dt = (1 - x)/tauD - u x S(t)        x: available resources
    du/dt = (U - u)/tauF + U (1 - u) S(t)  u: release probability

Ordering, identical in all three engines. With `stpOrder` 0, the default
and the published order: at spike time the released fraction is the
probability times the resources, `rel = R*X`, resources deplete,
`X -= rel`, the probability jumps, `R += U*(1-R)`, and the neuron's
synapses deliver `w * rel * stpScale`. With `stpOrder` 1 the jump comes
before the release, so a rested synapse releases U(2 - U). After the
millisecond's spikes and plasticity, every neuron recovers by exact
exponentials toward `X = 1` and `R = U`, every simulated millisecond,
whether or not plasticity is enabled. `stpScale` is 1 with `stpNorm` 0,
the default, so a rested synapse releases U of its weight as published;
`stpNorm` 1 sets it to the inverse of the rested release (1/U, or
1/(U(2 - U)) with `stpOrder` 1) so a rested synapse delivers exactly `w`,
an adaptation not in the source. State is seeded full and at `U` the
first time the term is on, on every engine; a later re-enable resumes
from the state the term was switched off with.

Tunables: `stp` (0/1), `stpU`, `stpTauD`, `stpTauF`, `stpNorm`,
`stpOrder`. The checkpoint node's defaults are the Markram 1998
depressing class, U 0.5, tauD 800 ms, tauF 1 ms, and a computed network
always carries its values; an init that omits them gets each engine's
fallback, U 0.2, tauD 200 ms, tauF 600 ms. On the CUDA tune frame the
order shares byte 100 with `stpNorm`: the float there is
`stpNorm + 2*stpOrder`. On the WebGPU Step struct it is the u32 after
`vmin`.

On the WebGPU engine the release lives in the delivery pass, not the
update pass: the update pass is at the eight storage buffers WebGPU
guarantees, and the delivery pass is the one place where exactly one
thread owns a spiking neuron, because partitions tile the presynaptic
range so the partition filter admits a given neuron in exactly one
dispatch. Delivery routes excitatory and inhibitory contributions on the
stored weight rather than the scaled one, since the release factor can
reach zero and a zero must not be read as inhibitory. Per-millisecond
recovery is its own dispatch (`stpRecover`) after delivery and
plasticity.

Gated by two checks in the cross-engine equivalence group: parity with
the term on, and a rate that moves when the term is switched on. Parity
alone passes when every engine ignores the term.

## 8b. Committed synapses

`commit` takes a synapse out of the plastic pool once it has
consolidated. The criterion carries no new state: consolidation already
drives a per-synapse reference weight to one of two stable points, so
committed means that reference has reached the upper one, at nine tenths
of the consolidated weight of the rule that synapse belongs to. A
committed synapse keeps its weight and keeps transmitting; both halves of
the weight update and the scaling pass skip it.

Implemented in all three engines: the reference loop skips committed
synapses in both halves of the update and in `applyScaling`, the WebGPU
`plastOut`, `plastIn` and `scalePass` shaders bind the reference weight
(`wRef`) so they can skip them too, and `plastOut`, `plastIn`
and `scaleK` in `cuda/kernels.cuh` share one `isCommitted` test.

Both settings that drive it are controls on the checkpoint node:
`commit after (sim-min)` (`tauCons`, the timescale of the drive into the
well) and `commit check every (sim-s)` (`consStep`, the cadence of the
consolidation step). On the CUDA tune frame bytes 104, 108 and 112 carry
the commit switch, the timescale and the cadence (section 13).

The consolidated reference is clamped to the two fixed points of its own
well, `[0, consW]`, in all three engines. The step is forward Euler over
a cubic, so a short timescale diverges without the clamp: at a twenty
second constant with the published well depth the reference overshoots,
the cubic grows with the overshoot, and the weights reach NaN inside ten
simulated seconds.

## 9. Every engine is seeded

The init message's `seed` keys two draws. The initial membrane state is
drawn by the host (`initState` in `src/rand.js`, an xorshift stream
keyed by the seed) and sent as `v0` and `u0`; an engine given an init
without them refuses it, and no engine draws its own.

The per-step noise is the other draw, and it is the same in all three
engines: `noiseDraw` in `src/rand.js`, a PCG output hash of
(neuron, t, seed) and the sum of two uniforms minus one, with a copy in
the WGSL update shader and a copy in `cuda/kernels.cuh`, all computed in
f32. The draw for a neuron is a pure function of its arguments, so it
does not depend on how many draws came before it and cannot drift
between engines. The three copies change together. The battery covers
the noisy case separately from the constant-drive one.

## 10. Device errors are engine errors

WebGPU reports validation failures asynchronously. A bind group that
does not match its layout does not throw: every dispatch using it
silently becomes a no-op while the engine keeps answering, reporting
state for a simulation that never ran, which is indistinguishable from a
very fast engine. The worker registers an `uncapturederror` listener and
raises device errors as engine `error` messages naming the failure. A
WebGPU change that is dramatically faster is checked for still
computing.

## 11. The fast plasticity mechanism set

The tune message carries four fields beyond the pair rule, all defaulting
to zero, where zero reproduces the pair rule byte for byte (MODEL.md
section 4 has the equations):

- `trip`: triplet LTP amplitude (Pfister and Gerstner 2006). LTP at a
  postsynaptic spike becomes (aP + trip*Kslow)*Kpre, with Kslow a slow
  postsynaptic trace read before the current spike. This is the
  potentiation-grows-with-rate behavior fit to Sjostrom, Turrigiano
  and Nelson 2001 and Wang et al. 2005, which pair STDP cannot produce.
- `tauY`: Kslow's time constant in ms, default 114, the Pfister and
  Gerstner visual cortex fit.
- `het`: heterosynaptic regression. At a postsynaptic spike every
  incoming plastic excitatory weight is pulled toward its reference by
  the fraction min(1, het*(Kslow/kRef)^3), kRef being Kslow's steady
  state at the homeostatic target rate, so competition engages when the
  neuron fires above its set point and the pull saturates at a full
  return to baseline (Lynch et al. 1977; Chistiakova et al. 2014;
  implementation form of Zenke, Agnes and Gerstner 2015). The reference
  is the weight vector at the moment het or consolidation first switches
  on; after a resume that is the resumed state. Section 12 says how
  consolidation moves it.
- `tin`: transmitter-induced potentiation added per presynaptic spike,
  the small non-Hebbian term that keeps low-rate synapses from silent
  collapse; the least measured of the set, and stated as such in the
  source.

Implemented in all three engines. The transmitter drip is a constant
added per presynaptic spike in the outgoing pass, which makes the wmax
clamp there load bearing. The triplet term needs one more per-neuron
trace, decayed with tauY and bumped alongside the fast traces, and read
in the incoming pass before that bump so potentiation is scaled by the
postsynaptic history rather than by the present spike.

WebGPU layout: Kpre and the slow trace are interleaved into one buffer,
two floats per neuron, which keeps the incoming pass within the eight
storage buffers WebGPU guarantees once the reference weight is bound;
Kpost stays a plain array because it is the hot read in the outgoing
pass. The reference weight vector is allocated when het or consolidation
switches on, not always, since it is a second copy of every weight. The
heterosynaptic pull is computed in the same pass as potentiation, from
the pre-update weight, with one clamp on the combined result, as the
reference engine does. The refractory period is an extra u32 storage
buffer at binding 8 of the update pass and a `refrac` field in the Step
uniform.

The battery gates the set with the fast plasticity mechanism group:
pairing-frequency dependence appears only with trip on, the transmitter
drip is exact, and the heterosynaptic pull returns weights toward
reference.

## 12. Consolidation of the reference weight

The heterosynaptic term regresses each plastic excitatory weight toward
a reference. With `cons` off that reference is captured once, when the
term switches on, and never updated. With `cons` on it follows the
weight through the double well of Zenke, Agnes and Gerstner 2015 (Nat
Commun 6:6922) equation 16:

    tauCons dwRef/dt = w - wRef - consP wRef (consW/2 - wRef) (consW - wRef)

The lower fixed point is zero, the upper is consW, and the midpoint at
consW/2 is unstable, so a synapse driven above the midpoint consolidates
to the upper state and one left below it decays to zero. `tauCons`
defaults to 20 simulated minutes. Every engine updates the reference
every `consStep` (default 1.2 simulated seconds) rather than every step,
as the published implementation does.

Applies to excitatory plastic synapses only; inhibitory weights follow
the Vogels rule and have no reference. `cons` is a gate: any value above
zero turns consolidation on in every engine and the rate is
`consStep/tauCons`.

## 13. The CUDA engine child

`host/host.mjs --cuda` runs each connection on `cuda/engine.exe` instead
of the reference worker. The child is pure compute: it speaks fixed
binary structs over stdio (framed as u32 cmd, u32 bytes, payload) and
owns no policy. `host/cudaengine.mjs` translates the messages of section
2: neuron types expand to per-neuron parameters, constant protocols fold
into a static drive, noise protocols into per-neuron amplitudes, and
input-map gains rebuild the external drive exactly as the reference's
`rebuildExt` does. A pulse or ramp protocol fails at the adapter. The
kernels (`cuda/kernels.cuh`) mirror the reference's step and plasticity
arithmetic; the plasticity decomposition is the WebGPU engine's
outgoing/incoming/trace sequence. Scaling (`scaleK`) reuses the reverse
index the child builds for `plastIn`, with a per-neuron spike
accumulator incremented in `traceStep`; measured set points are
`finishCalK`.

Commands and replies:

    1 INIT   (frame below)          -
    2 TUNE   (frame below)          -
    3 EXT    f32[n]                 -
    4 NAMP   f32[n]                 -
    5 TICK   u32 steps              100 STATE
    6 GETW                          101 WEIGHTS: f32 w[m], then f32 rho[n]
                                        when set points are measured
    7 QUERY  u32 idx, cap, outOnly  102 QUERYRESULT: u32 idx, outTotal,
                                        inTotal, outN, innN,
                                        u32 out[outN], u32 inn[innN]
    8 WATCH  i32 idx                (folded into STATE)
    9 SENDV  u32 on
    10 SENDT u32 on
    11 HELLO                        103 HELLO: u32 the contract number the
                                        child was built for

INIT frame: a 24-byte head (u32 n, m, syn; f32 tauE, tauI; u32 psc), then
f32 a, b, c, d, bias [n], i32 preStart [n+1], i32 post [m], f32 w [m], u8
delay [m], u8 pmask [m], f32 v0 [n], f32 u0 [n], u32 seed, u32 poolCount,
u16 pool [n], f32 poolK [poolCount] (a network without pools sends
poolCount 1 and zeros), f32 C, k, vr, vt, vpeak [n], u32 graded and,
when graded is 1, u8 grd [n], f32 thr [n], f32 slope [n]. The child
computes the expected length from n, m and poolCount and exits with a
message if the frame differs, so a host and a child built either side of
a protocol change fail at init rather than run on invented state.

TUNE frame, 184 bytes without a rule table (`_plastStruct` in
`host/cudaengine.mjs`):

| byte | field |
|---|---|
| 0, 4 | u32 plast, u32 wdep |
| 8 to 28 | f32 aP, aM, tauS, wmax, iEta, iRho |
| 32 to 44 | f32 trip, tauY, het, tin |
| 48 | u32 refrac |
| 52 to 60 | f32 cons, consW, consP |
| 64 to 76 | f32 stp, stpU, stpTauD, stpTauF |
| 80, 84 | f32 scale, sEta |
| 88, 92 | f32 rhoMode, calS |
| 96 | f32 tauM |
| 100 | f32 stpNorm + 2 stpOrder |
| 104 to 112 | f32 commit, tauCons, consStep |
| 116 | f32 vmin |
| 120 | u32 receptor channel count, then six f32 channel taus from 124 |
| 148, 152 | f32 eRevE, eRevI, then six f32 channel reversals from 156 |
| 184 | optional: u32 rowCount, then rows of 11 floats from 188 |

The child reads the rule table behind a length guard and builds row 1
from the scalars when none is sent.

STATE: u32 steps, u32 hasV, u64 spikes, u8 fired[n], then f32
vtrace[steps] if hasV (the watched neuron's membrane trace, written by a
one-thread kernel per step and copied back once per tick), then f32 v[n]
while SENDV is on and u8 lat[n] while SENDT is on, told apart by the
frame length.

QUERY: outgoing connections are a slice of the forward CSR; incoming ones
come from the reverse index the plasticity passes build, so the answer is
two small device-to-host copies.

`host/cudaengine.mjs` answers `hello` by sending HELLO and putting the
child's own number in the reply, so a child built for another contract
is refused by the hello check; a child that does not answer within five
seconds is reported as an older build.

`cuda/engine.exe` must be rebuilt after any change to `cuda/engine.cu`,
`cuda/kernels.cuh` or the INIT or TUNE layout (`nvcc -O3 -arch=sm_89 -o
cuda/engine.exe cuda/engine.cu`), and `cuda/parity.mjs` run against a
fresh `cuda/dumpnet.mjs` network (EXTENDING.md).

## 14. Feedback inhibition

A lumped inhibitory pool per population: every spike a pool's cells fire
in millisecond t delivers `poolK[g]` to every cell of the pool in
millisecond t + 1, on the inhibitory path (the input current in kick
mode, the I conductance in exp mode, so `tauI` and `psc` apply). The
wiring sets `poolK[g] = gain * 1000 / N_g`, so a cell receives gain times
its pool's mean rate in Hz per millisecond at any pool size. The circuit
is the APL neuron of the fly mushroom body (Lin et al. 2014) and the
basket cell recurrent inhibition of the hippocampus (Andersen, Eccles
and Loyning 1963); MODEL.md section 2 has the equation.

State is two per-pool count tables. A step reads the table the previous
step wrote and counts into the other, and the table it read is zeroed
before the next step. The reference engine swaps two arrays at the end
of the step; the WebGPU update shader and the CUDA kernels index the two
halves by the parity of t and zero the read half afterwards (a
`poolClear` dispatch after update; a `cudaMemsetAsync` after the step
kernel). Pools are init-time structure, carried by the init message and
the INIT frame, not by tune.

WebGPU: the update pass is at the eight storage buffers WebGPU
guarantees, so the pool rides in the refractory buffer, declared atomic:
the pool per cell in the top 24 bits of each cell's refractory word, then
the two count halves, then each pool's kick as float bits. CUDA: three
device arrays and five more kernel arguments; the INIT frame carries a
u32 count, u16 per cell and f32 per pool, and the child's length check
reads the count from the frame.

The battery's feedback inhibition group checks the reference and the
WebGPU engine against each other and checks that a cell outside every
pool computes the same bytes as it does with no pool in the network;
`cuda/parity.mjs` covers the CUDA child.

## 15. Writing an engine

An engine written in any language can run the networks the node graph
builds. It is a WebSocket server speaking the messages of section 2. The
page connects to it when a checkpoint's engine is `remote`, at the
checkpoint's engine address; `tools/linen.mjs run <scene> --engine remote
--host ws://host:port` connects to it from a shell. One connection is
one engine: the network arrives on the connection, and closing it ends
the engine. `tools/engine_template.py` is a complete engine to start from:
a leaky integrate-and-fire network in NumPy behind the `websockets`
package, about 200 lines.

### The framing

One message is one binary WebSocket frame:

    u32 headerLen (little endian) | JSON header | payloads

The header is the message as JSON with every typed array replaced by a
descriptor, `{ "__ta": { "d": "Float32Array", "n": 1000 } }`. `d` is one of
`Float32Array`, `Float64Array`, `Int8Array`, `Int16Array`, `Int32Array`,
`Uint8Array`, `Uint8ClampedArray`, `Uint16Array`, `Uint32Array`; `n` is the
element count. The payloads follow the header in the order their
descriptors appear in a depth-first walk of the header (object keys in
the order the JSON lists them, array elements by index). The first
payload starts at the first multiple of 8 at or after `4 + headerLen`,
and each following payload at the next multiple of 8 after the one
before. Every number in a payload is little endian. An engine's replies
use the same framing. `src/frame.js` is the codec the page uses, and
`encode` and `decode` in the template are the same in Python.

### The exchange

1. `{ cmd:'hello' }` arrives first. The reply is `{ cmd:'hello', protocol,
   engine, terms, optional }` (section 2): `protocol` is the contract
   number this engine implements (`PROTOCOL` in `src/protocol.js`, 3),
   `engine` a short name the status line shows, `terms` the terms beyond
   the baseline it implements, `optional` the optional queries it
   answers. The page refuses an engine of another contract, and sends
   the init only when the engine names every term the network uses.
2. `init` carries the network (section 1) and every tunable. The fields
   an engine needs for the baseline are `protocol` (refuse any other
   number with an `error` reply), `count`, `ntype` and `types` (each
   type's `f7` row: `C`, `k`, `vr`, `vt`, `vpeak`, `a`, `b`, `c`, `d`, in the
   units of section 3), `bias`, `preStart`, `post`, `w`, `delay` (whole
   milliseconds, 1 to 16), `seed`, `v0` and `u0`. An engine with another
   membrane reads the row fields it needs; the template takes `C`, `k`,
   `vr` and `vt`.
3. `tune` carries the tunables again; a tune without `bias` keeps the
   bias in force. It arrives whenever a setting changes on a node and
   never resets the state.
4. `{ cmd:'tick', steps }` advances `steps` milliseconds (1 to 20 from the
   page). The reply is `{ cmd:'state', fired, spikes, steps }`: `fired` a
   `Uint8Array` of `count` holding each cell's spike count over the tick,
   capped at 255; `spikes` the total; `steps` as asked. The page sends
   the next tick when the reply arrives.
5. An engine that cannot do what a message asks replies
   `{ cmd:'error', message }`, which the page shows and stops on. A
   message the engine does not handle is answered with
   `{ cmd:'unsupported', of:cmd }` or ignored. Messages whose top level
   carries `__host` belong to the run controller of `host/host.mjs`
   (section 7); they are sent only during a training run, which runs on
   `host/host.mjs`.

That is the minimum for the viewer: hello, init, tune of the bias, tick
and the state reply. Everything else is optional and named in the hello.

### What is optional

The terms (section 2, `TERMS` in `src/protocol.js`): every mechanism
beyond kick delivery, from exponential synapses (`syn:1`) and the
stimulus modes (`protocols:0` to `protocols:3`) to plasticity (`plast` and
the terms that ride on it). A network that uses a term the engine does
not name is not sent, the status line says which term, and the
checkpoint panel grays the settings that belong to it. The template
names `syn:1`, `psc:1`, `psc:2`, `refrac`, `vmin`, the four stimulus modes
and `custom`; the Tab menu scenes it cannot run are refused with the
term named.

The optional queries (`OPTIONAL`): `watch` (`{ cmd:'watch', idx }`, then a
`vtrace` `Float32Array` of one value per step on every state reply, the
viewer's scope), `sendV` (`{ cmd:'sendV', on }`, then `v`, every cell's
membrane potential, on every state reply, the potential filter and
graded probes), `sendT` (first-spike latency, `lat`), `query` (connection
lookup for a clicked cell, `queryResult`) and `getWeights` (`weights`, for
brain files, the weight plots and the recording's summaries). The page
sends none the engine does not name, and the status line says what is
unavailable when one is asked for.

### Settings the tune message does not have

The engine settings node (between connect and the checkpoint) puts a
block of keys and values on every init and tune as `custom` (section 2).
An engine reads what it understands there and names `custom` in its
hello; the template reads `lif_tau_ms`. An engine that does not name it
gets the block anyway, and the status line says how many keys it
ignores.

### The conformance check

    node tools/enginecheck.mjs ws://localhost:8890 [--seconds S]

sends hello and checks the contract number, the reply's fields and its
names; builds a fixed small scene through the scenario API (a balanced
network of 1,000 cells with kick synapses and noise drive, every node
named, every setting on its node); runs it on the candidate and on the
reference engine from the same init message, one millisecond a tick;
checks the shape of every state reply and of every optional query the
candidate names; and prints each population's rate, ISI CV and
population synchrony side by side, each passing or failing on the band
of the battery's cross-engine equivalence group (`src/equivalence.js`:
rate within 20%, ISI CV within 0.15, synchrony within a factor of 2).
Against `host/host.mjs` every check passes. Against the template the
protocol checks pass and the measures do not, because a leaky
integrate-and-fire membrane is a different model from the Izhikevich
membrane; the output says so. `tools/remotetest.mjs`, in the full test
suite, runs both.

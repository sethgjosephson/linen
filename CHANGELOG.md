# Changelog

One line per user-visible change, per release, newest first. The list
starts at the public release.

A release tag names the app. The files the app writes carry their own
numbers, checked on read and migrated where a migration exists, and a
release lists the number of each that it reads and writes:

| file | number | where it is checked |
|---|---|---|
| scene (`scenes/*.json`) | format 15 | `src/migrate.js`, every step from an older format runs on load |
| project (`project.json`) | format 1 | `src/project.js` |
| brain (`brains/*.npb`) | NPB1, NPB2, NPB3, version 1 | `src/brain.js` |
| scenario layout (`src/layouts/*.json`) | format 3 | `src/layoutfile.js` |
| recording (`recordings/*.json`) | linen-recording-1 | `src/recording.js` |
| trial matrix (`runs/*/trials.npt`) | np-trials-1 | `src/train.js`, `tools/np_trials.py` |
| headless run folder (`run.json`, `rates.csv`, `spikes.txt`, `cells.csv`, `settings.json`) | as written by `tools/linen.mjs` | `tools/linen_read.py` |
| engine init message | protocol 17 | `src/protocol.js`, refused on mismatch by every engine |

## 0.3.0 (2026-11-10)

- Neuromodulation, engine contract 17 (MODEL.md 4). A modulator node
  (before connect, like the pool) raises a level at every cell of a
  population, by a time course the engines hold, pulse or ramp, or while
  a presentation is on, or by the spikes of a releasing population. The
  checkpoint's modulation settings say what the level does: the Hebbian
  terms scale with base plus level (the three-factor rule as a field), or
  accumulate on an eligibility trace per synapse that every sweep pays
  into the weight by base plus level (Izhikevich 2007), the synaptic
  current scales with the level, and a current per level is added. All
  three engines; rebuild `cuda/engine.exe`. The learning gate and the
  rest of the rule are unchanged with no modulator in the scene. The
  every node scene has one: its interneurons release onto its pyramids
  and the rule learns faster while they are busy.
- `validation/modulator-0.3.0.md` measures the modulator on the
  released tree: the field under each kind of source against its
  equation (`tools/modsteps.mjs`), one pairing paid by a pulse at a
  delay under the gain and eligibility forms, the three engines and
  the Brian 2 transcription on the same cases, what the level does on
  the every node scene, and the cost on each engine on the balanced
  net with a modulator (`validation/modulated-net.json`).
- The settings panel names each dropdown choice in a word or two and
  each setting in a few; what a choice means is its tooltip. Saved
  scenes are unaffected, since a scene stores the choice's position.
- The mouse cortical column, the thalamocortical loop, CA3 pattern
  completion and the visual pathway run on the equations' membrane step
  (below). The thalamocortical loop's relay hold is -800 pA (was -500)
  and its cortical drive 86 (was 80); CA3's background noise is 210 (was
  240); the other two were not retuned. A copy of any of these saved
  before this release keeps the half steps and its old drives.
- The checkpoint names the brain a run starts from ("start from
  brain"), so an evaluation run on a trained brain is a scene setting;
  the training dialog takes it from the node.
- The learning gate has a fourth mode: on during presentations other
  than the single-sense probes.
- The curriculum records the first part of every gap as a fourth trial
  condition, and the readout reports delay activity per item.
- The WebGPU engine's population count lands on the state once it
  exists; before, a probe read before the first frame could see none.
- `validation/` holds one record per release: the output of the
  published-results checks, the CUDA parity check and the statistics
  battery, on the build that was released.
- `validation/membrane-step-0.3.0.md` measures the exact step on the
  released tree: one cell against the equations, Izhikevich 2003's
  cell classes, the four switched scenes against their bands, and
  the cost on each engine. `tools/published.mjs` runs the cell
  classes under both steps (60 checks, was 50).
- The checkpoint's `auto` engine choice counts the exact step's cost:
  each cell weighs 4,000 synapses in the breakeven when the exact step
  is on, so the column, the visual pathway and the thalamocortical loop
  run on the WebGPU engine at the default 2 steps per frame where a
  machine has one. CA3 stays on the reference engine until 4 steps.
- The statistics battery's groups that validate the maintainers'
  own measures on synthetic data are in a local file the public set
  does not carry; the groups that hold shipped mechanisms to the
  literature and to the reference engine remain.
- Opening a node's properties while a pane fills the window brings the
  properties panel back beside it, where it used to stay hidden.
- The CUDA engine's source no longer depends on Windows; on Linux it is
  built with the same command and the same file name.
- A curriculum block can carry its own presentation and gap (`on=`, `off=` in ms), item set (`set=`) and intensity (`gain=`), so a protocol whose phases differ in timing or stimuli runs as one network (Zenke et al. 2015: learn, test, cue). A set may label its items (the cues report their shape).
- Engine contract 16. The checkpoint's membrane step takes a third
  setting, the equations: a quadratic row is solved exactly over each
  half step, with the spike at the crossing, so a cell fires at the rate
  Izhikevich's equations give rather than his code's half steps (within
  1 percent of a 1 microsecond clock, where the half steps are 22
  percent off on average). Linear rows run as under the exponential
  step. Off by default, except in the four scenes named above. Rebuild
  `cuda/engine.exe`.
- Engine contract 15. A brain file keeps the consolidation reference,
  and a run resumed from it, or a brain loaded on the page, starts from
  it. Before, every reference started again from `refInit`, so a
  resumed run forgot which synapses had consolidated. Files without it
  load as before. Rebuild `cuda/engine.exe`.
- Engine contract 14. The checkpoint's membrane step (`integ`): two
  exponential half steps, exact for a cell whose conductances hold over
  the step, and the reset at the crossing inside the millisecond, on rows
  linear over the spiking range; quadratic rows keep the Euler half
  steps. Off by default. Rebuild `cuda/engine.exe`.
- Engine contract 13. A consolidating synapse's reference is held in
  [0, max(reference, weight, consW)] rather than at consW, so a strong
  synapse's reference settles above consW, where the weight's pull
  balances the well's, as in Zenke 2015's code. Rebuild
  `cuda/engine.exe`.
- Engine contract 12. Cell rows carry three features of Zenke, Agnes and
  Gerstner 2015's cells: a dynamic threshold (`thrJump`, `thrTau`: the
  threshold jumps at a spike and relaxes), adaptation as a conductance
  (`adapt` on the cell type node), and a Poisson output (the cell fires at
  its input current read in hertz). The checkpoint's inhibitory rule has a
  global factor (`iGlobal`, `iTauG`): one amplitude from the population
  rate in place of the per-cell target, their `GlobalPFConnection`. All
  three engines and Brian; rebuild `cuda/engine.exe`.
- The curriculum node has a `timing` control: fixed durations, or the random ones of Zenke, Agnes and Gerstner 2015 (a normal presentation, an exponential gap). The connect node's weight distribution has an `equal` option.
- The sliding threshold of Zenke, Agnes and Gerstner 2015 (their
  `P11Connection`) is a checkpoint setting: each cell keeps a slow
  estimate of its own rate, and the depression of its incoming
  excitatory synapses is scaled by the square of that estimate over a
  target rate. Off by default. Engine contract 11 (`slide`, `slideRate`,
  `slideTau`, `slideInit` on init and tune); rebuild `cuda/engine.exe`.
- The heterosynaptic term is the published one (Zenke, Agnes and
  Gerstner 2015): beta times the cube of the cell's fast postsynaptic
  trace, no normalization by a target rate and no ceiling on the step.
  Existing scenes with `het` above zero compete differently; the term's
  scale is the paper's (beta 0.05 at weights of order one).
- Where a consolidating synapse's reference starts is a checkpoint
  setting ("reference starts at"): the wired weight brought inside the
  well, zero (the lower fixed point, as the paper starts it), or the
  consolidated weight. Before, it started at the weight itself and the
  stepped equation threw a strong synapse's reference to zero. Engine
  contract 10 (`refInit` on init and tune); rebuild `cuda/engine.exe`.
- Soft depression can scale with a power of the weight ("depression
  weight exponent" on the checkpoint, 1 proportional, 0 additive, the
  Gutig et al. 2003 interpolation between). Engine contract 9 (`wmu`
  on init and tune); rebuild `cuda/engine.exe`.
- A third factor on learning: the checkpoint's "learning gate"
  setting holds the pair rule's terms at zero between a training run's
  presentations (or outside presentations of both senses), the way a
  neuromodulatory signal gates what is learned; homeostasis runs
  throughout. Engine contract 8 (`plastGain` on init and tune); rebuild
  `cuda/engine.exe`.
- A project node can name a receptor channel ("receptor channel" on the
  node): its synapses use that channel beside whatever the local wiring
  from the same cells uses, so a slow, magnesium-gated pathway can run
  beside a fast one. A channel line with no transmitter and no
  population is for the pathways that name it.
- A scene saved before the scatter's fill setting existed keeps the cell
  counts it was authored with (scene format 15). Such a scene loaded at
  the density default instead, which could multiply its cells and
  refuse to wire; the compatibility corpus is re-recorded at the
  authored counts.

## 0.2.0 (2026-10-21)

Engine contract 7: an engine built against an earlier contract is
refused at hello, so rebuild `cuda/engine.exe`. Scenes are format 14;
older scenes migrate on load.

Plasticity and synapses
- The pair rule can be timed at the synapse (the checkpoint's "stdp
  timed at"): a presynaptic spike counts when it arrives, after its
  delay, so a pairing is read from the arrival (Izhikevich 2006). It is
  the default for new scenes; scenes from before declare the soma
  reading they were tuned on.
- Short-term plasticity per rule: a plasticity node can give its synapses
  their own release settings or turn them off, so one axon can depress
  onto one target and facilitate onto another.
- Short-term plasticity can be confined to excitatory or to inhibitory
  synapses (the checkpoint's "stp scope").
- A receptor channel can carry the NMDA magnesium block (`mg:1` on its
  line in the receptors node), and can take the synapses from a
  population (`from:<tag>`), so connect-wired synapses can use it.
- Under measured set points the heterosynaptic gate uses each cell's own
  target, as the other homeostatic terms already did.
- A noise stimulus can have a time constant (an Ornstein-Uhlenbeck drive
  instead of a fresh draw every millisecond) and a shared fraction that
  every cell of the stimulus receives alike.

Scenes
- The mouse cortical column splits each layer's interneurons into PV
  (tag `Lxi`) and SST (tag `Lxs`) populations: excitatory synapses onto
  PV cells depress, onto SST cells facilitate, and SST inhibition is
  slower.

App and engines
- The CUDA engine delivers spikes and applies both learning passes over
  the step's spike list, a warp per spike; with learning it runs 1.5 to
  8x faster on the measured networks.
- The CUDA engine runs without a console window of its own.
- A scene link with `?train=1` starts a training run with the settings
  on the scene's checkpoint.
- Space fills the window with the pane under the cursor, the viewer or
  the node graph, and restores it.
- VIEWER WINDOW in the viewer bar moves the whole viewer, its bar, the
  oscilloscope, the panels and the raster, into a window of its own, to
  put on another screen; a VIEWER button in the top bar brings it back.
- The desktop app, from source: an Electron window with the server and
  the engine host inside it (`desktop/`, see the README). It takes
  `--root DIR` to run on another checkout's files and opens the
  developer tools on F12.
- Node modules for every project: the desktop app's startup folder
  (Folders menu, `init.js` by convention) and any folder added from
  that menu, or `--startup DIR` on either server, load before the
  project's own; a project's module of the same key wins.

## 0.1.0 (2026-10-10)

The first public release: the node graph, the three engines on one
contract, the scenes, the headless runner and the validation battery.
It read and wrote scene format 13 and engine protocol 3; the other file
numbers are as in the table above.

# Changelog

One line per user-visible change, per release, newest first. The list
starts at the public release.

A release tag names the app. The files the app writes carry their own
numbers, checked on read and migrated where a migration exists, and a
release lists the number of each that it reads and writes:

| file | number | where it is checked |
|---|---|---|
| scene (`scenes/*.json`) | format 14 | `src/migrate.js`, every step from an older format runs on load |
| project (`project.json`) | format 1 | `src/project.js` |
| brain (`brains/*.npb`) | NPB1, NPB2, NPB3, version 1 | `src/brain.js` |
| scenario layout (`src/layouts/*.json`) | format 3 | `src/layoutfile.js` |
| recording (`recordings/*.json`) | linen-recording-1 | `src/recording.js` |
| trial matrix (`runs/*/trials.npt`) | np-trials-1 | `src/train.js`, `tools/np_trials.py` |
| headless run folder (`run.json`, `rates.csv`, `spikes.txt`, `cells.csv`, `settings.json`) | as written by `tools/linen.mjs` | `tools/linen_read.py` |
| engine init message | protocol 7 | `src/protocol.js`, refused on mismatch by every engine |

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
  synapses (the checkpoint's "stp applies to").
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

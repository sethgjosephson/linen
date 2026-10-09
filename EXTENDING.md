# Extending linen

How to add a node, a mechanism or a measure without breaking the promises
the rest of the project depends on. The module map is in README.md
(Architecture); the contribution terms are in CONTRIBUTING.md; the engine
contract is ENGINE.md; every equation is in MODEL.md with its source and
the function that implements it. This document says what is frozen, what
is open, and the order of work for each kind of change.

## The four promises

Every change is held to these by the gates below.

1. **Determinism.** The same graph and seeds compute the same network, and
   the same network and seed simulate the same way on every engine up to
   summation order. Every seeded draw lives in `src/rand.js`; nothing in
   a compute function, an engine or the stimulus runtime reaches for `Math.random`.
2. **The engines agree.** A mechanism exists in all three engines or in
   none: `src/simworker.js` (the CPU reference, which defines correct
   behavior), `src/gpuworker.js` (WebGPU) and `cuda/engine.cu` behind
   `host/host.mjs` (the native child). No hidden constants, no CPU-only
   terms. An engine that lacks a term refuses the init rather than
   running without it.
3. **Fail loudly.** Wiring that clamps says so in the status line; an
   init without the fields it needs is refused; a node that cannot compute
   names itself and pulses red. Silence is the failure mode, not the
   error.
4. **Everything is on a node.** Every setting an experiment varies sits
   on a node in the scene and the run reads it from there. A launcher may
   press START; it never holds the settings.

## What is frozen

Change any of these and every consumer of the file format or the protocol
breaks. Each has a version or a migration path; use it.

- **The message protocol** between the page and an engine (ENGINE.md
  section 2): `hello`, `init`, `tune`, `tick`, `state`, `inputFrame`, `getWeights`
  and the rest, and every field they carry. Every engine-facing field of
  an init or tune comes from `engineConfig(net)` in `src/nodes.js`; never
  hand-list them at a call site. The contract carries a number,
  `PROTOCOL` in `src/protocol.js`, on every init; each engine refuses any
  other number with a message naming both, and the engine worker URLs
  carry the same number. A change to any message or frame bumps it in
  the same commit as the three engines, ENGINE.md and the battery. The
  CUDA child also checks the INIT frame length and exits on a mismatch.
- **The brain file** (`.npb`, `src/brain.js`): header, cell-order hash,
  weights, the per-synapse plasticity byte. A loader refuses a file from
  another tissue (the cell-order hash) rather than restoring by index
  onto the wrong cells.
- **The plasticity byte** per synapse (ENGINE.md section 1): the rule id
  in the low five bits (0 frozen, 1 the checkpoint's rule, 2 and up
  declared rules, at most 14, since the rule table holds 16 rows), the
  receptor channel in the top three. Every reader masks `& 31`.
- **The rule table**: 11 fields per rule in this order, identical in all
  three engines: `aP, aM, wmax, wdep, trip, het, tin, iEta, cons, consW,
  consP`.
- **The type table row** (`f7` on `NEURON_TYPES`): the Izhikevich 2007
  membrane, `C, k, vr, vt, vpeak, a, b, c, d`, and `graded: { thr,
  slope }` on a row that releases instead of spiking. Rows 0 to 6 are the
  2003 presets carried as 2007 rows and are what a scene means by RS, FS
  and the rest; do not renumber them. A scene's own rows join the table
  by name through the cell type node.
- **The scene format** (`src/migrate.js`): a scene file carries `format`,
  and every step from an old format to the current one runs on load.
  Never edit an existing step; append one and bump `SCENE_FORMAT`, with a
  case in `src/migrate.test.mjs`. A node's type key is part of the
  format: renaming one needs a migration.
- **Units.** Micrometers for every distance, milliseconds for time,
  millivolts for potentials. A current is in the unit of the row it lands
  on: picoamps on a measured row, the 2003 model's unit on a classic row.
  Conventions that could be rescales are parameters instead (`psc`,
  `tauS`/`tauM`, `stpNorm`, `stpOrder`), never applied silently.

## Adding a node

A node is a definition in `NODE_DEFS` (`src/nodes.js`) and a pure compute
function.

1. **The definition**: `title`, `cat`, `color`, `inputs` (and `side`
   for mask and signal ports), and `params`, each `{ k, label, t, def }`
   with `min`/`max`, `s:[lo, hi]` for a slider, `g:'group'` for a tab,
   `cond(params, node)` to hide one setting behind another. The panel
   renders from this; there is no separate UI code.
2. **The compute**: `compute(ins, p, node)` takes the computed upstream streams
   and the parameters and returns a stream. It is pure: no DOM, no
   randomness outside `rand.js`, no reading of anything below the node.
   Use `need(ins[0], 'points', 'what needs what')` and throw with a
   message that names the fix; the graph pulses the node and the status
   line shows the text.
3. **What reaches the wiring is in the hash.** The connect node's memo key
   is `streamHash` of what arrives at it. A field the node adds to the
   stream that changes the tissue must be in the hash, or an edit will be
   a silent memo hit; a display-only field (a hue) must stay out, or a
   color change will rewire. The stream copy the connect node's compute
   hands to the wiring lists its fields; add new ones there.
4. **Prose for every parameter** in `src/docs.js`, and a `blurb` for the
   node. `src/docs.test.mjs` and the documentation integrity group in
   the battery gate this both ways: a parameter without prose and prose
   without a parameter both fail.
5. **A test** in `src/<name>.test.mjs` importing `ok` and `report` from
   `tools/harness.mjs`; the runner discovers it the moment it exists.
6. **The settings audit**: `?scenario=<name>&audit=settings` moves every
   setting upstream of the checkpoint and reports which do not change the
   computed tissue (`src/audit-settings.js`, report in `runs/`). Run it
   after adding a parameter.

## A node as a plugin

A node can live outside the repository, in a project's `nodes` folder. It is
the same definition and the same compute function as a node in `NODE_DEFS`, held to the
same rules; what differs is where it is registered. The section above is for
a node that goes into the repository itself.

1. **The module**: an ES module that imports nothing. Its default export is
   a function that receives the plugin interface and calls
   `registerNode(key, definition)` once per node. A module that exports no
   function, registers no node, or throws is not loaded, and a module that
   fails part way leaves none of its nodes registered (`loadPlugin` in
   `src/plugins.js`).

   ```js
   export default function(linen){
     linen.registerNode('jitter', { title, cat, color, inputs, params, compute, doc });
   }
   ```

2. **What `registerNode` takes** (`src/nodes.js`): the shape of a
   `NODE_DEFS` entry (`title`, `cat`, `color`, `inputs`, `side`, `params`,
   `compute`, optionally `memo`) and `doc`, the prose `src/docs.js` holds for a
   built-in node: `{ io, blurb, params: { key: text } }`. `cat` is one of the
   ten categories (`NODE_CATS`) and every setting's `t` one of the kinds the
   panel draws (`PARAM_TYPES`). The documentation is held to the rule
   `src/docs.test.mjs` holds the built-in nodes to (`nodeDocProblems`): a
   blurb, prose for every setting and for nothing else. A key a built-in
   node has is refused, and so is a key another module registered, with both
   modules named; the same module registering again replaces its own node,
   which is how an edited module reloads. The node carries `plugin: {
   module, hash }`, the module's file name and a hash of its source.
3. **The interface**: `registerNode`, the stream helpers `need` and
   `clonePts`, and from `src/rand.js` `h32`, `pairHash`, `pairGauss` and
   `rng`.
4. **Purity**: the rules of a built-in compute function. No DOM, no randomness except
   the draws handed in (a module whose text names `Math.random` is refused),
   nothing read from below the node. A draw keyed on a cell's identity
   (`src`, `lidx`) is stable when the graph around the cell changes. A field
   a plugin adds to a stream reaches the nodes below it that read it; the
   built-in nodes copy the fields they know, and the wiring and the engines
   read only theirs (the stream copy in `connectCompute`, `engineConfig`).
5. **The memo and the wiring key**: a plugin node that sets `memo` is keyed
   on its module and the hash of the module's source as well as its params
   and inputs (`memoKey`), so an edited module never reuses what its
   earlier version computed. Below a plugin node the connect node's key is
   `streamHash` of the stream it receives, so an edit that changes the
   node's output changes the wiring key and one that does not leaves the wiring
   cached.
6. **How a project loads it**: every `.js` in `<project>/nodes` is loaded
   when the project opens, in name order, before any scene
   (`loadProjectPlugins`); `project.json` records nothing about them. The
   project menu lists each module with the nodes it registered or why it
   was not loaded, reloads them, and adds a module from a URL by fetching
   it, loading it and writing it into `nodes/` under its own name
   (`addPluginFromUrl`), which is how the hosted site takes one. Computation runs
   on the page's thread; the wiring workers run `wireConnect` on a finished
   stream and never read `NODE_DEFS`, so no worker imports a module.
   `tools/linen.mjs` loads the `nodes` folder beside a scene file's
   `scenes` folder and any folder given with `--nodes`.
7. **How a scene records it**: a scene file lists `plugins: [{ module,
   nodes: [key, ...] }]` (scene format 12, `scenePlugins`). Loading a scene
   whose modules are not registered throws before the editor changes,
   naming each module and its keys (`checkScenePlugins`); a node type no
   recorded module accounts for is refused the same way. The module name is
   the file name, so a module copied into `nodes/` and the same module added
   from a URL are one module.
8. **A test**: `src/plugins.test.mjs` loads `examples/nodes/jitter.js` from
   its text, computes a graph through it, and checks the scene round trip and
   the refusal; a custom module is tested the same way, with `importer`
   and `loadPlugin`.

A module runs in the page with the page's own access. It is the user's own
code, or code the user chose to add, and nothing sandboxes it.

## Adding a mechanism

A term in the dynamics is the most expensive kind of change, and it is
one commit or none.

1. **The reference first**: `src/simworker.js`, with the equation and its
   source in a comment. Then `src/gpuworker.js` (the WGSL, the Step
   struct, the per-neuron buffers: each pass is held to the eight storage
   buffers WebGPU guarantees, and the incoming plasticity pass is at that
   limit, so a new per-synapse array there needs a slot freed first),
   then `cuda/kernels.cuh` and `cuda/engine.cu` with the frame in
   `host/cudaengine.mjs` (ENGINE.md section 13).
2. **The field** goes into `engineConfig(net)` and nowhere else; the
   node that owns it gets its parameter and prose as above.
3. **ENGINE.md** gets the term: what it does, its frame layout, its
   refusal conditions. **MODEL.md** gets the formula, its symbols, its
   source, and the function that implements it; `src/modelcite.test.mjs`
   holds a token from that code so the citation cannot go stale
   silently.
4. **The battery** gets a group with two checks that both matter: a
   cross-engine parity check, and a check that the measured quantity
   moves when the term is switched on. Parity alone passes when every
   engine ignores the term. Forward the new field in `simRun`
   (`src/validate.js`) so the battery can vary it.
5. **Rebuild and parity**: after any change to `cuda/engine.cu`,
   `cuda/kernels.cuh` or the INIT or TUNE frame, rebuild (`nvcc -O3
   -arch=sm_89 -o cuda/engine.exe cuda/engine.cu`, with `-ccbin` pointing
   at the C++ compiler if it is not on the path) and run
   `node cuda/dumpnet.mjs 4000 10 testnet 1` then
   `node cuda/parity.mjs testnet 10`.
6. **A refusal path**: if a setting cannot mean the same thing in some
   configuration, refuse it at init with a message that names the fix.
   Do not run it as something else.

## Writing a custom engine

An engine outside the three (another simulator, another language, other
hardware) is a separate program the page drives over a WebSocket, so it
changes nothing frozen above and is held to none of the promises the three
keep with each other: it says in its hello which terms it implements, and
the page sends it only networks it can run. ENGINE.md section 15 is the
specification, the docs page "custom nodes and engines" the introduction,
`tools/engine_template.py` a working engine to start from, the engine
settings node the way to pass it settings of its own, and
`node tools/enginecheck.mjs ws://host:port` the check against the reference
engine.

## Adding a measure

A measure of spike trains goes in `src/spikestats.js`, with its
definition and source in MODEL.md section 5 and a case in
`src/spikestats.test.mjs` checked against a reference implementation
(the existing measures are checked against Elephant). A measure that is
a ratio states its minimum spike count and returns nothing below it.

## The gates

- `node tools/test.mjs` runs every automatable test and fails on any;
  `--quick` is the pure-JS set (about five seconds) and is what the
  pre-commit hook runs (`git config core.hooksPath tools/hooks`).
- `validate.html` is the statistics battery, a browser page: run it
  before and after any change to compute functions, engines or connectivity
  (`?topics=all` for every group, `?only=<substring>` for one). Do not
  run it while another tab drives WebGPU: rate checks over short windows
  feel GPU contention first, so a lone failure in a rate check is worth
  re-running alone before it is believed.
- `cuda/parity.mjs` is the CUDA gate.
- The settings audit, above, is the node gate.

## Citing sources

A claim in documentation or in a battery group's `source` names its
source beside it: the paper, the chapter, or the file and function in
this repository. A number taken from the literature carries the paper it
comes from; a number measured here carries the date and the machine.

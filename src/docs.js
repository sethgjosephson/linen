// Documentation content.
// Param tables are generated from NODE_DEFS at render time so they cannot drift from the code; the text lives here.
// Keys: NODES[type] = { blurb, io, params: { key: text } }

export const NODES = {
  sphere: { io:'geometry out',
    blurb:'Generates a spherical volume for a neuron scatter to fill or a mask to read.',
    params:{
      center:'Center of the sphere.',
      radius:'Radius of the sphere.' } },
  box: { io:'geometry out',
    blurb:'Generates a rectangular volume for a neuron scatter to fill or a mask to read.',
    params:{
      center:'Center of the box.',
      size:'Edge lengths along x, y and z.',
      rotate:'Euler rotation of the box.' } },
  ellipsoid: { io:'geometry out',
    blurb:'Generates an ellipsoidal volume with three independent semi-axes.',
    params:{
      center:'Center of the ellipsoid.',
      radii:'Semi-axis lengths along x, y and z.',
      rotate:'Euler rotation of the ellipsoid.' } },
  cylinder: { io:'geometry out',
    blurb:'Generates a cylindrical volume along its local Y axis.',
    params:{
      center:'Center of the cylinder.',
      radius:'Cross-section radius.',
      height:'Length along the axis.',
      rotate:'Euler rotation of the cylinder.' } },
  torus: { io:'geometry out',
    blurb:'Generates a ring-shaped volume lying in the XZ plane.',
    params:{
      center:'Center of the torus.',
      radius:'Distance from the center to the middle of the tube.',
      thickness:'Radius of the tube.',
      rotate:'Euler rotation of the torus.' } },
  mesh: { io:'geometry out',
    blurb:'Reads a closed triangle mesh from an OBJ or glTF file in the project folder as a region.',
    params:{
      file:'Path of the mesh file inside the project folder, for example meshes/brain.obj.',
      object:'Named object in the file to use; blank uses every object as one region.',
      scale:'Micrometers per file unit; a file in millimeters takes 1000.',
      up:'Up axis of the file; Z up turns the file\'s Z into this space\'s Y.',
      center:'Keeps the file coordinates or centers the bounds of the mesh at the origin.',
      reload:'Reads the file again.' } },
  spline: { io:'geometry out',
    blurb:'Generates a tube along a Catmull-Rom curve through four control points; double-click to drag them in the viewer.',
    params:{
      p0:'First control point, the start of the curve.',
      p1:'Second control point.',
      p2:'Third control point.',
      p3:'Fourth control point, the end of the curve.',
      radius:'Radius of the tube.' } },
  noisefield: { io:'geometry out',
    blurb:'Generates a procedural density pattern inside a bounds box, for use as a volume or a mask.',
    params:{
      center:'Center of the bounds box.',
      size:'Edge lengths of the bounds box; nothing is generated outside it.',
      pattern:'blobs: fractal noise. patches: blobs at one spacing. stripes: parallel bands.',
      scale:'Feature size of the pattern; patches and stripes repeat at about this spacing.',
      coverage:'Fraction of the box the pattern fills; for stripes, the duty cycle of the bands.',
      octaves:'Blobs only: number of noise octaves; more add finer detail.',
      direction:'Stripes only: the direction the bands repeat along.',
      seed:'Random seed of the field.' } },
  scatter: { io:'geometry in, points out',
    blurb:'Places neurons inside the incoming geometry, each with a stable identity.',
    params:{
      hue:'Hue of this population in the viewer, the pair table and the probes.',
      fill:'Fills the volume with a fixed count or at a density; the resolution slider scales either.',
      density:'Cell density of the volume at full resolution.',
      count:'Number of neurons at full resolution, when filling by count.',
      type:'Izhikevich preset for every neuron; TC and RZ fire under ordinary background input unless a negative current holds them.',
      tag:'Population name the pair table, probes and other nodes refer to.',
      spacing:'Minimum distance between two somata; 0 disables.',
      pattern:'uniform: independent random positions. minicolumns: positions on a jittered hexagonal lattice of strands.',
      pitch:'Minicolumns only: distance between strand centers.',
      jitter:'Minicolumns only: lateral scatter of somata around their strand.',
      axis:'Minicolumns only: the axis the strands run along.',
      seed:'Random seed of the positions.' } },
  pointfile: { io:'points out',
    blurb:'Reads neuron positions from a CSV, TSV or PLY file in the project folder as a population.',
    params:{
      hue:'Hue of the file\'s one population; a file split by column is colored through its population nodes.',
      file:'Path of the file inside the project folder, for example points/somas.csv.',
      columns:'Header names or 1-based numbers of the x, y and z columns; blank takes x, y, z or the first three numeric columns.',
      id:'Column holding each row\'s id, carried with the cells for a connections table to match.',
      scale:'Micrometers per file unit; a file in nanometers takes 0.001.',
      up:'Up axis of the file; Z up turns the file\'s Z into this space\'s Y.',
      type:'Izhikevich preset for every cell in the file.',
      tag:'Population tag of the cells; with a population column set, the prefix of every population\'s tag.',
      tagColumn:'Column whose every distinct value becomes its own population, tagged with the node\'s tag and the value; up to 1000 values.',
      filter:'Rows to keep, as column=value terms separated by semicolons; != keeps the others; blank keeps every row.',
      expand:'Lays out one population node per population below this node, gathered back into its place.',
      collapse:'Removes the population nodes and their gather and reconnects this node; refused once the fan is built on.',
      reload:'Reads the file again.' } },
  receptor: { io:'points in, points out',
    blurb:'Declares receptor channels, each a synaptic time constant and sign that transmitters map onto.',
    params:{
      table:'One channel per line: name, tau in ms, + or -, optional reversal in mV, then its transmitters; # starts a comment.' } },
  population: { io:'points in, points out',
    blurb:'Keeps one population of the incoming stream by its tag and drops the rest.',
    params:{
      hue:'Hue of the cells this node keeps.',
      tag:'Population to keep: a tag, a pattern with one *, or E or I for a class.' } },
  gather: { io:'points in, points out',
    blurb:'Concatenates point streams; a spare port appears when every port is taken.',
    params:{
      ports:'Number of input ports shown; a wired port is never removed.' } },
  move: { io:'points in, points out',
    blurb:'Translates, rotates and scales points or a geometry about a pivot.',
    params:{
      translate:'Offset applied after rotation and scale.',
      rotate:'Euler rotation about the pivot.',
      scale:'Per-axis multiplier about the pivot.',
      pivot:'Fixed point of the rotation and scale.' } },
  repeat: { io:'points in, points out',
    blurb:'Lays down copies of a stream, each stepped from the last by one move.',
    params:{
      copies:'Number of copies, counting the stream as it arrived.',
      pops:'numbered per copy: each copy is its own population, named by copy number. shared across copies: copies keep their populations.',
      tag:'Prefix for the copies\' population names, as in name1.RSexc; blank numbers the population itself, as in RSexc1.',
      translate:'Translation added for each copy.',
      rotate:'Rotation added for each copy, about the pivot.',
      scale:'Scale applied again for each copy, about the pivot.',
      pivot:'Point the rotation and scale act about.',
      mirror:'Reflects every second copy in the chosen axis about the pivot.' } },
  noisewarp: { io:'points in, points out',
    blurb:'Displaces each point by a smooth, seeded vector field.',
    params:{
      amplitude:'Maximum displacement per axis.',
      scale:'Feature size of the displacement field.',
      seed:'Seed of the field.' } },
  twist: { io:'points in, points out',
    blurb:'Rotates points around an axis by an angle proportional to their position along it.',
    params:{
      axis:'Axis to twist around.',
      angle:'Rotation per millimeter along the axis.',
      center:'A point the axis passes through.' } },
  gaussblur: { io:'points in, points out',
    blurb:'Adds an independent Gaussian offset to every coordinate of every point.',
    params:{
      sigma:'Standard deviation of the offset.',
      seed:'Seed of the offsets.' } },
  cull: { io:'points and mask in, points out',
    blurb:'Removes the points inside or outside the geometry wired to its side input.',
    params:{
      keep:'Side of the mask to keep.' } },
  gradient: { io:'points in, points out',
    blurb:'Removes points at random along an axis to make a linear density ramp.',
    params:{
      axis:'Axis of the ramp.',
      start:'Probability of keeping a point at the low end of the axis.',
      end:'Probability of keeping a point at the high end of the axis.',
      seed:'Seed of the per-neuron draws.' } },
  celltype: { io:'points in, points out',
    blurb:'Assigns a built-in or custom cell type to a stream, a seeded fraction of it, or one population.',
    params:{
      row:'Row to assign: the 2003 presets, the 2007 measured rows in picoamps, the fly rows, or custom from the numbers below.',
      frac:'Fraction of the matching cells that take the row, chosen by a seeded draw.',
      name:'Custom only: name of the row, in letters, digits and underscores, not a built-in name.',
      preset:'Custom only: a built-in row whose numbers are written below; custom writes nothing.',
      tag:'Population to apply it to: a tag, a pattern with one *, or E or I; blank is every cell.',
      sign:'Whether the cell excites or inhibits its targets when the sign comes from the cell type.',
      kind:'spiking, or graded: the cell never spikes and releases in proportion to its potential above the release threshold.',
      C:'Membrane capacitance.',
      k:'Slope of the instantaneous current-voltage curve.',
      vr:'Resting potential.',
      vt:'Instantaneous threshold; must sit above the rest.',
      vpeak:'Spike cutoff where the cell resets; for a graded row, the ceiling of its potential.',
      a:'Recovery time scale.',
      b:'Recovery sensitivity to the potential above rest.',
      c:'Reset potential after a spike.',
      d:'Recovery increment after a spike.',
      thr:'Graded rows: the potential where release begins.',
      slope:'Graded rows: the span from no release to full release.',
      hue:'Hue the viewer draws these cells in.' } },
  ops: { io:'points in, points out',
    blurb:'Applies a formula to every point, reading and writing its position, drive and type.',
    params:{
      shuffle:'shuffled: cells keep their identity and wiring and swap positions by a seeded permutation.',
      expr:'One assignment per line to x, y, z, bias or ntype; reads i, n, lidx, src, a to d, and the usual math functions.',
      a:'A free number the formula reads as a.',
      b:'A free number the formula reads as b.',
      c:'A free number the formula reads as c.',
      d:'A free number the formula reads as d.',
      seed:'Seed of rand and noise in the formula.' } },
  stimulus: { io:'network in (optional mask), network out',
    blurb:'Injects current into the neurons inside a region, retuning the running simulation without rewiring.',
    params:{
      tag:'Population to drive: a tag, a pattern with one *, or E or I; blank drives every neuron in the region.',
      center:'Center of the region when no mask is wired.',
      radius:'Radius of the region when no mask is wired.',
      current:'Injected current, or the noise amplitude in noise mode; negative inhibits.',
      mode:'constant: steady current. pulse train: square pulses. ramp: rises to full over the period, then holds. noise: independent random drive per neuron per millisecond.',
      period:'Pulse train: time between pulse onsets. Ramp: time to reach full strength.',
      width:'Pulse train only: pulse duration.',
      t0:'Onset delay from the start of the simulation.',
      duration:'Active time from the onset; 0 runs without end.',
      spread:'Per-neuron heterogeneity of the drive, as the sigma of a mean-preserving lognormal gain; 0 drives every neuron alike.',
      seedSpread:'Seed of the per-neuron gains, keyed to stable neuron identity.' } },
  plasticity: { io:'points in (place before connect)',
    blurb:'Declares a named learning rule and, optionally, the population pair it applies to.',
    params:{
      preset:'A published rule that writes the terms below and the rule name; custom writes nothing.',
      name:'Name the pair table and project nodes use for this rule; must be unique.',
      from:'Presynaptic population tag; blank leaves the rule declared for something else to name.',
      to:'Postsynaptic population tag; blank leaves the rule unscoped.',
      aP:'Potentiation amplitude for a pre-before-post pairing.',
      aM:'Depression amplitude for a post-before-pre pairing.',
      wmax:'Upper limit on weight magnitude, inhibitory weights included.',
      wdep:'hard: additive updates clamped at the limit. soft: updates scale with the room left toward the bound.',
      trip:'Triplet potentiation amplitude; 0 is the pair rule.',
      iEta:'Learning rate of inhibitory homeostasis.',
      het:'Strength of heterosynaptic competition among a cell\'s incoming weights when it fires above its set point.',
      tin:'Constant weight change at an active synapse on every presynaptic spike.',
      cons:'Rate of settling into a consolidated state.',
      consW:'Weight a consolidated synapse is drawn toward; at most the rule\'s weight limit.',
      consP:'Depth of the consolidation well.' } },
  project: { io:'source and target points in, both out with the projection (place before connect)',
    blurb:'Makes a pathway from one population to another for the connect node to wire.',
    params:{
      from:'Source population tag, or a pattern with one *; a * on both sides pairs the matching copies.',
      to:'Target population tag, or a pattern with one *; a * here alone targets every match from every source.',
      offset:'Inside a repeat, joins source copy k to target copy k plus this number; 0 joins each copy to itself.',
      across:'Tag of the repeat the offset applies across; blank is the first numbered repeat.',
      dispersed:'topographic: maps position onto position. dispersed: a seeded random subset, no map. proximity: by real distance. matched: around each cell\'s copy in population order.',
      axisFrom:'Axis the source population is viewed along; the other two form the map.',
      axisTo:'Axis the target population is viewed along.',
      flipU:'Mirrors the first mapped axis.',
      flipV:'Mirrors the second mapped axis.',
      sigma:'Topographic and proximity: Gaussian spread of each source arbor.',
      prob:'Topographic and proximity: connection probability at zero distance.',
      fanout:'Dispersed: average number of target cells each source cell reaches.',
      weight:'Synaptic weight magnitude; the sign follows the presynaptic type.',
      velocity:'Axon conduction velocity.',
      tract:'Physical tract length; 0 uses the distance between population centroids.',
      frozen:'frozen: exempts these synapses from every plasticity term.',
      rule:'Plasticity rule this tract learns by, named on a plasticity node; blank uses the checkpoint\'s.',
      seed:'Seed of this projection\'s pair draws.' } },
  pool: { io:'points in (place before connect)',
    blurb:'Inhibits every cell of a population in proportion to that population\'s mean rate, one millisecond later.',
    params:{
      tag:'Population the pool covers: a tag, a pattern with one * for one pool per match, E or I, or blank for every cell.',
      gain:'Inhibition per millisecond per Hz of the pool\'s mean rate, in weight units on the inhibitory path; 0 is off.' } },
  connectionsfile: { io:'points in (place before connect)',
    blurb:'Adds explicit synapses from a connections table, matched to the ids a points file carried.',
    params:{
      file:'Path of the table inside the project folder, for example tables/connections.csv.',
      pre:'Presynaptic id column; blank finds pre_root_id, pre, source, bodyId_pre or from.',
      post:'Postsynaptic id column; blank finds post_root_id, post, target, bodyId_post or to.',
      count:'Synapse count column; blank finds syn_count, count, weight, n or synapses, or counts every row as one.',
      signCol:'Transmitter column for the transmitter sign; blank finds nt_type, neurotransmitter, nt or sign.',
      weight:'Weight per synapse, in the connect node\'s units.',
      mode:'Turns a row with count c into one weighted synapse or c synapses.',
      signMode:'Sign from the presynaptic cell type or from the transmitter column.',
      wInh:'Multiplier on the weight of inhibitory synapses.',
      compress:'Transform applied to each row\'s count before it sets the weight or the synapse number.',
      cap:'Ceiling on the count before the transform; 0 leaves it.',
      velocity:'Axon conduction velocity; the delay is the distance between the cells over it.',
      rule:'Plasticity rule of these synapses: a rule name, frozen, or blank for the checkpoint\'s.',
      reload:'Reads the table again.' } },
  connect: { io:'points in, network out',
    blurb:'Wires synapses between neurons by a distance-dependent probability, with signs from the presynaptic type.',
    params:{
      radius:'Maximum connection distance.',
      sigma:'Width of the Gaussian falloff with distance.',
      prob:'Connection probability at zero distance.',
      wExc:'Synaptic weight of excitatory presynaptic neurons.',
      wInh:'Synaptic weight of inhibitory presynaptic neurons.',
      wdist:'uniform: within 50 percent of the base weight. lognormal: heavy-tailed, with the same mean.',
      wsigma:'Lognormal only: sigma of the underlying normal distribution.',
      cluster:'Fraction of extra synapses added by triadic closure, redrawn at every wiring.',
      table:'Pair rules per line: pre post probMul wMul, then 0, 1 or a rule name; keys are tags, tag* prefixes, E, I or *.',
      velocity:'Axon conduction velocity; the delay is distance over velocity, held to 1 to 16 ms.',
      kscale:'Fraction of full-scale in-degree this wiring realizes, fed to the same compensation as density.',
      density:'Fraction of biological density the incoming points represent, compensated as on the Scaling page.',
      nuE:'Excitatory rate the compensation assumes; 0 uses the rate measured at full resolution.',
      nuI:'Inhibitory rate the compensation assumes; 0 uses the rate measured at full resolution.',
      seed:'Random seed of the wiring.' } },
  pin: { io:'anything in, the same thing out',
    blurb:'Reroutes a wire, passing its input through unchanged; ctrl-click a wire to splice one in.',
    params:{} },
  note: { io:'no connections',
    blurb:'Draws a text annotation in the graph.',
    params:{
      text:'The note text; line breaks are kept.',
      near:'Label of the group this note sits beside when a scene is laid out; blank puts it in the margin.',
      width:'Box width; text wraps to fit.' } },
  curriculum: { io:'side input: another curriculum to follow · outputs a signal',
    blurb:'Generates a lesson stream of items in one sense for an input node\'s signal port.',
    params:{
      sense:'Presents each item as an image or as its spoken name.',
      offset:'For a following curriculum: delay behind the followed stream; negative runs ahead.',
      set:'The set of items presented.',
      order:'sequential recites the set in order; shuffled draws each presentation independently.',
      onMs:'Duration of each presentation.',
      offMs:'Blank gap between presentations.',
      jitter:'Position jitter as a fraction of the field, redrawn each presentation.',
      scaleVar:'Size variation per presentation.',
      rotVar:'Rotation range per presentation.',
      acuity:'Visual sharpness; 1 is sharp and 0 heavily blurred.',
      phases:'Training blocks such as 25 A; 25 B; 130 AB: simulated minutes, then the channels driven; blank drives every channel.',
      probeEvery:'Presents one sense alone, alternating, on every Nth presentation; 0 disables.',
      condNames:'Probe condition names, comma separated: the ordinary presentation first, then the probe conditions.',
      calCycles:'Passes of the calibration sweep before the schedule starts; 0 disables the sweep and the binding measure.',
      reSweepEvery:'Presentations between repeats of the calibration sweep; 0 for none.',
      place:'Whole sheet: items overlap fully. Own tile per item: each item small in its own tile. Own offset per item: full size, shifted per item.',
      placeOrder:'Whether sound uses the same tile order as sight or a seeded shuffle where no item keeps its tile.',
      scramble:'Redraws the tone paired with each item on every presentation.',
      voice:'Mixes recorded speech into the sound stream, in rotation with the synthetic formants.',
      seed:'Seed of the item order and the pose jitter.' } },
  live: { io:'no inputs · outputs a signal',
    blurb:'Turns a microphone or webcam on this machine into a signal for an input node.',
    params:{
      device:'microphone: log-spaced bands from 80 Hz to 8 kHz. webcam: the downsampled luminance image, mirrored.' } },
  testsignal: { io:'no inputs · outputs a signal',
    blurb:'Generates a test pattern signal for an input node, with no device or file.',
    params:{
      pattern:'bar sweep: a bright Gaussian bar sweeping across the channels.',
      period:'Duration of one full sweep.',
      direction:'Direction the bar moves; alternate every sweep makes each probe report a rate per direction and their index.' } },
  footage: { io:'no inputs · outputs a signal',
    blurb:'Plays the image files in a folder as a frame sequence signal for an input node.',
    params:{
      path:'The granted folder, plus optional subfolders as folder/sub.',
      choose:'Opens the browser folder picker and grants read access to the chosen folder.',
      fps:'Playback frame rate.',
      loop:'once holds the last frame; loop repeats.' } },
  input: { io:'network in · side inputs: mask (geometry), signal (curriculum, test signal, live or footage)',
    blurb:'Maps a signal\'s channels onto the neurons of a population inside the mask.',
    params:{
      tag:'Population the channels land on: a tag, a pattern with one *, or E or I; blank is every cell in the mask.',
      map:'sheet: a cols by rows grid across the axis. bands: cols slabs along the axis. distributed: seeded random scatter over the region.',
      code:'raw luminance passes the source through; retinal contrast sends center-surround differences on separate ON and OFF channels.',
      axis:'Sheet: the axis normal to the grid. Bands: the axis the bands divide.',
      cols:'Channels across the grid, or the number of bands.',
      rows:'Sheet and distributed: channels down the grid.',
      arbor:'Gaussian spread of each neuron\'s input across nearby channels; 0 gives one channel per neuron.',
      fanin:'Distributed: average number of channels converging on each neuron.',
      seed:'Distributed: seed of the random projection.',
      transient:'Adaptation strength; at 1 only changes in the input drive the network.',
      lagMs:'Delay of this map behind the lesson clock.',
      jitter:'Microsaccade amplitude, as small random raster shifts.',
      amp:'Injected current per neuron at gain 1; negative withdraws current where the signal is bright.' } },
  chart: { io:'network in (place in the readout chain)',
    blurb:'Plots what the network is doing or what a run measured, and saves the plot as a figure.',
    params:{
      source:'Run or sweep to draw from the project folder; empty is this page\'s current or last run.',
      plot:'The plot to draw; live plots read the probes, the others read a training run.',
      pop:'Population to plot, from those the run or the probes report.',
      window:'Span of the live recording the rate and raster plots show.',
      savePng:'Saves the figure, at the size and style on the figure tab, to the project\'s figures folder.',
      saveCsv:'Saves the numbers behind the plot as CSV, with a JSON file describing them.',
      smooth:'Moving-average window over the rate; 0 draws it as recorded.',
      title:'Title over the plot in a saved figure; blank leaves it out.',
      xText:'Title of the x axis in a saved figure.',
      yText:'Title of the y axis in a saved figure.',
      yMin:'Low end of the y axis or color scale.',
      yMax:'High end of the y axis or color scale.',
      grid:'Grid lines behind the data.',
      legend:'Legend naming each line when a plot draws several.',
      cmap:'Color map of a matrix plot.',
      preview:'Draws the working plot, or the figure as it will be saved.',
      format:'File format of the saved figure.',
      sizePreset:'Common figure size; choosing one writes the width and height.',
      figW:'Width of the saved figure, caption included.',
      figH:'Height of the saved figure.',
      dpi:'Resolution of a saved PNG.',
      theme:'Color scheme of the saved figure.',
      figFont:'Typeface of the figure.',
      fontPt:'Size of the figure text at the saved width.',
      linePt:'Width of data lines; axis lines scale with it.',
      caption:'Prints the provenance as a caption under the figure, or keeps it only in the file\'s metadata.',
      which:'Similarity matrix and PSTH: the condition to show.',
      field:'Measure to plot, by name or dotted path, from those in the run\'s checkpoints.',
      bins:'Number of bins in the weight distribution.',
      height:'Height of the plot on the panel.' } },
  analysis: { io:'network in',
    blurb:'Sets what a training run\'s readout measures, and how, at every checkpoint.',
    params:{
      cap:'Number of cells sampled from each probe for every measure.',
      setExpr:'Cells of interest as an expression over condition names with and, or, not; training runs ignore it and use the default.',
      probeWith:'Conditions to test recall with, comma separated; training runs ignore it and test sight and sound alone.',
      topQ:'Fraction of cells counted as responding to an item.',
      margin:'How much higher a cell must rank with both senses than with either alone to count as conjunctive.',
      minCells:'Fewest conjunctive cells an item needs before its timing correlation is reported.',
      nPerm:'Label permutations behind the cross-decoding p-value; 0 reports the accuracy without p.',
      minTrials:'Trials per item needed before decoding is attempted.',
      doDecode:'Decodes the item from each population against chance; binding, timing and like-to-like need it on.',
      doBind:'Measures the conjunctive cells and whether one sense alone reaches them; needs a calibration sweep.',
      doTiming:'Measures whether one sense wakes those cells in the order the pair does; needs first-spike latency from the engine.',
      doLike:'Measures whether connectivity tracks tuning similarity beyond distance.',
      exportTrials:'Writes the per-trial, per-cell spike counts beside the metrics at every checkpoint, one file per run.' } },
  probe: { io:'network in · side input: region mask (geometry)',
    blurb:'Reads the spike rate of a population or region live, as a number, a picture or sound.',
    params:{
      tag:'Population to read: a tag, a pattern with one *, or E or I; blank reads every cell the mask holds.',
      label:'Name shown in the overlay.',
      record:'Keeps a per-stimulus spike record for the analysis measures while a curriculum runs.',
      view:'rate: sparkline only. image: also decodes spikes onto a grid over the region.',
      audio:'spikes: rate-modulated crackle. bands: one oscillator per frequency band, playing at its band\'s activity.',
      axis:'Image view: the projection normal. Bands audio: the tonotopic axis.',
      cols:'Image cells across, or the number of audio bands.',
      rows:'Image cells down.' } },
  enginesettings: { io:'network in, the same network out (between connect and the checkpoint)',
    blurb:'Sends key and value settings to a custom engine on every init and tune.',
    params:{
      table:'One setting per line: key, space, value; keys are letters, digits and _ . : -, and # starts a comment.' } },
  checkpoint: { io:'network in',
    blurb:'Simulates the network fed into it and holds the settings of the simulation, plasticity and training.',
    params:{
      steps:'Simulated milliseconds per rendered frame.',
      liveHost:'WebSocket address of the remote engine.',
      engine:'auto picks cpu or gpu by size; cpu runs the reference engine; gpu runs WebGPU; remote runs an engine at the address below.',
      syn:'kick: charge in one step. exp decay: an exponentially decaying current. conductance: a decaying conductance toward each channel\'s reversal potential.',
      eRevE:'Conductance mode: reversal potential of the E channel.',
      eRevI:'Conductance mode: reversal potential of the I channel.',
      psc:'Meaning of the weight in exp mode.',
      tauE:'Exp mode: excitatory decay time constant.',
      tauI:'Exp mode: inhibitory decay time constant.',
      plast:'Switches learning on while the simulation runs: pair STDP and inhibitory homeostasis; can change mid-run.',
      aP:'STDP potentiation amplitude per pre-before-post pairing.',
      aM:'STDP depression amplitude per post-before-pre pairing.',
      tauS:'Presynaptic trace time constant, the potentiation window.',
      tauM:'Postsynaptic trace time constant, the depression window.',
      wdep:'hard: clamps weights at the limit. soft: scales each update by the room left toward the bound.',
      wmax:'Upper limit on any weight magnitude under learning.',
      iEta:'Learning rate of the inhibitory homeostasis rule.',
      rhoMode:'fixed target: one rate for every neuron. measured baseline: each neuron\'s own rate during a calibration window.',
      calS:'Measured set points only: length of the calibration window, with plasticity held off.',
      iRho:'Target excitatory rate for inhibitory homeostasis and synaptic scaling.',
      scale:'Scales each neuron\'s plastic excitatory inputs toward the target rate once per simulated second.',
      sEta:'Maximum fractional scaling step per second.',
      trip:'Triplet potentiation amplitude; 0 is the pair rule.',
      tauY:'Time constant of the slow postsynaptic trace read by the triplet and heterosynaptic terms.',
      het:'Heterosynaptic competition: pulls incoming weights toward their reference when a neuron fires above its set point.',
      stp:'Short-term plasticity on every outgoing synapse; any value above 0 switches it on.',
      stpPreset:'A published synapse class that writes the three numbers below; custom writes nothing.',
      stpU:'Release probability of a rested synapse, and the fraction of resources each spike consumes.',
      stpNorm:'Whether a rested synapse releases U of its weight or exactly its weight.',
      stpOrder:'Order of the release and the facilitation jump at a spike.',
      stpTauD:'Recovery time constant of synaptic resources.',
      stpTauF:'Decay time constant of the raised release probability.',
      cons:'Consolidates the heterosynaptic reference through a double well; any value above 0 is on, and all such values run alike.',
      consW:'Upper stable state of the consolidation well.',
      consStep:'Interval between consolidation steps.',
      tauCons:'Time a reference weight must stay past the midpoint before it consolidates.',
      commit:'Freezes a consolidated synapse at its weight; needs consolidation on.',
      consP:'Depth of the consolidation well; 0 leaves the reference tracking the weight.',
      tin:'Transmitter-induced potentiation added on every presynaptic spike.',
      vmin:'Floor on the membrane potential; 0 disables it.',
      refrac:'Absolute refractory period; 0 disables it.',
      trHours:'Simulated length of a training run; a run needs the resolution slider at 100 percent.',
      trCkptMin:'Interval between metrics files in a run.',
      trBrainMin:'Interval between brain files in a run.',
      runs:'Lists every run in the project folder in the training pane\'s FILES tab.',
      trReps:'Number of runs per condition, each with offset curriculum and noise seeds on the same wiring.',
      variants:'Sweep variants, each a list of node setting edits, run one after another on copies of the graph.',
      trSweep:'Conditions the sweep covers: the graph as configured, or each replicate paired and scrambled.',
      trKeep:'Number of brain files a run keeps; older ones are deleted.',
      trHost:'Address of the engine host the cuda option connects to; START starts one there when none listens.',
      trEngine:'Engine of a training run; cuda hands the run to the native engine host at the address below.',
      train:'Starts a training run on the graph upstream of this checkpoint.',
      brainSave:'Writes the graph and every synapse weight to a brain file in the project\'s brains folder.',
      exportNet:'Saves the wired network as two CSV files, one row per cell and one per synapse.',
      exportSecs:'Simulated length of the exported Brian 2 case.',
      exportBrian:'Saves the network as a Brian 2 case, with tools/brian_ref.py beside it to run it.',
      brainLoad:'Loads a brain file: a matching graph takes its weights, another graph opens as a new scene.' } },
};

export const PAGES = [
  { id:'overview',
title:'overview', html:`
<h1><img src="brand/linen-dark.svg" alt="linen" style="height:2.4em;display:block;margin:6px 0 14px 0"></h1>
<p>A sandbox for building and running 3D spiking neural networks.</p>
<h2>finding your way around</h2>
<p>The screen has three parts. The node graph is where you build a network, the viewer shows it running in 3D, and the properties panel holds the settings of whichever node you open. The guided tour is open when you first arrive, and the other scenes are in the Tab menu, each described on the scenarios page. The references page lists the papers behind them.</p>
<p>To give the viewer or the node graph the whole window, hover over it and press Space. Press Space again to bring the other panels back.</p>
<p>A network is described entirely by its graph, with data flowing from top to bottom. Everything is measured in micrometers, so the time a signal takes to reach another neuron depends on how far it has to travel.</p>
<h2>the viewer</h2>
<p>Use the mouse to orbit, pan and zoom around the tissue. Clicking a neuron shows its outgoing connections in white and its incoming ones in violet, and an oscilloscope appears at the top right with its membrane potential. Small bumps there are synaptic input that stayed below threshold, and a ramp followed by a sharp reset is a spike. Double-clicking a neuron selects the node that created it.</p>
<p>The strip under the 3D view is the spike raster. Time runs from left to right, each row is a neuron, and a white dot means that neuron fired. Vertical stripes mean many neurons are firing together, diagonal patterns show activity traveling through the tissue, and an even speckle is asynchronous, irregular firing.</p>
<h2>the viewer bar</h2>
<p>The bar above the viewer pauses the simulation, locks the weights and sets the resolution, which the scaling page explains. Per frame sets how much simulated time runs between screen updates. A larger value gets more done on a big network but looks less smooth, and underneath, the simulation still steps one millisecond at a time. The realtime factor beside it shows how well the engine is keeping up, where 1.00x is biological speed.</p>
<p>SHOW hides neurons whose membrane potential is below a level you choose, so at +40 mV you only see the cells that are firing. SLICE puts a box in the viewer and hides everything outside it, which is handy for looking inside dense tissue without changing the network. REC records the session, as described further down. VIEWER WINDOW moves the whole viewer, this bar included, into a window of its own, so you can put it on another screen while the graph and the properties take the room here. Close that window, or press the viewer button that appears in the top bar, to bring it back.</p>
<p>Locking the weights matters once a network has learned something. While they're unlocked, any change to the structure rebuilds the network from scratch and throws away what plasticity learned. Locked, setting changes still go through, but structural edits and the resolution slider are turned away until you unlock.</p>
<h2>building in the node graph</h2>
<p>Press Tab to open a searchable menu of every tool, grouped by what it does, with the scenarios at the end. Click a node to select it, and double-click to open its properties. On a shape or a stimulus, double-clicking also gives you a handle in the viewer that you can move, rotate and scale with W, E and R. Ctrl+F finds any node by name.</p>
<p>Wires connect by dragging from one port to another. Dropping a node onto a wire inserts it there, and giving a node a quick shake while you drag it pulls it out of its connections. Ctrl-clicking a wire adds a pin, which only reroutes the wire. Names are just labels, so renaming a node never changes the network. The keyboard page lists every shortcut.</p>
<p>The number keys bind nodes to the viewer. Select a node and press a digit to show it, then press that digit later to come back to it. A checkpoint shows the running simulation, a points node previews its neurons, and a shape shows its outline.</p>
<h2>changing settings</h2>
<p>The properties panel applies your edits as you make them. Changing a shape or the wiring rebuilds the network while the old one keeps running, and changing a stimulus or the checkpoint adjusts the running simulation in place.</p>
<h2>the top bar</h2>
<p>The top bar shows how many neurons and synapses are running and what just happened. The path in the middle, project / scene, opens menus for switching projects and scenes. KEYS changes key bindings, DOCS opens these pages, and the layout button cycles through four pane arrangements, each keeping the sizes you drag it to.</p>
<h2>projects and scenes</h2>
<p>When you run linen with serve.py, your work lives in a project folder on your disk, ./project unless you pick another from the project menu. Without the server, it's kept in your browser's storage instead, which belongs to the address you opened linen from.</p>
<p>A scene is one node graph, and it saves continuously as you work, so there's no save button. SAVE AS makes a copy to keep working in, and NEW starts a fresh scene from the guided tour. Scenes open in tabs under the graph. Switching tabs changes what you're editing, and the eye icon marks the scene the viewer is running; pressing a viewer key in another tab moves the viewer there. Each tab keeps its own undo history, and copy and paste works between them.</p>
<p>The scene menu can also copy a link that carries the whole scene inside it. Nothing is uploaded, and whoever opens the link gets the scene exactly as you had it, down to every seed. Scenarios from the Tab menu open with the same layout every time, and the scene menu can save your own arrangement as a scenario's default.</p>
<h2>brains, training and recordings</h2>
<p>A brain file holds a network's learned weights. SAVE BRAIN on the checkpoint's run tab writes one to the project's brains folder, and LOAD BRAIN brings it back. If the graph matches, only the weights load; otherwise the brain opens as a new scene. When you rewire after loading, connections that still exist keep their weights and new ones start fresh.</p>
<p>A training run is a checkpoint left running for a long time. Its settings sit on the checkpoint node, and START TRAINING at the bottom of its properties begins the run. As it goes, it writes metrics every few simulated minutes and brain checkpoints you can resume from, all in a folder under runs.</p>
<p>REC records whatever is running until you press it again: the firing rate of each probed population, every setting you change and when, and a summary of the weights before and after. It saves as JSON and CSV in the project's recordings folder, and a chart node can plot the recording as it happens.</p>
<h2>live input and output</h2>
<p>An input node feeds a signal into a population while the simulation runs. The signal can come from a curriculum, a test signal, your microphone or webcam, or a folder of images, and a mask decides which neurons receive it. Probe nodes work the other way, showing a population's firing rate as it runs, and they can even play it as sound once you click the audio button.</p>
<h2>the guided tour</h2>
<p>The scene you see first is a thin sheet of about 14,000 neurons at the density of mouse cortex. A cluster of chattering cells, a bursting cortical cell type (Gray and McCormick 1996), fires in bursts, and each burst crosses the sheet as a wave. A second source pulses on a different rhythm, so the two sets of waves collide.</p>
<p>This is closer to a slice of cortex in a dish than to an awake brain, where waves gently shape sparse firing instead of recruiting nearly every cell. The traveling waves scene is a larger version of the same sheet, and its wave fronts move at the speeds measured in cortical slices.</p>` },
  { id:'keys', title:'keyboard', html:'' },          // generated from keymap.js
  { id:'nodes', title:'node reference', html:'' },   // generated
  { id:'scenarios', title:'scenarios', html:`
<h1>scenarios</h1>
<p>Prebuilt networks opened from the Tab menu.</p>
<h2>common to all</h2>
<table>
<tr><td class="k">dimensions</td><td class="d">Micrometers.</td></tr>
<tr><td class="k">density</td><td class="d">Mouse cortical density with unitary synaptic weights.</td></tr>
<tr><td class="k">size against speed</td><td class="d">The resolution slider.</td></tr>
<tr><td class="k">layout</td><td class="d">Identical on every opening (see the overview).</td></tr>
<tr><td class="k">battery</td><td class="d">Holds each scene to its tissue's literature rates, and each teaching scene to the claim in its notes.</td></tr>
</table>
<h2>the Tab menu</h2>
<table>
<tr><th>scene</th><th>cells</th><th>synapses</th><th>shows</th></tr>
<tr><td class="k">guided tour</td><td class="v">14,000</td><td class="v">8 M</td><td class="d">Waves from a pacemaker across a sheet.</td></tr>
<tr><td class="k">traveling waves</td><td class="v">39,000</td><td class="v">27 M</td><td class="d">Fronts at slice speeds.</td></tr>
<tr><td class="k">mouse cortical column</td><td class="v">14,000</td><td class="v"></td><td class="d">Four layers at published proportions and rates.</td></tr>
<tr><td class="k">visual pathway</td><td class="v"></td><td class="v"></td><td class="d">Retina to thalamus to cortex.</td></tr>
<tr><td class="k">thalamocortical loop</td><td class="v"></td><td class="v"></td><td class="d">Activity crossing a fiber tract in hops.</td></tr>
<tr><td class="k">CA3 pattern completion</td><td class="v"></td><td class="v"></td><td class="d">Recall of a band from a cue at one end.</td></tr>
<tr><td class="k">balanced random net</td><td class="v"></td><td class="v"></td><td class="d">Brunel's asynchronous and synchronous regimes.</td></tr>
<tr><td class="k">sound localization</td><td class="v"></td><td class="v"></td><td class="d">Delay lines and coincidence detection.</td></tr>
<tr><td class="k">virtual patch rig</td><td class="v">7</td><td class="v"></td><td class="d">Current to rate curves.</td></tr>
<tr><td class="k">seizure and control</td><td class="v"></td><td class="v"></td><td class="d">Runaway volleys without inhibition.</td></tr>
<tr><td class="k">gamma from inhibition</td><td class="v"></td><td class="v"></td><td class="d">A gamma rhythm from fast spiking cells.</td></tr>
<tr><td class="k">working memory</td><td class="v"></td><td class="v"></td><td class="d">An item held in facilitated synapses.</td></tr>
<tr><td class="k">polychrony</td><td class="v"></td><td class="v"></td><td class="d">Firing order selecting a reader.</td></tr>
<tr><td class="k">every node</td><td class="v">43,000</td><td class="v">12 M</td><td class="d">A node of every kind in one structure.</td></tr>
</table>
<h2>guided tour</h2>
<p>The default scene: a sheet of cortex whose pacemaker bursts cross it as waves.</p>
<table>
<tr><td class="k">size</td><td class="d">700 by 300 by 700 µm.</td></tr>
<tr><td class="k">density</td><td class="d">72,000 excitatory and 18,000 inhibitory cells per mm³.</td></tr>
<tr><td class="k">cells, synapses</td><td class="d">About 14,000 and 8 million.</td></tr>
<tr><td class="k">pacemaker</td><td class="d">A cluster of chattering cells off to one side.</td></tr>
<tr><td class="k">sheet width</td><td class="d">600 µm still carries fronts; 500 µm does not.</td></tr>
<tr><td class="k">resolution</td><td class="d">100%; the waves depend on the full density.</td></tr>
<tr><td class="k">key 1</td><td class="d">The sheet.</td></tr>
<tr><td class="k">key 2</td><td class="d">A sculpture of shapes for the 3D manipulator: a spline tube, a ring, a tilted cylinder.</td></tr>
<tr><td class="k">key 3</td><td class="d">Thalamocortical relay cells silenced by an inhibitory pulse every three seconds.</td></tr>
<tr><td class="k">rebound</td><td class="d">The relay fires a synchronized burst on release, the post-inhibitory rebound the TC row models.</td></tr>
</table>
<h2>traveling waves</h2>
<p>The tour's sheet enlarged, with a second, pulsed source in the far corner.</p>
<table>
<tr><td class="k">size</td><td class="d">1200 by 300 by 1200 µm.</td></tr>
<tr><td class="k">cells, synapses</td><td class="d">About 39,000 and 27 million.</td></tr>
<tr><td class="k">front speed</td><td class="d">Inside the 1 to 10 cm/s band for cortical slices (Chervin 1988; Golomb and Amitai 1997); the battery measures it.</td></tr>
<tr><td class="k">kernel</td><td class="d">200 µm; at 100 nothing spreads.</td></tr>
<tr><td class="k">inhibition at a third</td><td class="d">The whole sheet bursts at once; the fronts are gone.</td></tr>
</table>
<h2>mouse cortical column</h2>
<p>A four-layer column at the population proportions of Potjans and Diesmann 2014.</p>
<table>
<tr><td class="k">size</td><td class="d">400 µm wide, 1.2 mm deep, four layer boxes.</td></tr>
<tr><td class="k">cells</td><td class="d">About 14,000.</td></tr>
<tr><td class="k">types</td><td class="d">Layer 5 pyramids intrinsically bursting; a tenth of layer 2/3 chattering.</td></tr>
<tr><td class="k">interneurons</td><td class="d">Split per layer between fast spiking and low threshold spiking by cell type nodes.</td></tr>
<tr><td class="k">wiring</td><td class="d">The 8 by 8 population pair table on the connect node; every probability as published.</td></tr>
<tr><td class="k">drive</td><td class="d">Background per layer and class; a weak pulse train into layer 4 through its box as a mask.</td></tr>
<tr><td class="k">excitatory rates</td><td class="d">0.34, 1.18, 2.30 and 0.92 Hz from layer 2/3 down, pulse off; inside de Kock and Sakmann 2009.</td></tr>
<tr><td class="k">inhibitory</td><td class="d">Every population above its excitatory partner; the state is asynchronous.</td></tr>
</table>
<h2>visual pathway</h2>
<p>The early visual route as a wired circuit, from retina through thalamus to the column.</p>
<table>
<tr><td class="k">retina</td><td class="d">A sheet of 1,200 ganglion cells.</td></tr>
<tr><td class="k">relay</td><td class="d">Thalamocortical cells at density inside a shell of reticular cells.</td></tr>
<tr><td class="k">cortex</td><td class="d">The four-layer column.</td></tr>
<tr><td class="k">tracts</td><td class="d">Four project nodes: optic tract (retina to relay), radiation (relay to layer 4), its collateral to layer 6, feedback to the relay.</td></tr>
<tr><td class="k">topography</td><td class="d">Retinotopy survives every stage; delays follow myelinated conduction.</td></tr>
<tr><td class="k">input</td><td class="d">A test signal's bar sweep as an ON and OFF contrast code on the retina; it reaches cortex through the anatomy.</td></tr>
<tr><td class="k">rates, sweep on</td><td class="d">Retina 13 Hz, relay 11, layer 4 13, layer 2/3 18.</td></tr>
</table>
<h2>thalamocortical loop</h2>
<p>A cortical patch and a thalamic nucleus joined by a fiber tract made of cells.</p>
<table>
<tr><td class="k">cortex</td><td class="d">A patch at density.</td></tr>
<tr><td class="k">relay</td><td class="d">Thalamocortical cells in a sphere, inside a reticular shell drawn as a torus.</td></tr>
<tr><td class="k">tract</td><td class="d">A cylinder of cells between patch and nucleus.</td></tr>
<tr><td class="k">wiring</td><td class="d">Every synapse local; connect radius 600 µm; the patch 805 µm from the nucleus.</td></tr>
<tr><td class="k">timing</td><td class="d">Activity crosses the tract in hops; its length shows in the timing.</td></tr>
<tr><td class="k">drive</td><td class="d">A holding current keeps relay cells below threshold; sensory pulses reach the nucleus every 120 ms.</td></tr>
<tr><td class="k">rates</td><td class="d">Cortex about 10 Hz, its interneurons 11, relay 18, shell 11.</td></tr>
</table>
<h2>CA3 pattern completion</h2>
<p>A recurrent pyramidal band that completes a cue at one end across its length.</p>
<table>
<tr><td class="k">band</td><td class="d">Pyramidal cells along a spline tube at density, 15% bursting, with feedback inhibition.</td></tr>
<tr><td class="k">cue</td><td class="d">Weak, to one end, every five seconds; the rest joins through the recurrent collaterals within about 100 ms.</td></tr>
<tr><td class="k">inhibitory pulse</td><td class="d">Global, two seconds after each cue.</td></tr>
<tr><td class="k">rates</td><td class="d">0.4 Hz between events; 100 ms after the cue, 38 Hz at the cued end, 24 Hz at the far end.</td></tr>
<tr><td class="k">containment</td><td class="d">An absolute refractory period, short-term depression and enough inhibition keep the recall an event.</td></tr>
<tr><td class="k">depression on</td><td class="d">The response moves 14 percent while the recurrent weight moves 64.</td></tr>
</table>
<h2>balanced random net</h2>
<p>Excitatory and inhibitory cells wired at random in one sphere, after Brunel (2000).</p>
<table>
<tr><td class="k">populations</td><td class="d">4 to 1 excitatory to inhibitory, at density.</td></tr>
<tr><td class="k">wiring</td><td class="d">Random, probability 0.1; inhibition five times excitation.</td></tr>
<tr><td class="k">drive</td><td class="d">Noise.</td></tr>
<tr><td class="k">regime</td><td class="d">Asynchronous irregular at 4.6 Hz.</td></tr>
<tr><td class="k">equal weights</td><td class="d">Volleys of 36 Hz every 200 to 300 ms.</td></tr>
<tr><td class="k">battery</td><td class="d">Measures both regimes and the population Fano factor between them.</td></tr>
</table>
<h2>sound localization</h2>
<p>Coincidence detectors between two ears read the interaural time difference by place.</p>
<table>
<tr><td class="k">circuit</td><td class="d">Two ears, a row of detectors, axons timed by length (Jeffress 1948; Carr and Konishi 1990 for the owl).</td></tr>
<tr><td class="k">stimulus</td><td class="d">Each ear fires a volley at every click.</td></tr>
<tr><td class="k">detector</td><td class="d">Fires where both volleys arrive in the same millisecond; one ear alone stays under threshold.</td></tr>
<tr><td class="k">example</td><td class="d">At 100 µm per ms, a detector 200 µm from the middle hears a 4 ms lead as simultaneous.</td></tr>
<tr><td class="k">control</td><td class="d">The right ear's onset.</td></tr>
<tr><td class="k">probes</td><td class="d">Left third, middle and right third of the row.</td></tr>
</table>
<h2>virtual patch rig</h2>
<p>Seven cells, one of each classic type, each alone with an electrode.</p>
<table>
<tr><td class="k">current</td><td class="d">Ramps from 0 to 12 over three seconds, then held.</td></tr>
<tr><td class="k">probes</td><td class="d">Each probe's rate is that cell's current to rate curve.</td></tr>
<tr><td class="k">rheobase</td><td class="d">The current where a cell starts firing (Connors and Gutnick 1990; Izhikevich 2003 Fig. 2).</td></tr>
<tr><td class="k">click a cell</td><td class="d">Its membrane potential.</td></tr>
</table>
<h2>seizure and control</h2>
<p>The balanced random net with its inhibitory cells silenced for two seconds.</p>
<table>
<tr><td class="k">silencing</td><td class="d">A hyperpolarizing current on every inhibitory cell from three seconds in: the model's GABA antagonist on a slice.</td></tr>
<tr><td class="k">inhibition out</td><td class="d">Recruitment runs away into synchronous volleys.</td></tr>
<tr><td class="k">recovery</td><td class="d">Within a second of inhibition returning (Brunel 2000's regimes).</td></tr>
<tr><td class="k">scope</td><td class="d">A mechanism, not a clinical simulation; it says nothing about any patient.</td></tr>
</table>
<h2>gamma from inhibition</h2>
<p>A tonically driven cortical patch at density, paced into the gamma band by its fast spiking cells.</p>
<table>
<tr><td class="k">cycle</td><td class="d">A pyramidal volley recruits the interneurons; their inhibition silences the patch for about one GABA-A decay; the next volley follows.</td></tr>
<tr><td class="k">sources</td><td class="d">Whittington, Traub and Jefferys 1995; Whittington et al. 2000.</td></tr>
<tr><td class="k">drive</td><td class="d">Sets the band.</td></tr>
<tr><td class="k">inhibitory tau</td><td class="d">On the checkpoint; sets the length of the silence.</td></tr>
<tr><td class="k">chart</td><td class="d">Rate over time shows the rhythm.</td></tr>
</table>
<h2>working memory</h2>
<p>An item held in synapses, not in firing (Mongillo, Barak and Tsodyks 2008).</p>
<table>
<tr><td class="k">network</td><td class="d">Two selective populations and one inhibitory pool; every synapse facilitates.</td></tr>
<tr><td class="k">synapses</td><td class="d">Each spike raises a low release probability for 1.5 s, against a short depression.</td></tr>
<tr><td class="k">class</td><td class="d">The facilitating class of prefrontal synapses (Wang et al. 2006).</td></tr>
<tr><td class="k">cue</td><td class="d">300 ms to one population every four seconds, taking turns; then silence.</td></tr>
<tr><td class="k">read pulse</td><td class="d">Weak, equal to both, 1.3 s after the cue ends; the cued population answers.</td></tr>
<tr><td class="k">controls</td><td class="d">No cue, or a facilitation of 50 ms: the two answer alike.</td></tr>
<tr><td class="k">against the paper</td><td class="d">Every synapse facilitates here; the paper facilitates excitatory to excitatory only.</td></tr>
</table>
<h2>polychrony</h2>
<p>Firing order alone decides which of two readers the same cells drive (Izhikevich 2006).</p>
<table>
<tr><td class="k">layout</td><td class="d">Three sources on a line, a reader beyond each end, every source wired to both.</td></tr>
<tr><td class="k">conduction</td><td class="d">100 µm per ms.</td></tr>
<tr><td class="k">delays</td><td class="d">4, 10 and 16 ms to one reader; 16, 10 and 4 to the other.</td></tr>
<tr><td class="k">input</td><td class="d">The sources fire in one order, six milliseconds apart.</td></tr>
<tr><td class="k">result</td><td class="d">Spikes reach one reader in the same millisecond and fire it; the opposite order fires the other.</td></tr>
<tr><td class="k">wiring</td><td class="d">By hand; in the paper the groups emerge through spike-timing plasticity in a random network with delays.</td></tr>
</table>
<h2>every node</h2>
<p>One structure with a node of every kind that needs no file or device.</p>
<table>
<tr><td class="k">shape</td><td class="d">A balanced core; six arms forking into three twigs; the crown in three tiers that shrink and turn.</td></tr>
<tr><td class="k">arm</td><td class="d">A flat box filled by a neuron scatter and passed through every arrange node.</td></tr>
<tr><td class="k">arrange nodes</td><td class="d">Cull cuts an eyelet, gradient thins the tip, twist turns, noise warp bends, gauss blur softens, operations lays a wave.</td></tr>
<tr><td class="k">twigs</td><td class="d">A noise field forked three ways by a repeat; a move tilts the ray.</td></tr>
<tr><td class="k">cell types</td><td class="d">A row of the scene's own for the arm; half the twig cells chattering.</td></tr>
<tr><td class="k">names</td><td class="d">Two more repeats make six rays and three tiers; each cell's name gives its place (tier2.ray4.fork1.bud).</td></tr>
<tr><td class="k">population nodes</td><td class="d">Three pick out the tiers; the top one is tilted on its own.</td></tr>
<tr><td class="k">wiring</td><td class="d">Gather, receptors, plasticity, feedback inhibition, project, connect.</td></tr>
<tr><td class="k">drive</td><td class="d">A noise current, a pulse by population, a pulse through a sphere mask, a bar sweep across the lowest tier, a lesson on the core.</td></tr>
<tr><td class="k">readout</td><td class="d">Probes, analysis, the checkpoint with plasticity on, a chart.</td></tr>
<tr><td class="k">size</td><td class="d">About 43,000 cells and 12 million synapses.</td></tr>
<tr><td class="k">notes</td><td class="d">One per group, on what its nodes add.</td></tr>
</table>
<h2>experiment scenes</h2>
<p>Learning experiments, not in the Tab menu.</p>
<p>Load by name from the address bar, <code>?scenario=the weave</code>, or from the trainer.</p>
<p>Built by cell count; alphabet school 2 is at mouse density.</p>
<table>
<tr><th>scene</th><th>cells</th><th>synapses</th><th>shows</th></tr>
<tr><td class="k">learning lab</td><td class="v">60 E, 15 I (patch)</td><td class="v"></td><td class="d">Cross-modal completion at growing scale.</td></tr>
<tr><td class="k">binding bench</td><td class="v">400 E, 100 I (association)</td><td class="v"></td><td class="d">Two senses converging on a learning area.</td></tr>
<tr><td class="k">the weave</td><td class="v">800 E, 200 FS</td><td class="v"></td><td class="d">Two codes in one tissue.</td></tr>
<tr><td class="k">the weave, letters</td><td class="v">1,600 E, 400 FS</td><td class="v"></td><td class="d">The weave with letters and spoken names.</td></tr>
<tr><td class="k">mandelbulb</td><td class="v">5,000 E, 1,250 FS (core)</td><td class="v"></td><td class="d">Twelve bulbs and a core from two repeats.</td></tr>
<tr><td class="k">PD microcircuit</td><td class="v">77,169</td><td class="v">300 M</td><td class="d">Potjans and Diesmann 2014 at their scale.</td></tr>
<tr><td class="k">alphabet school 2</td><td class="v"></td><td class="v"></td><td class="d">The training scenario.</td></tr>
</table>
<h2>learning lab</h2>
<p>A retina and a cochlea see the same item and project to one cortical patch.</p>
<table>
<tr><td class="k">patch</td><td class="d">About 60 excitatory and 15 inhibitory cells.</td></tr>
<tr><td class="k">dropped modality</td><td class="d">Every fourth presentation, alternating which.</td></tr>
<tr><td class="k">measure</td><td class="d">Whether one sense alone recreates the pattern both together produced.</td></tr>
<tr><td class="k">scale</td><td class="d">Sheets, patch and in-degree grow together; the pair table divides probabilities by the scale.</td></tr>
<tr><td class="k">in-degree</td><td class="d">Stays put while the populations grow.</td></tr>
<tr><td class="k">plasticity</td><td class="d">Depression fixed point on the wired mean weight, as in alphabet school 2; synaptic scaling on.</td></tr>
</table>
<h2>binding bench</h2>
<p>Two senses converge on one association area whose recurrent synapses learn.</p>
<table>
<tr><td class="k">senses</td><td class="d">A retinal sheet and a cochlear strip, each projecting with low convergence and strong weights onto its own area.</td></tr>
<tr><td class="k">V1 and A1</td><td class="d">About half of each area fires for a given item; every item decodes at 1.0.</td></tr>
<tr><td class="k">association</td><td class="d">400 excitatory, 100 inhibitory cells; about nine afferents per sense per cell, needing coincident input.</td></tr>
<tr><td class="k">plasticity</td><td class="d">Feedforward frozen; recurrence under the pair rule with soft bounds and a low ceiling.</td></tr>
<tr><td class="k">inhibition</td><td class="d">Held to 0.5 Hz by the Vogels rule.</td></tr>
<tr><td class="k">mechanism</td><td class="d">Paired cells fire together and potentiate (Hebb 1949; Litwin-Kumar and Doiron 2014; Zenke, Agnes and Gerstner 2015).</td></tr>
<tr><td class="k">measure</td><td class="d">Whether either sense alone evokes the other's pattern: cross classification (Kaplan, Man and Greening 2015).</td></tr>
<tr><td class="k">background drive</td><td class="d">None; a probe evokes the stimulus pattern and nothing else.</td></tr>
<tr><td class="k">timing</td><td class="d">250 ms presentations, 250 ms gaps; sight lags the tone by 10 ms.</td></tr>
</table>
<h2>the weave</h2>
<p>Two codes woven through one tissue, told apart only by which cells an input touches.</p>
<table>
<tr><td class="k">tissue</td><td class="d">800 excitatory and 200 fast-spiking cells in one box; no sheets, areas or hierarchy.</td></tr>
<tr><td class="k">codes</td><td class="d">Two noise fields each pick a scattered fifth of the excitatory cells; channel A drives one, B the other.</td></tr>
<tr><td class="k">mappings</td><td class="d">A bins its cells in two dimensions, B in one; B arrives 10 ms after A.</td></tr>
<tr><td class="k">overlap</td><td class="d">About a twentieth; almost no conjunctive cells.</td></tr>
<tr><td class="k">readout</td><td class="d">Cross classification of one channel against the other channel's centroids.</td></tr>
<tr><td class="k">plasticity</td><td class="d">Recurrence only, dense and flat, triplet and short-term terms on (Pokorny et al. 2019).</td></tr>
<tr><td class="k">curriculum</td><td class="d">Blocks: each channel alone, then both; the one-channel probes run through every block.</td></tr>
</table>
<h2>the weave, letters</h2>
<p>The weave with a letter for the spot and its spoken name for the tone.</p>
<table>
<tr><td class="k">tissue</td><td class="d">1,600 excitatory and 400 fast-spiking cells; recurrence the only plastic pathway.</td></tr>
<tr><td class="k">channel A</td><td class="d">The letter through a retina reporting center-surround contrast.</td></tr>
<tr><td class="k">channel B</td><td class="d">The name as formant structure with a consonant onset.</td></tr>
<tr><td class="k">contrast coding</td><td class="d">Drops the mean pairwise similarity of the 26 items from 0.58 to 0.37 (letters school grid).</td></tr>
<tr><td class="k">lesson set</td><td class="d">The five vowels, a control on the curriculum node.</td></tr>
<tr><td class="k">channel size</td><td class="d">About thirty cells each, from a scattered third of the excitatory cells.</td></tr>
<tr><td class="k">timing</td><td class="d">500 ms on, 500 ms off.</td></tr>
<tr><td class="k">drive</td><td class="d">A quarter of the abstract scene's.</td></tr>
<tr><td class="k">inhibitory set point</td><td class="d">1 Hz; the abstract scene uses 3.</td></tr>
<tr><td class="k">measured 2026-09-03</td><td class="d">Over fifteen simulated minutes, selectivity fell from 0.49 to 0.24 at 3 Hz and rose from 0.46 to 0.73 at 1 Hz.</td></tr>
<tr><td class="k">consolidation</td><td class="d">On; commit weight 1.2, committing after five simulated minutes.</td></tr>
</table>
<h2>mandelbulb</h2>
<p>A bulb of bulbs: twelve bulbs on two tiers around a core, from three shapes and two repeats.</p>
<table>
<tr><td class="k">bulb</td><td class="d">A sphere of 1,200 excitatory and 300 fast-spiking cells with a bud of 300 more.</td></tr>
<tr><td class="k">first repeat</td><td class="d">Six bulbs around the core.</td></tr>
<tr><td class="k">second repeat</td><td class="d">The ring again, smaller by 0.62, turned 30 degrees, raised 760 µm.</td></tr>
<tr><td class="k">tags</td><td class="d"><code>tier1.bulb1.RS</code> through <code>tier2.bulb6.bud</code>.</td></tr>
<tr><td class="k">core</td><td class="d">5,000 excitatory and 1,250 fast-spiking cells.</td></tr>
<tr><td class="k">surface</td><td class="d">A noise warp of 220 µm at a 380 µm feature size.</td></tr>
<tr><td class="k">inputs</td><td class="d">Six in the lower tier: sight in odd bulbs, sound in even, no retina or cochlea between.</td></tr>
<tr><td class="k">channels</td><td class="d">Eight per cell, contrast coded, no microsaccade; a channel keeps its cells for the whole run.</td></tr>
<tr><td class="k">bulb to bud</td><td class="d">24 contacts per source.</td></tr>
<tr><td class="k">lower to upper</td><td class="d">48 contacts per source, joining copy k to copy k plus one across the tier repeat.</td></tr>
<tr><td class="k">projections</td><td class="d">Project nodes made once inside the bulb and carried by the repeats.</td></tr>
<tr><td class="k">upper to core</td><td class="d">By proximity, 260 µm kernel; each bulb lands on the nearest face, six senses in six sectors.</td></tr>
<tr><td class="k">recurrence</td><td class="d">Local, 180 µm kernel, never between bulbs.</td></tr>
<tr><td class="k">class-key table</td><td class="d">E to E plastic; E to I and I to I frozen; I to E under the inhibitory rule.</td></tr>
<tr><td class="k">curriculum</td><td class="d">The vowels, one second on and one off; a unimodal probe every fifth presentation.</td></tr>
<tr><td class="k">probes</td><td class="d">Core, one sight bulb, one sound bulb, one upper bulb, one bud; the chart plots the core decode.</td></tr>
</table>
<h2>PD microcircuit</h2>
<p>The cortical microcircuit of Potjans and Diesmann (2014) at their own scale, untuned.</p>
<table>
<tr><td class="k">populations</td><td class="d">Eight, at their sizes.</td></tr>
<tr><td class="k">cells</td><td class="d">77,169 under a square millimeter.</td></tr>
<tr><td class="k">synapses</td><td class="d">About 300 million; raise the synapse budget and allow a few minutes to wire.</td></tr>
<tr><td class="k">parameters</td><td class="d">Their connection probabilities, synaptic strengths, delays and background; every number theirs.</td></tr>
<tr><td class="k">cell type</td><td class="d">A row matched to their integrate and fire neuron.</td></tr>
<tr><td class="k">record</td><td class="d">The replication in RESULTS.md.</td></tr>
</table>
<h2>alphabet school 2</h2>
<p>The training scenario: one laminar block of cortex at mouse density.</p>
<table>
<tr><td class="k">territories</td><td class="d">Visual, association and auditory, in the plane, each with the verified Potjans and Diesmann layers.</td></tr>
<tr><td class="k">beneath</td><td class="d">Thalamic relays and sensory surfaces.</td></tr>
<tr><td class="k">senses</td><td class="d">Segregated until the association territory.</td></tr>
<tr><td class="k">timing</td><td class="d">Sound leads sight by 15 ms through a weaker projection.</td></tr>
<tr><td class="k">plasticity</td><td class="d">Gated to the two convergent projections and the association territory's layer 2/3 inhibition.</td></tr>
<tr><td class="k">that inhibition</td><td class="d">Held toward a low fixed target.</td></tr>
<tr><td class="k">bring-up</td><td class="d">The spots-and-tones set, visual map floored at 32 cells across.</td></tr>
<tr><td class="k">blockamp, blockdrive</td><td class="d">On the training page; scale the sensory drive and the per-layer background.</td></tr>
</table>` },
  { id:'scaling', title:'scaling & proxy', html:`
<h1>scaling and the resolution slider</h1>
<p>A cubic millimeter of mouse cortex holds about a hundred thousand neurons and a billion synapses, which is a lot to run in a browser. The resolution slider lets you work with a smaller version of a scene that still behaves much like the full one.</p>
<p>At 40%, for example, every neuron scatter places 40% of its neurons. Each neuron then receives fewer connections, so linen makes up for it the way van Albada, Helias and Diesmann (2015) describe. The remaining weights grow by 1/&radic;0.4, which keeps the size of each neuron's input fluctuations, and a small steady current restores each neuron's average input, based on the firing rates measured at full size. Your own settings are never changed, and going back to a resolution you've already visited is instant.</p>
<p>Average firing rates and the overall pattern of activity carry over well. Correlations between neurons don't, because they depend on the size of the network itself, so synchrony and the strength of oscillations are only approximate below 100%. That's also why training only runs at full size.</p>
<p>linen also caps how many synapses it will build, based on the memory your machine reports and never below 50 million, and SYN in the top bar lets you change that cap. On the connect node, density % of real tells linen how much sparser your graph already is than real tissue, and target rates let you choose the firing rates the compensation aims for instead of measuring them.</p>` },
  { id:'headless', title:'without the browser', html:`
<h1>running a scene without the browser</h1>
<p>Runs a scene from a shell and writes what happened into a folder.</p>
<h2>the tool</h2>
<p><i>tools/linen.mjs</i>, in the repository (<a href="https://github.com/sethgjosephson/linen">github.com/sethgjosephson/linen</a>).</p>
<p>Takes a scene file or a Tab menu scene's name, computes it, and runs it on the reference or CUDA engine.</p>
<p>Drives an input node fed by a test signal on simulated time.</p>
<p>Leaves an input fed by a curriculum, a device or footage undriven, and reports it.</p>
<h2>commands</h2>
<pre>node tools/linen.mjs run "mouse cortical column" --seconds 10 --out runs/col</pre>
<p>Ten simulated seconds of the column, written to runs/col.</p>
<pre>node tools/linen.mjs run scenes/my-scene.json --seconds 60 --engine cuda --tick 20</pre>
<p>A scene file on the CUDA engine, 20 ms per tick.</p>
<pre>node tools/linen.mjs run "balanced random net" --over "connect:*:wInh=2.5"</pre>
<p>The balanced net with wInh at 2.5 on every connect node.</p>
<h2>the options</h2>
<table>
<tr><td class="k">--seconds</td><td class="d">Simulated time.</td></tr>
<tr><td class="k">--res</td><td class="d">Resolution, between 0 and 1.</td></tr>
<tr><td class="k">--seed</td><td class="d">The seed of every engine's noise and initial state.</td></tr>
<tr><td class="k">--bin</td><td class="d">Width of a rate bin, in milliseconds.</td></tr>
<tr><td class="k">--tick</td><td class="d">Simulated time per tick; also the resolution of the written spike times.</td></tr>
<tr><td class="k">--engine</td><td class="d">cpu, cuda or remote.</td></tr>
<tr><td class="k">--host</td><td class="d">The remote engine's address, ws://localhost:8801 unless given: an engine host, or a custom engine.</td></tr>
<tr><td class="k">--cells</td><td class="d">How many cells' spike trains to write.</td></tr>
<tr><td class="k">--out</td><td class="d">The folder to write into.</td></tr>
<tr><td class="k">--over</td><td class="d"><i>type:tag-or-name:key=value</i>; semicolons separate several; repeatable; * matches every node of the type.</td></tr>
</table>
<p>Reads every setting from the scene's nodes; a launcher may press start, never hold settings.</p>
<p>Stops with an error on a setting no node has; the sweep tab and training dialog follow the same rule.</p>
<h2>what it writes</h2>
<table>
<tr><td class="k">run.json</td><td class="d">The scene, the overridden settings, the size, the engine, the commit, the versions, the wall time.</td></tr>
<tr><td class="k">rates.csv</td><td class="d">One row per bin, one column per population, in Hz.</td></tr>
<tr><td class="k">release.csv</td><td class="d">One row per bin, one column per graded population, mean release in percent, read as a probe reads it.</td></tr>
<tr><td class="k">spikes.txt</td><td class="d">Spike times in milliseconds, one line per cell, the layout Neo's AsciiSpikeTrainIO reads.</td></tr>
<tr><td class="k">cells.csv</td><td class="d">Each line's cell: population, the node that made it, its index there, cell type, position.</td></tr>
<tr><td class="k">settings.json</td><td class="d">Every node in the computed graph with the settings it ran; repeats the run exactly.</td></tr>
<tr><td class="k">tools/linen_read.py</td><td class="d">Loads a folder into pandas DataFrames and Neo SpikeTrains by population.</td></tr>
<tr><td class="k">linen_read.py on a folder</td><td class="d">Prints a summary and the standard measures, with Elephant where Elephant has them.</td></tr>
</table>
<h2>a worked example</h2>
<p>The battery's balanced random net check, outside the browser: full density, two seconds, reference engine.</p>
<pre>node tools/linen.mjs run "balanced random net" --seconds 2 --out runs/brn</pre>
<table>
<tr><td class="k">excitatory band</td><td class="d">1 to 30 Hz (Brunel 2000).</td></tr>
<tr><td class="k">inhibitory band</td><td class="d">1 to 40 Hz.</td></tr>
<tr><td class="k">output</td><td class="d">Each population's rate as it finishes, and the folder beside it.</td></tr>
</table>
<h2>which engine</h2>
<table>
<tr><td class="k">cpu</td><td class="d">Runs in the same process; the reference every other engine is checked against.</td></tr>
<tr><td class="k">cuda</td><td class="d">A separate program, <i>cuda/engine.exe</i> built from <i>cuda/engine.cu</i>, spoken to over a pipe; a tick costs a message.</td></tr>
<tr><td class="k">large, coarse tick</td><td class="d">CUDA pays for itself.</td></tr>
<tr><td class="k">small, 1 ms tick</td><td class="d">The CPU is faster.</td></tr>
<tr><td class="k">agreement</td><td class="d">Up to floating point summation order; spike counts close, not identical.</td></tr>
</table>` },
  { id:'own', title:'custom nodes and engines', html:`
<h1>custom nodes and engines</h1>
<p>linen takes nodes and engines you write.</p>
<h2>a node</h2>
<p>A JavaScript module in the project's <i>nodes</i> folder. The object it receives carries <i>apiVersion</i>, 1 today, so a module written for a later interface can say so instead of failing partway.</p>
<pre>export default function(linen){
  linen.registerNode('shiftx', {
    title:'shift x', cat:'arrange', color:'hsl(270,75%,65%)', inputs:1,
    params:[ { k:'dx', label:'shift µm', t:'float', def:10 } ],
    compute(ins, p){
      const out = linen.clonePts(linen.need(ins[0], 'points', 'shift x needs points'));
      for(let i = 0; i &lt; out.count; i++) out.pos[i*3] += p.dx;
      return out;
    },
    doc:{ io:'points in, points out', blurb:'Moves every cell along x by a set distance.',
      params:{ dx:'The distance, micrometers.' } } });
}</pre>
<table>
<tr><td class="k">install</td><td class="d">Save it in <i>nodes</i>, or use <i>add a node module from a URL</i> in the project menu.</td></tr>
<tr><td class="k">example</td><td class="d"><i>examples/nodes/jitter.js</i>, served here under that path.</td></tr>
<tr><td class="k">appears in</td><td class="d">The Tab menu, the properties panel and the node reference.</td></tr>
<tr><td class="k">compute</td><td class="d">Reads its inputs and settings only.</td></tr>
<tr><td class="k">randomness</td><td class="d"><i>pairHash</i>, <i>pairGauss</i> or <i>rng</i>; never Math.random.</td></tr>
<tr><td class="k">scenes</td><td class="d">Record the modules they use and do not open without them.</td></tr>
</table>
<pre>node tools/linen.mjs run examples/scenes/jitter.json --seconds 1</pre>
<p>Runs the example scene for one simulated second.</p>
<h2>an engine</h2>
<p>Any program that listens on a WebSocket and speaks the engine messages.</p>
<pre>python tools/engine_template.py 8890
node tools/linen.mjs run "guided tour" --engine remote --host ws://localhost:8890
node tools/enginecheck.mjs ws://localhost:8890</pre>
<table>
<tr><td class="k">engine_template.py</td><td class="d">An engine to copy: a leaky integrate-and-fire network in NumPy, about 200 lines.</td></tr>
<tr><td class="k">in the app</td><td class="d">Set the checkpoint's <i>engine</i> to remote and its <i>engine address</i> to where yours listens.</td></tr>
<tr><td class="k">engine settings node</td><td class="d">Settings the checkpoint lacks, one key and value per line; they arrive as the <i>custom</i> block.</td></tr>
<tr><td class="k">enginecheck</td><td class="d">Asks hello, runs a small network on your engine and the reference, prints rate, ISI CV and synchrony.</td></tr>
<tr><td class="k">every message</td><td class="d">ENGINE.md section 15 (<a href="https://github.com/sethgjosephson/linen">github.com/sethgjosephson/linen</a>).</td></tr>
</table>` },
  { id:'limits', title:'limitations', html:`
<h1>limitations</h1>
<p>The equations are in <a href="https://github.com/sethgjosephson/linen/blob/main/MODEL.md">MODEL.md</a> in the repository.</p>
<table>
<tr><th>left out</th><th>what it costs</th><th>where measured</th></tr>
<tr><td class="k">dendrites</td><td class="d">Every neuron is a point: input sums linearly at the soma, with no cable filtering and no local nonlinearity. A cell answers the sum of its inputs, not their order or place. A two-pulse timing test a dendrite would pass, a point neuron fails.</td><td>MODEL.md 1; the fly direction-selectivity result</td></tr>
<tr><td class="k">ion channels</td><td class="d">The membrane is the Izhikevich 2007 fit: a quadratic in the potential, a linear recovery variable, a capacitance, rest and threshold per type. It reproduces firing patterns, not channels: no channel pharmacology, no temperature dependence, no spike shape. Measured rows carry chapter 8's currents and capacitances; classic rows are the 2003 presets in the same form.</td><td>MODEL.md 1</td></tr>
<tr><td class="k">steps below 1 ms</td><td class="d">Every engine advances in whole milliseconds: two membrane half steps, synaptic current held over the step. On the column against 0.05 ms on Brian 2: about 7% on the mean rate, up to 14% on one population. Excitatory populations run a little fast, inhibitory ones 5 to 11% slow, interval variability a few to 15% low. Classic rows: up to 80%. The exact convention on the checkpoint does not reduce it; the residual is the membrane of the fastest-firing populations, and only a smaller step would. Timing below a millisecond is not represented.</td><td>MODEL.md, what the one millisecond step costs; tools/steperr.mjs</td></tr>
<tr><td class="k">delays past 16 ms</td><td class="d">A delay is distance over axon velocity, rounded to a whole millisecond and clamped to 1 to 16. A path past 1.6 mm at 100 &micro;m/ms cannot carry its own delay. The status line reports the count clamped.</td><td>MODEL.md 2, delays</td></tr>
<tr><td class="k">receptor kinetics</td><td class="d">A spike delivers one exponential per channel, as a current or a conductance toward a reversal. No NMDA voltage dependence, no rise time. A kick synapse is a current for one step, its meaning tied to the millisecond.</td><td>MODEL.md 2, synapses</td></tr>
<tr><td class="k">plasticity mechanisms</td><td class="d">Every rule is a trace rule from its published source: pair and triplet STDP, heterosynaptic and transmitter terms, inhibitory plasticity toward a target rate, short-term plasticity, scaling, consolidation. None models receptors, calcium or proteins; none acts on the axonal delay. Synapses form at wiring time from distance and never form or prune during a run.</td><td>MODEL.md 4; MODEL.md, what is not modeled</td></tr>
<tr><td class="k">glia, neuromodulation, gap junctions</td><td class="d">No astrocytes, no modulatory signal changing a cell's or a rule's parameters during a run, no electrical coupling. Input arrives through a chemical synapse or a current node.</td><td>MODEL.md, what is not modeled</td></tr>
<tr><td class="k">external synapses</td><td class="d">Background is a constant current plus per-millisecond noise drawn per cell from a seeded counter, identical in all three engines. It stands in for thousands of external synapses, with no synaptic time course and no correlation between cells.</td><td>MODEL.md 2, noise; the column's per-population drive</td></tr>
<tr><td class="k">correlations below 100%</td><td class="d">Fewer cells, each cell's input mean and variance restored (van Albada, Helias and Diesmann 2015). Rates and the activity regime survive; correlations, synchrony and oscillation amplitudes do not. Training below 100 percent is refused.</td><td>MODEL.md 3, proxy downscaling; the scaling page</td></tr>
<tr><td class="k">bitwise agreement</td><td class="d">The reference, WebGPU and CUDA engines run the same mechanisms under one contract, statistically equivalent, each with its own summation order. The reference matches a second implementation on Brian 2 spike for spike. CUDA reports first-spike latency within a tick; WebGPU refuses that request. A step below 1 ms exists only in the Brian reference, for measurement.</td><td>ENGINE.md; the battery's cross-engine group</td></tr>
</table>` },
  { id:'references', title:'references', html:`
<h1>references</h1>
<h2>simulation methods</h2>
<p>Izhikevich, E. M. (2007). Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting. MIT Press, chapter 8. The membrane every engine runs, and the measured rows RS07 to RZ07.<br>
<a href="https://www.izhikevich.org/publications/dsn/index.htm">izhikevich.org/publications/dsn</a></p>
<p>Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks. The seven classic presets.<br>
<a href="https://www.izhikevich.org/publications/spikes.htm">izhikevich.org/publications/spikes.htm</a></p>
<p>NEST spatially structured networks. Gaussian distance kernels and masks for the connect node.<br>
<a href="https://nest-simulator.readthedocs.io/en/stable/tutorials/pynest_tutorial/part_4_spatially_structured_networks.html">nest-simulator.readthedocs.io</a></p>
<p>Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. Journal of Computational Neuroscience. The balanced net's regimes.</p>
<p>van Albada, S. J., Helias, M., Diesmann, M. (2015). Scalability of asynchronous networks is limited by one-to-one mapping between effective connectivity and correlations. PLOS Computational Biology. The resolution slider.<br>
<a href="https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004490">journals.plos.org</a></p>
<p>Song, S., Sjostrom, P. J., Reigl, M., Nelson, S., Chklovskii, D. B. (2005). Highly nonrandom features of synaptic connectivity in local cortical circuits. PLOS Biology. Lognormal weights.</p>
<p>Perin, R., Berger, T. K., Markram, H. (2011). A synaptic organizing principle for cortical neuronal groups. PNAS. Common-neighbor clustering.</p>
<p>Potjans, T. C., Diesmann, M. (2014). The cell-type specific cortical microcircuit. Cerebral Cortex. Layer population proportions.</p>
<p>Bi, G., Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. Journal of Neuroscience. The STDP window.</p>
<p>Song, S., Miller, K. D., Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. Nature Neuroscience. The online trace formulation.</p>
<p>Vogels, T. P., et al. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. Science. The homeostasis rule.</p>
<p>van Rossum, M. C. W., Bi, G. Q., Turrigiano, G. G. (2000). Stable Hebbian learning from spike timing-dependent plasticity. Journal of Neuroscience. Weight-dependent depression on the checkpoint node.</p>
<p>Gutig, R., Aharonov, R., Rotter, S., Sompolinsky, H. (2003). Learning input correlations through nonlinear temporally asymmetric Hebbian plasticity. Journal of Neuroscience. Weight-dependent depression on the checkpoint node.</p>
<h2>cortical statistics</h2>
<p>Schuz, A., Palm, G. (1989). Density of neurons and synapses in the cerebral cortex of the mouse. Journal of Comparative Neurology 286(4), 442-455. The density target: 9.2 x 10^4 neurons and 7.2 x 10^8 synapses per mm3, 8,200 synapses per neuron, 11 percent type II (mouse areas 8, 6 and 17).<br>
<a href="https://onlinelibrary.wiley.com/doi/10.1002/cne.902860404">onlinelibrary.wiley.com</a></p>
<p>Levy, R. B., Reyes, A. D. (2012). Spatial profile of excitatory and inhibitory synaptic connectivity in mouse primary auditory cortex. Journal of Neuroscience 32(16), 5609-5619. Gaussian fits of connection probability against distance: 114 um pyramid to pyramid, 92 to fast-spiking, 103 to non-fast-spiking, 95 and 85 back onto pyramids.<br>
<a href="https://www.jneurosci.org/content/32/16/5609">jneurosci.org</a></p>
<p>de Kock, C. P. J., Sakmann, B. (2009). Spiking in primary somatosensory cortex during natural whisking in awake head-restrained rats is cell-type specific. PNAS. Layer-specific spontaneous rates (sparse L2/3, higher L5), the background drive's target.</p>
<p>Bartol, T. M., et al. (2015). Nanoconnectomic upper bound on the variability of synaptic plasticity. eLife. About 4.7 bits of distinguishable synaptic state; the basis for f16 weight storage.</p>
<p>Muller, L., Chavane, F., Reynolds, J., Sejnowski, T. J. (2018). Cortical travelling waves: mechanisms and computational principles. Nature Reviews Neuroscience. Background for the default scene.</p>
<p>Gray, C. M., McCormick, D. A. (1996). Chattering cells. Science. The CH preset of the default scene's pacemaker.</p>
<p>Chervin, R. D., Pierce, P. A., Connors, B. W. (1988). Journal of Neurophysiology. Epileptiform discharge in cortical slices at 1-10 cm/s: the default scene's regime and speed band.</p>
<p>Golomb, D., Amitai, Y. (1997). Journal of Neurophysiology. Epileptiform discharge in cortical slices at 1-10 cm/s: the default scene's regime and speed band.</p>
<h2>cortical columns and dendrites</h2>
<p>Harris, K. D., Shepherd, G. M. G. (2015). The neocortical circuit: themes and variations. Nature Neuroscience 18(2), 170-181. Layer order, three excitatory projection classes, three interneuron groups; the column scene has the order and a two-way interneuron split.<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/">pmc.ncbi.nlm.nih.gov</a></p>
<p>Hawkins, J., Ahmad, S. (2016). Why neurons have thousands of synapses, a theory of sequence memory in neocortex. Frontiers in Neural Circuits 10:23. Background; not modeled.<br>
<a href="https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2016.00023/full">frontiersin.org</a></p>
<p>Hawkins, J., Ahmad, S., Cui, Y. (2017). A theory of how columns in the neocortex enable learning the structure of the world. Frontiers in Neural Circuits 11:81. Background; not modeled.<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5661005/">pmc.ncbi.nlm.nih.gov</a></p>
<p>Hawkins, J., Lewis, M., Klukas, M., Purdy, S., Ahmad, S. (2019). A framework for intelligence and cortical function based on grid cells in the neocortex. Frontiers in Neural Circuits 12:121. Background; not modeled.<br>
<a href="https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00121/full">frontiersin.org</a></p>
<p>Hawkins, J., Leadholm, N., Clay, V. (2025). Hierarchy or heterarchy? A theory of long-range connections for the sensorimotor brain. arXiv:2507.05888. Background; not modeled.<br>
<a href="https://arxiv.org/abs/2507.05888">arxiv.org</a></p>
<p>Haueis, P. (2016). The life of the cortical column: opening the domain of functional architecture of the cortex (1955 to 1981). History and Philosophy of the Life Sciences 38:2. The history of the column concept; the column scene is a column by geometry only.<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4914527/">pmc.ncbi.nlm.nih.gov</a></p>
<p>Wright, W. J., Hedrick, N. G., Komiyama, T. (2025). Distinct synaptic plasticity rules operate across dendritic compartments in vivo during learning. Science 388(6744), 322-328. Apical and basal rules in layer 2/3 of mouse motor cortex; the rules here are the basal, spike-timed kind.<br>
<a href="https://www.science.org/doi/10.1126/science.ads4706">science.org</a></p>
<h2>related tools</h2>
<table>
<tr><td class="k">NEST</td><td class="d">Gewaltig, M.-O., Diesmann, M. 2007, Scholarpedia 2(4):1430.</td></tr>
<tr><td class="k">Brian 2</td><td class="d">Stimberg, M., Brette, R., Goodman, D. F. M. 2019, eLife 8:e47314.</td></tr>
<tr><td class="k">NEST Desktop</td><td class="d">Spreizer et al. 2021, eNeuro.</td></tr>
<tr><td class="k">NetPyNE</td><td class="d">Dura-Bernal et al. 2019, eLife.</td></tr>
<tr><td class="k">BrainX3</td><td class="d">Arsiwalla et al. 2015, Frontiers in Neuroinformatics.</td></tr>
<tr><td class="k">The Virtual Brain</td><td class="d">On EBRAINS; Schirner et al. 2022, NeuroImage.</td></tr>
<tr><td class="k">Neuronify</td><td class="d">Dragly et al. 2017, eNeuro.</td></tr>
</table>
<h2>performance references</h2>
<p>Fidjeland, A. K., Shanahan, M. P. (2010). Accelerated simulation of spiking neural networks using GPUs. IJCNN.</p>
<p>Golosio, B., et al. (2021). Fast simulations of highly-connected spiking cortical models using GPUs. Frontiers in Computational Neuroscience.</p>` },
];

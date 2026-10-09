// Sample scenarios, loadable from the Tab menu.
// All spatial units are micrometers (µm); axon velocities in µm/ms (100 µm/ms = 0.1 m/s, unmyelinated intracortical range).
// Proportions and wiring logic follow the sources:
//  - cortical column: layer sizes / E-I ratios after Potjans & Diesmann (2014)
//    microcircuit proportions; L5 pyramids as intrinsic bursters, L2/3
//    chattering fraction, FS/LTS interneuron split per layer
//  - thalamocortical loop: TC relay + RZ resonator reticular shell
//    (Izhikevich 2003 types), pulsed sensory drive
//  - CA3 pattern completion: recurrent pyramidal band (15% bursting), a
//    weak cue to one end recruits the rest through the recurrent
//    collaterals; short-term depression is what keeps it bounded
//  - balanced random network: Brunel (2000) style 4:1 E/I with strong
//    inhibition, noise-driven asynchronous irregular firing
import { nameNodes } from './nodenames.js';
import { NEURON_TYPES } from './nodes.js';
export const RS=0, IB=1, CH=2, FS=3, LTS=4, TC=5, RZ=6;
// The measured rows of Izhikevich 2007 chapter 8, by key rather than by number, since the table grows: a scene on these carries its currents in picoamps and its weights in picoamps of synaptic current (MODEL.md 1).
const rowOf = key => NEURON_TYPES.findIndex(t => t.key === key);
export const RS07 = rowOf('RS07'), IB07 = rowOf('IB07'), CH07 = rowOf('CH07'),
  FS07 = rowOf('FS07'), LTS07 = rowOf('LTS07'), TC07 = rowOf('TC07'), RZ07 = rowOf('RZ07');

// Units for the exp-synapse scenes.
// Their checkpoints declare psc 1 (w is the total charge, exactly) and every quantity in weight units carries the factor 1/(tau(1 - e^(-1/tau))), so each scene delivers the charge it was measured with and the tuned numbers stay visible.
// EQ is the excitatory factor at tauE 3 and IQ the inhibitory one at tauI 8, which every one of these scenes uses.
// Weight-unit quantities scale (wExc, project weights, A+, the triplet and transmitter amplitudes, wmax, the inhibitory rate eta); multiplicative or dimensionless ones do not (A- under soft bounds, beta, the scaling rate, targets).
export const EQ = 1/(3*(1 - Math.exp(-1/3)));    // 1.1757
export const IQ = 1/(8*(1 - Math.exp(-1/8)));    // 1.0638
export function api(ed){
  // name: what the node is for, shown beside it in the graph.
  // A node left without one is named from its place in the graph when the scenario is opened (nodenames.js), so give one here when that would say too little.
  const add = (t, x, y, params={}, name='') => { const n = ed.addNode(t, x, y);
    if(name) n.name = name;
    // a scatter given a count is filled by count; the node's default is a density
    if(t === 'scatter' && params.count !== undefined && params.fill === undefined) params = { ...params, fill:0 };
    Object.assign(n.params, params); return n; };
  const wire = (dst, port, src) => {
    if(port >= dst.inputs.length)
      throw new Error(`scenario bug: ${dst.type} has ${dst.inputs.length} inputs, wired port ${port}`);
    dst.inputs[port] = { id: src.id };
  };
  // Notes sit in a column to the left of every scenario, clear of the graph itself (which starts at x 40 or beyond), so a loaded scene explains what it is without anyone having to read the source.
  const note = (y, text, width=340, near='') => add('note', -380, y, { text, width, near });
  // A labeled backdrop behind a set of nodes.
  // Purely a reading aid: it names a cluster the scenario already knows the meaning of, so the four nodes that build one cortical layer of one territory read as 'v.L4' rather than as four unlabeled boxes among sixty.
  const group = (label, nodes, hue) => {
    ed.groups = ed.groups || [];
    ed.groups.push({ label, hue,
      // a layer without a bursting fraction returns the same node for its excitatory in and out, so the same id can be listed twice
      members: [...new Set(nodes.filter(Boolean).map(n => n.id))] });
    return nodes;
  };
  return { add, wire, note, group };
}

// Potjans-Diesmann 2014 connection probabilities (rows = target, columns = source; population order L2/3e L2/3i L4e L4i L5e L5i L6e L6i).
// Verified against the reference implementation's network_params.py.
const PD_CONN = [
  [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0,    0.0076, 0.0],
  [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0,    0.0042, 0.0],
  [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0],
  [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0,    0.1057, 0.0],
  [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
  [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0],
  [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
  [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443],
];
const PD_TAGS = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i'];
// external in-degrees K_bg per population (same source), layer means used to scale the background noise drives
export function pdTable(){
  const lines = [];
  for(let t=0;t<8;t++) for(let s=0;s<8;s++){
    const wm = (s === 2 && t === 0) ? ' 2' : '';   // L4e -> L2/3e doubled weight
    lines.push(PD_TAGS[s] + ' ' + PD_TAGS[t] + ' ' + (PD_CONN[t][s]/0.05).toFixed(3) + wm);
  }
  return lines.join('\n');
}

export function corticalColumn(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Mouse cortical column\n\nFour layer bands are stacked at their real cortical depths and filled at the population proportions of the reference microcircuit (Potjans and Diesmann 2014). Layer 5 pyramids are intrinsically bursting, a tenth of layer 2/3 chatters, and every layer splits its interneurons between fast spiking and low threshold spiking. Every cell is one of the measured rows of Izhikevich 2007 chapter 8, so it carries a capacitance in picofarads and a rest in millivolts, and every current in the scene is in picoamps.');
  note(170, 'The connect node carries the 8 by 8 population pair table from the same paper, so every layer to layer connection probability is the published one. A pulse train drives layer 4, the input layer, through a geometry mask, as two nodes: the same current is five times the voltage step on a 20 pF interneuron that it is on a 100 pF pyramid, so the pulse is set per cell class.', 340, 'local wiring');
  note(320, 'The background drive is tuned layer by layer. With the thalamic pulse off, the spontaneous rates sit inside the bands that de Kock and Sakmann (2009) recorded: layer 2/3 excitatory cells at 0.67 Hz (band 0.2 to 1), layer 4 at 2.20 (0.5 to 3), layer 5 at 2.76 (2 to 4) and layer 6 at 1.22 (0.3 to 2). Every inhibitory population fires faster than its excitatory partner, at 4.22, 7.97, 12.52 and 12.14 Hz, and the population Fano factor per thousand cells is 0.1, which means the firing is asynchronous. The validation battery checks all of this.', 340, 'L2/3 drive');
  // 400x400 µm column, 1.2 mm deep; ~14k neurons at PD-2014 layer proportions
  const layers = [
    { name:'L2/3', c:[0,900,0], s:[400,300,400], e:3750, eType:RS07, chFrac:0.10, i:1060, ltsFrac:0.35 },
    { name:'L4',   c:[0,650,0], s:[400,200,400], e:4000, eType:RS07, chFrac:0,    i: 990, ltsFrac:0.20 },
    { name:'L5',   c:[0,400,0], s:[400,300,400], e: 880, eType:IB07, chFrac:0,    i: 200, ltsFrac:0.45 },
    { name:'L6',   c:[0,100,0], s:[400,300,400], e:2620, eType:RS07, chFrac:0,    i: 530, ltsFrac:0.35 },
  ];
  const streams = [], boxes = []; let l4box = null;
  layers.forEach((L, k) => {
    const x = 40 + k*300;
    const box = add('box', x, 20, { center:L.c, size:L.s });
    boxes.push(box);
    if(L.name === 'L4') l4box = box;
    // pyramidal cells sit in vertical minicolumn strands (Mountcastle 1997, 30 to 60 um pitch); interneurons get soma exclusion instead, which is the mosaic regularity measured in cell-body distributions
    const se = add('scatter', x, 75, { count:L.e, type:L.eType, seed:k*2+1, tag:L.name+'e',
      pattern:1, pitch:45, jitter:9, axis:1 });
    wire(se, 0, box);
    let eOut = se;
    if(L.chFrac){
      eOut = add('celltype', x, 130, { row:(L.chType || CH07) + 1, frac:L.chFrac });
      wire(eOut, 0, se);
    }
    const si = add('scatter', x+150, 75, { count:L.i, type:FS07, seed:k*2+2, tag:L.name+'i',
      spacing:10 });
    wire(si, 0, box);
    const iOut = add('celltype', x+150, 130, { row:(LTS07) + 1, frac:L.ltsFrac });
    wire(iOut, 0, si);
    group(L.name, [box, se, eOut, si, iOut], 205);
    streams.push(eOut, iOut);
  });
  const m1 = add('gather', 190, 210), m2 = add('gather', 490, 265), m3 = add('gather', 640, 320);
  streams.slice(0,4).forEach((s,i) => wire(m1, i, s));
  wire(m2, 0, m1); streams.slice(4,7).forEach((s,i) => wire(m2, i+1, s));
  wire(m3, 0, m2); wire(m3, 1, streams[7]);
  // PD probabilities are distance independent within the column, so the kernel is made effectively flat (sigma far above the column diagonal) and base prob 0.05 makes each table multiplier reproduce the PD value.
  // Weights scaled to the in-degree, and inhibition at 4.67 times excitation: a synapse has to be small relative to the distance to threshold when a neuron has hundreds of inputs.
  // On the measured rows the weights are 70 and -298 pA, a synapse a small fraction of the way to threshold on a 100 pF RS07 pyramid.
  // One current is not one voltage step here.
  // The 20 pF interneurons feel five times the jump the pyramids do from the same 70 pA (a small cell is depolarized more by the same synaptic current), and it is why the drive below is set per population and not per layer.
  const cn = add('connect', 490, 380,
    { radius:1400, sigma:4000, prob:0.05, wExc:70, wInh:-298, velocity:300, seed:1,
      table: pdTable() });
  wire(cn, 0, m3);
  // Background drive per layer.
  // In the reference microcircuit the external input is thousands of excitatory Poisson synapses per neuron, which is a positive mean current plus fluctuations; a zero-mean noise term alone cannot hold an inhibition-dominated column depolarized, and the network falls silent after its onset transient.
  // Each layer therefore gets a constant depolarizing drive and a noise term, both scaled by the PD external in-degrees K_bg (layer means 1550 / 2000 / 1950 / 2500).
  // Drive is population specific: excitatory cells here are RS or IB with strong spike-frequency adaptation while FS interneurons barely adapt, so a drive that is uniform across a layer recruits inhibition faster than excitation and silences the pyramids.
  // Splitting the drive by class (the E and I keys on the stimulus node) is the same thing the reference microcircuit does with its per-population external in-degrees.
  const kbg = [0.775, 1.0, 0.975, 1.25];
  // On the measured rows the drive is a picoamp figure per population rather than a factor times K_bg.
  // The K_bg ratios still set the noise, because fluctuation does come from the number of external synapses, but they cannot set the operating point any more: the rows differ in capacitance (RS07 100 pF, IB07 150, FS07 20) so the same product is a different depolarization in every layer, and L5 needs half again the current of an RS07 layer for the same rate because its pyramids are bursting cells with half again the membrane to charge.
  // Each layer is driven to the band de Kock and Sakmann 2009 report for it, at full density with the pulse off.
  // Per-layer operating point: the K_bg ratios fix the balance inside a layer, but each layer sits in a different amount of recurrent inhibition and one global scale cannot place them all.
  // Spontaneous rates on these drives, measured 2026-09-24 with the pulse off over 11.5 s at full density, against the de Kock and Sakmann 2009 bands: L2/3e 0.67 (0.2 to 1), L4e 2.20 (0.5 to 3), L5e 2.76 (2 to 4), L6e 1.22 (0.3 to 2), every inhibitory population above its excitatory partner at 4.22 / 7.97 / 12.52 / 12.14, L5e the highest excitatory rate, population Fano per thousand 0.1 and an ISI CV of 0.6 to 1.1.
  // The noise term is left alone: it sets fluctuation rather than the operating point, and scaling it with the steady current drops the column straight into silence.
  // The column is bistable, so the margin above these currents is not large: a quarter more layer 2/3 excitatory drive than the tuned value flips the whole column into the synchronized state at 7 to 12 Hz.
  // Layer 5 is the odd one out, at four and a half times as much current on the excitatory side as on the inhibitory one, because its pyramids are IB07 at 150 pF against FS07 at 20.
  const driveE = [210, 225, 540, 330];
  const driveI = [78, 96, 122, 90];
  let prev = cn;
  layers.forEach((L, k) => {
    const de = add('stimulus', 120 + k*150, 440,
      { mode:0, current:driveE[k], tag:'E' });
    wire(de, 0, prev); wire(de, 1, boxes[k]);
    const di = add('stimulus', 120 + k*150, 495,
      { mode:0, current:driveI[k], tag:'I' });
    wire(di, 0, de); wire(di, 1, boxes[k]);
    const ns = add('stimulus', 120 + k*150, 550, { mode:3, current:+(460*kbg[k]).toFixed(0) });
    wire(ns, 0, di); wire(ns, 1, boxes[k]);
    group(L.name + ' drive  E, I, noise', [de, di, ns], 55);
    prev = ns;
  });
  // Thalamic drive into layer 4, the input layer.
  // It has to modulate the activity the recurrent circuit generates rather than dictate it: too strong, every cell in the column fires once per pulse and the whole scene reads the pulse rate rather than a cortical rate.
  // The pulse is two nodes.
  // One current into the layer 4 box is a fifth of the voltage step on a 100 pF pyramid that it is on a 20 pF interneuron, so a single node at the level layer 4 excitatory cells need drives the interneurons five times as hard and the pulse arrives as feedforward inhibition: measured 2026-09-24, one node at 400 pA lifted L4i from 8.1 to 9.3 Hz and pushed L4e down from 2.18 to 1.95.
  // Split by class at the ratio of the capacitances, 80 and 16 pA, the pulse is the same few millivolts on both.
  // At 80 and 16 pA it lifts layer 4 excitatory cells from 2.18 to 2.58 Hz and the column stays asynchronous; at five times that the column locks to the pulse instead, at a population Fano per thousand of 14.
  const th = add('stimulus', 490, 500, { mode:1, current:80, period:100, width:10, tag:'E' });
  wire(th, 0, prev); wire(th, 1, l4box);
  const thI = add('stimulus', 490, 555, { mode:1, current:16, period:100, width:10, tag:'I' });
  wire(thI, 0, th); wire(thI, 1, l4box);
  // One probe per excitatory layer.
  // The result this circuit is known for is a set of per-layer rates, layer 2/3 lowest and layer 5 highest, and that is four numbers rather than a raster: with the populations named, the probes can say which layer each rate belongs to.
  // The excitatory cells only, since the published rates are stated that way.
  let prb = thI;
  const probes = [];
  layers.forEach((L, k) => {
    const pb = add('probe', 40 + k*150, 620, { label:L.name + 'e', tag:L.name + 'e' });
    wire(pb, 0, prb); prb = pb; probes.push(pb);
  });
  const out = add('checkpoint', 490, 690, { steps:2, syn:1, psc:1, tauE:3, tauI:8 });
  wire(out, 0, prb);
  const cha = add('chart', 490, 750, { plot:7, pop:'L4e', window:5, height:130 });
  wire(cha, 0, out);
  group('gather: eight populations into one stream', [m1, m2, m3], 280);
  group('local wiring  Potjans-Diesmann pair table', [cn], 130);
  group('thalamic pulse into L4', [th, thI], 55);
  group('readout  a rate per layer', [...probes, cha], 25);
  group('checkpoint  the simulation', [out], 0);
}

function thalamocortical(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Thalamocortical loop\n\nA relay nucleus and a patch of cortex are wired in both directions through a fiber tract, drawn here as a cylinder. Every synapse is local: the connect radius is 600 um and the patch sits 805 um from the nucleus, so no single synapse spans the gap. A signal crosses the tract in hops, and the length of the tract shows up in how long a trip around the loop takes.');
  note(170, 'The relay cells use the TC preset. It rests at -64.4 mV and fires steadily from a current of about 0.66, a sixth of what an RS cell needs, so the noise and input in the loop are enough to keep it going. The published model pairs the preset with a holding current that keeps the cell below threshold until something drives it, which is where a rebound burst comes from.', 340, 'thalamic relay');
  const cortexBox = add('box', 40, 20, { center:[0,550,0], size:[1000,250,1000] });
  const ce = add('scatter', 40, 75, { fill:1, density:72000, type:RS07, seed:1, tag:'ce' });
  wire(ce, 0, cortexBox);
  const ci = add('scatter', 190, 75, { fill:1, density:18000, type:FS07, seed:2, tag:'ci' });
  wire(ci, 0, cortexBox);
  const ciR = add('celltype', 190, 130, { row:(LTS07) + 1, frac:0.3 }); wire(ciR, 0, ci);
  const tract = add('cylinder', 340, 20, { center:[0,0,0], radius:80, height:900 });
  const tr = add('scatter', 340, 75, { fill:1, density:90000, type:RS07, seed:3, tag:'tract' }); wire(tr, 0, tract);
  // TC kept sparse: relay cells barely excite each other in vivo, and with d=0.05 they have almost no adaptation, so dense TC-TC coupling runs away
  const thal = add('sphere', 490, 20, { center:[0,-600,0], radius:220 });
  const tc = add('scatter', 490, 75, { fill:1, density:90000, type:TC07, seed:4, tag:'tc' });
  wire(tc, 0, thal);
  const shell = add('torus', 640, 20, { center:[0,-600,0], radius:300, thickness:80 });
  const rz = add('scatter', 640, 75, { fill:1, density:90000, type:RZ07, seed:5, tag:'rz' }); wire(rz, 0, shell);
  const m1 = add('gather', 190, 210), m2 = add('gather', 490, 265);
  wire(m1, 0, ce); wire(m1, 1, ciR); wire(m1, 2, tr); wire(m1, 3, tc);
  wire(m2, 0, m1); wire(m2, 1, rz);
  // Exponential synapses and inhibition at 4.67 times excitation: a lower ratio on an in-degree of 71 is net strongly positive, and one-millisecond kicks make every arrival a coincidence and set the whole loop ringing.
  // Velocity 120.
  // At 300 every distance inside the 240 um connect radius is under a millisecond and clamps to the 1 ms floor, so every synapse carries exactly the same delay and the spectrum spike timing needs does not exist anywhere in the scene.
  // At 120 the same radius spans 1 to 2 ms, which is also the right range for a local cortical axon once the synaptic delay is included.
  const cn = add('connect', 490, 330,
    { radius:600, sigma:200, prob:0.15, wExc:50, wInh:-150, wdist:1, velocity:120, seed:1 });
  wire(cn, 0, m2);
  const ns = add('stimulus', 340, 390, { mode:3, current:150, radius:100000, center:[0,0,0] });
  wire(ns, 0, cn);
  // Steady background for the cortical patch, split by class.
  // Fast spiking cells sit above their excitatory neighbors in cortex (Gentet et al. 2010) and receive less recurrent excitation than the pyramids do, so both classes get their own current.
  const dce = add('stimulus', 40, 390, { mode:0, current:80, tag:'ce' });
  wire(dce, 0, ns); wire(dce, 1, cortexBox);
  const dci = add('stimulus', 190, 390, { mode:0, current:10, tag:'ci' });
  wire(dci, 0, dce); wire(dci, 1, cortexBox);
  // The holding current.
  // The TC preset rests at -64.4 mV but fires steadily from a current of about 0.66, so under this scene's noise and input relay cells would free-run above the tonic 5 to 20 Hz awake.
  // A rebound burst has to rebound from somewhere, which is why Izhikevich drives this preset from a hyperpolarized hold.
  const hold = add('stimulus', 490, 390, { mode:0, current:-500, tag:'tc' });
  wire(hold, 0, dci); wire(hold, 1, thal);
  const sens = add('stimulus', 490, 450, { mode:1, current:1500, period:120, width:15 });
  wire(sens, 0, hold); wire(sens, 1, thal);   // sensory pulses into the thalamus
  // The loop as three rates.
  // Cortex drives the reticular shell, the shell inhibits the relay, and the relay drives cortex, so what the scene is about is the relation between these three numbers rather than any one of them.
  // The populations are named, so the probes can be too.
  const pbC = add('probe', 40, 450, { label:'cortex', tag:'ce' });
  wire(pbC, 0, sens);
  const pbT = add('probe', 240, 450, { label:'relay', tag:'tc' });
  wire(pbT, 0, pbC);
  const pbR = add('probe', 440, 450, { label:'reticular', tag:'rz' });
  wire(pbR, 0, pbT);
  const out = add('checkpoint', 490, 510, { steps:2, syn:1, psc:1, tauE:3, tauI:8, wmax:30 });
  wire(out, 0, pbR);
  group('cortical patch  RS + FS/LTS', [cortexBox, ce, ci, ciR], 130);
  group('fiber tract', [tract, tr], 280);
  group('thalamic relay  TC', [thal, tc], 205);
  group('reticular shell  RZ', [shell, rz], 205);
  group('gather: every population into one stream', [m1, m2], 280);
  group('local wiring', [cn], 130);
  const cha = add('chart', 490, 570, { plot:7, pop:'relay', window:5, height:130 });
  wire(cha, 0, out);
  group('background and sensory pulses', [ns, dce, dci, hold, sens], 55);
  group('readout  the three stations of the loop', [pbC, pbT, pbR, cha], 25);
  group('checkpoint  the simulation', [out], 0);
}

function ca3Attractor(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'CA3 pattern completion\n\nA recurrent excitatory population with strong internal connections and global inhibition, which is the classic autoassociative memory circuit. When a cue arrives at one end of the band, the recurrent collaterals recruit the rest of the population.');
  note(190, 'Watch the raster when the cue arrives. It drives a few hundred cells, the rest of the band joins within about a hundred milliseconds, and then everything settles back to its spontaneous rate. At full density the baseline is 0.4 Hz, and in the 100 ms after the cue the cued end reads 38 Hz while the far end, which the cue never touches, reads 24 Hz (measured 2026-09-15). For comparison, CA3 pyramidal cells in an awake animal fire at a median under 1 Hz (Mizuseki and Buzsaki 2013) and reach tens of Hz inside a place field, so the event rates here are in that range and the quiet baseline is below it.', 340, 'readout');
  note(430, 'Three things keep the recall an event and stop it from running away: an absolute refractory period, short-term depression, and enough inhibition. Depression is what makes it robust. With it on, raising the recurrent weight by 64 percent moves the spikes in the event by 18, and the recall is all or none below that: at two thirds of the weight the cue fires its own cells and recruits nothing.', 340, 'recurrent wiring');
  const band = add('spline', 40, 20,
    { p0:[-600,0,-200], p1:[-200,0,300], p2:[300,0,300], p3:[600,0,-300], radius:120 });
  const pyr = add('scatter', 40, 75, { fill:1, density:74000, type:RS07, seed:1, tag:'ca3' }); wire(pyr, 0, band);
  const pyrB = add('celltype', 40, 130, { row:(IB07) + 1, frac:0.15 }); wire(pyrB, 0, pyr);
  const inh = add('scatter', 190, 75, { fill:1, density:16000, type:FS07, seed:2, tag:'ca3i' }); wire(inh, 0, band);
  const inhR = add('celltype', 190, 130, { row:(LTS07) + 1, frac:0.3 }); wire(inhR, 0, inh);
  const m = add('gather', 115, 210); wire(m, 0, pyrB); wire(m, 1, inhR);
  // Inhibition at twice the excitatory weight; a loop that is net positive has only two states, silence and every cell at the refractory ceiling.
  // At density the band holds 7,500 cells against the rat's 300,000 CA3 pyramids, so the recurrent probability is cortex's 0.15 rather than CA3's few percent (Guzman et al. 2016): recall needs the in-degree (245 here), not the probability.
  // On the measured rows the unitary EPSP is 90 pA, 0.9 mV on a 100 pF pyramid, and the depression is the published depressing class at U 0.5 (Markram, Wang and Tsodyks 1998) rather than 0.2: recall is all or none here, and with the weaker depression it ran 200 ms at 37 Hz instead of 200 at 20 (2026-09-22).
  const cn = add('connect', 115, 275,
    { radius:600, sigma:200, prob:0.15, wExc:90, wInh:-180, wdist:1, velocity:100, seed:1 });
  wire(cn, 0, m);
  // Noise, and under it a holding current.
  // It is the holding current that matters and not the baseline: 30 pA fires nothing on its own, the pyramids sit silent between events, and taking it away drops the recall from 8.2 Hz in its 500 ms to 1.8, which is the cue's own cells and nothing recruited (2026-09-22).
  // What a cue completes from is a population close enough to threshold, which is what a holding current is.
  const bg = add('stimulus', 115, 320,
    { mode:3, current:240, radius:100000, center:[0,0,0] });
  wire(bg, 0, cn);
  // A holding current under the noise.
  // On the measured rows a pyramid has a rheobase near 60 pA and noise alone crosses it either never or in a volley: 240 pA of noise leaves the band silent between events and 500 ignites it.
  // The holding current sits the cells under threshold so the noise can trickle them over it, which is what a baseline is (2026-09-22).
  const hold = add('stimulus', 265, 320,
    { mode:0, current:30, tag:'ca3', radius:100000, center:[0,0,0] }); hold.name = 'holding current';
  wire(hold, 0, bg);
  // every 5 s from 500 ms in: cue one end, then a global silencing pulse 2.5 s later.
  // The cue is deliberately weak (10 for 40 ms on one end) so that most of what fires is recruited through the recurrent collaterals rather than driven directly by the electrode; it starts after the onset transient so the first recall is not confused with it.
  const ign = add('stimulus', 40, 340,
    { mode:1, current:1000, period:5000, width:40, t0:500, center:[-600,0,-200], radius:200 });
  wire(ign, 0, hold);
  const qch = add('stimulus', 190, 400,
    { mode:1, current:-2500, period:5000, width:60, t0:2500, center:[0,0,0], radius:100000 }); bg.name = 'background noise'; ign.name = 'cue'; qch.name = 'reset inhibition';
  wire(qch, 0, ign);
  // Exponential synapses, an absolute refractory period, and short-term depression.
  // Depression is the one that matters: it is a gain control on the recurrent loop, and it is what turns a scene that is either silent or saturated into one whose response barely moves when the recurrent weight does. wmax clears the inhibitory weight.
  // Two probes make the completion visible as two numbers rather than as a texture in the raster: the electrode drives the cued end, and the far end can only be reached through the recurrent collaterals, so the gap between when the two rise is the recall.
  const cueEnd = add('sphere', 40, 400, { center:[-600, 0, -200], radius:200 });
  const pbC = add('probe', 40, 455, { label:'cued end', tag:'ca3' });
  wire(pbC, 0, qch); wire(pbC, 1, cueEnd);
  const farEnd = add('sphere', 250, 400, { center:[600, 0, -300], radius:200 });
  const pbF = add('probe', 250, 455, { label:'far end', tag:'ca3' });
  wire(pbF, 0, pbC); wire(pbF, 1, farEnd);
  const out = add('checkpoint', 115, 510, { steps:2,
    syn:1, psc:1, tauE:3, tauI:8, refrac:2, wmax:40,
    stp:1, stpNorm:1, stpU:0.5, stpTauD:200, stpTauF:600 });
  wire(out, 0, pbF);
  const cha = add('chart', 115, 570, { plot:7, pop:'cued end', window:10, height:130 });
  wire(cha, 0, out);
  // the band feeds both populations, so it is left out of either backdrop rather than making one of them reach across the other
  group('CA3 band', [band], 280);
  group('pyramids  RS with a bursting fraction', [pyr, pyrB], 130);
  group('feedback inhibition  FS/LTS', [inh, inhR], 205);
  group('recurrent wiring', [m, cn], 280);
  group('spontaneous activity', [bg], 55);
  group('cue and silence', [ign, qch], 55);
  group('readout  cued against recruited', [cueEnd, pbC, farEnd, pbF, cha], 25);
  group('checkpoint  the simulation', [out], 0);
}

export function balancedNet(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Balanced random network\n\nSparse random connections, inhibition stronger than excitation, and a noise drive. This is the network from Brunel (2000), and it is the usual reference for what healthy cortical background activity looks like.');
  note(170, 'Try raising the inhibitory weight. The network settles into asynchronous irregular firing at a few Hz, which looks like a uniform speckle in the raster. Lower it and the population synchronizes into vertical stripes. The battery measures both regimes, along with the population Fano factor that tells them apart.', 340, 'sparse random wiring');
  const vol = add('sphere', 40, 20, { center:[0,0,0], radius:400 });
  const e = add('scatter', 40, 75, { fill:1, density:72000, type:RS07, seed:1, tag:'exc' }); wire(e, 0, vol);
  const i = add('scatter', 190, 75, { fill:1, density:18000, type:FS07, seed:2, tag:'inh' }); wire(i, 0, vol);
  const m = add('gather', 115, 155); wire(m, 0, e); wire(m, 1, i);
  // velocity 100 spreads conduction delays over 1-2.5 ms; uniform ~1 ms delays entrain a fast global oscillation that regularizes firing.
  // Brunel's wiring is random, not spatial: a kernel far wider than the radius is flat inside it, so every pair within 300 um connects with the same probability, 0.1 as in the paper (in-degree 597 here).
  // J is the unitary EPSP, g = 5 the inhibitory ratio (Brunel 2000).
  //
  // On the measured rows (2026-09-22) a weight is picoamps of synaptic current: 50 pA for one millisecond is 0.5 mV on a 100 pF pyramid, the unitary EPSP this scene was written with, and 2.5 mV on a 20 pF interneuron, which is a difference between the cells and not a choice.
  // At full density it reads 3.2 Hz excitatory and 10.8 inhibitory at g = 5, asynchronous and irregular; at g = 1 it runs at 16.8 Hz with a population Fano of 55 against 2.6, the synchronous regime.
  const cn = add('connect', 115, 220,
    { radius:300, sigma:3000, prob:0.1, wExc:50, wInh:-250, wdist:1, velocity:100, seed:1 });
  wire(cn, 0, m);
  // A background current per population.
  // The two rows differ in capacitance, so the same current is five times the drive on an interneuron, and any level that leaves the interneurons sane silences the pyramids.
  const nsE = add('stimulus', 40, 280, { mode:3, current:1800, tag:'exc', radius:100000, center:[0,0,0] });
  wire(nsE, 0, cn);
  const ns = add('stimulus', 190, 280, { mode:3, current:200, tag:'inh', radius:100000, center:[0,0,0] });
  wire(ns, 0, nsE);
  // The two populations read separately.
  // They share the volume, so a mask cannot tell them apart and the population field on the probe is what does: the excitatory rate is the one Brunel's regimes are stated in, and the inhibitory cells run several times faster, so one probe over both would report an average that is neither.
  const pbE = add('probe', 40, 400, { label:'exc', tag:'exc' });
  wire(pbE, 0, ns);
  const pbI = add('probe', 190, 400, { label:'inh', tag:'inh' });
  wire(pbI, 0, pbE);
  const out = add('checkpoint', 115, 460, { steps:2 });
  wire(out, 0, pbI);
  const cha = add('chart', 115, 520, { plot:7, pop:'exc', window:5, height:130 });
  wire(cha, 0, out);
  group('volume', [vol], 280);
  group('excitatory  8000 RS', [e], 130);
  group('inhibitory  2000 FS', [i], 205);
  group('sparse random wiring and noise drive', [m, cn, ns], 55);
  group('readout  a rate per population', [pbE, pbI, cha], 25);
  group('checkpoint  the simulation', [out], 0);
}

// The guided tour: also the first-boot default scene.
// Three outputs teach the interface; note nodes carry the instructions.
// The sheet tissue the guided tour and traveling waves share, on the measured rows (2026-09-24).
// A front needs tissue that is quiet in front of it and refractory behind it, and the measured rows RS07, FS07 and CH07 give neither at their published numbers.
//
// SHEET_ADAPT is the spike-triggered adaptation of the sheet's regular-spiking row, in picoamps added to the recovery current per spike.
// As a rate of depolarization it is what the 2003 preset had and the measured row does not: the classic RS row carries d 8 pA on a 1 pF cell, which is 8 pA per pF, and RS07 carries 100 on 100, which is 1.
// At the measured row's own d the sheet runs at 219 Hz with no front at all, because nothing behind the front falls silent.
// At 1600 pA, 16 pA per pF, the sheet is quiet between fronts and traveling waves reads 22 expanding fronts in three seconds.
// The regime is wide: with the interneuron weight at 0.6 of the base there were 20 fronts at 800 pA and 24 at 1600, at the chosen 0.5 there are 20 at 1200 and 22 at 1600, and below 600 pA it closes.
//
// SHEET_EI is the weight onto the sheet's interneurons, as a multiple of the base weight, carried by the connect node's pair table.
// One weight in picoamps is not one voltage step once the rows differ: 70 pA is 0.7 mV on a 100 pF pyramid and 3.5 mV on a 20 pF fast-spiking cell, so an excitatory surge recruits inhibition five times faster than it recruits pyramids and the front is extinguished as it forms.
// At 0.5 the interneurons get 1.75 mV per synapse, still more than the pyramids get, and the fronts return.
// The window is 0.4 to 0.6 and closes by 0.8, where the sheet falls to 4.5 Hz and nothing spreads.
// The front runs at 2.1 cm/s on the 1.2 mm sheet and at 0.8 on the guided tour's 700 um sheet.
// A measured pyramid charges over about 7 ms (100 pF against 14 nS near rest) where the classic row charges over 1.7, so each ring of tissue takes longer to reach threshold; the big sheet's 2.1 is inside the 1 to 10 cm/s reported for slices.
// On the tour's sheet a front has about 250 um to cross between the source and the far edge, which is too short a run to estimate a speed from, so the battery gates the speed on the big sheet and asks the tour only for fronts.
const SHEET_ADAPT = 1600, SHEET_EI = 0.5;
// the two pair table rows both scenes carry: excitation onto the interneurons, from the sheet and from the pacemaker
const sheetTable = () => 'sheet sheeti 1 ' + SHEET_EI + String.fromCharCode(10) + 'pacemaker sheeti 1 ' + SHEET_EI;
export function guidedTour(ed){
  const { add, wire, group } = api(ed);
  const note = (x, y, text, width=300, near='') => add('note', x, y, { text, width, near });

  note(40, 20, 'Welcome to linen\n\nYou are looking at a living spiking neural network. A small pacemaker cluster fires in rhythmic bursts, and each burst travels as a wave across a sheet of about 14,000 neurons packed at the density of mouse cortex.\n\nDrag the 3D view to orbit, scroll to zoom, and click any neuron to see its connections and voltage. The graph below builds everything you see. A single click selects a node and a double click opens its settings. The DOCS button (top right) explains every tool, and its references page lists the papers behind each scene, along with other simulators that are worth trying when you want to go further.');
  note(40, 220, 'Networks flow from top to bottom. Shape nodes define volumes (in micrometers), neuron scatter fills them with neurons, gather combines groups, connect grows the synapses, stimulus injects current, and the checkpoint runs the simulation.');
  note(40, 360, 'Something to try: double click the sphere node that feeds the pacemaker neuron scatter (shape nodes are orange), then drag the arrows that appear in the 3D view. The source of the waves moves along with it while the network keeps running.\n\nThe waves need the sheet at full density. If you lower RES in the viewer bar the network is recomputed smaller and faster, but the fronts stop making it across. For a bigger sheet, where the fronts have room to travel and collide, open traveling waves from the Tab menu.');
  note(40, 540, 'There are three checkpoint nodes in this graph, bound to the viewer keys 1, 2 and 3 (the digit shows on each node). Press a number to view that network. You can bind any node yourself by selecting it and pressing a digit. Checkpoint 2 is a sculpture of shapes, and checkpoint 3 shows thalamic rebound bursts.');

  // output 1: the wave sheet.
  // A 0.7 mm square of cortex 300 um thick at mouse density (90,000 per mm3, four fifths excitatory), wired with the paired-recording numbers: 0.15 at zero distance, a unitary EPSP of 0.5 and an IPSP of 1.5 (lognormal), and a kernel of 200 um, the reach of a local axon rather than the 100 um of a somatic pair.
  // The 1.2 mm square is the traveling waves scene.
  // Measured 2026-09-24 at full density on the reference engine: 13,881 cells, 7.5 M synapses, in-degree 542, the sheet at 11.1 Hz, 24 expanding fronts in three seconds, wired in 9 s on one thread.
  // The front speed is the membrane and is discussed where SHEET_ADAPT is set.
  // A 600 um square still carries fronts (5.5 M synapses); at half resolution the pacemaker fires at 78 Hz and nothing crosses, so the note about RES holds.
  const sheet = add('box', 460, 20, { center:[0,0,0], size:[700,300,700] });
  // Named populations.
  // A tag is what the rest of the app reads: REGIONS draws a box around each one with its name above it, POPULATIONS colors the neurons by it, the connect pair table addresses pathways by it, and a probe can read one population rather than an average over whatever shares a volume.
  // A scatter without a tag has none of that.
  const rs = add('scatter', 460, 90, { fill:1, density:72000, type:RS07, seed:1, tag:'sheet' }); wire(rs, 0, sheet);
  const fs = add('scatter', 610, 90, { fill:1, density:18000, type:FS07, seed:2, tag:'sheeti' }); wire(fs, 0, sheet);
  const pace = add('sphere', 760, 20, { center:[-220,0,-170], radius:120 });
  const ch = add('scatter', 760, 90, { fill:1, density:90000, type:CH07, seed:3, tag:'pacemaker' }); wire(ch, 0, pace);
  const beacon = add('sphere', 910, 20, { center:[220,0,170], radius:150 });
  const adapt = add('celltype', 460, 150, { row:0, name:'sheetcell', tag:'sheet', d:SHEET_ADAPT, hue:20 });
  wire(adapt, 0, rs);
  const m1 = add('gather', 535, 180); wire(m1, 0, adapt); wire(m1, 1, fs); wire(m1, 2, ch);
  // Exponential synapses with psc 1, so a weight is the total charge a spike delivers: 70 pA is a 0.7 mV EPSP on a 100 pF pyramid, and the pair table carries what that same EPSP has to be on a 20 pF interneuron (SHEET_EI).
  const cn1 = add('connect', 535, 260,
    { radius:600, sigma:200, prob:0.15, wExc:70, wInh:-150, wdist:1, velocity:250, seed:1,
      table:sheetTable() });
  wire(cn1, 0, m1);
  note(700, 255, 'Connect: nearby neurons wire up with a probability that falls off with distance. Try raising max dist and watch the synapse count in the top bar.', 200, 'checkpoint 1');
  const drive = add('stimulus', 535, 350, { mode:0, current:500 });
  wire(drive, 0, cn1); wire(drive, 1, pace);
  const pulse = add('stimulus', 535, 430, { mode:1, current:1000, period:900, width:25 });
  wire(pulse, 0, drive); wire(pulse, 1, beacon);
  note(700, 370, 'Stimulus: each of these injects current into a region. The first biases the pacemaker so it keeps bursting, the second pulses a distant spot on its own rhythm, and the third adds background noise. Edits apply to the running network right away.', 210, 'checkpoint 1');
  const ns1 = add('stimulus', 535, 510, { mode:3, current:200, radius:1000000, center:[0,0,0] });
  wire(ns1, 0, pulse);
  // Two probes and a chart, so the first scene shows what a measurement is as well as what a network is.
  // The pacemaker bursts and the sheet carries the wave, so the two rates are different numbers with an obvious meaning and the sparklines in the viewer overlay move visibly apart.
  const pbP = add('probe', 760, 520, { label:'pacemaker', tag:'pacemaker' });
  wire(pbP, 0, ns1);
  const pbS = add('probe', 910, 520, { label:'sheet', tag:'sheet' });
  wire(pbS, 0, pbP);
  const out1 = add('checkpoint', 535, 600, { steps:2, syn:1, psc:1, tauE:3, tauI:8 });
  wire(out1, 0, pbS);
  const cha1 = add('chart', 760, 660, { plot:7, pop:'sheet', window:5, height:120 });
  // the names a first visitor reads: what each part of the tour is for
  out1.name = 'waves'; adapt.name = 'the sheet row, adapting'; cn1.name = 'sheet wiring'; drive.name = 'pacemaker bias'; pulse.name = 'second wave source'; beacon.name = 'second source region'; ns1.name = 'background noise';
  wire(cha1, 0, out1);
  group('checkpoint 1  the wave sheet',
    [sheet, rs, fs, adapt, pace, ch, beacon, m1, cn1, drive, pulse, ns1, pbP, pbS, cha1, out1], 205);

  // output 2: shapes sculpture (gizmo practice)
  const spl = add('spline', 1150, 20,
    { p0:[-1500,0,-400], p1:[-500,0,500], p2:[500,0,-500], p3:[1500,0,300], radius:180 });
  // a sculpture, not a tissue: a fifth of cortical density so the shapes read as shapes, wired with the sheet's numbers
  const s2a = add('scatter', 1150, 90, { fill:1, density:18000, type:0, seed:1, tag:'ribbon' }); wire(s2a, 0, spl);
  const tor = add('torus', 1300, 20, { center:[0,0,0], radius:900, thickness:200 });
  const s2b = add('scatter', 1300, 90, { fill:1, density:18000, type:2, seed:2, tag:'ring' }); wire(s2b, 0, tor);
  const cyl = add('cylinder', 1450, 20, { center:[0,0,0], radius:220, height:1600, rotate:[25,0,20] });
  const s2c = add('scatter', 1450, 90, { fill:1, density:18000, type:3, seed:3, tag:'column' }); wire(s2c, 0, cyl);
  const m2 = add('gather', 1225, 180); wire(m2, 0, s2a); wire(m2, 1, s2b); wire(m2, 2, s2c);
  const cn2 = add('connect', 1225, 260,
    { radius:600, sigma:200, prob:0.15, wExc:0.5, wInh:-1.5, wdist:1, velocity:200, seed:1 });
  wire(cn2, 0, m2);
  // noise at 10 keeps the sculpture visibly alive (6 to 15 Hz per shape) at unitary weights
  const ns2 = add('stimulus', 1225, 350, { mode:3, current:10, radius:1000000, center:[0,0,0] });
  wire(ns2, 0, cn2);
  const out2 = add('checkpoint', 1225, 600, { steps:2 }); out2.name = 'sculpture'; cn2.name = 'sculpture wiring';
  wire(out2, 0, ns2);
  group('checkpoint 2  a sculpture of shapes',
    [spl, s2a, tor, s2b, cyl, s2c, m2, cn2, ns2, out2], 130);
  note(1150, 440, 'Press 2 for a sculpture of shapes: a tube spline, a ring and a tilted cylinder. Double click any orange shape node for a 3D gizmo, where w moves, e rotates and r scales. The first time you switch here the network has to be wired, and a progress bar shows in the top bar while it does.', 280, 'checkpoint 2');

  // output 3: thalamic rebound bursts (post-inhibitory rebound of TC cells)
  const ball = add('sphere', 1750, 20, { center:[0,0,0], radius:400 });
  // relay cells at density; a relay nucleus has almost no recurrent excitation of its own, so the local probability is a fiftieth of cortex and the rebound is the cell's, not the network's
  const s3a = add('scatter', 1750, 90, { fill:1, density:90000, type:5, seed:1, tag:'relay' }); wire(s3a, 0, ball);
  const cn3 = add('connect', 1825, 260,
    { radius:600, sigma:200, prob:0.003, wExc:0.5, wInh:-1.5, wdist:1, velocity:200, seed:1 });
  wire(cn3, 0, s3a);
  const inhib = add('stimulus', 1825, 350,
    { mode:1, current:-15, period:3000, width:200, t0:500, center:[0,0,0], radius:1000000 });
  wire(inhib, 0, cn3);
  const ns3 = add('stimulus', 1825, 430, { mode:3, current:1.45, radius:1000000, center:[0,0,0] });
  wire(ns3, 0, inhib);
  const out3 = add('checkpoint', 1825, 600, { steps:2 }); out3.name = 'rebound';
  wire(out3, 0, ns3);
  group('checkpoint 3  thalamic rebound bursts',
    [ball, s3a, cn3, inhib, ns3, out3], 25);
  note(1750, 500, 'Press 3 for thalamic rebound. These are thalamocortical (TC) relay cells. Every 3 seconds an inhibitory pulse silences them, and the moment it lets go they fire a synchronized rebound burst. Post-inhibitory rebound is a documented thalamic behavior, and the TC preset models it. Click a neuron and watch its voltage dip and then overshoot.', 280, 'checkpoint 3');
}

// the early visual pathway into the validated PD column: a retinal ganglion cell sheet (driven by the input node's ON/OFF contrast code) projects through a mirrored optic tract to a thalamic relay (TC cells in a reticular RZ shell), whose optic radiation targets L4e of the Potjans-Diesmann layered column, with the real side circuits: a weak LGN collateral into L6e and the L6e corticothalamic feedback to LGN.
// The column keeps its verified PD table and K_bg-scaled background; the thalamic drive is the actual pathway, not a pulse.
// Topography is preserved stage to stage; delays follow tract length at myelinated velocities.
function visualPathway(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Visual pathway\n\nThe early visual route as a wired circuit: a retinal sheet, the optic tract, a thalamic relay and a patch of cortex. The signal reaches cortex by traveling through the anatomy, and nothing is injected into cortex directly. Every cell is one of the measured rows of Izhikevich 2007 chapter 8, so each carries a capacitance in picofarads and every current and weight in the scene is in picoamps.');
  note(170, 'The project nodes carry the map. A cell\'s position in the source population maps onto the matching position in the target, so retinotopy survives every stage. The mirror options model the inversion that a lens and the chiasm produce.', 340, 'tracts');
  note(320, 'The input node encodes contrast instead of brightness, and it splits its cells into interleaved ON and OFF mosaics the way retinal ganglion cells divide. The geometry on its mask port sets which neurons it drives.', 340, 'stimulus');
  // The retina is a surface, so its density is per area: 1,200 ganglion cells on a 500 um square is 4,800 per mm2, the mouse retina's range (Jeon, Strettoi and Masland 1998), so it is filled by count.
  // The relay and the shell are volumes and fill at cortical density.
  const eyeBox = add('box', 40, 20, { center:[-1600, 800, 0], size:[500, 30, 500] });
  const rgc = add('scatter', 40, 75, { count:1200, type:RS07, seed:11, tag:'rgc' });
  wire(rgc, 0, eyeBox);
  const lgnBall = add('sphere', 190, 20, { center:[-800, 500, 0], radius:150 });
  const lgn = add('scatter', 190, 75, { fill:1, density:90000, type:TC07, seed:12, tag:'lgn' });
  wire(lgn, 0, lgnBall);
  const trnShell = add('torus', 340, 20, { center:[-800, 500, 0], radius:210, thickness:60 });
  const trn = add('scatter', 340, 75, { fill:1, density:90000, type:RZ07, seed:13, tag:'trn' });
  wire(trn, 0, trnShell);
  // the PD column, identical statistics to the cortical column scenario
  const layers = [
    { name:'L2/3', c:[0,900,0], s:[400,300,400], e:3750, eType:RS07, chFrac:0.10, i:1060, ltsFrac:0.35 },
    { name:'L4',   c:[0,650,0], s:[400,200,400], e:4000, eType:RS07, chFrac:0,    i: 990, ltsFrac:0.20 },
    { name:'L5',   c:[0,400,0], s:[400,300,400], e: 880, eType:IB07, chFrac:0,    i: 200, ltsFrac:0.45 },
    { name:'L6',   c:[0,100,0], s:[400,300,400], e:2620, eType:RS07, chFrac:0,    i: 530, ltsFrac:0.35 },
  ];
  const streams = [rgc, lgn, trn], boxes = [];
  layers.forEach((L, k) => {
    const x = 490 + k*300;
    const box = add('box', x, 20, { center:L.c, size:L.s });
    boxes.push(box);
    const se = add('scatter', x, 75, { count:L.e, type:L.eType, seed:k*2+1, tag:L.name+'e' });
    wire(se, 0, box);
    let eOut = se;
    if(L.chFrac){ eOut = add('celltype', x, 130, { row:(CH07) + 1, frac:L.chFrac }); wire(eOut, 0, se); }
    const si = add('scatter', x+150, 75, { count:L.i, type:FS07, seed:k*2+2, tag:L.name+'i' });
    wire(si, 0, box);
    const iOut = add('celltype', x+150, 130, { row:(LTS07) + 1, frac:L.ltsFrac });
    wire(iOut, 0, si);
    group(L.name, [box, se, eOut, si, iOut], 130);
    streams.push(eOut, iOut);
  });
  const m1 = add('gather', 190, 210), m2 = add('gather', 490, 265),
        m3 = add('gather', 640, 320), m4 = add('gather', 790, 340);
  streams.slice(0, 4).forEach((s, i) => wire(m1, i, s));
  wire(m2, 0, m1); streams.slice(4, 7).forEach((s, i) => wire(m2, i+1, s));
  wire(m3, 0, m2); streams.slice(7, 10).forEach((s, i) => wire(m3, i+1, s));
  wire(m4, 0, m3); wire(m4, 1, streams[10]);
  // optic tract (mirror u for the lens inversion), radiation into L4e, the weak collateral into L6e, and the L6e feedback to the thalamus.
  // Tract weights in unitary terms: a retinogeniculate input is one of a few strong ones per relay cell (Chen and Regehr 2000), so 5; a thalamocortical input onto L4 is many and about a millivolt (Bruno and Sakmann 2006), so 1; the collateral and the feedback are ordinary cortical unitaries.
  // On the measured rows the same unitary sizes in picoamps, each scaled by the capacitance of the cell the synapse lands on: 1000 pA into a 200 pF relay cell is the 5 mV retinogeniculate input, 100 pA into a 100 pF layer 4 pyramid is the millivolt of the radiation, the collateral is half that, and the feedback is 60 pA into the relay for 0.3 mV.
  // Measured with the bar sweep at full density 2026-09-24: retina 7.9 Hz, relay 7.5, reticular shell 2.8, L4 excitatory 7.7, L2/3 2.5.
  const opt = add('project', 490, 380, { from:'rgc', to:'lgn', axisFrom:1, axisTo:1,
    flipU:1, sigma:35, prob:0.5, weight:1000, velocity:8000, seed:1 });
  wire(opt, 0, m4);
  const rad = add('project', 640, 380, { from:'lgn', to:'L4e', axisFrom:1, axisTo:1,
    sigma:60, prob:0.8, weight:100, velocity:8000, seed:2 });
  wire(rad, 0, opt);
  const col6 = add('project', 790, 380, { from:'lgn', to:'L6e', axisFrom:1, axisTo:1,
    sigma:90, prob:0.15, weight:50, velocity:8000, seed:3 });
  wire(col6, 0, rad);
  const fb = add('project', 940, 380, { from:'L6e', to:'lgn', axisFrom:1, axisTo:1,
    sigma:60, prob:0.1, weight:60, velocity:8000, seed:4 });
  wire(fb, 0, col6);
  // local wiring: flat kernel + the verified PD table inside the column; retinal and relay cells do not synapse among themselves.
  // Same correction as the cortical column, which this scene's column is a copy of: weights scaled to an in-degree of 649 and inhibition at 4.67 times excitation.
  const cn = add('connect', 490, 440,
    { radius:1400, sigma:4000, prob:0.05, wExc:70, wInh:-298, velocity:300, seed:1,
      // beyond the PD rows: retinal cells never synapse locally, and the thalamic populations couple locally only to each other (relay to reticular and back); everything else between structures travels through the projection tracts.
      // The thalamic clamp is frozen against plasticity (fifth column 0): adult subcortical circuits are far less plastic than cortex.
      table: pdTable() + '\nrgc * 0\n* rgc 0\nlgn * 0\n* lgn 0\ntrn * 0\n* trn 0\nlgn trn 3 1 0\ntrn lgn 3 1 0' });
  wire(cn, 0, fb);
  const inp = add('input', 340, 500, { map:0, code:1, axis:1,
    cols:16, rows:16, arbor:0.8, transient:0.45, jitter:1, amp:1600 });
  const bar = add('testsignal', 170, 540, { pattern:0, period:1500 });
  wire(inp, 0, cn); wire(inp, 1, eyeBox); wire(inp, 2, bar);
  // K_bg-scaled background per layer (as the cortical column scenario) plus a small brainstem-like tone on the thalamus.
  // Noise sets fluctuation but it cannot set an operating point, so the column carries the same split background as the cortical column scenario, whose column this is a copy of, with the per-layer currents tuned there.
  const kbg = [0.775, 1.0, 0.975, 1.25];
  // The column's picoamp drives, unchanged: the same eight populations on the same rows want the same currents, and the radiation into layer 4 is a few percent on top of them rather than a second operating point.
  // What this scene adds is the retina, and the thalamus between them.
  const driveE = [210, 225, 540, 330];
  const driveI = [78, 96, 122, 90];
  // the cortical column's noise: it is the same column and the fluctuation that holds it asynchronous has to match, or the layers fall below threshold once the thalamus is held
  const bg = kbg.map(k => +(460*k).toFixed(0));
  let prev = inp;
  const drives = [];
  layers.forEach((L, k) => {
    const de = add('stimulus', 190 + k*150, 500,
      { mode:0, current:driveE[k], tag:L.name+'e' });
    wire(de, 0, prev); wire(de, 1, boxes[k]);
    const di = add('stimulus', 190 + k*150, 530,
      { mode:0, current:driveI[k], tag:L.name+'i' });
    wire(di, 0, de); wire(di, 1, boxes[k]);
    const ns = add('stimulus', 190 + k*150, 560, { mode:3, current:bg[k] });
    wire(ns, 0, di); wire(ns, 1, boxes[k]);
    drives.push(de, di, ns);
    prev = ns;
  });
  // Holding current on the relay and its shell.
  // Both rows rest close to firing, so without a hold they free-run and the reciprocal relay-to-shell loop (pair multiplier 3 in both directions) becomes an oscillator that paces the whole cortex through the radiation, every cell locked to one rhythm at 10 Hz with an ISI CV of 0.01.
  // The hold sets how much of the loop is allowed, and the two ends of it are not the same distance from threshold.
  // TC07 takes -600 pA, six times the rheobase of a 200 pF relay cell, and the shell only -50: RZ07 is a 100 pF resonator whose firing here comes from the relay rather than from its own drive, and holding it as hard as the relay silences it (at -300 pA the shell reads 0.07 Hz).
  // Measured 2026-09-24 across the hold on both: at -300 on the relay the loop wakes up and the population Fano rises from 1.4 per thousand to 8, at -900 the relay falls to 5.0 Hz and the shell to 0.2, and with the shell unheld the Fano is 3.4 and climbing.
  // Here the relay reads 7.5 Hz, the shell 2.8, and the ISI CVs are 2.0 and 1.3, which is not a locked loop.
  const holdL = add('stimulus', 640, 530, { mode:0, current:-600, tag:'lgn' });
  wire(holdL, 0, prev); wire(holdL, 1, lgnBall);
  const holdT = add('stimulus', 790, 530, { mode:0, current:-50, tag:'trn' });
  wire(holdT, 0, holdL); wire(holdT, 1, trnShell);
  const nsT = add('stimulus', 790, 560, { mode:3, current:600 });
  wire(nsT, 0, holdT); wire(nsT, 1, lgnBall);
  // Each station reads its own excitatory population.
  // The masks already put each probe in the right place; naming the population as well is what keeps a cortical probe off the fast-spiking cells that share the box, whose rate is several times higher and would sit in the same average.
  const pbR = add('probe', 190, 620, { label:'retina', tag:'rgc' });
  wire(pbR, 0, nsT); wire(pbR, 1, eyeBox);
  const pbT = add('probe', 340, 620, { label:'thalamus', tag:'lgn' });
  wire(pbT, 0, pbR); wire(pbT, 1, lgnBall);
  const pb4 = add('probe', 490, 620, { label:'L4', tag:'L4e' });
  wire(pb4, 0, pbT); wire(pb4, 1, boxes[1]);
  const pb23 = add('probe', 640, 620, { label:'L2/3', tag:'L2/3e' });
  wire(pb23, 0, pb4); wire(pb23, 1, boxes[0]);
  // plasticity kept gentle for long training runs: half-strength STDP and slow homeostasis targeting the circuit's designed sparse rates, so learning sculpts the pathway instead of re-tuning the thalamic clamp.
  // Every quantity in weight units is in picoamps: A+ 0.035, the bound 3500 pA, the inhibitory rate 0.106; A- is dimensionless, A+ over a weight.
  // The bound does not clear every wired weight.
  // Weights are drawn lognormal (sigma 1, mean preserving), and over eleven million synapses the tail runs to about a hundred and seventy times the mean: measured 2026-09-24 this scene wires weights from -26,400 to 33,800 pA, the top being the optic tract at a mean of 1000 and the largest local excitatory draw 11,900 against a mean of 70.
  // Those few are snapped to the bound on their first plasticity event.
  // Per-neuron measured set points, since a uniform target drives the column into synchronous bursting; cortical inhibition faster than excitatory scaling growth, the standard stability ordering.
  const out = add('checkpoint', 490, 680, { steps:2, syn:1, psc:1, tauE:3, tauI:8, tauS:20, tauM:20,
    // A- is A+ divided by the wired mean excitatory weight, so the weight-dependent fixed point lands on the weight the network was wired with rather than somewhere else.
    aP:0.035, aM:0.0005, wmax:3500, wdep:1, iEta:0.106, iRho:2.5,
    rhoMode:1, calS:20, scale:1, sEta:0.0003 });
  wire(out, 0, pb23);
  const cha = add('chart', 790, 680, { plot:7, pop:'L2/3', window:5, height:130 });
  wire(cha, 0, out);
  group('retina', [eyeBox, rgc], 205);
  group('LGN  thalamic relay', [lgnBall, lgn], 205);
  group('TRN  reticular shell', [trnShell, trn], 205);
  group('gather: every population into one stream', [m1, m2, m3, m4], 280);
  group('tracts  optic, radiation, collateral, feedback', [opt, rad, col6, fb], 205);
  group('local wiring  Potjans-Diesmann pair table', [cn], 130);
  group('stimulus  contrast onto the retinal sheet', [bar, inp], 55);
  group('background drive  scaled by K_bg', [...drives, holdL, holdT, nsT], 55);
  group('readout  what the observer measures', [pbR, pbT, pb4, pb23, cha], 280);
  group('checkpoint  the simulation and its plasticity', [out], 0);
}

// Alphabet school: the visual pathway with an ear added.
// One cortical column at full Potjans-Diesmann density, both senses converging on it.
//
// Why full density matters here rather than being an ambition.
// A synapse in cortex is tiny relative to threshold: the reference microcircuit uses a 0.15 mV post-synaptic potential against roughly 15 mV of distance to threshold, so a neuron fires only on the coordinated arrival of many inputs and the network settles into a smooth balanced regime.
// A reduced column carrying full-size weights has the opposite character: a handful of coincident spikes fires a cell, activity is fluctuation-dominated, and layers either burst or fall silent with nothing in between.
// The weights below are derived from the real in-degrees (about 1,900 to 4,700 recurrent inputs per neuron) rather than tuned by hand.
//
// Both senses reach the same column the way they reach cortex: a sensory sheet projects through its own thalamic relay, each relay sits inside a reticular shell that holds it in burst mode, and the radiation lands in layer 4.
// Retina and cochlea therefore converge on one piece of tissue, which is the minimal substrate for binding a letter's shape to its spoken name.
// In an animal these are separate areas converging downstream; one column standing in for that is the simplification, and it is stated rather than hidden.

// Sound localization by delay lines (Jeffress 1948; Carr and Konishi 1990 for the owl's nucleus laminaris).
// Two ears, a row of coincidence detectors between them, and axons whose conduction time is their length: a detector fires only where the two volleys arrive together, and the place that fires moves with the interaural time difference.
// At 100 um/ms a detector x micrometers from the middle needs the right ear to lead by 2x/100 ms, so a 4 ms difference puts the peak 200 um off center.
function soundLocalization(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Sound localization\n\nA row of coincidence detectors sits between two ears, wired by axons whose conduction time is set by their length (Jeffress 1948). Each ear fires a volley at every click. A detector fires only where the two volleys arrive in the same millisecond, and that place moves with the time difference between the ears.');
  note(210, 'Something to try: the right ear pulse has an onset of 2 ms. Set it to 0 and the middle detectors fire. Set it to 6 and the peak moves toward the right ear, since the left volley has to travel further to make up the lead. You can watch the three probes, or the cells in the viewer.', 340, 'clicks');
  note(430, 'The numbers: axons conduct at 100 um per ms, so a detector 200 um from the middle hears a 4 ms lead as simultaneous. Each ear alone lands under threshold (about 13 mV of the 20 between rest and threshold on these cells), and both together go over it. The barn owl does this with delays of a few hundred microseconds. The engine steps in milliseconds, so the clicks in this scene are further apart.', 340, 'delay lines');
  const row = add('box', 40, 20, { center:[0,0,0], size:[800,40,40] });
  const det = add('scatter', 40, 75, { fill:1, density:90000, type:RS07, seed:1, tag:'detectors' }); wire(det, 0, row);
  const lBall = add('sphere', 190, 20, { center:[-600,0,0], radius:40 });
  const left = add('scatter', 190, 75, { fill:1, density:90000, type:RS07, seed:2, tag:'left' }); wire(left, 0, lBall);
  const rBall = add('sphere', 340, 20, { center:[600,0,0], radius:40 });
  const right = add('scatter', 340, 75, { fill:1, density:90000, type:RS07, seed:3, tag:'right' }); wire(right, 0, rBall);
  const m = add('gather', 190, 160); wire(m, 0, det); wire(m, 1, left); wire(m, 2, right);
  // every ear cell reaches every detector (a flat kernel inside 1300 um), nothing else connects; the weight puts one ear's volley two thirds of the way from rest to threshold.
  // On the measured RS07 row that gap is 20 mV, rest -60 to threshold -40, and 42 pA of kick per synapse gets a detector there: at 45 every detector answers every click and the place code disappears, at 40 half of them stop answering at all (2026-09-24).
  const cn = add('connect', 190, 230,
    { radius:1300, sigma:5000, prob:1, wExc:42, wInh:-150, wdist:0, velocity:100, seed:1,
      table:'left detectors 1 1 0\nright detectors 1 1 0\n* * 0' });
  wire(cn, 0, m);
  // a click every 200 ms: the left ear at 0, the right ear 2 ms later
  const clickL = add('stimulus', 40, 300, { mode:1, current:2500, period:200, width:2, t0:0, tag:'left' });
  wire(clickL, 0, cn); wire(clickL, 1, lBall);
  const clickR = add('stimulus', 190, 300, { mode:1, current:2500, period:200, width:2, t0:2, tag:'right' }); clickL.name = 'left ear clicks'; clickR.name = 'right ear clicks';
  wire(clickR, 0, clickL); wire(clickR, 1, rBall);
  const lEnd = add('sphere', 40, 380, { center:[-260,0,0], radius:130 });
  const pbL = add('probe', 40, 435, { label:'left third', tag:'detectors' }); wire(pbL, 0, clickR); wire(pbL, 1, lEnd);
  const mid = add('sphere', 190, 380, { center:[0,0,0], radius:130 });
  const pbM = add('probe', 190, 435, { label:'middle', tag:'detectors' }); wire(pbM, 0, pbL); wire(pbM, 1, mid);
  const rEnd = add('sphere', 340, 380, { center:[260,0,0], radius:130 });
  const pbR = add('probe', 340, 435, { label:'right third', tag:'detectors' }); wire(pbR, 0, pbM); wire(pbR, 1, rEnd);
  const out = add('checkpoint', 190, 500, { steps:2 });
  wire(out, 0, pbR);
  const cha = add('chart', 340, 560, { plot:8, pop:'middle', window:2, height:130 });
  wire(cha, 0, out);
  group('coincidence detectors', [row, det], 130);
  group('left ear', [lBall, left], 205);
  group('right ear', [rBall, right], 25);
  group('clicks  one volley per ear, the right one late', [clickL, clickR], 55);
  group('delay lines  axons at 100 um per ms', [m, cn], 280);
  group('readout  three places along the row', [lEnd, pbL, mid, pbM, rEnd, pbR, cha], 280);
  group('checkpoint', [out], 0);
}

// The virtual patch rig: one cell of each classic type under a current ramp, the way a slice physiologist reads a cell's class off its response to current steps (Connors and Gutnick 1990; Izhikevich 2003 Fig. 2, the same seven presets the published-results suite checks).
//
// This scene stays on the classic rows because it is the reproduction: the seven presets are the 2003 paper's, the current is that paper's own unit, and the rheobase and rate of each cell are the numbers the figure shows.
// On the measured rows it would be a different claim about different cells.
// Every other scene's main network is on the measured rows of the 2007 book, where a current is picoamps and each row carries its own capacitance.
// The guided tour's second and third checkpoints, the sculpture of shapes and thalamic rebound, are on the classic rows too.
function patchRig(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Virtual patch rig\n\nSeven cells, one of each type, each alone in its own dish with an electrode in it. The current ramps from zero to 12 over the first three seconds and then holds. If you read each probe\'s rate over those seconds from left to right, you are reading that cell\'s current to rate curve.\n\nThis scene is a reproduction and stays on the classic rows: the seven cells are the presets of Izhikevich (2003) and the current is in the unit that model uses, so the curves are the ones figure 2 shows. The other scenes build their networks from the measured rows of the 2007 book, where a cell carries a capacitance in picofarads and a current is in picoamps.');
  note(200, 'Something to try: click a cell in the viewer to see its membrane potential. The current where a cell starts firing is its rheobase. TC and LTS start almost at once, RS needs about 4, and FS needs about the same but fires far faster once it goes. RZ is a resonator and fires only in a narrow band. Change the ramp\'s current to see how far each curve goes.', 340, 'the electrode');
  const names = ['RS', 'IB', 'CH', 'FS', 'LTS', 'TC', 'RZ'], types = [RS, IB, CH, FS, LTS, TC, RZ];
  const cells = [], dishes = [], probes = [];
  let prev = null;
  names.forEach((nm, k) => {
    const x = -600 + k*200;
    const dish = add('sphere', 40 + k*150, 20, { center:[x,0,0], radius:30 });
    const c = add('scatter', 40 + k*150, 75, { fill:0, count:1, type:types[k], seed:k + 1, tag:nm.toLowerCase() });
    wire(c, 0, dish); dishes.push(dish); cells.push(c);
  });
  const m = add('gather', 490, 160, { ports:7 });
  while(m.inputs.length < 7) m.inputs.push(null);   // the editor grows the ports on screen; a builder does it here
  cells.forEach((c, k) => wire(m, k, c));
  // no synapses: each cell answers its electrode alone
  const cn = add('connect', 490, 230, { radius:1, sigma:1, prob:0, wExc:0, wInh:0, velocity:200, seed:1 });
  wire(cn, 0, m);
  const ramp = add('stimulus', 490, 300, { mode:2, current:12, period:3000, radius:100000, center:[0,0,0] }); ramp.name = 'current ramp';
  wire(ramp, 0, cn);
  prev = ramp;
  names.forEach((nm, k) => {
    const pb = add('probe', 40 + k*150, 380, { label:nm, tag:nm.toLowerCase() });
    wire(pb, 0, prev); prev = pb; probes.push(pb);
  });
  const out = add('checkpoint', 490, 460, { steps:1 });
  wire(out, 0, prev);
  const cha = add('chart', 640, 520, { plot:7, pop:'RS', window:3, height:130 });
  wire(cha, 0, out);
  names.forEach((nm, k) => group(nm + '  ' + ['regular spiking', 'intrinsic burst', 'chattering', 'fast spiking', 'low threshold', 'thalamocortical', 'resonator'][k], [dishes[k], cells[k]], (k*50) % 360));
  group('readout  a probe per cell', probes, 25);
  group('the electrode  a ramp to 12 over 3 s, then held', [m, cn, ramp], 55);
  group('checkpoint', [out, cha], 0);
}

// Seizure and control: the balanced net with its inhibition blocked for two seconds.
// The same knobs as every other scene (Brunel 2000's regimes: inhibition dominated is asynchronous, excitation dominated runs away); the block is a hyperpolarizing current on every inhibitory cell, what a GABA antagonist does to a slice.
// A mechanism, not a clinic: nothing here is a prediction about a patient.
function seizure(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Seizure and control\n\nThis is the balanced random net, asynchronous at a few hertz, with its inhibitory cells silenced for two seconds starting three seconds in. Recruitment runs away until the population fires as fast as its refractory period allows, and when inhibition returns it settles back down within a second. The wiring is the same as the balanced net, and nothing is added for the seizure.');
  note(230, 'Keep in mind that this shows a mechanism and is not a clinical simulation. It is a runaway of recurrent excitation once inhibition fails, which is one of the things an epileptic focus can be, and it says nothing about any patient. The block is a hyperpolarizing current on the inhibitory cells, which is the model\'s version of a GABA antagonist on a slice, and it takes 30 nA to hold a cell under threshold against the 597 synapses reaching it.', 340, 'the block');
  note(440, 'Something to try: change the block\'s width (two seconds) or its current (-30 nA), or weaken it to -6 nA, which leaves the interneurons firing at about 38 Hz and still lets the excitatory population run away: what holds the network down is the balance, not the amount of inhibition on its own. At full density the excitatory rate is 3.2 Hz before the block, 256 Hz during it, which is most of the ceiling the 2 ms refractory period allows, and 3.2 Hz again within a second of it ending. The chart draws the excitatory rate over the last ten seconds.', 340, 'the block');
  const vol = add('sphere', 40, 20, { center:[0,0,0], radius:400 });
  const e = add('scatter', 40, 75, { fill:1, density:72000, type:RS07, seed:1, tag:'exc' }); wire(e, 0, vol);
  const i = add('scatter', 190, 75, { fill:1, density:18000, type:FS07, seed:2, tag:'inh' }); wire(i, 0, vol);
  const m = add('gather', 115, 155); wire(m, 0, e); wire(m, 1, i);
  // the balanced net's wiring and drive, in picoamps on the measured rows
  const cn = add('connect', 115, 220,
    { radius:300, sigma:3000, prob:0.1, wExc:50, wInh:-250, wdist:1, velocity:100, seed:1 });
  wire(cn, 0, m);
  const nsE = add('stimulus', 40, 280, { mode:3, current:1800, tag:'exc', radius:100000, center:[0,0,0] });
  wire(nsE, 0, cn);
  const ns = add('stimulus', 190, 280, { mode:3, current:200, tag:'inh', radius:100000, center:[0,0,0] });
  wire(ns, 0, nsE);
  // one pulse on every inhibitory cell, 2 s wide, 3 s in (the train repeats every 100 s, so it is one block in a watching session).
  //
  // On the measured rows it takes 30 nA to hold an interneuron under threshold at full density, where it is receiving 597 synapses (2026-09-22): at 6 nA the interneurons still fire at 38 Hz.
  // The scene reads 3.2 Hz excitatory before the block, 256 Hz during it, which is most of the ceiling a 2 ms refractory period allows, and 3.2 Hz within a second of it ending.
  const block = add('stimulus', 340, 280, { mode:1, current:-30000, period:100000, width:2000, t0:3000, tag:'inh', radius:100000, center:[0,0,0] }); block.name = 'inhibition block';
  wire(block, 0, ns);
  const pbE = add('probe', 40, 400, { label:'exc', tag:'exc' });
  wire(pbE, 0, block);
  const pbI = add('probe', 190, 400, { label:'inh', tag:'inh' });
  wire(pbI, 0, pbE);
  const out = add('checkpoint', 115, 460, { steps:2 });
  wire(out, 0, pbI);
  const cha = add('chart', 265, 520, { plot:7, pop:'exc', window:10, height:130 });
  wire(cha, 0, out);
  group('volume', [vol], 130);
  group('excitatory  RS at density', [e], 130);
  group('inhibitory  FS at density', [i], 205);
  group('random wiring and noise drive  as the balanced net', [m, cn, ns], 280);
  group('the block  inhibition silenced for 2 s', [block], 0);
  group('readout', [pbE, pbI, cha], 280);
  group('checkpoint', [out], 0);
}

// Gamma from inhibition (Whittington, Traub and Jefferys 1995; Whittington et al. 2000): a patch of cortex driven tonically, paced into the gamma band by its fast spiking cells.
// Every pyramidal volley recruits the interneurons, whose inhibition silences the patch for one GABA-A decay, about ten milliseconds, and the next volley follows: a period of 15 to 30 ms, 30 to 70 Hz, that shortens with drive.
function gammaPatch(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Gamma from inhibition\n\nA patch of cortex at density, driven tonically, and paced into the gamma band by its fast spiking cells. Every pyramidal volley recruits the interneurons, their inhibition silences the patch for about one GABA-A decay, and then the next volley follows (Whittington et al. 2000). The rate over time chart shows the rhythm, which is 35 Hz here.');
  note(230, 'Something to try: raise the drive and the band shifts up. Lengthen the inhibitory time constant on the checkpoint (tau I, 6 ms) and the rhythm slows, since the silence after each volley lasts longer. Set the inhibitory weight to zero on the connect node and the rhythm goes away with it.', 340, 'wiring');
  const vol = add('sphere', 40, 20, { center:[0,0,0], radius:200 });
  const e = add('scatter', 40, 75, { fill:1, density:72000, type:RS07, seed:1, tag:'pyr' }); wire(e, 0, vol);
  const i = add('scatter', 190, 75, { fill:1, density:18000, type:FS07, seed:2, tag:'fs' }); wire(i, 0, vol);
  const m = add('gather', 115, 155); wire(m, 0, e); wire(m, 1, i);
  // fast spiking cells are wired more densely and more strongly than pyramids in both directions (Holmgren 2003; Packer and Yuste 2011)
  const cn = add('connect', 115, 220,
    { radius:500, sigma:200, prob:0.15, wExc:50, wInh:-150, wdist:1, velocity:200, seed:1,
      table:'pyr fs 3 2\nfs pyr 3 2\nfs fs 3 1' });
  wire(cn, 0, m);
  // measured 2026-09-15: 34 Hz with 36 percent of the rate's power at the peak (pyramids 21 Hz, fast spiking 52); at a drive of 6 and 8 ms the rhythm sat at 25 Hz
  const drive = add('stimulus', 115, 280, { mode:0, current:1200, tag:'pyr', radius:100000, center:[0,0,0] }); drive.name = 'tonic drive';
  wire(drive, 0, cn);
  // picoamps on the measured rows: 1.2 nA holds the pyramids depolarized and the 50 pA of noise keeps the volleys from locking into lockstep.
  // Measured 2026-09-22 at full density over four seconds: the rate spectrum peaks at 35 Hz with the pyramids at 20 Hz and the interneurons at 63.
  const ns = add('stimulus', 265, 280, { mode:3, current:50, radius:100000, center:[0,0,0] });
  wire(ns, 0, drive);
  const pbE = add('probe', 40, 400, { label:'pyramids', tag:'pyr' });
  wire(pbE, 0, ns);
  const pbI = add('probe', 190, 400, { label:'fast spiking', tag:'fs' });
  wire(pbI, 0, pbE);
  const out = add('checkpoint', 115, 460, { steps:1, syn:1, psc:1, tauE:3, tauI:6 });
  wire(out, 0, pbI);
  const cha = add('chart', 265, 520, { plot:7, pop:'pyramids', window:0.5, height:130 });
  wire(cha, 0, out);
  group('volume', [vol], 130);
  group('pyramids  RS', [e], 130);
  group('fast spiking  FS', [i], 205);
  group('wiring  interneurons dense and strong', [m, cn], 280);
  group('drive  tonic and noise', [drive, ns], 55);
  group('readout', [pbE, pbI, cha], 280);
  group('checkpoint  exponential synapses, GABA-A at 6 ms', [out], 0);
}

// The Tab menu: the tutorial and the tissue scenes.
// The experiment scenes (the training blocks, the learning lab, the bench, the weaves, the mandelbulb) live in experiments.js and load by name from the address bar (?scenario=) and from the trainer, not from the menu.
// Every node type in one scene, apart from the ones that take a file or a device (footage, live, mesh, points file, connections file), as one structure: a balanced core, six arms around it that each fork into three buds, and the whole crown repeated into three tiers that shrink and turn as they rise.
// Every arrange node shapes an arm (a flat ribbon, so a twist shows), the buds come from a noise field, repeat is used three times and nests (tier2.ray4.fork1.bud), population picks the tiers apart, and move tilts a ray and the top tier.
// The numbers in the core follow the balanced net's.
export function everyNode(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Every node\n\nOne structure with a node of every kind that needs no file or device. It has a balanced core, six arms around it that each fork into three buds, and the whole crown repeated into three tiers that shrink and turn as they rise. Read the note beside each group, then double-click any node to see its settings. You can also bypass a node (d) to see what it was doing.');
  note(200, 'The core: a sphere region, a neuron scatter per class filling it at a density per cubic millimeter, and a cell type node that makes a fraction of the excitatory cells intrinsically bursting. The wiring and the drive follow, at a smaller weight, the balanced random net\'s.', 340, 'the core');
  note(340, 'One arm: a flat box region filled by a neuron scatter, then every arrange node in turn. Cull cuts an eyelet out of it with a small sphere, gradient thins it toward the tip, twist turns the ribbon half a turn along its length, noise warp bends it with a smooth field, gauss blur softens its edges, and operations lays a wave along it with a formula.', 340, 'one arm');
  note(520, 'The buds: a noise field region, a seeded density pattern inside a bounds box, filled by a neuron scatter beyond the arm\'s tip. Repeat lays down three copies turned about the tip, and a move turns the fan back so it sits astride the arm.', 340, 'the buds');
  note(660, 'One ray: the arm and its buds gathered, tilted upward by a move, and typed: a cell type node set to custom names a row of the scene\'s own for the arm (the measured regular-spiking numbers, so its currents are in picoamps), and another makes half of the bud cells chattering.', 340, 'one ray');
  note(800, 'The crown: repeat turns the ray six times about the vertical axis, and a second repeat stacks the six as three tiers, each two thirds the size of the one below, raised and turned by thirty degrees. The copies are numbered, so every cell has a population name that says where it is: tier2.ray4.fork1.bud. Three population nodes pick the tiers back out by name, and the top one is tilted on its own before all three are gathered with the core.', 340, 'the crown');
  note(1000, 'Wiring: gather joins the populations; receptors declares channels beyond E and I; plasticity names a pair STDP rule on the core\'s excitatory synapses; feedback inhibition pools the core\'s excitatory cells; project runs a tract from the core to the lowest tier\'s arms; connect grows the local synapses.', 340, 'wiring');
  note(1180, 'Drive: a noise current per population, a pulse train on every arm by its population name, a slower pulse on the top tier alone through a sphere as a mask, a bar sweep from the test signal fed across the lowest tier through a flat box as a mask, and a lesson of shapes from the curriculum fed onto the core. Readout: a probe per kind of tissue, analysis, the checkpoint with plasticity on, and a chart.', 340, 'drive');
  // ---- the core ----
  const ball = add('sphere', 40, 20, { center:[0, 0, 0], radius:250 });
  const e = add('scatter', 40, 75, { fill:1, density:72000, type:RS07, seed:1, tag:'exc' }); wire(e, 0, ball);
  const i = add('scatter', 190, 75, { fill:1, density:18000, type:FS07, seed:2, tag:'inh' }); wire(i, 0, ball);
  const nt = add('celltype', 40, 130, { row:(IB07) + 1, frac:0.15 }); wire(nt, 0, e);
  // ---- one arm: a flat ribbon from the core's edge outward, through every arrange node ----
  const slab = add('box', 340, 20, { center:[520, 0, 0], size:[520, 30, 170] });
  const hole = add('sphere', 490, 20, { center:[470, 0, 0], radius:45 });
  const sl = add('scatter', 340, 75, { fill:1, density:400000, type:RS, seed:3, tag:'arm' }); wire(sl, 0, slab);
  const cu = add('cull', 340, 130, { keep:0 }); wire(cu, 0, sl); wire(cu, 1, hole);
  const gr = add('gradient', 340, 185, { axis:0, start:1, end:0.35, seed:7 }); wire(gr, 0, cu);
  const tw = add('twist', 340, 240, { axis:0, angle:360, center:[520, 0, 0] }); wire(tw, 0, gr);
  const nw = add('noisewarp', 340, 295, { amplitude:22, scale:260, seed:6 }); wire(nw, 0, tw);
  const gb = add('gaussblur', 340, 350, { sigma:5, seed:5 }); wire(gb, 0, nw);
  const op = add('ops', 340, 405, { expr:'y = y + sin(x*0.014)*28' }); wire(op, 0, gb);
  // ---- the buds: a noise field beyond the tip, forked three ways ----
  const field = add('noisefield', 640, 20, { center:[935, 0, 0], size:[290, 80, 80], pattern:0, scale:60, coverage:0.6, seed:3 });
  const pa = add('scatter', 640, 75, { fill:1, density:300000, type:RS07, seed:4, tag:'bud' }); wire(pa, 0, field);
  const fk = add('repeat', 640, 130, { copies:3, tag:'fork', rotate:[0, 48, 0], pivot:[790, 0, 0] }); wire(fk, 0, pa);
  const fc = add('move', 640, 185, { rotate:[0, -48, 0], pivot:[790, 0, 0] }); wire(fc, 0, fk);
  // ---- one ray: arm and buds together, tilted, typed ----
  const gA = add('gather', 490, 460); wire(gA, 0, op); wire(gA, 1, fc);
  const tilt = add('move', 490, 515, { rotate:[0, 0, 14], pivot:[250, 0, 0] }); wire(tilt, 0, gA);
  const ct = add('celltype', 490, 570, { row:0, name:'armcell', tag:'arm', hue:285 }); wire(ct, 0, tilt);
  const ch = add('celltype', 490, 625, { row:(CH07) + 1, tag:'fork*.bud', frac:0.5 }); wire(ch, 0, ct);
  // ---- the crown: six rays, three tiers, the tiers picked apart ----
  const ring = add('repeat', 490, 680, { copies:6, tag:'ray', rotate:[0, 60, 0], pivot:[0, 0, 0] }); wire(ring, 0, ch);
  const tiers = add('repeat', 490, 735, { copies:3, tag:'tier', translate:[0, 340, 0], rotate:[0, 30, 0], scale:[0.66, 0.66, 0.66], pivot:[0, 0, 0] }); wire(tiers, 0, ring);
  const t1 = add('population', 340, 790, { tag:'tier1.*' }); wire(t1, 0, tiers);
  const t2 = add('population', 490, 790, { tag:'tier2.*' }); wire(t2, 0, tiers);
  const t3 = add('population', 640, 790, { tag:'tier3.*' }); wire(t3, 0, tiers);
  const crown = add('move', 640, 845, { rotate:[0, 0, 10], translate:[0, 40, 0], pivot:[0, 560, 0] }); wire(crown, 0, t3);
  // ---- wiring ----
  const g = add('gather', 190, 900, { ports:6 }); while(g.inputs.length < 6) g.inputs.push(null);
  wire(g, 0, nt); wire(g, 1, i); wire(g, 2, t1); wire(g, 3, t2); wire(g, 4, crown);
  const rc = add('receptor', 190, 955); wire(rc, 0, g);
  const pl = add('plasticity', 190, 1010, { name:'stdp', from:'exc', to:'exc', aP:0.8, aM:0.001, wmax:3000, wdep:1 }); wire(pl, 0, rc);
  const fb = add('pool', 190, 1065, { tag:'exc', gain:-1 }); wire(fb, 0, pl);
  const pr = add('project', 190, 1120, { from:'exc', to:'tier1.ray*.arm', axisFrom:1, axisTo:1, dispersed:1, fanout:20, weight:40 }); wire(pr, 0, fb);
  // Weights in picoamps on the measured rows, and smaller than a tissue scene's: 30 pA is 0.3 mV on a 100 pF cell where the balanced net uses 50.
  // The crown is the reason: the arms and the buds are geometry with no interneurons of their own, and at the balanced net's 50 pA the purely excitatory crown runs to the refractory ceiling and takes the core with it (measured 2026-09-24: core 333 Hz, arms 53).
  const cn = add('connect', 190, 1175, { radius:300, sigma:3000, prob:0.1, wExc:30, wInh:-150, wdist:1, velocity:100, seed:1 }); wire(cn, 0, pr);
  // ---- drive ----
  // A noise current per population rather than one for the whole scene.
  // The core is inhibition-dominated and bistable, the crown has no inhibition at all, and the two want different currents: 560 pA on the core's pyramids against 200 on its interneurons, and 800 on the buds, which have nothing but noise and their neighbors.
  // Measured 2026-09-24: at half the inhibitory drive the core flips to 207 Hz, at half again above it the core falls to 1 Hz, and the arms need no noise node at all (they are driven by the pulse trains and the bar, and read the same to two decimals with their noise at 0 or at 200).
  const ns = add('stimulus', 190, 1230, { mode:3, current:560, tag:'exc', radius:100000, center:[0, 0, 0] }); wire(ns, 0, cn);
  const nsI = add('stimulus', 190, 1258, { mode:3, current:200, tag:'inh', radius:100000, center:[0, 0, 0] }); wire(nsI, 0, ns);
  const nsB = add('stimulus', 190, 1286, { mode:3, current:800, tag:'*.bud', radius:100000, center:[0, 0, 0] }); wire(nsB, 0, nsI);
  // the arms' row is the measured one: currents in picoamps
  const pu = add('stimulus', 190, 1314, { mode:1, current:500, period:200, width:10, tag:'*.arm', radius:100000, center:[0, 0, 0] }); wire(pu, 0, nsB);
  const cap = add('sphere', 490, 1340, { center:[0, 640, 0], radius:330 });
  const pc = add('stimulus', 190, 1340, { mode:1, current:500, period:330, width:10, t0:100, tag:'*.arm' }); wire(pc, 0, pu); wire(pc, 1, cap);
  const plate = add('box', 490, 1395, { center:[0, 110, 0], size:[2600, 420, 2600] });
  const bar = add('testsignal', 640, 1395, { pattern:0, period:1500 });
  // 1000 doubles an arm's rate under a sweep, 3000 takes it to 20 Hz
  const inS = add('input', 190, 1395, { tag:'tier1.ray*.arm', map:0, axis:1, cols:16, rows:16, amp:2500 }); wire(inS, 0, pc); wire(inS, 1, plate); wire(inS, 2, bar);
  const cur = add('curriculum', 490, 1450, { sense:0, set:4, onMs:400, offMs:350 });
  // against the core's noise of 560: 120 pA takes the core from 1.9 to 3.2 Hz under a sweep, and 400 flips it to the runaway state at 235
  const inB = add('input', 190, 1450, { tag:'exc', map:0, axis:1, cols:12, rows:12, amp:120 }); wire(inB, 0, inS); wire(inB, 1, ball); wire(inB, 2, cur);
  // ---- readout ----
  const pbE = add('probe', 190, 1505, { label:'core exc', tag:'exc' }); wire(pbE, 0, inB);
  const pbI = add('probe', 190, 1560, { label:'core inh', tag:'inh' }); wire(pbI, 0, pbE);
  const pbS = add('probe', 190, 1615, { label:'arms', tag:'*.arm' }); wire(pbS, 0, pbI);
  const pbT = add('probe', 190, 1670, { label:'buds', tag:'*.bud' }); wire(pbT, 0, pbS);
  const an = add('analysis', 190, 1725, {}); wire(an, 0, pbT);
  const out = add('checkpoint', 190, 1780, { steps:2, syn:1, psc:1, tauE:3, tauI:8, plast:1 }); wire(out, 0, an);
  const cha = add('chart', 190, 1835, { plot:7, pop:'core exc', window:5, height:130 }); wire(cha, 0, out);
  group('the core  sphere, neuron scatters, cell type', [ball, e, i, nt], 25);
  group('one arm  a ribbon through every arrange node', [slab, hole, sl, cu, gr, tw, nw, gb, op], 270);
  group('the buds  noise field, neuron scatter, forked by a repeat', [field, pa, fk, fc], 190);
  group('one ray  arm and buds, tilted, typed', [gA, tilt, ct, ch], 300);
  group('the crown  six rays, three tiers, picked apart', [ring, tiers, t1, t2, t3, crown], 270);
  group('wiring  gather to connect', [g, rc, pl, fb, pr, cn], 140);
  group('drive  currents, a bar sweep, a lesson', [ns, nsI, nsB, pu, cap, pc, plate, bar, inS, cur, inB], 55);
  group('readout  a probe per tissue, analysis, chart', [pbE, pbI, pbS, pbT, an, cha], 175);
  group('run  the checkpoint, plasticity on', [out], 0);
}

// Working memory held in synapses rather than in firing (Mongillo, Barak and Tsodyks 2008).
// Two selective excitatory populations share a volume with one inhibitory pool; every synapse facilitates (a low release probability that each spike raises for a second and a half, against a short depression).
// A cue drives one population briefly; both then fall silent, but the cued one's synapses stay facilitated, and a later pulse given equally to both is answered by the cued one, whose inhibition holds the other down.
// Tuned on the reference engine (tools/tune.mjs, 200 ms bins).
// The release is as published (a rested synapse releases U of its weight, stpNorm 0), so a resting population's recurrence is weak and a facilitated one's is strong enough to ignite: within-population weights 8 times the base, across 0.05, inhibition -6, a noise drive of 5.5 (silent at rest), a read pulse of 1.5.
// Read at 2.6 s: cued 15.3 Hz, other 3.6.
// Cue the other instead: 15.2 against 4.5.
// No cue: 13.1 and 11.7, alike.
// Facilitation of 50 ms: 11.6 and 10.5, alike.
// Read at 1.8 s: 13.8 against 1.4; at 3.4 s: 15.7 against 5.5, fading with the facilitation.
// With the scaled release (stpNorm 1) a resting population ignites as readily as a facilitated one and the cued population reactivates by itself in population spikes (the paper's other regime); a read that lands just after one finds it depressed and the other population wins, so the scene uses the silent regime.
export function workingMemory(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Working memory in synapses\n\nTwo selective populations, a and b, share a patch of cortex with one inhibitory pool. Every four seconds a cue drives one of them for 300 ms, a and b taking turns, and then the whole patch goes silent. 1.6 s after each cue, a weak pulse is given equally to both. Without a cue the two would answer it alike, at about 12 Hz each. After a cue, the cued one answers at 15 Hz and the other at under 5. The cued population\'s synapses are still facilitated, so it ignites first and its inhibition holds the other one down. The item was held in the synapses and not in firing (Mongillo, Barak and Tsodyks 2008).\n\nThe answer is a burst of a few tens of milliseconds, so the raster under the viewer is the best place to watch: the two populations take turns answering the same pulse. Raise PER FRAME in the viewer bar to run it faster.');
  note(200, 'The mechanism is on the checkpoint: short-term plasticity with a low release probability (0.2), a short depression (200 ms) and a long facilitation (1500 ms), which is the facilitating class of prefrontal excitatory synapses (Wang et al. 2006). A rested synapse releases a fifth of its weight and a facilitated one up to three or four times that, and that is the difference between a population that cannot sustain a burst and one that can. One difference from the paper: in this model the setting applies to every synapse, inhibitory ones included, where the paper facilitates only the excitatory to excitatory ones.', 340, 'checkpoint');
  note(380, 'Something to try: set the facilitation tau on the checkpoint to 50 ms and the two populations answer every read pulse alike, since nothing of the cue is left by then. Move the read pulse later (its onset, in ms) and the uncued population recovers as the facilitation fades: 1.4 Hz with the read 0.8 s after the cue, 3.6 at 1.6 s and 5.5 at 2.4 s. Bypass one cue node (d) and that population never wins.', 340, 'cue');
  const vol = add('sphere', 40, 20, { center:[0, 0, 0], radius:250 });
  const a = add('scatter', 40, 75, { fill:1, density:36000, type:RS07, seed:1, tag:'mema' }); wire(a, 0, vol);
  const b = add('scatter', 190, 75, { fill:1, density:36000, type:RS07, seed:2, tag:'memb' }); wire(b, 0, vol);
  const i = add('scatter', 340, 75, { fill:1, density:18000, type:FS07, seed:3, tag:'inh' }); wire(i, 0, vol);
  const m = add('gather', 190, 155); wire(m, 0, a); wire(m, 1, b); wire(m, 2, i);
  // the balanced net's random wiring; each selective population's own synapses far stronger than the ones between them, the learned structure the paper assumes
  const cn = add('connect', 190, 220, { radius:300, sigma:3000, prob:0.1, wExc:50, wInh:-600, wdist:1, velocity:100, seed:1,
    table:'mema mema 1 8\nmemb memb 1 8\nmema memb 1 0.05\nmemb mema 1 0.05' });
  wire(cn, 0, m);
  const ns = add('stimulus', 190, 280, { mode:3, current:300, radius:100000, center:[0, 0, 0] }); wire(ns, 0, cn);
  // the protocol repeats so the running page keeps showing it: a cue every 4 s, a and b in turn, and the read 1.6 s after each (a single cue and read passed every gate and left the page silent; 6 s apart felt dead at a third of real time; at 3 s the last item has not faded and the contrast drops from 30 against 10 Hz to 30 against 20)
  const cue = add('stimulus', 190, 335, { mode:1, current:200, period:8000, width:300, t0:500, tag:'mema', radius:100000, center:[0, 0, 0] }); wire(cue, 0, ns);
  const cueB = add('stimulus', 340, 335, { mode:1, current:200, period:8000, width:300, t0:4500, tag:'memb', radius:100000, center:[0, 0, 0] }); wire(cueB, 0, cue);
  const read = add('stimulus', 190, 390, { mode:1, current:150, period:4000, width:60, t0:2100, tag:'mem*', radius:100000, center:[0, 0, 0] }); wire(read, 0, cueB); cue.name = 'cue a'; cueB.name = 'cue b'; read.name = 'read pulse';
  const pbA = add('probe', 190, 445, { label:'a', tag:'mema' }); wire(pbA, 0, read);
  const pbB = add('probe', 190, 500, { label:'b', tag:'memb' }); wire(pbB, 0, pbA);
  const pbI = add('probe', 190, 555, { label:'inh', tag:'inh' }); wire(pbI, 0, pbB);
  const out = add('checkpoint', 190, 610, { steps:2, syn:1, psc:1, tauE:3, tauI:8, stp:1, stpU:0.2, stpTauD:200, stpTauF:1500, stpNorm:0 }); wire(out, 0, pbI);
  const cha = add('chart', 190, 665, { plot:8, pop:'a', window:8, height:130 }); wire(cha, 0, out);
  group('volume', [vol], 25);
  group('two selective populations and an inhibitory pool', [a, b, i], 190);
  group('wiring  strong within a population, weak across', [m, cn], 140);
  group('background', [ns], 55);
  group('cue  a and b in turn, every 4 s', [cue, cueB], 55);
  group('read  a and b alike, 1.6 s after each cue', [read], 55);
  group('readout', [pbA, pbB, pbI, cha], 175);
  group('checkpoint  facilitating synapses', [out], 0);
}

// Polychrony: with conduction delays, which cells fire together depends on the order their inputs fired in, not only on which inputs fired (Izhikevich 2006).
// Three sources on a line, a reader beyond each end.
// Axons at 100 um per ms make the delays 4, 10 and 16 ms to the near reader and 16, 10 and 4 to the far one, so the sources firing c, b, a six milliseconds apart reach the reader beside a in the same millisecond, and a, b, c reach the one beside c.
// The same three sources, the same synapses, two timings, two groups.
export function polychrony(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Polychrony\n\nThree sources, a, b and c, sit on a line with a reader beyond each end, and every source reaches both readers. Axons conduct at 100 um per ms, so a spike from a takes 4 ms to reach the reader beside it and 16 ms to reach the far one, and for c it is the other way round. Twice a second the sources fire in the order c, b, a, six milliseconds apart. Their spikes arrive at the reader beside a in the same millisecond and it fires, while at the other reader they arrive spread over 24 ms and it stays silent. In between, the order a, b, c does the opposite. Which cells fire together is set by timing (Izhikevich 2006).');
  note(200, 'Something to try: change the axon velocity on the connect node. At 200 um per ms the delays halve, the six millisecond steps no longer line the spikes up, and neither reader fires. You can also change one onset (in ms) on a stimulus node and watch its reader drop out. In the paper these groups are not wired by hand. They emerge through spike-timing plasticity in a random network with delays, and there are far more of them than there are cells.', 340, 'order one');
  const P = { a:[-600, 0, 0], b:[0, 0, 0], c:[600, 0, 0], x:[-1000, 0, 0], y:[1000, 0, 0] };
  const ball = (k, x) => add('sphere', x, 20, { center:P[k], radius:60 });
  const cells = (k, tag, x, seed, geo) => { const s = add('scatter', x, 75, { fill:1, density:90000, type:RS07, seed, tag }); wire(s, 0, geo); return s; };
  const ga = ball('a', 40), gb = ball('b', 190), gc = ball('c', 340), gx = ball('x', 490), gy = ball('y', 640);
  const sa = cells('a', 'srca', 40, 1, ga), sb = cells('b', 'srcb', 190, 2, gb), sc = cells('c', 'srcc', 340, 3, gc);
  const rx = cells('x', 'readx', 490, 4, gx), ry = cells('y', 'ready', 640, 5, gy);
  const m = add('gather', 340, 155, { ports:6 }); while(m.inputs.length < 6) m.inputs.push(null);
  [sa, sb, sc, rx, ry].forEach((s, k) => wire(m, k, s));
  // every source to every reader and nothing else; the kernel is flat, so only the distance sets the delay.
  // The weight puts one volley well under threshold and three together over it.
  // Measured 2026-09-25: at 20 pA each reader reads 2.5 Hz over four seconds, one spike per volley of its own order and none for the other; at 15 they read 0.4 and 0.1, from 25 the other order begins to be answered.
  // At 200 um per ms neither fires.
  const cn = add('connect', 340, 220, { radius:1800, sigma:100000, prob:0.5, wExc:20, wInh:0, wdist:0, velocity:100, seed:1,
    table:'* * 0\nsrc* read* 1' });
  wire(cn, 0, m);
  let prev = cn;
  const pulse = (tag, t0, x, y, geo) => { const s = add('stimulus', x, y, { mode:1, current:2500, period:400, width:2, t0, tag }); wire(s, 0, prev); wire(s, 1, geo); prev = s; return s; };
  // order one: c, b, a, six milliseconds apart, from 0; order two: a, b, c, from 200
  const o1 = [pulse('srcc', 0, 190, 300, gc), pulse('srcb', 6, 190, 355, gb), pulse('srca', 12, 190, 410, ga)];
  const o2 = [pulse('srca', 200, 490, 300, ga), pulse('srcb', 206, 490, 355, gb), pulse('srcc', 212, 490, 410, gc)];
  const pbX = add('probe', 340, 480, { label:'reader beside a', tag:'readx' }); wire(pbX, 0, prev);
  const pbY = add('probe', 340, 535, { label:'reader beside c', tag:'ready' }); wire(pbY, 0, pbX);
  const pbS = add('probe', 340, 590, { label:'sources', tag:'src*' }); wire(pbS, 0, pbY);
  const out = add('checkpoint', 340, 645, { steps:1 }); wire(out, 0, pbS);
  const cha = add('chart', 340, 700, { plot:8, pop:'reader beside a', window:2, height:130 }); wire(cha, 0, out);
  group('source a', [ga, sa], 25);
  group('source b', [gb, sb], 25);
  group('source c', [gc, sc], 25);
  group('reader beside a', [gx, rx], 190);
  group('reader beside c', [gy, ry], 190);
  group('delay lines  every source to both readers', [m, cn], 140);
  group('order one  c, b, a', o1, 55);
  group('order two  a, b, c', o2, 55);
  group('readout', [pbX, pbY, pbS, cha], 175);
  group('checkpoint', [out], 0);
}

// Traveling waves: a 1.2 mm sheet.
// It is the scene the battery measures the front speed on.
// Measured 2026-09-15 at full density on the reference engine: 38,880 cells, 26.8 M synapses, in-degree 690, the sheet at 11 Hz, 18 expanding fronts in three seconds at 3.0 cm/s with inhibition intact.
// At a 100 um kernel (in-degree 150) the pacemaker fired and nothing spread; with inhibition cut to a third the sheet ran at 26 Hz and the fronts slowed to 1.7 cm/s.
export function travelingWaves(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Traveling waves\n\nA 1.2 mm square of cortex, 300 um thick, at mouse density: about 39,000 neurons and 27 million synapses, so give the wiring a moment. A cluster of chattering cells in one corner bursts rhythmically, and each burst crosses the sheet as a wave. A second source in the far corner pulses on its own rhythm, so the fronts collide.');
  note(200, 'Measured at full density: 22 fronts in three seconds at 2.1 cm/s, which is inside the 1 to 10 cm/s reported for propagation in cortical slices (Chervin 1988; Golomb and Amitai 1997). Fronts that recruit most of the sheet with quiet periods in between correspond to a disinhibited slice and not to awake cortex, where waves modulate sparse firing.');
  note(380, 'Something to try: on the connect node, lower sigma from 200 to 100 um and the pacemaker fires but nothing spreads (1.2 Hz across the sheet). Cut the inhibitory weight to a third and the sheet runs at 31 Hz with the fronts gone, because the whole sheet bursts at once instead. The waves also need the full density, so keep RES at 100%.');
  const sheet = add('box', 40, 20, { center:[0,0,0], size:[1200,300,1200] });
  const rs = add('scatter', 40, 90, { fill:1, density:72000, type:RS07, seed:1, tag:'sheet' }); wire(rs, 0, sheet);
  const fs = add('scatter', 190, 90, { fill:1, density:18000, type:FS07, seed:2, tag:'sheeti' }); wire(fs, 0, sheet);
  const pace = add('sphere', 340, 20, { center:[-400,0,-300], radius:150 });
  const ch = add('scatter', 340, 90, { fill:1, density:90000, type:CH07, seed:3, tag:'pacemaker' }); wire(ch, 0, pace);
  const beacon = add('sphere', 490, 20, { center:[400,0,300], radius:200 }, 'second source region');
  const adapt = add('celltype', 40, 150,
    { row:0, name:'sheetcell', tag:'sheet', d:SHEET_ADAPT, hue:20 }, 'the sheet row, adapting');
  wire(adapt, 0, rs);
  const m1 = add('gather', 115, 180); wire(m1, 0, adapt); wire(m1, 1, fs); wire(m1, 2, ch);
  // Exponential synapses with psc 1, so a weight is the total charge a spike delivers: 70 pA is a 0.7 mV EPSP on a 100 pF pyramid, and the pair table carries what that same EPSP has to be on a 20 pF interneuron (SHEET_EI).
  const cn = add('connect', 115, 260,
    { radius:600, sigma:200, prob:0.15, wExc:70, wInh:-150, wdist:1, velocity:250, seed:1,
      table:sheetTable() }, 'sheet wiring');
  wire(cn, 0, m1);
  const drive = add('stimulus', 115, 350, { mode:0, current:500 }, 'pacemaker bias');
  wire(drive, 0, cn); wire(drive, 1, pace);
  const pulse = add('stimulus', 115, 430, { mode:1, current:1000, period:900, width:25 }, 'second wave source');
  wire(pulse, 0, drive); wire(pulse, 1, beacon);
  const ns = add('stimulus', 115, 510, { mode:3, current:200, radius:1000000, center:[0,0,0] }, 'background noise');
  wire(ns, 0, pulse);
  const pbP = add('probe', 340, 520, { label:'pacemaker', tag:'pacemaker' }); wire(pbP, 0, ns);
  const pbS = add('probe', 490, 520, { label:'sheet', tag:'sheet' }); wire(pbS, 0, pbP);
  const out = add('checkpoint', 115, 600, { steps:2, syn:1, psc:1, tauE:3, tauI:8 }, 'waves'); wire(out, 0, pbS);
  const cha = add('chart', 340, 660, { plot:7, pop:'sheet', window:5, height:120 }); wire(cha, 0, out);
  group('the sheet', [sheet, rs, fs, adapt, m1], 205);
  group('the two sources', [pace, ch, beacon], 30);
  group('wiring and drive', [cn, drive, pulse, ns], 130);
  group('readout', [pbP, pbS, out, cha], 175);
}

// Every scenario opens with every node named: by its builder where it gave a name, from its place in the graph otherwise (nodenames.js).
export const named = build => ed => { build(ed); nameNodes(ed.nodes, ed.groups); };
export const SCENARIOS = [
  { name:'guided tour',           build: named(guidedTour) },
  { name:'traveling waves',       build: named(travelingWaves) },
  { name:'mouse cortical column', build: named(corticalColumn) },
  { name:'visual pathway',        build: named(visualPathway) },
  { name:'thalamocortical loop',  build: named(thalamocortical) },
  { name:'CA3 pattern completion', build: named(ca3Attractor) },
  { name:'balanced random net',   build: named(balancedNet) },
  { name:'sound localization',    build: named(soundLocalization) },
  { name:'virtual patch rig',     build: named(patchRig) },
  { name:'seizure and control',   build: named(seizure) },
  { name:'gamma from inhibition', build: named(gammaPatch) },
  { name:'working memory',        build: named(workingMemory) },
  { name:'polychrony',            build: named(polychrony) },
  { name:'every node',            build: named(everyNode) },
];

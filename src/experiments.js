// Experiment scenes: the training block, the learning lab, the binding bench, the two weaves and the mandelbulb.
// They build with the same api as the scenarios.
// The Tab menu is reserved for the tutorial and the tissue scenes, so these load by name from the address bar (?scenario=) and from the trainer, the audits and the battery, which see every scene through ALL_SCENARIOS.
import { api, EQ, IQ, RS, IB, CH, FS, LTS, TC, RZ, pdTable, SCENARIOS, named } from './scenarios.js';

// population counts are per square millimeter of surface, so a smaller footprint means proportionally fewer neurons at unchanged density, unchanged distances and unchanged delays.
// In-degree falls with it, because the kernel spans the box, so each footprint is its own network with its own working point and needs its background drive tuned the way full scale was tuned.
// This is the growth strategy read backwards: territory is the free variable, local statistics are not.
let FOOTPRINT = 1.0;
export function setFootprint(mm2){ FOOTPRINT = Math.max(0.05, Math.min(1, mm2)); }
export function getFootprint(){ return FOOTPRINT; }


// Alphabet school 2.
// One laminar block of cortex, parcellated in-plane into a visual, an association and an auditory territory, each carrying the verified PD layer structure; thalamus and sensors stacked beneath so the whole scenario reads as one specimen.
// Cross-modal convergence happens where anatomy puts it, in the association territory, with the conditioning arrangement the lab measured to be necessary: sound leads, its projection is deliberately the weaker one, and plasticity is gated to the convergent paths and the association territory inhibition.
// Background drive at 0.7 of the tuned currents.
// At 1.0 the cortex runs 3 to 5 Hz spontaneous, several times the awake rates of de Kock and Sakmann 2009, and the letter response rides on it at less than 2x contrast, vanishing one relay downstream.
// At 0.7 spontaneous lands at 0.7 to 1.3 Hz across layers and the letter drives v.L4 at seven times background.
// The contrast is the code.
let AS2_AMP = 30, AS2_DRIVE = 0.7;
// Per-layer multipliers on the tuned background drive, [L2/3, L4, L5, L6].
// Potjans and Diesmann 2014 report layer-specific spontaneous rates, sparse L2/3 under 1 Hz against higher L5.
// This is the knob that tunes the background to those rates.
let AS2_LAYER = [1, 1, 1, 1];
// Architecture switches for the binding experiments.
// In the learning lab the plastic convergence and the plastic E/I sparsening sit on the same population; school 2 splits them across a relay, convergence into L4 and E/I plasticity in L2/3. xl4ei gives the convergence site its own plastic E/I, the same generic inhibitory plasticity every other plastic pair uses. xrelay unfreezes the association territory's L4 to L2/3 projection, the canonical experience dependent synapse and the preparation where spike timing plasticity was characterized (Feldman 2000, Neuron), and the path by which layer 2/3 can learn to read what layer 4 has mixed.
let AS2_XL4EI = 0, AS2_XRELAY = 1;
// Association territory background drive multiplier.
// Inhibitory plasticity can only sparsen what varies, and an item independent background current works against it.
// Association cortex also runs quieter than primary sensory cortex, so a lower operating point is the realistic direction.
let AS2_XDRIVE = 1;
// Stream balance levers.
// Measured 2026-08-28: sound alone recreates the association pattern at 0.75 while sight reaches 0.33, with the visual weakness rooted upstream (v.L2/3 decode 0.34 against a.L2/3 at 0.56). vamp and aamp scale each sense's input gain; xvw and xaw scale each stream's convergent projection weight.
// All four land in node parameters and therefore in the saved graph.
let AS2_VAMP = 1, AS2_AAMP = 1, AS2_XVW = 1, AS2_XAW = 1;
// Arrival offset between the streams, milliseconds.
// Positive means sound leads (the lab's conditioning arrangement, default 15); negative means sight leads.
// The conduction delays are symmetric, so this is the whole nominal offset, but the effective offset at the convergence site also carries the integration asymmetry between a loud narrow tone and thin contrast rings, which only a sweep can find.
// The STDP window is tauS 20 ms; an effective offset outside it silences the cross-stream terms.
let AS2_LAG = 15;
// Association territory size and convergence sparsity.
// With full convergence every association layer 4 cell receives both streams, so every item drives every association neuron and no item-specific subset can exist for competition to separate.
// Pattern-separation circuits do the opposite: a large target population, each cell sampling few inputs, different cells sampling different ones (dentate granule cells, cerebellar granule layer). xshare enlarges the association territory at the unimodal territories' expense; xprob thins the convergent projections so each association cell sees a random subset.
let AS2_XSHARE = 0.25, AS2_XPROB = 0.3;
// Per-neuron lognormal spread on the background drive.
// Cortical firing rates are lognormal in every region and state examined, and a uniform current over statistically identical neurons gives a uniform population.
let AS2_SPREAD = 0, AS2_RELAY = 0;
// Diagnostic lesion of the association L4 recurrent scaffold. x.L4 carries the full PD table at wired weights.
// 1 removes x.L4e to x.L4e (lesion, diagnostic only), 2 makes it plastic instead.
let AS2_XREC = 0;
// Multiplier on the association territory's own L4e to L4e connection probability.
// Recurrence at the convergence site is what lets the two modalities bind to each other rather than each to the cell, and binding needs enough of it: measured 2026-08-28, x.L4e holds 342 cells with 5,757 recurrent synapses, 16.8 per cell at 4.9 percent density, which gives roughly 1.4 within-assembly recurrent inputs per cell against the 40 the reference model runs on.
// That is a factor of thirty, and it is a property of how much tissue the footprint builds rather than of any plasticity setting.
let AS2_XRECP = 1;
// Weight multiplier on the same pair.
// Raising xrecp alone raises the number of recurrent inputs per cell, and that count grows with the footprint at fixed density: 50 per cell at footprint 0.05 but 210 at 0.25.
// Tissue whose synaptic weights were tuned for the smaller number runs away at the larger one, measured 2026-08-28 at 24 Hz mean with synchrony 6455.
// Setting xrecw to the reciprocal of xrecp divides the same total recurrent excitation among more synapses, which is what "more tissue" should mean and what makes a connectivity change separable from a drive change.
let AS2_XRECW = 1;
// Mapping of the two convergent projections into the association territory.
// Topographic, which the project node's own reference calls the wrong mode for a higher-order target: each modality then maps its sensory topography onto x.L4e, so a letter's visual pattern lands where its retinal position sends it and the same letter's sound lands where its cochlear content sends it.
// Those two position sets are unrelated by construction and spatially separated, so the distance-limited recurrence may not even reach between them.
// Measured 2026-08-28: each modality builds a reliable item code at the site, split-half centered cosine 0.267 for vision and 0.553 for audio, while the cross-modal centered cosine is -0.072 with 149 trials.
// Reliable codes, statistically independent of each other, which is the disjoint failure. xdisp gives each source cell a seeded random subset of the target with no spatial map, so both modalities scatter across the whole population and can overlap.
let AS2_XDISP = 0;
// Plasticity on the association territory's own layer 2/3 recurrent excitation.
// That layer is the first site in the block never driven directly by either sense: it sees only what layer 4 has already mixed, which is where a modality-independent representation would have to live.
// With only the two inhibitory pairs plastic, the recurrent excitatory connections between exactly the cells that would form an assembly there exist and cannot learn.
// Measured 2026-08-29 at that site: the auditory item code survives with a split-half ceiling of 0.516 while the visual one collapses to 0.069, so there is also not much visual signal for an assembly to bind to.
// Recurrent excitatory connections in layer 2/3 are functionally specific and are the assembly substrate (Ko 2011, Cossell 2015).
// The layer's E to I and I to E pairs are already plastic and held to a low target, which is the stabilizer this needs.
let AS2_XL23REC = 1;
// Density and weight multipliers on the unimodal layer 4 to layer 2/3 projection, the v and a territories' own relay.
// Traced 2026-08-29 by split-half reliability along the visual pathway: LGN 0.971, v.L4 0.911, v.L2/3 0.412, x.L4 0.355, x.L2/3 -0.04.
// The visual code is near perfect through thalamus and layer 4 and loses more than half its reliability at this one projection, then decays to nothing.
// The auditory equivalent is far gentler, MGB 0.903 to a.L2/3 0.60, which is why audio still has a code at the deep site and vision does not.
// The loss is structural rather than caused by that synapse's plasticity: freezing it leaves the reliability unchanged at 0.412 against 0.382.
// More postsynaptic cells sampling the layer 4 code is the direct way to carry more of it.
// The relay multiplier is a footprint-dependent knob and its default is for the full block.
// Measured with the driven audit on 2026-08-31: at footprint 0.15 the published weight moved v.L2/3 not at all and 2.5x was needed for any propagation, while at footprint 1.0 the same 2.5x saturated every downstream stage (v.L2/3 92% driven, x.L4 100% at Jaccard 0.99) and 1.0 gives clean sparse propagation, because tract arbors and the pair-table radius grow with the sheet, so in-degree rises roughly sixfold between the two sizes with weights fixed.
// Tune at the footprint you run at; numbers from a slice do not transfer.
let AS2_URELAYP = 1, AS2_URELAYW = 1;
// Spatial spread of the two convergent projections into the association territory, as a multiplier on their published 200 um sigma.
// Measured 2026-08-29: at both association sites every item's representation correlates 0.6 to 0.92 with every other item's, decode is 32 to 37 percent against a chance of 20, and one item comes out anti-correlated with itself, while one stage upstream v.L2/3 decodes at 65 percent and shows none of it.
// The collapse happens at the convergence and it is not the recurrence: refreezing that layer leaves every number unchanged.
// A sigma wide relative to the territory gives every association cell nearly the same mixture of its inputs, so every cell responds nearly the same way and the population cannot carry more than about one dimension.
let AS2_XSIGMA = 1;
export function setBlockAmp(a){ AS2_AMP = Math.max(1, +a || 30); }
export function setBlockDrive(m){ AS2_DRIVE = Math.max(0, +m || 1); }
export function setBlockXl4ei(v){ AS2_XL4EI = (v|0) === 1 ? 1 : 0; }
export function setBlockXrelay(v){ AS2_XRELAY = (v|0) === 1 ? 1 : 0; }
export function setBlockXdrive(v){ AS2_XDRIVE = Math.max(0, +v || 1); }
export function setBlockXshare(v){ AS2_XSHARE = Math.max(0.1, Math.min(0.8, +v || 0.25)); }
export function setBlockXprob(v){ AS2_XPROB = Math.max(0.01, Math.min(1, +v || 0.3)); }
export function setBlockSpread(v){ AS2_SPREAD = Math.max(0, Math.min(2, +v || 0)); }
export function setBlockRelay(v){ AS2_RELAY = (v|0) === 1 ? 1 : 0; }
export function setBlockLag(v){ AS2_LAG = Math.max(-100, Math.min(100, +v)); }
export function setBlockXrec(v){ AS2_XREC = (v|0) === 1 ? 1 : (v|0) === 2 ? 2 : 0; }
export function setBlockXrecp(v){ if(v) AS2_XRECP = Math.max(0.1, +v); }
export function setBlockXrecw(v){ if(v) AS2_XRECW = Math.max(0.01, +v); }
export function setBlockXdisp(v){ AS2_XDISP = (v|0) === 1 ? 1 : 0; }
export function setBlockXl23rec(v){ AS2_XL23REC = (v|0) === 1 ? 1 : 0; }
export function setBlockXsigma(v){ if(v) AS2_XSIGMA = Math.max(0.05, +v); }
export function setBlockUrelay(o){
  if(o && o.p) AS2_URELAYP = Math.max(0.1, +o.p);
  if(o && o.w) AS2_URELAYW = Math.max(0.1, +o.w);
}
export function setBlockBalance(o){
  if(o.vamp) AS2_VAMP = Math.max(0.1, +o.vamp);
  if(o.aamp) AS2_AAMP = Math.max(0.1, +o.aamp);
  if(o.xvw) AS2_XVW = Math.max(0.1, +o.xvw);
  if(o.xaw) AS2_XAW = Math.max(0.05, +o.xaw);
}
export function setBlockLayers(v){
  const a = String(v).split(',').map(x => +x);
  if(a.length === 4 && a.every(x => isFinite(x) && x >= 0)) AS2_LAYER = a;
}

function alphabetSchool2(ed){
  const { add, wire, note, group } = api(ed);
  const F = FOOTPRINT, side = Math.round(1000*Math.sqrt(F));
  const pop = n => Math.max(1, Math.round(n*F));
  note(20, 'Alphabet school 2\n\nOne block of cortex at mouse density, parcellated in the plane into three territories: visual on the left, association in the middle and auditory on the right. Each one carries the verified Potjans and Diesmann layer structure. Areal borders in the plane and layers in depth is how the tissue itself is organized, and the hairline gaps are there so that you can see the borders in the viewer.');
  note(240, 'The senses stay segregated until the association territory, which is where multisensory neurons are found in cortex (Bruce, Desimone and Gross 1981). Sound leads sight by 15 ms through a weaker projection, which is the conditioning arrangement: the association cell fires from sight, the tone arrives just inside the plasticity window before the spike, and the tone gains the ability to evoke what it could not evoke alone.');
  note(500, 'Plasticity is gated here. The microcircuit within each territory and the sensory tracts are frozen. What learns is the two convergent projections into the association territory and the layer 2/3 inhibition of the association territory itself, which is held toward a low fixed target so that its code stays sparse.');
  note(760, 'The scene runs on the alphabet, with image and sound together on one clock, and with the retina in ON/OFF contrast so that two letters differ by their edges and not by their filled area. Each presentation is 1 s with a 1 s gap. At 200 ms nothing learns at all. The drive uses per-layer tuned currents, and blockdrive on the training page scales it.');

  // ---- geometry: one slab, three territories -------------------------
  const GAP = 30;
  const wV = side*0.36, wX = side*0.22, wA = side*0.36;
  const cV = -(wX/2 + GAP + wV/2), cX = 0, cA = (wX/2 + GAP + wA/2);
  // territory shares sum to one; the unimodal pair splits what the association territory does not take
  const uni = (1 - AS2_XSHARE)/2;
  const terr = [
    { key:'v', cx:cV, w:wV, share:uni },
    { key:'x', cx:cX, w:wX, share:AS2_XSHARE },
    { key:'a', cx:cA, w:wA, share:uni },
  ];
  const LAYERS = [
    { name:'L2/3', y:900, h:300, e:20683, i:5834, chFrac:0.10, ltsFrac:0.35 },
    { name:'L4',   y:650, h:200, e:21915, i:5479, chFrac:0,    ltsFrac:0.20 },
    { name:'L5',   y:400, h:300, e:4850,  i:1065, chFrac:0.22, ltsFrac:0.45 },
    { name:'L6',   y:100, h:300, e:14395, i:2948, chFrac:0,    ltsFrac:0.35 },
  ];
  const streams = [];
  let nodeX = 40;
  for(const T of terr){
    LAYERS.forEach((L, k) => {
      const box = add('box', nodeX, 20,
        { center:[T.cx, L.y, 0], size:[T.w, L.h, side] });
      const ne = Math.max(4, Math.round(L.e*F*T.share));
      const ni = Math.max(2, Math.round(L.i*F*T.share));
      const se = add('scatter', nodeX, 75,
        { count:ne, type:RS, seed:100 + k*10 + terr.indexOf(T),
          tag:T.key + '.' + L.name + 'e' });
      wire(se, 0, box);
      let eOut = se;
      if(L.chFrac){
        eOut = add('celltype', nodeX, 130,
          { row:(L.name === 'L5' ? IB : CH) + 1, frac:L.chFrac });
        wire(eOut, 0, se);
      }
      const si = add('scatter', nodeX + 75, 75,
        { count:ni, type:FS, seed:200 + k*10 + terr.indexOf(T),
          tag:T.key + '.' + L.name + 'i' });
      wire(si, 0, box);
      const iOut = add('celltype', nodeX + 75, 130, { row:(LTS) + 1, frac:L.ltsFrac });
      wire(iOut, 0, si);
      streams.push(eOut, iOut);
      // one backdrop per territory and layer: the excitatory and inhibitory populations that share a layer, and the box that bounds them
      group(T.key + '.' + L.name, [box, se, eOut, si, iOut],
        T.key === 'v' ? 205 : T.key === 'a' ? 25 : 130);
      nodeX += 160;
    });
  }

  // ---- thalamus and sensors, stacked beneath -------------------------
  const ball = v => Math.max(40, Math.round(v*Math.cbrt(F)));
  const sheet = v => Math.max(120, Math.round(v*Math.sqrt(F)));
  const lgnBall = add('sphere', 40, 200, { center:[cV, -350, 0], radius:ball(220) });
  const lgn = add('scatter', 40, 255,
    { count:Math.max(400, pop(2750)), spacing:10, type:TC, seed:13, tag:'lgn' });
  wire(lgn, 0, lgnBall);
  const lgnShell = add('torus', 190, 200,
    { center:[cV, -350, 0], radius:ball(300), thickness:ball(80) });
  const trn = add('scatter', 190, 255,
    { count:Math.max(200, pop(1375)), spacing:10, type:RZ, seed:14, tag:'trn' });
  wire(trn, 0, lgnShell);
  group('visual thalamus  LGN + TRN', [lgnBall, lgn, lgnShell, trn], 205);
  const mgbBall = add('sphere', 340, 200, { center:[cA, -350, 0], radius:ball(220) });
  const mgb = add('scatter', 340, 255,
    { count:Math.max(300, pop(2200)), spacing:10, type:TC, seed:15, tag:'mgb' });
  wire(mgb, 0, mgbBall);
  const mgbShell = add('torus', 490, 200,
    { center:[cA, -350, 0], radius:ball(300), thickness:ball(80) });
  const trna = add('scatter', 490, 255,
    { count:Math.max(150, pop(1100)), spacing:10, type:RZ, seed:16, tag:'trna' });
  wire(trna, 0, mgbShell);
  group('auditory thalamus  MGB + TRN', [mgbBall, mgb, mgbShell, trna], 25);
  // Resolution floors: 26 items are unreadable at coarse maps, so the visual map never drops below 32 cells across and the retina always carries at least two cells per channel; the cochlea gets the same floor.
  // At 40 per side at full footprint letter strokes are about a pixel wide once center-surround coding removes their interiors, legible but marginal.
  // Measured over the 26 items, mean pairwise similarity falls from 0.369 at a 32 grid to 0.271 at 48 and 0.259 at 64, most of it gained by the high forties, so the grid is 64 per side at a full square millimeter.
  // The cell count is still the density figure, max() only guarding the case where the grid would outrun it, and at every footprint the density term still wins (12000 cells against 8192 channels at F=1, 6000 against 4050 at F=0.5).
  const chV = Math.max(32, Math.round(64*Math.sqrt(F)));
  const chA = Math.max(32, Math.round(48*Math.sqrt(F)));
  const eyeBox = add('box', 640, 200,
    { center:[cV, -900, 0], size:[sheet(2000), 30, sheet(2000)] });
  const rgc = add('scatter', 640, 255,
    { count:Math.max(2*chV*chV, pop(12000)), type:RS, seed:11, tag:'rgc', spacing:8 });
  wire(rgc, 0, eyeBox);
  group('retina', [eyeBox, rgc], 205);
  // its own deck below the retina: at full footprint the retina sheet is two millimeters across and a strip at the same depth cuts through it
  const earBox = add('box', 790, 200,
    { center:[cA, -1080, 0], size:[100, 30, sheet(3000)] });
  const coch = add('scatter', 790, 255,
    { count:Math.max(8*chA, pop(6000)), type:RS, seed:12, tag:'coch', spacing:5 });
  wire(coch, 0, earBox);
  group('cochlea', [earBox, coch], 25);
  streams.push(lgn, trn, mgb, trna, rgc, coch);

  // ---- merge chain ---------------------------------------------------
  let mPrev = null, mi = 0, mx = 40;
  const merges = [];
  while(mi < streams.length){
    const m = add('gather', mx, 330);
    mx += 90;
    let port = 0;
    if(mPrev){ wire(m, 0, mPrev); port = 1; }
    while(port < 4 && mi < streams.length) wire(m, port++, streams[mi++]);
    mPrev = m;
    merges.push(m);
  }
  group('gather: every population into one stream', merges, 280);

  // ---- tracts and the convergence ------------------------------------
  let chain = mPrev;
  const tracts = [], learned = [];
  const proj = (o) => { const pnode = add('project', mx, 330, o); mx += 90;
    wire(pnode, 0, chain); chain = pnode;
    (o.frozen ? tracts : learned).push(pnode);
    return pnode; };
  const ts = v => Math.max(20, Math.round(v*Math.sqrt(F)));
  // Tract velocities are effective, not literal.
  // The block compresses the brain's geometry (the thalamus sits half a millimeter below cortex instead of centimeters away), so realistic axon speeds over these distances would give every tract a sub-millisecond time that clamps to the 1 ms floor, and no delay spectrum for spike timing to work on.
  // Velocities are set so the MEAN delay of each tract lands on its physiological latency (retina to LGN about 4 ms, thalamocortical 2.5, corticothalamic feedback 6, cortico-cortical association 5, association feedback 8), and the spread of real distances around the mean supplies the spectrum.
  // The delays are the measured biology; the velocities absorb the compression.
  // Driver synapses, not pooled ones.
  // A wide kernel would pool hundreds of ganglion cells onto every relay cell and average the letters into one another.
  // Retinogeniculate convergence is a handful of strong inputs per relay cell (Usrey, Reppas and Reid 1999), which is what preserves the retinotopic map instead of averaging it away.
  // The tight sigma does most of it: the ball maps a 1 mm sheet into a 440 um nucleus, so a fine retinal feature is finer still here.
  proj({ from:'rgc', to:'lgn', axisFrom:1, axisTo:1, flipU:1,
    sigma:25, prob:0.35, weight:8.0*EQ, velocity:190, seed:1, frozen:1 });
  proj({ from:'coch', to:'mgb', axisFrom:1, axisTo:1,
    sigma:30, prob:0.35, weight:9.0*EQ, velocity:225, seed:2, frozen:1 });
  proj({ from:'lgn', to:'v.L4e', axisFrom:1, axisTo:1,
    sigma:ts(150), prob:0.8, weight:1.0*EQ, velocity:400, seed:3, frozen:1 });
  proj({ from:'mgb', to:'a.L4e', axisFrom:1, axisTo:1,
    sigma:ts(150), prob:0.8, weight:1.0*EQ, velocity:400, seed:4, frozen:1 });
  // Feedforward inhibition: thalamic axons drive fast-spiking layer 4 interneurons at least as hard as they drive the principal cells (Cruikshank, Lewis and Connors 2007), and that disynaptic inhibition arriving one or two milliseconds behind the excitation is the main thing standing between a thalamic volley and the whole layer firing.
  // Without it the driven audit read layer 4 at 100% active for every letter.
  // Slightly wider than the excitatory arbor, as measured for the interneuron pool.
  proj({ from:'lgn', to:'v.L4i', axisFrom:1, axisTo:1,
    sigma:ts(180), prob:0.8, weight:1.3*EQ, velocity:400, seed:19, frozen:1 });
  proj({ from:'mgb', to:'a.L4i', axisFrom:1, axisTo:1,
    sigma:ts(180), prob:0.8, weight:1.3*EQ, velocity:400, seed:20, frozen:1 });
  proj({ from:'lgn', to:'v.L6e', axisFrom:1, axisTo:1,
    sigma:ts(220), prob:0.15, weight:0.4*EQ, velocity:195, seed:5, frozen:1 });
  proj({ from:'mgb', to:'a.L6e', axisFrom:1, axisTo:1,
    sigma:ts(220), prob:0.15, weight:0.4*EQ, velocity:190, seed:6, frozen:1 });
  proj({ from:'v.L6e', to:'lgn', axisFrom:1, axisTo:1,
    sigma:150, prob:0.1, weight:0.25*EQ, velocity:85, seed:7, frozen:1 });
  proj({ from:'a.L6e', to:'mgb', axisFrom:1, axisTo:1,
    sigma:150, prob:0.1, weight:0.25*EQ, velocity:85, seed:8, frozen:1 });
  // the paths that learn: unimodal layer 2/3 into the association territory layer 4, the auditory side at a third of the visual weight
  proj({ from:'v.L2/3e', to:'x.L4e', axisFrom:1, axisTo:1,
    dispersed:AS2_XDISP,
    sigma:ts(200*AS2_XSIGMA), prob:AS2_XPROB, weight:+(1.0*AS2_XVW*EQ).toFixed(3),
    velocity:75, seed:9 });
  proj({ from:'a.L2/3e', to:'x.L4e', axisFrom:1, axisTo:1,
    dispersed:AS2_XDISP,
    sigma:ts(200*AS2_XSIGMA), prob:AS2_XPROB, weight:+(0.33*AS2_XAW*EQ).toFixed(3),
    velocity:75, seed:10 });
  // feedback from the association territory to both unimodal areas, frozen: it is the readout path for completion, not a site of learning
  proj({ from:'x.L5e', to:'v.L2/3e', axisFrom:1, axisTo:1,
    sigma:ts(220), prob:0.1, weight:0.25*EQ, velocity:70, seed:17, frozen:1 });
  proj({ from:'x.L5e', to:'a.L2/3e', axisFrom:1, axisTo:1,
    sigma:ts(220), prob:0.1, weight:0.25*EQ, velocity:70, seed:18, frozen:1 });

  group('sensory and feedback tracts  frozen', tracts, 205);
  group('convergence into the association territory  learns', learned, 130);

  // ---- local wiring: PD per territory, frozen except assoc L2/3 E/I --
  const WE = 0.15, G = 4.67;
  const rows = [];
  // Plastic pairs keep their PD pair probabilities and only flip the plasticity flag (the PD multipliers for these pairs run 1.6 to 3.4, so an override row with multipliers at one would thin them to a third).
  // The association layer 2/3 E/I is always plastic; xl4ei extends that to the layer 4 convergence site and xrelay to the L4 to L2/3 projection.
  const NLCH = String.fromCharCode(10);
  const xPlast = new Set(['L2/3e|L2/3i', 'L2/3i|L2/3e']);
  if(AS2_XL4EI){ xPlast.add('L4e|L4i'); xPlast.add('L4i|L4e'); }
  if(AS2_XRELAY) xPlast.add('L4e|L2/3e');
  if(AS2_XL23REC) xPlast.add('L2/3e|L2/3e');
  // Unimodal layer 4 to layer 2/3.
  // Frozen, v.L2/3 is a fixed random projection of v.L4 that cannot learn to represent anything: measured 2026-08-28, decode falls from 72 percent at v.L4 to 23 at v.L2/3, and drive heterogeneity sparsens the layer without recovering the loss because the surviving cells are the ones with high background gain rather than the ones tuned to a feature.
  // This is the canonical experience dependent synapse and the preparation spike timing plasticity was characterized in (Feldman 2000, Neuron).
  const uniPlast = AS2_RELAY ? new Set(['L4e|L2/3e']) : new Set();
  for(const T of terr)
    for(const line of pdTable().split(NLCH)){
      const f = line.trim().split(/\s+/);
      if(f.length < 3) continue;
      if(AS2_XREC === 1 && T.key === 'x' && f[0] === 'L4e' && f[1] === 'L4e'){
        rows.push('x.L4e x.L4e 0 1 0');
        continue;
      }
      const recPlast = AS2_XREC === 2 && T.key === 'x'
        && f[0] === 'L4e' && f[1] === 'L4e';
      const key = f[0] + '|' + f[1];
      const pl = (recPlast || (T.key === 'x' && xPlast.has(key))
        || (T.key !== 'x' && uniPlast.has(key))) ? ' 1' : ' 0';
      // xrecp scales only the association territory's recurrence, so the rest of the column keeps its published pair densities.
      // The unimodal relay is the same pair in the v and a territories.
      const isURelay = T.key !== 'x' && f[0] === 'L4e' && f[1] === 'L2/3e';
      const isXRec = T.key === 'x' && f[0] === 'L4e' && f[1] === 'L4e';
      const pm = (AS2_XRECP !== 1 && isXRec)
        ? +(f[2]*AS2_XRECP).toFixed(4)
        : ((AS2_URELAYP !== 1 && isURelay)
          ? +(f[2]*AS2_URELAYP).toFixed(4) : f[2]);
      const wm = (AS2_XRECW !== 1 && isXRec)
        ? +((+(f[3] || 1))*AS2_XRECW).toFixed(4)
        : ((AS2_URELAYW !== 1 && isURelay)
          ? +((+(f[3] || 1))*AS2_URELAYW).toFixed(4) : (f[3] || '1'));
      rows.push(T.key + '.' + f[0] + ' ' + T.key + '.' + f[1] + ' ' + pm
        + ' ' + wm + pl);
    }
  rows.push('lgn trn 3 1 0', 'trn lgn 3 1 0');
  rows.push('mgb trna 3 1 0', 'trna mgb 3 1 0');
  // five columns so the catch-all is frozen as well as silent: a short row's plasticity column defaults to plastic in the parser
  rows.push('* * 0 1 0');
  const cn = add('connect', 40, 420,
    { radius:2200, sigma:9000, prob:0.05, wExc:WE*EQ, wInh:-WE*G*IQ,
      wdist:1, wsigma:1, cluster:0.2, velocity:300, seed:1,
      table:rows.join(NLCH) });
  wire(cn, 0, chain);

  // ---- lesson stream and encoders ------------------------------------
  // Letters, spoken by the formant synthesizer rather than a recording, presented for a second with a second of gap.
  // The timing is load bearing: at 200 ms on and 100 ms off learning does not happen at all.
  // STDP pairings per synapse go as pre rate times post rate times the window, and at the association site that is 3.64 x 1.14 x 0.02, about 0.083 a second, so a longer gap spends most of the run waiting; at aP 0.003 against wmax 60, 800 pairings move a weight a few percent of its range.
  // No jitter, no size or rotation variation and no recorded voice: nuisance variation is worth adding once something learns, not while asking whether anything can. calCycles 3, not 2: recording waits out a thirty-second settle for the startup transient, which eats the first ten presentations of the first bimodal pass, so with two passes the first ten letters end up with a single bimodal trial each and their centroids are one noisy sample.
  // A third pass puts every letter on at least two trials in every condition, which is what the conjunctive sets are built from. probeEvery 5 rather than 4 because probes land on k = probeEvery-1 mod probeEvery and the alphabet has 26 items: with 4 the shared factor of 2 means only the odd-numbered letters would be seen alone, and 5 is coprime with 26.
  const cur = add('curriculum', 640, 420, { sense:0, set:0, order:1, onMs:1000, offMs:1000,
    jitter:0, scaleVar:0, rotVar:0, acuity:1, probeEvery:5, voice:0, seed:1,
    calCycles:3,
    // the same sweep again every 2400 presentations, about 1.3 sim-hours at 1 s on and 1 s off, so drift in the conjunctive sets can be told from learning within them
    reSweepEvery:2400 });
  // Retinal contrast coding, ON and OFF blocks.
  // Ganglion cells report center-surround difference, not luminance, and for letters that is the difference between a stimulus and a silhouette: measured over the 26 items on this encoder grid, mean pairwise similarity falls from 0.582 to 0.369 and coverage from 0.168 to 0.123.
  // The spot sets are authored against raw luminance because a disc has no interior structure to lose; the retina here is sized 2*chV*chV for the doubled channels.
  const inV = add('input', 190, 420, { map:0, code:1, axis:1,
    cols:chV, rows:chV, arbor:0.7, transient:0.2, jitter:0,
    amp:+(AS2_AMP*AS2_VAMP).toFixed(2),
    lagMs:AS2_LAG > 0 ? AS2_LAG : 0 });
  wire(inV, 0, cn); wire(inV, 1, eyeBox); wire(inV, 2, cur);
  const inA = add('input', 340, 420, { map:1, code:1, axis:2,
    cols:chA, arbor:0.7, transient:0.2, jitter:0,
    amp:+(AS2_AMP*AS2_AAMP).toFixed(2),
    lagMs:AS2_LAG < 0 ? -AS2_LAG : 0 });
  // the sound stream: the same lesson, following the first curriculum's clock, carrying the item as its spoken name
  const curS = add('curriculum', 0, 0, { sense:1 }); wire(curS, 0, cur);
  curS.x = cur.x; curS.y = cur.y + 56;
  wire(inA, 0, inV); wire(inA, 1, earBox); wire(inA, 2, curS);
  group('local wiring  Potjans-Diesmann per territory', [cn], 130);
  group('curriculum and encoders  one clock, two senses', [cur, curS, inV, inA], 55);

  // ---- drive: the tuned per-layer currents, scaled by blockdrive ------
  const KEXT = [[1600/2100, 1500/2100], [1, 1900/2100],
    [2000/2100, 1900/2100], [2900/2100, 1]];
  const TUNED = (F >= 0.5 ? [0.545, 0.476, 0.651, 0.355]
    : [0.5379, 0.4117, 0.5325, 0.2723]);
  const IBASE = 8.0, NOISE = 2.0;
  let prev = inA;
  LAYERS.forEach((L, k) => {
    const band = [];
    // constant drive is masked per territory so the association territory can run at its own operating point (xdrive); the noise floor stays uniform across the block
    terr.forEach((T, ti) => {
      const mask = add('box', 440 + k*150 + ti*36, 500,
        { center:[T.cx, L.y, 0], size:[T.w, L.h, side] });
      band.push(mask);
      const mul = T.key === 'x' ? AS2_XDRIVE : 1;
      const de = add('stimulus', 440 + k*150 + ti*36, 560,
        { mode:0, current:+(IBASE*KEXT[k][0]*TUNED[k]*AS2_DRIVE*AS2_LAYER[k]*mul).toFixed(2),
          spread:AS2_SPREAD, seedSpread:40 + k*3 + ti });
      wire(de, 0, prev); wire(de, 1, mask);
      mask.name = T.key + '.' + L.name + ' drive region'; de.name = 'bias on ' + T.key + '.' + L.name;
      band.push(de);
      prev = de;
    });
    // noise is also masked per territory: the territory strips overhang the nominal block edge (their widths plus the two fixed gaps exceed side at every footprint below one), so a block-sized box would miss the outer edge of both unimodal territories, about a quarter of their cells at footprint 0.05.
    terr.forEach((T, ti) => {
      const nmask = add('box', 440 + k*150 + ti*36, 615,
        { center:[T.cx, L.y, 0], size:[T.w, L.h, side] });
      const ns = add('stimulus', 440 + k*150 + ti*36, 650,
        { mode:3, current:+(1.5*KEXT[k][0]*NOISE).toFixed(2) });
      wire(ns, 0, prev); wire(ns, 1, nmask);
      nmask.name = T.key + '.' + L.name + ' noise region'; ns.name = 'noise on ' + T.key + '.' + L.name;
      band.push(nmask, ns);
      prev = ns;
    });
    group(L.name + ' drive  constant and noise, per territory', band, 55);
  });
  const holds = [['lgn', lgnBall], ['trn', lgnShell], ['mgb', mgbBall], ['trna', mgbShell]];
  const held = [];
  holds.forEach(([tag, geo], k) => {
    const h = add('stimulus', 190 + k*140, 560, { mode:0, current:-6.0, tag });
    wire(h, 0, prev); wire(h, 1, geo);
    held.push(h);
    prev = h;
  });
  const tn = add('stimulus', 760, 560, { mode:3, current:1.2 });
  wire(tn, 0, prev); wire(tn, 1, lgnBall);
  const tn2 = add('stimulus', 900, 560, { mode:3, current:1.2 });
  wire(tn2, 0, tn); wire(tn2, 1, mgbBall);
  group('thalamic hold  keeps the relays below threshold',
    [...held, tn, tn2], 205);

  // ---- probes and checkpoint -----------------------------------------
  const probeDefs = [
    ['retina', eyeBox], ['cochlea', earBox], ['LGN', lgnBall], ['MGB', mgbBall],
  ];
  let pv = tn2, px = 40;
  const probes = [];
  for(const [label, geo] of probeDefs){
    const pb = add('probe', px, 690, { label });
    wire(pb, 0, pv); wire(pb, 1, geo);
    probes.push(pb);
    pv = pb; px += 140;
  }
  const areaProbe = (label, cx0, w0, y, h) => {
    const box = add('box', px, 660, { center:[cx0, y, 0], size:[w0, h, side] });
    const pb = add('probe', px, 720, { label });
    wire(pb, 0, pv); wire(pb, 1, box);
    probes.push(box, pb);
    pv = pb; px += 140;
    return pb;
  };
  areaProbe('v.L4', cV, wV, 650, 200);
  areaProbe('v.L2/3', cV, wV, 900, 300);
  // the auditory side gets the same two stages as the visual side, so a relay failure there is measured as one on the visual side is
  areaProbe('a.L4', cA, wA, 650, 200);
  areaProbe('a.L2/3', cA, wA, 900, 300);
  // Recorded as well as x.L2/3: the two plastic convergent projections land on x.L4e, so this is the population whose synapses learn; x.L2/3 sees it through frozen local hops.
  const pbX4 = areaProbe('x.L4', cX, wX, 650, 200);
  pbX4.params.record = 1;
  const xBox = add('box', px, 660, { center:[cX, 900, 0], size:[wX, 300, side] });
  const pbX = add('probe', px, 720,
    { label:'x.L2/3', record:1, view:1, axis:1, cols:24, rows:24 });
  wire(pbX, 0, pv); wire(pbX, 1, xBox);

  const out = add('checkpoint', px, 780, { steps:2, syn:1, psc:1, tauE:3, tauI:8, tauS:20, tauM:20,
    // aP 0.009: enough weight movement per sim-hour for a change to be visible inside a night rather than inferred from one. aM is left alone: the ratio aP/aM sets where the fixed point sits, and raising both would move the rate without moving the balance. wdep 1 keeps the soft bound, which is what makes a rate this high safe.
    aP:0.009*EQ, aM:0.001, wmax:60*EQ, wdep:1,
    // The Zenke, Agnes and Gerstner 2015 set.
    // Pair STDP and inhibitory homeostasis alone, with trip, het and tin at zero, is the configuration that paper shows does not form stable assemblies.
    // Its result is that the three work as an orchestrated set, not as independent extras, so they go on together or not at all.
    //
    // trip gives potentiation its growth with postsynaptic rate, which pair STDP cannot produce and pairing experiments show (Pfister and Gerstner 2006; Sjostrom 2001).
    // It is the driver.
    //
    // het is the stabilizer: competition among a cell's incoming weights that engages once it fires above its set point.
    // It is also the answer to the rate climbing through a run.
    //
    // tin keeps rarely coincident synapses from collapsing to silence under depression.
    // Small, and the least directly measured of the three by the authors own account.
    trip:0.006*EQ, tauY:114, het:0.001, tin:0.00005*EQ,
    rhoMode:0, iRho:2, iEta:0.01*IQ, scale:0, sEta:0.03 });
  // The measurement, in the graph, between the probes and the checkpoint.
  // Its values are the defaults, so this changes nothing about what is computed; it is here so the settings are visible and editable in the scene rather than living as constants in the trainer.
  const anz = add('analysis', 880, 545, {});
  wire(anz, 0, pbX);
  wire(out, 0, anz);
  // The decode curve on the convergence area, which is what this scene is for: whether the two senses come to name the same letter.
  // The chart reads the checkpoints as they are written, so it fills in during a run and can be read back afterwards from any run in the project.
  const cha = add('chart', 1120, 780, { plot:0, pop:'x.L4', field:'decode', height:150 });
  wire(cha, 0, out);
  group('readout  what the observer measures', [...probes, xBox, pbX, anz, cha], 280);
  group('checkpoint  the simulation and its plasticity', [out], 0);
}

// Learning lab.
// The node-built counterpart of the testbed in lab.js: same question, same measures, but computed through connect so distances, delays, the pair table and the weight distribution are the real ones.
// Deliberately small, because the point is to answer in seconds and then scale, and LAB_SCALE is the knob that scales it.
let LAB_SCALE = 1, LAB_AMP = 45, LAB_SEED = 0;
// Seed offset for replication.
// Shifts every scatter seed, the connect seed and the lesson order together, so a second run is a genuinely different network seeing a different item order rather than the same one twice.
export function setLabSeed(k){ LAB_SEED = Math.max(0, k | 0); }
function getLabSeed(){ return LAB_SEED; }
export function setLabScale(k){ LAB_SCALE = Math.max(1, +k || 1); }
function getLabScale(){ return LAB_SCALE; }
// Sensory drive per channel.
// Low enough that one modality alone leaves the patch below threshold is the interesting setting: that is what makes a cortical cell a coincidence detector, and it is the only regime in which cross-modal binding has anything to do.
// Too high and either sense alone reproduces the joint pattern before any learning happens.
export function setLabAmp(a){ LAB_AMP = Math.max(1, +a || 45); }
function getLabAmp(){ return LAB_AMP; }

function learningLab(ed){
  const { add, wire, note, group } = api(ed);
  const K = LAB_SCALE, n = v => Math.max(4, Math.round(v*K));
  // Density is held constant as the scale grows.
  // Counts scale by K, so area scales by K and every extent and separation in the sheet plane scales by its square root; thickness does not, because cortex gets wider rather than deeper.
  // Scaling the counts alone makes the tissue denser rather than larger, with distances shrinking relative to connectivity: measured that way, association firing moved from 1.6 Hz at scale 1 to 3.7 Hz at scale 4.
  // In-degree stays put because the pair table divides its probabilities by K, and a kernel of radius R*G over a sheet of constant density contains K times as many cells.
  const G = Math.sqrt(K);
  const ctr = (x, y, z) => [x*G, y, z*G];
  const ext = (x, y, z) => [x*G, y, z*G];
  // Conduction velocity scales with the geometry so delays stay put.
  // A larger sheet would have longer delays, but letting them grow here would vary two things at once and a scaling result could not say which one mattered.
  // Delay growth is its own experiment.
  const VEL = 200*G;
  note(20, 'Learning lab\n\nA paired-modality world for testing whether the learning rule builds a cross-modal association. One item is one place on the retina lighting up together with one tone. There are twelve items, and nothing else varies.');
  note(200, 'There are three stages, which is roughly how the senses meet: separate unimodal areas, V1 for sight and A1 for sound, and then convergence into an association area that sees both. Multisensory neurons are found in association cortex and not in primary areas (Bruce, Desimone and Gross 1981), and neurons that come to respond to both members of a learned arbitrary pair appear there too (Sakai and Miyashita 1991).');
  note(430, 'The association area projects back to both unimodal areas. Feedback projections are ubiquitous in cortex and denser than the feedforward ones. Here they are also what makes completion possible: without them, sound could never produce activity in the visual area, and there would be no way to ask whether it does.');
  note(640, 'The scene makes two simplifications. Primary cortex is not strictly unimodal: there are direct projections from auditory cortex to V1 (Falchier et al. 2002) and sound modulates V1 responses (Iurilli et al. 2012). Those inputs are left out, since they are mostly modulatory and do not drive the cells. Learning an arbitrary pairing also leans on perirhinal cortex and hippocampus, and the scene has neither.');
  note(880, 'To scale the scene, use labscale on the training page and leave the counts as they are. The pair table divides every probability by the scale, so the populations grow while in-degree stays put, and a scaling test measures one thing at a time.');

  // sensory sheets
  const eye = add('box', 40, 40, { center:ctr(-1400, 0, -520), size:ext(640, 40, 640) });
  const rgc = add('scatter', 40, 110, { count:n(256), type:RS, tag:'retina', seed:1 + LAB_SEED });
  wire(rgc, 0, eye);
  const ear = add('box', 240, 40, { center:ctr(-1400, 0, 520), size:ext(640, 40, 160) });
  const coch = add('scatter', 240, 110, { count:n(128), type:RS, tag:'cochlea', seed:2 + LAB_SEED });
  wire(coch, 0, ear);

  // unimodal areas
  const v1b = add('box', 440, 40, { center:ctr(-500, 0, -520), size:ext(420, 180, 420) });
  const v1e = add('scatter', 440, 110, { count:n(80), type:RS, tag:'v1e', seed:3 + LAB_SEED });
  wire(v1e, 0, v1b);
  const v1i = add('scatter', 600, 110, { count:n(20), type:FS, tag:'v1i', seed:4 + LAB_SEED });
  wire(v1i, 0, v1b);
  const a1b = add('box', 760, 40, { center:ctr(-500, 0, 520), size:ext(420, 180, 420) });
  const a1e = add('scatter', 760, 110, { count:n(80), type:RS, tag:'a1e', seed:5 + LAB_SEED });
  wire(a1e, 0, a1b);
  const a1i = add('scatter', 920, 110, { count:n(20), type:FS, tag:'a1i', seed:6 + LAB_SEED });
  wire(a1i, 0, a1b);

  // association area
  const asb = add('box', 1080, 40, { center:ctr(400, 0, 0), size:ext(460, 180, 460) });
  const asse = add('scatter', 1080, 110, { count:n(120), type:RS, tag:'asse', seed:7 + LAB_SEED });
  wire(asse, 0, asb);
  const assi = add('scatter', 1240, 110, { count:n(30), type:FS, tag:'assi', seed:8 + LAB_SEED });
  wire(assi, 0, asb);

  const m1 = add('gather', 240, 190); wire(m1, 0, rgc); wire(m1, 1, coch);
  const m2 = add('gather', 640, 190); wire(m2, 0, v1e); wire(m2, 1, v1i);
  wire(m2, 2, a1e); wire(m2, 3, a1i);
  const m3 = add('gather', 1080, 190); wire(m3, 0, asse); wire(m3, 1, assi);
  const m4 = add('gather', 640, 240); wire(m4, 0, m1); wire(m4, 1, m2); wire(m4, 2, m3);

  // Everything is off unless listed.
  // The senses stay apart until the association area, which is the whole point of the architecture: if both reached the same cells directly there would be no visual pattern and no auditory pattern to be distinct from each other, and cross-modal transfer would read at chance because it had nothing to compare.
  const P = v => (v/K).toFixed(4);
  const table = ['* * 0',
    // The fifth column gates plasticity per pair class.
    // Only the two convergent projections are plastic; the sensory projections, the recurrence, the feedback and every E/I loop are frozen.
    // That is the minimal thing that can bind.
    // When a spot and its tone occur together an association cell fires, and both the visual and the auditory afferents active at that moment potentiate, so a cell that started out driven by one comes to be driven by the other.
    // Neurons that respond to both members of a learned arbitrary pair are what Sakai and Miyashita 1991 report in inferotemporal cortex.
    // With everything learning at once the association area collapses, its decoding falling to chance while its rate wanders.
    'retina v1e ' + P(2.0) + ' 1 0',
    'cochlea a1e ' + P(2.0) + ' 1 0',
    'v1e asse ' + P(3.0) + ' 1 1',
    // The auditory projection is deliberately weaker than the visual one.
    // With sound leading and both able to fire an association cell, the cell fires from the tone and the later visual input lands after the spike, which is the depressing side of the window: the two senses compete and the area turns auditory.
    // Weak and leading is the conditioning arrangement instead.
    // Vision drives the spike, the tone arrives just before it, and the tone's afferents are the ones that potentiate, so the tone gains the ability to evoke a pattern it could not evoke on its own.
    'a1e asse ' + P(1.0) + ' 1 1',
    'asse asse ' + P(1.5) + ' 1 0',
    'asse v1e ' + P(0.6) + ' 1 0',
    'asse a1e ' + P(0.6) + ' 1 0',
    'v1e v1i ' + P(2.4) + ' 1 0', 'v1i v1e ' + P(2.4) + ' 1 0',
    'a1e a1i ' + P(2.4) + ' 1 0', 'a1i a1e ' + P(2.4) + ' 1 0',
    // The association area's own inhibition is plastic, and it is the only inhibition that is.
    // Sparse selective responses are not a setting to dial in, they are what feedback inhibition produces when each cell is held to a low rate: Vogels homeostasis weakens inhibition where a cell is too quiet and strengthens it where it is too active, so a cell ends up firing for the few items that drive it hardest and not the rest.
    // Frozen, no target can be enforced at all.
    // Only the inhibitory half of the loop learns.
    // Interneurons fire for everything, so a plastic E to I synapse under the excitatory rule potentiates to saturation within sim-hours and the Vogels weights double to hold the target against it.
    // Vogels et al. 2011 keep E to I static and let I to E carry the control, which is what this is.
    'asse assi ' + P(2.0) + ' 1 0', 'assi asse ' + P(1.0) + ' 1 1',
  ].join(String.fromCharCode(10));
  // wExc 5.
  // A cell that fires once in seven seconds meets almost no pairings inside a one-second presentation.
  // Input amplitude is not the lever (a spot lights a handful of retinal cells that already fire near their ceiling); wExc 5 sets the operating point (V1 1.2 to 1.6 Hz, A1 1.3 to 1.8, association 1.4 to 2.1, decode 0.98).
  const cn = add('connect', 640, 300,
    { radius:1700*G, sigma:850*G, prob:0.35, wExc:5*EQ, wInh:-15*IQ,
      wdist:1, wsigma:0.9, cluster:0, velocity:VEL, seed:1 + LAB_SEED, table });
  wire(cn, 0, m4);

  // One lesson stream, two encoders, one clock.
  // Timing is load bearing rather than cosmetic.
  // 200 ms on with a 100 ms gap blocks learning outright: STDP pairings per synapse go as pre rate times post rate times the window, and at 200 ms there are not enough of them for a weight to move.
  // 1 s on with a 1 s gap is the schedule that learns.
  //
  // probeEvery 5, not 4.
  // Probes land on k = probeEvery-1 mod probeEvery, and this set has 12 items, so a period of 4 shares a factor of 4 with the set and only items 3, 7 and 11 would be presented alone.
  // 5 is coprime with 12 and reaches every item, as with the alphabet.
  //
  // calCycles sweeps every item in all three conditions before anything learns, which is what the conjunctive sets are built from; reSweepEvery repeats that sweep so drift can be told apart from learning.
  // 600 rather than the alphabet's 2400 because this scene exists to be iterated on.
  const cur = add('curriculum', 1400, 300, { sense:0, set:8, order:1, onMs:1000, offMs:1000,
    jitter:0, scaleVar:0, rotVar:0, acuity:1, probeEvery:5, seed:1 + LAB_SEED,
    calCycles:3, reSweepEvery:600 });
  const side = Math.max(8, Math.round(16*Math.sqrt(K)));
  // Sound leads sight by 15 ms, which has to sit inside the plasticity window to do anything: at tauS 20 ms a 40 ms lead is two time constants out and carries about an eighth of the potentiation a lead near the peak does.
  // Simultaneous arrival is the worst case for a spike-timing rule: the connections between the two streams get equal potentiation and depression and nothing accumulates.
  // A lead gives the association a direction, and a cue preceding its partner is also what paired-associate learning looks like.
  const inV = add('input', 440, 380, { map:0, code:0, axis:1,
    cols:side, rows:side, arbor:0.3, transient:0.2, jitter:0, amp:LAB_AMP,
    lagMs:15 });
  wire(inV, 0, cn); wire(inV, 1, eye); wire(inV, 2, cur);
  const inA = add('input', 760, 380, { map:1, code:0, axis:0,
    cols:side, arbor:0.3, transient:0.2, jitter:0, amp:LAB_AMP });
  // the sound stream: the same lesson, following the first curriculum's clock, carrying the item as its spoken name
  const curS = add('curriculum', 0, 0, { sense:1 }); wire(curS, 0, cur);
  curS.x = cur.x; curS.y = cur.y + 56;
  wire(inA, 0, inV); wire(inA, 1, ear); wire(inA, 2, curS);

  // fluctuation only, no item-independent DC
  let prev = inA;
  const drives = [];
  [v1b, a1b, asb].forEach((geo, k) => {
    const s = add('stimulus', 440 + k*180, 450, { mode:3, current:2.0 });
    wire(s, 0, prev); wire(s, 1, geo);
    drives.push(s);
    prev = s;
  });

  const pbR = add('probe', 240, 530, { label:'retina' });
  wire(pbR, 0, prev); wire(pbR, 1, eye);
  const pbC = add('probe', 400, 530, { label:'cochlea' });
  wire(pbC, 0, pbR); wire(pbC, 1, ear);
  // V1 records too: sound alone reaching V1 through the association area's feedback is the measure that separates paired from scrambled most clearly, and it needs the V1 trials.
  const pbV = add('probe', 560, 530, { label:'V1', view:1, axis:1, cols:side, rows:side, record:1 });
  wire(pbV, 0, pbC); wire(pbV, 1, v1b);
  const pbA = add('probe', 720, 530, { label:'A1' });
  wire(pbA, 0, pbV); wire(pbA, 1, a1b);
  const pbX = add('probe', 880, 530, { label:'assoc', record:1 });
  wire(pbX, 0, pbA); wire(pbX, 1, asb);

  // What the observer asks of the recording.
  // The default expression is the cross-modal question this scene was built around, cells the pair drives that neither sense drives alone, but it is a field rather than a fixed function: the same scene answers a different question by being typed into rather than by being edited. cap is small because the whole association area is 120 cells at scale 1, so sampling is not the constraint here that it is in a block of a hundred thousand.
  const an = add('analysis', 1040, 530,
    { cap:2048, setExpr:'both and not sight and not sound',
      probeWith:'sight, sound', minCells:3, minTrials:2 });
  wire(an, 0, pbX);
  // The run trace, in the app: decoding at the association area against simulated time.
  // The chance line comes with it, because a decode number without one is not a claim about anything.
  const cha = add('chart', 1200, 610,
    { plot:0, pop:'assoc', field:'decode', height:150 });

  const out = add('checkpoint', 880, 610, { steps:2, syn:1, psc:1, tauE:3, tauI:8, tauS:20, tauM:20,
    // wmax has to clear the inhibitory weights and their lognormal tail, not just the excitatory ones.
    // At wmax 10 against wInh -15 half the inhibitory population read as pinned at the clamp before a single step of plasticity had run.
    // The excitatory fixed point is aP/aM and does not move with wmax.
    // A fixed low target rather than each neuron's own measured baseline.
    // Measured set points freeze whatever activity exists, which is the right choice for a column being kept in its operating band and the wrong one here, where the existing activity is the broad unselective firing this scene is meant to get away from.
    // The target is 2 Hz with a moderate learning rate: 1 Hz at iEta 0.02 silences the association area outright (0.13 Hz, decoding at chance).
    // The sparse direction (de Kock and Sakmann 2009; Potjans and Diesmann 2014) is kept, but at the rate this circuit survives. aP 0.009: this scene is for iterating, and a rule that takes a night to show a change cannot be iterated on.
    // The ratio aP/aM sets where the fixed point lands, so aM stays put and only the rate moves. wdep 1 keeps the soft bound, which is what makes a rate this high safe.
    aP:0.009*EQ, aM:0.001, wmax:60*EQ, wdep:1,
    // The Zenke, Agnes and Gerstner 2015 set.
    // Pair STDP and inhibitory homeostasis alone, with trip, het and tin at zero, is precisely the configuration that paper shows does not form stable assemblies.
    // The three are an orchestrated set rather than independent extras, so they go on together.
    //
    // trip gives potentiation its growth with postsynaptic rate, which pair STDP cannot produce and pairing experiments show (Pfister and Gerstner 2006). het is the stabilizer, competition among a cell's incoming weights once it fires above its set point. tin is the slow drift that keeps a silent synapse from being stranded at zero forever.
    trip:0.006*EQ, tauY:114, het:0.001, tin:0.00005*EQ,
    rhoMode:0, iRho:2, iEta:0.01*IQ, scale:0, sEta:0.03 });
  wire(out, 0, an);
  wire(cha, 0, out);
  group('retina', [eye, rgc], 205);
  group('cochlea', [ear, coch], 25);
  group('V1  visual unimodal area', [v1b, v1e, v1i], 205);
  group('A1  auditory unimodal area', [a1b, a1e, a1i], 25);
  group('association area  sees both senses', [asb, asse, assi], 130);
  group('gather: every population into one stream', [m1, m2, m3, m4], 280);
  group('wiring  only the two convergent projections learn', [cn], 130);
  group('curriculum and encoders  tone leads sight by 15 ms',
    [cur, curS, inV, inA], 55);
  group('background fluctuation', drives, 55);
  group('readout  what the observer measures',
    [pbR, pbC, pbV, pbA, pbX, an, cha], 280);
  group('checkpoint  the simulation and its plasticity', [out], 0);
}

// ---------------------------------------------------------------------------
// Binding bench.
// Built from the mechanism the assembly literature uses: two senses, each coding an item as a small selective set in its own area, converging on one association area whose recurrent excitation is plastic.
// A pair of stimuli drives two sets there at once; the recurrent synapses between them are the only ones that are ever co-active, so they potentiate and the two sets become one assembly (Hebb 1949; Litwin-Kumar and Doiron 2014; Zenke, Agnes and Gerstner 2015).
// Afterwards either sense alone recruits the whole assembly through the recurrence, which is what the cross-modal decoder asks for.
// Sparse codes come from low convergence and feedback inhibition under Vogels et al. 2011, and there is no background drive: the pattern a probe evokes is the stimulus and nothing else.
// Scale for the bench.
// Counts grow by K and the layout by its square root, as the lab does, and every pathway holds its in-degree (probabilities divided by K), the recurrence included.
// Letting the recurrence grow with the population instead (measured 2026-09-02, K 4): with four times the recurrent afferents the association area is bistable, quiet at 0.05 Hz below a recurrent weight of about 0.25 and dense and unselective above it, whatever the inhibition; the middle that scale 1 sits in does not exist at that in-degree.
// Held everywhere, the scale-4 network matches scale 1 population by population and the gain is decoder stability.
let BENCH_SCALE = 1;
export function setBenchScale(k){ BENCH_SCALE = Math.max(1, +k || 1); }
function bindingBench(ed){
  const { add, wire, note, group } = api(ed);
  const K = BENCH_SCALE, G = Math.sqrt(K), n = v => Math.max(4, Math.round(v*K));
  const ctr = (x, y, z) => [x*G, y, z*G];
  const ext = (x, y, z) => [x*G, y, z*G];
  // The encoder grid does not scale with the sheet.
  // A spot covers a fixed number of channels, so a grid that grew with the sheet would light a fixed number of cells in a sheet K times larger, and every downstream code would be K times sparser (measured at K 4 with a 32-wide grid: retina 0.41 Hz against 0.83, association 0.05 against 0.7).
  // At 16 channels each channel reaches K times as many cells and the lit fraction is the same at every scale.
  const side = 16;
  note(20, 'Binding bench\n\nA visual place and a tone together are one item. Each sense codes an item as a small set of cells in its own area, and both areas converge on one association area whose recurrent synapses learn. Cells driven by the place and cells driven by the tone fire together and wire together, and afterwards either sense alone recruits the whole assembly.');
  note(430, 'The codes are kept sparse, with few afferents per cell, strong weights, and feedback inhibition held to a low target. A cell that fires for every item cannot take part in an assembly for one of them. There is no background drive, so a probe evokes the stimulus pattern and nothing else.');
  note(880, 'The readout is the cross-classification test (Kaplan, Man and Greening 2015): a decoder is trained on sight-alone responses in the association area and tested on sound-alone responses, and then the reverse. Chance is one in twelve. V1 records too, so the same test tells you whether the tone reaches the visual area through the feedback.');

  // sensory sheets
  const eye = add('box', 40, 40, { center:ctr(-1500, 0, -600), size:ext(500, 40, 500) });
  const ret = add('scatter', 40, 110, { count:n(256), type:RS, tag:'ret', seed:11 });
  wire(ret, 0, eye);
  const ear = add('box', 240, 40, { center:ctr(-1500, 0, 600), size:ext(640, 40, 160) });
  const coc = add('scatter', 240, 110, { count:n(128), type:RS, tag:'coc', seed:12 });
  wire(coc, 0, ear);

  // unimodal areas
  const v1b = add('box', 440, 40, { center:ctr(-500, 0, -600), size:ext(500, 200, 500) });
  const v1e = add('scatter', 440, 110, { count:n(200), type:RS, tag:'v1e', seed:13 });
  wire(v1e, 0, v1b);
  const v1i = add('scatter', 600, 110, { count:n(50), type:FS, tag:'v1i', seed:14 });
  wire(v1i, 0, v1b);
  const a1b = add('box', 760, 40, { center:ctr(-500, 0, 600), size:ext(500, 200, 500) });
  const a1e = add('scatter', 760, 110, { count:n(200), type:RS, tag:'a1e', seed:15 });
  wire(a1e, 0, a1b);
  const a1i = add('scatter', 920, 110, { count:n(50), type:FS, tag:'a1i', seed:16 });
  wire(a1i, 0, a1b);

  // association area
  const xb = add('box', 1080, 40, { center:ctr(600, 0, 0), size:ext(600, 200, 600) });
  const xe = add('scatter', 1080, 110, { count:n(400), type:RS, tag:'xe', seed:17 });
  wire(xe, 0, xb);
  const xi = add('scatter', 1240, 110, { count:n(100), type:FS, tag:'xi', seed:18 });
  wire(xi, 0, xb);

  const m1 = add('gather', 240, 190); wire(m1, 0, ret); wire(m1, 1, coc);
  const m2 = add('gather', 640, 190); wire(m2, 0, v1e); wire(m2, 1, v1i);
  wire(m2, 2, a1e); wire(m2, 3, a1i);
  const m3 = add('gather', 1080, 190); wire(m3, 0, xe); wire(m3, 1, xi);
  const m4 = add('gather', 640, 240); wire(m4, 0, m1); wire(m4, 1, m2); wire(m4, 2, m3);

  // Rows are "pre post probMul wMul plastic" on a base probability of 0.15 with a kernel wide enough that every pair is in reach, so the numbers below are the connectivity.
  // Convergence into the unimodal areas is low (about 13 afferents per cell) and the weights strong, so a stimulus that lights a handful of sensory cells drives the few cortical cells that happen to receive two of them: a sparse, selective code.
  // The same arrangement feeds the association area from both sides.
  // Its recurrence is the plastic substrate; the two convergent projections learn as well; every E to I pathway is frozen (Vogels et al. 2011 keep it so) and only the association area's I to E pathway carries the rate control.
  // Weights are in the peak-current convention (psc 0, the default), a third of the charge the psc 1 scenes' numbers carry.
  // Measured 2026-09-02 with the curriculum driving: a single afferent above about 2 fires a pyramidal cell on its own, which makes every code dense; the sensory rows sit at 3.9 and 3.3 so a cell needs two coincident inputs, and the convergent rows at 0.8 with nine afferents a sense.
  // The feedforward rows are frozen: a plastic convergent projection potentiates every co-active input to the rule's fixed point within minutes and the sparse code becomes a dense rate code (sparseness 0.47 to 0.08 in ten sim-minutes).
  // The recurrence is the only excitatory pathway that learns.
  // Feedback to the unimodal areas is off: with it on, the loop through the association area ran the whole network as one oscillator.
  // Scaling: every row divides its probability by K, so convergence and the E/I loops keep their in-degree while the populations grow (measured at K 4 with all rows growing: the unimodal interneurons sat at their ceiling, V1 and A1 lost their item codes, decode 0.07, and the association area went silent; with only the recurrence growing it was bistable, see the note on the scale above).
  const P = v => (v/K).toFixed(4);
  const table = ['* * 0',
    'ret v1e ' + P(0.5) + ' 3 0',
    'coc a1e ' + P(0.66) + ' 2.5 0',
    'v1e xe ' + P(0.3) + ' 0.6 0',
    'a1e xe ' + P(0.3) + ' 0.6 0',
    'xe xe ' + P(0.5) + ' 0.3 1',
    'xe v1e 0 1 0',
    'xe a1e 0 1 0',
    'v1e v1i ' + P(2) + ' 1 0', 'v1i v1e ' + P(2) + ' 1 0',
    'a1e a1i ' + P(2) + ' 1 0', 'a1i a1e ' + P(2) + ' 1 0',
    // The inhibitory pathway carries its own rule (the plasticity node below) because a rule's weight ceiling applies to every synapse under it, inhibitory included: under the checkpoint's rule with wmax 2.5 to 4 the wired inhibition of 6 was clamped to the ceiling at the first plastic step, so lowering the ceiling to tame the recurrence weakened the inhibition meant to hold it (found 2026-09-02 at scale 4).
    'xe xi ' + P(1) + ' 0.5 0', 'xi xe ' + P(2) + ' 1 inh',
  ].join(String.fromCharCode(10));
  // The ceiling of 4 is load-bearing.
  // Measured 2026-09-02 at scale 1: with it the association area holds a sparse code (0.7 to 1.2 Hz, decode 1.0); at 30, with the wired weight at 6 or at 3, the area goes dense and unselective within twenty sim-minutes (6 to 8 Hz, selectivity 0).
  // The Vogels rule at a 0.5 Hz target weakens inhibition on every cell below target, and the low ceiling is what bounds that.
  const inh = add('plasticity', 460, 300, { name:'inh', wmax:4, wdep:1, iEta:0.02,
    aP:0, aM:0, trip:0, het:0, tin:0 });
  wire(inh, 0, m4);
  const cn = add('connect', 640, 300,
    { radius:4000*G, sigma:3000*G, prob:0.15, wExc:1.3, wInh:-6,
      wdist:1, wsigma:0.5, cluster:0, velocity:200*G, seed:19, table });
  wire(cn, 0, inh);

  // one lesson stream, two encoders, one clock; sight lags the tone by 10 ms
  const cur = add('curriculum', 1400, 300, { sense:0, set:8, order:1, onMs:250, offMs:250,
    jitter:0, scaleVar:0, rotVar:0, acuity:1, probeEvery:5, seed:21,
    calCycles:3, reSweepEvery:600 });
  const inV = add('input', 440, 380, { map:0, code:0, axis:1,
    cols:side, rows:side, arbor:1.0, transient:0.2, jitter:0, amp:600, lagMs:10 });
  wire(inV, 0, cn); wire(inV, 1, eye); wire(inV, 2, cur);
  const inA = add('input', 760, 380, { map:1, code:0, axis:0,
    cols:side, arbor:1.0, transient:0.2, jitter:0, amp:600 });
  // the sound stream: the same lesson, following the first curriculum's clock, carrying the item as its spoken name
  const curS = add('curriculum', 0, 0, { sense:1 }); wire(curS, 0, cur);
  curS.x = cur.x; curS.y = cur.y + 56;
  wire(inA, 0, inV); wire(inA, 1, ear); wire(inA, 2, curS);

  const pbR = add('probe', 240, 530, { label:'retina' });
  wire(pbR, 0, inA); wire(pbR, 1, eye);
  const pbC = add('probe', 400, 530, { label:'cochlea' });
  wire(pbC, 0, pbR); wire(pbC, 1, ear);
  const pbV = add('probe', 560, 530, { label:'V1', view:1, axis:1, cols:side, rows:side, record:1 });
  wire(pbV, 0, pbC); wire(pbV, 1, v1b);
  const pbA = add('probe', 720, 530, { label:'A1', record:1 });
  wire(pbA, 0, pbV); wire(pbA, 1, a1b);
  const pbX = add('probe', 880, 530, { label:'assoc', record:1 });
  wire(pbX, 0, pbA); wire(pbX, 1, xb);

  const an = add('analysis', 1040, 530,
    { cap:4096, setExpr:'both', probeWith:'sight, sound', minCells:3, minTrials:2 });
  wire(an, 0, pbX);
  const cha = add('chart', 1200, 610,
    { plot:0, pop:'assoc', field:'decode', height:150 });

  // Pair rule with soft bounds: additive potentiation, depression scaled by the weight, whose fixed point for a co-active pair is (A+ / A-) times the pre-to-post rate ratio (Gutig et al. 2003 at mu 1).
  // A+ / A- of 3 puts a co-active recurrent pair near 3, seven times its baseline of 0.4, and a pair that is never co-active stays where it was. wmax 4 is the soft ceiling: at wmax 40 the recurrence ran away at thirty sim-minutes (rate 0.7 to 2.3 Hz, synchrony index 500 to 1300, decode 1.0 to 0.13); at 2.5 only one direction of the cross-modal test moved.
  // The inhibitory target is low and its rate fast so the codes stay sparse.
  const out = add('checkpoint', 880, 610, { steps:2, syn:1, tauE:3, tauI:8,
    plast:1, aP:0.003, aM:0.001, wmax:4, wdep:1,
    trip:0, het:0, tin:0,
    rhoMode:0, iRho:0.5, iEta:0.02, scale:0,
    trHours:6, trCkptMin:30, trBrainMin:360, trEngine:1 });
  wire(out, 0, an);
  wire(cha, 0, out);
  group('retina', [eye, ret], 205);
  group('cochlea', [ear, coc], 25);
  group('V1  sparse visual code', [v1b, v1e, v1i], 205);
  group('A1  sparse auditory code', [a1b, a1e, a1i], 25);
  group('association area  plastic recurrence', [xb, xe, xi], 130);
  group('gather', [m1, m2, m3, m4], 280);
  group('wiring  low convergence, strong weights', [inh, cn], 130);
  group('curriculum and encoders', [cur, curS, inV, inA], 55);
  group('readout', [pbR, pbC, pbV, pbA, pbX, an, cha], 280);
  group('checkpoint', [out], 0);
}

// ---------------------------------------------------------------------------
// The weave.
// Built to test the rule rather than an anatomy.
//
// The other cross-modal scenes are small brains: two sensory surfaces, two unimodal areas, a convergence zone, a hierarchy.
// That shape carries a dozen assumptions, and when the answer comes out null there is no way to say which of them was wrong.
// This scene has none of it.
// There is one piece of tissue and two ways into it.
// A noise field picks a scattered third of the cells for one channel and a second field with a different seed picks a third for the other, so the two sets are woven through each other, share their neighbors, and are told apart only by which cells an input touches.
// Nothing is upstream of anything.
// The only plastic pathway is the tissue's own recurrence, so if the two codes come to predict each other it happened through synapses between cells that fired together and through nothing else.
//
// Three things here are taken from Pokorny, Ison, Rao, Legenstein, Papadimitriou and Maass 2019 (Cerebral Cortex 29(8):3577-3589, STDP forms associations between memory traces in networks of spiking neurons), which is the closest published result to what these runs have been asking for.
// Their network is 432 excitatory and 108 inhibitory spiking neurons with dense uniform recurrence, triplet STDP and short-term plasticity, and in it STDP alone builds an association between two assemblies.
// So the recurrence here is dense and flat rather than sparse and distance-tapered, the triplet and short-term terms are on, and the protocol has an order to it.
//
// They present each item alone first, until an assembly has formed for it, and only then present two items together; driving both channels from the first millisecond asks the rule to build two codes and the link between them at once out of a network that has neither.
// The training blocks parameter on the curriculum below reads '20 A; 20 B; 80 AB': channel A alone for twenty simulated minutes, then channel B alone, then the two together for eighty.
// The one-channel probes run through all three blocks unchanged, so the association measured during the pairing block has, in the same network and the same run, the two blocks before it as its own control.
// A rise that begins where the pairing begins is an association.
// A line already up during familiarization is the probe measuring something else, which a single number at the end of a run cannot tell apart.
function weave(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'The weave\n\nOne piece of tissue with two ways in. A noise field (patches, coverage 0.4) picks scattered patches of the cells for channel A, and a second field with a different seed picks other patches for channel B. The two sets are woven through each other and share their neighbors. There are no areas and no hierarchy, and the only plastic pathway is the recurrence.');
  note(560, 'The recurrence is dense and flat, with the triplet and short-term terms on, after Pokorny et al. 2019, where spike timing plasticity alone associates two assemblies in a recurrent network of about this size.');
  note(830, 'The training runs in blocks: channel A alone for 20 simulated minutes, then B alone for 20, then both together for 80. The one-channel probes run unchanged through every block.');

  const vol = add('box', 640, 40, { center:[0, 0, 0], size:[1200, 360, 1200] });
  const cells = add('scatter', 560, 110, { count:800, type:RS, tag:'weave', seed:31 });
  wire(cells, 0, vol);
  const inh = add('scatter', 720, 110, { count:200, type:FS, tag:'weavei', seed:32 });
  wire(inh, 0, vol);

  // The two channels are subsets of one tissue, not places in it.
  // A noise field at this feature size picks patches of a few dozen cells scattered through the volume; two seeds give two sets that interleave and overlap only by chance.
  const fieldA = add('noisefield', 260, 90, { center:[0, 0, 0], size:[1200, 360, 1200],
    pattern:1, scale:240, coverage:0.4, seed:7 });
  const fieldB = add('noisefield', 1020, 90, { center:[0, 0, 0], size:[1200, 360, 1200],
    pattern:1, scale:240, coverage:0.4, seed:23 });

  const m1 = add('gather', 640, 190); wire(m1, 0, cells); wire(m1, 1, inh);

  // Dense and flat: sigma far larger than the volume, so the Gaussian is level across it and a cell's partners are drawn from the whole tissue rather than from its neighborhood.
  // Uniform recurrence is what the published result runs on, and it is also what a woven code needs, since two cells carrying the same item can be a millimeter apart.
  const table = ['* * 0',
    'weave weave 0.8 0.25 1',              // the only plastic pathway
    'weave weavei 0.8 0.5 0', 'weavei weave 0.8 1 inh',
    'weavei weavei 0.5 0.5 0',
  ].join('\n');
  const inhRule = add('plasticity', 430, 250, { name:'inh', wmax:4, wdep:1, iEta:0.02,
    aP:0, aM:0, trip:0, het:0, tin:0 });
  wire(inhRule, 0, m1);
  const cn = add('connect', 640, 300,
    { radius:2600, sigma:4000, prob:0.6, wExc:1.3, wInh:-1.5,
      wdist:0, cluster:0, velocity:200, seed:19, table });
  wire(cn, 0, inhRule);

  // One clock, two encoders, two masks.
  // The mappings differ on purpose: channel A bins its cells in two dimensions and channel B in one, so the two codes are as unlike each other as a picture and a chord without either being a picture or a chord.
  // B arrives 10 ms after A, which is the asymmetry the spike timing window needs to have anything to work with.
  const cur = add('curriculum', 1400, 250, { sense:0, set:8, order:1, onMs:250, offMs:250,
    jitter:0, scaleVar:0, rotVar:0, acuity:1, probeEvery:5, seed:21,
    calCycles:3, reSweepEvery:0, condNames:'both, A, B',
    phases:'20 A; 20 B; 80 AB' });
  const inA = add('input', 420, 380, { map:0, code:0, axis:1,
    cols:5, rows:5, arbor:1.0, transient:0.2, jitter:0, amp:600, lagMs:0 });
  wire(inA, 0, cn); wire(inA, 1, fieldA); wire(inA, 2, cur);
  const inB = add('input', 800, 380, { map:1, code:0, axis:0,
    cols:12, arbor:1.0, transient:0.2, jitter:0, amp:600, lagMs:10 });
  // the sound stream: the same lesson, following the first curriculum's clock, carrying the item as its spoken name
  const curS = add('curriculum', 0, 0, { sense:1 }); wire(curS, 0, cur);
  curS.x = cur.x; curS.y = cur.y + 56;
  wire(inB, 0, inA); wire(inB, 1, fieldB); wire(inB, 2, curS);

  // the excitatory cells only: the readout is a claim about the code the tissue carries, and the fast-spiking cells are not part of it
  const pbW = add('probe', 640, 530, { label:'weave', tag:'weave', record:1 });
  wire(pbW, 0, inB); wire(pbW, 1, vol);
  const an = add('analysis', 900, 530,
    { cap:4096, setExpr:'both', probeWith:'A, B', minCells:3, minTrials:2 });
  wire(an, 0, pbW);

  const out = add('checkpoint', 640, 610, { steps:2, engine:0, syn:1, tauE:3, tauI:8,
    plast:1, aP:0.003, aM:0.001, tauS:16.8, tauM:33.7, wmax:4, wdep:1,
    trip:0.002, tauY:114, stp:0.5, stpOrder:1, stpU:0.2, stpTauD:200, stpTauF:600,
    het:0.001, tin:0, rhoMode:0, iRho:3, iEta:0.02, scale:0,
    trHours:2, trCkptMin:5, trBrainMin:20, trReps:5, trSweep:1, trEngine:1 });
  wire(out, 0, an);
  const cha = add('chart', 940, 690, { plot:0, pop:'weave', field:'decode', height:150 });
  wire(cha, 0, out);

  group('one tissue', [vol, cells, inh], 130);
  group('channel A: scattered patches of it', [fieldA], 205);
  group('channel B: other patches, woven through', [fieldB], 25);
  group('dense flat recurrence, only it learns', [m1, inhRule, cn], 130);
  group('two ways in, one clock, blocked schedule', [cur, curS, inA, inB], 55);
  group('readout', [pbW, an, cha], 280);
  group('checkpoint', [out], 0);
}

// ---------------------------------------------------------------------------
// The weave, with letters.
//
// Same tissue as the weave and the same question, with a real stimulus in place of the spot and the tone.
// One piece of tissue, two noise fields picking scattered patches of it each, and the only plastic pathway is the recurrence.
// Channel A sees the letter through a retina that reports center-surround contrast rather than luminance, which for letters is the difference between a stimulus and a silhouette: on the alphabet school's grid it drops the mean pairwise similarity of the twenty-six items from 0.58 to 0.37.
// Channel B hears its name, as formant structure with a consonant onset, not as a tone.
//
// It runs the vowels rather than the whole alphabet because the tissue has to hold the letter: five items over a scattered third of 1,600 excitatory cells gives each encoder channel a few cells to drive, where twenty-six on this grid would give the decoder five hundred cells to tell apart stimuli that already overlap.
// The lesson set is one control on the curriculum node, so the alphabet is one change away when the vowels answer.
//
// Presentations are a second on and a second off.
// Stimulus timing is load bearing: two hundred and fifty milliseconds blocks learning outright with a letter; the abstract weave can use the shorter cycle because a spot has no internal structure to resolve and a letter does.
//
// The plasticity is the abstract weave's: consolidation on, a commit weight of 1.2, committing after five simulated minutes, which holds a sparse, selective, fully decodable code for two hours while the coupling between the two codes rises.
// Whether it does the same thing for a letter and its name is the question this scene asks.
function weaveLetters(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'The weave, with letters\n\nThis is the abstract weave with a real stimulus. It is one piece of tissue with two scattered codes woven through it, where channel A sees the letter and channel B hears its name. The retina reports center-surround contrast, and the sound is formant structure with a consonant onset.');
  note(520, 'The lesson is the five vowels and not the whole alphabet, since five items leave each encoder channel about thirty cells to drive. The lesson set is a control on the curriculum node.');
  note(800, 'Plasticity has consolidation on, a commit weight of 1.2, and committing after five simulated minutes. Presentations are a second on and a second off, and the inhibitory set point is 1 Hz.');

  const vol = add('box', 640, 40, { center:[0, 0, 0], size:[1600, 400, 1600] });
  const cells = add('scatter', 560, 110, { count:1600, type:RS, tag:'weave', seed:31 });
  wire(cells, 0, vol);
  const inh = add('scatter', 720, 110, { count:400, type:FS, tag:'weavei', seed:32 });
  wire(inh, 0, vol);

  const fieldA = add('noisefield', 260, 90, { center:[0, 0, 0], size:[1600, 400, 1600],
    pattern:1, scale:300, coverage:0.4, seed:7 });
  const fieldB = add('noisefield', 1020, 90, { center:[0, 0, 0], size:[1600, 400, 1600],
    pattern:1, scale:300, coverage:0.4, seed:23 });

  const m1 = add('gather', 640, 190); wire(m1, 0, cells); wire(m1, 1, inh);

  const table = ['* * 0',
    'weave weave 0.8 0.25 1',
    'weave weavei 0.8 0.5 0', 'weavei weave 0.8 1 inh',
    'weavei weavei 0.5 0.5 0',
  ].join('\n');
  const inhRule = add('plasticity', 430, 250, { name:'inh', wmax:4, wdep:1, iEta:0.02,
    aP:0, aM:0, trip:0, het:0, tin:0 });
  wire(inhRule, 0, m1);
  const cn = add('connect', 640, 300,
    { radius:3400, sigma:5000, prob:0.6, wExc:1.3, wInh:-1.5,
      wdist:0, cluster:0, velocity:200, seed:19, table });
  wire(cn, 0, inhRule);

  const cur = add('curriculum', 1400, 250, { sense:0, set:2, order:1, onMs:1000, offMs:1000,
    jitter:0, scaleVar:0, rotVar:0, acuity:1, probeEvery:5, seed:21,
    calCycles:3, reSweepEvery:0, condNames:'both, sight, sound', voice:0,
    phases:'20 A; 20 B; 80 AB' });
  // ON/OFF contrast on both channels, as the letters school runs them: the retina reports a difference and the cochlea a band contrast, and a letter that is a silhouette to one is a shape to the other.
  //
  // A quarter of the abstract weave's drive.
  // A spot lights one channel of twenty-five and a letter lights a third of a hundred and twenty-eight, so the same amplitude ran this tissue at 21 Hz; measured on the live view, 150 sits between 2 and 7 Hz across a presentation, which is where the homeostat can hold it (the checkpoint's set point is 1 Hz).
  // Vision lands on 491 cells at 31 per channel and the audio on 414 at 42, so both codes have room.
  const inA = add('input', 420, 380, { map:0, code:1, axis:1,
    cols:8, rows:8, arbor:0.7, transient:0.2, jitter:0, amp:150, lagMs:0 });
  wire(inA, 0, cn); wire(inA, 1, fieldA); wire(inA, 2, cur);
  const inB = add('input', 800, 380, { map:1, code:1, axis:2,
    cols:16, arbor:0.7, transient:0.2, jitter:0, amp:150, lagMs:10 });
  // the sound stream: the same lesson, following the first curriculum's clock, carrying the item as its spoken name
  const curS = add('curriculum', 0, 0, { sense:1 }); wire(curS, 0, cur);
  curS.x = cur.x; curS.y = cur.y + 56;
  wire(inB, 0, inA); wire(inB, 1, fieldB); wire(inB, 2, curS);

  const pbW = add('probe', 640, 530, { label:'weave', tag:'weave', record:1 });
  wire(pbW, 0, inB); wire(pbW, 1, vol);
  const an = add('analysis', 900, 530,
    { cap:4096, setExpr:'both', probeWith:'sight, sound', minCells:3, minTrials:2 });
  wire(an, 0, pbW);

  const out = add('checkpoint', 640, 610, { steps:2, engine:0, syn:1, tauE:3, tauI:8,
    plast:1, aP:0.003, aM:0.001, tauS:16.8, tauM:33.7, wmax:4, wdep:1,
    trip:0.002, tauY:114, stp:0.5, stpOrder:1, stpU:0.2, stpTauD:200, stpTauF:600,
    // A 1 Hz set point, not the abstract weave's 3.
    // Measured 2026-09-03 over fifteen simulated minutes: at 3 Hz the tissue holds a dense state and selectivity falls away, 0.49 to 0.41 to 0.24, because a letter drives enough of the tissue that the homeostat spends the run feeding it; at 1 Hz the same fifteen minutes sharpen instead, 0.46 to 0.64 to 0.73.
    // The freeze can only lock the state it finds, so the set point has to put the code somewhere worth locking first.
    het:0.002, tin:0, rhoMode:0, iRho:1, iEta:0.02, scale:0,
    cons:1, consW:1.2, consP:10, tauCons:5, consStep:1.2, commit:1,
    trHours:2, trCkptMin:5, trBrainMin:20, trReps:1, trSweep:0, trEngine:1 });
  wire(out, 0, an);
  const cha = add('chart', 940, 690, { plot:0, pop:'weave', field:'decode', height:150 });
  wire(cha, 0, out);

  group('one tissue', [vol, cells, inh], 130);
  group('channel A: sees the letter', [fieldA], 205);
  group('channel B: hears its name', [fieldB], 25);
  group('dense flat recurrence, only it learns', [m1, inhRule, cn], 130);
  group('one clock, two senses, blocked schedule', [cur, curS, inA, inB], 55);
  group('readout', [pbW, an, cha], 280);
  group('checkpoint  committing synapses leave the plastic pool', [out], 0);
}

// A bulb of bulbs.
// One bulb is built once and repeated around a core, then the ring is repeated a second time smaller and higher, so twelve bulbs on two tiers come from three shape nodes and two repeats.
// Six inputs land in the lower tier, sight and sound alternating around the ring; each bulb projects up to the bulb above it and each of those into the core, where the two senses meet.
function mandelbulb(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Mandelbulb\n\nOne bulb (a sphere with a bud) is repeated six times around a core, and then the ring is repeated once more, smaller and higher. That makes twelve bulbs on two tiers, from three shapes and two repeat nodes. A noise warp gives every surface its texture.');
  note(560, 'Six inputs land in the lower tier, with sight and sound alternating around the ring, mapped straight onto the bulb cells with no retina or cochlea in between. Each channel keeps the same cells for the whole run, so a letter lights the same pixels every time it is shown.');
  note(820, 'Each lower bulb projects into the bulb above it and each upper bulb into the core, so the core is where a letter seen in one bulb meets its name heard in the next. Recurrence is local everywhere, and the kernel does not reach between bulbs.');

  // ---- one bulb, then the ring, then the tiers --------------------------
  const bulbGeo = add('sphere', 300, 40, { center:[1500, 0, 0], radius:340 });
  const budGeo = add('sphere', 500, 40, { center:[1900, 0, 190], radius:150 });
  const bRS = add('scatter', 260, 110, { count:1200, type:RS, tag:'RS', seed:41 });
  wire(bRS, 0, bulbGeo);
  const bFS = add('scatter', 420, 110, { count:300, type:FS, tag:'FS', seed:42 });
  wire(bFS, 0, bulbGeo);
  const bud = add('scatter', 580, 110, { count:300, type:RS, tag:'bud', seed:43 });
  wire(bud, 0, budGeo);
  // The bulb's own pathways, made once and carried by the repeats: the bud is the outermost layer, past the reach of the local kernel, so the bulb feeds it; and each bulb feeds the bulb above it, which is copy k to copy k plus one across the tier repeat.
  // Both project nodes pass their populations through, so the merge takes the pathway's stream and the FS cells.
  const budPath = add('project', 340, 150, { dispersed:1, fanout:24, weight:4, velocity:2000, seed:20 });
  wire(budPath, 0, bRS); wire(budPath, 1, bud);
  const upPath = add('project', 340, 200, { from:'RS', to:'RS', offset:1, across:'tier',
    dispersed:1, fanout:48, weight:4, velocity:2000, seed:1 });
  wire(upPath, 0, budPath); wire(upPath, 1, budPath);
  const bm = add('gather', 420, 250); wire(bm, 0, upPath); wire(bm, 1, bFS);
  const ring = add('repeat', 420, 250, { copies:6, pops:0, tag:'bulb',
    translate:[0, 0, 0], rotate:[0, 60, 0], scale:[1, 1, 1], pivot:[0, 0, 0], mirror:0 });
  wire(ring, 0, bm);
  const tiers = add('repeat', 420, 320, { copies:2, pops:0, tag:'tier',
    translate:[0, 760, 0], rotate:[0, 30, 0], scale:[0.62, 0.62, 0.62], pivot:[0, 0, 0], mirror:0 });
  wire(tiers, 0, ring);

  // ---- the core ---------------------------------------------------------
  const coreGeo = add('sphere', 900, 40, { center:[0, 260, 0], radius:900 });
  const cRS = add('scatter', 860, 110, { count:5000, type:RS, tag:'core', seed:51 });
  wire(cRS, 0, coreGeo);
  const cFS = add('scatter', 1020, 110, { count:1250, type:FS, tag:'corei', seed:52 });
  wire(cFS, 0, coreGeo);

  const all = add('gather', 640, 400); wire(all, 0, tiers); wire(all, 1, cRS); wire(all, 2, cFS);
  const warp = add('noisewarp', 640, 460, { amplitude:220, scale:380, seed:5 });
  wire(warp, 0, all);

  // ---- into the core: every upper bulb, one node -------------------------
  // By proximity: each upper bulb lands on the face of the core nearest it, so the six senses arrive in six sectors rather than every bulb over the whole core, which would leave nothing to be selective about.
  // The pattern names all six upper bulbs at once.
  const inn = add('project', 640, 520, { from:'tier2.bulb*.RS', to:'core',
    dispersed:2, sigma:260, prob:0.5, weight:4, velocity:2000, seed:10 });
  wire(inn, 0, warp);
  let prev = inn;
  const projs = [budPath, upPath, inn];
  // local recurrence only: the kernel reaches within a bulb, never across
  const table = ['E E 1 1', 'E I 1 0.6 0', 'I E 1 1 inh', 'I I 0.5 0.5 0'].join('\n');
  const inhRule = add('plasticity', 640, 660, { name:'inh', wmax:4, wdep:1, iEta:0.02,
    aP:0, aM:0, trip:0, het:0, tin:0 });
  wire(inhRule, 0, prev);
  const cn = add('connect', 640, 720, { radius:180, sigma:90, prob:0.35, wExc:1.2, wInh:-2.4,
    wdist:0, cluster:0, velocity:300, seed:19, table });
  wire(cn, 0, inhRule);

  // ---- six inputs, sight and sound alternating around the lower tier ------
  const cur = add('curriculum', 1500, 720, { sense:0, set:2, order:1, onMs:1000, offMs:1000,
    jitter:0, scaleVar:0, rotVar:0, acuity:1, probeEvery:5, seed:21,
    calCycles:3, reSweepEvery:0, condNames:'both, sight, sound', voice:0 });
  const curS = add('curriculum', 1500, 776, { sense:1, offset:0 }); wire(curS, 0, cur);
  const inputs = [];
  prev = cn;
  for(let k = 1; k <= 6; k++){
    const inp = add('input', 300 + (k-1)*120, 800, { tag:`tier1.bulb${k}.RS`,
      map:2, code:1, fanin:8, seed:k, cols:8, rows:8, transient:0.2, jitter:0, amp:28 });
    wire(inp, 0, prev); wire(inp, 2, (k % 2) ? cur : curS); prev = inp; inputs.push(inp);
  }

  // ---- readout --------------------------------------------------------------
  const pbC = add('probe', 640, 900, { label:'core', tag:'core', record:1 });
  wire(pbC, 0, prev);
  const pb1 = add('probe', 400, 900, { label:'bulb 1 sees', tag:'tier1.bulb1.RS', record:1 });
  wire(pb1, 0, pbC);
  const pb2 = add('probe', 880, 900, { label:'bulb 2 hears', tag:'tier1.bulb2.RS', record:1 });
  wire(pb2, 0, pb1);
  const pbU = add('probe', 1120, 900, { label:'upper bulb 1', tag:'tier2.bulb1.RS', record:1 });
  wire(pbU, 0, pb2);
  const pbB = add('probe', 1360, 900, { label:'bud 1', tag:'tier1.bulb1.bud', record:0 });
  wire(pbB, 0, pbU);
  const an = add('analysis', 900, 960,
    { cap:4096, setExpr:'both', probeWith:'sight, sound', minCells:3, minTrials:2 });
  wire(an, 0, pbB);
  const out = add('checkpoint', 640, 1030, { steps:2, engine:0, syn:1, tauE:3, tauI:8,
    plast:1, aP:0.003, aM:0.001, tauS:16.8, tauM:33.7, wmax:4, wdep:1,
    trip:0.002, tauY:114, stp:0.5, stpOrder:1, stpU:0.2, stpTauD:200, stpTauF:600,
    het:0.002, tin:0, rhoMode:0, iRho:2, iEta:0.02, scale:0,
    cons:1, consW:1.2, consP:10, tauCons:5, consStep:1.2, commit:1,
    trHours:2, trCkptMin:5, trBrainMin:20, trReps:1, trSweep:0, trEngine:1 });
  wire(out, 0, an);
  const cha = add('chart', 940, 1110, { plot:0, pop:'core', field:'decode', height:150 });
  wire(cha, 0, out);

  group('one bulb and its pathways', [bulbGeo, budGeo, bRS, bFS, bud, budPath, upPath, bm], 25);
  group('six around, then two tiers', [ring, tiers], 270);
  group('the core', [coreGeo, cRS, cFS], 130);
  group('every surface warped', [all, warp], 270);
  group('into the core', [inn], 160);
  group('local recurrence', [inhRule, cn], 130);
  group('six inputs, sight and sound alternating', [cur, curS, ...inputs], 55);
  group('readout', [pbC, pb1, pb2, pbU, pbB, an, cha], 280);
  group('checkpoint', [out], 0);
}

// ---- the Potjans and Diesmann 2014 microcircuit at its own scale ----------
// Their model as a scene, for the replication (RESULTS.md): the eight populations at the paper's sizes (77,169 cells under a square millimeter), the verified probability matrix, their synaptic strength (87.8 pA for 0.5 ms, so 43.9 pA ms of charge, delivered here as a kick), inhibition at four times excitation, the L4e to L2/3e weight doubled, and the background of 8 Hz Poisson input over K_bg external synapses per population, carried as a constant current (rate times K_bg times the charge) and a per-millisecond noise of the same variance.
// What differs from the paper, stated in RESULTS.md: the membrane is the 2007 quadratic form on a row matched to their leaky integrate-and-fire in capacitance (250 pF), rest (-65 mV), threshold (-50 mV), reset (-65 mV), refractory period (2 ms) and the time constant at rest (10 ms, k = C / (tau (vt - vr))), which gives a rheobase of 94 pA against their 375; delays come from distance at 400 um/ms in whole milliseconds (theirs are 1.5 and 0.8 ms with half that in spread); weights are lognormal at sigma 0.1 where theirs are Gaussian at ten percent; and the step is 1 ms against 0.1.
const PD_N = { 'L2/3e':20683, 'L2/3i':5834, 'L4e':21915, 'L4i':5479, 'L5e':4850, 'L5i':1065, 'L6e':14395, 'L6i':2948 };
const PD_KBG = { 'L2/3e':1600, 'L2/3i':1500, 'L4e':2100, 'L4i':1900, 'L5e':2000, 'L5i':1900, 'L6e':2900, 'L6i':2100 };
export const PD_CHARGE = 87.8*0.5;     // pA ms per spike: the PSC amplitude times its time constant
function pdMicrocircuit(ed){
  const { add, wire, note, group } = api(ed);
  note(20, 'Potjans and Diesmann 2014, at scale\n\nThe cell type specific cortical microcircuit as they published it: eight populations under one square millimeter of cortex at their sizes, their connection probabilities between every pair, their synaptic strength with inhibition four times excitation, and their background of 8 Hz Poisson input over a fixed number of external synapses per population, here as a constant current and a noise term of the same variance. The cells are on a row matched to their integrate and fire neuron in capacitance, rest, threshold, reset and time constant. What differs from the paper is stated in RESULTS.md.');
  note(320, 'This is the Potjans and Diesmann 2014 replication (RESULTS.md in the repository). The rates, the interval variability and the synchrony of each population are measured on the reference engine and on the CUDA engine, three seeds each, and set beside the paper\'s. The scene is not tuned: every number is theirs.');
  const layers = [
    { name:'L2/3', y:900, h:300 },
    { name:'L4',   y:650, h:200 },
    { name:'L5',   y:400, h:300 },
    { name:'L6',   y:100, h:300 },
  ];
  const streams = [], boxes = [];
  layers.forEach((L, k) => {
    const x = 40 + k*300;
    const box = add('box', x, 20, { center:[0, L.y, 0], size:[1000, L.h, 1000] }, L.name + ' box');
    boxes.push(box);
    const se = add('scatter', x, 75, { count:PD_N[L.name + 'e'], type:RS, seed:k*2 + 1, tag:L.name + 'e' }, L.name + 'e');
    wire(se, 0, box);
    const si = add('scatter', x + 150, 75, { count:PD_N[L.name + 'i'], type:FS, seed:k*2 + 2, tag:L.name + 'i' }, L.name + 'i');
    wire(si, 0, box);
    group(L.name, [box, se, si], 205);
    streams.push(se, si);
  });
  const m1 = add('gather', 190, 210, {}, 'L2/3 and L4'), m2 = add('gather', 490, 265, {}, 'L5 and L6'), m3 = add('gather', 640, 320, {}, 'the column');
  streams.slice(0, 4).forEach((s, i) => wire(m1, i, s));
  streams.slice(4, 8).forEach((s, i) => wire(m2, i, s));
  wire(m3, 0, m1); wire(m3, 1, m2);
  // one row for every cell, in two signs: the 2007 form matched to their integrate and fire neuron at rest
  const lif = { C:250, k:+(250/(10*15)).toFixed(4), vr:-65, vt:-50, vpeak:-50, a:0.02, b:0, c:-65, d:0, hue:30 };
  const rowE = add('celltype', 640, 375, { row:0, tag:'*e', name:'pde', sign:0, ...lif }, 'excitatory cells');
  wire(rowE, 0, m3);
  const rowI = add('celltype', 640, 430, { row:0, tag:'*i', name:'pdi', sign:1, ...lif, hue:200 }, 'inhibitory cells');
  wire(rowI, 0, rowE);
  group('one integrate and fire row, two signs', [rowE, rowI], 300);
  // the paper's probabilities, distance independent: the kernel is flat over the column and the base probability makes each table entry the published one.
  // Weights: 43.9 pA ms of charge per spike as a kick, at four times that for inhibition, lognormal at a tenth for their ten percent Gaussian.
  const cn = add('connect', 640, 490,
    { radius:3000, sigma:100000, prob:0.05, wExc:+PD_CHARGE.toFixed(2), wInh:+(-4*PD_CHARGE).toFixed(2),
      wdist:1, wsigma:0.1, velocity:400, seed:1, table:pdTable() }, 'their pair table');
  wire(cn, 0, rowI);
  group('local wiring  Potjans-Diesmann pair table', [cn], 130);
  // the background: 8 Hz over K_bg external synapses of the same strength, a mean current of rate times K_bg times the charge, and per-millisecond shot noise of variance rate times K_bg times the charge squared, drawn triangular (variance a sixth of the amplitude squared)
  let prev = cn; const drives = [];
  Object.keys(PD_N).forEach((tag, k) => {
    const K = PD_KBG[tag], nu = 0.008;
    const mean = +(nu*K*PD_CHARGE).toFixed(1), amp = +(PD_CHARGE*Math.sqrt(6*nu*K)).toFixed(1);
    const dc = add('stimulus', 40 + k*150, 580, { mode:0, current:mean, tag, radius:100000 }, 'background on ' + tag);
    wire(dc, 0, prev);
    const nz = add('stimulus', 40 + k*150, 635, { mode:3, current:amp, tag, radius:100000 }, 'noise on ' + tag);
    wire(nz, 0, dc);
    prev = nz; drives.push(dc, nz);
  });
  group('background  8 Hz over K_bg synapses, per population', drives, 55);
  let prb = prev; const probes = [];
  Object.keys(PD_N).forEach((tag, k) => {
    const pb = add('probe', 40 + k*150, 720, { label:tag, tag }, 'probe ' + tag);
    wire(pb, 0, prb); prb = pb; probes.push(pb);
  });
  const out = add('checkpoint', 640, 800, { steps:2, syn:0, refrac:2, plast:0 }, 'checkpoint');
  wire(out, 0, prb);
  const cha = add('chart', 640, 860, { plot:7, pop:'L5e', window:5, height:130 }, 'chart');
  wire(cha, 0, out);
  group('readout', [...probes, cha], 175);
  group('checkpoint', [out], 0);
}


export const EXPERIMENTS = [
  { name:'alphabet school 2',     build: named(alphabetSchool2) },
  { name:'learning lab',          build: named(learningLab) },
  { name:'binding bench',         build: named(bindingBench) },
  { name:'the weave',             build: named(weave) },
  { name:'the weave, letters',    build: named(weaveLetters) },
  { name:'mandelbulb',            build: named(mandelbulb) },
  { name:'PD microcircuit',       build: named(pdMicrocircuit) },
];
// Every scene that can be built, for anything that resolves a scene by name.
export const ALL_SCENARIOS = [...SCENARIOS, ...EXPERIMENTS];

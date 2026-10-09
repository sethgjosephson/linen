// What the chart node draws, for any context and any style: the panel's canvas, a PNG at a chosen size, or the SVG context.
// One function, so the figure that leaves the app is the plot that was on screen, redrawn at the size and in the style asked for, not a copy of the panel's pixels.
//
// paintChart reads the node's settings and the data sources it is handed and draws through chart.js.
// It touches no DOM and no app state, so src/chartpaint.test.mjs drives it under node with a stand-in recorder.
//
// src: { history, live() -> { rec, scope(), recording() }, weights() ->
//        Promise<{ w, pmask }>, matrices() -> Map, stale() }
// Returns { chart, note, smoothMs }: chart holds the numbers behind the plot (what "save data" writes), note is the line under the panel.

import * as C from './chart.js';

// a setting that is a number or blank: blank is "let the data decide"
export const numOrAuto = v => { const s = String(v === undefined || v === null ? '' : v).trim();
  if(s === '') return undefined; const n = +s; return Number.isFinite(n) ? n : undefined; };

export function chartSeries(history, pop, field){
  const points = [];
  for(const row of history){
    const x = row.simMin;
    if(!Number.isFinite(x)) continue;
    let y;
    if(field === 'hz'){
      const pr = (row.probes || []).find(p => p.l === pop);
      y = pr ? pr.hz : undefined;
    } else {
      const path = field.split('.');
      const dig = o => path.reduce((x, k) => (x && typeof x === 'object') ? x[k] : undefined, o);
      const v = row.readout && row.readout.pops && row.readout.pops[pop];
      y = v ? dig(v) : undefined;
      // Measures that belong to the network rather than to one population live on the checkpoint row: the rate, the synchrony, how far the weights moved, and the weight between and within the drive sets.
      // Looking there when the population has no such field makes those chartable too, under the same dotted names the files use.
      if(y === undefined) y = dig(row);
    }
    if(typeof y === 'number' && Number.isFinite(y)) points.push([x, y]);
  }
  return points;
}

// Several runs of one thing, as one line: the mean at each checkpoint and the range across the runs that reached it.
// The range rather than a standard error, because three runs do not say what distribution they came from and the range says exactly what was seen.
export function meanBand(seriesList){
  const at = new Map();
  for(const pts of seriesList) for(const [x, y] of pts){
    if(!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if(!at.has(x)) at.set(x, []);
    at.get(x).push(y);
  }
  const points = [], band = [];
  let widest = 0;
  for(const x of [...at.keys()].sort((a, b) => a - b)){
    const v = at.get(x);
    points.push([x, v.reduce((a, b) => a + b, 0)/v.length]);
    if(v.length > 1){ band.push([x, Math.min(...v), Math.max(...v)]); widest = Math.max(widest, v.length); }
  }
  return { points, band, runs:seriesList.length, most:widest };
}

export async function paintChart(ctx, w, h, P, src, S){
  const plot = P.plot | 0;
  const pop = String(P.pop || '').trim();
  const title = String(P.title || '').trim() || undefined;
  const xText = String(P.xText || '').trim(), yText = String(P.yText || '').trim();
  const yMin = numOrAuto(P.yMin), yMax = numOrAuto(P.yMax);
  // The compact layout of the panel has room for the x label only, in its short form.
  const lab = (x, y, xShort) => ({ title, xLabel: xText || (S.full ? x : (xShort || x)), yLabel: yText || y, style:S });
  const none = (text, note = '') => { C.drawLine(ctx, w, h, [], { empty:text, title, style:S }); return { chart:null, note }; };

  if(plot >= 7){
    const L = src.live ? src.live() : null;
    if(!L || !L.rec) return none('no live recording');
    const rec = L.rec, win = Math.max(100, (+P.window || 5)*1000);
    const smooth = Math.max(0, +P.smooth || 0);
    const noPop = rec.labels().length ? 'pick a probed population: ' + rec.labels().join(', ') : 'add a probe node: the page records what probes read';
    if(plot === 7){
      const labels = pop ? [pop] : rec.labels();
      const raw = labels.map(l => ({ label:l, points:rec.series(l, win) }));
      const series = raw.map(s => ({ label:s.label, points:C.boxcar(s.points, smooth).map(([t, hz]) => [t/1000, hz]) }));
      C.drawLine(ctx, w, h, series, { ...lab('time (s)', 'rate (Hz)', 'sim s'), yLo:0, yMin, yMax, xTight:true, empty:noPop });
      const pts = raw.length ? raw[0].points : [];
      const chart = !pts.length ? null : raw.length > 1 ? { kind:'multi', series:raw } : { kind:'series', points:pts };
      const one = pop || (labels.length === 1 ? labels[0] : '');
      return { chart, smoothMs:smooth, note: pts.length
        ? (one ? `${pts[pts.length-1][1].toFixed(1)} Hz now · mean ${rec.meanRate(one, win).toFixed(1)} Hz over the last ${(win/1000).toFixed(1)} s` : `${labels.length} populations over the last ${(win/1000).toFixed(1)} s`) +
          ` · ${rec.stepMs} ms frames` + (smooth ? ` · moving average of ${smooth} ms` : '') : '' };
    }
    if(plot === 8){
      const r = rec.raster(pop, win);
      C.drawRaster(ctx, w, h, r.events, { ...lab('time (s)', 'cell'), rows:r.rows, t0:rec.t - win, t1:rec.t, empty:noPop });
      const p = rec.pops.get(pop);
      return { chart:r.rows ? { kind:'raster', events:r.events, rows:r.rows, cells:r.cells } : null,
        note: r.rows ? `${r.events.length.toLocaleString()} spikes over ${r.rows} of ${p ? p.n : r.rows} cells` + (r.stride > 1 ? ` (every ${r.stride}th)` : '') + ` in the last ${(win/1000).toFixed(1)} s · ${rec.stepMs} ms frames` : '' };
    }
    if(plot === 9){
      const iv = rec.intervals(pop);
      const any = iv.total > 0;
      C.drawHistogram(ctx, w, h, any ? iv.counts : [], { ...lab('interval between spikes (ms)', 'intervals'), lo:0, hi:iv.ceiling + (iv.edges[1] || 4), yMax, empty:noPop });
      return { chart:any ? { kind:'hist', counts:iv.counts, edges:iv.edges } : null,
        note: any ? `${iv.total.toLocaleString()} intervals · bins of ${iv.edges[1]} ms, the last bin everything past ${iv.ceiling} ms · frame resolution` : '' };
    }
    if(plot === 11){
      const R = L.recording ? L.recording() : null;
      if(!R) return none('press REC in the viewer bar: the recording draws here as it runs');
      const labels = pop && R.labels().includes(pop) ? [pop] : R.labels();
      const series = labels.map(l => ({ label:l, points:C.boxcar(R.series(l), smooth).map(([t, hz]) => [t/1000, hz]) }));
      const vlines = R.events.map(e => ({ x:e.t/1000, strong:e.kind === 'edit', label:e.kind === 'edit' ? e.node + ' ' + e.key : '' }));
      C.drawLine(ctx, w, h, series, { ...lab('time (s)', 'rate (Hz)', 'sim s'), yLo:0, yMin, yMax, xTight:true, vlines, empty:'recording: no probe has reported yet' });
      const nEd = R.edits();
      return { chart:{ kind:'recording', csv:() => R.csv() }, smoothMs:smooth,
        note: R.seconds().toFixed(1) + ' s recorded · ' + nEd + ' edit' + (nEd === 1 ? '' : 's') + ' · ' + R.events.length + ' events · bins of ' + R.binMs + ' ms' + (smooth ? ' · moving average of ' + smooth + ' ms' : '') };
    }
    const sb = L.scope ? L.scope() : [];
    const pts = sb.map((v, i) => [i - sb.length, v]);
    C.drawLine(ctx, w, h, [{ points:pts, color:S.trace }], { ...lab('time before now (ms)', 'membrane potential (mV)', 'ms before now'),
      yLo: yMin === undefined ? -95 : undefined, yHi: yMax === undefined ? 40 : undefined, yMin, yMax,
      xTight:true, marks:[{ y:30, label:'peak' }, { y:-65, label:'reset' }], empty:'select a neuron in the viewer' });
    return { chart:pts.length ? { kind:'trace', points:pts } : null, note: pts.length ? `membrane potential of the selected cell, ${pts[pts.length-1][1].toFixed(0)} mV now` : '' };
  }

  if(plot === 1){
    // The weights the engine holds now, not the ones the wiring made: the whole interest of this plot is where learning moved them, and the wired array would show the starting distribution and call it the learned one.
    // Split by the rule byte, because a network that freezes most of its synapses draws one tall spike at the wired value and hides the plastic ones underneath it.
    if(!src.weights) return none('no engine');
    let d = null;
    try { d = await src.weights(); } catch(e){ return none('no weights available', 'weight readback failed: ' + e.message); }
    if(src.stale && src.stale()) return null;
    if(!d || !d.w) return none('no weights available', 'the engine did not return weights; a network has to be running');
    const plastic = [], frozen = [];
    const pm = d.pmask;
    for(let i = 0; i < d.w.length; i++){
      const v = d.w[i];
      if(v <= 0) continue;                 // excitatory only, as the papers plot it
      (!pm || pm[i] ? plastic : frozen).push(v);
    }
    const showFrozen = frozen.length > plastic.length;
    const b = C.bin(showFrozen ? frozen : plastic, P.bins | 0 || 40);
    C.drawHistogram(ctx, w, h, b.counts, { ...lab('excitatory weight', 'synapses'), lo:b.lo, hi:b.hi, yMax, empty:'no excitatory weights' });
    return { chart:{ kind:'hist', counts:[...b.counts], edges:Array.from(b.counts, (_, i) => b.lo + (b.hi - b.lo)*i/b.counts.length).concat([b.hi]) },
      note: `${plastic.length.toLocaleString()} plastic, ${frozen.length.toLocaleString()} frozen` +
        ` · showing the ${showFrozen ? 'frozen' : 'plastic'} ones · ${b.lo.toFixed(3)} to ${b.hi.toFixed(3)}` };
  }
  if(plot === 2){
    // Items down, cells across: the response matrix every selectivity and decoding number here was reduced from.
    const mats = src.matrices ? src.matrices() : null;
    const m = mats && mats.get(pop);
    if(!m) return none('no response matrix yet', mats && mats.size
      ? 'pick a population the run measures: ' + [...mats.keys()].join(', ')
      : 'a training run fills this in at its first checkpoint');
    C.drawMatrix(ctx, w, h, m.values, m.rows, m.cols, { ...lab('cell', 'item'), barLabel:'mean response', vMin:yMin, vMax:yMax });
    return { chart:{ kind:'matrix', values:m.values, rows:m.rows, cols:m.cols },
      note: `${m.rows} items x ${m.cols} cells at sim ${m.simMin} min · brighter is a stronger mean response` };
  }
  if(plot >= 3){
    // the figures the readout hands over as matrices, keyed by population and kind; drawn as heat maps at data resolution
    const mats = src.matrices ? src.matrices() : null;
    const which = P.which | 0;
    const key = plot === 3 ? pop + ':rdm:' + ['both', 'vis', 'aud', 'cross'][which]
      : plot === 4 ? pop + ':psth:' + ['both', 'vis', 'aud'][Math.min(2, which)]
      : plot === 5 ? pop + ':wblock' : pop + ':confusion';
    const m = mats && mats.get(key);
    if(!m) return none('nothing to draw yet', mats && mats.size
      ? 'a checkpoint where this measure exists fills this in (' + key + ')'
      : 'a training run fills this in at its first checkpoint');
    const names = plot === 3 ? [which === 3 ? 'item heard' : 'item', which === 3 ? 'item seen' : 'item', 'dissimilarity (1 - r)']
      : plot === 4 ? ['time from onset (ticks of ' + m.tickMs + ' ms)', 'item', 'spikes per tick']
      : plot === 5 ? ['target set', 'source set', 'mean excitatory weight']
      : ['decoded item', 'true item', 'trials'];
    C.drawMatrix(ctx, w, h, m.values, m.rows, m.cols, { ...lab(names[0], names[1]), barLabel:names[2], vMin:yMin, vMax:yMax });
    return { chart:{ kind:'matrix', values:m.values, rows:m.rows, cols:m.cols },
      note: plot === 3
        ? `${m.rows} items square at sim ${m.simMin} min · dissimilarity 1 - r, darker is more alike` + (which === 3 ? ' · rows seen, columns heard' : '')
        : plot === 4
        ? `${m.rows} items x ${m.cols} ticks of ${m.tickMs} ms from onset at sim ${m.simMin} min · mean population spikes per tick`
        : plot === 5
        ? `${m.rows} conjunctive sets square at sim ${m.simMin} min · mean excitatory weight, row set to column set`
        : `${m.rows} items square at sim ${m.simMin} min · rows true item, columns decoded item, split-half counts` };
  }
  const history = src.history || [];
  const field = P.field || 'decode';
  const pts = chartSeries(history, pop, field);
  const last = history[history.length - 1];
  const marks = [];
  // a decode number without its chance line is not a claim about anything
  if((field.startsWith('decode') || /\.acc$/.test(field) || field === 'crossModal.trialAcc')
     && last && last.readout && Number.isFinite(last.readout.chance))
    marks.push({ y:last.readout.chance, label:'chance' });
  // A sweep: one line per variant and condition, each the mean of its replicates with the range behind it.
  // A single run's line is one draw and says nothing about the spread; this is what the replicates were run for.
  const groups = (src.groups || []).filter(g => g && g.histories && g.histories.length);
  if(groups.length){
    const series = groups.map(g => {
      const A = meanBand(g.histories.map(x => chartSeries(x, pop, field)));
      return { label:g.label + ' (' + A.runs + ' run' + (A.runs === 1 ? '' : 's') + ')',
        points:A.points, band:A.band, runs:A.runs };
    }).filter(s => s.points.length);
    C.drawLine(ctx, w, h, series, { ...lab('simulated time (min)', field + (pop ? ', ' + pop : ''), 'sim min'),
      marks, yMin, yMax, empty: history.length ? 'nothing recorded for this population' : 'no checkpoints in this sweep' });
    const runs = series.reduce((n, s) => n + s.runs, 0);
    return { chart:series.length ? { kind:'bands', series, xName:'sim_min', yName:field } : null,
      note: series.length
        ? `${series.length} group${series.length === 1 ? '' : 's'} of runs · ${runs} runs · `
          + 'the line is the mean and the band is the range across them'
        : 'pick a population and a measure these runs record' };
  }
  C.drawLine(ctx, w, h, [{ points:pts }], { ...lab('simulated time (min)', field + (pop ? ', ' + pop : ''), 'sim min'), marks, yMin, yMax,
    empty: history.length ? 'nothing recorded for this population' : 'no checkpoints yet' });
  const lastY = pts.length ? pts[pts.length-1][1] : 0;
  return { chart:pts.length ? { kind:'series', points:pts, xName:'sim_min', yName:field } : null,
    note: pts.length
      ? `${pts.length} checkpoints · latest ${field} ` +
        (Math.abs(lastY) >= 100 ? lastY.toFixed(0) : Math.abs(lastY) >= 1 ? lastY.toFixed(2) : lastY.toFixed(4))
      : (history.length ? 'pick a population and a measure this run records' : 'a training run fills this in as it checkpoints') };
}

// ---- the figure that leaves the app ---------------------------------------
// The size of an exported figure.
// Drawing units are points (1/72 inch), so a font size on the node is a font size on the page whatever the resolution; the PNG is the same drawing at dpi/72 pixels per point.
const FONT_KEYS = ['sans', 'serif', 'mono'], THEME_KEYS = ['light', 'clear', 'dark'], CMAP_KEYS = [null, 'viridis', 'grey', 'teal'];
export const MAX_PIXELS = 120e6, MAX_SIDE = 16000;
export function figureSpec(P){
  const mmW = Math.min(1000, Math.max(20, +P.figW || 180)), mmH = Math.min(1000, Math.max(15, +P.figH || 100));
  const dpi = Math.min(1200, Math.max(72, Math.round(+P.dpi || 300)));
  const wPt = mmW/25.4*72, hPt = mmH/25.4*72, k = dpi/72;
  const pxW = Math.round(wPt*k), pxH = Math.round(hPt*k);
  const style = C.makeStyle({ theme:THEME_KEYS[P.theme | 0] || 'light', font:FONT_KEYS[P.figFont | 0] || 'sans',
    fontPx:Math.min(24, Math.max(4, +P.fontPt || 8)), lineW:Math.min(6, Math.max(0.1, +P.linePt || 1)),
    full:true, grid:(P.grid === undefined ? 1 : P.grid | 0) === 1, legend:(P.legend === undefined ? 1 : P.legend | 0) === 1,
    cmap:CMAP_KEYS[P.cmap | 0] || undefined });
  const tooBig = pxW > MAX_SIDE || pxH > MAX_SIDE || pxW*pxH > MAX_PIXELS;
  return { mmW, mmH, dpi, wPt, hPt, k, pxW, pxH, style, tooBig };
}
// the panel's style: the screen's compact look, or the figure's look scaled to the panel when the node previews the export
export function screenStyle(P){
  return C.makeStyle({ theme:'screen', grid:(P.grid === undefined ? 1 : P.grid | 0) === 1,
    legend:(P.legend === undefined ? 1 : P.legend | 0) === 1, cmap:CMAP_KEYS[P.cmap | 0] || undefined });
}
// Draw the whole figure (plot and caption) into a context w by h points.
export async function paintFigure(ctx, spec, P, src, captionLines){
  const S = spec.style;
  const band = C.captionHeight(S, captionLines);
  if(S.bg){ ctx.fillStyle = S.bg; ctx.fillRect(0, 0, spec.wPt, spec.hPt); }
  const r = await paintChart(ctx, spec.wPt, spec.hPt - band, P, src, S);
  if(band) C.drawCaption(ctx, S.fontPx*1.2, spec.hPt - band, captionLines, S);
  return r;
}

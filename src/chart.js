// Drawing for the chart node: a line plot, a raster, a histogram and a matrix.
//
// Everything ANALYSIS.md lists is one of these shapes: decode against time and rate against time are line plots, the weight and interval distributions are histograms, and the response matrix, the confusion matrix and the sorted weight matrix are all matrices drawn as heat maps.
// So there is a function per shape rather than one per measure, and adding a measure is a matter of handing one of them the numbers.
//
// Canvas rather than SVG on screen because a response matrix is thousands of cells and an SVG rect each is not a reasonable thing to ask a browser for.
// Pure functions of (context, size, data, style): no DOM, no reading of app state, so src/chart.test.mjs can check the scales and the layout without a browser, and the same functions draw the panel, a PNG at any size, and a vector figure through the SVG context in svgctx.js.
//
// A style (makeStyle) says how a figure looks: the colors of a theme, the font and its size, the line width, and whether the layout is the compact one of the panel (no axis titles, labels tucked into the tick row) or the full one of a figure (a title, axis titles with units, tick marks, a legend).
// Lengths are in the units of the context: CSS pixels on screen, points in an exported figure.

// Color sets.
// `screen` is the app's palette at the one saturation the rest of the interface uses.
// `light` is for print: black ink on white, and the Okabe and Ito (2008) color set for series, which stays distinguishable under the common forms of color blindness.
const SCREEN_SERIES = ['#6fd3b0', '#d3a86f', '#8fa8e8', '#e08aa0', '#c8d36f', '#9fd3e8'];
const OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000'];
export const THEMES = {
  screen: { bg:null, ink:'#c8c8c8', dim:'#555', grid:'#1c1c1c', axis:'#333', mark:'#6a5a2a',
    event:'#e8c46a', eventDim:'#777', series:SCREEN_SERIES, bar:'#6fd3b0', trace:'#e8e8e8', ramp:'teal' },
  dark: { bg:'#000', ink:'#e6e6e6', dim:'#b4bcc0', grid:'#242424', axis:'#9aa0a4', mark:'#c9a545',
    event:'#e8c46a', eventDim:'#888', series:SCREEN_SERIES, bar:'#6fd3b0', trace:'#e8e8e8', ramp:'teal' },
  light: { bg:'#fff', ink:'#000', dim:'#000', grid:'#e2e2e2', axis:'#000', mark:'#8a6a00',
    event:'#a06a00', eventDim:'#777', series:OKABE_ITO, bar:'#4d4d4d', trace:'#000', ramp:'grey' },
};
THEMES.clear = { ...THEMES.light, bg:null };

const FONTS = {
  mono: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
  sans: 'Arial, Helvetica, sans-serif',
  serif: '"Times New Roman", Times, serif',
};

// o: { theme, fontPx, font, lineW, full, grid, legend, cmap }
export function makeStyle(o = {}){
  const theme = THEMES[o.theme] || THEMES.screen;
  const fontPx = o.fontPx > 0 ? +o.fontPx : 10;
  const family = FONTS[o.font] || FONTS.mono;
  return { ...theme, fontPx, family, fontKey:FONTS[o.font] ? o.font : 'mono',
    font: fontPx + 'px ' + family, small: (fontPx*0.9) + 'px ' + family,
    lineW: o.lineW > 0 ? +o.lineW : 1,
    // showGrid, not grid: grid is the theme's grid color
    full: !!o.full, showGrid: o.grid !== false, legend: o.legend !== false,
    cmap: o.cmap || theme.ramp };
}
const SCREEN = makeStyle();

// Nice round numbers for an axis: the 1, 2, 5 sequence, which is what makes ticks land on values a person would have chosen.
export function niceStep(range, targetTicks){
  if(!(range > 0)) return 1;
  const raw = range/Math.max(1, targetTicks);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw/mag;
  // Heckbert's thresholds, which round the ideal step to the nearest of 1, 2, 5, 10 rather than up to the next one.
  // Rounding up looks harmless and is not: asked for four ticks over a range of 1 it returns a step of 0.5 and draws two, and an axis with two ticks on it is close to no axis.
  const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return step*mag;
}

// Data bounds padded out to the next round number, so the top of the plot is a value worth reading rather than whatever the maximum happened to be.
export function niceBounds(lo, hi, targetTicks = 4){
  if(!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo:0, hi:1, step:0.5 };
  if(lo === hi){ const p = Math.abs(lo) > 0 ? Math.abs(lo)*0.1 : 1; lo -= p; hi += p; }
  const step = niceStep(hi - lo, targetTicks);
  return { lo:Math.floor(lo/step)*step, hi:Math.ceil(hi/step)*step, step };
}
// Bounds a person fixed: kept exactly, with a round step between them.
// A side left open (undefined) is taken from the data and rounded as usual.
export function fixedBounds(dataLo, dataHi, fixLo, fixHi, targetTicks = 4){
  const hasLo = Number.isFinite(fixLo), hasHi = Number.isFinite(fixHi);
  if(!hasLo && !hasHi) return niceBounds(dataLo, dataHi, targetTicks);
  const auto = niceBounds(hasLo ? Math.min(fixLo, dataLo) : dataLo, hasHi ? Math.max(fixHi, dataHi) : dataHi, targetTicks);
  let lo = hasLo ? fixLo : auto.lo, hi = hasHi ? fixHi : auto.hi;
  if(!(hi > lo)) hi = lo + 1;
  return { lo, hi, step:niceStep(hi - lo, targetTicks), fixed:true };
}
// the tick values of bounds: multiples of the step inside [lo, hi]
export function ticksOf(B){
  const out = [], first = Math.ceil(B.lo/B.step - 1e-9)*B.step;
  for(let v = first; v <= B.hi + B.step*1e-9 && out.length < 200; v += B.step) out.push(Math.abs(v) < B.step*1e-9 ? 0 : v);
  return out;
}

export const fmt = v => {
  const a = Math.abs(v);
  if(a === 0) return '0';
  if(a >= 1e6) return v.toExponential(1);
  if(a >= 1000) return String(Math.round(v));
  if(a >= 1) return String(+v.toFixed(2));
  if(a >= 0.01) return String(+v.toFixed(3));
  return v.toExponential(1);
};

// A moving average over time: each point becomes the mean of the points within ms/2 either side of it.
// Frame rates are noisy at the frame length; a figure states the window it was smoothed over.
// Points are [t, y], t ascending; a window of zero returns the points as given.
export function boxcar(points, ms){
  if(!(ms > 0) || !points || points.length < 3) return points || [];
  const half = ms/2, out = [];
  let a = 0, b = 0, sum = 0;
  for(let i = 0; i < points.length; i++){
    const t = points[i][0];
    while(b < points.length && points[b][0] <= t + half){ sum += points[b][1]; b++; }
    while(points[a][0] < t - half){ sum -= points[a][1]; a++; }
    out.push([t, sum/(b - a)]);
  }
  return out;
}

// ---- layout ---------------------------------------------------------------
// Where the plot area sits.
// The compact layout is the panel's and keeps its fixed margins (scaled with the font); the full layout measures what it has to fit: the widest tick label, the axis titles, the title, a legend.
function legendRows(ctx, S, labels, width){
  if(!S.legend || !labels || labels.length < 2) return [];
  const f = S.fontPx, rows = [[]]; let x = 0;
  for(const l of labels){
    const wl = f*1.6 + ctx.measureText(l.label).width + f*1.2;
    if(x + wl > width && rows[rows.length - 1].length){ rows.push([]); x = 0; }
    rows[rows.length - 1].push({ ...l, w:wl }); x += wl;
  }
  return rows;
}
function layout(ctx, w, h, S, o = {}){
  const f = S.fontPx, k = f/10;
  ctx.font = S.font;
  let pad;
  if(!S.full) pad = { l:44*k, r:8*k, t:10*k, b:20*k };
  else {
    let tickW = 0;
    for(const t of o.yTicks || []) tickW = Math.max(tickW, ctx.measureText(t).width);
    pad = { l: Math.ceil(tickW + f*1.1 + (o.yLabel ? f*1.7 : 0) + f*0.3),
      r: Math.ceil(f*1.4 + (o.right || 0)),
      t: Math.ceil(f*0.9 + (o.title ? f*1.9 : 0)),
      b: Math.ceil(f*2.0 + (o.xLabel ? f*1.6 : 0)) };
  }
  const rows = legendRows(ctx, S, o.legend, w - pad.l - pad.r);
  const legendTop = pad.t;
  pad.t += rows.length*f*1.5;
  return { pad, rows, legendTop, iw: w - pad.l - pad.r, ih: h - pad.t - pad.b };
}
function ground(ctx, w, h, S){
  ctx.clearRect(0, 0, w, h);
  if(S.bg){ ctx.fillStyle = S.bg; ctx.fillRect(0, 0, w, h); }
  ctx.font = S.font; ctx.textBaseline = 'middle'; ctx.setLineDash([]);
}
const snap = (S, v) => S.full ? v : Math.round(v) + 0.5;
function axes(ctx, w, h, L, S){
  const { pad } = L;
  ctx.strokeStyle = S.axis; ctx.lineWidth = S.full ? Math.max(0.5, S.lineW*0.75) : 1;
  ctx.beginPath();
  ctx.moveTo(snap(S, pad.l), pad.t);
  ctx.lineTo(snap(S, pad.l), snap(S, h - pad.b));
  ctx.lineTo(w - pad.r, snap(S, h - pad.b));
  ctx.stroke();
}
function empty(ctx, w, h, S, text){
  ctx.fillStyle = S.dim; ctx.textAlign = 'center';
  ctx.fillText(text, w/2, h/2);
}
// the title, the axis titles and the legend of a full figure; in the compact layout the x label sits at the right end of the tick row and the y label is left out (the note under the panel carries the units)
function furniture(ctx, w, h, L, S, o){
  const { pad } = L, f = S.fontPx;
  ctx.font = S.font; ctx.fillStyle = S.ink;
  if(S.full){
    if(o.title){ ctx.textAlign = 'center'; ctx.font = (f*1.15) + 'px ' + S.family;
      ctx.fillText(o.title, pad.l + L.iw/2, f*1.3); ctx.font = S.font; }
    if(o.xLabel){ ctx.textAlign = 'center'; ctx.fillText(o.xLabel, pad.l + L.iw/2, h - f*0.9); }
    if(o.yLabel){ ctx.save(); ctx.translate(f*1.0, pad.t + L.ih/2); ctx.rotate(-Math.PI/2);
      ctx.textAlign = 'center'; ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
  } else if(o.xLabel){
    ctx.fillStyle = S.dim; ctx.textAlign = 'right';
    ctx.fillText(o.xLabel, w - pad.r, h - pad.b + 9*f/10);
  }
  let y = L.legendTop + f*0.6;
  for(const row of L.rows){
    let x = pad.l + f*0.3;
    for(const l of row){
      ctx.strokeStyle = l.color; ctx.lineWidth = Math.max(1, S.lineW*1.5);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + f*1.2, y); ctx.stroke();
      ctx.fillStyle = S.full ? S.ink : l.color; ctx.textAlign = 'left';
      ctx.fillText(l.label, x + f*1.6, y);
      x += l.w;
    }
    y += f*1.5;
  }
}
function yAxisTicks(ctx, w, h, L, S, Y, py){
  const { pad } = L, f = S.fontPx;
  ctx.textAlign = 'right'; ctx.lineWidth = S.full ? Math.max(0.35, S.lineW*0.5) : 1;
  for(const v of ticksOf(Y)){
    const y = snap(S, py(v));
    if(S.showGrid){ ctx.strokeStyle = S.grid; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); }
    if(S.full){ ctx.strokeStyle = S.axis; ctx.beginPath(); ctx.moveTo(pad.l - f*0.4, y); ctx.lineTo(pad.l, y); ctx.stroke(); }
    ctx.fillStyle = S.dim;
    ctx.fillText(fmt(v), pad.l - (S.full ? f*0.7 : 4*f/10), y);
  }
}
function xAxisTicks(ctx, w, h, L, S, X, px, o = {}){
  const { pad } = L, f = S.fontPx, base = h - pad.b;
  ctx.textAlign = 'center'; ctx.lineWidth = S.full ? Math.max(0.35, S.lineW*0.5) : 1;
  for(const v of ticksOf(X)){
    const x = snap(S, px(v));
    if(S.showGrid && o.grid !== false){ ctx.strokeStyle = S.grid; ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, base); ctx.stroke(); }
    if(S.full){ ctx.strokeStyle = S.axis; ctx.beginPath(); ctx.moveTo(x, base); ctx.lineTo(x, base + f*0.4); ctx.stroke(); }
    ctx.fillStyle = S.dim;
    ctx.fillText(fmt(v) + (o.unit || ''), x, base + (S.full ? f*1.1 : 9*f/10));
  }
}

// events: [[t, row], ...] over rows rows, between t0 and t1 (ms); one mark per spike, rows from the top, the time axis in seconds opts: { rows, t0, t1, color, empty, title, xLabel, yLabel, style }
export function drawRaster(ctx, w, h, events, opts = {}){
  const S = opts.style || SCREEN;
  const rows = opts.rows | 0, t0 = opts.t0 || 0, t1 = opts.t1 || 1;
  ground(ctx, w, h, S);
  const some = events && events.length && rows && t1 > t0;
  const Yb = { lo:0, hi:Math.max(1, rows), step:niceStep(Math.max(1, rows), 4) };
  const L = layout(ctx, w, h, S, { yTicks:some ? ticksOf(Yb).map(fmt) : ['0'], title:opts.title,
    xLabel:opts.xLabel, yLabel:opts.yLabel });
  axes(ctx, w, h, L, S);
  if(!some){ empty(ctx, w, h, S, opts.empty || 'no spikes yet'); return null; }
  const { pad, iw, ih } = L;
  const x = t => pad.l + (t - t0)/(t1 - t0)*iw;
  const y = r => pad.t + (r + 0.5)/rows*ih;
  const X = { lo:t0/1000, hi:t1/1000, step:niceStep((t1 - t0)/1000, 5) };
  xAxisTicks(ctx, w, h, L, S, X, v => x(v*1000), { grid:false, unit:S.full ? '' : ' s' });
  if(S.full){
    const f = S.fontPx;
    ctx.textAlign = 'right'; ctx.fillStyle = S.dim; ctx.strokeStyle = S.axis;
    for(const v of ticksOf(Yb)){ if(v > rows) continue;
      const yy = pad.t + v/rows*ih;
      ctx.beginPath(); ctx.moveTo(pad.l - f*0.4, yy); ctx.lineTo(pad.l, yy); ctx.stroke();
      ctx.fillText(fmt(v), pad.l - f*0.7, yy); }
  } else {
    ctx.fillStyle = S.dim; ctx.textAlign = 'right';
    ctx.fillText('0', pad.l - 4, pad.t + 4);
    ctx.fillText(String(rows), pad.l - 4, h - pad.b - 4);
  }
  ctx.fillStyle = opts.color || S.ink;
  const dot = Math.max(S.full ? 0.4 : 1, Math.min(S.full ? S.fontPx*0.35 : 2, ih/rows*0.8));
  const wide = S.full ? Math.max(0.4, S.lineW*0.6) : 1;
  for(const [t, r] of events) if(t >= t0 && t <= t1) ctx.fillRect(x(t), y(r) - dot/2, wide, dot);
  furniture(ctx, w, h, L, S, opts);
  return { x, y, pad };
}

// series: [{ label, color, points: [[x, y], ...], band: [[x, lo, hi], ...] }] A band is the spread around its series: drawn filled and faint behind the lines, and counted in the axis range.
// opts:   { title, xLabel, yLabel, yLo, yHi (extend the range), yMin, yMax
//           (fix it), xTight, marks: [{ y, label, color }], vlines: [{ x, label,
//           strong }], empty, style }
export function drawLine(ctx, w, h, series, opts = {}){
  const S = opts.style || SCREEN;
  const live = series.filter(s => s.points && s.points.length);
  ground(ctx, w, h, S);
  if(!live.length){
    axes(ctx, w, h, layout(ctx, w, h, S, { yTicks:['0'], title:opts.title, xLabel:opts.xLabel, yLabel:opts.yLabel }), S);
    empty(ctx, w, h, S, opts.empty || 'no data yet');
    return null;
  }
  let xLo = Infinity, xHi = -Infinity, yLo = Infinity, yHi = -Infinity;
  for(const s of live) for(const [x, y] of s.points){
    if(!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if(x < xLo) xLo = x; if(x > xHi) xHi = x;
    if(y < yLo) yLo = y; if(y > yHi) yHi = y;
  }
  // a band is part of the data: an axis that cuts the spread off understates it, which is the one thing a band must not do
  for(const s of live) for(const [x, lo, hi] of s.band || []){
    if(!Number.isFinite(x)) continue;
    if(x < xLo) xLo = x; if(x > xHi) xHi = x;
    if(Number.isFinite(lo) && lo < yLo) yLo = lo;
    if(Number.isFinite(hi) && hi > yHi) yHi = hi;
  }
  for(const m of opts.marks || []){
    if(!Number.isFinite(m.y)) continue;
    if(m.y < yLo) yLo = m.y; if(m.y > yHi) yHi = m.y;
  }
  if(opts.yLo !== undefined) yLo = Math.min(yLo, opts.yLo);
  if(opts.yHi !== undefined) yHi = Math.max(yHi, opts.yHi);
  // a time axis runs over exactly the data (xTight): rounding it out would leave empty plot at both ends of a window
  const X = opts.xTight && xHi > xLo ? { lo:xLo, hi:xHi, step:niceStep(xHi - xLo, 5) } : niceBounds(xLo, xHi === xLo ? xLo + 1 : xHi, 4);
  const Y = fixedBounds(yLo, yHi, opts.yMin, opts.yMax, 4);
  live.forEach((s, i) => { if(!s.color) s.color = live.length === 1 ? S.series[0] : S.series[i % S.series.length]; });
  const L = layout(ctx, w, h, S, { yTicks:ticksOf(Y).map(fmt), title:opts.title, xLabel:opts.xLabel,
    yLabel:opts.yLabel, legend:live.filter(s => s.label).map(s => ({ label:s.label, color:s.color })) });
  const { pad, iw, ih } = L;
  const px = v => pad.l + (v - X.lo)/(X.hi - X.lo)*iw;
  const py = v => h - pad.b - (v - Y.lo)/(Y.hi - Y.lo)*ih;
  yAxisTicks(ctx, w, h, L, S, Y, py);
  xAxisTicks(ctx, w, h, L, S, X, px);
  axes(ctx, w, h, L, S);

  // reference lines, chance being the one that matters most: a decode number without its chance line is not a claim about anything
  for(const m of opts.marks || []){
    if(!Number.isFinite(m.y) || m.y < Y.lo || m.y > Y.hi) continue;
    ctx.strokeStyle = m.color || S.mark; ctx.lineWidth = S.full ? Math.max(0.5, S.lineW*0.75) : 1;
    ctx.setLineDash([3, 3]);
    const y = snap(S, py(m.y));
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.setLineDash([]);
    if(m.label){
      ctx.fillStyle = m.color || S.mark; ctx.textAlign = 'left';
      ctx.fillText(m.label, pad.l + 3, y - S.fontPx*0.6);
    }
  }
  if(opts.vlines && opts.vlines.length){
    ctx.font = S.small; ctx.textAlign = 'left';
    let lastX = -Infinity;
    for(const e of opts.vlines){
      const x = snap(S, px(e.x));
      if(x < pad.l || x > w - pad.r) continue;
      ctx.strokeStyle = e.strong ? S.event : S.eventDim; ctx.lineWidth = S.full ? Math.max(0.35, S.lineW*0.5) : 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke(); ctx.setLineDash([]);
      if(e.strong && e.label && x - lastX > S.fontPx*4){ ctx.fillStyle = S.event; ctx.fillText(e.label, x + 2, pad.t + S.fontPx*0.9); lastX = x; }
    }
    ctx.font = S.font;
  }

  const clip = Y.fixed && ctx.clip;
  if(clip){ ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, iw, ih); ctx.clip(); }
  // Every band is drawn before every line, so a band never covers a neighbor's line, and the alpha is set back rather than saved, since the SVG context stacks transforms and clips and not this.
  for(const s of live){
    const b = (s.band || []).filter(([x, lo, hi]) => Number.isFinite(x) && Number.isFinite(lo) && Number.isFinite(hi));
    if(b.length < 2) continue;
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = s.color;
    ctx.beginPath();
    b.forEach(([x, , hi], i) => { const cx = px(x), cy = py(hi); if(i) ctx.lineTo(cx, cy); else ctx.moveTo(cx, cy); });
    for(let i = b.length - 1; i >= 0; i--) ctx.lineTo(px(b[i][0]), py(b[i][1]));
    ctx.closePath(); ctx.fill();
    ctx.globalAlpha = 1;
  }
  for(const s of live){
    ctx.strokeStyle = s.color; ctx.lineWidth = S.lineW;
    ctx.beginPath();
    let started = false;
    for(const [x, y] of s.points){
      if(!Number.isFinite(x) || !Number.isFinite(y)){ started = false; continue; }
      const cx = px(x), cy = py(y);
      if(started) ctx.lineTo(cx, cy); else { ctx.moveTo(cx, cy); started = true; }
    }
    ctx.stroke();
    // a single point draws nothing as a line, so mark it
    if(s.points.length === 1){
      const [x, y] = s.points[0], d = Math.max(3, S.lineW*3);
      ctx.fillStyle = s.color;
      ctx.fillRect(px(x) - d/2, py(y) - d/2, d, d);
    }
  }
  if(clip) ctx.restore();
  furniture(ctx, w, h, L, S, opts);
  return { X, Y, px, py, pad };
}

// counts: an array of bin heights; lo and hi are the values at the two ends of the bins, and put a ticked axis under them opts: { lo, hi, color, yMax, empty, title, xLabel, yLabel, style }
export function drawHistogram(ctx, w, h, counts, opts = {}){
  const S = opts.style || SCREEN;
  ground(ctx, w, h, S);
  if(!counts || !counts.length){
    axes(ctx, w, h, layout(ctx, w, h, S, { yTicks:['0'], title:opts.title, xLabel:opts.xLabel, yLabel:opts.yLabel }), S);
    empty(ctx, w, h, S, opts.empty || 'no data yet');
    return null;
  }
  let hi = 0;
  for(const c of counts) if(c > hi) hi = c;
  const Y = fixedBounds(0, hi || 1, 0, opts.yMax, 3);
  const L = layout(ctx, w, h, S, { yTicks:ticksOf(Y).map(fmt), title:opts.title, xLabel:opts.xLabel, yLabel:opts.yLabel });
  const { pad, iw, ih } = L;
  const bw = iw/counts.length;
  const py = v => h - pad.b - (Math.min(v, Y.hi) - Y.lo)/(Y.hi - Y.lo)*ih;
  yAxisTicks(ctx, w, h, L, S, Y, py);
  ctx.fillStyle = opts.color || S.bar;
  const gap = S.full ? Math.min(bw*0.12, S.fontPx*0.15) : 1;
  for(let i = 0; i < counts.length; i++){
    const x = pad.l + i*bw, top = py(counts[i]);
    ctx.fillRect(x + gap/2, top, Math.max(S.full ? 0.2 : 1, bw - gap), h - pad.b - top);
  }
  axes(ctx, w, h, L, S);
  if(opts.lo !== undefined && opts.hi !== undefined && opts.hi > opts.lo){
    if(S.full){
      const X = { lo:opts.lo, hi:opts.hi, step:niceStep(opts.hi - opts.lo, 5) };
      xAxisTicks(ctx, w, h, L, S, X, v => pad.l + (v - X.lo)/(X.hi - X.lo)*iw, { grid:false });
    } else {
      ctx.fillStyle = S.dim;
      ctx.textAlign = 'left'; ctx.fillText(fmt(opts.lo), pad.l, h - pad.b + 9*S.fontPx/10);
      ctx.textAlign = 'right'; ctx.fillText(fmt(opts.hi), w - pad.r, h - pad.b + 9*S.fontPx/10);
    }
  }
  furniture(ctx, w, h, L, S, S.full ? opts : { ...opts, xLabel:undefined });
  return { Y, pad };
}

// values: Float32Array/Array of rows*cols, row major.
// Drawn through a bitmap at data resolution and scaled up, so a matrix of thousands of cells costs one draw rather than thousands.
// A full figure carries a color bar with the value range on it. opts: { vMin, vMax (fix the color range), barLabel, title, xLabel,
//         yLabel, empty, style }
export function drawMatrix(ctx, w, h, values, rows, cols, opts = {}){
  const S = opts.style || SCREEN;
  ground(ctx, w, h, S);
  if(!values || !rows || !cols){ empty(ctx, w, h, S, opts.empty || 'no data yet'); return null; }
  let lo = Infinity, hi = -Infinity;
  for(let i = 0; i < rows*cols; i++){
    const v = values[i];
    if(!Number.isFinite(v)) continue;
    if(v < lo) lo = v; if(v > hi) hi = v;
  }
  if(Number.isFinite(opts.vMin)) lo = opts.vMin;
  if(Number.isFinite(opts.vMax)) hi = opts.vMax;
  if(!(hi > lo)){ hi = lo + 1; }
  const f = S.fontPx;
  const B = { lo, hi, step:niceStep(hi - lo, 4) };
  let barW = 0;
  if(S.full){ ctx.font = S.font; for(const v of ticksOf(B)) barW = Math.max(barW, ctx.measureText(fmt(v)).width);
    barW += f*3.0 + (opts.barLabel ? f*1.6 : 0); }
  const L = layout(ctx, w, h, S, { yTicks:[String(rows)], title:opts.title, xLabel:opts.xLabel, yLabel:opts.yLabel, right:barW });
  const { pad, iw, ih } = L;
  const rgba = new Uint8ClampedArray(rows*cols*4);
  for(let i = 0; i < rows*cols; i++){
    const t = Math.max(0, Math.min(1, (values[i] - lo)/(hi - lo)));
    // lightness carries the value: a rainbow map invents structure at its color boundaries that is not in the numbers
    const c = Number.isFinite(values[i]) ? ramp(t, S.cmap) : [128, 128, 128];
    rgba[i*4] = c[0]; rgba[i*4+1] = c[1]; rgba[i*4+2] = c[2]; rgba[i*4+3] = 255;
  }
  pixels(ctx, rgba, cols, rows, pad.l, pad.t, iw, ih);
  ctx.strokeStyle = S.axis; ctx.lineWidth = S.full ? Math.max(0.5, S.lineW*0.75) : 1;
  ctx.strokeRect(snap(S, pad.l), snap(S, pad.t), iw, ih);
  ctx.fillStyle = S.dim;
  if(S.full){
    // row and column counts at the corners of the axes, which is what a matrix of items or cells can honestly be ticked with
    ctx.textAlign = 'right'; ctx.fillText('1', pad.l - f*0.6, pad.t + f*0.5); ctx.fillText(String(rows), pad.l - f*0.6, h - pad.b - f*0.5);
    ctx.textAlign = 'left'; ctx.fillText('1', pad.l, h - pad.b + f*1.1);
    ctx.textAlign = 'right'; ctx.fillText(String(cols), pad.l + iw, h - pad.b + f*1.1);
    const bx = pad.l + iw + f*1.0, bwid = f*0.9, n = 64, bar = new Uint8ClampedArray(n*4);
    for(let i = 0; i < n; i++){ const c = ramp(1 - i/(n - 1), S.cmap); bar[i*4] = c[0]; bar[i*4+1] = c[1]; bar[i*4+2] = c[2]; bar[i*4+3] = 255; }
    pixels(ctx, bar, 1, n, bx, pad.t, bwid, ih);
    ctx.strokeRect(bx, pad.t, bwid, ih);
    ctx.textAlign = 'left';
    for(const v of ticksOf(B)){
      const y = pad.t + (1 - (v - lo)/(hi - lo))*ih;
      ctx.beginPath(); ctx.moveTo(bx + bwid, y); ctx.lineTo(bx + bwid + f*0.35, y); ctx.stroke();
      ctx.fillText(fmt(v), bx + bwid + f*0.6, y);
    }
    if(opts.barLabel){ ctx.save(); ctx.translate(w - f*0.9, pad.t + ih/2); ctx.rotate(-Math.PI/2);
      ctx.textAlign = 'center'; ctx.fillStyle = S.ink; ctx.fillText(opts.barLabel, 0, 0); ctx.restore(); }
  } else {
    ctx.textAlign = 'left';
    ctx.fillText(`${rows} x ${cols}`, pad.l, h - pad.b + 9*f/10);
    ctx.textAlign = 'right';
    ctx.fillText(`${fmt(lo)} to ${fmt(hi)}`, w - pad.r, h - pad.b + 9*f/10);
  }
  furniture(ctx, w, h, L, S, S.full ? opts : { ...opts, xLabel:undefined });
  return { lo, hi, pad };
}
// A block of pixels stretched over a rectangle, unsmoothed.
// A context that takes raw pixels (the SVG one, which embeds them as an image) is handed them; a canvas goes through a bitmap, since putImageData ignores transforms.
function pixels(ctx, rgba, cols, rows, x, y, w, h){
  if(ctx.drawPixels){ ctx.drawPixels(rgba, cols, rows, x, y, w, h); return; }
  const img = ctx.createImageData(cols, rows);
  img.data.set(rgba);
  ctx.imageSmoothingEnabled = false;
  const off = new OffscreenCanvas(cols, rows);
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.drawImage(off, x, y, w, h);
}

// Color maps, every one monotonic in lightness, so ordering is readable and no false edge appears mid-scale. teal: black through the app's teal to white. grey: white to black, for print. viridis: van der Walt and Smith's map (matplotlib), perceptually uniform and readable under color blindness, here as nine stops interpolated linearly.
const RAMPS = {
  teal: [[0, 0, 0], [16, 48, 44], [40, 120, 100], [111, 211, 176], [220, 250, 240]],
  grey: [[255, 255, 255], [0, 0, 0]],
  viridis: [[68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141],
    [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37]],
};
export function ramp(t, name = 'teal'){
  const stops = RAMPS[name] || RAMPS.teal;
  const x = Math.max(0, Math.min(0.999999, t))*(stops.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[i+1] || stops[i];
  return [a[0] + (b[0]-a[0])*f, a[1] + (b[1]-a[1])*f, a[2] + (b[2]-a[2])*f];
}

// Lines of provenance under a figure: what was plotted, of which population, at what simulated time, from which scene, and when.
// A figure pasted into a report carries its provenance on its face.
export function captionHeight(S, lines){ return lines && lines.length ? S.fontPx*(1.2 + lines.length*1.45) : 0; }
export function drawCaption(ctx, x, y, lines, S){
  ctx.font = S.font; ctx.textBaseline = 'middle'; ctx.textAlign = 'left'; ctx.fillStyle = S.dim;
  lines.forEach((ln, i) => ctx.fillText(ln, x, y + S.fontPx*(1.2 + i*1.45)));
}

// A histogram of arbitrary values, since every distribution in ANALYSIS.md arrives as a flat array and has to be binned before it can be drawn.
export function bin(values, nBins, lo, hi){
  const vals = values || [];
  if(lo === undefined || hi === undefined){
    lo = Infinity; hi = -Infinity;
    for(const v of vals){ if(!Number.isFinite(v)) continue;
      if(v < lo) lo = v; if(v > hi) hi = v; }
  }
  if(!Number.isFinite(lo) || !Number.isFinite(hi)) return { counts:[], lo:0, hi:1 };
  if(hi === lo) hi = lo + 1;
  const counts = new Float64Array(Math.max(1, nBins));
  for(const v of vals){
    if(!Number.isFinite(v)) continue;
    let b = Math.floor((v - lo)/(hi - lo)*nBins);
    if(b < 0) b = 0; if(b >= nBins) b = nBins - 1;
    counts[b]++;
  }
  return { counts, lo, hi };
}

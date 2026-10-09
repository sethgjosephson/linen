// Figures that leave the app: the numbers behind a chart as CSV or as spike trains in the plain ASCII layout Neo reads (one cell per line, spike times in ms separated by spaces).
// The text builders are pure and tested; the browser calls (a download, a canvas's bytes) are at the bottom.
// The figure itself is drawn by chartpaint.js, as a PNG (png.js adds its physical size and its provenance) or as SVG (svgctx.js).

// Cells are quoted only when they need it.
// Numbers are written as given.
export function csv(header, rows){
  const cell = v => { const s = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  return [header, ...rows].map(r => r.map(cell).join(',')).join('\n') + '\n';
}
// [t_ms, hz] points
export function seriesCsv(points, label){
  return csv(['t_ms', (label || 'population') + '_hz'], points.map(([t, y]) => [t, +y.toFixed(4)]));
}
// events [t_ms, row], with the cell each row stands for
export function rasterCsv(events, cells){
  return csv(['t_ms', 'row', 'cell'], events.map(([t, r]) => [t, r, cells[r] !== undefined ? cells[r] : '']));
}
// Neo's AsciiSpikeTrainIO layout: one line per cell, the spike times in ms separated by spaces; a silent cell is an empty line, so the row count is the cell count
export function spikeTrainsAscii(events, rows){
  const per = Array.from({ length:rows }, () => []);
  for(const [t, r] of events) if(r < rows) per[r].push(t);
  return per.map(a => a.join(' ')).join('\n') + '\n';
}
// Several means on one x axis, each with the range behind it: the columns are the mean, the low and the high of every group, so the spread survives the export rather than only the picture of it.
export function bandSeriesCsv(series, xName = 'x', yName = 'value'){
  const at = new Map();
  const cell = (x, k) => { if(!at.has(x)) at.set(x, new Map()); const r = at.get(x);
    if(!r.has(k)) r.set(k, {}); return r.get(k); };
  series.forEach((s, k) => {
    for(const [x, y] of s.points || []) cell(x, k).mean = y;
    for(const [x, lo, hi] of s.band || []){ const c = cell(x, k); c.lo = lo; c.hi = hi; }
  });
  const head = [xName];
  for(const s of series) head.push(s.label + ' ' + yName + ' mean', s.label + ' low', s.label + ' high');
  // rounded like the other series exports, so a mean of three numbers does not arrive as 0.30000000000000004
  const num = v => v === undefined ? '' : +(+v).toFixed(6);
  const rows = [...at.keys()].sort((a, b) => a - b).map(x => {
    const r = [x];
    series.forEach((s, k) => { const c = at.get(x).get(k) || {}; r.push(num(c.mean), num(c.lo), num(c.hi)); });
    return r;
  });
  return csv(head, rows);
}
export function histogramCsv(counts, edges, name = 'count'){
  return csv(['bin_from', 'bin_to', name], counts.map((c, i) => [edges[i], edges[i + 1] !== undefined ? edges[i + 1] : '', c]));
}
export function matrixCsv(values, rows, cols, rowName = 'row'){
  const out = [];
  for(let r = 0; r < rows; r++){ const line = [r]; for(let c = 0; c < cols; c++) line.push(values[r*cols + c]); out.push(line); }
  return csv([rowName, ...Array.from({ length:cols }, (_, c) => 'c' + c)], out);
}
// a file name from the parts, safe on every file system
export function figureName(parts, ext){
  return parts.filter(Boolean).map(s => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')).filter(Boolean).join('_') + '.' + ext;
}

// ---- browser side ----
export function download(name, blob){
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
export function downloadText(name, text){ download(name, new Blob([text], { type:'text/plain' })); }
// several named series on one time base: t, then a column each
export function multiSeriesCsv(series){
  const ts = new Map();
  series.forEach((s, k) => { for(const [t, y] of s.points){ if(!ts.has(t)) ts.set(t, []); ts.get(t)[k] = +y.toFixed(4); } });
  const rows = [...ts.keys()].sort((a, b) => a - b).map(t => [t, ...series.map((_, k) => ts.get(t)[k] === undefined ? '' : ts.get(t)[k])]);
  return csv(['t_ms', ...series.map(s => s.label + '_hz')], rows);
}
export function canvasPng(canvas){
  return new Promise((res, rej) => canvas.toBlob(b => b ? b.arrayBuffer().then(a => res(new Uint8Array(a)), rej) : rej(new Error('the canvas could not be encoded')), 'image/png'));
}

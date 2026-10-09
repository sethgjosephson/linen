// The figure that leaves the app, checked without a browser: the PNG bytes (size, provenance, a pixel encoder), the SVG the chart functions write through the SVG context, and the chart node's painter driven from a real recorder.
// What a reader of a figure relies on is what is pinned here: the physical size, the axis titles and units, the range asked for, a legend that names every series, text that stays inside the page.

import zlib from 'node:zlib';
import { ok, report } from '../tools/harness.mjs';
import { crc32, chunk, withMeta, readMeta, encodePng, base64 } from './png.js';
import { SvgContext } from './svgctx.js';
import * as C from './chart.js';
import { paintChart, paintFigure, figureSpec, screenStyle, numOrAuto, chartSeries, meanBand } from './chartpaint.js';
import { Recorder } from './record.js';
import { multiSeriesCsv, bandSeriesCsv } from './figure.js';

// ---- PNG bytes ----
{
  const iend = chunk('IEND', new Uint8Array(0));
  ok('the CRC of an empty IEND chunk is the known one (AE 42 60 82)', [...iend.subarray(8)].join() === '174,66,96,130', [...iend].join());
  ok('crc32 of "123456789" is CBF43926', ((crc32(Uint8Array.from('123456789', c => c.charCodeAt(0))) ^ 0xffffffff) >>> 0) === 0xcbf43926);
  const rgba = new Uint8ClampedArray(3*2*4);
  for(let i = 0; i < 6; i++){ rgba[i*4] = i*40; rgba[i*4+1] = 255 - i*40; rgba[i*4+2] = 7; rgba[i*4+3] = 255; }
  const png = encodePng(rgba, 3, 2);
  // read it back the way any decoder would: inflate the IDAT, strip the filter bytes
  let p = 8, idat = null;
  while(p < png.length){ const len = (png[p] << 24 | png[p+1] << 16 | png[p+2] << 8 | png[p+3]) >>> 0;
    if(String.fromCharCode(...png.subarray(p + 4, p + 8)) === 'IDAT') idat = png.subarray(p + 8, p + 8 + len); p += 12 + len; }
  const raw = zlib.inflateSync(Buffer.from(idat));
  const back = []; for(let y = 0; y < 2; y++) back.push(...raw.subarray(y*13 + 1, y*13 + 13));
  ok('a PNG written from pixels inflates to the same pixels', back.join() === [...rgba].join());
  // a big one crosses the 65535 byte stored block
  const bigPx = new Uint8ClampedArray(200*200*4).fill(9);
  ok('pixel data past one stored block still inflates', zlib.inflateSync(Buffer.from((() => { const b = encodePng(bigPx, 200, 200); let q = 8, d = null;
    while(q < b.length){ const len = (b[q] << 24 | b[q+1] << 16 | b[q+2] << 8 | b[q+3]) >>> 0; if(String.fromCharCode(...b.subarray(q + 4, q + 8)) === 'IDAT') d = b.subarray(q + 8, q + 8 + len); q += 12 + len; } return d; })())).length === 200*(200*4 + 1));
  const m = withMeta(png, { dpi:300, text:{ Title:'rate over time, L4e', Description:'linen · column · sim 12.0 s' } });
  const r = readMeta(m);
  ok('the physical size is written and read back: 300 dpi', r.dpi === 300 && r.width === 3 && r.height === 2, JSON.stringify(r));
  ok('the text survives, including characters outside Latin-1', r.text.Title === 'rate over time, L4e' && r.text.Description === 'linen · column · sim 12.0 s');
  ok('a second pass replaces the size instead of adding one', readMeta(withMeta(m, { dpi:600 })).dpi === 600 &&
    [...withMeta(m, { dpi:600 })].join().split([112, 72, 89, 115].join()).length === 2);
  ok('base64 of three bytes', base64(Uint8Array.from([1, 2, 3])) === 'AQID');
}

// ---- scales ----
{
  const B = C.fixedBounds(3, 47, 0, 100);
  ok('a fixed range is kept exactly', B.lo === 0 && B.hi === 100 && B.fixed === true && B.step === 20, JSON.stringify(B));
  const H = C.fixedBounds(3, 47, undefined, 40);
  ok('one fixed end, the other from the data on a round number', H.hi === 40 && H.lo === 0, JSON.stringify(H));
  ok('no fixed end is the automatic range', JSON.stringify(C.fixedBounds(3, 47)) === JSON.stringify(C.niceBounds(3, 47)));
  ok('ticks are the multiples of the step inside the range', C.ticksOf({ lo:0, hi:1, step:0.25 }).join() === '0,0.25,0.5,0.75,1' && C.ticksOf({ lo:-3, hi:7, step:5 }).join() === '0,5');
  ok('a tick label of a thousand or more is written out', C.fmt(1500) === '1500' && C.fmt(25000) === '25000' && C.fmt(0.5) === '0.5');
  const pts = [[0, 0], [10, 10], [20, 20], [30, 30], [40, 100]];
  const sm = C.boxcar(pts, 20);
  ok('a moving average keeps the times and averages within the window', sm.map(p => p[0]).join() === '0,10,20,30,40' && sm[1][1] === 10 && Math.abs(sm[3][1] - 50) < 1e-9 && sm[0][1] === 5, JSON.stringify(sm));
  ok('a zero window is the data', C.boxcar(pts, 0) === pts);
  for(const name of ['teal', 'grey', 'viridis']){
    const lum = t => { const c = C.ramp(t, name); return 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]; };
    let mono = true; const up = lum(1) > lum(0);
    for(let i = 1; i <= 50; i++) if((lum(i/50) - lum((i - 1)/50))*(up ? 1 : -1) < -0.5) mono = false;
    ok('the ' + name + ' color map is monotonic in lightness', mono);
  }
  ok('a blank setting is automatic and a number is a number', numOrAuto('') === undefined && numOrAuto('  ') === undefined && numOrAuto('0') === 0 && numOrAuto('-65.5') === -65.5 && numOrAuto('abc') === undefined);
}

// ---- the figure size ----
{
  const s = figureSpec({});
  ok('the default figure is two columns wide at 300 dpi: 2126 by 1181 pixels', s.pxW === 2126 && s.pxH === 1181 && s.dpi === 300 && Math.abs(s.wPt - 510.24) < 0.01, s.pxW + ' ' + s.pxH);
  ok('the default style is print: white ground, sans serif, 8 pt, full layout', s.style.bg === '#fff' && s.style.fontPx === 8 && /Arial/.test(s.style.family) && s.style.full);
  ok('a figure past what a canvas holds says so', figureSpec({ figW:1000, figH:1000, dpi:1200 }).tooBig && !s.tooBig);
  ok('settings are clamped to sane values', figureSpec({ dpi:5, figW:1 }).dpi === 72 && figureSpec({ dpi:5, figW:1 }).mmW === 20);
  ok('the panel style is the compact screen one', !screenStyle({}).full && screenStyle({}).bg === null && screenStyle({ grid:0 }).showGrid === false && screenStyle({}).grid === '#1c1c1c');
}

// ---- the painter, through the SVG context ----
const R = new Recorder();
R.setPopulations([{ label:'L4e', idx:Int32Array.from({ length:40 }, (_, i) => i) }, { label:'L5e', idx:Int32Array.from({ length:40 }, (_, i) => 40 + i) }]);
const fired = new Uint8Array(80);
for(let f = 0; f < 400; f++){ fired.fill(0); for(let i = 0; i < 80; i++) if((i*7 + f*3) % (i < 40 ? 23 : 11) === 0) fired[i] = 1; R.onState(fired, 5); }
const scope = Array.from({ length:200 }, (_, i) => -65 + 20*Math.sin(i/9));
const recording = { labels:() => ['L4e', 'L5e'], series:l => R.series(l, 1e9, 1), events:[{ t:500, kind:'edit', node:'stimulus1', key:'amp' }, { t:900, kind:'mark' }],
  edits:() => 1, seconds:() => 2, binMs:20, csv:() => 'x' };
const live = () => ({ rec:R, scope:() => scope, recording:() => recording });
const mats = new Map([['L4e', { values:Float32Array.from({ length:12*30 }, (_, i) => i % 17), rows:12, cols:30, simMin:4 }]]);
const history = [1, 2, 3, 4].map(k => ({ simMin:k*5, probes:[{ l:'L4e', hz:k }], readout:{ chance:0.1, pops:{ L4e:{ decode:0.1 + k*0.2 } } } }));
const src = { history, live, matrices:() => mats, weights:async () => ({ w:Float32Array.from({ length:500 }, (_, i) => 0.1 + (i % 50)/50), pmask:new Uint8Array(500).fill(1) }) };

const svgOf = async (P) => { const spec = figureSpec(P); const ctx = new SvgContext(spec.wPt, spec.hPt, { widthMm:spec.mmW, heightMm:spec.mmH, title:'t', meta:{ a:1 } });
  const r = await paintFigure(ctx, spec, P, src, (P.caption | 0) === 0 ? ['linen · scene · plot', 'sim 2.0 s'] : []); return { svg:ctx.toString(), r, spec }; };
const texts = svg => [...svg.matchAll(/<text[^>]*>([^<]*)<\/text>/g)].map(m => m[1]);
const inside = (svg, spec) => [...svg.matchAll(/<text x="([-\d.]+)" y="([-\d.]+)"[^>]*?(transform="[^"]*")?>/g)].every(m => m[3] || (+m[1] >= 0 && +m[1] <= spec.wPt && +m[2] >= 0 && +m[2] <= spec.hPt));

{
  const { svg, r, spec } = await svgOf({ plot:7, pop:'L4e', window:10 });
  ok('the SVG states its physical size and a viewBox in points', /width="180mm" height="100mm" viewBox="0 0 510\.24 283\.46"/.test(svg), svg.slice(0, 200));
  ok('the rate plot carries its axis titles with units', texts(svg).includes('time (s)') && texts(svg).includes('rate (Hz)'));
  ok('the y axis title is rotated', /<text[^>]*transform="matrix\(0 -1 1 0[^>]*>rate \(Hz\)</.test(svg));
  ok('the caption is stamped under the plot', texts(svg).includes('linen · scene · plot') && texts(svg).includes('sim 2.0 s'));
  ok('provenance rides in the file as well', /<title>t<\/title>/.test(svg) && /<metadata>\{&quot;a&quot;:1\}<\/metadata>/.test(svg));
  ok('the data is one path in the first series color', (svg.match(/stroke="#0072B2"/g) || []).length === 1);
  ok('every piece of text sits inside the page', inside(svg, spec));
  ok('the numbers behind it are handed back', r.chart.kind === 'series' && r.chart.points.length > 100 && /Hz now/.test(r.note));
  ok('no canvas-only call leaks into the SVG', !/undefined|NaN/.test(svg));
  ok('grid lines are drawn in the grid color, not the ink', (svg.match(/stroke="#e2e2e2"/g) || []).length >= 6 && !/stroke="true"/.test(svg));
}
{
  const { svg, r } = await svgOf({ plot:7, pop:'', window:10, title:'Layer rates', caption:1 });
  ok('no population named draws every probed one, with a legend naming each', texts(svg).includes('L4e') && texts(svg).includes('L5e') && r.chart.kind === 'multi');
  ok('a title is drawn and a caption can be left off', texts(svg).includes('Layer rates') && !texts(svg).includes('sim 2.0 s'));
  ok('two series take two colors of the print set', /stroke="#0072B2"/.test(svg) && /stroke="#D55E00"/.test(svg));
  const csvText = multiSeriesCsv(r.chart.series);
  ok('several series save as one table on a shared time column', csvText.split('\n')[0] === 't_ms,L4e_hz,L5e_hz' && csvText.split('\n').length > 100);
  const off = await svgOf({ plot:7, pop:'', legend:0, caption:1 });
  ok('the legend can be switched off', !texts(off.svg).includes('L5e'));
}
{
  const a = await svgOf({ plot:7, pop:'L4e', window:10, yMin:'0', yMax:'500', xText:'seconds', yText:'spikes per second per cell' });
  ok('a fixed range puts its ends on the axis', texts(a.svg).includes('0') && texts(a.svg).includes('500'));
  ok('typed axis titles replace the automatic ones', texts(a.svg).includes('seconds') && texts(a.svg).includes('spikes per second per cell') && !texts(a.svg).includes('rate (Hz)'));
  ok('data under a fixed range is clipped to the plot area', /<clipPath id="clip1">/.test(a.svg) && /clip-path="url\(#clip1\)"/.test(a.svg));
  const raw = await svgOf({ plot:7, pop:'L4e', window:10 }), sm = await svgOf({ plot:7, pop:'L4e', window:10, smooth:200 });
  const top = s => Math.max(...texts(s).map(Number).filter(Number.isFinite));
  ok('a moving average lowers the peaks and is named in the note', top(sm.svg) <= top(raw.svg) && /moving average of 200 ms/.test(sm.r.note) && sm.r.smoothMs === 200);
  ok('the saved data stays as recorded under a moving average', JSON.stringify(sm.r.chart.points) === JSON.stringify(raw.r.chart.points));
}
{
  const { svg, r, spec } = await svgOf({ plot:8, pop:'L5e', window:10 });
  ok('the raster is one merged path of spikes, not an element each', (svg.match(/<path d="M[^"]{2000,}" fill="#000"/g) || []).length === 1, String((svg.match(/<path/g) || []).length));
  ok('the raster has a time axis in seconds and a cell axis', texts(svg).includes('time (s)') && texts(svg).includes('cell') && inside(svg, spec));
  ok('the raster hands back its events and cells', r.chart.kind === 'raster' && r.chart.rows === 40 && r.chart.events.length > 100);
}
{
  const { svg, r } = await svgOf({ plot:9, pop:'L4e' });
  ok('the interval histogram has a ticked x axis in ms', texts(svg).includes('interval between spikes (ms)') && texts(svg).includes('intervals') && r.chart.kind === 'hist');
  const w = await svgOf({ plot:1, bins:20 });
  ok('the weight distribution draws from the engine readback', texts(w.svg).includes('excitatory weight') && texts(w.svg).includes('synapses') && w.r.chart.counts.length === 20);
}
{
  const { svg, r } = await svgOf({ plot:10 });
  ok('the membrane trace is in mV with its reference lines', texts(svg).includes('membrane potential (mV)') && texts(svg).includes('peak') && texts(svg).includes('reset') && r.chart.kind === 'trace');
  const rec = await svgOf({ plot:11, pop:'' });
  ok('the recording marks its edits in time and labels them', texts(rec.svg).includes('stimulus1 amp') && (rec.svg.match(/stroke-dasharray="2 3"/g) || []).length === 2);
}
{
  const { svg, r, spec } = await svgOf({ plot:2, pop:'L4e' });
  ok('a matrix is embedded as an unsmoothed PNG at data resolution', /<image [^>]*image-rendering:pixelated[^>]*href="data:image\/png;base64,/.test(svg) && (svg.match(/<image/g) || []).length === 2);
  ok('a matrix figure carries a color bar with its label and range', texts(svg).includes('mean response') && texts(svg).includes('15') && texts(svg).includes('item') && texts(svg).includes('cell') && inside(svg, spec));
  const v = await svgOf({ plot:2, pop:'L4e', yMin:'0', yMax:'40', cmap:1 });
  ok('a fixed color range is on the bar', texts(v.svg).includes('40') && r.chart.kind === 'matrix');
}
{
  // a sweep: several runs of a thing, drawn as the mean with the range behind it.
  // The replicates differ by a known amount, so the band's edges are known.
  const rep = d => history.map(row => ({ ...row,
    readout:{ ...row.readout, pops:{ L4e:{ decode:row.readout.pops.L4e.decode + d } } } }));
  const A = meanBand([rep(-0.1), rep(0), rep(0.1)].map(h => chartSeries(h, 'L4e', 'decode')));
  ok('the mean of the runs is the middle one and the band is their range',
    A.points.length === 4 && Math.abs(A.points[0][1] - 0.3) < 1e-9
    && Math.abs(A.band[0][1] - 0.2) < 1e-9 && Math.abs(A.band[0][2] - 0.4) < 1e-9 && A.runs === 3,
    JSON.stringify([A.points[0], A.band[0]]));
  const one = meanBand([chartSeries(history, 'L4e', 'decode')]);
  ok('one run has a mean and no band, since a band of one run is a lie',
    one.points.length === 4 && one.band.length === 0);
  const groups = [{ label:'as is', histories:[rep(-0.1), rep(0), rep(0.1)] },
    { label:'triplet x2', histories:[rep(0.2), rep(0.3)] }];
  const spec = figureSpec({ plot:0, pop:'L4e', field:'decode' });
  const ctx = new SvgContext(spec.wPt, spec.hPt);
  const rs = await paintFigure(ctx, spec, { plot:0, pop:'L4e', field:'decode' }, { ...src, groups }, []);
  const sv = ctx.toString();
  ok('a sweep draws a line and a faint band per group of runs',
    (sv.match(/fill-opacity="0\.18"/g) || []).length === 2 && rs.chart.kind === 'bands'
    && /5 runs/.test(rs.note), rs.note);
  ok('the groups are named with the number of runs behind each',
    texts(sv).includes('as is (3 runs)') && texts(sv).includes('triplet x2 (2 runs)'), texts(sv).join('|'));
  const table = bandSeriesCsv(rs.chart.series, rs.chart.xName, rs.chart.yName).split(String.fromCharCode(10));
  ok('the exported table carries the mean, the low and the high of every group',
    table[0] === 'sim_min,as is (3 runs) decode mean,as is (3 runs) low,as is (3 runs) high,'
      + 'triplet x2 (2 runs) decode mean,triplet x2 (2 runs) low,triplet x2 (2 runs) high', table[0]);
  ok('and a row per checkpoint', table.length === 6 && table[1].startsWith('5,0.3,0.2,0.4,'), table[1]);
}
{
  const { svg, r } = await svgOf({ plot:0, pop:'L4e', field:'decode' });
  ok('a decode trace has its chance line and its measure on the axis', texts(svg).includes('chance') && texts(svg).includes('decode, L4e') && texts(svg).includes('simulated time (min)'));
  ok('the run trace saves under its own column names', r.chart.xName === 'sim_min' && r.chart.yName === 'decode' && chartSeries(history, 'L4e', 'hz').length === 4);
  const none = await svgOf({ plot:0, pop:'nobody', field:'decode' });
  ok('a plot with nothing to draw says so and hands back no data', none.r.chart === null && texts(none.svg).includes('nothing recorded for this population'));
}
{
  // the panel: the compact style, the same painter
  const ctx = new SvgContext(320, 130);
  const r = await paintChart(ctx, 320, 130, { plot:7, pop:'L4e', window:10 }, src, screenStyle({}));
  const t = texts(ctx.toString());
  ok('the panel keeps its compact layout: no rotated title, the x label in the tick row', !/transform=/.test(ctx.toString()) && t.includes('sim s') && !t.includes('rate (Hz)') && r.chart.kind === 'series');
  const stale = await paintChart(new SvgContext(320, 130), 320, 130, { plot:1 }, { ...src, stale:() => true }, screenStyle({}));
  ok('a paint overtaken while it waited for the weights draws nothing', stale === null);
  for(const theme of [1, 2]){ const d = await svgOf({ plot:7, pop:'L4e', theme });
    ok('theme ' + theme + (theme === 1 ? ' has no ground' : ' has a black ground'), theme === 1 ? !/fill="#fff"/.test(d.svg) && !/fill="#000"\/>/.test(d.svg.split('\n')[5] || '') : /fill="#000"/.test(d.svg)); }
}
report('chartpaint');

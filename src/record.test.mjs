import { Recorder, ROWS, BIN_MS } from './record.js';
import { csv, seriesCsv, rasterCsv, spikeTrainsAscii, histogramCsv, matrixCsv, figureName } from './figure.js';
import { ok, report } from '../tools/harness.mjs';

// four cells probed out of six; frames of 2 ms
const R = new Recorder();
R.setPopulations([{ label:'a', idx:Int32Array.from([1, 2, 4, 5]) }, { label:'empty', idx:Int32Array.from([]) }]);
ok('a population with no cells is not recorded', R.labels().join() === 'a');
const fired = new Uint8Array(6);
fired[1] = 1; fired[4] = 1; R.onState(fired, 2);            // t 2: two of four fire
fired.fill(0); R.onState(fired, 2);                          // t 4: none
fired[1] = 1; R.onState(fired, 2);                           // t 6: cell 1 again, interval 4 ms
ok('the clock advances by the frame', R.t === 6);
const s = R.series('a', 10000, 1);
ok('a rate per frame: 2 of 4 cells in 2 ms is 250 Hz', s.length === 3 && Math.abs(s[0][1] - 250) < 1e-6 && s[1][1] === 0 && Math.abs(s[2][1] - 125) < 1e-6, JSON.stringify(s));
ok('the series carries simulated time', s.map(p => p[0]).join() === '2,4,6');
const sm = R.series('a', 10000, 4);
ok('smoothing averages over the last two frames', Math.abs(sm[1][1] - 125) < 1e-6 && Math.abs(sm[2][1] - 62.5) < 1e-6, JSON.stringify(sm));
const ra = R.raster('a', 10000);
ok('the raster has a row per probed cell', ra.rows === 4 && ra.cells.join() === '1,2,4,5');
ok('every spike is an event (t, row)', JSON.stringify(ra.events) === '[[2,0],[2,2],[6,0]]', JSON.stringify(ra.events));
ok('a window drops old events', R.raster('a', 3).events.length === 1);
const iv = R.intervals('a');
ok('one interval of 4 ms lands in its bin', iv.total === 1 && iv.counts[Math.floor(4/BIN_MS)] === 1);
ok('mean rate over the window', Math.abs(R.meanRate('a', 10000) - 125) < 1e-6);

// the same populations again (a tune) keep their recording; a changed one starts over
R.setPopulations([{ label:'a', idx:Int32Array.from([1, 2, 4, 5]) }]);
ok('a tune keeps the recording of an unchanged population', R.series('a', 10000, 1).length === 3 && R.raster('a', 10000).events.length === 3);
R.setPopulations([{ label:'a', idx:Int32Array.from([1, 2, 4]) }]);
ok('a population over different cells starts over', R.series('a', 10000, 1).length === 0);

// a big population is sampled to ROWS rows with a fixed stride
const big = new Recorder();
const idx = Int32Array.from({ length:5000 }, (_, i) => i);
big.setPopulations([{ label:'b', idx }]);
const f2 = new Uint8Array(5000).fill(1);
big.onState(f2, 2);
const rb = big.raster('b', 100);
ok('a 5000 cell population rasters ' + ROWS + ' rows', rb.rows === ROWS && rb.stride === 5 && rb.events.length === ROWS, rb.rows + ' ' + rb.events.length);
ok('the rate counts every cell, not only the sampled rows', Math.abs(big.series('b', 100, 1)[0][1] - 500) < 1e-6);

// the text builders
ok('csv quotes only what needs it', csv(['a', 'b'], [[1, 'x,y'], ['"q"', '']]) === 'a,b\n1,"x,y"\n"""q""",\n');
ok('series csv', seriesCsv([[2, 250], [4, 0]], 'a') === 't_ms,a_hz\n2,250\n4,0\n');
ok('raster csv names the cell', rasterCsv([[2, 0], [6, 1]], [7, 9]) === 't_ms,row,cell\n2,0,7\n6,1,9\n');
ok('spike trains: one line per cell, silent cells empty', spikeTrainsAscii([[2, 0], [6, 0], [4, 2]], 3) === '2 6\n\n4\n');
ok('histogram csv carries the edges', histogramCsv([3, 1], [0, 4, 8]) === 'bin_from,bin_to,count\n0,4,3\n4,8,1\n');
ok('matrix csv', matrixCsv([1, 2, 3, 4], 2, 2, 'item') === 'item,c0,c1\n0,1,2\n1,3,4\n');
ok('a figure name is safe', figureName(['guided tour', 'spike raster (live)', 'L2/3e'], 'png') === 'guided-tour_spike-raster-live_l2-3e.png');
report('record and figure');

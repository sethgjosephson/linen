import { Recording, BIN_MS, weightSummary, changedKeys, describe } from './recording.js';
import { ok, report } from '../tools/harness.mjs';

// two populations of three cells, frames of 2 ms
const rec = new Recording('test scene', [{ label:'a', idx:[0, 1, 2] }, { label:'b', idx:[3, 4, 5] }]);
const fired = new Uint8Array(6);
for(let f = 0; f < 50; f++){          // 100 ms
  fired.fill(0);
  if(f % 5 === 0) fired[0] = 1;       // cell 0: one spike per 10 ms, 100 Hz
  if(f % 10 === 0) fired[3] = 2;      // cell 3: two spikes per 20 ms
  if(f === 12) rec.edit('stimulus', 'current', 1, 2);
  if(f === 20) rec.mark('pause');
  rec.onState(fired, 2);
}
ok('the clock advanced by the frames', rec.t === 100 && rec.frames === 50);
ok('five bins of ' + BIN_MS + ' ms per population', rec.pops.get('a').hz.length === 5 && rec.pops.get('b').hz.length === 5);
// population a: 2 spikes per 20 ms over 3 cells = 2/(3*0.02) = 33.3 Hz
ok('the rate is spikes per cell per second', Math.abs(rec.pops.get('a').hz[0] - 33.333) < 0.01 && Math.abs(rec.pops.get('b').hz[0] - 33.333) < 0.01);
ok('a series is [t, hz] at the bin ends', rec.series('a')[0][0] === 20 && rec.series('a')[4][0] === 100);
ok('the edit and the mark carry their simulated time', rec.events.length === 2 && rec.events[0].t === 24 && rec.events[0].key === 'current' && rec.events[1].t === 40 && rec.events[1].what === 'pause');
ok('an edit reads as a sentence', describe(rec.events[0]) === 'stimulus current: 1 to 2');
const csv = rec.csv().trim().split('\n');
ok('the csv has a header, five bins and two event rows', csv.length === 8 && csv[0] === 't_ms,a_hz,b_hz,event' && csv[2].startsWith('24,,,"stimulus current: 1 to 2"') && csv[4].startsWith('40,,,"pause"'));
ok('event rows sit before the bin they fell in', csv[1].startsWith('20,') && csv[3].startsWith('40,33.333'));
const j = JSON.parse(JSON.stringify(rec));
ok('the file carries the scene, the bins and the events', j.format === 'linen-recording-1' && j.scene === 'test scene' && j.populations[0].hz.length === 5 && j.events.length === 2 && j.ms === 100);
// a rewiring in the middle: a new population joins, the existing one keeps its record
rec.mark('rewire'); rec.setPopulations([{ label:'a', idx:[0, 1, 2] }, { label:'c', idx:[1, 2] }]);
fired.fill(0); fired[1] = 1;
for(let f = 0; f < 10; f++) rec.onState(fired, 2);
ok('a population that joins later starts at the bin it joined', rec.pops.get('c').from === 100 && rec.pops.get('c').hz.length === 1 && rec.pops.get('a').hz.length === 6);
ok('the later population fires at its own rate', Math.abs(rec.pops.get('c').hz[0] - 250) < 0.01);   // 10 spikes over 2 cells in 20 ms
const ws = weightSummary(new Float32Array([0.5, 1, -1.5, 0, -0.5]));
ok('a weight summary splits by sign', ws.count === 5 && ws.e.n === 2 && Math.abs(ws.e.mean - 0.75) < 1e-6 && ws.i.n === 2 && ws.i.max === 1.5);
ok('changed keys compares by value', changedKeys({ a:1, b:[1, 2], c:'x' }, { a:1, b:[1, 3], c:'x', d:0 }).join() === 'b,d');
report('recording');

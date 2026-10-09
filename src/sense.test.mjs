// An input's sense is a control, not a consequence of its mapping.
//
// Reading the modality off map.sheet (a two dimensional encoder rendering the glyph, everything else the spectrum) would tie three separate things together, leaving no way to hear through a distributed map, to see through a one dimensional one, or to point both senses at the same population shape.
// These checks drive the runtime the way the page does and pin each of the four crossings.

import { IORuntime } from './io.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond) pass++;
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};

// Set 8 is the spot and tone pair set.
// Both renderers compute it in closed form, so this file runs without a canvas, the same as the engine host.
const SIG = { set:8, order:1, onMs:1000, offMs:2000, probeEvery:4, seed:1,
              calCycles:0 };
const PERIOD = 3000;

// source -2 is the curriculum
function run(map, { steps = 60, stepMs = 250 } = {}){
  const io = new IORuntime(e => { throw new Error(String(e)); });
  const posted = []; posted.times = [];
  let now = 0;
  io.attach({ postMessage: m => {
    if(m.cmd === 'inputFrame'){ posted.push(Float32Array.from(m.gains)); posted.times.push(now); }
  } }, [{ id:1, source:-2, code:0, jitter:0, transient:0, period:PERIOD,
          signal:SIG, ...map }], 1);
  io.last = -1e9;
  for(let t = 0; t < steps; t++){ now = t*stepMs; io.frame(now, false); }
  return posted;
}

const SHEET = { cols:8, rows:8, sheet:true };
const BANDS = { cols:16, rows:1, sheet:false };
const energy = f => f.reduce((a, v) => a + Math.abs(v), 0);
const live = fs => fs.filter(f => energy(f) > 0);
const same = (a, b) => a.length === b.length &&
  a.every((f, i) => f.length === b[i].length && f.every((v, k) => v === b[i][k]));

// --- every crossing of sense and mapping produces drive ---------------------
for(const [mname, m] of [['sheet', SHEET], ['bands', BANDS]]){
  for(const sense of ['sight', 'sound']){
    const fs = run({ ...m, sense });
    const n = m.cols * (m.sheet ? m.rows : 1);
    ok(`${sense} through a ${mname} mapping drives at all`, live(fs).length > 2,
       live(fs).length + ' non-empty frames of ' + fs.length);
    ok(`${sense} through a ${mname} mapping fills all ${n} channels`,
       fs.every(f => f.length === n),
       'saw ' + [...new Set(fs.map(f => f.length))].join(', '));
  }
}

// --- the sense decides the content, the mapping only decides the shape ------
{
  const sight = run({ ...BANDS, sense:'sight' });
  const sound = run({ ...BANDS, sense:'sound' });
  ok('the same one dimensional mapping gives different frames per sense',
     !same(sight, sound));
  const sheetSound = run({ ...SHEET, sense:'sound' });
  ok('sound through a sheet mapping is not silent', live(sheetSound).length > 2,
     live(sheetSound).length + ' non-empty frames');
}

// --- each sense carries item identity through either mapping ---------------
// Whatever the mapping, different items must produce different drive, or the downstream decode has nothing to separate.
{
  const items = fs => new Set(fs.map(f => f.join(','))).size;
  for(const [mname, m] of [['sheet', SHEET], ['bands', BANDS]])
    for(const sense of ['sight', 'sound']){
      const n = items(live(run({ ...m, sense }, { steps:400 })));
      ok(`${sense} through a ${mname} mapping distinguishes items`, n >= 4,
         'only ' + n + ' distinct frames over 400 steps');
    }
}

// --- probe muting follows the sense -----------------------------------------
// cond 1 shows the glyph with no sound, cond 2 the reverse.
// With probeEvery 4 a sight encoder must go silent on some presentation that a sound encoder drives, and the other way round.
{
  const sight = run({ ...BANDS, sense:'sight' }, { steps:200 });
  const sound = run({ ...BANDS, sense:'sound' }, { steps:200 });
  const q = fs => fs.map(f => energy(f) > 0 ? 1 : 0).join('');
  ok('the two senses are silenced on different presentations', q(sight) !== q(sound));
}

// --- a following stream's offset shifts its clock ---------------------------
// The signal carries the offset a following curriculum set; a stream 300 ms behind reads the lesson clock 300 ms earlier, frame for frame.
{
  const base = run({ ...BANDS, sense:'sound' }, { steps:200, stepMs:100 });
  const late = run({ ...BANDS, sense:'sound', signal:{ ...SIG, offset:300 } }, { steps:200, stepMs:100 });
  const key = f => f.join(',');
  const at = (fs, t) => { const i = fs.times.indexOf(t); return i < 0 ? null : key(fs[i]); };
  let checked = 0, agree = 0;
  for(let i = 0; i < base.length; i++){
    const t = base.times[i];
    if(t + 300 >= 200*100) continue;
    const l = at(late, t + 300);
    if(l === null) continue;
    checked++; if(l === key(base[i])) agree++;
  }
  ok('a stream 300 ms behind posts the same frame 300 ms later', checked > 10 && agree === checked,
     agree + ' of ' + checked);
}

// --- the same item lights the same channels every time ------------------------
// The channel-to-neuron map is fixed at compute time (inputCompute, seeded); this checks the other half: with no microsaccade, item k renders the same channel pattern on every presentation, one full cycle apart.
{
  const sig = { ...SIG, order:0, probeEvery:0 };      // sequential, no probes
  const fs = run({ ...SHEET, sense:'sight', signal:sig }, { steps:1000, stepMs:100 });
  const N = 12, cycle = N*PERIOD;                     // twelve spots in the set
  const key = f => f.join(',');
  const at = t => { const i = fs.times.indexOf(t); return i < 0 ? null : key(fs[i]); };
  let checked = 0, agree = 0;
  for(let i = 0; i < fs.length; i++){
    const t = fs.times[i];
    if(energy(fs[i]) === 0 || t + cycle >= 1000*100) continue;
    const later = at(t + cycle);
    if(later === null) continue;
    checked++; if(later === key(fs[i])) agree++;
  }
  ok('an item lights the same channels one cycle later', checked > 3 && agree === checked,
     agree + ' of ' + checked);
}

console.log(fail ? fail + ' failed, ' + pass + ' passed'
                 : 'all passed (' + pass + ' checks)');
process.exit(fail ? 1 : 0);

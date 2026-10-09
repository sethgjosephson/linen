// The stimulus runtime is deterministic, checked through the same path the page and the trainer use.
//
// An unseeded microsaccade jitter in encodeGains would give a training run of any scene with jitter on a different stimulus every time.
// The project's determinism rule is written for the compute functions and the engines; the stimulus runtime feeds the engines, and the reproducibility check in the battery runs a scene with jitter off, so it would not see it.
// This pins the property from the outside: a fake worker records every input frame, and two runs with the same seed must post identical frames while two seeds must not.

import { IORuntime } from './io.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond) pass++;
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};

// One map: an 8 by 8 sheet fed by the bar sweep (source 0), raw coding, with jitter of one channel.
// Only the fields encodeGains and frame() read.
const MAP = { id: 3, source: 0, cols: 8, rows: 8, sheet: true, code: 0,
              jitter: 1, transient: 0, period: 1500 };

function run(seed, { jitter = 1, stepMs = 40, frames = 200 } = {}){
  const io = new IORuntime(() => {});
  const posted = [], refreshes = [];
  io.attach({ postMessage: m => {
    if(m.cmd === 'inputFrame') posted.push(Float32Array.from(m.gains));
  } }, [{ ...MAP, jitter }], seed);
  io.last = -1e9;
  let lastJn = 0;
  for(let t = 0; t < frames; t++){
    io.frame(t*stepMs, false);
    const st = io.enc.get(MAP.id);
    if(st && (st.jn | 0) !== lastJn){ lastJn = st.jn | 0; refreshes.push([st.jx, st.jy]); }
  }
  return { posted, refreshes };
}
const sameFrames = (a, b) => a.length === b.length &&
  a.every((f, i) => f.length === b[i].length && f.every((v, k) => v === b[i][k]));
const sameOffsets = (a, b) => a.length === b.length &&
  a.every((r, i) => r[0] === b[i][0] && r[1] === b[i][1]);

// --- the property itself -------------------------------------------------
{
  const a = run(7), b = run(7);
  ok('the runtime posts frames at all', a.posted.length > 10, String(a.posted.length));
  ok('jitter refreshed more than once over the run', a.refreshes.length > 3,
     String(a.refreshes.length));
  ok('same seed, same frames, byte for byte', sameFrames(a.posted, b.posted));
  ok('same seed, same offset sequence', sameOffsets(a.refreshes, b.refreshes));

  const c = run(8);
  ok('a different seed gives a different offset sequence',
     !sameOffsets(a.refreshes, c.refreshes),
     'seeds 7 and 8 refreshed identically ' + a.refreshes.length + ' times');
  ok('and therefore different frames', !sameFrames(a.posted, c.posted));
}

// --- the seed reaches only the jitter --------------------------------------
{
  const a = run(7, { jitter: 0 }), b = run(8, { jitter: 0 });
  ok('with jitter off the seed changes nothing', sameFrames(a.posted, b.posted));
  ok('and nothing refreshed', a.refreshes.length === 0, String(a.refreshes.length));
}

// --- the offsets do not depend on how many steps a tick takes -------------
// The refresh *times* are quantized to the tick, so frames differ between step sizes; the offsets are keyed on a per-map refresh count and must not.
{
  const coarse = run(7, { stepMs: 40, frames: 200 });
  const fine = run(7, { stepMs: 20, frames: 400 });
  ok('the offset sequence is the same at 40 ms and 20 ms ticks',
     sameOffsets(coarse.refreshes, fine.refreshes),
     coarse.refreshes.length + ' vs ' + fine.refreshes.length + ' refreshes');
}

// --- a shifted frame is a shift, not an invention -------------------------
// Every value in a jittered frame must exist in the unjittered one at some neighboring position; jitter moves the raster, it never makes new values.
{
  const j = run(7), p = run(7, { jitter: 0 });
  let bad = 0;
  for(let i = 0; i < j.posted.length; i++){
    const src = new Set(Array.from(p.posted[i]).map(v => v.toFixed(6)));
    for(const v of j.posted[i]) if(!src.has(v.toFixed(6))){ bad++; break; }
  }
  ok('a jittered frame only contains values from the unjittered raster', bad === 0,
     bad + ' frame(s) carried a value that does not exist in the source');
}

// the shuffled tile order for sound: a bijection over the items with no item in its own tile, the same for a seed, different across seeds
{
  const { shuffledTile } = await import('./io.js');
  for(const [n, seed] of [[5, 1], [5, 23], [10, 7], [2, 3]]){
    const tiles = Array.from({ length:n }, (_, k) => shuffledTile(k, n, seed));
    ok(`shuffled tiles are a permutation (n ${n}, seed ${seed})`, new Set(tiles).size === n && tiles.every(t => t >= 0 && t < n), tiles.join(','));
    ok(`no item keeps its own tile (n ${n}, seed ${seed})`, tiles.every((t, k) => t !== k), tiles.join(','));
  }
  const a = Array.from({ length:5 }, (_, k) => shuffledTile(k, 5, 1)), b = Array.from({ length:5 }, (_, k) => shuffledTile(k, 5, 2));
  ok('the order depends on the seed', a.join(',') !== b.join(','), a.join(',') + ' vs ' + b.join(','));
}

console.log(fail ? fail + ' failed, ' + pass + ' passed'
                 : 'all passed (' + pass + ' checks)');
process.exit(fail ? 1 : 0);

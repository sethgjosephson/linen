// Do these spike statistics agree with the implementation the field uses?
//
// Every one of these measures has small choices inside it: population or sample variance, whether adjacent interval pairs are counted once or twice, what happens when a train is too short for the quantity to exist.
// A measure that differs from every published one by sqrt(N/(N-1)) is worse than no measure, because it looks right.
// So rather than reason about the choices, this generates spike trains with known properties, computes the statistics both here and in Elephant, and compares them value by value.
//
// Elephant runs in .venv-analysis and is not a dependency of the app: with it absent this falls back to checking the properties that hold by construction (a Poisson train has CV near 1, a regular one near 0) and says clearly that the cross-check did not run, rather than passing quietly.
//
//   node src/spikestats.test.mjs

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import * as S from './spikestats.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const PY = path.join(ROOT, '.venv-analysis', 'Scripts', 'python.exe');
const PY_NIX = path.join(ROOT, '.venv-analysis', 'bin', 'python');

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond) pass++;
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};

// A seeded generator, because a test that draws fresh randomness disagrees with itself and the disagreement gets read as a tolerance problem.
function rng(seed){
  let x = seed >>> 0;
  return () => {
    x ^= x << 13; x >>>= 0;
    x ^= x >> 17;
    x ^= x << 5; x >>>= 0;
    return x/4294967296;
  };
}
// Poisson: exponential intervals, so CV is 1 and LV is 1 by construction.
function poisson(rateHz, tStopMs, seed){
  const r = rng(seed), out = [];
  let t = 0;
  const mean = 1000/rateHz;
  for(;;){
    t += -Math.log(1 - r())*mean;
    if(t >= tStopMs) break;
    out.push(t);
  }
  return out;
}
// Perfectly regular: CV and LV both 0.
function regular(rateHz, tStopMs){
  const out = [], step = 1000/rateHz;
  for(let t = step; t < tStopMs; t += step) out.push(t);
  return out;
}
// Regular with jitter, which is the interesting middle case and the one where a variance convention actually shows up in the answer.
function jittered(rateHz, tStopMs, seed, jitterMs){
  const r = rng(seed), out = [], step = 1000/rateHz;
  for(let t = step; t < tStopMs; t += step)
    out.push(Math.max(0.1, t + (r() - 0.5)*2*jitterMs));
  return out.sort((a, b) => a - b);
}

const T_STOP = 20000, BIN = 50;
const trains = [
  poisson(20, T_STOP, 12345),
  poisson(20, T_STOP, 999),
  poisson(5, T_STOP, 4242),
  regular(20, T_STOP),
  jittered(20, T_STOP, 777, 12),
  jittered(50, T_STOP, 31337, 4),
  [100, 200],                       // too short for CV2 and LV
  [],                               // empty
];

// --- properties that hold by construction ---------------------------------
{
  ok('a Poisson train has CV near 1', Math.abs(S.cv(trains[0]) - 1) < 0.15,
    'cv = ' + S.cv(trains[0]).toFixed(3));
  ok('a Poisson train has LV near 1', Math.abs(S.lv(trains[0]) - 1) < 0.2,
    'lv = ' + S.lv(trains[0]).toFixed(3));
  ok('a regular train has CV 0', S.cv(trains[3]) < 1e-9,
    'cv = ' + S.cv(trains[3]));
  ok('a regular train has LV 0', S.lv(trains[3]) < 1e-9);
  ok('a jittered train sits between', S.cv(trains[4]) > 0 && S.cv(trains[4]) < 0.9,
    'cv = ' + S.cv(trains[4]).toFixed(3));
  ok('too few spikes gives NaN rather than a number',
    Number.isNaN(S.cv2(trains[6])) && Number.isNaN(S.lv(trains[6])));
  ok('an empty train gives NaN', Number.isNaN(S.cv(trains[7])));
  ok('mean rate is spikes over seconds',
    Math.abs(S.meanRate(trains[3], 0, T_STOP) - 20) < 0.1,
    S.meanRate(trains[3], 0, T_STOP).toFixed(3));
}

// --- the Fano factor is not the synchrony measure --------------------------
{
  const same = [poisson(20, T_STOP, 1), poisson(20, T_STOP, 2), poisson(20, T_STOP, 3)];
  const ff = S.fanoFactor(same);
  ok('the Fano factor of repeated Poisson trains is small',
    ff >= 0 && ff < 8, 'ff = ' + ff.toFixed(3));
  const sync = S.populationSynchrony(same, BIN, 0, T_STOP);
  ok('population synchrony is a different number', sync !== ff,
    'ff ' + ff.toFixed(3) + ' vs synchrony ' + sync.toFixed(3));
}

// --- the PSTH ---------------------------------------------------------------
{
  const h = S.timeHistogram(trains.slice(0, 3), BIN, 0, T_STOP);
  ok('the histogram has one bin per bin width', h.length === T_STOP/BIN,
    h.length + ' bins');
  let total = 0;
  for(const v of h) total += v;
  const spikes = trains.slice(0, 3).reduce((a, t) => a + t.length, 0);
  ok('every spike lands in exactly one bin', total === spikes,
    total + ' binned of ' + spikes);
  const rate = S.psth(trains.slice(0, 3), BIN, 0, T_STOP);
  const meanRate = rate.reduce((a, b) => a + b, 0)/rate.length;
  const expect = spikes/3/(T_STOP/1000);
  ok('the psth mean is the mean firing rate', Math.abs(meanRate - expect) < 0.5,
    meanRate.toFixed(2) + ' vs ' + expect.toFixed(2));
}

// --- against Elephant -------------------------------------------------------
const py = fs.existsSync(PY) ? PY : (fs.existsSync(PY_NIX) ? PY_NIX : null);
if(!py){
  console.log('');
  console.log('  Elephant not found at .venv-analysis, so the cross-check DID NOT RUN.');
  console.log('  Only the properties above were verified. To install:');
  console.log('    python -m venv .venv-analysis');
  console.log('    .venv-analysis/Scripts/python.exe -m pip install elephant neo');
} else {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'spikestats-'));
  const inFile = path.join(dir, 'in.json'), outFile = path.join(dir, 'out.json');
  fs.writeFileSync(inFile, JSON.stringify({ trains, tStopMs:T_STOP, binMs:BIN }));
  execFileSync(py, [path.join(ROOT, 'tools', 'elephant_check.py'), inFile, outFile],
    { stdio:'pipe' });
  const ref = JSON.parse(fs.readFileSync(outFile, 'utf8'));
  fs.rmSync(dir, { recursive:true, force:true });

  console.log('  checked against elephant ' + ref.versions.elephant +
    ', neo ' + ref.versions.neo);

  // null on their side and NaN on ours both mean undefined
  const agree = (a, b, tol, label) => {
    const aU = a === null || a === undefined, bU = Number.isNaN(b) || b === undefined;
    if(aU || bU) return ok(label + ' (both undefined)', aU && bU,
      'elephant ' + a + ' vs ours ' + b);
    ok(label, Math.abs(a - b) <= tol,
      'elephant ' + a.toFixed(6) + ' vs ours ' + b.toFixed(6));
  };
  trains.forEach((t, i) => {
    const r = ref.perTrain[i];
    agree(r.cv, S.cv(t), 1e-9, `train ${i}: cv`);
    agree(r.cv2, S.cv2(t), 1e-9, `train ${i}: cv2`);
    agree(r.lv, S.lv(t), 1e-9, `train ${i}: lv`);
    // Elephant's mean_firing_rate returns 1/ms for a train in ms; ours is Hz
    if(r.meanRate !== null)
      agree(r.meanRate*1000, S.meanRate(t, 0, T_STOP), 1e-9, `train ${i}: mean rate`);
  });
  agree(ref.fanoFactor, S.fanoFactor(trains), 1e-9, 'fano factor over all trains');
  if(ref.timeHistogram){
    const ours = S.timeHistogram(trains, BIN, 0, T_STOP);
    let maxDiff = 0;
    for(let i = 0; i < ours.length; i++)
      maxDiff = Math.max(maxDiff, Math.abs(ours[i] - ref.timeHistogram[i]));
    ok('time histogram matches bin for bin', ours.length === ref.timeHistogram.length && maxDiff === 0,
      ours.length + ' vs ' + ref.timeHistogram.length + ' bins, max diff ' + maxDiff);
  } else ok('time histogram computed', false, ref.timeHistogramError || 'absent');
}

console.log(fail ? (fail + ' failed, ' + pass + ' passed') : 'all passed (' + pass + ' checks)');
if(fail) process.exit(1);

// The 2003 model is a row of the 2007 form: proven on the arithmetic the engines run, before any engine loses its 2003 branch.
//
// Two claims, kept apart because they fail differently.
// The per-step update is the same function: from one state, one step of each form lands on the same voltage to rounding and the same spike decision, checked at every step by re-synchronizing the 2007 side to the 2003 side after each.
// And a free run is equivalent up to summation order: the two forms compute the same polynomial with the terms in a different order, which differs at 1e-13 per step, and a bursting preset near threshold amplifies that into a shifted spike within a few seconds.
// That is the standard the three engines are held to already (ENGINE.md 4), so the free run is judged on spike counts and the first spikes, not on every millisecond.

import { NEURON_TYPES } from './nodes.js';
import { classicRoots, classicRow, step2003, step2007 } from './form.js';
import { ok, report } from '../tools/harness.mjs';

const PRESETS = NEURON_TYPES.slice(0, 7);   // RS, IB, CH, FS, LTS, TC, RZ
const vmin = -90;

// --- the roots -------------------------------------------------------------
for(const t of PRESETS){
  const { vr, vt } = classicRoots(t.b);
  // RZ (b 0.26) is the degenerate case: rest and threshold coincide at
  // -62.5, the resonator sitting at its saddle-node, so the row is
  // k (v - vr)^2 and vt equals vr rather than exceeding it.
  ok(t.key + ': the rest is real and not above the threshold', Number.isFinite(vr) && vr <= vt + 1e-9,
     'vr ' + vr + ' vt ' + vt);
  ok(t.key + ': the rest solves 0.04 v^2 + (5 - b) v + 140 = 0',
     Math.abs(0.04*vr*vr + (5 - t.b)*vr + 140) < 1e-9);
  let worst = 0;
  for(const v of [-90, -70, -60, -50, -30, 0, 30]){
    const lhs = 0.04*(v - vr)*(v - vt), rhs = 0.04*v*v + 5*v + 140 - t.b*vr;
    worst = Math.max(worst, Math.abs(lhs - rhs));
  }
  ok(t.key + ': 0.04 (v - vr)(v - vt) equals the 2003 polynomial minus b vr', worst < 1e-9, String(worst));
}
{
  const { vr, vt } = classicRoots(0.2);
  ok('b = 0.2 gives the textbook rest of -70 and a threshold of -55',
     Math.abs(vr + 70) < 1e-9 && Math.abs(vt + 55) < 1e-9, vr + ' ' + vt);
  const rz = classicRoots(0.26);
  ok('RZ is the double root at -62.5', Math.abs(rz.vr + 62.5) < 1e-9 && Math.abs(rz.vt - rz.vr) < 1e-9);
}

// --- the row ---------------------------------------------------------------
{
  const r = classicRow(PRESETS[0]);
  ok('a classic row has C 1, k 0.04, peak 30 and carries a, b, c, d', r.C === 1 && r.k === 0.04 &&
     r.vpeak === 30 && r.a === 0.02 && r.b === 0.2 && r.c === -65 && r.d === 8);
  ok('and is marked classic', r.classic === true);
}

// --- claim 1: the same step function ----------------------------------------
// From the 2003 state at every step, one step of each; the 2007 side is then set back to the 2003 side so nothing accumulates.
{
  let worstV = 0, decisionMismatch = 0, steps = 0, spikes = 0;
  for(const t of PRESETS){
    const r = classicRow(t);
    for(const I of [0, 3, 6, 10, 20, 40]){
      let v3 = t.c, u3 = t.b*t.c;
      for(let s = 0; s < 3000; s++){
        const v7in = v3, u7in = u3 - t.b*r.vr;          // U = u - b vr
        const [n3v, n3u, sp3] = step2003(v3, u3, I, t, vmin);
        const [n7v, n7u, sp7] = step2007(v7in, u7in, I, r, vmin);
        if(sp3 !== sp7) decisionMismatch++;
        if(!sp3) worstV = Math.max(worstV, Math.abs(n3v - n7v), Math.abs((n3u - t.b*r.vr) - n7u));
        v3 = n3v; u3 = n3u; steps++; if(sp3) spikes++;
      }
    }
  }
  ok('one step of each form from the same state gives the same spike decision, every step',
     decisionMismatch === 0, decisionMismatch + ' of ' + steps);
  ok('and the same voltage and recovery to rounding', worstV < 1e-9, 'worst ' + worstV);
  ok('the comparison exercised spikes', spikes > 200, String(spikes));
}

// --- claim 2: a free run is equivalent up to summation order ----------------
// Same start, no re-synchronization, three seconds: the first five spikes land on the same millisecond, and the counts agree within ten percent.
// Ten, not one: the spike is a cutoff at 30 mV crossed inside a half step, so a 1e-13 difference in the polynomial decides which millisecond, and once the phase has slipped the adaptation variable follows it.
// That is the sensitivity ENGINE.md 4 holds the three engines to, at the same tolerances.
{
  let bad = [];
  for(const t of PRESETS){
    const r = classicRow(t);
    for(const I of [3, 6, 10, 20, 40]){
      let v3 = t.c, u3 = t.b*t.c, v7 = t.c, u7 = t.b*t.c - t.b*r.vr;
      const s3 = [], s7 = [];
      for(let s = 0; s < 3000; s++){
        let a, b;
        [v3, u3, a] = step2003(v3, u3, I, t, vmin); if(a) s3.push(s);
        [v7, u7, b] = step2007(v7, u7, I, r, vmin); if(b) s7.push(s);
      }
      // the first-spike comparison needs spikes to compare: at IB's rheobase (3 units) one side crosses once in three seconds and the other never does, which the count tolerance below allows
      const first = (s3.length < 5 || s7.length < 5) || s3.slice(0, 5).join() === s7.slice(0, 5).join();
      const count = Math.abs(s3.length - s7.length) <= Math.max(1, 0.10*s3.length);
      if(!(first && count)) bad.push(t.key + '@' + I + ': ' + s3.length + ' vs ' + s7.length + ' spikes, first ' + s3.slice(0,5) + ' vs ' + s7.slice(0,5));
    }
  }
  ok('free runs agree on the first five spikes and on counts within ten percent', bad.length === 0, bad.join(' | '));
}

// --- a control: a measured 2007 row is not the 2003 row ----------------------
{
  const t = PRESETS[0], r = t.f7;
  let v3 = t.c, u3 = t.b*t.c, v7 = r.vr, u7 = 0, n3 = 0, n7 = 0;
  for(let s = 0; s < 2000; s++){
    let a, b;
    [v3, u3, a] = step2003(v3, u3, 10, t, vmin); if(a) n3++;
    [v7, u7, b] = step2007(v7, u7, 10, r, vmin); if(b) n7++;
  }
  ok('the measured RS row under 10 pA is a different cell from classic RS under 10 units',
     n3 !== n7, n3 + ' vs ' + n7 + ' spikes');
}

report('form');

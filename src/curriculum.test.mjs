// The calibration sweep must see every item in all three conditions before anything learns, and the ordinary probe schedule must be checked for the aliasing that made it miss half the alphabet.
import { scheduleAt, parsePhases, phaseAt } from './curriculum.js';

let fail = 0;
const ok = (n, c, x) => { console.log((c?'ok   ':'FAIL ')+n+(x?'  '+x:'')); if(!c) fail++; };

const PERIOD = 3000;                       // 1000 on, 2000 off
const base = { set:0, order:1, onMs:1000, offMs:2000, probeEvery:4, seed:1 };
const at = (sig, k) => scheduleAt(sig, k*PERIOD + 10);

{
  const sig = { ...base, calCycles:2 };
  const N = at(sig, 0).count;
  const seen = new Map();                  // item -> Set(cond)
  const calK = 2*3*N;
  for(let k=0;k<calK;k++){
    const s = at(sig, k);
    if(!s.calibrating) continue;
    if(!seen.has(s.idx)) seen.set(s.idx, new Set());
    seen.get(s.idx).add(s.cond);
  }
  const complete = [...seen.values()].filter(c => c.has(0)&&c.has(1)&&c.has(2)).length;
  ok('sweep sees every item in all three conditions', complete === N,
    `${complete}/${N} items complete over ${calK} presentations ` +
    `(${(calK*PERIOD/60000).toFixed(1)} sim-min)`);
}

{
  const sig = { ...base, calCycles:1 };
  const N = at(sig, 0).count;
  ok('sweep ends after calCycles*3*N', !!at(sig, 3*N - 1).calibrating &&
    !at(sig, 3*N).calibrating);
}

{
  const a = at({ ...base }, 7), b = at({ ...base, calCycles:0 }, 7);
  ok('calCycles 0 leaves the schedule alone',
    a.idx === b.idx && a.cond === b.cond && !a.calibrating);
}

// the aliasing: probeEvery 4 over 26 items only ever probes odd-numbered letters, so half of them are never seen alone
{
  const sig = { ...base, order:0, calCycles:0 };   // sequential, k % N
  const N = at(sig, 0).count;
  const probed = new Set();
  for(let k=0;k<N*40;k++){ const s = at(sig, k); if(s.cond) probed.add(s.idx); }
  ok('probeEvery 4 misses half the alphabet (the bug)', probed.size < N,
    `${probed.size}/${N} items ever probed alone`);
  const sig5 = { ...sig, probeEvery:5 };
  const probed5 = new Set();
  for(let k=0;k<N*40;k++){ const s = at(sig5, k); if(s.cond) probed5.add(s.idx); }
  ok('probeEvery 5 reaches every item', probed5.size === N,
    `${probed5.size}/${N} items ever probed alone`);
}


// ---- repeat sweeps ----
{
  const sig = { ...base, order:0, calCycles:3, reSweepEvery:2400 };
  const N = at(sig, 0).count, calK = 3*3*N;
  const win = [];                          // [start,end) of every sweep window
  let cur = null;
  for(let k=0;k<7000;k++){
    const s = at(sig, k);
    if(s.sweep && !cur) cur = { start:k, re:!!s.resweep };
    if(!s.sweep && cur){ cur.end = k; win.push(cur); cur = null; }
  }
  ok('opening sweep is first and is not a repeat',
    win[0] && win[0].start === 0 && win[0].end === calK && !win[0].re,
    `first window ${win[0] && win[0].start}..${win[0] && win[0].end}`);
  ok('repeat sweeps recur on the configured period',
    win.length >= 3 && win[1].re && win[1].start === calK + 2400 &&
    win[2].start === calK + 4800,
    `starts ${win.map(w => w.start).join(',')}`);
  ok('every window is the same length', win.every(w => w.end - w.start === calK),
    `lengths ${win.map(w => w.end - w.start).join(',')}`);
  // each repeat must cover the whole alphabet in all three conditions too, or the drift comparison is against a different set of items
  const w = win[1];
  const seen = new Map();
  for(let k=w.start;k<w.end;k++){
    const s = at(sig, k);
    if(!seen.has(s.idx)) seen.set(s.idx, new Set());
    seen.get(s.idx).add(s.cond);
  }
  const complete = [...seen.values()].filter(c => c.size === 3).length;
  ok('a repeat sweep covers every item in all three conditions', complete === N,
    `${complete}/${N}`);
  ok('repeat period is about two sim-hours',
    Math.abs(2400*3000/3600000 - 2) < 0.01, `${(2400*3000/3600000).toFixed(2)} h`);
}


// the block schedule: training trials follow the blocks; probe trials do not, since they are the measurement and have to ask the same question in every block for the blocks before the pairing to be its control
{
  const list = parsePhases('25 A; 25 B; 130 AB');
  ok('a schedule parses to its blocks',
    JSON.stringify(list.map(b => [b.min, b.cond])) === '[[25,1],[25,2],[130,0]]',
    list.map(b => b.min + ':' + b.cond).join(' '));
  const M = 60000;
  ok('time falls in the block it is in',
    phaseAt(list, 0).cond === 1 && phaseAt(list, 24.9*M).cond === 1 &&
    phaseAt(list, 25.1*M).cond === 2 && phaseAt(list, 51*M).cond === 0);
  ok('the last block runs to the end of the run', phaseAt(list, 900*M).cond === 0);
  ok('no schedule is the ordinary stream',
    phaseAt(parsePhases(''), 0) === null && phaseAt(parsePhases('  '), 5*M) === null);
}
{
  const sig = { ...base, calCycles:0, phases:parsePhases('5 A; 5 B; 50 AB') };
  const conds = ph => {
    const k0 = Math.round(ph*60000/PERIOD), tr = new Set(), pr = new Set();
    for(let k=k0;k<k0+40;k++){
      const st = at(sig, k);
      (st.probe ? pr : tr).add(st.cond);
    }
    return { tr:[...tr].sort().join(''), pr:[...pr].sort().join('') };
  };
  const b1 = conds(1), b2 = conds(6), b3 = conds(11);   // 40 presentations is 2 min at this period
  ok('block A trains on channel A alone', b1.tr === '1', b1.tr);
  ok('block B trains on channel B alone', b2.tr === '2', b2.tr);
  ok('the pairing block trains on both', b3.tr === '0', b3.tr);
  // all three conditions, because every readout the trainer computes needs bimodal trials in the window and a block without them measures nothing
  ok('the same probes run in every block',
    b1.pr === '012' && b2.pr === '012' && b3.pr === '012',
    b1.pr + ' ' + b2.pr + ' ' + b3.pr);
  ok('an unscheduled run probes as it always did',
    (() => { const u = { ...base, calCycles:0 }, s = new Set();
      for(let k=0;k<40;k++){ const st = at(u, k); if(st.probe) s.add(st.cond); }
      return [...s].sort().join('') === '12'; })());
}

console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);

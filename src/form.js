// The Izhikevich 2003 model as a row of the 2007 form.
//
// The 2007 form (MODEL.md 1) integrates
//
//     C dv/dt = k (v - vr)(v - vt) - u + I,     du/dt = a (b (v - vr) - u)
//
// and the 2003 form
//
//     dv/dt = 0.04 v^2 + 5 v + 140 - u + I,     du/dt = a (b v - u).
//
// Put C = 1, k = 0.04 and measure the recovery variable from the rest, U = u - b vr.
// Then du/dt = a (b v - u) = a (b (v - vr) - U) is the 2007 recovery equation exactly, and the voltage equation asks for
//
//     0.04 (v - vr)(v - vt) = 0.04 v^2 + 5 v + 140 - b vr
//
// for every v, which fixes vr + vt = -125 and 0.04 vr vt = 140 - b vr.
// Eliminating vt, vr is a root of 0.04 v^2 + (5 - b) v + 140, the same quadratic whose roots are the 2003 model's rest and threshold on the recovery nullcline; the rest is the lower root, and vt = -125 - vr is then the 2007 instantaneous threshold (with u held), which is not the upper root.
// Every 2003 preset has a real lower root.
//
// So a 2003 neuron is a 2007 neuron with these numbers, term for term, and every scene written for the 2003 form runs unchanged on the 2007 engines once its types carry these rows.
// The one thing that is not physical about them is the capacitance: C = 1 pF is a unit choice, and under a classic row a picoamp means what the 2003 model's current meant.

export const CLASSIC_C = 1, CLASSIC_K = 0.04, CLASSIC_VPEAK = 30;

// The rest and instantaneous threshold of the 2003 model with recovery gain b, as the 2007 form names them.
export function classicRoots(b){
  const B = 5 - b;
  const D = B*B - 4*CLASSIC_K*140;
  if(!(D >= 0)) throw new Error('no 2007 row for b = ' + b + ': the 2003 rest is complex');
  const vr = (-B - Math.sqrt(D)) / (2*CLASSIC_K);
  return { vr, vt: -125 - vr };
}

// The 2007 row equal to a type's 2003 row (a, b, c, d).
// Marked classic so the initial state can start it where the 2003 form started.
export function classicRow(t){
  const { vr, vt } = classicRoots(+t.b);
  return { C:CLASSIC_C, k:CLASSIC_K, vr, vt, vpeak:CLASSIC_VPEAK,
    a:+t.a, b:+t.b, c:+t.c, d:+t.d, classic:true };
}

// One step of each form, written as the engines write it (src/simworker.js step, two half steps for v with the floor after each, one step for u), so the equivalence can be checked against the arithmetic that runs and not against the equations alone.
// Returns [v, u, spiked].
export function step2003(v, u, I, t, vmin){
  v += 0.5*(0.04*v*v + 5*v + 140 - u + I); if(v < vmin) v = vmin;
  v += 0.5*(0.04*v*v + 5*v + 140 - u + I); if(v < vmin) v = vmin;
  u += t.a*(t.b*v - u);
  if(v >= 30){ return [t.c, u + t.d, true]; }
  return [v, u, false];
}
export function step2007(v, u, I, r, vmin){
  v += 0.5*(r.k*(v - r.vr)*(v - r.vt) - u + I)/r.C; if(v < vmin) v = vmin;
  v += 0.5*(r.k*(v - r.vr)*(v - r.vt) - u + I)/r.C; if(v < vmin) v = vmin;
  u += r.a*(r.b*(v - r.vr) - u);
  if(v >= r.vpeak){ return [r.c, u + r.d, true]; }
  return [v, u, false];
}

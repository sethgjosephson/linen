// Per-pathway plasticity rules: resolution, precedence, and the invariant that a graph with no plasticity node resolves the gate byte as the boolean every saved scene carries.
//
// The other thing under test is that a scoped rule does not touch connection density.
// Overriding plasticity for a pathway by prepending a pair-table row with its multipliers at one would, because the winning row carries probability and weight along with the rule, silently give the association territory about a third of its biological connection count.

import { buildPairLUT, ruleIds, parseRuleRef, NODE_DEFS } from './nodes.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond){ pass++; }
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};

// Two populations, one excitatory and one inhibitory, one neuron each, with stable tags. buildPairLUT groups by source scatter id.
const pts = {
  count: 2,
  src: [10, 20],
  ntype: [0, 3],                     // 0 = RS (excitatory), 3 = FS (inhibitory)
  tags: { 10: 'a.e', 20: 'a.i' },
};
const cell = (T, pre, post) => T.plMul[pre * T.G + post];
const probOf = (T, pre, post) => T.probMul[pre * T.G + post];

// --- the compatibility invariant ------------------------------------------
{
  const table = 'a.e a.i 2.0 1 1\na.i a.e 3.0 1 0\n* * 0 1 0';
  const before = buildPairLUT(pts, table);              // no rules argument
  const after = buildPairLUT(pts, table, []);           // empty rule list
  ok('no rules: e->i stays plastic', cell(before, 0, 1) === 1, 'got ' + cell(before, 0, 1));
  ok('no rules: i->e stays frozen', cell(before, 1, 0) === 0, 'got ' + cell(before, 1, 0));
  ok('no rules: empty list is the same', cell(after, 0, 1) === cell(before, 0, 1));
  ok('no rules: probability untouched', probOf(before, 0, 1) === 2);
}

{
  const T = buildPairLUT(pts, 'a.e a.i 2.0 1', []);
  ok('short row still means plastic', cell(T, 0, 1) === 1, 'got ' + cell(T, 0, 1));
}

// --- named rules from the table -------------------------------------------
{
  const rules = [{ name: 'ff', from: '', to: '' }, { name: 'rec', from: '', to: '' }];
  const T = buildPairLUT(pts, 'a.e a.i 2.0 1 ff\na.i a.e 3.0 1 rec', rules);
  ok('table names rule ff', cell(T, 0, 1) === 2, 'got ' + cell(T, 0, 1));
  ok('table names rule rec', cell(T, 1, 0) === 3, 'got ' + cell(T, 1, 0));
}

// --- a scoped node assigns without touching density -----------------------
{
  const rules = [{ name: 'ff', from: 'a.e', to: 'a.i' }];
  const T = buildPairLUT(pts, 'a.e a.i 2.0 5 frozen\n* * 0 1 0', rules);
  ok('scoped rule wins over the frozen row', cell(T, 0, 1) === 2, 'got ' + cell(T, 0, 1));
  ok('scoped rule leaves probability alone', probOf(T, 0, 1) === 2, 'got ' + probOf(T, 0, 1));
  ok('scoped rule leaves weight alone', T.wMul[0 * T.G + 1] === 5, 'got ' + T.wMul[1]);
  ok('scoped rule does not leak to other pairs', cell(T, 1, 0) === 0, 'got ' + cell(T, 1, 0));
}

// --- specificity: a scoped rule loses to nothing, but only on its own pair --
{
  const rules = [{ name: 'ff', from: 'E', to: 'I' }];       // sign-level scope
  const T = buildPairLUT(pts, 'a.e a.i 2.0 1 frozen', rules);
  // tag/tag scores 4, sign/sign scores 2, so the explicit table row wins
  ok('a less specific scope loses to a tag row', cell(T, 0, 1) === 0, 'got ' + cell(T, 0, 1));
}
{
  const rules = [{ name: 'ff', from: 'a.e', to: 'a.i' }];
  const T = buildPairLUT(pts, 'E I 2.0 1 frozen', rules);
  ok('a more specific scope beats a sign row', cell(T, 0, 1) === 2, 'got ' + cell(T, 0, 1));
}

// --- unscoped rules are declared but assign nothing ------------------------
{
  const rules = [{ name: 'ff', from: '', to: '' }];
  const T = buildPairLUT(pts, 'a.e a.i 2.0 1 frozen', rules);
  ok('an unscoped rule assigns nothing by itself', cell(T, 0, 1) === 0, 'got ' + cell(T, 0, 1));
}

// --- reference parsing -----------------------------------------------------
{
  const ids = ruleIds([{ name: 'ff' }, { name: 'rec' }]);
  ok('ids start at 2', ids.get('ff') === 2 && ids.get('rec') === 3);
  ok('absent means default', parseRuleRef(undefined, ids) === 1);
  ok('0 means frozen', parseRuleRef('0', ids) === 0);
  ok('frozen means frozen', parseRuleRef('frozen', ids) === 0);
  ok('1 means default', parseRuleRef('1', ids) === 1);
  ok('default means default', parseRuleRef('default', ids) === 1);
  ok('a name resolves', parseRuleRef('ff', ids) === 2);

  let threw = '';
  try { parseRuleRef('typo', ids); } catch(e){ threw = e.message; }
  ok('an unknown name throws', /no plasticity rule named "typo"/.test(threw), threw);
  ok('and lists what exists', /ff, rec/.test(threw), threw);

  threw = '';
  try { ruleIds([{ name: 'x' }, { name: 'x' }]); } catch(e){ threw = e.message; }
  ok('duplicate names throw', /both declare a rule named "x"/.test(threw), threw);
}

// A byte holds 255 rules and zero is frozen.
{
  const many = Array.from({ length: 254 }, (_, i) => ({ name: 'r' + i }));
  const ids = ruleIds(many);
  ok('254 rules still fit the byte', ids.get('r253') === 255, 'got ' + ids.get('r253'));
}

// --- a bypassed rule node must not take the scene down -------------------
// Bypassing a plasticity node removes the rule from the stream while every row naming it stays put; throwing there would fail the whole wiring, so rows fall back to the default rule and the name is reported.
{
  const T = buildPairLUT(pts, ['a.e a.i 2.0 1 feedforward', '* * 0 1 0'].join('\n'), []);
  ok('a missing rule does not throw', true);
  ok('the pair falls back to the default rule', cell(T, 0, 1) === 1, 'got ' + cell(T, 0, 1));
  ok('and the name is reported', T.ruleWarn.includes('feedforward'),
    JSON.stringify(T.ruleWarn));
  ok('a resolvable table reports nothing', buildPairLUT(pts, 'a.e a.i 2.0 1 0', []).ruleWarn.length === 0);
}

// The preset selector writes the terms of a published rule and the name the pair table refers to, and the name is the part that can do damage: a table row already says 'weavei weave inh', so replacing a name someone chose would silently unbind the rule from the pathway it governs.
{
  const def = NODE_DEFS.plasticity.params.find(p => p.k === 'preset');
  const nameAfter = (name, i) => { const q = { name }; def.apply(q, i); return q.name; };
  const iTrip = def.options.findIndex(o => /triplet/.test(o));
  const iVog = def.options.findIndex(o => /inhibitory/.test(o));
  ok('a preset names an unnamed rule', nameAfter('', iVog) === 'inh');
  ok('a preset replaces the default name', nameAfter('rule', iTrip) === 'triplet');
  ok('a preset replaces another preset name', nameAfter('inh', iTrip) === 'triplet');
  ok('a preset keeps a name the scene chose', nameAfter('cortexRule', iTrip) === 'cortexRule',
    nameAfter('cortexRule', iTrip));
  const q = { name:'x', aP:0.008, aM:0.001, trip:0.005 };
  def.apply(q, iVog);
  ok('inhibitory homeostasis turns the timing terms off',
    q.aP === 0 && q.aM === 0 && q.trip === 0 && q.iEta === 0.02);
  const c = { name:'keep', aP:0.008 };
  def.apply(c, 0);
  ok('custom writes nothing', c.name === 'keep' && c.aP === 0.008);
}

console.log(fail ? (fail + ' failed, ' + pass + ' passed') : 'all passed (' + pass + ' checks)');
if(fail) process.exit(1);

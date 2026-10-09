// The per-synapse rule byte: this file pins the meaning of the table the engines read, which is the part the engines and the computation have to agree on.
// The engines obeying it is checked in the battery, where they run as Workers against a real network and one pathway switched to a rule with a different amplitude must diverge on every engine.

import { buildRuleTable, RULE_FIELDS, RULE_STRIDE, engineConfig } from './nodes.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail) => {
  if(cond) pass++;
  else { fail++; console.log('FAIL: ' + name + (detail ? '  ' + detail : '')); }
};
const near = (a, b, tol) => Math.abs(a - b) <= tol;

// --- the table itself ------------------------------------------------------
{
  const base = { aP:0.008, aM:0.001, wmax:30, wdep:1, trip:0, het:0, tin:0,
    iEta:0.002, cons:0, consW:0, consP:10 };
  const { ruleTable, ruleCount } = buildRuleTable(base, [
    { name:'fast', aP:0.02 },
    { name:'frozenish', aP:0, aM:0 },
  ]);
  ok('one row per rule plus the two reserved', ruleCount === 4, 'got ' + ruleCount);
  ok('table is stride by count', ruleTable.length === 4*RULE_STRIDE);
  const f = RULE_FIELDS.indexOf('aP'), g = RULE_FIELDS.indexOf('aM');
  ok('row 1 is the checkpoint', near(ruleTable[1*RULE_STRIDE + f], 0.008, 1e-9));
  ok('row 2 overrides what it names', near(ruleTable[2*RULE_STRIDE + f], 0.02, 1e-9));
  ok('row 2 inherits what it does not', near(ruleTable[2*RULE_STRIDE + g], 0.001, 1e-9));
  ok('row 3 can set a field to zero', ruleTable[3*RULE_STRIDE + f] === 0,
    'zero must not be read as absent');
}

// --- field order is the engine contract ------------------------------------
// The workers are self-contained and restate this order rather than importing it, so a change here without a change there would be silent and would look like a plasticity result.
{
  const expect = ['aP', 'aM', 'wmax', 'wdep', 'trip', 'het', 'tin', 'iEta',
    'cons', 'consW', 'consP'];
  ok('field order matches what the engines assume',
    RULE_FIELDS.join(',') === expect.join(','),
    RULE_FIELDS.join(','));
  ok('stride matches the field count', RULE_STRIDE === expect.length);
}

// Committed synapses are a tune field, not a rule-table column: the decision is per synapse but the switch is per network, and it reads the consolidated weight of whichever rule the synapse belongs to.
// If it stops reaching the engines the term silently does nothing, which is the failure this file is about.
{
  const net = { syn:1, plast:1, aP:0.008, aM:0.001, wmax:4, cons:0.5, consW:1.2,
    consP:10, commit:1, rules:[] };
  const cfg = engineConfig(net);
  ok('the commit switch reaches the engines', cfg.commit === 1, 'got ' + cfg.commit);
  ok('and is off unless asked for', engineConfig({ ...net, commit:0 }).commit === 0);
  ok('the consolidated weight is still a per-rule column',
    RULE_FIELDS.indexOf('consW') === 9, 'consW at ' + RULE_FIELDS.indexOf('consW'));
}

console.log(fail ? (fail + ' failed, ' + pass + ' passed') : 'all passed (' + pass + ' checks)');
if(fail) process.exit(1);

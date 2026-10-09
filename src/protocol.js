// The number of the engine contract (ENGINE.md), carried on every init and every hello reply.
//
// A page and an engine are loaded separately: the page's modules and the worker scripts can come from different builds when a deploy lands mid-session or a worker is served from a cache.
// With nothing to tell them apart, a stale engine handed a new type table runs every membrane to NaN in silence.
// So the sender writes the number of the contract it was written against into the init, every engine compares it with its own, and a mismatch is refused with both numbers in the message.
// The worker URLs carry the same number, so a page never loads an engine script from another contract by name.
//
// Bump it when any message or frame changes shape, in the same commit as the three engines, ENGINE.md and the battery.
//
//   1   one membrane (Izhikevich 2007); no form word
//   2   stpOrder on init and tune; the CUDA tune frame's byte
//       100 carries stpNorm + 2 stpOrder
//   3   hello before init (the engine's contract number, name, terms
//       and optional queries); the CUDA child answers HELLO; custom on
//       init and tune carries the engine settings node's entries

export const PROTOCOL = 3;

// The refusal, shared by every engine: null when the init matches, else the message to fail with.
// An init without the field is from before the number existed, and is refused for the same reason.
export function checkProtocol(m, who){
  const got = m && m.protocol;
  if(got !== PROTOCOL)
    return who + ': the init names engine contract ' + (got === undefined || got === null ? 'none' : String(got)) +
      ' and this engine is contract ' + PROTOCOL + '; the page and the engine are from different builds. Reload the page.';
  // The rule table's shape (ENGINE.md).
  // Every engine reads rows of RULE_ROW fields, and the WebGPU and CUDA engines hold RULE_ROWS rows.
  // Without this check a field added on the page shifts every later field in all three engines without an error, and a table longer than the GPU engines hold is read past its end there.
  if(m.ruleTable && m.ruleCount >= 2){
    if(m.ruleStride !== undefined && m.ruleStride !== RULE_ROW)
      return who + ': the rule table has rows of ' + m.ruleStride + ' fields and this engine reads ' +
        RULE_ROW + '; the page and the engine are from different builds. Reload the page.';
    if(m.ruleCount > RULE_ROWS)
      return who + ': the rule table has ' + m.ruleCount + ' rows and the engines hold ' + RULE_ROWS +
        ' (frozen, the checkpoint and ' + (RULE_ROWS - 2) + ' named rules).';
  }
  return null;
}

export const RULE_ROW = 11, RULE_ROWS = 16;

// ---- the hello exchange (ENGINE.md section 2) ----
// Before init the sender posts { cmd:'hello' }, and an engine answers with its contract number, its name, the terms it implements and the optional queries it answers.
// The sender refuses an engine of another contract and refuses to run a network that uses a term the engine does not name.

// Whether any rule row turns on the field at offset k of the rule table (the scalar when no table is sent).
function anyRule(m, k, scalar){
  if(m.ruleTable && m.ruleCount >= 2){
    for(let r = 1; r < m.ruleCount; r++) if(m.ruleTable[r*RULE_ROW + k] > 0) return true;
    return false;
  }
  return m[scalar] > 0;
}
const plastic = m => (m.plast|0) === 1;
function modes(m){
  const s = new Set();
  for(const p of m.protocols || []) s.add(p.mode|0);
  return s;
}
// A graded row in use: the type table and the cells are on an init only, so a tune never reports one.
function graded(m){
  if(!m.types || !m.ntype) return false;
  const rows = new Set();
  for(let i = 0; i < m.ntype.length; i++) rows.add(m.ntype[i]);
  for(const r of rows) if(m.types[r] && m.types[r].graded) return true;
  return false;
}

// The terms beyond the baseline, each named by the tune field (and value) that switches it on: [name, what it is, whether a message uses it].
// The baseline every engine runs is not listed: the init's network (cell count, type table, bias, CSR synapses with delays of 1 to 16 ms, seed, v0 and u0), kick delivery, tune of bias, tick and the state reply.
export const TERMS = [
  ['syn:1', 'exponential synapses', m => (m.syn|0) === 1],
  ['syn:2', 'conductance synapses', m => (m.syn|0) === 2],
  ['psc:1', 'the charge convention of exponential synapses (psc 1)', m => (m.syn|0) === 1 && (m.psc|0) === 1],
  ['psc:2', 'the exact convention of exponential synapses (psc 2)', m => (m.syn|0) === 1 && (m.psc|0) === 2],
  ['refrac', 'the absolute refractory period', m => m.refrac > 0],
  ['vmin', 'the membrane floor', m => m.vmin !== undefined && +m.vmin < 0],
  ['protocols:0', 'constant stimulus', m => modes(m).has(0)],
  ['protocols:1', 'pulse train stimulus', m => modes(m).has(1)],
  ['protocols:2', 'ramp stimulus', m => modes(m).has(2)],
  ['protocols:3', 'noise stimulus', m => modes(m).has(3)],
  ['inputs', 'input maps (inputFrame)', m => Array.isArray(m.inputs) && m.inputs.length > 0],
  ['graded', 'graded (non-spiking) rows', graded],
  ['plast', 'plasticity (pair STDP and the inhibitory rule)', plastic],
  ['rhoMode', 'measured rate set points', m => plastic(m) && (m.rhoMode|0) === 1],
  ['scale', 'synaptic scaling', m => plastic(m) && (m.scale|0) === 1],
  ['trip', 'the triplet term', m => plastic(m) && anyRule(m, 4, 'trip')],
  ['het', 'heterosynaptic regression', m => plastic(m) && anyRule(m, 5, 'het')],
  ['tin', 'transmitter-induced potentiation', m => plastic(m) && anyRule(m, 6, 'tin')],
  ['cons', 'consolidation', m => plastic(m) && anyRule(m, 8, 'cons')],
  ['commit', 'committed synapses', m => plastic(m) && (m.commit|0) === 1],
  ['ruleTable', 'named plasticity rules', m => plastic(m) && m.ruleCount > 2],
  ['stp', 'short-term plasticity', m => m.stp > 0],
  ['stpNorm', 'scaled short-term release (stpNorm 1)', m => m.stp > 0 && (m.stpNorm|0) === 1],
  ['stpOrder', 'facilitation before release (stpOrder 1)', m => m.stp > 0 && (m.stpOrder|0) === 1],
  ['chanTau', 'receptor channels', m => (Array.isArray(m.chanTau) || ArrayBuffer.isView(m.chanTau)) && m.chanTau.length > 0],
  ['pool', 'feedback inhibition', m => !!(m.pool && m.pool.length && m.poolK && m.poolK.length > 1)],
];
export const TERM_NAMES = TERMS.map(t => t[0]);
// 'custom' is named by an engine that reads the custom block; an engine that does not name it ignores the block, which the sender says.
export const KNOWN_TERMS = [...TERM_NAMES, 'custom'];
export const termLabel = name => (TERMS.find(t => t[0] === name) || [name, name])[1];
// The optional queries, by the message that asks: a watched cell's trace, every membrane potential, first-spike latency, connection lookup, weight readback.
// 'host' is the run controller of host/host.mjs (ENGINE.md section 7), which the host adds to its engine's reply.
export const OPTIONAL = ['watch', 'sendV', 'sendT', 'query', 'getWeights'];
export const KNOWN_OPTIONAL = [...OPTIONAL, 'host'];

export function helloReply(engine, terms, optional){
  return { cmd:'hello', protocol:PROTOCOL, engine, terms:[...terms], optional:[...optional] };
}
// null when the reply is a hello of this contract, else the message to fail with
export function checkHello(r){
  if(!r || r.cmd !== 'hello') return 'the engine did not answer hello';
  const who = typeof r.engine === 'string' && r.engine ? 'the ' + r.engine + ' engine' : 'the engine';
  if(r.protocol !== PROTOCOL)
    return who + ' answers engine contract ' + (r.protocol === undefined ? 'none' : String(r.protocol)) +
      ' and this page is contract ' + PROTOCOL + '; they are from different builds';
  if(typeof r.engine !== 'string' || !r.engine) return 'the engine\'s hello names no engine';
  if(!Array.isArray(r.terms) || r.terms.some(t => typeof t !== 'string'))
    return who + ': the hello reply\'s terms are not a list of names';
  if(!Array.isArray(r.optional) || r.optional.some(t => typeof t !== 'string'))
    return who + ': the hello reply\'s optional queries are not a list of names';
  return null;
}
// The terms a message uses that the engine's hello does not name, as [name, label] pairs.
export function missingTerms(m, hello){
  const have = new Set(hello && hello.terms || []);
  return TERMS.filter(([name, , used]) => used(m) && !have.has(name)).map(([name, label]) => [name, label]);
}
// How many keys of the custom block the engine ignores: all of them when its hello does not name 'custom', else none.
export function ignoredCustom(m, hello){
  const n = m && m.custom && typeof m.custom === 'object' ? Object.keys(m.custom).length : 0;
  return hello && (hello.terms || []).includes('custom') ? 0 : n;
}
export const customNote = (n, hello) => 'engine settings: ' + n + ' key' + (n === 1 ? '' : 's') + ' ignored by the ' + hello.engine + ' engine';
export const answers = (hello, q) => !!(hello && (hello.optional || []).includes(q));

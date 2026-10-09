// The engine settings node: its table, the custom block it puts on init and tune, and the note that an engine ignores it.
import { NODE_DEFS, computeNode, setResolution, engineConfig, parseEngineSettings, wireConnect, assembleConnect } from './nodes.js';
import { Doc, initMessage, tuneMessage } from './document.js';
import { helloReply, ignoredCustom, customNote, TERM_NAMES, OPTIONAL } from './protocol.js';
import { encodeMsg, decodeMsg } from './frame.js';
import { ok, report, threw, inProcessWorker } from '../tools/harness.mjs';
inProcessWorker(wireConnect, assembleConnect);
setResolution(1);

// the table
const t = parseEngineSettings('lif_tau_ms 12\nsolver  rk4   # a comment\n\n# a whole line\nlabel two words\nneg -1.5e-3');
ok('a number travels as a number', t.lif_tau_ms === 12 && t.neg === -1.5e-3, JSON.stringify(t));
ok('anything else travels as the text after the key', t.solver === 'rk4' && t.label === 'two words', JSON.stringify(t));
ok('comments and blank lines are skipped', Object.keys(t).length === 4);
ok('a key set twice is refused', /set twice/.test(threw(() => parseEngineSettings('a 1\na 2')) || ''));
ok('a key without a value is refused', /a key and a value/.test(threw(() => parseEngineSettings('lonely')) || ''));
ok('a key outside the alphabet is refused', /letters, digits/.test(threw(() => parseEngineSettings('9lives 1')) || '') &&
  /letters, digits/.test(threw(() => parseEngineSettings('__proto__ 1')) || ''));
ok('an empty table is an empty block', JSON.stringify(parseEngineSettings('')) === '{}');

// on a computed network
function graph(tables){
  const nodes = [];
  const add = (type, params) => {
    const def = NODE_DEFS[type], p = {};
    def.params.forEach(q => p[q.k] = structuredClone(q.def));
    Object.assign(p, params);
    const n = { id:nodes.length + 1, type, x:0, y:0, params:p, name:type + nodes.length, inputs:new Array(def.inputs).fill(null) };
    nodes.push(n); return n;
  };
  const wire = (to, from) => { to.inputs[0] = { id:from.id }; };
  const box = add('box', { center:[0, 0, 0], size:[200, 200, 200] });
  const e = add('scatter', { fill:0, count:60, type:0, tag:'e', seed:1 }); wire(e, box);
  const cn = add('connect', { radius:300, sigma:3000, prob:0.2, seed:1 }); wire(cn, e);
  let last = cn;
  const es = tables.map(table => { const n = add('enginesettings', { table }); wire(n, last); last = n; return n; });
  const cp = add('checkpoint', {}); wire(cp, last);
  return { cp, es, cn, byId:id => nodes.find(n => n.id === id) };
}
const g1 = graph(['lif_tau_ms 12\nsolver rk4']);
const net = await computeNode(g1.cp, g1.byId);
ok('the checkpoint carries the block', JSON.stringify(net.custom) === '{"lif_tau_ms":12,"solver":"rk4"}', JSON.stringify(net.custom));
ok('engineConfig puts it on init and tune as custom', JSON.stringify(initMessage(net).custom) === JSON.stringify(net.custom) &&
  JSON.stringify(tuneMessage(net).custom) === JSON.stringify(net.custom));
ok('a network with no engine settings node sends an empty block', JSON.stringify(engineConfig({ syn:0, protocols:[] }).custom) === '{}');
const g2 = graph(['a 1', 'b 2']);
ok('two nodes in one chain add their keys', JSON.stringify((await computeNode(g2.cp, g2.byId)).custom) === '{"a":1,"b":2}');
const g3 = graph(['a 1', 'a 2']);
ok('a key set by two nodes is refused', /already set/.test(await computeNode(g3.cp, g3.byId).then(() => '', e => e.message)));
{
  // on a points stream, before connect
  const def = NODE_DEFS.enginesettings;
  ok('placed on points it says where it goes', /between connect and the checkpoint/.test(threw(() => def.compute([{ kind:'points' }], { table:'' })) || ''));
}

// an edit tunes, never rewires
const doc = new Doc();
const view = g1.cp;
const first = doc.planCompute(net, view, false);
ok('a first computation starts an engine', first.action === 'start');
g1.es[0].params.table = 'lif_tau_ms 20';
for(const n of [g1.es[0], g1.cp]) n._cache = null;
const net2 = await computeNode(g1.cp, g1.byId);
const plan = doc.planCompute(net2, view, true);
ok('an edit of an entry is a tune of the running engine', plan.action === 'tune' && tuneMessage(net2).custom.lif_tau_ms === 20, plan.action);

// the block crosses the WebSocket framing as it is
const back = decodeMsg(encodeMsg(tuneMessage(net2)));
ok('the block round-trips through the frame codec', JSON.stringify(back.custom) === JSON.stringify(net2.custom));

// the stock engines ignore it, and the sender says how many keys
const ref = helloReply('reference', TERM_NAMES, OPTIONAL);
ok('an engine that does not name custom ignores every key', ignoredCustom(initMessage(net), ref) === 2);
ok('one that names it ignores none', ignoredCustom(initMessage(net), helloReply('mine', ['custom'], [])) === 0);
ok('the note names the count and the engine', customNote(3, ref) === 'engine settings: 3 keys ignored by the reference engine');
const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('./simworker.js');
globalThis.onmessage({ data:initMessage(net) });
globalThis.onmessage({ data:{ cmd:'tick', steps:5 } });
ok('the reference engine runs with the block on its init', !replies.some(d => d.cmd === 'error') && replies.some(d => d.cmd === 'state'),
  JSON.stringify(replies.find(d => d.cmd === 'error')));
report('enginesettings');

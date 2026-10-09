import { readFileSync } from 'node:fs';
import { PROTOCOL, checkProtocol, RULE_ROW, RULE_ROWS, TERMS, TERM_NAMES, KNOWN_TERMS, OPTIONAL, helloReply, checkHello, missingTerms, answers } from './protocol.js';
import { engineConfig } from './nodes.js';
import { encodeMsg, decodeMsg } from './frame.js';
import { CUDA_TERMS } from '../host/cudaengine.mjs';
import { ok, report } from '../tools/harness.mjs';

ok('the contract number is a positive integer', Number.isInteger(PROTOCOL) && PROTOCOL >= 1);
ok('a matching init passes', checkProtocol({ protocol:PROTOCOL }, 'x') === null);
const other = checkProtocol({ protocol:0 }, 'x') || '';
ok('another number is refused, naming both', /contract 0/.test(other) && new RegExp('contract ' + PROTOCOL).test(other), other);
ok('an init without the field is refused as none', /contract none/.test(checkProtocol({}, 'x') || ''));
ok('the message tells the user what to do', /Reload/.test(checkProtocol({ protocol:99 }, 'x') || ''));
ok('engineConfig carries the number', engineConfig({ syn:0, protocols:[] }).protocol === PROTOCOL);
// the rule table's shape, which the engines hard-code
const cfg = engineConfig({ syn:0, protocols:[], wmax:10 });
ok('the rule row the page sends is the width the engines read', cfg.ruleStride === RULE_ROW, cfg.ruleStride + ' vs ' + RULE_ROW);
ok('the table the page sends passes', checkProtocol(cfg, 'x') === null);
const wide = checkProtocol({ ...cfg, ruleStride:RULE_ROW + 1 }, 'x') || '';
ok('a table with another row width is refused, naming both', new RegExp('rows of ' + (RULE_ROW + 1)).test(wide) && new RegExp('reads ' + RULE_ROW).test(wide), wide);
const tall = checkProtocol({ ...cfg, ruleCount:RULE_ROWS + 1 }, 'x') || '';
ok('a table longer than the engines hold is refused', /rows and the engines hold 16/.test(tall), tall);
ok('an init with no table is not asked about one', checkProtocol({ protocol:PROTOCOL }, 'x') === null);

// ---- hello ----
// the CUDA child carries the number compiled in, and it has to be this one
const cu = readFileSync(new URL('../cuda/engine.cu', import.meta.url), 'utf8');
const cuNum = (/#define ENGINE_PROTOCOL (\d+)/.exec(cu) || [])[1];
ok('the CUDA child is built for this contract', +cuNum === PROTOCOL, 'engine.cu ENGINE_PROTOCOL ' + cuNum + ', protocol.js ' + PROTOCOL);
ok('the CUDA child answers HELLO', /cmd == 11/.test(cu) && /reply\(103/.test(cu));
// the WebGPU worker cannot load here (no device); its reply is read off the source
const gpu = readFileSync(new URL('./gpuworker.js', import.meta.url), 'utf8');
ok('the WebGPU engine answers hello with every term', /m\.cmd === 'hello'\) postMessage\(helloReply\('webgpu', TERM_NAMES/.test(gpu));

// the reference engine, in process
const replies = [];
globalThis.postMessage = d => replies.push(d);
globalThis.onmessage = null;
await import('./simworker.js');
globalThis.onmessage({ data:{ cmd:'hello' } });
const ref = replies.pop();
ok('the reference engine answers hello', ref && ref.cmd === 'hello' && ref.engine === 'reference', JSON.stringify(ref));
ok('and the reply passes the check', checkHello(ref) === null, checkHello(ref));
ok('the reference names every term', TERM_NAMES.every(t => ref.terms.includes(t)), ref.terms.join(' '));
ok('and answers every optional query', OPTIONAL.every(q => answers(ref, q)));
ok('the CUDA engine names every term but pulse and ramp stimuli', CUDA_TERMS.length === TERM_NAMES.length - 2 &&
  !CUDA_TERMS.includes('protocols:1') && !CUDA_TERMS.includes('protocols:2'));
ok('term names are unique', new Set(KNOWN_TERMS).size === KNOWN_TERMS.length);

// the check
ok('another contract is refused, naming both', /contract 2 and this page is contract/.test(checkHello({ ...ref, protocol:2 }) || ''));
ok('a reply without a name is refused', !!checkHello({ ...ref, engine:'' }));
ok('a reply whose terms are not names is refused', !!checkHello({ ...ref, terms:[1] }));
ok('a reply without optional queries is refused', !!checkHello({ ...ref, optional:null }));
ok('an error is not a hello', !!checkHello({ cmd:'error', message:'x' }));

// which terms a message uses
const base = engineConfig({ syn:0, protocols:[], refrac:2, vmin:-90 });
const none = helloReply('bare', [], []);
ok('the baseline asks for the floor and the refractory period only', JSON.stringify(missingTerms(base, none).map(m => m[0])) === '["refrac","vmin"]',
  JSON.stringify(missingTerms(base, none)));
ok('a bare kick network with neither needs no term', missingTerms(engineConfig({ syn:0, protocols:[], refrac:0, vmin:0 }), none).length === 0);
const uses = (extra, name) => missingTerms(engineConfig({ syn:0, protocols:[], refrac:0, vmin:0, ...extra }), none).some(m => m[0] === name);
ok('exp synapses are a term', uses({ syn:1 }, 'syn:1') && !uses({ syn:1 }, 'syn:2'));
ok('conductance is a term', uses({ syn:2 }, 'syn:2'));
ok('the charge convention is a term under exp only', uses({ syn:1, psc:1 }, 'psc:1') && !uses({ syn:0, psc:1 }, 'psc:1'));
ok('a pulse train is a term', uses({ protocols:[{ idx:new Uint32Array(1), amp:1, mode:1 }] }, 'protocols:1'));
ok('short-term plasticity is a term, and its conventions with it', uses({ stp:0.5, stpNorm:1 }, 'stp') && uses({ stp:0.5, stpNorm:1 }, 'stpNorm') && !uses({ stp:0, stpNorm:1 }, 'stpNorm'));
ok('plasticity terms count only with plasticity on', uses({ plast:1, trip:0.01 }, 'trip') && !uses({ plast:0, trip:0.01 }, 'trip'));
ok('a declared rule turns its term on through the table', uses({ plast:1, rules:[{ name:'r', het:0.01 }] }, 'het'));
ok('a graded row is a term on an init', missingTerms({ ntype:new Uint8Array([0, 1]), types:[{}, { graded:{ thr:-50, slope:10 } }] }, none).some(m => m[0] === 'graded'));
ok('the reference lacks nothing a message can use', missingTerms(engineConfig({ syn:2, plast:1, trip:1, het:1, tin:1, cons:1, commit:1, stp:1, stpNorm:1, stpOrder:1, scale:1, rhoMode:1,
  protocols:[0, 1, 2, 3].map(mode => ({ idx:new Uint32Array(1), amp:1, mode })) }), ref).length === 0);
ok('every term has a name, a label and a test', TERMS.every(t => typeof t[0] === 'string' && typeof t[1] === 'string' && typeof t[2] === 'function'));

// hello crosses the WebSocket framing unchanged, and the remote adapter names where it came from
const back = decodeMsg(encodeMsg(ref));
ok('a hello reply round-trips through the frame codec', JSON.stringify(back) === JSON.stringify(ref));
{
  let sock = null;
  globalThis.WebSocket = class { constructor(url){ this.url = url; sock = this; } send(){} close(){} };
  const { RemoteEngine } = await import('./remoteworker.js');
  const eng = new RemoteEngine('ws://example.test:9');
  const got = [];
  eng.onmessage = e => got.push(e.data);
  sock.onmessage({ data:encodeMsg(ref) });
  ok('the remote adapter passes the reply on with its address', got[0] && got[0].engine === 'reference' && got[0].url === 'ws://example.test:9', JSON.stringify(got[0]));
  sock.onerror();
  ok('an engine that cannot be reached is an engine error naming the address', got[1] && got[1].cmd === 'error' && /ws:\/\/example\.test:9/.test(got[1].message), got[1] && got[1].message);
}

report('protocol');

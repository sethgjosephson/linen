// The headless tools' settings parser (tools/overrides.mjs): a selector by tag, node name or probe label, type@k, JSON arrays, an empty value kept empty, and a refusal for a key the node lacks or a term that matches nothing.
import { splitTerms, applyOverrides, parseValue } from '../tools/overrides.mjs';
import { ok, report } from '../tools/harness.mjs';

const nodes = () => [
  { type:'stimulus', name:'pulse on L4 E', params:{ tag:'L4e', current:5, center:[0,0,0] } },
  { type:'stimulus', name:'drive', params:{ tag:'L23e', current:5, center:[0,0,0] } },
  { type:'probe', name:'p', params:{ tag:'x', label:'layer 4', mode:0 } },
];
const threw = fn => { try { fn(); return null; } catch(e){ return e.message; } };
{
  const ns = nodes();
  applyOverrides(ns, splitTerms('stimulus:pulse on L4 E:current=0'));
  ok('a node name selects', ns[0].params.current === 0 && ns[1].params.current === 5);
}
{
  const ns = nodes();
  applyOverrides(ns, splitTerms(['stimulus@1:*:center=[1,2,3]', 'probe:layer 4:mode=2']));
  ok('type@k and a JSON array', JSON.stringify(ns[1].params.center) === '[1,2,3]' && ns[0].params.center[0] === 0);
  ok('a probe label selects', ns[2].params.mode === 2);
}
ok('an empty value stays empty', parseValue('') === '' && parseValue('2.5') === 2.5 && parseValue('abc') === 'abc');
ok('a key the node lacks is refused', /no setting called nope/.test(threw(() => applyOverrides(nodes(), ['stimulus:*:nope=1'])) || ''));
ok('a term that matches nothing is refused', /no stimulus node matched/.test(threw(() => applyOverrides(nodes(), ['stimulus:nobody:current=1'])) || ''));
report('overrides');

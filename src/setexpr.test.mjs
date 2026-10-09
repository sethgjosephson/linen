// The set language, including the errors, because the error is what a person editing the field actually sees when they get it wrong.
import { parse, evaluate, compileSet, namesUsed } from './setexpr.js';

let fail = 0;
const ok = (n, c, x) => { console.log((c?'ok   ':'FAIL ')+n+(x?'  '+x:'')); if(!c) fail++; };

const N = 8;
const S = {
  both:  Uint8Array.from([1,1,1,1,0,0,0,0]),
  sight: Uint8Array.from([1,1,0,0,1,1,0,0]),
  sound: Uint8Array.from([1,0,1,0,1,0,1,0]),
};
const ev = src => Array.from(evaluate(parse(src), S, N));

ok('the conjunction, as the fixed code computed it',
  ev('both and not sight and not sound').join('') === '00010000',
  ev('both and not sight and not sound').join(''));
ok('or', ev('sight or sound').join('') === '11101110');
ok('not binds tighter than and', ev('not sight and both').join('') === '00110000');
ok('parentheses override', ev('not (sight and both)').join('') === '00111111');
ok('symbols are synonyms for the words',
  ev('both & !sight & !sound').join('') === ev('both and not sight and not sound').join(''));
ok('names are collected', [...namesUsed(parse('a and not b'))].sort().join(',') === 'a,b');
ok('labels with dots and slashes are one name',
  [...namesUsed(parse('x.L2/3 and not v.L4'))].sort().join(',') === 'v.L4,x.L2/3');

// the failures a person will actually hit
const err = src => (compileSet(src, ['both','sight','sound']).error || '').slice(0, 60);
ok('an unknown condition names itself and lists the real ones',
  /unknown condition "smell"/.test(err('both and not smell')), err('both and not smell'));
ok('an empty expression says so', /empty/.test(err('')), err(''));
ok('a dangling operator does not silently pass',
  err('both and').length > 0, err('both and'));
ok('an unbalanced parenthesis does not silently pass',
  err('(both and sight').length > 0, err('(both and sight'));
ok('a valid expression compiles clean', !compileSet('both and not sight', ['both','sight']).error);

console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);

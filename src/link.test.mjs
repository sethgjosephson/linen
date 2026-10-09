import { encodeSceneFragment, decodeSceneFragment, toBase64Url, fromBase64Url } from './link.js';
import { ok, report } from '../tools/harness.mjs';

const bytes = Uint8Array.from({ length: 300 }, (_, i) => (i*37) & 255);
ok('base64url round trip', fromBase64Url(toBase64Url(bytes)).join() === bytes.join());
ok('base64url has no padding or url characters', !/[+/=]/.test(toBase64Url(bytes)));

const scene = { format:9, resolution:25, hues:{ sheet:120 }, groups:[{ label:'g', members:[1, 2], hue:200 }],
  nodes:[{ id:1, type:'sphere', x:10, y:20, params:{ center:[0, 0, 0], radius:300 }, inputs:[] },
    { id:2, type:'scatter', x:10, y:80, params:{ fill:1, density:90000, type:0, seed:7, tag:'cells' }, inputs:[{ id:1 }] }] };
const json = JSON.stringify(scene);
const frag = await encodeSceneFragment(json);
ok('the fragment names the scene', frag.startsWith('scene='));
ok('the fragment is smaller than the JSON', frag.length < json.length, frag.length + ' vs ' + json.length);
const back = await decodeSceneFragment('#' + frag + '&other=1');
ok('the scene comes back exactly, seeds and resolution included', JSON.stringify(back) === json);
ok('a fragment without a scene is null', (await decodeSceneFragment('#res=50')) === null);
let threw = false;
try { await decodeSceneFragment('scene=' + toBase64Url(new Uint8Array([1, 2, 3]))); } catch(e){ threw = true; }
ok('a corrupt fragment throws rather than loading nothing', threw);
report('link');

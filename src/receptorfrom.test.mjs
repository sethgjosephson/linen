// A receptor channel that names a population (from:<tag> on the receptors node) takes the connect node's synapses from that population.
// The channel is the top three bits of the synapse byte, the rule id the low five, and both have to survive.
import { NODE_DEFS, NEURON_TYPES, computeNode, setResolution, wireConnect, assembleConnect } from './nodes.js';
import { ok, threw, report, inProcessWorker } from '../tools/harness.mjs';
inProcessWorker(wireConnect, assembleConnect);

function mkEditor(){
  return { nodes:[], nextId:1, groups:[], byId(id){ return this.nodes.find(n => n.id === id); },
    addNode(t, x, y){ const def = NODE_DEFS[t], params = {}; def.params.forEach(p => params[p.k] = structuredClone(p.def));
      const n = { id:this.nextId++, type:t, x, y, params, inputs:new Array(def.inputs).fill(null) }; this.nodes.push(n); return n; } };
}
setResolution(1);
const FS = NEURON_TYPES.findIndex(t => t.key === 'FS07'), LTS = NEURON_TYPES.findIndex(t => t.key === 'LTS07'), RS = NEURON_TYPES.findIndex(t => t.key === 'RS07');

// proj: a project node from Ae to As naming a receptor channel, placed before the receptors node
async function build(table, rule = false, proj = null){
  const ed = mkEditor();
  const sph = ed.addNode('sphere', 0, 0); Object.assign(sph.params, { center:[0, 0, 0], radius:150 });
  const pops = [['Ae', RS, 1], ['Ai', FS, 2], ['As', LTS, 3]].map(([tag, type, seed]) => {
    const s = ed.addNode('scatter', 0, 100); Object.assign(s.params, { count:60, fill:0, tag, type, seed });
    s.inputs[0] = { id:sph.id }; return s; });
  const g = ed.addNode('gather', 0, 150); pops.forEach((s, k) => g.inputs[k] = { id:s.id });
  let up = g;
  if(rule){
    const pl = ed.addNode('plasticity', 0, 170); Object.assign(pl.params, { name:'fac', aP:0, aM:0, stp:2, stpU:0.03, stpTauD:130, stpTauF:530 });
    pl.inputs[0] = { id:up.id }; up = pl;
  }
  if(proj){
    const pj = ed.addNode('project', 0, 175); Object.assign(pj.params, { from:'Ae', to:'As', dispersed:1, fanout:10, weight:1, velocity:300, seed:4, channel:proj.channel });
    pj.inputs[0] = { id:up.id }; up = pj;
  }
  const rc = ed.addNode('receptor', 0, 180); rc.params.table = table; rc.inputs[0] = { id:up.id };
  const cn = ed.addNode('connect', 0, 200);
  Object.assign(cn.params, { radius:1000, sigma:100000, prob:0.2, wExc:1, wInh:-1, wdist:0, cluster:0, seed:1,
    table:rule ? 'Ae As 1 1 fac' : '' });
  cn.inputs[0] = { id:rc.id };
  const cp = ed.addNode('checkpoint', 0, 300); Object.assign(cp.params, { syn:1 }); cp.inputs[0] = { id:cn.id };
  return computeNode(cp, id => ed.byId(id));
}
const chanOf = (net, tag) => {
  const seen = new Set();
  for(let i = 0; i < net.count; i++) if(net.tags[net.src[i]] === tag)
    for(let k = net.preStart[i]; k < net.preStart[i + 1]; k++) seen.add(net.pmask[k] >> 5);
  return [...seen].sort().join(',');
};

let net = await build('sst 20 - from:A*s');
ok('every synapse from the named population is on the channel (channel 0 is byte 2)', chanOf(net, 'As') === '2', chanOf(net, 'As'));
ok('synapses from the other populations stay on E and I', chanOf(net, 'Ae') === '0' && chanOf(net, 'Ai') === '0', chanOf(net, 'Ae') + ' ' + chanOf(net, 'Ai'));
net = await build('sst 20 - from:As', true);
const ruleOf = (pre, post) => { const ids = new Set();
  for(let i = 0; i < net.count; i++) if(net.tags[net.src[i]] === pre)
    for(let k = net.preStart[i]; k < net.preStart[i + 1]; k++) if(net.tags[net.src[net.post[k]]] === post) ids.add(net.pmask[k] & 31);
  return [...ids].join(','); };
ok('the rule id in the low five bits is kept beside the channel', ruleOf('Ae', 'As') === '2' && chanOf(net, 'As') === '2', ruleOf('Ae', 'As') + ' ' + chanOf(net, 'As'));
ok('a channel whose sign is not its population\'s is refused', /whose cells are inhibitory/.test(String(await build('slow 20 + from:As').then(() => '', e => e.message))));
// a projection naming a channel: its synapses carry the channel, the local wiring from the same cells does not
const pairChans = (net, pre, post) => { const seen = new Set();
  for(let i = 0; i < net.count; i++) if(net.tags[net.src[i]] === pre)
    for(let k = net.preStart[i]; k < net.preStart[i + 1]; k++) if(net.tags[net.src[net.post[k]]] === post) seen.add(net.pmask[k] >> 5);
  return [...seen].sort().join(','); };
net = await build('slow 80 + mg:1', false, { channel:'slow' });
ok('a channel with no transmitter and no population is accepted, for the project nodes that name it', !!net.receptors && net.receptors.channels[0].name === 'slow');
ok('the projection\'s synapses carry the channel beside the local ones from the same cells', pairChans(net, 'Ae', 'As') === '0,2', pairChans(net, 'Ae', 'As'));
ok('the same cells\' other synapses stay on E', pairChans(net, 'Ae', 'Ae') === '0' && pairChans(net, 'Ae', 'Ai') === '0', pairChans(net, 'Ae', 'Ae') + ' ' + pairChans(net, 'Ae', 'Ai'));
ok('a projection naming a channel the scene does not declare is refused', /no receptor channel named/.test(String(await build('slow 80 + mg:1', false, { channel:'nmda' }).then(() => '', e => e.message))));
ok('a projection whose source cells are not the channel\'s sign is refused', /are excitatory/.test(String(await build('slow 80 - mg:1', false, { channel:'slow' }).then(() => '', e => e.message))));
ok('a population on two channels is refused', /on two channels/.test(threw(() => NODE_DEFS.receptor.compute([{ kind:'points', count:0 }], { table:'x 20 - from:As\ny 30 - from:As' })) || ''));
report('receptorfrom');

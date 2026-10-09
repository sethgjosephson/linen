// The connections file node: a connections table becomes explicit synapses among the cells whose ids a points file carried, on top of the synapses the connect node wires.
import { ok, threw, report } from '../tools/harness.mjs';
import { parseConnectionsTable, NODE_DEFS, setFileReader, wireConnect, engineConfig, freezeGraded } from './nodes.js';

const pointsCsv = 'id,x,y,z\nc1,0,0,0\nc2,200,0,0\nc3,200,300,0\n';
const table = 'pre_root_id,post_root_id,syn_count,nt_type\nc1,c2,3,ACH\nc2,c3,2,GABA\nc9,c1,4,ACH\nc1,c1,1,ACH\n';
const t = parseConnectionsTable(table, {});
ok('columns found by the FlyWire names', t.rows === 4 && t.ids[t.pre[0]] === 'c1' && t.ids[t.post[1]] === 'c3' && t.count[0] === 3);
ok('ids are interned once', t.ids.length === 4, t.ids.join(','));
ok('the transmitter column gives a sign', t.sign && t.sign[0] === 1 && t.sign[1] === -1);
ok('histamine is inhibitory too (the fly photoreceptor synapse)', parseConnectionsTable('pre,post,nt\na,b,histamine\nb,a,ACH\n', {}).sign.join() === '-1,1');
ok('a named column that is missing is refused', /no column named/.test(threw(() => parseConnectionsTable(table, { pre:'nope' })) || ''));
ok('a table without a count counts one per row', parseConnectionsTable('pre,post\na,b\n', {}).count[0] === 1);

setFileReader(async path => path === 'p.csv' ? new TextEncoder().encode(pointsCsv) : path === 'w.csv' ? new TextEncoder().encode(table) : null);
const CN = { radius:1, sigma:1, prob:0, wExc:5, wInh:-10, wdist:0, wsigma:1, cluster:0, velocity:200, density:100, nuE:0, nuI:0, seed:1, table:'' };
const build = async (over) => {
  const pts = await NODE_DEFS.pointfile.compute([], { file:'p.csv', columns:'', id:'id', scale:1, up:0, type:0, tag:'c' }, { id:1 });
  const wired = await NODE_DEFS.connectionsfile.compute([pts], { file:'w.csv', pre:'', post:'', count:'', signCol:'', weight:5, mode:0, signMode:0, velocity:100, rule:'', ...over }, { id:2 });
  return wireConnect(wired, CN, null);   // prob 0: the sweep makes nothing, the table makes all
};
let net = await build({});
const syn = n => { const out = []; for(let i=0;i<n.count;i++) for(let s=n.preStart[i]; s<n.preStart[i+1]; s++) out.push([i, n.post[s], n.w[s], n.delay[s], n.pmask[s]]); return out; };
let S = syn(net);
ok('two rows matched, one unmatched, one self-connection skipped', net.connectionsReport.matched === 2 && net.connectionsReport.unmatched === 1 && net.connectionsReport.self === 1, JSON.stringify(net.connectionsReport));
ok('one synapse per row of weight times the count', S.length === 2 && S[0][2] === 15 && S[1][2] === 10, JSON.stringify(S));
ok('sign by the presynaptic cell type (all RS here)', S.every(s => s[2] > 0));
ok('delay is distance over velocity', S[0][3] === 2 && S[1][3] === 3, JSON.stringify(S));
{
  // at 10 um/ms the two pairs (200 and 300 um) are 20 and 30 ms: both past the 16 ms ring, clamped to it and counted
  const slow = await build({ velocity:10 });
  const ds = syn(slow).map(s => s[3]);
  ok('a table synapse delay past the ring is clamped and counted', ds.every(d => d === 16) &&
    slow.connectionsReport.clampedDelay === 2 && slow.delayClamped === 2, JSON.stringify({ ds, w:slow.connectionsReport, total:slow.delayClamped }));
}
ok('the default rule is the checkpoint\'s', S.every(s => s[4] === 1));
net = await build({ signMode:1, wInh:0.5 });
S = syn(net);
ok('the inhibitory factor scales inhibitory synapses only', S[0][2] === 15 && S[1][2] === -5, JSON.stringify(S));
net = await build({ mode:1 });
S = syn(net);
ok('count synapses of the weight in the other mode', S.length === 5 && S.every(s => s[2] === 5), JSON.stringify(S));
net = await build({ signMode:1 });
S = syn(net);
ok('sign by the transmitter column', S[0][2] === 15 && S[1][2] === -10, JSON.stringify(S));
net = await build({ rule:'frozen' });
ok('frozen table synapses carry rule 0', syn(net).every(s => s[4] === 0));
ok('the net reports what it wired', net.connectionsReport.synapses === 2);
let msg = null; try { await build({ file:'none.csv' }); } catch(e){ msg = e.message; }
ok('a missing table is refused', /not in the project folder/.test(msg || ''), msg);

// count compression and the ceiling
net = await build({ compress:1 });
ok('square root of the count', Math.abs(syn(net)[0][2] - 5*Math.sqrt(3)) < 1e-4);
net = await build({ compress:2 });
ok('log2 of one plus the count', Math.abs(syn(net)[0][2] - 5*Math.log2(4)) < 1e-4);
net = await build({ compress:3 });
ok('every pair alike', syn(net).every(s => s[2] === 5));
net = await build({ cap:2 });
ok('the ceiling applies before the weight', syn(net)[0][2] === 10 && syn(net)[1][2] === 10);
report('connections file');

// receptor channels: a receptor node on the stream gives each table synapse the channel its transmitter names (top three bits of the plasticity byte) and that channel's sign; an unnamed transmitter refuses the wire
const rc = NODE_DEFS.receptor.compute([await NODE_DEFS.pointfile.compute([], { file:'p.csv', columns:'', id:'id', scale:1, up:0, type:0, tag:'c' }, { id:1 })],
  { table:'ach 3 + ach\ngabaa 12 - gaba' }, { id:7 });
ok('the receptor node carries its channels', rc.receptors.channels.length === 2 && rc.receptors.channels[1].tau === 12 && rc.receptors.channels[1].sign === -1);
let rnet = await wireConnect(await NODE_DEFS.connectionsfile.compute([rc], { file:'w.csv', pre:'', post:'', count:'', signCol:'', weight:5, mode:0, signMode:1, velocity:100, rule:'' }, { id:2 }), CN, null);
let RS2 = syn(rnet);
ok('each synapse carries its channel above the rule and the channel sign', RS2[0][4] === (1 | (2 << 5)) && RS2[0][2] === 15 && RS2[1][4] === (1 | (3 << 5)) && RS2[1][2] === -10, JSON.stringify(RS2));
ok('the engine config lists the channel time constants', JSON.stringify(engineConfig(rnet).chanTau) === '[3,12]');
const rc2 = NODE_DEFS.receptor.compute([await NODE_DEFS.pointfile.compute([], { file:'p.csv', columns:'', id:'id', scale:1, up:0, type:0, tag:'c' }, { id:1 })], { table:'ach 3 + ach' }, { id:7 });
ok('a transmitter the node does not name refuses the wiring', /no channel on the receptor node/.test(await NODE_DEFS.connectionsfile.compute([rc2], { file:'w.csv', pre:'', post:'', count:'', signCol:'', weight:5, mode:0, signMode:1, velocity:100, rule:'' }, { id:2 }).then(w => wireConnect(w, CN, null)).then(() => '', e => e.message)));
ok('a second receptor node is refused', /already carries a receptor node/.test(threw(() => NODE_DEFS.receptor.compute([rc], { table:'x 1 + y' }, { id:8 })) || ''));
const rc3 = NODE_DEFS.receptor.compute([await NODE_DEFS.pointfile.compute([], { file:'p.csv', columns:'', id:'id', scale:1, up:0, type:0, tag:'c' }, { id:1 })],
  { table:'ach 3 + 0 ach\nglucl 30 - -75 gaba' }, { id:7 });
ok('a reversal potential after the sign is read', rc3.receptors.channels[1].erev === -75 && rc3.receptors.channels[1].transmitters.join() === 'gaba' && rc3.receptors.channels[0].erev === 0);
ok('a channel without one takes the sign default in the engine config', JSON.stringify(engineConfig(rnet).chanErev) === '[0,-70]' && JSON.stringify(engineConfig({ ...rnet, receptors:rc3.receptors }).chanErev) === '[0,-75]');
ok('freezing a graded cell keeps the channel bits', freezeGraded(Uint8Array.of(1 | (2 << 5), 1), Int32Array.of(0, 1, 2), Uint8Array.of(1, 0), 2).join() === [2 << 5, 1].join());
report('connections file (receptors)');

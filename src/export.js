// The wired network as plain files: one row per cell and one row per synapse, in CSV, for any tool that reads a table (Elephant and Neo through NumPy, a spreadsheet, R).
// The Brian 2 case is briancase.js.
import { csv } from './figure.js';

export function neuronsCsv(net, types){
  const rows = [];
  const tags = net.tags || {};
  for(let i = 0; i < net.count; i++){
    const t = types[net.ntype[i]];
    rows.push([i, +net.pos[i*3].toFixed(2), +net.pos[i*3+1].toFixed(2), +net.pos[i*3+2].toFixed(2),
      t ? t.key : net.ntype[i], t ? (t.sign > 0 ? 'E' : 'I') : '', tags[net.src[i]] || '', +net.bias[i].toFixed(4)]);
  }
  return csv(['index', 'x_um', 'y_um', 'z_um', 'type', 'sign', 'population', 'bias'], rows);
}
// weight is in the engine's unit (a kick in mV on a classic row, the charge or the peak current in exp mode, nS in conductance mode); the rule id and the receptor channel are the low five and the top three bits of the synapse byte
export function synapsesCsv(net){
  const rows = [];
  const pm = net.pmask;
  for(let i = 0; i < net.count; i++)
    for(let k = net.preStart[i]; k < net.preStart[i + 1]; k++)
      rows.push([i, net.post[k], +net.w[k].toFixed(5), net.delay[k], pm ? pm[k] & 31 : '', pm ? pm[k] >> 5 : '']);
  return csv(['pre', 'post', 'weight', 'delay_ms', 'rule', 'channel'], rows);
}

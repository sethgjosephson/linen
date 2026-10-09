// Dropping several free nodes on a wire inserts them as a chain in screen order, and the bypass key acts on the whole selection.
import { ok, report } from '../tools/harness.mjs';
const { NodeEditor } = await import('./editor.js');
const ed = Object.create(NodeEditor.prototype);
Object.assign(ed, { nodes:[], groups:[], nextId:1, cb:{ onSelect(){}, onChange(){}, onMoved(){} },
  viewSlots:{}, activeView:null, suggestResolution:0, selSet:new Set(), sel:null });
const g = ed.addNode('sphere', 0, 0), sc = ed.addNode('scatter', 0, 100), con = ed.addNode('connect', 0, 400);
sc.inputs[0] = { id:g.id }; con.inputs[0] = { id:sc.id };
const a = ed.addNode('noisewarp', 300, 250), b = ed.addNode('twist', 300, 180), c = ed.addNode('move', 320, 250);
ok('the three are free', [a, b, c].every(n => ed.isFree(n)));
const chain = ed.insertChain({ src:sc, dst:con, port:0 }, [a, b, c]);
ok('chained top to bottom, ties left to right', chain.map(n => n.type).join('>') === 'twist>noisewarp>move', chain.map(n => n.type).join('>'));
ok('the first reads the wire\'s source', b.inputs[0].id === sc.id);
ok('each reads the one above', a.inputs[0].id === b.id && c.inputs[0].id === a.id);
ok('the wire\'s target reads the last', con.inputs[0].id === c.id);
ok('the inserted nodes are no longer free', ![a, b, c].some(n => ed.isFree(n)));

// Copy and paste between scene tabs.
// A wire from outside the copied set is kept by the source's id, which names another node in another scene.
ed.docKey = 1; ed.cursorGraph = () => [600, 600]; ed.lastMouse = null;
ed.sel = sc; ed.selSet = new Set([sc]); sc.name = 'cells'; sc.params.tag = 'cells';
ed.copySelection();
ok('a copy remembers its scene and its outside wire', ed.clipboard.from === 1 && ed.clipboard[0].ext[0] === sc.inputs[0].id);
ed.sel = null; ed.selSet = new Set(); ed.pasteClipboard();
const home = ed.nodes[ed.nodes.length - 1];
ok('pasted where it came from, it reads what the original reads and is told apart from it',
  home.inputs[0] && home.inputs[0].id === sc.inputs[0].id && home.name !== 'cells' && home.params.tag !== 'cells' && home.params.seed !== sc.params.seed, home.name + ' ' + home.params.tag);
// another scene in the same editor: other nodes under the same ids
const keep = { nodes:ed.nodes, nextId:ed.nextId };
ed.nodes = []; ed.nextId = 1; ed.docKey = 2;
for(let k = 0; k < 8; k++) ed.addNode('sphere', 0, k*40);   // an id equal to the copied wire's source exists here too
ed.sel = null; ed.selSet = new Set(); ed.pasteClipboard();
const away = ed.nodes[ed.nodes.length - 1];
ok('pasted into another scene, the outside wire is dropped', away.type === 'scatter' && !away.inputs[0]);
ok('and the node keeps its name, tag and seed, since nothing there has them', away.name === 'cells' && away.params.tag === 'cells' && away.params.seed === sc.params.seed, away.name + ' ' + away.params.tag);
ed.sel = null; ed.selSet = new Set(); ed.pasteClipboard();
const again = ed.nodes[ed.nodes.length - 1];
ok('a second paste there steps on from the first', again.name !== 'cells' && again.params.tag !== 'cells', again.name + ' ' + again.params.tag);
Object.assign(ed, keep);
report('editorchain');

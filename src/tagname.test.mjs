import { tagToken, tagFollowsName, defaultNodeName, suggestSelfPairs, bumpName } from './tagname.js';
import { ok, report } from '../tools/harness.mjs';

// --- tagToken: a tag is one whitespace-free token --------------------------
ok('a plain name is its own tag', tagToken('relay') === 'relay');
ok('leading and trailing space is trimmed', tagToken('  relay  ') === 'relay');
ok('inner space is removed, since a tag cannot hold one',
   tagToken('V1 e') === 'V1e');
ok('runs of whitespace collapse away', tagToken('a\t  b') === 'ab');
ok('empty stays empty', tagToken('') === '' && tagToken(null) === '' && tagToken(undefined) === '');

// --- the derived link ------------------------------------------------------
ok('a fresh node, name and tag equal, is linked', tagFollowsName('sheet', 'sheet'));
ok('a node named but not yet tagged adopts the name (empty is linked)',
   tagFollowsName('sheet', ''));
ok('a hand-set tag that differs is not linked',
   !tagFollowsName('scatter2', 'v1e'));
ok('a name with a space still links to its token tag',
   tagFollowsName('V1 e', 'V1e'));
ok('an existing scene, name blank and tag set, is not linked',
   !tagFollowsName('', 'v1e'));

// --- unique defaults, so two populations never share a tag -----------------
ok('the first of a type takes the bare title', defaultNodeName('scatter', []) === 'scatter');
ok('a title with a space is tokenized', defaultNodeName('read node', []) === 'readnode');
ok('a duplicate gets 2', defaultNodeName('scatter', ['scatter']) === 'scatter2');
ok('and then 3, skipping the taken ones',
   defaultNodeName('scatter', ['scatter', 'scatter2']) === 'scatter3');
ok('a gap is filled rather than jumped',
   defaultNodeName('scatter', ['scatter', 'scatter3']) === 'scatter2');
ok('checkpoint is a fine token as is', defaultNodeName('checkpoint', []) === 'checkpoint');

// --- what a connect node can offer -----------------------------------------
{
  const s = suggestSelfPairs(['sheet', 'sheeti', 'pacemaker'], []);
  ok('one self-connection per population', s.length === 3);
  ok('and each is the population onto itself',
     s.every(([a, b]) => a === b) && s[0][0] === 'sheet');
  const s2 = suggestSelfPairs(['sheet', 'sheeti'], [['sheet', 'sheet']]);
  ok('a self-pair the table already has is skipped', s2.length === 1 && s2[0][0] === 'sheeti');
  ok('an empty upstream offers nothing', suggestSelfPairs([], []).length === 0);
  ok('a blank tag is not offered', suggestSelfPairs(['', 'x'], []).length === 1);
}


// --- through the real editor: only an identity tag adopts the name ----------
// A probe or stimulus tag is a filter where blank means every cell.
// Filling it with the node's name would make every scenario probe select nothing ("probe selects no neurons: no cell is tagged probe").
{
  const { NodeEditor } = await import('./editor.js');
  const ed = Object.create(NodeEditor.prototype);
  Object.assign(ed, { nodes:[], groups:[], nextId:1, cb:{ onSelect(){}, onChange(){} } });
  const sc = ed.addNode('scatter', 0, 0), pr = ed.addNode('probe', 0, 0), st = ed.addNode('stimulus', 0, 0);
  ok('a scatter adopts its name as its tag', sc.name === 'scatter' && sc.params.tag === 'scatter');
  ok('a probe is named but its tag stays blank (every cell)', pr.name === 'probe' && pr.params.tag === '');
  ok('a stimulus is named but its tag stays blank (every cell)', st.name === 'stimulus' && st.params.tag === '');
}

// --- a copy is a new population -------------------------------------------
// duplicate and paste must not clone the tag and the seed, or a pasted scatter is the same population twice, at the same coordinates.
{
  ok('a trailing number counts up', bumpName('RSexc1', ['RSexc1']) === 'RSexc2');
  ok('past every taken one', bumpName('RSexc1', ['RSexc1', 'RSexc2', 'rsexc3']) === 'RSexc4');
  ok('a plain name takes a 2', bumpName('scatter', ['scatter']) === 'scatter2');
  const { NodeEditor } = await import('./editor.js');
  const ed = Object.create(NodeEditor.prototype);
  Object.assign(ed, { nodes:[], groups:[], nextId:1, cb:{ onSelect(){}, onChange(){} } });
  const a = ed.addNode('scatter', 0, 0);
  a.name = 'RS exc 1'; a.params.tag = 'RSexc1'; a.params.seed = 3;
  const [b] = ed.duplicate([a]);
  ok('the copy takes the next name', b.name === 'RS exc 2', b.name);
  ok('and the next tag', b.params.tag === 'RSexc2', b.params.tag);
  ok('and a seed of its own', b.params.seed !== a.params.seed, String(b.params.seed));
  const c = ed.addNode('scatter', 0, 0);           // name and tag both 'scatter'
  const [d] = ed.duplicate([c]);
  ok('a tag that follows the name follows the new name',
     d.name === 'scatter2' && d.params.tag === 'scatter2', d.name + ' ' + d.params.tag);
  const m = ed.addNode('gather', 0, 0);
  const [m2] = ed.duplicate([m]);
  ok('a node without a tag only gets a new name', m2.name === 'gather2' && m2.params.tag === undefined);
}

report('tagname');

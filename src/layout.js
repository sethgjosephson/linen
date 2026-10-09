// Node-tree layout for a scenario graph.
//
// Position is display only and never reaches the computation, so it is derived from the graph rather than maintained by hand in nine scenarios.
//
// The arrangement follows the convention a compositing node tree uses, because that is what this graph is: signal flows down, each branch owns a column, and a node with several inputs sits in line with its first one while the others branch in from the side.
// Everything feeding one merge therefore sits together above it rather than being spread across a row with the rest of its layer, which is the property that makes a tree readable at a glance.
// Laying every layer out as a row packs tightly and reads badly, because a row says these nodes are alternatives when in fact they belong to different branches.

import { NODE_DEFS } from './nodes.js';
import { LABEL_ADV } from './editor.js';

const W = 130, H = 26;                   // must match editor.js
const COL = W + 30, ROW = H + 118;   // generous rows: these graphs are read
                                     // by following wires, not by scanning

// Row for each node, measured by the longest path to an output rather than from an input.
// Both keep a node below everything it consumes, but they place a branch very differently.
// Measuring from the input side pushes every branch to the top of the canvas, so a population that feeds the twelfth merge in a chain still sits in the second row and its wire crosses the whole graph to get there.
// Measuring to the output side puts each node as low as it can go, which lands a branch's tail directly above the merge it feeds and keeps its wire short.
function depths(nodes, parentsOf, childrenOf){
  const toSink = new Map(), state = new Map();
  const walk = n => {
    if(toSink.has(n.id)) return toSink.get(n.id);
    if(state.get(n.id) === 1) return 0;          // cycle: treat as an output
    state.set(n.id, 1);
    let d = 0;
    for(const c of childrenOf(n)) d = Math.max(d, walk(c) + 1);
    state.set(n.id, 2);
    toSink.set(n.id, d);
    return d;
  };
  for(const n of nodes) walk(n);
  const far = Math.max(...nodes.map(n => toSink.get(n.id)));
  const depth = new Map();
  for(const n of nodes) depth.set(n.id, far - toSink.get(n.id));
  return depth;
}

// Height a note will take once the editor wraps it.
// The editor measures the text on a canvas and this module has none, so the wrap point is estimated from the monospace advance the note is drawn at, paragraph by paragraph so a blank line between passages still costs a row.
// Slightly generous rather than exact: a note that is stacked a few pixels too far apart reads fine, one stacked too close overlaps its neighbor.
const NOTE_ADV = 6.4;                    // px per character at the drawn size
export function noteHeight(n){
  const w = Math.max(100, (n.params && n.params.width|0) || 220);
  const per = Math.max(8, Math.floor((w - 16)/NOTE_ADV));
  let lines = 0;
  for(const para of String((n.params && n.params.text) || '').split('\n'))
    lines += Math.max(1, Math.ceil(para.length/per));
  return Math.max(26, lines*13 + 12);
}

// One connected block of the graph: the nodes reachable from one another through wires or group membership.
// Laid out on its own and set at x0, y0; returns how many nodes moved and the block's right edge.
function layoutComponent(nodes, groups, x0, y0){
  // Groups are drawn as a backdrop around the bounding box of their members, so a group only reads as a group if its members end up next to each other.
  // Nothing in the dataflow says they should: clusters that all feed the same merge chain are pulled toward the same x by the centring pass and interleave, and their backdrops drawn over each other read as one slab.
  // Membership is therefore given to the ordering pass below, which keeps a group's members contiguous within each row.
  const gOf = new Map();
  groups.forEach((g, i) => {
    for(const id of (g.members || [])) if(!gOf.has(id)) gOf.set(id, i);
  });
  // Notes are prose about the scene, not part of it: they have no inputs and no outputs, so the dataflow pass reads them as sinks and stacks the lot in the bottom row, as far from what they describe as it is possible to put them.
  // They are held out here and given their own margin column once the graph's extent is known.
  const byId = new Map(nodes.map(n => [n.id, n]));
  const parentsOf = n => (n.inputs || [])
    .filter(Boolean).map(c => byId.get(c.id)).filter(Boolean);
  const kids = new Map(nodes.map(n => [n.id, []]));
  for(const n of nodes)
    for(const p of parentsOf(n)) kids.get(p.id).push(n);
  const childrenOf = n => kids.get(n.id) || [];
  // A mask arriving on a diamond side port is an auxiliary wire, not the pipe the node sits on.
  // Counting it equally when centring drags a chain of stimulus nodes sideways one step at a time, each pulled halfway toward a mask box parked off to the right, so the chain reads as a staircase instead of a strand.
  // Side wires still keep their source above the node that reads it; they just barely vote on where the node sits.
  const SIDE = 0.12;
  const isSide = (n, i) => {
    const d = NODE_DEFS[n.type];
    return !!d && d.side !== undefined && i >= d.side;
  };
  // weighted neighbors of n: [node, weight]
  const near = new Map(nodes.map(n => [n.id, []]));
  for(const n of nodes)
    (n.inputs || []).forEach((c, i) => {
      const p = c && byId.get(c.id);
      if(!p) return;
      const wgt = isSide(n, i) ? SIDE : 1;
      near.get(n.id).push([p, wgt]);
      near.get(p.id).push([n, wgt]);
    });
  const depth = depths(nodes, parentsOf, childrenOf);

  // Walk up from the outputs, laying each branch into its own column.
  // A node takes the column of its first input, so a chain stays in one line and reads as a single strand; the other inputs get columns of their own, so a merge's sources sit side by side directly above it.
  // Leaves claim the next free column in the order the walk reaches them, which is what groups each merge's feeders together: the walk descends one merge completely before it starts the next.
  const sinks = nodes.filter(n => childrenOf(n).length === 0);

  // Column for each node.
  // The first input keeps the node's own column so a chain reads as one strand down the page; the others are asked to sit one column to its right, beside the node they feed.
  // They ask for the same column rather than each taking a fresh one, because a branch that has been consumed frees its space: the placement pass below resolves the overlaps, so branches that occupy different rows share a column and only genuinely simultaneous ones spread sideways.
  // Allocating a new column per branch instead put the twentieth branch twenty columns from the trunk and traded long vertical wires for long horizontal ones.
  const col = new Map();
  const visit = (n, hint) => {
    if(col.has(n.id) && col.get(n.id) >= 0) return col.get(n.id);
    col.set(n.id, hint);
    const ins = parentsOf(n);
    ins.forEach((p, i) => visit(p, i === 0 ? hint : hint + 1));
    return hint;
  };
  // Sinks in the order the scene made them, so a scene with several checkpoints reads left to right in the order its notes number them.
  [...sinks].sort((a, b) => a.id - b.id)
    .forEach(n => visit(n, 0));
  for(const n of nodes) if(!col.has(n.id)) visit(n, 0);

  // Grid pass.
  // Two nodes can inherit the same column from a shared input; when they also share a row they would land on top of each other, so the later one steps right into the first free slot.
  const taken = new Set();
  const key = (r, c) => r + ':' + c;
  const rowsOf = new Map();
  const placed = [...nodes].sort((a, b) =>
    (depth.get(a.id) - depth.get(b.id)) || (col.get(a.id) - col.get(b.id))
    || (a.id - b.id));
  for(const n of placed){
    const r = depth.get(n.id);
    let c = Math.max(0, col.get(n.id));
    while(taken.has(key(r, c))) c++;
    taken.add(key(r, c));
    col.set(n.id, c);
    n.x = c*COL; n.y = r*ROW;
    if(!rowsOf.has(r)) rowsOf.set(r, []);
    rowsOf.get(r).push(n);
  }

  // Centring pass.
  // The grid alone leaves everything hard against the left edge, so a merge sits off to one side of the inputs it gathers and a fan of branches reads as a ragged staircase.
  // Each node is pulled toward the mean of whatever it connects to in the neighboring row, then the row is spread back out to keep a minimum gap in the order it already had, so nothing crosses that did not cross before.
  // Sweeping both directions settles chains and merges together.
  const rowKeys = [...rowsOf.keys()].sort((a, b) => a - b);
  // Each node wants to sit at the mean of everything it connects to.
  // The row is then laid out in that order with a minimum gap, and shifted so its center of mass lands where the targets wanted it.
  // That last shift is what actually centers anything: spreading a row rightward from a shared target puts the leftmost node on the target and everything else to its right, which is why a merge gathering three branches sat at the far left of them instead of above their middle.
  // How hard a group pulls its own members into one vertical band.
  // Ordering alone keeps members side by side within a row but says nothing across rows, so a cluster whose three rows each drifted toward a different merge spread its backdrop wide enough to swallow the neighboring one.
  const COHERE = 0.35;
  const relax = (row, gx) => {
    const want = new Map();
    for(const n of row){
      const rel = near.get(n.id) || [];
      let sw = 0, sx = 0;
      for(const [m, k] of rel){ sw += k; sx += k*m.x; }
      let t = sw > 0 ? sx/sw : n.x;
      const g = gOf.get(n.id);
      if(g !== undefined && gx.has(g)) t = t*(1 - COHERE) + gx.get(g)*COHERE;
      want.set(n.id, t);
    }
    // Order by where each node wants to be, except that members of one group move as a block: they sort on the block's mean rather than their own, so the row reads group by group and each backdrop encloses a contiguous run instead of a scatter with other groups' nodes between.
    const gWant = new Map(), gN = new Map();
    for(const n of row){
      const g = gOf.get(n.id);
      if(g === undefined) continue;
      gWant.set(g, (gWant.get(g) || 0) + want.get(n.id));
      gN.set(g, (gN.get(g) || 0) + 1);
    }
    const keyOf = n => {
      const g = gOf.get(n.id);
      return g === undefined ? want.get(n.id) : gWant.get(g)/gN.get(g);
    };
    row.sort((a, b) => (keyOf(a) - keyOf(b))
      || ((gOf.get(a.id) ?? -1) - (gOf.get(b.id) ?? -1))
      || (want.get(a.id) - want.get(b.id)) || (a.id - b.id));
    let prev = -Infinity, prevG = null;
    for(const n of row){
      const g = gOf.get(n.id);
      // clear of the neighboring backdrop's padding, which is 14 a side
      const gap = COL + (prevG !== null && g !== prevG ? 46 : 0);
      n.x = Math.max(want.get(n.id), prev + gap);
      prev = n.x; prevG = g === undefined ? null : g;
    }
    const mAct = row.reduce((s, n) => s + n.x, 0)/row.length;
    const mWant = row.reduce((s, n) => s + want.get(n.id), 0)/row.length;
    const d = mWant - mAct;
    for(const n of row) n.x += d;
  };
  for(let sweep = 0; sweep < 12; sweep++){
    const gSum = new Map(), gCount = new Map();
    for(const n of nodes){
      const g = gOf.get(n.id);
      if(g === undefined) continue;
      gSum.set(g, (gSum.get(g) || 0) + n.x);
      gCount.set(g, (gCount.get(g) || 0) + 1);
    }
    const gx = new Map();
    for(const [g, s] of gSum) gx.set(g, s/gCount.get(g));
    const seq = sweep % 2 === 0 ? rowKeys : [...rowKeys].reverse();
    for(const r of seq) relax(rowsOf.get(r), gx);
  }

  // Backdrops must not sit on top of each other, and per-row ordering cannot promise that: two clusters can be side by side in every row they share and still have overlapping bounding boxes, because one's widest row is a row the other does not reach at all.
  // Each group is therefore given a horizontal lane: taken left to right, a group slides right until its box clears every group placed before it whose rows it shares.
  // Overlap is resolved even when one box would completely contain the other, since a backdrop drawn inside another still lets their nodes land on the same pixels.
  {
    const PADX = 14, TOPB = 36, GAP = 18;
    const lanes = groups.map(g => {
      const ms = [...new Set(g.members || [])]
        .map(id => byId.get(id)).filter(Boolean);
      // the label widens the drawn box, and so does any room added by dragging its edges, so both have to widen the lane too
      const p = g.pad || {};
      return ms.length ? { ms, label: LABEL_ADV*(g.label || '').length + 16,
        l:p.l || 0, t:p.t || 0, r:p.r || 0, b:p.b || 0 } : null;
    }).filter(Boolean);
    const boxOf = g => {
      let bx0 = Infinity, bx1 = -Infinity, by0 = Infinity, by1 = -Infinity;
      for(const n of g.ms){
        bx0 = Math.min(bx0, n.x); bx1 = Math.max(bx1, n.x + W);
        by0 = Math.min(by0, n.y); by1 = Math.max(by1, n.y + H);
      }
      const x0 = bx0 - PADX - g.l;
      return { x0, x1:Math.max(bx1 + PADX + g.r, x0 + g.label),
        y0:by0 - TOPB - g.t, y1:by1 + PADX + g.b };
    };
    const separate = () => {
      lanes.sort((a, b) => boxOf(a).x0 - boxOf(b).x0);
      const done = [];
      for(const lane of lanes){
        const b = boxOf(lane);
        let shift = 0;
        for(let pass = 0; pass < 40; pass++){
          let hit = false;
          for(const h of done){
            const hb = boxOf(h);
            if(b.y0 >= hb.y1 || hb.y0 >= b.y1) continue;    // no shared rows
            const x0s = b.x0 + shift, x1s = b.x1 + shift;
            if(x0s >= hb.x1 || hb.x0 >= x1s) continue;      // already clear
            shift += hb.x1 - x0s + GAP;
            hit = true;
          }
          if(!hit) break;
        }
        if(shift) for(const n of lane.ms) n.x += shift;
        done.push(lane);
      }
    };
    // A group that moved can land on a node that belongs to no group, so every row is swept again and anything that now overlaps is pushed right.
    // A member takes its whole group with it, so a backdrop that was just cleared is not pulled apart to fix one node.
    const repair = () => {
      for(const r of rowKeys){
        const row = [...rowsOf.get(r)].sort((a, b) => a.x - b.x);
        let prev = null;
        for(const n of row){
          if(prev && n.x < prev.x + COL){
            const d = prev.x + COL - n.x, g = gOf.get(n.id);
            if(g === undefined) n.x += d;
            else for(const m of nodes) if(gOf.get(m.id) === g) m.x += d;
          }
          prev = n;
        }
      }
    };
    // Once the backdrops are clear of each other their members cannot collide either, so the last sweep moves only ungrouped nodes: each one steps right past any grouped node it lands on.
    // Nothing inside a group moves, so this cannot put two backdrops back on top of each other.
    const repairFree = () => {
      for(const r of rowKeys){
        const row = rowsOf.get(r);
        const fixed = row.filter(n => gOf.has(n.id)).map(n => [n.x, n.x + COL]);
        const free = row.filter(n => !gOf.has(n.id)).sort((a, b) => a.x - b.x);
        let prev = -Infinity;
        for(const n of free){
          let x = Math.max(n.x, prev);
          for(let guard = 0; guard < 200; guard++){
            let hit = false;
            for(const [f0, f1] of fixed)
              if(x < f1 && f0 < x + COL){ x = f1; hit = true; }
            if(!hit) break;
          }
          n.x = x; prev = x + COL;
        }
      }
    };
    // Each repair can push a group back into a neighbor, so the two run alternately, separation runs last, and only free nodes are tidied after it.
    // The extra rounds run only while two grouped nodes are still on each other, because each pass undoes a little of the other and running them to a fixed count widened the graph for nothing.
    const clash = () => {
      for(const r of rowKeys){
        const row = [...rowsOf.get(r)].filter(n => gOf.has(n.id))
          .sort((a, b) => a.x - b.x);
        for(let i = 1; i < row.length; i++)
          if(row[i].x < row[i-1].x + COL
             && gOf.get(row[i].id) !== gOf.get(row[i-1].id)) return true;
      }
      return false;
    };
    for(let i = 0; i < 8; i++){ separate(); repair(); }
    for(let i = 0; i < 8 && clash(); i++){ separate(); repair(); }
    separate();
    repairFree();
  }

  const minX = Math.min(...nodes.map(n => n.x));
  let moved = 0;
  for(const n of nodes){
    const nx = Math.round(x0 + n.x - minX), ny = Math.round(y0 + n.y);
    if(n.x !== nx || n.y !== ny) moved++;
    n.x = nx; n.y = ny;
  }

  return { moved, x1:Math.max(...nodes.map(n => n.x)) + W };
}


// The trunk layout, modeled on a hand arrangement of the cortical column: the populations sit in a row along the top, each with its shape above its scatters, the merges gather them at the left below the row, and from there one straight trunk runs down the page: connect, the stimuli, the probes, the checkpoint, the chart.
// A wire from the population row into the trunk (a region masking a stimulus, a stream from a population that is not the first) does not cross the page on a diagonal: it drops down a lane to the right of its own population through reroute dots, one level with each node it feeds, so every wire is a vertical run and a short hook.
// A shape that feeds only a trunk node (a probe's window, a pulse's target) sits to the right of that node, one row up, where its wire is short.
//
// Dots are made through `route(x, y)` when the caller can add nodes; the wires run direct otherwise.
// Returns null when the graph has no trunk (no node with exactly one main input above the sink), and the generic layout stands.
const LANE = 30, DOTW = 18, BLOCKGAP = 110, MROW = 96;
function layoutTrunk(nodes, groups, x0, y0, route){
  const byId = new Map(nodes.map(n => [n.id, n]));
  const isSide = (n, i) => { const d = NODE_DEFS[n.type]; return !!d && d.side !== undefined && i >= d.side; };
  const mains = n => (n.inputs || []).map((c, i) => [c, i])
    .filter(([c, i]) => c && byId.has(c.id) && !isSide(n, i)).map(([c]) => byId.get(c.id));
  const kids = new Map(nodes.map(n => [n.id, []]));
  for(const n of nodes) for(const c of n.inputs || []) if(c && byId.has(c.id)) kids.get(c.id).push(n);
  const sinks = nodes.filter(n => !kids.get(n.id).length).sort((a, b) => a.id - b.id);
  if(!sinks.length) return null;
  // the trunk: up the main input from the first sink while there is one
  const trunk = []; let cur = sinks[0];
  while(cur){ trunk.push(cur); const m = mains(cur); if(m.length !== 1) break; cur = m[0]; }
  const top = trunk[trunk.length - 1];
  const tree = [], streams = [], inTree = new Set();
  if(mains(top).length === 0){ trunk.pop(); streams.push(top); }
  else {
    // the merge tree above the trunk: every junction reachable through main inputs; what a junction takes that is not itself a junction is a stream
    const walk = m => {
      if(inTree.has(m.id)) return;
      inTree.add(m.id); tree.push(m);
      for(const p of mains(m)){ if(mains(p).length >= 2) walk(p); else streams.push(p); }
    };
    walk(top);
  }
  if(trunk.length < 2 && !tree.length) return null;
  const inTrunk = new Set(trunk.map(n => n.id));
  const fixed = id => inTrunk.has(id) || inTree.has(id);
  // population blocks: a stream and everything above it, joined when two streams share a node (the shape both scatters of a layer fill)
  const blockOf = new Map(), blocks = [];
  streams.forEach((s, k) => {
    const members = [];
    const seen = new Set();
    const up = n => {
      if(seen.has(n.id) || fixed(n.id)) return;
      seen.add(n.id); members.push(n);
      for(const c of n.inputs || []) if(c && byId.has(c.id)) up(byId.get(c.id));
    };
    up(s);
    let block = null;
    for(const n of members) if(blockOf.has(n.id)){ block = blockOf.get(n.id); break; }
    if(!block){ block = { ns:[], leaves:[] }; blocks.push(block); }
    for(const n of members) if(!blockOf.has(n.id)){ blockOf.set(n.id, block); block.ns.push(n); }
    block.leaves.push(s);
  });
  const inBlock = new Set([...blockOf.keys()]);
  // a shape that feeds only trunk nodes: beside its first consumer
  const free = nodes.filter(n => !fixed(n.id) && !inBlock.has(n.id));

  // the population row
  let x = x0, moved = 0;
  const made = [];
  let rowBottom = y0;
  const placedBlocks = [];
  for(const b of blocks){
    const gs = groups.filter(g => (g.members || []).some(id => b.ns.some(n => n.id === id)));
    let x1;
    if(b.ns.length === 1){ const n = b.ns[0]; if(n.x !== x || n.y !== y0) moved++; n.x = x; n.y = y0; x1 = x + W; }
    else { const r = layoutComponent(b.ns, gs, x, y0); moved += r.moved; x1 = r.x1; }
    for(const n of b.ns) rowBottom = Math.max(rowBottom, n.y + H);
    placedBlocks.push({ b, x0:x, x1 });
    x = x1;                               // lanes and the gap are added once the wires are known
  }
  const trunkX = blocks.length ? blocks[0].leaves[0].x : x0;
  // the merge tree at the trunk's x, the lowest merge nearest the trunk
  const level = new Map();
  const lvl = m => { if(level.has(m.id)) return level.get(m.id); let l = 1;
    for(const p of mains(m)) if(inTree.has(p.id)) l = Math.max(l, lvl(p) + 1);
    level.set(m.id, l); return l; };
  let deepest = 0;
  for(const m of tree) deepest = Math.max(deepest, lvl(m));
  let y = rowBottom + 70;
  for(const m of tree){
    const ny = y + (lvl(m) - 1)*MROW;
    if(m.x !== trunkX || m.y !== ny) moved++;
    m.x = trunkX; m.y = ny;
  }
  if(tree.length) y += (deepest - 1)*MROW + H + 90;
  // the trunk, top down
  for(let i = trunk.length - 1; i >= 0; i--){
    const n = trunk[i];
    if(n.x !== trunkX || n.y !== y) moved++;
    n.x = trunkX; n.y = y; y += ROW;
  }
  // free shapes: right of the trunk, one row above the node they feed
  const freeX = trunkX + W + 50;
  let rightmost = x;
  const rowUsed = new Map();
  for(const s of free.sort((a, b) => a.id - b.id)){
    const cs = kids.get(s.id).filter(c => fixed(c.id)).sort((a, b) => a.y - b.y);
    const c = cs[0] || trunk[0];
    const k = rowUsed.get(c.id) || 0; rowUsed.set(c.id, k + 1);
    const nx = freeX + k*(W + 30), ny = c.y - 50;
    if(s.x !== nx || s.y !== ny) moved++;
    s.x = nx; s.y = ny; rightmost = Math.max(rightmost, nx + W);
  }
  // lanes: every wire from a population into a trunk or tree node that is not the first population's own stream, one lane per source, a dot at the top and one level with each node it feeds
  const wiresFrom = src => { const out = [];
    for(const c of kids.get(src.id)) (c.inputs || []).forEach((w, i) => { if(w && w.id === src.id && fixed(c.id)) out.push({ dst:c, port:i }); });
    return out; };
  let laneX = 0;
  placedBlocks.forEach((pb, k) => {
    const sources = pb.b.ns.map(src => ({ src, wires:wiresFrom(src)
        .filter(w => !(k === 0 && inTree.has(w.dst.id) && !isSide(w.dst, w.port))) }))   // the first population's streams go straight down
      .filter(s => s.wires.length)
      .sort((a, b) => Math.min(...a.wires.map(w => w.dst.y)) - Math.min(...b.wires.map(w => w.dst.y)));
    let lane = Math.max(pb.x1 + 60, k === 0 && free.length ? rightmost + 40 : 0);
    for(const { src, wires } of sources){
      wires.sort((a, b) => a.dst.y - b.dst.y || a.port - b.port);
      if(route){
        let prev = route(lane - DOTW/2, src.y + H + 20); made.push(prev);
        prev.inputs[0] = { id:src.id };
        for(const w of wires){
          const py = isSide(w.dst, w.port) ? w.dst.y + H/2 - DOTW : w.dst.y - 70;
          const d = route(lane - DOTW/2, Math.round(py)); made.push(d);
          d.inputs[0] = { id:prev.id };
          w.dst.inputs[w.port] = { id:d.id };
          prev = d;
        }
      }
      laneX = lane; lane += LANE;
    }
    const right = sources.length ? laneX + DOTW : pb.x1;
    rightmost = Math.max(rightmost, right);
    const next = placedBlocks[k + 1];
    if(next){
      const shift = right + BLOCKGAP - next.x0;
      if(shift > 0) for(let j = k + 1; j < placedBlocks.length; j++){
        for(const n of placedBlocks[j].b.ns) n.x += shift;
        placedBlocks[j].x0 += shift; placedBlocks[j].x1 += shift;
      }
    }
  });
  // anything left (a second sink and its chain): under its input, to the right
  const placed = new Set([...trunk, ...tree, ...free, ...blocks.flatMap(b => b.ns)].map(n => n.id));
  for(let guard = 0; guard < nodes.length; guard++){
    const rest = nodes.filter(n => !placed.has(n.id) && mains(n).every(p => placed.has(p.id)));
    if(!rest.length) break;
    for(const n of rest){ const p = mains(n)[0] || trunk[0]; n.x = p.x + W + 40; n.y = p.y + ROW/2; placed.add(n.id); moved++; }
  }
  for(const n of nodes) if(!placed.has(n.id)){ n.x = trunkX; n.y = y; y += ROW; moved++; }
  const every = [...nodes, ...made];
  const minX = Math.min(...every.map(n => n.x));
  if(minX !== x0) for(const n of every) n.x = Math.round(n.x + x0 - minX);
  return { moved, x1:Math.max(...every.map(n => n.x)) + W, made };
}

// The scene's layout: every connected block of the graph laid out on its own, the blocks side by side left to right in the order the scene made them (by the id of each block's first sink, the checkpoint in every scene), then the notes.
// Blocks laid out together, with the lanes settled by position, would put the guided tour's third checkpoint left of its second.
export function layoutTopDown(all, opts = {}){
  if(!all || all.length < 2) return 0;
  const notes = all.filter(n => n.type === 'note');
  const nodes = all.filter(n => n.type !== 'note');
  if(nodes.length < 2) return 0;
  const x0 = opts.x0 ?? 60, y0 = opts.y0 ?? 60, groups = opts.groups || [];
  const byId = new Map(nodes.map(n => [n.id, n]));
  // union over wires and over group membership, so a backdrop never straddles two blocks
  const parent = new Map(nodes.map(n => [n.id, n.id]));
  const find = a => { while(parent.get(a) !== a){ parent.set(a, parent.get(parent.get(a))); a = parent.get(a); } return a; };
  const union = (a, b) => { a = find(a); b = find(b); if(a !== b) parent.set(a, b); };
  for(const n of nodes) for(const c of n.inputs || []) if(c && byId.has(c.id)) union(n.id, c.id);
  for(const g of groups){ const ms = (g.members || []).filter(id => byId.has(id)); for(let i = 1; i < ms.length; i++) union(ms[0], ms[i]); }
  const comps = new Map();
  for(const n of nodes){ const r = find(n.id); if(!comps.has(r)) comps.set(r, []); comps.get(r).push(n); }
  const blocks = [...comps.values()].map(ns => {
    const ids = new Set(ns.map(n => n.id)), consumed = new Set();
    for(const n of ns) for(const c of n.inputs || []) if(c && ids.has(c.id)) consumed.add(c.id);
    const sinks = ns.filter(n => !consumed.has(n.id)).map(n => n.id);
    return { ns, key:Math.min(...(sinks.length ? sinks : ns.map(n => n.id))) };
  }).sort((a, b) => a.key - b.key);
  let moved = 0, x = x0;
  const made = [];                          // reroute dots the trunk layout added
  // room between blocks for a note beside a group when the scene has one
  const GAPX = notes.some(n => n.params && String(n.params.near || '').trim()) ? 400 : 120;
  for(const b of blocks){
    if(b.ns.length === 1){
      const n = b.ns[0];
      if(n.x !== x || n.y !== y0) moved++;
      n.x = x; n.y = y0; x += W + GAPX; continue;
    }
    const gs = groups.filter(g => (g.members || []).some(id => b.ns.some(n => n.id === id)));
    const r = (opts.trunk && layoutTrunk(b.ns, gs, x, y0, opts.route)) || layoutComponent(b.ns, gs, x, y0);
    if(r.made) made.push(...r.made);
    moved += r.moved; x = r.x1 + GAPX;
  }
  // Notes.
  // One that names a group (its `near` parameter, the start of the group's label) sits to the right of that group's backdrop, top aligned, several stacking downward, provided the spot is clear of every node and every other backdrop; otherwise, and for a note that names nothing, the margin column to the left of everything, aligned with the group's top when it has one and stacked from the top when it does not.
  if(notes.length){
    const PADX = 14, TOPB = 36;
    const boxes = groups.map(g => {
      const ms = [...new Set(g.members || [])].map(id => byId.get(id)).filter(Boolean);
      if(!ms.length) return null;
      let bx0 = Infinity, bx1 = -Infinity, by0 = Infinity, by1 = -Infinity;
      for(const n of ms){ bx0 = Math.min(bx0, n.x); bx1 = Math.max(bx1, n.x + W); by0 = Math.min(by0, n.y); by1 = Math.max(by1, n.y + H); }
      const x0 = bx0 - PADX;
      return { label:String(g.label || '').toLowerCase(), x0, x1:Math.max(bx1 + PADX, x0 + LABEL_ADV*(g.label || '').length + 16), y0:by0 - TOPB, y1:by1 + PADX };
    }).filter(Boolean);
    const rects = [...boxes.map(b => [b.x0, b.y0, b.x1, b.y1]), ...nodes.map(n => [n.x, n.y, n.x + W, n.y + H]),
      ...made.map(n => [n.x, n.y, n.x + DOTW, n.y + DOTW])];
    const clear = (x, y, w, h) => !rects.some(([a0, b0, a1, b1]) => x < a1 && a0 < x + w && y < b1 && b0 < y + h);
    const left = Math.min(...nodes.map(n => n.x));
    const top = Math.min(...nodes.map(n => n.y));
    const wide = Math.max(...notes.map(n => +(n.params && n.params.width) || 340));
    const colX = Math.round(left - wide - 90);
    const placed = [];                        // [x, y, w, h] of every note placed so far
    const put = (n, x, y) => {
      const w = +(n.params && n.params.width) || 340, h = noteHeight(n);
      if(n.x !== x || n.y !== y) moved++;
      n.x = x; n.y = y; placed.push([x, y, w, h]); rects.push([x, y, x + w, y + h]);   // rects are corners
    };
    // the column: the first free y at or below the asked one
    const column = (n, wantY) => {
      const h = noteHeight(n);
      let y = wantY;
      for(let guard = 0; guard < 200; guard++){
        const hit = placed.find(([px, py, pw, ph]) => px === colX && y < py + ph && py < y + h);
        if(!hit) break;
        y = hit[1] + hit[3] + 34;
      }
      put(n, colX, Math.round(y));
    };
    let colY = top;
    for(const n of notes){
      const near = String((n.params && n.params.near) || '').trim().toLowerCase();
      const box = near ? boxes.find(b => b.label.startsWith(near)) : null;
      if(!box){ column(n, colY); colY = n.y + noteHeight(n) + 34; continue; }
      const w = +(n.params && n.params.width) || 340, h = noteHeight(n);
      let x = Math.round(box.x1 + 30), y = Math.round(box.y0), done = false;
      for(let guard = 0; guard < 12 && !done; guard++){
        if(clear(x, y, w, h)){ put(n, x, y); done = true; break; }
        // a note already beside this group: step below it
        const below = placed.find(([px, py, pw, ph]) => px === x && y < py + ph && py < y + h);
        if(below) y = below[1] + below[3] + 18; else break;
      }
      if(!done) column(n, box.y0 + TOPB);
    }
  }
  return moved;
}

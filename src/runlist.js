// The runs of a project folder, as a list in the FILES tab.
//
// It lives in its own module because two callers need the same list and one of them is not a run: a training run refreshes it as it writes files, and BROWSE RUNS on the checkpoint opens it with nothing running, which is how a finished run is looked at.
import * as P from './project.js';

// The runs a chart can draw, newest first, and the rows of one of them.
export async function listRunTags(ST){
  const tags = (await ST.list(P.runsDir())).filter(e => e.dir).map(e => e.name);
  return tags.sort().reverse();
}
export async function readRunMetrics(ST, tag){
  const rows = JSON.parse(new TextDecoder().decode(await ST.read(P.runFile(tag, P.RUN_METRICS))));
  return Array.isArray(rows) ? rows : [];
}

// The sweeps a chart can draw.
// A sweep is several runs of one graph, so its file is the list of them: drawing it means drawing the mean of a group of runs with the spread across them, which is the only honest reading of a chaotic system measured once per configuration.
export async function listSweepTags(ST){
  const files = (await ST.list(P.runsDir())).filter(e => !e.dir && /^sweep-.*\.json$/.test(e.name));
  return files.map(e => e.name.replace(/\.json$/, '')).sort().reverse();
}
// Returns { stamp, hours, engine, groups:[{ label, tags, histories }] }: one group per variant and condition, since the mean of two conditions together is a number about nothing.
export async function readSweep(ST, tag){
  const file = JSON.parse(new TextDecoder().decode(await ST.read(P.runsDir() + '/' + tag + '.json')));
  const rows = (file.rows || []).filter(r => r && r.tag && !r.failed);
  const groups = new Map();
  for(const r of rows){
    const label = [r.variant || '', r.cond && r.cond !== 'as configured' ? r.cond : ''].filter(Boolean).join(' · ') || 'all runs';
    if(!groups.has(label)) groups.set(label, { label, tags:[], histories:[] });
    groups.get(label).tags.push(r.tag);
  }
  for(const g of groups.values())
    for(const t of g.tags){
      // a run whose metrics file is missing is left out of its group rather than counted as a run with no checkpoints
      try { g.histories.push(await readRunMetrics(ST, t)); } catch(e){}
    }
  return { stamp:file.stamp || tag, hours:file.hours, engine:file.engine,
    replicates:file.replicates, groups:[...groups.values()].filter(g => g.histories.length) };
}

// UTC tag to local start time, so a run is identifiable at a glance
export const tagLabel = tag => {
  const m = tag.match(/^(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)$/);
  if(!m) return tag;
  const d = new Date(Date.UTC(+m[1], +m[2]-1, +m[3], +m[4], +m[5]));
  return `${tag} (started ${d.toLocaleString()})`;
};

// ul: the <ul> to fill.
// ST: an open store. currentTag marks the run in progress. onChart, when given, adds a second link to every metrics file that hands its parsed rows over, which is what draws a finished run in the chart nodes.
export async function renderRuns(ul, ST, { currentTag = null, onChart = null,
  onError = () => {}, whereText = null } = {}){
  ul.innerHTML = '';
  // Browser storage is invisible from outside the tab, so a run that went there has to say so here or it reads as a run that went to disk.
  if(whereText){
    const where = document.createElement('li');
    where.className = 'fwhere' + (ST.mode === 'project' ? '' : ' browser');
    where.textContent = whereText;
    ul.appendChild(where);
  }
  const tags = (await ST.list(P.runsDir())).filter(e => e.dir).map(e => e.name);
  tags.sort().reverse();                    // newest run first
  for(const tag of tags){
    // The run's files and the checkpoints a directory down are listed together because to a reader they are one run's output; the split exists so retention can cull one half and not the other.
    const files = (await ST.list(P.runDir(tag))).filter(e => !e.dir)
      .map(e => [e.name, e, P.runFile(tag, e.name)]);
    for(const e of await ST.list(P.runBrainsDir(tag)))
      if(!e.dir) files.push([e.name, e, P.runBrainPath(tag, e.name)]);
    files.sort((a, b) => a[0] < b[0] ? -1 : 1);
    const bytes = files.reduce((t, f) => t + f[1].size, 0);
    const head = document.createElement('li');
    head.className = 'frun' + (tag === currentTag ? ' cur' : '');
    head.textContent = `${tagLabel(tag)} · ${files.length} files · ` +
      `${(bytes/1073741824).toFixed(2)} GB` + (tag === currentTag ? ' · this run' : '');
    ul.appendChild(head);
    for(const [name, ent, path] of files){
      const li = document.createElement('li');
      li.className = 'ffile';
      const a = document.createElement('a');
      a.textContent = name + ' (' + (ent.size/1048576).toFixed(1) + ' MB)';
      a.href = '#';
      a.onclick = async e => {
        e.preventDefault();
        const url = await ST.url(path);
        if(!url) return;
        const dl = document.createElement('a');
        dl.href = url; dl.download = name; dl.click();
        // A project-folder url points at the file on disk and must not be revoked; only the blob copy browser storage hands back is ours.
        if(ST.mode !== 'project') setTimeout(() => URL.revokeObjectURL(url), 5000);
      };
      li.appendChild(a);
      // Every run writes the same rows to metrics.json, so that file gets a second link which loads them into the chart nodes.
      if(onChart && name === P.RUN_METRICS){
        li.appendChild(document.createTextNode(' '));
        const ch = document.createElement('a');
        ch.textContent = 'chart';
        ch.className = 'fchart';
        ch.href = '#';
        ch.title = 'draw this run in the chart nodes on the graph';
        ch.onclick = async e => {
          e.preventDefault();
          try {
            const rows = JSON.parse(new TextDecoder().decode(await ST.read(path)));
            onChart(Array.isArray(rows) ? rows : [], tag);
          } catch(err){ onError('could not chart ' + tag + ': ' + err.message); }
        };
        li.appendChild(ch);
      }
      ul.appendChild(li);
    }
  }
  return tags.length;
}

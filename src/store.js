// Where the app's files go.
//
// There are two places a file can land, and which one is in use is not a preference: it is whatever the page can reach.
// Served by serve.py with a project folder, everything goes into that folder as ordinary files anyone can open in a file manager.
// Opened any other way (the Cloudflare deploy, a file:// page, a static server), there is nowhere on disk to write, so it falls back to the browser's own storage.
//
// The fallback is labeled everywhere it is used rather than being silent.
// Browser storage is per-origin and invisible: a run written from localhost:8173 is not there on localhost:8174, and nothing in the file manager shows it exists.
//
// Both modes use the same relative paths, so the two layouts are identical and moving a browser-storage run into a project folder is a copy rather than a translation.
//
//   runs/<tag>/...     training checkpoints, metrics, status
//   brains/...         saved weights
//   scenes/...         saved graphs
//
// Directories are implied by the paths; nothing has to create them first.

const BASE = '';                       // same origin; overridable for tests

// One shared normalizer, because a path that means one thing to the server and another to the fallback is a bug that only shows up in one of them.
export function normPath(p){
  const parts = [];
  for(const seg of String(p || '').split(/[\\/]+/)){
    if(!seg || seg === '.') continue;
    if(seg === '..'){
      if(!parts.length) throw new Error('path escapes the project root: ' + p);
      parts.pop(); continue;
    }
    parts.push(seg);
  }
  if(!parts.length) throw new Error('empty path');
  return parts.join('/');
}

function toBytes(data){
  if(typeof data === 'string') return new TextEncoder().encode(data);
  if(data instanceof Uint8Array) return data;
  if(data instanceof ArrayBuffer) return new Uint8Array(data);
  if(ArrayBuffer.isView(data))
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  throw new Error('write expects a string, ArrayBuffer or typed array');
}

// ---- project folder, over serve.py ----------------------------------------

function serverStore(info, base, fetchFn){
  const enc = p => normPath(p).split('/').map(encodeURIComponent).join('/');
  return {
    mode: 'project',
    root: info.root,
    label: info.root,
    note: '',
    async write(path, data){
      const body = toBytes(data);
      const r = await fetchFn(base + '/project/file/' + enc(path),
        { method:'PUT', body });
      if(!r.ok) throw new Error('write failed (' + r.status + '): ' + path);
      return (await r.json()).bytes;
    },
    async read(path){
      const r = await fetchFn(base + '/project/file/' + enc(path));
      if(r.status === 404) return null;
      if(!r.ok) throw new Error('read failed (' + r.status + '): ' + path);
      return new Uint8Array(await r.arrayBuffer());
    },
    async list(dir){
      const d = String(dir || '').replace(/^[\\/]+|[\\/]+$/g, '');
      const url = d ? base + '/project/list/' + enc(d) : base + '/project/list';
      const r = await fetchFn(url);
      if(!r.ok) throw new Error('list failed (' + r.status + '): ' + dir);
      return (await r.json()).entries;
    },
    async remove(path){
      const r = await fetchFn(base + '/project/file/' + enc(path),
        { method:'DELETE' });
      // a file that is already gone is the state the caller asked for, so this succeeds either way and returns whether it removed anything
      if(r.status === 404) return false;
      if(!r.ok) throw new Error('delete failed (' + r.status + '): ' + path);
      return (await r.json()).removed !== false;
    },
    // A real URL, so a link points at the file on disk instead of copying it into memory to hand back a blob.
    async url(path){ return base + '/project/file/' + enc(path); }
  };
}

// ---- browser storage, over OPFS -------------------------------------------

// The same store over any directory handle: the browser's private origin storage (OPFS), or a folder the person picked with showDirectoryPicker on the hosted site, where there is no server and so no path, but a handle reads and writes the real folder all the same.
async function dirFor(rootOf, path, create){
  const segs = normPath(path).split('/');
  const name = segs.pop();
  let h = await rootOf();
  for(const s of segs){
    try { h = await h.getDirectoryHandle(s, { create }); }
    catch(e){ if(!create) return null; throw e; }
  }
  return { dir: h, name };
}

function opfsStore(why){
  return handleStore(() => navigator.storage.getDirectory(),
    { mode:'browser', root:null, label:'browser storage', note:why });
}

// A folder on this machine reached through a directory handle. root is the folder's name, which is all the page can know of it; the handle itself is kept in IndexedDB so a reload finds the same folder (see fsstore.js).
export function folderStore(handle){
  return handleStore(async () => handle,
    { mode:'folder', root:handle.name, label:'folder ' + handle.name, note:'',
      handle });
}

function handleStore(rootOf, head){
  return {
    ...head,
    async write(path, data){
      const at = await dirFor(rootOf, path, true);
      const fh = await at.dir.getFileHandle(at.name, { create:true });
      const w = await fh.createWritable();
      const body = toBytes(data);
      await w.write(body); await w.close();
      return body.byteLength;
    },
    async read(path){
      const at = await dirFor(rootOf, path, false);
      if(!at) return null;
      try {
        const fh = await at.dir.getFileHandle(at.name);
        return new Uint8Array(await (await fh.getFile()).arrayBuffer());
      } catch(e){ return null; }
    },
    async list(dir){
      const d = String(dir || '').replace(/^[\\/]+|[\\/]+$/g, '');
      let h = await rootOf();
      if(d) for(const s of normPath(d).split('/')){
        try { h = await h.getDirectoryHandle(s); }
        catch(e){ return []; }        // a missing directory lists as empty,
      }                               // the same answer the server gives
      const out = [];
      for await (const [name, handle] of h.entries()){
        if(handle.kind === 'directory'){
          out.push({ name, dir:true, size:0, mtime:0 }); continue;
        }
        const f = await handle.getFile();
        out.push({ name, dir:false, size:f.size, mtime:f.lastModified });
      }
      out.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
      return out;
    },
    async remove(path){
      const at = await dirFor(rootOf, path, false);
      if(!at) return false;
      try { await at.dir.removeEntry(at.name); return true; }
      catch(e){ return false; }
    },
    // No file on disk to point at, so the bytes are copied into a blob URL.
    // Callers revoke it; holding one pins a whole checkpoint in memory.
    async url(path){
      const bytes = await this.read(path);
      if(!bytes) return null;
      return URL.createObjectURL(new Blob([bytes],
        { type:'application/octet-stream' }));
    }
  };
}

// ---- what is already in browser storage -----------------------------------
//
// Runs written before there was a project folder live in OPFS under training/.
// They are not deleted and not moved on their own: a checkpoint directory is measured in gigabytes and copying one is a decision, not a side effect of opening a page.
// What happens automatically is that they are counted and reported, so nothing sits there unnoticed.

const LEGACY_ROOT = 'training';

export async function browserRuns(root = LEGACY_ROOT){
  if(typeof navigator === 'undefined' || !navigator.storage) return [];
  let h;
  try { h = await (await navigator.storage.getDirectory())
    .getDirectoryHandle(root); }
  catch(e){ return []; }               // nothing was ever written there
  const runs = [];
  for await (const [tag, dh] of h.entries()){
    if(dh.kind !== 'directory') continue;
    const files = [];
    let bytes = 0;
    for await (const [name, fh] of dh.entries()){
      if(fh.kind !== 'file') continue;
      const f = await fh.getFile();
      files.push({ name, size:f.size, mtime:f.lastModified });
      bytes += f.size;
    }
    runs.push({ tag, files, bytes });
  }
  runs.sort((a, b) => a.tag < b.tag ? 1 : -1);   // newest first
  return runs;
}

// Copy those runs into a destination store, one file at a time so a large one does not have to fit in memory all at once alongside its copy.
// The source is left alone: deleting gigabytes is the user's call, and a copy that is verified before anything is removed is the only safe order.
export async function importBrowserRuns(dest, onLog = () => {}, root = LEGACY_ROOT){
  const runs = await browserRuns(root);
  let copied = 0, skipped = 0, bytes = 0;
  const src = opfsStore('');
  for(const run of runs){
    for(const f of run.files){
      const to = 'runs/' + run.tag + '/' + f.name;
      // A file already there at the same size is the same file: this is resumable, so an import interrupted half way through does not start over and does not overwrite what it already moved.
      const have = await dest.read(to);
      if(have && have.length === f.size){ skipped++; continue; }
      const data = await src.read(root + '/' + run.tag + '/' + f.name);
      if(!data) continue;
      await dest.write(to, data);
      copied++; bytes += data.length;
      onLog('copied ' + to + ' (' + (data.length/1048576).toFixed(1) + ' MB)');
    }
  }
  return { runs:runs.length, copied, skipped, bytes };
}

// ---- picking one ----------------------------------------------------------

let _store = null, _pending = null;

// Exported for the tests, which need a store pointed at a server of their own rather than whatever origin the page happens to be on.
export async function probe(base = BASE, fetchFn = fetch){
  let info = null;
  try {
    const r = await fetchFn(base + '/project');
    if(r.ok) info = await r.json();
  } catch(e){ /* not served by serve.py */ }
  if(info && info.ok && info.writable) return serverStore(info, base, fetchFn);
  // no server: a folder picked earlier in this browser, if it still lets us in without asking (asking needs a click, which the project menu gives)
  if(projectHandle){
    try {
      const h = await projectHandle();
      if(h && h.queryPermission &&
         await h.queryPermission({ mode:'readwrite' }) === 'granted') return folderStore(h);
    } catch(e){}
  }
  const why = !info
    ? 'this page is not served by serve.py, so there is no project folder to write to'
    : 'the project folder is only writable from the machine serving it';
  return opfsStore(why);
}

// Where a remembered project handle comes from, set by the app (fsstore.js keeps it); the store module stays free of IndexedDB.
let projectHandle = null;
export function setProjectHandleSource(fn){ projectHandle = fn; }
// Adopt a picked folder as the store, in place of whatever probe found.
export function useFolder(handle){ _store = folderStore(handle); _pending = null; return _store; }

// Resolved once and shared: concurrent callers await the same probe rather than each running their own and racing to install a different store.
export function store(){
  if(_store) return Promise.resolve(_store);
  if(!_pending) _pending = probe().then(s => (_store = s, _pending = null, s));
  return _pending;
}

// The project folder can be switched while the page is open.
// The endpoints do not move, but which folder they answer for does, so what was found here is stale: forget it and let the next caller probe again.
export function resetStore(){ _store = null; _pending = null; }

// One line for the UI, so nobody has to guess where a night's run went.
export async function whereLabel(){
  const s = await store();
  return s.mode === 'project' ? 'project folder: ' + s.root
    : s.mode === 'folder' ? 'folder ' + s.root + ' (through the browser)'
    : 'browser storage (this browser only): ' + s.note;
}

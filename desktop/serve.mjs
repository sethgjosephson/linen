// The app's server in Node: serve.py's routes behind the same URLs, so the page is the same page in the desktop app as on the dev server.
// Static files from the checkout with caching off, the project folder file API (/project, /project/browse, /project/list, /project/file GET PUT DELETE, /project/open, /project/pick), /layout, and the engine host's status and launch (/host/status, /host/start).
// Writes, the directory browser, the picker and the host launch answer loopback only, whatever address the server is bound to.
// The desktop app starts it in process (startServer) with its own folder picker and launches the host through its own binary; on its own it is a second dev server:
//
//   node desktop/serve.mjs [port] [--project PATH] [--host ADDR]
//
// tools/servetest.py runs against either server (NPTEST_SERVER names the command), which is what keeps the two answering alike.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import net from 'node:net';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export const HOST_PORT = 8801;
const MIME = {
  '.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8', '.mjs':'text/javascript; charset=utf-8',
  '.css':'text/css; charset=utf-8', '.json':'application/json', '.svg':'image/svg+xml', '.png':'image/png',
  '.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.gif':'image/gif', '.ico':'image/x-icon', '.webp':'image/webp',
  '.txt':'text/plain; charset=utf-8', '.md':'text/plain; charset=utf-8', '.csv':'text/csv; charset=utf-8',
  '.py':'text/plain; charset=utf-8', '.obj':'text/plain; charset=utf-8', '.wasm':'application/wasm',
  '.woff2':'font/woff2', '.woff':'font/woff', '.mp3':'audio/mpeg', '.wav':'audio/wav', '.mp4':'video/mp4',
  '.glb':'model/gltf-binary', '.gltf':'model/gltf+json', '.ply':'application/octet-stream', '.bin':'application/octet-stream',
};
const LOCAL = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

// Somewhere to start browsing from.
// On Windows that is the drives, since there is no single root to walk down from.
export function driveRoots(){
  if(process.platform !== 'win32') return ['/'];
  const out = [];
  for(const c of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'){
    const d = c + ':\\';
    try { if(fs.statSync(d).isDirectory()) out.push(d); } catch(e){}
  }
  return out;
}
// Is this folder one the server may adopt?
// Empty, or already a project.
// The confinement guard on every read and write is only worth something while the project folder is a folder this app owns: pointed at a drive root or a home directory, the guard still holds and the page reads the whole disk, because everything is inside the root.
export function looksLikeProject(dir){
  let entries;
  try { entries = fs.readdirSync(dir); } catch(e){ return { ok:false, why:e.message }; }
  if(!entries.length) return { ok:true, why:null };
  if(entries.includes('project.json')) return { ok:true, why:null };
  for(const e of entries){
    if(!['runs', 'scenes', 'brains', 'nodes'].includes(e)) continue;
    try { if(fs.statSync(path.join(dir, e)).isDirectory()) return { ok:true, why:null }; } catch(err){}
  }
  return { ok:false, why:'this folder already holds other files. Pick an empty folder, or one that already has a project.json in it.' };
}
// The real path of a file that may not exist yet: the deepest ancestor that does is resolved and the rest is appended.
function realpathLoose(p){
  const rest = [];
  let cur = p;
  for(;;){
    try { return path.join(fs.realpathSync.native(cur), ...rest); }
    catch(e){
      const parent = path.dirname(cur);
      if(parent === cur) return path.join(cur, ...rest);
      rest.unshift(path.basename(cur));
      cur = parent;
    }
  }
}
// Absolute path for a project-relative path, or null if it escapes.
// Confinement is checked on the resolved path rather than by looking for dot-dot in the text, because that is not the only way out: a symlink, a Windows short name or a drive-relative path all get there too, and resolving collapses every one of them before the comparison.
export function projectPath(project, rel){
  let r;
  try { r = decodeURIComponent(rel || ''); } catch(e){ return null; }
  r = r.replace(/^\/+/, '');
  if(!r) return null;
  const parts = path.posix.normalize(r).split('/').filter(p => p !== '' && p !== '.');
  // a component that is only dots is never a name anyone asked for, and Windows and the \\?\ file layer disagree about what it means
  if(parts.some(p => /^\.+$/.test(p))) return null;
  const full = realpathLoose(path.join(project, ...parts));
  const root = realpathLoose(project);
  if(full !== root && !full.startsWith(root + path.sep)) return null;
  return full;
}
// The node binary, from PATH or the vendored copy.
export function nodeExe(){
  const names = process.platform === 'win32' ? ['node.exe', 'node.cmd', 'node'] : ['node'];
  for(const dir of (process.env.PATH || '').split(path.delimiter)){
    if(!dir) continue;
    for(const n of names){ const p = path.join(dir, n); try { if(fs.statSync(p).isFile()) return p; } catch(e){} }
  }
  for(const cand of ['D:\\Tools\\node\\node.exe', 'C:\\Program Files\\nodejs\\node.exe', '/usr/bin/node', '/usr/local/bin/node'])
    if(fs.existsSync(cand)) return cand;
  return null;
}
export function hostUp(port = HOST_PORT, timeout = 300){
  return new Promise(res => {
    const s = net.connect({ host:'127.0.0.1', port });
    const done = v => { s.destroy(); res(v); };
    s.setTimeout(timeout, () => done(false));
    s.once('connect', () => done(true));
    s.once('error', () => done(false));
  });
}
const readBody = req => new Promise(res => { const c = []; req.on('data', d => c.push(d)); req.on('end', () => res(Buffer.concat(c))); });
const stamp = () => { const d = new Date(), p = x => String(x).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`; };

// The root is the checkout the page is served from, the project is the project folder, port 0 takes a free one, and host is the bind address.
// The pick({ start, title }) callback resolves to a chosen folder, '' for cancelled; without it /project/pick says there is no picker.
// The hostLaunch { exe, env } is the process that runs host/host.mjs, the vendored or PATH node by default (the desktop app passes its own binary as node).
// The host's log goes to host/host.log under the root, or into logDir when given.
// Resolves with { server, port, project(), setProject(p) } once listening.
export function startServer({ root, project, port = 0, host = '127.0.0.1', pick = null, hostLaunch = null, logDir = null, log = () => {} }){
  root = path.resolve(root);
  let PROJECT = path.resolve(project);
  const HOST_LOG = logDir ? path.join(path.resolve(logDir), 'host.log') : path.join(root, 'host', 'host.log');
  const json = (res, code, obj) => {
    const body = Buffer.from(JSON.stringify(obj));
    res.writeHead(code, { 'Content-Type':'application/json', 'Content-Length':body.length, 'Cache-Control':'no-store' });
    res.end(body);
  };
  const plain = (res, code, text) => {
    res.writeHead(code, { 'Content-Type':'text/plain; charset=utf-8', 'Cache-Control':'no-store' });
    res.end(text);
  };
  // Starting a process, or writing a file, is not something a page on another machine gets to ask for, whatever address this is bound to.
  const localOnly = (req, res) => {
    if(LOCAL.has(req.socket.remoteAddress)) return true;
    json(res, 403, { ok:false, error:'local requests only' });
    return false;
  };
  const confined = (res, rel) => {
    const full = projectPath(PROJECT, rel);
    if(full === null) json(res, 400, { ok:false, error:'path outside the project folder' });
    return full;
  };
  const relOut = full => path.relative(PROJECT, full).split(path.sep).join('/');

  const serveStatic = (req, res, urlPath) => {
    let p;
    try { p = decodeURIComponent(urlPath); } catch(e){ return plain(res, 400, 'bad path'); }
    const parts = path.posix.normalize(p).split('/').filter(x => x && x !== '.');
    if(parts.some(x => x === '..')) return plain(res, 404, 'not found');
    const full = path.join(root, ...parts);
    let st;
    try { st = fs.statSync(full); } catch(e){ return plain(res, 404, 'not found'); }
    if(st.isDirectory()){
      if(!urlPath.endsWith('/')){ res.writeHead(301, { Location:urlPath + '/', 'Cache-Control':'no-store' }); return res.end(); }
      const index = path.join(full, 'index.html');
      if(fs.existsSync(index)) return sendFile(res, index);
      // a listing, links only, the shape the attach page reads
      const names = fs.readdirSync(full).sort();
      const items = names.map(n => { const d = fs.statSync(path.join(full, n)).isDirectory() ? '/' : ''; return `<li><a href="${encodeURIComponent(n)}${d}">${n}${d}</a></li>`; });
      res.writeHead(200, { 'Content-Type':'text/html; charset=utf-8', 'Cache-Control':'no-store' });
      return res.end(`<!doctype html><title>${urlPath}</title><ul>${items.join('')}</ul>`);
    }
    sendFile(res, full);
  };
  const sendFile = (res, full) => {
    const type = MIME[path.extname(full).toLowerCase()] || 'application/octet-stream';
    const st = fs.statSync(full);
    res.writeHead(200, { 'Content-Type':type, 'Content-Length':st.size, 'Cache-Control':'no-store' });
    fs.createReadStream(full).pipe(res);
  };

  const onGet = async (req, res, urlPath, query) => {
    if(urlPath === '/host/status') return json(res, 200, { ok:true, up:await hostUp(), port:HOST_PORT, node:!!(hostLaunch || nodeExe()) });
    // Answered to any client: knowing the folder is not the same as being able to write into it.
    if(urlPath === '/project') return json(res, 200, { ok:true, root:PROJECT, name:path.basename(PROJECT), writable:LOCAL.has(req.socket.remoteAddress) });
    // Server side because the browser's own folder picker deliberately never exposes an absolute path, and an absolute path is exactly what the engine host needs.
    if(urlPath === '/project/browse'){
      if(!localOnly(req, res)) return;
      const want = query.get('path') || '';
      if(!want) return json(res, 200, { ok:true, path:'', parent:null, dirs:driveRoots(), roots:true });
      const full = path.resolve(want);
      let st;
      try { st = fs.statSync(full); } catch(e){ st = null; }
      if(!st || !st.isDirectory()) return json(res, 200, { ok:false, error:'not a folder: ' + full });
      let names;
      try {
        names = fs.readdirSync(full).filter(n => { if(n.startsWith('.')) return false; try { return fs.statSync(path.join(full, n)).isDirectory(); } catch(e){ return false; } }).sort();
      } catch(e){ return json(res, 200, { ok:false, error:e.message }); }
      let parent = path.dirname(full.replace(/[\\/]+$/, '') || full);
      if(parent === full) parent = '';
      const { ok, why } = looksLikeProject(full);
      return json(res, 200, { ok:true, path:full, parent, dirs:names, adoptable:ok, why });
    }
    if(urlPath === '/project/list' || urlPath.startsWith('/project/list/')){
      if(!localOnly(req, res)) return;
      const rel = urlPath.length > '/project/list'.length ? urlPath.slice('/project/list/'.length) : '';
      const full = rel ? confined(res, rel) : realpathLoose(PROJECT);
      if(full === null) return;
      let st;
      try { st = fs.statSync(full); } catch(e){ st = null; }
      if(!st || !st.isDirectory()) return json(res, 200, { ok:true, entries:[] });
      const out = [];
      for(const name of fs.readdirSync(full).sort()){
        let s;
        try { s = fs.statSync(path.join(full, name)); } catch(e){ continue; }
        out.push({ name, dir:s.isDirectory(), size:s.size, mtime:Math.floor(s.mtimeMs) });
      }
      return json(res, 200, { ok:true, entries:out });
    }
    if(urlPath.startsWith('/project/file/')){
      if(!localOnly(req, res)) return;
      const full = confined(res, urlPath.slice('/project/file/'.length));
      if(full === null) return;
      let st;
      try { st = fs.statSync(full); } catch(e){ st = null; }
      if(!st || !st.isFile()) return plain(res, 404, 'not found');
      res.writeHead(200, { 'Content-Type':'application/octet-stream', 'Content-Length':st.size, 'Cache-Control':'no-store' });
      return fs.createReadStream(full).pipe(res);
    }
    serveStatic(req, res, urlPath);
  };

  const onPut = async (req, res, urlPath) => {
    if(!urlPath.startsWith('/project/file/')) return plain(res, 404, 'not found');
    // The body is read before anything else, including before the path is judged.
    // A refusal that leaves the request body unread stalls the client, which turns a clean 400 into a hang.
    const body = await readBody(req);
    if(!localOnly(req, res)) return;
    const full = confined(res, urlPath.slice('/project/file/'.length));
    if(full === null) return;
    try {
      fs.mkdirSync(path.dirname(full), { recursive:true });
      // through a temporary file in the same directory, so a checkpoint interrupted part way through does not leave a truncated file that reads as a complete one
      const tmp = full + '.part';
      fs.writeFileSync(tmp, body);
      fs.renameSync(tmp, full);
    } catch(e){ return json(res, 500, { ok:false, error:e.message }); }
    json(res, 200, { ok:true, bytes:body.length, path:relOut(full) });
  };

  const onDelete = async (req, res, urlPath) => {
    if(!urlPath.startsWith('/project/file/')) return plain(res, 404, 'not found');
    if(!localOnly(req, res)) return;
    const full = confined(res, urlPath.slice('/project/file/'.length));
    if(full === null) return;
    // Deleting something already gone is the state the caller asked for, so it succeeds rather than erroring.
    // It still says whether a file was there, since a caller cleaning up old checkpoints reports what it freed.
    let existed = false;
    try {
      try { existed = fs.statSync(full).isFile(); } catch(e){ existed = false; }
      if(existed) fs.unlinkSync(full);
    } catch(e){ return json(res, 500, { ok:false, error:e.message }); }
    json(res, 200, { ok:true, removed:existed });
  };

  const onPost = async (req, res, urlPath, query) => {
    // A scenario's saved layout (src/layouts/<slug>.json): the scene menu writes the arrangement on screen so it ships as the scenario's default.
    // Loopback only, like every write; the slug is letters, digits and dashes, so the path cannot leave the folder.
    if(urlPath === '/layout'){
      if(!localOnly(req, res)) return;
      const slug = query.get('name') || '';
      if(!slug || !/^[A-Za-z0-9-]+$/.test(slug)) return json(res, 400, { ok:false, error:'bad layout name' });
      const body = await readBody(req);
      let data;
      try { data = JSON.parse(body.length ? body.toString('utf8') : 'null'); } catch(e){ data = null; }
      if(!data || typeof data !== 'object' || Array.isArray(data) || !Array.isArray(data.nodes)) return json(res, 400, { ok:false, error:'not a layout' });
      const folder = path.join(root, 'src', 'layouts');
      fs.mkdirSync(folder, { recursive:true });
      const full = path.join(folder, slug + '.json');
      fs.writeFileSync(full, JSON.stringify(data, null, 1) + '\n');
      return json(res, 200, { ok:true, path:path.relative(root, full).split(path.sep).join('/') });
    }
    // Everything the page and the host write goes through PROJECT, so this is the one place that has to change; a host already running keeps the folder it was launched with, which the reply says so the UI can pass it on.
    if(urlPath === '/project/open'){
      if(!localOnly(req, res)) return;
      const body = await readBody(req);
      let want = null, create = false;
      try { const r = JSON.parse(body.length ? body.toString('utf8') : '{}'); want = (r.path || '').trim(); create = !!r.create; }
      catch(e){ return json(res, 400, { ok:false, error:'bad request body' }); }
      if(!want) return json(res, 400, { ok:false, error:'no path given' });
      const full = path.resolve(want);
      let st;
      try { st = fs.statSync(full); } catch(e){ st = null; }
      if(st && !st.isDirectory()) return json(res, 400, { ok:false, error:'that is a file, not a folder' });
      let created = false;
      if(!st){
        if(!create) return json(res, 200, { ok:false, missing:true, error:'no such folder: ' + full });
        try { fs.mkdirSync(full, { recursive:true }); created = true; }
        catch(e){ return json(res, 400, { ok:false, error:e.message }); }
      }
      const { ok, why } = looksLikeProject(full);
      if(!ok) return json(res, 400, { ok:false, error:why });
      PROJECT = full;
      log('project folder: ' + PROJECT);
      return json(res, 200, { ok:true, root:PROJECT, created, name:path.basename(PROJECT), hostRunning:await hostUp() });
    }
    // The machine's own folder picker, opened by this process, since a page cannot learn a path from the picker it can open itself and the path is what everything here writes to.
    if(urlPath === '/project/pick'){
      if(!localOnly(req, res)) return;
      const body = await readBody(req);
      let r = {};
      try { r = JSON.parse(body.length ? body.toString('utf8') : '{}'); } catch(e){ r = {}; }
      if(!pick) return json(res, 200, { ok:false, error:'no folder picker in this server; type the path, or use the desktop app' });
      let chosen;
      try { chosen = await pick({ start:r.start || PROJECT || process.cwd(), title:r.title || 'project folder' }); }
      catch(e){ return json(res, 200, { ok:false, error:'the folder picker failed: ' + e.message }); }
      return json(res, 200, { ok:true, path:chosen ? path.resolve(chosen) : '', cancelled:!chosen });
    }
    if(urlPath !== '/host/start') return plain(res, 404, 'not found');
    if(!localOnly(req, res)) return;
    const body = await readBody(req);
    let r = {};
    try { r = JSON.parse(body.length ? body.toString('utf8') : '{}'); } catch(e){ r = {}; }
    // The port is the request's to choose within reason: a host keeps the code it loaded, so a page that needs current code while an older host is still busy asks for a fresh one beside it.
    let port = parseInt(r.port, 10);
    if(!Number.isFinite(port)) port = HOST_PORT;
    if(!(port >= 1024 && port <= 65535)) return json(res, 200, { ok:false, up:false, error:'port out of range' });
    if(await hostUp(port)) return json(res, 200, { ok:true, up:true, started:false, note:'a host was already listening' });
    const exe = hostLaunch ? hostLaunch.exe : nodeExe();
    if(!exe) return json(res, 200, { ok:false, up:false, error:'no node binary found on PATH or in D:/Tools/node' });
    // The request chooses the engine and nothing else: the rest of the command is fixed here rather than taken from a page.
    const args = [path.join('host', 'host.mjs'), String(port), '--project', PROJECT];
    if(r.cuda) args.push('--cuda');
    try {
      fs.mkdirSync(path.dirname(HOST_LOG), { recursive:true });
      const fd = fs.openSync(HOST_LOG, 'a');
      fs.writeSync(fd, '\n=== started ' + stamp() + ' ===\n');
      // Detached on purpose: a run can outlast this server, and it already survives the page that started it.
      const child = spawn(exe, args, { cwd:root, stdio:['ignore', fd, fd], detached:true, windowsHide:true,
        env:hostLaunch && hostLaunch.env ? { ...process.env, ...hostLaunch.env } : process.env });
      child.unref();
      fs.closeSync(fd);
    } catch(e){ return json(res, 200, { ok:false, up:false, error:e.message }); }
    // Report reachable rather than launched: the caller is about to connect, and a pid that exited is not an answer it can use.
    for(let i = 0; i < 60; i++){
      if(await hostUp(port)) return json(res, 200, { ok:true, up:true, started:true, cuda:!!r.cuda, log:path.relative(root, HOST_LOG).split(path.sep).join('/') });
      await new Promise(w => setTimeout(w, 250));
    }
    json(res, 200, { ok:false, up:false, error:'host did not come up within 15 s; see ' + path.relative(root, HOST_LOG).split(path.sep).join('/') });
  };

  const server = http.createServer((req, res) => {
    // the path as sent, not resolved: the URL parser would fold a ".." away before the guard could judge it
    const q = req.url.indexOf('?');
    const urlPath = q < 0 ? req.url : req.url.slice(0, q);
    const query = new URLSearchParams(q < 0 ? '' : req.url.slice(q + 1));
    const run = req.method === 'GET' ? onGet : req.method === 'PUT' ? onPut : req.method === 'DELETE' ? onDelete : req.method === 'POST' ? onPost : null;
    if(!run){ res.writeHead(405, { 'Cache-Control':'no-store' }); return res.end(); }
    run(req, res, urlPath, query).catch(e => { if(!res.headersSent) json(res, 500, { ok:false, error:e.message }); else res.end(); });
  });
  return new Promise((res, rej) => {
    server.once('error', rej);
    server.listen(port, host, () => res({ server, port:server.address().port, host, project:() => PROJECT,
      setProject:p => { PROJECT = path.resolve(p); } }));
  });
}

// On its own: a dev server beside serve.py, from this checkout.
if(process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)){
  const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
  const argv = process.argv.slice(2);
  let project = path.join(ROOT, 'project'), host = '127.0.0.1';
  let i = argv.indexOf('--project');
  if(i >= 0){ project = path.resolve(argv[i + 1]); argv.splice(i, 2); }
  i = argv.indexOf('--host');
  if(i >= 0){ host = argv[i + 1]; argv.splice(i, 2); }
  // a positional port wins; otherwise the PORT environment variable, otherwise 8173
  const port = argv[0] ? parseInt(argv[0], 10) : parseInt(process.env.PORT || '8173', 10);
  fs.mkdirSync(project, { recursive:true });
  const s = await startServer({ root:ROOT, project, port, host, log:m => console.log(m) });
  console.log('project folder: ' + project);
  console.log('listening on ' + host + ':' + s.port + (host === '127.0.0.1' ? ' (loopback only)' : ''));
}

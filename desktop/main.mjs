// The desktop app: an Electron window on the app served by desktop/serve.mjs from this checkout, so the page is the one linen.build serves.
// What the shell adds is what a tab cannot have: the server and the engine host inside the app (the host runs on the app's own binary as node), the project folder from the machine's own picker with the last one remembered, no throttling when the window is in the background, a larger heap, and the viewer window put back where it was.
//
//   npm start          (in desktop/, after npm install)
import { app, BrowserWindow, dialog, shell, Menu } from 'electron';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startServer, looksLikeProject } from './serve.mjs';

// From source the app's files are the checkout above this folder; packaged, they are the linen folder electron-builder copies into resources (the build section of package.json says which).
// --root DIR runs the app on another checkout's files instead, so a fork or an edited copy gets this window, its server and its host without a build of its own.
const rootArg = (() => { const k = process.argv.indexOf('--root'); return k >= 0 && process.argv[k + 1] ? path.resolve(process.argv[k + 1]) : null; })();
const ROOT = rootArg || (app.isPackaged ? path.join(process.resourcesPath, 'linen') : path.join(path.dirname(fileURLToPath(import.meta.url)), '..'));
if(rootArg && !fs.existsSync(path.join(ROOT, 'index.html'))){ console.error('linen: no index.html under --root ' + ROOT); app.exit(1); }
// A tab's heap is capped near 4 GB; the wiring of a large network holds more than that (the synapse budget is derived from the machine's memory).
app.commandLine.appendSwitch('js-flags', '--max-old-space-size=16384');

// settings.json in the app's data folder: the project folder, the window and viewer window placements.
const settingsFile = () => path.join(app.getPath('userData'), 'settings.json');
function readSettings(){
  try { return JSON.parse(fs.readFileSync(settingsFile(), 'utf8')); } catch(e){ return {}; }
}
function writeSettings(s){
  try { fs.mkdirSync(path.dirname(settingsFile()), { recursive:true }); fs.writeFileSync(settingsFile(), JSON.stringify(s, null, 1)); } catch(e){}
}
const settings = readSettings();
const remember = (key, win) => {
  const save = () => { if(!win.isDestroyed() && !win.isMinimized()){ settings[key] = { ...win.getNormalBounds(), max:win.isMaximized() }; writeSettings(settings); } };
  win.on('resize', save); win.on('move', save); win.on('close', save);
};
const placed = (key, dflt) => {
  const b = settings[key];
  return b && Number.isFinite(b.x) ? { x:b.x, y:b.y, width:b.width, height:b.height } : dflt;
};

// The first launch asks where the project should live, before there is a window: a folder of the person's choosing, or linen under Documents.
// A chosen folder must be empty or already a project, the rule the server applies to every folder it is pointed at, so the confinement of its file API means something.
async function chooseProject(){
  const dflt = path.join(app.getPath('documents'), 'linen');
  for(;;){
    const ask = await dialog.showMessageBox({ type:'question', title:'linen', message:'Where should your linen project live?',
      detail:'A project is a folder that holds its scenes, runs and brains. Choose an empty folder, or one that already holds a project, or use ' + dflt + '. You can open another project any time from the project menu.',
      buttons:['Choose a folder', 'Use ' + dflt], defaultId:0, cancelId:1 });
    if(ask.response === 1) return dflt;
    const r = await dialog.showOpenDialog({ title:'project folder', defaultPath:app.getPath('documents'), properties:['openDirectory', 'createDirectory', 'promptToCreate'] });
    if(r.canceled || !r.filePaths.length) continue;
    const chosen = r.filePaths[0];
    try { fs.mkdirSync(chosen, { recursive:true }); } catch(e){ await dialog.showMessageBox({ type:'warning', title:'linen', message:e.message, buttons:['OK'] }); continue; }
    const { ok, why } = looksLikeProject(chosen);
    if(!ok){ await dialog.showMessageBox({ type:'warning', title:'linen', message:why, buttons:['OK'] }); continue; }
    return chosen;
  }
}

app.whenReady().then(async () => {
  let project = settings.project;
  if(!project){ project = await chooseProject(); settings.project = project; writeSettings(settings); }
  fs.mkdirSync(project, { recursive:true });
  // the startup folder beside the settings: every node module in it loads with every project (init.js by convention); the folders added from the menu are listed after it
  const startupDir = path.join(app.getPath('userData'), 'startup');
  fs.mkdirSync(startupDir, { recursive:true });
  if(!Array.isArray(settings.nodePaths)) settings.nodePaths = [];
  const s = await startServer({
    root:ROOT, project, port:0, host:'127.0.0.1',
    startupDirs:() => [startupDir, ...settings.nodePaths],
    // the machine's own folder picker, over the app's windows
    pick: async ({ start, title }) => {
      const r = await dialog.showOpenDialog(BrowserWindow.getFocusedWindow() || undefined, { title, defaultPath:start, properties:['openDirectory', 'createDirectory', 'promptToCreate'] });
      return r.canceled || !r.filePaths.length ? '' : r.filePaths[0];
    },
    // the engine host on this binary as node, so the app needs no node install; its log beside the settings, since a packaged app's files may not be writable
    hostLaunch:{ exe:process.execPath, env:{ ELECTRON_RUN_AS_NODE:'1' } },
    logDir:app.getPath('userData'),
    log:m => console.log(m),
  });
  // the server's project folder is the one to remember, wherever the page moves it
  const poll = setInterval(() => { if(s.project() !== settings.project){ settings.project = s.project(); writeSettings(settings); } }, 2000);
  app.on('will-quit', () => clearInterval(poll));

  const win = new BrowserWindow({
    ...placed('window', { width:1600, height:960 }), backgroundColor:'#000', title:'linen', autoHideMenuBar:true,
    webPreferences:{ backgroundThrottling:false, contextIsolation:true, nodeIntegration:false, sandbox:true },
  });
  if(settings.window && settings.window.max) win.maximize();
  remember('window', win);
  // an application menu, hidden until Alt: the edit roles for the text fields, the developer tools on F12 as in a browser, since the page is the same page and people will want to look
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label:'Folders', submenu:[
      { label:'Open the project folder', click:() => shell.openPath(s.project()) },
      { label:'Open the startup folder', click:() => shell.openPath(startupDir) },
      // a folder of node modules served to every project, kept in the settings; the page's "reload node modules" picks it up
      { label:'Add a folder of node modules...', click:async () => {
        const r = await dialog.showOpenDialog(win, { title:'a folder of node modules', properties:['openDirectory'] });
        if(r.canceled || !r.filePaths.length) return;
        const d = r.filePaths[0];
        if(!settings.nodePaths.includes(d)){ settings.nodePaths.push(d); writeSettings(settings); }
      } },
    ] },
    { role:'editMenu' },
    { label:'View', submenu:[ { role:'toggleDevTools', accelerator:'F12' }, { role:'togglefullscreen' } ] },
    { label:'Help', submenu:[ { label:'linen.build', click:() => shell.openExternal('https://linen.build') } ] },
  ]));
  // VIEWER WINDOW opens with window.open and the DOM moves across: the same realm, kept; it gets a window of its own where it was last put.
  // A new-window link to a page of the app (docs.html) gets an ordinary window; a link to anywhere else goes to the system browser.
  const local = url => url.startsWith('http://127.0.0.1:' + s.port + '/');
  const outside = url => /^https?:/.test(url) && !local(url);
  win.webContents.setWindowOpenHandler(({ url, frameName }) => {
    console.log('window open: ' + (frameName || '(no name)') + ' ' + url);
    if(frameName === 'linen-viewer') return { action:'allow', overrideBrowserWindowOptions:{
      ...placed('viewer', { width:1200, height:800 }), backgroundColor:'#000', title:'linen viewer', autoHideMenuBar:true,
      webPreferences:{ backgroundThrottling:false } } };
    // a page of the app in a window of its own, opened here rather than through the native path, which made no window for a link with noopener
    if(local(url)){
      const w = new BrowserWindow({ width:1100, height:900, backgroundColor:'#000', autoHideMenuBar:true, webPreferences:{ contextIsolation:true, nodeIntegration:false, sandbox:true } });
      w.loadURL(url);
    } else if(outside(url)) shell.openExternal(url);
    return { action:'deny' };
  });
  win.webContents.on('did-create-window', (child, { frameName }) => { if(frameName === 'linen-viewer') remember('viewer', child); });
  // a plain link to the outside world must not replace the app in its own window
  win.webContents.on('will-navigate', (e, url) => { if(outside(url)){ e.preventDefault(); shell.openExternal(url); } });
  await win.loadURL('http://127.0.0.1:' + s.port + '/index.html');
});
app.on('window-all-closed', () => app.quit());

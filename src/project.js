// The shape of a project folder, in one place.
//
// Every path the app writes is built here rather than spelled out at the call site, so a layout change touches one file.
//
//   <project>/
//     project.json          what this project is, and the scenes in it
//     scenes/               one file per node tree
//     brains/               brains you saved or imported, kept until you say
//     nodes/                node modules (*.js) the project brings, loaded when it opens (src/plugins.js)
//     runs/
//       <tag>/
//         run.json          how the run was configured and how it ended
//         metrics.json      the per-checkpoint rows
//         trials.npt        the trial matrix, when a probe exports one
//         calibration.json  measured set points, when the run calibrates
//         brains/           automatic checkpoints, culled by the keep setting
//
// Two brains directories is not duplication, it is two lifetimes.
// The ones under a run are written on a cadence and deleted on a cadence: retention drops the oldest as new ones arrive.
// The top-level ones are never culled.
// Sharing one directory means retention eventually deletes something that was meant to be kept, and it would do it quietly.
//
// Inside a run the file names carry no tag.
// Checkpoints keep it: a .npb is the one file here that travels, and once it has been copied somewhere else its name is the only thing left saying which run produced it.

export const PROJECT_FORMAT = 1;
export const PROJECT_FILE = 'project.json';

export const scenesDir = () => 'scenes';
export const scenePath = name => 'scenes/' + name;
export const brainsDir = () => 'brains';
export const brainPath = name => 'brains/' + name;
export const nodesDir = () => 'nodes';
export const nodePath = name => 'nodes/' + name;

export const figuresDir = () => 'figures';
export const figurePath = name => 'figures/' + name;
export const runsDir = () => 'runs';
export const runDir = tag => 'runs/' + tag;
export const runFile = (tag, name) => 'runs/' + tag + '/' + name;
export const runBrainsDir = tag => 'runs/' + tag + '/brains';
export const runBrainPath = (tag, name) => 'runs/' + tag + '/brains/' + name;

// The fixed names inside a run, so a reader does not have to guess and a writer cannot invent a variant.
export const RUN_STATUS = 'run.json';
export const RUN_METRICS = 'metrics.json';
export const RUN_TRIALS = 'trials.npt';
export const RUN_CALIBRATION = 'calibration.json';

// A checkpoint's filename keeps the run tag for the reason above.
export const ckptName = (tag, simMs) =>
  'ckpt-' + tag + '-' + String(Math.round(simMs/60000)).padStart(5, '0') + 'min.npb';

// ---- the project file ------------------------------------------------------

export function newProject(name){
  return { format: PROJECT_FORMAT, name: name || 'project',
           created: new Date().toISOString(),
           scenes: [], activeScene: null };
}

export async function readProject(st){
  const bytes = await st.read(PROJECT_FILE);
  if(!bytes) return null;
  let p;
  try { p = JSON.parse(new TextDecoder().decode(bytes)); }
  catch(e){ throw new Error(PROJECT_FILE + ' is not valid JSON: ' + e.message); }
  // A newer format is not something to guess at.
  // Reading it as if it were this one would write it back in this one, which is how a project written by a later version gets silently downgraded.
  if(typeof p.format === 'number' && p.format > PROJECT_FORMAT)
    throw new Error(PROJECT_FILE + ' is format ' + p.format + ', and this ' +
      'build understands ' + PROJECT_FORMAT + '. Update before opening it.');
  if(!Array.isArray(p.scenes)) p.scenes = [];
  return p;
}

export async function writeProject(st, proj){
  proj.format = PROJECT_FORMAT;
  await st.write(PROJECT_FILE, JSON.stringify(proj, null, 1));
  return proj;
}

// Read it, or create it.
// A folder someone pointed the server at is a project the moment the app writes into it, so this never refuses to start.
export async function openProject(st, name){
  const found = await readProject(st);
  if(found) return found;
  return writeProject(st, newProject(name));
}

export function addScene(proj, file, label){
  if(!proj.scenes.some(s => s.file === file))
    proj.scenes.push({ file, label: label || file.replace(/^scenes\//, '')
      .replace(/\.json$/, '') });
  if(!proj.activeScene) proj.activeScene = file;
  return proj;
}

// ---- migration from the earlier run layout ---------------------------------

// What an earlier run layout holds, and what it becomes:
//
//   runs/<tag>/status.json            -> runs/<tag>/run.json
//   runs/<tag>/metrics-<tag>.json     -> runs/<tag>/metrics.json
//   runs/<tag>/trials-<tag>.npt       -> runs/<tag>/trials.npt
//   runs/<tag>/calibration-<tag>.json -> runs/<tag>/calibration.json
//   runs/<tag>/ckpt-*.npb             -> runs/<tag>/brains/ckpt-*.npb
//
// Only these are touched.
// A file that layout never wrote is left where it is rather than swept somewhere on a guess, because a folder the user put something in is not the app's to tidy.
const RENAMES = [
  [/^status\.json$/, RUN_STATUS],
  [/^metrics-.*\.json$/, RUN_METRICS],
  [/^trials-.*\.npt$/, RUN_TRIALS],
  [/^calibration-.*\.json$/, RUN_CALIBRATION],
];

// Copy, verify, then remove.
// Never the other way round: a move that deletes first and fails second loses a checkpoint, and these are the only copy of a night's training.
async function moveFile(st, from, to, log){
  const bytes = await st.read(from);
  if(!bytes) return false;
  const already = await st.read(to);
  if(already && already.length === bytes.length){
    await st.remove(from);                 // resumable: the copy is done
    log('already moved, dropped the original: ' + from);
    return true;
  }
  await st.write(to, bytes);
  const back = await st.read(to);
  if(!back || back.length !== bytes.length)
    throw new Error('copy of ' + from + ' to ' + to + ' did not verify; ' +
      'the original has been left alone');
  await st.remove(from);
  log(from + ' -> ' + to);
  return true;
}

export async function migrate(st, log = () => {}){
  let moved = 0, runs = 0;
  for(const d of await st.list(runsDir())){
    if(!d.dir) continue;
    const tag = d.name;
    let touched = false;
    for(const f of await st.list(runDir(tag))){
      if(f.dir) continue;
      if(/^ckpt-.*\.npb$/.test(f.name)){
        if(await moveFile(st, runFile(tag, f.name), runBrainPath(tag, f.name), log)){
          moved++; touched = true;
        }
        continue;
      }
      const hit = RENAMES.find(([re]) => re.test(f.name));
      if(!hit) continue;
      if(f.name === hit[1]) continue;
      if(await moveFile(st, runFile(tag, f.name), runFile(tag, hit[1]), log)){
        moved++; touched = true;
      }
    }
    if(touched) runs++;
  }
  return { moved, runs };
}


// Which scene a page shows at startup, given what the project names and which scene the browser buffer belongs to.
//
// The buffer in localStorage is the last graph on screen, kept so a reload is instant.
// It also records the scene it was saved for.
// If the project names that same scene, the buffer is that scene, at most two seconds newer than the file, and it stands.
// If the project names another scene and that scene has a file, the file is loaded: the buffer is some other scene's content, and saving it under the project's name would overwrite that scene's file with a scratch scene.
// If the file is missing, the buffer keeps its own scene name so it saves back where it came from, never under a name that is not its own.
// A buffer with no tag (a browser from before the tag) is trusted only when there is no file to prefer.
export function startupScene(projectActive, bufferFor, diskHas){
  if(!projectActive) return { scene: bufferFor || null, load: false };
  if(bufferFor === projectActive) return { scene: projectActive, load: false };
  if(diskHas) return { scene: projectActive, load: true };
  if(bufferFor) return { scene: bufferFor, load: false };
  return { scene: projectActive, load: false };
}

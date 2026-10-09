// Keyboard bindings.
// One table drives the editor, the viewer, the settings panel and the documentation, so a listed shortcut is always the one that runs.
// Bindings persist per browser.

const STORE = 'neuron-playground-keys-v1';

// action: { label, group, def, fixed } def is a binding string: optional "ctrl+" and "shift+" prefixes then a key name, matched case insensitively. fixed actions cannot be rebound.
export const ACTIONS = {
  addNode:      { group:'graph',    label:'add node menu',        def:'Tab' },
  frameAll:     { group:'graph',    label:'frame all',            def:'f' },
  findNode:     { group:'graph',    label:'find a node',          def:'ctrl+f' },
  deleteNode:   { group:'graph',    label:'delete selected',      def:'Delete', alt:'Backspace' },
  bypass:       { group:'graph',    label:'bypass selected',      def:'d' },
  addDot:       { group:'graph',    label:'insert reroute pin',   def:'.' },
  copy:         { group:'graph',    label:'copy',                 def:'ctrl+c' },
  paste:        { group:'graph',    label:'paste',                def:'ctrl+v' },
  clone:        { group:'graph',    label:'clone selected',       def:'ctrl+d' },
  // ctrl+u rather than the conventional ctrl+shift+g: shift is not part of a plain letter's binding string here, so ctrl+shift+g and ctrl+g are the same event to this table and one would shadow the other.
  groupSel:     { group:'graph',    label:'group selected',       def:'ctrl+g' },
  ungroup:      { group:'graph',    label:'ungroup under cursor', def:'ctrl+u' },
  undo:         { group:'graph',    label:'undo',                 def:'ctrl+z' },
  redo:         { group:'graph',    label:'redo',                 def:'ctrl+y', alt:'ctrl+shift+z' },
  cancel:       { group:'graph',    label:'cancel drag or menu',  def:'Escape', fixed:true },
  gizmoMove:    { group:'viewer',   label:'gizmo translate',      def:'w' },
  gizmoRotate:  { group:'viewer',   label:'gizmo rotate',         def:'e' },
  gizmoScale:   { group:'viewer',   label:'gizmo scale',          def:'r' },
  closeGizmo:   { group:'viewer',   label:'close gizmo',          def:'Escape', fixed:true },
  viewSlots:    { group:'viewer',   label:'bind or recall view',  def:'1 to 9', fixed:true },
};

export const GROUPS = [['graph', 'node graph'], ['viewer', 'viewer']];

let custom = {};
try { custom = JSON.parse(localStorage.getItem(STORE)) || {}; } catch(e){}

export function binding(action){
  const a = ACTIONS[action];
  if(!a) return '';
  return (!a.fixed && custom[action]) || a.def;
}
export function isCustom(action){ return !!custom[action]; }
export function setBinding(action, str){
  const a = ACTIONS[action];
  if(!a || a.fixed) return false;
  if(!str || str === a.def) delete custom[action]; else custom[action] = str;
  try { localStorage.setItem(STORE, JSON.stringify(custom)); } catch(e){}
  return true;
}
export function resetBindings(){
  custom = {};
  try { localStorage.removeItem(STORE); } catch(e){}
}

// Turn a keydown into the same string form the table uses, so a captured key and a configured one compare directly.
export function eventToBinding(e){
  if(['Control','Shift','Alt','Meta'].includes(e.key)) return '';
  let s = '';
  if(e.ctrlKey || e.metaKey) s += 'ctrl+';
  if(e.shiftKey && e.key.length > 1) s += 'shift+';
  return s + (e.key.length === 1 ? e.key.toLowerCase() : e.key);
}

// Does this event fire this action?
// Accepts the primary binding and the optional alternate, which exists for keys that differ across keyboards (Delete and Backspace) or have a second conventional form (redo).
export function matches(e, action){
  const a = ACTIONS[action];
  if(!a) return false;
  const got = eventToBinding(e);
  if(!got) return false;
  const want = binding(action);
  if(got === want.toLowerCase() || got === want) return true;
  if(a.alt && (got === a.alt.toLowerCase() || got === a.alt)) return true;
  // shift is not significant for plain letters, so ctrl+shift+z reaches redo
  if(want.startsWith('ctrl+') && (e.ctrlKey || e.metaKey)){
    const bare = want.slice(5).toLowerCase();
    if(e.key.toLowerCase() === bare && !e.shiftKey) return true;
  }
  return false;
}

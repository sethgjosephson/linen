// The app's icons: inline SVG, one pixel strokes, square corners, drawn on a 16 by 16 grid, in currentColor so they take the button's color and its hover.
// A button that carries one keeps its meaning in its title, and the ones whose meaning is not universal keep a word beside the icon.

const ICONS = {
  play:    '<path d="M4 2 L13 8 L4 14 Z" fill="currentColor" stroke="none"/>',
  pause:   '<path d="M4 2 V14 M12 2 V14" stroke-width="2.5"/>',
  lock:    '<rect x="3" y="7" width="10" height="7"/><path d="M5 7 V5 a3 3 0 0 1 6 0 V7"/>',
  unlock:  '<rect x="3" y="7" width="10" height="7"/><path d="M5 7 V5 a3 3 0 0 1 6 0"/>',
  add:     '<path d="M8 3 V13 M3 8 H13"/>',
  remove:  '<path d="M4 4 L12 12 M12 4 L4 12"/>',
  copy:    '<rect x="5" y="5" width="8" height="8"/><path d="M3 11 V3 H11"/>',
  save:    '<path d="M8 2 V10 M4.5 6.5 L8 10 L11.5 6.5"/><path d="M2 12 V14 H14 V12"/>',
  load:    '<path d="M8 10 V2 M4.5 5.5 L8 2 L11.5 5.5"/><path d="M2 12 V14 H14 V12"/>',
  folder:  '<path d="M2 4 H6 L7.5 6 H14 V13 H2 Z"/>',
  runs:    '<path d="M2 3 H14 M2 8 H14 M2 13 H14"/>',
  up:      '<path d="M8 13 V3 M4 7 L8 3 L12 7"/>',
  box:     '<rect x="2.5" y="2.5" width="11" height="11" stroke-dasharray="2 2"/>',
  slice:   '<rect x="2.5" y="2.5" width="11" height="11"/><path d="M8 1.5v13" stroke-dasharray="2 1.5"/>',
  palette: '<circle cx="5" cy="5" r="2.2"/><circle cx="11" cy="5" r="2.2"/><circle cx="5" cy="11" r="2.2"/><circle cx="11" cy="11" r="2.2"/>',
  keys:    '<rect x="1.5" y="4.5" width="13" height="7"/><path d="M4 7 H5 M7 7 H8 M10 7 H12 M4 9.5 H12"/>',
  docs:    '<path d="M3 2 H10 L13 5 V14 H3 Z M10 2 V5 H13 M5 8 H11 M5 11 H11"/>',
  close:   '<path d="M4 4 L12 12 M12 4 L4 12"/>',
  eye:     '<path d="M1.5 8 C4 3.5 12 3.5 14.5 8 C12 12.5 4 12.5 1.5 8 Z"/><circle cx="8" cy="8" r="1.8"/>',
  expand:  '<path d="M2 6 V2 H6 M10 2 H14 V6 M14 10 V14 H10 M6 14 H2 V10"/>',
  text:    '<path d="M2 3 H14 M2 6 H10 M2 9 H14 M2 12 H8"/>',
  browse:  '<path d="M2 4 H6 L7.5 6 H14 V13 H2 Z"/><path d="M8 8 V11.5 M6.5 10 L8 11.5 L9.5 10"/>',
  // spike display: a flash toward white, and a flash in color from gray
  bright:  '<circle cx="8" cy="8" r="2.5"/><path d="M8 1.5 V3.5 M8 12.5 V14.5 M1.5 8 H3.5 M12.5 8 H14.5 M3.4 3.4 L4.8 4.8 M11.2 11.2 L12.6 12.6 M3.4 12.6 L4.8 11.2 M11.2 4.8 L12.6 3.4"/>',
  sat:     '<circle cx="8" cy="8" r="5.5"/><path d="M8 2.5 A5.5 5.5 0 0 1 8 13.5 Z" fill="currentColor" stroke="none"/>',
  // which spikes: both signs, a depolarizing spike, a hyperpolarizing dip
  spikeEI: '<path d="M1 8 H4 L5.5 2 L7 8 L8.5 14 L10 8 H15"/>',
  spikeE:  '<path d="M1 12 H6 L8 3 L10 12 H15"/>',
  spikeI:  '<path d="M1 4 H6 L8 13 L10 4 H15"/>',
  // pane layouts: the outline of the window with the viewer's pane filled
  layoutRight:   '<rect x="1.5" y="2.5" width="13" height="11"/><rect x="6.5" y="2.5" width="8" height="11" fill="currentColor" stroke="none" opacity=".5"/><path d="M6.5 2.5 V13.5 M1.5 8 H6.5"/>',
  layoutClassic: '<rect x="1.5" y="2.5" width="13" height="11"/><rect x="1.5" y="2.5" width="8.5" height="6" fill="currentColor" stroke="none" opacity=".5"/><path d="M10 2.5 V13.5 M1.5 8.5 H10"/>',
  layoutLeft:    '<rect x="1.5" y="2.5" width="13" height="11"/><rect x="1.5" y="2.5" width="8" height="11" fill="currentColor" stroke="none" opacity=".5"/><path d="M9.5 2.5 V13.5 M9.5 8 H14.5"/>',
  layoutWide:    '<rect x="1.5" y="2.5" width="13" height="11"/><rect x="1.5" y="2.5" width="13" height="5.5" fill="currentColor" stroke="none" opacity=".5"/><path d="M1.5 8 H14.5 M9 8 V13.5"/>',
};

export function icon(name, cls = ''){
  const d = ICONS[name];
  if(!d) return '';
  return `<svg class="ic${cls ? ' ' + cls : ''}" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" ` +
    `fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="square">${d}</svg>`;
}

// The button's words are kept in the title, so a hover still says the whole thing.
export function iconify(el, name, text = ''){
  if(!el) return;
  const words = el.textContent.trim();
  if(!el.title && words) el.title = words.toLowerCase();
  if(words && !el.getAttribute('aria-label')) el.setAttribute('aria-label', words.toLowerCase());
  el.innerHTML = icon(name) + (text ? `<span>${text}</span>` : '');
  el.classList.add('icb');
  if(!text) el.classList.add('icon-only');
}

// The browser's own title tooltip waits about a second and cannot be styled, so the title moves into data-tip on hover and shows at once under the button, its name above the description.
// A button whose state changes sets a new title; the next show picks it up.
// The shortcut hints in the pane headers are cut off once the text is large, so hovering one lists every shortcut in it, one per line.
export function installTips(){
  const tip = document.createElement('div');
  tip.id = 'tip'; tip.hidden = true;
  document.body.appendChild(tip);
  const HINT = '.panehead .hint, #graphHint';
  let cur = null, timer = 0;
  const show = () => {
    if(!cur || !cur.isConnected || cur.classList.contains('open')){ tip.hidden = true; return; }   // open: its menu is showing, and the tip would sit over it
    tip.textContent = '';
    if(cur.matches(HINT)){
      const items = cur.textContent.split(/\s*\u00a0\s*/).map(t => t.trim()).filter(Boolean);
      if(!items.length){ tip.hidden = true; return; }
      const n = document.createElement('div'); n.className = 'tipname'; n.textContent = 'shortcuts'; tip.appendChild(n);
      for(const t of items){ const d = document.createElement('div'); d.className = 'tipline'; d.textContent = t; tip.appendChild(d); }
      return place();
    }
    if(cur.title){ cur.dataset.tip = cur.title; cur.removeAttribute('title'); }
    const name = cur.getAttribute('aria-label') || '', body = cur.dataset.tip || '';
    if(!name && !body){ tip.hidden = true; return; }
    if(name){ const n = document.createElement('div'); n.className = 'tipname'; n.textContent = name; tip.appendChild(n); }
    if(body && body !== name){ const b = document.createElement('div'); b.textContent = body; tip.appendChild(b); }
    place();
  };
  const place = () => {
    tip.hidden = false;
    const r = cur.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    const x = Math.min(Math.max(4, r.left + r.width/2 - w/2), innerWidth - w - 4);
    let y = r.bottom + 6;
    if(y + h > innerHeight - 4) y = r.top - h - 6;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  document.addEventListener('pointerover', e => {
    const el = e.target.closest ? e.target.closest('button.icon-only, ' + HINT) : null;
    if(el === cur) return;
    cur = el; clearTimeout(timer);
    if(!el){ tip.hidden = true; return; }
    timer = setTimeout(show, 120);
  });
  document.addEventListener('pointerout', e => { if(!e.relatedTarget){ cur = null; clearTimeout(timer); tip.hidden = true; } });
  document.addEventListener('click', () => { if(cur && !tip.hidden) setTimeout(show, 0); }, true);
  addEventListener('scroll', () => { tip.hidden = true; }, true);
}

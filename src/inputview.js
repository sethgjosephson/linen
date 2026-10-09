// Input monitor: shows what is actually being fed to the tissue, per input node, with a mute toggle on each row.
// Visual maps draw as the encoder grid; band maps draw as a spectrum over the tonotopic axis.
// Both show the encoded gains, which is the signal the engine receives, not the raw source, so retinal contrast coding and adaptation are visible in the picture.

let el = null, panel = null, io = null, rows = [];

// the panel carries a header and a body; the rows go in the body, the panel shows and hides as a whole
export function attachInputs(container){ panel = container; el = container.querySelector('.ovbody') || container; }

export function setupInputs(net, runtime){
  io = runtime;
  rows = [];
  if(!el) return;
  el.innerHTML = '';
  const maps = (net && net.inputMaps) || [];
  (panel || el).style.display = maps.length ? 'block' : 'none';
  for(const m of maps){
    const sheet = !!m.sheet;
    const wrap = document.createElement('div'); wrap.className = 'inrow';
    const head = document.createElement('div'); head.className = 'inhead';
    const name = document.createElement('span');
    // A curriculum presents the item in whichever sense the encoder is set to; every other source names itself. -1 is a read node, -2 a curriculum.
    const SRC = ['test signal', 'microphone', 'webcam'];
    name.textContent = (m.source === -2 ? (m.sense === 'sound' ? 'sound' : 'sight')
      : m.source === -1 ? 'file' : (SRC[m.source] || 'input')) +
      (m.code === 1 ? ' (ON/OFF contrast)' : '');
    const item = document.createElement('span'); item.className = 'initem';
    const btn = document.createElement('button');
    btn.textContent = 'on';
    btn.onclick = () => {
      const muted = io.toggleMute(m.id);
      btn.textContent = muted ? 'muted' : 'on';
      btn.classList.toggle('muted', muted);
    };
    head.append(name, item, btn);
    const cv = document.createElement('canvas');
    // retinal coding doubles the channels (ON block then OFF block), so the monitor shows the two blocks side by side
    const blocks = m.code === 1 ? 2 : 1;
    cv.width = m.cols * blocks;
    cv.height = sheet ? m.rows : 24;
    cv.className = 'incanvas';
    wrap.append(head, cv);
    el.appendChild(wrap);
    rows.push({ map:m, cv, item, sheet, blocks });
  }
}

export function updateInputs(){
  if(!io || !rows.length) return;
  for(const r of rows){
    const g = io.lastGains && io.lastGains.get(r.map.id);
    const src = io.sources && io.sources.get(r.map.id);
    r.item.textContent = src && src.inst && src.inst.item ? src.inst.item : '';
    if(!g) continue;
    const c = r.cv.getContext('2d');
    const W = r.cv.width, H = r.cv.height;
    if(r.sheet){
      const cols = r.map.cols, rowsN = r.map.rows, G = cols*rowsN;
      const im = c.createImageData(W, H);
      for(let b=0;b<r.blocks;b++)
        for(let y=0;y<rowsN;y++)
          for(let x=0;x<cols;x++){
            const v = Math.min(255, Math.round((g[b*G + y*cols + x] || 0)*255));
            // flip vertically so the picture reads the same way up as the source image did
            const o = (((rowsN-1-y))*W + b*cols + x)*4;
            im.data[o] = b ? v*0.5 : v;          // ON white, OFF dimmer/red
            im.data[o+1] = b ? v*0.2 : v;
            im.data[o+2] = b ? v*0.2 : v;
            im.data[o+3] = 255;
          }
      c.putImageData(im, 0, 0);
    } else {
      c.fillStyle = '#000'; c.fillRect(0, 0, W, H);
      const cols = r.map.cols;
      for(let b=0;b<r.blocks;b++)
        for(let x=0;x<cols;x++){
          const v = Math.min(1, g[b*cols + x] || 0);
          const h = Math.max(v > 0.004 ? 1 : 0, Math.round(v*(H-1)));
          c.fillStyle = b ? 'hsl(10,70%,45%)' : 'hsl(190,70%,60%)';
          c.fillRect(b*cols + x, H-h, 1, h);
        }
    }
  }
}

// A drawing context that writes SVG: the part of the canvas 2D interface chart.js uses, recorded as vector elements, so the functions that draw the panel also write a figure a journal or a drawing program can take.
// Pure: no DOM, so it runs under node and the chart tests read its output.
//
// Text stays text (selectable, editable, in the chosen font).
// The baseline is placed by arithmetic rather than by dominant-baseline, which several drawing programs ignore.
// Runs of filled rectangles in one color (a raster's spikes, a histogram's bars) are merged into one path, so a raster of fifty thousand spikes is one element.
// A block of pixels (a matrix) is embedded as a PNG at data resolution and scaled unsmoothed.

import { encodePng, base64 } from './png.js';

const r2 = v => { const s = (+v).toFixed(2); return s.indexOf('.') < 0 ? s : s.replace(/0+$/, '').replace(/\.$/, ''); };
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
// the average advance of a character, as a fraction of the font size
const ADVANCE = { mono:0.6, sans:0.53, serif:0.48 };

export class SvgContext {
  // w, h: the drawing's size in its own units (points); o: { widthMm, heightMm, title, desc, meta (an object written as JSON) }
  constructor(w, h, o = {}){
    this.w = w; this.h = h; this.o = o;
    this.out = []; this.defs = []; this.clipN = 0;
    this.fillStyle = '#000'; this.strokeStyle = '#000'; this.lineWidth = 1;
    this.globalAlpha = 1;                 // fills only, which is what the band needs
    this.font = '10px monospace'; this.textAlign = 'left'; this.textBaseline = 'alphabetic';
    this.imageSmoothingEnabled = false;
    this._dash = []; this._path = []; this._m = [1, 0, 0, 1, 0, 0]; this._stack = [];
    this._clip = null; this._run = null;
  }
  // ---- state ----
  save(){ this._flush(); this._stack.push({ m:this._m.slice(), clip:this._clip }); }
  restore(){ this._flush(); const s = this._stack.pop(); if(s){ this._m = s.m; this._clip = s.clip; } }
  translate(x, y){ this._flush(); const m = this._m; m[4] += m[0]*x + m[2]*y; m[5] += m[1]*x + m[3]*y; }
  rotate(a){ this._flush(); const m = this._m, c = Math.cos(a), s = Math.sin(a);
    this._m = [m[0]*c + m[2]*s, m[1]*c + m[3]*s, -m[0]*s + m[2]*c, -m[1]*s + m[3]*c, m[4], m[5]]; }
  scale(x, y){ this._flush(); const m = this._m; m[0] *= x; m[1] *= x; m[2] *= y; m[3] *= y; }
  setTransform(a, b, c, d, e, f){ this._flush(); this._m = [a, b, c, d, e, f]; }
  setLineDash(d){ this._dash = d || []; }
  // a fill's opacity, written only when it is not 1 so nothing else changes
  _op(){ return this.globalAlpha < 1 ? ` fill-opacity="${r2(this.globalAlpha)}"` : ''; }
  _attrs(){
    const m = this._m, id = m[0] === 1 && m[1] === 0 && m[2] === 0 && m[3] === 1 && m[4] === 0 && m[5] === 0;
    return (id ? '' : ` transform="matrix(${m.map(r2).join(' ')})"`) + (this._clip ? ` clip-path="url(#${this._clip})"` : '');
  }
  // ---- rectangles ----
  clearRect(){}
  fillRect(x, y, w, h){
    if(!(w > 0) || !(h > 0)) return;
    const key = this.fillStyle + this.globalAlpha + this._attrs();
    if(!this._run || this._run.key !== key){ this._flush(); this._run = { key, fill:this.fillStyle, op:this._op(), attrs:this._attrs(), d:[] }; }
    this._run.d.push(`M${r2(x)} ${r2(y)}h${r2(w)}v${r2(h)}h${r2(-w)}z`);
  }
  _flush(){
    const r = this._run; this._run = null;
    if(r) this.out.push(`<path d="${r.d.join('')}" fill="${esc(r.fill)}"${r.op || ''}${r.attrs}/>`);
  }
  strokeRect(x, y, w, h){
    this._flush();
    this.out.push(`<rect x="${r2(x)}" y="${r2(y)}" width="${r2(w)}" height="${r2(h)}" fill="none" ${this._stroke()}${this._attrs()}/>`);
  }
  _stroke(){
    return `stroke="${esc(this.strokeStyle)}" stroke-width="${r2(this.lineWidth)}"` +
      (this._dash.length ? ` stroke-dasharray="${this._dash.map(r2).join(' ')}"` : '');
  }
  // ---- paths ----
  beginPath(){ this._path = []; }
  moveTo(x, y){ this._path.push(`M${r2(x)} ${r2(y)}`); }
  lineTo(x, y){ this._path.push(`L${r2(x)} ${r2(y)}`); }
  rect(x, y, w, h){ this._path.push(`M${r2(x)} ${r2(y)}h${r2(w)}v${r2(h)}h${r2(-w)}z`); }
  closePath(){ this._path.push('z'); }
  stroke(){
    if(!this._path.length) return;
    this._flush();
    this.out.push(`<path d="${this._path.join('')}" fill="none" ${this._stroke()} stroke-linejoin="round"${this._attrs()}/>`);
  }
  fill(){
    if(!this._path.length) return;
    this._flush();
    this.out.push(`<path d="${this._path.join('')}" fill="${esc(this.fillStyle)}"${this._op()}${this._attrs()}/>`);
  }
  clip(){
    this._flush();
    const id = 'clip' + (++this.clipN);
    this.defs.push(`<clipPath id="${id}"><path d="${this._path.join('')}"/></clipPath>`);
    this._clip = id;
  }
  // ---- text ----
  _font(){
    const m = /([\d.]+)px\s+(.*)$/.exec(this.font) || [0, 10, 'monospace'];
    const family = m[2].trim();
    const kind = /mono|consolas|menlo/i.test(family) ? 'mono' : /times|serif/i.test(family) && !/sans/i.test(family) ? 'serif' : 'sans';
    return { px:+m[1], family, kind };
  }
  measureText(s){ const f = this._font(); return { width:String(s).length*f.px*ADVANCE[f.kind] }; }
  fillText(s, x, y){
    this._flush();
    const f = this._font();
    const dy = this.textBaseline === 'middle' ? f.px*0.35 : this.textBaseline === 'top' ? f.px*0.8 : 0;
    const anchor = this.textAlign === 'center' ? 'middle' : this.textAlign === 'right' ? 'end' : 'start';
    this.out.push(`<text x="${r2(x)}" y="${r2(y + dy)}" font-family="${esc(f.family)}" font-size="${r2(f.px)}" text-anchor="${anchor}" fill="${esc(this.fillStyle)}"${this._attrs()}>${esc(s)}</text>`);
  }
  // ---- pixels ----
  drawPixels(rgba, cols, rows, x, y, w, h){
    this._flush();
    const png = encodePng(rgba, cols, rows);
    this.out.push(`<image x="${r2(x)}" y="${r2(y)}" width="${r2(w)}" height="${r2(h)}" preserveAspectRatio="none" ` +
      `style="image-rendering:pixelated" href="data:image/png;base64,${base64(png)}"${this._attrs()}/>`);
  }
  // ---- the document ----
  toString(){
    this._flush();
    const o = this.o;
    const size = o.widthMm ? ` width="${r2(o.widthMm)}mm" height="${r2(o.heightMm)}mm"` : ` width="${r2(this.w)}" height="${r2(this.h)}"`;
    return `<?xml version="1.0" encoding="UTF-8"?>\n` +
      `<svg xmlns="http://www.w3.org/2000/svg"${size} viewBox="0 0 ${r2(this.w)} ${r2(this.h)}">\n` +
      (o.title ? `<title>${esc(o.title)}</title>\n` : '') +
      (o.desc ? `<desc>${esc(o.desc)}</desc>\n` : '') +
      (o.meta ? `<metadata>${esc(JSON.stringify(o.meta))}</metadata>\n` : '') +
      (this.defs.length ? `<defs>${this.defs.join('')}</defs>\n` : '') +
      this.out.join('\n') + '\n</svg>\n';
  }
}

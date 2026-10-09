// Exercises the child-stdout frame reader: frames split across many chunks, several frames inside one chunk, a header straddling a chunk boundary, and a body large enough that a quadratic path would stall.
import { makeFrameReader } from './cudaengine.mjs';

let fail = 0;
const ok = (name, cond) => { console.log((cond ? 'ok   ' : 'FAIL ') + name); if(!cond) fail++; };

const mkFrame = (cmd, body) => {
  const h = Buffer.alloc(8);
  h.writeUInt32LE(cmd, 0); h.writeUInt32LE(body.length, 4);
  return Buffer.concat([h, body]);
};

const drive = (stream, size) => {
  const got = [];
  const read = makeFrameReader((cmd, body) => got.push({ cmd, body: Buffer.from(body) }));
  for(let o = 0; o < stream.length; o += size)
    read(stream.subarray(o, Math.min(o + size, stream.length)));
  return got;
};

// one byte at a time is the worst case for header and body straddling
{
  const a = mkFrame(100, Buffer.from('alpha'));
  const b = mkFrame(101, Buffer.from('bravo-body-longer'));
  const c = mkFrame(102, Buffer.alloc(0));
  const got = drive(Buffer.concat([a, b, c]), 1);
  ok('one byte at a time: three frames', got.length === 3);
  ok('one byte at a time: cmds', got.map(g => g.cmd).join(',') === '100,101,102');
  ok('one byte at a time: bodies', got[0].body.toString() === 'alpha' &&
    got[1].body.toString() === 'bravo-body-longer' && got[2].body.length === 0);
}

{
  const s = Buffer.concat([mkFrame(1, Buffer.from('x')), mkFrame(2, Buffer.from('yy')),
    mkFrame(3, Buffer.from('zzz'))]);
  const got = drive(s, s.length);
  ok('one chunk, three frames', got.length === 3 &&
    got[2].body.toString() === 'zzz');
}

{
  const f = mkFrame(101, Buffer.from('split-header'));
  const got = drive(f, 4);
  ok('header straddling a chunk', got.length === 1 &&
    got[0].body.toString() === 'split-header');
}

// the real shape: a large body in many chunks, where a quadratic path would stall
{
  const M = 4_000_000;                       // 16 MB of floats
  const payload = Buffer.alloc(M*4);
  for(let i = 0; i < M; i++) payload.writeFloatLE(i % 1000, i*4);
  const f = mkFrame(101, payload);
  const t0 = Date.now();
  const got = drive(f, 65536);               // the pipe's chunk size
  const ms = Date.now() - t0;
  ok('large frame reassembles', got.length === 1 && got[0].body.length === M*4);
  const w = new Float32Array(got[0].body.buffer.slice(
    got[0].body.byteOffset, got[0].body.byteOffset + M*4));
  ok('large frame content exact', w[0] === 0 && w[999] === 999 &&
    w[1000] === 0 && w[M-1] === (M-1) % 1000);
  console.log(`     16 MB in ${Math.ceil(M*4/65536)} chunks took ${ms} ms`);
}

// a chunk cut inside the second body, so the tail handling is exercised rather than the clean-boundary case
{
  const s = Buffer.concat([mkFrame(100, Buffer.from('first')), mkFrame(101, Buffer.from('second'))]);
  const got = drive(s, s.length - 3);        // cut inside the second body
  ok('tail carried across chunks', got.length === 2 &&
    got[0].body.toString() === 'first' && got[1].body.toString() === 'second');
}

console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);

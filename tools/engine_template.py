"""A template engine: a leaky integrate-and-fire network that linen's node graph can drive.

    python tools/engine_template.py [port]          (default 8890)

Then set a checkpoint's engine to remote with the engine address ws://localhost:8890, or run a scene without the browser:

    node tools/linen.mjs run "guided tour" --engine remote --host ws://localhost:8890

The transport is the websockets package (its asyncio server); the arithmetic is NumPy.
It speaks the protocol in ENGINE.md: one message per binary WebSocket frame, u32 headerLen | JSON header | payloads, each payload 8-byte aligned, typed arrays in the header replaced by {"__ta": {"d": dtype, "n": length}} descriptors in depth-first order (src/frame.js).
It answers hello, takes init, tune and tick, and steps C dv/dt = -gL (v - vr) + I with a spike at vt and a reset to vr, read from each cell type's f7 row, gL = k (vt - vr) (the slope of the row at rest), one Euler step per millisecond.
That is a different model from linen's Izhikevich membrane, so its rates differ from the reference engine's; tools/enginecheck.mjs shows by how much.
The engine settings node's key lif_tau_ms, when present, sets every cell's membrane time constant instead (gL = C / tau).
"""
import asyncio, json, struct, sys
import numpy as np
from websockets.asyncio.server import serve

PROTOCOL = 3
# the terms beyond the baseline this engine implements (TERMS in src/protocol.js), and the optional queries it answers
TERMS = ['syn:1', 'psc:1', 'psc:2', 'refrac', 'vmin', 'protocols:0', 'protocols:1', 'protocols:2', 'protocols:3', 'custom']
OPTIONAL = ['watch', 'sendV']
DTYPES = {'Float32Array': np.float32, 'Float64Array': np.float64, 'Int8Array': np.int8, 'Int16Array': np.int16,
          'Int32Array': np.int32, 'Uint8Array': np.uint8, 'Uint8ClampedArray': np.uint8, 'Uint16Array': np.uint16,
          'Uint32Array': np.uint32}
NAMES = {np.dtype(t).str: n for n, t in DTYPES.items() if n != 'Uint8ClampedArray'}
ROWS = 17                                    # delays are 1 to 16 ms


def decode(buf):
    hlen = struct.unpack_from('<I', buf, 0)[0]
    off = (4 + hlen + 7) & ~7

    def walk(v):
        nonlocal off
        if isinstance(v, dict):
            ta = v.get('__ta')
            if isinstance(ta, dict) and ta.get('d') in DTYPES:
                dt = np.dtype(DTYPES[ta['d']])
                a = np.frombuffer(buf, dt, ta['n'], off)
                off = (off + ta['n'] * dt.itemsize + 7) & ~7
                return a
            return {k: walk(x) for k, x in v.items()}
        return [walk(x) for x in v] if isinstance(v, list) else v
    return walk(json.loads(buf[4:4 + hlen]))


def encode(msg):
    bufs = []

    def walk(v):
        if isinstance(v, np.ndarray):
            bufs.append(np.ascontiguousarray(v))
            return {'__ta': {'d': NAMES[v.dtype.str], 'n': int(v.size)}}
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return [walk(x) for x in v] if isinstance(v, list) else v
    head = json.dumps(walk(msg)).encode()
    out = bytearray(struct.pack('<I', len(head)) + head)
    for b in [b''] + [x.tobytes() for x in bufs]:
        out += b + bytes(-(len(out) + len(b)) % 8)
    return bytes(out)


class Refused(Exception):
    pass


def scale(p, t):
    # a stimulus protocol's factor at millisecond t (ENGINE.md section 2): pulse train, ramp, or on while it lasts
    dt = t - p['t0']
    if dt < 0 or (p['duration'] > 0 and dt >= p['duration']):
        return 0.0
    if p['mode'] == 1:
        return 1.0 if dt % p['period'] < p['width'] else 0.0
    return min(1.0, dt / p['period']) if p['mode'] == 2 else 1.0


class LIF:
    def __init__(self):
        self.n, self.watch, self.send_v = 0, -1, False

    def init(self, m):
        if m.get('protocol') != PROTOCOL:
            raise Refused('lif-template is engine contract %d and the init names %s' % (PROTOCOL, m.get('protocol')))
        n, nt = int(m['count']), np.asarray(m['ntype'], np.int64)
        row = lambda k: np.array([t['f7'][k] for t in m['types']], np.float64)[nt]
        self.n, self.C, self.vr, self.vt = n, row('C'), row('vr'), row('vt')
        self.gl0 = row('k') * (self.vt - self.vr)
        self.v = np.array(m['v0'], np.float64)
        self.pre, self.post = np.asarray(m['preStart'], np.int64), np.asarray(m['post'], np.int64)
        self.w, self.delay = np.asarray(m['w'], np.float64), np.asarray(m['delay'], np.int64)
        self.ringE, self.ringI = np.zeros((ROWS, n)), np.zeros((ROWS, n))
        self.gE, self.gI, self.hold = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
        self.rng, self.t = np.random.default_rng(int(m.get('seed') or 1)), 0
        self.tune(m)

    def tune(self, m):
        syn = int(m.get('syn') or 0)
        if syn == 2 or int(m.get('plast') or 0) or (m.get('stp') or 0) > 0:
            raise Refused('lif-template runs kick and exponential synapses only, without plasticity')
        if m.get('bias') is not None:
            self.bias = np.asarray(m['bias'], np.float64)
        self.exp, psc = syn == 1, int(m.get('psc') or 0)
        tE, tI = m.get('tauE') or 3, m.get('tauI') or 8
        self.decE, self.decI = np.exp(-1 / tE), np.exp(-1 / tI)
        # what w means under exp synapses: the peak (psc 0), the charge (1), the peak integrated over the step (2)
        self.itauE, self.itauI = [1.0 if psc == 0 else (1 - d if psc == 1 else tau * (1 - d)) for d, tau in ((self.decE, tE), (self.decI, tI))]
        self.refrac = min(20, int(m.get('refrac') or 0))
        self.vmin = float(m['vmin']) if m.get('vmin') is not None and float(m['vmin']) < 0 else -np.inf
        self.protos = [{**p, 'mode': int(p['mode']), 'cells': np.asarray(p['idx'], np.int64),
                        'each': p['amp'] * (np.asarray(p['gain'], np.float64) if p.get('gain') is not None else 1.0)}
                       for p in m.get('protocols') or []]
        self.scales = None
        tau = (m.get('custom') or {}).get('lif_tau_ms')
        self.gl = self.C / float(tau) if isinstance(tau, (int, float)) and tau > 0 else self.gl0

    def deliver(self, spiking):
        starts, lens = self.pre[spiking], self.pre[spiking + 1] - self.pre[spiking]
        if not lens.sum():
            return
        s = np.arange(lens.sum()) + np.repeat(starts - np.cumsum(lens) + lens, lens)
        slot, w = (self.t + self.delay[s]) % ROWS, self.w[s]
        np.add.at(self.ringE, (slot[w > 0], self.post[s][w > 0]), w[w > 0])
        np.add.at(self.ringI, (slot[w <= 0], self.post[s][w <= 0]), w[w <= 0])

    def tick(self, steps):
        fired, vtrace, spikes = np.zeros(self.n, np.uint8), np.zeros(steps, np.float32), 0
        for k in range(steps):
            r = self.t % ROWS
            if self.exp:
                self.gE, self.gI = self.gE * self.decE + self.ringE[r], self.gI * self.decI + self.ringI[r]
                syn = self.gE * self.itauE + self.gI * self.itauI
            else:
                syn = self.ringE[r] + self.ringI[r]
            self.ringE[r] = 0
            self.ringI[r] = 0
            scales = [scale(p, self.t) for p in self.protos]
            if scales != self.scales:
                self.scales, self.dc, self.namp = scales, np.zeros(self.n), np.zeros(self.n)
                for p, f in zip(self.protos, scales):
                    np.add.at(self.namp if p['mode'] == 3 else self.dc, p['cells'], p['each'] * f)
            noise = self.namp * (self.rng.random(self.n) + self.rng.random(self.n) - 1)
            v = self.v + (-self.gl * (self.v - self.vr) + syn + self.bias + self.dc + noise) / self.C
            v = np.maximum(v, self.vmin)
            held = self.hold > 0
            v[held] = self.vr[held]
            self.hold[held] -= 1
            spiking = np.flatnonzero((v >= self.vt) & ~held)
            v[spiking] = self.vr[spiking]
            self.hold[spiking] = self.refrac
            fired[spiking] += fired[spiking] < 255
            spikes += spiking.size
            self.v = v
            self.deliver(spiking)
            if self.watch >= 0:
                vtrace[k] = 30.0 if self.watch in spiking else v[self.watch]
            self.t += 1
        out = {'cmd': 'state', 'fired': fired, 'spikes': spikes, 'steps': steps, 'vtrace': vtrace}
        if self.send_v:
            out['v'] = self.v.astype(np.float32)
        return out


async def session(ws):
    eng = LIF()
    async for raw in ws:
        if isinstance(raw, str):
            continue
        m = decode(raw)
        cmd = m.get('cmd')
        try:
            if cmd == 'hello':
                await ws.send(encode({'cmd': 'hello', 'protocol': PROTOCOL, 'engine': 'lif-template', 'terms': TERMS, 'optional': OPTIONAL}))
            elif cmd == 'init':
                eng.init(m)
            elif cmd == 'tune' and eng.n:
                eng.tune(m)
            elif cmd == 'tick':
                if not eng.n:
                    raise Refused('lif-template: tick before init')
                await ws.send(encode(eng.tick(int(m.get('steps') or 1))))
            elif cmd == 'watch':
                eng.watch = int(m.get('idx', -1))
            elif cmd == 'sendV':
                eng.send_v = bool(m.get('on'))
            elif cmd is not None:
                await ws.send(encode({'cmd': 'unsupported', 'of': cmd}))
        except Refused as e:
            await ws.send(encode({'cmd': 'error', 'message': str(e)}))


async def main(port):
    async with serve(session, 'localhost', port, max_size=None, compression=None):
        print('lif-template engine listening on ws://localhost:%d' % port, flush=True)
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8890))

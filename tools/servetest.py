"""Does the project file API do what it says, and refuse what it should?

The confinement check is the part worth testing hardest. This is a localhost
dev server, but a path guard that can be walked out of is not a guard, and
the failure is silent: a write lands somewhere outside the project and nobody
notices until something else is overwritten.
"""
import json, os, shutil, subprocess, sys, tempfile, time
import urllib.request, urllib.error
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A port nothing else holds, asked of the OS, so a server another project left on a fixed number cannot answer in place of the one spawned here.
import socket
with socket.socket() as _s:
    _s.bind(('127.0.0.1', 0)); PORT = _s.getsockname()[1]
PROJ = tempfile.mkdtemp(prefix='nptest-')

passed = failed = 0
VERBOSE = os.environ.get('NPTEST_VERBOSE')
def ok(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        if VERBOSE:
            print('  ok   ' + name, flush=True)
    else:
        failed += 1
        print('FAIL: ' + name + ('  ' + str(detail) if detail else ''), flush=True)

def req(method, path, body=None):
    r = urllib.request.Request('http://localhost:%d%s' % (PORT, path),
                               data=body, method=method)
    try:
        with urllib.request.urlopen(r, timeout=5) as f:
            return f.status, f.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

# The server's own output is kept rather than thrown away: when it fails to start, its traceback is the only thing that says why, and discarding it turns a clear error into a probe loop that times out.
LOG = os.path.join(PROJ, 'server.log')
logf = open(LOG, 'wb')
srv = subprocess.Popen([sys.executable, '-u', os.path.join(ROOT, 'serve.py'),
                        str(PORT), '--project', PROJ],
                       stdout=logf, stderr=logf, cwd=ROOT)
try:
    up = False
    for _ in range(40):
        if srv.poll() is not None:
            break                      # it exited; the log below says why
        try:
            req('GET', '/project')
            up = True
            break
        except Exception:
            time.sleep(0.25)
    if not up:
        logf.flush()
        with open(LOG, 'rb') as f:
            print('FAIL: the server never came up on port %d. Its output:' % PORT)
            print(f.read().decode('utf-8', 'replace')[-2000:])
        raise SystemExit(1)

    code, body = req('GET', '/project')
    info = json.loads(body)
    ok('the server names its project folder', code == 200 and info['ok'])
    ok('and it is the one it was given',
       os.path.realpath(info['root']) == os.path.realpath(PROJ), info['root'])
    ok('and says it is writable from localhost', info['writable'] is True)

    payload = b'{"hello":"world"}'
    code, body = req('PUT', '/project/file/runs/t1/metrics.json', payload)
    ok('a file writes', code == 200 and json.loads(body)['ok'], body)
    ok('and lands on disk where it was asked to',
       os.path.isfile(os.path.join(PROJ, 'runs', 't1', 'metrics.json')))
    code, body = req('GET', '/project/file/runs/t1/metrics.json')
    ok('and reads back byte for byte', code == 200 and body == payload, body)

    code, body = req('GET', '/project/list/runs/t1')
    ents = json.loads(body)['entries']
    ok('a listing sees it', any(e['name'] == 'metrics.json' for e in ents), ents)
    ok('with its size', any(e['size'] == len(payload) for e in ents), ents)

    blob = bytes(range(256)) * 40
    req('PUT', '/project/file/runs/t1/ckpt.npb', blob)
    code, body = req('GET', '/project/file/runs/t1/ckpt.npb')
    ok('binary survives the round trip', body == blob, len(body))

    ok('no .part files remain',
       not any(f.endswith('.part')
               for _, _, fs in os.walk(PROJ) for f in fs))

    # The confinement, which is the point.
    # What is asserted here is the invariant, not a status code: an attempt may be refused outright, or it may be normalized down to a harmless path and written, but it must never land outside the project folder.
    # Pinning a specific code would be testing the client's URL normalization rather than the guard, since some of these forms collapse before the server ever sees them.
    root = os.path.realpath(PROJ)
    outside = os.path.join(os.path.dirname(PROJ), 'ESCAPED.txt')
    for attempt in ('../ESCAPED.txt', '..%2fESCAPED.txt', 'a/../../ESCAPED.txt',
                    '/etc/passwd', 'a/b/../../../ESCAPED.txt',
                    '..%2F..%2FESCAPED.txt', '....//ESCAPED.txt',
                    '%2e%2e%2fESCAPED.txt'):
        code, body = req('PUT', '/project/file/' + attempt, b'nope')
        if code == 200:
            landed = os.path.realpath(
                os.path.join(PROJ, json.loads(body)['path']))
            contained = landed == root or landed.startswith(root + os.sep)
            ok('contained: ' + attempt, contained, landed)
        else:
            ok('refused: ' + attempt, code in (400, 404),
               str(code) + ' ' + str(body[:60]))
    ok('and nothing escaped', not os.path.exists(outside))
    ok('and no ESCAPED.txt anywhere above the project',
       not any(f == 'ESCAPED.txt'
               for _, _, fs in os.walk(os.path.dirname(PROJ)) for f in fs))

    # Node modules: the project's nodes folder is listed and read through the same file API as scenes and brains (src/plugins.js loadProjectPlugins).
    module = b'export default function(linen){}\n'
    code, body = req('PUT', '/project/file/nodes/jitter.js', module)
    ok('a node module writes into nodes/', code == 200 and json.loads(body)['ok'], body)
    req('PUT', '/project/file/nodes/notes.txt', b'not a module')
    code, body = req('GET', '/project/list/nodes')
    ents = json.loads(body)['entries']
    ok('the nodes folder lists its files',
       sorted(e['name'] for e in ents if not e['dir']) == ['jitter.js', 'notes.txt'], ents)
    code, body = req('GET', '/project/file/nodes/jitter.js')
    ok('and a module reads back as written', code == 200 and body == module, body[:60])
    code, body = req('GET', '/project/list/nodes/missing')
    ok('a missing folder lists as empty', code == 200 and json.loads(body)['entries'] == [], body[:60])

    code, _ = req('DELETE', '/project/file/runs/t1/metrics.json')
    ok('a file deletes', code == 200 and
       not os.path.isfile(os.path.join(PROJ, 'runs', 't1', 'metrics.json')))

    code, _ = req('GET', '/project/file/runs/t1/missing.json')
    ok('a missing file is 404', code == 404, code)

    code, body = req('GET', '/index.html')
    ok('static files still serve', code == 200 and b'<' in body, code)

    # --- browsing for a folder, and adopting one -------------------------
    # The folder is chosen on the server because the browser's own picker never reveals an absolute path, and the engine host needs one.
    code, body = req('GET', '/project/browse')
    roots = json.loads(body)
    ok('browsing with no path offers somewhere to start',
       code == 200 and roots['ok'] and len(roots['dirs']) > 0, body[:120])

    code, body = req('GET', '/project/browse?path=' + quote(PROJ))
    at = json.loads(body)
    ok('browsing a real folder lists its subfolders',
       at['ok'] and isinstance(at['dirs'], list), body[:120])
    ok('and offers a way back up', bool(at['parent']), at.get('parent'))
    ok('and says the project folder is adoptable', at['adoptable'] is True)

    code, body = req('GET', '/project/browse?path=' + quote(
        os.path.join(PROJ, 'runs', 't1', 'metrics.json')))
    ok('browsing a file is refused, not guessed at',
       json.loads(body)['ok'] is False, body[:120])

    # A folder holding unrelated files is not a project.
    # This is the guard that keeps the confinement check meaningful: every read and write is confined to the project root, so pointing the root at a home directory would confine it to the whole home directory.
    busy = os.path.join(os.path.dirname(PROJ), 'nptest-busy')
    os.makedirs(busy, exist_ok=True)
    with open(os.path.join(busy, 'someones-taxes.txt'), 'w') as f:
        f.write('not a project')
    code, body = req('POST', '/project/open',
                     json.dumps({'path': busy}).encode())
    ok('a folder of unrelated files is refused', code == 400, body[:160])
    ok('and the refusal says what would be accepted',
       b'empty' in body or b'project.json' in body, body[:160])
    code, body = req('GET', '/project')
    ok('and the project folder did not move',
       os.path.realpath(json.loads(body)['root']) == os.path.realpath(PROJ))

    fresh = os.path.join(os.path.dirname(PROJ), 'nptest-fresh')
    shutil.rmtree(fresh, ignore_errors=True)
    code, body = req('POST', '/project/open',
                     json.dumps({'path': fresh}).encode())
    ok('a missing folder is reported rather than created',
       json.loads(body)['ok'] is False and json.loads(body).get('missing'),
       body[:160])
    ok('and nothing was created', not os.path.exists(fresh))

    code, body = req('POST', '/project/open',
                     json.dumps({'path': fresh, 'create': True}).encode())
    ok('asking for it explicitly creates it', code == 200 and
       json.loads(body)['ok'] and json.loads(body)['created'], body[:160])
    ok('and it is there', os.path.isdir(fresh))
    code, body = req('GET', '/project')
    ok('and the server moved to it',
       os.path.realpath(json.loads(body)['root']) == os.path.realpath(fresh),
       body[:160])

    req('PUT', '/project/file/scenes/moved.json', b'{"here":true}')
    ok('a write lands in the newly opened folder',
       os.path.isfile(os.path.join(fresh, 'scenes', 'moved.json')))
    ok('and not in the previous one',
       not os.path.isfile(os.path.join(PROJ, 'scenes', 'moved.json')))

    code, body = req('POST', '/project/open',
                     json.dumps({'path': PROJ}).encode())
    ok('a folder that is already a project is adopted back',
       code == 200 and json.loads(body)['ok'], body[:160])

    # A folder holding only a nodes folder is a project's start, not someone's unrelated files.
    nodesonly = PROJ + '-nodes'
    os.makedirs(os.path.join(nodesonly, 'nodes'), exist_ok=True)
    code, body = req('GET', '/project/browse?path=' + quote(nodesonly))
    ok('a folder with only nodes/ in it is adoptable',
       json.loads(body).get('adoptable') is True, body[:160])

    shutil.rmtree(busy, ignore_errors=True)
    shutil.rmtree(fresh, ignore_errors=True)
    shutil.rmtree(nodesonly, ignore_errors=True)
finally:
    srv.terminate()
    try:
        srv.wait(timeout=5)
    except Exception:
        srv.kill()
    shutil.rmtree(PROJ, ignore_errors=True)

print(('%d failed, %d passed' % (failed, passed)) if failed
      else 'all passed (%d checks)' % passed)
sys.exit(1 if failed else 0)

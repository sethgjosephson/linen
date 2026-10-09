#!/usr/bin/env python3
"""Dev server: static files with caching disabled. Serves the app from any cwd.

It also starts the engine host on request, so a training run does not need a
second terminal opened by hand before it will go. A page cannot spawn a
process, and this server is the only thing already running on the machine
when someone is using the app, so the capability lives here.

And it owns the project folder: the one place a run's files go.

That last part needs saying, because the obvious alternative does not work.
The browser way to let someone choose a folder is showDirectoryPicker, and it
deliberately never exposes the absolute path: you get a handle, not a
location. The CUDA host is a separate process writing hundred-megabyte
checkpoints to a real path, and a handle cannot tell it where. So the page and
the host would land in different places, which is exactly the split this is
meant to close. This server already knows real paths and already launches the
host, so the folder lives here and both write through the one location.

    python serve.py [port] [--project PATH] [--host ADDR]

Default project is ./project beside this file. It listens on 127.0.0.1
only; writes were always loopback-only, but static files, the project root
path and the directory browser were served to anything on the network,
which is fine at home and not on shared wifi. --host 0.0.0.0 exposes it on
purpose.
"""
import http.server, os, sys, json, socket, subprocess, shutil, time, posixpath
from urllib.parse import unquote
import urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
HOST_PORT = 8801
HOST_LOG = os.path.join(ROOT, 'host', 'host.log')
PROJECT = os.path.join(ROOT, 'project')


def drive_roots():
    """Somewhere to start browsing from. On Windows that is the drives, since
    there is no single root to walk down from."""
    if os.name != 'nt':
        return ['/']
    out = []
    for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        d = c + ':\\'
        if os.path.isdir(d):
            out.append(d)
    return out


def looks_like_project(path):
    """Is this folder one the server may adopt?

    Empty, or already a project. The point is not to stop anyone doing what
    they meant to: it is that the confinement guard on every read and write is
    only worth something while the project folder is a folder this app owns.
    Pointed at a drive root or a home directory, the guard still holds and
    the page read the whole disk, because everything is inside the root. So
    the root itself is what has to be chosen carefully, and a folder holding
    someone's unrelated files is not a project.
    """
    try:
        entries = os.listdir(path)
    except OSError as e:
        return False, str(e)
    if not entries:
        return True, None
    if 'project.json' in entries:
        return True, None
    if any(e in ('runs', 'scenes', 'brains', 'nodes') and
           os.path.isdir(os.path.join(path, e)) for e in entries):
        return True, None
    return False, ('this folder already holds other files. Pick an empty '
                   'folder, or one that already has a project.json in it.')


def project_path(rel):
    """Absolute path for a project-relative path, or None if it escapes.

    Confinement is checked on the resolved path rather than by looking for
    dot-dot in the text, because that is not the only way out: a symlink, a
    Windows short name or a drive-relative path all get there too, and
    realpath collapses every one of them before the comparison.
    """
    rel = unquote(rel or '').lstrip('/')
    if not rel:
        return None
    parts = [p for p in posixpath.normpath(rel).split('/') if p not in ('', '.')]
    if any(p == '..' for p in parts):
        return None
    full = os.path.realpath(os.path.join(PROJECT, *parts))
    root = os.path.realpath(PROJECT)
    if full != root and not full.startswith(root + os.sep):
        return None
    return full


def node_exe():
    """The node binary, from PATH or the vendored copy."""
    found = shutil.which('node')
    if found:
        return found
    for cand in (r'D:\Tools\node\node.exe', r'C:\Program Files\nodejs\node.exe',
                 '/usr/bin/node', '/usr/local/bin/node'):
        if os.path.exists(cand):
            return cand
    return None


def host_up(port=HOST_PORT, timeout=0.3):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex(('127.0.0.1', port)) == 0


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _local_only(self):
        """Starting a process, or writing a file, is not something a page on
        another machine gets to ask for, whatever address this is bound to."""
        if self.client_address[0] in ('127.0.0.1', '::1'):
            return True
        self._json(403, {'ok': False, 'error': 'local requests only'})
        return False

    def _confined(self, rel):
        full = project_path(rel)
        if full is None:
            self._json(400, {'ok': False,
                             'error': 'path outside the project folder'})
        return full

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/host/status':
            self._json(200, {'ok': True, 'up': host_up(), 'port': HOST_PORT,
                             'node': bool(node_exe())})
            return
        # Answered to any client: knowing the folder is not the same as being able to write into it.
        if path == '/project':
            self._json(200, {'ok': True, 'root': PROJECT,
                             'name': os.path.basename(PROJECT.rstrip(os.sep)),
                             'writable': self.client_address[0] in ('127.0.0.1', '::1')})
            return
        # Server side because the browser's own folder picker deliberately never exposes an absolute path, and an absolute path is exactly what the engine host needs.
        if path == '/project/browse':
            if not self._local_only():
                return
            q = self.path.split('?', 1)
            want = unquote(q[1][5:]) if len(q) > 1 and q[1].startswith('path=') else ''
            if not want:
                self._json(200, {'ok': True, 'path': '', 'parent': None,
                                 'dirs': drive_roots(), 'roots': True})
                return
            full = os.path.abspath(want)
            if not os.path.isdir(full):
                self._json(200, {'ok': False, 'error': 'not a folder: ' + full})
                return
            try:
                names = sorted(n for n in os.listdir(full)
                               if os.path.isdir(os.path.join(full, n))
                               and not n.startswith('.'))
            except OSError as e:
                self._json(200, {'ok': False, 'error': str(e)})
                return
            parent = os.path.dirname(full.rstrip(os.sep))
            if parent == full:
                parent = ''
            adoptable, why = looks_like_project(full)
            self._json(200, {'ok': True, 'path': full, 'parent': parent,
                             'dirs': names, 'adoptable': adoptable,
                             'why': why})
            return
        if path == '/project/list' or path.startswith('/project/list/'):
            if not self._local_only():
                return
            rel = path[len('/project/list/'):] if len(path) > len('/project/list') else ''
            full = os.path.realpath(PROJECT) if not rel else self._confined(rel)
            if full is None:
                return
            if not os.path.isdir(full):
                self._json(200, {'ok': True, 'entries': []})
                return
            out = []
            for name in sorted(os.listdir(full)):
                fp = os.path.join(full, name)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                out.append({'name': name, 'dir': os.path.isdir(fp),
                            'size': st.st_size, 'mtime': int(st.st_mtime * 1000)})
            self._json(200, {'ok': True, 'entries': out})
            return
        if path.startswith('/project/file/'):
            if not self._local_only():
                return
            full = self._confined(path[len('/project/file/'):])
            if full is None:
                return
            if not os.path.isfile(full):
                self.send_error(404)
                return
            with open(full, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_PUT(self):
        """Write one file into the project folder, creating its parents."""
        path = self.path.split('?')[0]
        if not path.startswith('/project/file/'):
            self.send_error(404)
            return
        # The body is read before anything else, including before the path is judged.
        # A refusal that leaves the request body unread stalls the client, which turns a clean 400 into a hang.
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else b''
        if not self._local_only():
            return
        full = self._confined(path[len('/project/file/'):])
        if full is None:
            return
        try:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            # through a temporary file in the same directory, so a checkpoint interrupted part way through does not leave a truncated file that reads as a complete one
            tmp = full + '.part'
            with open(tmp, 'wb') as f:
                f.write(body)
            os.replace(tmp, full)
        except Exception as e:
            self._json(500, {'ok': False, 'error': str(e)})
            return
        self._json(200, {'ok': True, 'bytes': len(body),
                         'path': os.path.relpath(full, PROJECT).replace(os.sep, '/')})

    def do_DELETE(self):
        path = self.path.split('?')[0]
        if not path.startswith('/project/file/'):
            self.send_error(404)
            return
        if not self._local_only():
            return
        full = self._confined(path[len('/project/file/'):])
        if full is None:
            return
        # Deleting something already gone is the state the caller asked for, so it succeeds rather than erroring.
        # It still says whether a file was there, since a caller cleaning up old checkpoints reports what it freed.
        try:
            existed = os.path.isfile(full)
            if existed:
                os.remove(full)
        except Exception as e:
            self._json(500, {'ok': False, 'error': str(e)})
            return
        self._json(200, {'ok': True, 'removed': existed})

    def do_POST(self):
        route = self.path.split('?')[0]
        # A scenario's saved layout (src/layouts/<slug>.json): the scene menu writes the arrangement on screen so it ships as the scenario's default.
        # Loopback only, like every write; the slug is letters, digits and dashes, so the path cannot leave the folder.
        if route == '/layout':
            if not self._local_only():
                return
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            slug = (q.get('name') or [''])[0]
            if not slug or not all(c.isalnum() or c == '-' for c in slug):
                self._json(400, {'ok': False, 'error': 'bad layout name'})
                return
            n = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(n) if n else b''
            try:
                data = json.loads(body or b'null')
                assert isinstance(data, dict) and isinstance(data.get('nodes'), list)
            except Exception:
                self._json(400, {'ok': False, 'error': 'not a layout'})
                return
            folder = os.path.join(ROOT, 'src', 'layouts')
            os.makedirs(folder, exist_ok=True)
            full = os.path.join(folder, slug + '.json')
            with open(full, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(data, f, indent=1)
                f.write('\n')
            self._json(200, {'ok': True, 'path': os.path.relpath(full, ROOT).replace(os.sep, '/')})
            return
        # Everything the page and the host write goes through PROJECT, so this is the one place that has to change; a host already running keeps the folder it was launched with, which the reply says so the UI can pass it on.
        if route == '/project/open':
            if not self._local_only():
                return
            n = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(n) if n else b'{}'
            try:
                req = json.loads(body or b'{}')
            except Exception:
                self._json(400, {'ok': False, 'error': 'bad request body'})
                return
            want = (req.get('path') or '').strip()
            if not want:
                self._json(400, {'ok': False, 'error': 'no path given'})
                return
            full = os.path.abspath(want)
            if os.path.exists(full) and not os.path.isdir(full):
                self._json(400, {'ok': False,
                                 'error': 'that is a file, not a folder'})
                return
            created = False
            if not os.path.exists(full):
                if not req.get('create'):
                    self._json(200, {'ok': False, 'missing': True,
                                     'error': 'no such folder: ' + full})
                    return
                try:
                    os.makedirs(full)
                    created = True
                except OSError as e:
                    self._json(400, {'ok': False, 'error': str(e)})
                    return
            adoptable, why = looks_like_project(full)
            if not adoptable:
                self._json(400, {'ok': False, 'error': why})
                return
            global PROJECT
            PROJECT = full
            print('project folder: ' + PROJECT)
            self._json(200, {'ok': True, 'root': PROJECT, 'created': created,
                             'name': os.path.basename(PROJECT.rstrip(os.sep)),
                             'hostRunning': host_up()})
            return
        # The machine's own folder picker, opened by this process, since a page cannot learn a path from the picker it can open itself and the path is what everything here writes to.
        # A child process runs the dialog so a dismissed or stuck one never holds the server.
        if route == '/project/pick':
            if not self._local_only():
                return
            n = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(n) if n else b'{}'
            try:
                req = json.loads(body or b'{}')
            except Exception:
                req = {}
            start = (req.get('start') or PROJECT or os.getcwd())
            title = req.get('title') or 'project folder'
            code = '; '.join([
                'import sys, tkinter, tkinter.filedialog as fd',
                'r = tkinter.Tk()', 'r.withdraw()', 'r.attributes("-topmost", True)',
                'p = fd.askdirectory(initialdir=sys.argv[1], title=sys.argv[2], mustexist=False)',
                'sys.stdout.write(p or "")'])
            try:
                out = subprocess.run([sys.executable, '-c', code, start, title],
                                     capture_output=True, text=True, timeout=600)
                path = out.stdout.strip()
            except subprocess.TimeoutExpired:
                self._json(200, {'ok': False, 'error': 'the picker was left open'})
                return
            except Exception as e:
                self._json(200, {'ok': False, 'error': 'no folder picker on this machine: ' + str(e)})
                return
            self._json(200, {'ok': True, 'path': os.path.abspath(path) if path else '',
                             'cancelled': not path})
            return
        if route != '/host/start':
            self.send_error(404)
            return
        if not self._local_only():
            return
        try:
            n = int(self.headers.get('Content-Length') or 0)
            req = json.loads(self.rfile.read(n) or b'{}') if n else {}
        except Exception:
            req = {}
        # The port is the request's to choose within reason: a host keeps the code it loaded, so a page that needs current code while an older host is still busy asks for a fresh one beside it.
        try:
            port = int(req.get('port') or HOST_PORT)
        except (TypeError, ValueError):
            port = HOST_PORT
        if not (1024 <= port <= 65535):
            self._json(200, {'ok': False, 'up': False, 'error': 'port out of range'})
            return
        if host_up(port):
            self._json(200, {'ok': True, 'up': True, 'started': False,
                             'note': 'a host was already listening'})
            return
        node = node_exe()
        if not node:
            self._json(200, {'ok': False, 'up': False,
                             'error': 'no node binary found on PATH or in D:/Tools/node'})
            return
        # The request chooses the engine and nothing else: the rest of the command is fixed here rather than taken from a page.
        args = [node, os.path.join('host', 'host.mjs'), str(port),
                '--project', PROJECT]
        if req.get('cuda'):
            args.append('--cuda')
        try:
            os.makedirs(os.path.dirname(HOST_LOG), exist_ok=True)
            logf = open(HOST_LOG, 'ab')
            logf.write(('\n=== started %s ===\n'
                        % time.strftime('%Y-%m-%d %H:%M:%S')).encode())
            logf.flush()
            kw = {}
            if os.name == 'nt':
                kw['creationflags'] = (subprocess.CREATE_NEW_PROCESS_GROUP |
                                       subprocess.DETACHED_PROCESS)
            else:
                kw['start_new_session'] = True
            # Detached on purpose: a run can outlast this server, and it already survives the page that started it.
            subprocess.Popen(args, cwd=ROOT, stdout=logf, stderr=logf,
                             stdin=subprocess.DEVNULL, **kw)
        except Exception as e:
            self._json(200, {'ok': False, 'up': False, 'error': str(e)})
            return
        # Report reachable rather than launched: the caller is about to connect, and a pid that exited is not an answer it can use.
        for _ in range(60):
            if host_up(port):
                self._json(200, {'ok': True, 'up': True, 'started': True,
                                 'cuda': bool(req.get('cuda')),
                                 'log': os.path.relpath(HOST_LOG, ROOT)})
                return
            time.sleep(0.25)
        self._json(200, {'ok': False, 'up': False,
                         'error': 'host did not come up within 15 s; see '
                                  + os.path.relpath(HOST_LOG, ROOT)})


if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    if '--project' in argv:
        i = argv.index('--project')
        PROJECT = os.path.abspath(argv[i + 1])
        del argv[i:i + 2]
    host = '127.0.0.1'
    if '--host' in argv:
        i = argv.index('--host')
        host = argv[i + 1]
        del argv[i:i + 2]
    # a positional port wins; otherwise the PORT environment variable (the preview harness assigns one), otherwise 8173
    port = int(argv[0]) if argv else int(os.environ.get('PORT') or 8173)
    os.makedirs(PROJECT, exist_ok=True)
    # flush=True: stdout is block-buffered under a pipe, so without it these two lines never reach a captured log while the request log, which is stderr, does.
    print('project folder: ' + PROJECT, flush=True)
    print('listening on ' + host + (' (loopback only)' if host == '127.0.0.1' else ''), flush=True)
    http.server.test(HandlerClass=Handler, port=port, bind=host)

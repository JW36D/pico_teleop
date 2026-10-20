"""Bounded asynchronous telemetry transport, separate from robot feedback TCP."""
import json
import queue
import socket
import socketserver
import struct
import threading
import time

HEADER = struct.Struct('!II')


def read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        chunk = stream.read(count-len(result))
        if not chunk:
            raise EOFError('source disconnected')
        result.extend(chunk)
    return bytes(result)


class Publisher:
    def __init__(self, path, source, session_id, capacity=256, metadata=None):
        self.path, self.source, self.session_id = str(path), source, session_id
        self.metadata = metadata
        self.queue = queue.Queue(capacity)
        self.dropped = self.sequence = self.peak = 0
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name='collection-publish-'+source, daemon=True)
        self.thread.start()

    def submit(self, record, blob=b''):
        self.sequence += 1
        item = (dict(record, session_id=self.session_id, source=self.source,
                     source_sequence=self.sequence), blob)
        try:
            self.queue.put_nowait(item)
            self.peak = max(self.peak, self.queue.qsize())
            return True
        except queue.Full:
            self.dropped += 1
            self.error = 'publisher_queue_overflow'
            return False

    def _send(self, sock, record, blob=b''):
        record = dict(record, transport_dropped=self.dropped, publisher_queue_peak=self.peak)
        data = json.dumps(record, ensure_ascii=False, allow_nan=False).encode()
        sock.sendall(HEADER.pack(len(data), len(blob))+data+blob)

    def _run(self):
        sock = None
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    if sock is None:
                        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        sock.settimeout(.5)
                        sock.connect(self.path)
                        self._send(sock, {'kind':'metadata', 'source':self.source,
                            'session_id':self.session_id, 'metadata':self.metadata,
                            'receive_monotonic_ns':time.monotonic_ns()})
                    try:
                        record, blob = self.queue.get(timeout=.1)
                    except queue.Empty:
                        self._send(sock, {'kind':'heartbeat', 'source':self.source,
                            'session_id':self.session_id, 'receive_monotonic_ns':time.monotonic_ns()})
                        continue
                    try:
                        self._send(sock, record, blob)
                    finally:
                        self.queue.task_done()
                except (OSError, ValueError, TypeError) as exc:
                    self.error = str(exc)
                    self.dropped += 1
                    if sock:
                        sock.close()
                    sock = None
                    if self.stopping.wait(.1):
                        break
        finally:
            if sock:
                try:
                    self._send(sock, {'kind':'source_closed','source':self.source,
                        'session_id':self.session_id,'receive_monotonic_ns':time.monotonic_ns(),
                        'last_source_sequence':self.sequence,'undelivered':self.queue.qsize()})
                except OSError as exc:
                    self.error = str(exc)
                sock.close()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            self.error = 'publisher_shutdown_timeout'
        if self.queue.qsize() or self.thread.is_alive():
            raise RuntimeError('collection publisher not drained: '+str(self.error))


class IngestServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path, accept):
        self.accept_record = accept
        self.clients = set()
        self.clients_lock = threading.Lock()
        super().__init__(str(path), IngestHandler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.shutdown()
        deadline = time.monotonic()+3
        while time.monotonic()<deadline:
            with self.clients_lock:
                if not self.clients:
                    break
            time.sleep(.005)
        self.server_close()
        self.thread.join(timeout=2)


class IngestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        with self.server.clients_lock:
            self.server.clients.add(self)
        self.request.settimeout(2)
        try:
            while True:
                size, blob_size = HEADER.unpack(read_exact(self.rfile, HEADER.size))
                if size > 256*1024 or blob_size > 4*1024*1024:
                    raise ValueError('oversize collection message')
                data = json.loads(read_exact(self.rfile, size))
                blob = read_exact(self.rfile, blob_size)
                self.server.accept_record(data, blob)
        except (EOFError, OSError, ValueError):
            return
        finally:
            with self.server.clients_lock:
                self.server.clients.discard(self)

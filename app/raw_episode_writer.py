"""Async raw episode writer with bounded pre-roll and delayed boundary commit.

All filesystem work occurs on the writer thread. A/X calls enqueue commands;
they never call a robot interface. Completion is an atomic integrity commit.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

SCHEMA = 'pico_raw_v1'
CAMERAS = ('front', 'left_wrist', 'right_wrist')


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def append_json(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')


class RawEpisodeWriter:
    def __init__(self, root, metadata, history_s=2., commit_lag_s=.5,
                 source_capacity=1024, max_pending=4):
        self.session_id = metadata.get('session_id') or str(uuid.uuid4())
        self.directory = Path(root).resolve()/'sessions'/self.session_id
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory/'episodes').mkdir()
        (self.directory/'.trash').mkdir()
        self.metadata = dict(metadata, session_id=self.session_id, schema=SCHEMA,
                             closed=False, started_wall_time_ns=time.time_ns())
        atomic_json(self.directory/'session.json', self.metadata)
        self.commands = queue.Queue(128)
        self.sources = queue.Queue(source_capacity)
        self.history_ns, self.lag_ns = int(history_s*1e9), int(commit_lag_s*1e9)
        if self.history_ns <= self.lag_ns or self.lag_ns <= 0:
            raise ValueError('history must exceed positive commit lag')
        self.max_pending = max_pending
        self.pending = set()
        self.registered = set()
        self.deleted = set()
        self.lock = threading.Lock()
        self.dropped = self.queue_peak = 0
        self.error = None
        self.losses = collections.deque(maxlen=256)
        self.ready = True
        self.active = {}
        self.history = collections.defaultdict(collections.deque)
        self.last_source_time = {}
        self.last_source_sequence = {}
        self.boundary_watermark_ns = None
        self.notices = queue.Queue(128)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name='raw-episode-writer', daemon=True)
        self.thread.start()

    def _command(self, command):
        try:
            self.commands.put_nowait(command)
            return True
        except queue.Full:
            self.error = 'writer_event_queue_overflow'
            self.ready = False
            return False

    def begin(self, event, number, task):
        with self.lock:
            if not self.ready or len(self.pending) >= self.max_pending:
                return None
            episode_id = str(uuid.uuid4())
            self.pending.add(episode_id)
            self.registered.add(episode_id)
        if not self._command(('begin', episode_id, event, number, task)):
            with self.lock:
                self.pending.discard(episode_id)
            return None
        return episode_id

    def end(self, episode_id, event, reason='A_END', interrupted=False, failed=False):
        return self._command(('end', episode_id, event, reason, interrupted, failed))

    def delete(self, episode_id, event):
        # Only registered UUIDs from this writer may be passed to this method.
        if episode_id not in self.registered:
            return False
        return self._command(('delete', episode_id, event))

    def event(self, event):
        return self._command(('event', event))

    def update_metadata(self, source, data):
        return self._command(('metadata', source, data))

    def advance_boundary_watermark(self, stamp):
        return self._command(('watermark', stamp)) if type(stamp) is int else True

    def submit(self, record, blob=b''):
        try:
            self.sources.put_nowait((record, blob))
            self.queue_peak = max(self.queue_peak, self.sources.qsize())
            return True
        except queue.Full:
            self.dropped += 1
            self.losses.append({'source':record.get('source'),'source_sequence':record.get('source_sequence'),
                'receive_monotonic_ns':record.get('receive_monotonic_ns'),'reason':'writer_source_queue_overflow'})
            self.error = 'writer_source_queue_overflow'
            self.ready = False
            return False

    def fail(self, reason, source=None):
        self.ready = False
        self.error = reason
        return self._command(('failure', reason, source))

    def _notice(self, data):
        try:
            self.notices.put_nowait(data)
        except queue.Full:
            self.ready = False
            self.error = 'writer_notice_overflow'

    def _open_episode(self, episode_id, event, number, task):
        start = event['receive_monotonic_ns']
        if time.monotonic_ns()-start > self.history_ns:
            raise ValueError('start event older than pre-roll buffer')
        path = self.directory/'episodes'/episode_id
        path.mkdir()
        (path/'frames').mkdir()
        for camera in CAMERAS:
            (path/'images'/camera).mkdir(parents=True)
        info = {'schema':SCHEMA, 'session_id':self.session_id, 'episode_id':episode_id,
                'episode_number':number, 'start_ns':start, 'end_ns':None,
                'start_event':event, 'end_event':None, 'task':task,
                'mode':self.metadata['mode'], 'retained':True, 'deleted':False,
                'task_success':None, 'closed':False, 'termination':None, 'integrity':'pending'}
        ctx = {'path':path, 'info':info, 'streams':{}, 'last':{}, 'stats':{},
               'errors':[], 'end_arrival':None, 'dropped_baseline':self.dropped}
        initial = {}
        for source in ('left', 'right'):
            before = [r for r, _ in self.history[source] if r['receive_monotonic_ns'] < start]
            initial[source] = before[-1] if before else None
        atomic_json(path/'episode.json', info)
        atomic_json(path/'initial_state.json', {'boundary_ns':start, 'sides':initial,
            'semantics':'latest available telemetry at or before start; original source times retained'})
        self.active[episode_id] = ctx
        # The main writer pass flushes only after ALL queued boundary commands.

    def _write(self, ctx, record, blob):
        source, stamp = record['source'], record['receive_monotonic_ns']
        info = ctx['info']
        if stamp < info['start_ns'] or info['end_ns'] is not None and stamp >= info['end_ns']:
            return
        seq = record.get('source_sequence', record.get('event_sequence'))
        if source in ctx['last'] and seq <= ctx['last'][source][0]:
            return
        stats = ctx['stats'].setdefault(source, {'submitted':0, 'written':0, 'sequence_gaps':[],
            'frame_number_gaps':[], 'duplicates':0, 'write_errors':0, 'last_persisted':None})
        stats['submitted'] += 1
        if source in ctx['last']:
            prev_seq, prev_stamp, prev_frame = ctx['last'][source]
            if seq != prev_seq+1:
                stats['sequence_gaps'].append([prev_seq, seq])
            if stamp < prev_stamp:
                ctx['errors'].append(source+':time_regression')
            if source in CAMERAS:
                frame = record.get('frame_number')
                if frame == prev_frame:
                    stats['duplicates'] += 1
                elif frame != prev_frame+1:
                    stats['frame_number_gaps'].append([prev_frame, frame])
        relative = 'pico_events.jsonl' if source == 'pico_events' else source+'.jsonl'
        row = dict(record)
        if source in CAMERAS:
            relative = 'frames/'+source+'.jsonl'
            name = '%012d_%012d.jpg' % (record['frame_number'], seq)
            image_path = Path('images')/source/name
            try:
                if not blob:
                    raise ValueError('empty JPEG')
                with (ctx['path']/image_path).open('xb') as image:
                    image.write(blob)
                row.update(image_path=str(image_path), write_ok=True,
                    sha256=hashlib.sha256(blob).hexdigest(), size_bytes=len(blob))
            except Exception as exc:
                row.update(image_path=str(image_path), write_ok=False, write_error=str(exc))
                ctx['errors'].append(source+':image_write_failure')
                stats['write_errors'] += 1
        if relative not in ctx['streams']:
            ctx['streams'][relative] = (ctx['path']/relative).open('a')
        append_json(ctx['streams'][relative], row)
        stats['written'] += 1
        stats['last_persisted'] = {'source_sequence':seq, 'receive_monotonic_ns':stamp}
        ctx['last'][source] = (seq, stamp, record.get('frame_number'))

    def _finish(self, episode_id, ctx):
        for stream in ctx['streams'].values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        info = ctx['info']
        if self.error:
            ctx['errors'].append(self.error)
        for source in self.metadata.get('required_sources', []):
            if self.last_source_time.get(source, -1) < info['end_ns']:
                ctx['errors'].append(source+':end_watermark_not_observed')
        # Empty/short episodes are raw facts, not an I/O error. Training eligibility
        # and required-source presence are checked separately by the exporter.
        if self.dropped > ctx['dropped_baseline']:
            ctx['errors'].append('writer_source_queue_overflow')
        for source, stats in ctx['stats'].items():
            if stats['sequence_gaps'] or stats['frame_number_gaps'] or stats['duplicates']:
                ctx['errors'].append(source+':sequence_discontinuity')
        complete = not ctx['errors'] and info['termination'] == 'A_END'
        info.update(closed=True, integrity='complete' if complete else 'incomplete')
        integrity = {'schema':SCHEMA, 'closed':True, 'normal_close':info['termination']=='A_END',
            'complete':complete, 'sources':ctx['stats'], 'errors':sorted(set(ctx['errors'])),
            'writer_queue_peak':self.queue_peak, 'writer_dropped':self.dropped,
            'located_losses':[r for r in self.losses if type(r.get('receive_monotonic_ns')) is int
                              and info['start_ns'] <= r['receive_monotonic_ns'] < info['end_ns']],
            'boundary_semantics':'[start_ns,end_ns); host monotonic receive times',
            'completed_wall_time_ns':time.time_ns()}
        atomic_json(ctx['path']/'episode.json', info)
        atomic_json(ctx['path']/'integrity.json', integrity)  # authoritative final commit
        del self.active[episode_id]
        with self.lock:
            self.pending.discard(episode_id)
        if info['deleted']:
            self._trash(episode_id)
        self._notice({'event':'episode_closed', 'episode_id':episode_id, 'complete':complete})

    def _trash(self, episode_id):
        if str(uuid.UUID(episode_id)) != episode_id:
            raise ValueError('invalid registered UUID')
        path = self.directory/'episodes'/episode_id
        if path.is_symlink() or path.resolve().parent != (self.directory/'episodes').resolve():
            raise ValueError('delete target escapes episode directory')
        if path.exists():
            info = json.loads((path/'episode.json').read_text())
            info.update(deleted=True, retained=False)
            atomic_json(path/'episode.json', info)
            os.rename(str(path), str(self.directory/'.trash'/episode_id))
        self.deleted.add(episode_id)

    def _handle(self, command, events):
        kind, *args = command
        if kind == 'event':
            append_json(events, args[0]); events.flush()
        elif kind == 'watermark':
            self.boundary_watermark_ns = max(self.boundary_watermark_ns or 0,args[0])
        elif kind == 'metadata':
            self.metadata.setdefault('sources', {})[args[0]] = args[1]
            atomic_json(self.directory/'session.json', self.metadata)
        elif kind == 'failure':
            for ctx in self.active.values():
                ctx['errors'].append(str(args))
        elif kind == 'begin':
            self._open_episode(*args)
        elif kind == 'end':
            episode_id, event, reason, interrupted, failed = args
            ctx = self.active[episode_id]
            end = event['receive_monotonic_ns']
            if end < ctx['info']['start_ns']:
                ctx['errors'].append('end_before_start')
            if any(value[1] >= end for value in ctx['last'].values()):
                ctx['errors'].append('end_event_arrived_after_commit_lag')
            ctx['info'].update(end_ns=end, end_event=event, termination=reason,
                               interrupted=interrupted, failed=failed)
            ctx['end_arrival'] = time.monotonic_ns()
            atomic_json(ctx['path']/'episode.json', ctx['info'])
        elif kind == 'delete':
            episode_id, event = args
            if episode_id in self.active:
                self.active[episode_id]['info'].update(deleted=True, retained=False)
            elif (self.directory/'episodes'/episode_id/'integrity.json').is_file():
                self._trash(episode_id)
            else:
                raise ValueError('episode not registered or not logically closed')
            append_json(events, dict(event, event='delete_requested', episode_id=episode_id))
            events.flush()

    def _run(self):
        events = (self.directory/'events.jsonl').open('a')
        try:
            while not self.stopping.is_set() or not self.sources.empty() or not self.commands.empty():
                # Source ingestion precedes boundaries so initial_state sees the
                # available pre-roll. Each pass has a bounded work budget.
                for _ in range(128):
                    try:
                        record, blob = self.sources.get_nowait()
                    except queue.Empty:
                        break
                    source, stamp = record['source'], record['receive_monotonic_ns']
                    sequence = record['source_sequence']
                    if source in self.last_source_sequence and sequence <= self.last_source_sequence[source]:
                        self.fail('source_sequence_regression',source)
                    self.last_source_sequence[source] = sequence
                    if source in self.last_source_time and stamp < self.last_source_time[source]:
                        self.fail('source_time_regression', source)
                    self.last_source_time[source] = stamp
                    history = self.history[source]
                    history.append((record, blob))
                    self.sources.task_done()
                for _ in range(128):
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    self._handle(command, events)
                    self.commands.task_done()
                now = time.monotonic_ns()
                cutoff = now-self.lag_ns
                for ctx in list(self.active.values()):
                    # A receiver watermark is enqueued AFTER its complete button
                    # batch. No future command can introduce an earlier boundary.
                    limit = ctx['info']['end_ns']
                    if limit is None:
                        limit = cutoff
                        if self.metadata.get('boundary_watermarks'):
                            limit = min(limit,self.boundary_watermark_ns or 0)
                    for source, history in self.history.items():
                        for record, blob in history:
                            if record['receive_monotonic_ns'] <= limit:
                                self._write(ctx, record, blob)
                for source, history in self.history.items():
                    while history and (history[0][0]['receive_monotonic_ns'] < now-self.history_ns or len(history) > 512):
                        history.popleft()
                for episode_id, ctx in list(self.active.items()):
                    if ctx['end_arrival'] and now-ctx['end_arrival'] >= self.lag_ns:
                        end = ctx['info']['end_ns']
                        drained = all(self.last_source_time.get(s,-1) >= end
                                      for s in self.metadata.get('required_sources', []))
                        # Per-source FIFO watermarks cover IPC/encoding delays.
                        # A timeout produces an explicitly incomplete episode,
                        # never a premature complete marker.
                        if drained or now-ctx['end_arrival'] >= self.history_ns:
                            self._finish(episode_id, ctx)
                self.stopping.wait(.005)
            # Stop is requested only after robot stop/cleanup and publisher drain.
            for episode_id, ctx in list(self.active.items()):
                if ctx['info']['end_ns'] is None:
                    ctx['info'].update(end_ns=time.monotonic_ns(), termination='PROGRAM_EXIT', interrupted=True)
                for history in self.history.values():
                    for record, blob in history:
                        self._write(ctx, record, blob)
                self._finish(episode_id, ctx)
            self.metadata.update(closed=True, closed_wall_time_ns=time.time_ns())
            atomic_json(self.directory/'session.json', self.metadata)
        except Exception as exc:
            self.error = type(exc).__name__+': '+str(exc)
            self.ready = False
            self._notice({'event':'writer_failed', 'error':self.error})
            # No final complete marker is written after any unhandled I/O error.
        finally:
            for ctx in self.active.values():
                for stream in ctx['streams'].values():
                    try:
                        stream.close()
                    except OSError:
                        pass
            events.close()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('writer shutdown timeout; incomplete records retained')
        if self.error:
            raise RuntimeError(self.error)

"""Bounded diagnostic output; filesystem and console work never run in tick()."""
import collections
import json
import os
from pathlib import Path
import sys
import threading
import time

from incremental import MappingError


class AsyncDiagnosticLogger:
    def __init__(self, log_path, status_path, critical_queue_size=256, console=None,
                 writer=None):
        if critical_queue_size < 1:
            raise ValueError('critical_queue_size must be positive')
        self.log_path, self.status_path = Path(log_path), Path(status_path)
        self.file = writer if writer is not None else self.log_path.open('a', buffering=65536)
        self.console = sys.stdout if console is None else console
        try:
            console_fileno = self.console.fileno()
        except (AttributeError, OSError, ValueError):
            console_fileno = None
        self.console_fd = os.dup(console_fileno) if console_fileno is not None else None
        self.console_encoding = getattr(self.console, 'encoding', None) or 'utf-8'
        self.condition = threading.Condition()
        self.critical = collections.deque()
        self.capacity = critical_queue_size
        self.pending = None
        self.console_pending = None
        self.closing = False
        self.console_closing = False
        self.error = None
        self.console_error = None
        self.submitted = self.written = self.coalesced = self.console_coalesced = 0
        self.critical_overflow = 0
        self.thread = threading.Thread(target=self._run, name='teleop-diagnostic-writer', daemon=True)
        self.console_thread = threading.Thread(target=self._run_console,
                                               name='teleop-console-writer', daemon=True)
        self.thread.start()
        self.console_thread.start()

    def check_error(self):
        with self.condition:
            error = self.error
        if error:
            raise MappingError('Diagnostic logging failed: ' + error)

    def append(self, event, critical=False, loop_start_ns=None):
        # Callers provide a finalized, detached report; serialization is background work.
        enqueue_start = time.monotonic_ns()
        with self.condition:
            if self.error:
                raise MappingError('Diagnostic logging failed: ' + self.error)
            if self.closing:
                raise MappingError('Diagnostic logging is closed')
            if critical and len(self.critical) >= self.capacity:
                self.critical_overflow += 1
                raise MappingError('Critical diagnostic queue full; event was not silently dropped')
            self.submitted += 1
            item = (self.submitted, event)
            if critical:
                self.critical.append(item)
            else:
                if self.pending is not None:
                    self.coalesced += 1
                self.pending = item
            if 'stage_timing_ms' in event:
                now_ns = time.monotonic_ns()
                event['stage_timing_ms']['diagnostic_enqueue'] = round((now_ns-enqueue_start)/1e6, 3)
                if loop_start_ns is not None:
                    event['stage_timing_ms']['loop'] = round((now_ns-loop_start_ns)/1e6, 3)
            self.condition.notify_all()

    def stats(self):
        with self.condition:
            return {'submitted_events': self.submitted, 'written_events': self.written,
                    'coalesced_regular_events': self.coalesced,
                    'pending_critical_events': len(self.critical),
                    'pending_regular_event': self.pending is not None,
                    'critical_queue_overflow': self.critical_overflow,
                    'console_coalesced_summaries': self.console_coalesced,
                    'error': self.error, 'console_error': self.console_error}

    def console_message(self, message):
        # A slow SSH terminal can only delay this independent daemon thread.
        with self.condition:
            if self.console_pending is not None:
                self.console_coalesced += 1
            self.console_pending = message
            self.condition.notify_all()

    @staticmethod
    def summary(event):
        mapping = event.get('mapping') or {}
        label = event.get('event') or mapping.get('state', 'STATUS')
        parts = ['[%s]' % event.get('mode', 'TELEOP'), str(label)]
        if event.get('arm') in ('left','right'):
            parts.insert(1,'左臂' if event['arm']=='left' else '右臂')
        hint = {
            'CONTROL_GAP_PAUSED':'短暂控制延迟，保持 Grip 等待恢复',
            'COMMAND_GAP_PAUSED':'短暂发送延迟，保持 Grip 等待恢复',
            'CONTROL_COMPUTATION_GAP_PAUSED':'短暂计算延迟，保持 Grip 等待恢复',
            'WAIT_ROBOT_STATIONARY':'等待机械臂停稳，保持 Grip',
            'IK_REFUSED':'目标求解失败，可往回移动',
            'JOINT_SESSION_BOUND':'超出启动时的软件关节范围，请往回移动',
            'BOUNDARY_HOLD':'当前方向到可达边界，保持 Grip 可往回或沿有效方向移动',
            'BOUNDARY_LIMITED':'部分方向受限，继续跟随可执行方向',
            'WAIT_FRESH_INPUT':'手柄输入失效，请恢复连接并松开 Grip 重新接管',
            'CONTROL_GAP':'控制长时间中断，请松开 Grip 重新接管',
            'COMMAND_GAP':'发送长时间中断，请松开 Grip 重新接管',
            'CONTROL_COMPUTATION_GAP':'控制计算长时间中断，请松开 Grip 重新接管',
            'RESUMED':'已恢复',
        }.get(label)
        if hint:
            parts.append(hint)
        if event.get('control_ticks') is not None:
            parts.append('tick=%s' % event['control_ticks'])
        if event.get('motion_command_count') is not None:
            parts.append('commands=%s' % event['motion_command_count'])
        error = event.get('error') or mapping.get('reason')
        if error:
            parts.append(str(error).replace('\n', ' ')[:240])
        return ' '.join(parts)

    def _next(self):
        with self.condition:
            while not self.critical and self.pending is None and not self.closing:
                self.condition.wait()
            if self.pending is not None and (not self.critical or self.pending[0] < self.critical[0][0]):
                item, self.pending = self.pending, None
                return item[1]
            if self.critical:
                return self.critical.popleft()[1]
            return None

    def _write(self, event):
        data = json.dumps(event, allow_nan=False, separators=(',', ':'))
        self.file.write(data + '\n')
        # Publish the latest complete JSON snapshot with one atomic replacement.
        temporary = self.status_path.with_name(self.status_path.name + '.tmp')
        temporary.write_text(data + '\n')
        temporary.replace(self.status_path)
        # Flush each diagnostic event in this background thread, so filesystem
        # errors are observed promptly rather than deferred to session shutdown.
        self.file.flush()

    def _run(self):
        try:
            while True:
                event = self._next()
                if event is None:
                    break
                self._write(event)
                with self.condition:
                    self.written += 1
                self.console_message(self.summary(event))
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
            self.console_message('Diagnostic logging failed: ' + str(exc))
        finally:
            try:
                self.file.close()
            except Exception as exc:
                with self.condition:
                    self.error = self.error or str(exc)

    def _run_console(self):
        try:
            while True:
                with self.condition:
                    while self.console_pending is None and not self.console_closing:
                        self.condition.wait()
                    if self.console_pending is None:
                        return
                    message, self.console_pending = self.console_pending, None
                if self.console_fd is not None:
                    # A daemon may block in this raw syscall under SSH backpressure,
                    # but cannot hold Python's stdout buffered lock during exit.
                    data = (message + '\n').encode(self.console_encoding, errors='backslashreplace')
                    while data:
                        written = os.write(self.console_fd, data)
                        if written <= 0:
                            raise OSError('Console descriptor accepted zero bytes')
                        data = data[written:]
                else:
                    # In-memory/custom test streams have no OS descriptor.
                    self.console.write(message + '\n')
                    self.console.flush()
        except Exception as exc:
            with self.condition:
                self.console_error = str(exc)
        finally:
            if self.console_fd is not None:
                os.close(self.console_fd)

    def close(self, timeout=5.):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout=max(0., timeout))
        with self.condition:
            if self.thread.is_alive():
                self.error = self.error or 'Diagnostic writer did not finish within shutdown timeout'
        result = self.stats()
        result['complete'] = (not result['error'] and not result['critical_queue_overflow'] and
                              self.written + self.coalesced == self.submitted)
        if not result['complete']:
            self.console_message('Diagnostic output incomplete: ' + str(result))
        with self.condition:
            self.console_closing = True
            self.condition.notify_all()
        return result

"""Read-only TCP feedback, with state sampling separate from configuration checks.

No robot settings or motion commands are sent. Receive timestamps use the local
monotonic clock: the JSON state response does not supply a robot sample clock.
Protocol: https://develop.realman-robotics.com/robot/json/armState/
"""
import copy
import json
import socket
import threading
import time

from incremental import MappingError, vector

QUERIES = {'get_current_arm_state': 'current_arm_state',
           'get_current_work_frame': 'current_work_frame',
           'get_current_tool_frame': 'current_tool_frame',
           'get_arm_power_state': 'arm_power_state'}


class ReadOnlyJSONConnection:
    """One thread owns this connection; outstanding queries are never pipelined."""
    def __init__(self, host, port=8080, connect=socket.create_connection):
        self.host, self.port, self.connect = host, port, connect
        self.connection = None
        self.buffer = b''

    def ask(self, command, timeout=.15):
        if command not in QUERIES:
            raise MappingError('Not a read-only feedback query')
        requested_ns = time.monotonic_ns()
        deadline = time.monotonic()+timeout
        try:
            if self.connection is None:
                self.connection = self.connect((self.host, self.port), timeout=timeout)
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(max(.001, deadline-time.monotonic()))
            self.connection.sendall((json.dumps({'command': command})+'\r\n').encode())
            total = len(self.buffer)
            while True:
                while b'\n' in self.buffer:
                    line, self.buffer = self.buffer.split(b'\n', 1)
                    if not line.strip():
                        continue
                    try:
                        response = json.loads(line)
                    except (ValueError, UnicodeError) as exc:
                        raise MappingError('Malformed robot feedback JSON: '+str(exc))
                    if not isinstance(response, dict):
                        raise MappingError('Robot feedback response must be an object')
                    if (response.get('state') == QUERIES[command] or
                            response.get('command') in (command, QUERIES[command])):
                        return {**response, '_request_monotonic_ns':requested_ns,
                                '_receive_monotonic_ns':time.monotonic_ns()}
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Feedback deadline exceeded')
                self.connection.settimeout(max(.001, remaining))
                part = self.connection.recv(4096)
                if not part:
                    raise ConnectionError('Feedback connection closed')
                total += len(part)
                if total > 65536:
                    raise MappingError('Feedback exceeds size limit')
                self.buffer += part
        except BaseException:
            # A late response from a timed-out request must never become a new
            # sample with a fresh timestamp after reconnecting.
            self.close()
            raise

    def close(self):
        connection, self.connection = self.connection, None
        self.buffer = b''
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass


def query(host, command, timeout=.15):
    """Compatibility helper for occasional one-off read-only queries."""
    connection = ReadOnlyJSONConnection(host)
    try:
        return connection.ask(command, timeout)
    finally:
        connection.close()


def decode_state(response):
    state = response.get('arm_state')
    if not isinstance(state, dict):
        raise MappingError('Missing arm state')
    for key in ('arm_err', 'sys_err'):
        if type(state.get(key)) is not int or state[key] != 0:
            raise MappingError('Nonzero or missing ' + key)
    joints = [x/1000. for x in vector(state.get('joint'), 6)]
    raw_pose = vector(state.get('pose'), 6)
    return {'joints':joints, 'position_m':[x/1e6 for x in raw_pose[:3]],
            'orientation_rad':[x/1000 for x in raw_pose[3:]],
            'request_monotonic_ns':response.get('_request_monotonic_ns'),
            'receive_monotonic_ns':response.get('_receive_monotonic_ns', time.monotonic_ns()),
            'source':'tcp_query_response', 'robot_sample_timestamp_ns':None,
            'sample_time_semantics':'local_receive_time_robot_clock_unavailable'}


def frame_signature(work, tool):
    # Integer protocol values avoid floating-point comparison jitter.
    result = []
    for response, name in ((work, 'frame_name'), (tool, 'tool_name')):
        if not isinstance(response.get(name), str):
            raise MappingError('Missing coordinate frame name')
        p = vector(response.get('pose'), 6)
        result.append((response[name], tuple(p)))
    result.append((tool.get('payload'), tuple(tool.get('position', []))))
    return tuple(result)


class FeedbackMonitor:
    def __init__(self, host, config, query_fn=query, process_check=None):
        self.host, self.config, self.query_fn = host, config, query_fn
        self.process_check = process_check
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.checks_ready = threading.Event()
        self.current = self.previous = self.frames = self.error = None
        self.sequence = 0
        self.transport_errors = self.transport_recoveries = 0
        self.transport_error = None
        self.check_receive_ns = None
        self.check_transport_error = None
        self.check_transport_errors = self.check_transport_recoveries = 0
        self.check_recovered = False
        self.state_connection = ReadOnlyJSONConnection(host) if query_fn is query else None
        self.check_connection = ReadOnlyJSONConnection(host) if query_fn is query else None
        self.thread = threading.Thread(target=self.run, name='readonly-arm-feedback', daemon=True)
        self.check_thread = threading.Thread(target=self.run_checks, name='readonly-arm-config-checks', daemon=True)

    def ask(self, command, checks=False):
        connection = self.check_connection if checks else self.state_connection
        if connection is not None:
            return connection.ask(command, self.config['feedback_timeout_s'])
        return self.query_fn(self.host, command, self.config['feedback_timeout_s'])

    def fail(self, exc):
        with self.lock:
            self.error = self.error or str(exc)
        self.ready.set()
        self.checks_ready.set()
        self.stop_event.set()

    def start(self):
        self.check_thread.start()
        self.thread.start()
        if not self.ready.wait(2):
            self.close()
            raise MappingError('Feedback monitor did not initialize')
        return self.snapshot()

    def run_checks(self):
        initialized_ns = time.monotonic_ns()
        interval = self.config.get('feedback_checks_interval_s',1.)
        deadline_s = interval+self.config['max_feedback_age_ms']/1000.
        try:
            while not self.stop_event.is_set():
                try:
                    power = self.ask('get_arm_power_state', checks=True)
                    if type(power.get('power_state')) is not int or power['power_state'] != 1:
                        raise MappingError('Arm must already be powered; no automatic enabling')
                    frames = frame_signature(self.ask('get_current_work_frame', checks=True),
                                             self.ask('get_current_tool_frame', checks=True))
                    if self.frames is not None and frames != self.frames:
                        raise MappingError('Robot work/tool frame changed')
                except (OSError,TimeoutError) as exc:
                    with self.lock:
                        self.check_transport_errors += 1
                        self.check_transport_error = str(exc)
                        last_checked_ns = self.check_receive_ns or initialized_ns
                    if (time.monotonic_ns()-last_checked_ns)/1e9 > deadline_s:
                        raise MappingError('Robot configuration feedback transport deadline exceeded: '+str(exc))
                    self.stop_event.wait(1/self.config['feedback_hz'])
                    continue
                # Process conflicts and process-reader errors are distinct
                # from a robot TCP read timeout and must not be retried here.
                if self.process_check:
                    self.process_check()
                with self.lock:
                    self.frames = frames
                    self.check_receive_ns = time.monotonic_ns()
                    self.check_recovered = self.check_transport_error is not None
                    if self.check_recovered:
                        self.check_transport_recoveries += 1
                    self.check_transport_error = None
                self.checks_ready.set()
                # Checks cannot block the state sampler or command thread.
                self.stop_event.wait(interval)
        except Exception as exc:
            self.fail(exc)
        finally:
            if self.check_connection:
                self.check_connection.close()

    def publish(self, state):
        with self.lock:
            previous = self.current
            self.sequence += 1
            state['sequence'] = self.sequence
            state['target_feedback_hz'] = self.config['feedback_hz']
            state['joint_velocity_deg_s'] = None
            state['velocity_interval_s'] = None
            if previous is not None:
                dt = (state['receive_monotonic_ns']-previous['receive_monotonic_ns'])/1e9
                if dt <= 0:
                    raise MappingError('Robot feedback receive time did not advance')
                state['velocity_interval_s'] = dt
                state['joint_velocity_deg_s'] = [(a-b)/dt for a,b in zip(state['joints'],previous['joints'])]
            recovered = self.transport_error is not None
            if recovered:
                self.transport_recoveries += 1
            state['transport_recovered_this_sample'] = recovered
            self.transport_error = None
            self.previous, self.current = previous, state

    def run(self):
        try:
            if not self.checks_ready.wait(2):
                raise MappingError('Feedback configuration checks did not initialize')
            period = 1/self.config['feedback_hz']
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    state = decode_state(self.ask('get_current_arm_state'))
                except (OSError, TimeoutError) as exc:
                    with self.lock:
                        self.transport_errors += 1
                        self.transport_error = str(exc)
                        last_received = self.current['receive_monotonic_ns'] if self.current else None
                    if (last_received is not None and
                            (time.monotonic_ns()-last_received)/1e6 > self.config['max_feedback_age_ms']):
                        raise MappingError('Robot feedback transport deadline exceeded: '+str(exc))
                else:
                    self.publish(state)
                    self.ready.set()
                self.stop_event.wait(max(0., period-(time.monotonic()-started)))
        except Exception as exc:
            self.fail(exc)
        finally:
            if self.state_connection:
                self.state_connection.close()

    def snapshot(self):
        with self.lock:
            current = copy.deepcopy(self.current)
            error = self.error
            transport = {'transient_error':self.transport_error,
                         'error_count':self.transport_errors,
                         'recovery_count':self.transport_recoveries}
            checked_ns = self.check_receive_ns
            check_transport = {'transient_error':self.check_transport_error,
                'error_count':self.check_transport_errors,
                'recovery_count':self.check_transport_recoveries,
                'recovered_this_check':self.check_recovered}
        if error:
            raise MappingError('Feedback monitor: ' + error)
        if current is None:
            raise MappingError('No robot feedback')
        age = (time.monotonic_ns()-current['receive_monotonic_ns'])/1e6
        if not 0 <= age <= self.config['max_feedback_age_ms']:
            raise MappingError('Robot feedback is stale: %.1f ms' % age)
        check_age = (time.monotonic_ns()-checked_ns)/1e6 if checked_ns is not None else None
        check_deadline = self.config.get('feedback_checks_interval_s',1.)*1000+self.config['max_feedback_age_ms']
        if check_age is not None and not 0 <= check_age <= check_deadline:
            raise MappingError('Robot configuration feedback is stale: %.1f ms' % check_age)
        velocity = current['joint_velocity_deg_s']
        # Legacy stationary decisions retain their nominal-period tolerance;
        # the velocity estimate itself always uses measured receive intervals.
        stationary = velocity is not None and max(abs(v) for v in velocity) <= (
            self.config['stationary_tolerance_deg']*self.config['feedback_hz'])
        return {**current, 'age_ms':age, 'stationary':stationary,
                'transport':transport, 'configuration_checked_monotonic_ns':checked_ns,
                'configuration_check_interval_s':self.config.get('feedback_checks_interval_s',1.),
                'configuration_check_age_ms':check_age,
                'configuration_check_deadline_ms':check_deadline,
                'configuration_transport':check_transport}

    def close(self):
        self.stop_event.set()
        for thread in (self.thread, self.check_thread):
            if thread.is_alive():
                thread.join(timeout=2)
        # Normally each owning thread closes its socket in finally.
        for connection in (self.state_connection, self.check_connection):
            if connection:
                connection.close()

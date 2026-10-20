"""INSPIRE gripper through RealMan's built-in end-tool protocol.

No Modbus registers, power changes, persistent limits or arm SDK are used.
Only this worker owns its TCP connection; the arm loop only updates a mailbox.
"""
import copy
import json
import math
import socket
import threading
import time

from incremental import MappingError, input_health


GRIPPER_DEFAULTS = {'enabled': False, 'driver': 'realman_inspire',
                    'speed': 1000, 'force': 500,
                    'trigger_close': .6, 'trigger_open': .4,
                    'control_mode': 'binary', 'command_hz': 50.,
                    'feedback_hz': 20., 'position_deadband': 2}


def gripper_config(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value)-set(GRIPPER_DEFAULTS):
        raise MappingError('Invalid gripper configuration keys')
    result = {**GRIPPER_DEFAULTS, **value}
    if type(result['enabled']) is not bool or result['driver'] != 'realman_inspire':
        raise MappingError('Gripper requires enabled boolean and realman_inspire driver')
    if result['control_mode'] not in ('binary', 'proportional'):
        raise MappingError('gripper.control_mode must be binary or proportional')
    for key in ('command_hz', 'feedback_hz'):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]) or result[key] <= 0:
            raise MappingError('gripper.'+key+' must be finite and positive')
    if type(result['position_deadband']) is not int or not 1 <= result['position_deadband'] <= 999:
        raise MappingError('gripper.position_deadband must be 1..999 controller units')
    for key, lower in (('speed', 1), ('force', 50)):
        if type(result[key]) is not int or not lower <= result[key] <= 1000:
            raise MappingError('gripper.%s must be %d..1000' % (key, lower))
    lo, hi = result['trigger_open'], result['trigger_close']
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (lo, hi)) or not 0 <= lo < hi <= 1:
        raise MappingError('Gripper trigger thresholds require 0 <= open < close <= 1')
    return result


def action_request(action, config):
    if type(action) is int and 1 <= action <= 1000:
        # This API has no speed/force parameters. Do not invent JSON fields
        # or precede each target with a transient fully-closing pick command.
        return {'command': 'set_gripper_position', 'position': action, 'block': False}
    if action == 'open':
        return {'command': 'set_gripper_release', 'speed': config['speed'], 'block': False}
    if action == 'close':
        return {'command': 'set_gripper_pick', 'speed': config['speed'],
                'force': config['force'], 'block': False}
    raise MappingError('Unknown gripper action')


def trigger_position(trigger):
    """Absolute trigger closure fraction; the position API's minimum is 1."""
    return max(1, min(1000, round(1000*(1-trigger))))


class GripperConnection:
    """One outstanding request; an ambiguous/timed-out write is never retried."""
    REPLIES = {'get_tool_voltage': 'tool_voltage_state',
               'get_tool_RS485_mode': 'get_tool_RS485_mode',
               'get_gripper_state': 'get_gripper_state',
               'set_gripper_release': 'set_gripper', 'set_gripper_pick': 'set_gripper',
               'set_gripper_position': 'set_gripper'}

    def __init__(self, host, execute=False, connect=socket.create_connection):
        self.host, self.execute, self.connect = host, execute, connect
        self.socket = None
        self.buffer = b''

    def ask(self, request, timeout=.3, before_send=None):
        command = request.get('command')
        if command not in self.REPLIES:
            raise MappingError('Unsupported gripper request')
        if command.startswith('set_') and not self.execute:
            raise MappingError('Gripper motion prohibited in preview')
        expected = self.REPLIES[command]
        requested_ns = time.monotonic_ns()
        deadline = time.monotonic()+timeout
        try:
            if self.socket is None:
                self.socket = self.connect((self.host, 8080), timeout=timeout)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.settimeout(max(.001, deadline-time.monotonic()))
            if before_send is not None and not before_send():
                return None
            self.socket.sendall((json.dumps(request)+'\r\n').encode())
            total = len(self.buffer)
            while True:
                while b'\n' in self.buffer:
                    line, self.buffer = self.buffer.split(b'\n', 1)
                    if not line.strip():
                        continue
                    response = json.loads(line)
                    if not isinstance(response, dict):
                        raise MappingError('Invalid gripper response')
                    if response.get('command') == expected or response.get('state') == expected:
                        return {**response, '_request_monotonic_ns': requested_ns,
                                '_receive_monotonic_ns': time.monotonic_ns()}
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Gripper response timeout; action outcome may be unknown')
                self.socket.settimeout(remaining)
                part = self.socket.recv(4096)
                if not part:
                    raise ConnectionError('Gripper TCP connection closed')
                total += len(part)
                if total > 65536:
                    raise MappingError('Oversized gripper response')
                self.buffer += part
        except BaseException:
            self.close()
            raise

    def close(self):
        connection, self.socket = self.socket, None
        self.buffer = b''
        if connection is not None:
            connection.close()


def decode_feedback(raw):
    enabled = raw.get('enable', raw.get('enable_state'))
    integers = ('status', 'error', 'mode', 'actpos', 'current_force', 'temperature')
    valid = (all(type(raw.get(k)) is int for k in integers) and
             type(enabled) is int and enabled == 1 and raw['status'] == 1 and
             raw['error'] == 0 and 1 <= raw['mode'] <= 6)
    # The target uses nominal 0..1000 units, but real feedback can report 1001
    # at the open endpoint. Preserve measured values instead of clipping them
    # or mistaking a target-domain bound for a feedback validity requirement.
    return {'valid': bool(valid), 'opening_controller_units': raw['actpos'] if valid else None,
            'force_g': raw['current_force'] if valid else None,
            'temperature_c': raw['temperature'] if valid else None,
            'raw': raw, 'receive_monotonic_ns': raw.get('_receive_monotonic_ns'),
            'request_monotonic_ns': raw.get('_request_monotonic_ns'),
            'source': 'realman_get_gripper_state', 'robot_sample_timestamp_ns': None,
            'time_semantics': 'local_TCP_receive_time_not_device_sample_time'}


def inspect_port(connection):
    power = connection.ask({'command': 'get_tool_voltage'})
    mode = connection.ask({'command': 'get_tool_RS485_mode'})
    feedback = decode_feedback(connection.ask({'command': 'get_gripper_state'}))
    return {'power': power, 'serial': mode, 'feedback': feedback}


def check_port(port):
    if port['power'].get('voltage_type') != 3:
        raise MappingError('Tool supply is not 24V; no automatic power change')
    if port['serial'].get('tool_RS485_mode') != 0:
        raise MappingError('Built-in gripper conflicts with tool Modbus/other serial mode')


class GripperWorker:
    def __init__(self, host, side, config, execute=False, max_input_age_ms=100.,
                 notify=None, connection=None):
        self.config = gripper_config(config)
        self.side, self.execute, self.max_age_ms = side, execute, max_input_age_ms
        self.connection = connection or GripperConnection(host, execute)
        self.notify = notify or (lambda message: None)
        self.lock = threading.Lock()
        self.wake, self.stopped = threading.Event(), threading.Event()
        self.was_active = False
        self.intent = self.pending = None
        self.valid_until_ns = 0
        self.generation = 0
        self.state = {'enabled': True, 'arm': side, 'driver': self.config['driver'],
                      'control_mode': self.config['control_mode'],
                      'command_hz': self.config['command_hz'], 'feedback_hz': self.config['feedback_hz'],
                      'mode': 'EXECUTE' if execute else 'PREVIEW_NO_MOTION',
                      'status': 'CONNECTING', 'fault': None, 'feedback': None,
                      'last_attempt': None, 'last_accepted_command': None,
                      'accepted_command_count': 0, 'speed': self.config['speed'],
                      'force': self.config['force'],
                      'speed_force_source': ('not_set_by_position_command' if self.config['control_mode']=='proportional'
                                             else 'explicit_command_parameters')}
        self.thread = threading.Thread(target=self.run, name=side+'-gripper', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self, status, active, now_ns):
        health = input_health(status, now_ns, self.max_age_ms, require_head=False, side=self.side)
        status = status if isinstance(status, dict) else {}
        ctrl = (status.get('latest') or {}).get(self.side) or {}
        trigger, grip = ctrl.get('trigger'), ctrl.get('grip')
        valid = (active and health['ready'] and not status.get('session_exit_requested', False) and
                 not getattr(self, 'stop_requested', lambda: False)() and
                 type(grip) in (int, float) and .2 < grip <= 1 and
                 type(trigger) in (int, float) and math.isfinite(trigger) and 0 <= trigger <= 1)
        with self.lock:
            self.state['raw_trigger'] = trigger if type(trigger) in (int, float) and math.isfinite(trigger) else None
            if not valid or self.stopped.is_set():
                self.was_active = False
                self.pending = None
                self.valid_until_ns = 0
            else:
                self.valid_until_ns = now_ns+int((self.max_age_ms-health['input_age_ms'])*1e6)
                if self.config['control_mode'] == 'proportional':
                    new_intent = trigger_position(trigger)
                    changed = (type(self.intent) is not int or
                               abs(new_intent-self.intent) >= self.config['position_deadband'] or
                               new_intent in (1, 1000) and new_intent != self.intent)
                else:
                    new_intent = ('close' if trigger >= self.config['trigger_close'] else
                                  'open' if trigger <= self.config['trigger_open'] else self.intent)
                    changed = new_intent is not None and new_intent != self.intent
                # A new clutch captures input without making the gripper move.
                if self.was_active and changed and not self.state['fault']:
                    self.generation += 1
                    self.pending = (self.generation, new_intent)
                    self.wake.set()
                if changed or not self.was_active:
                    self.intent = new_intent
                self.was_active = True
            self.state['trigger_intent'] = self.intent
            self.state['target_position_controller_units'] = self.intent if type(self.intent) is int else None
        return self.snapshot(now_ns)

    def snapshot(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        with self.lock:
            result = copy.deepcopy(self.state)
            result['pending_action'] = self.pending[1] if self.pending else None
        feedback = result['feedback']
        if feedback:
            stamp = feedback.get('receive_monotonic_ns')
            feedback['age_ms'] = (now_ns-stamp)/1e6 if type(stamp) is int else None
            feedback['valid'] = bool(feedback['valid'] and feedback['age_ms'] is not None and 0 <= feedback['age_ms'] <= 1000.)
        return result

    def fail(self, exc):
        with self.lock:
            self.state.update(status='FAULT', fault=str(exc))
            self.pending = None
        try:
            self.notify(self.side+'夹爪不可用，机械臂遥操继续：'+str(exc))
        except Exception:
            pass

    def can_send(self, generation, action):
        with self.lock:
            return (not self.stopped.is_set() and not getattr(self, 'stop_requested', lambda: False)() and self.was_active and
                    self.generation == generation and self.intent == action and
                    time.monotonic_ns() <= self.valid_until_ns)

    def run(self):
        try:
            port = inspect_port(self.connection)
            check_port(port)
            with self.lock:
                self.state.update(initial_port=port, feedback=port['feedback'],
                                  status='ONLINE' if port['feedback']['valid'] else 'OFFLINE_UNVERIFIED')
            if not port['feedback']['valid']:
                self.notify(self.side+('夹爪反馈未在线；首次主动扳机动作将尝试内置因时接口，未验证成功会停用夹爪。'
                    if self.execute else '夹爪反馈未在线；预览只读，不会发送开合指令。'))
            next_poll = 0.
            next_send = 0.
            verify_deadline = None
            while not self.stopped.is_set():
                self.wake.clear()
                with self.lock:
                    pending = self.pending
                    if self.valid_until_ns < time.monotonic_ns():
                        self.pending = pending = None
                    # While verifying the first action, keep only the latest intent.
                    if verify_deadline is not None:
                        pending = None
                    if time.monotonic() < next_send:
                        pending = None
                    if pending is not None:
                        self.pending = None
                        request = action_request(pending[1], self.config)
                        attempt = {'action': 'position' if type(pending[1]) is int else pending[1],
                                   'target_position_controller_units': pending[1] if type(pending[1]) is int else None,
                                   'request': request,
                                   'send_monotonic_ns': time.monotonic_ns(), 'outcome': 'UNKNOWN'}
                        if self.execute:
                            self.state['last_attempt'] = attempt
                        else:
                            self.state['preview_request'] = request
                if pending is not None and self.execute and not self.stopped.is_set():
                    # No retries, including if the TCP acknowledgement is lost.
                    ack = self.connection.ask(request,
                        before_send=lambda: self.can_send(*pending))
                    if ack is None:
                        with self.lock:
                            self.state['last_attempt'] = {**attempt, 'outcome': 'CANCELLED_BEFORE_SEND'}
                        continue
                    if ack.get('state') is not True:
                        raise MappingError('Controller rejected gripper command: '+str(ack))
                    with self.lock:
                        attempt = {**attempt, 'outcome': 'CONTROLLER_ACCEPTED',
                                   'ack_monotonic_ns': ack['_receive_monotonic_ns']}
                        self.state['last_attempt'] = attempt
                        self.state['last_accepted_command'] = attempt
                        self.state['accepted_command_count'] += 1
                        if not self.state['feedback'] or not self.state['feedback']['valid']:
                            verify_deadline = time.monotonic()+2.
                            next_poll = 0.
                    next_send = attempt['send_monotonic_ns']/1e9+1/self.config['command_hz']
                if time.monotonic() >= next_poll:
                    feedback = decode_feedback(self.connection.ask({'command': 'get_gripper_state'}))
                    with self.lock:
                        self.state['feedback'] = feedback
                        self.state['status'] = 'ONLINE' if feedback['valid'] else 'OFFLINE_UNVERIFIED'
                    if feedback['valid']:
                        verify_deadline = None
                    elif verify_deadline is not None and time.monotonic() >= verify_deadline:
                        raise MappingError('Command accepted but gripper feedback remains invalid; model/wiring/protocol needs confirmation')
                    if feedback['raw'].get('error'):
                        raise MappingError('Gripper reports error: '+str(feedback['raw']['error']))
                    next_poll = time.monotonic()+1/self.config['feedback_hz']
                deadline = next_poll
                with self.lock:
                    if self.pending is not None and verify_deadline is None:
                        deadline = min(deadline, next_send)
                self.wake.wait(max(0., deadline-time.monotonic()))
        except Exception as exc:
            self.fail(exc)
        finally:
            self.connection.close()

    def cancel(self):
        self.stopped.set()
        with self.lock:
            self.pending = None
            self.valid_until_ns = 0
        self.wake.set()

    def close(self):
        self.cancel()
        if self.thread.ident is not None:
            self.thread.join(timeout=1.5)
        if self.thread.is_alive():
            raise MappingError('Gripper worker did not stop; in-flight command outcome unknown')
        # Never release, disable or remove power here: an object may be held.

"""Offline gripper protocol, input cancellation and arm-loop isolation checks."""
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gripper_control import (GripperConnection, GripperWorker, decode_feedback,
                             gripper_config, action_request, trigger_position)
from incremental import MappingError


def sample(side='right', trigger=0., grip=1., focus=True):
    return {'server_connected': True, 'device_connected': True, 'latest': {
        'sequence': 1, 'receive_monotonic_ns': time.monotonic_ns(),
        'input_mode': 1, 'app_focus': focus,
        side: {'grip': grip, 'trigger': trigger, 'pose': [0, 0, 0, 0, 0, 0, 1]}}}


def feedback(online=True):
    return {'command': 'get_gripper_state', 'enable': int(online), 'status': int(online),
            'error': 0, 'mode': 3 if online else 0, 'actpos': 600 if online else 0,
            'temperature': 25 if online else 0, 'current_force': 0,
            '_receive_monotonic_ns': time.monotonic_ns()}


def wait_for(predicate, timeout=1.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.002)
    raise AssertionError('Worker did not reach expected state')


class Transport:
    def __init__(self, block_poll=False, write_error=False, online=True):
        self.calls, self.writes = [], []
        self.write_times = []
        self.polling, self.unblock = threading.Event(), threading.Event()
        self.block_poll, self.write_error, self.online = block_poll, write_error, online
        self.poll_count = 0
        self.closed = False

    def ask(self, request, **kwargs):
        command = request['command']
        if kwargs.get('before_send') and not kwargs['before_send']():
            return None
        self.calls.append(request.copy())
        if command == 'get_tool_voltage':
            return {'voltage_type': 3}
        if command == 'get_tool_RS485_mode':
            return {'tool_RS485_mode': 0, 'baudrate': 115200}
        if command == 'get_gripper_state':
            self.poll_count += 1
            if self.poll_count >= 2 and self.block_poll:
                self.polling.set()
                if not self.unblock.wait(1.):
                    raise TimeoutError('test blocked read')
            return feedback(self.online)
        self.writes.append(request.copy())
        self.write_times.append(time.monotonic())
        if self.write_error:
            raise TimeoutError('write acknowledgement lost')
        self.online = True  # Simulate built-in driver's first-use initialization.
        return {'command': 'set_gripper', 'state': True,
                '_receive_monotonic_ns': time.monotonic_ns()}

    def close(self):
        self.closed = True


class GripperTests(unittest.TestCase):
    def worker(self, side='right', execute=True, transport=None, config=None):
        transport = transport or Transport()
        result = GripperWorker('unused', side, gripper_config(config), execute,
                               connection=transport)
        self.addCleanup(result.close)
        return result, transport

    def update(self, worker, trigger=0., **kwargs):
        return worker.update(sample(worker.side, trigger, **kwargs), True, time.monotonic_ns())

    def test_hysteresis_no_startup_action_regrip_or_cross_arm_input(self):
        for side in ('left', 'right'):
            worker, transport = self.worker(side)
            self.update(worker, 1.)  # Holding trigger on initial clutch is not a command.
            self.assertIsNone(worker.pending)
            self.update(worker, .5)
            self.assertIsNone(worker.pending)
            self.update(worker, .2)
            self.assertEqual(worker.pending[1], 'open')
            self.update(worker, .5)
            self.assertEqual(worker.pending[1], 'open')
            self.update(worker, .9)
            self.assertEqual(worker.pending[1], 'close')  # One latest target, no queue.
            self.update(worker, .9, grip=0.)
            self.assertIsNone(worker.pending)
            self.update(worker, 0.)
            self.assertIsNone(worker.pending)
            other = 'left' if side == 'right' else 'right'
            worker.update(sample(other, 1.), True, time.monotonic_ns())
            self.assertIsNone(worker.pending)
            self.assertEqual(transport.calls, [])

    def test_input_loss_and_expiration_cancel_commands_at_send_gate(self):
        for reason in ('focus', 'stale', 'grip', 'stop'):
            with self.subTest(reason=reason):
                worker, _ = self.worker()
                self.update(worker, 0.)
                self.update(worker, 1.)
                generation, action = worker.pending
                self.assertTrue(worker.can_send(generation, action))
                if reason == 'stop':
                    worker.cancel()
                elif reason == 'stale':
                    worker.valid_until_ns = time.monotonic_ns()-1
                else:
                    self.update(worker, 1., **({'focus': False} if reason == 'focus' else {'grip': 0.}))
                self.assertFalse(worker.can_send(generation, action))

    def test_blocked_gripper_io_does_not_block_updates_or_send_old_intent(self):
        transport = Transport(block_poll=True)
        worker, _ = self.worker(transport=transport)
        self.addCleanup(transport.unblock.set)
        worker.start()
        self.assertTrue(transport.polling.wait(1.))
        started = time.monotonic()
        self.update(worker, 0.)
        self.update(worker, 1.)
        self.update(worker, 1., grip=0.)
        self.assertLess(time.monotonic()-started, .05)
        transport.unblock.set()
        wait_for(lambda: worker.snapshot()['feedback'] is not None)
        worker.close()
        self.assertEqual(transport.writes, [])

    def test_first_deliberate_action_can_initialize_builtin_driver(self):
        worker, transport = self.worker(transport=Transport(online=False))
        worker.start()
        wait_for(lambda: worker.snapshot()['status'] == 'OFFLINE_UNVERIFIED')
        self.assertEqual(transport.writes, [])
        self.update(worker, 0.)
        self.update(worker, 1.)
        wait_for(lambda: worker.snapshot()['accepted_command_count'] == 1)
        wait_for(lambda: worker.snapshot()['feedback']['valid'])
        self.assertEqual(transport.writes[0], {'command': 'set_gripper_pick',
                                              'speed': 1000, 'force': 500, 'block': False})
        worker.close()
        self.assertEqual(len(transport.writes), 1)  # Shutdown never opens a held object.

    def test_preview_never_writes_and_unknown_write_outcome_is_not_retried(self):
        preview, transport = self.worker(execute=False)
        preview.start()
        wait_for(lambda: preview.snapshot()['status'] == 'ONLINE')
        self.update(preview, 0.)
        self.update(preview, 1.)
        wait_for(lambda: preview.snapshot().get('preview_request') is not None)
        self.assertEqual(transport.writes, [])
        self.assertIsNone(preview.snapshot()['last_attempt'])
        worker, transport = self.worker(transport=Transport(write_error=True))
        worker.start()
        wait_for(lambda: worker.snapshot()['status'] == 'ONLINE')
        self.update(worker, 0.)
        self.update(worker, 1.)
        wait_for(lambda: worker.snapshot()['fault'] is not None)
        self.update(worker, 0.)
        self.update(worker, 1.)
        self.assertEqual(len(transport.writes), 1)
        self.assertEqual(worker.snapshot()['last_attempt']['outcome'], 'UNKNOWN')
        self.assertEqual(worker.snapshot()['accepted_command_count'], 0)

    def test_offline_zero_is_not_actual_closed_position_or_training_label(self):
        decoded = decode_feedback(feedback(False))
        self.assertFalse(decoded['valid'])
        self.assertIsNone(decoded['opening_controller_units'])
        raw = feedback(True)
        self.assertEqual(decode_feedback(raw)['opening_controller_units'], 600)
        raw['actpos'] = 1001
        raw['current_force'] = -16
        self.assertTrue(decode_feedback(raw)['valid'])
        self.assertEqual(decode_feedback(raw)['opening_controller_units'], 1001)
        self.assertEqual(decode_feedback(raw)['force_g'], -16)
        raw['error'] = 4
        self.assertIsNone(decode_feedback(raw)['opening_controller_units'])

    def test_proportional_mapping_deadband_accumulates_and_regrip_does_not_jump(self):
        self.assertEqual([trigger_position(x) for x in (0., .25, .3, .5, .75, 1.)],
                         [1000, 750, 700, 500, 250, 1])
        worker, _ = self.worker(config={'control_mode':'proportional'})
        self.update(worker, 0.)
        self.assertIsNone(worker.pending)
        self.update(worker, .001)
        self.assertIsNone(worker.pending)
        self.update(worker, .002)
        self.assertEqual(worker.pending[1], 998)
        self.update(worker, .5)
        self.assertEqual(worker.pending[1], 500)
        self.assertEqual(action_request(500, worker.config),
                         {'command':'set_gripper_position','position':500,'block':False})
        self.update(worker, .5, grip=0.)
        self.update(worker, .25)
        self.assertIsNone(worker.pending)
        self.update(worker, .75)
        self.assertEqual(worker.pending[1], 250)

    def test_position_stream_keeps_latest_target_without_waiting_for_arrival(self):
        worker, transport = self.worker(config={'control_mode':'proportional','command_hz':20.})
        worker.start()
        wait_for(lambda: worker.snapshot()['status']=='ONLINE')
        self.update(worker, 0.)
        self.update(worker, .25)
        wait_for(lambda: len(transport.writes)==1)
        for trigger in (.5, .75, .8):
            self.update(worker, trigger)
        wait_for(lambda: len(transport.writes)==2)
        self.assertEqual([r['position'] for r in transport.writes], [750, 200])
        self.assertGreaterEqual(transport.write_times[1]-transport.write_times[0], .045)
        # Actual feedback remains at 600: arrival is never a prerequisite.
        self.assertEqual(worker.snapshot()['feedback']['opening_controller_units'], 600)
        self.assertEqual(worker.snapshot()['speed_force_source'], 'not_set_by_position_command')


class JSONTransportTests(unittest.TestCase):
    def test_shared_ack_name_unrelated_packet_and_fragmented_response(self):
        sock = mock.Mock()
        sock.recv.side_effect = [b'{"state":"current_trajectory_state"}\r\n{"command":"set_',
                                 b'gripper","state":true}\r\n']
        client = GripperConnection('unused', True, connect=mock.Mock(return_value=sock))
        self.addCleanup(client.close)
        reply = client.ask({'command': 'set_gripper_release', 'speed': 1000, 'block': False})
        self.assertIs(reply['state'], True)
        self.assertEqual(sock.sendall.call_count, 1)

    def test_preview_guard_cancel_and_timeout_never_duplicate_write(self):
        connect = mock.Mock()
        client = GripperConnection('unused', False, connect)
        for command in ('set_gripper_release', 'set_gripper_pick', 'set_gripper_position'):
            with self.assertRaises(MappingError):
                client.ask({'command': command})
        connect.assert_not_called()
        sock = mock.Mock()
        connect.return_value = sock
        client = GripperConnection('unused', True, connect)
        self.addCleanup(client.close)
        self.assertIsNone(client.ask({'command': 'set_gripper_release'}, before_send=lambda: False))
        sock.sendall.assert_not_called()
        sock.recv.side_effect = socket.timeout('late ack')
        with self.assertRaises(socket.timeout):
            client.ask({'command': 'set_gripper_release'})
        self.assertEqual(sock.sendall.call_count, 1)
        self.assertIsNone(client.socket)
        sock.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

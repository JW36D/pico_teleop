import json
from pathlib import Path
import socket
import socketserver
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from incremental import MappingError
from robot_feedback import FeedbackMonitor, ReadOnlyJSONConnection, decode_state


CONFIG = {'feedback_hz':100., 'feedback_timeout_s':.03, 'max_feedback_age_ms':250.,
          'stationary_tolerance_deg':.05, 'feedback_checks_interval_s':.02}


def reply(command, joints=None):
    if command == 'get_current_arm_state':
        return {'state':'current_arm_state', 'arm_state':{'joint':joints or [0]*6,
                   'pose':[0]*6, 'arm_err':0, 'sys_err':0}}
    if command == 'get_arm_power_state':
        return {'state':'arm_power_state', 'power_state':1}
    if command == 'get_current_work_frame':
        return {'state':'current_work_frame', 'frame_name':'Base', 'pose':[0]*6}
    return {'state':'current_tool_frame', 'tool_name':'Arm_Tip', 'pose':[0]*6,
            'payload':0, 'position':[0]*3}


class LocalServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ReadOnlyConnectionTests(unittest.TestCase):
    def server(self, response):
        owner = self
        self.connections = self.requests = 0
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                owner.connections += 1
                for line in self.rfile:
                    owner.requests += 1
                    request = json.loads(line)
                    data = response(request, owner.requests)
                    if isinstance(data, dict):
                        data = json.dumps(data).encode()+b'\r\n'
                    self.wfile.write(data)
                    self.wfile.flush()
        server = LocalServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_repeated_queries_reuse_connection_and_keep_unique_local_timestamps(self):
        port = self.server(lambda request, index:reply(request['command'], [index]*6))
        connection = ReadOnlyJSONConnection('127.0.0.1', port)
        self.addCleanup(connection.close)
        samples = [decode_state(connection.ask('get_current_arm_state')) for _ in range(4)]
        self.assertEqual(self.connections, 1)
        self.assertEqual(self.requests, 4)
        self.assertEqual([sample['joints'][0] for sample in samples], [.001,.002,.003,.004])
        self.assertTrue(all(a['receive_monotonic_ns'] < b['receive_monotonic_ns']
                            for a,b in zip(samples, samples[1:])))
        self.assertTrue(all(sample['robot_sample_timestamp_ns'] is None for sample in samples))
        self.assertTrue(all(sample['request_monotonic_ns'] <= sample['receive_monotonic_ns'] for sample in samples))

    def test_malformed_feedback_is_fault_not_relabelled_as_transport_retry(self):
        port = self.server(lambda request, index:b'{bad json}\r\n')
        connection = ReadOnlyJSONConnection('127.0.0.1', port)
        self.addCleanup(connection.close)
        with self.assertRaisesRegex(MappingError, 'Malformed'):
            connection.ask('get_current_arm_state')
        self.assertIsNone(connection.connection)

    def test_only_allowlisted_queries_can_open_a_connection(self):
        connection = ReadOnlyJSONConnection('invalid.example')
        with self.assertRaisesRegex(MappingError, 'read-only'):
            connection.ask('set_arm_power')
        self.assertIsNone(connection.connection)


class FeedbackQualityTests(unittest.TestCase):
    def monitor(self, query_fn=None, process_check=None, **config):
        monitor = FeedbackMonitor('localhost', {**CONFIG, **config},
            query_fn=query_fn or (lambda host, command, timeout:reply(command)), process_check=process_check)
        self.addCleanup(monitor.close)
        return monitor

    def wait_error(self, monitor):
        deadline = time.monotonic()+1.
        while not monitor.error and time.monotonic() < deadline:
            time.sleep(.001)
        self.assertIsNotNone(monitor.error)

    def test_velocity_uses_actual_response_interval_not_control_or_nominal_frequency(self):
        monitor = self.monitor()
        first = decode_state(reply('get_current_arm_state', [1000]*6))
        first['receive_monotonic_ns'] = 1_000_000_000
        second = decode_state(reply('get_current_arm_state', [2000]*6))
        second['receive_monotonic_ns'] = 1_050_000_000
        monitor.publish(first)
        monitor.publish(second)
        with patch('robot_feedback.time.monotonic_ns', return_value=1_060_000_000):
            result = monitor.snapshot()
        self.assertEqual(result['sequence'], 2)
        self.assertAlmostEqual(result['velocity_interval_s'], .05)
        self.assertEqual(result['joint_velocity_deg_s'], [20.]*6)
        self.assertEqual(result['age_ms'], 10.)
        self.assertFalse(result['stationary'])

    def test_snapshot_reuse_preserves_sequence_and_timestamp_and_cannot_mutate_monitor(self):
        monitor = self.monitor()
        monitor.publish(decode_state(reply('get_current_arm_state')))
        first, second = monitor.snapshot(), monitor.snapshot()
        self.assertEqual(first['sequence'], second['sequence'])
        self.assertEqual(first['receive_monotonic_ns'], second['receive_monotonic_ns'])
        first['joints'][0] = 123.
        self.assertEqual(monitor.snapshot()['joints'][0], 0.)

    def test_low_frequency_checks_do_not_stall_state_sampling(self):
        blocked = threading.Event()
        release = threading.Event()
        calls = [0]
        def slow_check():
            calls[0] += 1
            if calls[0] > 1:
                blocked.set()
                release.wait(.5)
        monitor = self.monitor(process_check=slow_check)
        monitor.start()
        self.assertTrue(blocked.wait(.5))
        before = monitor.snapshot()['sequence']
        time.sleep(.08)
        after = monitor.snapshot()['sequence']
        release.set()
        self.assertGreaterEqual(after-before, 5)

    def test_alarm_is_terminal_even_when_feedback_is_fresh(self):
        alarm = threading.Event()
        def ask(host, command, timeout):
            data = reply(command)
            if command == 'get_current_arm_state' and alarm.is_set():
                data['arm_state']['arm_err'] = 4
            return data
        monitor = self.monitor(ask)
        monitor.start()
        alarm.set()
        self.wait_error(monitor)
        with self.assertRaisesRegex(MappingError, 'arm_err'):
            monitor.snapshot()

    def test_power_and_frame_changes_are_terminal_and_cannot_be_recovered_as_tcp_glitches(self):
        for field in ('power', 'frame'):
            with self.subTest(field=field):
                changed = threading.Event()
                def ask(host, command, timeout):
                    data = reply(command)
                    if changed.is_set():
                        if command == 'get_arm_power_state' and field == 'power':
                            data['power_state'] = 0
                        if command == 'get_current_work_frame' and field == 'frame':
                            data['pose'][0] = 1
                    return data
                monitor = self.monitor(ask)
                monitor.start()
                changed.set()
                self.wait_error(monitor)
                with self.assertRaises(MappingError):
                    monitor.snapshot()
                monitor.close()

    def test_single_transport_timeout_is_explicit_and_does_not_refresh_old_feedback(self):
        fail_next = threading.Event()
        failed = threading.Event()
        block_retry = threading.Event()
        def ask(host, command, timeout):
            if command == 'get_current_arm_state' and fail_next.is_set():
                fail_next.clear()
                failed.set()
                raise socket.timeout('temporary timeout')
            if command == 'get_current_arm_state' and failed.is_set():
                block_retry.wait(.2)
            return reply(command)
        monitor = self.monitor(ask)
        monitor.start()
        before = monitor.snapshot()
        fail_next.set()
        self.assertTrue(failed.wait(.5))
        deadline = time.monotonic()+.5
        while monitor.snapshot()['transport']['error_count'] == 0 and time.monotonic() < deadline:
            time.sleep(.001)
        during = monitor.snapshot()
        self.assertEqual(during['sequence'], before['sequence'])
        self.assertEqual(during['receive_monotonic_ns'], before['receive_monotonic_ns'])
        self.assertEqual(during['transport']['error_count'], 1)
        self.assertIn('temporary', during['transport']['transient_error'])
        block_retry.set()
        deadline = time.monotonic()+.5
        while monitor.snapshot()['sequence'] == before['sequence'] and time.monotonic() < deadline:
            time.sleep(.001)
        recovered = monitor.snapshot()
        self.assertGreater(recovered['sequence'], before['sequence'])
        self.assertEqual(recovered['transport']['recovery_count'], 1)

    def test_stale_feedback_remains_hard_error_regardless_transport_retry(self):
        monitor = self.monitor()
        state = decode_state(reply('get_current_arm_state'))
        state['receive_monotonic_ns'] -= 300_000_000
        monitor.publish(state)
        with self.assertRaisesRegex(MappingError, 'stale'):
            monitor.snapshot()

    def test_configuration_read_timeout_retries_and_recovers_without_stopping_sampler(self):
        fail_next = threading.Event()
        failed = threading.Event()
        release_retry = threading.Event()
        def ask(host,command,timeout):
            if command == 'get_current_tool_frame' and fail_next.is_set():
                fail_next.clear()
                failed.set()
                raise socket.timeout('tool query briefly timed out')
            if command == 'get_current_tool_frame' and failed.is_set():
                release_retry.wait(.15)
            return reply(command)
        monitor = self.monitor(ask)
        initial = monitor.start()
        fail_next.set()
        self.assertTrue(failed.wait(.5))
        deadline = time.monotonic()+.5
        while monitor.snapshot()['configuration_transport']['error_count'] == 0 and time.monotonic() < deadline:
            time.sleep(.001)
        during = monitor.snapshot()
        self.assertEqual(during['configuration_transport']['error_count'],1)
        self.assertGreater(during['sequence'],initial['sequence'])
        self.assertEqual(during['configuration_check_deadline_ms'],270.)
        self.assertIn('timed out',during['configuration_transport']['transient_error'])
        release_retry.set()
        deadline = time.monotonic()+.5
        while monitor.snapshot()['configuration_transport']['recovery_count'] == 0 and time.monotonic() < deadline:
            time.sleep(.001)
        recovered = monitor.snapshot()
        self.assertEqual(recovered['configuration_transport']['recovery_count'],1)
        self.assertIsNone(monitor.error)

    def test_configuration_continuous_transport_loss_expires_derived_deadline(self):
        lost = threading.Event()
        def ask(host,command,timeout):
            if lost.is_set() and command == 'get_current_tool_frame':
                raise socket.timeout('tool query disconnected')
            return reply(command)
        monitor = self.monitor(ask,max_feedback_age_ms=30.)
        monitor.start()
        lost.set()
        self.wait_error(monitor)
        with self.assertRaisesRegex(MappingError,'configuration feedback transport deadline'):
            monitor.snapshot()

    def test_process_check_error_is_not_reclassified_as_robot_network_timeout(self):
        calls = [0]
        def check():
            calls[0] += 1
            if calls[0] > 1:
                raise OSError('controller ownership could not be checked')
        monitor = self.monitor(process_check=check)
        monitor.start()
        self.wait_error(monitor)
        with self.assertRaisesRegex(MappingError,'ownership'):
            monitor.snapshot()
        self.assertEqual(monitor.check_transport_errors,0)


if __name__ == '__main__':
    unittest.main()

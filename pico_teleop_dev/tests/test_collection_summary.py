"""Session counts, real export eligibility, safe commands and parent lifecycle."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collection_summary
import collection_tmux
from raw_episode_writer import SCHEMA


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='summary with spaces ')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / 'raw' / 'sessions' / str(uuid.uuid4())
        self.source.mkdir(parents=True)
        (self.source / 'episodes').mkdir()
        (self.source / '.trash').mkdir()
        self.write(self.source / 'session.json', {'schema': SCHEMA, 'session_id': self.source.name,
                   'mode': 'EXECUTE', 'closed': True, 'started_wall_time_ns': 1792144369000000000})
        self.write(self.source / 'runtime_summary.json', {'episode_number': 4, 'ended_monotonic_ns': 123})
        self.episode(1)
        self.episode(2, target=False)
        self.episode(3, deleted=True)
        self.episode(4, complete=False)

    def write(self, path, data):
        path.write_text(json.dumps(data) + '\n')

    def episode(self, number, target=True, deleted=False, complete=True):
        root = self.source / ('.trash' if deleted else 'episodes') / str(uuid.uuid4())
        root.mkdir()
        (root / 'frames').mkdir()
        start = 1000000000
        self.write(root / 'episode.json', {'schema': SCHEMA, 'session_id': self.source.name,
            'episode_id': root.name, 'episode_number': number, 'mode': 'EXECUTE', 'task': '分拣',
            'closed': True, 'retained': not deleted, 'deleted': deleted,
            'start_ns': start, 'end_ns': start + 100000000,
            'termination': 'A_END' if complete else 'SIGNAL'})
        self.write(root / 'initial_state.json', {'boundary_ns': start, 'sides': {'left': None, 'right': None}})
        counts = {}
        for side in ('left', 'right'):
            rows = []
            for i in range(3):
                stamp = start + i * 33333333
                row = {'source': side, 'session_id': self.source.name, 'source_sequence': i + 1,
                       'receive_monotonic_ns': stamp, 'robot_feedback_monotonic_ns': stamp,
                       'robot_joints_deg': [0] * 6, 'robot_feedback_stationary': True,
                       'clutch_active': False, 'motion_streaming_active': False,
                       'gripper': {'feedback': {'valid': True, 'opening_controller_units': 500,
                                               'receive_monotonic_ns': stamp},
                                   'last_accepted_command': {'outcome': 'CONTROLLER_ACCEPTED',
                                       'ack_monotonic_ns': start - 1, 'target_position_controller_units': 500}}}
                if side == 'left' and not target:
                    row['gripper']['last_accepted_command'] = {}
                rows.append(row)
            (root / (side + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
            counts[side] = {'written': 3}
        payload = b'synthetic JPEG fixture for hash and sampling checks'
        for camera in ('front', 'left_wrist', 'right_wrist'):
            folder = root / 'images' / camera
            folder.mkdir(parents=True)
            rows = []
            for i in range(3):
                image = folder / (str(i) + '.jpg')
                image.write_bytes(payload)
                rows.append({'source': camera, 'session_id': self.source.name, 'source_sequence': i + 1,
                    'receive_monotonic_ns': start + i * 33333333, 'frame_number': i + 1,
                    'write_ok': True, 'image_path': str(image.relative_to(root)),
                    'sha256': hashlib.sha256(payload).hexdigest()})
            (root / 'frames' / (camera + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
            counts[camera] = {'written': 3}
        counts['pico_events'] = {'written': 0}
        (root / 'pico_events.jsonl').write_text('')
        self.write(root / 'integrity.json', {'closed': True, 'complete': complete, 'sources': counts})
        return root

    def test_counts_use_actual_deletions_and_actual_export_rules(self):
        before = {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        summary = collection_summary.build_summary(self.source)
        self.assertEqual({key: summary[key] for key in ('recorded', 'deleted', 'retained', 'complete', 'eligible')},
                         dict(recorded=4, deleted=1, retained=3, complete=2, eligible=1))
        self.assertEqual(before, {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()})
        output = collection_summary.render_summary(summary)
        self.assertIn('左夹爪缺少已确认的控制目标', output)
        self.assertIn('录制未用 A 正常结束', output)
        self.assertNotIn('no_causal_accepted_gripper_target', output)

    def test_normal_a_end_during_acknowledged_deceleration_keeps_all_grid_points(self):
        episode=next(p for p in (self.source/'episodes').iterdir()
                     if json.loads((p/'episode.json').read_text())['episode_number']==1)
        path=episode/'left.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.update(state='RELEASED',robot_feedback_stationary=False,
                       stop_confirmed_monotonic_ns=999999999,
                       last_actually_sent_command={'sequence':1,'send_monotonic_ns':999999990,
                                                   'joints_deg':[30]*6})
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        plan=collection_summary.make_plan(episode)['episodes'][0]
        self.assertTrue(plan['accepted'],plan['reasons'])
        self.assertEqual(len(plan['samples']),plan['grid_count'])
        self.assertEqual(len(plan['samples']),2)
        self.assertTrue(all(sample['sources']['left']['action_source']=='derived_stop_hold'
                            for sample in plan['samples']))
        self.assertTrue(all(sample['sources']['left']['feedback_stationary'] is False
                            for sample in plan['samples']))

    def test_image_corruption_is_not_reported_complete_or_eligible(self):
        image = next((self.source / 'episodes').glob('*/images/front/*.jpg'))
        episode_id = image.parents[2].name
        image.write_bytes(b'corrupt')
        summary = collection_summary.build_summary(self.source)
        ep = next(ep for ep in summary['episodes'] if ep['episode_id'] == episode_id)
        self.assertFalse(ep['complete'])
        self.assertFalse(ep['eligible'])

    def test_commands_select_only_this_session_quote_paths_and_avoid_overwrite(self):
        summary = collection_summary.build_summary(self.source)
        previous = {}
        for version, command in summary['commands'].items():
            args = shlex.split(command)
            self.assertEqual(args[args.index('--source') + 1], str(self.source))
            self.assertEqual(args[args.index('--format') + 1], version)
            output = Path(args[args.index('--output') + 1])
            self.assertNotIn(self.source, output.parents)
            output.mkdir(parents=True)
            previous[version] = output
        commands = collection_summary.export_commands(self.source, json.loads((self.source / 'session.json').read_text()))
        for version, command in commands.items():
            args = shlex.split(command)
            self.assertNotEqual(Path(args[args.index('--output') + 1]), previous[version])

    def test_missing_record_is_reported_unresolved_not_silently_lost(self):
        self.write(self.source / 'runtime_summary.json', {'episode_number': 5, 'ended_monotonic_ns': 123})
        summary = collection_summary.build_summary(self.source)
        self.assertEqual(summary['recorded'], 5)
        self.assertEqual(summary['unresolved'], 1)

    def test_empty_session_is_zero_not_export_success(self):
        for folder in ('episodes', '.trash'):
            import shutil
            shutil.rmtree(self.source / folder)
            (self.source / folder).mkdir()
        self.write(self.source / 'runtime_summary.json', {'episode_number': 0, 'ended_monotonic_ns': 123})
        summary = collection_summary.build_summary(self.source)
        self.assertEqual((summary['recorded'], summary['eligible']), (0, 0))
        self.assertIn('暂不能生成 LeRobot 数据集', collection_summary.render_summary(summary))

    def test_parent_prints_report_after_client_exit_and_cleans_only_scratch(self):
        runtime = self.directory / 'runtime'
        runtime.mkdir()
        manifest = {'directory': str(runtime), 'mode': 'EXECUTE', 'tmux_session': 'pico-test'}
        self.write(runtime / 'status.json', {'session_directory': str(self.source)})
        state = collection_tmux.final_result(manifest, 0)
        self.write(runtime / 'control.json', state)
        output = io.StringIO()
        def attached(*_):
            self.assertEqual(output.getvalue(), '')
            self.assertTrue(runtime.exists())
            return 0
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(collection_tmux.subprocess, 'call', side_effect=attached), \
             contextlib.redirect_stdout(output):
            code = collection_tmux.attach_and_report(manifest)
        self.assertEqual(code, 0)
        self.assertIn('可转换（本地检查通过）：1 条', output.getvalue())
        self.assertFalse(runtime.exists())
        self.assertTrue((self.source / 'collection_summary.txt').is_file())
        self.assertTrue((self.source / 'collection_summary.json').is_file())
        self.assertEqual(len(list((self.source / 'episodes').iterdir())), 3)

    def test_detach_does_not_claim_finished_or_delete_runtime(self):
        runtime = self.directory / 'runtime'
        runtime.mkdir()
        self.write(runtime / 'control.json', {'phase': 'running', 'pid': 123})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            finished = collection_tmux.show_result(runtime)
        self.assertFalse(finished)
        self.assertIn('采集仍在运行', output.getvalue())
        self.assertNotIn('本次采集结果', output.getvalue())
        self.assertTrue(runtime.exists())

    def test_failed_start_does_not_report_a_different_session(self):
        runtime = self.directory / 'runtime'
        runtime.mkdir()
        result = collection_tmux.final_result({'directory': str(runtime), 'mode': 'EXECUTE'}, 1)
        self.assertNotIn('raw_directory', result)
        self.assertIn('本次未生成 raw 记录', result['summary_text'])


if __name__ == '__main__':
    unittest.main()

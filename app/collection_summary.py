"""Summarize one raw session with the same eligibility checks as the exporter."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shlex
import time

from runtime_paths import PROJECT_DIR
from collection_viewer import clean, read_json
from export_core import make_plan
from raw_episode_writer import atomic_json


TASK_DIR = Path(__file__).resolve().parent
TRAINING_REASONS = {'quality_failed_on_fixed_grid', 'preview_excluded', 'missing_task',
                    'zero_training_frames', 'interrupted_or_failed'}


def reason_label(reason):
    labels = {
        'raw_incomplete': '记录未完整保存', 'interrupted_or_failed': '录制未用 A 正常结束',
        'preview_excluded': '预览记录，不作为正式采集导出', 'missing_task': '缺少任务说明',
        'zero_training_frames': '记录太短，没有可转换的数据帧',
        'quality_failed_on_fixed_grid': '图像或机械臂数据未通过时间对齐检查',
        'missing_completion_commit': '记录未完成保存', 'not_closed': '记录未完成收尾',
        'source_changed_during_snapshot': '检查时原始文件发生变化',
        'check_failed': '转换检查未完成，请重新检查',
    }
    if reason in labels:
        return labels[reason]
    side, _, detail = reason.partition(':')
    name = {'left': '左臂', 'right': '右臂', 'front': '头部相机',
            'left_wrist': '左腕相机', 'right_wrist': '右腕相机'}.get(side)
    if name:
        if detail == 'no_causal_accepted_gripper_target':
            return ('左' if side == 'left' else '右') + '夹爪缺少已确认的控制目标'
        if detail == 'stop_unconfirmed_or_action_ambiguous':
            return name + '的停止状态或控制目标无法确认'
        if 'missing_or_stale' in detail or detail in ('missing_telemetry', 'missing_image'):
            return name + '数据缺失或更新不及时'
        if 'gripper' in detail:
            return name + '夹爪反馈缺失或无效'
        if detail == 'image_hash_mismatch':
            return name + '图像文件校验失败'
        if detail == 'missing_joint_feedback':
            return name + '关节反馈缺失'
        return name + '原始数据未通过检查'
    return '原始记录未通过检查，详情见汇总文件'


def export_commands(source, metadata):
    """Commands name this session explicitly and never reuse an existing output."""
    source = Path(source).resolve()
    stamp = metadata.get('started_wall_time_ns')
    date = datetime.fromtimestamp(stamp / 1e9).strftime('%Y%m%d_%H%M%S') if type(stamp) is int else 'session'
    data_parent = source.parent.parent.parent if source.parent.name == 'sessions' else source.parent
    output_parent = data_parent / 'lerobot'
    commands = {}
    for version, tag in (('v2.1', 'v21'), ('v3.0', 'v30')):
        output = output_parent / ('pico_' + date + '_' + source.name[:8] + '_' + tag)
        number = 2
        while output.exists():
            output = output_parent / ('pico_' + date + '_' + source.name[:8] + '_' + tag + '_' + str(number))
            number += 1
        commands[version] = shlex.join(['bash', str(PROJECT_DIR / 'export_dataset.sh'),
            '--source', str(source), '--format', version, '--fps', '30', '--output', str(output)])
    return commands


def build_summary(source):
    source = Path(source).resolve()
    metadata = read_json(source / 'session.json')
    runtime_path = source / 'runtime_summary.json'
    runtime = read_json(runtime_path) if runtime_path.exists() else {}
    result = {'source': str(source), 'mode': metadata.get('mode'),
              'finished': bool(metadata.get('closed') or runtime.get('ended_monotonic_ns')),
              'recorded': 0, 'deleted': 0, 'retained': 0, 'complete': 0,
              'eligible': 0, 'unchecked': 0, 'unresolved': 0, 'episodes': [],
              'generated_wall_time_ns': time.time_ns()}
    max_number = 0
    directories = 0
    for folder in ('episodes', '.trash'):
        parent = source / folder
        if not parent.is_dir():
            continue
        for path in sorted(parent.iterdir()):
            if not path.is_dir() or path.is_symlink():
                continue
            directories += 1
            try:
                info = read_json(path / 'episode.json')
                number = info.get('episode_number')
                if type(number) is int:
                    max_number = max(max_number, number)
            except (OSError, ValueError):
                info, number = {}, None
            entry = {'episode_id': path.name, 'number': number, 'path': str(path)}
            if folder == '.trash':
                result['deleted'] += 1
                entry['state'] = 'deleted'
            elif not info or info.get('deleted') or not info.get('retained'):
                result['unresolved'] += 1
                entry['state'] = 'unresolved'
            else:
                result['retained'] += 1
                entry.update(state='retained', complete=False, eligible=False)
                try:
                    integrity = read_json(path / 'integrity.json')
                except (OSError, ValueError):
                    integrity = {}
                try:
                    # Per-episode plans bound memory to one record and use exactly
                    # the same 30 Hz selection, image hashes and action rules.
                    plan = make_plan(path)['episodes'][0]
                    entry['eligible'] = plan['accepted']
                    entry['reasons'] = plan['reasons']
                    entry['quality_reasons'] = dict(Counter(
                        row['reason'] for row in plan.get('quality_failures', [])).most_common())
                    entry['frames'] = len(plan['samples']) if plan['accepted'] else 0
                    raw_errors = [reason for reason in plan['reasons'] if reason not in TRAINING_REASONS]
                    entry['complete'] = bool(info.get('closed') and integrity.get('closed')
                                             and integrity.get('complete') and not raw_errors)
                except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                    entry.update(reasons=['check_failed'], check_error=str(exc))
                    result['unchecked'] += 1
                result['complete'] += int(entry['complete'])
                result['eligible'] += int(entry['eligible'])
            result['episodes'].append(entry)
    recorded = runtime.get('episode_number')
    result['recorded'] = max(recorded if type(recorded) is int else 0, max_number, directories)
    result['unresolved'] += max(0, result['recorded'] - directories)
    result['commands'] = export_commands(source, metadata)
    return result


def render_summary(summary):
    lines = ['', '本次采集结果',
             '录制：%s 条    删除：%s 条    保留：%s 条' % (
                 summary['recorded'], summary['deleted'], summary['retained']),
             '完整保存：%s 条    可转换（本地检查通过）：%s 条' % (
                 summary['complete'], summary['eligible'])]
    if not summary['finished']:
        lines.append('本次采集尚未确认正常收尾，以下为当前文件状态。')
    if summary['unresolved']:
        lines.append('另有 %s 条记录未能确认保存或删除结果，请检查 raw 目录。' % summary['unresolved'])
    if summary['unchecked']:
        lines.append('有 %s 条转换检查未完成。' % summary['unchecked'])
    rejected = [ep for ep in summary['episodes'] if ep['state'] == 'retained' and not ep['eligible']]
    if rejected:
        lines.extend(['', '未通过转换检查：'])
        for ep in sorted(rejected, key=lambda ep: ep['number'] if type(ep['number']) is int else -1)[:10]:
            reasons = list(ep.get('quality_reasons', {})) or ep.get('reasons', [])
            labels = list(dict.fromkeys(reason_label(reason) for reason in reasons))
            lines.append('  第%s条：%s' % (ep['number'] if ep['number'] is not None else '?', '；'.join(labels[:3])))
        if len(rejected) > 10:
            lines.append('  其余 %s 条详见汇总文件。' % (len(rejected) - 10))
    lines.extend(['', 'raw 目录：' + summary['source'],
                  '汇总文件：' + str(Path(summary['source']) / 'collection_summary.txt'),
                  '原始画面按 JPG 帧保存；转换 LeRobot 时会自动生成视频。', ''])
    if not summary['eligible']:
        lines.append('本次暂无通过检查的记录，暂不能生成 LeRobot 数据集。')
    lines.extend(['本次转换命令（直接复制执行）：' if summary['eligible'] else '本次转换命令（当前无可转换记录）：', '',
                  'LeRobot v2.1：', summary['commands']['v2.1'], '',
                  'LeRobot v3.0：', summary['commands']['v3.0'], ''])
    return '\n'.join(clean(line) for line in lines)


def save_summary(source):
    summary = build_summary(source)
    text = render_summary(summary)
    source = Path(summary['source'])
    atomic_json(source / 'collection_summary.json', summary)
    temporary = source / 'collection_summary.txt.new'
    temporary.write_text(text + '\n')
    temporary.replace(source / 'collection_summary.txt')
    return text

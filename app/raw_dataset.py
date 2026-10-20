"""Read-only raw validation by default; explicit interrupted-record recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid

from raw_episode_writer import atomic_json, CAMERAS, SCHEMA


def jsonl(path):
    if not path.is_file():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inspect_episode(path, hashes=True):
    result = {'path':str(path),'episode_id':path.name,'errors':[],'warnings':[],'sources':{}}
    try:
        info = json.loads((path/'episode.json').read_text())
        result['episode'] = info
        initial = json.loads((path/'initial_state.json').read_text())
        if initial.get('boundary_ns') != info.get('start_ns'):
            result['errors'].append('initial_state_boundary_mismatch')
        if info.get('schema') != SCHEMA:
            result['errors'].append('unsupported_schema')
        if not (path/'integrity.json').is_file():
            result['errors'].append('missing_completion_commit')
            integrity = {}
        else:
            integrity = json.loads((path/'integrity.json').read_text())
        result['integrity'] = integrity
        if not info.get('closed') or not integrity.get('closed'):
            result['errors'].append('not_closed')
        if not integrity.get('complete'):
            result['warnings'].append('incomplete_raw_record')
        if info.get('task_success') is not None and type(info['task_success']) is not bool:
            result['errors'].append('invalid_task_success')
        start,end = info.get('start_ns'),info.get('end_ns')
        if type(start) is not int or type(end) is not int or end < start:
            result['errors'].append('invalid_boundaries')
        for source in ('left','right','pico_events')+CAMERAS:
            name = 'frames/'+source+'.jsonl' if source in CAMERAS else source+'.jsonl'
            rows = jsonl(path/name)
            previous = None
            for row in rows:
                stamp = row['receive_monotonic_ns']
                seq = row['source_sequence']
                if type(start) is int and type(end) is int and not start <= stamp < end:
                    result['errors'].append(source+':out_of_boundary')
                if previous and (stamp < previous[1] or seq <= previous[0]):
                    result['errors'].append(source+':nonmonotonic_source')
                previous = (seq,stamp)
                if row.get('session_id') != info['session_id'] and source != 'pico_events':
                    result['errors'].append(source+':session_mismatch')
                if source in CAMERAS:
                    image = path/row['image_path']
                    try:
                        image.resolve().relative_to(path.resolve())
                    except ValueError:
                        result['errors'].append(source+':image_path_escape')
                        continue
                    if image.is_symlink() or not image.is_file() or not row.get('write_ok'):
                        result['errors'].append(source+':missing_image')
                    elif hashes and hashlib.sha256(image.read_bytes()).hexdigest() != row.get('sha256'):
                        result['errors'].append(source+':image_hash_mismatch')
            result['sources'][source] = len(rows)
            expected = integrity.get('sources',{}).get(source,{}).get('written',0)
            if expected != len(rows):
                result['errors'].append(source+':integrity_count_mismatch')
            if not rows and source != 'pico_events':
                result['warnings'].append(source+':no_samples')
    except (ValueError,KeyError,TypeError,OSError) as exc:
        result['errors'].append(type(exc).__name__+':'+str(exc))
    result['errors'] = sorted(set(result['errors']))
    result['valid'] = not result['errors']
    return result


def episode_paths(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('source must be an existing non-symlink raw directory')
    root = root.resolve()
    if (root/'episode.json').is_file():
        return [root]
    pattern = 'episodes/*' if (root/'session.json').is_file() else 'sessions/*/episodes/*'
    return sorted(p for p in root.glob(pattern) if p.is_dir() and not p.is_symlink())


def recover(path):
    if (path/'integrity.json').exists():
        raise ValueError('completion commit already exists; recovery refuses to overwrite it')
    info = json.loads((path/'episode.json').read_text())
    rows = [r for name in ('left.jsonl','right.jsonl','pico_events.jsonl')+
            tuple('frames/'+c+'.jsonl' for c in CAMERAS) for r in jsonl(path/name)]
    observed_end = max([info['start_ns']]+[r['receive_monotonic_ns']+1 for r in rows])
    info.update(closed=True,integrity='incomplete',termination='RECOVERED_INTERRUPTION',interrupted=True,
        recovery_wall_time_ns=time.time_ns(),end_ns=info.get('end_ns') or observed_end,
        recovery_end_semantics='original end if present; otherwise lower bound from last persisted data, not observed A event')
    atomic_json(path/'episode.json',info)
    counts = {}
    for source in ('left','right','pico_events')+CAMERAS:
        selected = [r for r in rows if r['source']==source]
        counts[source] = {'written':len(selected),'submitted':None,'sequence_gaps':[],
                          'last_persisted':selected[-1].get('source_sequence') if selected else None}
    atomic_json(path/'integrity.json',{'schema':SCHEMA,'closed':True,'complete':False,'normal_close':False,
        'recovered':True,'sources':counts,'errors':['process_interruption','pre_crash_queue_loss_unknown']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--recover',metavar='EPISODE_UUID')
    parser.add_argument('--skip-image-hashes',action='store_true')
    args = parser.parse_args()
    paths = episode_paths(args.source)
    if args.recover:
        episode_id = str(uuid.UUID(args.recover))
        selected = [p for p in paths if p.name==episode_id]
        if len(selected)!=1:
            parser.error('expected exactly one registered episode UUID')
        recover(selected[0])
    reports = [inspect_episode(p,not args.skip_image_hashes) for p in paths]
    result = {'schema':SCHEMA,'source':str(args.source.resolve()),'read_only':not bool(args.recover),
        'episode_count':len(reports),'valid_count':sum(r['valid'] for r in reports),'episodes':reports,
        'errors':[] if reports else ['no_episodes_found']}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if reports and all(r['valid'] for r in reports) else 1


if __name__=='__main__':
    raise SystemExit(main())

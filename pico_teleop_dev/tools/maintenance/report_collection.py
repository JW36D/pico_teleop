"""Report actual source timing and integrity; never infer headset display latency."""
import argparse
import json
from pathlib import Path

from raw_dataset import episode_paths, inspect_episode


def rows(path):
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if line.strip(): yield json.loads(line)


def percentiles(values):
    values=sorted(values)
    if not values: return None
    def percentile(q):
        index=(len(values)-1)*q
        lo=int(index); hi=min(lo+1,len(values)-1)
        return round(values[lo]+(values[hi]-values[lo])*(index-lo),3)
    return {'count':len(values),'p50':percentile(.5),'p95':percentile(.95),
        'p99':percentile(.99),'maximum':round(values[-1],3)}


def episode_report(path):
    checked=inspect_episode(path,hashes=False)
    info=checked.get('episode',{})
    report={'episode_id':path.name,'mode':info.get('mode'),'task':info.get('task'),
        'closed':info.get('closed'),'integrity_complete':checked.get('integrity',{}).get('complete'),
        'index_errors':checked['errors'],'image_hashes_checked':False,
        'duration_s':(info['end_ns']-info['start_ns'])/1e9 if type(info.get('end_ns')) is int else None,
        'sources':{},'writer_queue_peak':checked.get('integrity',{}).get('writer_queue_peak'),
        'writer_dropped':checked.get('integrity',{}).get('writer_dropped')}
    for source in ('front','left_wrist','right_wrist','left','right'):
        relative=Path('frames')/(source+'.jsonl') if source not in ('left','right') else Path(source+'.jsonl')
        intervals=[]; feedback_ages=[]; send_intervals=[]; continuous_intervals=[]; frame_gaps=0; duplicates=0
        count=0; first=last=last_send=last_frame=continuous_send=None
        for row in rows(path/relative):
            stamp=row['receive_monotonic_ns']
            if first is None: first=stamp
            if last is not None: intervals.append((stamp-last)/1e6)
            last=stamp; count+=1
            if source in ('left','right'):
                feedback=row.get('robot_feedback_monotonic_ns')
                if type(feedback) is int: feedback_ages.append((stamp-feedback)/1e6)
                if row.get('motion_sent'):
                    sent=(row.get('last_actually_sent_command') or {}).get('send_monotonic_ns')
                    if type(sent) is int and sent!=last_send:
                        if last_send is not None: send_intervals.append((sent-last_send)/1e6)
                        if continuous_send is not None: continuous_intervals.append((sent-continuous_send)/1e6)
                        last_send=sent
                        continuous_send=sent
                if not row.get('clutch_active') or not row.get('motion_streaming_active'):
                    continuous_send=None
            else:
                number=row['frame_number']
                if last_frame is not None:
                    frame_gaps+=max(0,number-last_frame-1)
                    duplicates+=number==last_frame
                last_frame=number
        result={'received':count,'observed_hz':round((count-1)*1e9/(last-first),3)
            if count>1 and last>first else None,'receive_interval_ms':percentiles(intervals)}
        if source in ('left','right'):
            result.update(feedback_receive_age_ms=percentiles(feedback_ages),
                actual_send_interval_ms=percentiles(send_intervals),
                continuous_actual_send_interval_ms=percentiles(continuous_intervals))
        else: result.update(frame_number_gaps=frame_gaps,duplicate_frames=duplicates)
        report['sources'][source]=result
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    reports=[episode_report(p) for p in episode_paths(args.source)]
    result={'source':str(args.source.resolve()),'episodes':reports,
        'timing_axis':'host monotonic receive; camera exposure and robot sample time unavailable',
        'headset_end_to_end_latency_measured':False,
        'preview_is_real_robot_action_evidence':False}
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output: args.output.write_text(text)
    else: print(text,end='')
    return 0 if reports and all(not r['index_errors'] and r['integrity_complete'] for r in reports) else 1


if __name__=='__main__': raise SystemExit(main())

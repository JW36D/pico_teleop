"""Observe a bounded, continuous preview/execute load run; send no commands."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--wait-seconds',type=float,default=300)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 10<=args.seconds<=3600 or not 1<=args.wait_seconds<=900:
        parser.error('duration outside bounded monitor range')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():
        parser.error('report exists; choose a new output')
    wait_deadline=time.monotonic()+args.wait_seconds
    started=None; initial=None; final=None; samples=0; failure=None
    next_notice=0; max_send_age=0; max_queue=0
    with args.output.with_suffix('.jsonl').open('x') as trace:
        while True:
            try:
                state=json.loads(args.status.read_text())
            except (OSError,ValueError) as exc:
                failure='status_unavailable:'+str(exc); break
            now=time.monotonic_ns()
            sources=state.get('sources',{})
            display=sources.get('display',{}).get('display_status',{})
            healthy=(state.get('ready') and state.get('recording') and display.get('connected')
                and now-state['time_monotonic_ns']<500_000_000
                # Display status arrives at 5Hz and coordinator snapshots at
                # 10Hz. Allow their publication delay; this is a liveness
                # check, not a measurement of photon-to-photon latency.
                and 0<=now-display.get('last_sent_monotonic_ns',0)<750_000_000)
            if started is None:
                if healthy:
                    started=time.monotonic(); initial=state
                    print('LOAD_STARTED '+str(state.get('session_id')),flush=True)
                elif time.monotonic()>wait_deadline:
                    failure='waiting_for_recording_and_live_video_timed_out'; break
                else:
                    time.sleep(.5); continue
            final=state; samples+=1
            elapsed=time.monotonic()-started
            send_age=(now-display.get('last_sent_monotonic_ns',0))/1e6
            max_send_age=max(max_send_age,send_age)
            max_queue=max(max_queue,state.get('writer_queue_peak',0))
            row={'observed_monotonic_ns':now,'elapsed_s':round(elapsed,3),'healthy':bool(healthy),
                'ready':state.get('ready'),'recording':state.get('recording'),
                'writer_queue_peak':state.get('writer_queue_peak'),'writer_dropped':state.get('writer_dropped'),
                'display':display,'sources':{s:{k:v for k,v in record.items() if k!='metadata'}
                    for s,record in sources.items() if s!='display'}}
            trace.write(json.dumps(row,ensure_ascii=False)+'\n'); trace.flush()
            if not healthy:
                failure='recording_or_source_or_display_discontinuity'; break
            initial_display=initial['sources']['display']['display_status']
            if display.get('stream_connections')!=initial_display.get('stream_connections'):
                failure='video_reconnected_during_load'; break
            if display.get('error_history')!=initial_display.get('error_history'):
                failure='video_error_during_load'; break
            if state.get('writer_dropped') or state.get('writer_error'):
                failure='writer_loss_or_failure'; break
            for source in ('left','right','front','left_wrist','right_wrist'):
                record=sources[source]; baseline=initial['sources'][source]
                if record.get('transport_dropped',0)>baseline.get('transport_dropped',0):
                    failure=source+':transport_loss'
                if record.get('frame_number_gaps',0)>baseline.get('frame_number_gaps',0):
                    failure=source+':camera_frame_gap'
            if failure or elapsed>=args.seconds: break
            if elapsed>=next_notice:
                print('LOAD %.0fs video=%s queue_peak=%s'%(elapsed,display.get('sent_frames'),max_queue),flush=True)
                next_notice=elapsed+60
            time.sleep(.5)
    elapsed=time.monotonic()-started if started is not None else 0
    result={'passed':failure is None and elapsed>=args.seconds,'failure':failure,
        'observed_seconds':round(elapsed,3),'required_seconds':args.seconds,'samples':samples,
        'initial':initial,'final':final,'writer_queue_peak':max_queue,
        'maximum_sampled_time_since_last_video_send_ms':round(max_send_age,2),
        'headset_end_to_end_latency_measured':False,'motion_commands_sent_by_monitor':0}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('initial','final')},ensure_ascii=False),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__': raise SystemExit(main())

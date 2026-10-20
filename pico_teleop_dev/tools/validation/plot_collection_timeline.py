"""Plot preserved camera/feedback/accepted-command times for a raw episode."""
import argparse
import json
from pathlib import Path


def rows(path):
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if line.strip(): yield json.loads(line)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='PNG path; SVG saved alongside')
    parser.add_argument('--start-s',type=float,default=0)
    parser.add_argument('--seconds',type=float,default=20)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    info=json.loads((args.episode/'episode.json').read_text())
    if not info.get('closed'):
        parser.error('episode must be closed')
    start=info['start_ns']
    end=(info['end_ns']-start)/1e9
    lo,hi=args.start_s,min(end,args.start_s+args.seconds)
    if not 0<=lo<hi:
        parser.error('requested window does not intersect episode')
    def seconds(stamp): return (stamp-start)/1e9
    fig,axes=plt.subplots(3,2,figsize=(14,10),sharex=True)
    for column,side in enumerate(('left','right')):
        observed=[]; sent=[]; grip=[]; gripper=[]; target=[]; ages=[]
        seen_feedback=seen_send=seen_gripper=seen_target=None
        for row in rows(args.episode/(side+'.jsonl')):
            t=seconds(row['receive_monotonic_ns'])
            if t<lo-.5: continue
            if t>hi+.5: break
            feedback=row.get('robot_feedback_monotonic_ns')
            joints=row.get('robot_joints_deg')
            if type(feedback) is int:
                ages.append((t,(row['receive_monotonic_ns']-feedback)/1e6))
                if feedback!=seen_feedback and isinstance(joints,list) and len(joints)==6:
                    observed.append([seconds(feedback)]+joints);seen_feedback=feedback
            command=row.get('last_actually_sent_command') or {}
            stamp=command.get('send_monotonic_ns')
            if row.get('clutch_active') and row.get('motion_streaming_active') and type(stamp) is int:
                if stamp!=seen_send and isinstance(command.get('joints_deg'),list):
                    sent.append([seconds(stamp)]+command['joints_deg']);seen_send=stamp
            grip.append((t,float(bool(row.get('clutch_active')))))
            g=row.get('gripper') or {}
            response=g.get('feedback') or {}
            stamp=response.get('receive_monotonic_ns')
            if response.get('valid') and type(stamp) is int and stamp!=seen_gripper:
                opening=response.get('opening_controller_units')
                if type(opening) in (int,float):
                    gripper.append((seconds(stamp),opening/1000));seen_gripper=stamp
            accepted=g.get('last_accepted_command') or {}
            stamp=accepted.get('ack_monotonic_ns')
            if accepted.get('outcome')=='CONTROLLER_ACCEPTED' and type(stamp) is int and stamp!=seen_target:
                value=accepted.get('target_position_controller_units')
                if type(value) in (int,float):
                    target.append((seconds(stamp),value/1000));seen_target=stamp
        colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        for joint in range(6):
            if observed:
                a=np.asarray(observed);axes[0,column].plot(a[:,0],a[:,joint+1],color=colors[joint],lw=1,label='J%d feedback'%(joint+1))
            if sent:
                a=np.asarray(sent);axes[0,column].step(a[:,0],a[:,joint+1],where='post',color=colors[joint],ls='--',lw=.8)
        axes[0,column].set_title(side+' joints (solid feedback; dashed successful sends)')
        axes[0,column].set_ylabel('degrees; J6 unwrapped')
        if observed: axes[0,column].legend(ncol=3,fontsize=7)
        for data,label,style in ((gripper,'opening response','-'),(target,'accepted target','--'),(grip,'Grip',':')):
            if data:
                a=np.asarray(data);axes[1,column].step(a[:,0],a[:,1],where='post',ls=style,lw=1,label=label)
        axes[1,column].set_title(side+' gripper / Grip')
        axes[1,column].set_ylabel('controller opening / 1000')
        axes[1,column].legend(fontsize=8)
        if ages:
            a=np.asarray(ages);axes[2,1].plot(a[:,0],a[:,1],lw=.7,label=side)
    for i,camera in enumerate(('front','left_wrist','right_wrist')):
        times=[seconds(r['receive_monotonic_ns']) for r in rows(args.episode/'frames'/(camera+'.jsonl'))
            if lo<=seconds(r['receive_monotonic_ns'])<=hi]
        axes[2,0].eventplot(times,lineoffsets=i,linelengths=.7,linewidths=.5)
    axes[2,0].set_yticks(range(3),['front','left_wrist','right_wrist'])
    axes[2,0].set_title('Actual camera host receives (not exposure times)')
    axes[2,1].axhline(50,color='red',ls='--',lw=.7,label='export threshold')
    axes[2,1].set_title('Age of latest robot feedback response')
    axes[2,1].set_ylabel('ms');axes[2,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlim(lo,hi);ax.grid(alpha=.2)
        if lo==0: ax.axvline(0,color='black',lw=1)
        if hi==end: ax.axvline(end,color='black',lw=1)
    for ax in axes[2]: ax.set_xlabel('Host monotonic seconds from A start')
    fig.suptitle(info['episode_id'][:8]+' | '+info['mode']+' | raw receive/ack times; no smoothing')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(args.output),dpi=150)
    fig.savefig(str(args.output.with_suffix('.svg')))
    plt.close(fig)
    print(args.output)
    return 0


if __name__=='__main__': raise SystemExit(main())

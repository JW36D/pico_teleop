"""Explicit synthetic, no-hardware writer/reader smoke in a pinned export image."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from export_core import make_plan, DEFAULT_PROFILE
from export_dataset import export
from raw_episode_writer import RawEpisodeWriter, CAMERAS


def fixture(root):
    import numpy as np
    from PIL import Image
    import io
    colors={'front':(20,210,30),'left_wrist':(220,25,20),'right_wrist':(20,30,220)}
    images={}
    for role,color in colors.items():
        b=io.BytesIO();Image.new('RGB',(64,48),color).save(b,format='JPEG',quality=95)
        images[role]=b.getvalue()
    w=RawEpisodeWriter(root,{'mode':'SYNTHETIC_OFFLINE','synthetic':True,
        'required_sources':['left','right']+list(CAMERAS)},history_s=10)
    base=time.monotonic_ns()
    seq={s:0 for s in ('left','right')+CAMERAS}
    for episode_number in (1,2):
        start=base+(episode_number-1)*1_000_000_000
        ep=w.begin({'receive_monotonic_ns':start},episode_number,'synthetic color sorting fixture')
        for i in range(17):  # final sample is an explicit post-end source watermark
            stamp=start+i*33_333_333
            for side,offset in (('left',0),('right',10)):
                seq[side]+=1
                w.submit({'kind':'telemetry','synthetic':True,'source':side,'source_sequence':seq[side],
                    'session_id':w.session_id,'receive_monotonic_ns':stamp,'control_tick':seq[side],
                    'mode':'SYNTHETIC_OFFLINE','robot_joints_deg':[offset,1,2,3,4,370+i],
                    'robot_feedback_monotonic_ns':stamp,'robot_feedback_sequence':seq[side],
                    'robot_feedback_stationary':True,'clutch_active':side=='left','motion_streaming_active':side=='left',
                    'state':'TRACKING' if side=='left' else 'RELEASED',
                    'last_actually_sent_command':{'sequence':seq[side],'send_monotonic_ns':stamp,
                        'joints_deg':[offset+1,2,3,4,5,380+i],'synthetic':True},
                    'gripper':{'feedback':{'valid':True,'receive_monotonic_ns':stamp,'opening_controller_units':1003},
                        'last_accepted_command':{'outcome':'CONTROLLER_ACCEPTED','ack_monotonic_ns':stamp,
                            'target_position_controller_units':750,'synthetic':True}}})
            for camera in CAMERAS:
                seq[camera]+=1
                w.submit({'kind':'frame','synthetic':True,'source':camera,'camera_role':camera,
                    'session_id':w.session_id,'source_sequence':seq[camera],'receive_monotonic_ns':stamp,
                    'frame_number':seq[camera],'width':64,'height':48,'color_space':'RGB'},images[camera])
        w.end(ep,{'receive_monotonic_ns':start+500_000_000})
    w.close()
    return w.directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format',choices=('v2.1','v3.0'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture-root',type=Path,required=True)
    args=parser.parse_args()
    if not list(args.fixture_root.glob('sessions/*')):
        fixture(args.fixture_root)
    plan=make_plan(args.fixture_root,{'include_preview':True})
    plan['synthetic_smoke']=True
    plan['training_eligible']=False
    if not all(ep['accepted'] for ep in plan['episodes']):
        raise RuntimeError(str([(ep['episode_id'],ep['reasons'],ep.get('quality_failures')) for ep in plan['episodes']]))
    output=export(plan,args.format,args.output,'local/pico_synthetic_smoke')
    print(json.dumps({'output':str(output),'synthetic':True,'training_eligible':False,
        'episodes':len(plan['episodes']),'frames':sum(len(ep['samples']) for ep in plan['episodes']),
        'selection_sha256':hashlib.sha256(json.dumps([ep['samples'] for ep in plan['episodes']],sort_keys=True).encode()).hexdigest()}))
    return 0


if __name__=='__main__':raise SystemExit(main())

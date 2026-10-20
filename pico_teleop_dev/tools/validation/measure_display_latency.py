"""Measure optical scene/display delay from one externally recorded video.

The phone must simultaneously see a physical black/white change and the same
change through the headset lens. This tool does not use sender encode duration
as a substitute for end-to-end display latency.
"""
import argparse
import json
from pathlib import Path


def transitions(values, fps):
    import numpy as np
    low,high=np.percentile(values,[10,90])
    if high-low<35:
        raise ValueError('optical ROI contrast too low; use a black/white target')
    state=None; edges=[]
    for frame,value in enumerate(values):
        new=False if value<low+.3*(high-low) else True if value>low+.7*(high-low) else state
        if new is not None and state is not None and new!=state:
            edges.append((frame,bool(new)))
        state=new
    if any(b[0]-a[0]<fps*.15 for a,b in zip(edges,edges[1:])):
        raise ValueError('ambiguous rapid transitions; hold each target state for >=0.5s')
    return edges


def match_edges(source,display,fps,max_delay_ms=1000):
    matches=[]; used=set(); unmatched=[]
    for frame,polarity in source:
        candidates=[(index,event) for index,event in enumerate(display) if index not in used
            and event[1]==polarity and 0<=event[0]-frame<=max_delay_ms*fps/1000]
        if not candidates:
            unmatched.append({'source_frame':frame,'polarity':polarity}); continue
        index,event=candidates[0]; used.add(index)
        matches.append({'source_frame':frame,'headset_frame':event[0],
            'polarity':polarity,'delay_ms':(event[0]-frame)*1000/fps})
    return matches,unmatched,[event for i,event in enumerate(display) if i not in used]


def rectangle(value):
    parts=tuple(int(x) for x in value.split(','))
    if len(parts)!=4 or any(x<0 for x in parts) or min(parts[2:])==0:
        raise argparse.ArgumentTypeError('ROI must be x,y,width,height in video pixels')
    return parts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--source-roi',type=rectangle,required=True)
    parser.add_argument('--headset-roi',type=rectangle,required=True)
    parser.add_argument('--capture-fps',type=float,required=True,
        help='Actual phone capture rate, not slow-motion playback/export rate')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--synthetic',action='store_true',help='Mark generated fixtures explicitly')
    args=parser.parse_args()
    if args.output.suffix!='.json': parser.error('--output must end in .json')
    if not 60<=args.capture_fps<=1000:
        parser.error('use an actual capture rate >=60fps for this measurement')
    if args.output.exists(): parser.error('output exists; refusing overwrite')
    a,b=args.source_roi,args.headset_roi
    if max(a[0],b[0])<min(a[0]+a[2],b[0]+b[2]) and max(a[1],b[1])<min(a[1]+a[3],b[1]+b[3]):
        parser.error('physical and headset ROIs must be separate')
    import cv2
    import numpy as np
    capture=cv2.VideoCapture(str(args.video))
    values=[[],[]]; first=None
    try:
        while True:
            ok,image=capture.read()
            if not ok: break
            if first is None: first=image.copy()
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            for result,(x,y,w,h) in zip(values,(args.source_roi,args.headset_roi)):
                if x+w>image.shape[1] or y+h>image.shape[0]:
                    raise ValueError('ROI outside recorded frame')
                result.append(float(gray[y:y+h,x:x+w].mean()))
    finally:
        capture.release()
    if first is None: raise ValueError('no decoded video frames')
    source=transitions(values[0],args.capture_fps)
    display=transitions(values[1],args.capture_fps)
    matches,unmatched,extra=match_edges(source,display,args.capture_fps)
    if len(matches)<20 or unmatched or extra:
        raise ValueError('need >=20 unambiguous transitions; matched=%d source_missing=%d display_extra=%d'
            %(len(matches),len(unmatched),len(extra)))
    delays=[event['delay_ms'] for event in matches]
    frame_ms=1000/args.capture_fps
    p95=float(np.percentile(delays,95))
    report={'measurement':'physical scene and actual headset lens in same external recording',
        'operator_video':str(args.video.resolve()),'synthetic':args.synthetic,
        'capture_fps_operator_supplied':args.capture_fps,'decoded_frames':len(values[0]),
        'source_roi':args.source_roi,'headset_roi':args.headset_roi,
        'delays_ms':{name:float(np.percentile(delays,q)) for name,q in [('p50',50),('p95',95),('p99',99)]},
        'frame_quantization_ms':frame_ms,'p95_plus_one_frame_ms':p95+frame_ms,
        'target_p95_ms':150,'meets_target_with_one_frame_margin':not args.synthetic and p95+frame_ms<=150,
        'matches':matches,'unmatched_source':unmatched,'unmatched_display':extra,
        'limitations':'ROI identity and capture FPS require operator verification; rolling shutter and exposure differ'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4))
    for values_,label in zip(values,('physical target','headset image')):
        ax.plot(np.arange(len(values_))/args.capture_fps,values_,label=label,lw=.8)
    ax.set(xlabel='External capture time (s)',ylabel='ROI mean luminance',
        title='Optical scene/display transitions; P95 %.1f ms%s'%(p95,' | SYNTHETIC' if args.synthetic else ''))
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(str(args.output.with_suffix('.png')),dpi=150);plt.close(fig)
    for roi,color in ((args.source_roi,(0,255,0)),(args.headset_roi,(0,0,255))):
        x,y,w,h=roi;cv2.rectangle(first,(x,y),(x+w,y+h),color,2)
    cv2.imwrite(str(args.output.with_name(args.output.stem+'_rois.jpg')),first)
    print(json.dumps({k:v for k,v in report.items() if k!='matches'},indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())

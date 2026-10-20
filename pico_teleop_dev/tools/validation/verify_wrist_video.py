"""Bounded NVENC/native TCP/decoder smoke; synthetic colors, no robot APIs."""
import argparse
import json
from pathlib import Path
import socket
import struct
import time

from wrist_video import WristVideoSender, recv_exact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=180)
    parser.add_argument('--source',choices=('ZEDMINI','PICO4U'),default='ZEDMINI')
    args=parser.parse_args()
    if not 30<=args.frames<=600:
        parser.error('--frames must be 30..600')
    import cv2
    import numpy as np
    width,height = (2560,720) if args.source=='ZEDMINI' else (2160,810)
    camera = 'ZED' if args.source=='ZEDMINI' else 'VR'
    args.output.mkdir(parents=True,exist_ok=False)
    images={role:np.full((480,640,3),color,np.uint8) for role,color in
        [('left_wrist',(20,30,220)),('right_wrist',(220,30,20))]}
    def latest():
        now=time.monotonic_ns()
        return {role:(now,image) for role,image in images.items()}
    server=socket.socket()
    server.bind(('127.0.0.1',0)); server.listen(1); server.settimeout(10)
    sender=WristVideoSender(latest,{'width':width,'height':height,'fps':60,'bitrate':4000000,
        'decoder_profiles':[{'camera':camera,'width':width,'height':height,'fps':60}],
        'mode':'listen','listen_port':0,'bind_ip':'127.0.0.1'})
    sender.start()
    control=None; arrivals=[]
    try:
        deadline=time.monotonic()+3
        while not sender.status['listening']:
            if sender.status['error'] or time.monotonic()>deadline:
                raise RuntimeError('native control listener failed: '+str(sender.status))
            time.sleep(.01)
        control=socket.create_connection(('127.0.0.1',sender.status['listen_port']),1)
        data=b'\xca\xfe\x01'+struct.pack('<7i',width,height,60,4000000,0,2,server.getsockname()[1])
        data+=bytes([len(camera)])+camera.encode()+b'\x09127.0.0.1'
        command=b'OPEN_CAMERA'
        body=struct.pack('<I',len(command))+command+struct.pack('<I',len(data))+data
        control.sendall(struct.pack('!I',len(body))+body)
        connection,_=server.accept(); connection.settimeout(2)
        with connection,(args.output/'stream.h264').open('wb') as stream:
            deadline=time.monotonic()+args.frames/20+5
            while len(arrivals)<args.frames and time.monotonic()<deadline:
                size=struct.unpack('!I',recv_exact(connection,4))[0]
                if not 0<size<=5_000_000:
                    raise ValueError('invalid H.264 access-unit length')
                stream.write(recv_exact(connection,size))
                arrivals.append(time.monotonic_ns())
    finally:
        if control: control.close()
        sender.close(); server.close()
    decoder=cv2.VideoCapture(str(args.output/'stream.h264'))
    count=0; first=None
    try:
        while True:
            ok,image=decoder.read()
            if not ok: break
            if first is None: first=image
            count+=1
    finally:
        decoder.release()
    if first is None:
        raise RuntimeError('native H.264 stream produced no decoded image')
    cv2.imwrite(str(args.output/'decoded.jpg'),first)
    intervals=np.diff(arrivals)/1e6
    left_pixel=first[height//2,width//4]
    right_pixel=first[height//2,width*3//4]
    passed=(count==args.frames and first.shape==(height,width,3) and
        left_pixel[2]>180 and right_pixel[0]>180)
    report={'passed':bool(passed),'synthetic_input':True,'robot_commands':0,
        'profile':{'source':args.source,'width':width,'height':height,'fps':60},'wire_frames':len(arrivals),
        'decoded_frames':count,'observed_fps':round(1000/float(intervals.mean()),2),
        'interarrival_ms':{p:round(float(np.percentile(intervals,n)),2)
            for p,n in [('p50',50),('p95',95),('p99',99)]},
        'decoded_shape':list(first.shape),'left_center_bgr':left_pixel.tolist(),
        'right_center_bgr':right_pixel.tolist(),
        'end_to_end_headset_latency_measured':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())

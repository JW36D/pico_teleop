"""One owner for three RGB streams; capture/encoding/display never run in control."""
import argparse
import fcntl
import json
from pathlib import Path
import queue
import signal
import sys
import threading
import time

from collection_ipc import Publisher

ROLES = ('front', 'left_wrist', 'right_wrist')


def load_camera_config(path, require_roles=True):
    config = json.loads(Path(path).read_text())
    cameras = config.get('cameras', {})
    if set(cameras) != set(ROLES):
        raise ValueError('camera roles must be front/left_wrist/right_wrist')
    serials = [cameras[r].get('serial') for r in ROLES]
    if require_roles and (any(not isinstance(s, str) or not s for s in serials) or len(set(serials)) != 3):
        raise ValueError('相机角色尚未确认或序列号重复，请先运行 --identify 并显式绑定角色')
    for role in ROLES:
        if cameras[role].get('rotation_deg', 0) not in (0, 90, 180, 270):
            raise ValueError('rotation_deg must be 0/90/180/270')
    for key in ('width', 'height', 'fps', 'jpeg_quality'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError('invalid camera '+key)
    if config['jpeg_quality'] > 100:
        raise ValueError('JPEG quality exceeds 100')
    return config


def device_info(device, rs):
    result = {}
    for key in ('serial_number','name','firmware_version','usb_type_descriptor','physical_port'):
        item = getattr(rs.camera_info, key)
        result[key] = device.get_info(item) if device.supports(item) else None
    return result


class CameraService:
    def __init__(self, config, ipc, session_id, video=None, status_path=None):
        import cv2
        import numpy as np
        import pyrealsense2 as rs
        self.cv2, self.np, self.rs = cv2, np, rs
        self.config, self.ipc, self.session_id = config, ipc, session_id
        self.stop = threading.Event()
        self.latest, self.latest_lock = {}, threading.Lock()
        self.pipelines, self.publishers, self.threads = [], [], []
        self.failure = None
        self.video_config, self.status_path, self.video = video, status_path, None
        self.video_publisher = None

    def start(self):
        rs = self.rs
        ctx = rs.context()
        devices = {d.get_info(rs.camera_info.serial_number):d for d in ctx.query_devices()}
        wanted = {self.config['cameras'][role]['serial'] for role in ROLES}
        if not wanted.issubset(devices):
            raise RuntimeError('missing configured camera: '+str(wanted-set(devices)))
        for role in ROLES:
            settings = self.config['cameras'][role]
            serial = settings['serial']
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, self.config['width'], self.config['height'], rs.format.bgr8, self.config['fps'])
            pipe = rs.pipeline(ctx)
            profile = pipe.start(cfg)
            self.pipelines.append(pipe)
            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = stream.get_intrinsics()
            sensor = profile.get_device().first_color_sensor()
            exposure = settings.get('exposure', {'auto':True})
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1 if exposure.get('auto', True) else 0)
            if not exposure.get('auto', True) and 'value' in exposure:
                sensor.set_option(rs.option.exposure, exposure['value'])
            actual_options = {}
            for name in ('exposure','gain','enable_auto_exposure','white_balance','enable_auto_white_balance'):
                option = getattr(rs.option, name)
                actual_options[name] = sensor.get_option(option) if sensor.supports(option) else None
            metadata = {'role':role, 'device':device_info(devices[serial], rs),
                'profile':{'width':stream.width(), 'height':stream.height(), 'fps':stream.fps(), 'format':str(stream.format())},
                'intrinsics':{name:getattr(intrinsics,name) for name in ('width','height','ppx','ppy','fx','fy','coeffs')},
                'distortion_model':str(intrinsics.model), 'options':actual_options,
                'rotation_deg':settings.get('rotation_deg',0), 'extrinsics':None,
                'extrinsics_unknown_reason':'not measured', 'encoding':{'codec':'JPEG','quality':self.config['jpeg_quality'],
                'decoded_color':'RGB', 'encoder_input_color':'BGR', 'lossless':False}}
            pub = Publisher(self.ipc, role, self.session_id, capacity=64, metadata=metadata)
            self.publishers.append(pub)
            encode_queue = queue.Queue(8)
            for target, args in ((self._capture,(role,pipe,encode_queue,pub)),
                                 (self._encode,(role,encode_queue,pub))):
                worker = threading.Thread(target=target,args=args,name=target.__name__+role,daemon=True)
                self.threads.append(worker)
                worker.start()
        if self.video_config and self.video_config.get('enabled'):
            from wrist_video import WristVideoSender
            self.video = WristVideoSender(self.latest_snapshot, self.video_config, self.status_path)
            self.video.start()
            self.video_publisher = Publisher(self.ipc,'display',self.session_id,capacity=16,
                metadata={'config':self.video_config,'protocol':'xrtoolkit_tcp_h264_length_u32be'})

    def publish_display_status(self):
        if self.video_publisher:
            self.video_publisher.submit({'kind':'display_status','receive_monotonic_ns':time.monotonic_ns(),
                'status':dict(self.video.status)})

    def latest_snapshot(self):
        with self.latest_lock:
            return self.latest.copy()

    def _capture(self, role, pipe, target, pub):
        rs = self.rs
        frame_meta = ('actual_exposure','gain_level','sensor_timestamp','frame_timestamp','time_of_arrival')
        last_frame, last_receive = None, None
        last_seen = time.monotonic_ns()
        try:
            while not self.stop.is_set():
                frames = pipe.poll_for_frames()
                frame = frames.get_color_frame() if frames else None
                if not frame:
                    timeout_ns = 10_000_000_000 if last_frame is None else 1_000_000_000
                    if time.monotonic_ns()-last_seen > timeout_ns:
                        raise RuntimeError('camera_stream_timeout')
                    self.stop.wait(.002)
                    continue
                now = time.monotonic_ns()
                last_seen = now
                number = frame.get_frame_number()
                image = self.np.asanyarray(frame.get_data()).copy()
                rotation = self.config['cameras'][role].get('rotation_deg',0)
                if rotation:
                    image = self.np.rot90(image, -(rotation//90)).copy()
                metadata = {}
                for name in frame_meta:
                    field = getattr(rs.frame_metadata_value,name)
                    metadata[name] = frame.get_frame_metadata(field) if frame.supports_frame_metadata(field) else None
                record = {'kind':'frame','source':role,'camera_role':role,
                    'serial':self.config['cameras'][role]['serial'],'frame_number':number,
                    'device_timestamp_ms':frame.get_timestamp(), 'device_timestamp_domain':str(frame.get_frame_timestamp_domain()),
                    'receive_monotonic_ns':now, 'time_semantics':'host_receive_not_exposure',
                    'metadata':metadata, 'width':image.shape[1], 'height':image.shape[0],
                    'codec':'JPEG','quality':self.config['jpeg_quality'],'color_space':'RGB',
                    'frame_number_gap':max(0,number-last_frame-1) if last_frame is not None else 0,
                    'duplicate':number == last_frame, 'receive_interval_ns':now-last_receive if last_receive else None}
                last_frame, last_receive = number, now
                with self.latest_lock:
                    self.latest[role] = (now, image)
                try:
                    target.put_nowait((record,image))
                except queue.Full:
                    raise RuntimeError('camera_encode_queue_overflow')
        except Exception as exc:
            self.failure = role+':'+str(exc)
            pub.submit({'kind':'failure','receive_monotonic_ns':time.monotonic_ns(),'error':self.failure})
            self.stop.set()

    def _encode(self, role, source, pub):
        try:
            while not self.stop.is_set() or not source.empty():
                try:
                    record,image = source.get(timeout=.1)
                except queue.Empty:
                    continue
                ok, encoded = self.cv2.imencode('.jpg',image,[self.cv2.IMWRITE_JPEG_QUALITY,self.config['jpeg_quality']])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                pub.submit(record, encoded.tobytes())
                source.task_done()
        except Exception as exc:
            self.failure = role+':'+str(exc)
            pub.submit({'kind':'failure','receive_monotonic_ns':time.monotonic_ns(),'error':self.failure})
            self.stop.set()

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=3)
        for pipe in self.pipelines:
            pipe.stop()
        if self.video:
            self.video.close()
        if self.video_publisher:
            self.video_publisher.close()
        for pub in self.publishers:
            pub.close()


def identify(config, output, seconds=3):
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    output.mkdir(parents=True,exist_ok=False)
    ctx = rs.context()
    records = []
    for device in ctx.query_devices():
        info = device_info(device,rs)
        serial = info['serial_number']
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
        pipe = rs.pipeline(ctx)
        pipe.start(cfg)
        try:
            deadline = time.monotonic()+seconds
            image = None
            while time.monotonic() < deadline:
                frame = pipe.wait_for_frames(1000).get_color_frame()
                if frame:
                    image = np.asanyarray(frame.get_data()).copy()
            if image is None:
                raise RuntimeError('no color frame: '+serial)
            cv2.putText(image,serial,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
            filename = serial+'.jpg'
            if not cv2.imwrite(str(output/filename),image):
                raise RuntimeError('snapshot write failed')
            records.append(dict(info,snapshot=filename,role=None))
        finally:
            pipe.stop()
    (output/'devices.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'index.html').write_text('<meta charset="utf-8"><h1>按真实安装位置确认角色</h1>'+''.join(
        '<p>'+r['serial_number']+'</p><img width="640" src="'+r['snapshot']+'">' for r in records))
    print(output/'index.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--identify',action='store_true')
    parser.add_argument('--enumerate',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--ipc')
    parser.add_argument('--session-id')
    parser.add_argument('--video-config',type=Path)
    parser.add_argument('--status-path',type=Path)
    args = parser.parse_args()
    if args.enumerate:
        import pyrealsense2 as rs
        print(json.dumps([device_info(d,rs) for d in rs.context().query_devices()]))
        return 0
    config = load_camera_config(args.config, require_roles=not args.identify)
    with open('/tmp/pico-realsense-owner.lock','a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.identify:
            identify(config,args.output or Path(__file__).resolve().parents[2]/'data/camera-identify'/str(time.time_ns()))
            return 0
        if not args.ipc or not args.session_id:
            parser.error('--ipc and --session-id required for collection')
        video = json.loads(args.video_config.read_text())['video'] if args.video_config else None
        service = CameraService(config,args.ipc,args.session_id,video,args.status_path)
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,lambda *_:service.stop.set())
        try:
            service.start()
            while not service.stop.wait(.2):
                service.publish_display_status()
        finally:
            service.close()
        if service.failure:
            print(service.failure,file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

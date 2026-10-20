"""XRoboToolkit TCP H.264 adapter fed by the single camera owner.

Wire format follows pinned Orin sender: network-u32 length + H.264 access unit.
Control: network-u32 body length; little-endian command/data lengths; CA FE v1.
Only the pre-encoder raw-frame queue drops old frames. A slow TCP link is reset
with a fresh encoder, never resumed by replaying an accumulated video backlog.
"""
import json
from pathlib import Path
import socket
import struct
import threading
import time


def recv_exact(sock, count, stop=None):
    result = bytearray()
    deadline = time.monotonic()+2
    while len(result)<count:
        try:
            chunk = sock.recv(count-len(result))
        except socket.timeout:
            if stop and stop.is_set():
                raise EOFError('video control stopped')
            if result and time.monotonic() >= deadline:
                raise EOFError('video control idle timeout')
            continue
        if not chunk:
            raise EOFError('video control disconnected')
        result.extend(chunk)
    return bytes(result)


def parse_control(body):
    if len(body)<8:
        raise ValueError('short control body')
    size = struct.unpack_from('<I',body)[0]
    if size>128 or 4+size+4>len(body):
        raise ValueError('invalid command length')
    command = body[4:4+size].decode().rstrip('\0')
    data_size = struct.unpack_from('<I',body,4+size)[0]
    data = body[8+size:]
    if data_size!=len(data):
        raise ValueError('invalid control payload length')
    return command,data


def parse_camera_request(data):
    if len(data)<33 or data[:3]!=b'\xca\xfe\x01':
        raise ValueError('unsupported camera request header')
    values = struct.unpack_from('<7i',data,3)
    offset = 31
    strings = []
    for _ in range(2):
        length = data[offset]
        offset+=1
        if offset+length>len(data):
            raise ValueError('short camera request string')
        strings.append(data[offset:offset+length].decode())
        offset+=length
    return dict(zip(('width','height','fps','bitrate','mv_hevc','render_mode','port'),values),
                camera=strings[0],ip=strings[1])


def make_pipeline(width,height,fps,bitrate=4000000,input_width=None,input_height=None):
    import gi
    gi.require_version('Gst','1.0')
    from gi.repository import Gst
    Gst.init(None)
    input_width,input_height = input_width or width,input_height or height
    description = (
        'appsrc name=source is-live=true block=false format=time do-timestamp=true '
        'max-bytes=%d caps=video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ! '
        'queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
        'videoconvert ! video/x-raw,format=I420 ! nvvidconv ! '
        'video/x-raw(memory:NVMM),format=NV12,width=%d,height=%d ! '
        'nvv4l2h264enc maxperf-enable=true bitrate=%d insert-sps-pps=true '
        'iframeinterval=30 idrinterval=30 num-B-Frames=0 ! '
        'h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au ! '
        'appsink name=encoded sync=false max-buffers=1 drop=false'
    ) % (input_width*input_height*3*2,input_width,input_height,fps,width,height,bitrate)
    pipe = Gst.parse_launch(description)
    if pipe.set_state(Gst.State.PLAYING)==Gst.StateChangeReturn.FAILURE:
        raise RuntimeError('hardware H.264 pipeline failed to start')
    return Gst,pipe,pipe.get_by_name('source'),pipe.get_by_name('encoded')


class WristVideoSender:
    def __init__(self,latest,config,status_path=None):
        self.latest,self.config,self.status_path = latest,config,status_path
        self.stop = threading.Event()
        self.stream_stop = threading.Event()
        self.thread = self.stream_thread = None
        self.status = {'connected':False,'listening':False,'error':None,'sent_frames':0,
            'error_history':[],
            'projection_verified':bool(config.get('operator_two_eye_panel_verified',False)),
            'profile':{k:config[k] for k in ('width','height','fps')},
            'control_requests':0,'stream_connections':0}

    def _error(self,exc):
        error=str(exc)
        self.status.update(connected=False,error=error)
        history=self.status['error_history']
        if not history or history[-1]['error']!=error:
            self.status['error_history']=(history+[{'error':error,'monotonic_ns':time.monotonic_ns()}])[-8:]

    def start(self):
        self.thread = threading.Thread(target=self._run_listener,name='wrist-video-control',daemon=True)
        self.thread.start()

    def _run_listener(self):
        try:
            self._listen()
        except Exception as exc:
            self._error(exc)

    def _listen(self):
        if self.config.get('mode','listen')=='direct':
            self._stream(self.config['headset_ip'],self.config['port'])
            return
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind((self.config.get('bind_ip','0.0.0.0'),self.config.get('listen_port',13579)))
            server.listen(1)
            server.settimeout(.2)
            self.status['listen_port'] = server.getsockname()[1]
            self.status['listening'] = True
            while not self.stop.is_set():
                try:
                    client,peer = server.accept()
                except socket.timeout:
                    continue
                with client:
                    client.settimeout(.5)
                    try:
                        while not self.stop.is_set():
                            try:
                                size = struct.unpack('!I',recv_exact(client,4,self.stop))[0]
                            except socket.timeout:
                                continue
                            if size>65536:
                                raise ValueError('oversize video control body')
                            command,data = parse_control(recv_exact(client,size,self.stop))
                            if command=='OPEN_CAMERA':
                                request = parse_camera_request(data)
                                self.status.update(last_request=request,peer=peer[0],
                                    control_requests=self.status['control_requests']+1)
                                if request['ip']!=peer[0] or self.config.get('headset_ip') not in (None,peer[0]):
                                    raise ValueError('video peer/IP mismatch')
                                profiles = self.config.get('decoder_profiles',[
                                    dict(camera='ZED',**{k:self.config[k] for k in ('width','height','fps')})])
                                if request['mv_hevc'] or not any(all(request[k]==profile[k]
                                        for k in ('camera','width','height','fps')) for profile in profiles):
                                    raise ValueError('APK decoder profile differs from configured width/height/fps: '+str(request))
                                if not 1<=request['port']<=65535:
                                    raise ValueError('invalid video port')
                                self._stop_stream()
                                self.stream_stop = threading.Event()
                                self.status.update(error=None,profile={k:request[k] for k in ('width','height','fps')})
                                self.stream_thread = threading.Thread(target=self._stream,args=(peer[0],request['port'],request),daemon=True)
                                self.stream_thread.start()
                            elif command=='CLOSE_CAMERA':
                                self._stop_stream()
                    except (OSError,EOFError,ValueError,IndexError) as exc:
                        self._error(exc)
                    finally:
                        self._stop_stream()

    def _panel(self,width,height):
        import cv2
        import numpy as np
        views = self.latest()
        now = time.monotonic_ns()
        panels = []
        self.status['camera_age_ms'] = {role:round((now-views[role][0])/1e6,1)
            if role in views else None for role in ('left_wrist','right_wrist')}
        for role in ('left_wrist','right_wrist'):
            stamp,image = views.get(role,(0,None))
            if image is None or now-stamp>self.config.get('stale_ms',150)*1e6:
                panel = np.zeros((height,width//2,3),np.uint8)
                cv2.putText(panel,'VIDEO STALE',(20,70),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
            else:
                # Preserve the 4:3 camera view inside each 16:9 APK half.
                scale = min((width//2)/image.shape[1],height/image.shape[0])
                resized = cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
                panel = np.zeros((height,width//2,3),np.uint8)
                y,x = (height-resized.shape[0])//2,(width//2-resized.shape[1])//2
                panel[y:y+resized.shape[0],x:x+resized.shape[1]] = resized
            cv2.putText(panel,role,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
            panels.append(panel)
        image = np.hstack(panels)
        text = 'STATUS UNKNOWN'
        try:
            state = json.loads(Path(self.status_path).read_text())
            if now-state['time_monotonic_ns']<500_000_000:
                text = ('REC '+str(state['episode_id'])[:8]+' %.1fs'%((now-state['started_ns'])/1e9)
                    if state['recording'] else 'IDLE')
                if state.get('writer_error'):
                    text += ' RECORDING ERROR'
                elif not state.get('ready'):
                    text += ' NOT READY'
        except (TypeError,OSError,ValueError,KeyError):
            pass
        cv2.putText(image,text,(12,height-14),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        return image

    def _stream(self,ip,port,profile=None):
        profile = profile or self.config
        width,height,fps = (profile[k] for k in ('width','height','fps'))
        # Composite once at camera-sized resolution; VIC scales to the APK's
        # decoder dimensions. Upscaling BGR in Python wastes control CPU time.
        scale = min(1,1280/width)
        input_width = int(width*scale)//2*2
        input_height = int(height*scale)//2*2
        stream_stop = self.stream_stop
        while not self.stop.is_set() and not stream_stop.is_set():
            pipe = None
            try:
                with socket.create_connection((ip,port),timeout=.3) as network:
                    network.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                    network.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,128*1024)
                    network.settimeout(self.config.get('send_timeout_ms',150)/1000)
                    Gst,pipe,source,sink = make_pipeline(width,height,fps,self.config.get('bitrate',4000000),
                        input_width,input_height)
                    self.status.update(connected=True,error=None,
                        stream_connections=self.status['stream_connections']+1)
                    tick = time.monotonic()
                    while not self.stop.is_set() and not stream_stop.is_set():
                        image = self._panel(input_width,input_height)
                        data = image.tobytes()
                        buffer = Gst.Buffer.new_allocate(None,len(data),None)
                        buffer.fill(0,data)
                        if source.emit('push-buffer',buffer)!=Gst.FlowReturn.OK:
                            raise RuntimeError('appsrc push failed')
                        sample = sink.emit('try-pull-sample',100_000_000)
                        if sample:
                            encoded = sample.get_buffer()
                            ok,mapped = encoded.map(Gst.MapFlags.READ)
                            if not ok:
                                raise RuntimeError('H.264 map failed')
                            send_start = time.monotonic_ns()
                            try:
                                network.sendall(struct.pack('!I',len(mapped.data))+mapped.data)
                            finally:
                                encoded.unmap(mapped)
                            self.status['sent_frames']+=1
                            self.status['last_sent_monotonic_ns'] = time.monotonic_ns()
                            self.status['last_send_duration_ms'] = round((time.monotonic_ns()-send_start)/1e6,2)
                            self.status['largest_send_duration_ms'] = max(self.status.get('largest_send_duration_ms',0),
                                self.status['last_send_duration_ms'])
                        else:
                            message = pipe.get_bus().pop_filtered(Gst.MessageType.ERROR)
                            if message:
                                raise RuntimeError(str(message.parse_error()))
                        tick += 1/fps
                        if tick < time.monotonic()-1/fps:
                            tick = time.monotonic()
                        stream_stop.wait(max(0,tick-time.monotonic()))
            except Exception as exc:
                self._error(exc)
            finally:
                if pipe:
                    pipe.set_state(Gst.State.NULL)
                self.status['connected'] = False
            stream_stop.wait(.3)

    def _stop_stream(self):
        self.stream_stop.set()
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
            self.stream_thread = None

    def close(self):
        self.stop.set()
        self._stop_stream()
        if self.thread:
            self.thread.join(timeout=2)
        self.status['listening'] = False

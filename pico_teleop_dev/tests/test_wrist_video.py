"""Native APK control over real TCP, without opening cameras or robot SDKs."""
from pathlib import Path
import socket
import struct
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'camera'))
from wrist_video import WristVideoSender, parse_control, parse_camera_request


def message(command, width=2560, ip='127.0.0.1'):
    # BinaryWriter field order from pinned v1.1.1 CameraRequestSerializer.cs.
    request = b'\xca\xfe\x01'+struct.pack('<7i',width,720,60,4000000,0,2,12345)
    for value in ('ZED',ip):
        encoded=value.encode(); request+=bytes([len(encoded)])+encoded
    data=request if command=='OPEN_CAMERA' else b''
    command=command.encode()
    body=struct.pack('<I',len(command))+command+struct.pack('<I',len(data))+data
    return struct.pack('!I',len(body))+body


class VideoProtocolTests(unittest.TestCase):
    def setUp(self):
        self.sender=WristVideoSender(lambda:{},{'width':2560,'height':720,'fps':60,
            'mode':'listen','listen_port':0,'bind_ip':'127.0.0.1'})
        self.started=threading.Event(); self.stopped=threading.Event()
        def stream(ip,port,profile=None):
            self.started.set()
            self.sender.stream_stop.wait(3)
            self.stopped.set()
        self.patch=mock.patch.object(self.sender,'_stream',side_effect=stream)
        self.patch.start(); self.sender.start()
        deadline=time.monotonic()+2
        while not self.sender.status.get('listening') and time.monotonic()<deadline:
            time.sleep(.005)
        self.assertTrue(self.sender.status['listening'])
        self.client=socket.create_connection(('127.0.0.1',self.sender.status['listen_port']),1)

    def tearDown(self):
        self.client.close(); self.sender.close(); self.patch.stop()

    def test_fragmented_native_open_close_and_peer_disconnect(self):
        packet=message('OPEN_CAMERA')
        for offset in range(0,len(packet),3):
            self.client.sendall(packet[offset:offset+3])
        self.assertTrue(self.started.wait(1))
        self.assertEqual(self.sender.status['last_request']['fps'],60)
        self.client.sendall(message('CLOSE_CAMERA'))
        self.assertTrue(self.stopped.wait(1))
        self.started.clear(); self.stopped.clear()
        self.client.sendall(message('OPEN_CAMERA'))
        self.assertTrue(self.started.wait(1))
        self.client.close()
        self.assertTrue(self.stopped.wait(1))

    def test_profile_mismatch_visible_and_never_starts_stream(self):
        self.client.sendall(message('OPEN_CAMERA',width=1280))
        deadline=time.monotonic()+1
        while not self.sender.status['error'] and time.monotonic()<deadline:
            time.sleep(.005)
        self.assertIn('profile differs',self.sender.status['error'])
        self.assertFalse(self.started.is_set())

    def test_forged_video_destination_rejected(self):
        self.client.sendall(message('OPEN_CAMERA',ip='192.0.2.1'))
        deadline=time.monotonic()+1
        while not self.sender.status['error'] and time.monotonic()<deadline:
            time.sleep(.005)
        self.assertIn('peer/IP mismatch',self.sender.status['error'])
        self.assertFalse(self.started.is_set())

    def test_pico4u_profile_negotiated_without_reusing_zed_decoder_dimensions(self):
        self.sender.config['decoder_profiles']=[{'camera':'VR','width':2160,'height':810,'fps':60}]
        packet=message('OPEN_CAMERA').replace(struct.pack('<i',2560),struct.pack('<i',2160),1)
        packet=packet.replace(struct.pack('<i',720),struct.pack('<i',810),1).replace(b'\x03ZED',b'\x02VR',1)
        packet=struct.pack('!I',len(packet)-4)+packet[4:]
        command_size=len('OPEN_CAMERA')
        data_offset=4+4+command_size
        packet=packet[:data_offset]+struct.pack('<I',len(packet)-data_offset-4)+packet[data_offset+4:]
        self.client.sendall(packet)
        self.assertTrue(self.started.wait(1))
        self.assertEqual(self.sender.status['profile'],{'width':2160,'height':810,'fps':60})

    def test_truncated_or_oversized_request_rejected(self):
        with self.assertRaises((ValueError,IndexError)):
            parse_camera_request(b'\xca\xfe\x01'+bytes(29))
        with self.assertRaises(ValueError):
            parse_control(struct.pack('<I',500)+bytes(16))


if __name__=='__main__': unittest.main()

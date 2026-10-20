"""Pure task-episode/button state machine, independent of Grip and robot APIs."""
import time


class ButtonCursor:
    def __init__(self):
        self.receiver_id = None
        self.cursor = 0
        self.generation = None
        self.armed = {k:False for k in ('A','X','B','Y')}

    def request(self):
        return {'command':'button_events', 'receiver_id':self.receiver_id, 'cursor':self.cursor}

    def consume(self, reply):
        generation_changed = self.generation is not None and reply['connection_generation'] != self.generation
        reset = reply.get('reset') or reply.get('overflow') or reply['receiver_id'] != self.receiver_id or generation_changed
        self.receiver_id = reply['receiver_id']
        self.generation = reply['connection_generation']
        self.cursor = reply['head']
        if reset:
            self.armed = {k:reply['valid'].get(k) is True and reply['buttons'].get(k) is False
                          for k in self.armed}
            return [], ('EVENT_OVERFLOW' if reply.get('overflow') else
                        'CONNECTION_CHANGED' if generation_changed else 'EVENT_BASELINE')
        result = []
        for event in reply['events']:
            pressed = []
            for name in self.armed:
                valid, down = event['valid'].get(name), event['buttons'].get(name)
                if not valid:
                    self.armed[name] = False
                elif down is False:
                    self.armed[name] = True
                elif name in event['pressed']:
                    if self.armed[name]:
                        pressed.append(name)
                    self.armed[name] = False
            result.append(dict(event, pressed=pressed))
        return result, None


class CollectionSession:
    def __init__(self, writer, task, stop, clock_ns=time.monotonic_ns, max_event_age_ms=150):
        self.writer, self.task, self.stop, self.clock_ns = writer, task, stop, clock_ns
        self.max_event_age_ns = int(max_event_age_ms*1e6)
        self.active_episode_id = self.deletable_episode_id = None
        self.number = 0
        self.exiting = False
        self.message = '未录制'
        self.started_ns = None

    def _log(self, input_event, **extra):
        self.writer.event(dict(input_event, **extra))

    def handle_batch(self, events, ready=True, readiness_reason=None):
        if self.exiting or not events:
            return
        if any(e.get('exit_chord') for e in events):
            event = next(e for e in events if e.get('exit_chord'))
            self.exit(event, 'B_Y_CHORD')
            return
        # A takes precedence over X across the complete receiver poll batch.
        has_a = any('A' in e['pressed'] for e in events)
        for event in events:
            if event.get('valid') and not all(event['valid'].values()) and self.active_episode_id:
                self.interrupt('PICO_FOCUS_OR_BUTTON_VALIDITY_LOST',event=event)
            self.writer.submit(dict(event, source='pico_events',
                                    source_sequence=event['event_sequence']))
            self._log(event, event='pico_input')
            if not 0 <= self.clock_ns()-event['receive_monotonic_ns'] <= self.max_event_age_ns:
                self._log(event, event='stale_command_ignored')
                continue
            if 'A' in event['pressed']:
                if self.active_episode_id:
                    episode_id = self.active_episode_id
                    if not self.writer.end(episode_id, event):
                        self.message = '录制关闭失败，原始数据保留待检查'
                    else:
                        self.message = '已结束并保留 '+episode_id[:8]
                    self.deletable_episode_id, self.active_episode_id = episode_id, None
                    self.started_ns = None
                elif ready and self.writer.ready:
                    episode_id = self.writer.begin(event, self.number+1, self.task)
                    if episode_id:
                        self.number += 1
                        self.deletable_episode_id = None
                        self.active_episode_id, self.started_ns = episode_id, event['receive_monotonic_ns']
                        self.message = '正在录制 '+episode_id[:8]
                    else:
                        self.message = '未开始录制：后台收尾资源不足或写入器不可用'
                else:
                    self.message = '未开始录制：'+str(readiness_reason or self.writer.error or '采集源未就绪')
                self._log(event, event='record_toggle', message=self.message,
                          episode_id=self.active_episode_id or self.deletable_episode_id)
            elif 'X' in event['pressed'] and not has_a:
                if self.active_episode_id:
                    self.message = '正在录制，X无效'
                elif self.deletable_episode_id:
                    target = self.deletable_episode_id
                    if self.writer.delete(target, event):
                        self.deletable_episode_id = None
                        self.message = '已移入回收目录 '+target[:8]
                    else:
                        self.message = '删除失败：写入器事件队列不可用'
                else:
                    self.message = '没有可删除的新记录'
                self._log(event, event='delete_button', message=self.message)

    def interrupt(self, reason, failed=False, event=None):
        event = event or {'receive_monotonic_ns':self.clock_ns()}
        episode_id = self.active_episode_id
        if self.active_episode_id:
            self.writer.end(episode_id, event, reason=reason, interrupted=not failed, failed=failed)
            self.active_episode_id = None
            self.deletable_episode_id = episode_id
            self.started_ns = None
        self.message = '录制中断：'+reason
        self._log(event, event='recording_interrupted', reason=reason,episode_id=episode_id)

    def exit(self, event=None, reason='SIGNAL'):
        if self.exiting:
            return
        self.exiting = True
        self.stop()  # latch and signal robots BEFORE writer or image cleanup
        if self.active_episode_id:
            self.interrupt(reason, event=event)
        else:
            self.message = '会话已结束：'+reason
            self._log(event or {'receive_monotonic_ns':self.clock_ns()},
                      event='session_exit', reason=reason)

    def status(self):
        return {'recording':bool(self.active_episode_id), 'episode_id':self.active_episode_id,
                'episode_number':self.number,
                'deletable_episode_id':self.deletable_episode_id, 'message':self.message,
                'started_ns':self.started_ns, 'exiting':self.exiting}

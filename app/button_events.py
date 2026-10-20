"""Bounded receiver-side button history. No robot APIs or filesystem work.

Caller serializes access with the receiver lock. A snapshot's poses remain
latest-only; this stream preserves changes that occur between control ticks.
"""
import collections
import uuid

BUTTONS = {'A': ('right', 'primaryButton'), 'X': ('left', 'primaryButton'),
           'B': ('right', 'secondaryButton'), 'Y': ('left', 'secondaryButton')}


class ButtonEventRing:
    def __init__(self, capacity=512, chord_window_ms=200, max_gap_ms=150):
        self.receiver_id = str(uuid.uuid4())
        self.events = collections.deque(maxlen=capacity)
        self.sequence = self.generation = self.exit_sequence = 0
        self.window_ns = int(chord_window_ms * 1e6)
        self.max_gap_ns = int(max_gap_ms * 1e6)
        self.last_ns = self.device_id = None
        self.buttons = {k: None for k in BUTTONS}
        self.valid = {k: False for k in BUTTONS}
        self.armed = {k: False for k in BUTTONS}
        self.down_ns = {}
        self.chord_latched = False

    def _append(self, now_ns, pico_sequence, reason, pressed=(), released=()):
        self.sequence += 1
        event = {'receiver_id': self.receiver_id, 'connection_generation': self.generation,
                 'event_sequence': self.sequence, 'pico_sequence': pico_sequence,
                 'receive_monotonic_ns': now_ns, 'buttons': self.buttons.copy(),
                 'valid': self.valid.copy(), 'pressed': list(pressed),
                 'released': list(released), 'reason': reason, 'exit_chord': False}
        if not self.chord_latched and all(self.buttons[k] is True and self.valid[k]
                                         and k in self.down_ns for k in ('B', 'Y')):
            if abs(self.down_ns['B'] - self.down_ns['Y']) <= self.window_ns:
                self.exit_sequence = self.sequence
                self.chord_latched = True
                event['exit_chord'] = True
        self.events.append(event)

    def invalidate(self, now_ns, reason='DISCONNECTED'):
        self.generation += 1
        self.buttons = {k: None for k in BUTTONS}
        self.valid = {k: False for k in BUTTONS}
        self.armed = {k: False for k in BUTTONS}
        self.down_ns.clear()
        self.chord_latched = False
        self.last_ns = None
        self._append(now_ns, None, reason)

    def observe(self, sample):
        now = sample['receive_monotonic_ns']
        dev = sample.get('device_id')
        if self.last_ns is not None and (now <= self.last_ns or
                now-self.last_ns > self.max_gap_ns or dev != self.device_id):
            self.invalidate(now, 'INPUT_DISCONTINUITY')
        self.last_ns, self.device_id = now, dev
        pressed, released, changed = [], [], False
        for name, (side, field) in BUTTONS.items():
            ctrl = sample.get(side) or {}
            value = ctrl.get(field)
            valid = (sample.get('app_focus') is True and sample.get('input_mode') == 1
                     and ctrl.get('status') in (None, 3) and type(value) is bool)
            if valid != self.valid[name] or value != self.buttons[name]:
                changed = True
            if not valid:
                self.armed[name] = False
                self.down_ns.pop(name, None)
            elif value is False:
                if self.valid[name] and self.buttons[name] is True:
                    released.append(name)
                self.armed[name] = True
                self.down_ns.pop(name, None)
            elif self.armed[name] and self.buttons[name] is not True:
                pressed.append(name)
                self.armed[name] = False
                self.down_ns[name] = now
            self.buttons[name], self.valid[name] = value if valid else None, valid
        if not all(self.buttons[k] is True for k in ('B', 'Y')):
            self.chord_latched = False
        if changed:
            self._append(now, sample.get('sequence'), 'BUTTON_CHANGE', pressed, released)

    def summary(self):
        return {'schema': 'pico_button_events_v1', 'receiver_id': self.receiver_id,
                'connection_generation': self.generation, 'head': self.sequence,
                'oldest': self.events[0]['event_sequence'] if self.events else self.sequence+1,
                'exit_sequence': self.exit_sequence, 'buttons': self.buttons.copy(),
                'valid': self.valid.copy(), 'chord_window_ms': self.window_ns/1e6}

    def read(self, cursor, receiver_id=None):
        summary = self.summary()
        reset = receiver_id != self.receiver_id
        overflow = cursor < summary['oldest']-1 or cursor > self.sequence
        return {**summary, 'event_watermark_ns':self.last_ns,
                'overflow': overflow, 'reset': reset,
                'events': [] if reset or overflow else [e for e in self.events
                                                       if e['event_sequence'] > cursor]}


class ExitGuard:
    """Latch only chords newer than this program's startup snapshot."""
    def __init__(self):
        self.receiver_id = self.baseline = None
        self.requested = False

    def observe(self, status):
        info = status.get('button_stream') or {}
        receiver_id = info.get('receiver_id')
        if receiver_id is None:
            return self.requested
        if receiver_id != self.receiver_id:
            self.receiver_id, self.baseline = receiver_id, info['exit_sequence']
        elif info['exit_sequence'] > self.baseline:
            self.requested = True
        return self.requested

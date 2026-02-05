from collections import deque


class FrameBuffer:
    def __init__(self, maxlen):
        self.buf = deque(maxlen=maxlen)

    def push(self, ts, frame):
        self.buf.append((ts, frame))

    def clip(self, start_ts, end_ts):
        return [(t, f) for (t, f) in list(self.buf) if start_ts <= t <= end_ts]


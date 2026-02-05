from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    if not np.all(np.isfinite(bb_test)) or not np.all(np.isfinite(bb_gt)):
        return 0.0
    if bb_test[2] <= bb_test[0] or bb_test[3] <= bb_test[1]:
        return 0.0
    if bb_gt[2] <= bb_gt[0] or bb_gt[3] <= bb_gt[1]:
        return 0.0
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    denom = ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
             (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh + 1e-6)
    if denom <= 0:
        return 0.0
    o = wh / denom
    return float(max(0.0, min(1.0, o)))


@dataclass
class KalmanBox:
    x: np.ndarray
    P: np.ndarray


def create_kalman(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    s = w * h
    r = w / (h + 1e-6)
    x = np.array([cx, cy, s, r, 0., 0., 0.])
    P = np.eye(7) * 10.0
    return KalmanBox(x=x, P=P)


def kf_predict(kf: KalmanBox):
    F = np.eye(7)
    dt = 1.0
    F[0, 4] = dt
    F[1, 5] = dt
    F[2, 6] = dt
    Q = np.eye(7) * 0.01
    kf.x = F @ kf.x
    kf.P = F @ kf.P @ F.T + Q


def kf_update(kf: KalmanBox, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    s = w * h
    r = w / (h + 1e-6)
    z = np.array([cx, cy, s, r])
    H = np.zeros((4, 7))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    H[3, 3] = 1
    R = np.eye(4) * 0.1
    y = z - H @ kf.x
    S = H @ kf.P @ H.T + R
    K = kf.P @ H.T @ np.linalg.inv(S)
    kf.x = kf.x + K @ y
    kf.P = (np.eye(7) - K @ H) @ kf.P


def to_bbox(kf: KalmanBox):
    cx, cy, s, r = kf.x[:4]
    s = float(np.clip(s, 1.0, 1e7))
    r = float(np.clip(abs(r), 0.1, 10.0))
    w = float(np.sqrt(max(1e-6, s * r)))
    h = float(max(1e-6, s / w))
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.array([x1, y1, x2, y2], dtype=float)


class Track:
    _next_id = 1

    def __init__(self, bbox):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = create_kalman(bbox)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    def predict(self):
        kf_predict(self.kf)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return to_bbox(self.kf)

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        kf_update(self.kf, bbox)

    def bbox(self):
        return to_bbox(self.kf)


class SORT:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    def update(self, dets):
        for t in self.tracks:
            t.predict()

        N = len(self.tracks)
        M = len(dets)
        if N == 0 or M == 0:
            matches = []
            unmatched_dets = list(range(M))
        else:
            iou_mat = np.zeros((N, M), dtype=np.float32)
            for i, t in enumerate(self.tracks):
                tb = t.bbox()
                for j, d in enumerate(dets):
                    iou_mat[i, j] = iou(tb, d)
            iou_mat = np.nan_to_num(iou_mat, nan=0.0, posinf=0.0, neginf=0.0)
            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            matches = []
            unmatched_tracks = set(range(N))
            unmatched_dets = set(range(M))
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] < self.iou_threshold:
                    continue
                matches.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)
            unmatched_dets = list(unmatched_dets)

        for r, c in matches:
            self.tracks[r].update(dets[c])

        for idx in unmatched_dets:
            self.tracks.append(Track(dets[idx]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for t in self.tracks:
            bb = t.bbox()
            results.append(np.concatenate([bb, [t.id]], axis=0))
        return np.array(results) if results else np.zeros((0, 5))

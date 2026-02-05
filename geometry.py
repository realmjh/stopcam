import cv2
import numpy as np


def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(np.asarray(poly, dtype=np.int32), (int(pt[0]), int(pt[1])), False) >= 0


def seg_cross(p0, p1, a, b):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p0, a, b) != ccw(p1, a, b) and ccw(p0, p1, a) != ccw(p0, p1, b)


def draw_overlays(frame, cfg):
    if 'approach_roi' in cfg and cfg['approach_roi']:
        cv2.polylines(frame, [np.asarray(cfg['approach_roi'], np.int32)], True, (0, 255, 255), 1)
    if 'stop_zone' in cfg and cfg['stop_zone']:
        cv2.polylines(frame, [np.asarray(cfg['stop_zone'], np.int32)], True, (0, 255, 0), 2)
    if 'line_segment' in cfg and cfg['line_segment']:
        a, b = cfg['line_segment']
        cv2.line(frame, tuple(a), tuple(b), (0, 0, 255), 2)
    return frame


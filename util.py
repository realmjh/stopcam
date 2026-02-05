import os
import yaml
import cv2


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def open_video(video=None, rtsp=None):
    src = video if video else rtsp
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {src}")
    return cap


def write_clip(frames, out_path, fps=30, transform=None):
    if not frames:
        return False
    first = frames[0][1] if isinstance(frames[0], (tuple, list)) else frames[0]
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for item in frames:
        fr = item[1] if isinstance(item, (tuple, list)) else item
        if transform is not None:
            fr = transform(fr)
        vw.write(fr)
    vw.release()
    return True

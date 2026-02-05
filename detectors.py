import cv2
import numpy as np


class BackgroundDetector:

    def __init__(self, min_area=500, dilate_iter=2, var_threshold=16):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=var_threshold, detectShadows=True)
        self.min_area = min_area
        self.dilate_iter = dilate_iter

    def detect(self, frame, mask=None):
        fg = self.bg.apply(frame)
        if mask is not None:
            fg = cv2.bitwise_and(fg, mask)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        fg = cv2.dilate(fg, None, iterations=self.dilate_iter)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area:
                continue
            boxes.append([x, y, x + w, y + h])
        return np.array(boxes, dtype=np.float32)


class YOLODetector:

    def __init__(self, model_name="yolov8n.pt", conf=0.25, classes=(2, 3, 5, 7)):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("ultralytics not installed: pip install ultralytics") from e
        self.model = YOLO(model_name)
        self.conf = conf
        self.classes = set(classes)

    def detect(self, frame):
        res = self.model(frame, conf=self.conf, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        sel = [i for i, c in enumerate(cls) if c in self.classes]
        return boxes[sel] if len(sel) else np.zeros((0, 4))

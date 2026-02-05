import cv2


class PrivacyBlurrer:
    def __init__(self, blur_faces=True, blur_plates=True, ksize=25):
        self.blur_faces = blur_faces
        self.blur_plates = blur_plates
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1

        base = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(base + 'haarcascade_frontalface_default.xml') if blur_faces else None
        self.plate_cascade = cv2.CascadeClassifier(base + 'haarcascade_russian_plate_number.xml') if blur_plates else None

    def __call__(self, frame):
        if not (self.blur_faces or self.blur_plates):
            return frame
        out = frame.copy()
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        regions = []
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                regions.append((x, y, w, h))
        if self.plate_cascade is not None and not self.plate_cascade.empty():
            plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in plates:
                regions.append((x, y, w, h))
        for (x, y, w, h) in regions:
            roi = out[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi = cv2.GaussianBlur(roi, (self.ksize, self.ksize), 0)
            out[y:y+h, x:x+w] = roi
        return out

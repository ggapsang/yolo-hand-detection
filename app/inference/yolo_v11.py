import time

import cv2
import numpy as np
import onnxruntime as ort

from app.inference.base import AbstractDetector


class YOLOv11(AbstractDetector):

    def __init__(self, model_path, labels, size=640, confidence=0.25, threshold=0.45):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size
        self.labels = labels

        preferred = ['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        available = ort.get_available_providers()
        providers = [p for p in preferred if p in available] or ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._provider = self.session.get_providers()[0]

    @property
    def provider(self) -> str:
        return self._provider

    def inference_from_file(self, file):
        mat = cv2.imread(file)
        return self.inference(mat)

    def inference(self, image: np.ndarray) -> tuple:
        ih, iw = image.shape[:2]

        img = cv2.resize(image, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        start = time.time()
        outputs = self.session.run(None, {self.input_name: img})
        inference_time = time.time() - start

        # YOLOv8/v11 output: [1, 4+num_classes, num_anchors]
        # squeeze batch dim → [4+num_classes, num_anchors], transpose → [num_anchors, 4+num_classes]
        output = outputs[0].squeeze(0).T
        x_scale = iw / self.size
        y_scale = ih / self.size

        boxes, confidences, class_ids = [], [], []

        for row in output:
            scores = row[4:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > self.confidence:
                cx, cy, w, h = row[:4]
                x = int((cx - w / 2) * x_scale)
                y = int((cy - h / 2) * y_scale)
                boxes.append([x, y, int(w * x_scale), int(h * y_scale)])
                confidences.append(float(conf))
                class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                results.append((class_ids[i], self.labels[class_ids[i]], confidences[i], x, y, w, h))

        return iw, ih, inference_time, results

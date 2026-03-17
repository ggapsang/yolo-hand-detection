import time

import cv2
import numpy as np

from app.inference.base import AbstractDetector


class YOLODarknet(AbstractDetector):

    def __init__(self, config, model, labels, size=416, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size
        self.labels = labels
        self._output_names = []

        try:
            self.net = cv2.dnn.readNetFromDarknet(config, model)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        except Exception:
            raise ValueError(
                "Couldn't find the models!\nDid you forget to download them manually "
                "(and keep in the correct directory, models/) or run the shell script?"
            )

        ln = self.net.getLayerNames()
        for i in self.net.getUnconnectedOutLayers():
            self._output_names.append(ln[int(i) - 1])

    @property
    def provider(self) -> str:
        return "OpenCV/OpenCL"

    def inference_from_file(self, file):
        mat = cv2.imread(file)
        return self.inference(mat)

    def inference(self, image: np.ndarray) -> tuple:
        ih, iw = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layer_outputs = self.net.forward(self._output_names)
        inference_time = time.time() - start

        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence:
                    box = detection[0:4] * np.array([iw, ih, iw, ih])
                    cx, cy, w, h = box.astype("int")
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                results.append((class_ids[i], self.labels[class_ids[i]], confidences[i], x, y, w, h))

        return iw, ih, inference_time, results

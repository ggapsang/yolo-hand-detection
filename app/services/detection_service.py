import cv2
import numpy as np

from app.inference.base import AbstractDetector
from app.schemas.detection import BBox, Detection, DetectionResponse


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data: could not decode file")
    return image


def draw_detections(image: np.ndarray, results: list, max_hands: int = -1) -> np.ndarray:
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    count = len(sorted_results) if max_hands == -1 else max_hands
    for _, name, confidence, x, y, w, h in sorted_results[:count]:
        color = (0, 255, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{name} ({confidence:.2f})", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def run_detection(
    image: np.ndarray,
    detector: AbstractDetector,
    min_confidence: float | None,
    max_hands: int,
) -> DetectionResponse:
    width, height, inference_time, results = detector.inference(image)

    if min_confidence is not None:
        results = [r for r in results if r[2] >= min_confidence]

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    count = len(sorted_results) if max_hands == -1 else max_hands

    detections = [
        Detection(
            class_id=class_id,
            label=label,
            confidence=round(conf, 4),
            bbox=BBox(x=x, y=y, w=w, h=h),
        )
        for class_id, label, conf, x, y, w, h in sorted_results[:count]
    ]

    return DetectionResponse(
        width=width,
        height=height,
        inference_time=round(inference_time, 4),
        detections=detections,
    )


def annotate_image(
    image: np.ndarray,
    detector: AbstractDetector,
    min_confidence: float | None,
    max_hands: int,
) -> bytes:
    _, _, _, results = detector.inference(image)

    if min_confidence is not None:
        results = [r for r in results if r[2] >= min_confidence]

    draw_detections(image, results, max_hands)
    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes()

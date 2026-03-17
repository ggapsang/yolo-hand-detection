import tempfile
import uuid
from typing import Generator

import cv2

from app.inference.base import AbstractDetector
from app.services.detection_service import draw_detections

# stream_id → 임시 영상 파일 경로
_stream_registry: dict[str, str] = {}


def register_video(video_path: str) -> str:
    stream_id = uuid.uuid4().hex
    _stream_registry[stream_id] = video_path
    return stream_id


def get_video_path(stream_id: str) -> str | None:
    return _stream_registry.get(stream_id)


def _annotate_frame(frame, detector: AbstractDetector, min_confidence: float | None, max_hands: int):
    _, _, inference_time, results = detector.inference(frame)
    if min_confidence is not None:
        results = [r for r in results if r[2] >= min_confidence]
    draw_detections(frame, results, max_hands)
    fps = round(1 / inference_time, 1) if inference_time > 0 else 0
    cv2.putText(frame, f"{fps} FPS", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame


def process_video(
    video_path: str,
    detector: AbstractDetector,
    min_confidence: float | None,
    max_hands: int,
) -> str:
    """모든 프레임을 처리해 바운딩박스가 그려진 mp4 파일 경로를 반환한다."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _annotate_frame(frame, detector, min_confidence, max_hands)
        writer.write(frame)

    cap.release()
    writer.release()
    return out_path


def generate_mjpeg(
    video_path: str,
    detector: AbstractDetector,
    min_confidence: float | None,
    max_hands: int,
) -> Generator[bytes, None, None]:
    """프레임별로 처리하며 MJPEG 스트림을 yield한다."""
    cap = cv2.VideoCapture(video_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _annotate_frame(frame, detector, min_confidence, max_hands)
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()

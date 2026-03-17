from pathlib import Path

from app.inference.base import AbstractDetector
from app.inference.yolo_darknet import YOLODarknet
from app.inference.yolo_v11 import YOLOv11

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

_MODEL_CONFIGS = {
    "v11": {
        "cls": YOLOv11,
        "args": [str(MODELS_DIR / "hand_detect_yolov11.onnx"), ["hand"]],
    },
    "normal": {
        "cls": YOLODarknet,
        "args": [str(MODELS_DIR / "cross-hands.cfg"), str(MODELS_DIR / "cross-hands.weights"), ["hand"]],
    },
    "tiny": {
        "cls": YOLODarknet,
        "args": [str(MODELS_DIR / "cross-hands-tiny.cfg"), str(MODELS_DIR / "cross-hands-tiny.weights"), ["hand"]],
    },
    "prn": {
        "cls": YOLODarknet,
        "args": [str(MODELS_DIR / "cross-hands-tiny-prn.cfg"), str(MODELS_DIR / "cross-hands-tiny-prn.weights"), ["hand"]],
    },
    "v4-tiny": {
        "cls": YOLODarknet,
        "args": [str(MODELS_DIR / "cross-hands-yolov4-tiny.cfg"), str(MODELS_DIR / "cross-hands-yolov4-tiny.weights"), ["hand"]],
    },
}

NETWORK_CHOICES = list(_MODEL_CONFIGS.keys())


def create_detector(network: str, size: int, confidence: float, threshold: float) -> AbstractDetector:
    if network not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown network: '{network}'. Choose from {NETWORK_CHOICES}")

    config = _MODEL_CONFIGS[network]
    detector = config["cls"](*config["args"])
    detector.size = size
    detector.confidence = confidence
    detector.threshold = threshold
    return detector

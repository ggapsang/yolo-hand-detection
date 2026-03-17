# 하위호환: 기존 demo_*.py 스크립트가 동일하게 import할 수 있도록 re-export
from app.inference.yolo_darknet import YOLODarknet
from app.inference.yolo_v11 import YOLOv11

# 기존 코드와 완전히 호환되는 별칭
YOLO = YOLODarknet

__all__ = ["YOLO", "YOLODarknet", "YOLOv11"]

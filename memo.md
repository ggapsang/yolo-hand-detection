## demo~ 파일들 명령어 정리

**demo.py**

| 인자 | 전체 | 기본값 | 설명 |
|---|---|---|---|
| `-i` | `--images` | `"images"` | 이미지 폴더 경로 또는 목록 텍스트 파일 |
| `-n` | `--network` | `"v11"` | 네트워크 종류 (v11 / normal / tiny / prn / v4-tiny) |
| `-d` | `--device` | `0` | 사용 장치 번호 |
| `-s` | `--size` | `640` | YOLO 입력 이미지 크기 |
| `-c` | `--confidence` | `0.25` | 탐지 신뢰도 임계값 |

**demo_webcam.py**

| 인자 | 전체 | 기본값 | 설명 |
|---|---|---|---|
| `-n` | `--network` | `"v11"` | 네트워크 종류 (v11 / normal / tiny / prn / v4-tiny) |
| `-d` | `--device` | `0` | 웹캠 장치 번호 |
| `-s` | `--size` | `640` | YOLO 입력 이미지 크기 |
| `-c` | `--confidence` | `0.2` | 탐지 신뢰도 임계값 |
| `-nh` | `--hands` | `-1` | 프레임당 최대 탐지 손 개수 (-1은 전부) |

**demo_video.py**

| 인자 | 전체 | 기본값 | 설명 |
|---|---|---|---|
| `-v` | `--video` | 없음 (필수) | 영상 파일 경로 |
| `-n` | `--network` | `"v11"` | 네트워크 종류 (v11 / normal / tiny / prn / v4-tiny) |
| `-s` | `--size` | `640` | YOLO 입력 이미지 크기 |
| `-c` | `--confidence` | `0.2` | 탐지 신뢰도 임계값 |
| `-nh` | `--hands` | `-1` | 프레임당 최대 탐지 손 개수 (-1은 전부) |
| `-o` | `--output` | `None` | 파일 이름을 만들면 해당 경로로 분석 영상이 저장됨 |

---

## 네트워크 종류

| 옵션 | 모델 파일 | 클래스 | 비고 |
|---|---|---|---|
| `v11` | `models/hand_detect_yolov11.onnx` | hand | YOLOv11 파인튜닝, **기본값** |
| `normal` | `models/cross-hands.cfg/.weights` | hand | YOLOv3 |
| `tiny` | `models/cross-hands-tiny.cfg/.weights` | hand | YOLOv3-Tiny |
| `prn` | `models/cross-hands-tiny-prn.cfg/.weights` | hand | YOLOv3-Tiny-PRN, 작은 사이즈에서 잘 작동 |
| `v4-tiny` | `models/cross-hands-yolov4-tiny.cfg/.weights` | hand | YOLOv4-Tiny |

---

## 예시 명령어

```bash
# 웹캠 기본 실행 (YOLOv11, 640x640)
python demo_webcam.py

# 웹캠 - 기존 v4-tiny 모델, 빠른 속도용 (256x256)
python demo_webcam.py -n v4-tiny -s 256

# 웹캠 - 두 번째 카메라 장치 사용
python demo_webcam.py -d 1

# 웹캠 - 신뢰도 임계값 조정
python demo_webcam.py -c 0.4

# 영상 파일 분석 (YOLOv11 기본)
python demo_video.py -v sample.mp4

# 영상 파일 분석 후 결과 저장
python demo_video.py -v sample.mp4 -o output.mp4

# 영상 - 기존 v3 모델 사용
python demo_video.py -v sample.mp4 -n normal -s 416

# 이미지 폴더 분석 (YOLOv11)
python demo.py -i images/

# 이미지 분석 - 기존 v4-tiny 모델
python demo.py -i images/ -n v4-tiny -s 416

# 이미지 목록 파일로 분석
python demo.py -i filelist.txt -n v11
```

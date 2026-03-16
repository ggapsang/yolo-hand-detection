## demo~ 파일들 명령어 정리

**demo.py**

| 인자 | 전체 | 기본값 | 설명 |
|---|---|---|---|
| `-i` | `--images` | `"images"` | 이미지 폴더 경로 또는 목록 텍스트 파일 |
| `-n` | `--network` | `"normal"` | 네트워크 종류 (normal / tiny / prn / v4-tiny) |
| `-d` | `--device` | `0` | 사용 장치 번호 |
| `-s` | `--size` | `416` | YOLO 입력 이미지 크기 |
| `-c` | `--confidence` | `0.25` | 탐지 신뢰도 임계값 |

**demo_webcam.py**

| 인자 | 전체 | 기본값 | 설명 |
|---|---|---|---|
| `-n` | `--network` | `"normal"` | 네트워크 종류 (normal / tiny / prn / v4-tiny) |
| `-d` | `--device` | `0` | 웹캠 장치 번호 |
| `-s` | `--size` | `416` | YOLO 입력 이미지 크기 |
| `-c` | `--confidence` | `0.2` | 탐지 신뢰도 임계값 |
| `-nh` | `--hands` | `-1` | 프레임당 최대 탐지 손 개수 (-1은 전부) |

**demo_video.py**

| 인자 | 전체 | 기본값 | 설명 |
|---|---|---|---|
| `-v` | `--video` | 없음 (필수) | 영상 파일 경로 |
| `-n` | `--network` | `"normal"` | 네트워크 종류 (normal / tiny / prn / v4-tiny) |
| `-s` | `--size` | `416` | YOLO 입력 이미지 크기 |
| `-c` | `--confidence` | `0.2` | 탐지 신뢰도 임계값 |
| `-nh` | `--hands` | `-1` | 프레임당 최대 탐지 손 개수 (-1은 전부) |


## CUDA (GPU) 사용 시 설치 라이브러리

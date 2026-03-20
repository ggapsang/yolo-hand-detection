# Hand Detection API — QA 테스트 리스트

서버 기동 명령어:
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8911
# 또는 환경변수로 모델/파라미터 변경
$env:YOLO_NETWORK="v11"; $env:YOLO_SIZE="640"; uvicorn app.main:app --port 8911
```

Swagger UI: http://localhost:8911/docs

---

## 1. 서버 기동

| # | 항목 | 확인 방법 | 기대 결과 | 결과 |
|---|------|-----------|-----------|------|
| 1-1 | 정상 기동 | 콘솔 로그 확인 | `[startup] loading model: v11` 출력, provider 표시 | ■ |
| 1-2 | 모델 파일 로드 | 콘솔 로그 확인 | `hand_detect_yolov11.onnx` 로드 오류 없음 | ■ |
| 1-3 | GPU 가속 여부 | 콘솔 로그 확인 | provider가 `DmlExecutionProvider` 또는 `CUDAExecutionProvider`인지 확인 (없으면 CPU도 무방) | ■ |
| 1-4 | Swagger 접근 | 브라우저에서 `/docs` 열기 | Swagger UI 정상 표시 | ■ |

---

## 2. GET /api/v1/health

```bash
curl http://localhost:8911/api/v1/health
```

| # | 항목 | 확인 방법 | 기대 결과 | 결과 |
|---|------|-----------|-----------|------|
| 2-1 | 정상 응답 | curl 또는 브라우저 | HTTP 200, `{"status":"ok","model":"v11","provider":"..."}` | ■ |
| 2-2 | model 필드 | 응답 JSON 확인 | `"model": "v11"` | ■ |
| 2-3 | provider 필드 | 응답 JSON 확인 | provider 값이 비어있지 않음 | ■ |

---

## 3. POST /api/v1/detect (이미지 → JSON)

```bash
# 손이 있는 이미지
curl -X POST http://localhost:8911/api/v1/detect \
  -F "file=@images/hand.jpg"

# confidence 파라미터 지정
curl -X POST "http://localhost:8911/api/v1/detect?confidence=0.5" \
  -F "file=@images/hand.jpg"

# max_hands=1 제한
curl -X POST "http://localhost:8911/api/v1/detect?max_hands=1" \
  -F "file=@images/hand.jpg"

# 잘못된 파일 (텍스트 파일)
curl -X POST http://localhost:8911/api/v1/detect -F "file=@README.md"

# 빈 파일
echo $null > empty.bin
curl -X POST http://localhost:8911/api/v1/detect -F "file=@empty.bin"
```

| # | 항목 | 확인 방법 | 기대 결과 | 결과 |
|---|------|-----------|-----------|------|
| 3-1 | 정상 감지 (손 있는 이미지) | 응답 JSON | `detections` 배열에 1개 이상, `label: "hand"`, `confidence > 0` | ■ |
| 3-2 | 응답 스키마 | 응답 JSON 필드 | `width`, `height`, `inference_time`, `detections[].class_id/label/confidence/bbox(x,y,w,h)` 모두 포함 | ■ |
| 3-3 | 손 없는 이미지 | 응답 JSON | `detections: []`, HTTP 200 | ☐ |
| 3-4 | confidence 파라미터 필터링 | 높은 confidence 지정 후 비교 | 기본값 대비 `detections` 수 감소 또는 동일 | ☐ |
| 3-5 | max_hands=1 제한 | 응답 JSON | `detections` 길이가 1 이하 | ☐ |
| 3-6 | inference_time | 응답 JSON | 0보다 큰 float 값 | ■ |
| 3-7 | bbox 좌표 유효성 | 응답 JSON | x,y,w,h 모두 0 이상, 이미지 크기 초과 없음 | ■ |
| 3-8 | 잘못된 파일 (txt 등) | 응답 확인 | HTTP 400, `"Invalid image data"` 메시지 | ☐ |
| 3-9 | 빈 파일 전송 | 응답 확인 | HTTP 400 | ☐ |
| 3-10 | 대용량 이미지 (4K) | 응답 확인 | 오류 없이 처리, 응답 width/height가 원본 일치 | ☐ |

---

## 4. POST /api/v1/detect/annotated (이미지 → JPEG)

```bash
curl -X POST http://localhost:8911/api/v1/detect/annotated \
  -F "file=@images/hand.jpg" \
  --output annotated_result.jpg
```

| # | 항목 | 확인 방법 | 기대 결과 | 결과 |
|---|------|-----------|-----------|------|
| 4-1 | 응답 Content-Type | 응답 헤더 | `image/jpeg` | ■ |
| 4-2 | 이미지 저장 후 확인 | 저장된 파일 열기 | 손 위에 바운딩 박스 + 라벨 표시 | ■ |
| 4-3 | 손 없는 이미지 | 저장된 파일 열기 | 원본과 동일한 이미지 (박스 없음), HTTP 200 | ■ |
| 4-4 | max_hands=1 | 저장된 파일 열기 | 박스 1개만 표시 | ■ |
| 4-5 | 잘못된 파일 | 응답 확인 | HTTP 400 | ☐ |

---

## 5. POST /api/v1/detect/video (영상 → MP4 다운로드)

```bash
curl -X POST http://localhost:8911/api/v1/detect/video \
  -F "file=@sample.mp4" \
  --output annotated_video.mp4
```

| # | 항목 | 확인 방법 | 기대 결과 | 결과 |
|---|------|-----------|-----------|------|
| 5-1 | 응답 Content-Type | 응답 헤더 | `video/mp4` | ■ |
| 5-2 | 응답 파일명 | Content-Disposition 헤더 | `annotated.mp4` | ■ |
| 5-3 | 결과 영상 재생 | 저장된 mp4 열기 | 손에 바운딩 박스 + FPS 표시 없이 처리된 영상 | ■ |
| 5-4 | 임시 파일 정리 | 요청 전후 `Get-ChildItem $env:TEMP -Filter "tmp*.mp4"` 실행해서 파일 수 비교 | 요청 후 임시 mp4 파일 수 증가 없음 | ☐ |
| 5-5 | confidence 파라미터 | 높은 값으로 요청 | 낮은 confidence 박스 제거됨 | ☐ |
| 5-6 | 잘못된 파일 (이미지 등) | 응답 확인 | 오류 응답 (4xx 또는 5xx) | ☐ |

---

## 6. MJPEG 스트리밍 플로우

### 6-1. POST /api/v1/detect/video/upload

```bash
curl -X POST http://localhost:8911/api/v1/detect/video/upload \
  -F "file=@sample.mp4"
```

| # | 항목 | 확인 방법 | 기대 결과 | 결과 |
|---|------|-----------|-----------|------|
| 6-1 | 응답 스키마 | 응답 JSON | `stream_id`, `stream_url`, `player_url` 모두 포함 | ■ |
| 6-2 | URL 형식 | 응답 JSON | `stream_url`, `player_url`이 `http://localhost:8911/api/v1/...` 형태 | ■ |

### 6-2. GET /api/v1/detect/video/stream/{stream_id}

```bash
# stream_url을 브라우저 또는 curl로 접근
curl "http://localhost:8911/api/v1/detect/video/stream/{stream_id}" --output stream_dump.bin
```

| # | 항목 | 확인 방법 | 기대 결과 | 결과 |
|---|------|-----------|-----------|------|
| 6-3 | Content-Type | 응답 헤더 | `multipart/x-mixed-replace; boundary=frame` | ■ |
| 6-4 | 스트림 데이터 흐름 | curl 출력 확인 | 데이터가 연속으로 흘러옴 (스트리밍 중단 없음) | ■ |
| 6-5 | 존재하지 않는 stream_id | 응답 확인 | HTTP 404, `"stream_id not found"` | ■ |

### 6-3. GET /api/v1/detect/video/player/{stream_id}

```bash
# 브라우저에서 player_url 열기
```

| # | 항목 | 확인 방법 | 기대 결과 | 결과 |
|---|------|-----------|-----------|------|
| 6-6 | HTML 응답 | 브라우저 열기 | 검정 배경에 스트리밍 영상 표시 | ■ |
| 6-7 | 바운딩 박스 표시 | 영상 확인 | 손 위에 박스 + 라벨 | ■ |
| 6-8 | 존재하지 않는 stream_id | 브라우저 확인 | HTTP 404 | ■ |

---

## 7. 환경변수 설정 테스트

> ⚠️ 각 테스트 후 반드시 환경변수를 제거하고 다음 항목으로 진행할 것

| # | 항목 | 기동 방법 | 정리 명령어 | 기대 결과 | 결과 |
|---|------|-----------|-------------|-----------|------|
| 7-1 | 모델 변경 (normal) | `$env:YOLO_NETWORK="normal"; uvicorn app.main:app --port 8911` | `Remove-Item Env:YOLO_NETWORK` | 콘솔에 `loading model: normal` | ■ |
| 7-2 | 기본 confidence 변경 | `$env:YOLO_CONFIDENCE="0.5"; uvicorn app.main:app --port 8911` | `Remove-Item Env:YOLO_CONFIDENCE` | `/health` 응답 기준으로 낮은 감도 적용 확인 | ■ |
| 7-3 | 잘못된 network 값 | `$env:YOLO_NETWORK="invalid"; uvicorn app.main:app --port 8911` | `Remove-Item Env:YOLO_NETWORK` | 기동 실패 또는 에러 메시지 출력 | ■ |

---

## 8. 비고 / 이슈 기록

- 멀리 떨어져 있는 손일수록 감지 능력이 약한 것 같음. v3, v4 버전 모델이 어떤 데이터로 학습했는지를 확인하고 해당 데이터까지 추가하여, 현재 v11이 학습한 데이터와 하나로 합쳐서 v11버전으로 재훈련 필요


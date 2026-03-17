import os
import tempfile

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app.schemas.detection import StreamUploadResponse
from app.services.video_service import (
    generate_mjpeg,
    get_video_path,
    process_video,
    register_video,
)

router = APIRouter()


@router.post("/detect/video")
async def detect_video(
    request: Request,
    file: UploadFile = File(...),
    confidence: float | None = None,
    max_hands: int = -1,
):
    """영상 업로드 → 처리 완료된 mp4 파일 반환"""
    suffix = os.path.splitext(file.filename or "video")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    try:
        output_path = process_video(input_path, request.app.state.detector, confidence, max_hands)
    finally:
        os.unlink(input_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="annotated.mp4",
        background=_cleanup_task(output_path),
    )


@router.post("/detect/video/upload", response_model=StreamUploadResponse)
async def upload_for_stream(
    request: Request,
    file: UploadFile = File(...),
):
    """영상 업로드 → stream_id + player_url 반환 (MJPEG 스트리밍용)"""
    suffix = os.path.splitext(file.filename or "video")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    stream_id = register_video(video_path)
    base_url = str(request.base_url).rstrip("/")

    return StreamUploadResponse(
        stream_id=stream_id,
        stream_url=f"{base_url}/api/v1/detect/video/stream/{stream_id}",
        player_url=f"{base_url}/api/v1/detect/video/player/{stream_id}",
    )


@router.get("/detect/video/stream/{stream_id}")
def stream_video(
    stream_id: str,
    request: Request,
    confidence: float | None = None,
    max_hands: int = -1,
):
    """MJPEG 스트리밍 — 브라우저 또는 <img> 태그에서 직접 재생 가능"""
    video_path = get_video_path(stream_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="stream_id not found")

    return StreamingResponse(
        generate_mjpeg(video_path, request.app.state.detector, confidence, max_hands),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/detect/video/player/{stream_id}", response_class=HTMLResponse)
def video_player(stream_id: str, request: Request):
    """MJPEG 스트림을 임베드한 최소 HTML 뷰어"""
    if not get_video_path(stream_id):
        raise HTTPException(status_code=404, detail="stream_id not found")

    base_url = str(request.base_url).rstrip("/")
    stream_url = f"{base_url}/api/v1/detect/video/stream/{stream_id}"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hand Detection Stream</title>
  <style>
    body {{ margin: 0; background: #111; display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
    img {{ max-width: 100%; max-height: 100vh; }}
  </style>
</head>
<body>
  <img src="{stream_url}" alt="stream">
</body>
</html>"""
    return HTMLResponse(content=html)


class _cleanup_task:
    """FileResponse 전송 완료 후 임시 파일을 삭제하는 background task"""

    def __init__(self, path: str):
        self.path = path

    async def __call__(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

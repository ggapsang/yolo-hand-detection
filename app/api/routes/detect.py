from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.schemas.detection import DetectionResponse
from app.services.detection_service import annotate_image, decode_image, run_detection

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request,
    file: UploadFile = File(...),
    confidence: float | None = None,
    max_hands: int = -1,
):
    data = await file.read()
    try:
        image = decode_image(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return run_detection(image, request.app.state.detector, confidence, max_hands)


@router.post("/detect/annotated")
async def detect_annotated(
    request: Request,
    file: UploadFile = File(...),
    confidence: float | None = None,
    max_hands: int = -1,
):
    data = await file.read()
    try:
        image = decode_image(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    jpeg_bytes = annotate_image(image, request.app.state.detector, confidence, max_hands)
    return Response(content=jpeg_bytes, media_type="image/jpeg")

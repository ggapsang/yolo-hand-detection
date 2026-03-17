from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    detector = request.app.state.detector
    return {
        "status": "ok",
        "model": request.app.state.network,
        "provider": detector.provider,
    }

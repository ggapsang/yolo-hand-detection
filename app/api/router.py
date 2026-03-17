from fastapi import APIRouter

from app.api.routes import detect, health, video

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(detect.router, tags=["detection"])
api_router.include_router(video.router, tags=["video"])

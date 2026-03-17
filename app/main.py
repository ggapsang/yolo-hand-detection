from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.inference.factory import create_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] loading model: {settings.network}")
    app.state.network = settings.network
    app.state.detector = create_detector(
        network=settings.network,
        size=settings.size,
        confidence=settings.confidence,
        threshold=settings.threshold,
    )
    print(f"[startup] provider: {app.state.detector.provider}")
    yield
    print("[shutdown] releasing model")
    del app.state.detector


app = FastAPI(
    title="Hand Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")

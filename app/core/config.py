from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    network: str = "v11"
    confidence: float = 0.25
    threshold: float = 0.45
    size: int = 640
    port: int = 8911

    model_config = {"env_prefix": "YOLO_"}


settings = Settings()

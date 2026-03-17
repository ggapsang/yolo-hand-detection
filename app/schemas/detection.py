from pydantic import BaseModel


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox: BBox


class DetectionResponse(BaseModel):
    width: int
    height: int
    inference_time: float
    detections: list[Detection]


class StreamUploadResponse(BaseModel):
    stream_id: str
    stream_url: str
    player_url: str

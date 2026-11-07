from pydantic import BaseModel


class HeartData(BaseModel):
    seconds: int
    heart_rate_bpm: float


class MotionData(BaseModel):
    seconds: int
    acceleration_x_g: float
    acceleration_y_g: float
    acceleration_z_g: float


class SleepPredictionRequest(BaseModel):
    heart: list[HeartData]
    motion: list[MotionData]


class SleepPredictionResponse(BaseModel):
    message: str
    total_predictions: int
    predicted_stages: list[str]

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class SleepPrediction(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)

    seconds: int

    heart_rate_bpm: float | None = None

    acceleration_x_g: float | None = None
    acceleration_y_g: float | None = None
    acceleration_z_g: float | None = None

    predicted_stage: str

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

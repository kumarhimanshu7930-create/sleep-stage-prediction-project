from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from scalar_fastapi import get_scalar_api_reference
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from app.database import create_db_tables, get_session
from app.db_models import SleepPrediction
from app.feature_engineering import create_features
from app.model import predict_sleep_stage
from app.schemas import SleepPredictionRequest, SleepPredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):

    create_db_tables()

    yield


app = FastAPI(title="Sleep Stage Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/scalar", include_in_schema=False)
def scalar_docs():

    return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)


@app.get("/")
def home():

    return {"message": "Sleep Stage Prediction API is running"}


@app.post("/predict", response_model=SleepPredictionResponse)
def predict(
    data: SleepPredictionRequest, session: Annotated[Session, Depends(get_session)]
):

    try:

        # Convert incoming data into DataFrames
        heart = pd.DataFrame([item.model_dump() for item in data.heart])

        motion = pd.DataFrame([item.model_dump() for item in data.motion])

        # Generate features
        features = create_features(heart, motion)

        if features.empty:
            raise HTTPException(
                status_code=400, detail="Not enough data to generate features"
            )

        # Predict sleep stages
        predictions = predict_sleep_stage(features)

        predictions = [str(stage) for stage in predictions]

        # Match original sensor values by 30-second epoch
        heart["epoch"] = (heart["seconds"] // 30) * 30
        motion["epoch"] = (motion["seconds"] // 30) * 30

        heart_epoch = heart.groupby("epoch")["heart_rate_bpm"].mean()

        motion_epoch = motion.groupby("epoch").agg(
            {
                "acceleration_x_g": "mean",
                "acceleration_y_g": "mean",
                "acceleration_z_g": "mean",
            }
        )

        # Prepare database records
        records = []

        for epoch, stage in zip(features["epoch"], predictions):

            records.append(
                SleepPrediction(
                    seconds=int(epoch),
                    heart_rate_bpm=(
                        float(heart_epoch[epoch])
                        if epoch in heart_epoch.index
                        else None
                    ),
                    acceleration_x_g=(
                        float(motion_epoch.loc[epoch, "acceleration_x_g"])
                        if epoch in motion_epoch.index
                        else None
                    ),
                    acceleration_y_g=(
                        float(motion_epoch.loc[epoch, "acceleration_y_g"])
                        if epoch in motion_epoch.index
                        else None
                    ),
                    acceleration_z_g=(
                        float(motion_epoch.loc[epoch, "acceleration_z_g"])
                        if epoch in motion_epoch.index
                        else None
                    ),
                    predicted_stage=stage,
                )
            )

        # Save records into MySQL
        session.add_all(records)
        session.commit()

        return SleepPredictionResponse(
            message="Sleep stage prediction successful",
            total_predictions=len(predictions),
            predicted_stages=predictions,
        )

    except HTTPException:
        raise

    except SQLAlchemyError as e:

        session.rollback()

        raise HTTPException(status_code=500, detail="Database operation failed") from e

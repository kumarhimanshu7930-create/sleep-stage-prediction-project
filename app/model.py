from pathlib import Path

import joblib
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

model = CatBoostClassifier()

model.load_model(str(MODEL_DIR / "catboost_sleep_stage.cbm"))

feature_columns = joblib.load(MODEL_DIR / "catboost_feature_columns.pkl")

# Convert numeric predictions into sleep-stage names
SLEEP_STAGE_MAPPING = {0: "Wake", 1: "Sleep", 2: "Sleep", 3: "Sleep", 5: "REM"}


def predict_sleep_stage(features):

    features = features[feature_columns]

    predictions = model.predict(features)

    predictions = predictions.flatten().astype(int)

    sleep_stages = [SLEEP_STAGE_MAPPING[prediction] for prediction in predictions]

    return sleep_stages

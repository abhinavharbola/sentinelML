import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from serving.db import get_engine, init_db, insert_prediction
from serving.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Fraud Detection API")

_engine = None
_model = None
_model_version = None


def load_production_model():
    global _model, _model_version
    client = MlflowClient()
    mv = client.get_model_version_by_alias(config.MODEL_NAME, "production")
    if mv.version != _model_version:
        _model = mlflow.xgboost.load_model(f"models:/{config.MODEL_NAME}@production")
        _model_version = mv.version
    return _model, _model_version


@app.on_event("startup")
def startup():
    global _engine
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    _engine = get_engine()
    init_db(_engine)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model, version = load_production_model()
    features = request.features.dict()

    score = float(model.predict_proba(pd.DataFrame([features]))[0][1])
    predicted_label = int(score >= config.PREDICTION_THRESHOLD)
    transaction_id = f"{request.batch_id:04d}-{request.row_index}"

    insert_prediction(
        _engine, transaction_id, request.batch_id, request.row_index,
        features, score, predicted_label, str(version),
    )

    return PredictionResponse(
        transaction_id=transaction_id,
        score=score,
        predicted_label=predicted_label,
        model_version=str(version),
    )
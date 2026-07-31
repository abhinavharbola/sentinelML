import sys
from pathlib import Path

from pydantic import BaseModel, create_model

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

Features = create_model("Features", **{col: (float, ...) for col in config.FEATURE_COLUMNS})


class PredictionRequest(BaseModel):
    batch_id: int
    row_index: int
    features: Features


class PredictionResponse(BaseModel):
    transaction_id: str
    score: float
    predicted_label: int
    model_version: str
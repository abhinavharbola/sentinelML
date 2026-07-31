import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from serving.db import get_current_batch, get_engine, get_labeled_predictions

FEATURE_COLUMNS = config.FEATURE_COLUMNS


def emit_output(name, value):
    output_path = os.environ.get("GITHUB_OUTPUT")
    line = f"{name}={value}"
    if output_path:
        with open(output_path, "a") as f:
            f.write(line + "\n")
    print(line)


def load_pretrain_data():
    frames = []
    for i in range(config.PRETRAIN_BATCHES):
        df = pd.read_parquet(config.BATCHES_DIR / f"batch_{i:04d}.parquet")
        df["batch_id"] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_online_labeled_data(engine):
    df = get_labeled_predictions(engine)
    if df.empty:
        return df
    return df.rename(columns={"true_label": "Class"})


def build_training_set(engine):
    columns = FEATURE_COLUMNS + ["Class", "batch_id"]
    pretrain = load_pretrain_data()[columns]
    online = load_online_labeled_data(engine)
    if online.empty:
        return pretrain
    return pd.concat([pretrain, online[columns]], ignore_index=True)


def compute_sample_weights(df, current_batch):
    recent_cutoff = current_batch - config.PERFORMANCE_WINDOW_BATCHES
    weights = pd.Series(1.0, index=df.index)
    weights[df["batch_id"] >= recent_cutoff] = config.RECENT_SAMPLE_WEIGHT
    return weights


def train_model(df, sample_weight=None):
    X, y = df[FEATURE_COLUMNS], df["Class"]
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud-detection")

    engine = get_engine()
    train_df = build_training_set(engine)
    sample_weight = compute_sample_weights(train_df, get_current_batch(engine))

    with mlflow.start_run() as run:
        model = train_model(train_df, sample_weight=sample_weight)

        mlflow.log_param("n_rows", len(train_df))
        mlflow.log_param("n_fraud", int(train_df["Class"].sum()))
        mlflow.log_param("recent_sample_weight", config.RECENT_SAMPLE_WEIGHT)
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=config.MODEL_NAME)

        client = MlflowClient()
        version = client.search_model_versions(f"run_id='{run.info.run_id}'")[0].version
        emit_output("challenger_version", version)

        print(f"trained challenger, run_id={run.info.run_id}, version={version}, rows={len(train_df)}")


if __name__ == "__main__":
    main()
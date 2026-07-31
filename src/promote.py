import argparse
import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from serving.db import get_current_batch, get_engine, get_labeled_predictions, get_performance_window, get_state, log_audit_event, set_state
from src.evaluate import evaluate_model, run_gate


def emit_output(name, value):
    output_path = os.environ.get("GITHUB_OUTPUT")
    line = f"{name}={value}"
    if output_path:
        with open(output_path, "a") as f:
            f.write(line + "\n")
    print(line)


def get_champion_or_none(client):
    try:
        return client.get_model_version_by_alias(config.MODEL_NAME, "production")
    except MlflowException:
        return None


def bootstrap_promote(engine, client, challenger_version):
    challenger_model = mlflow.xgboost.load_model(f"models:/{config.MODEL_NAME}/{challenger_version}")
    holdout = pd.read_parquet(config.FROZEN_HOLDOUT_PATH)
    holdout_perf = evaluate_model(challenger_model, holdout[config.FEATURE_COLUMNS], holdout["Class"])

    client.set_registered_model_alias(config.MODEL_NAME, "production", challenger_version)
    set_state(engine, "champion_version", float(challenger_version))
    set_state(engine, "champion_holdout_recall", holdout_perf["recall"])
    set_state(engine, "champion_holdout_precision", holdout_perf["precision"])

    log_audit_event(engine, "initial_promotion", {"challenger_version": challenger_version, "holdout": holdout_perf})
    return True


def evaluate_and_promote(engine, client, challenger_version):
    champion_mv = get_champion_or_none(client)
    if champion_mv is None:
        return bootstrap_promote(engine, client, challenger_version)

    champion_model = mlflow.xgboost.load_model(f"models:/{config.MODEL_NAME}/{champion_mv.version}")
    challenger_model = mlflow.xgboost.load_model(f"models:/{config.MODEL_NAME}/{challenger_version}")

    holdout = pd.read_parquet(config.FROZEN_HOLDOUT_PATH)
    window_start = max(0, get_current_batch(engine) - config.PERFORMANCE_WINDOW_BATCHES)
    recent_labeled = get_labeled_predictions(engine, min_batch_id=window_start)

    if recent_labeled.empty or (recent_labeled["true_label"] == 1).sum() < config.MIN_FRAUD_COUNT_FOR_PERF_CHECK:
        details = {
            "challenger_version": challenger_version,
            "champion_version": champion_mv.version,
            "reason": "insufficient_recent_labels",
        }
        log_audit_event(engine, "promotion_rejected", details)
        return False

    result = run_gate(
        champion_model, challenger_model,
        holdout[config.FEATURE_COLUMNS], holdout["Class"],
        recent_labeled[config.FEATURE_COLUMNS], recent_labeled["true_label"],
    )
    result["challenger_version"] = challenger_version
    result["champion_version"] = champion_mv.version

    if not result["passed"]:
        log_audit_event(engine, "promotion_rejected", result)
        return False

    set_state(engine, "previous_champion_version", float(champion_mv.version))
    set_state(engine, "previous_champion_holdout_recall", result["champion_holdout"]["recall"])
    set_state(engine, "previous_champion_holdout_precision", result["champion_holdout"]["precision"])
    client.set_registered_model_alias(config.MODEL_NAME, "production", challenger_version)
    set_state(engine, "champion_version", float(challenger_version))
    set_state(engine, "champion_holdout_recall", result["challenger_holdout"]["recall"])
    set_state(engine, "champion_holdout_precision", result["challenger_holdout"]["precision"])

    log_audit_event(engine, "promotion", result)
    return True


def check_rollback(engine, client):
    window_start = max(0, get_current_batch(engine) - config.PERFORMANCE_WINDOW_BATCHES)
    perf_window = get_performance_window(engine, min_batch_id=window_start)
    n_fraud = int((perf_window["true_label"] == 1).sum()) if not perf_window.empty else 0

    if n_fraud < config.MIN_FRAUD_COUNT_FOR_PERF_CHECK:
        print("not enough recent labeled fraud to evaluate rollback")
        return False

    baseline_recall = get_state(engine, "champion_holdout_recall")
    baseline_precision = get_state(engine, "champion_holdout_precision")
    if baseline_recall is None:
        return False

    current_recall = recall_score(perf_window["true_label"], perf_window["predicted_label"])
    current_precision = precision_score(perf_window["true_label"], perf_window["predicted_label"], zero_division=0)

    regressed = (
        current_recall < baseline_recall * (1 - config.RECALL_RELATIVE_DROP_THRESHOLD)
        or current_precision < baseline_precision * (1 - config.PRECISION_RELATIVE_DROP_THRESHOLD)
    )
    if not regressed:
        return False

    previous_version = get_state(engine, "previous_champion_version")
    if previous_version is None:
        print("no previous champion to roll back to")
        return False

    previous_version = int(previous_version)
    client.set_registered_model_alias(config.MODEL_NAME, "production", previous_version)
    set_state(engine, "champion_version", float(previous_version))

    previous_recall = get_state(engine, "previous_champion_holdout_recall")
    previous_precision = get_state(engine, "previous_champion_holdout_precision")
    if previous_recall is not None:
        set_state(engine, "champion_holdout_recall", previous_recall)
    if previous_precision is not None:
        set_state(engine, "champion_holdout_precision", previous_precision)

    log_audit_event(engine, "rollback", {
        "reverted_to_version": previous_version,
        "current_recall": current_recall,
        "current_precision": current_precision,
        "baseline_recall": baseline_recall,
        "baseline_precision": baseline_precision,
    })
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["promote", "check-rollback"])
    parser.add_argument("--challenger-version", type=int)
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    engine = get_engine()
    client = MlflowClient()

    if args.mode == "check-rollback":
        rolled_back = check_rollback(engine, client)
        emit_output("rolled_back", str(rolled_back).lower())
        return

    if args.challenger_version is None:
        raise SystemExit("--challenger-version is required for promote mode")

    promoted = evaluate_and_promote(engine, client, args.challenger_version)
    emit_output("promoted", str(promoted).lower())


if __name__ == "__main__":
    main()
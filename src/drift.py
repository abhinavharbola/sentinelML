import os
import sys
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sklearn.metrics import precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from serving.db import get_current_batch, get_engine, get_performance_window, get_recent_predictions, get_state, log_audit_event


def emit_output(name, value):
    output_path = os.environ.get("GITHUB_OUTPUT")
    line = f"{name}={value}"
    if output_path:
        with open(output_path, "a") as f:
            f.write(line + "\n")
    print(line)


def load_reference():
    return pd.read_parquet(config.FROZEN_HOLDOUT_PATH)[config.FEATURE_COLUMNS]


def compute_feature_drift(reference, current):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report.as_dict()["metrics"][0]["result"]["share_of_drifted_columns"]


def feature_drift_triggered(drift_share):
    return drift_share > config.FEATURE_DRIFT_SHARE_THRESHOLD


def performance_drift_triggered(current_recall, current_precision, baseline_recall, baseline_precision):
    if baseline_recall is None or baseline_recall == 0:
        return False
    recall_drop = (baseline_recall - current_recall) / baseline_recall
    precision_drop = (baseline_precision - current_precision) / baseline_precision if baseline_precision else 0
    return (
        recall_drop > config.RECALL_RELATIVE_DROP_THRESHOLD
        or precision_drop > config.PRECISION_RELATIVE_DROP_THRESHOLD
    )


def should_retrain(drift_share, n_fraud, current_recall, current_precision, baseline_recall, baseline_precision):
    feat_triggered = feature_drift_triggered(drift_share)
    perf_triggered = (
        n_fraud >= config.MIN_FRAUD_COUNT_FOR_PERF_CHECK
        and current_recall is not None
        and performance_drift_triggered(current_recall, current_precision, baseline_recall, baseline_precision)
    )
    return feat_triggered or perf_triggered


def main():
    engine = get_engine()
    current_batch = get_current_batch(engine)
    window_start = max(0, current_batch - config.PERFORMANCE_WINDOW_BATCHES)

    recent = get_recent_predictions(engine, min_batch_id=window_start)
    if recent.empty:
        print("no recent predictions to check drift on")
        emit_output("retrain_needed", "false")
        return

    reference = load_reference()
    drift_share = compute_feature_drift(reference, recent[config.FEATURE_COLUMNS])

    perf_window = get_performance_window(engine, min_batch_id=window_start)
    n_fraud = int((perf_window["true_label"] == 1).sum()) if not perf_window.empty else 0

    baseline_recall = get_state(engine, "champion_holdout_recall")
    baseline_precision = get_state(engine, "champion_holdout_precision")

    current_recall = current_precision = None
    if n_fraud >= config.MIN_FRAUD_COUNT_FOR_PERF_CHECK and baseline_recall is not None:
        current_recall = recall_score(perf_window["true_label"], perf_window["predicted_label"])
        current_precision = precision_score(perf_window["true_label"], perf_window["predicted_label"], zero_division=0)

    retrain_needed = should_retrain(
        drift_share, n_fraud, current_recall, current_precision, baseline_recall, baseline_precision
    )

    log_audit_event(engine, "drift_check", {
        "window_start_batch": window_start,
        "current_batch": current_batch,
        "drift_share": drift_share,
        "feature_drift_triggered": feature_drift_triggered(drift_share),
        "n_fraud_in_window": n_fraud,
        "current_recall": current_recall,
        "current_precision": current_precision,
        "baseline_recall": baseline_recall,
        "baseline_precision": baseline_precision,
        "retrain_needed": retrain_needed,
    })

    emit_output("retrain_needed", str(retrain_needed).lower())


if __name__ == "__main__":
    main()
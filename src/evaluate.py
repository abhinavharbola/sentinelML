import sys
from pathlib import Path

from sklearn.metrics import precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def evaluate_model(model, X, y):
    preds = (model.predict_proba(X)[:, 1] >= config.PREDICTION_THRESHOLD).astype(int)
    return {
        "recall": float(recall_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
    }


def within_tolerance(challenger, champion, tolerance):
    return (
        challenger["recall"] >= champion["recall"] * (1 - tolerance)
        and challenger["precision"] >= champion["precision"] * (1 - tolerance)
    )


def strictly_better_or_equal(challenger, champion):
    not_worse = challenger["recall"] >= champion["recall"] and challenger["precision"] >= champion["precision"]
    strictly_better = challenger["recall"] > champion["recall"] or challenger["precision"] > champion["precision"]
    return not_worse and strictly_better


def run_gate(champion_model, challenger_model, holdout_X, holdout_y, window_X, window_y):
    champion_holdout = evaluate_model(champion_model, holdout_X, holdout_y)
    challenger_holdout = evaluate_model(challenger_model, holdout_X, holdout_y)
    champion_window = evaluate_model(champion_model, window_X, window_y)
    challenger_window = evaluate_model(challenger_model, window_X, window_y)

    holdout_ok = within_tolerance(challenger_holdout, champion_holdout, config.HOLDOUT_TOLERANCE)
    window_ok = strictly_better_or_equal(challenger_window, champion_window)

    return {
        "champion_holdout": champion_holdout,
        "challenger_holdout": challenger_holdout,
        "champion_window": champion_window,
        "challenger_window": challenger_window,
        "holdout_ok": holdout_ok,
        "window_ok": window_ok,
        "passed": holdout_ok and window_ok,
    }
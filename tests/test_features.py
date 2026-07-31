import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.simulate_stream import (
    compute_legit_centroid,
    inject_concept_drift,
    inject_feature_drift,
    split_frozen_holdout,
)


def make_synthetic_df(n_legit=100, n_fraud=20):
    legit = pd.DataFrame({
        "V1": range(n_legit), "V2": range(n_legit), "V3": range(n_legit),
        "Amount": range(n_legit), "Class": 0,
    })
    fraud = pd.DataFrame({
        "V1": [1000] * n_fraud, "V2": [1000] * n_fraud, "V3": [1000] * n_fraud,
        "Amount": [500] * n_fraud, "Class": 1,
    })
    return pd.concat([legit, fraud], ignore_index=True)


def test_split_frozen_holdout_preserves_class_ratio(monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_FRACTION", 0.2)
    monkeypatch.setattr(config, "SEED", 0)
    df = make_synthetic_df(n_legit=100, n_fraud=20)

    holdout, remaining = split_frozen_holdout(df)

    assert len(holdout) + len(remaining) == len(df)
    assert len(holdout) == pytest.approx(24, abs=2)
    assert (holdout.Class == 1).sum() == pytest.approx(4, abs=1)
    assert (holdout.Class == 0).sum() == pytest.approx(20, abs=2)


def test_inject_feature_drift_only_applies_from_start_batch(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_DRIFT_START_BATCH", 5)
    monkeypatch.setattr(config, "FEATURE_DRIFT_FEATURES", ["V1"])
    monkeypatch.setattr(config, "FEATURE_DRIFT_SHIFT", 10.0)
    monkeypatch.setattr(config, "FEATURE_DRIFT_SCALE", 2.0)

    batch = pd.DataFrame({"V1": [1.0, 2.0, 3.0], "Class": [0, 0, 1]})

    unchanged, flagged_before = inject_feature_drift(batch, batch_id=4)
    drifted, flagged_after = inject_feature_drift(batch, batch_id=5)

    assert flagged_before is False
    assert flagged_after is True
    pd.testing.assert_frame_equal(unchanged, batch)
    assert list(drifted["V1"]) == [12.0, 14.0, 16.0]


def test_inject_concept_drift_only_shifts_fraud_rows_in_window(monkeypatch):
    monkeypatch.setattr(config, "CONCEPT_DRIFT_BATCHES", [7])
    monkeypatch.setattr(config, "FEATURE_DRIFT_FEATURES", ["V1"])
    monkeypatch.setattr(config, "CONCEPT_DRIFT_ALPHA", 0.5)

    batch = pd.DataFrame({"V1": [100.0, 200.0], "Class": [0, 1]})
    legit_centroid = pd.Series({"V1": 0.0})

    drifted, flagged_in_window = inject_concept_drift(batch, batch_id=7, legit_centroid=legit_centroid)
    untouched, flagged_out_of_window = inject_concept_drift(batch, batch_id=8, legit_centroid=legit_centroid)

    assert flagged_in_window is True
    assert flagged_out_of_window is False
    assert drifted.loc[0, "V1"] == 100.0
    assert drifted.loc[1, "V1"] == 100.0
    pd.testing.assert_frame_equal(untouched, batch)


def test_compute_legit_centroid_uses_only_class_zero(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_DRIFT_FEATURES", ["V1"])
    holdout = pd.DataFrame({"V1": [10.0, 20.0, 999.0], "Class": [0, 0, 1]})

    centroid = compute_legit_centroid(holdout)

    assert centroid["V1"] == 15.0
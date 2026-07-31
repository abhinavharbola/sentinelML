import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def load_raw():
    return pd.read_csv(config.RAW_DATA_PATH)


def split_frozen_holdout(df):
    holdout = df.groupby("Class", group_keys=False).apply(
        lambda g: g.sample(frac=config.HOLDOUT_FRACTION, random_state=config.SEED)
    )
    remaining = df.drop(holdout.index).reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)
    return holdout, remaining


def compute_legit_centroid(holdout):
    return holdout.loc[holdout.Class == 0, config.FEATURE_DRIFT_FEATURES].mean()


def make_batches(df):
    return np.array_split(df, config.N_BATCHES)


def inject_feature_drift(batch, batch_id):
    if batch_id < config.FEATURE_DRIFT_START_BATCH:
        return batch, False
    batch = batch.copy()
    for feat in config.FEATURE_DRIFT_FEATURES:
        batch[feat] = batch[feat] * config.FEATURE_DRIFT_SCALE + config.FEATURE_DRIFT_SHIFT
    return batch, True


def inject_concept_drift(batch, batch_id, legit_centroid):
    if batch_id not in config.CONCEPT_DRIFT_BATCHES:
        return batch, False
    batch = batch.copy()
    fraud_mask = batch.Class == 1
    alpha = config.CONCEPT_DRIFT_ALPHA
    for feat in config.FEATURE_DRIFT_FEATURES:
        batch.loc[fraud_mask, feat] = (
            (1 - alpha) * batch.loc[fraud_mask, feat] + alpha * legit_centroid[feat]
        )
    return batch, True


def main():
    df = load_raw()
    holdout, remaining = split_frozen_holdout(df)

    config.FROZEN_HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_parquet(config.FROZEN_HOLDOUT_PATH, index=False)

    legit_centroid = compute_legit_centroid(holdout)
    batches = make_batches(remaining)

    config.BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for batch_id, batch in enumerate(batches):
        batch, feature_drift = inject_feature_drift(batch, batch_id)
        batch, concept_drift = inject_concept_drift(batch, batch_id, legit_centroid)

        path = config.BATCHES_DIR / f"batch_{batch_id:04d}.parquet"
        batch.to_parquet(path, index=False)

        manifest.append({
            "batch_id": batch_id,
            "n_rows": len(batch),
            "n_fraud": int(batch.Class.sum()),
            "feature_drift": feature_drift,
            "concept_drift": concept_drift,
        })

    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {len(batches)} batches, holdout size {len(holdout)}")


if __name__ == "__main__":
    main()
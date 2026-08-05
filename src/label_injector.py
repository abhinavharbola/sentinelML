import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from serving.db import get_engine, get_current_batch, get_unlabeled_predictions, update_labels


def load_batch_labels(batch_id):
    df = pd.read_parquet(config.BATCHES_DIR / f"batch_{batch_id:04d}.parquet", columns=["Class"])
    df["row_index"] = df.index
    df["batch_id"] = batch_id
    return df


def main():
    engine = get_engine()
    current_batch = get_current_batch(engine)
    threshold = current_batch - config.LABEL_DELAY_BATCHES

    if threshold < 0:
        print("no batches old enough to release labels for yet")
        return

    pending = get_unlabeled_predictions(engine, max_batch_id=threshold)
    if pending.empty:
        print("no pending predictions to label")
        return

    released = 0
    for batch_id, group in pending.groupby("batch_id"):
        labels = load_batch_labels(batch_id)
        updates = group.merge(labels, on=["batch_id", "row_index"], how="inner")
        updates = updates.rename(columns={"Class": "true_label"})
        update_labels(engine, updates[["transaction_id", "true_label"]])
        released += len(updates)

    print(f"released labels for {released} predictions (batches up to {threshold})")


if __name__ == "__main__":
    main()
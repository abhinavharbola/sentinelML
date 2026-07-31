import os
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from serving.db import get_current_batch, get_engine


def load_batch(batch_id):
    return pd.read_parquet(config.BATCHES_DIR / f"batch_{batch_id:04d}.parquet")


def replay(batch_id, api_url):
    df = load_batch(batch_id)
    sent = 0
    for row_index, row in df.iterrows():
        payload = {
            "batch_id": batch_id,
            "row_index": int(row_index),
            "features": {col: float(row[col]) for col in config.FEATURE_COLUMNS},
        }
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        response.raise_for_status()
        sent += 1
    return sent


def main():
    api_url = os.environ["API_URL"]
    engine = get_engine()
    current_batch = get_current_batch(engine)

    if current_batch < config.STREAM_START_BATCH:
        print(f"batch {current_batch} is pretrain material, not replaying")
        return

    sent = replay(current_batch, api_url)
    print(f"replayed {sent} rows from batch {current_batch}")


if __name__ == "__main__":
    main()
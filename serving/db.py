import json
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def _parse_features(features):
    return features if isinstance(features, dict) else json.loads(features)


def get_engine():
    return create_engine(os.environ["NEON_DATABASE_URL"])


def init_db(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                transaction_id TEXT PRIMARY KEY,
                batch_id INT NOT NULL,
                row_index INT NOT NULL,
                features JSONB NOT NULL,
                score FLOAT NOT NULL,
                predicted_label INT NOT NULL,
                model_version TEXT NOT NULL,
                predicted_at TIMESTAMP NOT NULL DEFAULT now(),
                true_label INT,
                label_inserted_at TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_state (
                key TEXT PRIMARY KEY,
                value DOUBLE PRECISION NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                event_type TEXT NOT NULL,
                details JSONB NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("""
            INSERT INTO pipeline_state (key, value) VALUES ('current_batch', 0)
            ON CONFLICT (key) DO NOTHING
        """))


def get_state(engine, key, default=None):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT value FROM pipeline_state WHERE key = :key"), {"key": key}).fetchone()
    return row[0] if row else default


def set_state(engine, key, value):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO pipeline_state (key, value) VALUES (:key, :value)
            ON CONFLICT (key) DO UPDATE SET value = :value
        """), {"key": key, "value": value})


def get_current_batch(engine):
    return int(get_state(engine, "current_batch", 0))


def advance_batch(engine):
    next_batch = get_current_batch(engine) + 1
    set_state(engine, "current_batch", next_batch)
    return next_batch


def insert_prediction(engine, transaction_id, batch_id, row_index, features, score, predicted_label, model_version):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO predictions
                (transaction_id, batch_id, row_index, features, score, predicted_label, model_version)
            VALUES
                (:transaction_id, :batch_id, :row_index, CAST(:features AS JSONB), :score, :predicted_label, :model_version)
            ON CONFLICT (transaction_id) DO NOTHING
        """), {
            "transaction_id": transaction_id,
            "batch_id": batch_id,
            "row_index": row_index,
            "features": json.dumps(features),
            "score": score,
            "predicted_label": predicted_label,
            "model_version": model_version,
        })


def insert_predictions_bulk(engine, records):
    if not records:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO predictions
                (transaction_id, batch_id, row_index, features, score, predicted_label, model_version)
            VALUES
                (:transaction_id, :batch_id, :row_index, CAST(:features AS JSONB), :score, :predicted_label, :model_version)
            ON CONFLICT (transaction_id) DO NOTHING
        """), records)


def get_unlabeled_predictions(engine, max_batch_id):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT transaction_id, batch_id, row_index
            FROM predictions
            WHERE true_label IS NULL AND batch_id <= :max_batch_id
        """), {"max_batch_id": max_batch_id}).fetchall()
    return pd.DataFrame(rows, columns=["transaction_id", "batch_id", "row_index"])


def update_labels(engine, updates: pd.DataFrame):
    with engine.begin() as conn:
        for row in updates.itertuples(index=False):
            conn.execute(text("""
                UPDATE predictions
                SET true_label = :true_label, label_inserted_at = now()
                WHERE transaction_id = :transaction_id
            """), {"true_label": int(row.true_label), "transaction_id": row.transaction_id})


def get_recent_predictions(engine, min_batch_id, max_batch_id=None):
    query = "SELECT batch_id, features FROM predictions WHERE batch_id >= :min_batch_id"
    params = {"min_batch_id": min_batch_id}
    if max_batch_id is not None:
        query += " AND batch_id <= :max_batch_id"
        params["max_batch_id"] = max_batch_id

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    if not rows:
        return pd.DataFrame()

    records = []
    for batch_id, features in rows:
        record = dict(_parse_features(features))
        record["batch_id"] = batch_id
        records.append(record)
    return pd.DataFrame(records)


def get_performance_window(engine, min_batch_id, max_batch_id=None):
    query = """
        SELECT batch_id, score, predicted_label, true_label
        FROM predictions
        WHERE true_label IS NOT NULL AND batch_id >= :min_batch_id
    """
    params = {"min_batch_id": min_batch_id}
    if max_batch_id is not None:
        query += " AND batch_id <= :max_batch_id"
        params["max_batch_id"] = max_batch_id

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    return pd.DataFrame(rows, columns=["batch_id", "score", "predicted_label", "true_label"])


def get_labeled_predictions(engine, min_batch_id=None, max_batch_id=None):
    query = "SELECT batch_id, features, true_label FROM predictions WHERE true_label IS NOT NULL"
    params = {}
    if min_batch_id is not None:
        query += " AND batch_id >= :min_batch_id"
        params["min_batch_id"] = min_batch_id
    if max_batch_id is not None:
        query += " AND batch_id <= :max_batch_id"
        params["max_batch_id"] = max_batch_id

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    if not rows:
        return pd.DataFrame()

    records = []
    for batch_id, features, true_label in rows:
        record = dict(_parse_features(features))
        record["batch_id"] = batch_id
        record["true_label"] = true_label
        records.append(record)
    return pd.DataFrame(records)


def get_audit_events(engine, event_type=None, limit=500):
    query = "SELECT id, event_type, details, created_at FROM audit_log"
    params = {"limit": limit}
    if event_type is not None:
        query += " WHERE event_type = :event_type"
        params["event_type"] = event_type
    query += " ORDER BY created_at DESC LIMIT :limit"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    return pd.DataFrame(rows, columns=["id", "event_type", "details", "created_at"])


def log_audit_event(engine, event_type, details: dict):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO audit_log (event_type, details) VALUES (:event_type, CAST(:details AS JSONB))
        """), {"event_type": event_type, "details": json.dumps(details)})
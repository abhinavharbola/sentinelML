import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from dashboard.llm_explain import explain_event
from serving.db import get_audit_events, get_current_batch, get_engine, get_state

st.set_page_config(page_title="Fraud Detection Pipeline", layout="wide")


@st.cache_resource
def get_db_engine():
    return get_engine()


def load_events(engine):
    events = get_audit_events(engine, limit=500)
    return events.sort_values("created_at")


def render_overview(engine):
    st.subheader("Pipeline status")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current simulated batch", get_current_batch(engine))
    col2.metric("Champion version", int(get_state(engine, "champion_version", 0)))

    baseline_recall = get_state(engine, "champion_holdout_recall")
    baseline_precision = get_state(engine, "champion_holdout_precision")
    col3.metric("Champion holdout recall", f"{baseline_recall:.2%}" if baseline_recall is not None else "n/a")
    col4.metric("Champion holdout precision", f"{baseline_precision:.2%}" if baseline_precision is not None else "n/a")


def extract_drift_rows(events):
    drift_events = events[events.event_type == "drift_check"]
    rows = []
    for _, row in drift_events.iterrows():
        d = row["details"]
        rows.append({
            "batch": d.get("current_batch"),
            "drift_share": d.get("drift_share"),
            "recall": d.get("current_recall"),
            "precision": d.get("current_precision"),
            "retrain_needed": d.get("retrain_needed"),
        })
    return pd.DataFrame(rows).sort_values("batch") if rows else pd.DataFrame()


def render_drift_charts(events):
    st.subheader("Feature drift over simulated time")
    df = extract_drift_rows(events)
    if df.empty:
        st.info("No drift checks logged yet.")
        return

    st.line_chart(df.set_index("batch")[["drift_share"]])
    st.caption(f"Retrain fires above {config.FEATURE_DRIFT_SHARE_THRESHOLD} share of drifted columns")

    perf_df = df.dropna(subset=["recall"])
    if not perf_df.empty:
        st.subheader("Rolling recall / precision")
        st.line_chart(perf_df.set_index("batch")[["recall", "precision"]])


def render_promotion_history(events):
    st.subheader("Champion / challenger history")
    history = events[events.event_type.isin(["initial_promotion", "promotion", "promotion_rejected", "rollback"])]
    if history.empty:
        st.info("No promotion events logged yet.")
        return

    rows = []
    for _, row in history.iterrows():
        d = row["details"]
        rows.append({
            "time": row["created_at"],
            "event": row["event_type"],
            "challenger_version": d.get("challenger_version"),
            "champion_version": d.get("champion_version"),
            "holdout_ok": d.get("holdout_ok"),
            "window_ok": d.get("window_ok"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_explanations(events):
    st.subheader("Ask the pipeline to explain a decision")
    explainable = events[events.event_type.isin(["drift_check", "promotion", "promotion_rejected", "rollback"])]
    if explainable.empty:
        st.info("Nothing to explain yet.")
        return

    labels = [f"{row.created_at} : {row.event_type}" for _, row in explainable.iterrows()]
    selected_label = st.selectbox("Select an event", labels)
    selected_row = explainable.iloc[labels.index(selected_label)]

    if st.button("Explain this decision"):
        with st.spinner("Asking the model..."):
            explanation = explain_event(selected_row.event_type, selected_row.details)
        st.write(explanation)


def main():
    engine = get_db_engine()
    st.title("Continuous Fraud Detection: MLOps Dashboard")

    render_overview(engine)
    events = load_events(engine)
    render_drift_charts(events)
    render_promotion_history(events)
    render_explanations(events)


if __name__ == "__main__":
    main()
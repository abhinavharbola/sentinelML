import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from dashboard.llm_explain import explain_event
from serving.db import get_audit_events, get_current_batch, get_engine, get_state, init_db

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon="🛡️", layout="wide")

EVENT_STYLE = {
    "initial_promotion": ("#e6f4ea", "#1e7e34", "🟢 initial promotion"),
    "promotion": ("#e6f4ea", "#1e7e34", "🟢 promotion"),
    "promotion_rejected": ("#fdecea", "#b3261e", "🔴 rejected"),
    "rollback": ("#fff4e5", "#b25e00", "🟠 rollback"),
    "drift_check": ("#e8eef7", "#2952a3", "🔵 drift check"),
}

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: #f8f9fb;
        border: 1px solid #e6e8eb;
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    h1 { font-weight: 700; }
    h3 { margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_engine():
    engine = get_engine()
    init_db(engine)
    return engine


def load_events(engine):
    events = get_audit_events(engine, limit=500)
    return events.sort_values("created_at")


def render_header(engine):
    st.title("🛡️ Continuous Fraud Detection")
    st.caption("MLOps dashboard: drift, promotion gate, rollback, and audit trail")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current simulated batch", get_current_batch(engine))
    col2.metric("Champion version", int(get_state(engine, "champion_version", 0)))

    baseline_recall = get_state(engine, "champion_holdout_recall")
    baseline_precision = get_state(engine, "champion_holdout_precision")
    col3.metric("Champion holdout recall", f"{baseline_recall:.2%}" if baseline_recall is not None else "n/a")
    col4.metric("Champion holdout precision", f"{baseline_precision:.2%}" if baseline_precision is not None else "n/a")


def render_sidebar(engine):
    with st.sidebar:
        st.header("Pipeline")
        st.metric("Simulated batch", get_current_batch(engine))
        st.metric("Champion version", int(get_state(engine, "champion_version", 0)))
        st.divider()
        st.caption(f"Feature drift threshold: {config.FEATURE_DRIFT_SHARE_THRESHOLD}")
        st.caption(f"Holdout tolerance: {config.HOLDOUT_TOLERANCE:.0%}")
        st.caption(f"Min fraud in window: {config.MIN_FRAUD_COUNT_FOR_PERF_CHECK}")
        st.divider()
        if st.button("🔄 Refresh data", use_container_width=True):
            st.rerun()


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


def render_drift_tab(events):
    df = extract_drift_rows(events)
    if df.empty:
        st.info("📭 No drift checks logged yet. Run `src/drift.py` to generate one.")
        return

    st.subheader("Feature drift over simulated time")
    chart_df = df.set_index("batch")[["drift_share"]].copy()
    chart_df["threshold"] = config.FEATURE_DRIFT_SHARE_THRESHOLD
    st.line_chart(chart_df, color=["#2952a3", "#b3261e"])
    st.caption(f"Retrain fires when drift share crosses the threshold line ({config.FEATURE_DRIFT_SHARE_THRESHOLD})")

    latest = df.iloc[-1]
    if bool(latest.get("retrain_needed")):
        st.warning(f"⚠️ Most recent check (batch {int(latest['batch'])}) flagged `retrain_needed = true`")
    else:
        st.success(f"✅ Most recent check (batch {int(latest['batch'])}) did not trigger a retrain")

    perf_df = df.dropna(subset=["recall"])
    if not perf_df.empty:
        st.subheader("Rolling recall / precision")
        st.line_chart(perf_df.set_index("batch")[["recall", "precision"]])
    else:
        st.caption("No rolling performance data yet — needs enough labeled fraud in the recent window.")


def render_history_tab(events):
    history = events[events.event_type.isin(["initial_promotion", "promotion", "promotion_rejected", "rollback"])]
    if history.empty:
        st.info("📭 No promotion events logged yet. Run `src/promote.py` to generate one.")
        return

    st.subheader("Champion / challenger history")
    rows = []
    for _, row in history.iterrows():
        d = row["details"]
        rows.append({
            "time": row["created_at"],
            "event": EVENT_STYLE.get(row.event_type, ("", "", row.event_type))[2],
            "challenger_version": d.get("challenger_version"),
            "champion_version": d.get("champion_version"),
            "holdout_ok": d.get("holdout_ok"),
            "window_ok": d.get("window_ok"),
        })
    display_df = pd.DataFrame(rows).sort_values("time", ascending=False)

    def style_event_column(val):
        for _, (bg, fg, label) in EVENT_STYLE.items():
            if val == label:
                return f"background-color: {bg}; color: {fg}; font-weight: 600; border-radius: 6px;"
        return ""

    styled = display_df.style.map(style_event_column, subset=["event"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_explanations_tab(events):
    explainable = events[events.event_type.isin(["drift_check", "promotion", "promotion_rejected", "rollback"])]
    if explainable.empty:
        st.info("📭 Nothing to explain yet.")
        return

    st.subheader("Ask the pipeline to explain a decision")
    st.caption("One isolated, stateless LLM call — only fires when you click the button below.")

    explainable = explainable.sort_values("created_at", ascending=False)
    labels = [
        f"{EVENT_STYLE.get(row.event_type, ('', '', row.event_type))[2]}  ·  {row.created_at}"
        for _, row in explainable.iterrows()
    ]
    selected_label = st.selectbox("Select an event", labels)
    selected_row = explainable.iloc[labels.index(selected_label)]

    if st.button("💬 Explain this decision", type="primary"):
        with st.spinner("Asking the model..."):
            explanation = explain_event(selected_row.event_type, selected_row.details)
        st.markdown(f"> {explanation}")


def main():
    engine = get_db_engine()
    render_sidebar(engine)
    render_header(engine)

    events = load_events(engine)

    tab_drift, tab_history, tab_explain = st.tabs(["📈 Drift", "🗂️ Promotion History", "💬 Ask the Pipeline"])
    with tab_drift:
        render_drift_tab(events)
    with tab_history:
        render_history_tab(events)
    with tab_explain:
        render_explanations_tab(events)


if __name__ == "__main__":
    main()
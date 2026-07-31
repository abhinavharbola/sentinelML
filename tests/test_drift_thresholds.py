import config
from src.drift import feature_drift_triggered, performance_drift_triggered, should_retrain


def test_feature_drift_triggered_above_threshold():
    assert feature_drift_triggered(config.FEATURE_DRIFT_SHARE_THRESHOLD + 0.01) is True


def test_feature_drift_not_triggered_at_or_below_threshold():
    assert feature_drift_triggered(config.FEATURE_DRIFT_SHARE_THRESHOLD) is False
    assert feature_drift_triggered(config.FEATURE_DRIFT_SHARE_THRESHOLD - 0.01) is False


def test_performance_drift_not_triggered_without_baseline():
    assert performance_drift_triggered(0.5, 0.5, None, None) is False


def test_performance_drift_triggered_on_recall_drop():
    baseline_recall = 0.80
    dropped_recall = baseline_recall * (1 - config.RECALL_RELATIVE_DROP_THRESHOLD - 0.01)
    assert performance_drift_triggered(dropped_recall, 0.70, baseline_recall, 0.70) is True


def test_performance_drift_not_triggered_within_threshold():
    baseline_recall = 0.80
    slightly_lower_recall = baseline_recall * (1 - config.RECALL_RELATIVE_DROP_THRESHOLD + 0.01)
    assert performance_drift_triggered(slightly_lower_recall, 0.70, baseline_recall, 0.70) is False


def test_should_retrain_true_when_only_feature_drift_triggers():
    drift_share = config.FEATURE_DRIFT_SHARE_THRESHOLD + 0.1
    retrain = should_retrain(
        drift_share, n_fraud=0, current_recall=None, current_precision=None,
        baseline_recall=None, baseline_precision=None,
    )
    assert retrain is True


def test_should_retrain_ignores_performance_when_fraud_count_too_low():
    baseline_recall = 0.80
    dropped_recall = baseline_recall * 0.5
    retrain = should_retrain(
        drift_share=0.0,
        n_fraud=config.MIN_FRAUD_COUNT_FOR_PERF_CHECK - 1,
        current_recall=dropped_recall,
        current_precision=0.70,
        baseline_recall=baseline_recall,
        baseline_precision=0.70,
    )
    assert retrain is False


def test_should_retrain_true_when_performance_drops_with_enough_fraud():
    baseline_recall = 0.80
    dropped_recall = baseline_recall * 0.5
    retrain = should_retrain(
        drift_share=0.0,
        n_fraud=config.MIN_FRAUD_COUNT_FOR_PERF_CHECK,
        current_recall=dropped_recall,
        current_precision=0.70,
        baseline_recall=baseline_recall,
        baseline_precision=0.70,
    )
    assert retrain is True
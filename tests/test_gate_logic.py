import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.evaluate import strictly_better_or_equal, within_tolerance


def test_within_tolerance_allows_small_regression():
    champion = {"recall": 0.80, "precision": 0.80}
    challenger = {"recall": 0.77, "precision": 0.80}

    assert within_tolerance(challenger, champion, tolerance=0.05) is True


def test_within_tolerance_rejects_large_regression():
    champion = {"recall": 0.80, "precision": 0.80}
    challenger = {"recall": 0.60, "precision": 0.80}

    assert within_tolerance(challenger, champion, tolerance=0.05) is False


def test_strictly_better_or_equal_rejects_mixed_result():
    champion = {"recall": 0.70, "precision": 0.70}
    challenger = {"recall": 0.75, "precision": 0.65}

    assert strictly_better_or_equal(challenger, champion) is False


def test_strictly_better_or_equal_accepts_pareto_improvement():
    champion = {"recall": 0.70, "precision": 0.70}
    challenger = {"recall": 0.75, "precision": 0.70}

    assert strictly_better_or_equal(challenger, champion) is True


def test_strictly_better_or_equal_rejects_identical_performance():
    champion = {"recall": 0.70, "precision": 0.70}
    challenger = {"recall": 0.70, "precision": 0.70}

    assert strictly_better_or_equal(challenger, champion) is False
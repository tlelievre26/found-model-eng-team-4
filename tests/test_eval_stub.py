import math

import pytest

from src.eval.stub import metrics


def test_hand_worked_metrics():
    probs = [{"A": 0.8, "B": 0.2}, {"A": 0.4, "B": 0.6}]
    m = metrics(probs, ["A", "A"], n_bins=10)
    assert m["accuracy"] == 0.5
    assert m["nll"] == pytest.approx((-math.log(0.8) - math.log(0.4)) / 2)
    # Two bins with one question each: |1 - 0.8| and |0 - 0.6|, weighted 1/2 each.
    assert m["ece"] == pytest.approx((0.2 + 0.6) / 2)


def test_perfect_calibration_has_zero_ece():
    probs = [{"A": 0.75, "B": 0.25}] * 4
    assert metrics(probs, ["A", "A", "A", "B"])["ece"] == pytest.approx(0)

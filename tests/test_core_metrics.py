import pytest

from core.metrics import compute_core_metrics


def test_max_drawdown_includes_loss_from_initial_nav():
    metrics = compute_core_metrics([-10.0, 5.0])

    assert metrics["max_drawdown"] == pytest.approx(-10.0)

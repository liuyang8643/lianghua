import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "results" / "money_flow_20260609_analysis.py"
    spec = importlib.util.spec_from_file_location("money_flow_20260609_analysis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rank_ic_by_day_detects_positive_monotonic_relation():
    mod = _load_module()
    rows_in = []
    for day in range(5):
        date = f"2024-01-{day + 2:02d}"
        for value in range(25):
            rows_in.append({"date": date, "feature": value, "fwd5_excess": value * 2})
    df = pd.DataFrame(rows_in)

    rows = mod._rank_ic_by_day(df, ["feature"], ["fwd5_excess"])

    assert rows[0]["mean_ic"] == 1.0
    assert rows[0]["hit"] == 1.0


def test_quantile_spread_uses_high_minus_low_feature_bucket():
    mod = _load_module()
    rows_in = []
    for day in range(5):
        date = f"2024-01-{day + 2:02d}"
        for value in range(25):
            rows_in.append({"date": date, "feature": value, "fwd5_excess": value})
    df = pd.DataFrame(rows_in)

    rows = mod._quantile_spread(df, ["feature"], "fwd5_excess")

    assert rows[0]["q5_minus_q1"] > 0
    assert rows[0]["q5"] > rows[0]["q1"]

import numpy as np

from research_holdout_trade_open_audit import audit_run


def _trade_row(date="2019-01-02", price=10.0):
    return [
        {"sort": 0},
        {"sort": date},
        {"sort": "000001.SZ Ping An"},
        {"sort": "buy"},
        {"sort": "open"},
        {"sort": price},
        {"sort": 100},
        {"sort": price * 100},
    ]


def test_holdout_trade_audit_accepts_exact_runtime_open(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "research_holdout_trade_open_audit._report_payload",
        lambda _: {"tables": {"trades": {"rows": [_trade_row()]}}},
    )

    result = audit_run(
        tmp_path,
        "validation",
        np.array([[10.0]]),
        {"2019-01-02": 0},
        {"000001.SZ": 0},
    )

    assert result["passes"] is True


def test_holdout_trade_audit_exposes_price_and_period_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "research_holdout_trade_open_audit._report_payload",
        lambda _: {
            "tables": {"trades": {"rows": [_trade_row("2018-12-28", 9.0)]}}
        },
    )

    result = audit_run(
        tmp_path,
        "validation",
        np.array([[10.0]]),
        {"2018-12-28": 0},
        {"000001.SZ": 0},
    )

    assert result["passes"] is False
    assert result["issues"]["open_price_mismatches"] == 1
    assert result["issues"]["outside_frozen_period"] == 1

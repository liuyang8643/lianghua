from datetime import date, datetime

import numpy as np

from trading.post_close import _run_continuous_backtest


def test_continuous_backtest_uses_current_timing_api(monkeypatch):
    trade_day = date(2026, 8, 13)
    valid_dates = [datetime(2026, 8, 13)]
    data = {
        "stock_codes": np.array(["600000.SH"]),
        "trade_dates": np.array(["2026-08-13"], dtype="datetime64[D]"),
        "open": np.array([[10.0]]),
    }
    scores = {"Factor": np.array([[1.0]])}
    timing_calls = []
    overlay_calls = []

    monkeypatch.setattr(
        "utils.stock.time.get_trading_date_span",
        lambda start, end: [trade_day],
    )
    monkeypatch.setattr(
        "core.runtime.load_runtime_stock_codes",
        lambda: ["600000.SH"],
    )
    monkeypatch.setattr(
        "core.backtest._compute_factor_scores",
        lambda *args, **kwargs: (
            data, scores, {}, valid_dates, [0],
            ["600000.SH"], {"600000.SH": 0},
        ),
    )

    def fake_timing(config, dates, index_data=None):
        timing_calls.append((config, dates, index_data))
        return np.array([0.75])

    monkeypatch.setattr("core.backtest._compute_timing_multipliers", fake_timing)

    def fake_overlay(**kwargs):
        overlay_calls.append(kwargs)
        return kwargs["base_multipliers"] * 0.5

    monkeypatch.setattr(
        "core.trend_timing.compute_configured_timing_multipliers",
        fake_overlay,
    )
    monkeypatch.setattr(
        "core.backtest._backtest_direct",
        lambda *args, **kwargs: {"position_multipliers": kwargs["position_multipliers"]},
    )

    result = _run_continuous_backtest(
        trade_day, trade_day,
        {"weights": {"Factor": 1.0}, "buy_n": 1},
        [object],
    )

    assert timing_calls == [(
        {"weights": {"Factor": 1.0}, "buy_n": 1}, valid_dates, None,
    )]
    assert len(overlay_calls) == 1
    assert overlay_calls[0]["data"] is data
    assert overlay_calls[0]["date_indices"] == [0]
    assert overlay_calls[0]["base_multipliers"].tolist() == [0.75]
    assert result["position_multipliers"].tolist() == [0.375]

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.update_ah_history import attach_h_preclose
from testback.research_ah_premium import (
    AHConfig,
    CostModel,
    Period,
    build_pair_panel,
    causal_adjusted_a_prices,
    causal_adjusted_h_prices,
    current_pair_signals,
    derive_contiguous_periods,
    event_metrics,
    generate_pair_events,
    period_events,
    prior_rolling_zscore,
    require_executable_mode,
    select_training_candidate,
    training_fitness,
)


def _a_history(dates, *, price=100.0):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": np.full(len(dates), price),
            "close": np.full(len(dates), price),
            "preClose": np.full(len(dates), price),
        }
    )


def _h_history(dates, *, price=100.0):
    values = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "h_code": "00001.HK",
            "raw_open": np.full(len(dates), price),
            "raw_close": np.full(len(dates), price),
            "raw_high": np.full(len(dates), price + 1.0),
            "raw_low": np.full(len(dates), price - 1.0),
            "qfq_open": np.full(len(dates), price),
            "qfq_close": np.full(len(dates), price),
            "qfq_high": np.full(len(dates), price + 1.0),
            "qfq_low": np.full(len(dates), price - 1.0),
            "hfq_open": np.full(len(dates), price),
            "hfq_close": np.full(len(dates), price),
            "volume": 1.0,
        }
    )
    return attach_h_preclose(values)


def _fx_history(dates, *, rate=1.0):
    return pd.DataFrame(
        {"date": pd.to_datetime(dates), "mid_rate": np.full(len(dates), rate)}
    )


def test_causal_a_adjustment_absorbs_split_without_future_factor():
    values = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "open": [100.0, 50.0, 51.0],
            "close": [100.0, 50.0, 51.0],
            "preClose": [100.0, 50.0, 50.0],
        }
    )
    result = causal_adjusted_a_prices(values)
    np.testing.assert_allclose(result["a_adj_open"], [100.0, 100.0, 102.0])
    np.testing.assert_allclose(result["a_adj_close"], [100.0, 100.0, 102.0])


def test_prior_zscore_excludes_current_and_future_values():
    values = np.array([1.0, 2.0, 1.5, 2.5, 2.0, 10.0, 4.0])
    original = prior_rolling_zscore(values, 5)
    changed_current = values.copy()
    changed_current[5] = 1000.0
    changed = prior_rolling_zscore(changed_current, 5)
    assert original[5] != changed[5]
    np.testing.assert_allclose(original[:5], changed[:5], equal_nan=True)

    changed_future = values.copy()
    changed_future[6] = -999.0
    future = prior_rolling_zscore(changed_future, 5)
    np.testing.assert_allclose(original[:6], future[:6], equal_nan=True)


def test_pair_panel_uses_only_common_a_h_fx_dates_and_correct_fx_unit():
    a_dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    h_dates = ["2024-01-02", "2024-01-04"]
    fx_dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    panel = build_pair_panel(
        _a_history(a_dates, price=100.0),
        _h_history(h_dates, price=100.0),
        _fx_history(fx_dates, rate=0.8),
    )
    assert panel["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-04",
    ]
    np.testing.assert_allclose(panel["spread_close"], np.log(100.0 / 80.0))


def test_h_causal_index_uses_qfq_preclose_not_broken_hfq_level():
    history = _h_history(pd.date_range("2025-06-02", periods=2), price=80.0)
    history.loc[0, ["raw_open", "raw_close"]] = [81.85, 82.85]
    history.loc[0, ["raw_high", "raw_low"]] = [83.0, 81.0]
    history.loc[0, ["qfq_open", "qfq_close", "qfq_high", "qfq_low"]] = [
        73.121,
        74.121,
        74.271,
        72.271,
    ]
    history.loc[1, ["raw_open", "raw_close"]] = [79.50, 77.60]
    history.loc[1, ["raw_high", "raw_low"]] = [80.0, 77.0]
    history.loc[1, ["qfq_open", "qfq_close", "qfq_high", "qfq_low"]] = [
        74.586,
        72.686,
        75.086,
        72.086,
    ]
    history.loc[:, "hfq_close"] = [180.576, 81.415]
    history = attach_h_preclose(history.drop(columns=[
        "qfq_scale", "qfq_intercept", "qfq_affine_valid", "h_pre_close"
    ]))
    result = causal_adjusted_h_prices(history)
    assert history.loc[1, "h_pre_close"] == pytest.approx(79.035, abs=0.002)
    assert result.loc[1, "h_adj_close"] / result.loc[0, "h_adj_close"] - 1 == (
        pytest.approx(-0.01816, abs=0.0001)
    )
    assert abs(np.log(result.loc[1, "h_adj_close"] / result.loc[0, "h_adj_close"])) < 0.03


def _event_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12)
    spread = np.array([0.00, 0.02, -0.01, 0.01, -0.02, 0.50, 0.10, 0.05,
                       0.02, 0.01, 0.00, -0.01])
    a_open = np.full(12, 100.0)
    h_open = np.full(12, 100.0)
    h_open[6:] = 102.0
    return pd.DataFrame(
        {
            "date": dates,
            "a_adj_open": a_open,
            "h_adj_open": h_open,
            "mid_rate": np.ones(12),
            "spread_close": spread,
            "spread_open": spread,
            "segment_id": np.zeros(12, dtype=np.int64),
        }
    )


def test_signal_side_does_not_read_next_execution_rows_close_or_volume():
    config = AHConfig(lookback=5, entry_z=1.0, horizon=1)
    base = _event_panel()
    events = generate_pair_events(
        base,
        config,
        CostModel(0.0, 0.0, 0.0),
        a_code="600000.SH",
        h_code="00001.HK",
    )
    target = events.loc[events["signal_date"] == pd.Timestamp("2024-01-06")].iloc[0]
    assert target["cheap_leg"] == "H"
    assert target["entry_date"] == pd.Timestamp("2024-01-07")

    polluted = base.copy()
    polluted.loc[6, "spread_close"] = -1000.0
    polluted["volume"] = 1.0
    polluted.loc[6, "volume"] = 1e30
    changed = generate_pair_events(
        polluted,
        config,
        CostModel(0.0, 0.0, 0.0),
        a_code="600000.SH",
        h_code="00001.HK",
    )
    target_changed = changed.loc[
        changed["signal_date"] == pd.Timestamp("2024-01-06")
    ].iloc[0]
    assert target_changed["cheap_leg"] == target["cheap_leg"]
    assert target_changed["entry_date"] == target["entry_date"]


def test_event_never_crosses_an_adjustment_segment_boundary():
    panel = _event_panel()
    panel.loc[6:, "segment_id"] = 1
    events = generate_pair_events(
        panel,
        AHConfig(lookback=5, entry_z=1.0, horizon=1),
        CostModel(0.0, 0.0, 0.0),
        a_code="600000.SH",
        h_code="00001.HK",
    )
    assert events.empty


def test_cost_increase_cannot_improve_event_net_return():
    config = AHConfig(lookback=5, entry_z=1.0, horizon=1)
    low = generate_pair_events(
        _event_panel(),
        config,
        CostModel(0.0, 0.0, 0.0),
        a_code="600000.SH",
        h_code="00001.HK",
    )
    high = generate_pair_events(
        _event_panel(),
        config,
        CostModel(100.0, 100.0, 100.0),
        a_code="600000.SH",
        h_code="00001.HK",
    )
    np.testing.assert_array_less(
        high["long_only_net_return"].to_numpy(),
        low["long_only_net_return"].to_numpy(),
    )
    np.testing.assert_array_less(
        high["theoretical_pair_net_return"].to_numpy(),
        low["theoretical_pair_net_return"].to_numpy(),
    )


def test_long_short_is_blocked_without_point_in_time_borrow_data():
    require_executable_mode("long_only_switch")
    with pytest.raises(ValueError, match="theoretical_only"):
        require_executable_mode("long_short_pair")
    require_executable_mode("long_short_pair", has_pit_borrow_data=True)


def _candidate_frame(train_value: float, test_value: float) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    values = np.r_[np.full(40, train_value), np.full(20, test_value)]
    # Add deterministic dispersion so the t-stat is finite.
    values = values + np.tile([-0.001, 0.001], 30)
    return pd.DataFrame(
        {
            "a_code": "600000.SH",
            "h_code": "00001.HK",
            "signal_date": dates,
            "entry_date": dates,
            "exit_date": dates,
            "long_only_net_return": values,
            "theoretical_pair_net_return": values,
            "signed_spread_reversion": values,
        }
    )


def test_training_selection_is_invariant_to_test_outcomes():
    first = AHConfig(20, 1.0, 5)
    second = AHConfig(60, 1.0, 5)
    period = Period(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-04-01"))
    candidates = {
        first: _candidate_frame(0.02, -100.0),
        second: _candidate_frame(0.01, 100.0),
    }
    selected, _ = select_training_candidate(candidates, period)
    assert selected == first

    mutated = {key: value.copy() for key, value in candidates.items()}
    for frame in mutated.values():
        frame.loc[frame["signal_date"] > period.end, "theoretical_pair_net_return"] *= -999
    selected_after, _ = select_training_candidate(mutated, period)
    assert selected_after == first


def test_training_selection_uses_executable_long_only_not_theoretical_pair():
    first = AHConfig(20, 1.0, 5)
    second = AHConfig(60, 1.0, 5)
    period = Period(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-04-01"))
    first_frame = _candidate_frame(0.02, 0.0)
    second_frame = _candidate_frame(0.01, 0.0)
    first_frame["theoretical_pair_net_return"] *= -100.0
    second_frame["theoretical_pair_net_return"] *= 100.0
    selected, _ = select_training_candidate(
        {first: first_frame, second: second_frame},
        period,
    )
    assert selected == first


def test_period_seal_requires_exit_inside_same_period():
    frame = _candidate_frame(0.01, 0.01).iloc[:2].copy()
    frame.loc[0, "signal_date"] = pd.Timestamp("2020-01-02")
    frame.loc[0, "exit_date"] = pd.Timestamp("2020-02-01")
    frame.loc[1, "signal_date"] = pd.Timestamp("2020-01-03")
    frame.loc[1, "exit_date"] = pd.Timestamp("2020-01-04")
    sealed = period_events(
        frame,
        Period(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31")),
    )
    assert len(sealed) == 1
    assert sealed.iloc[0]["signal_date"] == pd.Timestamp("2020-01-03")


def test_month_aggregation_removes_cross_sectional_event_duplication():
    period = Period(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31"))
    base = _candidate_frame(0.01, 0.01).iloc[:48].copy()
    duplicated = pd.concat(
        [base.assign(a_code=f"{index:06d}.SH") for index in range(100)],
        ignore_index=True,
    )
    base_fitness, base_folds = training_fitness(base, period)
    duplicate_fitness, duplicate_folds = training_fitness(duplicated, period)
    assert duplicate_fitness == pytest.approx(base_fitness)
    assert duplicate_folds == pytest.approx(base_folds)


def test_many_events_in_one_month_are_not_enough_training_history():
    frame = pd.concat([_candidate_frame(0.01, 0.01).iloc[[0]]] * 100)
    fitness, folds = training_fitness(
        frame,
        Period(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31")),
    )
    assert fitness == float("-inf")
    assert folds == []


def test_single_month_metrics_are_json_safe():
    import json

    frame = pd.concat([_candidate_frame(0.01, 0.01).iloc[[0]]] * 2)
    metrics = event_metrics(frame)
    assert metrics["signal_months"] == 1
    assert metrics["theoretical_pair_monthly_t_stat"] is None
    json.dumps(metrics, allow_nan=False)


def test_current_signal_uses_last_completed_close_without_future_row():
    panel = _event_panel().iloc[:6].copy()
    panel.loc[5, "spread_close"] = 0.50
    config = AHConfig(lookback=5, entry_z=1.0, horizon=20)
    signals = current_pair_signals(
        {("600000.SH", "00001.HK"): panel},
        config,
    )
    assert len(signals) == 1
    assert signals.iloc[0]["name"] == ""
    assert signals.iloc[0]["signal_date"] == pd.Timestamp("2024-01-06")
    assert signals.iloc[0]["cheap_leg"] == "H"

    polluted = panel.copy()
    polluted.loc[5, "a_adj_open"] = 1e30
    polluted.loc[5, "spread_open"] = -1e30
    polluted["volume"] = 1.0
    polluted.loc[5, "volume"] = 1e30
    changed = current_pair_signals(
        {("600000.SH", "00001.HK"): polluted},
        config,
    )
    pd.testing.assert_frame_equal(signals, changed)


def test_current_signals_keep_each_pairs_own_latest_common_date():
    first = _event_panel().iloc[:6].copy()
    second = _event_panel().iloc[:7].copy()
    first.loc[5, "spread_close"] = 0.50
    second.loc[6, "spread_close"] = -0.50
    signals = current_pair_signals(
        {
            ("600000.SH", "00001.HK"): first,
            ("000001.SZ", "00002.HK"): second,
        },
        AHConfig(lookback=5, entry_z=1.0, horizon=5),
    )
    by_code = signals.set_index("a_code")
    assert by_code.loc["600000.SH", "signal_date"] == pd.Timestamp("2024-01-06")
    assert by_code.loc["000001.SZ", "signal_date"] == pd.Timestamp("2024-01-07")
    assert by_code.loc["600000.SH", "cheap_leg"] == "H"
    assert by_code.loc["000001.SZ", "cheap_leg"] == "A"


def test_contiguous_split_is_chronological_and_outcome_independent():
    dates = pd.date_range("2020-01-01", periods=100)
    periods = derive_contiguous_periods(dates)
    assert periods["train"].end < periods["validation"].start
    assert periods["validation"].end < periods["test"].start
    assert (periods["train"].end - periods["train"].start).days == 59
    assert (periods["validation"].end - periods["validation"].start).days == 19

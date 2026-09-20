import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rl_test_data import financial_runtime_arrays

from ai.ga.config import canonicalize_individual_config
from testback.backtest import run_single_mode, run_static_config_backtest
from env.action_schema import ActionSchema
from env.backtest import (
    EpisodeSession,
    PreparedEpisode,
    prepare_episode_from_runtime,
    run_day_config_episode,
)


ROOT = Path(__file__).resolve().parents[1]


def write_canonical_runtime(path: Path, stocks: int = 30) -> None:
    dates = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-06-29"),
        dtype="datetime64[D]",
    )
    codes = np.asarray([f"{600001 + index:06d}.SH" for index in range(stocks)])
    day = np.arange(len(dates), dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    close = 10.0 + day * 0.002 + stock * 0.05
    open_prices = close * (1.0 + 0.0001 * ((stock % 5) - 2.0))
    preclose = np.empty_like(close)
    preclose[0] = close[0]
    preclose[1:] = close[:-1]
    volume = 1_000_000.0 + day * 1_000.0 + stock * 100.0
    panel = np.broadcast_to(1.0 + day * 0.001 + stock * 0.01, close.shape).copy()
    total_share = np.broadcast_to(
        (1e9 - np.arange(stocks, dtype=np.float64) * 1e7)[None, :], close.shape
    )
    np.savez_compressed(
        path,
        stock_codes=codes,
        trade_dates=dates,
        open=open_prices,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=volume,
        amount=volume * close,
        preClose=preclose,
        issue_price=np.full(stocks, 8.0),
        issue_date=np.full(stocks, dates[0], dtype="datetime64[D]"),
        st_mask=np.zeros(close.shape, dtype=np.bool_),
        listing_age=np.broadcast_to(
            np.arange(len(dates), dtype=np.int32)[:, None],
            close.shape,
        ).copy(),
        delisted_mask=np.zeros(close.shape, dtype=np.bool_),
        total_share=total_share,
        bps=panel,
        eps=panel,
        roe=panel,
        profit_yoy=panel,
        revenue_yoy=panel,
        operating_cf_ps=panel,
        gross_margin=panel,
        **financial_runtime_arrays(total_share),
    )


@pytest.fixture
def canonical_episode(tmp_path: Path) -> PreparedEpisode:
    runtime_path = tmp_path / "runtime.npz"
    write_canonical_runtime(runtime_path)
    return prepare_episode_from_runtime(
        runtime_path,
        "2020-06-20",
        "2020-06-24",
        prefilter_n=300,
    )


def current4_payload() -> dict:
    return json.loads((ROOT / "configs" / "config.json").read_text("utf-8"))


def assert_trace_equal(left, right) -> None:
    assert left.decision_dates == right.decision_dates
    assert left.next_decision_dates == right.next_decision_dates
    assert left.day_configs == right.day_configs
    assert left.residual_cash_reasons == right.residual_cash_reasons
    assert left.order_plans == right.order_plans
    assert left.fills == right.fills
    assert left.fee_breakdowns == right.fee_breakdowns
    assert left.account_events == right.account_events
    np.testing.assert_array_equal(left.actions, right.actions)
    np.testing.assert_array_equal(left.rewards, right.rewards)
    np.testing.assert_array_equal(left.portfolio_returns, right.portfolio_returns)
    np.testing.assert_array_equal(left.nav, right.nav)
    np.testing.assert_array_equal(left.cash, right.cash)
    np.testing.assert_array_equal(left.exposure, right.exposure)
    np.testing.assert_array_equal(
        left.full_investment_contract,
        right.full_investment_contract,
    )


def test_fixed_backtest_is_elementwise_the_same_env_session(
    canonical_episode: PreparedEpisode,
) -> None:
    payload = current4_payload()
    result = run_static_config_backtest(canonical_episode, payload)
    schema = ActionSchema()
    canonical, day_config = canonicalize_individual_config(
        payload,
        action_schema=schema,
    )
    direct_trace = run_day_config_episode(
        EpisodeSession(canonical_episode, action_schema=schema),
        lambda _observation: day_config,
    )

    assert result["individual_config"] == canonical
    assert_trace_equal(result["trace"], direct_trace)
    assert result["signal_dates"] == list(direct_trace.decision_dates)
    assert result["trade_dates"] == list(direct_trace.decision_dates)
    assert result["nav_dates"] == list(direct_trace.next_decision_dates)
    np.testing.assert_array_equal(
        result["daily_returns"],
        direct_trace.portfolio_returns * 100.0,
    )
    np.testing.assert_array_equal(result["daily_assets"], direct_trace.nav[1:])
    for index, snapshot in enumerate(result["daily_snapshots"]):
        assert snapshot["cash"] == pytest.approx(direct_trace.cash[index])
        assert snapshot["signal_date"] == direct_trace.decision_dates[index]
        assert snapshot["trade_date"] == direct_trace.decision_dates[index]
        assert snapshot["settlement_date"] == direct_trace.next_decision_dates[index]
        assert snapshot["total_asset"] == pytest.approx(direct_trace.nav[index + 1])
        assert snapshot["cash"] + snapshot["market_value"] == pytest.approx(
            snapshot["total_asset"]
        )
    assert result["full_investment_contract_satisfied"] is True
    assert result["position_lot_analytics_available"] is False
    assert result["holding_period_available"] is False
    assert result["benchmark_available"] is False
    assert result["per_year_metrics_available"] is False
    assert result["round_trip_count"] is None
    assert result["cleared_positions_count"] is None
    assert result["trade_log"]
    assert sum(row["total_fee"] for row in result["trade_log"]) > 0.0
    for row in result["trade_log"]:
        assert row["total_fee"] == pytest.approx(
            row["broker_commission"]
            + row["transfer_fee"]
            + row["stamp_tax"]
            + row["slippage"]
        )
    assert all(result["full_investment_contract"])
    # Only 30 stocks exist in this fixture; fixed buy_n=50 caps each at 2%.
    assert all(0.58 < exposure < 0.60 for exposure in result["daily_exposures"])
    assert set(result["trace"].residual_cash_reasons) == {
        "concentration_or_lot_capacity_exhausted"
    }


def test_trace_actions_are_owned_float32_and_independent_between_runs(canonical_episode):
    schema=ActionSchema()
    config=schema.decode(np.zeros(schema.action_dim,dtype=np.float32))
    traces=[run_day_config_episode(EpisodeSession(canonical_episode),lambda _:config) for _ in range(2)]
    left,right=traces
    assert_trace_equal(left,right)
    assert left.actions.dtype==np.float32 and left.actions.flags.owndata
    assert not np.shares_memory(left.actions,right.actions)
    expected=np.tile(schema.encode(config),(len(left.actions),1))
    np.testing.assert_array_equal(left.actions,expected)
    left.actions[:]=99
    np.testing.assert_array_equal(right.actions,expected)
    np.testing.assert_array_equal(schema.encode(config),expected[0])


def test_backtest_reports_real_delist_account_events(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime-delist.npz"
    write_canonical_runtime(runtime_path)
    with np.load(runtime_path, allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    delist_index = int(
        np.searchsorted(arrays["trade_dates"], np.datetime64("2020-06-21"))
    )
    arrays["delisted_mask"][delist_index:] = True
    np.savez_compressed(runtime_path, **arrays)
    episode = prepare_episode_from_runtime(
        runtime_path,
        "2020-06-20",
        "2020-06-22",
        prefilter_n=300,
    )

    result = run_static_config_backtest(episode, current4_payload())

    assert result["account_events"] == [
        dict(event) for event in result["trace"].account_events
    ]
    assert result["delist_events"]
    assert result["delist_count"] == len(result["delist_events"])
    assert all(
        event["type"] == "delist_write_off"
        for event in result["delist_events"]
    )
    assert all(event["effective_date"] >= "2020-06-21" for event in result["delist_events"])


def test_backtest_rejects_the_legacy_array_signature() -> None:
    with pytest.raises(TypeError, match="PreparedEpisode"):
        run_static_config_backtest({}, {"weights": {}})


def test_prefilter_metadata_does_not_change_the_canonical_day_config() -> None:
    payload = current4_payload()
    with_prefilter, _ = canonicalize_individual_config(payload)
    payload["individual_config"].pop("prefilter_n")
    without_prefilter, _ = canonicalize_individual_config(payload)

    assert with_prefilter == without_prefilter
    assert "prefilter_n" not in with_prefilter


def test_fixed_cli_adapter_writes_a_canonical_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "env.backtest.ObservationBuilder",
        lambda *_args, **_kwargs: pytest.fail("fixed CLI must not build unused actor inputs"),
    )
    runtime_path = tmp_path / "runtime.npz"
    write_canonical_runtime(runtime_path)
    output_dir = tmp_path / "result"
    result = run_single_mode(
        SimpleNamespace(
            individual_config=str(ROOT / "configs" / "config.json"),
            runtime_path=str(runtime_path),
            output_dir=str(output_dir),
            start_date="20200620",
            end_date="20200624",
            lookback=64,
            initial_cash=1_000_000.0,
        ),
        {"save_charts": False},
    )

    record = json.loads((output_dir / "record.json").read_text("utf-8"))
    assert result["full_investment_contract_satisfied"] is True
    assert record["full_investment_contract_satisfied"] is True
    assert record["individual_config"] == result["individual_config"]
    assert all("cash" in snapshot for snapshot in result["daily_snapshots"])
    expected_total_fees = sum(row["total_fee"] for row in result["trade_log"])
    assert record["metrics"]["total_fees"] == pytest.approx(expected_total_fees)

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from env.action_schema import CORE_FACTOR_NAMES
from env.contracts import AccountState
from env.observation import (
    MARKET_FEATURE_NAMES,
    PORTFOLIO_FEATURE_NAMES,
    POSITION_FEATURE_NAMES,
    STOCK_FEATURE_NAMES,
    ObservationBuilder,
)
from factor import precompute_factors
from offline_data import load_runtime_slice
from test_rl_runtime_slice import _runtime_arrays


def synthetic_inputs(
    *,
    date_count: int = 8,
    stock_codes: tuple[str, ...] = ("000001.SZ", "600000.SH", "300001.SZ"),
):
    stock_count = len(stock_codes)
    row = np.arange(date_count, dtype=np.float32)[:, None]
    column = np.arange(stock_count, dtype=np.float32)[None, :]
    base = row * 2.0 + column + 10.0
    data = {
        "open": base.copy(),
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.5,
        "volume": 1_000.0 + base * 10.0,
        "amount": 100_000.0 + base * 1_000.0,
        "preClose": base - 0.25,
        "total_share": 1_000_000.0 + base * 100.0,
        "bps": 2.0 + base / 100.0,
        "eps": 0.5 + base / 1_000.0,
        "roe": 0.1 + base / 10_000.0,
        "profit_yoy": base / 100.0,
        "revenue_yoy": base / 200.0,
        "operating_cf_ps": base / 300.0,
        "gross_margin": 0.2 + base / 10_000.0,
        "st_mask": np.zeros((date_count, stock_count), dtype=np.bool_),
        "issue_price": np.linspace(8.0, 12.0, stock_count, dtype=np.float32),
        "stock_names": np.asarray([f"stock-{i}" for i in range(stock_count)]),
    }
    dates = np.arange(
        np.datetime64("2024-01-02"),
        np.datetime64("2024-01-02") + np.timedelta64(date_count, "D"),
    )
    ranks = np.empty((date_count, len(CORE_FACTOR_NAMES), stock_count), dtype=np.float32)
    for factor_index in range(len(CORE_FACTOR_NAMES)):
        ranks[:, factor_index, :] = (
            0.1 * factor_index + row / (date_count * 10.0) + column / (stock_count * 10.0)
        )
    validity = np.ones_like(ranks, dtype=np.bool_)
    filters = np.zeros((date_count, 3, stock_count), dtype=np.bool_)
    runtime_schema_hash = "1" * 64
    runtime = SimpleNamespace(
        stock_codes=stock_codes,
        trade_dates=dates,
        data=data,
        manifest=SimpleNamespace(schema_hash=runtime_schema_hash),
    )
    factors = SimpleNamespace(
        factor_names=tuple(CORE_FACTOR_NAMES),
        stock_codes=stock_codes,
        trade_dates=dates.copy(),
        ranks=ranks,
        validity=validity,
        filters=filters,
        schema_hash="2" * 64,
        runtime_schema_hash=runtime_schema_hash,
        rank_universe_sha256="3" * 64,
    )
    return runtime, factors


def copy_inputs(runtime, factors):
    copied_runtime = SimpleNamespace(
        stock_codes=tuple(runtime.stock_codes),
        trade_dates=runtime.trade_dates.copy(),
        data={name: values.copy() for name, values in runtime.data.items()},
        manifest=SimpleNamespace(schema_hash=runtime.manifest.schema_hash),
    )
    copied_factors = SimpleNamespace(
        factor_names=tuple(factors.factor_names),
        stock_codes=tuple(factors.stock_codes),
        trade_dates=factors.trade_dates.copy(),
        ranks=factors.ranks.copy(),
        validity=factors.validity.copy(),
        filters=factors.filters.copy(),
        schema_hash=factors.schema_hash,
        runtime_schema_hash=factors.runtime_schema_hash,
        rank_universe_sha256=factors.rank_universe_sha256,
    )
    return copied_runtime, copied_factors


def sample_account(runtime, decision_index: int = 5) -> AccountState:
    code = runtime.stock_codes[0]
    quantity = 10
    mark = float(runtime.data["open"][decision_index, 0])
    market_value = mark * quantity
    nav = 500.0 + market_value
    return AccountState(
        cash=500.0,
        positions={code: quantity},
        sellable_positions={code: 8},
        average_costs={code: mark - 1.0},
        last_prices={code: mark - 0.5},
        nav=nav,
        peak_nav=nav + 50.0,
    )


def assert_static_equal(left, right) -> None:
    assert left.decision_date == right.decision_date
    np.testing.assert_array_equal(left.stock_panel, right.stock_panel)
    np.testing.assert_array_equal(left.market_panel, right.market_panel)
    np.testing.assert_array_equal(left.feature_mask, right.feature_mask)
    np.testing.assert_array_equal(left.stock_mask, right.stock_mask)
    np.testing.assert_array_equal(left.time_mask, right.time_mask)


def test_observation_builder_shapes_masks_and_account_state() -> None:
    runtime, factors = synthetic_inputs()
    runtime.data["eps"][5, 0] = np.nan
    runtime.data["eps"][5, 1] = 0.0
    runtime.data["open"][5, 2] = np.nan
    factors.validity[5, 0, 1] = False
    factors.filters[5, :, :] = True  # Soft filters must not alter stock_mask.
    builder = ObservationBuilder(runtime, factors, lookback=6)

    observation = builder.build(5, sample_account(runtime))

    assert observation.stock_panel.shape == (6, 3, len(STOCK_FEATURE_NAMES))
    assert observation.market_panel.shape == (6, len(MARKET_FEATURE_NAMES))
    assert observation.position_panel.shape == (3, len(POSITION_FEATURE_NAMES))
    assert observation.portfolio.shape == (len(PORTFOLIO_FEATURE_NAMES),)
    assert observation.feature_mask.shape == observation.stock_panel.shape
    assert observation.stock_mask[-1].tolist() == [True, True, False]
    eps_index = STOCK_FEATURE_NAMES.index("eps")
    assert observation.stock_panel[-1, 0, eps_index] == 0.0
    assert not observation.feature_mask[-1, 0, eps_index]
    assert observation.stock_panel[-1, 1, eps_index] == 0.0
    assert observation.feature_mask[-1, 1, eps_index]
    factor_index = STOCK_FEATURE_NAMES.index(f"factor_rank.{CORE_FACTOR_NAMES[0]}")
    assert not observation.feature_mask[-1, 1, factor_index]
    assert observation.position_panel[0, 0] == 1.0
    assert observation.position_panel[0, 4] == pytest.approx(0.8)
    assert observation.portfolio[0] == 500.0
    assert np.isfinite(observation.stock_panel).all()
    assert np.isfinite(observation.market_panel).all()
    account_part = builder.build_account(5, sample_account(runtime))
    assert account_part.current_factor_ranks.shape == (len(CORE_FACTOR_NAMES), 3)
    assert account_part.current_factor_validity.shape == (len(CORE_FACTOR_NAMES), 3)


def test_observation_padding_and_issue_price_are_masked_before_listing() -> None:
    runtime, factors = synthetic_inputs()
    runtime.data["open"][:2, 0] = np.nan
    builder = ObservationBuilder(runtime, factors, lookback=6)

    static = builder.build_static(2)

    assert static.time_mask.tolist() == [False, False, False, True, True, True]
    issue_index = STOCK_FEATURE_NAMES.index("issue_price")
    assert not static.feature_mask[3, 0, issue_index]
    assert not static.feature_mask[4, 0, issue_index]
    assert static.feature_mask[5, 0, issue_index]
    assert np.all(static.stock_panel[~static.feature_mask] == 0.0)


def test_future_and_t_post_open_fields_cannot_change_t_observation() -> None:
    runtime, factors = synthetic_inputs()
    baseline = ObservationBuilder(runtime, factors, lookback=4).build_static(5)

    changed_runtime, changed_factors = copy_inputs(runtime, factors)
    for values in changed_runtime.data.values():
        if values.ndim == 2 and np.issubdtype(values.dtype, np.number):
            values[6:] = 9_999.0
    changed_factors.ranks[6:] = 0.999
    future_changed = ObservationBuilder(changed_runtime, changed_factors, lookback=4).build_static(5)
    assert_static_equal(baseline, future_changed)

    changed_runtime, changed_factors = copy_inputs(runtime, factors)
    for field in ("high", "low", "close", "volume", "amount", "total_share"):
        changed_runtime.data[field][5] = 8_888.0
    post_open_changed = ObservationBuilder(changed_runtime, changed_factors, lookback=4).build_static(5)
    assert_static_equal(baseline, post_open_changed)

    changed_runtime, changed_factors = copy_inputs(runtime, factors)
    changed_runtime.data["close"][4, 0] += 7.0
    lag_changed = ObservationBuilder(changed_runtime, changed_factors, lookback=4).build_static(5)
    assert not np.array_equal(baseline.stock_panel, lag_changed.stock_panel)

    changed_runtime, changed_factors = copy_inputs(runtime, factors)
    changed_runtime.data["open"][5, 0] += 7.0
    open_changed = ObservationBuilder(changed_runtime, changed_factors, lookback=4).build_static(5)
    assert not np.array_equal(baseline.stock_panel, open_changed.stock_panel)


def test_builder_fails_fast_on_alignment_or_unregistered_numeric_field() -> None:
    runtime, factors = synthetic_inputs()
    factors.stock_codes = tuple(reversed(factors.stock_codes))
    with pytest.raises(ValueError, match="stock order"):
        ObservationBuilder(runtime, factors)

    runtime, factors = synthetic_inputs()
    factors.factor_names = tuple(reversed(factors.factor_names))
    with pytest.raises(ValueError, match="factor order"):
        ObservationBuilder(runtime, factors)

    runtime, factors = synthetic_inputs()
    factors.runtime_schema_hash = "9" * 64
    with pytest.raises(ValueError, match="different runtime schema"):
        ObservationBuilder(runtime, factors)

    runtime, factors = synthetic_inputs()
    runtime.data["new_unregistered_panel"] = np.zeros_like(runtime.data["open"])
    with pytest.raises(ValueError, match="unregistered runtime field"):
        ObservationBuilder(runtime, factors)


def test_registered_optional_runtime_field_upgrades_schema_instead_of_being_dropped() -> None:
    runtime, factors = synthetic_inputs()
    runtime.data["star_st_mask"] = np.zeros_like(runtime.data["st_mask"])
    runtime.data["star_st_mask"][5, 1] = True

    builder = ObservationBuilder(runtime, factors, lookback=4)
    static = builder.build_static(5)

    assert "star_st_mask" in builder.schema.stock_feature_names
    feature_index = builder.schema.stock_feature_names.index("star_st_mask")
    assert static.stock_panel[-1, 1, feature_index] == 1.0
    assert static.feature_mask[-1, 1, feature_index]
    assert builder.schema.stock_feature_count == len(STOCK_FEATURE_NAMES) + 1


def test_static_part_can_be_reused_but_not_for_another_date() -> None:
    runtime, factors = synthetic_inputs()
    builder = ObservationBuilder(runtime, factors, lookback=4)
    static = builder.build_static(5)
    account = sample_account(runtime, 5)

    direct = builder.build(5, account)
    reused = builder.build(5, account, static=static)
    np.testing.assert_array_equal(direct.stock_panel, reused.stock_panel)
    np.testing.assert_array_equal(direct.position_panel, reused.position_panel)

    with pytest.raises(ValueError, match="different decision date"):
        builder.build(4, sample_account(runtime, 4), static=static)


def test_builder_consumes_public_runtime_and_factor_contracts(tmp_path) -> None:
    data = _runtime_arrays()
    path = tmp_path / "runtime.npz"
    np.savez(path, **data)
    runtime = load_runtime_slice(path, data["trade_dates"][140], data["trade_dates"][150])
    factors = precompute_factors(runtime)
    builder = ObservationBuilder(runtime, factors, lookback=64)

    observation = builder.build(
        runtime.decision_start,
        AccountState(cash=1_000_000.0, nav=1_000_000.0, peak_nav=1_000_000.0),
    )

    assert observation.schema_version == builder.schema.identifier
    assert builder.schema.runtime_schema_hash == runtime.manifest.schema_hash
    assert builder.schema.factor_schema_hash == factors.schema_hash
    assert observation.stock_panel.shape == (64, runtime.n_stocks, len(STOCK_FEATURE_NAMES))

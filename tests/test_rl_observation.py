from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from env.action_schema import ActionSchema, CORE_FACTOR_NAMES, CORE_FILTER_NAMES
from env.contracts import AccountState, PolicyMemory
from env.observation import (ObservationBuilder, ObservationSchema, LAGGED_RUNTIME_FIELDS,
    OBSERVATION_SCHEMA_VERSION, POSITION_FEATURE_NAMES, PORTFOLIO_FEATURE_NAMES,
    QUARANTINED_RUNTIME_FIELDS, STOCK_FEATURE_NAMES, RAW_MISSING_VALUE)
from offline_data.financial_versions import RAW_FINANCIAL_VALUE_NAMES
from env.planner import DayMarketData
from rl_test_data import financial_runtime_arrays


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
        "listing_age": np.broadcast_to(
            np.arange(date_count, dtype=np.int32)[:, None],
            (date_count, stock_count),
        ).copy(),
        "delisted_mask": np.zeros((date_count, stock_count), dtype=np.bool_),
        "issue_price": np.linspace(8.0, 12.0, stock_count, dtype=np.float32),
        "issue_date": np.full(
            stock_count,
            np.datetime64("2024-01-02"),
            dtype="datetime64[D]",
        ),
    }
    data.update(financial_runtime_arrays(data["total_share"]))
    dates = np.arange(
        np.datetime64("2024-01-02"),
        np.datetime64("2024-01-02") + np.timedelta64(date_count, "D"),
    )
    ranks = np.empty((date_count, len(CORE_FACTOR_NAMES), stock_count), dtype=np.float32)
    for factor_index in range(len(CORE_FACTOR_NAMES)):
        ranks[:, factor_index, :] = (
            0.075 * factor_index + row / (date_count * 10.0) + column / (stock_count * 10.0)
        )
    binary_index = CORE_FACTOR_NAMES.index("PBBelowTwoROEAbove10Signal")
    ranks[:, binary_index, :] = ((row + column) % 3 == 0).astype(np.float32)
    validity = np.ones_like(ranks, dtype=np.bool_)
    filters = np.zeros(
        (date_count, len(CORE_FILTER_NAMES), stock_count),
        dtype=np.bool_,
    )
    runtime_schema_hash = "1" * 64
    runtime = SimpleNamespace(
        stock_codes=stock_codes,
        trade_dates=dates,
        data=data,
        manifest=SimpleNamespace(schema_hash=runtime_schema_hash),
    )
    factors = SimpleNamespace(
        factor_names=tuple(CORE_FACTOR_NAMES),
        filter_names=tuple(CORE_FILTER_NAMES),
        stock_codes=stock_codes,
        trade_dates=dates.copy(),
        ranks=ranks,
        validity=validity,
        filters=filters,
        schema_hash="2" * 64,
        runtime_schema_hash=runtime_schema_hash,
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
        filter_names=tuple(factors.filter_names),
        stock_codes=tuple(factors.stock_codes),
        trade_dates=factors.trade_dates.copy(),
        ranks=factors.ranks.copy(),
        validity=factors.validity.copy(),
        filters=factors.filters.copy(),
        schema_hash=factors.schema_hash,
        runtime_schema_hash=factors.runtime_schema_hash,
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


def make_builder(runtime, factors, *, lookback=4, action_schema=None):
    markets = tuple(DayMarketData(
        decision_date=str(date), stock_codes=runtime.stock_codes,
        factor_ranks={name:factors.ranks[i,j] for j,name in enumerate(factors.factor_names)},
        factor_validity={name:factors.validity[i,j] for j,name in enumerate(factors.factor_names)},
        filter_masks={name:factors.filters[i,j] for j,name in enumerate(factors.filter_names)},
        open_prices=runtime.data["open"][i], preclose_prices=runtime.data["preClose"][i],
        issue_prices=np.where(runtime.data["issue_date"]==date,runtime.data["issue_price"],np.nan),
        st_mask=runtime.data["st_mask"][i], delisted_mask=runtime.data["delisted_mask"][i],
        listing_age=runtime.data["listing_age"][i]).seal(borrow_readonly=True)
        for i,date in enumerate(runtime.trade_dates))
    return ObservationBuilder(runtime,factors,lookback=lookback,day_markets=markets,action_schema=action_schema)


def assert_static_equal(left, right):
    assert left.decision_date == right.decision_date
    for name in ("stock_panel","time_mask","pit_universe_mask"):
        np.testing.assert_array_equal(getattr(left,name),getattr(right,name))


def test_full_raw_vocabulary_has_no_statistical_market_panel():
    runtime,factors=synthetic_inputs()
    builder=make_builder(runtime,factors)
    obs=builder.build(5,sample_account(runtime))
    assert builder.schema.version==OBSERVATION_SCHEMA_VERSION
    assert obs.stock_panel.shape==(4,3,37)
    assert not any(name.startswith(("factor_rank.", "filter_pass.")) for name in builder.schema.stock_feature_names)
    assert obs.position_panel.shape==(3,4)
    assert obs.portfolio.shape==(4,)
    assert obs.policy_history.shape==(4,15)
    assert not hasattr(obs,"market_panel")
    assert not any(name in builder.schema.stock_feature_names for name in QUARANTINED_RUNTIME_FIELDS)
    assert ObservationSchema.from_dict(builder.schema.to_dict())==builder.schema


def test_raw_rows_preserve_current_open_and_completed_ohlcva():
    runtime,factors=synthetic_inputs()
    builder=make_builder(runtime,factors)
    obs=builder.build_static(5)
    names=builder.schema.stock_feature_names
    for offset,day in enumerate(range(2,6)):
        for name in ("open","preClose"):
            np.testing.assert_array_equal(obs.stock_panel[offset,:,names.index(name)],runtime.data[name][day])
        for name in LAGGED_RUNTIME_FIELDS:
            np.testing.assert_array_equal(obs.stock_panel[offset,:,names.index(name+"_lag1")],runtime.data[name][day-1])
        for name in RAW_FINANCIAL_VALUE_NAMES:
            np.testing.assert_array_equal(obs.stock_panel[offset,:,names.index(name)],runtime.data[name][day].astype(np.float32))


def test_current_completed_prices_and_future_rows_cannot_leak():
    runtime,factors=synthetic_inputs()
    before=make_builder(runtime,factors).build_static(5)
    changed,cf=copy_inputs(runtime,factors)
    for name in LAGGED_RUNTIME_FIELDS:
        if name not in ("open","preClose"):
            changed.data[name][5:]*=900
        changed.data[name][6:]*=20
    cf.ranks[6:]=0.99
    assert_static_equal(before,make_builder(changed,cf).build_static(5))


def test_pit_padding_and_first_listed_lag_are_unavailable():
    runtime,factors=synthetic_inputs()
    runtime.data["listing_age"][:3,1]=-1
    runtime.data["listing_age"][3:,1]=np.arange(5)
    builder=make_builder(runtime,factors,lookback=6)
    obs=builder.build_static(3)
    assert obs.time_mask.tolist()==[False,False,True,True,True,True]
    assert not obs.pit_universe_mask[:5,1].any()
    assert not obs.stock_panel[:5,1].any()
    for name in LAGGED_RUNTIME_FIELDS:
        assert obs.stock_panel[-1,1,builder.schema.stock_feature_names.index(name+"_lag1")]==RAW_MISSING_VALUE
    assert obs.stock_panel[-1,1,builder.schema.stock_feature_names.index("open")]==runtime.data["open"][3,1]


def test_real_zero_negative_financial_and_missing_volume_are_distinct():
    runtime,factors=synthetic_inputs()
    runtime.data["volume"][4,0]=0
    runtime.data["volume"][4,1]=np.nan
    financial = RAW_FINANCIAL_VALUE_NAMES[0]
    runtime.data[financial][5,:] = [0, -1, np.nan]
    builder=make_builder(runtime,factors)
    raw=builder.build_static(5).stock_panel[-1]
    names=builder.schema.stock_feature_names
    assert raw[0,names.index("volume_lag1")]==0
    assert raw[1,names.index("volume_lag1")]==RAW_MISSING_VALUE
    np.testing.assert_array_equal(raw[:,names.index(financial)], [0, -1, RAW_MISSING_VALUE])


def test_legality_is_shared_with_planner_and_fixed_schema_controls():
    runtime,factors=synthetic_inputs()
    runtime.data["listing_age"][:]=1000
    runtime.data["preClose"][5]=10
    runtime.data["open"][5]=[11,9,0]
    schema=ActionSchema()
    builder=make_builder(runtime,factors,action_schema=schema)
    raw=builder.build_static(5).stock_panel[-1]
    legality=builder.day_markets[5].trade_legality(schema.fixed_limit_up_protection)
    names=builder.schema.stock_feature_names
    np.testing.assert_array_equal(raw[:,names.index("price_buy_allowed")],legality.buy_allowed)
    np.testing.assert_array_equal(raw[:,names.index("price_sell_allowed")],legality.sell_allowed)
    assert builder.day_markets[5].trade_legality(schema.fixed_limit_up_protection) is legality
    disabled=make_builder(runtime,factors,action_schema=replace(schema,fixed_limit_up_protection=False))
    assert disabled.schema.action_schema_hash!=builder.schema.action_schema_hash
    assert disabled.build_static(5).stock_panel[-1,0,names.index("price_sell_allowed")]==1
    assert raw[0,names.index("price_sell_allowed")]==0


def test_account_preserves_full_axis_raw_holdings_even_nonmember():
    runtime,factors=synthetic_inputs()
    runtime.data["delisted_mask"][5,0]=True
    account=sample_account(runtime)
    builder=make_builder(runtime,factors)
    obs=builder.build(5,account)
    np.testing.assert_array_equal(obs.position_panel[0],[10,19,8,19.5])
    np.testing.assert_array_equal(obs.portfolio,[account.cash,account.nav,account.peak_nav,account.max_drawdown])
    assert not obs.pit_universe_mask[-1,0]
    assert not obs.stock_panel[-1,0].any()
    assert not obs.policy_history.any()


def test_schema_and_date_mismatch_fail_closed():
    runtime,factors=synthetic_inputs()
    builder=make_builder(runtime,factors)
    with pytest.raises(ValueError,match="identity"):
        builder.build(4,sample_account(runtime,4),static=builder.build_static(5))
    with pytest.raises(ValueError,match="unsupported"):
        replace(builder.schema,version="old")

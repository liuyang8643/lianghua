from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from env.action_schema import CORE_FACTOR_NAMES, CORE_FILTER_NAMES
from env.contracts import AccountState, DayConfig, RebalanceMode
from env.planner import DayMarketData, DayPlanner
from env.simulator import DaySimulator


def _config(
    *,
    primary_factor: str = CORE_FACTOR_NAMES[0],
    factor_weights: dict[str, float] | None = None,
    filter_flags: dict[str, bool] | None = None,
    target_exposure: float = 0.8,
    buy_n: int = 1,
    sell_m: int = 1,
    rebalance_now: bool = True,
    mode: RebalanceMode = RebalanceMode.EQUALIZE,
    limit_up_protection: bool = False,
    band: float = 0.0,
) -> DayConfig:
    if factor_weights is None:
        factor_weights = {
            name: 1.0 if name == primary_factor else 0.0
            for name in CORE_FACTOR_NAMES
        }
    enabled = {name: factor_weights[name] > 0.0 for name in CORE_FACTOR_NAMES}
    return DayConfig(
        factor_weights=factor_weights,
        factor_enabled=enabled,
        filter_flags=filter_flags
        or {name: False for name in CORE_FILTER_NAMES},
        target_exposure=target_exposure,
        buy_n=buy_n,
        sell_m=sell_m,
        rebalance_now=rebalance_now,
        rebalance_mode=mode,
        limit_up_protection=limit_up_protection,
        rebalance_band_pct=band,
    )


def _market(
    *,
    codes: tuple[str, ...] = (
        "600000.SH",
        "600001.SH",
        "600002.SH",
        "600003.SH",
    ),
    ranks: dict[str, np.ndarray] | None = None,
    validity: dict[str, np.ndarray] | None = None,
    filters: dict[str, np.ndarray] | None = None,
    opens: np.ndarray | None = None,
    precloses: np.ndarray | None = None,
    issues: np.ndarray | None = None,
    st: np.ndarray | None = None,
    listing_age: np.ndarray | None = None,
    decision_date: str = "2026-08-20",
) -> DayMarketData:
    size = len(codes)
    default_ranks = np.linspace(1.0, 0.1, size)
    ranks = ranks or {
        name: default_ranks.copy() for name in CORE_FACTOR_NAMES
    }
    validity = validity or {
        name: np.ones(size, dtype=bool) for name in CORE_FACTOR_NAMES
    }
    filters = filters or {
        name: np.ones(size, dtype=bool) for name in CORE_FILTER_NAMES
    }
    return DayMarketData(
        decision_date=decision_date,
        stock_codes=codes,
        factor_ranks=ranks,
        factor_validity=validity,
        filter_masks=filters,
        open_prices=np.full(size, 10.0) if opens is None else opens,
        preclose_prices=(
            np.full(size, 10.0) if precloses is None else precloses
        ),
        issue_prices=np.full(size, 10.0) if issues is None else issues,
        st_mask=np.zeros(size, dtype=bool) if st is None else st,
        listing_age=(
            np.full(size, 10, dtype=np.int32)
            if listing_age is None
            else listing_age
        ),
    )


def _cash_account(cash: float = 100_000.0) -> AccountState:
    return AccountState(cash=cash, nav=cash, peak_nav=cash)


def test_dynamic_factor_weights_change_ranking_without_recomputing_factors():
    ranks = {
        CORE_FACTOR_NAMES[0]: np.array([1.0, 0.7, 0.4, 0.1]),
        CORE_FACTOR_NAMES[1]: np.array([0.1, 0.4, 0.7, 1.0]),
        CORE_FACTOR_NAMES[2]: np.array([0.1, 0.1, 0.1, 0.1]),
        CORE_FACTOR_NAMES[3]: np.array([0.1, 0.1, 0.1, 0.1]),
    }
    market = _market(ranks=ranks)
    planner = DayPlanner()

    first = planner.plan(market, _cash_account(), _config(primary_factor=CORE_FACTOR_NAMES[0]))
    second = planner.plan(market, _cash_account(), _config(primary_factor=CORE_FACTOR_NAMES[1]))

    assert first.diagnostics["buy_n_stocks"] == ("600000.SH",)
    assert second.diagnostics["buy_n_stocks"] == ("600003.SH",)
    assert set(first.buy_orders) == {"600000.SH"}
    assert set(second.buy_orders) == {"600003.SH"}


@pytest.mark.parametrize("filter_name", CORE_FILTER_NAMES)
def test_each_soft_filter_toggle_changes_eligibility_only_when_enabled(filter_name):
    filters = {
        name: np.ones(4, dtype=bool) for name in CORE_FILTER_NAMES
    }
    filters[filter_name] = np.array([False, True, True, True])
    market = _market(filters=filters)
    planner = DayPlanner()

    disabled = planner.plan(market, _cash_account(), _config())
    flags = {name: name == filter_name for name in CORE_FILTER_NAMES}
    enabled = planner.plan(market, _cash_account(), _config(filter_flags=flags))

    assert disabled.diagnostics["buy_n_stocks"] == ("600000.SH",)
    assert enabled.diagnostics["buy_n_stocks"] == ("600001.SH",)
    assert enabled.diagnostics["filter_rejected"][filter_name] == 1


def test_legality_skips_limit_up_suspended_unlisted_and_missing_preclose():
    codes = (
        "600000.SH",
        "600001.SH",
        "600002.SH",
        "600003.SH",
        "600004.SH",
    )
    ranks = {
        name: np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        for name in CORE_FACTOR_NAMES
    }
    market = _market(
        codes=codes,
        ranks=ranks,
        opens=np.array([11.0, np.nan, 10.0, 10.0, 10.0]),
        precloses=np.array([10.0, 10.0, 10.0, np.nan, 10.0]),
        listing_age=np.array([10, 10, -1, 10, 10], dtype=np.int32),
    )

    plan = DayPlanner(diagnostics="full").plan(
        market, _cash_account(), _config()
    )

    assert plan.diagnostics["buy_n_stocks"] == ("600004.SH",)
    assert plan.diagnostics["buy_legality_rejections"] == {
        "600000.SH": "limit_up",
        "600003.SH": "missing_preclose",
    }
    assert plan.diagnostics["market_rejections"] == {
        "600001.SH": "suspended_or_missing_open",
        "600002.SH": "not_listed",
    }


def test_sell_m_prefers_buy_legal_names_before_raw_rank_fill():
    ranks = {
        name: np.array([1.0, 0.9, 0.8, 0.7])
        for name in CORE_FACTOR_NAMES
    }
    market = _market(
        ranks=ranks,
        opens=np.array([11.0, 10.0, 10.0, 10.0]),
    )
    plan = DayPlanner().plan(
        market,
        _cash_account(),
        _config(buy_n=1, sell_m=2),
    )

    assert plan.diagnostics["buy_n_stocks"] == ("600001.SH",)
    assert plan.diagnostics["sell_m_stocks"] == (
        "600001.SH",
        "600002.SH",
    )


def test_limit_down_blocks_sell_and_limit_up_sell_protection_is_dynamic():
    codes = ("600003.SH", "600000.SH", "600001.SH", "600002.SH")
    market = _market(
        codes=codes,
        opens=np.array([10.0, 9.0, 11.0, 10.0]),
    )
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 100, "600001.SH": 100, "600002.SH": 100},
        sellable_positions={
            "600000.SH": 100,
            "600001.SH": 100,
            "600002.SH": 100,
        },
        last_prices={
            "600000.SH": 9.0,
            "600001.SH": 11.0,
            "600002.SH": 10.0,
        },
        nav=3_000.0,
        peak_nav=3_000.0,
    )

    unprotected = DayPlanner().plan(
        market,
        account,
        _config(target_exposure=0.0, limit_up_protection=False),
    )
    protected = DayPlanner().plan(
        market,
        account,
        _config(target_exposure=0.0, limit_up_protection=True),
    )

    assert unprotected.sell_orders == (
        ("600001.SH", -1),
        ("600002.SH", -1),
    )
    assert unprotected.diagnostics["sell_legality_rejections"] == {
        "600000.SH": "limit_down"
    }
    assert protected.sell_orders == (("600002.SH", -1),)
    assert protected.diagnostics["sell_legality_rejections"] == {
        "600000.SH": "limit_down",
        "600001.SH": "limit_up_protected",
    }


def test_rebalance_now_false_always_returns_no_orders():
    account = AccountState(
        cash=0.0,
        positions={"600003.SH": 1000},
        sellable_positions={"600003.SH": 1000},
        last_prices={"600003.SH": 10.0},
        nav=10_000.0,
        peak_nav=10_000.0,
    )

    plan = DayPlanner().plan(
        _market(), account, _config(rebalance_now=False)
    )

    assert plan.sell_orders == ()
    assert plan.buy_orders == {}
    assert plan.diagnostics["no_trade_reason"] == "rebalance_now_false"


def test_empty_sellable_mapping_means_no_position_is_sellable():
    account = AccountState(
        cash=0.0,
        positions={"600003.SH": 100},
        sellable_positions={},
        last_prices={"600003.SH": 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )

    plan = DayPlanner().plan(
        _market(),
        account,
        _config(target_exposure=0.0),
    )

    assert plan.sell_orders == ()


def test_equalize_and_replace_only_preserve_distinct_legacy_semantics():
    account = AccountState(
        cash=1_000.0,
        positions={"600000.SH": 1000, "600003.SH": 1000},
        sellable_positions={"600000.SH": 1000, "600003.SH": 1000},
        average_costs={"600000.SH": 10.0, "600003.SH": 10.0},
        last_prices={"600000.SH": 10.0, "600003.SH": 10.0},
        nav=21_000.0,
        peak_nav=21_000.0,
    )
    market = _market()

    equalize = DayPlanner().plan(
        market,
        account,
        _config(target_exposure=0.0, mode=RebalanceMode.EQUALIZE),
    )
    replace = DayPlanner().plan(
        market,
        account,
        _config(target_exposure=0.0, mode=RebalanceMode.REPLACE_ONLY),
    )

    assert equalize.sell_orders == (("600000.SH", -1), ("600003.SH", -1))
    assert replace.sell_orders == (("600003.SH", -1),)
    assert replace.diagnostics["skip_reasons"]["_target_exposure"] == (
        "replace_only_does_not_trim_retained_positions"
    )


def test_target_exposure_and_rebalance_band_control_equalization():
    account = AccountState(
        cash=2_000.0,
        positions={"600000.SH": 900},
        sellable_positions={"600000.SH": 900},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=11_000.0,
        peak_nav=11_000.0,
    )
    market = _market()

    no_band = DayPlanner().plan(
        market,
        account,
        _config(target_exposure=1.0, band=0.0),
    )
    wide_band = DayPlanner().plan(
        market,
        account,
        _config(target_exposure=1.0, band=0.10),
    )

    assert no_band.diagnostics["base_target"] == pytest.approx(10_000.0)
    assert no_band.buy_orders == {"600000.SH": 100}
    assert wide_band.buy_orders == {}
    assert wide_band.diagnostics["skip_reasons"]["600000.SH"] == (
        "within_or_above_target_band"
    )


def test_kcb_buy_uses_200_minimum_then_one_share_increments():
    market = _market(codes=("688001.SH",))
    planner = DayPlanner()

    below_minimum = planner.plan(
        market,
        _cash_account(10_000.0),
        _config(target_exposure=0.10),
    )
    one_share_increment = planner.plan(
        market,
        _cash_account(10_000.0),
        _config(target_exposure=0.3996),
    )

    assert below_minimum.buy_orders == {"688001.SH": 200}
    assert one_share_increment.buy_orders == {"688001.SH": 333}

    executed = DaySimulator().step(
        _cash_account(10_000.0),
        one_share_increment,
        {"688001.SH": 10.0},
        {"688001.SH": 10.0},
        close_prices={"688001.SH": 10.0},
        next_preclose_prices={"688001.SH": 10.0},
    )
    assert executed.fills[0].quantity == 333
    assert executed.account_state.positions == {"688001.SH": 333}


def test_kcb_partial_sell_uses_200_minimum_then_one_share_increments():
    code = "688001.SH"
    market = _market(codes=(code,))
    account = AccountState(
        cash=0.0,
        positions={code: 1000},
        sellable_positions={code: 1000},
        average_costs={code: 10.0},
        last_prices={code: 10.0},
        nav=10_000.0,
        peak_nav=10_000.0,
    )
    planner = DayPlanner()

    legal_partial = planner.plan(
        market,
        account,
        _config(target_exposure=0.90),
    )
    below_minimum = planner.plan(
        market,
        account,
        _config(target_exposure=1.0),
    )

    assert legal_partial.sell_orders == ((code, 250),)
    assert below_minimum.sell_orders == ()


def test_kcb_full_liquidation_allows_sub_200_residual():
    code = "688001.SH"
    account = AccountState(
        cash=0.0,
        positions={code: 150},
        sellable_positions={code: 150},
        last_prices={code: 10.0},
        nav=1_500.0,
        peak_nav=1_500.0,
    )

    plan = DayPlanner().plan(
        _market(codes=(code,)),
        account,
        _config(target_exposure=0.0),
    )

    assert plan.sell_orders == ((code, -1),)


def test_kcb_replace_only_partial_sell_keeps_one_share_increment():
    held = "688001.SH"
    replacement = "600000.SH"
    account = AccountState(
        cash=0.0,
        positions={held: 1000},
        sellable_positions={held: 333},
        last_prices={held: 10.0},
        nav=10_000.0,
        peak_nav=10_000.0,
    )
    ranks = {
        name: np.array([0.1, 1.0]) for name in CORE_FACTOR_NAMES
    }

    plan = DayPlanner().plan(
        _market(codes=(held, replacement), ranks=ranks),
        account,
        _config(
            target_exposure=1.0,
            mode=RebalanceMode.REPLACE_ONLY,
        ),
    )

    assert plan.sell_orders == ((held, 333),)


def test_replace_only_exposure_counts_suspended_position_fallback_value():
    market = _market(opens=np.array([10.0, 10.0, 10.0, np.nan]))
    account = AccountState(
        cash=5_000.0,
        positions={"600003.SH": 900},
        sellable_positions={"600003.SH": 900},
        average_costs={"600003.SH": 10.0},
        last_prices={"600003.SH": 10.0},
        nav=14_000.0,
        peak_nav=14_000.0,
    )

    plan = DayPlanner().plan(
        market,
        account,
        _config(
            target_exposure=0.5,
            mode=RebalanceMode.REPLACE_ONLY,
        ),
    )

    assert plan.sell_orders == ()
    assert plan.buy_orders == {}
    assert plan.diagnostics["valuation_fallbacks"] == {
        "600003.SH": "account.last_prices"
    }
    assert plan.diagnostics["skip_reasons"]["_target_exposure"] == (
        "replace_only_does_not_trim_retained_positions"
    )


def test_universe_cache_and_minimal_diagnostics_do_not_change_plan():
    market = _market()
    account = _cash_account()
    config = _config(buy_n=2, sell_m=3)
    full_planner = DayPlanner(diagnostics="full")

    first = full_planner.plan(market, account, config)
    second = full_planner.plan(market, account, config)
    minimal = DayPlanner(diagnostics="minimal").plan(market, account, config)

    assert first.diagnostics["universe_cache_reused"] is False
    assert second.diagnostics["universe_cache_reused"] is True
    assert first.sell_orders == minimal.sell_orders
    assert first.buy_orders == minimal.buy_orders
    assert first.diagnostics["buy_n_stocks"] == minimal.diagnostics["buy_n_stocks"]
    assert first.diagnostics["sell_m_stocks"] == minimal.diagnostics["sell_m_stocks"]
    assert "ranked_stocks" in first.diagnostics
    assert "final_scores" in first.diagnostics
    assert "ranked_stocks" not in minimal.diagnostics
    assert "final_scores" not in minimal.diagnostics


def test_future_rows_cannot_change_the_t_plan():
    full_ranks = {
        name: np.array(
            [[1.0, 0.8, 0.4, 0.2], [0.1, 0.2, 0.8, 1.0]],
            dtype=np.float64,
        )
        for name in CORE_FACTOR_NAMES
    }
    first_market = _market(
        ranks={name: matrix[0].copy() for name, matrix in full_ranks.items()}
    )
    first = DayPlanner().plan(first_market, _cash_account(), _config())

    for matrix in full_ranks.values():
        matrix[1] = np.array([1000.0, -1000.0, 500.0, -500.0])
    second_market = _market(
        ranks={name: matrix[0].copy() for name, matrix in full_ranks.items()}
    )
    second = DayPlanner().plan(second_market, _cash_account(), _config())

    assert first.sell_orders == second.sell_orders
    assert first.buy_orders == second.buy_orders
    assert first.diagnostics["buy_n_stocks"] == second.diagnostics["buy_n_stocks"]
    with pytest.raises(TypeError, match="unexpected keyword argument 'close_prices'"):
        DayMarketData(  # type: ignore[call-arg]
            decision_date="2026-08-20",
            stock_codes=("600000.SH",),
            factor_ranks={name: np.ones(1) for name in CORE_FACTOR_NAMES},
            factor_validity={name: np.ones(1, dtype=bool) for name in CORE_FACTOR_NAMES},
            filter_masks={name: np.ones(1, dtype=bool) for name in CORE_FILTER_NAMES},
            open_prices=np.ones(1),
            preclose_prices=np.ones(1),
            issue_prices=np.ones(1),
            st_mask=np.zeros(1, dtype=bool),
            close_prices=np.ones(1),
        )


def test_fixed_day_config_matches_existing_shared_planner_on_simple_fixture():
    from core.legality import LegalityChecker
    from core.strategy import build_rebalance_day

    codes = np.array(
        ["600000.SH", "600001.SH", "600002.SH", "600003.SH"],
        dtype="U12",
    )
    dates = np.array(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ],
        dtype="datetime64[D]",
    )
    shape = (len(dates), len(codes))
    data = {
        "stock_codes": codes,
        "trade_dates": dates,
        "open": np.full(shape, 10.0),
        "close": np.full(shape, 10.0),
        "preClose": np.full(shape, 10.0),
        "st_mask": np.zeros(shape, dtype=bool),
        "issue_price": np.full(len(codes), 10.0),
    }
    last_ranks = {
        CORE_FACTOR_NAMES[0]: np.array([1.0, 0.8, 0.4, 0.2]),
        CORE_FACTOR_NAMES[1]: np.array([0.9, 0.7, 0.5, 0.3]),
        CORE_FACTOR_NAMES[2]: np.array([0.8, 0.6, 0.4, 0.2]),
        CORE_FACTOR_NAMES[3]: np.array([0.7, 0.5, 0.3, 0.1]),
    }
    all_scores = {
        name: np.vstack([np.zeros((len(dates) - 1, len(codes))), row])
        for name, row in last_ranks.items()
    }
    weights = {name: 0.25 for name in CORE_FACTOR_NAMES}
    stock_indices = {str(code): idx for idx, code in enumerate(codes)}
    checker = LegalityChecker(data, stock_indices)
    legacy = build_rebalance_day(
        data=data,
        all_scores=all_scores,
        date_idx=5,
        trade_idx=5,
        signal_date=date(2024, 1, 9),
        valid_stocks=[str(code) for code in codes],
        valid_cols=np.arange(len(codes), dtype=np.intp),
        stock_indices=stock_indices,
        weights=weights,
        buy_n=2,
        sell_m=3,
        checker=checker,
        positions={},
        sellable_volumes={},
        cash=100_000.0,
        position_multiplier=0.8,
        rebalance=True,
        is_rebalance_day=True,
        rebalance_band_pct=0.0,
    )
    market = _market(
        codes=tuple(str(code) for code in codes),
        ranks=last_ranks,
        decision_date="2024-01-09",
    )
    current = DayPlanner().plan(
        market,
        _cash_account(),
        _config(
            factor_weights=weights,
            target_exposure=0.8,
            buy_n=2,
            sell_m=3,
        ),
    )

    assert current.diagnostics["buy_n_stocks"] == tuple(legacy.buy_n_stocks)
    assert current.diagnostics["sell_m_stocks"] == tuple(legacy.sell_m_stocks)
    assert current.sell_orders == tuple(legacy.sell_orders)
    assert current.buy_orders == legacy.buy_orders
    assert current.diagnostics["base_target"] == pytest.approx(legacy.base_target)

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from env.action_schema import CORE_FACTOR_NAMES, CORE_FILTER_NAMES
from env.contracts import AccountState, DayConfig
from env.fees import DEFAULT_FEE_SCHEDULE
from env.planner import DayMarketData, DayPlanner
from env.simulator import DaySimulator


def test_candidate_snapshot_reuses_validated_market_and_freezes_new_mask():
    market = _market()
    mask = np.asarray([True, False, True, False])
    updated = market.with_candidate_mask(mask)
    expected = replace(market, candidate_mask=mask)
    account = AccountState(cash=1_000_000.0, nav=1_000_000.0)
    assert DayPlanner().plan(updated, account, _config()) == DayPlanner().plan(
        expected, account, _config()
    )
    assert updated.stock_codes is market.stock_codes
    assert updated.factor_ranks is market.factor_ranks
    assert updated.open_prices is market.open_prices
    assert updated.candidate_mask.dtype == np.bool_
    mask[:] = False
    np.testing.assert_array_equal(updated.candidate_mask, [True, False, True, False])
    np.testing.assert_array_equal(market.candidate_mask, np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="read-only"):
        updated.candidate_mask[0] = False
    with pytest.raises(ValueError, match="candidate_mask must have shape"):
        market.with_candidate_mask(np.ones(3, dtype=bool))


def test_cached_freeze_ratios_follow_changed_stock_universe():
    planner = DayPlanner()
    account = AccountState(cash=1_000_000.0, nav=1_000_000.0)
    for codes in (("600000.SH", "300001.SZ"), ("688001.SH", "430001.BJ")):
        market = _market(codes=codes)
        plan = planner.plan(market, account, _config(buy_n=2, turnover_rate=1.0))
        assert plan.diagnostics["limit_prices"] == {
            code: 10.0 * (1.3 if code.endswith(".BJ") else 1.2 if code.startswith(("30", "68")) else 1.1)
            for code in codes
        }


def _config(
    *,
    primary_factor: str = CORE_FACTOR_NAMES[0],
    factor_weights: dict[str, float] | None = None,
    filter_flags: dict[str, bool] | None = None,
    buy_n: int = 1,
    turnover_rate: float = 1.0,
    limit_up_protection: bool = False,
    band: float = 0.0,
    single_buy_pct: float | None = None,
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
        buy_n=buy_n,
        turnover_rate=turnover_rate,
        limit_up_protection=limit_up_protection,
        rebalance_band_pct=band,
        single_buy_pct=(1.0 / buy_n if single_buy_pct is None else single_buy_pct),
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
    delisted: np.ndarray | None = None,
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
        delisted_mask=(
            np.zeros(size, dtype=bool) if delisted is None else delisted
        ),
        listing_age=(
            np.full(size, 10, dtype=np.int32)
            if listing_age is None
            else listing_age
        ),
    )


def _cash_account(cash: float = 100_000.0) -> AccountState:
    return AccountState(cash=cash, nav=cash, peak_nav=cash)


def _assert_full_investment_proof(plan, reason: str) -> None:
    diagnostics = plan.diagnostics
    assert diagnostics["full_investment_contract_satisfied"] is True
    assert diagnostics["residual_cash_reason"] == reason
    assert diagnostics["planned_post_order_cash"] >= 0.0

    if reason == "below_next_legal_frozen_lot_cost":
        next_cost = diagnostics["cheapest_next_legal_buy_cost"]
        assert next_cost is not None
        assert diagnostics["planned_post_order_cash"] < next_cost
    else:
        assert diagnostics["cheapest_next_legal_buy_cost"] is None


def test_single_buy_pct_decouples_per_stock_target_from_buy_n():
    planner = DayPlanner()
    market = _market()
    account = _cash_account()

    narrow = planner.plan(
        market,
        account,
        _config(buy_n=2, turnover_rate=1.0, single_buy_pct=0.60),
    )
    broad = planner.plan(
        market,
        account,
        _config(buy_n=4, turnover_rate=1.0, single_buy_pct=0.60),
    )

    assert narrow.diagnostics["base_target"] == pytest.approx(
        broad.diagnostics["base_target"]
    )


def test_dynamic_factor_weights_change_ranking_without_recomputing_factors():
    ranks = {
        **{name: np.full(4, 0.1) for name in CORE_FACTOR_NAMES},
        CORE_FACTOR_NAMES[0]: np.array([1.0, 0.7, 0.4, 0.1]),
        CORE_FACTOR_NAMES[1]: np.array([0.1, 0.4, 0.7, 1.0]),
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


def test_soft_filters_cannot_become_a_hidden_cash_switch():
    market = _market(
        filters={name: np.zeros(4, dtype=bool) for name in CORE_FILTER_NAMES}
    )
    config = _config(
        buy_n=2,
        turnover_rate=1.0,
        filter_flags={name: True for name in CORE_FILTER_NAMES},
    )

    plan = DayPlanner().plan(market, _cash_account(), config)

    assert len(plan.diagnostics["buy_n_stocks"]) == 2
    assert plan.diagnostics["filter_backfill_buy_count"] == 2
    assert plan.buy_orders
    assert plan.diagnostics["full_investment_contract_satisfied"] is True


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


def test_full_rank_top_is_independent_of_buy_legality():
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
        _config(buy_n=1, turnover_rate=1.0),
    )

    assert plan.diagnostics["buy_n_stocks"] == ("600001.SH",)
    assert plan.diagnostics["full_rank_top_stocks"] == ("600000.SH",)


@pytest.mark.parametrize("reverse_insertion", (False, True))
def test_retained_order_follows_full_rank_then_unknown_codes(reverse_insertion):
    market=_market(ranks={name:np.array([0.2,0.8,0.8,0.1]) for name in CORE_FACTOR_NAMES})
    names=["outside-z","600002.SH","600000.SH","outside-a","600001.SH"]
    if reverse_insertion:
        names.reverse()
    account=AccountState(cash=100_000.,positions={code:100 for code in names},
        sellable_positions={code:100 for code in names},last_prices={code:10. for code in names},
        nav=105_000.,peak_nav=105_000.)
    plan=DayPlanner().plan(market,account,_config(buy_n=5,turnover_rate=0.))
    assert plan.diagnostics["retained_stocks"]==("600001.SH","600002.SH","600000.SH","outside-a","outside-z")
    assert plan.diagnostics["examined_worst_holdings"]==()
    assert plan.diagnostics["sell_legality_rejections"]=={
        "outside-a":"outside_market_universe","outside-z":"outside_market_universe"}


def test_limit_down_blocks_sell_and_limit_up_sell_protection_is_dynamic():
    codes = ("600003.SH", "600000.SH", "600001.SH", "600002.SH")
    market = _market(
        codes=codes,
        opens=np.array([10.0, 9.0, 11.0, 10.0]),
        ranks={name: np.array([1.0, 0.8, 0.1, 0.5]) for name in CORE_FACTOR_NAMES},
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
        _config(limit_up_protection=False),
    )
    protected = DayPlanner().plan(
        market,
        account,
        _config(limit_up_protection=True),
    )

    assert unprotected.sell_orders == (("600001.SH", 100),)
    assert unprotected.diagnostics["sell_legality_rejections"] == {
        "600000.SH": "limit_down"
    }
    assert protected.sell_orders == ()
    assert protected.diagnostics["sell_legality_rejections"] == {
        "600000.SH": "limit_down",
        "600001.SH": "limit_up_protected",
    }


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
        _config(),
    )

    assert plan.sell_orders == ()


def test_planner_replaces_exits_and_reinvests_cash():
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

    plan = DayPlanner().plan(market, account, _config())

    assert plan.sell_orders == (("600003.SH", 1000),)
    assert set(plan.buy_orders) == {"600000.SH"}
    assert not hasattr(plan.day_config, "target_exposure")
    assert not hasattr(plan.day_config, "rebalance_now")
    assert not hasattr(plan.day_config, "rebalance_mode")


def test_full_investment_cash_sweep_takes_precedence_over_rebalance_band():
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
        _config(band=0.0),
    )
    wide_band = DayPlanner().plan(
        market,
        account,
        _config(band=0.10),
    )

    assert no_band.diagnostics["base_target"] == pytest.approx(11_000.0)
    assert no_band.buy_orders == {"600000.SH": 100}
    assert wide_band.buy_orders == no_band.buy_orders


def test_full_investment_cash_below_one_frozen_lot_has_auditable_residual():
    market = _market(codes=("600000.SH",))

    plan = DayPlanner().plan(market, _cash_account(1_000.0), _config())

    assert plan.buy_orders == {}
    assert plan.diagnostics["planned_post_order_cash"] == pytest.approx(1_000.0)
    assert plan.diagnostics["cheapest_next_legal_buy_cost"] == pytest.approx(
        DEFAULT_FEE_SCHEDULE.buy_total_cost(100 * 11.0)
    )
    _assert_full_investment_proof(
        plan,
        "below_next_legal_frozen_lot_cost",
    )


def test_full_investment_limit_up_rank_one_falls_back_to_next_legal_stock():
    codes = ("600000.SH", "600001.SH")
    ranks = {
        name: np.array([1.0, 0.5], dtype=np.float64)
        for name in CORE_FACTOR_NAMES
    }
    market = _market(
        codes=codes,
        ranks=ranks,
        opens=np.array([11.0, 10.0]),
    )

    plan = DayPlanner(diagnostics="full").plan(
        market,
        _cash_account(2_000.0),
        _config(),
    )

    assert plan.diagnostics["buy_legality_rejections"] == {
        "600000.SH": "limit_up"
    }
    assert plan.diagnostics["buy_n_stocks"] == ("600001.SH",)
    assert plan.buy_orders == {"600001.SH": 100}
    _assert_full_investment_proof(
        plan,
        "below_next_legal_frozen_lot_cost",
    )


def test_full_investment_all_targets_illegal_explains_untouched_cash():
    codes = ("600000.SH", "600001.SH")
    ranks = {
        name: np.array([1.0, 0.5], dtype=np.float64)
        for name in CORE_FACTOR_NAMES
    }
    market = _market(
        codes=codes,
        ranks=ranks,
        opens=np.array([11.0, np.nan]),
    )

    plan = DayPlanner(diagnostics="full").plan(
        market,
        _cash_account(2_000.0),
        _config(),
    )

    assert plan.buy_orders == {}
    assert plan.diagnostics["buy_n_stocks"] == ()
    assert plan.diagnostics["planned_post_order_cash"] == pytest.approx(2_000.0)
    assert plan.diagnostics["buy_legality_rejections"] == {
        "600000.SH": "limit_up"
    }
    assert plan.diagnostics["market_rejections"] == {
        "600001.SH": "suspended_or_missing_open"
    }
    _assert_full_investment_proof(plan, "no_legal_buy_target")


@pytest.mark.parametrize(
    (
        "locked_open",
        "expected_sell_reason",
        "expected_equity",
        "expected_residual_reason",
    ),
    [
        (9.0, "limit_down", 900.0, "concentration_or_lot_capacity_exhausted"),
        (
            np.nan,
            "suspended_or_missing_open",
            1_000.0,
            "no_legal_buy_target",
        ),
    ],
)
def test_full_investment_treats_limit_down_and_suspended_holdings_as_locked(
    locked_open,
    expected_sell_reason,
    expected_equity,
    expected_residual_reason,
):
    held = "600003.SH"
    opens = np.array([10.0, 10.0, 10.0, locked_open])
    account = AccountState(
        cash=0.0,
        positions={held: 100},
        sellable_positions={held: 100},
        average_costs={held: 10.0},
        last_prices={held: 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )

    plan = DayPlanner().plan(_market(opens=opens), account, _config())

    assert plan.sell_orders == ()
    assert plan.buy_orders == {}
    assert plan.diagnostics["sell_legality_rejections"] == {
        held: expected_sell_reason
    }
    assert plan.diagnostics["total_equity"] == pytest.approx(expected_equity)
    _assert_full_investment_proof(plan, expected_residual_reason)


def test_full_investment_reports_concentration_capacity_exhaustion():
    market = _market(codes=("600000.SH",))

    plan = DayPlanner().plan(
        market,
        _cash_account(100_000.0),
        _config(buy_n=2, turnover_rate=1.0, single_buy_pct=0.50),
    )

    assert plan.buy_orders == {"600000.SH": 5_000}
    assert plan.diagnostics["planned_buy_notional"] == pytest.approx(50_000.0)
    assert plan.diagnostics["planned_post_order_cash"] > 49_000.0
    _assert_full_investment_proof(
        plan,
        "concentration_or_lot_capacity_exhausted",
    )


def test_full_investment_fee_residual_matches_the_executed_account():
    market = _market(codes=("600000.SH",))
    account = _cash_account(2_000.0)
    plan = DayPlanner().plan(market, account, _config())
    expected_cash = 2_000.0 - DEFAULT_FEE_SCHEDULE.buy_total_cost(1_000.0)

    assert plan.buy_orders == {"600000.SH": 100}
    assert plan.diagnostics["planned_post_order_cash"] == pytest.approx(
        expected_cash
    )
    _assert_full_investment_proof(
        plan,
        "below_next_legal_frozen_lot_cost",
    )

    result = DaySimulator().step(
        account,
        plan,
        {"600000.SH": 10.0},
        {"600000.SH": 10.0},
        close_prices={"600000.SH": 10.0},
        next_preclose_prices={"600000.SH": 10.0},
    )
    assert result.account_state.cash == pytest.approx(expected_cash)
    assert result.account_state.cash == pytest.approx(
        plan.diagnostics["planned_post_order_cash"]
    )


def test_kcb_buy_uses_200_minimum_then_one_share_increments():
    market = _market(codes=("688001.SH",))
    planner = DayPlanner()

    below_minimum = planner.plan(
        market,
        _cash_account(10_000.0),
        _config(buy_n=10, turnover_rate=1.0, single_buy_pct=0.10),
    )
    one_share_increment = planner.plan(
        market,
        _cash_account(10_000.0),
        _config(buy_n=3, turnover_rate=1.0, single_buy_pct=0.3996),
    )

    assert below_minimum.buy_orders == {}
    assert below_minimum.diagnostics["skip_reasons"]["688001.SH"] == (
        "below_exchange_minimum_or_concentration_cap"
    )
    assert one_share_increment.buy_orders == {"688001.SH": 399}

    executed = DaySimulator().step(
        _cash_account(10_000.0),
        one_share_increment,
        {"688001.SH": 10.0},
        {"688001.SH": 10.0},
        close_prices={"688001.SH": 10.0},
        next_preclose_prices={"688001.SH": 10.0},
    )
    assert executed.fills[0].quantity == 399
    assert executed.account_state.positions == {"688001.SH": 399}


def test_retained_overweight_shares_trim_to_target_subject_to_minimum():
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
        _config(buy_n=2, turnover_rate=1.0, single_buy_pct=0.75),
    )
    below_minimum = planner.plan(
        market,
        account,
        _config(buy_n=2, turnover_rate=1.0, single_buy_pct=0.99),
    )

    assert legal_partial.sell_orders == ((code, 250),)
    assert below_minimum.sell_orders == ()


def test_kcb_full_liquidation_allows_sub_200_residual():
    code = "688001.SH"
    replacement = "600000.SH"
    account = AccountState(
        cash=0.0,
        positions={code: 150},
        sellable_positions={code: 150},
        last_prices={code: 10.0},
        nav=1_500.0,
        peak_nav=1_500.0,
    )

    ranks = {name: np.array([0.1, 1.0]) for name in CORE_FACTOR_NAMES}
    plan = DayPlanner().plan(
        _market(codes=(code, replacement), ranks=ranks), account, _config()
    )

    assert plan.sell_orders == ((code, 150),)


def test_partial_sellable_holding_does_not_free_a_replacement_slot():
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
        _config(),
    )

    assert plan.sell_orders == ()
    assert plan.buy_orders == {}
    assert plan.diagnostics["retained_stocks"] == (held,)


def test_full_investment_counts_suspended_position_fallback_value():
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
        _config(),
    )

    assert plan.sell_orders == ()
    assert plan.buy_orders == {}  # The suspended position already occupies the only slot.
    assert plan.diagnostics["total_equity"] == 14_000.0
    assert plan.diagnostics["valuation_fallbacks"] == {
        "600003.SH": "account.last_prices"
    }


def test_equalize_empty_position_target_union_keeps_cash():
    market = _market(opens=np.full(4, np.nan))
    plan = DayPlanner().plan(market, _cash_account(12345.0), _config(buy_n=2))
    assert plan.sell_orders == ()
    assert dict(plan.buy_orders) == {}
    assert plan.diagnostics["skip_reasons"] == {}
    assert plan.diagnostics["planned_post_order_cash"] == 12345.0
    _assert_full_investment_proof(plan, "no_legal_buy_target")


@pytest.mark.parametrize("code", ["600000.SH", "688001.SH", "430001.BJ"])
def test_equalize_full_liquidation_preserves_integer_above_float64_precision(code):
    quantity = 2**53 + 201
    sells, buys, skipped, cash, next_cost = DayPlanner()._equalize_orders(
        market=_market(codes=(code,)), account_cash=0.0,
        positions={code: quantity}, sellable={code: quantity},
        position_values={code: quantity * 10.0}, prices={code: 10.0},
        limit_prices={}, buy_targets=(), target_codes=(), keep_codes=(),
        sellable_ok={code}, base_target=0.0, band=0.0,
    )
    assert sells == [(code, quantity)]
    assert type(sells[0][1]) is int
    assert buys == skipped == {}
    assert cash > 0.0
    assert next_cost is None


def test_equalize_buy_execution_order_follows_rank_across_exchange_lot_rules():
    codes = ("600000.SH", "688001.SH", "430001.BJ")
    ranks = {name: np.array([0.1, 0.5, 1.0]) for name in CORE_FACTOR_NAMES}
    plan = DayPlanner().plan(
        _market(codes=codes, ranks=ranks), _cash_account(), _config(buy_n=3),
    )
    assert tuple(plan.buy_orders) == ("430001.BJ", "688001.SH", "600000.SH")
    assert all(type(amount) is int and amount > 0 for amount in plan.buy_orders.values())
    assert plan.buy_orders["600000.SH"] % 100 == 0
    assert plan.buy_orders["688001.SH"] >= 200
    assert plan.buy_orders["430001.BJ"] >= 100


@pytest.mark.parametrize("turnover_rate", (0.0, 0.17, 1.0))
@pytest.mark.parametrize("zero_weights", (False, True))
def test_universe_cache_and_minimal_diagnostics_do_not_change_plan(turnover_rate, zero_weights):
    market = _market().with_candidate_mask(np.asarray([True, False, True, True]))
    account = AccountState(
        cash=90_000.0, nav=100_000.0, peak_nav=100_000.0,
        positions={"600003.SH": 1_000}, sellable_positions={"600003.SH": 1_000},
        last_prices={"600003.SH": 10.0},
    )
    config = _config(buy_n=2, turnover_rate=turnover_rate,
                     factor_weights=dict.fromkeys(CORE_FACTOR_NAMES, 0.0) if zero_weights else None)
    full_planner = DayPlanner(diagnostics="full")

    first = full_planner.plan(market, account, config)
    second = full_planner.plan(market, account, config)
    minimal = DayPlanner(diagnostics="minimal").plan(market, account, config)

    assert first.diagnostics["universe_cache_reused"] is False
    assert second.diagnostics["universe_cache_reused"] is True
    assert first.sell_orders == minimal.sell_orders
    assert first.buy_orders == minimal.buy_orders
    assert first.diagnostics["buy_n_stocks"] == minimal.diagnostics["buy_n_stocks"]
    assert first.diagnostics["retained_stocks"] == minimal.diagnostics["retained_stocks"]
    assert "ranked_stocks" in first.diagnostics
    assert "final_scores" in first.diagnostics
    assert "ranked_stocks" not in minimal.diagnostics
    assert "final_scores" not in minimal.diagnostics
    assert all(first.diagnostics[name] == value for name, value in minimal.diagnostics.items())
    scores = first.diagnostics["final_scores"]
    assert scores[1] == -np.inf
    np.testing.assert_array_equal(scores[[0, 2, 3]], market.factor_scores(config)[[0, 2, 3]])
    scores[:] = 123.0
    assert not np.any(second.diagnostics["final_scores"] == 123.0)
    assert not np.any(market.factor_scores(config) == 123.0)


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
            delisted_mask=np.zeros(1, dtype=bool),
            listing_age=np.full(1, 10, dtype=np.int32),
            close_prices=np.ones(1),
        )


@pytest.mark.parametrize("cash", [0.0, 2000.0])
def test_retained_drift_rebalances_to_fund_underweight_holdings(cash):
    account = AccountState(
        cash=cash,
        positions={"600000.SH": 1600, "600001.SH": 400},
        sellable_positions={"600000.SH": 1600, "600001.SH": 400},
        last_prices={"600000.SH": 10.0, "600001.SH": 10.0},
        nav=20000.0 + cash, peak_nav=20000.0 + cash,
    )
    plan = DayPlanner().plan(_market(), account, _config(buy_n=2, turnover_rate=0.0))
    assert dict(plan.sell_orders)["600000.SH"] > 0
    assert plan.diagnostics["full_investment_contract_satisfied"]
    assert set(plan.buy_orders) == {"600001.SH"}
    assert plan.diagnostics["planned_post_order_cash"] >= 0

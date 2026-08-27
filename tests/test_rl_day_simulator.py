from __future__ import annotations

import hashlib
import inspect
import json
import math

import pytest

from env.contracts import AccountState, OrderPlan
from env.simulator import (
    ACCOUNTING_MODE,
    ACCOUNTING_SCHEMA_HASH,
    ACCOUNTING_SCHEMA_VERSION,
    DaySimulator,
    FeeSchedule,
    accounting_schema_manifest,
)


def _plan(
    *,
    sells: tuple[tuple[str, int], ...] = (),
    buys: dict[str, int] | None = None,
    decision_date: str = "2026-08-20",
) -> OrderPlan:
    return OrderPlan(
        decision_date=decision_date,
        sell_orders=sells,
        buy_orders=buys or {},
    )


def test_default_buy_fees_are_deducted_once_and_reward_is_next_open_log_return():
    result = DaySimulator().step(
        AccountState(cash=2_000.0, nav=2_000.0, peak_nav=2_000.0),
        _plan(buys={"600000.SH": 100}),
        {"600000.SH": 10.0},
        {"600000.SH": 10.0},
        close_prices={"600000.SH": 10.0},
        next_preclose_prices={"600000.SH": 10.0},
    )

    expected_fee = 0.1 + 1_000.0 * 0.00002 + 1_000.0 * 0.001
    expected_nav = 2_000.0 - expected_fee
    assert result.fills[0].fee == pytest.approx(expected_fee)
    assert result.account_state.cash == pytest.approx(1_000.0 - expected_fee)
    assert result.account_state.positions == {"600000.SH": 100}
    assert result.account_state.nav == pytest.approx(expected_nav)
    assert result.portfolio_return == pytest.approx(expected_nav / 2_000.0 - 1.0)
    assert result.reward == pytest.approx(math.log(expected_nav / 2_000.0))
    assert result.diagnostics["total_fees"] == pytest.approx(expected_fee)
    assert result.account_state.average_costs["600000.SH"] == pytest.approx(
        (1_000.0 + expected_fee) / 100
    )


def test_default_sell_fees_include_stamp_tax_and_full_sell_allows_odd_lot():
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 150},
        sellable_positions={"600000.SH": 150},
        average_costs={"600000.SH": 8.0},
        last_prices={"600000.SH": 10.0},
        nav=1_500.0,
        peak_nav=1_500.0,
    )
    result = DaySimulator().step(
        account,
        _plan(sells=(("600000.SH", -1),)),
        {"600000.SH": 10.0},
        {},
    )

    expected_fee = (
        max(1_500.0 * 0.0000854, 0.1)
        + 1_500.0 * 0.0005
        + 1_500.0 * 0.00002
        + 1_500.0 * 0.001
    )
    assert result.fills[0].quantity == 150
    assert result.fills[0].fee == pytest.approx(expected_fee)
    assert result.account_state.positions == {}
    assert result.account_state.cash == pytest.approx(1_500.0 - expected_fee)
    assert result.account_state.nav == pytest.approx(1_500.0 - expected_fee)


def test_sells_execute_before_buys_and_fund_them():
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 100},
        sellable_positions={"600000.SH": 100},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )
    result = DaySimulator().step(
        account,
        _plan(
            sells=(("600000.SH", -1),),
            buys={"600001.SH": 100},
        ),
        {"600000.SH": 10.0, "600001.SH": 9.0},
        {"600001.SH": 9.0},
        close_prices={"600001.SH": 9.0},
        next_preclose_prices={"600001.SH": 9.0},
    )

    assert [(fill.side, fill.code) for fill in result.fills] == [
        ("sell", "600000.SH"),
        ("buy", "600001.SH"),
    ]
    assert result.account_state.positions == {"600001.SH": 100}
    assert result.diagnostics["fill_sequence"] == (
        ("sell", "600000.SH"),
        ("buy", "600001.SH"),
    )


def test_empty_sellable_mapping_does_not_fall_back_to_full_position():
    account = AccountState(
        cash=1.0,
        positions={"600000.SH": 100},
        sellable_positions={},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=1_001.0,
        peak_nav=1_001.0,
    )
    result = DaySimulator().step(
        account,
        _plan(sells=(("600000.SH", -1),)),
        {"600000.SH": 10.0},
        {"600000.SH": 10.0},
        close_prices={"600000.SH": 10.0},
        next_preclose_prices={"600000.SH": 10.0},
    )

    assert result.fills == ()
    assert result.account_state.positions == {"600000.SH": 100}
    assert result.diagnostics["skipped_orders"] == (
        {"code": "600000.SH", "side": "sell", "reason": "not_sellable"},
    )


def test_partial_orders_are_floored_to_100_share_lots():
    account = AccountState(
        cash=2_000.0,
        positions={"600000.SH": 250},
        sellable_positions={"600000.SH": 250},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=4_500.0,
        peak_nav=4_500.0,
    )
    result = DaySimulator().step(
        account,
        _plan(sells=(("600000.SH", 150),), buys={"600001.SH": 150}),
        {"600000.SH": 10.0, "600001.SH": 10.0},
        {"600000.SH": 10.0, "600001.SH": 10.0},
        close_prices={"600000.SH": 10.0, "600001.SH": 10.0},
        next_preclose_prices={"600000.SH": 10.0, "600001.SH": 10.0},
    )

    assert [(fill.side, fill.quantity) for fill in result.fills] == [
        ("sell", 100),
        ("buy", 100),
    ]
    assert result.account_state.positions == {
        "600000.SH": 150,
        "600001.SH": 100,
    }


def test_kcb_direct_buy_enforces_200_minimum_and_preserves_one_share_step():
    simulator = DaySimulator()
    account = AccountState(cash=10_000.0, nav=10_000.0, peak_nav=10_000.0)
    prices = {"688001.SH": 10.0}

    rejected = simulator.step(
        account,
        _plan(buys={"688001.SH": 199}),
        prices,
        prices,
        close_prices=prices,
        next_preclose_prices=prices,
    )
    accepted = simulator.step(
        account,
        _plan(buys={"688001.SH": 333}),
        prices,
        prices,
        close_prices=prices,
        next_preclose_prices=prices,
    )

    assert rejected.fills == ()
    assert rejected.diagnostics["skipped_orders"] == (
        {
            "code": "688001.SH",
            "side": "buy",
            "reason": "below_exchange_minimum",
        },
    )
    assert accepted.fills[0].quantity == 333
    assert accepted.account_state.positions == {"688001.SH": 333}


def test_kcb_direct_partial_sell_enforces_200_minimum_and_one_share_step():
    code = "688001.SH"
    account = AccountState(
        cash=0.0,
        positions={code: 500},
        sellable_positions={code: 500},
        average_costs={code: 10.0},
        last_prices={code: 10.0},
        nav=5_000.0,
        peak_nav=5_000.0,
    )
    prices = {code: 10.0}
    simulator = DaySimulator()

    rejected = simulator.step(
        account,
        _plan(sells=((code, 199),)),
        prices,
        prices,
        close_prices=prices,
        next_preclose_prices=prices,
    )
    accepted = simulator.step(
        account,
        _plan(sells=((code, 333),)),
        prices,
        prices,
        close_prices=prices,
        next_preclose_prices=prices,
    )

    assert rejected.fills == ()
    assert accepted.fills[0].quantity == 333
    assert accepted.account_state.positions == {code: 167}


def test_missing_opens_use_explained_mark_fallbacks_but_do_not_fill_orders():
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 100},
        sellable_positions={"600000.SH": 100},
        average_costs={"600000.SH": 8.0},
        last_prices={"600000.SH": 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )
    result = DaySimulator().step(
        account,
        _plan(sells=(("600000.SH", -1),)),
        {"600000.SH": float("nan")},
        {"600000.SH": float("nan")},
        close_prices={"600000.SH": 10.5},
        next_preclose_prices={"600000.SH": 10.5},
    )

    assert result.fills == ()
    assert result.account_state.positions == {"600000.SH": 100}
    assert result.account_state.last_prices == {"600000.SH": 10.5}
    assert result.account_state.nav == pytest.approx(1_050.0)
    assert result.reward == pytest.approx(math.log(1.05))
    assert result.diagnostics["current_mark_fallbacks"]["600000.SH"] == (
        "open[T]_missing; used_account.last_prices"
    )
    assert result.diagnostics["settlement_fallbacks"]["600000.SH"] == (
        "open[T+1]_missing_or_suspended; used_preClose[T+1]"
    )
    assert result.diagnostics["skipped_orders"] == (
        {"code": "600000.SH", "side": "sell", "reason": "missing_open[T]"},
    )


def test_reward_ends_at_next_open_not_current_close():
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 100},
        sellable_positions={"600000.SH": 100},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )
    result = DaySimulator().step(
        account,
        _plan(),
        {"600000.SH": 10.0},
        {"600000.SH": 11.0},
        close_prices={"600000.SH": 12.0},
        next_preclose_prices={"600000.SH": 12.0},
    )

    assert result.account_state.nav == pytest.approx(1_100.0)
    assert result.portfolio_return == pytest.approx(0.10)
    assert result.reward == pytest.approx(math.log(1.10))
    assert result.diagnostics["reward_interval"] == (
        "pretrade_open[T]_to_pretrade_open[T+1]"
    )


def test_accounting_schema_manifest_is_stable_and_bundle_bindable():
    manifest = accounting_schema_manifest()
    hash_payload = dict(manifest)
    declared_hash = hash_payload.pop("schema_hash")
    actual_hash = hashlib.sha256(
        json.dumps(
            hash_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert manifest["schema_version"] == ACCOUNTING_SCHEMA_VERSION
    assert manifest["mode"] == ACCOUNTING_MODE == "total_return_reinvested"
    assert manifest["broker_exact"] is False
    assert declared_hash == actual_hash == ACCOUNTING_SCHEMA_HASH
    # Freeze the exact accounting identity used by policy bundles. Any change
    # here must be an intentional accounting-schema migration.
    assert ACCOUNTING_SCHEMA_HASH == (
        "5d59f4a6970f25682e7d9c3e016ae64131cb9682f8f5f52138b31948270e32ae"
    )
    assert len(ACCOUNTING_SCHEMA_HASH) == 64
    assert manifest["settlement_only_price_inputs"] == [
        "close[T]",
        "preClose[T+1]",
        "open[T+1]",
    ]
    manifest["mode"] = "mutated-by-caller"
    assert accounting_schema_manifest()["mode"] == ACCOUNTING_MODE


def test_corporate_action_factor_one_keeps_integer_quantity_and_cash_unchanged():
    account = AccountState(
        cash=23.0,
        positions={"600000.SH": 137},
        sellable_positions={"600000.SH": 137},
        average_costs={"600000.SH": 8.0},
        last_prices={"600000.SH": 10.0},
        nav=1_393.0,
        peak_nav=1_393.0,
    )
    result = DaySimulator().step(
        account,
        _plan(),
        {"600000.SH": 10.0},
        {"600000.SH": 11.0},
        close_prices={"600000.SH": 10.0},
        next_preclose_prices={"600000.SH": 10.0},
    )

    assert result.account_state.positions == {"600000.SH": 137}
    assert result.account_state.cash == 23.0
    assert result.diagnostics["corporate_action_cash_residual"] == 0.0
    assert result.diagnostics["corporate_action_adjustments"] == {}
    assert result.diagnostics["accounting_schema"] == {
        "schema_version": ACCOUNTING_SCHEMA_VERSION,
        "schema_hash": ACCOUNTING_SCHEMA_HASH,
        "mode": ACCOUNTING_MODE,
        "broker_exact": False,
    }
    assert result.diagnostics["broker_exact"] is False


def test_corporate_action_rebases_economic_shares_and_stays_consistent_next_step():
    simulator = DaySimulator()
    initial = AccountState(
        cash=0.0,
        positions={"600000.SH": 100},
        sellable_positions={"600000.SH": 100},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )
    first = simulator.step(
        initial,
        _plan(decision_date="2026-08-20"),
        {"600000.SH": 10.0},
        {"600000.SH": 5.5},
        close_prices={"600000.SH": 10.0},
        next_preclose_prices={"600000.SH": 5.0},
        next_decision_date="2026-08-21",
    )

    assert first.account_state.positions == {"600000.SH": 200}
    assert first.account_state.last_prices == {"600000.SH": 5.5}
    assert first.account_state.cash == pytest.approx(0.0)
    assert first.account_state.nav == pytest.approx(1_100.0)
    assert first.reward == pytest.approx(math.log(1.10))
    expected_economic_value = 100 * (10.0 / 5.0) * 5.5
    realized_economic_value = (
        first.account_state.positions["600000.SH"]
        * first.account_state.last_prices["600000.SH"]
        + first.diagnostics["corporate_action_cash_residual"]
    )
    assert realized_economic_value == pytest.approx(expected_economic_value)
    adjustment = first.diagnostics["corporate_action_adjustments"]["600000.SH"]
    assert adjustment["mode"] == ACCOUNTING_MODE
    assert adjustment["accounting_schema_hash"] == ACCOUNTING_SCHEMA_HASH
    assert adjustment["broker_exact"] is False
    assert adjustment["broker_quantity_exact"] is False
    assert adjustment["quantity_before"] == 100
    assert adjustment["synthetic_quantity_after_integer_rebase"] == 200
    assert first.diagnostics["accounting_model"] == ACCOUNTING_MODE
    assert first.diagnostics["accounting_schema"]["schema_hash"] == (
        ACCOUNTING_SCHEMA_HASH
    )
    assert first.diagnostics["broker_exact"] is False
    assert first.diagnostics["broker_quantity_exact"] is False

    second = simulator.step(
        first.account_state,
        _plan(decision_date="2026-08-21"),
        {"600000.SH": 5.5},
        {"600000.SH": 6.6},
        close_prices={"600000.SH": 6.0},
        next_preclose_prices={"600000.SH": 6.0},
        next_decision_date="2026-08-24",
    )

    assert second.diagnostics["balance_sheet_pretrade_nav"] == pytest.approx(
        first.account_state.nav
    )
    assert second.account_state.positions == {"600000.SH": 200}
    assert second.account_state.last_prices == {"600000.SH": 6.6}
    assert second.account_state.nav == pytest.approx(1_320.0)
    assert second.reward == pytest.approx(math.log(1.20))
    assert second.diagnostics["corporate_action_adjustments"] == {}
    assert second.diagnostics["broker_quantity_exact"] is False


def test_fractional_economic_share_rebase_books_rounding_residual_to_cash():
    account = AccountState(
        cash=10.0,
        positions={"600000.SH": 101},
        sellable_positions={"600000.SH": 101},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=1_020.0,
        peak_nav=1_020.0,
    )
    result = DaySimulator().step(
        account,
        _plan(),
        {"600000.SH": 10.0},
        {"600000.SH": 11.0},
        close_prices={"600000.SH": 15.0},
        next_preclose_prices={"600000.SH": 10.0},
    )

    # Exact economic quantity is 151.5, rounded to 152. The half-share
    # over-allocation is offset by -5.5 cash so NAV stays exact.
    assert result.account_state.positions == {"600000.SH": 152}
    assert result.account_state.cash == pytest.approx(4.5)
    assert result.account_state.nav == pytest.approx(10.0 + 151.5 * 11.0)
    assert result.diagnostics["corporate_action_cash_residual"] == pytest.approx(-5.5)
    exact_value = 101 * 1.5 * 11.0
    rounded_value_plus_residual = (
        result.account_state.positions["600000.SH"] * 11.0
        + result.diagnostics["corporate_action_cash_residual"]
    )
    assert rounded_value_plus_residual == pytest.approx(exact_value)


def test_half_share_rounding_is_ties_to_even_as_schema_declares():
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 100},
        sellable_positions={"600000.SH": 100},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )
    result = DaySimulator().step(
        account,
        _plan(),
        {"600000.SH": 10.0},
        {"600000.SH": 11.0},
        close_prices={"600000.SH": 15.05},
        next_preclose_prices={"600000.SH": 10.0},
    )

    # 100 * 1.505 = 150.5; ties-to-even rounds down to 150, then the
    # unrepresented half share is preserved as +5.5 cash.
    assert result.account_state.positions == {"600000.SH": 150}
    assert result.account_state.cash == pytest.approx(5.5)
    assert result.account_state.nav == pytest.approx(150.5 * 11.0)
    assert result.diagnostics["corporate_action_cash_residual"] == pytest.approx(5.5)


def test_upward_rounding_falls_back_to_floor_when_cash_cannot_fund_residual():
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 101},
        sellable_positions={"600000.SH": 101},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=1_010.0,
        peak_nav=1_010.0,
    )
    simulator = DaySimulator()
    result = simulator.step(
        account,
        _plan(decision_date="2026-08-20"),
        {"600000.SH": 10.0},
        {"600000.SH": 11.0},
        close_prices={"600000.SH": 15.0},
        next_preclose_prices={"600000.SH": 10.0},
        next_decision_date="2026-08-21",
    )

    # 151.5 normally rounds upward to 152, requiring a -5.5 cash residual.
    # With no cash buffer, the synthetic account uses 151 shares and carries
    # +5.5 cash, preserving economic value without producing invalid cash.
    assert result.account_state.positions == {"600000.SH": 151}
    assert result.account_state.cash == pytest.approx(5.5)
    assert result.account_state.nav == pytest.approx(151.5 * 11.0)
    adjustment = result.diagnostics["corporate_action_adjustments"]["600000.SH"]
    assert adjustment["nearest_ties_to_even_quantity"] == 152
    assert adjustment["cash_safe_floor_applied"] is True
    assert adjustment["rounding_cash_residual"] == pytest.approx(5.5)

    # The resulting account is directly usable by the next transition; in
    # particular it cannot trip the Observation layer's non-negative-cash
    # invariant.
    following = simulator.step(
        result.account_state,
        _plan(decision_date="2026-08-21"),
        {"600000.SH": 11.0},
        {"600000.SH": 12.0},
        close_prices={"600000.SH": 11.0},
        next_preclose_prices={"600000.SH": 11.0},
    )
    assert following.diagnostics["balance_sheet_pretrade_nav"] == pytest.approx(
        result.account_state.nav
    )
    assert following.account_state.cash == pytest.approx(5.5)


def test_company_action_inputs_are_keyword_only_settlement_fields():
    signature = inspect.signature(DaySimulator.step)
    assert signature.parameters["close_prices"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        signature.parameters["next_preclose_prices"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    account = AccountState(cash=2_000.0, nav=2_000.0, peak_nav=2_000.0)
    order = _plan(buys={"600000.SH": 100})
    ordinary = DaySimulator().step(
        account,
        order,
        {"600000.SH": 10.0},
        {"600000.SH": 10.0},
        close_prices={"600000.SH": 10.0},
        next_preclose_prices={"600000.SH": 10.0},
    )
    adjusted = DaySimulator().step(
        account,
        order,
        {"600000.SH": 10.0},
        {"600000.SH": 10.0},
        close_prices={"600000.SH": 10.0},
        next_preclose_prices={"600000.SH": 5.0},
    )

    assert ordinary.fills == adjusted.fills
    assert ordinary.diagnostics["pretrade_nav"] == adjusted.diagnostics["pretrade_nav"]
    assert ordinary.diagnostics["total_fees"] == adjusted.diagnostics["total_fees"]
    assert ordinary.account_state.positions == {"600000.SH": 100}
    assert adjusted.account_state.positions == {"600000.SH": 200}


def test_cash_only_terminal_transition_has_zero_reward_and_is_deterministic():
    account = AccountState(cash=1_000.0, nav=1_000.0, peak_nav=1_000.0)
    simulator = DaySimulator(FeeSchedule())

    first = simulator.step(account, _plan(), {}, {}, terminated=True)
    second = simulator.step(account, _plan(), {}, {}, terminated=True)

    assert first == second
    assert first.reward == 0.0
    assert first.portfolio_return == 0.0
    assert first.terminated is True
    assert first.account_state == account
    assert first.diagnostics["broker_exact"] is False
    assert first.diagnostics["broker_quantity_exact"] is False


def test_missing_company_action_reference_never_claims_broker_exact_quantity():
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 100},
        sellable_positions={"600000.SH": 100},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )
    result = DaySimulator().step(
        account,
        _plan(),
        {"600000.SH": 10.0},
        {"600000.SH": 11.0},
        close_prices={"600000.SH": 10.5},
        next_preclose_prices={},
    )

    assert result.diagnostics["broker_quantity_exact"] is False
    assert result.diagnostics["quantity_rebase_uncertainty"] == {
        "600000.SH": "missing preClose[T+1]"
    }


def test_reward_denominator_is_recomputed_from_open_not_stale_cached_nav():
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 100},
        sellable_positions={"600000.SH": 100},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )
    result = DaySimulator().step(
        account,
        _plan(),
        {"600000.SH": 20.0},
        {"600000.SH": 20.0},
        close_prices={"600000.SH": 20.0},
        next_preclose_prices={"600000.SH": 20.0},
    )

    assert result.reward == 0.0
    assert result.portfolio_return == 0.0
    assert result.diagnostics["pretrade_nav"] == 2_000.0
    assert result.diagnostics["cached_account_nav_difference"] == -1_000.0


def test_non_positive_reward_boundary_is_rejected_explicitly():
    with pytest.raises(ValueError, match=r"pretrade NAV at open\[T\]"):
        DaySimulator().step(AccountState(cash=0.0), _plan(), {}, {})

from __future__ import annotations

import hashlib
import inspect
import json
import math

import numpy as np
import pytest

from env.contracts import AccountState, OrderPlan
from env.simulator import (
    ACCOUNTING_MODE,
    ACCOUNTING_SCHEMA_HASH,
    ACCOUNTING_SCHEMA_VERSION,
    DaySimulator,
    FeeSchedule,
    accounting_schema_manifest,
    settlement_economics,
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


def test_settlement_economics_broadcasts_all_fallback_and_action_cases():
    economics = settlement_economics(
        current_mark=np.asarray((10.0, 10.0, 10.0, 10.0, 10.0, np.nan)),
        current_close=np.asarray((10.0, 10.0, np.nan, 10.0, np.nan, 10.0)),
        next_preclose=np.asarray((10.0, 5.0, 5.0, np.nan, np.nan, 10.0)),
        next_open=np.asarray((11.0, 5.5, np.nan, 12.0, np.nan, 11.0)),
    )

    np.testing.assert_allclose(
        economics.settlement_mark[:5],
        (11.0, 5.5, 5.0, 12.0, 10.0),
    )
    np.testing.assert_allclose(
        economics.reference_ratio,
        (1.0, 2.0, 2.0, 1.0, 1.0, 1.0),
    )
    np.testing.assert_allclose(
        economics.effective_corporate_action_ratio,
        (1.0, 2.0, 2.0, 1.0, 1.0, 1.0),
    )
    np.testing.assert_allclose(
        economics.gross_return[:5],
        (1.1, 1.1, 1.0, 1.2, 1.0),
    )
    assert np.isnan(economics.gross_return[5])
    np.testing.assert_array_equal(
        economics.gross_return_valid,
        (True, True, True, True, True, False),
    )
    assert economics.mark_source.tolist() == [
        "open[T+1]",
        "open[T+1]",
        "preClose[T+1]",
        "open[T+1]",
        "current_mark[T]",
        "open[T+1]",
    ]
    assert economics.ratio_source.tolist() == [
        "close[T]/preClose[T+1]",
        "close[T]/preClose[T+1]",
        "current_mark[T]_fallback/preClose[T+1]",
        "unavailable; ratio=1",
        "unavailable; ratio=1",
        "close[T]/preClose[T+1]",
    ]


    lightweight = settlement_economics(
        current_mark=np.asarray((10.0, 10.0, 10.0, 10.0, 10.0, np.nan)).reshape(2, 3),
        current_close=np.asarray((10.0, 10.0, np.nan, 10.0, np.nan, 10.0)).reshape(2, 3),
        next_preclose=np.asarray((10.0, 5.0, 5.0, np.nan, np.nan, 10.0)).reshape(2, 3),
        next_open=np.asarray((11.0, 5.5, np.nan, 12.0, np.nan, 11.0)).reshape(2, 3),
        diagnostics=False,
        chunk_rows=1,
    )
    np.testing.assert_array_equal(
        lightweight.gross_return,
        economics.gross_return.reshape(2, 3),
    )
    np.testing.assert_array_equal(
        lightweight.gross_return_valid,
        economics.gross_return_valid.reshape(2, 3),
    )
    assert lightweight.settlement_mark is None
    assert lightweight.mark_source is None
    assert lightweight.ratio_source is None


def test_next_day_delist_is_written_off_without_fabricating_a_fill():
    simulator = DaySimulator(FeeSchedule())
    account = AccountState(
        cash=100.0,
        positions={"000001.SZ": 100},
        sellable_positions={"000001.SZ": 100},
        average_costs={"000001.SZ": 10.0},
        last_prices={"000001.SZ": 10.0},
        nav=1100.0,
        peak_nav=1100.0,
    )

    result = simulator.step(
        account,
        _plan(),
        {"000001.SZ": 10.0},
        {"000001.SZ": np.nan},
        close_prices={"000001.SZ": 10.0},
        next_preclose_prices={"000001.SZ": np.nan},
        next_delisted_codes={"000001.SZ"},
        next_decision_date="2026-08-21",
    )

    assert result.account_state.positions == {}
    assert result.account_state.cash == 100.0
    assert result.account_state.nav == 100.0
    assert result.fills == ()
    assert result.diagnostics["delist_write_offs"] == (
        {
            "type": "delist_write_off",
            "code": "000001.SZ",
            "effective_date": "2026-08-21",
            "quantity": 100,
            "average_cost": 10.0,
            "proceeds": 0.0,
        },
    )


def test_settlement_economics_snaps_near_one_ratio_and_matches_simulator_nav():
    economics = settlement_economics(
        current_mark=10.0,
        current_close=10.00000009,
        next_preclose=10.0,
        next_open=11.0,
    )
    assert economics.reference_ratio.item() == pytest.approx(1.000000009)
    assert economics.effective_corporate_action_ratio.item() == 1.0
    assert economics.gross_return.item() == pytest.approx(1.1)

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
        close_prices={"600000.SH": 10.00000009},
        next_preclose_prices={"600000.SH": 10.0},
    )

    assert result.account_state.nav / account.nav == pytest.approx(
        economics.gross_return.item()
    )
    assert result.diagnostics["corporate_action_adjustments"] == {}


def test_settlement_economics_uses_scalar_math_isclose_boundary():
    reference_ratio = 1.000000015
    economics = settlement_economics(
        current_mark=10.0,
        current_close=reference_ratio * 10.0,
        next_preclose=10.0,
        next_open=11.0,
    )

    assert not math.isclose(reference_ratio, 1.0, rel_tol=1e-8, abs_tol=1e-8)
    assert economics.effective_corporate_action_ratio.item() == pytest.approx(
        reference_ratio
    )
    assert economics.gross_return.item() == pytest.approx(reference_ratio * 1.1)


def test_settlement_economics_does_not_snap_infinite_reference_ratio():
    with np.errstate(over="ignore", invalid="ignore"):
        economics = settlement_economics(
            current_mark=10.0,
            current_close=np.finfo(np.float64).max,
            next_preclose=np.nextafter(0.0, 1.0),
            next_open=11.0,
        )

    assert np.isinf(economics.reference_ratio.item())
    assert np.isinf(economics.effective_corporate_action_ratio.item())
    assert not economics.gross_return_valid.item()
    assert np.isnan(economics.gross_return.item())


def test_default_buy_fees_enter_net_nav_once_before_episode_reward():
    result = DaySimulator().step(
        AccountState(cash=2_000.0, nav=2_000.0, peak_nav=2_000.0),
        _plan(buys={"600000.SH": 100}),
        {"600000.SH": 10.0},
        {"600000.SH": 10.0},
        close_prices={"600000.SH": 10.0},
        next_preclose_prices={"600000.SH": 10.0},
    )

    expected_fee = 0.1 + 1_000.0 * 0.00002 + 1_000.0 * 0.0025
    expected_nav = 2_000.0 - expected_fee
    assert result.fills[0].fee == pytest.approx(expected_fee)
    assert result.account_state.cash == pytest.approx(1_000.0 - expected_fee)
    assert result.account_state.positions == {"600000.SH": 100}
    assert result.account_state.nav == pytest.approx(expected_nav)
    assert result.portfolio_return == pytest.approx(expected_nav / 2_000.0 - 1.0)
    drawdown = 1.0 - expected_nav / 2_000.0
    assert result.reward == 0.0
    assert result.diagnostics["net_log_return"] == pytest.approx(
        math.log(expected_nav / 2_000.0)
    )
    assert result.account_state.max_drawdown == pytest.approx(drawdown)
    assert result.diagnostics["total_fees"] == pytest.approx(expected_fee)
    assert result.diagnostics["gross_traded_notional"] == pytest.approx(1_000.0)
    assert result.diagnostics["gross_turnover_ratio"] == pytest.approx(0.5)
    assert result.diagnostics["total_cost_ratio"] == pytest.approx(
        expected_fee / 2_000.0
    )
    assert not result.policy_memory.initialized
    assert result.account_state.average_costs["600000.SH"] == pytest.approx(
        (1_000.0 + expected_fee) / 100
    )


def test_weighted_average_cost_includes_buy_fees_and_survives_partial_sale():
    fees = FeeSchedule(
        commission_rate=0.01,
        minimum_commission=2.0,
        stamp_tax_rate=0.03,
        transfer_fee_rate=0.005,
        slippage_rate=0.02,
    )
    simulator = DaySimulator(fees)
    account = AccountState(
        cash=3_000.0,
        positions={"600000.SH": 100},
        sellable_positions={"600000.SH": 100},
        average_costs={"600000.SH": 10.0},
        last_prices={"600000.SH": 20.0},
        nav=5_000.0,
        peak_nav=5_000.0,
    )

    added = simulator.step(
        account,
        _plan(buys={"600000.SH": 100}),
        {"600000.SH": 20.0},
        {"600000.SH": 20.0},
        close_prices={"600000.SH": 20.0},
        next_preclose_prices={"600000.SH": 20.0},
    )
    expected_average = (100 * 10.0 + fees.buy_total_cost(100 * 20.0)) / 200
    assert added.account_state.positions == {"600000.SH": 200}
    assert added.account_state.average_costs["600000.SH"] == pytest.approx(
        expected_average
    )

    partially_sold = simulator.step(
        added.account_state,
        _plan(sells=(("600000.SH", 100),)),
        {"600000.SH": 30.0},
        {"600000.SH": 30.0},
        close_prices={"600000.SH": 30.0},
        next_preclose_prices={"600000.SH": 30.0},
    )
    assert partially_sold.account_state.positions == {"600000.SH": 100}
    assert partially_sold.account_state.average_costs["600000.SH"] == pytest.approx(
        expected_average
    )


def test_simulator_keeps_running_drawdown_but_owns_no_episode_reward():
    simulator = DaySimulator()
    account = AccountState(
        cash=0.0,
        positions={"600000.SH": 100},
        sellable_positions={"600000.SH": 100},
        last_prices={"600000.SH": 10.0},
        nav=1_000.0,
        peak_nav=1_000.0,
    )
    first = simulator.step(
        account,
        _plan(),
        {"600000.SH": 10.0},
        {"600000.SH": 9.0},
    )
    assert first.account_state.max_drawdown == pytest.approx(0.10)
    assert first.reward == 0.0

    recovery = simulator.step(
        first.account_state,
        _plan(),
        {"600000.SH": 9.0},
        {"600000.SH": 9.5},
    )
    assert recovery.account_state.max_drawdown == pytest.approx(0.10)
    assert recovery.reward == 0.0


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
        _plan(sells=(("600000.SH", 150),)),
        {"600000.SH": 10.0},
        {},
    )

    expected_fee = (
        max(1_500.0 * 0.0000854, 0.1)
        + 1_500.0 * 0.0005
        + 1_500.0 * 0.00002
        + 1_500.0 * 0.0025
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
            sells=(("600000.SH", 100),),
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
        _plan(sells=(("600000.SH", 100),)),
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


@pytest.mark.parametrize("invalid_open", (float("nan"), float("inf"), -float("inf"), 0.0, -1.0))
def test_missing_opens_use_explained_mark_fallbacks_but_do_not_fill_orders(invalid_open):
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
        _plan(sells=(("600000.SH", 100),)),
        {"600000.SH": invalid_open},
        {"600000.SH": float("nan")},
        close_prices={"600000.SH": 10.5},
        next_preclose_prices={"600000.SH": 10.5},
    )

    assert result.fills == ()
    assert result.account_state.positions == {"600000.SH": 100}
    assert result.account_state.last_prices == {"600000.SH": 10.5}
    assert result.account_state.nav == pytest.approx(1_050.0)
    assert result.reward == 0.0
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
    assert result.reward == 0.0
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
    assert first.reward == 0.0
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
    assert second.reward == 0.0
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

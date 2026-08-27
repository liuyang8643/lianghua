"""Deterministic one-day execution and next-open settlement.

Orders decided for T are filled at raw ``open[T]``.  Reward is the log return
from the account immediately before those fills to its value at raw
``open[T+1]``.  Current-day close is accepted only by the settlement half of
the transition, after the decision, to bridge corporate-action reference
prices without exposing it to :mod:`env.planner`. Corporate actions use the
versioned ``total_return_reinvested`` synthetic-account contract below; its
integer quantities are never represented as broker-exact holdings.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

from env.contracts import AccountState, Fill, OrderPlan, StepResult
from env.fees import FeeSchedule
from env.quantity import floor_buy_quantity, floor_partial_sell_quantity


ACCOUNTING_MODE = "total_return_reinvested"
ACCOUNTING_SCHEMA_VERSION = "total_return_reinvested-v1"


def _accounting_schema_payload() -> dict[str, object]:
    return {
        "schema_version": ACCOUNTING_SCHEMA_VERSION,
        "mode": ACCOUNTING_MODE,
        "broker_exact": False,
        "position_semantics": "integer_synthetic_total_return_reinvested_shares",
        "reward_interval": "pretrade_open[T]_to_pretrade_open[T+1]",
        "execution_price_inputs": ["open[T]"],
        "settlement_only_price_inputs": [
            "close[T]",
            "preClose[T+1]",
            "open[T+1]",
        ],
        "current_mark_priority": ["open[T]", "account.last_prices"],
        "corporate_action_factor": {
            "symbol": "A",
            "formula": "close[T]/preClose[T+1]",
            "ordinary_day_relative_tolerance": 1e-8,
            "ordinary_day_absolute_tolerance": 1e-8,
        },
        "quantity_transition": {
            "economic_quantity": "quantity_after_open_fills*A",
            "integer_quantity_candidate": (
                "round_ties_to_even(economic_quantity)"
            ),
            "cash_safety_constraint": (
                "process_positions_in_ascending_code_order; if the candidate "
                "cash residual would make account cash negative, use "
                "floor(economic_quantity)"
            ),
            "cash_residual": (
                "(economic_quantity-synthetic_quantity)*settlement_mark[T+1]"
            ),
        },
        "cost_basis_transition": (
            "preserve_total_cost_across_nonzero_synthetic_quantity_rebase"
        ),
        "sellable_transition": "all_synthetic_positions_sellable_at_T+1",
        "nav_transition": (
            "cash_after_fills_and_rounding_residual+"
            "sum(synthetic_quantity*settlement_mark[T+1])"
        ),
        "settlement_mark_priority": [
            "open[T+1]",
            "preClose[T+1]",
            "close[T]",
            "current_mark",
        ],
        "missing_reference_policy": {
            "missing_close[T]": (
                "approximate_A_with_current_mark[T]/preClose[T+1]"
            ),
            "missing_preClose[T+1]": "A=1_and_record_uncertainty",
        },
    }


def _canonical_schema_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


ACCOUNTING_SCHEMA_HASH = _canonical_schema_hash(_accounting_schema_payload())


def accounting_schema_manifest() -> dict[str, object]:
    """Return the immutable-semantics manifest a policy bundle must bind.

    A fresh JSON-serialisable mapping is returned so callers cannot mutate the
    module's accounting identity. ``schema_hash`` excludes only itself.
    """

    payload = _accounting_schema_payload()
    payload["schema_hash"] = ACCOUNTING_SCHEMA_HASH
    return payload


def _valid_price(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


class DaySimulator:
    """Execute and settle using the synthetic total-return accounting schema."""

    def __init__(self, fees: FeeSchedule | None = None) -> None:
        self.fees = fees or FeeSchedule()

    def step(
        self,
        account: AccountState,
        order_plan: OrderPlan,
        open_prices: Mapping[str, float],
        next_open_prices: Mapping[str, float],
        *,
        close_prices: Mapping[str, float] | None = None,
        next_preclose_prices: Mapping[str, float] | None = None,
        next_decision_date: str = "",
        terminated: bool = False,
    ) -> StepResult:
        """Fill at T-open and mark the resulting account at T+1-open.

        ``close_prices`` and ``next_preclose_prices`` are settlement inputs. If
        both are present, holdings are economically rebased by
        ``close[T] / preClose[T+1]`` before raw ``open[T+1]`` valuation. The
        rebased quantity is rounded to an integer and its value residual moves
        to cash. Diagnostics bind this transition to
        :func:`accounting_schema_manifest` and always report ``broker_exact``
        false; these synthetic quantities must not be sent to a broker.
        """

        current_open = {str(code): float(value) for code, value in open_prices.items()}

        positions = {
            str(code): int(quantity)
            for code, quantity in account.positions.items()
            if int(quantity) > 0
        }
        sellable = {
            str(code): max(0, int(quantity))
            for code, quantity in account.sellable_positions.items()
        }
        average_costs = {
            str(code): float(value)
            for code, value in account.average_costs.items()
            if _valid_price(value)
        }
        current_marks, current_mark_fallbacks = self._mark_prices(
            positions,
            primary=current_open,
            fallback=account.last_prices,
            primary_name="open[T]",
            fallback_name="account.last_prices",
        )
        balance_sheet_pretrade_nav = float(account.cash) + sum(
            quantity * current_marks[code]
            for code, quantity in positions.items()
            if code in current_marks
        )
        missing_current_marks = tuple(code for code in positions if code not in current_marks)
        if missing_current_marks:
            raise ValueError(
                "cannot compute pretrade open[T] NAV for positions without a mark: "
                f"{missing_current_marks}"
            )
        pretrade_nav = balance_sheet_pretrade_nav
        nav_source = "open[T]_mark"
        if not math.isfinite(pretrade_nav) or pretrade_nav <= 0.0:
            raise ValueError("pretrade NAV at open[T] must be finite and positive")

        cash = float(account.cash)
        if not math.isfinite(cash):
            raise ValueError("account cash must be finite")

        fills: list[Fill] = []
        skipped_orders: list[dict[str, object]] = []
        fee_breakdown: list[dict[str, float | str | int]] = []
        seen_sell_codes: set[str] = set()
        total_fees = 0.0

        # Sells always execute before any buy so their proceeds are available.
        for raw_code, raw_requested in order_plan.sell_orders:
            code = str(raw_code)
            if code in seen_sell_codes:
                raise ValueError(f"duplicate sell order for {code}")
            seen_sell_codes.add(code)
            requested = int(raw_requested)
            held = positions.get(code, 0)
            available = min(held, sellable.get(code, 0))
            if held <= 0 or available <= 0:
                skipped_orders.append(
                    {"code": code, "side": "sell", "reason": "not_sellable"}
                )
                continue
            if not _valid_price(current_open.get(code)):
                skipped_orders.append(
                    {"code": code, "side": "sell", "reason": "missing_open[T]"}
                )
                continue
            if requested < 0:
                quantity = available
            else:
                quantity = min(requested, available)
                if quantity < held:
                    quantity = floor_partial_sell_quantity(code, quantity)
            if quantity <= 0:
                skipped_orders.append(
                    {
                        "code": code,
                        "side": "sell",
                        "reason": "below_exchange_minimum",
                    }
                )
                continue

            price = current_open[code]
            notional = quantity * price
            commission = self.fees.broker_commission(notional)
            stamp_tax = notional * self.fees.stamp_tax_rate
            transfer_fee = notional * self.fees.transfer_fee_rate
            slippage = notional * self.fees.slippage_rate
            fee = commission + stamp_tax + transfer_fee + slippage
            cash += notional - fee
            total_fees += fee
            remaining = held - quantity
            if remaining > 0:
                positions[code] = remaining
                sellable[code] = max(0, available - quantity)
            else:
                positions.pop(code, None)
                sellable.pop(code, None)
                average_costs.pop(code, None)
            fills.append(
                Fill(
                    code=code,
                    side="sell",
                    quantity=quantity,
                    price=price,
                    fee=fee,
                    timestamp=order_plan.decision_date,
                )
            )
            fee_breakdown.append(
                {
                    "code": code,
                    "side": "sell",
                    "quantity": quantity,
                    "commission": commission,
                    "stamp_tax": stamp_tax,
                    "transfer_fee": transfer_fee,
                    "slippage": slippage,
                    "total": fee,
                }
            )

        seen_buy_codes: set[str] = set()
        for raw_code, raw_requested in order_plan.buy_orders.items():
            code = str(raw_code)
            if code in seen_buy_codes:
                raise ValueError(f"duplicate buy order for {code}")
            seen_buy_codes.add(code)
            requested = floor_buy_quantity(code, int(raw_requested))
            if requested <= 0:
                skipped_orders.append(
                    {
                        "code": code,
                        "side": "buy",
                        "reason": "below_exchange_minimum",
                    }
                )
                continue
            if not _valid_price(current_open.get(code)):
                skipped_orders.append(
                    {"code": code, "side": "buy", "reason": "missing_open[T]"}
                )
                continue

            price = current_open[code]
            affordable = self._affordable_buy_quantity(code, cash, price)
            quantity = min(requested, affordable)
            if quantity <= 0:
                skipped_orders.append(
                    {"code": code, "side": "buy", "reason": "insufficient_cash"}
                )
                continue
            notional = quantity * price
            commission = self.fees.broker_commission(notional)
            transfer_fee = notional * self.fees.transfer_fee_rate
            slippage = notional * self.fees.slippage_rate
            stamp_tax = 0.0
            fee = commission + transfer_fee + slippage
            total_cost = notional + fee
            cash -= total_cost
            total_fees += fee

            old_quantity = positions.get(code, 0)
            old_average = average_costs.get(
                code,
                float(account.last_prices.get(code, price))
                if _valid_price(account.last_prices.get(code))
                else price,
            )
            new_quantity = old_quantity + quantity
            average_costs[code] = (
                old_quantity * old_average + total_cost
            ) / new_quantity
            positions[code] = new_quantity
            # T-day buys are deliberately not sellable until T+1 settlement.
            sellable.setdefault(code, 0)
            fills.append(
                Fill(
                    code=code,
                    side="buy",
                    quantity=quantity,
                    price=price,
                    fee=fee,
                    timestamp=order_plan.decision_date,
                )
            )
            fee_breakdown.append(
                {
                    "code": code,
                    "side": "buy",
                    "quantity": quantity,
                    "commission": commission,
                    "stamp_tax": stamp_tax,
                    "transfer_fee": transfer_fee,
                    "slippage": slippage,
                    "total": fee,
                }
            )

        # These fields become visible only to settlement, after all T-open
        # fills are fixed. They cannot influence order quantity, price, or fee.
        next_open = {
            str(code): float(value) for code, value in next_open_prices.items()
        }
        current_close = {
            str(code): float(value)
            for code, value in (close_prices or {}).items()
        }
        next_preclose = {
            str(code): float(value)
            for code, value in (next_preclose_prices or {}).items()
        }
        next_positions: dict[str, int] = {}
        next_average_costs: dict[str, float] = {}
        next_marks: dict[str, float] = {}
        settlement_fallbacks: dict[str, str] = {}
        quantity_rebase_uncertainty: dict[str, str] = {}
        corporate_action_adjustments: dict[str, dict[str, object]] = {}
        corporate_action_cash_residual = 0.0
        # Stable code order makes the cash-safe integerisation deterministic
        # even when callers construct their position mappings differently.
        for code in sorted(positions):
            quantity = positions[code]
            if _valid_price(current_open.get(code)):
                current_mark = current_open[code]
            elif _valid_price(current_marks.get(code)):
                current_mark = current_marks[code]
            else:
                raise ValueError(f"cannot establish current economic mark for {code}")

            has_next_open = _valid_price(next_open.get(code))
            has_close = _valid_price(current_close.get(code))
            has_next_preclose = _valid_price(next_preclose.get(code))
            if not has_close or not has_next_preclose:
                missing_fields = []
                if not has_close:
                    missing_fields.append("close[T]")
                if not has_next_preclose:
                    missing_fields.append("preClose[T+1]")
                quantity_rebase_uncertainty[code] = (
                    "missing " + ", ".join(missing_fields)
                )
            if has_next_open:
                raw_next_mark = next_open[code]
            elif has_next_preclose:
                raw_next_mark = next_preclose[code]
                settlement_fallbacks[code] = (
                    "open[T+1]_missing_or_suspended; used_preClose[T+1]"
                )
            elif has_close:
                raw_next_mark = current_close[code]
                settlement_fallbacks[code] = (
                    "open[T+1]_and_preClose[T+1]_missing; used_close[T]"
                )
            else:
                raw_next_mark = current_mark
                settlement_fallbacks[code] = (
                    "open[T+1]_preClose[T+1]_close[T]_missing; carried_current_mark"
                )

            if has_close and has_next_preclose:
                reference_mark = current_close[code]
                reference_ratio = reference_mark / next_preclose[code]
                ratio_source = "close[T]/preClose[T+1]"
            elif has_next_preclose:
                reference_mark = current_mark
                reference_ratio = reference_mark / next_preclose[code]
                ratio_source = "current_mark[T]_fallback/preClose[T+1]"
                settlement_fallbacks[code] = (
                    settlement_fallbacks.get(code, "")
                    + ("; " if code in settlement_fallbacks else "")
                    + "close[T]_missing; economic share ratio approximated from open[T]"
                )
            else:
                reference_mark = current_mark
                reference_ratio = 1.0
                ratio_source = "unavailable; ratio=1"
                settlement_fallbacks[code] = (
                    settlement_fallbacks.get(code, "")
                    + ("; " if code in settlement_fallbacks else "")
                    + "preClose[T+1]_missing; corporate-action rebase unavailable"
                )

            ordinary_day = math.isclose(
                reference_ratio, 1.0, rel_tol=1e-8, abs_tol=1e-8
            )
            effective_ratio = 1.0 if ordinary_day else reference_ratio
            exact_economic_quantity = quantity * effective_ratio
            nearest_quantity = int(round(exact_economic_quantity))
            floored_quantity = int(math.floor(exact_economic_quantity))
            target_economic_value = exact_economic_quantity * raw_next_mark
            nearest_position_value = nearest_quantity * raw_next_mark
            nearest_cash_residual = target_economic_value - nearest_position_value
            cash_safe_floor_applied = (
                nearest_quantity > floored_quantity
                and cash + nearest_cash_residual < 0.0
            )
            rebased_quantity = (
                floored_quantity if cash_safe_floor_applied else nearest_quantity
            )
            rebased_position_value = rebased_quantity * raw_next_mark
            cash_residual = target_economic_value - rebased_position_value
            cash += cash_residual
            corporate_action_cash_residual += cash_residual

            if rebased_quantity > 0:
                next_positions[code] = rebased_quantity
                next_marks[code] = raw_next_mark
                old_total_cost = average_costs.get(code, current_mark) * quantity
                next_average_costs[code] = old_total_cost / rebased_quantity
            if not ordinary_day:
                corporate_action_adjustments[code] = {
                    "mode": ACCOUNTING_MODE,
                    "accounting_schema_hash": ACCOUNTING_SCHEMA_HASH,
                    "broker_exact": False,
                    "broker_quantity_exact": False,
                    "ratio_source": ratio_source,
                    "reference_mark": reference_mark,
                    "preClose[T+1]": next_preclose.get(code),
                    "raw_mark[T+1]": raw_next_mark,
                    "reference_ratio": reference_ratio,
                    "quantity_before": quantity,
                    "synthetic_economic_quantity": exact_economic_quantity,
                    "synthetic_quantity_after_integer_rebase": rebased_quantity,
                    "nearest_ties_to_even_quantity": nearest_quantity,
                    "cash_safe_floor_applied": cash_safe_floor_applied,
                    "rounding_cash_residual": cash_residual,
                    "target_economic_value": target_economic_value,
                }

        next_nav = cash + sum(
            quantity * next_marks[code]
            for code, quantity in next_positions.items()
        )
        if not math.isfinite(next_nav) or next_nav <= 0.0:
            raise ValueError("pretrade NAV at open[T+1] must be finite and positive")
        portfolio_return = next_nav / pretrade_nav - 1.0
        reward = math.log(next_nav / pretrade_nav)
        peak_nav = max(float(account.peak_nav), pretrade_nav, next_nav)
        next_account = AccountState(
            cash=cash,
            positions=next_positions,
            sellable_positions=dict(next_positions),
            average_costs=next_average_costs,
            last_prices=next_marks,
            nav=next_nav,
            peak_nav=peak_nav,
        )
        accounting_manifest = accounting_schema_manifest()
        accounting_identity = {
            "schema_version": accounting_manifest["schema_version"],
            "schema_hash": accounting_manifest["schema_hash"],
            "mode": accounting_manifest["mode"],
            "broker_exact": accounting_manifest["broker_exact"],
        }
        diagnostics: dict[str, object] = {
            "reward_interval": "pretrade_open[T]_to_pretrade_open[T+1]",
            "decision_date": order_plan.decision_date,
            "next_decision_date": next_decision_date,
            "pretrade_nav": pretrade_nav,
            "pretrade_nav_source": nav_source,
            "cached_account_nav": float(account.nav),
            "cached_account_nav_difference": float(account.nav) - pretrade_nav,
            "balance_sheet_pretrade_nav": balance_sheet_pretrade_nav,
            "next_pretrade_nav": next_nav,
            "current_mark_fallbacks": current_mark_fallbacks,
            "settlement_fallbacks": settlement_fallbacks,
            "corporate_action_adjustments": corporate_action_adjustments,
            "quantity_rebase_uncertainty": quantity_rebase_uncertainty,
            "accounting_schema": accounting_identity,
            "accounting_model": accounting_manifest["mode"],
            "corporate_action_quantity_semantics": accounting_manifest[
                "position_semantics"
            ],
            "corporate_action_cash_residual": corporate_action_cash_residual,
            "broker_exact": False,
            "broker_quantity_exact": False,
            "fee_breakdown": tuple(fee_breakdown),
            "total_fees": total_fees,
            "skipped_orders": tuple(skipped_orders),
            "fill_sequence": tuple((fill.side, fill.code) for fill in fills),
        }
        return StepResult(
            account_state=next_account,
            reward=reward,
            portfolio_return=portfolio_return,
            fills=tuple(fills),
            terminated=bool(terminated),
            diagnostics=diagnostics,
        )

    def _affordable_buy_quantity(
        self,
        code: str,
        cash: float,
        price: float,
    ) -> int:
        low = 0
        high = int(cash / price) + 1
        while low + 1 < high:
            middle = (low + high) // 2
            notional = middle * price
            if notional + self.fees.buy_fee(notional) <= cash:
                low = middle
            else:
                high = middle
        return floor_buy_quantity(code, low)

    @staticmethod
    def _mark_prices(
        positions: Mapping[str, int],
        *,
        primary: Mapping[str, float],
        fallback: Mapping[str, float],
        primary_name: str,
        fallback_name: str,
    ) -> tuple[dict[str, float], dict[str, str]]:
        marks: dict[str, float] = {}
        fallbacks: dict[str, str] = {}
        for code in positions:
            if _valid_price(primary.get(code)):
                marks[code] = float(primary[code])
            elif _valid_price(fallback.get(code)):
                marks[code] = float(fallback[code])
                fallbacks[code] = f"{primary_name}_missing; used_{fallback_name}"
            else:
                fallbacks[code] = f"{primary_name}_and_{fallback_name}_missing"
        return marks, fallbacks


__all__ = [
    "ACCOUNTING_MODE",
    "ACCOUNTING_SCHEMA_HASH",
    "ACCOUNTING_SCHEMA_VERSION",
    "DaySimulator",
    "FeeSchedule",
    "accounting_schema_manifest",
]

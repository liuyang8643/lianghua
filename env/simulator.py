"""Deterministic one-day execution and next-open settlement.

Orders decided for T are filled at raw ``open[T]``.  This module owns only
execution and settlement; episode-level training rewards are computed by the
canonical session from the resulting net next-open NAV. Current-day close is
accepted only by the settlement half of
the transition, after the decision, to bridge corporate-action reference
prices without exposing it to :mod:`env.planner`. Corporate actions use the
versioned ``total_return_reinvested`` synthetic-account contract below; its
integer quantities are never represented as broker-exact holdings.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from env.contracts import (
    AccountState,
    Fill,
    OrderPlan,
    PolicyMemory,
    StepResult,
)
from env.fees import FeeSchedule, summarize_fill_costs
from env.quantity import floor_buy_quantity, floor_partial_sell_quantity


ACCOUNTING_MODE = "total_return_reinvested"
ACCOUNTING_SCHEMA_VERSION = "total_return_reinvested-v1"


@dataclass(frozen=True)
class SettlementEconomics:
    """Broadcast settlement terms shared by observations and account updates."""

    settlement_mark: NDArray[np.float64] | None
    reference_ratio: NDArray[np.float64] | None
    effective_corporate_action_ratio: NDArray[np.float64] | None
    gross_return: NDArray[np.float64]
    current_mark_valid: NDArray[np.bool_] | None
    current_close_valid: NDArray[np.bool_] | None
    next_preclose_valid: NDArray[np.bool_] | None
    next_open_valid: NDArray[np.bool_] | None
    gross_return_valid: NDArray[np.bool_]
    mark_source: NDArray[np.str_] | None
    ratio_source: NDArray[np.str_] | None


@njit(cache=True, fastmath=False, parallel=False, error_model='numpy')
def _settlement_terms(current, close, preclose, following_open):
    """Single numeric settlement law for a panel or a held-position vector."""
    size = len(current)
    numeric = np.empty((4, size), dtype=np.float64)
    valid = np.empty((5, size), dtype=np.bool_)
    sources = np.empty((2, size), dtype=np.int8)
    for index in range(size):
        current_valid = np.isfinite(current[index]) and current[index] > 0.0
        close_valid = np.isfinite(close[index]) and close[index] > 0.0
        preclose_valid = np.isfinite(preclose[index]) and preclose[index] > 0.0
        next_valid = np.isfinite(following_open[index]) and following_open[index] > 0.0
        valid[0, index] = current_valid
        valid[1, index] = close_valid
        valid[2, index] = preclose_valid
        valid[3, index] = next_valid
        if next_valid:
            mark, mark_source = following_open[index], 0
        elif preclose_valid:
            mark, mark_source = preclose[index], 1
        elif close_valid:
            mark, mark_source = close[index], 2
        elif current_valid:
            mark, mark_source = current[index], 3
        else:
            mark, mark_source = np.nan, 4
        if close_valid and preclose_valid:
            ratio, ratio_source = close[index] / preclose[index], 0
        elif current_valid and preclose_valid:
            ratio, ratio_source = current[index] / preclose[index], 1
        else:
            ratio, ratio_source = 1.0, 2
        distance = abs(ratio - 1.0)
        tolerance = max(1e-8 * max(abs(ratio), 1.0), 1e-8)
        effective = 1.0 if np.isfinite(ratio) and distance <= tolerance else ratio
        gross_valid = current_valid and np.isfinite(mark) and mark > 0.0 and np.isfinite(effective) and effective > 0.0
        numeric[0, index] = mark
        numeric[1, index] = ratio
        numeric[2, index] = effective
        numeric[3, index] = effective * mark / current[index] if gross_valid else np.nan
        valid[4, index] = gross_valid
        sources[0, index] = mark_source
        sources[1, index] = ratio_source
    return numeric, valid, sources


_MARK_SOURCES = np.array(('open[T+1]', 'preClose[T+1]', 'close[T]', 'current_mark[T]', 'unavailable'), dtype='U20')
_RATIO_SOURCES = np.array(('close[T]/preClose[T+1]', 'current_mark[T]_fallback/preClose[T+1]', 'unavailable; ratio=1'), dtype='U48')


def _complete_settlement_economics(
    *, current_mark: ArrayLike, current_close: ArrayLike,
    next_preclose: ArrayLike, next_open: ArrayLike, include_sources: bool = True,
) -> SettlementEconomics:
    """Broadcast inputs to the single serial total-return settlement kernel."""
    arrays = np.broadcast_arrays(
        np.asarray(current_mark, dtype=np.float64), np.asarray(current_close, dtype=np.float64),
        np.asarray(next_preclose, dtype=np.float64), np.asarray(next_open, dtype=np.float64),
    )
    shape = arrays[0].shape
    numeric, valid, sources = _settlement_terms(*(values.ravel() for values in arrays))
    return SettlementEconomics(
        settlement_mark=numeric[0].reshape(shape),
        reference_ratio=numeric[1].reshape(shape),
        effective_corporate_action_ratio=numeric[2].reshape(shape),
        gross_return=numeric[3].reshape(shape),
        current_mark_valid=valid[0].reshape(shape),
        current_close_valid=valid[1].reshape(shape),
        next_preclose_valid=valid[2].reshape(shape),
        next_open_valid=valid[3].reshape(shape),
        gross_return_valid=valid[4].reshape(shape),
        mark_source=_MARK_SOURCES[sources[0]].reshape(shape) if include_sources else None,
        ratio_source=_RATIO_SOURCES[sources[1]].reshape(shape) if include_sources else None,
    )


def settlement_economics(
    *,
    current_mark: ArrayLike,
    current_close: ArrayLike,
    next_preclose: ArrayLike,
    next_open: ArrayLike,
    diagnostics: bool = True,
    chunk_rows: int = 64,
) -> SettlementEconomics:
    """Resolve settlement terms, optionally retaining only gross returns.

    ``diagnostics=False`` is the historical-panel path. It processes the first
    axis in bounded chunks and returns only ``gross_return`` and
    ``gross_return_valid``; scalar simulator settlement keeps the complete
    marks, ratios, flags, and source labels by default.
    """

    if diagnostics:
        return _complete_settlement_economics(
            current_mark=current_mark,
            current_close=current_close,
            next_preclose=next_preclose,
            next_open=next_open,
        )
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive int")
    current, close, preclose, following_open = np.broadcast_arrays(
        np.asarray(current_mark),
        np.asarray(current_close),
        np.asarray(next_preclose),
        np.asarray(next_open),
    )
    if any(
        not np.issubdtype(values.dtype, np.number)
        for values in (current, close, preclose, following_open)
    ):
        raise ValueError("settlement price inputs must be numeric")
    gross = np.empty(current.shape, dtype=np.float64)
    valid = np.empty(current.shape, dtype=np.bool_)
    row_count = current.shape[0] if current.ndim > 1 else 1
    for start in range(0, row_count, chunk_rows):
        stop = min(start + chunk_rows, row_count)
        selector = slice(start, stop) if current.ndim > 1 else (...,)
        complete = _complete_settlement_economics(
            current_mark=current[selector],
            current_close=close[selector],
            next_preclose=preclose[selector],
            next_open=following_open[selector],
            include_sources=False,
        )
        gross[selector] = complete.gross_return
        valid[selector] = complete.gross_return_valid
    return SettlementEconomics(
        settlement_mark=None,
        reference_ratio=None,
        effective_corporate_action_ratio=None,
        gross_return=gross,
        current_mark_valid=None,
        current_close_valid=None,
        next_preclose_valid=None,
        next_open_valid=None,
        gross_return_valid=valid,
        mark_source=None,
        ratio_source=None,
    )


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
    if value is None:
        return False
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
        next_delisted_codes: Iterable[str] = (),
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
        # Normalize the public mapping before filtering, so duplicate keys
        # after str conversion retain the same last-value semantics. Every
        # later T-open consumer can reuse this one finite-positive check.
        current_open = {
            code: price for code, price in current_open.items()
            if math.isfinite(price) and price > 0.0
        }

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

        # Sells always execute before any buy so their proceeds are available.
        for raw_code, raw_requested in order_plan.sell_orders:
            code = str(raw_code)
            if code in seen_sell_codes:
                raise ValueError(f"duplicate sell order for {code}")
            seen_sell_codes.add(code)
            requested = int(raw_requested)
            if requested <= 0:
                raise ValueError("sell order quantity must be a positive int")
            held = positions.get(code, 0)
            available = min(held, sellable.get(code, 0))
            if held <= 0 or available <= 0:
                skipped_orders.append(
                    {"code": code, "side": "sell", "reason": "not_sellable"}
                )
                continue
            if code not in current_open:
                skipped_orders.append(
                    {"code": code, "side": "sell", "reason": "missing_open[T]"}
                )
                continue
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
            if code not in current_open:
                skipped_orders.append(
                    {"code": code, "side": "buy", "reason": "missing_open[T]"}
                )
                continue

            price = current_open[code]
            # Plans reserve frozen-price costs already. When the requested
            # lot fits actual cash, avoid solving for a larger unused maximum.
            # Retain the inverse's integer upper bound at float boundaries.
            requested_fits = False
            if cash > 0.0 and requested <= int(cash / price):
                notional = requested * price
                commission, transfer_fee, slippage, fee = self.fees.buy_fee_components(notional)
                requested_fits = notional + fee <= cash
            if requested_fits:
                quantity = requested
            else:
                quantity = min(requested, self._affordable_buy_quantity(code, cash, price))
            if quantity <= 0:
                skipped_orders.append(
                    {"code": code, "side": "buy", "reason": "insufficient_cash"}
                )
                continue
            if not requested_fits:
                notional = quantity * price
                commission, transfer_fee, slippage, fee = self.fees.buy_fee_components(notional)
            stamp_tax = 0.0
            total_cost = notional + fee
            cash -= total_cost

            old_quantity = positions.get(code, 0)
            old_average = average_costs.get(code)
            if old_average is None:
                last_price = account.last_prices.get(code)
                old_average = float(last_price) if _valid_price(last_price) else price
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

        gross_traded_notional, total_fees = summarize_fill_costs(fills)
        post_fill_cash = cash
        post_fill_invested_value = 0.0
        post_fill_marks: dict[str, float] = {}
        for code, quantity in positions.items():
            if code in current_open:
                mark = current_open[code]
            elif code in current_marks:
                mark = current_marks[code]
            else:
                raise ValueError(f"cannot mark post-fill position {code}")
            post_fill_invested_value += quantity * mark
            post_fill_marks[code] = mark
        post_fill_nav = post_fill_cash + post_fill_invested_value
        if not math.isfinite(post_fill_nav) or post_fill_nav <= 0.0:
            raise ValueError("post-fill NAV at open[T] must be finite and positive")
        post_fill_exposure = post_fill_invested_value / post_fill_nav

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
        delist_write_offs: list[dict[str, object]] = []
        corporate_action_cash_residual = 0.0
        delisted = frozenset(str(code) for code in next_delisted_codes)
        settlement_codes = sorted(positions)
        # Price economics are independent across positions. Evaluate the same
        # vectorised authority once; cash-safe integerisation below stays serial.
        # These four rows are constructed from the same held-code axis. Use
        # the sole settlement law directly; the public broadcasting adapter
        # remains for arbitrary market panels. Source labels are materialized
        # only where an actual corporate-action diagnostic consumes them.
        settlement_numeric, settlement_valid, settlement_sources = _settlement_terms(
            np.asarray([post_fill_marks[code] for code in settlement_codes], dtype=np.float64),
            np.asarray([current_close.get(code, math.nan) for code in settlement_codes], dtype=np.float64),
            np.asarray([next_preclose.get(code, math.nan) for code in settlement_codes], dtype=np.float64),
            np.asarray([next_open.get(code, math.nan) for code in settlement_codes], dtype=np.float64),
        )
        # Stable code order makes the cash-safe integerisation deterministic
        # even when callers construct their position mappings differently.
        settlement_rows = zip(
            settlement_codes, settlement_valid[3].tolist(),
            settlement_valid[1].tolist(), settlement_valid[2].tolist(),
            settlement_numeric[0].tolist(), settlement_numeric[2].tolist(),
        )
        next_invested_value = 0.0
        for settlement_index, (code, has_next_open, has_close, has_next_preclose, raw_next_mark, effective_ratio) in enumerate(settlement_rows):
            quantity = positions[code]
            if code in delisted:
                delist_write_offs.append(
                    {
                        "type": "delist_write_off",
                        "code": code,
                        "effective_date": next_decision_date,
                        "quantity": quantity,
                        "average_cost": average_costs.get(code),
                        "proceeds": 0.0,
                    }
                )
                continue
            current_mark = post_fill_marks[code]
            if not has_close or not has_next_preclose:
                missing_fields = []
                if not has_close:
                    missing_fields.append("close[T]")
                if not has_next_preclose:
                    missing_fields.append("preClose[T+1]")
                quantity_rebase_uncertainty[code] = (
                    "missing " + ", ".join(missing_fields)
                )
            if not has_next_open and has_next_preclose:
                settlement_fallbacks[code] = (
                    "open[T+1]_missing_or_suspended; used_preClose[T+1]"
                )
            elif not has_next_open and has_close:
                settlement_fallbacks[code] = (
                    "open[T+1]_and_preClose[T+1]_missing; used_close[T]"
                )
            elif not has_next_open:
                settlement_fallbacks[code] = (
                    "open[T+1]_preClose[T+1]_close[T]_missing; carried_current_mark"
                )

            if has_close and has_next_preclose:
                reference_mark = current_close[code]
            elif has_next_preclose:
                reference_mark = current_mark
                settlement_fallbacks[code] = (
                    settlement_fallbacks.get(code, "")
                    + ("; " if code in settlement_fallbacks else "")
                    + "close[T]_missing; economic share ratio approximated from open[T]"
                )
            else:
                reference_mark = current_mark
                settlement_fallbacks[code] = (
                    settlement_fallbacks.get(code, "")
                    + ("; " if code in settlement_fallbacks else "")
                    + "preClose[T+1]_missing; corporate-action rebase unavailable"
                )

            ordinary_day = effective_ratio == 1.0
            if ordinary_day and quantity <= 2**53:
                # Integers in this range are represented exactly by float64;
                # multiplying by one leaves zero rounding cash by construction.
                rebased_quantity = quantity
                cash_residual = 0.0
            else:
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
                next_invested_value += rebased_quantity * raw_next_mark
                old_total_cost = average_costs.get(code, current_mark) * quantity
                next_average_costs[code] = old_total_cost / rebased_quantity
            if not ordinary_day:
                corporate_action_adjustments[code] = {
                    "mode": ACCOUNTING_MODE,
                    "accounting_schema_hash": ACCOUNTING_SCHEMA_HASH,
                    "broker_exact": False,
                    "broker_quantity_exact": False,
                    "ratio_source": str(_RATIO_SOURCES[settlement_sources[1, settlement_index]]),
                    "reference_mark": reference_mark,
                    "preClose[T+1]": next_preclose.get(code),
                    "raw_mark[T+1]": raw_next_mark,
                    "reference_ratio": float(settlement_numeric[1, settlement_index]),
                    "quantity_before": quantity,
                    "synthetic_economic_quantity": exact_economic_quantity,
                    "synthetic_quantity_after_integer_rebase": rebased_quantity,
                    "nearest_ties_to_even_quantity": nearest_quantity,
                    "cash_safe_floor_applied": cash_safe_floor_applied,
                    "rounding_cash_residual": cash_residual,
                    "target_economic_value": target_economic_value,
                }

        next_nav = cash + next_invested_value
        if not math.isfinite(next_nav) or next_nav <= 0.0:
            raise ValueError("pretrade NAV at open[T+1] must be finite and positive")
        portfolio_return = next_nav / pretrade_nav - 1.0
        peak_nav = max(float(account.peak_nav), pretrade_nav, next_nav)
        current_peak = max(float(account.peak_nav), pretrade_nav)
        current_drawdown = 1.0 - pretrade_nav / current_peak
        running_max_drawdown = max(float(account.max_drawdown), current_drawdown)
        next_drawdown = 1.0 - next_nav / peak_nav
        next_max_drawdown = max(running_max_drawdown, next_drawdown)
        drawdown_increment = next_max_drawdown - running_max_drawdown
        net_log_return = math.log(next_nav / pretrade_nav)
        next_account = AccountState(
            cash=cash,
            positions=next_positions,
            sellable_positions=dict(next_positions),
            average_costs=next_average_costs,
            last_prices=next_marks,
            nav=next_nav,
            peak_nav=peak_nav,
            max_drawdown=next_max_drawdown,
        )
        accounting_manifest = accounting_schema_manifest()
        accounting_identity = {
            "schema_version": accounting_manifest["schema_version"],
            "schema_hash": accounting_manifest["schema_hash"],
            "mode": accounting_manifest["mode"],
            "broker_exact": accounting_manifest["broker_exact"],
        }
        policy_memory = (
            PolicyMemory()
            if order_plan.day_config is None
            else PolicyMemory(
                previous_day_config=order_plan.day_config,
                previous_gross_turnover_ratio=(
                    gross_traded_notional / pretrade_nav
                ),
                previous_total_cost_ratio=total_fees / pretrade_nav,
            )
        )
        diagnostics: dict[str, object] = {
            "reward_interval": "pretrade_open[T]_to_pretrade_open[T+1]",
            "decision_date": order_plan.decision_date,
            "next_decision_date": next_decision_date,
            "pretrade_nav": pretrade_nav,
            "pretrade_nav_source": nav_source,
            "cached_account_nav": float(account.nav),
            "cached_account_nav_difference": float(account.nav) - pretrade_nav,
            "balance_sheet_pretrade_nav": balance_sheet_pretrade_nav,
            "post_fill_cash": post_fill_cash,
            "post_fill_invested_value": post_fill_invested_value,
            "post_fill_nav": post_fill_nav,
            "post_fill_exposure": post_fill_exposure,
            "next_pretrade_nav": next_nav,
            "net_log_return": net_log_return,
            "running_max_drawdown": next_max_drawdown,
            "drawdown_increment": drawdown_increment,
            "current_mark_fallbacks": current_mark_fallbacks,
            "settlement_fallbacks": settlement_fallbacks,
            "corporate_action_adjustments": corporate_action_adjustments,
            "account_events": tuple(delist_write_offs),
            "delist_write_offs": tuple(delist_write_offs),
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
            "gross_traded_notional": gross_traded_notional,
            "gross_turnover_ratio": gross_traded_notional / pretrade_nav,
            "total_cost_ratio": total_fees / pretrade_nav,
            "skipped_orders": tuple(skipped_orders),
            "fill_sequence": tuple((fill.side, fill.code) for fill in fills),
        }
        return StepResult(
            account_state=next_account,
            reward=0.0,
            portfolio_return=portfolio_return,
            policy_memory=policy_memory,
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
        return floor_buy_quantity(code, self.fees.affordable_buy_shares(cash, price))

    @staticmethod
    def _mark_prices(
        positions: Mapping[str, int],
        *,
        primary: Mapping[str, float],
        fallback: Mapping[str, float],
        primary_name: str,
        fallback_name: str,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Mark held codes using already validated T-open prices."""
        marks: dict[str, float] = {}
        fallbacks: dict[str, str] = {}
        for code in positions:
            if code in primary:
                marks[code] = primary[code]
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
    "SettlementEconomics",
    "accounting_schema_manifest",
    "settlement_economics",
]

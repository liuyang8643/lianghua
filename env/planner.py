"""Causal, single-day stock selection and rebalance planning.

The planner intentionally receives only values that may be known at the
decision open.  In particular there is no slot for current-day close, high,
low, volume, or amount.  Factor ranks and validity masks are computed outside
this module once and selected here for the current row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from env.contracts import AccountState, DayConfig, OrderPlan, RebalanceMode
from env.fees import DEFAULT_FEE_SCHEDULE
from env.quantity import (
    floor_buy_quantity,
    floor_partial_sell_quantity,
    round_buy_quantity,
)


_PRICE_EPS = 0.001

_IPO_44_START = date(2014, 1, 1)
_KCB_OPEN = date(2019, 7, 22)
_CYB_REG = date(2020, 8, 24)
_MB_REG = date(2023, 4, 10)
_MB_ST_10_START = date(2026, 7, 6)

_BOARD_MAIN = 0
_BOARD_CYB = 1
_BOARD_KCB = 2
_BOARD_BJ = 3


def _as_float_row(name: str, values, size: int) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return np.ascontiguousarray(result)


def _as_bool_row(name: str, values, size: int) -> NDArray[np.bool_]:
    result = np.asarray(values, dtype=np.bool_)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class DayMarketData:
    """The complete causal market input for one T-open decision.

    ``listing_age`` is measured in trading rows: ``-1`` means not listed yet,
    ``0`` is the first listed row.  If omitted, a positive open/pre-close is
    conservatively treated as an established listing; callers that need IPO
    first-five-day rules must provide the exact age.
    """

    decision_date: str
    stock_codes: Sequence[str]
    factor_ranks: Mapping[str, NDArray[np.floating]]
    factor_validity: Mapping[str, NDArray[np.bool_]]
    filter_masks: Mapping[str, NDArray[np.bool_]]
    open_prices: NDArray[np.floating]
    preclose_prices: NDArray[np.floating]
    issue_prices: NDArray[np.floating]
    st_mask: NDArray[np.bool_]
    listing_age: NDArray[np.integer] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.decision_date)
        except ValueError as exc:
            raise ValueError("decision_date must be ISO YYYY-MM-DD") from exc

        codes = (
            self.stock_codes
            if isinstance(self.stock_codes, tuple)
            and all(isinstance(code, str) for code in self.stock_codes)
            else tuple(str(code) for code in self.stock_codes)
        )
        if not codes or len(set(codes)) != len(codes):
            raise ValueError("stock_codes must be non-empty and unique")
        size = len(codes)
        opens = _as_float_row("open_prices", self.open_prices, size)
        precloses = _as_float_row("preclose_prices", self.preclose_prices, size)
        issues = _as_float_row("issue_prices", self.issue_prices, size)
        st = _as_bool_row("st_mask", self.st_mask, size)

        if set(self.factor_ranks) != set(self.factor_validity):
            raise ValueError("factor_ranks and factor_validity must have identical keys")
        ranks: dict[str, NDArray[np.float64]] = {}
        validity: dict[str, NDArray[np.bool_]] = {}
        for name in self.factor_ranks:
            rank = _as_float_row(f"factor_ranks[{name!r}]", self.factor_ranks[name], size)
            valid = _as_bool_row(
                f"factor_validity[{name!r}]", self.factor_validity[name], size
            )
            if not np.isfinite(rank[valid]).all():
                raise ValueError(f"valid factor ranks for {name!r} must be finite")
            ranks[str(name)] = rank
            validity[str(name)] = valid

        filters = {
            str(name): _as_bool_row(f"filter_masks[{name!r}]", values, size)
            for name, values in self.filter_masks.items()
        }
        if self.listing_age is None:
            established = (
                (np.isfinite(opens) & (opens > 0.0))
                | (np.isfinite(precloses) & (precloses > 0.0))
            )
            listing_age = np.where(established, 5, -1).astype(np.int32)
        else:
            listing_age = np.asarray(self.listing_age, dtype=np.int32)
            if listing_age.shape != (size,):
                raise ValueError(f"listing_age must have shape ({size},)")
            listing_age = np.ascontiguousarray(listing_age)

        object.__setattr__(self, "stock_codes", codes)
        object.__setattr__(self, "factor_ranks", ranks)
        object.__setattr__(self, "factor_validity", validity)
        object.__setattr__(self, "filter_masks", filters)
        object.__setattr__(self, "open_prices", opens)
        object.__setattr__(self, "preclose_prices", precloses)
        object.__setattr__(self, "issue_prices", issues)
        object.__setattr__(self, "st_mask", st)
        object.__setattr__(self, "listing_age", listing_age)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _bare_code(code: str) -> str:
    return code.split(".", 1)[0]


def _board_type(code: str) -> int:
    bare = _bare_code(code)
    if bare.startswith(("300", "301")):
        return _BOARD_CYB
    if bare.startswith("688"):
        return _BOARD_KCB
    if bare.startswith(("43", "83", "87", "92")):
        return _BOARD_BJ
    return _BOARD_MAIN


def _ordinary_limit_ratio(code: str) -> float:
    board = _board_type(code)
    if board in (_BOARD_CYB, _BOARD_KCB):
        return 0.20
    if board == _BOARD_BJ:
        return 0.30
    return 0.10


def _floor_price(values: float) -> float:
    return math.floor(values * 100.0 + 1e-9) / 100.0


def _ceil_price(values: float) -> float:
    return math.ceil(values * 100.0 - 1e-9) / 100.0


class DayPlanner:
    """Produce one deterministic order plan from a causal T-open snapshot."""

    def __init__(self, diagnostics: str = "minimal") -> None:
        if diagnostics not in ("minimal", "full"):
            raise ValueError("diagnostics must be 'minimal' or 'full'")
        self.diagnostics_mode = diagnostics
        self._cached_stock_codes: tuple[str, ...] | None = None
        self._cached_code_to_idx: dict[str, int] = {}
        self._cached_board_types = np.empty(0, dtype=np.int8)

    def _universe_metadata(
        self, stock_codes: tuple[str, ...]
    ) -> tuple[dict[str, int], NDArray[np.int8], bool]:
        cache_reused = False
        if self._cached_stock_codes is stock_codes:
            cache_reused = True
        elif self._cached_stock_codes == stock_codes:
            # Equivalent daily tuples reuse the O(N) Python structures. Tuple
            # comparison itself runs in C and avoids rebuilding ~5k entries.
            self._cached_stock_codes = stock_codes
            cache_reused = True
        else:
            self._cached_stock_codes = stock_codes
            self._cached_code_to_idx = {
                code: idx for idx, code in enumerate(stock_codes)
            }
            self._cached_board_types = np.fromiter(
                (_board_type(code) for code in stock_codes),
                dtype=np.int8,
                count=len(stock_codes),
            )
        return self._cached_code_to_idx, self._cached_board_types, cache_reused

    def plan(
        self,
        market: DayMarketData,
        account: AccountState,
        config: DayConfig,
    ) -> OrderPlan:
        codes = market.stock_codes
        size = len(codes)
        code_to_idx, board_types, universe_cache_reused = self._universe_metadata(codes)
        self._validate_config_inputs(market, config)

        enabled_factors = [
            name for name, enabled in config.factor_enabled.items() if enabled
        ]
        listed = np.asarray(market.listing_age >= 0, dtype=np.bool_)
        valid_open = np.isfinite(market.open_prices) & (market.open_prices > 0.0)
        eligible = listed & valid_open
        market_rejection_counts = {
            "not_listed": int((~listed).sum()),
            "suspended_or_missing_open": int((listed & ~valid_open).sum()),
        }
        market_rejection_counts = {
            reason: count
            for reason, count in market_rejection_counts.items()
            if count
        }
        market_rejections = (
            {
                codes[idx]: (
                    "not_listed" if not listed[idx] else "suspended_or_missing_open"
                )
                for idx in np.flatnonzero(~(listed & valid_open))
            }
            if self.diagnostics_mode == "full"
            else {}
        )
        factor_rejected: dict[str, int] = {}
        for name in enabled_factors:
            before = int(eligible.sum())
            eligible &= market.factor_validity[name]
            factor_rejected[name] = before - int(eligible.sum())

        filter_rejected: dict[str, int] = {}
        for name, enabled in config.filter_flags.items():
            if not enabled:
                continue
            before = int(eligible.sum())
            eligible &= market.filter_masks[name]
            filter_rejected[name] = before - int(eligible.sum())

        scores = np.zeros(size, dtype=np.float64)
        for name in enabled_factors:
            scores += market.factor_ranks[name] * float(config.factor_weights[name])
        scores[~eligible] = -np.inf
        finite_indices = np.flatnonzero(np.isfinite(scores))
        # Stable input-order tie breaking is deterministic and matches the
        # historical stock-universe ordering used by the legacy planner.
        ranked_indices = finite_indices[
            np.argsort(-scores[finite_indices], kind="stable")
        ]
        buy_legality: dict[str, str] = {}
        buy_legality_counts: dict[str, int] = {}
        sell_legality: dict[str, str] = {}
        buy_targets: list[str] = []
        legal_keep_candidates: list[str] = []
        for idx in ranked_indices:
            ok, reason = self._trade_legality(
                market,
                int(idx),
                is_buy=True,
                config=config,
                board=int(board_types[idx]),
            )
            if ok:
                code = codes[idx]
                if len(buy_targets) < config.buy_n:
                    buy_targets.append(code)
                if len(legal_keep_candidates) < config.sell_m:
                    legal_keep_candidates.append(code)
                if (
                    len(buy_targets) >= config.buy_n
                    and len(legal_keep_candidates) >= config.sell_m
                ):
                    break
            else:
                buy_legality_counts[reason] = buy_legality_counts.get(reason, 0) + 1
                if self.diagnostics_mode == "full":
                    buy_legality[codes[idx]] = reason

        # Match the established retention contract: prefer buy-legal names,
        # then fill a short sell-M list from raw rank without applying buy
        # legality (sell legality is checked independently below).
        keep_codes = list(legal_keep_candidates)
        keep_seen = set(keep_codes)
        for idx in ranked_indices:
            if len(keep_codes) >= config.sell_m:
                break
            code = codes[idx]
            if code not in keep_seen:
                keep_codes.append(code)
                keep_seen.add(code)

        valuation_prices: dict[str, float] = {}
        valuation_fallbacks: dict[str, str] = {}
        positions = {
            str(code): int(quantity)
            for code, quantity in account.positions.items()
            if int(quantity) > 0
        }
        for code in positions:
            idx = code_to_idx.get(code)
            if idx is not None and valid_open[idx]:
                valuation_prices[code] = float(market.open_prices[idx])
            else:
                fallback = float(account.last_prices.get(code, math.nan))
                if math.isfinite(fallback) and fallback > 0.0:
                    valuation_prices[code] = fallback
                    valuation_fallbacks[code] = "account.last_prices"
                else:
                    valuation_fallbacks[code] = "missing"

        balance_sheet_nav = float(account.cash) + sum(
            positions[code] * valuation_prices[code]
            for code in positions
            if code in valuation_prices
        )
        missing_valuation_marks = tuple(
            code for code in positions if code not in valuation_prices
        )
        if missing_valuation_marks:
            raise ValueError(
                "cannot compute open[T] account equity for positions without a mark: "
                f"{missing_valuation_marks}"
            )
        total_equity = balance_sheet_nav
        nav_source = "open[T]_mark"
        if not math.isfinite(total_equity) or total_equity < 0.0:
            raise ValueError("account pretrade NAV must be finite and non-negative")

        reserve_ratio = max(
            (_ordinary_limit_ratio(code) for code in buy_targets), default=0.0
        )
        base_target = (
            total_equity * config.target_exposure / (config.buy_n + reserve_ratio)
        )
        prices = {
            code: float(market.open_prices[code_to_idx[code]])
            for code in set(positions) | set(buy_targets) | set(keep_codes)
            if code in code_to_idx and valid_open[code_to_idx[code]]
        }
        limit_prices = {
            code: self._freeze_price(
                code,
                prices[code],
                float(market.preclose_prices[code_to_idx[code]]),
            )
            for code in buy_targets
            if code in prices
        }

        diagnostics: dict[str, object] = {
            "enabled_factors": tuple(enabled_factors),
            "enabled_filters": tuple(
                name for name, enabled in config.filter_flags.items() if enabled
            ),
            "factor_rejected": factor_rejected,
            "filter_rejected": filter_rejected,
            "eligible_count": int(eligible.sum()),
            "buy_n_stocks": tuple(buy_targets),
            "sell_m_stocks": tuple(keep_codes),
            "sell_legality_rejections": sell_legality,
            "prices": prices,
            "limit_prices": limit_prices,
            "valuation_fallbacks": valuation_fallbacks,
            "balance_sheet_nav": balance_sheet_nav,
            "total_equity": total_equity,
            "nav_source": nav_source,
            "cached_account_nav": float(account.nav),
            "cached_account_nav_difference": float(account.nav) - total_equity,
            "base_target": base_target,
            "reserve_ratio": reserve_ratio,
            "universe_cache_reused": universe_cache_reused,
        }
        diagnostics["market_rejection_counts"] = market_rejection_counts
        diagnostics["buy_legality_rejection_counts"] = dict(
            sorted(buy_legality_counts.items())
        )
        if self.diagnostics_mode == "full":
            diagnostics.update(
                {
                    "market_rejections": market_rejections,
                    "ranked_stocks": tuple(codes[idx] for idx in ranked_indices),
                    "buy_legality_rejections": buy_legality,
                    "final_scores": scores.copy(),
                }
            )
        if not config.rebalance_now:
            diagnostics["no_trade_reason"] = "rebalance_now_false"
            return OrderPlan(
                decision_date=market.decision_date,
                day_config=config,
                diagnostics=diagnostics,
            )

        sellable = {
            str(code): max(0, int(value))
            for code, value in account.sellable_positions.items()
        }
        sellable_ok: set[str] = set()
        for code in positions:
            idx = code_to_idx.get(code)
            if idx is None:
                sell_legality[code] = "outside_market_universe"
                continue
            ok, reason = self._trade_legality(
                market,
                idx,
                is_buy=False,
                config=config,
                board=int(board_types[idx]),
            )
            if ok:
                sellable_ok.add(code)
            else:
                sell_legality[code] = reason

        position_values = {
            code: positions[code] * valuation_prices[code]
            for code in positions
            if code in valuation_prices
        }
        if config.rebalance_mode is RebalanceMode.EQUALIZE:
            sell_orders, buy_orders, skip_reasons = self._equalize_orders(
                market=market,
                account_cash=float(account.cash),
                positions=positions,
                sellable=sellable,
                position_values=position_values,
                prices=prices,
                limit_prices=limit_prices,
                buy_targets=buy_targets,
                keep_codes=keep_codes,
                sellable_ok=sellable_ok,
                base_target=base_target,
                band=config.rebalance_band_pct,
            )
        else:
            sell_orders, buy_orders, skip_reasons = self._replacement_orders(
                account_cash=float(account.cash),
                positions=positions,
                sellable=sellable,
                position_values=position_values,
                prices=prices,
                limit_prices=limit_prices,
                buy_targets=buy_targets,
                keep_codes=keep_codes,
                sellable_ok=sellable_ok,
                desired_invested=total_equity * config.target_exposure,
            )

        diagnostics["sell_legality_rejections"] = sell_legality
        diagnostics["skip_reasons"] = skip_reasons
        diagnostics["planned_sell_notional"] = sum(
            (positions.get(code, 0) if quantity < 0 else quantity)
            * prices.get(code, 0.0)
            for code, quantity in sell_orders
        )
        diagnostics["planned_buy_notional"] = sum(
            quantity * prices.get(code, 0.0) for code, quantity in buy_orders.items()
        )
        return OrderPlan(
            decision_date=market.decision_date,
            sell_orders=tuple(sell_orders),
            buy_orders=buy_orders,
            day_config=config,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _validate_config_inputs(market: DayMarketData, config: DayConfig) -> None:
        missing_factors = set(config.factor_weights) - set(market.factor_ranks)
        if missing_factors:
            raise ValueError(f"market is missing factors: {sorted(missing_factors)}")
        missing_filters = {
            name
            for name, enabled in config.filter_flags.items()
            if enabled and name not in market.filter_masks
        }
        if missing_filters:
            raise ValueError(f"market is missing enabled filters: {sorted(missing_filters)}")

    @staticmethod
    def _freeze_price(code: str, open_price: float, preclose: float) -> float:
        base = preclose
        ratio = _ordinary_limit_ratio(code)
        if (
            not math.isfinite(base)
            or base <= 0.0
            or abs(open_price - base) / base > ratio
        ):
            base = open_price
        return base * (1.0 + ratio)

    @staticmethod
    def _trade_legality(
        market: DayMarketData,
        idx: int,
        *,
        is_buy: bool,
        config: DayConfig,
        board: int | None = None,
    ) -> tuple[bool, str]:
        age = int(market.listing_age[idx])
        if age < 0:
            return False, "not_listed"
        open_price = float(market.open_prices[idx])
        if not math.isfinite(open_price) or open_price <= 0.0:
            return False, "suspended_or_missing_open"

        code = market.stock_codes[idx]
        board = _board_type(code) if board is None else board
        decision_day = date.fromisoformat(market.decision_date)
        ratio = _ordinary_limit_ratio(code)
        if board == _BOARD_CYB and decision_day < _CYB_REG:
            ratio = 0.10
        if bool(market.st_mask[idx]):
            if board == _BOARD_CYB:
                ratio = 0.05 if decision_day < _CYB_REG else 0.20
            elif board == _BOARD_KCB:
                ratio = 0.20
            elif board == _BOARD_BJ:
                ratio = 0.30
            else:
                ratio = 0.05 if decision_day < _MB_ST_10_START else 0.10

        first_day = age == 0
        exempt = board == _BOARD_BJ and first_day
        if board == _BOARD_KCB and decision_day >= _KCB_OPEN and 0 <= age <= 4:
            exempt = True
        if board == _BOARD_CYB:
            if decision_day >= _CYB_REG and 0 <= age <= 4:
                exempt = True
            elif decision_day < _IPO_44_START and first_day:
                exempt = True
        if board == _BOARD_MAIN:
            if decision_day >= _MB_REG and 0 <= age <= 4:
                exempt = True
            elif decision_day < _IPO_44_START and first_day:
                exempt = True

        old_ipo_first = first_day and decision_day >= _IPO_44_START and not exempt
        preclose = float(market.preclose_prices[idx])
        if first_day:
            issue = float(market.issue_prices[idx])
            if math.isfinite(issue) and issue > 0.0:
                preclose = issue
        if exempt:
            return True, "ok_no_daily_limit"
        if not math.isfinite(preclose) or preclose <= 0.0:
            return False, "missing_preclose"

        if old_ipo_first:
            ratio = 0.44
        up_limit = _floor_price(preclose * (1.0 + ratio))
        down_limit = _ceil_price(preclose * (1.0 - ratio))
        if is_buy:
            if open_price >= up_limit - _PRICE_EPS:
                return False, "limit_up"
            if old_ipo_first:
                ipo_open_limit = _floor_price(preclose * 1.20)
                if open_price >= ipo_open_limit - _PRICE_EPS:
                    return False, "ipo_open_limit"
            return True, "ok"
        if open_price <= down_limit + _PRICE_EPS:
            return False, "limit_down"
        if config.limit_up_protection and open_price >= up_limit - _PRICE_EPS:
            return False, "limit_up_protected"
        return True, "ok"

    @staticmethod
    def _ordered_position_codes(
        positions: Mapping[str, int], preferred: Sequence[str], stock_codes: Sequence[str]
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for code in preferred:
            if code in positions and code not in seen:
                result.append(code)
                seen.add(code)
        for code in stock_codes:
            if code in positions and code not in seen:
                result.append(code)
                seen.add(code)
        for code in sorted(set(positions) - seen):
            result.append(code)
        return result

    @staticmethod
    def _affordable_buy_shares(
        code: str,
        cash: float,
        unit_price: float,
    ) -> int:
        if cash <= 0.0 or unit_price <= 0.0:
            return 0
        low, high = 0, int(cash / unit_price) + 1
        while low + 1 < high:
            middle = (low + high) // 2
            if DEFAULT_FEE_SCHEDULE.buy_total_cost(middle * unit_price) <= cash:
                low = middle
            else:
                high = middle
        return floor_buy_quantity(code, low)

    def _try_buy(
        self,
        *,
        code: str,
        requested: int,
        prices: Mapping[str, float],
        limit_prices: Mapping[str, float],
        cash_sim: float,
        buy_orders: dict[str, int],
        skip_reasons: dict[str, str],
    ) -> float:
        quantity = round_buy_quantity(code, requested)
        if quantity <= 0:
            skip_reasons[code] = "below_lot_or_band"
            return cash_sim
        budget_price = max(prices[code], limit_prices.get(code, prices[code]))
        affordable = self._affordable_buy_shares(
            code,
            cash_sim,
            budget_price,
        )
        quantity = min(quantity, affordable)
        if quantity <= 0:
            skip_reasons[code] = "insufficient_frozen_cash"
            return cash_sim
        buy_orders[code] = quantity
        return cash_sim - DEFAULT_FEE_SCHEDULE.buy_total_cost(
            quantity * prices[code]
        )

    def _equalize_orders(
        self,
        *,
        market: DayMarketData,
        account_cash: float,
        positions: Mapping[str, int],
        sellable: Mapping[str, int],
        position_values: Mapping[str, float],
        prices: Mapping[str, float],
        limit_prices: Mapping[str, float],
        buy_targets: Sequence[str],
        keep_codes: Sequence[str],
        sellable_ok: set[str],
        base_target: float,
        band: float,
    ) -> tuple[list[tuple[str, int]], dict[str, int], dict[str, str]]:
        target_set, keep_set = set(buy_targets), set(keep_codes)
        sell_orders: list[tuple[str, int]] = []
        cash_sim = account_cash
        sell_sequence = self._ordered_position_codes(
            positions, buy_targets, market.stock_codes
        )
        for code in sell_sequence:
            if code not in prices or code not in sellable_ok:
                continue
            current_value = position_values[code]
            target = base_target if code in target_set else (
                current_value if code in keep_set else 0.0
            )
            if current_value <= target * (1.0 + band):
                continue
            available = min(int(positions[code]), max(0, int(sellable.get(code, 0))))
            if available <= 0:
                continue
            if target == 0.0 and available == int(positions[code]):
                quantity = -1
            else:
                quantity = min(
                    floor_partial_sell_quantity(
                        code,
                        (current_value - target) / prices[code],
                    ),
                    floor_partial_sell_quantity(code, available),
                )
            if quantity == 0:
                continue
            sell_orders.append((code, quantity))
            executed_quantity = available if quantity < 0 else quantity
            cash_sim += DEFAULT_FEE_SCHEDULE.sell_net_proceeds(
                executed_quantity * prices[code]
            )

        buy_orders: dict[str, int] = {}
        skip_reasons: dict[str, str] = {}
        for code in buy_targets:
            if code not in prices:
                skip_reasons[code] = "missing_open"
                continue
            current_value = position_values.get(code, 0.0)
            if current_value >= base_target * (1.0 - band):
                skip_reasons[code] = "within_or_above_target_band"
                continue
            cash_sim = self._try_buy(
                code=code,
                requested=int((base_target - current_value) / prices[code]),
                prices=prices,
                limit_prices=limit_prices,
                cash_sim=cash_sim,
                buy_orders=buy_orders,
                skip_reasons=skip_reasons,
            )
        return sell_orders, buy_orders, skip_reasons

    def _replacement_orders(
        self,
        *,
        account_cash: float,
        positions: Mapping[str, int],
        sellable: Mapping[str, int],
        position_values: Mapping[str, float],
        prices: Mapping[str, float],
        limit_prices: Mapping[str, float],
        buy_targets: Sequence[str],
        keep_codes: Sequence[str],
        sellable_ok: set[str],
        desired_invested: float,
    ) -> tuple[list[tuple[str, int]], dict[str, int], dict[str, str]]:
        keep_set = set(keep_codes)
        sell_orders: list[tuple[str, int]] = []
        cash_sim = account_cash
        remaining_values = dict(position_values)
        for code in positions:
            if code in keep_set or code not in prices or code not in sellable_ok:
                continue
            available = min(int(positions[code]), max(0, int(sellable.get(code, 0))))
            if available <= 0:
                continue
            quantity = (
                -1
                if available == int(positions[code])
                else floor_partial_sell_quantity(code, available)
            )
            if quantity == 0:
                continue
            sell_orders.append((code, quantity))
            executed_quantity = available if quantity < 0 else quantity
            cash_sim += DEFAULT_FEE_SCHEDULE.sell_net_proceeds(
                executed_quantity * prices[code]
            )
            remaining_values[code] = max(
                0.0,
                remaining_values.get(code, 0.0)
                - executed_quantity * prices[code],
            )

        new_codes = [
            code for code in buy_targets if code not in positions and code in prices
        ]
        invested_after_sells = sum(remaining_values.values())
        desired_new = max(0.0, desired_invested - invested_after_sells)
        buy_budget = min(cash_sim, desired_new)
        cash_per_new = buy_budget / len(new_codes) if new_codes else 0.0
        buy_orders: dict[str, int] = {}
        skip_reasons: dict[str, str] = {}
        for code in buy_targets:
            if code in positions:
                skip_reasons[code] = "replace_only_existing_position"
            elif code not in prices:
                skip_reasons[code] = "missing_open"
        for code in new_codes:
            cash_sim = self._try_buy(
                code=code,
                requested=int(cash_per_new / prices[code]),
                prices=prices,
                limit_prices=limit_prices,
                cash_sim=cash_sim,
                buy_orders=buy_orders,
                skip_reasons=skip_reasons,
            )
        if invested_after_sells > desired_invested:
            skip_reasons["_target_exposure"] = (
                "replace_only_does_not_trim_retained_positions"
            )
        return sell_orders, buy_orders, skip_reasons


__all__ = ["DayMarketData", "DayPlanner"]

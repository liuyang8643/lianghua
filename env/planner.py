"""Causal, single-day stock selection and rebalance planning.

The planner intentionally receives only values that may be known at the
decision open.  In particular there is no slot for current-day close, high,
low, volume, or amount.  Factor ranks and validity masks are computed outside
this module once and selected here for the current row.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numba import njit, literal_unroll
from numpy.typing import NDArray

from env.legality import (
    classify_board_types,
    evaluate_trade_legality,
    legality_reason_text,
    ordinary_limit_ratios,
    TradeLegalityResult,
)
from env.contracts import AccountState, DayConfig, OrderPlan, PortfolioExecutionSpec
from env.fees import (
    DEFAULT_FEE_SCHEDULE, FeeSchedule,
    affordable_buy_shares, buy_total_cost, sell_net_proceeds,
)
from env.scoring import FactorScoreRow, score_factor_ranks
from env.prefilter import PrefilterUniverse, rank_scored_universe_indices
from env.quantity import (
    minimum_buy_quantity,
    buy_quantity_step,
    floor_quantity,
)


_EQUALIZE_INPUT_DTYPE = np.dtype([
    ("quantity", np.int64), ("available", np.int64),
    ("values", np.float64), ("price", np.float64), ("limits", np.float64),
    ("minimum", np.int64), ("step", np.int64),
    ("target", np.bool_), ("keep", np.bool_), ("sell", np.bool_),
], align=True)


def _as_float_row(name: str, values, size: int) -> NDArray[np.floating]:
    result = np.asarray(values)
    if result.dtype not in (np.dtype('float32'), np.dtype('float64')):
        result = np.asarray(result, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return np.ascontiguousarray(result)


def _as_bool_row(name: str, values, size: int) -> NDArray[np.bool_]:
    result = np.asarray(values, dtype=np.bool_)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return np.ascontiguousarray(result)


@lru_cache(maxsize=8)
def _sealed_board_types(codes: tuple[str, ...]) -> NDArray[np.int8]:
    values = classify_board_types(codes)
    values.flags.writeable = False
    return values


@njit(cache=True, fastmath=False, parallel=False)
def _eligibility_rows(listing_age, delisted, opens, candidates, factor_masks, filter_masks):
    """Single serial scan for eligibility and all rejection counters."""
    size = len(opens)
    listed = np.empty(size, dtype=np.bool_)
    valid_open = np.empty(size, dtype=np.bool_)
    factor_eligible = np.empty(size, dtype=np.bool_)
    eligible = np.empty(size, dtype=np.bool_)
    counts = np.zeros(4, dtype=np.int64)
    missing = np.zeros(len(factor_masks), dtype=np.int64)
    rejected = np.zeros(len(filter_masks), dtype=np.int64)
    for stock in range(size):
        listed[stock] = listing_age[stock] >= 0
        valid_open[stock] = np.isfinite(opens[stock]) and opens[stock] > 0.0
        base = listed[stock] and not delisted[stock] and valid_open[stock]
        factor_eligible[stock] = base and candidates[stock]
        counts[0] += not listed[stock]
        counts[1] += listed[stock] and delisted[stock]
        counts[2] += listed[stock] and not delisted[stock] and not valid_open[stock]
        counts[3] += base and not candidates[stock]
        allowed = factor_eligible[stock]
        if allowed:
            if len(factor_masks):
                factor = 0
                for factor_mask in literal_unroll(factor_masks):
                    missing[factor] += not factor_mask[stock]
                    factor += 1
            for filt in range(len(filter_masks)):
                rejected[filt] += allowed and not filter_masks[filt, stock]
                allowed = allowed and filter_masks[filt, stock]
        eligible[stock] = allowed
    return listed, valid_open, factor_eligible, eligible, counts, missing, rejected


@njit(cache=True, fastmath=False, parallel=False)
def _try_buy_quantity(requested, price, budget_price, cash, minimum, step, fees):
    quantity = floor_quantity(requested, minimum, step)
    if quantity <= 0:
        return 0, cash, 1
    affordable = floor_quantity(affordable_buy_shares(cash, budget_price, fees), minimum, step)
    quantity = min(quantity, affordable)
    if quantity <= 0:
        return 0, cash, 2
    return quantity, cash - buy_total_cost(quantity * price, fees), 0


@njit(cache=True, fastmath=False, parallel=False)
def _rebalance_quantities(positions, sellable, values, prices, budget_prices, buy_indices,
                        target_mask, keep_mask, sell_allowed, minimums, steps, cash, targets, band, fees):
    """Canonical sell-first target allocation and mandatory cash sweep, serial."""
    count = len(positions)
    sells = np.zeros(count, dtype=np.int64)
    buys = np.zeros(count, dtype=np.int64)
    buy_order = np.full(count, -1, dtype=np.int64)
    skip = np.zeros(count, dtype=np.int8)
    post_values = values.copy()
    order_counter = 0
    for index in range(count):
        if not np.isfinite(prices[index]) or not sell_allowed[index]:
            continue
        current = values[index]
        goal = targets[index] if target_mask[index] else (current if keep_mask[index] else 0.0)
        if current <= goal * (1.0 + band):
            continue
        available = min(positions[index], max(0, sellable[index]))
        if available <= 0:
            continue
        if goal == 0.0 and available == positions[index]:
            quantity = available
        else:
            quantity = min(floor_quantity((current - goal) / prices[index], minimums[index], steps[index]),
                           floor_quantity(available, minimums[index], steps[index]))
        if quantity == 0:
            continue
        sells[index] = quantity
        cash += sell_net_proceeds(quantity * prices[index], fees)
        post_values[index] = max(0.0, post_values[index] - quantity * prices[index])
    for index in buy_indices:
        target = targets[index]
        if not np.isfinite(prices[index]):
            skip[index] = 3
            continue
        if post_values[index] >= target * (1.0 - band):
            skip[index] = 4
            continue
        quantity, cash, reason = _try_buy_quantity(
            int((target - post_values[index]) / prices[index]), prices[index], budget_prices[index],
            cash, minimums[index], steps[index], fees,
        )
        skip[index] = reason
        if quantity:
            buy_order[index] = order_counter
            order_counter += 1
            buys[index] += quantity
    progress = True
    while progress:
        progress = False
        for index in buy_indices:
            target = targets[index]
            if not np.isfinite(prices[index]):
                continue
            planned = post_values[index] + buys[index] * prices[index]
            shortfall = target - planned
            if shortfall <= 0.0:
                continue
            quantity, cash, reason = _try_buy_quantity(
                int(shortfall / prices[index]), prices[index], budget_prices[index], cash,
                minimums[index], steps[index], fees,
            )
            skip[index] = reason
            if quantity:
                if buys[index] == 0:
                    buy_order[index] = order_counter
                    order_counter += 1
                buys[index] += quantity
                progress = True
    cheapest = 0.0
    has_increment = False
    for index in buy_indices:
        target = targets[index]
        if not np.isfinite(prices[index]):
            continue
        planned = post_values[index] + buys[index] * prices[index]
        remaining = int(max(0.0, target - planned) / prices[index])
        if floor_quantity(remaining, minimums[index], steps[index]) < minimums[index]:
            continue
        next_cost = buy_total_cost(minimums[index] * budget_prices[index], fees)
        cheapest = min(cheapest, next_cost) if has_increment else next_cost
        has_increment = True
    return sells, buys, buy_order, skip, cash, cheapest, has_increment


@dataclass(frozen=True)
class DayMarketData:
    """The complete causal market input for one T-open decision.

    ``listing_age`` is measured on the full runtime trading axis: ``-1`` means
    not listed yet and ``0`` is the first listed row. It is mandatory; a
    sliced panel may not infer IPO age from its own first row.
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
    delisted_mask: NDArray[np.bool_]
    listing_age: NDArray[np.integer]
    candidate_mask: NDArray[np.bool_] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    _sealed: bool = field(default=False, init=False, repr=False, compare=False)
    _legality_cache: dict[bool, TradeLegalityResult] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _factor_score_row: FactorScoreRow | None = field(default=None, init=False, repr=False, compare=False)

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
        delisted = _as_bool_row("delisted_mask", self.delisted_mask, size)
        candidates = (
            np.ones(size, dtype=np.bool_)
            if self.candidate_mask is None
            else _as_bool_row("candidate_mask", self.candidate_mask, size)
        )

        if set(self.factor_ranks) != set(self.factor_validity):
            raise ValueError("factor_ranks and factor_validity must have identical keys")
        ranks: dict[str, NDArray[np.float64]] = {}
        validity: dict[str, NDArray[np.bool_]] = {}
        for name in self.factor_ranks:
            # FactorBatch already owns a contiguous numeric cache. Preserve its
            # dtype: expanding every daily row to float64 in each worker would
            # turn shared factors into gigabytes of private copies. Scoring
            # owns the float64 arithmetic conversion.
            rank = np.asarray(self.factor_ranks[name])
            if rank.dtype not in (np.dtype('float32'), np.dtype('float64')):
                rank = np.asarray(rank, dtype=np.float64)
            if rank.shape != (size,):
                raise ValueError(f"factor_ranks[{name!r}] must have shape ({size},)")
            rank = np.ascontiguousarray(rank)
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
        object.__setattr__(self, "delisted_mask", delisted)
        object.__setattr__(self, "listing_age", listing_age)
        object.__setattr__(self, "candidate_mask", candidates)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def seal(self, *, borrow_readonly: bool = False) -> "DayMarketData":
        """Freeze a decision snapshot so all policies can reuse it.

        Public callers get owned arrays. PreparedEpisode may explicitly borrow
        readonly arrays whose lifetime/immutability are owned by its sealed
        RuntimeSlice/FactorBatch, including shared-memory-backed snapshots.
        A readonly view alone is not proof that its external buffer is immutable.
        """
        result = copy(self)
        def frozen(values):
            owner = values
            aliased_writable = values.flags.writeable
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
                aliased_writable |= owner.flags.writeable
            array = values.copy() if aliased_writable or not borrow_readonly else values.view()
            array.flags.writeable = False
            return array
        for name in ("open_prices", "preclose_prices", "issue_prices", "st_mask",
                     "delisted_mask", "listing_age", "candidate_mask"):
            object.__setattr__(result, name, frozen(getattr(self, name)))
        for name in ("factor_ranks", "factor_validity", "filter_masks"):
            object.__setattr__(result, name, MappingProxyType({
                key: frozen(values) for key, values in getattr(self, name).items()
            }))
        object.__setattr__(result, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(result, "_legality_cache", {})
        object.__setattr__(result, "_factor_score_row", FactorScoreRow(
            result.factor_ranks, result.factor_validity, len(result.stock_codes),
        ))
        object.__setattr__(result, "_sealed", True)
        return result

    def factor_scores(self, config: DayConfig) -> NDArray[np.float64]:
        if self._sealed:
            return self._factor_score_row.score(config)
        return score_factor_ranks(self.factor_ranks, self.factor_validity, config, len(self.stock_codes))

    def trade_legality(self, limit_up_protection: bool) -> TradeLegalityResult:
        """Reuse policy-independent legality only for a sealed snapshot."""
        if self._sealed and limit_up_protection in self._legality_cache:
            return self._legality_cache[limit_up_protection]
        result = evaluate_trade_legality(
            decision_date=self.decision_date, stock_codes=self.stock_codes,
            listing_age=self.listing_age, open_prices=self.open_prices,
            preclose_prices=self.preclose_prices, issue_prices=self.issue_prices,
            st_mask=self.st_mask, delisted_mask=self.delisted_mask,
            limit_up_protection=limit_up_protection,
            precomputed_board_types=_sealed_board_types(self.stock_codes),
        )
        if self._sealed:
            for values in vars(result).values():
                if isinstance(values, np.ndarray):
                    values.flags.writeable = False
            self._legality_cache[limit_up_protection] = result
        return result

    def with_candidate_mask(self, values: NDArray[np.bool_]) -> "DayMarketData":
        """Replace only the candidate row of this already validated snapshot.

        The independent read-only mask prevents a caller's scratch buffer from
        changing a frozen decision. Other market fields keep their validated
        storage rather than repeating all full-universe checks for a prefilter.
        """

        candidates = _as_bool_row("candidate_mask", values, len(self.stock_codes)).copy()
        candidates.flags.writeable = False
        result = copy(self)
        object.__setattr__(result, "candidate_mask", candidates)
        return result



def mark_open_positions(market,account,code_to_idx=None):
    """One open-time equity/holding mark shared by allocation and execution."""
    code_to_idx=({code:i for i,code in enumerate(market.stock_codes)} if code_to_idx is None else code_to_idx)
    positions={str(code):int(q) for code,q in account.positions.items() if int(q)>0}
    prices={};marks={};fallbacks={}
    for code in positions:
        index=code_to_idx.get(code)
        value=float(market.open_prices[index]) if index is not None else math.nan
        if math.isfinite(value) and value>0:prices[code]=value;marks[code]=value
        else:
            value=float(account.last_prices.get(code,math.nan))
            if math.isfinite(value) and value>0:marks[code]=value;fallbacks[code]='account.last_prices'
            else:fallbacks[code]='missing'
    missing=tuple(code for code in positions if code not in marks)
    if missing:raise ValueError(f'cannot compute open[T] account equity for positions without a mark: {missing}')
    values={code:positions[code]*marks[code] for code in positions}
    equity=float(account.cash)+sum(values.values())
    if not math.isfinite(equity) or equity<0:raise ValueError('account pretrade NAV must be finite and non-negative')
    return prices,marks,fallbacks,values,equity


class DayPlanner:
    """Produce one deterministic order plan from a causal T-open snapshot."""

    def __init__(
        self,
        diagnostics: str = "minimal",
        *,
        fees: FeeSchedule = DEFAULT_FEE_SCHEDULE,
    ) -> None:
        if diagnostics not in ("minimal", "full"):
            raise ValueError("diagnostics must be 'minimal' or 'full'")
        self.diagnostics_mode = diagnostics
        self.fees = fees
        self._cached_stock_codes: tuple[str, ...] | None = None
        self._cached_code_to_idx: dict[str, int] = {}
        self._cached_board_types = np.empty(0, dtype=np.int8)
        self._cached_limit_ratios = np.empty(0, dtype=np.float64)
        self._cached_universe: PrefilterUniverse | None = None

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
            self._cached_board_types = classify_board_types(stock_codes)
            self._cached_limit_ratios = ordinary_limit_ratios(self._cached_board_types)
            self._cached_universe = PrefilterUniverse(stock_codes)
        return self._cached_code_to_idx, self._cached_board_types, cache_reused

    def plan(
        self,
        market: DayMarketData,
        account: AccountState,
        config: DayConfig,
        *, target_weights: Mapping[str, float] | None = None,
        portfolio_spec: PortfolioExecutionSpec | None = None,
    ) -> OrderPlan:
        if target_weights is None and portfolio_spec is None:
            return self._plan(market,account,config)[0]
        return self._plan(market, account, config, target_weights=target_weights,portfolio_spec=portfolio_spec)[0]

    def plan_and_rank(
        self, market: DayMarketData, account: AccountState, config: DayConfig,
        universe: PrefilterUniverse,
        *, prefilter_n: int | None = None,
    ) -> tuple[OrderPlan, NDArray[np.intp]]:
        """One score calculation feeds today's plan and tomorrow's prefilter."""
        if universe.stock_codes != market.stock_codes:
            raise ValueError("prefilter universe differs from the decision market")
        if prefilter_n is not None and (type(prefilter_n) is not int or prefilter_n <= 0):
            raise ValueError("prefilter_n must be a positive int")
        plan, ranking = self._plan(market, account, config)
        return plan, ranking[:prefilter_n]

    def _plan(
        self, market: DayMarketData, account: AccountState, config: DayConfig,
        *, target_weights: Mapping[str, float] | None = None,
        portfolio_spec: PortfolioExecutionSpec | None = None,
    ) -> tuple[OrderPlan, NDArray[np.intp]]:
        codes = market.stock_codes
        if (target_weights is None)!=(portfolio_spec is None):
            raise ValueError('explicit portfolio weights require a PortfolioExecutionSpec')
        if portfolio_spec is not None and portfolio_spec.max_positions<config.buy_n:
            raise ValueError('portfolio capacity must cover the entry-ranking size')
        size = len(codes)
        code_to_idx, board_types, universe_cache_reused = self._universe_metadata(codes)
        self._validate_config_inputs(market, config)
        trade_legality = market.trade_legality(config.limit_up_protection)
        if (
            trade_legality.sell_allowed is None
            or trade_legality.buy_reason_codes is None
            or trade_legality.sell_reason_codes is None
        ):
            raise RuntimeError("planner requires complete trade legality diagnostics")

        enabled_factors = [
            name for name, enabled in config.factor_enabled.items() if enabled
        ]
        enabled_filters = [name for name, enabled in config.filter_flags.items() if enabled]
        factor_masks = tuple(market.factor_validity[name] for name in enabled_factors)
        filter_masks = np.asarray([market.filter_masks[name] for name in enabled_filters], dtype=np.bool_).reshape(-1, size)
        listed, valid_open, factor_eligible, eligible, market_counts, factor_counts, filter_counts = _eligibility_rows(
            market.listing_age, market.delisted_mask, market.open_prices, market.candidate_mask,
            factor_masks, filter_masks,
        )
        market_rejection_counts = dict(zip(
            ("not_listed", "delisted", "suspended_or_missing_open", "outside_t1_prefilter"),
            map(int, market_counts),
        ))
        market_rejection_counts = {
            reason: count
            for reason, count in market_rejection_counts.items()
            if count
        }
        market_rejections = (
            {
                codes[idx]: (
                    "not_listed"
                    if not listed[idx]
                    else (
                        "delisted"
                        if market.delisted_mask[idx]
                        else (
                            "suspended_or_missing_open"
                            if not valid_open[idx]
                            else "outside_t1_prefilter"
                        )
                    )
                )
                for idx in np.flatnonzero(
                    ~(listed & ~market.delisted_mask & valid_open & market.candidate_mask)
                )
            }
            if self.diagnostics_mode == "full"
            else {}
        )
        factor_rejected = dict(zip(enabled_factors, map(int, factor_counts)))
        filter_rejected = dict(zip(enabled_filters, map(int, filter_counts)))

        universe_scores = market.factor_scores(config)
        member = listed & ~market.delisted_mask
        full_ranking = rank_scored_universe_indices(
            self._cached_universe, universe_scores, pit_universe_mask=member,
        )
        full_members = full_ranking[member[full_ranking]]
        positions = {str(code): int(quantity) for code, quantity in account.positions.items() if int(quantity) > 0}
        held_mask = np.zeros(size, dtype=bool)
        held_mask[[code_to_idx[code] for code in positions if code in code_to_idx]] = True
        buyable = factor_eligible & trade_legality.buy_allowed
        # Current holdings remain candidates even when temporarily not buyable.
        pool = held_mask | buyable
        pool_ranking = full_ranking[pool[full_ranking]]
        preferred = held_mask[pool_ranking] | eligible[pool_ranking]
        ranked_indices = np.concatenate((pool_ranking[preferred], pool_ranking[~preferred]))
        top_codes = {codes[index] for index in ranked_indices[:config.sell_m]}
        held_order = [codes[index] for index in ranked_indices[held_mask[ranked_indices]]]
        held_order.extend(sorted(code for code in positions if code not in code_to_idx))
        replacement_limit = config.replacement_limit
        examined = list(reversed(held_order))
        exits = []
        voluntary_exits = 0
        for code in examined:
            over_target = len(positions) - len(exits) > config.buy_n
            rank_exit = code not in top_codes and voluntary_exits < replacement_limit
            if (rank_exit or over_target) and code in code_to_idx and trade_legality.sell_allowed[code_to_idx[code]] and account.sellable_positions.get(code, 0) >= positions[code]:
                exits.append(code)
                if not over_target:
                    voluntary_exits += 1
        exit_set = set(exits)
        keep_codes = [code for code in held_order if code not in exit_set]
        vacancies = max(0, config.buy_n - len(keep_codes))
        legal_ranked = ranked_indices[buyable[ranked_indices]]
        new_indices = legal_ranked[~held_mask[legal_ranked]][:vacancies]
        buy_legality = {}
        buy_legality_counts = {}
        sell_legality = {}
        for index in np.flatnonzero(factor_eligible & ~trade_legality.buy_allowed):
            reason = legality_reason_text(int(trade_legality.buy_reason_codes[index]))
            buy_legality_counts[reason] = buy_legality_counts.get(reason, 0) + 1
            if self.diagnostics_mode == 'full': buy_legality[codes[index]] = reason
        target_codes = [*keep_codes, *(codes[index] for index in new_indices)]
        if target_weights is not None:
            weights = {str(k):float(v) for k,v in target_weights.items()}
            if any(k not in code_to_idx for k in weights):
                raise ValueError('target weights contain a stock outside the market vocabulary')
            if any(not math.isfinite(v) or v < 0 or v > 1 for v in weights.values()) or sum(weights.values()) > 1+1e-8:
                raise ValueError('target weights must be finite, long-only and unlevered')
            if not portfolio_spec.allow_cash and abs(sum(weights.values())-1)>1e-8:
                raise ValueError('explicit targets must allocate one NAV; intentional cash is not full investment')
            if any(v > portfolio_spec.max_weight+1e-8 for v in weights.values()):
                raise ValueError('target weights exceed the configured concentration cap')
            target_codes = [codes[i] for i in full_ranking if weights.get(codes[i],0.) > 0]
            if len(target_codes)>portfolio_spec.max_positions:
                raise ValueError('explicit targets exceed the maximum position count')
            keep_codes = [code for code in target_codes if code in positions]
            exits = [code for code in positions if weights.get(code,0.) == 0]
            if any(code not in code_to_idx or not trade_legality.sell_allowed[code_to_idx[code]]
                   or account.sellable_positions.get(code,0)<positions[code] for code in exits):
                raise ValueError('locked or partly unsellable holdings must retain a target slot')
            if len(exits)>config.replacement_limit:
                raise ValueError('explicit targets exceed the replacement budget')
            new_indices = np.asarray([code_to_idx[code] for code in target_codes if code not in positions],dtype=np.intp)
            examined = exits
            voluntary_exits = len(exits)
            # Explicit model targets never use the legacy soft-filter backfill.
            legal_ranked = legal_ranked[eligible[legal_ranked]]
        target_mask = np.zeros(size, dtype=bool)
        target_mask[[code_to_idx[code] for code in target_codes if code in code_to_idx]] = True
        buy_indices = legal_ranked[target_mask[legal_ranked]]
        buy_targets = [codes[index] for index in buy_indices]
        filter_backfill_buy_count = int((~eligible[buy_indices]).sum())
        prices,valuation_prices,valuation_fallbacks,position_values,total_equity=mark_open_positions(market,account,code_to_idx)
        balance_sheet_nav = total_equity
        nav_source = "open[T]_mark"
        if not math.isfinite(total_equity) or total_equity < 0.0:
            raise ValueError("account pretrade NAV must be finite and non-negative")

        # ``single_buy_pct`` is a concentration control, not an exposure
        # control.  Its conditional lower bound (1 / buy_n) guarantees that
        # the target list has at least one NAV of aggregate buy capacity.
        # Affordability below still reserves the exchange freeze price and all
        # fees, so any residual cash is operationally unavoidable rather than
        # a learned cash allocation.
        base_target = total_equity * config.single_buy_pct
        # Holdings already supplied their valid T-open price above; retained
        # names are a subset of holdings, so only new buy targets need a read.
        for code in buy_targets:
            if code not in prices:
                idx = code_to_idx.get(code)
                if idx is not None and valid_open[idx]:
                    prices[code] = float(market.open_prices[idx])
        limit_prices = {
            code: self._freeze_price(
                prices[code],
                float(market.preclose_prices[code_to_idx[code]]),
                ratio=float(self._cached_limit_ratios[code_to_idx[code]]),
            )
            for code in buy_targets
            if code in prices
        }

        diagnostics: dict[str, object] = {
            "enabled_factors": tuple(enabled_factors),
            "enabled_filters": tuple(enabled_filters),
            "factor_rejected": factor_rejected,
            "factor_missing": dict(factor_rejected),
            "factor_missing_policy": "available_absolute_weight_centered_rank_normalization_no_signal_stable_tail",
            "filter_rejected": filter_rejected,
            "eligible_count": int(eligible.sum()),
            "filter_backfill_buy_count": filter_backfill_buy_count,
            "buy_n_stocks": tuple(buy_targets),
            "retained_stocks": tuple(keep_codes),
            "target_stocks": tuple(target_codes),
            "turnover_rate": config.turnover_rate,
            "replacement_limit": replacement_limit,
            "voluntary_replacement_count": voluntary_exits,
            "retention_rank_limit": config.retention_rank_limit,
            "examined_worst_holdings": tuple(examined),
            "replacement_exit_stocks": tuple(exits),
            "replacement_entry_stocks": tuple(codes[index] for index in new_indices),
            "full_rank_top_stocks": tuple(codes[index] for index in full_members[:config.buy_n]),
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
            "universe_cache_reused": universe_cache_reused,
        }
        if target_weights is not None:
            diagnostics['target_weights'] = weights
            diagnostics['allocation_mode'] = 'explicit_target_weights'
            diagnostics['portfolio_execution'] = portfolio_spec.as_dict()
            diagnostics['intentional_cash_weight'] = max(0.,1-sum(weights.values()))
            diagnostics['full_investment_required'] = not portfolio_spec.allow_cash
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
                    "final_scores": np.where(factor_eligible, universe_scores, -np.inf),
                }
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
            ok = bool(trade_legality.sell_allowed[idx])
            if ok:
                sellable_ok.add(code)
            else:
                sell_legality[code] = legality_reason_text(int(trade_legality.sell_reason_codes[idx]))

        sell_orders, buy_orders, skip_reasons, planned_cash, cheapest_next_legal_buy_cost = self._equalize_orders(
            market=market,
            account_cash=float(account.cash),
            positions=positions,
            sellable=sellable,
            position_values=position_values,
            prices=prices,
            limit_prices=limit_prices,
            buy_targets=buy_targets,
            target_codes=target_codes,
            keep_codes=keep_codes,
            sellable_ok=sellable_ok,
            base_target=base_target,
            band=config.rebalance_band_pct,
            target_values=({code:total_equity*weight for code,weight in weights.items()}
                           if target_weights is not None else None),
        )

        if not buy_targets:
            residual_reason = "no_legal_buy_target"
        elif cheapest_next_legal_buy_cost is None:
            residual_reason = "concentration_or_lot_capacity_exhausted"
        elif planned_cash + 1e-9 < cheapest_next_legal_buy_cost:
            residual_reason = "below_next_legal_frozen_lot_cost"
        else:
            residual_reason = "cash_sweep_incomplete"
        full_investment_contract_satisfied = residual_reason != "cash_sweep_incomplete"

        diagnostics["sell_legality_rejections"] = sell_legality
        diagnostics["skip_reasons"] = skip_reasons
        diagnostics["planned_post_order_cash"] = planned_cash
        diagnostics["cheapest_next_legal_buy_cost"] = cheapest_next_legal_buy_cost
        diagnostics["residual_cash_reason"] = residual_reason
        diagnostics["full_investment_contract_satisfied"] = (
            full_investment_contract_satisfied
        )
        diagnostics["planned_sell_notional"] = sum(
            quantity * prices.get(code, 0.0)
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
        ), full_ranking

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
    def _freeze_price(
        open_price: float, preclose: float, *, ratio: float
    ) -> float:
        base = preclose
        if (
            not math.isfinite(base)
            or base <= 0.0
            or abs(open_price - base) / base > ratio
        ):
            base = open_price
        return base * (1.0 + ratio)

    def _ordered_position_codes(
        self, positions: Mapping[str, int], preferred: Sequence[str], stock_codes: Sequence[str]
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for code in preferred:
            if code in positions and code not in seen:
                result.append(code)
                seen.add(code)
        code_to_idx, _, _ = self._universe_metadata(tuple(stock_codes))
        remaining = set(positions) - seen
        result.extend(sorted((code for code in remaining if code in code_to_idx), key=code_to_idx.__getitem__))
        result.extend(sorted(code for code in remaining if code not in code_to_idx))
        return result

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
        target_codes: Sequence[str],
        keep_codes: Sequence[str],
        sellable_ok: set[str],
        base_target: float,
        band: float,
        target_values: Mapping[str,float] | None = None,
    ) -> tuple[list[tuple[str, int]], dict[str, int], dict[str, str], float, float | None]:
        # Only the small position/target set crosses the numeric boundary;
        # selection and the stock vocabulary retain the complete PIT axis.
        codes = self._ordered_position_codes(positions, buy_targets, market.stock_codes)
        codes.extend(code for code in buy_targets if code not in positions)
        indices = {code: index for index, code in enumerate(codes)}
        buy_indices = np.asarray([indices[code] for code in buy_targets], dtype=np.int64)
        keep = set(keep_codes)
        targets = set(target_codes)
        # Pack each code once. Integer fields never pass through float64;
        # strided field views feed the same numeric equalization authority.
        packed = np.asarray([
            (positions.get(code, 0), sellable.get(code, 0),
             position_values.get(code, 0.0), prices.get(code, math.nan),
             limit_prices.get(code, prices.get(code, math.nan)),
             minimum_buy_quantity(code), buy_quantity_step(code),
             code in targets, code in keep, code in sellable_ok)
            for code in codes
        ], dtype=_EQUALIZE_INPUT_DTYPE)
        quantity, available, values, price, limits, minimum, step, target_mask, keep_mask, sell_mask = (
            packed[name] for name in _EQUALIZE_INPUT_DTYPE.names
        )
        # Like max(price, limit): a NaN operand never replaces the first item.
        budget = np.where(limits > price, limits, price)
        targets = np.asarray([base_target if target_values is None else target_values.get(code,0.) for code in codes],dtype=np.float64)
        result = _rebalance_quantities(
            quantity, available, values, price, budget, buy_indices,
            target_mask, keep_mask, sell_mask,
            minimum, step, account_cash, targets, band, self.fees.parameters,
        )
        sells, buys, buy_order, skipped, cash, cheapest, has_increment = result
        sell_orders = [(code, amount) for code, amount in zip(codes, sells.tolist()) if amount]
        # The kernel assigns a unique index at each code's first actual buy.
        buy_orders = {code: amount for _, code, amount in sorted(
            (order, code, amount)
            for order, code, amount in zip(buy_order.tolist(), codes, buys.tolist()) if amount
        )}
        reasons = ('', 'below_exchange_minimum_or_concentration_cap', 'insufficient_frozen_cash',
                   'missing_open', 'within_or_above_target_band')
        skip_values = skipped.tolist()
        skip_reasons = {codes[index]: reasons[skip_values[index]]
                        for index in buy_indices.tolist() if skip_values[index]}
        return sell_orders, buy_orders, skip_reasons, float(cash), (float(cheapest) if has_increment else None)


__all__ = ["DayMarketData", "DayPlanner"]

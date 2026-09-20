"""Causal T-1 candidate prefilter shared by backtest and live trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numba import njit
from numpy.typing import NDArray

from env.contracts import DayConfig
from env.scoring import score_factor_ranks
from utils.stable_sort import stable_radix_order


@dataclass(frozen=True)
class PrefilterUniverse:
    """Validated immutable axis shared by ranking and candidate selection."""

    stock_codes: tuple[str, ...]
    code_to_index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        codes = tuple(str(code) for code in self.stock_codes)
        if not codes or len(set(codes)) != len(codes):
            raise ValueError("stock_codes must be non-empty and unique")
        object.__setattr__(self, "stock_codes", codes)
        object.__setattr__(self, "code_to_index", MappingProxyType(
            {code: index for index, code in enumerate(codes)}
        ))


def prefilter_n_from_config(payload: Mapping[str, object]) -> int:
    """Read the fixed operational prefilter size from a strategy config."""

    if not isinstance(payload, Mapping):
        raise TypeError("strategy config must be a mapping")
    config = payload.get("individual_config", payload)
    if not isinstance(config, Mapping):
        raise TypeError("individual_config must be a mapping")
    value = config.get("prefilter_n")
    if type(value) is not int or value <= 0:
        raise ValueError("prefilter_n must be a positive int")
    return value


def rank_complete_universe(
    stock_codes: Sequence[str],
    factor_ranks: Mapping[str, NDArray[np.floating]],
    factor_validity: Mapping[str, NDArray[np.bool_]],
    config: DayConfig,
    *,
    pit_universe_mask: NDArray[np.bool_],
) -> tuple[str, ...]:
    """Rank the complete stock vocabulary using only one causal factor row.

    Filters and T-day trade legality are intentionally absent: the result is
    the full T ranking that becomes the T+1 fetch/selection candidate pool.
    Stocks without any enabled factor remain in a stable tail so the result
    is always a permutation of the sealed stock axis.
    """

    universe = PrefilterUniverse(stock_codes)
    order = rank_complete_universe_indices(
        universe, factor_ranks, factor_validity, config,
        pit_universe_mask=pit_universe_mask,
    )
    return tuple(universe.stock_codes[index] for index in order)


def rank_complete_universe_indices(
    universe: PrefilterUniverse,
    factor_ranks: Mapping[str, NDArray[np.floating]],
    factor_validity: Mapping[str, NDArray[np.bool_]],
    config: DayConfig,
    *,
    pit_universe_mask: NDArray[np.bool_],
) -> NDArray[np.intp]:
    """The same stable complete ranking, represented by read-only indices."""

    if not isinstance(universe, PrefilterUniverse):
        raise TypeError("universe must be a PrefilterUniverse")
    size = len(universe.stock_codes)
    member = np.asarray(pit_universe_mask, dtype=np.bool_)
    if member.shape != (size,):
        raise ValueError("PIT membership must match the sealed stock axis")
    enabled = tuple(
        name for name, is_enabled in config.factor_enabled.items() if is_enabled
    )
    missing = set(enabled) - set(factor_ranks)
    if missing or set(factor_ranks) != set(factor_validity):
        raise ValueError("prefilter factor inputs do not match DayConfig")

    scores = score_factor_ranks(factor_ranks, factor_validity, config, size)
    return rank_scored_universe_indices(universe, scores, pit_universe_mask=member)


@njit(cache=True, fastmath=False, parallel=False)
def _descending_score_order(values):
    """Exact stable order for the public entry's finite native float64 row."""
    size = len(values)
    bits = values.view(np.uint64)
    keys = np.empty(size, dtype=np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    for index in range(size):
        # +/-zero compare equal and must preserve their original axis order.
        raw = np.uint64(0) if values[index] == 0.0 else bits[index]
        keys[index] = raw if raw & sign else (~raw) ^ sign
    order = np.arange(size, dtype=np.intp)
    scratch = np.empty(size, dtype=np.intp)
    buckets = np.empty(256, dtype=np.intp)
    order, _ = stable_radix_order(keys, order, scratch, size, buckets)
    return order


def rank_scored_universe_indices(
    universe: PrefilterUniverse,
    scores: NDArray[np.floating],
    *,
    pit_universe_mask: NDArray[np.bool_],
    limit: int | None = None,
) -> NDArray[np.intp]:
    """Stable ranking shared by standalone scoring and the day planner."""
    size = len(universe.stock_codes)
    scores = np.asarray(scores, dtype=np.float64)
    member = np.asarray(pit_universe_mask, dtype=np.bool_)
    if scores.shape != (size,) or not np.isfinite(scores).all():
        raise ValueError("universe scores must be finite and match the stock axis")
    if member.shape != (size,):
        raise ValueError("PIT membership must match the sealed stock axis")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("ranking limit must be a positive int or None")
    count = size if limit is None else min(limit, size)
    selected = np.flatnonzero(member)
    if 0 < count < len(selected):
        values = scores[selected]
        threshold = np.partition(values, len(values) - count)[len(values) - count]
        above = selected[values > threshold]
        ties = selected[values == threshold][:count - len(above)]
        selected = np.concatenate((above, ties))
    order = selected[_descending_score_order(scores[selected])]
    if len(order) < count:
        order = np.concatenate((order, np.flatnonzero(~member)[:count - len(order)]))
    order.flags.writeable = False
    return order


def candidate_mask_from_previous_ranking(
    stock_codes: Sequence[str],
    previous_ranking: Sequence[str] | None,
    prefilter_n: int,
    *,
    held_codes: Sequence[str] = (),
) -> NDArray[np.bool_]:
    """Return T candidates from T-1 top-N plus every current holding.

    ``previous_ranking=None`` is the explicit cold-start state and therefore
    keeps the complete axis for the first decision of an independent chain.
    """

    if type(prefilter_n) is not int or prefilter_n <= 0:
        raise ValueError("prefilter_n must be a positive int")
    universe = PrefilterUniverse(stock_codes)
    codes = universe.stock_codes
    if previous_ranking is None:
        return np.ones(len(codes), dtype=np.bool_)
    ranking = tuple(str(code) for code in previous_ranking)
    if len(ranking) != len(codes) or set(ranking) != set(codes):
        raise ValueError("previous ranking must be a permutation of stock_codes")
    indices = np.fromiter(
        (universe.code_to_index[code] for code in ranking),
        dtype=np.intp, count=len(codes),
    )
    return candidate_mask_from_previous_indices(
        universe, indices, prefilter_n, held_codes=held_codes,
    )


@njit(cache=True, fastmath=False, parallel=False)
def _checked_ranking_mask(ranking, size, prefilter_n):
    seen = np.zeros(size, dtype=np.bool_)
    mask = seen if len(ranking) <= prefilter_n else np.zeros(size, dtype=np.bool_)
    for offset in range(len(ranking)):
        index = ranking[offset]
        if index < 0 or index >= size:
            return mask, 1
        if seen[index]:
            return mask, 2
        seen[index] = True
        if offset < prefilter_n:
            mask[index] = True
    return mask, 0


def candidate_mask_from_previous_indices(
    universe: PrefilterUniverse,
    previous_ranking: NDArray[np.integer] | None,
    prefilter_n: int,
    *,
    held_codes: Sequence[str] = (),
    ranking_is_prefix: bool = False,
) -> NDArray[np.bool_]:
    """T-1 top-N plus held names, with a fully checked integer permutation."""

    if type(prefilter_n) is not int or prefilter_n <= 0:
        raise ValueError("prefilter_n must be a positive int")
    if not isinstance(universe, PrefilterUniverse):
        raise TypeError("universe must be a PrefilterUniverse")
    size = len(universe.stock_codes)
    if previous_ranking is None:
        return np.ones(size, dtype=np.bool_)
    ranking = np.asarray(previous_ranking)
    if type(ranking_is_prefix) is not bool:
        raise TypeError("ranking_is_prefix must be bool")
    expected_count = min(size, prefilter_n) if ranking_is_prefix else size
    if ranking.shape != (expected_count,) or not np.issubdtype(ranking.dtype, np.integer):
        raise ValueError("previous ranking must be an integer permutation of the stock axis")
    ranking = np.asarray(ranking, dtype=np.intp)
    mask, error = _checked_ranking_mask(ranking, size, min(prefilter_n, size))
    if error == 1:
        raise ValueError("previous ranking indices are outside the sealed stock axis")
    if error == 2:
        raise ValueError("previous ranking must be a permutation of the stock axis")
    held = {str(code) for code in held_codes}
    unknown = {code for code in held if code not in universe.code_to_index}
    if unknown:
        raise ValueError(f"held codes are outside the sealed stock axis: {sorted(unknown)}")
    for code in held:
        mask[universe.code_to_index[code]] = True
    return mask


__all__ = [
    "PrefilterUniverse",
    "candidate_mask_from_previous_indices",
    "candidate_mask_from_previous_ranking",
    "prefilter_n_from_config",
    "rank_complete_universe",
    "rank_complete_universe_indices",
    "rank_scored_universe_indices",
]

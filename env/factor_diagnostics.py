"""Offline cross-sectional factor diagnostics; callers own split boundaries.

These statistics do not select stocks, construct prices, or settle accounts.
Positive IC means a larger supplied score predicts a larger supplied return.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


MIN_CROSS_SECTION = 30
MIN_PHASE_DAYS = 100


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """One-based average ranks of finite values, including exact ties."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], values.size]
    result = np.empty(values.size, dtype=np.float64)
    result[order] = np.repeat((starts + ends + 1.0) / 2.0, ends - starts)
    return result


def _score_inputs(scores: ArrayLike, eligible: ArrayLike):
    matrix = np.asarray(scores, dtype=np.float64)
    member = np.asarray(eligible, dtype=bool)
    if matrix.ndim != 2 or member.shape != (matrix.shape[1],):
        raise ValueError("scores must be [F,N] and eligible must be [N]")
    return matrix, member


def daily_rank_ic(
    scores: ArrayLike, forward_returns: ArrayLike, eligible: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Spearman IC and pair counts, reranking each common finite universe.

    Fewer than 30 pairs or a constant score/return produces NaN. Average
    ranks ensure binary events cannot acquire artificial ordering within ties.
    """
    matrix, member = _score_inputs(scores, eligible)
    returns = np.asarray(forward_returns, dtype=np.float64)
    if returns.shape != member.shape:
        raise ValueError("forward_returns must be [N]")
    common = np.isfinite(matrix) & (member & np.isfinite(returns))[None, :]
    counts = common.sum(axis=1, dtype=np.int64)
    ic = np.full(matrix.shape[0], np.nan)
    for factor_index in np.flatnonzero(counts >= MIN_CROSS_SECTION):
        valid = common[factor_index]
        x = _average_ranks(matrix[factor_index, valid])
        y = _average_ranks(returns[valid])
        x -= x.mean()
        y -= y.mean()
        denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
        if denominator > 0:
            ic[factor_index] = np.clip(np.dot(x, y) / denominator, -1, 1)
    return ic, counts


def rank_similarity(
    scores: ArrayLike, eligible: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Pearson similarity of each factor's own eligible finite average ranks.

    Each pair uses common valid stocks, at least 30. Ranks are computed once
    per factor, not reranked per pair: with different missingness this is NOT
    exact pairwise Spearman correlation. Constants produce NaN.
    """
    matrix, member = _score_inputs(scores, eligible)
    valid = np.isfinite(matrix) & member[None, :]
    ranks = np.zeros_like(matrix)
    for factor_index in np.flatnonzero(valid.any(axis=1)):
        mask = valid[factor_index]
        ranks[factor_index, mask] = _average_ranks(matrix[factor_index, mask])
    mask_float = valid.astype(np.float64)
    counts_float = mask_float @ mask_float.T
    divisor = np.maximum(counts_float, 1)
    sums = ranks @ mask_float.T
    squares = (ranks * ranks) @ mask_float.T
    cross = ranks @ ranks.T - sums * sums.T / divisor
    variance = np.maximum(squares - sums * sums / divisor, 0)
    denominator = np.sqrt(variance * variance.T)
    usable = (counts_float >= MIN_CROSS_SECTION) & (denominator > 0)
    result = np.full(counts_float.shape, np.nan)
    np.divide(cross, denominator, out=result, where=usable)
    return np.clip(result, -1, 1), counts_float.astype(np.int64)


def _mean_statistics(values: NDArray[np.float64], lag: int) -> dict:
    finite = np.isfinite(values)
    n = int(finite.sum())
    mean = float(values[finite].mean()) if n else float("nan")
    # Missing days stay in place: an unavailable IC must not turn a lag-2
    # pair into a lag-1 pair. Zero residuals give no covariance contribution.
    residual = np.where(finite, values - mean, 0.0)
    effective_lag = min(lag, max(values.size - 1, 0))
    long_run_sum = float(np.dot(residual, residual))
    for distance in range(1, effective_lag + 1):
        weight = 1.0 - distance / (lag + 1.0)
        long_run_sum += 2.0 * weight * float(
            np.dot(residual[distance:], residual[:-distance])
        )
    se = float(np.sqrt(max(long_run_sum, 0.0)) / n) if n >= 2 else float("nan")
    return {
        "mean_ic": mean,
        "positive_fraction": float(np.mean(values[finite] > 0)) if n else float("nan"),
        "n": n,
        "nw_lag": lag,
        "nw_effective_lag": effective_lag,
        "nw_standard_error": se,
        "nw_t": mean / se if se > 0 else float("nan"),
        "phase_eligible": n >= MIN_PHASE_DAYS,
    }


def summarize_ic(dates: ArrayLike, ic: ArrayLike, horizon: int) -> list[dict]:
    """All-period, fixed calendar-year and half-year IC summaries by factor.

    Newey-West uses Bartlett weights, lag=max(horizon-1,20), no small-sample
    adjustment and no multiple-testing correction. A phase needs 100 valid
    days to be a reference; this flag is not a significance or success claim.
    The caller must supply only authorized split dates and forward labels.
    """
    days = np.asarray(dates, dtype="datetime64[D]")
    matrix = np.asarray(ic, dtype=np.float64)
    if days.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != days.size:
        raise ValueError("dates must be [T] and ic must be [T,F]")
    if days.size == 0 or np.isnat(days).any() or np.any(days[1:] <= days[:-1]):
        raise ValueError("dates must be nonempty, valid and strictly increasing")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    years = days.astype("datetime64[Y]").astype(int) + 1970
    months = days.astype("datetime64[M]").astype(int) % 12 + 1
    halves = np.where(months <= 6, 1, 2)
    periods = [("all", "all", np.ones(days.size, dtype=bool))]
    for year in np.unique(years):
        periods.append(("year", str(year), years == year))
        for half in (1, 2):
            selected = (years == year) & (halves == half)
            if selected.any():
                periods.append(("halfyear", f"{year}-H{half}", selected))
    output = []
    lag = max(int(horizon) - 1, 20)
    for kind, name, selected in periods:
        for factor_index in range(matrix.shape[1]):
            output.append({
                "factor_index": factor_index,
                "period_kind": kind,
                "period": name,
                "start": str(days[selected][0]),
                "end": str(days[selected][-1]),
                "horizon": int(horizon),
                **_mean_statistics(matrix[selected, factor_index], lag),
            })
    return output

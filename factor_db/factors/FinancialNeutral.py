"""PIT financial quality factor with neutral treatment of missing reports.

The ordinary multi-factor path excludes a stock whenever any active factor is
NaN.  Historical financial coverage changes materially through time, so that
behavior turns a small financial tilt into an era-dependent universe filter.
This factor instead emits an already-ranked 0.5 score when a financial
component is unavailable, while retaining NaN for stocks that are not otherwise
tradable.  It therefore changes ranking, not universe membership.
"""

from __future__ import annotations

import numpy as np

MIN_RAW_PRICE = 2.0


def _percentile_rank_or_neutral(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Rank each date cross-sectionally and fill unavailable values with 0.5."""
    masked = np.where(valid, values, np.nan)
    nan = ~np.isfinite(masked)
    sortable = np.where(nan, -np.inf, masked.astype(np.float64, copy=False))
    order = np.argsort(np.argsort(-sortable, axis=1), axis=1).astype(np.float32)
    counts = (~nan).sum(axis=1, keepdims=True).astype(np.float32)
    ranks = 1.0 - (order + 0.5) / np.where(counts > 0.0, counts, 1.0)
    ranks[nan] = 0.5
    return ranks


class FinancialQualityGrowthNeutralPIT:
    """Five-to-one quality/growth blend; unavailable components score neutral."""

    hist_days = 0
    scores_are_ranks = True

    def calc_batch(self, panel: dict) -> np.ndarray:
        eps = np.asarray(panel["eps"], dtype=np.float64)
        ocfps = np.asarray(panel["operating_cf_ps"], dtype=np.float64)
        gross_margin = np.asarray(panel["gross_margin"], dtype=np.float64)
        profit_yoy = np.asarray(panel["profit_yoy"], dtype=np.float64)
        open_price = np.asarray(panel["open"], dtype=np.float64)
        base_valid = (
            np.isfinite(open_price)
            & (open_price >= MIN_RAW_PRICE)
            & ~np.asarray(panel["st_mask"], dtype=bool)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            sloan = np.clip(-(eps - ocfps) / np.abs(eps), -10.0, 10.0)
            cash_coverage = np.clip(
                ocfps / np.maximum(np.abs(eps), 0.01),
                -5.0,
                5.0,
            )

        earnings_cash_valid = (
            base_valid & np.isfinite(eps) & np.isfinite(ocfps)
        )
        sloan_rank = _percentile_rank_or_neutral(
            sloan,
            earnings_cash_valid & (np.abs(eps) > 0.005) & np.isfinite(sloan),
        )
        cash_rank = _percentile_rank_or_neutral(
            cash_coverage,
            earnings_cash_valid & np.isfinite(cash_coverage),
        )
        margin_rank = _percentile_rank_or_neutral(
            gross_margin,
            base_valid
            & np.isfinite(gross_margin)
            & (gross_margin > -10.0)
            & (gross_margin < 100.0),
        )
        growth_rank = _percentile_rank_or_neutral(
            profit_yoy,
            base_valid & np.isfinite(profit_yoy),
        )

        quality_rank = (sloan_rank + cash_rank + margin_rank) / 3.0
        score = (5.0 * quality_rank + growth_rank) / 6.0
        return np.where(base_valid, score, np.nan).astype(np.float32)

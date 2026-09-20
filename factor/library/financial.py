"""Production scores over float64 financial primitives already aligned as-of < T.

The offline snapshot owns disclosure/revision replay and growth definitions.
Only total_share is lagged by the factor preparation layer: financial rows
already describe information available before the corresponding decision.
"""

import numpy as np


FINANCIAL_SEMANTICS_VERSION = "financial-strict-before-t-soft-score-v1"


class PBBelowTwoROEAbove10Signal:
    """Known PB < 2 and ROE > 10% is 1; known rejection is a valid 0.

    ``total_share`` must be the preparation layer's T-1 share count.
    """

    hist_days = 1

    def calc_batch(self, panel: dict) -> np.ndarray:
        opening, shares, profit, equity = (
            np.asarray(panel[name], dtype=np.float64)
            for name in ("open", "total_share", "financial_profit_ttm", "financial_equity")
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cap = opening * shares
            pb = cap / equity
            roe = profit / equity
        valid = (
            np.isfinite(opening) & (opening > 0)
            & np.isfinite(shares) & (shares > 0)
            & np.isfinite(cap) & (cap > 0)
            & np.isfinite(equity) & (equity > 0)
            & np.isfinite(profit)
        )
        return np.where(valid, ((pb < 2.0) & (roe > 0.1)).astype(np.float64), np.nan)


def _finite_spread(preferred: np.ndarray, other: np.ndarray) -> np.ndarray:
    preferred = np.asarray(preferred, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        spread = preferred - other
    return np.where(
        np.isfinite(preferred) & np.isfinite(other) & np.isfinite(spread),
        spread,
        np.nan,
    )


class LowCashOutflowProfitGrowthSpread:
    """Prefer lower cash-outflow growth minus profit growth."""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        return _finite_spread(panel["financial_profit_yoy"], panel["financial_cash_outflow_yoy"])


class HighOperatingProfitRevenueGrowthSpread:
    """Prefer higher operating-profit growth minus revenue growth."""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        return _finite_spread(panel["financial_operating_profit_yoy"], panel["financial_revenue_yoy"])

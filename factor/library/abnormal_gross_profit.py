"""Video abnormal gross profit score over sealed, as-of quarterly primitives.

The offline data layer supplies as-of single-quarter values and period-end
assets. Research and production consume the same formula and validity rule.
"""
from collections.abc import Mapping

import numpy as np

from offline_data import ABNORMAL_GROSS_PROFIT_PANEL_FIELDS


ABNORMAL_GROSS_PROFIT_VERSION = "abnormal-gross-profit-single-quarter-cash-scaled-v1"


def abnormal_gross_profit(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return (GP_q - GP_q-4 * cash_q / cash_q-4) / assets_q.

    Missing operands, nonpositive prior sales cash or assets, and negative
    current sales cash are unavailable. Current zero cash is allowed. Revenue
    and cost retain source signs; no clipping or cross-sectional fill occurs.
    """
    operands = [np.asarray(fields[name], dtype=np.float64) for name in ABNORMAL_GROSS_PROFIT_PANEL_FIELDS]
    if any(value.shape != operands[0].shape for value in operands[1:]):
        raise ValueError("abnormal gross profit operand axes differ")
    revenue, cost, cash, prior_revenue, prior_cost, prior_cash, assets = operands
    valid = np.logical_and.reduce([np.isfinite(value) for value in operands])
    valid &= (cash >= 0) & (prior_cash > 0) & (assets > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = ((revenue - cost) - (prior_revenue - prior_cost) * cash / prior_cash) / assets
    result = np.where(valid & np.isfinite(result), result, np.nan)
    result.flags.writeable = False
    return result


class HighAbnormalGrossProfit:
    """Prefer high abnormal gross profit using already-causal raw primitives.

    Announcement availability and single-quarter alignment belong to
    offline_data. Inputs are available at T-open and must not be lagged again.
    """
    hist_days = 0

    def calc_batch(self, panel: Mapping[str, np.ndarray]) -> np.ndarray:
        return abnormal_gross_profit(panel)


__all__ = ["ABNORMAL_GROSS_PROFIT_VERSION", "ABNORMAL_GROSS_PROFIT_PANEL_FIELDS",
           "abnormal_gross_profit", "HighAbnormalGrossProfit"]

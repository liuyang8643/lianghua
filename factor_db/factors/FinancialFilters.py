"""慢频财务买入过滤器。

runtime 财务字段已经按法定最晚披露日后的首个交易日做 PIT 对齐，因此过滤器
可在 T 日开盘直接使用 T 行财务值。返回正数表示允许买入，NaN 表示排除。
"""

from __future__ import annotations

import numpy as np


def _base(panel: dict) -> np.ndarray:
    open_ = panel["open"]
    return np.isfinite(open_) & (open_ >= 2.0) & ~panel["st_mask"]


class FilterFinancialCoreCoverage:
    """仅允许 EPS、经营现金流/股、毛利率三项均有同一期有效值的股票。"""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        valid = (
            _base(panel)
            & np.isfinite(panel["eps"])
            & np.isfinite(panel["operating_cf_ps"])
            & np.isfinite(panel["gross_margin"])
        )
        return np.where(valid, 1.0, np.nan)


class FilterPositiveEarnings:
    """排除 EPS 不为正或缺失的公司。"""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        eps = panel["eps"]
        valid = _base(panel) & np.isfinite(eps) & (eps > 0)
        return np.where(valid, 1.0, np.nan)


class FilterPositiveOperatingCashFlow:
    """排除每股经营现金流不为正或缺失的公司。"""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        cash_flow = panel["operating_cf_ps"]
        valid = (
            _base(panel)
            & np.isfinite(cash_flow)
            & (cash_flow > 0)
        )
        return np.where(valid, 1.0, np.nan)


class FilterPositiveEarningsAndCashFlow:
    """同时要求 EPS 和每股经营现金流为正。"""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        eps = panel["eps"]
        cash_flow = panel["operating_cf_ps"]
        valid = (
            _base(panel)
            & np.isfinite(eps)
            & (eps > 0)
            & np.isfinite(cash_flow)
            & (cash_flow > 0)
        )
        return np.where(valid, 1.0, np.nan)


class FilterPositiveROE:
    """排除摊薄 ROE 不为正或缺失的公司。"""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        roe = panel["roe"]
        valid = _base(panel) & np.isfinite(roe) & (roe > 0)
        return np.where(valid, 1.0, np.nan)


class FilterFinancialQualityFloor:
    """盈利、现金流、ROE、毛利率均为正的保守质量底线。"""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        eps = panel["eps"]
        cash_flow = panel["operating_cf_ps"]
        roe = panel["roe"]
        gross_margin = panel["gross_margin"]
        valid = (
            _base(panel)
            & np.isfinite(eps)
            & (eps > 0)
            & np.isfinite(cash_flow)
            & (cash_flow > 0)
            & np.isfinite(roe)
            & (roe > 0)
            & np.isfinite(gross_margin)
            & (gross_margin > 0)
        )
        return np.where(valid, 1.0, np.nan)

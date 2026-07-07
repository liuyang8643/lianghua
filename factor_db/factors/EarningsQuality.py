"""盈余质量因子 — 应计利润越低、现金流覆盖越好的公司得分越高。
高分=盈利由真实现金支撑，低分=纸上利润。
"""
import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class SloanAccruals:
    """Sloan应计 — (eps-ocfps)/|eps|，正应计=利润虚高(差)，负应计=利润保守(好)，取反使高分=优质"""
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        eps = _shift(panel["eps"])
        ocfps = _shift(panel["operating_cf_ps"])
        open_p = panel["close"]
        valid = (
            ~np.isnan(open_p) & (open_p >= MIN_RAW_PRICE)
            & ~panel["st_mask"]
            & np.isfinite(eps) & np.isfinite(ocfps)
            & (np.abs(eps) > 0.005)
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            accruals = (eps - ocfps) / np.abs(eps)
            score = -accruals
            score = np.clip(score, -10, 10)
        return np.where(valid, score, np.nan)


class CashFlowCoverage:
    """现金流覆盖度 — ocfps/max(|eps|,0.01)，>1=利润有现金支撑"""
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        eps = _shift(panel["eps"])
        ocfps = _shift(panel["operating_cf_ps"])
        open_p = panel["close"]
        valid = (
            ~np.isnan(open_p) & (open_p >= MIN_RAW_PRICE)
            & ~panel["st_mask"]
            & np.isfinite(eps) & np.isfinite(ocfps)
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.maximum(np.abs(eps), 0.01)
            score = ocfps / denom
            score = np.clip(score, -5, 5)
        return np.where(valid, score, np.nan)


class AccrualsYield:
    """应计收益率 — (ocfps-eps)/open，负应计(现金流>利润)=盈利保守=高分"""
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        eps = _shift(panel["eps"])
        ocfps = _shift(panel["operating_cf_ps"])
        open_p = panel["close"]
        valid = (
            ~np.isnan(open_p) & (open_p >= MIN_RAW_PRICE)
            & ~panel["st_mask"]
            & np.isfinite(eps) & np.isfinite(ocfps)
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            accruals = (eps - ocfps) / open_p
            score = -accruals
            score = np.clip(score, -2, 2)
        return np.where(valid, score, np.nan)


class GrossMarginQuality:
    """毛利率 — 高毛利率在小盘股中=真生意而非壳，直接读字段"""
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        gm = _shift(panel["gross_margin"])
        open_p = panel["close"]
        valid = (
            ~np.isnan(open_p) & (open_p >= MIN_RAW_PRICE)
            & ~panel["st_mask"]
            & np.isfinite(gm)
            & (gm > -10) & (gm < 100)
        )
        return np.where(valid, gm, np.nan)


class EarningsQualityComposite:
    """盈余质量复合 — Sloan应计取负+现金流覆盖+毛利率 三路等权排名平均"""
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        eps = _shift(panel["eps"])
        ocfps = _shift(panel["operating_cf_ps"])
        gm = _shift(panel["gross_margin"])
        open_p = panel["close"]
        st = panel["st_mask"]

        bv = ~np.isnan(open_p) & (open_p >= MIN_RAW_PRICE) & ~st
        fin_ok = np.isfinite(eps) & np.isfinite(ocfps)

        # Sloan
        with np.errstate(divide='ignore', invalid='ignore'):
            accruals = (eps - ocfps) / np.abs(eps)
            sloan = np.clip(-accruals, -10, 10)
        sv = bv & fin_ok & (np.abs(eps) > 0.005) & np.isfinite(sloan)

        # CashFlowCoverage
        with np.errstate(divide='ignore', invalid='ignore'):
            cfc = ocfps / np.maximum(np.abs(eps), 0.01)
            cfc = np.clip(cfc, -5, 5)
        cv = bv & fin_ok & np.isfinite(cfc)

        # GrossMargin
        gv = bv & np.isfinite(gm) & (gm > -10) & (gm < 100)

        def _pct_rank(x):
            nan = np.isnan(x)
            xx = np.where(nan, -np.inf, x.astype(np.float64))
            order = np.argsort(np.argsort(-xx, axis=1), axis=1).astype(np.float32)
            n = (~nan).sum(axis=1, keepdims=True).astype(np.float32)
            r = 1.0 - order / np.where(n > 0, n, 1.0)
            r[nan] = np.nan
            return r

        s1 = np.where(sv, sloan, np.nan)
        s2 = np.where(cv, cfc, np.nan)
        s3 = np.where(gv, gm, np.nan)
        comp = np.nanmean(np.stack([_pct_rank(s1), _pct_rank(s2), _pct_rank(s3)]), axis=0)
        out_valid = bv & np.isfinite(comp)
        return np.where(out_valid, comp, np.nan)

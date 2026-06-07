import numpy as np

HIST_WINDOW = 252
QUALITY_WEIGHT = 0.25
SIZE_JITTER = 0.05

__thesis__ = "52周低点反转叠加质量过滤，捕捉锚定效应修复"


class Factor_20260605_192540_g4_0:
    hist_days = HIST_WINDOW

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_arr = panel['open'].astype(np.float64)
        n_dates, n_stocks = open_arr.shape
        w = self.hist_days

        base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']

        pad = (w - n_dates % w) % w
        padded = np.pad(open_arr, ((0, pad), (0, 0)), constant_values=np.nan)
        n2 = padded.shape[0]
        blocks = padded.reshape(n2 // w, w, n_stocks)

        prefix = np.maximum.accumulate(blocks, axis=1).reshape(n2, n_stocks)[:n_dates]
        suffix = np.maximum.accumulate(blocks[:, ::-1, :], axis=1)[:, ::-1, :].reshape(n2, n_stocks)[:n_dates]

        suf_lag = np.empty((n_dates, n_stocks))
        suf_lag[:w - 1] = np.nan
        suf_lag[w - 1:] = suffix[:n_dates - (w - 1)]

        mask = np.arange(n_dates)[:, None] >= (w - 1)
        rolling_max = np.where(mask, np.fmax(suf_lag, prefix), np.nan)

        with np.errstate(divide='ignore', invalid='ignore'):
            proximity = open_arr / rolling_max

        def _cross_rank(x):
            x64 = x.astype(np.float64)
            nan_mask = np.isnan(x64)
            x_filled = np.where(nan_mask, 0.0, x64)
            order = np.argsort(np.argsort(x_filled, axis=1), axis=1).astype(np.float64)
            n_valid = (~nan_mask).sum(axis=1, keepdims=True).astype(np.float64)
            rank = 2.0 * order / np.maximum(n_valid - 1.0, 1.0) - 1.0
            return np.where(nan_mask, np.nan, rank).astype(np.float32)

        rev_sig = _cross_rank(-proximity)
        rev_sig = np.where(np.isfinite(rev_sig), rev_sig, 0.0)

        roe_f = np.where(np.isfinite(panel['roe']), panel['roe'], 0.0)
        gm_f = np.where(np.isfinite(panel['gross_margin']), panel['gross_margin'], 0.0)
        quality_raw = np.sign(roe_f) * np.abs(roe_f) ** 0.5 + np.sign(gm_f) * np.abs(gm_f) ** 0.5
        quality_sig = _cross_rank(quality_raw)
        quality_sig = np.where(np.isfinite(quality_sig), quality_sig, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_mcap = np.log(panel['total_share'] * panel['open'])
        size_sig = _cross_rank(log_mcap)
        size_sig = np.where(np.isfinite(size_sig), size_sig, 0.0)

        score = (1.0 - QUALITY_WEIGHT - SIZE_JITTER) * rev_sig + QUALITY_WEIGHT * quality_sig + SIZE_JITTER * size_sig
        return np.where(base_valid & np.isfinite(score), score.astype(np.float32), np.nan)

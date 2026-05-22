"""SAR抛物线转向因子 — open到SAR的距离."""
import numpy as np

MIN_PRICE = 2.0
ACCEL_STEP = 0.02
ACCEL_MAX = 0.20


class SAR:
    hist_days = 10

    def calc_batch(self, panel: dict) -> np.ndarray:
        high = panel["high"]
        low = panel["low"]
        opn = panel["open"]
        st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        N, S = high.shape

        hp = np.roll(high, 1, axis=0); hp[0] = np.nan
        lp = np.roll(low, 1, axis=0); lp[0] = np.nan

        sar_arr = np.full((N, S), np.nan, dtype=np.float32)
        is_long = np.ones(S, dtype=bool)
        ep = np.full(S, np.nan)
        af = np.full(S, ACCEL_STEP)
        sar_prev = np.full(S, np.nan)

        for t in range(1, N):
            h_t = hp[t]; l_t = lp[t]
            active = ~np.isnan(h_t) & ~np.isnan(l_t)
            if not active.any():
                continue

            first_bar = active & np.isnan(sar_prev)
            if first_bar.any():
                sar_prev[first_bar] = l_t[first_bar]
                ep[first_bar] = h_t[first_bar]
                sar_arr[t, first_bar] = np.nan
                continue

            ongoing = active & ~first_bar
            if not ongoing.any():
                continue

            long_mask = ongoing & is_long
            if long_mask.any():
                sar_new = sar_prev[long_mask] + af[long_mask] * (ep[long_mask] - sar_prev[long_mask])
                sar_new = np.minimum(sar_new, l_t[long_mask])
                sar_arr[t, long_mask] = sar_new

                new_high = h_t[long_mask] > ep[long_mask]
                ep[long_mask] = np.where(new_high, h_t[long_mask], ep[long_mask])
                af[long_mask] = np.where(new_high, np.minimum(af[long_mask] + ACCEL_STEP, ACCEL_MAX), af[long_mask])

                flip = sar_new > l_t[long_mask]
                if flip.any():
                    idx = long_mask.copy()
                    idx[idx] = flip
                    is_long[idx] = False
                    sar_arr[t, idx] = ep[idx]
                    ep[idx] = l_t[idx]
                    af[idx] = ACCEL_STEP

            short_mask = ongoing & ~is_long
            if short_mask.any():
                sar_new = sar_prev[short_mask] - af[short_mask] * (sar_prev[short_mask] - ep[short_mask])
                sar_new = np.maximum(sar_new, h_t[short_mask])
                sar_arr[t, short_mask] = sar_new

                new_low = l_t[short_mask] < ep[short_mask]
                ep[short_mask] = np.where(new_low, l_t[short_mask], ep[short_mask])
                af[short_mask] = np.where(new_low, np.minimum(af[short_mask] + ACCEL_STEP, ACCEL_MAX), af[short_mask])

                flip = sar_new < h_t[short_mask]
                if flip.any():
                    idx = short_mask.copy()
                    idx[idx] = flip
                    is_long[idx] = True
                    sar_arr[t, idx] = ep[idx]
                    ep[idx] = h_t[idx]
                    af[idx] = ACCEL_STEP

            updated = ~np.isnan(sar_arr[t])
            sar_prev[updated] = sar_arr[t, updated]

        score = opn - sar_arr
        return np.where(valid, score, np.nan)

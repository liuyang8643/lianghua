"""Strict 20-day completed signed-RMB-amount imbalance factor."""

from __future__ import annotations

import numpy as np


class CompletedSignedAmountImbalance20Strict:
    """Prefer completed windows whose RMB amount is concentrated on up days.

    Row ``T`` uses exactly ``[T-20, T)``.  A completed day's direction is the
    sign of its official ``close / preClose`` return, determined by direct
    positive-price comparison.  Zero-return days contribute no signed amount
    but retain their positive amount in the denominator.  Every close,
    official pre-close, and amount observation must be finite and strictly
    positive or the entire affected window is invalid.
    """

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = np.asarray(panel["close"])
        pre_close = np.asarray(panel["preClose"])
        amount = np.asarray(panel["amount"])
        if close.ndim != 2:
            raise ValueError("close, preClose, and amount must be two-dimensional")
        if not (close.shape == pre_close.shape == amount.shape):
            raise ValueError("close, preClose, and amount must have matching shapes")

        rows, stocks = close.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        window = self.hist_days
        if rows <= window:
            return result

        # For row ``block_start + offset``, the 20-day history is exactly the
        # suffix ``previous_block[offset:]`` plus the prefix
        # ``current_block[:offset]``.  Scaled summaries for those two pieces
        # avoid rolling subtraction and use only fixed 20-row working state.
        suffix_scale = np.zeros((window + 1, stocks), dtype=np.float64)
        suffix_total = np.zeros((window + 1, stocks), dtype=np.float64)
        suffix_signed = np.zeros((window + 1, stocks), dtype=np.float64)
        suffix_count = np.zeros((window + 1, stocks), dtype=np.int16)

        for block_start in range(window, rows, window):
            suffix_scale[window].fill(0.0)
            suffix_total[window].fill(0.0)
            suffix_signed[window].fill(0.0)
            suffix_count[window].fill(0)

            for offset in range(window - 1, -1, -1):
                history_row = block_start - window + offset
                close_row = close[history_row]
                pre_close_row = pre_close[history_row]
                amount_row = amount[history_row]
                daily_valid = (
                    np.isfinite(close_row)
                    & (close_row > 0.0)
                    & np.isfinite(pre_close_row)
                    & (pre_close_row > 0.0)
                    & np.isfinite(amount_row)
                    & (amount_row > 0.0)
                )
                daily_amount = np.where(daily_valid, amount_row, 0.0)
                daily_direction = np.greater(
                    close_row, pre_close_row
                ).astype(np.int8)
                daily_direction -= np.less(close_row, pre_close_row)
                daily_direction[~daily_valid] = 0

                old_scale = suffix_scale[offset + 1]
                new_scale = np.maximum(old_scale, daily_amount)
                old_weight = np.zeros(stocks, dtype=np.float64)
                daily_weight = np.zeros(stocks, dtype=np.float64)
                np.divide(
                    old_scale,
                    new_scale,
                    out=old_weight,
                    where=new_scale > 0.0,
                )
                np.divide(
                    daily_amount,
                    new_scale,
                    out=daily_weight,
                    where=new_scale > 0.0,
                )
                suffix_scale[offset] = new_scale
                suffix_total[offset] = (
                    suffix_total[offset + 1] * old_weight + daily_weight
                )
                suffix_signed[offset] = (
                    suffix_signed[offset + 1] * old_weight
                    + daily_weight * daily_direction
                )
                suffix_count[offset] = (
                    suffix_count[offset + 1] + daily_valid
                )

            prefix_scale = np.zeros(stocks, dtype=np.float64)
            prefix_total = np.zeros(stocks, dtype=np.float64)
            prefix_signed = np.zeros(stocks, dtype=np.float64)
            prefix_count = np.zeros(stocks, dtype=np.int16)
            block_rows = min(window, rows - block_start)

            for offset in range(block_rows):
                combined_scale = np.maximum(
                    suffix_scale[offset], prefix_scale
                )
                suffix_weight = np.zeros(stocks, dtype=np.float64)
                prefix_weight = np.zeros(stocks, dtype=np.float64)
                np.divide(
                    suffix_scale[offset],
                    combined_scale,
                    out=suffix_weight,
                    where=combined_scale > 0.0,
                )
                np.divide(
                    prefix_scale,
                    combined_scale,
                    out=prefix_weight,
                    where=combined_scale > 0.0,
                )
                combined_total = (
                    suffix_total[offset] * suffix_weight
                    + prefix_total * prefix_weight
                )
                combined_signed = (
                    suffix_signed[offset] * suffix_weight
                    + prefix_signed * prefix_weight
                )
                combined_count = suffix_count[offset] + prefix_count
                with np.errstate(
                    divide="ignore", invalid="ignore", over="ignore"
                ):
                    score = combined_signed / combined_total
                valid = (
                    (combined_count == window)
                    & np.isfinite(combined_total)
                    & (combined_total > 0.0)
                    & np.isfinite(combined_signed)
                    & np.isfinite(score)
                )
                result[block_start + offset, valid] = score[valid]

                if offset + 1 == block_rows:
                    continue
                current_row = block_start + offset
                close_row = close[current_row]
                pre_close_row = pre_close[current_row]
                amount_row = amount[current_row]
                daily_valid = (
                    np.isfinite(close_row)
                    & (close_row > 0.0)
                    & np.isfinite(pre_close_row)
                    & (pre_close_row > 0.0)
                    & np.isfinite(amount_row)
                    & (amount_row > 0.0)
                )
                daily_amount = np.where(daily_valid, amount_row, 0.0)
                daily_direction = np.greater(
                    close_row, pre_close_row
                ).astype(np.int8)
                daily_direction -= np.less(close_row, pre_close_row)
                daily_direction[~daily_valid] = 0

                new_scale = np.maximum(prefix_scale, daily_amount)
                old_weight = np.zeros(stocks, dtype=np.float64)
                daily_weight = np.zeros(stocks, dtype=np.float64)
                np.divide(
                    prefix_scale,
                    new_scale,
                    out=old_weight,
                    where=new_scale > 0.0,
                )
                np.divide(
                    daily_amount,
                    new_scale,
                    out=daily_weight,
                    where=new_scale > 0.0,
                )
                prefix_total = prefix_total * old_weight + daily_weight
                prefix_signed = (
                    prefix_signed * old_weight
                    + daily_weight * daily_direction
                )
                prefix_scale = new_scale
                prefix_count += daily_valid

        return result

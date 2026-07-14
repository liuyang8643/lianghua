"""Breadth-gated, risk-adjusted V3 of the open-auction trend signal."""

import numpy as np

from factor_db.factors.TrendOpenSignal import (
    _buy_signal,
    _lag,
    _lag_bool,
    _rolling_mean,
)
from factor_db.factors.TrendOpenSignalV2 import _quality_score, _sell_signal_v2


MIN_RAW_OPEN = 2.0
MIN_ACTIVE_STOCKS = 20
BREADTH20_ON_FLOOR = 0.43
BREADTH20_OFF_FLOOR = 0.36
BREADTH60_FLOOR = 0.35
BREADTH20_CHANGE5_FLOOR = -0.05
SCORE_WEIGHTS = (0.0, 0.0, 0.0, 0.0)


class TrendOpenSignalV3:
    """Trend state ranked by momentum quality and gated by T-1 breadth."""

    hist_days = 71
    pre_ranked = True
    requires_full_history = True

    def calc_batch(self, panel: dict) -> np.ndarray:
        components = _build_v3_components(panel)
        return _compose_v3_scores(
            components,
            min_active_stocks=MIN_ACTIVE_STOCKS,
            breadth20_on_floor=BREADTH20_ON_FLOOR,
            breadth20_off_floor=BREADTH20_OFF_FLOOR,
            breadth60_floor=BREADTH60_FLOOR,
            breadth20_change5_floor=BREADTH20_CHANGE5_FLOOR,
            score_weights=SCORE_WEIGHTS,
        )


def _build_v3_components(panel: dict) -> dict:
    open_ = np.asarray(panel["open"], dtype=np.float64)
    high = np.asarray(panel["high"], dtype=np.float64)
    low = np.asarray(panel["low"], dtype=np.float64)
    close = np.asarray(panel["close"], dtype=np.float64)
    amount = np.asarray(panel["amount"], dtype=np.float64)
    st_mask = np.asarray(panel["st_mask"], dtype=bool)

    maw5 = _rolling_mean(close * amount, 5) / _rolling_mean(amount, 5)
    maw20 = _rolling_mean(close * amount, 20) / _rolling_mean(amount, 20)
    buy_at_close = _buy_signal(panel, open_, high, low, close, amount, maw5, maw20)
    buy_at_open = _lag_bool(buy_at_close, 1) & np.isfinite(open_) & (open_ > 0)
    sell_at_open = _sell_signal_v2(panel, open_, high, low, close, amount, maw20)

    rows = np.arange(open_.shape[0], dtype=np.int32)[:, None]
    last_buy = np.maximum.accumulate(np.where(buy_at_open, rows, -1), axis=0)
    last_sell = np.maximum.accumulate(np.where(sell_at_open, rows, -1), axis=0)
    holding = (last_buy > last_sell) & (last_buy >= 0)
    eligible = holding & np.isfinite(open_) & (open_ >= MIN_RAW_OPEN) & ~st_mask

    completed_close = _lag(close, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum20 = completed_close / _lag(close, 21) - 1.0
        momentum60 = completed_close / _lag(close, 61) - 1.0
        daily_change = close / _lag(close, 1) - 1.0
    volatility20 = np.sqrt(_lag(_rolling_mean(daily_change * daily_change, 20), 1))

    quality = _quality_score(open_, high, close, maw20)
    quality = np.where(np.isfinite(quality), quality, 0.0)
    momentum20_q = np.clip((momentum20 + 0.05) / 0.35, 0.0, 1.0)
    momentum60_q = np.clip((momentum60 + 0.10) / 0.70, 0.0, 1.0)
    stability_q = np.clip((0.055 - volatility20) / 0.045, 0.0, 1.0)

    valid20 = np.isfinite(close) & np.isfinite(maw20) & ~st_mask
    ma60 = _rolling_mean(close, 60)
    valid60 = np.isfinite(close) & np.isfinite(ma60) & ~st_mask
    breadth20_completed = _cross_section_ratio(valid20 & (close >= maw20), valid20)
    breadth60_completed = _cross_section_ratio(valid60 & (close >= ma60), valid60)
    breadth20 = _lag_1d(breadth20_completed, 1)
    breadth60 = _lag_1d(breadth60_completed, 1)
    breadth20_smoothed = _rolling_mean(breadth20_completed[:, None], 3)[:, 0]
    breadth20_smoothed = _lag_1d(breadth20_smoothed, 1)
    breadth20_change5 = breadth20 - _lag_1d(breadth20, 5)

    return {
        "eligible": eligible,
        "active_count": np.sum(eligible, axis=1),
        "quality": quality.astype(np.float32),
        "momentum20": np.nan_to_num(momentum20_q, nan=0.0).astype(np.float32),
        "momentum60": np.nan_to_num(momentum60_q, nan=0.0).astype(np.float32),
        "stability": np.nan_to_num(stability_q, nan=0.0).astype(np.float32),
        "breadth20": breadth20,
        "breadth20_smoothed": breadth20_smoothed,
        "breadth60": breadth60,
        "breadth20_change5": breadth20_change5,
    }


def _compose_v3_scores(
    components: dict,
    *,
    min_active_stocks: int,
    breadth20_on_floor: float,
    breadth20_off_floor: float,
    breadth60_floor: float,
    breadth20_change5_floor: float,
    score_weights: tuple[float, float, float, float],
) -> np.ndarray:
    quality_w, momentum20_w, momentum60_w, stability_w = score_weights
    score = (
        quality_w * components["quality"]
        + momentum20_w * components["momentum20"]
        + momentum60_w * components["momentum60"]
        + stability_w * components["stability"]
    )
    market_on = _hysteresis_state(
        components["breadth20_smoothed"],
        on_floor=breadth20_on_floor,
        off_floor=breadth20_off_floor,
    )
    regime = (
        (components["active_count"] >= min_active_stocks)
        & market_on
        & (components["breadth60"] >= breadth60_floor)
        & (components["breadth20_change5"] >= breadth20_change5_floor)
    )
    bounded_score = np.clip(0.05 + 0.95 * score, 0.05, 1.0)
    return np.where(
        components["eligible"] & regime[:, None], bounded_score, 0.0
    ).astype(np.float32)


def _hysteresis_state(
    values: np.ndarray, *, on_floor: float, off_floor: float
) -> np.ndarray:
    if on_floor <= 0.0:
        return np.ones(values.shape, dtype=bool)
    rows = np.arange(values.shape[0], dtype=np.int32)
    last_on = np.maximum.accumulate(np.where(values >= on_floor, rows, -1))
    last_off = np.maximum.accumulate(np.where(values <= off_floor, rows, -1))
    return (last_on > last_off) & (last_on >= 0)


def _cross_section_ratio(condition: np.ndarray, universe: np.ndarray) -> np.ndarray:
    numerator = np.sum(condition, axis=1)
    denominator = np.sum(universe, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _lag_1d(values: np.ndarray, periods: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[periods:] = values[:-periods]
    return result

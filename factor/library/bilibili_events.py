"""Research-only completed-bar 2560, candle and volume-ratio events.

The event is aligned to the decision date: output row ``T`` describes the
signal found on the completed bar ``d = T - 1``.  It uses raw (unadjusted)
close, as the source video does not specify an adjustment convention.
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np

from factor.base import FactorDefinition, FactorMetadata


NAME = "Bili2560Events"
VERSION = "bilibili-2560-unadjusted-close-v2-valid-windows"
CANDLE_VERSION = "bilibili-candle-events-v2-doji-ma20"
VOLUME_RATIO_VERSION = "bilibili-volume-ratio-v1-assumed-bounds-2p5-inclusive-5-inclusive"
REQUIRED_FIELDS = ("close", "volume")
HIST_DAYS = 61
CANDLE_NAMES = (
    "BiliLongUpperShadow", "BiliLongLowerShadow", "BiliBigBull",
    "BiliBigBear", "BiliHighDoji", "BiliLowDoji",
)
VOLUME_RATIO_NAMES = ("BiliVolumeRatio25To5Inclusive", "BiliVolumeRatioAbove5")


def _rolling_mean(values: np.ndarray, window: int, *, allow_zero: bool = False) -> np.ndarray:
    """Return complete positive-price/nonnegative-volume trailing means."""
    rows, stocks = values.shape
    finite = np.isfinite(values) & (values >= 0 if allow_zero else values > 0)
    safe = np.where(finite, values, 0.0)
    cumulative = np.concatenate(
        (np.zeros((1, stocks), dtype=np.float64), np.cumsum(safe, axis=0, dtype=np.float64)),
        axis=0,
    )
    counts = np.concatenate(
        (np.zeros((1, stocks), dtype=np.int64), np.cumsum(finite, axis=0, dtype=np.int64)),
        axis=0,
    )
    result = np.full((rows, stocks), np.nan, dtype=np.float64)
    if rows >= window:
        totals = cumulative[window:] - cumulative[:-window]
        valid = counts[window:] - counts[:-window] == window
        result[window - 1 :] = np.divide(
            totals,
            float(window),
            out=np.full(totals.shape, np.nan, dtype=np.float64),
            where=valid,
        )
    return result


def calculate_bilibili_events(panel: dict[str, np.ndarray]) -> np.ndarray:
    """Compute decision-date scores for the 2560 event.

    A score is 1 when both strict upward crosses occurred on completed bar
    ``d=T-1``: close crossed above its 25-day price mean and the 5-day volume
    mean crossed above the 60-day volume mean. Valid non-events score 0;
    unavailable or malformed windows score NaN.
    """
    close = np.asarray(panel["close"], dtype=np.float64)
    volume = np.asarray(panel["volume"], dtype=np.float64)
    if close.ndim != 2 or volume.shape != close.shape:
        raise ValueError("close and volume must be matching date-by-stock matrices")
    rows, stocks = close.shape
    result = np.full((rows, stocks), np.nan, dtype=np.float64)
    if rows <= HIST_DAYS:
        return result

    close_ma25 = _rolling_mean(close, 25)
    volume_ma5 = _rolling_mean(volume, 5, allow_zero=True)
    volume_ma60 = _rolling_mean(volume, 60, allow_zero=True)
    # d=T-1.  The first usable decision row is T=61 (d=60).
    current = slice(HIST_DAYS - 1, rows - 1)
    prior = slice(HIST_DAYS - 2, rows - 2)
    required = (
        np.isfinite(close_ma25[current]) & np.isfinite(close_ma25[prior])
        & np.isfinite(volume_ma5[current]) & np.isfinite(volume_ma5[prior])
        & np.isfinite(volume_ma60[current]) & np.isfinite(volume_ma60[prior])
    )
    signal = (
        (close[current] > close_ma25[current]) & (close[prior] < close_ma25[prior])
        & (volume_ma5[current] > volume_ma60[current])
        & (volume_ma5[prior] < volume_ma60[prior])
    )
    result[HIST_DAYS:] = np.where(required, signal.astype(np.float64), np.nan)
    return result


class Bili2560Events:
    """Research factor for the completed-bar 2560 event."""

    hist_days = HIST_DAYS
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict[str, np.ndarray]) -> np.ndarray:
        return calculate_bilibili_events(panel)


def _candle_valid(panel: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    opening, high, low, close = (np.asarray(panel[name], dtype=np.float64) for name in ("open", "high", "low", "close"))
    if opening.ndim != 2 or any(value.shape != opening.shape for value in (high, low, close)):
        raise ValueError("OHLC must be matching date-by-stock matrices")
    valid = (
        np.isfinite(opening) & (opening > 0) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
        & (low > 0) & (close > 0)
        & (high >= np.maximum(opening, close)) & (low <= np.minimum(opening, close))
    )
    body = np.abs(close - opening)
    upper = high - np.maximum(opening, close)
    lower = np.minimum(opening, close) - low
    return opening, close, body, upper, lower, valid


def _candle_scores(panel: dict[str, np.ndarray], kind: str) -> np.ndarray:
    opening, close, body, upper, lower, valid = _candle_valid(panel)
    rows, stocks = opening.shape
    result = np.full((rows, stocks), np.nan, dtype=np.float64)
    d = slice(0, rows - 1)
    valid_bar = valid[d]
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = body[d] / opening[d]
        upper_ratio = upper[d] / opening[d]
        lower_ratio = lower[d] / opening[d]
    if kind == "upper":
        signal = (body_ratio < .01) & (upper_ratio > .05) & (lower_ratio < .01)
    elif kind == "lower":
        signal = (body_ratio < .01) & (lower_ratio > .05) & (upper_ratio < .01)
    elif kind == "bull":
        signal = (close[d] - opening[d] > .05 * opening[d]) & (upper[d] <= .2 * body[d]) & (lower[d] <= .2 * body[d])
    elif kind == "bear":
        signal = (opening[d] - close[d] > .05 * opening[d]) & (upper[d] <= .2 * body[d]) & (lower[d] <= .2 * body[d])
    elif kind in ("high", "low"):
        # MA20 includes the completed signal bar; no previous MA is used.
        # Positive lower shadow is already required, so multiplication
        # expresses the strict ratio bounds without dividing by zero.
        doji = (
            (body_ratio < .01) & (upper_ratio > .01) & (lower_ratio > .01)
            & (upper[d] > .33 * lower[d]) & (upper[d] < 3.0 * lower[d])
        )
        close_ma = _rolling_mean(close, 20)[d]
        valid_bar = valid_bar & np.isfinite(close_ma)
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation = close[d] / close_ma - 1.0
        signal = doji & (deviation > .05 if kind == "high" else deviation < -.05)
    else:
        raise ValueError(kind)
    result[1:] = np.where(valid_bar, signal.astype(np.float64), np.nan)
    return result


class _CandleFactor:
    pre_ranked = False
    requires_full_history = False
    kind = "upper"
    hist_days = 1

    def calc_batch(self, panel: dict[str, np.ndarray]) -> np.ndarray:
        return _candle_scores(panel, self.kind)


class BiliLongUpperShadow(_CandleFactor):
    kind = "upper"


class BiliLongLowerShadow(_CandleFactor):
    kind = "lower"


class BiliBigBull(_CandleFactor):
    kind = "bull"


class BiliBigBear(_CandleFactor):
    kind = "bear"


class BiliHighDoji(_CandleFactor):
    kind = "high"
    hist_days = 20


class BiliLowDoji(_CandleFactor):
    kind = "low"
    hist_days = 20


class BiliVolumeRatio25To5Inclusive:
    """V[d]/mean(V[d-5:d]) in [2.5,5]; inclusive bounds are an assumption.

    The video does not disclose interval closure. The separate above-five
    event uses the explicitly stated strict >5 threshold. Output T uses
    d=T-1; six finite nonnegative volumes and a positive mean are required.
    """

    hist_days = 6
    pre_ranked = False
    requires_full_history = False
    above_five = False

    def calc_batch(self, panel: dict[str, np.ndarray]) -> np.ndarray:
        volume = np.asarray(panel["volume"], dtype=np.float64)
        if volume.ndim != 2:
            raise ValueError("volume must be a date-by-stock matrix")
        result = np.full(volume.shape, np.nan, dtype=np.float64)
        if len(volume) <= self.hist_days:
            return result
        prior_mean = _rolling_mean(volume, 5, allow_zero=True)[4:-2]
        current = volume[5:-1]
        valid = np.isfinite(current) & (current >= 0) & np.isfinite(prior_mean) & (prior_mean > 0)
        ratio = np.divide(current, prior_mean, out=np.full(current.shape, np.nan), where=valid)
        signal = ratio > 5 if self.above_five else (ratio >= 2.5) & (ratio <= 5)
        result[self.hist_days:] = np.where(valid, signal.astype(np.float64), np.nan)
        return result


class BiliVolumeRatioAbove5(BiliVolumeRatio25To5Inclusive):
    """Completed-day volume ratio strictly above five."""

    above_five = True


_SOURCE_HASH = hashlib.sha256(inspect.getsource(inspect.getmodule(Bili2560Events)).encode()).hexdigest()
_CANDLE_CLASSES = (BiliLongUpperShadow, BiliLongLowerShadow, BiliBigBull, BiliBigBear, BiliHighDoji, BiliLowDoji)
BILIBILI_EVENT_DEFINITION = FactorDefinition(metadata=FactorMetadata(name=NAME, version=VERSION, hist_days=HIST_DAYS, required_fields=REQUIRED_FIELDS, implementation_hash=_SOURCE_HASH), implementation=Bili2560Events)
RESEARCH_FACTOR_DEFINITIONS = (BILIBILI_EVENT_DEFINITION,) + tuple(
    FactorDefinition(metadata=FactorMetadata(name=cls.__name__, version=CANDLE_VERSION, hist_days=cls.hist_days, required_fields=("open", "high", "low", "close"), implementation_hash=_SOURCE_HASH), implementation=cls)
    for cls in _CANDLE_CLASSES
) + tuple(
    FactorDefinition(metadata=FactorMetadata(name=cls.__name__, version=VOLUME_RATIO_VERSION, hist_days=cls.hist_days, required_fields=("volume",), implementation_hash=_SOURCE_HASH), implementation=cls)
    for cls in (BiliVolumeRatio25To5Inclusive, BiliVolumeRatioAbove5)
)

__all__ = [
    "Bili2560Events", *CANDLE_NAMES, *VOLUME_RATIO_NAMES, "BILIBILI_EVENT_DEFINITION",
    "RESEARCH_FACTOR_DEFINITIONS", "calculate_bilibili_events",
]

"""Causal market-trend position overlay shared by backtest and live paths."""

from __future__ import annotations

import numpy as np


def _lag(values: np.ndarray, periods: int = 1) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    if periods < values.shape[0]:
        result[periods:] = values[:-periods]
    return result


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    if values.shape[0] < window:
        return result
    cumulative = np.cumsum(values, axis=0, dtype=np.float64)
    result[window - 1:] = (
        cumulative[window - 1:]
        - np.vstack((np.zeros((1, values.shape[1])), cumulative[:-window]))
    ) / window
    return result


def _apply_recovery_limit(target: np.ndarray, recovery_step: float) -> np.ndarray:
    """De-risk immediately, but rebuild exposure by at most one step per day."""
    result = np.asarray(target, dtype=np.float64).copy()
    for row in range(1, len(result)):
        if target[row] < result[row - 1]:
            result[row] = target[row]
        else:
            result[row] = min(target[row], result[row - 1] + recovery_step)
    return result


def _adjusted_open_return(prev_open, prev_close, current_open, current_pre_close):
    """Corporate-action-safe open-to-open return for selected values."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (prev_close / prev_open) * (current_open / current_pre_close) - 1.0


def market_open_index(data: dict, block_size: int = 256) -> np.ndarray:
    """Equal-weight broad-market Open index, computed once in row blocks."""
    open_price = np.asarray(data["open"], dtype=np.float64)
    close = np.asarray(data["close"], dtype=np.float64)
    pre_close = np.asarray(data["preClose"], dtype=np.float64)
    st_mask = np.asarray(data["st_mask"], dtype=bool)
    market_return = np.zeros(open_price.shape[0], dtype=np.float64)
    for start in range(1, open_price.shape[0], block_size):
        stop = min(start + block_size, open_price.shape[0])
        returns = _adjusted_open_return(
            open_price[start - 1:stop - 1], close[start - 1:stop - 1],
            open_price[start:stop], pre_close[start:stop],
        )
        valid = (
            np.isfinite(returns)
            & np.isfinite(open_price[start:stop])
            & (open_price[start:stop] > 0)
            & ~st_mask[start:stop]
        )
        clipped = np.where(valid, np.clip(returns, -0.20, 0.20), 0.0)
        count = valid.sum(axis=1)
        market_return[start:stop] = np.divide(
            clipped.sum(axis=1), count,
            out=np.zeros(stop - start, dtype=np.float64), where=count > 0,
        )
    return np.cumprod(1.0 + market_return)


def market_completed_index(data: dict, block_size: int = 256) -> np.ndarray:
    """Equal-weight market index known before each row's open.

    Index row ``T`` compounds official completed-day returns only through
    ``T-1``.  It never uses ``open[T] / preClose[T]`` or any T-day HLCVA.
    """
    close = np.asarray(data["close"], dtype=np.float64)
    pre_close = np.asarray(data["preClose"], dtype=np.float64)
    st_mask = np.asarray(data["st_mask"], dtype=bool)
    if close.shape != pre_close.shape or close.shape != st_mask.shape:
        raise ValueError("close, preClose, and st_mask must have matching shapes")

    daily_return = np.full(close.shape[0], np.nan, dtype=np.float64)
    for start in range(0, close.shape[0], block_size):
        stop = min(start + block_size, close.shape[0])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            returns = close[start:stop] / pre_close[start:stop] - 1.0
        valid = (
            np.isfinite(returns)
            & np.isfinite(close[start:stop])
            & (close[start:stop] > 0.0)
            & np.isfinite(pre_close[start:stop])
            & (pre_close[start:stop] > 0.0)
            & ~st_mask[start:stop]
        )
        count = valid.sum(axis=1)
        daily_return[start:stop] = np.divide(
            np.where(valid, returns, 0.0).sum(axis=1),
            count,
            out=np.full(stop - start, np.nan, dtype=np.float64),
            where=count > 0,
        )

    completed = np.cumprod(1.0 + daily_return)
    signal_index = np.full_like(completed, np.nan)
    signal_index[0] = 1.0
    signal_index[1:] = completed[:-1]
    return signal_index


def _continuous_score_multiplier(data: dict, settings: dict,
                                 open_index: np.ndarray | None = None) -> np.ndarray:
    """Map the market Open index, including open[T], to exposure."""
    market_index = market_open_index(data) if open_index is None else np.asarray(open_index)
    momentum_window = int(settings.get("momentum_window", 5))
    ma_window = int(settings.get("ma_window", 20))
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum = market_index / _lag(market_index, momentum_window) - 1.0
        ma = _rolling_mean(market_index[:, None], ma_window)[:, 0]
        ma_ratio = market_index / ma
    return _smooth_health(
        momentum, ma_ratio,
        momentum_center=float(settings.get("momentum_center", -0.05)),
        momentum_scale=float(settings.get("momentum_scale", 0.004)),
        ma_center=float(settings.get("ma_center", 1.0)),
        ma_scale=float(settings.get("ma_scale", 0.012)),
        sharpness=float(settings.get("softmin_sharpness", 4.0)),
        slope=float(settings.get("slope", 2.0)),
        floor=float(settings.get("floor", 0.15)),
        ceiling=float(settings.get("ceiling", 1.0)),
        strict_history=bool(settings.get("strict_history", False)),
        warmup_multiplier=float(settings.get("strict_warmup_multiplier", 1.0)),
    )


def _smooth_health(momentum: np.ndarray, ma_ratio: np.ndarray, *,
                   momentum_center: float, momentum_scale: float,
                   ma_center: float, ma_scale: float,
                   sharpness: float, slope: float,
                   floor: float, ceiling: float = 1.0,
                   strict_history: bool = False,
                   warmup_multiplier: float = 1.0) -> np.ndarray:
    if strict_history:
        if momentum_scale <= 0.0 or ma_scale <= 0.0:
            raise ValueError("strict trend scales must be positive")
        if sharpness <= 0.0 or slope <= 0.0:
            raise ValueError("strict trend sharpness and slope must be positive")
        if not 0.0 <= floor <= ceiling <= 1.0:
            raise ValueError("strict trend floor and ceiling must be ordered in [0, 1]")
        if not floor <= warmup_multiplier <= ceiling:
            raise ValueError("strict trend warmup multiplier must lie within bounds")

    momentum_health = (momentum - momentum_center) / momentum_scale
    ma_health = (ma_ratio - ma_center) / ma_scale
    with np.errstate(invalid="ignore"):
        weak = -np.logaddexp(-sharpness * momentum_health,
                             -sharpness * ma_health) / sharpness
    if strict_history:
        valid = np.isfinite(weak)
        if not np.any(valid):
            raise ValueError("strict trend inputs never become finite")
        first_valid = int(np.flatnonzero(valid)[0])
        if not np.all(valid[first_valid:]):
            raise ValueError("strict trend inputs contain invalid data after warmup")
        result = np.full_like(weak, warmup_multiplier, dtype=np.float64)
        health = 1.0 / (
            1.0 + np.exp(-np.clip(slope * weak[first_valid:], -40.0, 40.0))
        )
        result[first_valid:] = floor + (ceiling - floor) * health
        return result

    weak = np.nan_to_num(weak, nan=8.0, posinf=8.0, neginf=-8.0)
    health = 1.0 / (1.0 + np.exp(-np.clip(slope * weak, -40.0, 40.0)))
    return floor + (ceiling - floor) * health


def strategy_trend_multipliers(strategy_open_index: np.ndarray,
                               settings: dict) -> np.ndarray:
    """Return exposure from the strategy-target Open index."""
    strategy_index = np.asarray(strategy_open_index, dtype=np.float64)
    momentum_window = int(settings.get("strategy_momentum_window", 3))
    ma_window = int(settings.get("strategy_ma_window", 20))
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum = strategy_index / _lag(strategy_index, momentum_window) - 1.0
        ma = _rolling_mean(strategy_index[:, None], ma_window)[:, 0]
        ma_ratio = strategy_index / ma
    return _smooth_health(
        momentum, ma_ratio,
        momentum_center=float(settings.get("strategy_momentum_center", -0.044)),
        momentum_scale=float(settings.get("strategy_momentum_scale", 0.0075)),
        ma_center=float(settings.get("strategy_ma_center", 1.014)),
        ma_scale=float(settings.get("strategy_ma_scale", 0.009)),
        sharpness=float(settings.get("strategy_softmin_sharpness", 4.0)),
        slope=float(settings.get("strategy_slope", 2.0)),
        floor=float(settings.get("floor", 0.03)),
        ceiling=float(settings.get("ceiling", 1.0)),
        strict_history=bool(settings.get("strict_history", False)),
        warmup_multiplier=float(settings.get("strict_warmup_multiplier", 1.0)),
    )


def _quantize_dual_exposure(
    exposure: np.ndarray,
    settings: dict,
) -> np.ndarray:
    """Optionally snap a mixed dual exposure to a zero-anchored grid."""
    if "exposure_step" not in settings:
        return exposure
    raw_step = settings["exposure_step"]
    if (
        isinstance(raw_step, (bool, np.bool_))
        or not isinstance(
            raw_step,
            (int, float, np.integer, np.floating),
        )
    ):
        raise ValueError(
            "trend_risk_overlay exposure_step must be 0 or finite in (0, 1]"
        )
    step = float(raw_step)
    if not np.isfinite(step) or step < 0.0 or step > 1.0:
        raise ValueError(
            "trend_risk_overlay exposure_step must be 0 or finite in (0, 1]"
        )
    if step == 0.0:
        return exposure

    floor = float(settings.get("floor", 0.03))
    ceiling = float(settings.get("ceiling", 1.0))
    if (
        not np.isfinite(floor)
        or not np.isfinite(ceiling)
        or not 0.0 <= floor <= ceiling <= 1.0
    ):
        raise ValueError(
            "quantized dual trend floor and ceiling must be ordered in [0, 1]"
        )
    values = np.asarray(exposure, dtype=np.float64)
    quantized = np.floor(values / step + 0.5) * step
    return np.clip(quantized, floor, ceiling)


def _blend_dual_exposure(
    broad: np.ndarray,
    strategy: np.ndarray,
    settings: dict,
) -> np.ndarray:
    """Blend dual signals once, then apply the shared opt-in quantizer."""
    weight = float(settings.get("strategy_weight", 0.60))
    mixed = weight * strategy + (1.0 - weight) * broad
    return _quantize_dual_exposure(mixed, settings)


def dual_trend_multipliers(market_index: np.ndarray,
                           strategy_index: np.ndarray,
                           settings: dict) -> np.ndarray:
    """Blend broad-market and strategy-target trends."""
    broad = _continuous_score_multiplier({}, settings, open_index=market_index)
    strategy = strategy_trend_multipliers(strategy_index, settings)
    return _blend_dual_exposure(broad, strategy, settings)


def strategy_open_index(data: dict, daily_topn: list[list[str]],
                        date_indices: np.ndarray | list[int],
                        stock_indices: dict[str, int]) -> np.ndarray:
    """Build an equal-weight strategy-target index sampled at each open.

    The T open-to-open return uses the names selected at T-1.  No T-day
    high/low/close/volume/amount participates in the signal.
    """
    rows = np.asarray(date_indices, dtype=np.intp)
    returns = np.zeros(len(rows), dtype=np.float64)
    if len(rows) < 2:
        return np.ones(len(rows), dtype=np.float64)

    width = max((len(codes) for codes in daily_topn[:-1]), default=0)
    if width == 0:
        return np.ones(len(rows), dtype=np.float64)
    cols = np.zeros((len(rows) - 1, width), dtype=np.intp)
    selected = np.zeros_like(cols, dtype=bool)
    for i, codes in enumerate(daily_topn[:-1]):
        mapped = [stock_indices[code] for code in codes if code in stock_indices]
        if mapped:
            cols[i, :len(mapped)] = mapped
            selected[i, :len(mapped)] = True

    prev_rows = rows[:-1, None]
    current_rows = rows[1:, None]
    values = _adjusted_open_return(
        np.asarray(data["open"])[prev_rows, cols],
        np.asarray(data["close"])[prev_rows, cols],
        np.asarray(data["open"])[current_rows, cols],
        np.asarray(data["preClose"])[current_rows, cols],
    )
    valid = selected & np.isfinite(values)
    clipped = np.where(valid, np.clip(values, -0.20, 0.20), 0.0)
    count = valid.sum(axis=1)
    returns[1:] = np.divide(
        clipped.sum(axis=1), count,
        out=np.zeros(len(rows) - 1, dtype=np.float64), where=count > 0,
    )
    return np.cumprod(1.0 + returns)


def strategy_completed_index(data: dict, daily_topn: list[list[str]],
                             date_indices: np.ndarray | list[int],
                             stock_indices: dict[str, int]) -> np.ndarray:
    """Build the target portfolio index known before each next open.

    Target names selected at open ``T`` contribute only their official
    ``close[T] / preClose[T] - 1`` return to index row ``T+1``.  Thus the
    multiplier used at row T contains no T-day HLCVA or gap return.  A day
    with no legal target is represented as cash with a zero completed return.
    """
    rows = np.asarray(date_indices, dtype=np.intp)
    if len(rows) == 0:
        return np.empty(0, dtype=np.float64)
    if len(daily_topn) != len(rows):
        raise ValueError(
            "strict completed strategy index requires one target list per row"
        )
    if len(rows) == 1:
        return np.ones(1, dtype=np.float64)

    width = max((len(codes) for codes in daily_topn[:-1]), default=0)
    if width == 0:
        return np.ones(len(rows), dtype=np.float64)
    cols = np.zeros((len(rows) - 1, width), dtype=np.intp)
    selected = np.zeros_like(cols, dtype=bool)
    for i, codes in enumerate(daily_topn[:-1]):
        mapped = [stock_indices[code] for code in codes if code in stock_indices]
        if len(mapped) != len(codes):
            raise ValueError("strict completed strategy index found an unmapped target")
        if mapped:
            cols[i, :len(mapped)] = mapped
            selected[i, :len(mapped)] = True

    completed_rows = rows[:-1, None]
    close = np.asarray(data["close"], dtype=np.float64)[completed_rows, cols]
    pre_close = np.asarray(data["preClose"], dtype=np.float64)[completed_rows, cols]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = close / pre_close - 1.0
    valid = (
        selected
        & np.isfinite(values)
        & np.isfinite(close)
        & (close > 0.0)
        & np.isfinite(pre_close)
        & (pre_close > 0.0)
    )
    selected_count = selected.sum(axis=1)
    valid_count = valid.sum(axis=1)
    if np.any(valid_count != selected_count):
        raise ValueError("strict completed strategy index found invalid target returns")
    daily_return = np.divide(
        np.where(valid, values, 0.0).sum(axis=1),
        selected_count,
        out=np.zeros(len(rows) - 1, dtype=np.float64),
        where=selected_count > 0,
    )
    if not np.isfinite(daily_return).all():
        raise ValueError("strict completed strategy index produced invalid returns")

    signal_index = np.ones(len(rows), dtype=np.float64)
    signal_index[1:] = np.cumprod(1.0 + daily_return)
    return signal_index


def build_strategy_topn_path(
    *, data: dict, all_scores: dict, valid_dates: list,
    date_indices: np.ndarray | list[int], valid_stocks: list[str],
    stock_indices: dict[str, int], weights: dict, buy_n: int,
    limit_up_protection: bool = False,
    filter_masks: dict | None = None,
) -> list[list[str]]:
    """Build the legal daily target path without simulating an account."""
    from core.legality import LegalityChecker
    from core.scoring import (
        compute_weighted_score_matrix, select_topn_legal_from_scores,
    )

    indices = np.asarray(date_indices, dtype=np.intp)
    cols = np.asarray([stock_indices[stock] for stock in valid_stocks], dtype=np.intp)
    scores = compute_weighted_score_matrix(all_scores, indices, cols, weights)
    if filter_masks:
        allowed = np.logical_and.reduce([
            np.asarray(mask)[np.ix_(indices, cols)] for mask in filter_masks.values()
        ])
        scores[~allowed] = -np.inf

    checker = LegalityChecker(
        data, stock_indices, limit_up_protection=limit_up_protection,
    )
    day_open = np.asarray(data["open"])
    result = []
    for i, (dt, row) in enumerate(zip(valid_dates, indices)):
        buy, _, _ = select_topn_legal_from_scores(
            scores[i], valid_stocks, cols, buy_n, buy_n,
            checker, int(row), dt.date(), day_open[row],
        )
        result.append(buy)
    return result


def compute_dual_trend_multipliers(
    *, data: dict, all_scores: dict, valid_dates: list,
    date_indices: np.ndarray | list[int], valid_stocks: list[str],
    stock_indices: dict[str, int], weights: dict, buy_n: int,
    settings: dict, limit_up_protection: bool = False,
    filter_masks: dict | None = None,
) -> np.ndarray:
    """Build target Open trend and return exposure without a shadow account."""
    indices = np.asarray(date_indices, dtype=np.intp)
    targets = build_strategy_topn_path(
        data=data, all_scores=all_scores, valid_dates=valid_dates,
        date_indices=indices, valid_stocks=valid_stocks,
        stock_indices=stock_indices, weights=weights, buy_n=buy_n,
        limit_up_protection=limit_up_protection, filter_masks=filter_masks,
    )
    strategy_index = strategy_open_index(data, targets, indices, stock_indices)
    full_market_index = data.get("_market_open_index")
    if full_market_index is None:
        full_market_index = market_open_index(data)
    broad = _continuous_score_multiplier(
        {}, settings, open_index=np.asarray(full_market_index),
    )[indices]
    strategy = strategy_trend_multipliers(strategy_index, settings)
    return _blend_dual_exposure(broad, strategy, settings)


def compute_dual_completed_trend_multipliers(
    *, data: dict, all_scores: dict, valid_dates: list,
    date_indices: np.ndarray | list[int], valid_stocks: list[str],
    stock_indices: dict[str, int], weights: dict, buy_n: int,
    settings: dict, limit_up_protection: bool = False,
    filter_masks: dict | None = None,
) -> np.ndarray:
    """Return dual-trend exposure from information completed before each open."""
    indices = np.asarray(date_indices, dtype=np.intp)
    targets = build_strategy_topn_path(
        data=data, all_scores=all_scores, valid_dates=valid_dates,
        date_indices=indices, valid_stocks=valid_stocks,
        stock_indices=stock_indices, weights=weights, buy_n=buy_n,
        limit_up_protection=limit_up_protection, filter_masks=filter_masks,
    )
    strategy_index = strategy_completed_index(
        data, targets, indices, stock_indices,
    )
    full_market_index = data.get("_market_completed_index")
    if full_market_index is None:
        full_market_index = market_completed_index(data)
    broad = _continuous_score_multiplier(
        {}, settings, open_index=np.asarray(full_market_index),
    )[indices]
    strategy = strategy_trend_multipliers(strategy_index, settings)
    return _blend_dual_exposure(broad, strategy, settings)


def compute_configured_timing_multipliers(
    *, data: dict, all_scores: dict, valid_dates: list,
    date_indices: np.ndarray | list[int], valid_stocks: list[str],
    stock_indices: dict[str, int], config: dict,
    filter_masks: dict | None = None,
    base_multipliers: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build the configured trend overlay and combine it with base timing."""
    settings = config.get("trend_risk_overlay") or {}
    if not settings.get("enabled", False):
        return base_multipliers

    indices = np.asarray(date_indices, dtype=np.intp)
    mode = str(settings.get("mode", "discrete")).lower()
    if mode == "dual_strategy":
        overlay = compute_dual_trend_multipliers(
            data=data, all_scores=all_scores, valid_dates=valid_dates,
            date_indices=indices, valid_stocks=valid_stocks,
            stock_indices=stock_indices, weights=config["weights"],
            buy_n=config["buy_n"], settings=settings,
            limit_up_protection=config.get("limit_up_protection", False),
            filter_masks=filter_masks,
        )
    elif mode == "dual_completed":
        overlay = compute_dual_completed_trend_multipliers(
            data=data, all_scores=all_scores, valid_dates=valid_dates,
            date_indices=indices, valid_stocks=valid_stocks,
            stock_indices=stock_indices, weights=config["weights"],
            buy_n=config["buy_n"], settings=settings,
            limit_up_protection=config.get("limit_up_protection", False),
            filter_masks=filter_masks,
        )
    else:
        full_overlay = trend_overlay_multipliers(data, config)
        overlay = None if full_overlay is None else full_overlay[indices]

    if overlay is None:
        return base_multipliers
    if base_multipliers is None:
        return overlay
    base = np.asarray(base_multipliers, dtype=np.float64)
    if len(base) != len(overlay):
        raise ValueError("base timing and trend overlay lengths must match")
    return base * overlay


def trend_overlay_multipliers(data: dict, config: dict) -> np.ndarray | None:
    """Return one causal multiplier per runtime row, or ``None`` when disabled.

    The state at row T uses the Open index through ``open[T]``. T-day
    high/low/close/volume/amount cannot affect it.
    """
    settings = config.get("trend_risk_overlay")
    if not settings or not settings.get("enabled", False):
        return None

    mode = str(settings.get("mode", "discrete")).lower()
    if mode == "continuous_score":
        return _continuous_score_multiplier(data, settings)
    if mode in {"dual_strategy", "dual_completed"}:
        raise ValueError(f"{mode} timing requires a strategy target path")

    momentum_window = int(settings.get("momentum_window", 20))
    ma_window = int(settings.get("ma_window", 20))
    momentum_floor = float(settings.get("momentum_floor", 0.0))
    ma_ratio_floor = float(settings.get("ma_ratio_floor", 1.0))
    risk_floor = float(settings.get("risk_floor", 0.3))
    recovery_step = float(settings.get("recovery_step", 1.0))
    risk_rule = str(settings.get("risk_rule", "and")).lower()
    if momentum_window < 1 or ma_window < 2:
        raise ValueError("trend_risk_overlay windows must be positive")
    if not 0.0 <= risk_floor <= 1.0:
        raise ValueError("trend_risk_overlay risk_floor must be in [0, 1]")
    if not 0.0 < recovery_step <= 1.0:
        raise ValueError("trend_risk_overlay recovery_step must be in (0, 1]")
    if risk_rule not in {"and", "or"}:
        raise ValueError("trend_risk_overlay risk_rule must be 'and' or 'or'")

    market_index = market_open_index(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum = market_index / _lag(market_index, momentum_window) - 1.0
        ma = _rolling_mean(market_index[:, None], ma_window)[:, 0]
        ma_ratio = market_index / ma
    momentum_risk = np.isfinite(momentum) & (momentum <= momentum_floor)
    ma_risk = np.isfinite(ma_ratio) & (ma_ratio <= ma_ratio_floor)
    # ``and`` preserves the original conservative overlay. ``or`` is useful
    # for a deliberately simple insurance rule: either a sharp short-term
    # break or a sustained moving-average break is enough to reduce exposure.
    risk_off = (momentum_risk & ma_risk) if risk_rule == "and" else (momentum_risk | ma_risk)
    target = np.where(risk_off, risk_floor, 1.0).astype(np.float64)
    return _apply_recovery_limit(target, recovery_step)


def trend_overlay_multiplier_for_row(data: dict, row: int, config: dict) -> float:
    multipliers = trend_overlay_multipliers(data, config)
    if multipliers is None:
        return 1.0
    value = multipliers[row]
    return float(value) if np.isfinite(value) else 1.0

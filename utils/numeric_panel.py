"""Shared matrix validation, calendar validation and causal row shift."""
from datetime import date, datetime
import numpy as np

def _numeric_matrix(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must have a real numeric dtype")
    return array

def _validated_trade_dates(values: object, rows: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("trade_dates must be one-dimensional")
    if len(raw) != rows:
        raise ValueError("trade_dates length must match the price panel rows")
    if raw.dtype.kind not in "MOSU":
        raise ValueError(
            "trade_dates must contain datetime64 or date-like values"
        )
    if raw.dtype.kind == "O" and any(
        not isinstance(
            value,
            (date, datetime, np.datetime64, str, np.str_),
        )
        for value in raw
    ):
        raise ValueError(
            "trade_dates must contain datetime64 or date-like values"
        )
    try:
        dates = raw.astype("datetime64[D]")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "trade_dates must contain valid calendar dates"
        ) from exc
    if np.isnat(dates).any():
        raise ValueError("trade_dates must not contain NaT")
    if len(dates) > 1 and np.any(dates[1:] <= dates[:-1]):
        raise ValueError("trade_dates must be strictly increasing")
    return dates

def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result

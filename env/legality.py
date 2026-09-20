"""Causal, vectorised A-share T-open trading legality.

This is the single legality implementation shared by planning, observation,
backtest and compatibility entry points. Only T-open data is accepted:
``open[T]``, official ``preClose[T]``, issue price, listing age, and the PIT
ST flag.
The point-in-time delisted mask is also mandatory so stale historical bars can
never make a written-off security buyable again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


_EPS = 0.001

_IPO_44_START = np.datetime64("2014-01-01", "D")
_KCB_OPEN = np.datetime64("2019-07-22", "D")
_CYB_REG = np.datetime64("2020-08-24", "D")
_MB_REG = np.datetime64("2023-04-10", "D")
_MB_ST_10_START = np.datetime64("2026-07-06", "D")

BOARD_MAIN = 0
BOARD_CYB = 1
BOARD_KCB = 2
BOARD_BJ = 3

REASON_OK = 0
REASON_OK_NO_DAILY_LIMIT = 1
REASON_NOT_LISTED = 2
REASON_SUSPENDED_OR_MISSING_OPEN = 3
REASON_MISSING_PRECLOSE = 4
REASON_LIMIT_UP = 5
REASON_IPO_OPEN_LIMIT = 6
REASON_LIMIT_DOWN = 7
REASON_LIMIT_UP_PROTECTED = 8
REASON_DELISTED = 9
LEGALITY_REASON_TEXT = (
    "ok",
    "ok_no_daily_limit",
    "not_listed",
    "suspended_or_missing_open",
    "missing_preclose",
    "limit_up",
    "ipo_open_limit",
    "limit_down",
    "limit_up_protected",
    "delisted",
)


def _floor_2(values: ArrayLike) -> NDArray[np.float64]:
    """Round a limit-up price down to cents (the conservative boundary)."""

    return np.floor(np.asarray(values, dtype=np.float64) * 100.0 + 1e-9) / 100.0


def _ceil_2(values: ArrayLike) -> NDArray[np.float64]:
    """Round a limit-down price up to cents (the conservative boundary)."""

    return np.ceil(np.asarray(values, dtype=np.float64) * 100.0 - 1e-9) / 100.0


def classify_board_types(stock_codes: Sequence[str]) -> NDArray[np.int8]:
    """Classify a complete stock vocabulary without a per-stock Python loop."""

    codes = np.asarray(stock_codes, dtype="U32")
    if codes.ndim != 1 or codes.size == 0:
        raise ValueError("stock_codes must be a non-empty one-dimensional sequence")
    bare = np.char.partition(codes, ".")[:, 0]
    cyb = np.char.startswith(bare, "300") | np.char.startswith(bare, "301")
    kcb = np.char.startswith(bare, "688") | np.char.startswith(bare, "689")
    bj = (
        np.char.startswith(bare, "43")
        | np.char.startswith(bare, "83")
        | np.char.startswith(bare, "87")
        | np.char.startswith(bare, "92")
    )
    result = np.full(codes.shape, BOARD_MAIN, dtype=np.int8)
    result[cyb] = BOARD_CYB
    result[kcb] = BOARD_KCB
    result[bj] = BOARD_BJ
    return np.ascontiguousarray(result)


def ordinary_limit_ratios(board_types: ArrayLike) -> NDArray[np.float64]:
    """Return post-reform ordinary daily limit ratios for board codes."""

    boards = np.asarray(board_types, dtype=np.int8)
    if np.any(~np.isin(boards, (BOARD_MAIN, BOARD_CYB, BOARD_KCB, BOARD_BJ))):
        raise ValueError("board_types contains an unknown board code")
    return np.where(
        boards == BOARD_BJ,
        0.30,
        np.where((boards == BOARD_CYB) | (boards == BOARD_KCB), 0.20, 0.10),
    )


@dataclass(frozen=True)
class TradeLegalityResult:
    """Vectorised masks, reasons, and conservative exchange boundaries."""

    buy_allowed: NDArray[np.bool_]
    sell_allowed: NDArray[np.bool_] | None
    buy_reason_codes: NDArray[np.uint8] | None
    sell_reason_codes: NDArray[np.uint8] | None
    limit_up: NDArray[np.bool_] | None
    limit_down: NDArray[np.bool_] | None
    up_limit_prices: NDArray[np.float64] | None
    down_limit_prices: NDArray[np.float64] | None
    daily_limit_exempt: NDArray[np.bool_] | None
    board_types: NDArray[np.int8] | None

    @property
    def buy_reasons(self) -> NDArray[np.str_] | None:
        return _reason_text_array(self.buy_reason_codes)

    @property
    def sell_reasons(self) -> NDArray[np.str_] | None:
        return _reason_text_array(self.sell_reason_codes)


def _reason_text_array(
    reason_codes: NDArray[np.uint8] | None,
) -> NDArray[np.str_] | None:
    if reason_codes is None:
        return None
    return np.asarray(LEGALITY_REASON_TEXT, dtype="U32")[reason_codes]


def legality_reason_text(reason_code: int) -> str:
    if not 0 <= int(reason_code) < len(LEGALITY_REASON_TEXT):
        raise ValueError("unknown legality reason code")
    return LEGALITY_REASON_TEXT[int(reason_code)]


def _decision_date_column(
    decision_date: date | str | np.datetime64 | ArrayLike,
    leading_shape: tuple[int, ...],
) -> NDArray[np.datetime64]:
    dates = np.asarray(decision_date, dtype="datetime64[D]")
    if dates.ndim == 0:
        dates = np.broadcast_to(dates, leading_shape)
    else:
        try:
            dates = np.broadcast_to(dates, leading_shape)
        except ValueError as exc:
            raise ValueError(
                "decision_date must be scalar or match the price leading dimensions"
            ) from exc
    if np.isnat(dates).any():
        raise ValueError("decision_date must not contain NaT")
    return dates[..., None]


def _broadcast_numeric(
    name: str,
    values: ArrayLike,
    shape: tuple[int, ...],
    dtype,
):
    try:
        return np.broadcast_to(np.asarray(values, dtype=dtype), shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc


def evaluate_trade_legality(
    *,
    decision_date: date | str | np.datetime64 | ArrayLike,
    stock_codes: Sequence[str],
    listing_age: ArrayLike,
    open_prices: ArrayLike,
    preclose_prices: ArrayLike,
    issue_prices: ArrayLike,
    st_mask: ArrayLike,
    delisted_mask: ArrayLike,
    limit_up_protection: bool = False,
    precomputed_board_types: ArrayLike | None = None,
    diagnostics: bool = True,
) -> TradeLegalityResult:
    """Evaluate buy and sell legality for one row or a broadcast price panel.

    Price arrays use ``[..., N]`` with the stock vocabulary on the final axis.
    ``decision_date`` may be scalar or match the leading dimensions. Every
    branch is vectorised; the returned reason array has one reason per stock.
    """

    codes = np.asarray(stock_codes, dtype="U32")
    if codes.ndim != 1 or codes.size == 0:
        raise ValueError("stock_codes must be non-empty and one-dimensional")
    opens = np.asarray(open_prices, dtype=np.float64)
    if opens.ndim < 1 or opens.shape[-1] != len(codes):
        raise ValueError("open_prices must have stock_codes on its final axis")
    shape = opens.shape
    precloses = _broadcast_numeric(
        "preclose_prices", preclose_prices, shape, np.float64
    )
    issues = _broadcast_numeric("issue_prices", issue_prices, shape, np.float64)
    st = _broadcast_numeric("st_mask", st_mask, shape, np.bool_)
    delisted = _broadcast_numeric("delisted_mask", delisted_mask, shape, np.bool_)
    ages = _broadcast_numeric("listing_age", listing_age, shape, np.int32)
    dates = _decision_date_column(decision_date, shape[:-1])

    if precomputed_board_types is None:
        boards_row = classify_board_types(codes)
    else:
        boards_row = np.asarray(precomputed_board_types, dtype=np.int8)
        if boards_row.shape != (len(codes),):
            raise ValueError("precomputed_board_types must have shape [N]")
        ordinary_limit_ratios(boards_row)
    boards = np.broadcast_to(boards_row, shape)

    ratios = np.broadcast_to(ordinary_limit_ratios(boards_row), shape).copy()
    ratios[(boards == BOARD_CYB) & (dates < _CYB_REG)] = 0.10
    main_st_ratio = np.where(dates < _MB_ST_10_START, 0.05, 0.10)
    cyb_st_ratio = np.where(dates < _CYB_REG, 0.05, 0.20)
    st_ratios = np.where(
        boards == BOARD_CYB,
        cyb_st_ratio,
        np.where(
            boards == BOARD_KCB,
            0.20,
            np.where(boards == BOARD_BJ, 0.30, main_st_ratio),
        ),
    )
    ratios = np.where(st, st_ratios, ratios)

    listed = ages >= 0
    first_day = ages == 0
    exempt = (boards == BOARD_BJ) & first_day
    exempt |= (
        (boards == BOARD_KCB)
        & (dates >= _KCB_OPEN)
        & (ages >= 0)
        & (ages <= 4)
    )
    exempt |= (
        (boards == BOARD_CYB)
        & (dates >= _CYB_REG)
        & (ages >= 0)
        & (ages <= 4)
    )
    exempt |= (
        (boards == BOARD_MAIN)
        & (dates >= _MB_REG)
        & (ages >= 0)
        & (ages <= 4)
    )
    exempt |= first_day & (dates < _IPO_44_START) & (boards != BOARD_BJ)

    old_ipo_first = first_day & (dates >= _IPO_44_START) & ~exempt
    valid_issue = np.isfinite(issues) & (issues > 0.0)
    limit_reference = np.where(first_day & valid_issue, issues, precloses)
    valid_preclose = np.isfinite(limit_reference) & (limit_reference > 0.0)
    ratios = np.where(old_ipo_first, 0.44, ratios)
    has_daily_limit = valid_preclose & ~exempt
    up_limit_prices = np.where(
        has_daily_limit,
        _floor_2(limit_reference * (1.0 + ratios)),
        np.nan,
    )
    valid_open = np.isfinite(opens) & (opens > 0.0)
    active = listed & ~delisted & valid_open
    limit_up = active & has_daily_limit & (opens >= up_limit_prices - _EPS)
    ipo_open_limit = _floor_2(limit_reference * 1.20)
    ipo_open_blocked = (
        active
        & old_ipo_first
        & valid_preclose
        & ~limit_up
        & (opens >= ipo_open_limit - _EPS)
    )
    resolved = exempt | valid_preclose
    buy_allowed = active & resolved & ~limit_up & ~ipo_open_blocked
    if not diagnostics:
        return TradeLegalityResult(
            buy_allowed=np.ascontiguousarray(buy_allowed),
            sell_allowed=None,
            buy_reason_codes=None,
            sell_reason_codes=None,
            limit_up=None,
            limit_down=None,
            up_limit_prices=None,
            down_limit_prices=None,
            daily_limit_exempt=None,
            board_types=None,
        )

    down_limit_prices = np.where(
        has_daily_limit,
        _ceil_2(limit_reference * (1.0 - ratios)),
        np.nan,
    )
    limit_down = active & has_daily_limit & (opens <= down_limit_prices + _EPS)
    sell_allowed = active & resolved & ~limit_down
    if limit_up_protection:
        sell_allowed &= ~limit_up

    buy_reason_codes = np.full(shape, REASON_OK, dtype=np.uint8)
    sell_reason_codes = np.full(shape, REASON_OK, dtype=np.uint8)
    buy_reason_codes[active & exempt] = REASON_OK_NO_DAILY_LIMIT
    sell_reason_codes[active & exempt] = REASON_OK_NO_DAILY_LIMIT
    missing_preclose = active & ~exempt & ~valid_preclose
    buy_reason_codes[missing_preclose] = REASON_MISSING_PRECLOSE
    sell_reason_codes[missing_preclose] = REASON_MISSING_PRECLOSE
    buy_reason_codes[limit_up] = REASON_LIMIT_UP
    buy_reason_codes[ipo_open_blocked] = REASON_IPO_OPEN_LIMIT
    sell_reason_codes[limit_down] = REASON_LIMIT_DOWN
    if limit_up_protection:
        sell_reason_codes[limit_up & ~limit_down] = REASON_LIMIT_UP_PROTECTED
    suspended = listed & ~delisted & ~valid_open
    buy_reason_codes[suspended] = REASON_SUSPENDED_OR_MISSING_OPEN
    sell_reason_codes[suspended] = REASON_SUSPENDED_OR_MISSING_OPEN
    buy_reason_codes[~listed] = REASON_NOT_LISTED
    sell_reason_codes[~listed] = REASON_NOT_LISTED
    buy_reason_codes[delisted & listed] = REASON_DELISTED
    sell_reason_codes[delisted & listed] = REASON_DELISTED

    return TradeLegalityResult(
        buy_allowed=np.ascontiguousarray(buy_allowed),
        sell_allowed=np.ascontiguousarray(sell_allowed),
        buy_reason_codes=np.ascontiguousarray(buy_reason_codes),
        sell_reason_codes=np.ascontiguousarray(sell_reason_codes),
        limit_up=np.ascontiguousarray(limit_up),
        limit_down=np.ascontiguousarray(limit_down),
        up_limit_prices=np.ascontiguousarray(up_limit_prices),
        down_limit_prices=np.ascontiguousarray(down_limit_prices),
        daily_limit_exempt=np.ascontiguousarray(exempt),
        board_types=np.ascontiguousarray(boards),
    )


def evaluate_buy_legality_mask(
    *,
    decision_date: date | str | np.datetime64 | ArrayLike,
    stock_codes: Sequence[str],
    listing_age: ArrayLike,
    open_prices: ArrayLike,
    preclose_prices: ArrayLike,
    issue_prices: ArrayLike,
    st_mask: ArrayLike,
    delisted_mask: ArrayLike,
    precomputed_board_types: ArrayLike | None = None,
    chunk_rows: int = 64,
) -> NDArray[np.bool_]:
    """Return only hard-buy legality with bounded full-panel peak memory."""

    opens = np.asarray(open_prices)
    if opens.ndim == 1:
        return evaluate_trade_legality(
            decision_date=decision_date,
            stock_codes=stock_codes,
            listing_age=listing_age,
            open_prices=opens,
            preclose_prices=preclose_prices,
            issue_prices=issue_prices,
            st_mask=st_mask,
            delisted_mask=delisted_mask,
            precomputed_board_types=precomputed_board_types,
            diagnostics=False,
        ).buy_allowed
    if opens.ndim != 2:
        raise ValueError("panel open_prices must have shape [D, N]")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive int")
    row_count = opens.shape[0]
    dates = np.asarray(decision_date)
    ages = np.asarray(listing_age)
    precloses = np.asarray(preclose_prices)
    st = np.asarray(st_mask)
    delisted = np.asarray(delisted_mask)
    issues = np.asarray(issue_prices)
    if (
        ages.shape != opens.shape
        or precloses.shape != opens.shape
        or st.shape != opens.shape
        or delisted.shape != opens.shape
    ):
        raise ValueError(
            "listing_age, preclose_prices, st_mask, and delisted_mask must match [D, N]"
        )
    if dates.ndim not in (0, 1) or (dates.ndim == 1 and dates.shape != (row_count,)):
        raise ValueError("decision_date must be scalar or have shape [D]")
    if issues.shape not in ((opens.shape[-1],), opens.shape):
        raise ValueError("issue_prices must have shape [N] or [D, N]")
    boards = (
        classify_board_types(stock_codes)
        if precomputed_board_types is None
        else np.asarray(precomputed_board_types, dtype=np.int8)
    )
    if boards.shape != (opens.shape[-1],):
        raise ValueError("precomputed_board_types must have shape [N]")
    ordinary_limit_ratios(boards)

    result = np.empty(opens.shape, dtype=np.bool_)
    for start in range(0, row_count, chunk_rows):
        stop = min(start + chunk_rows, row_count)
        chunk = evaluate_trade_legality(
            decision_date=decision_date if dates.ndim == 0 else dates[start:stop],
            stock_codes=stock_codes,
            listing_age=ages[start:stop],
            open_prices=opens[start:stop],
            preclose_prices=precloses[start:stop],
            issue_prices=issues if issues.ndim == 1 else issues[start:stop],
            st_mask=st[start:stop],
            delisted_mask=delisted[start:stop],
            precomputed_board_types=boards,
            diagnostics=False,
        )
        result[start:stop] = chunk.buy_allowed
    return result


def compute_limit_up_matrix(data):
    """Return full-panel limit-up prices using the authoritative rule engine."""

    opens = np.asarray(data["open"])
    if opens.ndim != 2:
        raise ValueError("open must have shape [D, N]")
    ages = np.asarray(data["listing_age"], dtype=np.int32)
    dates = np.asarray(data["trade_dates"], dtype="datetime64[D]")
    precloses = np.asarray(data["preClose"])
    st = np.asarray(data["st_mask"])
    delisted = np.asarray(data["delisted_mask"], dtype=np.bool_)
    codes = tuple(str(code) for code in data["stock_codes"])
    issue_dates = np.asarray(data["issue_date"], dtype="datetime64[D]")
    for name, values in (
        ("listing_age", ages),
        ("preClose", precloses),
        ("st_mask", st),
        ("delisted_mask", delisted),
    ):
        if values.shape != opens.shape:
            raise ValueError(f"{name} must match open shape {opens.shape}")
    if dates.shape != (opens.shape[0],):
        raise ValueError("trade_dates must match open date axis")
    if len(codes) != opens.shape[1]:
        raise ValueError("stock_codes must match open stock axis")
    if issue_dates.shape != (opens.shape[1],):
        raise ValueError("issue_date must match open stock axis")
    boards = classify_board_types(codes)
    output = np.empty(opens.shape, dtype=np.float64)
    for start in range(0, len(dates), 64):
        stop = min(start + 64, len(dates))
        chunk = evaluate_trade_legality(
            decision_date=dates[start:stop],
            stock_codes=codes,
            listing_age=ages[start:stop],
            open_prices=opens[start:stop],
            preclose_prices=precloses[start:stop],
            issue_prices=np.where(
                dates[start:stop, None] == issue_dates[None, :],
                np.asarray(data["issue_price"])[None, :],
                np.nan,
            ),
            st_mask=st[start:stop],
            delisted_mask=delisted[start:stop],
            precomputed_board_types=boards,
        )
        if chunk.up_limit_prices is None:
            raise RuntimeError("complete legality result omitted limit prices")
        output[start:stop] = chunk.up_limit_prices
    return output


__all__ = [
    "BOARD_BJ",
    "BOARD_CYB",
    "BOARD_KCB",
    "BOARD_MAIN",
    "TradeLegalityResult",
    "classify_board_types",
    "compute_limit_up_matrix",
    "evaluate_buy_legality_mask",
    "evaluate_trade_legality",
    "ordinary_limit_ratios",
]

"""财务报告期到交易日的 point-in-time 对齐。"""

from __future__ import annotations

import numpy as np
import pandas as pd


_DISCLOSURE_MONTH_DAY = {
    (3, 31): (0, 4, 30),
    (6, 30): (0, 8, 31),
    (9, 30): (0, 10, 31),
    (12, 31): (1, 4, 30),
}


def standard_report_period_mask(
    report_periods: pd.Series,
) -> np.ndarray:
    """标记可按法定季度披露期限对齐的标准报告期。"""
    periods = pd.to_datetime(
        report_periods.astype(int).astype(str),
        format="%Y%m%d",
    )
    return np.fromiter(
        (
            (month, day) in _DISCLOSURE_MONTH_DAY
            for month, day in zip(
                periods.dt.month,
                periods.dt.day,
            )
        ),
        dtype=np.bool_,
        count=len(periods),
    )


def statutory_disclosure_deadlines(
    report_periods: pd.Series,
) -> np.ndarray:
    """返回报告期对应的法定最晚披露日。

    年报最晚于次年 4 月 30 日披露，一季报、半年报、三季报分别最晚于
    当年 4 月 30 日、8 月 31 日、10 月 31 日披露。源数据不含公告时间，
    因此使用最晚披露日，并由调用方从其后的首个交易日开始使用。
    """
    periods = pd.to_datetime(
        report_periods.astype(int).astype(str),
        format="%Y%m%d",
    )
    month_day = list(zip(periods.dt.month, periods.dt.day))
    unexpected = sorted(set(month_day) - set(_DISCLOSURE_MONTH_DAY))
    if unexpected:
        raise ValueError(f"unexpected financial report periods: {unexpected}")

    years = periods.dt.year.to_numpy(dtype=np.int32)
    deadline_years = np.empty(len(periods), dtype=np.int32)
    deadline_months = np.empty(len(periods), dtype=np.int8)
    deadline_days = np.empty(len(periods), dtype=np.int8)
    for i, key in enumerate(month_day):
        year_offset, month, day = _DISCLOSURE_MONTH_DAY[key]
        deadline_years[i] = years[i] + year_offset
        deadline_months[i] = month
        deadline_days[i] = day

    deadlines = pd.to_datetime(
        {
            "year": deadline_years,
            "month": deadline_months,
            "day": deadline_days,
        }
    )
    return deadlines.to_numpy(dtype="datetime64[D]")


def build_pit_source_indices(
    financial_rows: pd.DataFrame,
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> np.ndarray:
    """为每个交易日和股票返回当时可用的最新财报行号。

    返回矩阵中的非负整数是 ``financial_rows.iloc`` 的行号，-1 表示尚无
    已披露报告。同一生效日同时出现年报和一季报时，较新的报告期覆盖较旧
    报告期。所有财务字段随后都通过该行号取值，避免不同字段跨报告期混拼。
    """
    codes = {str(code): i for i, code in enumerate(stock_codes)}
    standard = standard_report_period_mask(
        financial_rows["report_period"]
    )
    source_rows = np.flatnonzero(standard)
    rows = financial_rows.iloc[source_rows]
    stock_cols = rows["stock_code"].map(codes)
    deadlines = statutory_disclosure_deadlines(
        rows["report_period"]
    )
    dates = trade_dates.astype("datetime64[D]")

    events = pd.DataFrame(
        {
            "source_row": source_rows,
            "stock_col": stock_cols,
            "effective_row": np.searchsorted(
                dates,
                deadlines,
                side="right",
            ),
            "report_period": rows[
                "report_period"
            ].to_numpy(dtype=np.int64),
        }
    )
    events = events[
        events["stock_col"].notna()
        & (events["effective_row"] < len(dates))
    ].copy()
    events["stock_col"] = events["stock_col"].astype(np.intp)
    events = (
        events.sort_values("report_period", kind="stable")
        .drop_duplicates(
            ["effective_row", "stock_col"],
            keep="last",
        )
    )

    source_at_event = np.full(
        (len(dates), len(stock_codes)),
        -1,
        dtype=np.int32,
    )
    source_at_event[
        events["effective_row"].to_numpy(dtype=np.intp),
        events["stock_col"].to_numpy(dtype=np.intp),
    ] = events["source_row"].to_numpy(dtype=np.int32)

    has_event = source_at_event >= 0
    event_rows = np.where(
        has_event,
        np.arange(len(dates), dtype=np.int32)[:, None],
        -1,
    )
    np.maximum.accumulate(event_rows, axis=0, out=event_rows)
    valid = event_rows >= 0
    source_indices = np.take_along_axis(
        source_at_event,
        np.maximum(event_rows, 0),
        axis=0,
    )
    source_indices[~valid] = -1
    return source_indices


def materialize_pit_field(
    financial_rows: pd.DataFrame,
    source_indices: np.ndarray,
    field: str,
) -> np.ndarray:
    """按统一报告期行号展开一个财务字段，源报告缺失时保持 NaN。"""
    values = pd.to_numeric(
        financial_rows[field],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    result = np.full(source_indices.shape, np.nan, dtype=np.float64)
    valid = source_indices >= 0
    result[valid] = values[source_indices[valid]]
    return result

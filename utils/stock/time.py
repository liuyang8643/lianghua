from bisect import bisect_right
from datetime import date
from pathlib import Path

_TRADING_CALENDAR_STATE = None
_CALENDAR_PATH = Path(__file__).resolve().parents[2] / "data" / "trading_calendar.parquet"


def _get_trading_calendar_state():
    """Read the authoritative calendar once; never infer sessions from weekdays."""
    global _TRADING_CALENDAR_STATE
    if _TRADING_CALENDAR_STATE is None:
        import pyarrow.parquet as pq
        dates = tuple(sorted(pq.read_table(_CALENDAR_PATH).column('trade_date').to_pylist()))
        if not dates or any(not isinstance(value, date) for value in dates):
            raise ValueError('Trading calendar must contain nonempty date values')
        if len(set(dates)) != len(dates):
            raise ValueError('Trading calendar contains duplicate dates')
        _TRADING_CALENDAR_STATE = dates
    return _TRADING_CALENDAR_STATE


def _require_covered(value, dates):
    if not isinstance(value, date):
        raise ValueError('Trading calendar query requires a date')
    if not dates[0] <= value <= dates[-1]:
        raise ValueError(f'Trading calendar does not cover {value}; update the calendar first')


def get_last_trading_day(base_date: date = None) -> date:
    value = date.today() if base_date is None else base_date
    dates = _get_trading_calendar_state()
    _require_covered(value, dates)
    return dates[bisect_right(dates, value) - 1]


def get_trading_date_span(start_date: date, end_date: date) -> list[date]:
    dates = _get_trading_calendar_state()
    _require_covered(start_date, dates)
    _require_covered(end_date, dates)
    if start_date > end_date:
        raise ValueError('start_date不能大于end_date')
    return [value for value in dates if start_date <= value <= end_date]

from datetime import date
import pandas as pd
import pytest
import utils.stock.time as st


@pytest.fixture
def calendar(tmp_path, monkeypatch):
    path = tmp_path / 'calendar.parquet'
    dates = [date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 21)]
    pd.DataFrame({'trade_date': dates}).to_parquet(path)
    monkeypatch.setattr(st, '_CALENDAR_PATH', path)
    monkeypatch.setattr(st, '_TRADING_CALENDAR_STATE', None)
    return path, dates


def test_calendar_loaded_once_and_holidays_are_not_invented(calendar, monkeypatch):
    path, dates = calendar
    assert st.get_last_trading_day(date(2026, 9, 20)) == dates[1]
    import pyarrow.parquet as pq
    monkeypatch.setattr(pq, 'read_table', lambda *a, **k: pytest.fail('calendar reloaded'))
    assert st.get_trading_date_span(dates[0], dates[-1]) == dates


def test_calendar_missing_empty_and_out_of_bounds_fail(calendar):
    path, dates = calendar
    with pytest.raises(ValueError, match='does not cover'):
        st.get_last_trading_day(date(2026, 9, 22))
    st._TRADING_CALENDAR_STATE = None
    pd.DataFrame({'trade_date': []}).to_parquet(path)
    with pytest.raises(ValueError, match='nonempty'):
        st.get_last_trading_day(dates[0])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        st.get_last_trading_day(dates[0])


def test_calendar_rejects_duplicate_dates_and_reversed_query(calendar):
    path, dates = calendar
    with pytest.raises(ValueError):
        st.get_trading_date_span(dates[-1], dates[0])
    st._TRADING_CALENDAR_STATE = None
    pd.DataFrame({'trade_date': dates + dates[:1]}).to_parquet(path)
    with pytest.raises(ValueError, match='duplicate'):
        st.get_last_trading_day(dates[0])

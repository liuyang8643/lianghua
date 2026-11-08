"""Regression: verified suspension must never excuse a missing resumption bar."""
import pandas as pd
import pytest

from data import update_kline as source


def test_verified_301139_exact_bounds_and_code():
    days = pd.to_datetime(['20260828', '20260831', '20260929', '20260930'])
    cache = {day.strftime('%Y-%m-%d'): [] for day in days}
    exempt = source._suspended_dates('301139.SZ', days, cache)
    assert exempt.tolist() == days[1:3].tolist()
    assert source._suspended_dates('301138.SZ', days, cache).empty


def test_resumption_day_still_requires_real_bar():
    days = pd.to_datetime(['20260828', '20260831', '20260929', '20260930'])
    cache = {day.strftime('%Y-%m-%d'): [] for day in days}
    expected = days.difference(source._suspended_dates('301139.SZ', days, cache))
    frame = pd.DataFrame({'close': [1.]}, index=['20260828'])
    with pytest.raises(RuntimeError, match='20260930'):
        source._validate_qmt_dates(frame, expected, '301139.SZ')
    frame.loc['20260930'] = 1.
    source._validate_qmt_dates(frame, expected, '301139.SZ')

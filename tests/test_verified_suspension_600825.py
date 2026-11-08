"""新华传媒重组停牌缺口可豁免，复牌日仍必须有真实日线。"""
import pandas as pd
import pytest

from data import update_kline as source


def test_verified_suspension_excludes_adjacent_sessions_and_other_stocks():
    days = pd.to_datetime(['20260904', '20260907', '20260918', '20260921'])
    cache = {day.strftime('%Y-%m-%d'): [] for day in days}
    exempt = source._suspended_dates('600825.SH', days, cache)
    assert exempt.tolist() == days[1:3].tolist()
    assert source._suspended_dates('600826.SH', days, cache).empty


def test_resumption_bar_is_still_required():
    days = pd.bdate_range('2026-09-04', '2026-09-21')
    cache = {day.strftime('%Y-%m-%d'): [] for day in days}
    expected = days.difference(source._suspended_dates('600825.SH', days, cache))
    frame = pd.DataFrame({'close': [10.]}, index=['20260904'])
    with pytest.raises(RuntimeError, match='20260921'):
        source._validate_qmt_dates(frame, expected, '600825.SH')
    frame.loc['20260921'] = 11.
    source._validate_qmt_dates(frame, expected, '600825.SH')

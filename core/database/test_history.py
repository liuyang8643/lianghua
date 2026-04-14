import unittest
from datetime import date, datetime
from unittest.mock import patch

import pandas as pd

from core.database import history as history_module
from utils.stock.time import get_trading_date_span


def _make_daily_df(*dates: datetime) -> pd.DataFrame:
  return pd.DataFrame(
    {
      'time': [int(dt.timestamp() * 1000) for dt in dates],
      'suspendFlag': [0] * len(dates),
    }
  )


class TestHistoryAfterDownload(unittest.TestCase):
  def test_count_complete_new_stock_skips_download(self):
    base_time = datetime(2026, 4, 10, 15, 0, 0)
    trading_days = get_trading_date_span(date(2026, 3, 16), date(2026, 4, 10))
    data = _make_daily_df(*[
      datetime.combine(trade_day, datetime.min.time()).replace(hour=15)
      for trade_day in trading_days
    ])

    with patch.object(history_module, 'get_history_data', return_value={'301667.SZ': data}), \
         patch.object(history_module, '_get_stock_date_range', return_value=(date(2026, 3, 16), None)), \
         patch.object(history_module, 'is_latest_data', return_value=True), \
         patch.object(history_module.xtdata, 'download_history_data2') as mock_download:
      result = history_module.get_history_data_after_download(
        ['301667.SZ'],
        60,
        base_time,
        '1d',
        'back',
      )

    self.assertIs(result['301667.SZ'], data)
    mock_download.assert_not_called()

  def test_count_none_complete_new_stock_skips_download(self):
    base_time = datetime(2026, 4, 10, 15, 0, 0)
    data = _make_daily_df(
      datetime(2025, 12, 23, 15, 0, 0),
      datetime(2026, 4, 10, 15, 0, 0),
    )

    with patch.object(history_module, 'get_history_data', return_value={'301667.SZ': data}), \
         patch.object(history_module, '_get_stock_date_range', return_value=(date(2025, 12, 23), None)), \
         patch.object(history_module, 'is_latest_data', return_value=True), \
         patch.object(history_module.xtdata, 'download_history_data2') as mock_download:
      result = history_module.get_history_data_after_download(
        ['301667.SZ'],
        None,
        base_time,
        '1d',
        'back',
      )

    self.assertIs(result['301667.SZ'], data)
    mock_download.assert_not_called()

  def test_count_none_incomplete_history_downloads_from_list_date(self):
    base_time = datetime(2026, 4, 10, 15, 0, 0)
    data = _make_daily_df(
      datetime(2026, 1, 10, 15, 0, 0),
      datetime(2026, 4, 10, 15, 0, 0),
    )

    with patch.object(history_module, 'get_history_data', side_effect=[{'301667.SZ': data}, {'301667.SZ': data}]), \
         patch.object(history_module, '_get_stock_date_range', return_value=(date(2025, 12, 23), None)), \
         patch.object(history_module, 'is_latest_data', return_value=True), \
         patch.object(history_module.xtdata, 'download_history_data2') as mock_download:
      history_module.get_history_data_after_download(
        ['301667.SZ'],
        None,
        base_time,
        '1d',
        'back',
      )

    mock_download.assert_called_once_with(
      ['301667.SZ'],
      '1d',
      start_time='20251223',
      end_time='20260410',
    )

  def test_minute_period_download_uses_datetime_boundaries(self):
    base_time = datetime(2026, 4, 10, 14, 37, 0)
    empty_df = pd.DataFrame(columns=['time', 'suspendFlag'])
    expected_start = history_module.format_qmt_datetime(
      history_module.get_target_period_backward(base_time, '1m', 10)
    )

    with patch.object(history_module, 'get_history_data', side_effect=[{'000001.SZ': empty_df}, {'000001.SZ': empty_df}]), \
         patch.object(history_module.xtdata, 'download_history_data2') as mock_download:
      history_module.get_history_data_after_download(
        ['000001.SZ'],
        10,
        base_time,
        '1m',
        'back',
      )

    mock_download.assert_called_once_with(
      ['000001.SZ'],
      '1m',
      start_time=expected_start,
      end_time='20260410143700',
    )

  def test_count_shortfall_downloads_before_first_returned_bar(self):
    base_time = datetime(2026, 4, 30, 15, 0, 0)
    short_df = _make_daily_df(
      datetime(2026, 4, 21, 15, 0, 0),
      datetime(2026, 4, 22, 15, 0, 0),
      datetime(2026, 4, 23, 15, 0, 0),
    )
    filled_df = _make_daily_df(
      datetime(2026, 4, 17, 15, 0, 0),
      datetime(2026, 4, 20, 15, 0, 0),
      datetime(2026, 4, 21, 15, 0, 0),
      datetime(2026, 4, 22, 15, 0, 0),
      datetime(2026, 4, 23, 15, 0, 0),
    )

    with patch.object(history_module, 'get_history_data', side_effect=[{'300108.SZ': short_df}, {'300108.SZ': filled_df}]), \
         patch.object(history_module, '_get_stock_date_range', return_value=(date(2020, 1, 1), None)), \
         patch.object(history_module, 'is_latest_data', return_value=True), \
         patch.object(history_module.xtdata, 'download_history_data2') as mock_download:
      result = history_module.get_history_data_after_download(
        ['300108.SZ'],
        5,
        base_time,
        '1d',
        'back',
      )

    self.assertIs(result['300108.SZ'], filled_df)
    mock_download.assert_called_once_with(
      ['300108.SZ'],
      '1d',
      start_time='20260417',
      end_time='20260430',
    )

  def test_placeholder_tail_expands_fetch_count_without_download(self):
    base_time = datetime(2026, 4, 30, 15, 0, 0)
    placeholder_df = pd.DataFrame(
      {
        'time': [int(dt.timestamp() * 1000) for dt in [
          datetime(2026, 4, 17, 15, 0, 0),
          datetime(2026, 4, 20, 15, 0, 0),
          datetime(2026, 4, 21, 15, 0, 0),
          datetime(2026, 4, 22, 15, 0, 0),
          datetime(2026, 4, 23, 15, 0, 0),
        ]],
        'open': [0.0] * 5,
        'high': [0.0] * 5,
        'low': [0.0] * 5,
        'close': [0.0] * 5,
        'suspendFlag': [0] * 5,
      }
    )
    expanded_df = pd.DataFrame(
      {
        'time': [int(dt.timestamp() * 1000) for dt in [
          datetime(2026, 4, 10, 15, 0, 0),
          datetime(2026, 4, 13, 15, 0, 0),
          datetime(2026, 4, 14, 15, 0, 0),
          datetime(2026, 4, 15, 15, 0, 0),
          datetime(2026, 4, 16, 15, 0, 0),
          datetime(2026, 4, 17, 15, 0, 0),
          datetime(2026, 4, 20, 15, 0, 0),
          datetime(2026, 4, 21, 15, 0, 0),
          datetime(2026, 4, 22, 15, 0, 0),
          datetime(2026, 4, 23, 15, 0, 0),
        ]],
        'open': [1.0] * 5 + [0.0] * 5,
        'high': [1.1] * 5 + [0.0] * 5,
        'low': [0.9] * 5 + [0.0] * 5,
        'close': [1.0] * 5 + [0.0] * 5,
        'suspendFlag': [0] * 10,
      }
    )
    expected_start = history_module.format_qmt_date(
      history_module.get_target_period_backward(datetime(2026, 4, 17, 15, 0, 0), '1d', 5)
    )

    with patch.object(history_module, 'get_history_data', side_effect=[{'300108.SZ': placeholder_df}, {'300108.SZ': expanded_df}]), \
         patch.object(history_module, '_get_stock_date_range', return_value=(date(2020, 1, 1), None)), \
         patch.object(history_module, 'is_latest_data', return_value=True), \
         patch.object(history_module.xtdata, 'download_history_data2') as mock_download:
      result = history_module.get_history_data_after_download(
        ['300108.SZ'],
        5,
        base_time,
        '1d',
        'back',
      )

    self.assertIs(result['300108.SZ'], expanded_df)
    mock_download.assert_not_called()

  def test_widen_local_window_before_second_download(self):
    base_time = datetime(2026, 6, 20, 15, 0, 0)
    empty_df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'suspendFlag'])
    raw20 = pd.DataFrame(
      {
        'time': [int(dt.timestamp() * 1000) for dt in [
          datetime(2026, 5, 23, 15, 0, 0),
          datetime(2026, 5, 26, 15, 0, 0),
          datetime(2026, 5, 27, 15, 0, 0),
          datetime(2026, 5, 28, 15, 0, 0),
          datetime(2026, 5, 29, 15, 0, 0),
          datetime(2026, 5, 30, 15, 0, 0),
          datetime(2026, 6, 3, 15, 0, 0),
          datetime(2026, 6, 4, 15, 0, 0),
          datetime(2026, 6, 5, 15, 0, 0),
          datetime(2026, 6, 6, 15, 0, 0),
          datetime(2026, 6, 9, 15, 0, 0),
          datetime(2026, 6, 10, 15, 0, 0),
          datetime(2026, 6, 11, 15, 0, 0),
          datetime(2026, 6, 12, 15, 0, 0),
          datetime(2026, 6, 13, 15, 0, 0),
          datetime(2026, 6, 16, 15, 0, 0),
          datetime(2026, 6, 17, 15, 0, 0),
          datetime(2026, 6, 18, 15, 0, 0),
          datetime(2026, 6, 19, 15, 0, 0),
          datetime(2026, 6, 20, 15, 0, 0),
        ]],
        'open': [10.0] + [20.0] * 10 + [30.0] * 9,
        'high': [10.5] + [20.5] * 10 + [30.5] * 9,
        'low': [9.5] + [19.5] * 10 + [29.5] * 9,
        'close': [10.0] + [20.0] * 10 + [30.0] * 9,
        'suspendFlag': [0] + [1] * 10 + [0] * 9,
      }
    )
    raw32 = pd.DataFrame(
      {
        'time': [int(dt.timestamp() * 1000) for dt in [
          datetime(2026, 5, 7, 15, 0, 0),
          datetime(2026, 5, 8, 15, 0, 0),
          datetime(2026, 5, 9, 15, 0, 0),
          datetime(2026, 5, 12, 15, 0, 0),
          datetime(2026, 5, 13, 15, 0, 0),
          datetime(2026, 5, 14, 15, 0, 0),
          datetime(2026, 5, 15, 15, 0, 0),
          datetime(2026, 5, 16, 15, 0, 0),
          datetime(2026, 5, 19, 15, 0, 0),
          datetime(2026, 5, 20, 15, 0, 0),
          datetime(2026, 5, 21, 15, 0, 0),
          datetime(2026, 5, 22, 15, 0, 0),
          datetime(2026, 5, 23, 15, 0, 0),
          datetime(2026, 5, 26, 15, 0, 0),
          datetime(2026, 5, 27, 15, 0, 0),
          datetime(2026, 5, 28, 15, 0, 0),
          datetime(2026, 5, 29, 15, 0, 0),
          datetime(2026, 5, 30, 15, 0, 0),
          datetime(2026, 6, 3, 15, 0, 0),
          datetime(2026, 6, 4, 15, 0, 0),
          datetime(2026, 6, 5, 15, 0, 0),
          datetime(2026, 6, 6, 15, 0, 0),
          datetime(2026, 6, 9, 15, 0, 0),
          datetime(2026, 6, 10, 15, 0, 0),
          datetime(2026, 6, 11, 15, 0, 0),
          datetime(2026, 6, 12, 15, 0, 0),
          datetime(2026, 6, 13, 15, 0, 0),
          datetime(2026, 6, 16, 15, 0, 0),
          datetime(2026, 6, 17, 15, 0, 0),
          datetime(2026, 6, 18, 15, 0, 0),
          datetime(2026, 6, 19, 15, 0, 0),
          datetime(2026, 6, 20, 15, 0, 0),
        ]],
        'open': [10.0] * 13 + [20.0] * 10 + [30.0] * 9,
        'high': [10.5] * 13 + [20.5] * 10 + [30.5] * 9,
        'low': [9.5] * 13 + [19.5] * 10 + [29.5] * 9,
        'close': [10.0] * 13 + [20.0] * 10 + [30.0] * 9,
        'suspendFlag': [0] * 13 + [1] * 10 + [0] * 9,
      }
    )

    expected_start = history_module.format_qmt_date(
      history_module.get_target_period_backward(base_time, '1d', 20)
    )

    with patch.object(history_module, 'get_history_data', side_effect=[{'603019.SH': empty_df}, {'603019.SH': raw20}, {'603019.SH': raw32}]), \
         patch.object(history_module, '_get_stock_date_range', return_value=(date(2020, 1, 1), None)), \
         patch.object(history_module.xtdata, 'download_history_data2') as mock_download:
      result = history_module.get_history_data_after_download(
        ['603019.SH'],
        20,
        base_time,
        '1d',
        'back',
      )

    self.assertIs(result['603019.SH'], raw32)
    mock_download.assert_called_once_with(
      ['603019.SH'],
      '1d',
      start_time=expected_start,
      end_time='20260620',
    )


if __name__ == '__main__':
  unittest.main()

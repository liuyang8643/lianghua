import unittest
from datetime import date, datetime
from unittest.mock import patch

import pandas as pd

from core.database import history as history_module


def _make_daily_df(*dates: datetime) -> pd.DataFrame:
  return pd.DataFrame(
    {
      'time': [int(dt.timestamp() * 1000) for dt in dates],
      'suspendFlag': [0] * len(dates),
    }
  )


class TestHistoryAfterDownload(unittest.TestCase):
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
      incrementally=True,
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
      incrementally=True,
    )


if __name__ == '__main__':
  unittest.main()

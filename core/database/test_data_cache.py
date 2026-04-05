import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from core.database import data as data_module


class TestMarketDataFromCache(unittest.TestCase):
  def test_get_market_data_from_cache_passes_base_time_to_full_data(self):
    base_time = datetime(2024, 8, 30, 15, 0, 0)
    df = pd.DataFrame(
      [
        {'time': int(datetime(2024, 8, 30, 15, 0, 0).timestamp() * 1000), 'open': 1.23},
      ]
    )

    with patch.object(data_module, 'get_full_market_data', return_value=df) as mock_full, \
         patch.object(data_module, 'check_stock_valid_at_date', return_value=True):
      result = data_module.get_market_data_from_cache(
        stock_code='000023.SZ',
        count=1,
        base_time=base_time,
        period='1d',
        allow_tainted=True,
        dividend_type='back',
      )

    self.assertIsNotNone(result)
    self.assertEqual(len(result), 1)
    self.assertEqual(float(result.iloc[-1]['open']), 1.23)
    self.assertEqual(mock_full.call_count, 1)
    self.assertEqual(mock_full.call_args.kwargs['target_time'], base_time)

  def test_historical_daily_data_uses_tainted_fast_path(self):
    base_time = datetime(2024, 8, 30, 15, 0, 0)
    df = pd.DataFrame(
      [
        {'time': int(datetime(2024, 8, 30, 15, 0, 0).timestamp() * 1000), 'open': 1.0},
      ]
    )

    with patch.object(data_module, 'get_full_market_data', return_value=df) as mock_full, \
         patch.object(data_module, 'check_stock_valid_at_date', return_value=True):
      _ = data_module.get_market_data_from_cache(
        stock_code='000023.SZ',
        count=1,
        base_time=base_time,
        period='1d',
        allow_tainted=False,
        dividend_type='back',
      )

    self.assertTrue(mock_full.call_args.kwargs['allow_tainted'])


if __name__ == '__main__':
  unittest.main()

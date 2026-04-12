import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from core.database import data as data_module


class TestMarketDataFromCache(unittest.TestCase):
  def test_init_market_data_range_uses_window_count_and_filters_suspend_rows(self):
    end_time = datetime(2024, 8, 30, 15, 0, 0)
    df = pd.DataFrame(
      [
        {'time': int(datetime(2024, 8, 29, 15, 0, 0).timestamp() * 1000), 'open': 1.0, 'suspendFlag': 0},
        {'time': int(datetime(2024, 8, 30, 15, 0, 0).timestamp() * 1000), 'open': 2.0, 'suspendFlag': 1},
      ]
    )

    with patch.object(data_module, '_GLOBAL_DAILY_CACHE') as mock_cache, \
         patch.object(data_module, 'get_market_data', return_value=df) as mock_get:
      mock_cache.contains.side_effect = [False, True]  # 股票缓存未命中，基准缓存跳过
      mock_cache.stat.return_value = {'count': 1, 'total_size_mb': 0.01}

      loaded = data_module.init_market_data_range(
        stock_codes=['000023.SZ'],
        start_time=datetime(2024, 8, 29, 0, 0, 0),
        end_time=end_time,
        period='1d',
        max_workers=1,
        dividend_type='back',
      )

    self.assertEqual(loaded, 1)
    self.assertEqual(mock_get.call_count, 1)
    self.assertEqual(mock_get.call_args.args[0], '000023.SZ')
    self.assertEqual(mock_get.call_args.args[1], 2)
    self.assertEqual(mock_get.call_args.args[2], data_module.get_latest_trading_time(end_time))

    cached_df = mock_cache.put.call_args.args[1]
    self.assertEqual(len(cached_df), 1)
    self.assertEqual(float(cached_df.iloc[-1]['open']), 1.0)

  def test_get_market_data_from_cache_passes_base_time_to_window_loader(self):
    base_time = datetime(2024, 8, 30, 15, 0, 0)
    df = pd.DataFrame(
      [
        {'time': int(datetime(2024, 8, 30, 15, 0, 0).timestamp() * 1000), 'open': 1.23},
      ]
    )

    with patch.object(data_module, '_get_cached_market_window', return_value=df) as mock_window, \
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
    self.assertEqual(mock_window.call_count, 1)
    self.assertEqual(mock_window.call_args.args[0], '000023.SZ')
    self.assertEqual(mock_window.call_args.args[1], 1)
    self.assertEqual(mock_window.call_args.args[2], base_time)

  def test_historical_daily_data_uses_tainted_fast_path(self):
    base_time = datetime(2024, 8, 30, 15, 0, 0)
    df = pd.DataFrame(
      [
        {'time': int(datetime(2024, 8, 30, 15, 0, 0).timestamp() * 1000), 'open': 1.0},
      ]
    )

    with patch.object(data_module, '_get_cached_market_window', return_value=df) as mock_window, \
         patch.object(data_module, 'check_stock_valid_at_date', return_value=True):
      _ = data_module.get_market_data_from_cache(
        stock_code='000023.SZ',
        count=1,
        base_time=base_time,
        period='1d',
        allow_tainted=False,
        dividend_type='back',
      )

    self.assertTrue(mock_window.call_args.args[4])

  def test_get_market_data_from_cache_strict_trade_date_rejects_fallback_bar(self):
    base_time = datetime(2024, 8, 30, 15, 0, 0)
    fallback_df = pd.DataFrame(
      [
        {'time': int(datetime(2024, 8, 29, 15, 0, 0).timestamp() * 1000), 'open': 9.99},
      ]
    )

    with patch.object(data_module, '_get_cached_market_window', return_value=fallback_df), \
         patch.object(data_module, 'check_stock_valid_at_date', return_value=True):
      result = data_module.get_market_data_from_cache(
        stock_code='000023.SZ',
        count=1,
        base_time=base_time,
        period='1d',
        allow_tainted=True,
        dividend_type='back',
        strict_trade_date=True,
      )

    self.assertIsNone(result)

  def test_get_market_data_batch_strict_trade_date_rejects_fallback_bar(self):
    base_time = datetime(2024, 8, 30, 15, 0, 0)
    fallback_df = pd.DataFrame(
      [
        {'time': int(datetime(2024, 8, 29, 15, 0, 0).timestamp() * 1000), 'open': 9.99},
      ]
    )

    with patch.object(data_module, 'get_history_data', return_value={'000023.SZ': fallback_df}):
      result = data_module.get_market_data_batch(
        stock_codes=['000023.SZ'],
        count=1,
        base_time=base_time,
        period='1d',
        allow_tainted=True,
        dividend_type='back',
        strict_trade_date=True,
      )

    self.assertEqual({'000023.SZ': None}, result)

  def test_get_market_data_range_from_cache_slices_requested_window(self):
    start_time = datetime(2024, 8, 29, 0, 0, 0)
    end_time = datetime(2024, 8, 30, 15, 0, 0)
    df = pd.DataFrame(
      [
        {'time': int(datetime(2024, 8, 28, 15, 0, 0).timestamp() * 1000), 'open': 0.8},
        {'time': int(datetime(2024, 8, 29, 15, 0, 0).timestamp() * 1000), 'open': 1.0},
        {'time': int(datetime(2024, 8, 30, 15, 0, 0).timestamp() * 1000), 'open': 1.2},
      ]
    )

    with patch.object(data_module, '_get_cached_market_window', return_value=df) as mock_window, \
         patch.object(data_module, 'check_stock_valid_at_date', return_value=True):
      result = data_module.get_market_data_range_from_cache(
        stock_code='000023.SZ',
        start_time=start_time,
        end_time=end_time,
        period='1d',
        allow_tainted=True,
        dividend_type='back',
      )

    self.assertEqual(mock_window.call_args.args[1], 2)
    self.assertEqual(len(result), 2)
    self.assertEqual(float(result.iloc[0]['open']), 1.0)
    self.assertEqual(float(result.iloc[-1]['open']), 1.2)


if __name__ == '__main__':
  unittest.main()

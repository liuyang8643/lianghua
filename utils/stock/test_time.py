import unittest
from datetime import date, datetime
from unittest.mock import patch

from . import time as time_module
from .time import get_target_period_backward

class TestTime(unittest.TestCase):
  def setUp(self):
    time_module._get_trading_calendar_state.cache_clear()
    time_module.is_trading_day.cache_clear()
    time_module.get_last_trading_day.cache_clear()
    time_module.get_next_trading_day.cache_clear()

  def tearDown(self):
    time_module._get_trading_calendar_state.cache_clear()
    time_module.is_trading_day.cache_clear()
    time_module.get_last_trading_day.cache_clear()
    time_module.get_next_trading_day.cache_clear()

  def test_get_period_start_time(self):
    self.assertEqual(
      get_target_period_backward(datetime(2025, 3, 6, 13, 1, 4), '1m', 2),
      datetime(2025, 3, 6, 11, 29, 4)
    )
    self.assertEqual(
      get_target_period_backward(datetime(2025, 3, 3, 13, 1, 4), '1d', 1),
      datetime(2025, 2, 28, 13, 1, 4)
    )
    self.assertEqual(
      get_target_period_backward(datetime(2025, 3, 7, 13, 1, 4), '5m', 143),
      datetime(2025, 3, 4, 13, 6, 4)
    )

  def test_is_trading_day_uses_historical_calendar_and_weekday_fallback(self):
    calendar = frozenset([
      date(2006, 1, 23),
      date(2006, 1, 24),
      date(2006, 1, 25),
      date(2006, 2, 6),
    ])

    with patch.object(
        time_module,
        '_get_trading_calendar_state',
        return_value=(calendar, date(2006, 1, 23), date(2006, 2, 6)),
    ):
      self.assertTrue(time_module.is_trading_day(date(2006, 1, 25)))
      self.assertFalse(time_module.is_trading_day(date(2006, 1, 26)))
      self.assertTrue(time_module.is_trading_day(date(2006, 2, 7)))
      self.assertFalse(time_module.is_trading_day(date(2006, 2, 11)))

  def test_get_last_trading_day_prefers_calendar_for_history(self):
    calendar = frozenset([
      date(2006, 1, 23),
      date(2006, 1, 24),
      date(2006, 1, 25),
      date(2006, 2, 6),
    ])

    with patch.object(
        time_module,
        '_get_trading_calendar_state',
        return_value=(calendar, date(2006, 1, 23), date(2006, 2, 6)),
    ):
      self.assertEqual(time_module.get_last_trading_day(date(2006, 1, 26)), date(2006, 1, 25))
      self.assertEqual(time_module.get_last_trading_day(date(2006, 2, 7)), date(2006, 2, 7))
      self.assertEqual(time_module.get_next_trading_day(date(2006, 1, 25)), date(2006, 2, 6))

  def test_get_trading_calendar_state_uses_process_local_cache(self):
    calendar = (
      frozenset([date(2026, 4, 10), date(2026, 4, 13)]),
      date(2026, 4, 10),
      date(2026, 4, 13),
    )

    with patch.object(time_module, '_fetch_xt_trading_calendar_state', return_value=calendar) as mock_fetch:
      first = time_module._get_trading_calendar_state()
      second = time_module._get_trading_calendar_state()

    self.assertEqual(first, calendar)
    self.assertEqual(second, calendar)
    self.assertEqual(mock_fetch.call_count, 1)

if __name__ == '__main__':
  unittest.main()

import unittest
from datetime import datetime

from .time import get_target_period_backward

class TestTime(unittest.TestCase):
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

if __name__ == '__main__':
  unittest.main()

"""测试 stock_list 模块 - 时间准确性与并发性能测试"""
import unittest
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .stock_list import get_all_stock_code_list
import core.database.stock_list as stock_list_module


def query_single_date(date: datetime) -> tuple[str, int, float]:
  """单次查询函数（用于多进程测试）"""
  start = time.time()
  stock_list = get_all_stock_code_list(date)
  elapsed = (time.time() - start) * 1000
  return (date.strftime('%Y-%m-%d'), len(stock_list), elapsed)


class TestStockList(unittest.TestCase):
  """股票列表模块测试 - 时间准确性与并发性能"""
  
  # 测试股票样本
  TEST_STOCKS = [
    ('300799.SZ', '2024-07-26', '已退市股票'),
    ('600532.SH', '2023-06-19', '已退市股票'),
    ('000001.SZ', None, '平安银行（一直在市）'),
    ('301667.SZ', '2025-12-23', '新股'),
  ]
  
  # 测试日期点
  TEST_DATES = [
    datetime(2023, 1, 1),   # 300799在市，600532在市，000001在市，301667未上市
    datetime(2023, 7, 1),   # 300799在市，600532已退市，000001在市，301667未上市
    datetime(2024, 1, 1),   # 300799在市，600532已退市，000001在市，301667未上市
    datetime(2024, 8, 1),   # 300799已退市，600532已退市，000001在市，301667未上市
    datetime(2025, 11, 1),  # 300799已退市，600532已退市，000001在市，301667未上市
    datetime(2025, 12, 30), # 300799已退市，600532已退市，000001在市，301667已上市
    datetime(2026, 1, 9),   # 300799已退市，600532已退市，000001在市，301667已上市（当前）
  ]
  
  # 期望结果（True=应在列表中，False=不应在列表中）
  EXPECTED_RESULTS = {
    '2023-01-01': {'300799.SZ': True,  '600532.SH': True,  '000001.SZ': True,  '301667.SZ': False},
    '2023-07-01': {'300799.SZ': True,  '600532.SH': False, '000001.SZ': True,  '301667.SZ': False},
    '2024-01-01': {'300799.SZ': True,  '600532.SH': False, '000001.SZ': True,  '301667.SZ': False},
    '2024-08-01': {'300799.SZ': False, '600532.SH': False, '000001.SZ': True,  '301667.SZ': False},
    '2025-11-01': {'300799.SZ': False, '600532.SH': False, '000001.SZ': True,  '301667.SZ': False},
    '2025-12-30': {'300799.SZ': False, '600532.SH': False, '000001.SZ': True,  '301667.SZ': True},
    '2026-01-09': {'300799.SZ': False, '600532.SH': False, '000001.SZ': True,  '301667.SZ': True},
  }
  
  def test_stock_date_info(self):
    """验证测试股票的日期信息"""
    _get_stock_date_range = stock_list_module._get_stock_date_range
    
    for code, expected_delist, desc in self.TEST_STOCKS:
      date_range = _get_stock_date_range(code)
      self.assertIsNotNone(date_range, f'股票 {code} ({desc}) 应有日期信息')
  
  def test_time_accuracy(self):
    """时间准确性测试 - 不同时间点的股票列表正确性"""
    for test_date in self.TEST_DATES:
      date_str = test_date.strftime('%Y-%m-%d')
      stock_list = get_all_stock_code_list(test_date)
      expected = self.EXPECTED_RESULTS[date_str]
      
      for code, should_exist in expected.items():
        exists = code in stock_list
        self.assertEqual(
          exists, 
          should_exist, 
          f'日期 {date_str}: 股票 {code} 预期{"在" if should_exist else "不在"}列表，实际{"在" if exists else "不在"}列表'
        )
  
  def test_cache_consistency(self):
    """缓存一致性测试 - 验证缓存前后结果一致"""
    test_date = datetime(2024, 1, 1)
    
    stock_list_1 = get_all_stock_code_list(test_date)
    stock_list_2 = get_all_stock_code_list(test_date)
    
    self.assertEqual(stock_list_1, stock_list_2, '缓存前后结果应一致')
  
  def test_multithread_consistency(self):
    """多线程一致性测试"""
    dates_to_query = self.TEST_DATES[:3]
    
    results_serial = []
    for date in dates_to_query:
      results_serial.append(query_single_date(date))
    
    with ThreadPoolExecutor(max_workers=2) as executor:
      results_thread = list(executor.map(query_single_date, dates_to_query))
    
    for (date_s, count_s, _), (date_t, count_t, _) in zip(results_serial, results_thread):
      self.assertEqual(date_s, date_t, '日期应一致')
      self.assertEqual(count_s, count_t, f'日期 {date_s} 的股票数量应一致')
  
  def test_multiprocess_consistency(self):
    """多进程一致性测试"""
    dates_to_query = self.TEST_DATES[:3]
    
    results_serial = []
    for date in dates_to_query:
      results_serial.append(query_single_date(date))
    
    with ProcessPoolExecutor(max_workers=2) as executor:
      results_process = list(executor.map(query_single_date, dates_to_query))
    
    for (date_s, count_s, _), (date_p, count_p, _) in zip(results_serial, results_process):
      self.assertEqual(date_s, date_p, '日期应一致')
      self.assertEqual(count_s, count_p, f'日期 {date_s} 的股票数量应一致')
  
  def test_large_scale_query(self):
    """大规模查询测试 - 查询多个年份"""
    dates = []
    for year in range(2024, 2026):
      for month in range(1, 4):
        dates.append(datetime(year, month, 1))
    
    for date in dates:
      stock_list = get_all_stock_code_list(date)
      self.assertGreater(len(stock_list), 0, f'日期 {date.strftime("%Y-%m-%d")} 应返回股票列表')
  
  def test_lru_cache_mechanism(self):
    """LRU缓存淘汰测试 - 验证缓存机制不崩溃"""
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(150)]
    
    for date in dates:
      stock_list = get_all_stock_code_list(date)
      self.assertIsInstance(stock_list, list, '应返回列表类型')


if __name__ == '__main__':
  unittest.main()

import time as sys_time
import concurrent.futures
from datetime import datetime, time
from typing import List, Optional,Callable

from trading.logger import trading_logger
from utils.recorder import recorder
from utils.stock.time import is_current_trading
from utils.stock.holiday import is_trading_day
from .trader import Trader
from .lark.sender import lark_sender

class TradingScheduler:
  def __init__(
      self,
      trader: Trader,
      before_trade: Callable = None,  # 交易时间前执行一次
      while_trade: List[Callable] = None,  # 交易时间内每 检查间隔时间 执行一次
      after_trade: Callable = None,  # 交易时间后执行一次
      check_interval: int = 30  # 检查间隔时间
  ):
    self.trader = trader
    self.trading = False
    self.prepared = False

    """ Hooks """
    self.before_trade = before_trade
    self.while_trade = while_trade
    self.after_trade = after_trade
    self.check_interval = check_interval

    """ Stores"""
    self.whole_sub_id = None  # 全市场行情订阅 ID

  def start_check_trading(self):
    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    current_check_interval = self.check_interval
    while True:
      if not self.prepared:
        if is_trading_day() and time(9, 25) < datetime.now().time() < time(15, 0):
          current_check_interval = self.check_interval
          trading_logger.debug(f"今日交易准备开始")
          executor = concurrent.futures.ThreadPoolExecutor()
          if self.before_trade:
            future = executor.submit(self.before_trade, self)
            future.result()
            trading_logger.debug(f"盘前准备完成，等待开盘")
          self.prepared = True
        else:
          current_check_interval = min(10 * self.check_interval, 5 * 60)  # 非交易时间检查间隔时间加长

      if is_current_trading():
        if not self.trading:
          lark_sender.send_msg("今日交易开始")
          trading_logger.debug(f"今日交易开始")
          self.trading = True
        if self.while_trade:
          futures = [executor.submit(func, self) for func in self.while_trade]
          # concurrent.futures.wait(futures)
      elif self.trading and time(15, 0) < datetime.now().time():
        self.trading = False
        self.prepared = False
        trading_logger.debug("今日交易结束")
        if self.after_trade:
          future = executor.submit(self.after_trade, self)
          future.result()
          trading_logger.debug(f"盘后处理完成")
        executor.shutdown(wait=False, cancel_futures=True)
        rc = recorder.flush()
        msg = f"今日交易完成: \n{'\n'.join([f'{k}@{v}' for k, v in sorted(rc.items())])}"
        lark_sender.send_msg(msg)
        trading_logger.debug(msg)

      sys_time.sleep(current_check_interval)

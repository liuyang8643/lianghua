import time as sys_time
import concurrent.futures
from datetime import datetime, time
from typing import Callable, List, Optional

from trading.logger import trading_logger
from utils.recorder import recorder
from utils.stock.time import is_current_trading
from utils.stock.holiday import is_trading_day
from .trader import Trader
from .lark.sender import lark_sender

PREPARE_REBALANCE_START = time(9, 25)
EXECUTE_REBALANCE_START = time(9, 30)
EXECUTE_REBALANCE_END = time(10, 0)


class TradingScheduler:
  def __init__(
      self,
      trader: Trader,
      before_trade: Callable = None,
      execute_trade: Callable = None,
      while_trade: List[Callable] = None,
      after_trade: Callable = None,
      check_interval: int = 30
  ):
    self.trader = trader
    self.trading = False
    self.prepared = False
    self.executed = False

    self.before_trade = before_trade
    self.execute_trade = execute_trade
    self.while_trade = while_trade
    self.after_trade = after_trade
    self.check_interval = check_interval

    self.whole_sub_id = None

  def _in_prepare_window(self, current_time: datetime) -> bool:
    return PREPARE_REBALANCE_START <= current_time.time() < EXECUTE_REBALANCE_END

  def _in_execute_window(self, current_time: datetime) -> bool:
    return EXECUTE_REBALANCE_START <= current_time.time() < EXECUTE_REBALANCE_END

  def start_check_trading(self):
    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    current_check_interval = self.check_interval
    while True:
      now = datetime.now()
      if not self.prepared:
        if is_trading_day(now.date()) and self._in_prepare_window(now):
          current_check_interval = self.check_interval
          trading_logger.debug("今日开盘预计算开始")
          executor = concurrent.futures.ThreadPoolExecutor()
          if self.before_trade:
            future = executor.submit(self.before_trade, self)
            future.result()
            trading_logger.debug("开盘预计算完成")
          self.prepared = True
        elif is_trading_day(now.date()) and now.time() >= EXECUTE_REBALANCE_END:
          current_check_interval = self.check_interval
          trading_logger.warning("已错过开盘调仓窗口，今日跳过调仓")
          self.prepared = True
          self.executed = True
        else:
          current_check_interval = min(10 * self.check_interval, 5 * 60)

      if self.prepared and not self.executed and is_trading_day(now.date()) and self._in_execute_window(now):
        current_check_interval = self.check_interval
        trading_logger.debug("今日开盘调仓执行开始")
        if self.execute_trade and executor is not None:
          future = executor.submit(self.execute_trade, self)
          future.result()
          trading_logger.debug("开盘调仓执行完成")
        self.executed = True

      if is_current_trading(now):
        if not self.trading:
          lark_sender.send_msg("今日交易开始")
          trading_logger.debug("今日交易开始")
          self.trading = True
        if self.while_trade and executor is not None:
          futures = [executor.submit(func, self) for func in self.while_trade]
      elif self.trading and time(15, 0) < now.time():
        self.trading = False
        self.prepared = False
        self.executed = False
        trading_logger.debug("今日交易结束")
        if self.after_trade and executor is not None:
          future = executor.submit(self.after_trade, self)
          future.result()
          trading_logger.debug("盘后处理完成")
        if executor is not None:
          executor.shutdown(wait=False, cancel_futures=True)
          executor = None
        rc = recorder.flush()
        msg = f"今日交易完成: \n{'\n'.join([f'{k}@{v}' for k, v in sorted(rc.items())])}"
        lark_sender.send_msg(msg)
        trading_logger.debug(msg)

      sys_time.sleep(current_check_interval)

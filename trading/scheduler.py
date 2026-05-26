import time as sys_time
import concurrent.futures
from datetime import datetime, time
from typing import Callable, List, Optional

from trading.logger import trading_logger
from utils.recorder import recorder
from utils.stock.time import is_current_trading, is_trading_day
from .trader import Trader
from .lark.sender import lark_sender

PREPARE_REBALANCE_START = time(9, 25)
EXECUTE_REBALANCE_START = time(9, 30)
EXECUTE_REBALANCE_END = time(10, 0)
POST_CLOSE_START = time(15, 0)
POST_CLOSE_END = time(16, 0)
UPDATE_ALL_START = time(16, 0)


class TradingScheduler:
  def __init__(
      self,
      trader: Trader,
      before_trade: Callable = None,
      execute_trade: Callable = None,
      while_trade: List[Callable] = None,
      after_trade: Callable = None,
      post_close: Callable = None,
      update_all: Callable = None,
      check_interval: int = 30,
      time_provider: Callable[[], datetime] = None,
      fast_forward: bool = False,
  ):
    self.trader = trader
    self.trading = False
    self.prepared = False
    self.executed = False
    self.post_close_done = False
    self.update_all_done = False

    self.before_trade = before_trade
    self.execute_trade = execute_trade
    self.while_trade = while_trade
    self.after_trade = after_trade
    self.post_close = post_close
    self.update_all = update_all
    self.check_interval = check_interval

    self.time_provider = time_provider or datetime.now
    self.fast_forward = fast_forward

    self.whole_sub_id = None

  def _now(self) -> datetime:
    return self.time_provider()

  def _in_prepare_window(self, current_time: datetime) -> bool:
    return PREPARE_REBALANCE_START <= current_time.time() < EXECUTE_REBALANCE_END

  def _in_execute_window(self, current_time: datetime) -> bool:
    return EXECUTE_REBALANCE_START <= current_time.time() < EXECUTE_REBALANCE_END

  def _advance_time(self, target: time):
    """快进模式：跳到下一个触发时间。"""
    if not self.fast_forward:
      return
    now = self._now()
    new_dt = datetime.combine(now.date(), target)
    if new_dt <= now:
      return  # 不倒退
    if hasattr(self.time_provider, 'set'):
      self.time_provider.set(new_dt)

  def start_check_trading(self):
    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    current_check_interval = self.check_interval
    while True:
      now = self._now()
      # ---- 09:25 准备 ----
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
          self._advance_time(EXECUTE_REBALANCE_START)
        elif is_trading_day(now.date()) and now.time() >= EXECUTE_REBALANCE_END:
          current_check_interval = self.check_interval
          trading_logger.warning("已错过开盘调仓窗口，今日跳过调仓")
          self.prepared = True
          self.executed = True
          self.post_close_done = True
          self._advance_time(POST_CLOSE_START)
        else:
          current_check_interval = min(10 * self.check_interval, 5 * 60)

      # ---- 09:30 执行 ----
      if self.prepared and not self.executed and is_trading_day(now.date()) and self._in_execute_window(now):
        current_check_interval = self.check_interval
        trading_logger.debug("今日开盘调仓执行开始")
        if self.execute_trade and executor is not None:
          future = executor.submit(self.execute_trade, self)
          future.result()
          trading_logger.debug("开盘调仓执行完成")
        self.executed = True
        self._advance_time(POST_CLOSE_START)

      # ---- 盘中 ----
      if is_current_trading(now):
        if not self.trading:
          lark_sender.send_msg("今日交易开始")
          trading_logger.debug("今日交易开始")
          self.trading = True
        if self.while_trade and executor is not None:
          futures = [executor.submit(func, self) for func in self.while_trade]

      # ---- 15:00 交易结束 + 盘后对比 ----
      elif self.trading and time(15, 0) <= now.time():
        self.trading = False
        self.prepared = False
        self.executed = False
        trading_logger.debug("今日交易结束")
        if self.after_trade and executor is not None:
          future = executor.submit(self.after_trade, self)
          future.result()
          trading_logger.debug("盘后处理完成")

        if not self.post_close_done and self.post_close and POST_CLOSE_START <= now.time() < POST_CLOSE_END:
          trading_logger.debug("盘后对比分析开始")
          if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor()
          future = executor.submit(self.post_close, self)
          future.result()
          trading_logger.debug("盘后对比分析完成")
          self.post_close_done = True

        self._advance_time(UPDATE_ALL_START)
        rc = recorder.flush()
        msg = f"今日交易完成: \n{'\n'.join([f'{k}@{v}' for k, v in sorted(rc.items())])}"
        lark_sender.send_msg(msg)
        trading_logger.debug(msg)

      # ---- 16:00 全量更新 ----
      if self.post_close_done and not self.update_all_done and self.update_all and UPDATE_ALL_START <= now.time():
        trading_logger.debug("全量数据更新开始")
        if executor is None:
          executor = concurrent.futures.ThreadPoolExecutor()
        future = executor.submit(self.update_all, self)
        future.result()
        trading_logger.debug("全量数据更新完成")
        self.update_all_done = True

      # ---- 清理 ----
      if self.update_all_done and executor is not None:
        executor.shutdown(wait=False)
        executor = None
        if self.fast_forward:
          trading_logger.info("快进模拟完成，退出")
          return

      sys_time.sleep(0 if self.fast_forward else current_check_interval)

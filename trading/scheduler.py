import time as sys_time
from datetime import datetime, time
from typing import Callable

from trading.logger import trading_logger
from utils.recorder import recorder
from utils.stock.time import is_current_trading, is_trading_day
from .trader import Trader
from .lark.sender import LarkMsgLevel, lark_sender

PREPARE_REBALANCE_START = time(9, 25, 10)
EXECUTE_REBALANCE_START = PREPARE_REBALANCE_START
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
      after_trade: Callable = None,
      post_close: Callable = None,
      update_all: Callable = None,
      check_interval: int = 1,
      time_provider: Callable[[], datetime] = None,
      fast_forward: bool = False,
  ):
    self.trader = trader
    self.trading = False
    self.prepared = False
    self.executed = False
    self.post_close_done = False
    self.update_all_done = False
    # 当前状态标志所归属的日期。用于检测跨日运行（进程未被 watchdog/人工重启）
    # 时主动重置当日状态，避免上一交易日残留的 True 标志让新交易日跳过调仓。
    self._state_day = None

    self.before_trade = before_trade
    self.execute_trade = execute_trade
    self.after_trade = after_trade
    self.post_close = post_close
    self.update_all = update_all
    self.check_interval = check_interval

    self.time_provider = time_provider or datetime.now
    self.fast_forward = fast_forward

    self.whole_sub_id = None

  def _now(self) -> datetime:
    return self.time_provider()

  def _reset_daily_state(self):
    """重置「当日」调度状态，使新交易日能重新走 预计算→执行→盘后→全量更新。

    动机：scheduler 主循环常驻、跨日不退出。若进程未被 watchdog/人工在 16:00 后
    重启，上一交易日跑完后 prepared/executed/post_close_done/update_all_done 全为
    True，会让新交易日的 line `if not self.prepared` / `not self.executed` 条件不
    成立而静默跳过调仓（仅 self.trading 被 15:00 重置为 False，导致只误发一张
    「盘中交易开始」卡片却零成交）。这里在跨日时把这些标志清零。
    """
    self.trading = False
    self.prepared = False
    self.executed = False
    self.post_close_done = False
    self.update_all_done = False

  def _check_day_rollover(self, now: datetime) -> bool:
    """检测是否跨入新的一天；若跨日则重置当日状态。返回是否发生了重置。

    首次调用仅记录当前日期、不重置（避免影响「启动时已错过窗口」等既有逻辑）。
    """
    today = now.date()
    if self._state_day is None:
      self._state_day = today
      return False
    if today == self._state_day:
      return False
    trading_logger.info(f"检测到跨日 {self._state_day} → {today}，重置当日调度状态")
    self._reset_daily_state()
    self._state_day = today
    return True

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

  def _handle_missed_open_window(self, now: datetime):
    """启动时已错过 09:25:10-10:00 调仓窗口的状态置位。

    根据当前时间精确设置 prepared/executed/post_close_done，
    保证「未到的环节」（15:00 实盘/回测 diff、16:00 update_all）后续仍能触发：
    - 启动时 10:00 <= now < 16:00：仅置 prepared/executed=True，post_close_done 保持 False，
      15:00 之后 line 126 elif 触发 post_close。
    - 启动时 now >= 16:00：post_close 窗口也错过，直接置位 post_close_done=True，
      line 171 触发 update_all（如果还在 update_all 时间范围内）。
    """
    trading_logger.warning("已错过开盘调仓窗口，今日跳过调仓")
    self.prepared = True
    self.executed = True
    if now.time() >= POST_CLOSE_END:
      self.post_close_done = True

  def start_check_trading(self):
    import os as _os
    current_check_interval = self.check_interval

    def _run_stage(callback):
      if callback is None:
        return
      try:
        callback(self)
      except KeyboardInterrupt:
        trading_logger.info("收到 Ctrl+C 中断，正在退出...")
        _os._exit(1)

    while True:
      now = self._now()
      # ---- 跨日重置：进程常驻跨日时，清掉上一交易日残留的状态标志 ----
      if self._check_day_rollover(now):
        current_check_interval = self.check_interval

      # ---- 09:25:10 准备 ----
      if not self.prepared:
        if is_trading_day(now.date()) and self._in_prepare_window(now):
          current_check_interval = self.check_interval
          trading_logger.debug("今日开盘预计算开始")
          _run_stage(self.before_trade)
          trading_logger.debug("开盘预计算完成")
          self.prepared = True
          self._advance_time(EXECUTE_REBALANCE_START)
        elif is_trading_day(now.date()) and now.time() >= EXECUTE_REBALANCE_END:
          current_check_interval = self.check_interval
          self._handle_missed_open_window(now)
          self._advance_time(POST_CLOSE_START)
        elif self.fast_forward and is_trading_day(now.date()) and now.time() < PREPARE_REBALANCE_START:
          current_check_interval = self.check_interval
          self._advance_time(PREPARE_REBALANCE_START)
        else:
          current_check_interval = min(10 * self.check_interval, 5 * 60)

      # ---- 预计算完成立即执行 ----
      if self.prepared and not self.executed and is_trading_day(now.date()) and self._in_execute_window(now):
        current_check_interval = self.check_interval
        trading_logger.debug("今日调仓执行开始")
        _run_stage(self.execute_trade)
        trading_logger.debug("今日调仓执行完成")
        self.executed = True
        self._advance_time(POST_CLOSE_START)

      # ---- 盘中 ----
      if is_current_trading(now):
        if not self.trading:
          lark_sender.send_notification_card(
            level=LarkMsgLevel.Info,
            title=f"📈 盘中交易开始 @ {now.strftime('%Y-%m-%d')}",
            sub_title=f"开盘时刻: {now.strftime('%H:%M:%S')}",
            content="盘中交易已开始，订单/成交回调进入飞书通知。")
          trading_logger.debug("今日交易开始")
          self.trading = True

      # ---- 15:00 交易结束 + 盘后对比 ----
      elif self.trading and time(15, 0) <= now.time():
        self.trading = False
        # 注意：不要 reset prepared/executed — 它们应保持 True 表示「今日已处理」，
        # 否则下个循环会进 line 91 的「已错过开盘调仓窗口」分支重复刷屏。
        # 跨日通过 _check_day_rollover 自动重置，支持连续多天运行。
        trading_logger.debug("今日交易结束")
        _run_stage(self.after_trade)
        trading_logger.debug("盘后处理完成")

        rc = recorder.flush()
        # v2 table：各阶段触发次数
        try:
            stage_rows = [{'stage': k, 'count': str(v + 1)} for k, v in sorted(rc.items())]
            lark_sender.send_table_card(
                title=f"✅ 今日交易完成 @ {now.strftime('%Y-%m-%d')}",
                level=LarkMsgLevel.Success,
                subtitle=f"共完成 {len(stage_rows)} 个阶段",
                tables=[{
                    'title': '**🕐 各阶段触发次数**',
                    'element_id': 'stages',
                    'columns': [
                        {'name': 'stage', 'display_name': '阶段', 'horizontal_align': 'left'},
                        {'name': 'count', 'display_name': '次数', 'horizontal_align': 'right'},
                    ],
                    'rows': stage_rows,
                }] if stage_rows else None,
                summary_md=None if stage_rows else '<font color="grey">无阶段标记</font>',
            )
        except Exception as e:
            trading_logger.warning(f"今日交易完成卡片发送失败: {e}")
        msg = '\n'.join([f'{k}@{v}' for k, v in sorted(rc.items())])
        trading_logger.debug(msg)

      # ---- 15:00-16:00 实盘/回测 diff（与 self.trading 解耦）----
      # 不再要求"今日交易过"才触发：只要在 post_close 窗口 + 交易日，diff 都跑一次。
      # 修复场景：--skip 202605281500（启动时间正好 15:00 整点，is_current_trading 严格 <15:00
      # 永远拿不到 self.trading=True）；或 13:00 启动后一直跑到 15:00 都能命中。
      if (not self.post_close_done and self.post_close
          and is_trading_day(now.date())
          and POST_CLOSE_START <= now.time() < POST_CLOSE_END):
        trading_logger.debug("盘后对比分析开始")
        _run_stage(self.post_close)
        trading_logger.debug("盘后对比分析完成")
        self.post_close_done = True
        self._advance_time(UPDATE_ALL_START)

      # ---- 16:00 全量更新 ----
      if self.post_close_done and not self.update_all_done and self.update_all and UPDATE_ALL_START <= now.time():
        trading_logger.debug("全量数据更新开始")
        _run_stage(self.update_all)
        trading_logger.debug("全量数据更新完成")
        self.update_all_done = True

      # ---- 清理 ----
      if self.update_all_done:
        if self.fast_forward:
          trading_logger.info("快进模拟完成，退出")
          return

      try:
        sys_time.sleep(0 if self.fast_forward else current_check_interval)
      except KeyboardInterrupt:
        trading_logger.info("收到 Ctrl+C 中断，正在退出...")
        _os._exit(1)

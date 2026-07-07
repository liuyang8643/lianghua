"""跨日重置回归测试 —— 防止「进程常驻跨日运行 → 新交易日带着上一日 True 标志静默跳过调仓」复发。

背景：2026-06-03 启动的实盘进程未在 16:00 被重启，跑到 2026-06-04 时
prepared/executed/post_close_done/update_all_done 全为 True，导致 06-04 尾盘
不再触发选股(before_trade)与下单(execute_trade)，只误发一张「行情开始」卡片却零成交。
"""
from datetime import datetime

from trading.scheduler import (
    EXECUTE_REBALANCE_START, PREPARE_REBALANCE_START, TradingScheduler,
)


class _Clock:
  """可控模拟时钟，暴露 __call__ 供 time_provider 使用。"""
  def __init__(self, dt: datetime):
    self._dt = dt
  def __call__(self) -> datetime:
    return self._dt
  def set(self, dt: datetime):
    self._dt = dt


def _make_scheduler(dt: datetime) -> TradingScheduler:
  clock = _Clock(dt)
  sch = TradingScheduler(trader=object(), time_provider=clock)
  return sch


def test_first_rollover_call_only_records_day():
  """首次调用只记录日期、不重置（不应干扰启动时既有窗口逻辑）。"""
  sch = _make_scheduler(datetime(2026, 6, 3, 9, 30))
  sch.prepared = sch.executed = True
  changed = sch._check_day_rollover(sch._now())
  assert changed is False
  assert sch._state_day == datetime(2026, 6, 3).date()
  # 首次不重置，残留标志保持原样
  assert sch.prepared is True and sch.executed is True


def test_same_day_no_reset():
  """同一交易日内多次轮询不触发重置。"""
  clock = _Clock(datetime(2026, 6, 3, 9, 30))
  sch = TradingScheduler(trader=object(), time_provider=clock)
  sch._check_day_rollover(sch._now())  # 记录 06-03
  sch.prepared = sch.executed = sch.post_close_done = sch.update_all_done = True

  clock.set(datetime(2026, 6, 3, 14, 59))
  changed = sch._check_day_rollover(sch._now())
  assert changed is False
  assert sch.prepared and sch.executed and sch.post_close_done and sch.update_all_done


def test_cross_day_resets_all_daily_flags():
  """跨日时清零全部当日状态，使新交易日能重新走完整流程。"""
  clock = _Clock(datetime(2026, 6, 3, 16, 5))
  sch = TradingScheduler(trader=object(), time_provider=clock)
  sch._check_day_rollover(sch._now())  # 记录 06-03
  # 模拟 06-03 跑完全流程后的残留状态
  sch.trading = False
  sch.prepared = True
  sch.executed = True
  sch.post_close_done = True
  sch.update_all_done = True

  clock.set(datetime(2026, 6, 4, 0, 5))
  changed = sch._check_day_rollover(sch._now())

  assert changed is True
  assert sch._state_day == datetime(2026, 6, 4).date()
  assert sch.prepared is False
  assert sch.executed is False
  assert sch.post_close_done is False
  assert sch.update_all_done is False
  assert sch.trading is False


def test_cross_day_allows_new_day_prepare_condition():
  """重置后，新交易日 15:00:30 的『not prepared』前置条件应重新成立。"""
  clock = _Clock(datetime(2026, 6, 3, 16, 5))
  sch = TradingScheduler(trader=object(), time_provider=clock)
  sch._check_day_rollover(sch._now())
  sch.prepared = True  # 上一日已准备

  clock.set(datetime(2026, 6, 4, 15, 0, 30))
  sch._check_day_rollover(sch._now())
  # 主循环 line `if not self.prepared` 的判定恢复为 True → 会重新触发 before_trade
  assert (not sch.prepared) is True


def test_execute_window_starts_at_1505():
  """盘后固定价格撮合 15:05 起才进入执行窗口。"""
  sch = _make_scheduler(datetime(2026, 6, 4, 15, 0, 30))
  assert sch._in_prepare_window(sch._now()) is True
  assert sch._in_execute_window(sch._now()) is False

  sch = _make_scheduler(datetime(2026, 6, 4, 15, 5, 0))
  assert sch._in_execute_window(sch._now()) is True


def test_rebalance_waits_until_150030():
  sch = _make_scheduler(datetime(2026, 6, 4, 15, 0, 29))
  assert sch._in_prepare_window(sch._now()) is False
  assert sch._in_execute_window(sch._now()) is False

  sch = _make_scheduler(datetime(2026, 6, 4, 15, 0, 30))
  assert sch._in_prepare_window(sch._now()) is True
  assert sch._in_execute_window(sch._now()) is False

  sch = _make_scheduler(datetime(2026, 6, 4, 15, 4, 59))
  assert sch._in_prepare_window(sch._now()) is True
  assert sch._in_execute_window(sch._now()) is False

  sch = _make_scheduler(datetime(2026, 6, 4, 15, 5, 0))
  assert sch._in_execute_window(sch._now()) is True


def test_fast_forward_runs_prepare_and_execute_synchronously(monkeypatch):
  calls = []
  clock = _Clock(datetime(2026, 6, 4, 15, 0, 30))
  sch = TradingScheduler(
      trader=object(), time_provider=clock, fast_forward=True,
      before_trade=lambda store: calls.append("before"),
      execute_trade=lambda store: calls.append("execute"),
      post_close=lambda store: calls.append("post_close"),
      update_all=lambda store: calls.append("update_all"),
  )
  monkeypatch.setattr("trading.scheduler.is_trading_day", lambda d: True)
  monkeypatch.setattr("trading.scheduler.is_current_trading", lambda now: False)

  sch.start_check_trading()

  assert calls == ["before", "execute", "post_close", "update_all"]


def test_fast_forward_advances_from_before_window_to_rebalance_start(monkeypatch):
  calls = []
  clock = _Clock(datetime(2026, 6, 4, 9, 25))
  sch = TradingScheduler(
      trader=object(), time_provider=clock, fast_forward=True,
      before_trade=lambda store: calls.append(("before", store._now().time())),
      execute_trade=lambda store: calls.append(("execute", store._now().time())),
      post_close=lambda store: calls.append(("post_close", store._now().time())),
      update_all=lambda store: calls.append(("update_all", store._now().time())),
  )
  monkeypatch.setattr("trading.scheduler.is_trading_day", lambda d: True)
  monkeypatch.setattr("trading.scheduler.is_current_trading", lambda now: False)

  sch.start_check_trading()

  assert calls[0] == ("before", PREPARE_REBALANCE_START)
  assert calls[1] == ("execute", EXECUTE_REBALANCE_START)
  assert [name for name, _ in calls] == ["before", "execute", "post_close", "update_all"]

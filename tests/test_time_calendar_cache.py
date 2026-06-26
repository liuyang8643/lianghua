"""交易日历读一次放内存 + is_current_trading 先判时段再查日历 的单元测试。"""
from datetime import datetime

import pytest

import utils.stock.time as st


def test_is_current_trading_short_circuits_outside_hours(monkeypatch):
    """不在交易时段时直接短路，绝不调用 is_trading_day（不读日历 parquet）。"""
    def _boom(*a, **k):
        raise AssertionError("不在交易时段不应查交易日历")

    monkeypatch.setattr(st, "is_trading_day", _boom)

    # 12:00 处于午休（11:30~13:00 之外的交易时段），trading_hours=False
    assert st.is_current_trading(datetime(2026, 6, 1, 12, 0)) is False
    # 08:00 开盘前
    assert st.is_current_trading(datetime(2026, 6, 1, 8, 0)) is False
    # 16:00 收盘后
    assert st.is_current_trading(datetime(2026, 6, 1, 16, 0)) is False


def test_is_current_trading_checks_calendar_inside_hours(monkeypatch):
    """处于交易时段时才查交易日历，结果取决于 is_trading_day。"""
    calls = []

    monkeypatch.setattr(st, "is_trading_day", lambda d: (calls.append(d) or True))
    assert st.is_current_trading(datetime(2026, 6, 1, 10, 0)) is True
    assert len(calls) == 1  # 交易时段内确实查了日历

    monkeypatch.setattr(st, "is_trading_day", lambda d: False)
    assert st.is_current_trading(datetime(2026, 6, 1, 14, 0)) is False


def test_calendar_state_cached_in_memory():
    """连续调用返回同一个对象（读一次放内存，不重复读 parquet）。"""
    st._TRADING_CALENDAR_STATE = None  # 复位，确保从磁盘读一次
    first = st._get_trading_calendar_state()
    if first[1] is None:
        pytest.skip("trading_calendar.parquet 不存在，跳过缓存断言")
    second = st._get_trading_calendar_state()
    assert first is second  # 同一对象 → 命中内存缓存

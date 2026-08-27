"""LegalityChecker 合成单元测试（快速、不依赖真实 runtime）。

覆盖核心分支逻辑：
  - 取整偏严：涨停价向下取整(_floor_2)、跌停价向上取整(_ceil_2)
  - 板块日常涨跌幅 + ST 按板块（主板5%/创业板20%/科创板20%/北交所30%；创业板注册制前ST5%；主板2026-07-06起ST10%）
  - 注册制免限期（前5日/北交所首日）可买、不参与涨跌停判定
  - 老规则 IPO 首日 +44% / 开盘+20% 封板（一字/秒封）禁买

真实数据样本测试见 tests/test_legality_realdata_*.py。
"""
from datetime import date

import numpy as np

from core.legality import LegalityChecker, _floor_2, _ceil_2

# 各板块用真实前缀代码（board_type 由前缀推断）
BOARD_CODE = {0: '600519.SH', 1: '300750.SZ', 2: '688981.SH', 3: '830799.BJ'}
N = 80
TRADE_DATES = (np.datetime64('2000-01-01') + np.arange(N)).astype('datetime64[D]')


def _make_checker(board, *, trade_idx, open_t, preclose=np.nan, st=False,
                  volume_t=1.0, list_tidx=-1, high_t=np.nan, low_t=np.nan,
                  close_t=np.nan, issue_price=np.nan,
                  limit_up_protection=False):
    code = BOARD_CODE[board]
    o = np.full((N, 1), np.nan); c = np.full((N, 1), np.nan)
    v = np.zeros((N, 1))
    pc = np.full((N, 1), np.nan)
    h = np.full((N, 1), np.nan); l = np.full((N, 1), np.nan)
    o[trade_idx, 0] = open_t; h[trade_idx, 0] = high_t
    l[trade_idx, 0] = low_t; c[trade_idx, 0] = close_t
    v[trade_idx, 0] = volume_t
    if trade_idx > 0 and not np.isnan(preclose):
        c[trade_idx - 1, 0] = preclose
        pc[trade_idx, 0] = preclose
    st_mask = np.zeros((N, 1), dtype=bool); st_mask[trade_idx, 0] = st
    data = dict(stock_codes=np.array([code]), trade_dates=TRADE_DATES,
                open=o, close=c, high=h, low=l, volume=v, preClose=pc, st_mask=st_mask,
                issue_price=np.array([issue_price]))
    effective_list_tidx = list_tidx if list_tidx >= 0 else 0
    list_map = {code: TRADE_DATES[effective_list_tidx].item()}
    return LegalityChecker(
        data, {code: 0}, list_map,
        limit_up_protection=limit_up_protection,
    )


def _buy(board, signal_date, **kw):
    ck = _make_checker(board, **kw)
    ok, _ = ck.check([0], kw['trade_idx'], signal_date, is_buy=True)
    return bool(ok[0])


def _sell(board, signal_date, **kw):
    ck = _make_checker(board, **kw)
    ok, _ = ck.check([0], kw['trade_idx'], signal_date, is_buy=False)
    return bool(ok[0])


SIG = date(2018, 6, 1)        # 普通时段（注册制前）
SIG_REG = date(2023, 6, 1)    # 创业板/科创板注册制后


# ---------- 取整助手 ----------
def test_floor_ceil_helpers():
    assert abs(_floor_2(np.array([11.055]))[0] - 11.05) < 1e-9   # 涨停向下
    assert abs(_ceil_2(np.array([9.032]))[0] - 9.04) < 1e-9      # 跌停向上


def test_legality_requires_no_current_hlcv_or_amount_fields():
    code = BOARD_CODE[0]
    open_price = np.full((N, 1), np.nan)
    pre_close = np.full((N, 1), np.nan)
    open_price[30, 0] = 10.0
    pre_close[30, 0] = 10.0
    data = {
        'stock_codes': np.array([code]),
        'trade_dates': TRADE_DATES,
        'open': open_price,
        'preClose': pre_close,
        'st_mask': np.zeros((N, 1), dtype=bool),
        'issue_price': np.array([np.nan]),
    }
    checker = LegalityChecker(
        data, {code: 0}, {code: TRADE_DATES[0].item()},
    )

    buy_ok, _ = checker.check([0], 30, SIG, is_buy=True)
    sell_ok, _ = checker.check([0], 30, SIG, is_buy=False)

    assert buy_ok.tolist() == [True]
    assert sell_ok.tolist() == [True]


# ---------- 涨停取整偏严 ----------
def test_zero_volume_does_not_change_open_legality():
    assert _buy(0, SIG, trade_idx=30, open_t=10.0, preclose=10.0, volume_t=0.0) is True


def test_buy_uptick_floor_strict():
    # 主板 preclose=10.05 → 真实涨停 round(11.055)=11.06；严格 floor=11.05
    assert _buy(0, SIG, trade_idx=30, open_t=11.05, preclose=10.05) is False
    assert _buy(0, SIG, trade_idx=30, open_t=11.04, preclose=10.05) is True


def test_sell_downtick_ceil_strict():
    assert _sell(0, SIG, trade_idx=30, open_t=9.04, preclose=10.04) is False
    assert _sell(0, SIG, trade_idx=30, open_t=9.05, preclose=10.04) is True


def test_limit_up_protection_for_001260_next_open_scenarios():
    """001260.SZ is protected only if the simulated next open is limit-up."""
    next_day = date(2026, 8, 14)
    # 2026-08-13 close=21.03; the strict main-board next-day limit is 23.13.
    assert _sell(
        0, next_day, trade_idx=30, open_t=23.13, preclose=21.03,
        limit_up_protection=True,
    ) is False
    assert _sell(
        0, next_day, trade_idx=30, open_t=22.00, preclose=21.03,
        limit_up_protection=True,
    ) is True


# ---------- ST 按板块 ----------
def test_main_board_st_5pct():
    assert _buy(0, SIG, trade_idx=30, open_t=10.60, preclose=10.0, st=True) is False
    assert _buy(0, SIG, trade_idx=30, open_t=10.60, preclose=10.0, st=False) is True


def test_main_board_st_10pct_after_2026_07_06():
    assert _buy(0, date(2026, 8, 1), trade_idx=30, open_t=10.60, preclose=10.0, st=True) is True
    assert _buy(0, date(2026, 7, 1), trade_idx=30, open_t=10.60, preclose=10.0, st=True) is False


def test_cyb_st_20pct_after_registration():
    assert _buy(1, SIG_REG, trade_idx=30, open_t=11.50, preclose=10.0, st=True) is True


def test_cyb_st_5pct_before_registration():
    assert _buy(1, SIG, trade_idx=30, open_t=10.60, preclose=10.0, st=True) is False
    assert _buy(1, SIG, trade_idx=30, open_t=10.60, preclose=10.0, st=False) is True


def test_kcb_st_20pct():
    assert _buy(2, SIG_REG, trade_idx=30, open_t=11.50, preclose=10.0, st=True) is True


def test_bj_st_30pct():
    assert _buy(3, SIG_REG, trade_idx=30, open_t=12.50, preclose=10.0, st=True) is True
    assert _buy(3, SIG_REG, trade_idx=30, open_t=13.10, preclose=10.0, st=True) is False


# ---------- 注册制免限期 ----------
def test_registration_new_stock_first5_buyable():
    assert _buy(1, SIG_REG, trade_idx=20, open_t=30.0, preclose=10.0, list_tidx=18) is True


def test_bj_first_day_exempt_buyable():
    assert _buy(3, date(2022, 1, 5), trade_idx=20, open_t=50.0, list_tidx=20, issue_price=10.0) is True


# ---------- 老规则 IPO 首日：开盘顶发行价 120% 时无法成交 ----------
def test_ipo_old_rule_open_cap_is_blocked_without_using_hlc():
    assert _buy(0, date(2017, 1, 13), trade_idx=20, open_t=2.08, low_t=2.08,
                high_t=2.49, close_t=2.49, list_tidx=20, issue_price=1.73) is False


def test_ipo_old_rule_not_sealed_buyable():
    assert _buy(0, date(2017, 1, 13), trade_idx=20, open_t=1.90, low_t=1.90,
                high_t=2.49, close_t=2.49, list_tidx=20, issue_price=1.73) is True


def test_ipo_open_cap_is_blocked_when_current_hlc_is_unknown():
    assert _buy(0, date(2017, 1, 13), trade_idx=20, open_t=2.08,
                list_tidx=20, issue_price=1.73) is False

"""主板（沪深主板，含原中小板 002/003）买卖合法性——真实 runtime 数据回归测试。

每个 case 用真实股票 + 真实交易日硬编码，注释写明该 bar 的关键数值（open/preclose/
涨跌幅/st/board/issue_price），断言 LegalityChecker（经 conftest 的 market fixture）判定
与制度预期一致，并核对 market.bar 关键字段确认样本确属该 case。

涨停价向下取整、跌停价向上取整（与 core/legality.py 一致）：
  floor2(x)=floor(x*100)/100, ceil2(x)=ceil(x*100)/100
"""
import math
from datetime import date

import pytest


def _floor2(v):
    return math.floor(v * 100 + 1e-9) / 100.0


def _ceil2(v):
    return math.ceil(v * 100 - 1e-9) / 100.0


# ====================================================================
# 规则 1：日常普通股 ±10%（主板，非 ST，远离上市初期）
# ====================================================================

def test_main_daily_limit_up_buy_false(market):
    """涨停禁买：主板普通股开盘价 >= floor(前收×1.10) → buy=False。

    样本 000001.SZ(平安银行) 2024-10-08：open=13.43, preclose=12.21,
    涨幅 +9.99%（floor(12.21×1.10)=13.43，开盘一字顶涨停），board=0, st=False。
    """
    code, d = '000001.SZ', date(2024, 10, 8)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is False
    assert abs(b['open'] - 13.43) < 1e-6 and abs(b['preclose'] - 12.21) < 1e-6
    assert b['open'] >= _floor2(b['preclose'] * 1.10) - 1e-9      # 触及 +10% 涨停
    assert market.buy(code, d) is False


def test_main_daily_limit_down_sell_false(market):
    """跌停禁卖：主板普通股开盘价 <= ceil(前收×0.90) → sell=False。

    样本 000002.SZ(万科A) 2016-07-05：open=19.79, preclose=21.99,
    跌幅 -10.00%（ceil(21.99×0.90)=19.80，开盘一字跌停），board=0, st=False。
    """
    code, d = '000002.SZ', date(2016, 7, 5)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is False
    assert abs(b['open'] - 19.79) < 1e-6 and abs(b['preclose'] - 21.99) < 1e-6
    assert b['open'] <= _ceil2(b['preclose'] * 0.90) + 1e-9      # 触及 -10% 跌停
    assert market.sell(code, d) is False


def test_main_daily_normal_buy_true(market):
    """普通日可买：开盘价在 ±10% 区间内且未触涨跌停 → buy=True。

    样本 000001.SZ(平安银行) 2014-12-25：open=14.35, preclose=14.07,
    涨幅 +1.99%（远低于 +10% 涨停 floor(14.07×1.10)=15.47），board=0, st=False。
    """
    code, d = '000001.SZ', date(2014, 12, 25)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is False
    assert abs(b['open'] - 14.35) < 1e-6 and abs(b['preclose'] - 14.07) < 1e-6
    assert b['open'] < _floor2(b['preclose'] * 1.10) - 0.02     # 未到涨停
    assert b['open'] > _ceil2(b['preclose'] * 0.90) + 0.02      # 未到跌停
    assert market.buy(code, d) is True


# ====================================================================
# 规则 2：ST/*ST ±5%（2026-07-06 前历史样本一律 ±5%）
#   关键：开盘恰好顶 5% 涨停但 < 10% 线 → 证明用的是 5% 而非 10%。
# ====================================================================

def test_main_st_limit_up_5pct_buy_false_recent(market):
    """ST 涨停禁买(±5%)：open 顶 floor(前收×1.05) 但 < floor(前收×1.10) → buy=False。

    样本 000048.SZ(*ST,当时为康达尔) 2018-08-06：open=21.02, preclose=20.02,
    涨幅 +5.00%（floor(20.02×1.05)=21.02 触 5% 涨停；floor(20.02×1.10)=22.02，
    若按 10% 规则则尚未涨停应可买）→ 证明主板 ST 用 ±5%。board=0, st=True。
    """
    code, d = '000048.SZ', date(2018, 8, 6)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is True
    assert abs(b['open'] - 21.02) < 1e-6 and abs(b['preclose'] - 20.02) < 1e-6
    assert b['open'] >= _floor2(b['preclose'] * 1.05) - 1e-9    # 顶 5% 涨停
    assert b['open'] < _floor2(b['preclose'] * 1.10) - 0.02     # 但未到 10% 线
    assert market.buy(code, d) is False


def test_main_st_limit_up_5pct_buy_false_old(market):
    """ST 涨停禁买(±5%)——另一只一字板样本。

    样本 000008.SZ(当时 ST 渤海) 2002-06-24：open=12.73, preclose=12.12,
    涨幅 +5.03%（floor(12.12×1.05)=12.72 触 5% 涨停，一字 high=low=12.73），
    board=0, st=True。floor(12.12×1.10)=13.33，按 10% 规则不会涨停。
    """
    code, d = '000008.SZ', date(2002, 6, 24)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is True
    assert abs(b['open'] - 12.73) < 1e-6 and abs(b['preclose'] - 12.12) < 1e-6
    assert b['open'] >= _floor2(b['preclose'] * 1.05) - 1e-9
    assert b['open'] < _floor2(b['preclose'] * 1.10) - 0.02
    assert market.buy(code, d) is False


def test_main_st_normal_buy_true(market):
    """ST 普通日可买：开盘在 ±5% 区间内且未触涨跌停 → buy=True。

    样本 000008.SZ 2002-05-08：open=13.78, preclose=13.94, 涨幅 -1.15%，
    board=0, st=True（在 ±5% 区间内，可正常买入）。
    """
    code, d = '000008.SZ', date(2002, 5, 8)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is True
    assert abs(b['open'] - 13.78) < 1e-6 and abs(b['preclose'] - 13.94) < 1e-6
    assert b['open'] < _floor2(b['preclose'] * 1.05) - 0.02     # 未到 5% 涨停
    assert b['open'] > _ceil2(b['preclose'] * 0.95) + 0.02      # 未到 5% 跌停
    assert market.buy(code, d) is True


# ====================================================================
# 规则 3：老规则 IPO 首日（主板 2014-01-01 ~ 2023-04-09）
#   开盘上限=发行价×1.20、涨停=发行价×1.44；一字/秒封 → buy=False。
# ====================================================================

def test_main_ipo_firstday_sealed_known(market):
    """老规则 IPO 首日一字/秒封禁买（已知样本）。

    样本 603690.SH 2017-01-13：issue_price=1.73, open=2.08, low=2.08,
    high=2.49, close=2.49, board=0。
    floor(1.73×1.20)=2.07≈open（顶 +20% 开盘上限），floor(1.73×1.44)=2.49=close
    （封 +44% 涨停），low==open（一字/秒封，9:30 排不上买单）→ buy=False。
    """
    code, d = '603690.SH', date(2017, 1, 13)
    b = market.bar(code, d)
    assert b['board'] == 0
    ip = b['issue_price']
    assert abs(ip - 1.73) < 1e-6 and abs(b['open'] - 2.08) < 1e-6
    assert b['open'] >= _floor2(ip * 1.20) - 1e-9               # 顶 +20% 开盘上限
    assert b['close'] >= _floor2(ip * 1.44) - 1e-9             # 封 +44% 涨停
    assert b['low'] >= b['open'] - 1e-9                         # 一字：low==open
    assert market.buy(code, d) is False


def test_main_ipo_firstday_sealed_2(market):
    """老规则 IPO 首日一字/秒封禁买（第二个样本）。

    样本 001208.SZ(华菱线缆) 2021-06-24：issue_price=3.67, open=4.40,
    low=4.40, high=5.28, close=5.28, board=0。
    floor(3.67×1.20)=4.40=open，floor(3.67×1.44)=5.28=close，low==open
    → 一字/秒封 → buy=False。
    """
    code, d = '001208.SZ', date(2021, 6, 24)
    b = market.bar(code, d)
    assert b['board'] == 0
    ip = b['issue_price']
    assert abs(ip - 3.67) < 1e-6 and abs(b['open'] - 4.40) < 1e-6
    assert b['open'] >= _floor2(ip * 1.20) - 1e-9
    assert b['close'] >= _floor2(ip * 1.44) - 1e-9
    assert b['low'] >= b['open'] - 1e-9
    assert market.buy(code, d) is False


def test_main_ipo_firstday_open_not_sealed_buy_true(market):
    """老规则 IPO 首日「非一字、开盘未顶 +20%」→ 可买（对照）。

    样本 601598.SH(中国外运) 2019-01-18：issue_price=5.24, open=5.30,
    low=4.77, board=0。open=5.30 << floor(5.24×1.20)=6.28（开盘远未顶上限），
    且 low<open（盘中破发，非一字）→ 9:30 能买到 → buy=True。
    """
    code, d = '601598.SH', date(2019, 1, 18)
    b = market.bar(code, d)
    assert b['board'] == 0
    ip = b['issue_price']
    assert abs(ip - 5.24) < 1e-6 and abs(b['open'] - 5.30) < 1e-6
    assert b['open'] < _floor2(ip * 1.20) - 0.10              # 开盘未顶 +20%
    assert b['low'] < b['open']                                # 盘中破开盘价，非一字
    assert market.buy(code, d) is True


# ====================================================================
# 规则 4：主板全面注册制（2023-04-10 起上市）前 5 日不设涨跌幅
#   上市第 1~5 日即便大涨 → buy=True（不受涨停拦截）。
# ====================================================================

def test_main_reg_firstday_no_limit_buy_true(market):
    """注册制后主板新股上市首日不设涨跌幅 → buy=True。

    样本 001239.SZ(永达股份) 2023-12-12（上市首日）：issue_price=12.05,
    open=27.51（较发行价 +128%，远超 ±20%/±44% 任何老规则上限），board=0。
    上市日 >= 2023-04-10，属全面注册制新股前 5 日不设限 → buy=True。
    """
    code, d = '001239.SZ', date(2023, 12, 12)
    assert market.list_date(code) == date(2023, 12, 12)        # 确认为上市首日
    assert market.list_date(code) >= date(2023, 4, 10)
    b = market.bar(code, d)
    assert b['board'] == 0
    assert abs(b['open'] - 27.51) < 1e-6
    assert b['open'] > _floor2(b['issue_price'] * 1.20)        # 远超老规则 +20% 上限
    assert market.buy(code, d) is True


def test_main_reg_first5_big_gain_buy_true(market):
    """注册制后主板新股上市前 5 日内大涨仍可买（开盘 +29% vs 前收，不受 10% 涨停拦截）。

    样本 603275.SH(众辰科技) 2023-08-28（上市于 2023-08-23，为第 4 个交易日 ds=3）：
    open=63.20, preclose=49.00, 涨幅 +28.98%, board=0。
    若按日常 ±10%（floor(49×1.10)=53.90）早已涨停禁买，但处于注册制前 5 日不设限
    窗口 → buy=True。
    """
    code, d = '603275.SH', date(2023, 8, 28)
    assert market.list_date(code) == date(2023, 8, 23)         # 上市日，d 为上市后第4个交易日
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is False
    assert abs(b['open'] - 63.20) < 1e-6 and abs(b['preclose'] - 49.00) < 1e-6
    assert b['open'] > _floor2(b['preclose'] * 1.10)          # 远超 +10% 涨停线
    assert market.buy(code, d) is True


# ====================================================================
# 规则 5：2014-01-01 前上市的主板股首日不设涨跌幅 → buy=True
# ====================================================================

def test_main_pre2014_firstday_no_limit_buy_true(market):
    """2014 年前主板新股上市首日不设涨跌幅 → buy=True。

    样本 002334.SZ(英威腾) 2010-01-13（上市首日，runtime 内有该日数据）：
    issue_price=48.00, open=65.00（较发行价 +35%，远超后来 +20% 开盘上限），
    board=0。上市日 < 2014-01-01，首日不设涨跌幅 → buy=True。
    """
    code, d = '002334.SZ', date(2010, 1, 13)
    assert market.list_date(code) == date(2010, 1, 13)
    assert market.list_date(code) < date(2014, 1, 1)
    b = market.bar(code, d)
    assert b['board'] == 0
    assert abs(b['issue_price'] - 48.00) < 1e-6 and abs(b['open'] - 65.00) < 1e-6
    assert b['open'] > _floor2(b['issue_price'] * 1.20)       # 超 +20%，证明首日无限制
    assert market.buy(code, d) is True

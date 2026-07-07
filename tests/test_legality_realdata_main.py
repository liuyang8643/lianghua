"""主板（沪深主板，含原中小板 002/003）买卖合法性——真实 runtime 数据回归测试。

尾盘收盘交易口径：合法性只看 close[T] 与 preClose[T]（前收）。每个 case 用真实股票 +
真实交易日硬编码，注释写明该 bar 的关键数值（close/preclose/涨跌幅/st/board/issue_price），
断言 LegalityChecker（经 conftest 的 market fixture）判定与制度预期一致，并核对 market.bar
关键字段确认样本确属该 case。

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
    """涨停禁买：主板普通股收盘价 >= floor(前收×1.10) → buy=False。

    样本 000021.SZ(深科技) 2024-10-08：close=18.18, preclose=16.53,
    收盘涨幅 +9.98%（floor(16.53×1.10)=18.18，收盘封涨停），board=0, st=False。
    """
    code, d = '000021.SZ', date(2024, 10, 8)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is False
    assert abs(b['close'] - 18.18) < 1e-6 and abs(b['preclose'] - 16.53) < 1e-6
    assert b['close'] >= _floor2(b['preclose'] * 1.10) - 1e-9      # 收盘触及 +10% 涨停
    assert market.buy(code, d) is False


def test_main_daily_limit_down_sell_false(market):
    """跌停禁卖：主板普通股收盘价 <= ceil(前收×0.90) → sell=False。

    样本 000002.SZ(万科A) 2016-07-05：close=19.79, preclose=21.99,
    收盘跌幅 -10.00%（ceil(21.99×0.90)=19.80，一字跌停收盘封死），board=0, st=False。
    """
    code, d = '000002.SZ', date(2016, 7, 5)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is False
    assert abs(b['close'] - 19.79) < 1e-6 and abs(b['preclose'] - 21.99) < 1e-6
    assert b['close'] <= _ceil2(b['preclose'] * 0.90) + 1e-9      # 触及 -10% 跌停
    assert market.sell(code, d) is False


def test_main_daily_normal_buy_true(market):
    """普通日可买：收盘价在 ±10% 区间内且未触涨跌停 → buy=True。

    样本 000001.SZ(平安银行) 2014-12-25：close=14.69, preclose=14.07,
    收盘涨幅 +4.41%（远低于 +10% 涨停 floor(14.07×1.10)=15.47），board=0, st=False。
    """
    code, d = '000001.SZ', date(2014, 12, 25)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is False
    assert abs(b['close'] - 14.69) < 1e-6 and abs(b['preclose'] - 14.07) < 1e-6
    assert b['close'] < _floor2(b['preclose'] * 1.10) - 0.02     # 未到涨停
    assert b['close'] > _ceil2(b['preclose'] * 0.90) + 0.02      # 未到跌停
    assert market.buy(code, d) is True


# ====================================================================
# 规则 2：ST/*ST ±5%（2026-07-06 前历史样本一律 ±5%）
#   关键：收盘恰好顶 5% 涨停但 < 10% 线 → 证明用的是 5% 而非 10%。
# ====================================================================

def test_main_st_limit_up_5pct_buy_false_recent(market):
    """ST 涨停禁买(±5%)：close 顶 floor(前收×1.05) 但 < floor(前收×1.10) → buy=False。

    样本 000048.SZ(*ST,当时为康达尔) 2018-08-06：close=21.02, preclose=20.02,
    收盘涨幅 +5.00%（floor(20.02×1.05)=21.02 收盘触 5% 涨停；floor(20.02×1.10)=22.02，
    若按 10% 规则则尚未涨停应可买）→ 证明主板 ST 用 ±5%。board=0, st=True。
    """
    code, d = '000048.SZ', date(2018, 8, 6)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is True
    assert abs(b['close'] - 21.02) < 1e-6 and abs(b['preclose'] - 20.02) < 1e-6
    assert b['close'] >= _floor2(b['preclose'] * 1.05) - 1e-9    # 收盘顶 5% 涨停
    assert b['close'] < _floor2(b['preclose'] * 1.10) - 0.02     # 但未到 10% 线
    assert market.buy(code, d) is False


def test_main_st_limit_up_5pct_buy_false_old(market):
    """ST 涨停禁买(±5%)——另一只一字板样本。

    样本 000008.SZ(当时 ST 渤海) 2002-06-24：close=12.73, preclose=12.12,
    收盘涨幅 +5.03%（floor(12.12×1.05)=12.72 收盘触 5% 涨停，一字 high=low=close=12.73），
    board=0, st=True。floor(12.12×1.10)=13.33，按 10% 规则不会涨停。
    """
    code, d = '000008.SZ', date(2002, 6, 24)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is True
    assert abs(b['close'] - 12.73) < 1e-6 and abs(b['preclose'] - 12.12) < 1e-6
    assert b['close'] >= _floor2(b['preclose'] * 1.05) - 1e-9
    assert b['close'] < _floor2(b['preclose'] * 1.10) - 0.02
    assert market.buy(code, d) is False


def test_main_st_normal_buy_true(market):
    """ST 普通日可买：收盘在 ±5% 区间内且未触涨跌停 → buy=True。

    样本 000008.SZ 2002-05-08：close=14.13, preclose=13.94, 收盘涨幅 +1.36%，
    board=0, st=True（在 ±5% 区间内，可正常买入）。
    """
    code, d = '000008.SZ', date(2002, 5, 8)
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is True
    assert abs(b['close'] - 14.13) < 1e-6 and abs(b['preclose'] - 13.94) < 1e-6
    assert b['close'] < _floor2(b['preclose'] * 1.05) - 0.02     # 未到 5% 涨停
    assert b['close'] > _ceil2(b['preclose'] * 0.95) + 0.02      # 未到 5% 跌停
    assert market.buy(code, d) is True


# ====================================================================
# 规则 3：老规则 IPO 首日（主板 2014-01-01 ~ 2023-04-09）
#   盘中涨停=发行价×1.44；收盘封 +44%（close>=floor(发行价×1.44)）→ buy=False。
# ====================================================================

def test_main_ipo_firstday_sealed_known(market):
    """老规则 IPO 首日收盘封 +44% 涨停 → 禁买（已知样本）。

    样本 603690.SH 2017-01-13：issue_price=1.73, close=2.49, board=0。
    首日涨跌幅基准=发行价，floor(1.73×1.44)=2.49=close（收盘封 +44% 涨停，
    由通用涨停判定 ratios=0.44 覆盖）→ buy=False。
    """
    code, d = '603690.SH', date(2017, 1, 13)
    b = market.bar(code, d)
    assert b['board'] == 0
    ip = b['issue_price']
    assert abs(ip - 1.73) < 1e-6 and abs(b['close'] - 2.49) < 1e-6
    assert b['close'] >= _floor2(ip * 1.44) - 1e-9             # 收盘封 +44% 涨停
    assert market.buy(code, d) is False


def test_main_ipo_firstday_sealed_2(market):
    """老规则 IPO 首日收盘封 +44% 涨停 → 禁买（第二个样本）。

    样本 001208.SZ(华菱线缆) 2021-06-24：issue_price=3.67, close=5.28, board=0。
    floor(3.67×1.44)=5.28=close（收盘封 +44% 涨停）→ buy=False。
    """
    code, d = '001208.SZ', date(2021, 6, 24)
    b = market.bar(code, d)
    assert b['board'] == 0
    ip = b['issue_price']
    assert abs(ip - 3.67) < 1e-6 and abs(b['close'] - 5.28) < 1e-6
    assert b['close'] >= _floor2(ip * 1.44) - 1e-9
    assert market.buy(code, d) is False


def test_main_ipo_firstday_open_not_sealed_buy_true(market):
    """老规则 IPO 首日「收盘未封 +44% 涨停」→ 可买（对照）。

    样本 601598.SH(中国外运) 2019-01-18：issue_price=5.24, close=4.89, board=0。
    盘中破发回落，close=4.89 << floor(5.24×1.44)=7.54（收盘远未封涨停）
    → 尾盘可成交 → buy=True。
    """
    code, d = '601598.SH', date(2019, 1, 18)
    b = market.bar(code, d)
    assert b['board'] == 0
    ip = b['issue_price']
    assert abs(ip - 5.24) < 1e-6 and abs(b['close'] - 4.89) < 1e-6
    assert b['close'] < _floor2(ip * 1.44) - 0.10             # 收盘未封 +44% 涨停
    assert market.buy(code, d) is True


# ====================================================================
# 规则 4：主板全面注册制（2023-04-10 起上市）前 5 日不设涨跌幅
#   上市第 1~5 日即便收盘大涨 → buy=True（不受涨停拦截）。
# ====================================================================

def test_main_reg_firstday_no_limit_buy_true(market):
    """注册制后主板新股上市首日不设涨跌幅 → buy=True。

    样本 001239.SZ(永达股份) 2023-12-12（上市首日）：issue_price=12.05,
    close=23.72（较发行价 +97%，远超 ±20%/±44% 任何老规则上限），board=0。
    上市日 >= 2023-04-10，属全面注册制新股前 5 日不设限 → buy=True。
    """
    code, d = '001239.SZ', date(2023, 12, 12)
    assert market.list_date(code) == date(2023, 12, 12)        # 确认为上市首日
    assert market.list_date(code) >= date(2023, 4, 10)
    b = market.bar(code, d)
    assert b['board'] == 0
    assert abs(b['close'] - 23.72) < 1e-6
    assert b['close'] > _floor2(b['issue_price'] * 1.44)       # 收盘远超老规则 +44% 涨停
    assert market.buy(code, d) is True


def test_main_reg_first5_big_gain_buy_true(market):
    """注册制后主板新股上市前 5 日内收盘大涨仍可买（收盘 +34.5% vs 前收，不受 10% 涨停拦截）。

    样本 001373.SZ(慧智微) 2023-06-02（上市于 2023-06-01，为第 2 个交易日 ds=1）：
    close=61.94, preclose=46.05, 收盘涨幅 +34.51%, board=0, st=False。
    若按日常 ±10%（floor(46.05×1.10)=50.65）收盘早已远超涨停线应禁买，但处于注册制
    前 5 日不设限窗口 → buy=True。
    """
    code, d = '001373.SZ', date(2023, 6, 2)
    list_d = market.list_date(code)
    assert list_d == date(2023, 6, 1)                          # 上市日
    assert list_d >= date(2023, 4, 10)
    ds = market.didx(d) - market.didx(list_d)
    assert 1 <= ds <= 4                                        # 上市后前 5 日窗口内
    b = market.bar(code, d)
    assert b['board'] == 0 and b['st'] is False
    assert abs(b['close'] - 61.94) < 1e-6 and abs(b['preclose'] - 46.05) < 1e-6
    assert b['close'] > _floor2(b['preclose'] * 1.10)          # 收盘远超 +10% 涨停线
    assert market.buy(code, d) is True


# ====================================================================
# 规则 5：2014-01-01 前上市的主板股首日不设涨跌幅 → buy=True
# ====================================================================

def test_main_pre2014_firstday_no_limit_buy_true(market):
    """2014 年前主板新股上市首日不设涨跌幅 → buy=True。

    样本 002334.SZ(英威腾) 2010-01-13（上市首日，runtime 内有该日数据）：
    issue_price=48.00, close=68.83（较发行价 +43.4%，超 +20% 上限），board=0。
    上市日 < 2014-01-01，首日不设涨跌幅（若按日常 10% 则 floor(48×1.10)=52.80 应涨停禁买）
    → buy=True。
    """
    code, d = '002334.SZ', date(2010, 1, 13)
    assert market.list_date(code) == date(2010, 1, 13)
    assert market.list_date(code) < date(2014, 1, 1)
    b = market.bar(code, d)
    assert b['board'] == 0
    assert abs(b['issue_price'] - 48.00) < 1e-6 and abs(b['close'] - 68.83) < 1e-6
    assert b['close'] > _floor2(b['issue_price'] * 1.20)       # 超 +20%，证明首日无涨跌幅限制
    assert market.buy(code, d) is True

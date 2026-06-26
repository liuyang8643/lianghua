"""创业板（300/301）买卖合法性闸门真实数据回归测试。

用生产 runtime NPZ 里真实存在的（股票, 日期）样本，验证 `core.legality.LegalityChecker`
对创业板各时段/各规则的判定与监管制度一致：
  1. 注册制前(2014-01-01~2020-08-23) 日常 ±10%
  2. 注册制后(2020-08-24 起) 日常 ±20%
  3. 注册制前 ST ±5%
  4. 注册制后 ST ±20%
  5. 注册制后上市前 5 日不设涨跌幅
  6. 老规则 IPO 首日：开盘≤发行价×1.20、盘中≤发行价×1.44（一字/秒封买不进）

所有样本 (code, date) 硬编码，注释标注 bar 关键数值（open/preclose/涨跌幅/st/board），
可追溯。创业板注册制切换基准日 = 2020-08-24。
"""
from datetime import date

import numpy as np
import pytest


def _floor2(v):
    """与 LegalityChecker 一致的向下取整到分（涨停价偏严）。"""
    return np.floor(v * 100.0 + 1e-9) / 100.0


# ===================== CASE 1：注册制前 日常 ±10% =====================

def test_cyb_reg_before_daily_limit_up_reject(market):
    """注册制前(<2020-08-24)创业板日常涨跌幅 ±10%：开盘一字涨停 → 禁买。

    样本 300001.SZ 2014-02-17：
      open=26.38 preclose=23.98 涨幅≈+10.01% st=False board=1(创业板)
      涨停价 floor(23.98×1.10)=26.37，open≥涨停价 → 涨停禁买。
    """
    code, d = '300001.SZ', date(2014, 2, 17)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is False
    assert d < date(2020, 8, 24)
    assert bar['open'] >= _floor2(bar['preclose'] * 1.10) - 1e-3   # 触 +10% 涨停
    assert market.buy(code, d) is False


def test_cyb_reg_before_daily_normal_buyable(market):
    """注册制前创业板普通交易日（开盘未触涨跌停）→ 可买（对照）。

    样本 300001.SZ 2015-01-16：
      open=20.14 preclose=19.84 涨幅≈+1.51% st=False board=1
      open 远低于 +10% 涨停价 → 正常可成交。
    """
    code, d = '300001.SZ', date(2015, 1, 16)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is False
    assert bar['open'] < _floor2(bar['preclose'] * 1.10) - 0.02
    assert market.buy(code, d) is True


# ===================== CASE 2：注册制后 日常 ±20% =====================

def test_cyb_reg_after_daily_15pct_buyable(market):
    """注册制后(≥2020-08-24)创业板日常涨跌幅 ±20%：开盘 +16% 未涨停 → 可买。

    样本 300001.SZ 2024-10-08：
      open=25.52 preclose=22.00 涨幅≈+16.00% st=False board=1
      +16% 已超旧 ±10% 上限，却低于 ±20% 涨停价 floor(22.00×1.20)=26.40
      → 仅当采用 20% 才可买，证明用的是 20% 而非 10%。
    """
    code, d = '300001.SZ', date(2024, 10, 8)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is False
    assert d >= date(2020, 8, 24)
    pct = (bar['open'] / bar['preclose'] - 1.0) * 100.0
    assert 13.0 < pct < 18.0                                        # 介于 10% 与 20% 之间
    assert bar['open'] < _floor2(bar['preclose'] * 1.20) - 0.02     # 未触 +20%
    assert market.buy(code, d) is True


def test_cyb_reg_after_daily_limit_up_reject(market):
    """注册制后创业板 +20% 一字涨停 → 禁买（对照）。

    样本 300008.SZ 2024-10-08：
      open=5.39 preclose=4.49 涨幅≈+20.04% st=False board=1
      涨停价 floor(4.49×1.20)=5.38，open≥涨停价 → 禁买。
    """
    code, d = '300008.SZ', date(2024, 10, 8)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is False
    assert bar['open'] >= _floor2(bar['preclose'] * 1.20) - 1e-3
    assert market.buy(code, d) is False


# ===================== CASE 3：注册制前 ST ±5% =====================

def test_cyb_reg_before_st_5pct(market):
    """注册制前创业板 ST 股 ±5% 限制。

    runtime st_mask 在 2020-08-24 之前对全部创业板股票均无 ST 标记（实测命中 0 条），
    无法构造真实样本，跳过。
    """
    pytest.skip('runtime st_mask 在 2020-08-24 前对创业板无任何 ST 标记（共 0 条），无真实样本')


# ===================== CASE 4：注册制后 ST ±20% =====================

def test_cyb_reg_after_st_15pct_buyable(market):
    """注册制后创业板 ST 股涨跌幅 ±20%：开盘 +14.8% 未涨停 → 可买。

    样本 300029.SZ 2020-11-17：
      open=6.90 preclose=6.01 涨幅≈+14.81% st=True board=1
      +14.8% 远超旧 ST ±5% 上限，却低于 ST ±20% 涨停价 floor(6.01×1.20)=7.21
      → 仅当 ST 采用 20% 才可买，证明用的是 20% 而非 5%。
    """
    code, d = '300029.SZ', date(2020, 11, 17)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is True
    assert d >= date(2020, 8, 24)
    pct = (bar['open'] / bar['preclose'] - 1.0) * 100.0
    assert pct > 5.0                                               # 已超旧 ST 5% 上限
    assert bar['open'] < _floor2(bar['preclose'] * 1.20) - 0.02    # 未触 ST 20% 涨停
    assert market.buy(code, d) is True


def test_cyb_reg_after_st_limit_up_reject(market):
    """注册制后创业板 ST 股 +20% 一字涨停 → 禁买（对照）。

    样本 300044.SZ 2022-04-28：
      open=2.39 preclose=1.99 涨幅≈+20.10% st=True board=1
      ST 涨停价 floor(1.99×1.20)=2.38，open≥涨停价 → 禁买。
    """
    code, d = '300044.SZ', date(2022, 4, 28)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is True
    assert bar['open'] >= _floor2(bar['preclose'] * 1.20) - 1e-3
    assert market.buy(code, d) is False


# ===================== CASE 5：注册制后上市前 5 日不设涨跌幅 =====================

def test_cyb_reg_after_new_listing_first_day_buyable(market):
    """注册制后创业板新股上市首日（首批 300861）不设涨跌幅 → 大涨仍可买。

    样本 300861.SZ：上市日 2020-08-24（注册制首批），发行价 43.76。
      首日 open=60.00，较发行价 +37% 远超 ±20%，但首日(ds=0)不设涨跌幅 → 可买。
    """
    code, d = '300861.SZ', date(2020, 8, 24)
    assert market.list_date(code) == d                            # 确认确是上市首日
    bar = market.bar(code, d)
    assert bar['board'] == 1
    assert bar['open'] / bar['issue_price'] - 1.0 > 0.20          # 较发行价大涨超 20%
    assert market.buy(code, d) is True


def test_cyb_reg_after_new_listing_day2to5_buyable(market):
    """注册制后创业板新股上市第 2~5 日仍不设涨跌幅 → 开盘破 20% 仍可买。

    样本 301618.SZ：上市日 2020-08-24 之后的注册制新股，上市日 2024-09-30。
      第 2 个交易日 2024-10-08(ds=1)：open=500.22 preclose=381.00 涨幅≈+31.3%
      已突破 ±20% 涨停价 floor(381.00×1.20)=457.20，按日常规则应禁买；
      但上市前 5 日不设涨跌幅 → 仍可买，证明 ds≤4 豁免生效。
    """
    code, d = '301618.SZ', date(2024, 10, 8)
    list_d = market.list_date(code)
    assert list_d == date(2024, 9, 30)                            # 上市日
    # d 为上市后第 2 个交易日（ds=1，仍在前 5 日窗口内）
    ds = market.didx(d) - market.didx(list_d)
    assert 1 <= ds <= 4
    bar = market.bar(code, d)
    assert bar['board'] == 1
    assert bar['open'] >= _floor2(bar['preclose'] * 1.20) - 1e-3  # 若按日常 20% 则属涨停
    assert market.buy(code, d) is True                            # 但前 5 日豁免 → 可买


# ===================== CASE 6：老规则 IPO 首日（2014~2020-08-23）=====================

def test_cyb_old_ipo_first_day_sealed_reject(market):
    """老规则创业板 IPO 首日一字/秒封 → 买不进。

    样本 300357.SZ 上市首日 2014-01-21：发行价 20.05。
      开盘上限=floor(20.05×1.20)=24.06，盘中涨停=floor(20.05×1.44)=28.87。
      open=24.06(顶开盘上限) close=29.15(封盘中涨停) low=24.06(==open，全天未下探)
      → 一字/秒封，实盘集合竞价排不上买单 → 禁买。
    """
    code, d = '300357.SZ', date(2014, 1, 21)
    assert market.list_date(code) == d                            # 确认是 IPO 首日
    bar = market.bar(code, d)
    assert bar['board'] == 1
    assert date(2014, 1, 1) <= d < date(2020, 8, 24)
    ip = bar['issue_price']
    assert bar['open'] >= _floor2(ip * 1.20) - 1e-3               # 顶 +20% 开盘上限
    assert bar['close'] >= _floor2(ip * 1.44) - 1e-3              # 封 +44% 盘中涨停
    assert bar['low'] >= bar['open'] - 1e-3                       # low==open，未下探
    assert market.buy(code, d) is False


def test_cyb_old_ipo_first_day_unsealed_buyable(market):
    """老规则创业板 IPO 首日非一字（盘中回落、未封涨停）→ 可买（对照）。

    样本 300360.SZ 上市首日 2014-01-21：发行价 55.11。
      盘中涨停=floor(55.11×1.44)=79.35。
      open=66.13 close=65.40(未封 79.35 涨停) low=64.88(<open，盘中下探)
      → 非一字，实盘可成交 → 可买。
    """
    code, d = '300360.SZ', date(2014, 1, 21)
    assert market.list_date(code) == d
    bar = market.bar(code, d)
    assert bar['board'] == 1
    ip = bar['issue_price']
    assert bar['close'] < _floor2(ip * 1.44) - 0.02              # 未封 +44% 涨停
    assert bar['low'] < bar['open'] - 0.02                       # 盘中曾下探，非一字
    assert market.buy(code, d) is True

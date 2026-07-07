"""创业板（300/301）买卖合法性闸门真实数据回归测试。

尾盘收盘交易口径：合法性只看 close[T] 与 preClose[T]（前收）。用生产 runtime NPZ 里真实
存在的（股票, 日期）样本，验证 `core.legality.LegalityChecker` 对创业板各时段/各规则的判定
与监管制度一致：
  1. 注册制前(2014-01-01~2020-08-23) 日常 ±10%
  2. 注册制后(2020-08-24 起) 日常 ±20%
  3. 注册制前 ST ±5%
  4. 注册制后 ST ±20%
  5. 注册制后上市前 5 日不设涨跌幅
  6. 老规则 IPO 首日：盘中涨停=发行价×1.44（收盘封 +44% 买不进）

所有样本 (code, date) 硬编码，注释标注 bar 关键数值（close/preclose/涨跌幅/st/board），
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
    """注册制前(<2020-08-24)创业板日常涨跌幅 ±10%：收盘涨停 → 禁买。

    样本 300001.SZ 2014-02-17：
      close=26.38 preclose=23.98 收盘涨幅≈+10.01% st=False board=1(创业板)
      涨停价 floor(23.98×1.10)=26.37，close≥涨停价 → 收盘封涨停禁买。
    """
    code, d = '300001.SZ', date(2014, 2, 17)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is False
    assert d < date(2020, 8, 24)
    assert bar['close'] >= _floor2(bar['preclose'] * 1.10) - 1e-3   # 收盘触 +10% 涨停
    assert market.buy(code, d) is False


def test_cyb_reg_before_daily_normal_buyable(market):
    """注册制前创业板普通交易日（收盘未触涨跌停）→ 可买（对照）。

    样本 300001.SZ 2015-01-16：
      close=20.39 preclose=19.84 收盘涨幅≈+2.77% st=False board=1
      close 远低于 +10% 涨停价 → 正常可成交。
    """
    code, d = '300001.SZ', date(2015, 1, 16)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is False
    assert bar['close'] < _floor2(bar['preclose'] * 1.10) - 0.02
    assert market.buy(code, d) is True


# ===================== CASE 2：注册制后 日常 ±20% =====================

def test_cyb_reg_after_daily_above10_below20_buyable(market):
    """注册制后(≥2020-08-24)创业板日常涨跌幅 ±20%：收盘超 +10% 未触 +20% → 可买。

    样本 300001.SZ 2024-10-08：
      close=24.52 preclose=22.00 收盘涨幅≈+11.45% st=False board=1
      收盘已超旧 ±10% 涨停价 floor(22.00×1.10)=24.20，却低于 ±20% 涨停价
      floor(22.00×1.20)=26.40 → 仅当采用 20% 才可买，证明用的是 20% 而非 10%。
    """
    code, d = '300001.SZ', date(2024, 10, 8)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is False
    assert d >= date(2020, 8, 24)
    assert bar['close'] > _floor2(bar['preclose'] * 1.10)          # 收盘超旧 10% 涨停线
    assert bar['close'] < _floor2(bar['preclose'] * 1.20) - 0.02   # 未触 +20%
    assert market.buy(code, d) is True


def test_cyb_reg_after_daily_limit_up_reject(market):
    """注册制后创业板 +20% 收盘涨停 → 禁买（对照）。

    样本 300006.SZ 2024-10-08：
      close=4.27 preclose=3.56 收盘涨幅≈+19.94% st=False board=1
      涨停价 floor(3.56×1.20)=4.27，close≥涨停价 → 收盘封涨停禁买。
    """
    code, d = '300006.SZ', date(2024, 10, 8)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is False
    assert bar['close'] >= _floor2(bar['preclose'] * 1.20) - 1e-3
    assert market.buy(code, d) is False


# ===================== CASE 3：注册制前 ST ±5% =====================

def test_cyb_reg_before_st_5pct(market):
    """注册制前创业板 ST 股 ±5% 限制。

    runtime st_mask 在 2020-08-24 之前对全部创业板股票均无 ST 标记（实测命中 0 条），
    无法构造真实样本，跳过。
    """
    pytest.skip('runtime st_mask 在 2020-08-24 前对创业板无任何 ST 标记（共 0 条），无真实样本')


# ===================== CASE 4：注册制后 ST ±20% =====================

def test_cyb_reg_after_st_above5_below20_buyable(market):
    """注册制后创业板 ST 股涨跌幅 ±20%：收盘超 +5% 未触 +20% → 可买。

    样本 300029.SZ 2020-11-17：
      close=6.37 preclose=6.01 收盘涨幅≈+5.99% st=True board=1
      收盘已超旧 ST ±5% 涨停价 floor(6.01×1.05)=6.31，却低于 ST ±20% 涨停价
      floor(6.01×1.20)=7.21 → 仅当 ST 采用 20% 才可买，证明用的是 20% 而非 5%。
    """
    code, d = '300029.SZ', date(2020, 11, 17)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is True
    assert d >= date(2020, 8, 24)
    assert bar['close'] > _floor2(bar['preclose'] * 1.05)          # 收盘超旧 ST 5% 涨停线
    assert bar['close'] < _floor2(bar['preclose'] * 1.20) - 0.02   # 未触 ST 20% 涨停
    assert market.buy(code, d) is True


def test_cyb_reg_after_st_limit_up_reject(market):
    """注册制后创业板 ST 股 +20% 收盘涨停 → 禁买（对照）。

    样本 300044.SZ 2022-04-28：
      close=2.39 preclose=1.99 收盘涨幅≈+20.10% st=True board=1
      ST 涨停价 floor(1.99×1.20)=2.38，close≥涨停价 → 收盘封涨停禁买。
    """
    code, d = '300044.SZ', date(2022, 4, 28)
    bar = market.bar(code, d)
    assert bar['board'] == 1 and bar['st'] is True
    assert bar['close'] >= _floor2(bar['preclose'] * 1.20) - 1e-3
    assert market.buy(code, d) is False


# ===================== CASE 5：注册制后上市前 5 日不设涨跌幅 =====================

def test_cyb_reg_after_new_listing_first_day_buyable(market):
    """注册制后创业板新股上市首日（首批 300861）不设涨跌幅 → 大涨仍可买。

    样本 300861.SZ：上市日 2020-08-24（注册制首批），发行价 43.76。
      首日 close=70.06，较发行价 +60% 远超 ±20%，但首日(ds=0)不设涨跌幅 → 可买。
    """
    code, d = '300861.SZ', date(2020, 8, 24)
    assert market.list_date(code) == d                            # 确认确是上市首日
    bar = market.bar(code, d)
    assert bar['board'] == 1
    assert bar['close'] / bar['issue_price'] - 1.0 > 0.20         # 收盘较发行价大涨超 20%
    assert market.buy(code, d) is True


def test_cyb_reg_after_new_listing_day2to5_buyable(market):
    """注册制后创业板新股上市第 2~5 日仍不设涨跌幅 → 收盘破 20% 仍可买。

    样本 301618.SZ：上市日 2024-09-30 的注册制新股。
      第 2 个交易日 2024-10-08(ds=1)：close=505.55 preclose=381.00 收盘涨幅≈+32.7%
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
    assert bar['close'] >= _floor2(bar['preclose'] * 1.20) - 1e-3  # 若按日常 20% 收盘属涨停
    assert market.buy(code, d) is True                            # 但前 5 日豁免 → 可买


# ===================== CASE 6：老规则 IPO 首日（2014~2020-08-23）=====================

def test_cyb_old_ipo_first_day_sealed_reject(market):
    """老规则创业板 IPO 首日收盘封 +44% 涨停 → 买不进。

    样本 300357.SZ 上市首日 2014-01-21：发行价 20.05。
      盘中涨停=floor(20.05×1.44)=28.87，close=29.15≥28.87（收盘封 +44% 涨停，
      由通用涨停判定 ratios=0.44 覆盖）→ 禁买。
    """
    code, d = '300357.SZ', date(2014, 1, 21)
    assert market.list_date(code) == d                            # 确认是 IPO 首日
    bar = market.bar(code, d)
    assert bar['board'] == 1
    assert date(2014, 1, 1) <= d < date(2020, 8, 24)
    ip = bar['issue_price']
    assert bar['close'] >= _floor2(ip * 1.44) - 1e-3              # 收盘封 +44% 涨停
    assert market.buy(code, d) is False


def test_cyb_old_ipo_first_day_unsealed_buyable(market):
    """老规则创业板 IPO 首日收盘未封涨停 → 可买（对照）。

    样本 300360.SZ 上市首日 2014-01-21：发行价 55.11。
      盘中涨停=floor(55.11×1.44)=79.35，close=65.40<79.35（收盘未封涨停）
      → 尾盘可成交 → 可买。
    """
    code, d = '300360.SZ', date(2014, 1, 21)
    assert market.list_date(code) == d
    bar = market.bar(code, d)
    assert bar['board'] == 1
    ip = bar['issue_price']
    assert bar['close'] < _floor2(ip * 1.44) - 0.02             # 收盘未封 +44% 涨停
    assert market.buy(code, d) is True

"""科创板（688）与北交所（83/87/43/92）买卖合法性真实数据测试。

尾盘收盘交易口径：合法性只看 close[T] 与 preClose[T]（前收）。用生产 runtime NPZ 里的真实
股票 + 真实日期样本，断言 core/legality.LegalityChecker（经 conftest 的 `market` fixture
封装）对各涨跌停规则 case 的判定与预期一致。

样本均为 (code, date) 硬编码，注释写明 bar 关键数值（close/preclose/涨跌幅%/st/board），
自解释可追溯。

板块编码：board==2 科创板 / board==3 北交所。

== 北交所说明 ==
当前 runtime 股票池里 **没有任何** 83/87/43/92 前缀代码（n_bj==0），故北交所全部 case
（首日不设限 / 日常±30% / ST±30%）均无法用真实样本覆盖，统一 pytest.skip。一旦 runtime
纳入北交所代码即可补样本。
"""
import math
from datetime import date

import pytest


def _floor2(v):
    """与 LegalityChecker 一致的向下取整到分（涨停价偏严）。"""
    return math.floor(v * 100 + 1e-9) / 100.0


# ============================================================================
# 科创板（688）—— 开市 2019-07-22，board==2
# ============================================================================


def test_kcb_ipo_first5days_no_limit_big_gain_buyable(market):
    """规则1：科创板上市前 5 日（ds<=4）不设涨跌幅，即便单日收盘大涨仍 buy=True。

    三个真实新股样本，收盘涨幅均远超日常 ±20%，证明前 5 日确无涨跌幅限制：
      - 688026.SH 洁特生物 2020-02-03：ds=2，close=122.04 / preclose=58.52 / +108.5% / 非ST / board=2
      - 688010.SH 福光股份 2019-07-24：ds=2，close=81.40 / preclose=54.50 / +49.4% / 非ST / board=2
      - 688615.SH 合合信息 2024-10-08：ds=3，close=350.00 / preclose=245.01 / +42.9% / 非ST / board=2
    """
    samples = [
        ('688026.SH', date(2020, 2, 3)),
        ('688010.SH', date(2019, 7, 24)),
        ('688615.SH', date(2024, 10, 8)),
    ]
    for code, d in samples:
        if not market.has(code):
            pytest.skip(f'{code} 不在 runtime 股票池')
        bar = market.bar(code, d)
        assert bar['board'] == 2, f'{code} 应为科创板'
        assert bar['st'] is False
        # 收盘涨幅远超 20%，只有"不设涨跌幅"窗口才可能出现
        pct = (bar['close'] / bar['preclose'] - 1) * 100
        assert pct > 22.0, f'{code} {d} 收盘涨幅 {pct:.2f}% 应远超日常 20%'
        # 确认确实落在上市前 5 个交易日内（ds<=4）
        ds = market.didx(d) - market.didx(market.list_date(code))
        assert 0 <= ds <= 4, f'{code} {d} ds={ds} 不在前 5 日'
        # 不设涨跌幅 → 即便大涨仍可买
        assert market.buy(code, d) is True, f'{code} {d} 上市初期应可买'


def test_kcb_daily_limit_up_20pct_not_buyable(market):
    """规则2：科创板第 6 日起日常 ±20%，收盘涨停（close>=floor(preclose*1.20)）→ buy=False。

    2024-10-08（国庆后首个交易日）科创板收盘一字封涨停样本：
      - 688002.SH 睿创微纳 close=47.32 / preclose=39.43 / +20.01% / 非ST / board=2 / ds=1262
      - 688008.SH 澜起科技 close=80.26 / preclose=66.88 / +20.01% / 非ST / board=2 / ds=1262
    收盘涨停禁买，但同一日并未跌停，故 sell 仍应为 True（验证只禁买不禁卖）。
    """
    samples = [
        ('688002.SH', date(2024, 10, 8)),
        ('688008.SH', date(2024, 10, 8)),
    ]
    for code, d in samples:
        if not market.has(code):
            pytest.skip(f'{code} 不在 runtime 股票池')
        bar = market.bar(code, d)
        assert bar['board'] == 2
        assert bar['st'] is False
        pct = (bar['close'] / bar['preclose'] - 1) * 100
        # 收盘恰好顶在 +20% 涨停
        assert 19.9 < pct < 20.2, f'{code} {d} 收盘涨幅 {pct:.2f}% 应≈+20%'
        assert bar['close'] >= _floor2(bar['preclose'] * 1.20) - 1e-3
        # 上市已逾 5 日（日常涨跌幅生效）
        assert market.didx(d) - market.didx(market.list_date(code)) >= 5
        assert market.buy(code, d) is False, f'{code} {d} 涨停应禁买'
        assert market.sell(code, d) is True, f'{code} {d} 涨停不影响卖出'


def test_kcb_daily_normal_day_buyable(market):
    """规则2 对照：科创板第 6 日起、非涨跌停的普通交易日 → buy=True。

      - 688001.SH 华兴源创 2019-07-29：close=55.48 / preclose=50.66 / +9.51% / 非ST / board=2 / ds=5
      - 688003.SH 天准科技 2019-07-31：close=52.20 / preclose=50.65 / +3.06% / 非ST / board=2 / ds=7
    收盘涨幅均在 ±20% 区间内且非涨停 → 正常可买。
    """
    samples = [
        ('688001.SH', date(2019, 7, 29)),
        ('688003.SH', date(2019, 7, 31)),
    ]
    for code, d in samples:
        if not market.has(code):
            pytest.skip(f'{code} 不在 runtime 股票池')
        bar = market.bar(code, d)
        assert bar['board'] == 2
        assert bar['st'] is False
        pct = (bar['close'] / bar['preclose'] - 1) * 100
        assert 0 < pct < 19.0, f'{code} {d} 收盘涨幅 {pct:.2f}% 应在区间内非涨停'
        assert market.didx(d) - market.didx(market.list_date(code)) >= 5
        assert market.buy(code, d) is True, f'{code} {d} 普通日应可买'


def test_kcb_st_uses_20pct_not_5pct(market):
    """规则3：科创板 ST 股日常仍按 ±20%（而非主板 ST 的 ±5%）。

    688282.SH 理工导航（科创板 ST 股）：
      - 2024-10-08：close=27.85 / preclose=26.12 / 收盘+6.62% / st=True / board=2
      - 2024-06-05：close=26.23 / preclose=22.96 / 收盘+14.24% / st=True / board=2
    两日收盘均已超 ST ±5% 涨停线 floor(preclose×1.05)（若按 5% 应涨停禁买），
    却低于 ±20% 涨停线 → 实际 buy=True，证明科创板 ST 用 ±20% 而非 ±5%。
    """
    code = '688282.SH'
    if not market.has(code):
        pytest.skip(f'{code} 不在 runtime 股票池')
    for d in [date(2024, 10, 8), date(2024, 6, 5)]:
        bar = market.bar(code, d)
        assert bar['board'] == 2
        assert bar['st'] is True, f'{code} {d} 应为 ST'
        # 收盘已超 5% 涨停线（5% 限幅下应涨停禁买）
        assert bar['close'] > _floor2(bar['preclose'] * 1.05), (
            f'{code} {d} 收盘应超 ST 5% 涨停线')
        # 但未触 20% 涨停线
        assert bar['close'] < _floor2(bar['preclose'] * 1.20) - 0.02
        # 在 20% 限幅内、未触顶 → 可买；证明用的是 20% 而非 5%
        assert market.buy(code, d) is True, (
            f'{code} {d} 收盘仍可买，证明科创板 ST 用 ±20% 而非 ±5%')


# ============================================================================
# 北交所（83/87/43/92）—— 开市 2021-11-15，board==3
# 当前 runtime 股票池无北交所代码，全部 skip
# ============================================================================

_BJ_PREFIXES = ('83', '87', '43', '92')


def _bj_codes(market):
    return [c for c in market.codes if c.startswith(_BJ_PREFIXES)]


def test_bj_first_day_no_limit_buyable(market):
    """规则4：北交所上市首日（ds==0）不设涨跌幅 → buy=True。

    当前 runtime 股票池无任何北交所代码，无真实样本可用，skip。
    """
    bj = _bj_codes(market)
    if not bj:
        pytest.skip('runtime 股票池无北交所（83/87/43/92）代码，无法验证首日不设限')
    pytest.skip(f'存在北交所代码 {bj[:5]} 但本测试样本待补充')


def test_bj_daily_limit_up_30pct_not_buyable(market):
    """规则5：北交所次日起日常 ±30%，收盘涨停（close>=floor(preclose*1.30)）→ buy=False。

    当前 runtime 股票池无北交所代码，无真实样本可用，skip。
    """
    bj = _bj_codes(market)
    if not bj:
        pytest.skip('runtime 股票池无北交所（83/87/43/92）代码，无法验证日常 ±30% 涨停')
    pytest.skip(f'存在北交所代码 {bj[:5]} 但本测试样本待补充')


def test_bj_st_uses_30pct_not_5pct(market):
    """规则6：北交所 ST 股仍按 ±30%（收盘≈+25% 仍 buy=True，证明非 5%）。

    当前 runtime 股票池无北交所代码，无真实样本可用，skip。
    """
    bj = _bj_codes(market)
    if not bj:
        pytest.skip('runtime 股票池无北交所（83/87/43/92）代码，无法验证 ST ±30%')
    pytest.skip(f'存在北交所代码 {bj[:5]} 但本测试样本待补充')

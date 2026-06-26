"""科创板（688）与北交所（83/87/43/92）买卖合法性真实数据测试。

用生产 runtime NPZ 里的真实股票 + 真实日期样本，断言 core/legality.LegalityChecker
（经 conftest 的 `market` fixture 封装）对各涨跌停规则 case 的判定与预期一致。

样本均为 (code, date) 硬编码，注释写明 bar 关键数值（open/preclose/涨跌幅%/st/board），
自解释可追溯。所有数值来自 runtime_1990-12-19_2026-05-29.npz 实扫描确认。

板块编码：board==2 科创板 / board==3 北交所。

== 北交所说明 ==
当前 runtime（截至 2026-05-29）股票池里 **没有任何** 83/87/43/92 前缀代码
（n_bj==0），故北交所全部 case（首日不设限 / 日常±30% / ST±30%）均无法用真实样本
覆盖，统一 pytest.skip。一旦 runtime 纳入北交所代码即可补样本。
"""
from datetime import date

import pytest

# ============================================================================
# 科创板（688）—— 开市 2019-07-22，board==2
# ============================================================================


def test_kcb_ipo_first5days_no_limit_big_gain_buyable(market):
    """规则1：科创板上市前 5 日（ds<=4）不设涨跌幅，即便单日大涨仍 buy=True。

    三个真实新股样本，开盘涨幅均远超日常 ±20%，证明前 5 日确无涨跌幅限制：
      - 688026.SH 洁特生物 2020-02-03：ds=2，open=76.08 / preclose=58.52 / +30.01% / 非ST / board=2
      - 688591.SH 泰凌微   2023-08-28：ds=1，open=42.05 / preclose=32.70 / +28.59% / 非ST / board=2
      - 688615.SH 合合信息 2024-10-08：ds=3，open=390.00 / preclose=245.01 / +59.18% / 非ST / board=2
    """
    samples = [
        ('688026.SH', date(2020, 2, 3)),
        ('688591.SH', date(2023, 8, 28)),
        ('688615.SH', date(2024, 10, 8)),
    ]
    for code, d in samples:
        if not market.has(code):
            pytest.skip(f'{code} 不在 runtime 股票池')
        bar = market.bar(code, d)
        assert bar['board'] == 2, f'{code} 应为科创板'
        assert bar['st'] is False
        # 开盘涨幅远超 20%，只有"不设涨跌幅"窗口才可能出现
        pct = (bar['open'] / bar['preclose'] - 1) * 100
        assert pct > 22.0, f'{code} {d} 开盘涨幅 {pct:.2f}% 应远超日常 20%'
        # 确认确实落在上市前 5 个交易日内（ds<=4）
        ds = market.didx(d) - market.didx(market.list_date(code))
        assert 0 <= ds <= 4, f'{code} {d} ds={ds} 不在前 5 日'
        # 不设涨跌幅 → 即便大涨仍可买
        assert market.buy(code, d) is True, f'{code} {d} 上市初期应可买'


def test_kcb_daily_limit_up_20pct_not_buyable(market):
    """规则2：科创板第 6 日起日常 ±20%，开盘涨停（open>=floor(preclose*1.20)）→ buy=False。

    2024-10-08（国庆后首个交易日）大量科创板一字/秒封涨停样本：
      - 688002.SH 睿创微纳 open=47.32 / preclose=39.43 / +20.01% / 非ST / board=2 / ds=1262
      - 688008.SH 澜起科技 open=80.26 / preclose=66.88 / +20.01% / 非ST / board=2 / ds=1262
    涨停禁买，但同一日并未跌停，故 sell 仍应为 True（验证只禁买不禁卖）。
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
        pct = (bar['open'] / bar['preclose'] - 1) * 100
        # 开盘恰好顶在 +20% 涨停
        assert 19.9 < pct < 20.2, f'{code} {d} 开盘涨幅 {pct:.2f}% 应≈+20%'
        # 上市已逾 5 日（日常涨跌幅生效）
        assert market.didx(d) - market.didx(market.list_date(code)) >= 5
        assert market.buy(code, d) is False, f'{code} {d} 涨停应禁买'
        assert market.sell(code, d) is True, f'{code} {d} 涨停不影响卖出'


def test_kcb_daily_normal_day_buyable(market):
    """规则2 对照：科创板第 6 日起、非涨跌停的普通交易日 → buy=True。

      - 688001.SH 华兴源创 2019-07-29：open=51.40 / preclose=50.66 / +1.46% / 非ST / board=2 / ds=5
      - 688003.SH 天准科技 2019-07-31：open=54.50 / preclose=50.65 / +7.60% / 非ST / board=2 / ds=7
    开盘涨幅均在 ±20% 区间内且非涨停 → 正常可买。
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
        pct = (bar['open'] / bar['preclose'] - 1) * 100
        assert 0 < pct < 19.0, f'{code} {d} 开盘涨幅 {pct:.2f}% 应在区间内非涨停'
        assert market.didx(d) - market.didx(market.list_date(code)) >= 5
        assert market.buy(code, d) is True, f'{code} {d} 普通日应可买'


def test_kcb_st_uses_20pct_not_5pct(market):
    """规则3：科创板 ST 股日常仍按 ±20%（而非主板 ST 的 ±5%）。

    688282.SH 理工导航（科创板 ST 股）：
      - 2024-10-08：open=30.10 / preclose=26.12 / +15.24% / st=True / board=2 / ds=617
      - 2024-06-05：open=25.50 / preclose=22.96 / +11.06% / st=True / board=2 / ds=536（佐证样本）
    若 ST 限幅为 5%，+15.24% / +11.06% 的开盘根本不可能出现且必被判涨停禁买；
    实际 buy=True，证明科创板 ST 用的是 ±20% 而非 ±5%。
    """
    code = '688282.SH'
    if not market.has(code):
        pytest.skip(f'{code} 不在 runtime 股票池')
    for d, expect_pct_gt in [(date(2024, 10, 8), 14.0), (date(2024, 6, 5), 10.0)]:
        bar = market.bar(code, d)
        assert bar['board'] == 2
        assert bar['st'] is True, f'{code} {d} 应为 ST'
        pct = (bar['open'] / bar['preclose'] - 1) * 100
        # 开盘涨幅远超 5%，5% 限幅下不可能成立
        assert pct > expect_pct_gt, f'{code} {d} 涨幅 {pct:.2f}% 应远超 5%'
        assert pct < 20.0, f'{code} {d} 涨幅 {pct:.2f}% 仍在 20% 内（未涨停）'
        # 在 20% 限幅内、未触顶 → 可买；证明用的是 20% 而非 5%
        assert market.buy(code, d) is True, (
            f'{code} {d} 开盘 +{pct:.2f}% 仍可买，证明科创板 ST 用 ±20% 而非 ±5%')


# ============================================================================
# 北交所（83/87/43/92）—— 开市 2021-11-15，board==3
# 当前 runtime 股票池无北交所代码，全部 skip
# ============================================================================

_BJ_PREFIXES = ('83', '87', '43', '92')


def _bj_codes(market):
    return [c for c in market.codes if c.startswith(_BJ_PREFIXES)]


def test_bj_first_day_no_limit_buyable(market):
    """规则4：北交所上市首日（ds==0）不设涨跌幅 → buy=True。

    当前 runtime（截至 2026-05-29）股票池无任何北交所代码，无真实样本可用，skip。
    """
    bj = _bj_codes(market)
    if not bj:
        pytest.skip('runtime 股票池无北交所（83/87/43/92）代码，无法验证首日不设限')
    pytest.skip(f'存在北交所代码 {bj[:5]} 但本测试样本待补充')


def test_bj_daily_limit_up_30pct_not_buyable(market):
    """规则5：北交所次日起日常 ±30%，开盘涨停（open>=floor(preclose*1.30)）→ buy=False。

    当前 runtime 股票池无北交所代码，无真实样本可用，skip。
    """
    bj = _bj_codes(market)
    if not bj:
        pytest.skip('runtime 股票池无北交所（83/87/43/92）代码，无法验证日常 ±30% 涨停')
    pytest.skip(f'存在北交所代码 {bj[:5]} 但本测试样本待补充')


def test_bj_st_uses_30pct_not_5pct(market):
    """规则6：北交所 ST 股仍按 ±30%（open≈+25% 仍 buy=True，证明非 5%）。

    当前 runtime 股票池无北交所代码，无真实样本可用，skip。
    """
    bj = _bj_codes(market)
    if not bj:
        pytest.skip('runtime 股票池无北交所（83/87/43/92）代码，无法验证 ST ±30%')
    pytest.skip(f'存在北交所代码 {bj[:5]} 但本测试样本待补充')

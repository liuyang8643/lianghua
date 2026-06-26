"""真实 runtime 数据下的 LegalityChecker 边界场景验收。

覆盖三类容易出错、且只能靠真实样本证伪的判定：
  A. 临退禁买（退市日前 35 交易日窗口）——窗口内禁买、窗口外不受影响。
  B. 跳空突破涨停的"实际不设限"日（重组/复牌等）——open 远超理论涨停价，被涨停判定一并拦下。
  C. 涨跌停取整偏严——真实一字涨停禁买、一字跌停禁卖。

样本均为 (代码, 日期) 硬编码，并在注释中写明 bar 关键数值与归类依据，自解释可追溯。
数据源：data/runtime 下最新 runtime_*.npz（由 conftest 的 market fixture 加载）。
"""
from datetime import date

import pytest


def test_临退禁买_窗口内禁买窗口外放行(market):
    """A. 临退禁买：退市日前 35 交易日窗口内 buy==False，窗口之前正常交易日 buy==True。

    样本1 600001.SH 邯郸钢铁，退市日 2009-12-29（吸收合并退市）：
      - 窗口内 2009-12-15（距退市 10 个交易日）：open=5.29 < preclose=5.35（非涨停，
        当日明显未触顶），buy 仍为 False，证明拦截来自临退禁买而非涨停。
      - 窗口外 2009-08-03（距退市 100 个交易日）：open=preclose=7.45（平开非涨停），
        buy==True，证明 35 日窗口之前不受临退影响。
    样本2 600068.SH 葛洲坝，退市日 2021-09-13（换股吸收合并退市）：
      - 窗口内 2021-08-30（距退市 10 个交易日）：open=9.28，preclose=9.23，buy==False。
      - 窗口外 2021-04-20（距退市 100 个交易日）：open=7.39，preclose=7.37，buy==True。
    """
    # —— 样本1：600001.SH 邯郸钢铁 ——
    if not market.has('600001.SH'):
        pytest.skip('当前 runtime 未包含历史退市样本 600001.SH')
    assert market.delist_date('600001.SH') == date(2009, 12, 29)

    d_in1 = date(2009, 12, 15)      # 退市前 10 个交易日，落在 35 日禁买窗口内
    bar_in1 = market.bar('600001.SH', d_in1)
    assert bar_in1['open'] == pytest.approx(5.29, abs=1e-2)
    assert bar_in1['preclose'] == pytest.approx(5.35, abs=1e-2)
    assert bar_in1['open'] < bar_in1['preclose']   # 当日下跌，绝非涨停，故禁买只能来自临退
    assert market.buy('600001.SH', d_in1) is False

    d_out1 = date(2009, 8, 3)       # 退市前 100 个交易日，在 35 日窗口之外
    bar_out1 = market.bar('600001.SH', d_out1)
    assert bar_out1['open'] == pytest.approx(7.45, abs=1e-2)
    assert bar_out1['preclose'] == pytest.approx(7.45, abs=1e-2)
    assert market.buy('600001.SH', d_out1) is True

    # —— 样本2：600068.SH 葛洲坝 ——
    assert market.has('600068.SH')
    assert market.delist_date('600068.SH') == date(2021, 9, 13)

    d_in2 = date(2021, 8, 30)       # 退市前 10 个交易日，窗口内
    bar_in2 = market.bar('600068.SH', d_in2)
    assert bar_in2['open'] == pytest.approx(9.28, abs=1e-2)
    assert market.buy('600068.SH', d_in2) is False

    d_out2 = date(2021, 4, 20)      # 退市前 100 个交易日，窗口外
    bar_out2 = market.bar('600068.SH', d_out2)
    assert bar_out2['open'] == pytest.approx(7.39, abs=1e-2)
    assert market.buy('600068.SH', d_out2) is True


def test_跳空突破涨停的实际不设限日被拦(market):
    """B. 跳空突破涨停的"实际不设限"日：open 远高于理论涨停价 → 被涨停判定一并拦下。

    样本 600705.SH（中航资本，借壳重组致股价跳变）2012-08-30：
      preclose=close[T-1]=4.44，open=14.00（ratio≈3.15），主板理论涨停价仅
      floor(4.44*1.10)=4.88。该日属"实际不设涨跌幅"的重组/复牌跳空日，runtime 无事件
      字段无法单独识别，但因 open 远超 4.88 会被涨停（limit_up）判定拦下，buy==False。
      这正是模块文档所述"跳空高开形态已被涨停判定一并拦下"的真实写照。
    """
    if not market.has('600705.SH'):
        pytest.skip('当前 runtime 未包含历史重组样本 600705.SH')
    d = date(2012, 8, 30)
    bar = market.bar('600705.SH', d)

    assert bar['board'] == 0                        # 主板，日常 ±10%
    assert bar['preclose'] == pytest.approx(4.44, abs=1e-2)
    assert bar['open'] == pytest.approx(14.00, abs=1e-2)

    # open 远高于理论涨停价（preclose*1.10≈4.88），跳空幅度 >200%
    theoretical_up_limit = bar['preclose'] * 1.10
    assert bar['open'] > theoretical_up_limit * 2

    # 被涨停判定一并拦下
    assert market.buy('600705.SH', d) is False


def test_涨跌停取整偏严_一字板禁买禁卖(market):
    """C. 涨跌停取整偏严：真实一字涨停禁买、真实一字跌停禁卖。

    样本1（涨停）002920.SZ 德赛西威 2018-01-02：
      preclose=39.13，主板涨停价 floor(39.13*1.10)=43.04，当日 open=high=low=43.04
      （一字板封死），buy==False。
    样本2（跌停）603016.SH 新宏泰 2018-01-02：
      preclose=36.14，主板跌停价 ceil(36.14*0.90)=32.53，当日 open=high=low=32.53
      （一字跌停），sell==False。
    """
    # —— 一字涨停：禁买 ——
    d_up = date(2018, 1, 2)
    bar_up = market.bar('002920.SZ', d_up)
    assert bar_up['board'] == 0
    assert bar_up['preclose'] == pytest.approx(39.13, abs=1e-2)
    assert bar_up['open'] == pytest.approx(43.04, abs=1e-2)
    # 一字板：开=高=低，封死涨停价
    assert bar_up['open'] == pytest.approx(bar_up['high'], abs=1e-6)
    assert bar_up['open'] == pytest.approx(bar_up['low'], abs=1e-6)
    # 取整偏严：涨停价向下取整 floor(39.13*1.10)=floor(43.043)=43.04
    assert bar_up['open'] == pytest.approx(43.04, abs=1e-2)
    assert market.buy('002920.SZ', d_up) is False

    # —— 一字跌停：禁卖 ——
    d_dn = date(2018, 1, 2)
    bar_dn = market.bar('603016.SH', d_dn)
    assert bar_dn['board'] == 0
    assert bar_dn['preclose'] == pytest.approx(36.14, abs=1e-2)
    assert bar_dn['open'] == pytest.approx(32.53, abs=1e-2)
    assert bar_dn['open'] == pytest.approx(bar_dn['high'], abs=1e-6)
    assert bar_dn['open'] == pytest.approx(bar_dn['low'], abs=1e-6)
    # 取整偏严：跌停价向上取整 ceil(36.14*0.90)=ceil(32.526)=32.53
    assert bar_dn['open'] == pytest.approx(32.53, abs=1e-2)
    assert market.sell('603016.SH', d_dn) is False

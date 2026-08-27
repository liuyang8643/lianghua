"""dim5 逐股盈亏:单边持有的票(如回测买进、实盘没买进)差额口径回归。

防止「单边持有票在逐行差额显示成 None、却计入合计」导致逐行加不出合计、方向看反。
"""
from datetime import date

import numpy as np
import pandas as pd

import trading.report as report_mod
from trading.report import PostCloseReport


def _bt_with_positions(positions):
    """构造最小回测结果:无 daily_snapshots → _rebuild 用 positions 当 T 日快照,
    daily_pnl = current_value(无昨日、无当日成交)。"""
    return {'positions': positions, 'trade_log': [], 'daily_snapshots': []}


def test_single_side_holding_diff_and_total_consistent():
    rpt = PostCloseReport(date(2026, 6, 1))

    # 实盘:持有 AAA(日盈亏 800),没持有 BBB
    rpt.feed_positions_df(pd.DataFrame([
        {'code': 'AAA', 'name': 'A', 'volume': 100, 'market_value': 1000.0,
         'daily_pnl': 800.0, 'daily_return_pct': 8.0},
    ]))
    # 回测:持有 AAA(1000)和 BBB(500)
    rpt.feed_backtest(_bt_with_positions([
        {'code': 'AAA', 'volume': 100, 'current_price': 10.0, 'current_value': 1000.0, 'avg_price': 9.0},
        {'code': 'BBB', 'volume': 100, 'current_price': 5.0, 'current_value': 500.0, 'avg_price': 4.0},
    ]))

    d5 = rpt.build_dim5_pnl()
    by_code = {r['code']: r for r in d5['rows']}

    # AAA 两边都有 → 差额 = 800 - 1000
    assert by_code['AAA']['pnl_diff'] == 800.0 - 1000.0
    # BBB 仅回测持有 → 实盘贡献 0,差额 = 0 - 500(不再是 None)
    assert by_code['BBB']['pnl_diff'] == -500.0

    # 合计:实盘 800、回测 1500;逐行差额之和 == 合计差额
    assert d5['live_total_pnl'] == 800.0
    assert d5['bt_total_pnl'] == 1500.0
    row_diff_sum = sum(r['pnl_diff'] for r in d5['rows'] if r['pnl_diff'] is not None)
    assert abs(row_diff_sum - (d5['live_total_pnl'] - d5['bt_total_pnl'])) < 1e-9


def test_reconcile_ignores_not_held_stocks():
    rpt = PostCloseReport(date(2026, 6, 1))
    rpt.feed_positions_df(pd.DataFrame([
        {'code': 'AAA', 'name': 'A', 'volume': 100, 'market_value': 1000.0,
         'daily_pnl': 800.0, 'daily_return_pct': 8.0},
    ]))
    rpt.feed_backtest(_bt_with_positions([
        {'code': 'AAA', 'volume': 100, 'current_price': 10.0, 'current_value': 1000.0, 'avg_price': 9.0},
        {'code': 'BBB', 'volume': 100, 'current_price': 5.0, 'current_value': 500.0, 'avg_price': 4.0},
    ]))
    rpt.feed_asset(total_asset=1_000_800.0, prev_asset=1_000_000.0, net_cash_flow=0.0)

    data = rpt.build()
    # 实盘没持有 BBB → 不应被列为"无法计算 P&L"
    assert 'BBB' not in (data['reconcile'].get('unreconcilable_codes') or [])


# ════════════════════════════════════════════════════════════
# 总盈亏以「个股盈亏总和」为准（免疫未记账出入金）+ 残差告警
# ════════════════════════════════════════════════════════════

def _rpt_with_live_aaa(daily_pnl, total_asset, prev_asset=1_000_000.0):
    """实盘只持有 AAA(给定 daily_pnl)，回测空仓，账户资产可调。"""
    rpt = PostCloseReport(date(2026, 6, 1))
    rpt.feed_positions_df(pd.DataFrame([
        {'code': 'AAA', 'name': 'A', 'volume': 100, 'market_value': 1000.0,
         'daily_pnl': daily_pnl, 'daily_return_pct': 8.0},
    ]))
    rpt.feed_backtest(_bt_with_positions([]))
    rpt.feed_asset(total_asset=total_asset, prev_asset=prev_asset, net_cash_flow=0.0)
    return rpt


def test_summary_uses_per_stock_total_as_authoritative_pnl():
    # 账户层日变化 = 50,800（含未记账的 5w 入金），个股口径 = 800
    rpt = _rpt_with_live_aaa(daily_pnl=800.0, total_asset=1_050_800.0)
    s = rpt.build()['summary']

    assert s['live_pnl_source'] == 'per_stock'
    assert abs(s['live_daily_pnl'] - 800.0) < 1e-6          # 今日盈亏 = 个股口径
    assert abs(s['live_account_pnl'] - 50_800.0) < 1e-6     # 账户口径单独保留
    # 收益率按个股口径 / prev_asset
    assert abs(s['live_daily_return_pct'] - 800.0 / 1_000_000.0 * 100) < 1e-9


def test_reconcile_residual_is_per_stock_minus_account():
    rpt = _rpt_with_live_aaa(daily_pnl=800.0, total_asset=1_050_800.0)
    rec = rpt.build()['reconcile']
    # 残差 = Σ个股 - 账户；不能因 summary 切到个股口径而自比为 0
    assert not rec['within_tolerance']
    assert abs(rec['diff'] - (800.0 - 50_800.0)) < 1e-6
    assert abs(rec['per_stock_pnl_sum'] - 800.0) < 1e-6
    assert abs(rec['account_pnl'] - 50_800.0) < 1e-6


def test_summary_falls_back_to_account_when_unreconcilable():
    # 持有 AAA 却算不出 daily_pnl(NaN) → 个股总和不可信 → 回退账户口径
    rpt = _rpt_with_live_aaa(daily_pnl=np.nan, total_asset=1_000_500.0)
    data = rpt.build()
    s = data['summary']
    assert s['live_pnl_source'] == 'account'
    assert abs(s['live_daily_pnl'] - 500.0) < 1e-6
    assert 'AAA' in (data['reconcile'].get('unreconcilable_codes') or [])


def test_summary_falls_back_to_account_when_snapshot_chain_is_broken(monkeypatch):
    rpt = _rpt_with_live_aaa(daily_pnl=40_000.0, total_asset=1_002_500.0)
    monkeypatch.setattr(rpt, '_chain_broken', lambda: True)
    dim5 = {
        'live_total_pnl': 40_000.0,
        'rows': [{
            'code': 'AAA', 'live_daily_pnl': 40_000.0,
            'live_volume': 100, 'live_yesterday_volume': 100,
        }],
    }

    summary = rpt.build_summary(dim5)
    reconcile = rpt.reconcile_pnl(dim5, summary)

    assert summary['live_pnl_source'] == 'account'
    assert summary['live_daily_pnl'] == 2_500.0
    assert reconcile['per_stock_pnl_sum'] is None
    assert reconcile['diff'] is None
    assert reconcile['unreconcilable_codes'] == ['AAA']


def test_reconcile_alert_fires_on_suspected_deposit(monkeypatch):
    calls = []
    monkeypatch.setattr(report_mod.lark_sender, 'send_notification_card',
                        lambda **k: calls.append(k))
    rpt = _rpt_with_live_aaa(daily_pnl=800.0, total_asset=1_050_800.0)
    data = rpt.build()
    rpt._maybe_alert_reconcile(data)

    assert len(calls) == 1
    content = calls[0]['content']
    # 账户 50,800 - 个股 800 = +50,000 疑似净入金
    assert '入金' in content
    assert '50,000' in content


def test_reconcile_alert_silent_when_within_tolerance(monkeypatch):
    calls = []
    monkeypatch.setattr(report_mod.lark_sender, 'send_notification_card',
                        lambda **k: calls.append(k))
    # 账户层 = 个股层 = 800，账平 → 不告警
    rpt = _rpt_with_live_aaa(daily_pnl=800.0, total_asset=1_000_800.0)
    data = rpt.build()
    rpt._maybe_alert_reconcile(data)
    assert calls == []


# ════════════════════════════════════════════════════════════
# 个股盈亏差异 & 滑点明细卡（每日盘后随 diff 报告增发）
# ════════════════════════════════════════════════════════════

def _fills_df(rows):
    return pd.DataFrame(rows, columns=[
        'code', 'name', 'direction', 'price', 'shares', 'amount',
        'fee_est', 'est_price', 'slippage_pct'])


def test_pnl_slippage_card_contents(monkeypatch):
    calls = []
    monkeypatch.setattr(report_mod.lark_sender, 'send_table_card',
                        lambda **k: calls.append(k))
    rpt = PostCloseReport(date(2026, 6, 12))
    rpt.feed_positions_df(pd.DataFrame([
        {'code': 'AAA', 'name': '甲股', 'volume': 100, 'market_value': 1000.0,
         'daily_pnl': 800.0, 'daily_return_pct': 8.0},
    ]))
    rpt.feed_backtest(_bt_with_positions([
        {'code': 'AAA', 'volume': 100, 'current_price': 10.0,
         'current_value': 1000.0, 'avg_price': 9.0},
    ]))
    # 两笔分批买入同一只票：滑点按 vwap 聚合（(10.5×100+10.7×100)/200=10.6 vs 开盘 10.0 → +6%）
    rpt.feed_fills_df(_fills_df([
        {'code': 'BBB', 'name': '乙股', 'direction': 'buy', 'price': 10.5,
         'shares': 100, 'amount': 1050.0, 'fee_est': 0.1, 'est_price': 10.0, 'slippage_pct': 5.0},
        {'code': 'BBB', 'name': '乙股', 'direction': 'buy', 'price': 10.7,
         'shares': 100, 'amount': 1070.0, 'fee_est': 0.1, 'est_price': 10.0, 'slippage_pct': 7.0},
        {'code': 'CCC', 'name': '丙股', 'direction': 'sell', 'price': 9.8,
         'shares': 200, 'amount': 1960.0, 'fee_est': 0.2, 'est_price': 10.0, 'slippage_pct': -2.0},
    ]))
    rpt.feed_asset(total_asset=1_000_800.0, prev_asset=1_000_000.0)

    rpt._send_pnl_slippage_card(rpt.build())

    assert len(calls) == 1
    card = calls[0]
    assert '盈亏差异 & 滑点' in card['title']
    tables = {t['element_id']: t for t in card['tables']}
    assert set(tables) == {'pnl_diff_tbl', 'slippage_tbl'}

    # 表1：AAA 实盘 800 vs 回测 1000 → 差 -200
    pnl_rows = tables['pnl_diff_tbl']['rows']
    aaa = next(r for r in pnl_rows if r['name'] == '甲股')
    assert '-200' in aaa['diff']

    # 表2：BBB 买入 vwap 10.6 vs 开盘 10.0 → 滑点 +6%、成本 +120；
    #       CCC 卖出 9.8 vs 10.0 → 滑点 -2%、成本 +40（卖便宜也是成本）
    slip = {r['name']: r for r in tables['slippage_tbl']['rows']}
    assert slip['乙股 买']['vwap'] == '10.60'
    assert '+6.00%' in slip['乙股 买']['sp']
    assert '+120' in slip['乙股 买']['cost']
    assert '+40' in slip['丙股 卖']['cost']
    # 汇总：总滑点成本 120 + 40 = 160
    assert '160' in card['summary_md']


def test_pnl_slippage_card_skipped_when_no_data(monkeypatch):
    calls = []
    monkeypatch.setattr(report_mod.lark_sender, 'send_table_card',
                        lambda **k: calls.append(k))
    rpt = PostCloseReport(date(2026, 6, 12))
    rpt.feed_backtest(_bt_with_positions([]))
    rpt._send_pnl_slippage_card(rpt.build())
    assert calls == []

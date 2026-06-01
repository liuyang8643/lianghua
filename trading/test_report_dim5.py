"""dim5 逐股盈亏:单边持有的票(如回测买进、实盘没买进)差额口径回归。

防止「单边持有票在逐行差额显示成 None、却计入合计」导致逐行加不出合计、方向看反。
"""
from datetime import date

import pandas as pd

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

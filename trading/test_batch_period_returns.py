"""batch 区间涨幅口径单测。"""
from scripts.batch_dryrun_postclose import _compound_daily_pct, _period_returns


def test_compound_daily_pct():
    # (+1%) 与 (-1%) 两日 → 约 0%
    assert abs(_compound_daily_pct([1.0, -1.0]) - 0.0) < 0.01


def test_period_diff_live_beats_bt_when_daily_sum_says_so():
    """逐日复利差应与「实盘日涨幅优于回测」同向。"""
    rows = [
        {'prev_asset': 100.0, 'live_asset': 110.0, 'bt_asset': 108.0,
         'net_cash_flow': 0, 'live_ret': 1.0, 'bt_ret': -1.0},
        {'prev_asset': 110.0, 'live_asset': 109.0, 'bt_asset': 107.0,
         'net_cash_flow': 0, 'live_ret': -0.5, 'bt_ret': -1.5},
    ]
    live_acc, bt_c, diff = _period_returns(rows)
    assert bt_c is not None and live_acc is not None
    assert diff is not None
    # 两日 live 均好于 bt → 复利差应为正
    assert diff > 0

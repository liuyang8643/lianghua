"""战报「回测 vs 实盘」对账（T日操作 + T日持仓）单测。

覆盖：
1. extract_bt_reference 从 seed-replay 结果抽取 回测操作/持仓 参考
2. _compute_comparison 实盘成交累计 vs 回测、实盘T日持仓 = T-1 + 买 - 卖
3. 实盘进度推进时差额收敛、对齐计数
4. 无 bt_ref（首日/缺快照）时退回纯订单进度战报
5. run_seed_replay_for_open 缺 T-1 种子时优雅返回 None
"""
import json
from datetime import date, datetime

from xtquant import xtconstant

from trading.day_board import TradingDayBoard, extract_bt_reference


def _plan_rows():
    return [
        {'code': 'A.SZ', 'name': '甲', 'direction': 'buy', 'est_volume': 500,
         'est_price': 10.0, 'est_amount': 5000.0, 'reason': 'topN换入',
         'plan_seq': 1, 'limit_status': 'ok'},
        {'code': 'B.SZ', 'name': '乙', 'direction': 'buy', 'est_volume': 1000,
         'est_price': 5.0, 'est_amount': 5000.0, 'reason': 'topN换入',
         'plan_seq': 2, 'limit_status': 'ok'},
        {'code': 'C.SH', 'name': '丙', 'direction': 'sell', 'est_volume': 2000,
         'est_price': 8.0, 'est_amount': 16000.0, 'reason': '换出',
         'plan_seq': 3, 'limit_status': 'ok'},
    ]


def _bt_result():
    """回测（继承T-1: A 1000股, C 2000股）：买 A500/B1000, 卖 C2000全清。
    T日EOD持仓: A 1500, B 1000。"""
    t_snap = {
        'executed_buy_details': [
            {'code': 'A.SZ', 'shares': 500, 'price': 10.0},
            {'code': 'B.SZ', 'shares': 1000, 'price': 5.0},
        ],
        'executed_sell_details': [
            {'code': 'C.SH', 'shares': 2000, 'price': 8.0},
        ],
        'positions_eod': [
            {'code': 'A.SZ', 'volume': 1500},
            {'code': 'B.SZ', 'volume': 1000},
        ],
    }
    return {'daily_snapshots': [{'positions_eod': []}, t_snap]}


def _board(bt_ref=None, y_positions=None):
    b = TradingDayBoard()
    b._push_now = lambda: None  # 隔离飞书网络 + 定时器
    b.start_session(trade_date=date(2026, 6, 2), plan_rows=_plan_rows(),
                    buy_n=2, bt_ref=bt_ref, y_positions=y_positions or {})
    return b


def _trade(code, otype, vol, price):
    return {'order_id': 1, 'code': code, 'order_type': otype,
            'price': price, 'volume': vol, 'amount': vol * price,
            'at': datetime.now()}


def test_extract_bt_reference():
    ref = extract_bt_reference(_bt_result())
    assert ref['buy']['A.SZ'] == {'shares': 500, 'amount': 5000.0}
    assert ref['buy']['B.SZ'] == {'shares': 1000, 'amount': 5000.0}
    assert ref['sell']['C.SH'] == {'shares': 2000, 'amount': 16000.0}
    assert ref['positions'] == {'A.SZ': 1500, 'B.SZ': 1000}


def test_extract_bt_reference_empty():
    assert extract_bt_reference({}) == {'buy': {}, 'sell': {}, 'positions': {}}
    assert extract_bt_reference(None) == {'buy': {}, 'sell': {}, 'positions': {}}


def test_compute_comparison_partial_progress():
    """T-1: A1000/C2000。实盘已买 A500、B600(部分)，C 未卖。"""
    ref = extract_bt_reference(_bt_result())
    b = _board(bt_ref=ref, y_positions={'A.SZ': 1000, 'C.SH': 2000})
    b._trades = [
        _trade('A.SZ', xtconstant.STOCK_BUY, 500, 10.0),
        _trade('B.SZ', xtconstant.STOCK_BUY, 600, 5.0),
    ]
    cmp = b._compute_comparison()

    # ── T日操作对比（净买卖额）──
    op = {r['code']: r for r in cmp['op_rows']}
    assert op['A.SZ']['bt_net'] == 5000.0 and op['A.SZ']['live_net'] == 5000.0
    assert op['A.SZ']['diff'] == 0.0
    assert op['B.SZ']['bt_net'] == 5000.0 and op['B.SZ']['live_net'] == 3000.0
    assert op['B.SZ']['diff'] == -2000.0
    assert op['C.SH']['bt_net'] == -16000.0 and op['C.SH']['live_net'] == 0.0
    assert op['C.SH']['diff'] == 16000.0
    assert cmp['op_totals']['bt'] == -6000.0
    assert cmp['op_totals']['live'] == 8000.0

    # ── T日持仓对比（股数）= T-1 + 买 - 卖 ──
    pos = {r['code']: r for r in cmp['pos_rows']}
    assert pos['A.SZ']['bt_shares'] == 1500 and pos['A.SZ']['live_shares'] == 1500
    assert pos['A.SZ']['diff'] == 0
    assert pos['B.SZ']['bt_shares'] == 1000 and pos['B.SZ']['live_shares'] == 600
    assert pos['B.SZ']['diff'] == -400
    assert pos['C.SH']['bt_shares'] == 0 and pos['C.SH']['live_shares'] == 2000
    assert pos['C.SH']['diff'] == 2000
    assert cmp['pos_totals']['aligned'] == 1   # 仅 A 对齐
    assert cmp['pos_totals']['total'] == 3


def test_compute_comparison_full_alignment():
    """实盘完全执行回测计划 → 操作 + 持仓 差额全部归零、对齐数 = 总数。"""
    ref = extract_bt_reference(_bt_result())
    b = _board(bt_ref=ref, y_positions={'A.SZ': 1000, 'C.SH': 2000})
    b._trades = [
        _trade('A.SZ', xtconstant.STOCK_BUY, 500, 10.0),
        _trade('B.SZ', xtconstant.STOCK_BUY, 1000, 5.0),
        _trade('C.SH', xtconstant.STOCK_SELL, 2000, 8.0),
    ]
    cmp = b._compute_comparison()
    assert all(r['diff'] == 0.0 for r in cmp['op_rows'])
    # 持仓：A1500/B1000 对齐，C 已清(0/0 被过滤)
    assert all(r['diff'] == 0 for r in cmp['pos_rows'])
    assert cmp['pos_totals']['aligned'] == cmp['pos_totals']['total']
    # 卡片可正常渲染且含两张对比表
    blob = json.dumps(b._build_card(), ensure_ascii=False)
    assert 'T日操作对比' in blob
    assert 'T日持仓对比' in blob


def test_no_bt_ref_falls_back():
    """无 bt_ref（首日/缺T-1快照）→ 不做对账，卡片仅含订单进度。"""
    b = _board(bt_ref=None)
    assert b._compute_comparison() is None
    blob = json.dumps(b._build_card(), ensure_ascii=False)
    assert 'T日操作对比' not in blob
    assert '目标→成交' in blob  # 退回原始战报


def test_run_seed_replay_for_open_missing_seed(monkeypatch):
    """缺 T-1 种子时 run_seed_replay_for_open 优雅返回 None（首个交易日）。"""
    from trading import post_close
    monkeypatch.setattr(post_close, '_load_seed', lambda trade_date: None)
    out = post_close.run_seed_replay_for_open(
        date(2026, 6, 2), {'weights': {}, 'temperatures': {}, 'buy_n': 2},
        data=None, all_scores=None, date_idx=0, valid_stocks=[], stock_indices={})
    assert out is None

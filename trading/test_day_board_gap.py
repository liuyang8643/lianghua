"""调仓战报「目标→成交」缺口聚合 + 买卖结束定稿 的单测。"""
import json
from datetime import date, datetime

from xtquant import xtconstant

from trading.day_board import TradingDayBoard, _gap_md


def _plan_rows():
    return [
        {'code': 'A.SZ', 'name': '甲', 'direction': 'buy', 'est_volume': 1000,
         'est_price': 10.0, 'est_amount': 10000.0, 'reason': 'topN换入',
         'plan_seq': 1, 'limit_status': 'ok'},
        {'code': 'B.SZ', 'name': '乙', 'direction': 'buy', 'est_volume': 2000,
         'est_price': 5.0, 'est_amount': 10000.0, 'reason': 'topN换入',
         'plan_seq': 2, 'limit_status': 'ok'},
        {'code': 'C.SH', 'name': '丙', 'direction': 'sell', 'est_volume': 1000,
         'est_price': 8.0, 'est_amount': 8000.0, 'reason': '换出',
         'plan_seq': 3, 'limit_status': 'ok'},
    ]


def _board():
    b = TradingDayBoard()
    b._push_now = lambda: None  # 隔离飞书网络 + 定时器
    return b


def _order(oid, code, otype, vol):
    return {'order_id': oid, 'code': code, 'order_type': otype,
            'order_status': xtconstant.ORDER_SUCCEEDED, 'order_volume': vol,
            'traded_volume': vol, 'price': 0.0, 'traded_price': 0.0,
            'status_msg': '', 'updated_at': datetime.now()}


def test_gap_md_marks_shortfall_red_and_full_check():
    assert '缺' in _gap_md(20000, 10000)   # 缺 1w → 标红
    assert '红' not in _gap_md(20000, 10000)  # 用的是 font color,不含"红"字
    assert _gap_md(8000, 8000).endswith('✓')
    assert _gap_md(8000, 8200).endswith('✓')  # 成交略多于计划也算打满


def test_aggregate_plan_vs_done_gap():
    """A 计划买1w全成、B 计划买1w未成、C 计划卖8k全成 → 买缺口1w、卖无缺口。"""
    b = _board()
    b.start_session(trade_date=date(2026, 6, 2), plan_rows=_plan_rows(), buy_n=2)
    b._orders = {1: _order(1, 'A.SZ', xtconstant.STOCK_BUY, 1000),
                 3: _order(3, 'C.SH', xtconstant.STOCK_SELL, 1000)}
    b._trades = [
        {'order_id': 1, 'code': 'A.SZ', 'order_type': xtconstant.STOCK_BUY,
         'price': 10.0, 'volume': 1000, 'amount': 10000.0, 'at': datetime.now()},
        {'order_id': 3, 'code': 'C.SH', 'order_type': xtconstant.STOCK_SELL,
         'price': 8.0, 'volume': 1000, 'amount': 8000.0, 'at': datetime.now()},
    ]
    agg = b._aggregate()
    assert agg['plan_buy_amt'] == 20000.0
    assert agg['plan_sell_amt'] == 8000.0
    assert agg['buy_done_amt'] == 10000.0   # 仅 A 成交
    assert agg['sell_done_amt'] == 8000.0

    blob = json.dumps(b._build_card(), ensure_ascii=False)
    assert '目标→成交' in blob
    assert '缺' in blob  # 买入缺口 1w 应出现在卡片


def test_finalize_locks_board_and_ignores_late_events():
    b = _board()
    b.start_session(trade_date=date.today(), plan_rows=_plan_rows())
    b.finalize()
    assert b._finalized is True
    # 定稿后迟到的回调被忽略,不污染收口快照
    before = dict(b._orders)
    b.record_order(type('O', (), {
        'order_id': 9, 'stock_code': 'Z.SZ', 'order_type': xtconstant.STOCK_BUY,
        'order_status': xtconstant.ORDER_SUCCEEDED, 'order_volume': 100,
        'traded_volume': 100, 'price': 1.0, 'traded_price': 1.0, 'status_msg': ''})())
    assert b._orders == before

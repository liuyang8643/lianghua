"""战报展示：名称、订单过滤、零值格式。"""
import json
from datetime import date

from xtquant import xtconstant

from trading.day_board import TradingDayBoard, _fmt_price_qty, _fmt_shares


def _board(**kwargs):
    b = TradingDayBoard()
    b._push_now = lambda: None
    plan = [
        {'code': 'A.SZ', 'name': '甲', 'direction': 'buy', 'est_volume': 0,
         'est_price': 10.0, 'est_amount': 0.0, 'reason': '已达标(持仓已接近目标仓)',
         'plan_seq': 1, 'limit_status': 'skipped'},
        {'code': 'B.SZ', 'name': '乙', 'direction': 'buy', 'est_volume': 500,
         'est_price': 5.0, 'est_amount': 2500.0, 'reason': 'topN换入',
         'plan_seq': 2, 'limit_status': 'ok'},
        {'code': 'C.SH', 'name': '丙', 'direction': 'sell', 'est_volume': 1000,
         'est_price': 8.0, 'est_amount': 8000.0, 'reason': '换出',
         'plan_seq': 3, 'limit_status': 'ok'},
    ]
    bt_ref = {'buy': {'B.SZ': {'shares': 500, 'amount': 2500.0}},
              'sell': {'C.SH': {'shares': 1000, 'amount': 8000.0}},
              'positions': {'B.SZ': 500}}
    b.start_session(trade_date=date(2026, 6, 4), plan_rows=plan, bt_ref=bt_ref, **kwargs)
    return b


def test_plan_visible_hides_skipped_buy_when_bt_also_idle():
    b = _board()
    assert b._plan_visible('A.SZ', b._plan['A.SZ']) is False
    assert b._plan_visible('B.SZ', b._plan['B.SZ']) is True
    assert b._plan_visible('C.SH', b._plan['C.SH']) is True
    agg = b._aggregate()
    codes = {r['code'] for r in agg['rows']}
    assert 'A.SZ' not in codes
    assert codes == {'B.SZ', 'C.SH'}


def test_zero_formats_not_dash():
    assert _fmt_shares(0) == '0'
    assert _fmt_price_qty(0, 0) == '0'
    assert _fmt_price_qty(10.0, 0) == '10.00 × 0'


def test_position_table_shows_zero_shares_not_dash():
    b = TradingDayBoard()
    b._push_now = lambda: None
    b._trade_date = date(2026, 6, 4)
    b._name_cache['Y.SZ'] = '测试'
    cmp = {
        'op_rows': [], 'op_totals': {'bt': 0, 'live': 0},
        'pos_rows': [{'code': 'Y.SZ', 'bt_shares': 100, 'live_shares': 0, 'diff': -100}],
        'pos_totals': {'bt': 100, 'live': 0, 'aligned': 0, 'total': 1},
    }
    blob = json.dumps(b._build_comparison_elements(cmp), ensure_ascii=False)
    assert '"live":"0"' in blob.replace(' ', '')
    assert '"-"' not in blob.split('live')[1][:20]

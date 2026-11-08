from datetime import date, datetime, timedelta
from types import SimpleNamespace as NS

from xtquant import xtconstant

from trading.persistence import EVT_TRADE, LiveTradeRecorder
from trading import watcher


def test_delayed_trade_keeps_original_day_and_does_not_use_todays_plan(monkeypatch):
    recorder = LiveTradeRecorder()
    recorder._today_plan_prices = {'688092.SH': 999.0}
    events, fills = [], []
    monkeypatch.setattr(recorder, '_append_event', events.append)
    monkeypatch.setattr(recorder, '_append_fill', fills.append)
    original = datetime.combine(date.today() - timedelta(days=1), datetime.min.time())
    recorder.record_event(EVT_TRADE, event_time=original, code='688092.SH', direction='buy',
                          order_id=1, traded_id='prior-day-fill', traded_volume=200,
                          traded_price=20.0, amount=4000.0)
    assert events[0]['date'] == original.date() and events[0]['ts'] == original
    assert fills[0]['date'] == original.date() and fills[0]['fill_time'] == original
    assert fills[0]['est_price'] is None and fills[0]['slippage_pct'] is None


def test_watcher_preserves_bridge_time_and_skips_previous_day_board(monkeypatch):
    from trading.day_board import day_board
    original = datetime.now() - timedelta(days=1)
    recorded, board = [], []
    monkeypatch.setattr(watcher, '_stock_name', lambda code: '')
    monkeypatch.setattr(watcher.live_trade_recorder, 'record_event',
                        lambda *a, **kw: recorded.append(kw))
    monkeypatch.setattr(day_board, 'record_trade', board.append)
    callback = watcher.TraderCallback(NS(strategy_name='FlashMan'))
    callback.on_stock_trade(NS(traded_id='delayed-bridge-trade', order_id=1,
        stock_code='688092.SH', order_type=xtconstant.STOCK_BUY,
        traded_volume=200, traded_price=20.0, traded_amount=4000.0,
        strategy_name='FlashMan', _bridge_event_time=original.timestamp()))
    assert recorded[0]['event_time'] == original
    assert board == []
    # Some broker counters restart each day; yesterday's trade ID must not
    # suppress today's real fill when a bridge consumer spans midnight.
    callback.on_stock_trade(NS(traded_id='delayed-bridge-trade', order_id=1,
        stock_code='688092.SH', order_type=xtconstant.STOCK_BUY,
        traded_volume=200, traded_price=20.0, traded_amount=4000.0,
        strategy_name='FlashMan', _bridge_event_time=datetime.now().timestamp()))
    assert len(recorded) == 2 and len(board) == 1

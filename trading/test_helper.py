from xtquant import xtconstant

from trading.helper import get_price_type


def test_close_trade_uses_after_fix_price_type():
    assert get_price_type(xtconstant.STOCK_BUY, '600000.SH', 10.0) == xtconstant.OPT_AFTER_FIX_BUY
    assert get_price_type(xtconstant.STOCK_SELL, '688296.SH', 22.83) == xtconstant.OPT_AFTER_FIX_SELL


def test_no_price_falls_back_to_market_peer():
    assert get_price_type(xtconstant.STOCK_BUY, '600000.SH', None) == xtconstant.MARKET_PEER_PRICE_FIRST

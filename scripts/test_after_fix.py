"""测试 OPT_AFTER_FIX_BUY/SELL — 最小手数 100 股，不影响实盘。

用法：明天 15:05 后跑一次，看 QMT 到底认不认这个 price_type。
    uv run python scripts/test_after_fix.py
"""
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from configs import TRADE_ACCOUNT
import os, time

qmt_data_dir = os.path.dirname(__import__('xtquant.xtdata', fromlist=['xtdata']).get_data_dir())
client = XtQuantTrader(qmt_data_dir, int(time.time()))
client.start()
assert client.connect() == 0, "连接失败"
acc = StockAccount(TRADE_ACCOUNT, 'STOCK')
assert client.subscribe(acc) == 0, "订阅失败"

# 用一只持仓股做卖测试，用一只自选股做买测试
# 从持仓中找一只可卖的
pos = client.query_stock_positions(acc)
if pos:
    test_code = pos[0].stock_code
    test_shares = min(100, pos[0].can_use_volume)

    print(f"\n=== 测试1: FIX_PRICE + price=收盘价 (对照组) ===")
    oid = client.order_stock(acc, test_code, xtconstant.STOCK_SELL, test_shares,
                             xtconstant.FIX_PRICE, 10.0, 'test', 'test_fix_price')
    print(f"FIX_PRICE + price=10.0  → order_id={oid}")

    print(f"\n=== 测试2: OPT_AFTER_FIX_SELL + price=0 ===")
    oid = client.order_stock(acc, test_code, xtconstant.STOCK_SELL, test_shares,
                             xtconstant.OPT_AFTER_FIX_SELL, 0, 'test', 'test_after_fix_0')
    print(f"OPT_AFTER_FIX_SELL + price=0 → order_id={oid}")

    print(f"\n=== 测试3: OPT_AFTER_FIX_SELL + price=收盘价 ===")
    oid = client.order_stock(acc, test_code, xtconstant.STOCK_SELL, test_shares,
                             xtconstant.OPT_AFTER_FIX_SELL, 10.0, 'test', 'test_after_fix_price')
    print(f"OPT_AFTER_FIX_SELL + price=10.0 → order_id={oid}")

    print(f"\n=== 测试4: order_stock_async + OPT_AFTER_FIX_SELL + price=0 ===")
    seq = client.order_stock_async(acc, test_code, xtconstant.STOCK_SELL, test_shares,
                                   xtconstant.OPT_AFTER_FIX_SELL, 0, 'test', 'test_async')
    print(f"order_stock_async seq={seq}")
else:
    print("无持仓，跳过测试")

client.stop()
print("\n测试完成 — 如果4个测试 OPT_AFTER_FIX 全都 -1，则确认 QMT 版本不支持。")

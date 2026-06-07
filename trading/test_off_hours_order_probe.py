"""闭市/非交易时段：真实 QMT 下单 → 终态耗时探针（集成测试，需本地 QMT）。

用法（仓库根目录）:
  uv run python -m trading.test_off_hours_order_probe

输出：每笔委托从 submit 到进入 TERMINAL_STATUS 的秒数、终态名、status_msg。
"""
from __future__ import annotations

import time
from datetime import datetime

from xtquant import xtconstant

from configs import TRADE_ACCOUNT
from trading.executor import TERMINAL_STATUS
from trading.helper import get_order_status_label
from trading.trader import Trader
from utils.stock.time import is_current_trading

POLL_SEC = 0.05
MAX_WAIT_SEC = 5.0


def _poll_until_terminal(trader: Trader, order_id: int) -> dict:
    t0 = time.perf_counter()
    last_status = None
    last = None
    history: list[tuple[float, str, str]] = []
    while time.perf_counter() - t0 < MAX_WAIT_SEC:
        o = trader.query_order(order_id)
        if o is None:
            history.append((time.perf_counter() - t0, 'None', ''))
            time.sleep(POLL_SEC)
            continue
        last = o
        st = int(o.order_status)
        label = get_order_status_label(st)
        msg = (getattr(o, 'status_msg', '') or '').strip()
        if st != last_status:
            history.append((time.perf_counter() - t0, label, msg))
            last_status = st
        if st in TERMINAL_STATUS:
            elapsed = time.perf_counter() - t0
            return {
                'elapsed_sec': elapsed,
                'terminal': label,
                'status_msg': msg,
                'history': history,
                'order': o,
            }
        time.sleep(POLL_SEC)
    return {
        'elapsed_sec': time.perf_counter() - t0,
        'terminal': '超时未终态',
        'status_msg': (getattr(last, 'status_msg', '') or '') if last else '',
        'history': history,
        'order': last,
    }


def _pick_sell_probe(trader: Trader) -> tuple[str, int] | None:
    positions = trader.query_positions() or []
    for p in sorted(positions, key=lambda x: x.stock_code):
        avail = int(getattr(p, 'can_use_volume', 0) or 0)
        if avail >= 100:
            return p.stock_code, 100
    return None


def _submit_with_timeout(trader: Trader, order_type: int, code: str, vol: int,
                         remark: str, timeout_sec: float = 8.0) -> int | None:
    """order_stock 在闭市可能同步阻塞数分钟,用 thread.join 超时测返回时间。"""
    import threading

    box: dict = {}

    def _do():
        try:
            box['oid'] = trader.order(order_type, code, vol, order_remark=remark)
        except Exception as e:
            box['err'] = e

    th = threading.Thread(target=_do, daemon=True)
    th.start()
    th.join(timeout_sec)
    if th.is_alive():
        return None
    if 'err' in box:
        raise box['err']
    return box.get('oid')


def main():
    wall = datetime.now()
    print(f"真实时钟: {wall:%Y-%m-%d %H:%M:%S}  交易时段={is_current_trading(wall)}", flush=True)

    td = Trader(TRADE_ACCOUNT)
    sell_probe = _pick_sell_probe(td)
    if not sell_probe:
        print("SKIP 卖出探针: 无可用持仓 >= 100 股", flush=True)
        return

    code, vol = sell_probe
    remark = f'off_hours_probe sell {wall.isoformat()}'
    print(f"\n=== 卖出探针 {code} {vol}股 ===", flush=True)
    t_submit = time.perf_counter()
    oid = _submit_with_timeout(td, xtconstant.STOCK_SELL, code, vol, remark)
    if oid is None:
        print(f"submit 在 8s 内未返回 (order_stock 同步阻塞)", flush=True)
        print("结论: 不是轮询等 600s, 而是 QMT 下单 API 本身卡住。", flush=True)
        return
    print(f"submit 返回 order_id={oid}  (+{(time.perf_counter()-t_submit)*1000:.0f}ms)", flush=True)
    r = _poll_until_terminal(td, oid)
    print(f"→ 终态: {r['terminal']}  耗时 {r['elapsed_sec']:.3f}s  msg={r['status_msg']!r}")
    for dt, label, msg in r['history']:
        print(f"   {dt:6.3f}s  {label}  {msg}")

    # 买入探针：同一只、最小 100 股（闭市应废单或拒）
    print(f"\n=== 买入探针 {code} {vol}股 ===", flush=True)
    t_submit = time.perf_counter()
    try:
        oid2 = _submit_with_timeout(td, xtconstant.STOCK_BUY, code, vol, remark + ' buy')
        if oid2 is None:
            print("买入 submit 8s 超时未返回", flush=True)
            return
        print(f"submit 返回 order_id={oid2}  (+{(time.perf_counter()-t_submit)*1000:.0f}ms)", flush=True)
        r2 = _poll_until_terminal(td, oid2)
        print(f"→ 终态: {r2['terminal']}  耗时 {r2['elapsed_sec']:.3f}s  msg={r2['status_msg']!r}")
        for dt, label, msg in r2['history']:
            print(f"   {dt:6.3f}s  {label}  {msg}")
    except Exception as e:
        print(f"买入 submit 异常(可能柜台直接拒): {e}")

    print("\n结论: 若 submit 8s 超时 → 瓶颈在 order_stock 同步 API, 不是轮询 600s。", flush=True)
    print("闭市 --skip 已改为跳过真实委托, 盘中再测秒废单。", flush=True)


if __name__ == '__main__':
    main()

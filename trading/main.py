from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
from xtquant import xtconstant, xtdata

from configs import TRADE_ACCOUNT
from data.db import allow_buy_stock_code_list
from core.runtime import load_runtime_npz
from core.scoring import scores_to_ranks, batch_limit_check, precompute_limit_helpers
from data.db.stock_name import get_stock_name_at_date
from core.ga import get_profile_factor_classes, resolve_profile_name

def _get_name(code, signal_date):
    name = get_stock_name_at_date(code, signal_date)
    if name:
        return name
    # 兜底：CNINFO parquet 缺 bare_code 列时直接从 xtdata 查
    try:
        from xtquant import xtdata
        detail = xtdata.get_instrument_detail(code)
        if detail:
            name = detail.get('InstrumentName', '').strip()
    except Exception:
        pass
    return name or ''
from trading.logger import trading_logger
from utils.recorder import recorder

from .lark.receiver import create_lark_handler
from .lark.sender import lark_sender, LarkMsgLevel
from .manual_confirm import build_manual_confirmation_text, is_manual_confirmation_approved
from .persistence import live_trade_recorder
from .post_close import run_post_close, run_update_all
from .scheduler import TradingScheduler
from .trader import Trader
from .helper import get_order_status_label


def _submit_sell_order(store, code, shares, signal_date, trade_date):
    """提交卖出委托。shares=-1 表示全部清仓。返回 {code, order_type, order_id, shares} 或 None。"""
    try:
        if shares < 0:
            order_id = store.trader.clear_position(code, reason=f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}')
        else:
            order_id = store.trader.order(
                xtconstant.STOCK_SELL, code, shares, None,
                order_remark=f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
            )
        if order_id is None:
            trading_logger.info(f"{code} 无需卖出或委托未发出")
            return None
        trading_logger.info(f"已提交卖出委托: {code} {'全仓' if shares < 0 else f'{shares}股'} order_id={order_id}")
        recorder.mark("提交卖出委托")
        return {'code': code, 'order_type': 'SELL', 'order_id': order_id, 'shares': shares}
    except ValueError as e:
        trading_logger.info(f"{code} 卖出前校验拦截: {e}")
    except Exception as e:
        trading_logger.exception(f"{code} 卖出委托失败: {e}")
    return None


def _submit_buy_order(store, code, shares, signal_date, trade_date):
    try:
        order_id = store.trader.order(
            xtconstant.STOCK_BUY, code, shares, None,
            order_remark=f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
        )
        trading_logger.info(f"已提交买入委托: {code} * {shares} 股 order_id={order_id}")
        recorder.mark("提交买入委托")
        return {'code': code, 'order_type': 'BUY', 'order_id': order_id, 'shares': shares}
    except ValueError as e:
        trading_logger.info(f"{code} 买入前校验拦截: {e}")
    except Exception as e:
        trading_logger.exception(f"{code} 买入委托失败: {e}")
    return None


def _print_and_confirm_manual_plan(pending: dict, execute_buy: bool, execute_sell: bool) -> bool:
    message = build_manual_confirmation_text(pending, execute_buy=execute_buy, execute_sell=execute_sell)
    print(message)
    try:
        user_input = input("确认执行> ")
    except EOFError:
        trading_logger.warning("无法读取交互输入（非交互环境），已取消确认")
        return False
    return is_manual_confirmation_approved(user_input)


def _compute_live_scores(signal_datetime, all_stocks, weights, factor_classes):
    """实盘因子计算，与回测 _compute_factor_scores 逻辑一致。"""
    max_lookback = max((getattr(c, 'hist_days', 0) for c in factor_classes), default=0) or None
    data = load_runtime_npz([signal_datetime], max_lookback=max_lookback)
    if data is None:
        raise FileNotFoundError(f"未找到覆盖 {signal_datetime} 的 runtime npz 文件")

    npz_stocks = [str(s) for s in data['stock_codes']]
    stock_indices = {c: i for i, c in enumerate(npz_stocks)}
    valid_stocks = [s for s in all_stocks if s in stock_indices]

    npz_dates = data['trade_dates']
    date_to_idx = {}
    for i, d in enumerate(npz_dates):
        date_to_idx[d.astype('datetime64[D]').item()] = i
    sd = signal_datetime.date() if hasattr(signal_datetime, 'date') else signal_datetime
    date_idx = date_to_idx.get(sd)
    if date_idx is None:
        raise ValueError(f"信号日期 {sd} 不在 runtime npz 日期范围内")

    import time
    t0 = time.time()

    py_dates = [d.astype('datetime64[D]').item() for d in npz_dates]
    factor_data = {**data, 'stock_codes': npz_stocks, 'trade_dates': py_dates}
    all_scores = {}
    for f_cls in factor_classes:
        f = f_cls()
        name = f.__class__.__name__
        if weights is not None and weights.get(name, 0.0) == 0:
            continue
        raw = f.calc_batch(factor_data)
        all_scores[name] = scores_to_ranks(raw.astype(np.float32, copy=False))

    trading_logger.info(f"实时因子计算完成 ({time.time() - t0:.1f}s), {len(all_scores)} 个因子")
    return data, all_scores, date_idx, valid_stocks, stock_indices


class MutableTime:
    """可变的模拟时钟。暴露 __call__ 用作 time_provider，set() 用于快进。"""
    def __init__(self, dt: datetime):
        self._dt = dt
    def __call__(self) -> datetime:
        return self._dt
    def set(self, dt: datetime):
        self._dt = dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual-config', type=str, required=True, help='Individual_config JSON文件路径')
    parser.add_argument('--skip', type=str, help='跳过数据更新，按指定系统时间(YYYYMMDDHHMM)模拟实盘流程(快进模式)')
    parser.add_argument('--buy', action='store_true', help='忽略时间窗口，立即执行一次买入（需手工yes确认）')
    parser.add_argument('--sell', action='store_true', help='忽略时间窗口，立即执行一次卖出（需手工yes确认）')
    parser.add_argument('--update-all', action='store_true', help='执行全量数据更新（删除昨日不完整数据→全量下载→构建runtime NPZ）后退出')
    args = parser.parse_args()

    if args.update_all:
        run_update_all(None)
        sys.exit(0)

    skip_update = args.skip is not None
    is_manual = args.buy or args.sell
    if skip_update:
        skip_dt = datetime.strptime(args.skip, '%Y%m%d%H%M')
        trading_logger.info(f"跳过数据更新, 模拟时间: {skip_dt}")
    else:
        skip_dt = None

    with open(args.individual_config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    profile_name = resolve_profile_name(config_data)
    factor_classes = get_profile_factor_classes(profile_name)
    individual_config = config_data['individual_config']
    weights = individual_config['weights']
    temperatures = individual_config['temperatures']
    buy_n = individual_config['buy_n']
    sell_m = individual_config.get('sell_m', buy_n)

    trading_logger.info(f"加载Individual_config: {args.individual_config}")
    trading_logger.info(f"配置参数: buy_n={buy_n}, sell_m={sell_m}, factors={sorted(weights.keys())}")

    td = Trader(TRADE_ACCOUNT)

    threading.Thread(target=create_lark_handler, args=[td], daemon=True).start()

    def before_trade(store):
        trade_now = datetime.now()
        trade_date = trade_now.date()
        signal_date = trade_date
        signal_datetime = datetime.combine(signal_date, datetime.min.time())

        trading_logger.info(
            f"开始预计算调仓: signal_date={signal_date.isoformat()}, trade_date={trade_date.isoformat()}"
        )
        recorder.mark("开始选股")

        # 快速数据更新（--skip 模式下跳过）
        if not skip_update:
            from data.update_live import update_live_quick
            try:
                update_live_quick()
            except Exception as e:
                trading_logger.warning(f"快速数据更新失败, 继续使用已有数据: {e}")

        if store.whole_sub_id is None:
            store.whole_sub_id = xtdata.subscribe_whole_quote(['SH', 'SZ'])

        asset = store.trader.query_asset()
        if asset is None:
            trading_logger.warning("资产查询失败，跳过本轮调仓")
            store.pending_rebalance = None
            return

        all_stocks = allow_buy_stock_code_list(target_date=trade_date)
        trading_logger.info(f"候选股票池: {len(all_stocks)} 只")
        # 因子计算（与回测一致）
        try:
            data, all_scores, score_date_idx, valid_stocks, stock_indices = _compute_live_scores(
                signal_datetime, all_stocks, weights, factor_classes)
        except Exception as e:
            if is_manual:
                from datetime import timedelta
                from utils.stock.time import get_last_trading_day
                fallback = get_last_trading_day(trade_date)
                if fallback >= trade_date:
                    fallback = get_last_trading_day(trade_date - timedelta(days=1))
                trading_logger.warning(
                    f"今日({trade_date.isoformat()})无NPZ数据，回退至前一交易日({fallback.isoformat()})计算因子"
                )
                signal_date = fallback
                trading_logger.info(f"更新: signal_date={signal_date.isoformat()}, trade_date={trade_date.isoformat()}")
                signal_datetime = datetime.combine(signal_date, datetime.min.time())
                data, all_scores, score_date_idx, valid_stocks, stock_indices = _compute_live_scores(
                    signal_datetime, all_stocks, weights, factor_classes)
            else:
                trading_logger.error(f"因子计算失败: {e}")
                store.pending_rebalance = None
                return

        valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)

        # 加权选股（与回测一致）
        final_score = np.zeros(len(valid_stocks))
        for name, ranks_mat in all_scores.items():
            w = weights.get(name, 0.0)
            if w == 0:
                continue
            ranks = ranks_mat[score_date_idx][valid_cols]
            temp = temperatures.get(name, 1.0)
            if temp != 1.0:
                np.power(ranks, 1.0 / temp, out=ranks)
            final_score += ranks * w

        top_idx = np.argsort(-final_score)
        buy_n_stocks = [valid_stocks[i] for i in top_idx[:buy_n]]
        trading_logger.info(f"选股完成 Top{buy_n}: {buy_n_stocks[:5]}...")

        # 涨跌停检查预计算
        board_type, base_ratio, list_tidx = precompute_limit_helpers(data, stock_indices)
        open_all = data['open']
        close_all = data['close']
        high_all = data['high']
        low_all = data['low']
        st_all = data.get('st_mask')
        issue_price_all = data.get('issue_price')

        trade_idx = score_date_idx + 1 if score_date_idx + 1 < len(data['trade_dates']) else score_date_idx
        trade_dt64 = data['trade_dates'][trade_idx]
        exec_date = trade_dt64.astype('datetime64[D]').item() if hasattr(trade_dt64, 'astype') else trade_date

        # 获取当日价格
        day_open = open_all[trade_idx]
        # 获取持仓（手动触发时 QMT 可能尚未同步，等最多 5s）
        import time
        positions = {}
        for attempt in range(6):
            raw = store.trader.query_positions()
            if raw:
                positions = {p.stock_code: p for p in raw if p.volume > 0}
                break
            if attempt == 0:
                trading_logger.warning("QMT 持仓为空，等待同步...")
            time.sleep(1)
        can_sell = sum(1 for p in positions.values() if p.can_use_volume > 0)
        trading_logger.info(f"持仓: {len(positions)} 只 (可卖{can_sell}), 总市值={sum(p.market_value for p in positions.values()):.0f}")
        _last_valid_price = {}

        prices = {}
        for code in set(positions.keys()) | set(buy_n_stocks):
            si = stock_indices.get(code)
            if si is None:
                continue
            open_val = day_open[si]
            if np.isnan(open_val) or open_val <= 0:
                if code in _last_valid_price:
                    prices[code] = _last_valid_price[code]
                continue
            prices[code] = _last_valid_price[code] = float(open_val)

        # 多退少补 rebalance（与回测一致）
        # 注意：用 volume(总持仓) 算市值，避免 T+0 买入后 can_use_volume=0 导致重复买入
        pos_vals = {c: p.volume * prices.get(c, 0) for c, p in positions.items()}
        total_eq = asset.total_asset
        timing_mult = 1.0
        base_target = total_eq * timing_mult / buy_n
        buy_n_set = set(buy_n_stocks)

        sell_orders = []   # [(code, shares)]  shares=-1 表示全清
        buy_orders = {}    # {code: shares}

        all_codes = set(positions.keys()) | buy_n_set
        for code in all_codes:
            if code not in prices:
                continue
            cv = pos_vals.get(code, 0)
            tgt = base_target if code in buy_n_set else 0.0

            # 多退
            if code in positions and cv > tgt * 1.01:
                if tgt == 0:
                    sell_orders.append((code, -1))  # 不在 buy_n → 全清
                else:
                    sv = int((cv - tgt) / prices[code] / 100) * 100
                    pos = positions[code]
                    if 0 < sv < pos.can_use_volume:
                        sell_orders.append((code, sv))

            # 少补 — 仅 buy_n 内的
            if code in buy_n_set and cv < base_target * 0.99:
                bv = int((base_target - cv) / prices[code] / 100) * 100
                if bv > 0:
                    buy_orders[code] = bv

        # 涨跌停过滤卖出
        if sell_orders:
            sc = [c for c, _ in sell_orders]
            si_list = [stock_indices[c] for c in sc if c in stock_indices]
            ok, _ = batch_limit_check(
                sc, si_list, trade_idx, exec_date,
                board_type, base_ratio, list_tidx,
                open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy=False)
            sell_orders = [(c, s) for j, (c, s) in enumerate(sell_orders) if ok[j]]

        # 涨跌停过滤买入
        if buy_orders:
            bc = list(buy_orders.keys())
            bi_list = [stock_indices[c] for c in bc if c in stock_indices]
            ok, _ = batch_limit_check(
                bc, bi_list, trade_idx, exec_date,
                board_type, base_ratio, list_tidx,
                open_all, close_all, high_all, low_all, st_all, issue_price_all, is_buy=True)
            buy_orders = {c: s for j, (c, s) in enumerate(buy_orders.items()) if ok[j]}

        # 准备 pending_rebalance
        sell_details = []
        for code, shares in sell_orders:
            pos = positions.get(code)
            vol = shares if shares > 0 else (pos.can_use_volume if pos else 0)
            reason = '换出' if shares < 0 else f'多退({shares}股)'
            sell_details.append({
                'code': code,
                'name': _get_name(code, signal_date),
                'board': '',
                'volume': vol,
                'est_price': prices.get(code, 0),
                'est_amount': vol * prices.get(code, 0),
                'reason': reason,
            })

        buy_details = []
        for code, shares in buy_orders.items():
            buy_details.append({
                'code': code,
                'name': _get_name(code, signal_date),
                'board': '',
                'shares': shares,
                'est_price': prices.get(code, 0),
                'est_amount': shares * prices.get(code, 0),
            })

        # 打印多退少补详情
        trading_logger.info(
            f"====== 多退少补计划: target={base_target:.0f}, 权益={total_eq:.0f}, 持仓={len(positions)}只 ======"
        )
        if sell_details:
            for d in sell_details:
                trading_logger.info(
                    f"  [卖] {d['code']} {d.get('name','')} {d['reason']}: "
                    f"{d['volume']}股 @{d['est_price']:.2f} ≈{d['est_amount']:.0f}"
                )
        else:
            trading_logger.info("  [卖] 无")
        if buy_details:
            for d in buy_details:
                trading_logger.info(
                    f"  [买] {d['code']} {d.get('name','')} 少补: "
                    f"{d['shares']}股 @{d['est_price']:.2f} ≈{d['est_amount']:.0f}"
                )
        else:
            trading_logger.info("  [买] 无")
        trading_logger.info("=" * 60)

        store.pending_rebalance = {
            'signal_date': signal_date,
            'trade_date': trade_date,
            'sell_orders': sell_orders,
            'buy_allocations': buy_orders,
            'sell_details': sell_details,
            'buy_details': buy_details,
            'prices': prices,
            'stock_indices': stock_indices,
        }

        trading_logger.info(
            f"调仓预计算完成: sell={len(sell_orders)}, buy={len(buy_orders)}, "
            f"target={base_target:.0f}, equity={total_eq:.0f}"
        )
        recorder.mark("完成调仓预计算")

        # 飞书通知：开盘准备完成
        try:
            lines = [f"开盘准备完成 @ {trade_date.isoformat()}"]
            lines.append(f"持仓 {len(positions)} 只 | 权益 ¥{total_eq:,.0f}")
            lines.append(f"计划卖出 {len(sell_orders)} 只 | 买入 {len(buy_orders)} 只")
            if buy_details:
                lines.append(f"Top{buy_n}: {', '.join(d['code'] for d in buy_details[:5])}{'...' if len(buy_details) > 5 else ''}")
            lark_sender.send_msg('\n'.join(lines))
        except Exception as e:
            trading_logger.warning(f"飞书通知失败: {e}")

    def execute_trade(store, execute_sell=True, execute_buy=True):
        import time
        pending = getattr(store, 'pending_rebalance', None)
        if not pending:
            trading_logger.warning("没有可执行的调仓计划，跳过本轮下单")
            return

        signal_date = pending['signal_date']
        trade_date = pending['trade_date']
        sell_orders = pending.get('sell_orders', [])
        buy_allocations = pending.get('buy_allocations', {})
        prices = pending.get('prices', {})

        trading_logger.info(
            f"开始调仓: sell={len(sell_orders)} buy={len(buy_allocations)} "
            f"signal={signal_date.isoformat()} trade={trade_date.isoformat()}"
        )

        submitted = []  # [{code, order_type, order_id, shares}]

        if execute_sell and sell_orders:
            with ThreadPoolExecutor(max_workers=min(16, len(sell_orders))) as executor:
                futures = [
                    executor.submit(_submit_sell_order, store, code, shares, signal_date, trade_date)
                    for code, shares in sell_orders
                ]
                for future in as_completed(futures):
                    r = future.result()
                    if r:
                        submitted.append(r)

        if execute_buy and buy_allocations:
            buy_items = list(buy_allocations.items())
            for i, (code, shares) in enumerate(buy_items):
                if i > 0:
                    time.sleep(0.3)
                # 资金检查：不足则降量
                asset = store.trader.query_asset()
                if asset and asset.cash > 0:
                    est_price = prices.get(code, 0)
                    if est_price > 0:
                        max_shares = int(asset.cash * 0.99 / est_price / 100) * 100
                        if max_shares < shares:
                            if max_shares >= 100:
                                trading_logger.warning(f"{code} 资金不足 {shares}→{max_shares}股")
                                shares = max_shares
                            else:
                                trading_logger.warning(f"{code} 资金不足,跳过")
                                continue
                r = _submit_buy_order(store, code, shares, signal_date, trade_date)
                if r:
                    submitted.append(r)

        store.pending_rebalance = None

        # 轮询等待所有委托进入终态
        TERMINAL = {xtconstant.ORDER_SUCCEEDED, xtconstant.ORDER_CANCELED,
                    xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL}
        waited = 0
        while waited < 30:
            pending_orders = []
            for s in submitted:
                o = store.trader.query_order(s['order_id'])
                if o and o.order_status not in TERMINAL:
                    pending_orders.append(s['code'])
            if not pending_orders:
                break
            time.sleep(1)
            waited += 1
        if waited >= 30:
            trading_logger.warning(f"等待成交超时, 仍有{len(pending_orders)}笔未终态: {pending_orders[:5]}...")

        # 汇总
        ok_list = []
        fail_list = []
        partial_list = []
        for s in submitted:
            o = store.trader.query_order(s['order_id'])
            status = get_order_status_label(o.order_status) if o else '查询失败'
            msg = o.status_msg if o else ''
            traded = o.traded_volume if o else 0
            price = o.traded_price if o else 0
            vol = o.order_volume if o else 0
            line = f"{s['order_type']:4s} {s['code']} {s['shares']}股 → {status}"
            if traded and traded != vol:
                line += f" {traded}/{vol}股"
            if price:
                line += f" @{price:.2f}"
            if status in ('已成',):
                ok_list.append(line)
            elif status in ('废单', '已撤', '部撤'):
                fail_list.append(f"{line} {msg}")
            else:
                partial_list.append(f"{line} {msg}")

        if ok_list:
            trading_logger.info(f"=== 成交 {len(ok_list)} 笔 ===")
            for l in ok_list:
                trading_logger.info(f"  {l}")
        if partial_list:
            trading_logger.warning(f"=== 未完成 {len(partial_list)} 笔 ===")
            for l in partial_list:
                trading_logger.warning(f"  {l}")
        if fail_list:
            trading_logger.error(f"=== 失败 {len(fail_list)} 笔 ===")
            for l in fail_list:
                trading_logger.error(f"  {l}")

    def after_trade(store):
        if store.whole_sub_id is not None:
            xtdata.unsubscribe_quote(store.whole_sub_id)
            store.whole_sub_id = None
        store.pending_rebalance = None

    def post_close(store):
        try:
            run_post_close(store)
        except Exception as e:
            trading_logger.exception(f"盘后对比失败: {e}")

    def update_all(store):
        try:
            run_update_all(store)
        except Exception as e:
            trading_logger.exception(f"全量更新失败: {e}")

    sim_clock = MutableTime(skip_dt) if skip_update else None
    scheduler = TradingScheduler(
        td,
        before_trade=before_trade,
        execute_trade=execute_trade,
        while_trade=[],
        after_trade=after_trade,
        post_close=post_close,
        update_all=update_all,
        time_provider=sim_clock,
        fast_forward=skip_update,
    )
    scheduler._individual_config = individual_config
    scheduler._factor_classes = factor_classes

    if is_manual:
        execute_buy = args.buy
        execute_sell = args.sell
        if args.buy and args.sell:
            execute_buy = True
            execute_sell = True

        store = SimpleNamespace(trader=td, whole_sub_id=None, pending_rebalance=None)
        try:
            before_trade(store)
            pending = getattr(store, 'pending_rebalance', None)
            if not pending:
                trading_logger.warning("没有生成可执行计划，已取消本次手动触发")
                sys.exit(0)

            confirmed = _print_and_confirm_manual_plan(
                pending,
                execute_buy=execute_buy,
                execute_sell=execute_sell,
            )
            if not confirmed:
                trading_logger.warning("手动确认未通过，已取消本次交易")
                sys.exit(0)

            execute_trade(store, execute_sell=execute_sell, execute_buy=execute_buy)
            trading_logger.info("手动单次触发执行完成")
        finally:
            after_trade(store)
        sys.exit(0)

    scheduler.start_check_trading()

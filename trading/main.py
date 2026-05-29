from datetime import datetime
import argparse
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
from xtquant import xtdata

from configs import TRADE_ACCOUNT
from data.db import allow_buy_stock_code_list
from core.runtime import load_runtime_npz
from core.scoring import scores_to_ranks, batch_limit_check, build_legality_context, select_topn
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


def _build_plan_rows(*, trade_date, buy_n_stocks, buy_details, sell_details,
                     prices, final_score, valid_stocks, signal_date):
    """组装 plan_{T}.parquet 行 — 全量 topN + 实际下单意图。
    
    语义：
      - `est_volume > 0`：本次实际下单股数（合法性+少补+资金都通过）
      - `est_volume == 0`：未下单。`reason` 说明原因（合法性 / 已达标 / 资金不足）
      - `limit_status`：保留字段兼容 plan parquet schema；'ok' = 实际下单
    """
    score_by_code = {code: float(final_score[i]) for i, code in enumerate(valid_stocks)}
    buy_actual = {d['code']: d for d in buy_details}

    rows = []
    for seq, code in enumerate(buy_n_stocks, start=1):
        d = buy_actual.get(code)
        if d:
            status = 'ok'; reason = 'topN换入'
            est_vol = int(d['shares']); est_amt = float(d['est_amount'])
        else:
            status = 'skipped'; reason = 'topN未下单(已达标/资金不足/合法性)'
            est_vol = 0; est_amt = 0.0
        rows.append({
            'date': trade_date, 'code': code,
            'name': _get_name(code, signal_date),
            'direction': 'buy', 'est_price': float(prices.get(code, 0.0)),
            'est_volume': est_vol, 'est_amount': est_amt,
            'factor_score': score_by_code.get(code),
            'limit_status': status, 'reason': reason, 'plan_seq': seq,
        })
    # 2. 卖出计划
    for seq, d in enumerate(sell_details, start=1):
        rows.append({
            'date': trade_date, 'code': d['code'], 'name': d.get('name', ''),
            'direction': 'sell', 'est_price': float(d['est_price']),
            'est_volume': int(d['volume']), 'est_amount': float(d['est_amount']),
            'factor_score': None,
            'limit_status': 'ok', 'reason': d['reason'],
            'plan_seq': seq + len(buy_n_stocks),
        })
    return rows
from trading.logger import trading_logger
from utils.recorder import recorder

from .lark.receiver import create_lark_handler
from .lark.sender import lark_sender, LarkMsgLevel
from .manual_confirm import build_manual_confirmation_text, is_manual_confirmation_approved
from .persistence import live_trade_recorder
from .post_close import run_post_close, run_update_all
from .scheduler import TradingScheduler
from .trader import Trader
from .executor import RebalanceExecutor


def _print_and_confirm_manual_plan(pending: dict) -> bool:
    message = build_manual_confirmation_text(pending)
    print(message)
    try:
        user_input = input("确认执行> ")
    except EOFError:
        trading_logger.warning("无法读取交互输入（非交互环境），已取消确认")
        return False
    return is_manual_confirmation_approved(user_input)


def _compute_live_scores(signal_datetime, all_stocks, weights, factor_classes):
    """实盘因子计算，与回测 _compute_factor_scores 逻辑一致。"""
    max_lookback = max((c.hist_days for c in factor_classes), default=0) or None
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
    sd = signal_datetime.date()
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
    parser.add_argument('--confirm', action='store_true',
                        help='单次手动模式：忽略时间窗口，立即跑一遍完整选股+买卖，且买卖前需手工 yes 确认。不加此参数则进入 scheduler 自动模式')
    parser.add_argument('--update-all', action='store_true', help='执行全量数据更新（删除昨日不完整数据→全量下载→构建runtime NPZ）后退出')
    parser.add_argument('--dry-run', action='store_true', help='只计算选股计划，跳过实际买卖下单')
    args = parser.parse_args()

    if args.update_all:
        run_update_all(None)
        sys.exit(0)

    skip_update = args.skip is not None
    dry_run = args.dry_run
    is_manual = args.confirm
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
    rebalance_executor = RebalanceExecutor(td)

    threading.Thread(target=create_lark_handler, args=[td], daemon=True).start()

    def before_trade(store):
        trade_now = store._now()
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
                # 注意：trade_idx = score_date_idx，回退后 day_open 也变成 open[T-1]，
                # 因此 plan 里的 est_price/est_amount/合法性检查全部基于 T-1，
                # 与 T 日 09:30 真实开盘价必然有 diff——仅供手动 dry-run 排演用。
                trading_logger.warning(
                    f"NPZ 缺 {trade_date.isoformat()} 行(--skip 跳过了 update_live_quick)，"
                    f"signal/价格/合法性全部回退到 {fallback.isoformat()}；"
                    f"plan 中的 est_price 是 T-1 开盘价，与今日真实开盘价会 diff。"
                )
                signal_date = fallback
                signal_datetime = datetime.combine(signal_date, datetime.min.time())
                data, all_scores, score_date_idx, valid_stocks, stock_indices = _compute_live_scores(
                    signal_datetime, all_stocks, weights, factor_classes)
            else:
                trading_logger.error(f"因子计算失败: {e}")
                store.pending_rebalance = None
                return

        valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)

        # 加权选股（与回测共用 select_topn）
        buy_n_stocks, final_score = select_topn(
            all_scores, score_date_idx, valid_stocks, valid_cols,
            weights, temperatures, buy_n,
        )
        trading_logger.info(f"选股完成 Top{buy_n}: {buy_n_stocks[:5]}...")

        # 涨跌停检查上下文（与回测共用 build_legality_context）
        legal_ctx = build_legality_context(data, stock_indices)
        open_all = legal_ctx['open_all']
        close_all = legal_ctx['close_all']
        st_all = legal_ctx['st_all']
        issue_price_all = legal_ctx['issue_price_all']
        board_type = legal_ctx['board_type']
        base_ratio = legal_ctx['base_ratio']
        list_tidx = legal_ctx['list_tidx']

        # T 日开盘契约：signal_date == trade_date == score_date_idx → trade_idx=date_idx，与回测对齐。
        # 例外：is_manual + NPZ 缺 T 日行时，signal_date 已被回退到 T-1，
        # 此时 trade_idx 也指向 T-1 行，day_open=open[T-1]（仅用于排演，非生产路径）。
        trade_idx = score_date_idx
        trade_dt64 = data['trade_dates'][trade_idx]
        exec_date = trade_dt64.astype('datetime64[D]').item()

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

        # 多退少补 rebalance（与回测严格对齐）
        # 1. 用 volume(总持仓) 算市值，避免 T+0 买入后 can_use_volume=0 导致重复买入
        # 2. total_eq 用 open[T] 重算（不用 QMT.total_asset），跟回测口径一致；
        #    否则 QMT 的 last_price/close[T-1] 与 open[T] 之间跳空会让 base_target 偏移，
        #    导致回测/实盘 base_target 不一致 → 阈值穿越触发不一致的少补。
        pos_vals = {c: p.volume * prices.get(c, 0) for c, p in positions.items()}
        total_eq = float(asset.cash) + sum(pos_vals.values())
        timing_mult = 1.0
        base_target = total_eq * timing_mult / buy_n
        trading_logger.info(
            f"[多退少补] total_eq={total_eq:.0f} (cash={asset.cash:.0f} + "
            f"open市值={sum(pos_vals.values()):.0f}), base_target={base_target:.0f} "
            f"(QMT.total_asset={asset.total_asset:.0f}, 差={total_eq-asset.total_asset:+.0f})"
        )
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
                # 圆到板块最小买入手数（科创/创业市价单 200 起）
                # 否则 QMT 会以「最小买入数量为200」拒单
                from utils.stock.info import min_buy_shares as _min_buy
                min_lot = _min_buy(code)
                if 0 < bv < min_lot:
                    bv = min_lot
                if bv > 0:
                    buy_orders[code] = bv

        # 涨跌停过滤卖出
        if sell_orders:
            sc = [c for c, _ in sell_orders]
            si_list = [stock_indices[c] for c in sc if c in stock_indices]
            ok, _ = batch_limit_check(
                sc, si_list, trade_idx, exec_date,
                board_type, base_ratio, list_tidx,
                open_all, close_all, st_all, issue_price_all, is_buy=False)
            sell_orders = [(c, s) for j, (c, s) in enumerate(sell_orders) if ok[j]]

        # 涨跌停过滤买入
        if buy_orders:
            bc = list(buy_orders.keys())
            bi_list = [stock_indices[c] for c in bc if c in stock_indices]
            ok, _ = batch_limit_check(
                bc, bi_list, trade_idx, exec_date,
                board_type, base_ratio, list_tidx,
                open_all, close_all, st_all, issue_price_all, is_buy=True)
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
            'buy_n_stocks': buy_n_stocks,  # topN 顺序，买入按此串行优先级
            'sell_details': sell_details,
            'buy_details': buy_details,
            'prices': prices,
            'stock_indices': stock_indices,
        }

        # AL-5: 落地盘前调仓计划（候选股 + 订单 + 合法性状态）
        # 关键约束：plan 是「真实下单意图」记录，不允许被后续 dry-run / 二次跑覆盖
        # - dry-run 不下单 → 不写 plan
        # - 当日 plan 已存在 → 不覆盖（保护 9:30 真实记录不被 13:30/15:00 sim 覆盖）
        plan_rows = _build_plan_rows(
            trade_date=trade_date,
            buy_n_stocks=buy_n_stocks, buy_details=buy_details,
            sell_details=sell_details, prices=prices, final_score=final_score,
            valid_stocks=valid_stocks, signal_date=signal_date,
        )
        plan_path = live_trade_recorder.plan_path(trade_date)
        if dry_run:
            trading_logger.info(f"[dry-run] 不写入 plan_{trade_date.isoformat()}.parquet")
        elif plan_path.exists():
            trading_logger.warning(
                f"[plan 保护] {plan_path.name} 已存在（{plan_path.stat().st_mtime}），"
                f"跳过覆盖（避免 9:30 真实下单意图被后续 sim 重写）"
            )
        else:
            live_trade_recorder.record_plan(plan_rows, trade_date=trade_date)

        trading_logger.info(
            f"调仓预计算完成: sell={len(sell_orders)}, buy={len(buy_orders)}, "
            f"target={base_target:.0f}, equity={total_eq:.0f}"
        )
        recorder.mark("完成调仓预计算")

        # 飞书：开启「调仓战报」聚合卡片
        # 取代之前的「开盘准备完成」独立卡片 + 零散订单/成交回调卡片，
        # 全天只发一张可更新卡片，订单/成交回调实时刷新（debounce 300ms）。
        try:
            from .day_board import day_board
            day_board.start_session(
                trade_date=trade_date,
                plan_rows=plan_rows,
                equity=total_eq,
                position_count=len(positions),
                base_target=base_target,
                buy_n=buy_n,
            )
        except Exception as e:
            trading_logger.warning(f"飞书战报初始化失败: {e}")

    def execute_trade(store, execute_sell=True, execute_buy=True):
        pending = store.pending_rebalance
        if not pending:
            trading_logger.warning("没有可执行的调仓计划，跳过本轮下单")
            return

        if dry_run:
            trading_logger.info("=== DRY RUN: 跳过实际买卖下单 ===")
            store._dry_run_plan = pending
            store.pending_rebalance = None
            return

        rebalance_executor.execute(pending, execute_sell=execute_sell, execute_buy=execute_buy)
        store.pending_rebalance = None

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
    scheduler._skip_update = skip_update

    if is_manual:
        store = SimpleNamespace(trader=td, whole_sub_id=None, pending_rebalance=None,
                                _now=sim_clock if skip_update else datetime.now,
                                _skip_update=skip_update)
        try:
            before_trade(store)
            pending = store.pending_rebalance
            if not pending:
                trading_logger.warning("没有生成可执行计划，已取消本次手动触发")
                sys.exit(0)

            confirmed = _print_and_confirm_manual_plan(pending)
            if not confirmed:
                trading_logger.warning("手动确认未通过，已取消本次交易")
                sys.exit(0)

            execute_trade(store)
            trading_logger.info("手动单次触发执行完成")
        finally:
            after_trade(store)
        sys.exit(0)

    scheduler.start_check_trading()

from datetime import datetime
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from xtquant import xtdata

from configs import TRADE_ACCOUNT
from data.db import allow_buy_stock_code_list
from core.backtest import _compute_factor_scores
from core.legality import LegalityChecker
from core.strategy_config import load_strategy_config
from core.strategy import build_rebalance_day
from core.timing import compute_position_multiplier_for_date
from data.db.stock_name import get_stock_name_at_date


def _build_plan_rows(*, trade_date, buy_n_stocks, buy_details, sell_details,
                     prices, final_score, valid_stocks, signal_date,
                     buy_skip_reasons: dict | None = None):
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
        if code in buy_actual:
            d = buy_actual[code]
            status = 'ok'; reason = 'topN换入'
            est_vol = int(d['shares']); est_amt = float(d['est_amount'])
        else:
            status = 'skipped'
            reason = buy_skip_reasons[code] if buy_skip_reasons is not None and code in buy_skip_reasons else '未纳入少补计划'
            est_vol = 0; est_amt = 0.0
        rows.append({
            'date': trade_date, 'code': code,
            'name': get_stock_name_at_date(code, signal_date) or '',
            'direction': 'buy', 'est_price': float(prices.get(code, 0.0)),
            'est_volume': est_vol, 'est_amount': est_amt,
            'factor_score': score_by_code[code],
            'limit_status': status, 'reason': reason, 'plan_seq': seq,
        })
    # 2. 卖出计划
    for seq, d in enumerate(sell_details, start=1):
        rows.append({
            'date': trade_date, 'code': d['code'], 'name': d['name'],
            'direction': 'sell', 'est_price': float(d['est_price']),
            'est_volume': int(d['volume']), 'est_amount': float(d['est_amount']),
            'factor_score': None,
            'limit_status': 'ok', 'reason': d['reason'],
            'plan_seq': seq + len(buy_n_stocks),
        })
    return rows
from trading.logger import trading_logger
from utils.recorder import recorder

from .manual_confirm import build_manual_confirmation_text, is_manual_confirmation_approved
from .persistence import _TRADE_DIR, live_trade_recorder
from .post_close import run_post_close, run_update_all
from .scheduler import PREPARE_REBALANCE_START, TradingScheduler
from .trader import Trader
from .executor import RebalanceExecutor


def _print_and_confirm_manual_plan(pending: dict) -> bool:
    message = build_manual_confirmation_text(pending)
    print(message)
    user_input = input("确认执行> ")
    return is_manual_confirmation_approved(user_input)


def _load_prev_eod_baseline(trade_date):
    """T-1 收盘基线：昨日持仓股数 + 昨日现金，用于把「今日目标」锚定到昨日收盘状态。

    动机：base_target 必须与「当天运行几次」无关——否则重复运行会用实时持仓/资金重算目标，
    叠加「低买被开盘价高估权益」的正反馈，导致重跑不断追价 churn（见 603810 case）。
    数据源与回测 seed-replay 同口径：positions_{T-1}.parquet + daily_summary[T-1].cash。

    Returns: (prev_cash, {code: yesterday_shares}) 或 (None, None)（缺快照时回退实时）。
    """
    from datetime import timedelta
    from utils.stock.time import get_last_trading_day
    from trading.persistence import _TRADE_DIR

    prev = get_last_trading_day(trade_date - timedelta(days=1))
    pos_path = _TRADE_DIR / f"positions_{prev.isoformat()}.parquet"
    summary_path = _TRADE_DIR / "daily_summary.parquet"
    if not pos_path.exists() or not summary_path.exists():
        return None, None
    import pandas as pd
    sdf = pd.read_parquet(summary_path)
    prev_rows = sdf[sdf['date'] == prev]
    if prev_rows.empty or 'cash' not in prev_rows.columns:
        return None, None
    prev_cash = float(prev_rows['cash'].iloc[-1])
    pdf = pd.read_parquet(pos_path)
    y_shares = {r['code']: int(r['volume'])
                for _, r in pdf.iterrows() if int(r['volume']) > 0}
    return prev_cash, y_shares


class MutableTime:
    """可变的模拟时钟。暴露 __call__ 用作 time_provider，set() 用于快进。"""
    def __init__(self, dt: datetime):
        self._dt = dt
    def __call__(self) -> datetime:
        return self._dt
    def set(self, dt: datetime):
        self._dt = dt


def _parse_skip_datetime(skip_arg: str) -> datetime:
    """解析 --skip 的快进时间；分钟级 09:25 自动对齐到开盘调仓触发秒。"""
    if len(skip_arg) == 12:
        skip_dt = datetime.strptime(skip_arg, '%Y%m%d%H%M')
        if skip_dt.hour == 9 and skip_dt.minute == 25 and skip_dt.second == 0:
            return datetime.combine(skip_dt.date(), PREPARE_REBALANCE_START)
        return skip_dt
    if len(skip_arg) == 14:
        return datetime.strptime(skip_arg, '%Y%m%d%H%M%S')
    raise ValueError(f'--skip 快进时间格式错误: {skip_arg}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual-config', type=str, default='configs/config.json', help='最终策略 config JSON 文件路径')
    parser.add_argument('--skip', type=str, help='回放/模拟模式。YYYYMMDD(8位)=逐日读parquet发飞书日报; YYYYMMDDHHMM(12位)/YYYYMMDDHHMMSS(14位)=快进scheduler模拟')
    parser.add_argument('--confirm', action='store_true',
                        help='单次手动模式：忽略时间窗口，立即跑一遍完整选股+买卖，且买卖前需手工 yes 确认。不加此参数则进入 scheduler 自动模式')
    parser.add_argument('--update', action='store_true', help='全量数据更新（K线+财务+股本+指数+退市→构建runtime NPZ）后退出')
    parser.add_argument('--dry-run', action='store_true', help='只计算选股计划，跳过实际买卖下单')
    parser.add_argument('--trade', action='store_true', help='实际执行QMT交易（覆盖--skip闭市跳过和--dry-run）')
    args = parser.parse_args()

    strategy_config = load_strategy_config(args.individual_config)
    factor_classes = strategy_config['factor_classes']
    individual_config = strategy_config['individual_config']
    if individual_config.get('filter_factors'):
        raise ValueError("filter_factors 目前仅支持研究回测，实盘路径未启用，禁止静默忽略")
    weights = individual_config['weights']
    buy_n = individual_config['buy_n']
    sell_m = individual_config['sell_m'] if 'sell_m' in individual_config else buy_n
    trading_logger.info(f"加载Individual_config: {args.individual_config}")
    rebalance_enabled = individual_config['rebalance'] if 'rebalance' in individual_config else True
    trading_logger.info(f"配置参数: buy_n={buy_n}, sell_m={sell_m}, rebalance={rebalance_enabled}, factors={sorted(weights.keys())}")

    if args.skip and len(args.skip) == 8:
        from datetime import datetime as _dt
        from trading.replay import replay_reports
        start = _dt.strptime(args.skip, '%Y%m%d').date()
        replay_reports(start, individual_config=individual_config, factor_classes=factor_classes)
        sys.exit(0)
    if args.skip and len(args.skip) not in (12, 14):
        raise ValueError(f'--skip 格式错误: {args.skip}')

    skip_update = args.skip is not None and len(args.skip) in (12, 14)
    # --skip 无 --update: 快进时间 + 跳过所有数据拉取
    # --skip --update: 快进时间 + 按实盘时间点正常拉数据
    # --update 单独: 全量更新后退出（不走 scheduler）
    no_data_fetch = skip_update and not args.update

    if args.update and not skip_update:
        run_update_all(None)
        sys.exit(0)

    dry_run = args.dry_run
    is_manual = args.confirm
    if skip_update:
        skip_dt = _parse_skip_datetime(args.skip)
        tag = "无数据拉取" if no_data_fetch else "含数据更新"
        trading_logger.info(f"快进模式 ({tag}), 模拟时间: {skip_dt}")
    else:
        skip_dt = None

    td = Trader(TRADE_ACCOUNT)
    rebalance_executor = RebalanceExecutor(td)
    _TRADE_DIR.mkdir(parents=True, exist_ok=True)
    (_TRADE_DIR / "trading_main.pid").write_text(str(os.getpid()), encoding="utf-8")

    def before_trade(store):
        prepare_t0 = time.time()
        stage_t = prepare_t0
        trade_now = store._now()
        trade_date = trade_now.date()
        signal_date = trade_date
        signal_datetime = datetime.combine(signal_date, datetime.min.time())

        trading_logger.info(
            f"开始预计算调仓: signal_date={signal_date.isoformat()}, trade_date={trade_date.isoformat()}"
        )
        recorder.mark("开始选股")

        # T-1 收盘基线（昨持仓+昨现金）：把今日目标锚定到昨日收盘，重复运行不漂移
        prev_cash, y_shares = _load_prev_eod_baseline(trade_date)

        if no_data_fetch:
            trading_logger.info("[盘前] 跳过 QMT 资产查询")
            prev_cash_qmt = None
            positions = {}
        else:
            asset = store.trader.query_asset()
            trading_logger.info(f"[盘前耗时] QMT资产查询: {time.time() - stage_t:.1f}s")
            stage_t = time.time()
            if asset is None:
                trading_logger.warning("资产查询失败，跳过本轮调仓")
                store.pending_rebalance = None
                return
            prev_cash_qmt = float(asset.cash)

            positions = {p.stock_code: p for p in store.trader.query_positions() if p.volume > 0}
            can_sell = sum(1 for p in positions.values() if p.can_use_volume > 0)
            trading_logger.info(f"持仓: {len(positions)} 只 (可卖{can_sell}), 总市值={sum(p.market_value for p in positions.values()):.0f}")
            trading_logger.info(f"[盘前耗时] QMT持仓查询: {time.time() - stage_t:.1f}s")
            stage_t = time.time()

        kline_overlay = None
        if not no_data_fetch:
            from data.update_live import update_live_quick
            anchor = skip_dt.date() if skip_update else None
            kline_overlay = update_live_quick(patch_npz=False, anchor_date=anchor)
            trading_logger.info(f"[盘前耗时] 快速数据更新: {time.time() - stage_t:.1f}s")
            stage_t = time.time()
        store._kline_overlay = kline_overlay

        if store.whole_sub_id is None and not skip_update:
            store.whole_sub_id = xtdata.subscribe_whole_quote(['SH', 'SZ'])
        trading_logger.info(f"[盘前耗时] 行情订阅: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        all_stocks = allow_buy_stock_code_list(target_date=trade_date)
        trading_logger.info(f"候选股票池: {len(all_stocks)} 只")
        trading_logger.info(f"[盘前耗时] 候选池加载: {time.time() - stage_t:.1f}s")
        stage_t = time.time()
        result = _compute_factor_scores(
            [signal_datetime], all_stocks, weights, factor_classes,
            kline_data=getattr(store, '_kline_overlay', None))
        if result is None:
            raise ValueError(f"信号日期 {signal_datetime.date()} 不在 runtime npz 日期范围内")
        data, all_scores, _, valid_dates, date_indices, valid_stocks, stock_indices = result
        score_date_idx = date_indices[0]
        trading_logger.info(f"[盘前耗时] runtime加载+因子计算: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)

        # 合法性闸门（实盘不传退市日，由 allow_buy 名单剔除）
        limit_up_protection = individual_config['limit_up_protection'] if 'limit_up_protection' in individual_config else False
        checker = LegalityChecker(data, stock_indices, limit_up_protection=limit_up_protection)

        # T 日开盘契约：signal_date == trade_date == score_date_idx → trade_idx=date_idx，与回测对齐。
        # 实盘严禁缺 T 日开盘价时回退到历史价格；当天 K 线不完整应中止调仓。
        trade_idx = score_date_idx
        trade_dt64 = data['trade_dates'][trade_idx]
        exec_date = trade_dt64.astype('datetime64[D]').item()

        _cash = float(asset.cash) if not no_data_fetch else (prev_cash or prev_cash_qmt or 0)
        pos_volumes = {c: int(p.volume) for c, p in positions.items()}
        sellable_volumes = {c: int(p.can_use_volume) for c, p in positions.items()}
        timing_mult = compute_position_multiplier_for_date(individual_config, exec_date)
        day_plan = build_rebalance_day(
            data=data, all_scores=all_scores, date_idx=score_date_idx, trade_idx=trade_idx,
            signal_date=exec_date, valid_stocks=valid_stocks, valid_cols=valid_cols,
            stock_indices=stock_indices, weights=weights, buy_n=buy_n, sell_m=sell_m,
            checker=checker, positions=pos_volumes, sellable_volumes=sellable_volumes,
            cash=_cash, rebalance=rebalance_enabled, position_multiplier=timing_mult,
            target_cash=prev_cash if prev_cash is not None else None,
            target_positions=y_shares if prev_cash is not None else None,
            price_codes_extra=set(y_shares or {}),
            limit_up_protection=limit_up_protection,
        )
        buy_n_stocks = day_plan.buy_n_stocks
        final_score = day_plan.final_score
        prices = day_plan.prices
        limit_prices = day_plan.limit_prices
        tradable_buy_stocks = day_plan.tradable_buy_stocks
        sell_orders = day_plan.sell_orders
        buy_orders = day_plan.buy_orders
        plan_skip_reasons = day_plan.skip_reasons
        total_eq = day_plan.total_eq
        base_target = day_plan.base_target
        live_eq = _cash + sum(day_plan.pos_vals.values())
        eq_note = (f"T-1基线 cash={prev_cash:.0f}+昨持仓@开盘"
                   if (prev_cash is not None and y_shares) else "实时(缺T-1快照,非幂等)")
        _log_extra = f", QMT.total_asset={asset.total_asset:.0f}" if not skip_update else ""
        trading_logger.info(f"选股完成 Top{buy_n}: {buy_n_stocks[:5]}...")
        trading_logger.info(
            f"[多退少补] total_eq={total_eq:.0f} ({eq_note}), base_target={base_target:.0f} "
            f"(reserve_L={day_plan.reserve_L:.2f}, timing={timing_mult:.2f}, 实时权益={live_eq:.0f}{_log_extra})"
        )
        trading_logger.info(f"[盘前耗时] 共享策略决策: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        # 准备 pending_rebalance
        sell_details = []
        for code, shares in sell_orders:
            pos = positions[code]
            vol = shares if shares > 0 else pos.can_use_volume
            reason = '换出' if shares < 0 else f'多退({shares}股)'
            sell_details.append({
                'code': code,
                'name': get_stock_name_at_date(code, signal_date) or '',
                'board': '',
                'volume': vol,
                'est_price': prices[code],
                'est_amount': vol * prices[code],
                'reason': reason,
            })

        buy_details = []
        for code, shares in buy_orders.items():
            buy_details.append({
                'code': code,
                'name': get_stock_name_at_date(code, signal_date) or '',
                'board': '',
                'shares': shares,
                'est_price': prices[code],
                'est_amount': shares * prices[code],
            })

        # 打印多退少补详情
        trading_logger.info(
            f"====== 多退少补计划: target={base_target:.0f}, 权益={total_eq:.0f}, 持仓={len(positions)}只 ======"
        )
        if sell_details:
            for d in sell_details:
                trading_logger.info(
                    f"  [卖] {d['code']} {d['name']} {d['reason']}: "
                    f"{d['volume']}股 @{d['est_price']:.2f} ≈{d['est_amount']:.0f}"
                )
        else:
            trading_logger.info("  [卖] 无")
        if buy_details:
            for d in buy_details:
                trading_logger.info(
                    f"  [买] {d['code']} {d['name']} 少补: "
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
            'limit_prices': limit_prices,
            'stock_indices': stock_indices,
        }

        # AL-5: 落地盘前调仓计划（候选股 + 订单 + 合法性状态）
        # 关键约束：plan 是「真实下单意图」记录，不允许被后续 dry-run / 二次跑覆盖
        # - dry-run 不下单 → 不写 plan
        # - 当日 plan 已存在 → 不覆盖（保护 9:30 真实记录不被 13:30/15:00 sim 覆盖）
        buy_skip_reasons = {}
        for code in buy_n_stocks:
            if code in buy_orders:
                continue
            px = float(prices.get(code, 0.0))
            if px <= 0:
                buy_skip_reasons[code] = '缺开盘价/停牌'
                continue
            if base_target <= 0:
                # sim / 无实盘资金基线：base_target=0 时所有 cv>=target 恒成立，
                # 不能误判「已达标」。此时无实盘下单意图，标注为无基线。
                buy_skip_reasons[code] = '无实盘基线'
            elif code not in tradable_buy_stocks:
                buy_skip_reasons[code] = '合法性过滤'
            else:
                # 已达标 / 未触发少补 / 冻结资金不足（来自统一多退少补实现）
                buy_skip_reasons[code] = plan_skip_reasons[code] if code in plan_skip_reasons else '未触发少补'

        plan_rows = _build_plan_rows(
            trade_date=trade_date,
            buy_n_stocks=buy_n_stocks, buy_details=buy_details,
            sell_details=sell_details, prices=prices, final_score=final_score,
            valid_stocks=valid_stocks, signal_date=signal_date,
            buy_skip_reasons=buy_skip_reasons,
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
        trading_logger.info(f"[盘前耗时] plan落地: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        trading_logger.info(
            f"调仓预计算完成: sell={len(sell_orders)}, buy={len(buy_orders)}, "
            f"target={base_target:.0f}, equity={total_eq:.0f}"
        )
        recorder.mark("完成调仓预计算")

        # 回测对账种子：复用上面已算好的因子分数跑 T-1 种子单日回放（继承前一日实盘状态），
        # 供战报「回测 vs 实盘」实时对账（T日操作 + T日持仓）。与盘后 diff 同口径。
        from .post_close import run_seed_replay_for_open
        from .day_board import extract_bt_reference
        bt_result = run_seed_replay_for_open(
            trade_date, individual_config,
            data=data, all_scores=all_scores, date_idx=score_date_idx,
            valid_stocks=valid_stocks, stock_indices=stock_indices)
        bt_ref = extract_bt_reference(bt_result) if bt_result is not None else None
        snaps = bt_result['daily_snapshots'] if bt_result is not None else None
        bt_daily_return = snaps[-1]['daily_return_pct'] if snaps else None
        trading_logger.info(f"[盘前耗时] 回测对账种子: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        # 飞书：开启「调仓战报」聚合卡片
        # 取代之前的「开盘准备完成」独立卡片 + 零散订单/成交回调卡片，
        # 全天只发一张可更新卡片，订单/成交回调实时刷新（debounce 300ms）。
        from .day_board import day_board
        day_board.start_session(
            trade_date=trade_date,
            plan_rows=plan_rows,
            equity=total_eq,
            position_count=len(positions),
            base_target=base_target,
            buy_n=buy_n,
            bt_ref=bt_ref,
            bt_daily_return=bt_daily_return,
            y_positions=y_shares or {},
        )
        trading_logger.info(f"[盘前耗时] 战报初始化: {time.time() - stage_t:.1f}s")
        trading_logger.info(f"[盘前耗时] 预计算总耗时: {time.time() - prepare_t0:.1f}s")

    def execute_trade(store, execute_sell=True, execute_buy=True):
        pending = store.pending_rebalance
        if not pending:
            trading_logger.warning("没有可执行的调仓计划，跳过本轮下单")
            return

        if dry_run and not args.trade:
            trading_logger.info("=== DRY RUN: 跳过实际买卖下单 ===")
            store.pending_rebalance = None
            return

        from datetime import datetime as _dt
        from utils.stock.time import is_current_trading as _is_trading
        wall_now = _dt.now()
        off_hours_fast = skip_update and not _is_trading(wall_now) and not args.trade
        if off_hours_fast:
            trading_logger.info(
                f"[执行] --skip 闭市演练: 真实时钟 {wall_now:%H:%M:%S} 非交易时段, "
                "跳过 order_stock(闭市会同步阻塞数分钟), 直接战报定稿+盘后报告")
        rebalance_executor.execute(
            pending, execute_sell=execute_sell, execute_buy=execute_buy,
            off_hours_fast=off_hours_fast)
        store.pending_rebalance = None

    def after_trade(store):
        if store.whole_sub_id is not None:
            xtdata.unsubscribe_quote(store.whole_sub_id)
            store.whole_sub_id = None
        store.pending_rebalance = None

    def post_close(store):
        run_post_close(store)

    def update_all(store):
        run_update_all(store)

    sim_clock = MutableTime(skip_dt) if skip_update else None
    scheduler = TradingScheduler(
        td,
        before_trade=before_trade,
        execute_trade=execute_trade,
        after_trade=after_trade,
        post_close=post_close,
        update_all=update_all,
        time_provider=sim_clock,
        fast_forward=skip_update,
    )
    scheduler._individual_config = individual_config
    scheduler._factor_classes = factor_classes
    scheduler._skip_update = no_data_fetch

    if is_manual:
        store = SimpleNamespace(trader=td, whole_sub_id=None, pending_rebalance=None,
                                _now=sim_clock if skip_update else datetime.now,
                                _skip_update=skip_update)
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
        after_trade(store)
        sys.exit(0)

    scheduler.start_check_trading()

from datetime import datetime
import argparse
import json
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from xtquant import xtdata

from configs import TRADE_ACCOUNT
from data.db import allow_buy_stock_code_list
from core.backtest import _compute_factor_scores
from core.scoring import select_topn
from core.legality import LegalityChecker
from core.ga import get_profile_factor_classes, resolve_profile_name
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
        d = buy_actual.get(code)
        if d:
            status = 'ok'; reason = 'topN换入'
            est_vol = int(d['shares']); est_amt = float(d['est_amount'])
        else:
            status = 'skipped'
            reason = (buy_skip_reasons or {}).get(code) or '未纳入少补计划'
            est_vol = 0; est_amt = 0.0
        rows.append({
            'date': trade_date, 'code': code,
            'name': get_stock_name_at_date(code, signal_date) or '',
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
    try:
        import pandas as pd
        sdf = pd.read_parquet(summary_path)
        prev_rows = sdf[sdf['date'] == prev]
        if prev_rows.empty or 'cash' not in prev_rows.columns:
            return None, None
        prev_cash = float(prev_rows['cash'].iloc[-1])
        pdf = pd.read_parquet(pos_path)
        y_shares = {r['code']: int(r['volume'])
                    for _, r in pdf.iterrows() if int(r['volume']) > 0}
        return (prev_cash, y_shares) if y_shares else (None, None)
    except Exception as e:
        trading_logger.warning(f"读取 T-1 基线失败，回退实时口径: {e}")
        return None, None


def _target_equity(prev_cash, y_shares, prices, live_fallback_eq):
    """今日目标权益：优先 T-1 现金 + 昨持仓×今开盘价（与运行次数无关）；缺基线时回退实时。"""
    if prev_cash is not None and y_shares:
        return prev_cash + sum(sh * prices.get(c, 0.0) for c, sh in y_shares.items())
    return live_fallback_eq


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
    parser.add_argument('--skip', type=str, help='回放/模拟模式。YYYYMMDD(8位)=逐日读parquet发飞书日报; YYYYMMDDHHMM(12位)=快进scheduler模拟')
    parser.add_argument('--confirm', action='store_true',
                        help='单次手动模式：忽略时间窗口，立即跑一遍完整选股+买卖，且买卖前需手工 yes 确认。不加此参数则进入 scheduler 自动模式')
    parser.add_argument('--update', action='store_true', help='全量数据更新（K线+财务+股本+指数+退市→构建runtime NPZ）后退出')
    parser.add_argument('--dry-run', action='store_true', help='只计算选股计划，跳过实际买卖下单')
    parser.add_argument('--trade', action='store_true', help='实际执行QMT交易（覆盖--skip闭市跳过和--dry-run）')
    args = parser.parse_args()

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

    if args.skip and len(args.skip) == 8:
        from datetime import datetime as _dt
        from trading.replay import replay_reports
        start = _dt.strptime(args.skip, '%Y%m%d').date()
        replay_reports(start, individual_config=individual_config, factor_classes=factor_classes)
        sys.exit(0)

    skip_update = args.skip is not None and len(args.skip) == 12
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
        skip_dt = datetime.strptime(args.skip, '%Y%m%d%H%M')
        tag = "无数据拉取" if no_data_fetch else "含数据更新"
        trading_logger.info(f"快进模式 ({tag}), 模拟时间: {skip_dt}")
    else:
        skip_dt = None

    td = Trader(TRADE_ACCOUNT)
    rebalance_executor = RebalanceExecutor(td)

    threading.Thread(target=create_lark_handler, args=[td], daemon=True).start()

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

        kline_overlay = None
        if not no_data_fetch:
            from data.update_live import update_live_quick
            anchor = skip_dt.date() if skip_update else None
            try:
                kline_overlay = update_live_quick(patch_npz=False, anchor_date=anchor)
            except Exception as e:
                trading_logger.warning(f"快速数据更新失败, 继续使用已有数据: {e}")
            finally:
                trading_logger.info(f"[盘前耗时] 快速数据更新: {time.time() - stage_t:.1f}s")
                stage_t = time.time()
        store._kline_overlay = kline_overlay

        if store.whole_sub_id is None and not skip_update:
            store.whole_sub_id = xtdata.subscribe_whole_quote(['SH', 'SZ'])
        trading_logger.info(f"[盘前耗时] 行情订阅: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        if no_data_fetch:
            trading_logger.info("[盘前] 跳过 QMT 资产查询")
            prev_cash_qmt = None
        else:
            asset = store.trader.query_asset()
            trading_logger.info(f"[盘前耗时] QMT资产查询: {time.time() - stage_t:.1f}s")
            stage_t = time.time()
            if asset is None:
                trading_logger.warning("资产查询失败，跳过本轮调仓")
                store.pending_rebalance = None
                return
            prev_cash_qmt = float(asset.cash)

        all_stocks = allow_buy_stock_code_list(target_date=trade_date)
        trading_logger.info(f"候选股票池: {len(all_stocks)} 只")
        trading_logger.info(f"[盘前耗时] 候选池加载: {time.time() - stage_t:.1f}s")
        stage_t = time.time()
        # 因子计算（与回测共用 _compute_factor_scores）
        try:
            result = _compute_factor_scores(
                [signal_datetime], all_stocks, weights, factor_classes,
                kline_data=getattr(store, '_kline_overlay', None))
            if result is None:
                raise ValueError(f"信号日期 {signal_datetime.date()} 不在 runtime npz 日期范围内")
            data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices = result
            score_date_idx = date_indices[0]
        except Exception as e:
            if is_manual:
                from datetime import timedelta
                from utils.stock.time import get_last_trading_day
                fallback = get_last_trading_day(trade_date)
                if fallback >= trade_date:
                    fallback = get_last_trading_day(trade_date - timedelta(days=1))
                trading_logger.warning(
                    f"NPZ 缺 {trade_date.isoformat()} 行(--skip 跳过了 update_live_quick)，"
                    f"signal/价格/合法性全部回退到 {fallback.isoformat()}；"
                    f"plan 中的 est_price 是 T-1 开盘价，与今日真实开盘价会 diff。"
                    f" 原始异常: {e}"
                )
                signal_date = fallback
                signal_datetime = datetime.combine(signal_date, datetime.min.time())
                result = _compute_factor_scores(
                    [signal_datetime], all_stocks, weights, factor_classes,
                    kline_data=getattr(store, '_kline_overlay', None))
                if result is None:
                    raise ValueError(f"回退日期 {signal_date} 也不在 runtime npz 日期范围内")
                data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices = result
                score_date_idx = date_indices[0]
            else:
                trading_logger.error(f"因子计算失败: {e}")
                store.pending_rebalance = None
                return
        trading_logger.info(f"[盘前耗时] runtime加载+因子计算: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        valid_cols = np.array([stock_indices[s] for s in valid_stocks], dtype=np.intp)

        # 加权选股（与回测共用 select_topn）
        buy_n_stocks, final_score = select_topn(
            all_scores, score_date_idx, valid_stocks, valid_cols,
            weights, temperatures, buy_n,
        )
        trading_logger.info(f"选股完成 Top{buy_n}: {buy_n_stocks[:5]}...")
        trading_logger.info(f"[盘前耗时] TopN选择: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        # 合法性闸门（与回测共用 LegalityChecker；实盘不传退市日，由 allow_buy 名单剔除）
        checker = LegalityChecker(data, stock_indices)
        open_all = data['open']

        # T 日开盘契约：signal_date == trade_date == score_date_idx → trade_idx=date_idx，与回测对齐。
        # 例外：is_manual + NPZ 缺 T 日行时，signal_date 已被回退到 T-1，
        # 此时 trade_idx 也指向 T-1 行，day_open=open[T-1]（仅用于排演，非生产路径）。
        trade_idx = score_date_idx
        trade_dt64 = data['trade_dates'][trade_idx]
        exec_date = trade_dt64.astype('datetime64[D]').item()

        # 获取当日价格
        day_open = open_all[trade_idx]

        # T-1 收盘基线（昨持仓+昨现金）：把今日目标锚定到昨日收盘，重复运行不漂移
        prev_cash, y_shares = _load_prev_eod_baseline(trade_date)

        # 获取持仓
        if no_data_fetch:
            trading_logger.info("[盘前] 跳过 QMT 持仓查询")
            positions = {}
        else:
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
        trading_logger.info(f"[盘前耗时] QMT持仓查询: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        prices = {}
        price_codes = set(positions.keys()) | set(buy_n_stocks)
        if y_shares:
            price_codes |= set(y_shares.keys())  # 昨持仓也要估值（含今日已清空的）
        for code in price_codes:
            si = stock_indices.get(code)
            if si is None:
                continue
            open_val = day_open[si]
            if not np.isnan(open_val) and open_val > 0:
                prices[code] = float(open_val)
                continue
            # T 日停牌/无开盘价：回退到 NPZ 中该股最近一个有效开盘价。
            # 与回测 _last_valid_price 跨日累积口径一致（实盘单次调用无法靠局部缓存累积，
            # 必须直接扫 NPZ 历史），否则停牌持仓会被按 0 估值，污染 total_eq/base_target。
            col = open_all[:trade_idx + 1, si]
            valid = col[(~np.isnan(col)) & (col > 0)]
            if valid.size:
                prices[code] = float(valid[-1])
        trading_logger.info(f"[盘前耗时] 开盘价映射: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        # 多退少补 rebalance（与回测严格对齐）
        # 1. 用 volume(总持仓) 算市值，避免 T+0 买入后 can_use_volume=0 导致重复买入
        # 2. total_eq 用 open[T] 重算（不用 QMT.total_asset），跟回测口径一致；
        #    否则 QMT 的 last_price/close[T-1] 与 open[T] 之间跳空会让 base_target 偏移，
        #    导致回测/实盘 base_target 不一致 → 阈值穿越触发不一致的少补。
        # cv（多退少补的当前缺口）仍用「当前持仓×开盘价」——这样重跑只补齐到目标、收敛不重复。
        pos_vals = {c: p.volume * prices.get(c, 0) for c, p in positions.items()}
        _cash = float(asset.cash) if not no_data_fetch else (prev_cash or prev_cash_qmt or 0)
        live_eq = _cash + sum(pos_vals.values())
        # total_eq（决定 base_target/目标）锚定 T-1 收盘基线，与当天运行次数无关；缺基线时回退实时。
        total_eq = _target_equity(prev_cash, y_shares, prices, live_eq)
        eq_note = (f"T-1基线 cash={prev_cash:.0f}+昨持仓@开盘"
                   if (prev_cash is not None and y_shares) else "实时(缺T-1快照,非幂等)")
        timing_mult = 1.0
        # 市价单涨停价冻结预留：均匀满仓时最后一只也要冻结得起 → base_target = E/(buy_n+reserve_L)
        # （与回测 _backtest_direct 的 market_order_freeze 口径一致）
        from utils.stock.info import board_limit_ratio as _blr, limit_up_price as _lup
        reserve_L = max((_blr(c) for c in buy_n_stocks), default=0.0)
        base_target = total_eq * timing_mult / (buy_n + reserve_L)
        _log_extra = f", QMT.total_asset={asset.total_asset:.0f}" if not skip_update else ""
        trading_logger.info(
            f"[多退少补] total_eq={total_eq:.0f} ({eq_note}), base_target={base_target:.0f} "
            f"(reserve_L={reserve_L:.2f}, 实时权益={live_eq:.0f}{_log_extra})"
        )
        # 市价买单资金冻结单价 = 前收×(1+板块涨跌幅)，供执行器按涨停价做预算/冻结，
        # 避免按开盘价低估占用 → 末位标的反复「资金可用数不足」废单。
        close_all = data['close']
        prev_close_row = close_all[trade_idx - 1] if trade_idx >= 1 else day_open
        limit_prices = {}
        for code in prices:
            si = stock_indices.get(code)
            pc = float(prev_close_row[si]) if si is not None else 0.0
            lp = _lup(code, pc) if (pc and not np.isnan(pc)) else 0.0
            limit_prices[code] = lp if lp > 0 else prices[code]
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
            ok, _ = checker.check(si_list, trade_idx, exec_date, is_buy=False)
            sell_orders = [(c, s) for j, (c, s) in enumerate(sell_orders) if ok[j]]

        # 涨跌停过滤买入
        buy_orders_pre_legality = dict(buy_orders)
        if buy_orders:
            bc = list(buy_orders.keys())
            bi_list = [stock_indices[c] for c in bc if c in stock_indices]
            ok, _ = checker.check(bi_list, trade_idx, exec_date, is_buy=True)
            buy_orders = {c: s for j, (c, s) in enumerate(buy_orders.items()) if ok[j]}
        trading_logger.info(f"[盘前耗时] 多退少补+合法性: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

        # 准备 pending_rebalance
        sell_details = []
        for code, shares in sell_orders:
            pos = positions.get(code)
            vol = shares if shares > 0 else (pos.can_use_volume if pos else 0)
            reason = '换出' if shares < 0 else f'多退({shares}股)'
            sell_details.append({
                'code': code,
                'name': get_stock_name_at_date(code, signal_date) or '',
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
                'name': get_stock_name_at_date(code, signal_date) or '',
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
            px = float(prices.get(code, 0) or 0)
            if px <= 0:
                buy_skip_reasons[code] = '缺开盘价'
                continue
            cv = float(pos_vals.get(code, 0) or 0)
            if base_target <= 0:
                # sim / 无实盘资金基线：base_target=0 时所有 cv>=target 恒成立，
                # 不能误判「已达标」。此时无实盘下单意图，标注为无基线。
                buy_skip_reasons[code] = '无实盘基线'
            elif cv >= base_target * 0.99:
                buy_skip_reasons[code] = '已达标'
            elif code in buy_orders_pre_legality:
                buy_skip_reasons[code] = '合法性过滤'
            else:
                buy_skip_reasons[code] = '未触发少补'

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
        bt_ref = None
        bt_daily_return = None
        try:
            from .post_close import (run_seed_replay_for_open,
                                     _run_continuous_backtest, _resolve_backtest_start)
            from .day_board import extract_bt_reference
            bt_result = run_seed_replay_for_open(
                trade_date, individual_config,
                data=data, all_scores=all_scores, date_idx=score_date_idx,
                valid_stocks=valid_stocks, stock_indices=stock_indices)
            if bt_result is None:
                # 无 T-1 种子（sim / 首日 / 缺快照）：回退连续回测，
                # 让战报「目标」列仍有值（与盘后 run_post_close 同口径）。
                bt_start = _resolve_backtest_start(trade_date)
                bt_result = _run_continuous_backtest(
                    bt_start, trade_date, individual_config, factor_classes,
                    kline_data=getattr(store, '_kline_overlay', None))
            if bt_result is not None:
                bt_ref = extract_bt_reference(bt_result)
                snaps = bt_result.get('daily_snapshots') or []
                if snaps:
                    bt_daily_return = snaps[-1].get('daily_return_pct')
        except Exception as e:
            trading_logger.warning(f"盘中回测对账种子构建失败: {e}")
        trading_logger.info(f"[盘前耗时] 回测对账种子: {time.time() - stage_t:.1f}s")
        stage_t = time.time()

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
                bt_ref=bt_ref,
                bt_daily_return=bt_daily_return,
                y_positions=y_shares or {},
            )
        except Exception as e:
            trading_logger.warning(f"飞书战报初始化失败: {e}")
        trading_logger.info(f"[盘前耗时] 战报初始化: {time.time() - stage_t:.1f}s")
        trading_logger.info(f"[盘前耗时] 预计算总耗时: {time.time() - prepare_t0:.1f}s")

    def execute_trade(store, execute_sell=True, execute_buy=True):
        pending = store.pending_rebalance
        if not pending:
            trading_logger.warning("没有可执行的调仓计划，跳过本轮下单")
            return

        if dry_run and not args.trade:
            trading_logger.info("=== DRY RUN: 跳过实际买卖下单 ===")
            store._dry_run_plan = pending
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

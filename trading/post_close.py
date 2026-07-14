"""15:00 盘后工作流：拉收盘K线 → 单日回测 → 实盘vs回测对比 → 飞书报告。

同时负责 16:00 update-all（带重试）。
"""
import time
import traceback
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from data.db.stock_list import allow_buy_stock_code_list
from trading.lark.sender import LarkMsgLevel, lark_sender
from trading.logger import trading_logger
from utils.recorder import recorder


# ================================================================
# 15:00 盘后K线更新
# ================================================================

def _post_close_update_kline(trade_date: date | None = None):
    """15:00 收盘 K 线 → 只写 parquet，不落盘 NPZ（16:00 update_all 再全量写）。"""
    from data.update_live import _download_kline_all
    t0 = time.time()
    kline_data = _download_kline_all(anchor_date=trade_date)
    trading_logger.info(
        f"[PostClose] K线→parquet ({len(kline_data)}只, {time.time()-t0:.0f}s, 锚定={trade_date})"
    )
    return kline_data


# ================================================================
# 多日连续回测：从实盘首调仓日 → T 日，模拟实盘演化形成对齐的 T-1 持仓
# ================================================================

def _resolve_backtest_start(trade_date: date) -> date:
    """从 daily_summary 推断回测起点。
    
    取第一个发生过调仓的日期作为起点（buy_count>0 或 sell_count>0）。
    若无 daily_summary 或无调仓记录，回退到 trade_date（单日模式）。
    """
    from trading.persistence import _TRADE_DIR
    summary_path = _TRADE_DIR / "daily_summary.parquet"
    if not summary_path.exists():
        return trade_date
    sdf = pd.read_parquet(summary_path)
    if sdf.empty:
        return trade_date
    moved = sdf[(sdf['buy_count'] > 0) | (sdf['sell_count'] > 0)]
    if moved.empty:
        return trade_date
    first_move = moved['date'].min()
    return first_move if isinstance(first_move, date) else first_move.date()


def _extract_live_seed(pos_df, summary_df, prev_date: date):
    """从 positions_{T-1} 与 daily_summary 提取单日回放的种子。

    Returns (seed_cash, seed_positions, y_positions_eod) 或 None：
      - seed_cash: T-1 日终现金（实盘真实可用现金基线）
      - seed_positions: {code: {'volume', 'avg_price'}}，喂给 _backtest_direct 做起始持仓
      - y_positions_eod: T-1 收盘持仓快照（last_price 估值），合成 T-1 snapshot 算 daily_pnl 基线
    任一关键数据缺失返回 None（调用方回退连续回测）。
    """
    if pos_df is None or pos_df.empty or summary_df is None or summary_df.empty:
        return None
    prev_rows = summary_df[summary_df['date'] == prev_date]
    if prev_rows.empty or 'cash' not in prev_rows.columns:
        return None
    seed_cash = float(prev_rows['cash'].iloc[-1])

    seed_positions: dict[str, dict] = {}
    y_positions_eod: list[dict] = []
    for _, r in pos_df.iterrows():
        vol = int(r['volume'])
        if vol <= 0:
            continue
        code = r['code']
        avg = float(r['avg_price']) if pd.notna(r.get('avg_price')) else 0.0
        lp = float(r['last_price']) if pd.notna(r.get('last_price')) else 0.0
        seed_positions[code] = {'volume': vol, 'avg_price': avg}
        y_positions_eod.append({
            'code': code, 'volume': vol, 'avg_price': avg,
            'current_price': lp, 'current_value': lp * vol,
        })
    if not seed_positions:
        return None
    return seed_cash, seed_positions, y_positions_eod


def _load_seed(trade_date: date):
    """读取 T-1 实盘种子（现金 + 持仓 + 收盘估值）。

    Returns (prev_date, seed_cash, seed_positions, y_positions_eod) 或 None。
    数据源：positions_{T-1}.parquet + daily_summary[T-1].cash —— 盘前/盘后共用同一种子，
    保证「开盘 diff」与「收盘 diff」基于完全相同的回测起点。
    """
    from datetime import timedelta
    from trading.persistence import _TRADE_DIR
    from utils.stock.time import get_last_trading_day

    prev_date = get_last_trading_day(trade_date - timedelta(days=1))
    pos_path = _TRADE_DIR / f"positions_{prev_date.isoformat()}.parquet"
    summary_path = _TRADE_DIR / "daily_summary.parquet"
    if not pos_path.exists() or not summary_path.exists():
        return None
    seed = _extract_live_seed(pd.read_parquet(pos_path),
                              pd.read_parquet(summary_path), prev_date)
    if seed is None:
        return None
    seed_cash, seed_positions, y_positions_eod = seed
    return prev_date, seed_cash, seed_positions, y_positions_eod


def _load_latest_seed(trade_date: date):
    """找 T 日之前**最近一个可用**的实盘持仓快照做种子（T-1 缺失时的兜底）。

    正常应取 T-1（_load_seed）；当 T-1 快照缺失（链路断裂）时，退而取更早一天的真实
    持仓+现金，配合多日连续回测从该真实资金基线演化到 T 日，使「目标持仓」与实盘的
    资金体量对齐（避免从默认 70 万空仓起步导致每只系统性偏小）。

    Returns (seed_date, seed_cash, seed_positions) 或 None。
    """
    from trading.persistence import _TRADE_DIR
    summary_path = _TRADE_DIR / "daily_summary.parquet"
    if not summary_path.exists():
        return None
    summary_df = pd.read_parquet(summary_path)
    cand = []
    for p in _TRADE_DIR.glob("positions_*.parquet"):
        try:
            d = date.fromisoformat(p.stem.split('positions_')[1])
        except (ValueError, IndexError):
            continue
        if d < trade_date:
            cand.append((d, p))
    for seed_date, p in sorted(cand, reverse=True):  # 由近及远
        seed = _extract_live_seed(pd.read_parquet(p), summary_df, seed_date)
        if seed is not None:
            seed_cash, seed_positions, _ = seed
            return seed_date, seed_cash, seed_positions
    return None


def _load_or_rebuild_seed(trade_date: date, individual_config: dict,
                          factor_classes: list, kline_data: dict | None = None):
    """加载 T-1 种子；缺失时从最近真实快照回放到 T-1。"""
    loaded = _load_seed(trade_date)
    if loaded is not None:
        return loaded

    from datetime import timedelta
    from utils.stock.time import get_last_trading_day

    latest = _load_latest_seed(trade_date)
    if latest is None:
        return None
    seed_date, seed_cash, seed_positions = latest
    prev_date = get_last_trading_day(trade_date - timedelta(days=1))
    if seed_date >= prev_date:
        return None

    rebuilt = _run_continuous_backtest(
        seed_date + timedelta(days=1), prev_date,
        individual_config, factor_classes,
        kline_data=kline_data,
        init_cash=seed_cash,
        init_positions=seed_positions,
    )
    snaps = rebuilt.get('daily_snapshots') if rebuilt else None
    if not snaps or snaps[-1].get('date') != prev_date.isoformat():
        trading_logger.warning(
            f"[PostClose] 无法重建 T-1 种子: {seed_date} -> {prev_date}")
        return None

    snap = snaps[-1]
    y_positions_eod = snap.get('positions_eod') or []
    rebuilt_positions = {
        p['code']: {'volume': int(p['volume']), 'avg_price': float(p['avg_price'])}
        for p in y_positions_eod if int(p.get('volume', 0)) > 0
    }
    if not rebuilt_positions:
        return None
    trading_logger.info(
        f"[PostClose] 已由 {seed_date} 回放重建 T-1 种子 {prev_date}: "
        f"现金={snap['cash']:,.0f}, 持仓={len(rebuilt_positions)} 只")
    return prev_date, float(snap['cash']), rebuilt_positions, y_positions_eod


def _get_filter_factor_classes(individual_config: dict) -> list:
    from core.factors.registry import get_factor_class

    return [
        get_factor_class(name)
        for name, enabled in individual_config.get('filter_factors', {}).items()
        if enabled
    ]


def _seed_replay_core(*, prev_date: date, seed_cash: float, seed_positions: dict,
                      y_positions_eod: list, decision,
                      individual_config: dict) -> dict | None:
    """用 T-1 种子 + 已算好的因子分数跑单日回放，并合成 [y_snap, t_snap]。

    盘前（before_trade，复用实时分数）与盘后（run_post_close，重算分数）共用此核心，
    使「开盘 diff」与「收盘 diff」的回测口径完全一致。
    """
    from core.backtest import _backtest_direct
    import numpy as np

    weights = individual_config['weights']
    buy_n = individual_config['buy_n']
    sell_m = individual_config.get('sell_m', buy_n)

    # 只保留 NPZ 覆盖到的种子持仓（其余无法估值/交易）。
    seed_in_npz = {c: v for c, v in seed_positions.items() if c in decision.stock_indices}
    dropped = sorted(set(seed_positions) - set(seed_in_npz))
    if dropped:
        trading_logger.warning(
            f"[SeedReplay] {len(dropped)} 只种子持仓不在 NPZ，回放忽略: {dropped[:5]}")

    bt_result = _backtest_direct(
        decision.data, decision.all_scores,
        [datetime.combine(decision.trade_date, datetime.min.time())],
        [decision.date_idx], decision.valid_stocks, decision.stock_indices,
        weights=weights, buy_n=buy_n, sell_m=sell_m,
        init_cash=seed_cash, init_positions=seed_in_npz,
        position_multipliers=np.array([decision.position_multiplier]),
        limit_up_protection=individual_config.get('limit_up_protection', False),
        rebalance=individual_config.get('rebalance', True),
        filter_masks=decision.filter_masks,
    )
    snaps = bt_result.get('daily_snapshots') or []
    if not snaps:
        return None
    t_snap = snaps[-1]

    # 合成 T-1 快照（实盘 T-1 收盘价值的种子持仓），供 _rebuild_backtest_per_stock_pnl
    # 取 snaps[-2] 作 T 日 daily_pnl 基线，并据此重算 T 日 daily_return_pct
    # （单日回放下 _backtest_direct 的 daily_return 基于 init_cash，未含种子持仓，需覆盖）。
    y_total = seed_cash + sum(p['current_value'] for p in y_positions_eod)
    y_snap = {
        'date': prev_date.strftime('%Y-%m-%d'),
        'signal_date': prev_date.strftime('%Y-%m-%d'),
        'trade_date': prev_date.strftime('%Y-%m-%d'),
        'total_asset': y_total, 'cash': seed_cash,
        'market_value': y_total - seed_cash,
        'positions_eod': y_positions_eod,
        'daily_return_pct': 0.0, 'cumulative_return_pct': 0.0,
    }
    t_total = t_snap.get('total_asset', y_total)
    t_snap['daily_return_pct'] = ((t_total - y_total) / y_total * 100) if y_total else 0.0
    bt_result['daily_snapshots'] = [y_snap, t_snap]
    return bt_result


def _run_seed_replay(trade_date: date, individual_config: dict,
                     factor_classes: list, kline_data: dict | None = None) -> dict | None:
    """盘后单日回放：用实盘 T-1 真实持仓 + 现金做种子，只回放 T 日多退少补。

    目的：让回测端 T 日的持仓/下单手数与实盘可比 —— 两边起点（资金+持仓）完全相同，
    手数差只来自执行层（废单/部成/资金不足/min_lot），价格差只来自滑点。
    无 T-1 种子（首个交易日 / 缺快照）时返回 None，调用方回退连续回测。
    """
    from core.backtest import is_rebalance_day_index, stock_pool_prefixes
    from core.strategy import build_strategy_day
    from core.timing import compute_position_multiplier_for_date
    from trading.persistence import get_live_rebalance_index

    loaded = _load_or_rebuild_seed(
        trade_date, individual_config, factor_classes, kline_data=kline_data)
    if loaded is None:
        trading_logger.info("[PostClose] 缺 T-1 种子，无法构造单日回放")
        return None
    prev_date, seed_cash, seed_positions, y_positions_eod = loaded

    boards = stock_pool_prefixes(individual_config.get('stock_pool'))
    all_stocks = allow_buy_stock_code_list(target_date=trade_date, boards=boards)
    filter_factor_classes = _get_filter_factor_classes(individual_config)
    volumes = {code: int(info['volume']) for code, info in seed_positions.items()}
    rebalance_idx = get_live_rebalance_index(trade_date)
    decision = build_strategy_day(
        trade_date=trade_date, all_stocks=all_stocks,
        individual_config=individual_config, factor_classes=factor_classes,
        filter_factor_classes=filter_factor_classes,
        positions=volumes, sellable_volumes=volumes, cash=seed_cash,
        kline_data=kline_data,
        is_rebalance_day=is_rebalance_day_index(
            rebalance_idx, individual_config.get('holding_period')),
        position_multiplier=compute_position_multiplier_for_date(
            individual_config, trade_date),
    )

    bt_result = _seed_replay_core(
        prev_date=prev_date, seed_cash=seed_cash, seed_positions=seed_positions,
        y_positions_eod=y_positions_eod, decision=decision,
        individual_config=individual_config)
    if bt_result is None:
        return None
    t_snap = bt_result['daily_snapshots'][-1]
    y_total = bt_result['daily_snapshots'][0]['total_asset']
    trading_logger.info(
        f"[PostClose] 单日回放完成 (种子: {prev_date} 现金¥{seed_cash:,.0f} + "
        f"{len(seed_positions)}只持仓, 权益¥{y_total:,.0f}) · T日回测收益={t_snap['daily_return_pct']:+.2f}%"
    )
    return bt_result


def run_seed_replay_for_open(trade_date: date, individual_config: dict, *,
                             decision, factor_classes: list | None = None) -> dict | None:
    """盘前/盘中：复用 before_trade 已算好的因子分数，跑 T-1 种子单日回放。

    供战报做「回测 vs 实盘」实时对账。回测端继承 T-1 实盘现金+持仓，与盘后口径一致；
    因子分数直接复用（不重算），几乎不增加盘前耗时。无 T-1 种子时返回 None（首日/缺快照）。
    """
    loaded = (_load_or_rebuild_seed(trade_date, individual_config, factor_classes)
              if factor_classes is not None else _load_seed(trade_date))
    if loaded is None:
        trading_logger.info("[SeedReplay@open] 缺 T-1 种子，跳过盘中回测对账")
        return None
    prev_date, seed_cash, seed_positions, y_positions_eod = loaded

    bt_result = _seed_replay_core(
        prev_date=prev_date, seed_cash=seed_cash, seed_positions=seed_positions,
        y_positions_eod=y_positions_eod, decision=decision,
        individual_config=individual_config)
    if bt_result is None:
        return None
    t_snap = bt_result['daily_snapshots'][-1]
    trading_logger.info(
        f"[SeedReplay@open] 完成 (种子 {prev_date} 现金¥{seed_cash:,.0f} + "
        f"{len(seed_positions)}只持仓) · T日回测收益={t_snap['daily_return_pct']:+.2f}%")
    return bt_result


def _run_continuous_backtest(start_date: date, end_date: date,
                              individual_config: dict, factor_classes: list,
                              kline_data: dict | None = None,
                              init_cash: float = 1_000_000.0,
                              init_positions: dict | None = None) -> dict | None:
    """连续多日回测 [start_date, end_date]，让回测演化形成对齐 T-1 的持仓。

    init_cash / init_positions: 实盘真实资金基线种子（T-1 缺失时取更早一天的真实持仓+
    现金），使回测资金体量与实盘对齐，目标持仓股数不再系统性偏小。默认 100 万空仓。

    返回 dict 与 _backtest_direct 一致；其中 daily_snapshots[-1] 为 T 日 snapshot。
    PostCloseReport._bt_snap() 默认取 [-1]，_rebuild_backtest_per_stock_pnl
    用 [-1] vs [-2] 算 T 日 daily_pnl。
    """
    from core.backtest import (
        _compute_factor_scores, _backtest_direct, _compute_timing_multipliers,
        build_list_dates_map,
    )
    from core.runtime import load_runtime_stock_codes
    from utils.stock.time import get_trading_date_span

    trading_days = get_trading_date_span(start_date, end_date)
    if not trading_days:
        trading_logger.warning(f"[PostClose] {start_date} → {end_date} 无交易日")
        return None

    signal_datetimes = [datetime.combine(d, datetime.min.time()) for d in trading_days]
    all_stocks = load_runtime_stock_codes()
    weights = individual_config['weights']
    filter_factor_classes = _get_filter_factor_classes(individual_config)
    buy_n = individual_config['buy_n']
    sell_m = individual_config.get('sell_m', buy_n)

    scores_result = _compute_factor_scores(
        signal_datetimes, all_stocks, weights, factor_classes,
        kline_data=kline_data,
        filter_factor_classes=filter_factor_classes or None,
    )
    if scores_result is None:
        return None

    data, all_scores, filter_masks, valid_dates, date_indices, valid_stocks, stock_indices = scores_result
    if not valid_dates:
        return None

    timing_multipliers = _compute_timing_multipliers(individual_config, valid_dates)
    prefilter_n = individual_config.get('prefilter_n')
    seed_in_npz = ({c: v for c, v in init_positions.items() if c in stock_indices}
                   if init_positions else None)
    return _backtest_direct(
        data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
        weights=weights, buy_n=buy_n, sell_m=sell_m,
        holding_period=individual_config.get('holding_period'),
        position_multipliers=timing_multipliers,
        list_dates_map=build_list_dates_map(data),
        lightweight=False, init_cash=init_cash, init_positions=seed_in_npz,
        limit_up_protection=individual_config.get('limit_up_protection', False),
        rebalance=individual_config.get('rebalance', True),
        filter_masks=filter_masks,
        prefilter_n=prefilter_n,
    )


# ================================================================
# 主入口
# ================================================================

def run_post_close(store) -> dict | None:
    """AL-5 盘后 5 维 Diff 调度。

    时序：
      1. 拉收盘 K 线 + NPZ 增量修补
      2. 单日回测（用最新 NPZ）
      3. 写 positions_{T}.parquet 快照（含 daily_pnl）
      4. 读 plan_{T} / fills_{T} / positions_{T}
      5. 组装 PostCloseReport 5 维 diff
      6. 飞书卡片 + HTML 报告
      7. 写 daily_summary
    """
    trade_date = store._now().date()
    trading_logger.info("=" * 50)
    trading_logger.info(f"盘后 Diff @ {trade_date.isoformat()}")
    trading_logger.info("=" * 50)

    individual_config = getattr(store, '_individual_config', None)
    factor_classes = getattr(store, '_factor_classes', None)
    if individual_config is None or factor_classes is None:
        trading_logger.error("[PostClose] 缺少 config/factor_classes, 跳过")
        return None

    _skip = getattr(store, '_skip_update', False)

    # 1. 拉收盘 K 线 → parquet only（NPZ 等 16:00 update_all）
    recorder.mark("盘后K线更新")
    kline_data = None
    if not _skip:
        try:
            kline_data = _post_close_update_kline(trade_date)
        except Exception:
            trading_logger.exception("[PostClose] K线更新失败，继续使用已有 NPZ 回测")

    # 2. 回测端：优先「单日回放」——用实盘 T-1 真实持仓+现金做种子，只回放 T 日多退少补，
    #    使持仓/下单手数与实盘可比（手数差归因执行层、价格差归因滑点）。
    #    无 T-1 种子（首个交易日 / 缺快照）时回退到多日连续回测。
    recorder.mark("盘后回测")
    bt_result = _run_seed_replay(
        trade_date, individual_config, factor_classes, kline_data=kline_data)
    if bt_result is None:
        trading_logger.warning("[PostClose] 回测失败")
        return None
    snaps = bt_result.get('daily_snapshots') or []
    trading_logger.info(f"[PostClose] 回测完成: {len(snaps)}日快照")

    # 3. 构建对比报告
    recorder.mark("盘后对比")
    from trading.persistence import live_trade_recorder, _TRADE_DIR
    from trading.report import PostCloseReport

    report = PostCloseReport(trade_date)

    # 3.1 实盘资产
    prev_asset = None
    summary_path = _TRADE_DIR / "daily_summary.parquet"
    if summary_path.exists():
        prev = pd.read_parquet(summary_path)
        prev_rows = prev[prev['date'] < trade_date]
        if not prev_rows.empty:
            prev_asset = float(prev_rows['total_asset'].iloc[-1])

    # 资产仅查一次，feed_asset / write_daily_summary 共用，避免 QMT 时刻漂移
    live_asset = None
    asset_snapshot = None
    if _skip:
        trading_logger.info("[PostClose] --skip 模式跳过 QMT 资产查询")
    else:
        try:
            asset_snapshot = store.trader.query_asset()
            if asset_snapshot:
                live_asset = float(asset_snapshot.total_asset)
        except Exception as e:
            trading_logger.warning(f"[PostClose] 资产查询失败: {e}")

    # 3.1.1 同步 QMT 银证流水到 cash_flows（自动识别日内入金/出金）
    if _skip:
        trading_logger.info("[PostClose] --skip 模式跳过 银证流水同步")
    else:
        try:
            live_trade_recorder.sync_bank_transfers_from_qmt(store.trader, trade_date=trade_date)
        except Exception as e:
            trading_logger.warning(f"[PostClose] 银证流水同步失败: {e}")

    # 3.1.2 兜底从 QMT 回填当日成交，弥补 watcher.on_stock_order 漏接
    # 这一步必须在 snapshot_positions 之前，否则 daily_pnl 公式会基于不完整 fills 算错。
    if _skip:
        trading_logger.info("[PostClose] --skip 模式跳过 QMT 成交回填")
    else:
        try:
            live_trade_recorder.backfill_fills_from_qmt(store.trader, trade_date=trade_date)
        except Exception as e:
            trading_logger.warning(f"[PostClose] QMT 成交回填失败: {e}")

    report.feed_asset(
        live_asset, prev_asset,
        net_cash_flow=live_trade_recorder.get_today_cash_flows(trade_date=trade_date))

    # 3.2 实盘 fills（回填后再取，给 snapshot_positions 用）— 支持 sim/进程重启回退 parquet
    live_fills = live_trade_recorder.get_today_fills_df(trade_date=trade_date)
    report.feed_fills_df(live_fills)

    # 3.3 QMT 持仓快照（含 daily_pnl 计算 + 落地 positions_{T}.parquet）
    positions = []
    if _skip:
        trading_logger.info("[PostClose] --skip 模式跳过 QMT 持仓查询")
    else:
        try:
            positions = store.trader.query_positions()
        except Exception as e:
            trading_logger.warning(f"[PostClose] 持仓查询失败: {e}")
            positions = []
    if positions:
        live_trade_recorder.snapshot_positions(positions, fills_df=live_fills, trade_date=trade_date)
    # positions_{T}.parquet 为权威实盘持仓源：实盘由上面 snapshot 落地，
    # sim/replay 直接读盘上已有快照（不查 QMT，避免非交易时段卡死）。
    pos_path = _TRADE_DIR / f"positions_{trade_date.isoformat()}.parquet"
    live_positions_df = pd.read_parquet(pos_path) if pos_path.exists() else None
    if live_positions_df is not None:
        report.feed_positions_df(live_positions_df)

    # 3.4 实盘 plan
    plan_path = _TRADE_DIR / f"plan_{trade_date.isoformat()}.parquet"
    if plan_path.exists():
        report.feed_plan_df(pd.read_parquet(plan_path))
    else:
        trading_logger.warning(f"[PostClose] {plan_path.name} 不存在，维度 1/2 数据缺失")

    # 3.5 回测结果
    report.feed_backtest(bt_result)

    # 4. HTML 报告（飞书卡片统一由 day_board 发送）
    html_dir = Path(__file__).resolve().parents[1] / "data" / "live_trades" / "reports"
    html_path = html_dir / f"diff_{trade_date.isoformat()}.html"
    report.to_html(html_path)
    report_data = report.send(html_path=html_path)

    # 5. 持久化日终摘要（复用上面已查到的 asset_snapshot，保证与 report 中的账户 P&L 完全一致）
    #    daily_pnl 以报告的「个股盈亏总和」口径为准（免疫未记账出入金），与飞书/HTML 一致；
    #    个股口径不可信（有漏记成交）时 per_stock_pnl=None，write_daily_summary 回退账户口径。
    if asset_snapshot:
        per_stock_pnl = None
        if report_data:
            _s = report_data.get('summary', {})
            if _s.get('live_pnl_source') == 'per_stock':
                per_stock_pnl = _s.get('live_daily_pnl')
        try:
            live_trade_recorder.write_daily_summary(
                total_asset=float(asset_snapshot.total_asset),
                cash=float(asset_snapshot.cash),
                market_value=float(asset_snapshot.market_value),
                trade_date=trade_date,
                per_stock_pnl=per_stock_pnl,
            )
        except Exception as e:
            trading_logger.warning(f"[PostClose] 日终摘要失败: {e}")
    else:
        trading_logger.warning("[PostClose] 资产快照缺失，跳过日终摘要")

    # 6. 更新战报卡片 P&L 数据并锁定
    try:
        from trading.day_board import day_board, extract_bt_reference
        if day_board._trade_date is None:
            trading_logger.info(
                "[PostClose] 战报会话未初始化（直接从盘后时间启动），跳过战报刷新")
            return report_data
        # 6.1 用盘后回测结果回灌「目标」对账参考：盘前若缺 T-1 种子（sim / 首日），
        #     start_session 时 bt_ref 可能为空，导致目标/操作/diff 三列全为「-」。
        snaps = bt_result.get('daily_snapshots') or []
        day_board.feed_bt_reference(
            extract_bt_reference(bt_result),
            bt_daily_return=snaps[-1].get('daily_return_pct') if snaps else None)
        # 6.2 用 parquet 灌入实盘 成交/持仓（不走 QMT），填齐战报「实盘」侧：
        #     成交 → fills_{T}.parquet；持仓 → positions_{T}.parquet。
        day_board.feed_live_fills(live_fills)
        if live_positions_df is not None:
            day_board.feed_live_positions(live_positions_df)
        if report_data:
            s = report_data.get('summary', {})
            day_board.feed_close_data(
                live_pnl=s.get('live_daily_pnl'),
                live_return=s.get('live_daily_return_pct'))
        day_board.finalize()
    except Exception as e:
        trading_logger.warning(f"[PostClose] 战报 finalize 失败: {e}\n{traceback.format_exc()}")

    trading_logger.info("[PostClose] 完成")
    return report_data


# ================================================================
# 重试工具
# ================================================================

def _retry_with_backoff(func, max_retries=3, base_delay=2.0):
    """指数退避重试，返回 (result, success)。"""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(), True
        except Exception as e:
            last_exc = e
            if attempt == max_retries - 1:
                trading_logger.error(f"[Retry] {getattr(func, '__name__', func)} 全部{max_retries}次失败: {e}")
            else:
                delay = base_delay * (2 ** attempt)
                trading_logger.warning(
                    f"[Retry] {getattr(func, '__name__', func)} "
                    f"第{attempt+1}/{max_retries}次失败: {e}, {delay:.0f}s后重试"
                )
                time.sleep(delay)
    return None, False


# ================================================================
# 16:00 全量更新
# ================================================================

def run_update_all(store):
    """16:00 全量数据更新（带重试+断点续跑），更新完发送飞书通知。"""
    if getattr(store, '_skip_update', False):
        trading_logger.info("[UpdateAll] 跳过全量更新 (--skip 模式)")
        return True

    import pandas as pd
    target_date = store._now().date() if store is not None else date.today()
    from data.update_all import (
        _update_stock_list, _update_stock_name, _update_kline,
        _update_balance, _update_issue_price,
        _update_financial_deep, _update_indices, _update_delist, _update_trading_calendar,
        _build_runtime,
        DATA_DIR, TODAY, YESTERDAY,
    )

    if target_date != date.today():
        trading_logger.info(
            f"[UpdateAll] 历史模拟 {target_date}: 盘后 K 线已更新，仅重建 runtime")
        _, runtime_ok = _retry_with_backoff(_build_runtime)
        ok = runtime_ok
        trading_logger.info(
            f"[UpdateAll] 历史模拟 {target_date} {'完成' if ok else '失败'}")
        return ok

    t0 = time.time()
    trading_logger.info("=" * 50)
    trading_logger.info(f"全量更新 @ {TODAY.isoformat()}")
    trading_logger.info("=" * 50)
    recorder.mark("全量更新")

    # 断点续跑：每步完成后写标记，重启跳过已完成步骤
    CKPT_DIR = DATA_DIR / "runtime" / ".update_checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    today_str = TODAY.isoformat()
    # 清理非今日的旧日标记；今日的 step 标记保留用于断点续跑
    for old in CKPT_DIR.glob("*.done"):
        if old.stem != today_str:
            old.unlink(missing_ok=True)
            for s in CKPT_DIR.glob("step_*"):
                s.unlink(missing_ok=True)

    def _step_done(name: str) -> bool:
        return (CKPT_DIR / f"step_{name.replace('/', '_')}").exists()

    def _mark_step(name: str):
        (CKPT_DIR / f"step_{name.replace('/', '_')}").touch()
        (CKPT_DIR / f"{today_str}.done").touch()

    # 首次启用时（无标记），从产出文件时间戳自动推断已完成的步骤
    def _was_updated_today(glob_pattern: str) -> bool:
        for p in DATA_DIR.glob(glob_pattern):
            from datetime import datetime as _dt
            mtime = _dt.fromtimestamp(p.stat().st_mtime).date()
            if mtime == TODAY:
                return True
        return False

    if not (CKPT_DIR / f"{today_str}.done").exists():
        auto_steps = [
            ("股票列表", lambda: _was_updated_today("stock_list/stock_list.parquet")),
            ("退市列表", lambda: _was_updated_today("delist/delist.parquet")),
            ("股票名称/ST", lambda: _was_updated_today("stock_name/current_names.parquet")),
            ("资产负债表", lambda: _was_updated_today("financial/balance.parquet")),
            ("深历史财务", lambda: _was_updated_today("financial/deep_indicators.parquet")),
            ("发行价", lambda: _was_updated_today("issue_price/issue_price.parquet")),
            ("大盘指数", lambda: _was_updated_today("index_*_daily.parquet")),
        ]
        for auto_name, check in auto_steps:
            try:
                if check():
                    _mark_step(auto_name)
                    trading_logger.info(f"[Checkpoint] 自动补标记: {auto_name} (产出文件今日已更新)")
            except Exception:
                pass

    # Phase 1: 清理最近 N 个交易日的指数不完整数据。
    # K 线无需在此清理：mootdx update_recent 会按 time 合并覆盖最近 N 个交易日。
    if not _step_done("phase1_clean"):
        trading_logger.info("--- Phase 1: 清理指数最近数据 ---")
        for f in DATA_DIR.glob("index_*_daily.parquet"):
            try:
                df = pd.read_parquet(f)
                dates = pd.to_datetime(df['trade_date'])
                mask = dates.dt.date < YESTERDAY
                if mask.sum() < len(df):
                    df[mask].reset_index(drop=True).to_parquet(f, index=False)
            except Exception as e:
                trading_logger.warning(f"[指数清理] {f.name} 失败: {e}")
        _mark_step("phase1_clean")

    # Phase 2: 逐步下载（每个步骤独立重试）
    trading_logger.info("--- Phase 2: 下载更新 ---")

    def _run_update_step(name: str, func) -> tuple[bool, str | None]:
        """同步运行 func，保留独立重试与状态记录。"""
        return _retry_with_backoff(func)

    steps = [
        ("股票列表", _update_stock_list),
        ("退市列表", _update_delist),
        ("股票名称/ST", _update_stock_name),
        ("K线日线", _update_kline),
        ("资产负债表", _update_balance),
        ("深历史财务", _update_financial_deep),
        ("发行价", _update_issue_price),
        ("大盘指数", _update_indices),
        ("交易日历", _update_trading_calendar),
    ]

    steps_ok = True
    for name, func in steps:
        if _step_done(name):
            trading_logger.info(f">>> {name} <<< [跳过: 已完成]")
            continue
        t1 = time.time()
        trading_logger.info(f">>> {name} <<<")
        _, ok = _run_update_step(name, func)
        if ok:
            _mark_step(name)
        else:
            steps_ok = False
        trading_logger.info(f"<<< {name} {'OK' if ok else 'FAIL'} ({time.time()-t1:.0f}s) >>>")

    # Phase 3: 构建 Runtime（不做断点续跑——runtime 必须基于当日前面的全部数据重新构建）
    trading_logger.info("--- Phase 3: 构建 Runtime ---")
    _, runtime_ok = _retry_with_backoff(_build_runtime)
    ok = steps_ok and runtime_ok

    # 全部完成，清理标记
    for m in CKPT_DIR.glob("step_*"):
        m.unlink(missing_ok=True)

    elapsed = time.time() - t0
    npz_files = sorted((DATA_DIR / "runtime").glob("runtime_*.npz"))
    size_info = ""
    if npz_files:
        latest = npz_files[-1]
        size_info = f", NPZ: {latest.name} ({latest.stat().st_size/(1024*1024):.1f}MB)"

    try:
        npz_name = npz_files[-1].name if npz_files else '-'
        npz_size = f"{npz_files[-1].stat().st_size/(1024*1024):.1f} MB" if npz_files else '-'
        rows = [
            {'metric': '⏱ 耗时', 'value': f"{elapsed:.0f} s"},
            {'metric': '📋 步骤', 'value': f"{len(steps)} 项"},
            {'metric': '📦 NPZ', 'value': npz_name},
            {'metric': '💾 大小', 'value': npz_size},
        ]
        lark_sender.send_table_card(
            title=f"{'✅' if ok else '❌'} 全量更新{'完成' if ok else '失败'} @ {TODAY.isoformat()}",
            level=LarkMsgLevel.Success if ok else LarkMsgLevel.Danger,
            tables=[{
                'title': '**📊 更新汇总**',
                'element_id': 'update_summary',
                'columns': [
                    {'name': 'metric', 'display_name': '指标', 'horizontal_align': 'left'},
                    {'name': 'value', 'display_name': '数值', 'horizontal_align': 'right'},
                ],
                'rows': rows,
            }],
        )
    except Exception as e:
        trading_logger.warning(f"[UpdateAll] 飞书失败: {e}")

    trading_logger.info(f"全量更新{'完成' if ok else '失败'}! {elapsed:.0f}s")
    recorder.mark("全量更新完成" if ok else "全量更新失败")
    return ok

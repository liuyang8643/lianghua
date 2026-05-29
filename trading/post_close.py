"""15:00 盘后工作流：拉收盘K线 → 单日回测 → 实盘vs回测对比 → 飞书报告。

同时负责 16:00 update-all（带重试）。
"""
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.db.stock_list import allow_buy_stock_code_list
from trading.lark.sender import LarkMsgLevel, lark_sender
from trading.logger import trading_logger
from utils.recorder import recorder


# ================================================================
# 15:00 盘后K线更新
# ================================================================

def _post_close_update_kline():
    from data.update_live import _download_kline_all, _patch_npz_incremental
    t0 = time.time()
    kline_data = _download_kline_all()
    trading_logger.info(f"[PostClose] K线下载完成 ({len(kline_data)}只, {time.time()-t0:.0f}s)")
    _patch_npz_incremental(kline_data)
    trading_logger.info(f"[PostClose] NPZ增量修补完成 ({time.time()-t0:.0f}s)")
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


def _run_continuous_backtest(start_date: date, end_date: date,
                              individual_config: dict, factor_classes: list) -> dict | None:
    """连续多日回测 [start_date, end_date]，让回测演化形成对齐 T-1 的持仓。
    
    返回 dict 与 _backtest_direct 一致；其中 daily_snapshots[-1] 为 T 日 snapshot。
    PostCloseReport._bt_snap() 默认取 [-1]，_rebuild_backtest_per_stock_pnl
    用 [-1] vs [-2] 算 T 日 daily_pnl。
    """
    from core.backtest import _compute_factor_scores, _backtest_direct
    from utils.stock.time import get_trading_date_span

    trading_days = get_trading_date_span(start_date, end_date)
    if not trading_days:
        trading_logger.warning(f"[PostClose] {start_date} → {end_date} 无交易日")
        return None

    signal_datetimes = [datetime.combine(d, datetime.min.time()) for d in trading_days]
    all_stocks = allow_buy_stock_code_list(target_date=end_date)
    weights = individual_config['weights']
    temperatures = individual_config['temperatures']
    buy_n = individual_config['buy_n']
    sell_m = individual_config.get('sell_m', buy_n)

    scores_result = _compute_factor_scores(
        signal_datetimes, all_stocks, weights, factor_classes)
    if scores_result is None:
        return None

    data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices = scores_result
    if not valid_dates:
        return None

    return _backtest_direct(
        data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
        weights=weights, buy_n=buy_n, sell_m=sell_m, temperatures=temperatures,
        lightweight=False,
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

    # 1. 拉收盘K线 & 增量修补NPZ
    recorder.mark("盘后K线更新")
    if not getattr(store, '_skip_update', False):
        _post_close_update_kline()
    else:
        trading_logger.info("[PostClose] 跳过K线更新 (--skip 模式)")

    # 2. 多日连续回测：从实盘首调仓日 → T 日，让回测演化出 T-1 持仓
    # 这样 T 日的回测换手是"多退少补"，跟实盘对齐口径
    recorder.mark("盘后回测")
    bt_start = _resolve_backtest_start(trade_date)
    if bt_start < trade_date:
        trading_logger.info(f"[PostClose] 连续回测窗口: {bt_start} → {trade_date} "
                            f"(模拟实盘 T-1 持仓基线)")
    else:
        trading_logger.info(f"[PostClose] 单日回测: {trade_date} (实盘尚无历史调仓)")
    bt_result = _run_continuous_backtest(bt_start, trade_date,
                                          individual_config, factor_classes)
    if bt_result is None:
        trading_logger.warning("[PostClose] 回测失败")
        return None
    snaps = bt_result.get('daily_snapshots') or []
    trading_logger.info(f"[PostClose] 回测完成: {len(snaps)}日, 总收益={bt_result.get('total_return', 0):.2f}%")

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
    try:
        asset_snapshot = store.trader.query_asset()
        if asset_snapshot:
            live_asset = float(asset_snapshot.total_asset)
    except Exception as e:
        trading_logger.warning(f"[PostClose] 资产查询失败: {e}")

    # 3.1.1 同步 QMT 银证流水到 cash_flows（自动识别日内入金/出金）
    try:
        live_trade_recorder.sync_bank_transfers_from_qmt(store.trader, trade_date=trade_date)
    except Exception as e:
        trading_logger.warning(f"[PostClose] 银证流水同步失败: {e}")

    # 3.1.2 兜底从 QMT 回填当日成交，弥补 watcher.on_stock_order 漏接
    # 这一步必须在 snapshot_positions 之前，否则 daily_pnl 公式会基于不完整 fills 算错。
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
    try:
        positions = store.trader.query_positions()
    except Exception as e:
        trading_logger.warning(f"[PostClose] 持仓查询失败: {e}")
        positions = []
    if positions:
        live_trade_recorder.snapshot_positions(positions, fills_df=live_fills, trade_date=trade_date)
        pos_path = _TRADE_DIR / f"positions_{trade_date.isoformat()}.parquet"
        if pos_path.exists():
            report.feed_positions_df(pd.read_parquet(pos_path))

    # 3.4 实盘 plan
    plan_path = _TRADE_DIR / f"plan_{trade_date.isoformat()}.parquet"
    if plan_path.exists():
        report.feed_plan_df(pd.read_parquet(plan_path))
    else:
        trading_logger.warning(f"[PostClose] {plan_path.name} 不存在，维度 1/2 数据缺失")

    # 3.5 回测结果
    report.feed_backtest(bt_result)

    # 4. 飞书报告 + HTML
    html_dir = Path(__file__).resolve().parents[1] / "data" / "live_trades" / "reports"
    html_path = html_dir / f"diff_{trade_date.isoformat()}.html"
    report.to_html(html_path)
    report_data = report.send(html_path=html_path)

    # 5. 持久化日终摘要（复用上面已查到的 asset_snapshot，保证与 report 中的账户 P&L 完全一致）
    if asset_snapshot:
        try:
            live_trade_recorder.write_daily_summary(
                total_asset=float(asset_snapshot.total_asset),
                cash=float(asset_snapshot.cash),
                market_value=float(asset_snapshot.market_value),
                trade_date=trade_date,
            )
        except Exception as e:
            trading_logger.warning(f"[PostClose] 日终摘要失败: {e}")
    else:
        trading_logger.warning("[PostClose] 资产快照缺失，跳过日终摘要")

    # 6. 锁定战报卡片（盘后最终状态，不再被新事件污染）
    try:
        from trading.day_board import day_board
        day_board.finalize()
    except Exception as e:
        trading_logger.warning(f"[PostClose] 战报 finalize 失败: {e}")

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
    """16:00 全量数据更新（带重试），更新完发送飞书通知。"""
    if getattr(store, '_skip_update', False):
        trading_logger.info("[UpdateAll] 跳过全量更新 (--skip 模式)")
        return True

    import pandas as pd
    from data.update_all import (
        _update_stock_list, _update_stock_name, _update_kline,
        _update_balance, _update_issue_price,
        _update_indices, _update_delist, _update_trading_calendar,
        _build_runtime, DATA_DIR, TODAY, YESTERDAY,
    )

    t0 = time.time()
    trading_logger.info("=" * 50)
    trading_logger.info(f"全量更新 @ {TODAY.isoformat()}")
    trading_logger.info("=" * 50)
    recorder.mark("全量更新")

    # Phase 1: 清理昨日数据
    trading_logger.info("--- Phase 1: 清理昨日数据 ---")
    kline_dir = DATA_DIR / "k-line"
    if kline_dir.exists():
        cutoff_ms = int(pd.Timestamp(YESTERDAY).timestamp() * 1000)
        for f in kline_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(f)
                before = len(df)
                df_clean = df[df['time'] < cutoff_ms]
                if len(df_clean) < before:
                    df_clean.to_parquet(f, index=False)
            except Exception:
                pass

    for f in DATA_DIR.glob("index_*_daily.parquet"):
        try:
            df = pd.read_parquet(f)
            dates = pd.to_datetime(df['trade_date'])
            mask = dates.dt.date < YESTERDAY
            if mask.sum() < len(df):
                df[mask].reset_index(drop=True).to_parquet(f, index=False)
        except Exception as e:
            trading_logger.warning(f"[指数清理] {f.name} 失败: {e}")

    # Phase 2: 逐步下载（每个步骤独立重试）
    trading_logger.info("--- Phase 2: 下载更新 ---")
    steps = [
        ("股票列表", _update_stock_list),
        ("股票名称/ST", _update_stock_name),
        ("K线日线", _update_kline),
        ("资产负债表", _update_balance),
        ("发行价", _update_issue_price),
        ("大盘指数", _update_indices),
        ("退市列表", _update_delist),
        ("交易日历", _update_trading_calendar),
    ]

    for name, func in steps:
        t1 = time.time()
        trading_logger.info(f">>> {name} <<<")
        _, ok = _retry_with_backoff(func)
        trading_logger.info(f"<<< {name} {'OK' if ok else 'FAIL'} ({time.time()-t1:.0f}s) >>>")

    # Phase 3: 构建 Runtime
    trading_logger.info("--- Phase 3: 构建 Runtime ---")
    _, ok = _retry_with_backoff(_build_runtime)

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
            title=f"✅ 全量更新完成 @ {TODAY.isoformat()}",
            level=LarkMsgLevel.Success,
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

    trading_logger.info(f"全量更新完成! {elapsed:.0f}s")
    recorder.mark("全量更新完成")
    return ok

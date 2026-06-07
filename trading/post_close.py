"""15:00 盘后工作流：拉收盘K线 → 单日回测 → 实盘vs回测对比 → 飞书报告。

同时负责 16:00 update-all（带重试）。
"""
import time
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


def _seed_replay_core(*, prev_date: date, seed_cash: float, seed_positions: dict,
                      y_positions_eod: list, data, all_scores, valid_dates, date_indices,
                      valid_stocks, stock_indices, individual_config: dict) -> dict | None:
    """用 T-1 种子 + 已算好的因子分数跑单日回放，并合成 [y_snap, t_snap]。

    盘前（before_trade，复用实时分数）与盘后（run_post_close，重算分数）共用此核心，
    使「开盘 diff」与「收盘 diff」的回测口径完全一致。
    """
    from core.backtest import _backtest_direct

    weights = individual_config['weights']
    temperatures = individual_config['temperatures']
    buy_n = individual_config['buy_n']
    sell_m = individual_config.get('sell_m', buy_n)

    # 只保留 NPZ 覆盖到的种子持仓（其余无法估值/交易）。
    seed_in_npz = {c: v for c, v in seed_positions.items() if c in stock_indices}
    dropped = sorted(set(seed_positions) - set(seed_in_npz))
    if dropped:
        trading_logger.warning(
            f"[SeedReplay] {len(dropped)} 只种子持仓不在 NPZ，回放忽略: {dropped[:5]}")

    bt_result = _backtest_direct(
        data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
        weights=weights, buy_n=buy_n, sell_m=sell_m, temperatures=temperatures,
        init_cash=seed_cash, init_positions=seed_in_npz,
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
                     factor_classes: list) -> dict | None:
    """盘后单日回放：用实盘 T-1 真实持仓 + 现金做种子，只回放 T 日多退少补。

    目的：让回测端 T 日的持仓/下单手数与实盘可比 —— 两边起点（资金+持仓）完全相同，
    手数差只来自执行层（废单/部成/资金不足/min_lot），价格差只来自滑点。
    无 T-1 种子（首个交易日 / 缺快照）时返回 None，调用方回退连续回测。
    """
    from core.backtest import _compute_factor_scores

    loaded = _load_seed(trade_date)
    if loaded is None:
        trading_logger.info("[PostClose] 缺 T-1 种子(positions / daily_summary)，回退连续回测")
        return None
    prev_date, seed_cash, seed_positions, y_positions_eod = loaded

    all_stocks = allow_buy_stock_code_list(target_date=trade_date)
    weights = individual_config['weights']

    signal_dt = [datetime.combine(trade_date, datetime.min.time())]
    scores_result = _compute_factor_scores(signal_dt, all_stocks, weights, factor_classes)
    if scores_result is None:
        return None
    data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices = scores_result
    if not valid_dates:
        return None

    bt_result = _seed_replay_core(
        prev_date=prev_date, seed_cash=seed_cash, seed_positions=seed_positions,
        y_positions_eod=y_positions_eod, data=data, all_scores=all_scores,
        valid_dates=valid_dates, date_indices=date_indices,
        valid_stocks=valid_stocks, stock_indices=stock_indices,
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
                             data, all_scores, date_idx: int,
                             valid_stocks, stock_indices) -> dict | None:
    """盘前/盘中：复用 before_trade 已算好的因子分数，跑 T-1 种子单日回放。

    供战报做「回测 vs 实盘」实时对账。回测端继承 T-1 实盘现金+持仓，与盘后口径一致；
    因子分数直接复用（不重算），几乎不增加盘前耗时。无 T-1 种子时返回 None（首日/缺快照）。
    """
    loaded = _load_seed(trade_date)
    if loaded is None:
        trading_logger.info("[SeedReplay@open] 缺 T-1 种子，跳过盘中回测对账")
        return None
    prev_date, seed_cash, seed_positions, y_positions_eod = loaded

    valid_dates = [datetime.combine(trade_date, datetime.min.time())]
    bt_result = _seed_replay_core(
        prev_date=prev_date, seed_cash=seed_cash, seed_positions=seed_positions,
        y_positions_eod=y_positions_eod, data=data, all_scores=all_scores,
        valid_dates=valid_dates, date_indices=[date_idx],
        valid_stocks=valid_stocks, stock_indices=stock_indices,
        individual_config=individual_config)
    if bt_result is None:
        return None
    t_snap = bt_result['daily_snapshots'][-1]
    trading_logger.info(
        f"[SeedReplay@open] 完成 (种子 {prev_date} 现金¥{seed_cash:,.0f} + "
        f"{len(seed_positions)}只持仓) · T日回测收益={t_snap['daily_return_pct']:+.2f}%")
    return bt_result


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

    # 1. 拉收盘 K 线 → parquet only（NPZ 等 16:00 update_all）
    recorder.mark("盘后K线更新")
    if not getattr(store, '_skip_update', False):
        _post_close_update_kline(trade_date)
    else:
        trading_logger.info("[PostClose] 跳过K线更新 (--skip 模式)")

    # 2. 回测端：优先「单日回放」——用实盘 T-1 真实持仓+现金做种子，只回放 T 日多退少补，
    #    使持仓/下单手数与实盘可比（手数差归因执行层、价格差归因滑点）。
    #    无 T-1 种子（首个交易日 / 缺快照）时回退到多日连续回测。
    recorder.mark("盘后回测")
    bt_result = _run_seed_replay(trade_date, individual_config, factor_classes)
    if bt_result is None:
        bt_start = _resolve_backtest_start(trade_date)
        if bt_start < trade_date:
            trading_logger.info(f"[PostClose] 回退连续回测窗口: {bt_start} → {trade_date}")
        else:
            trading_logger.info(f"[PostClose] 回退单日回测: {trade_date} (实盘尚无历史调仓)")
        bt_result = _run_continuous_backtest(bt_start, trade_date,
                                              individual_config, factor_classes)
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

    _skip = getattr(store, '_skip_update', False)

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
        from trading.day_board import day_board
        if report_data:
            s = report_data.get('summary', {})
            day_board.feed_close_data(
                live_pnl=s.get('live_daily_pnl'),
                live_return=s.get('live_daily_return_pct'))
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
        _build_runtime,
        DATA_DIR, TODAY, YESTERDAY,
    )

    t0 = time.time()
    trading_logger.info("=" * 50)
    trading_logger.info(f"全量更新 @ {TODAY.isoformat()}")
    trading_logger.info("=" * 50)
    recorder.mark("全量更新")

    # Phase 1: 清理最近 N 个交易日的指数不完整数据。
    # K 线无需在此清理：QMT _update_kline(update_recent) 会按 time 合并覆盖最近 N 个交易日。
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

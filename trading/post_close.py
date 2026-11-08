"""15:00 盘后工作流：拉收盘K线 → 单日回测 → 实盘vs回测对比 → 飞书报告。

同时负责 16:00 update-all（带重试）。
"""
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from trading.logger import trading_logger
from utils.recorder import recorder


# ================================================================
# 15:00 盘后K线更新
# ================================================================

def _post_close_update_kline(trade_date: date | None = None, *, allow_qmt=True):
    """15:00 收盘 K 线 → 只写 parquet，不落盘 NPZ（16:00 update_all 再全量写）。"""
    from data.update_live import _download_kline_all
    t0 = time.time()
    if allow_qmt:
        kline_data = _download_kline_all(anchor_date=trade_date)
    else:
        from data.update_kline import update_recent
        kline_data = update_recent(1, anchor_date=trade_date, collect=True, allow_qmt=False)
    trading_logger.info(
        f"[PostClose] K线→parquet ({len(kline_data)}只, {time.time()-t0:.0f}s, 锚定={trade_date})"
    )
    return kline_data


# ================================================================
# 多日连续回测：从实盘首调仓日 → T 日，模拟实盘演化形成对齐的 T-1 持仓
# ================================================================

def _capture_live_close_facts(store, trade_date: date, *, skip_account: bool) -> dict:
    """Capture broker facts before any slow or fallible analytics work."""
    from trading.persistence import POSITION_COLS, live_trade_recorder, _TRADE_DIR

    capture_errors = []
    asset_snapshot = None
    live_asset = None
    if skip_account:
        trading_logger.info("[PostClose] 回放模式跳过 QMT 账户采集")
    else:
        try:
            asset_snapshot = store.trader.query_asset()
            if asset_snapshot:
                live_asset = float(asset_snapshot.total_asset)
            else:
                capture_errors.append("资产快照为空")
        except Exception as e:
            trading_logger.warning(f"[PostClose] 资产查询失败: {e}")
            capture_errors.append(f"资产查询: {e}")

        try:
            live_trade_recorder.sync_bank_transfers_from_qmt(
                store.trader, trade_date=trade_date)
        except Exception as e:
            trading_logger.warning(f"[PostClose] 银证流水同步失败: {e}")
            capture_errors.append(f"银证流水: {e}")

        try:
            backfilled = live_trade_recorder.backfill_fills_from_qmt(
                store.trader, trade_date=trade_date)
            if backfilled is None:
                capture_errors.append("成交查询未成功")
        except Exception as e:
            trading_logger.warning(f"[PostClose] QMT 成交回填失败: {e}")
            capture_errors.append(f"成交回填: {e}")

    live_fills = live_trade_recorder.get_today_fills_df(trade_date=trade_date)

    if not skip_account:
        try:
            positions = store.trader.query_positions()
            # Trader returns [] on query failure. Only accept an empty list as
            # authoritative when the asset snapshot also says this is an
            # all-cash account; otherwise preserve the existing snapshot.
            empty_account = (
                asset_snapshot is not None
                and float(getattr(asset_snapshot, 'market_value', 0) or 0) <= 0
            )
            if positions or empty_account:
                live_trade_recorder.snapshot_positions(
                    positions, fills_df=live_fills, trade_date=trade_date)
            elif not positions:
                trading_logger.warning(
                    "[PostClose] 持仓返回空/None但账户仍有市值，拒绝覆盖当日持仓快照")
                capture_errors.append("持仓为空但账户仍有市值")
        except Exception as e:
            trading_logger.warning(f"[PostClose] 持仓查询/快照失败: {e}")
            capture_errors.append(f"持仓快照: {e}")

    pos_path = _TRADE_DIR / f"positions_{trade_date.isoformat()}.parquet"
    if (
        not skip_account
        and asset_snapshot is not None
        and float(getattr(asset_snapshot, 'market_value', 0) or 0) <= 0
        and not pos_path.exists()
    ):
        # snapshot_positions 没有任何当日/昨日持仓行时不会建文件。
        # 但对全现金账户，空 parquet 本身就是权威快照，次日回放不能把
        # “确实空仓”误判为“快照缺失”。
        pd.DataFrame(columns=POSITION_COLS).to_parquet(pos_path, index=False)
    live_positions_df = pd.read_parquet(pos_path) if pos_path.exists() else None

    # Persist the account baseline immediately. A later report may overwrite
    # the same date with reconciled per-stock P&L, but analytics failure must
    # never lose the broker account snapshot.
    summary_written = False
    if asset_snapshot is not None:
        try:
            live_trade_recorder.write_daily_summary(
                total_asset=float(asset_snapshot.total_asset),
                cash=float(asset_snapshot.cash),
                market_value=float(asset_snapshot.market_value),
                trade_date=trade_date,
            )
            summary_written = True
        except Exception as e:
            trading_logger.warning(f"[PostClose] 日终账户快照失败: {e}")
            capture_errors.append(f"日终账户快照: {e}")

    trading_logger.info(
        f"[PostClose] 券商事实已采集: asset={'ok' if asset_snapshot else 'missing'}, "
        f"fills={len(live_fills)}, positions="
        f"{len(live_positions_df) if live_positions_df is not None else 'missing'}"
    )
    return {
        'trade_dir': _TRADE_DIR,
        'recorder': live_trade_recorder,
        'asset_snapshot': asset_snapshot,
        'live_asset': live_asset,
        'live_fills': live_fills,
        'live_positions_df': live_positions_df,
        'summary_written': summary_written,
        'ok': skip_account or not capture_errors,
        'errors': capture_errors,
    }


def run_post_close(store) -> dict | None:
    """AL-5 盘后 5 维 Diff 调度。

    时序：
      1. 立即从 QMT 采集资产/成交/持仓并落盘
      2. 拉收盘 K 线 + 运行回测
      3. 组装 PostCloseReport 5 维 diff
      4. 飞书卡片 + HTML 报告
      5. 用对账后的个股盈亏补写 daily_summary
    """
    trade_date = store._now().date()
    trading_logger.info(f"[PostClose] 开始 · {trade_date.isoformat()}")

    _skip = getattr(store, '_skip_update', False)
    _skip_account = _skip or getattr(store, '_historical_replay', False)
    facts = _capture_live_close_facts(
        store, trade_date, skip_account=_skip_account)
    store._close_facts_ready = facts['ok']
    live_trade_recorder = facts['recorder']
    _TRADE_DIR = facts['trade_dir']
    asset_snapshot = facts['asset_snapshot']
    live_asset = facts['live_asset']
    live_fills = facts['live_fills']
    live_positions_df = facts['live_positions_df']
    if not facts['ok']:
        trading_logger.error(
            f"[PostClose] 券商事实未完整落盘，留待重试: {'; '.join(facts['errors'])}")
        return None

    # 1. 拉收盘 K 线 → parquet only（NPZ 等 16:00 update_all）
    recorder.mark("盘后K线更新")
    kline_data = None
    if not _skip:
        try:
            if getattr(store, '_file_mode', False):
                kline_data = _post_close_update_kline(trade_date, allow_qmt=False)
            else:
                kline_data = _post_close_update_kline(trade_date)
        except Exception:
            if getattr(store, '_file_mode', False):
                raise  # Never publish a final report after an unverified data update.
            trading_logger.exception("[PostClose] K线更新失败，继续使用已有 NPZ 回测")

    if hasattr(store, '_prepare_close_valuation'):
        store._prepare_close_valuation()
        facts = _capture_live_close_facts(store, trade_date, skip_account=False)
        if not facts['ok']:
            raise RuntimeError('Validated closing facts could not be persisted')
        live_positions_df = facts['live_positions_df']
        live_fills = facts['live_fills']
        asset_snapshot = facts['asset_snapshot']
        live_asset = facts['live_asset']

    recorder.mark("冻结计划零磨损回放")
    bt_result = store._frozen_reference()

    recorder.mark("盘后对比")
    from trading.report import PostCloseReport

    report = PostCloseReport(trade_date)
    if hasattr(store, '_valuation_proof'):
        report._valuation_proof = store._valuation_proof

    # 3.1 实盘资产
    prev_asset = None
    summary_path = _TRADE_DIR / "daily_summary.parquet"
    if summary_path.exists():
        prev = pd.read_parquet(summary_path)
        prev_rows = prev[prev['date'] < trade_date]
        if not prev_rows.empty:
            prev_asset = float(prev_rows['total_asset'].iloc[-1])

    report.feed_asset(
        live_asset, prev_asset,
        net_cash_flow=live_trade_recorder.get_today_cash_flows(trade_date=trade_date))

    # 3.2 实盘 fills / positions：复用开头已落盘的权威快照。
    fills_verified = not _skip_account or (_TRADE_DIR / f'fills_{trade_date.isoformat()}.parquet').exists()
    report.feed_fills_df(live_fills if fills_verified else None)
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
    from trading.execution_reconciliation import load_execution_reference
    previous, execution_targets, close_prices = load_execution_reference(
        trade_date, _TRADE_DIR, Path(__file__).resolve().parents[1] / 'data' / 'k-line')
    report.feed_execution_reference(previous, execution_targets, close_prices)

    # 4. HTML 报告（飞书卡片统一由 day_board 发送）
    html_dir = Path(__file__).resolve().parents[1] / "data" / "live_trades" / "reports"
    html_path = html_dir / f"diff_{trade_date.isoformat()}.html"
    report.to_html(html_path)
    report_data = (report.build() if getattr(store, '_local_reports_only', False)
                   else report.send(html_path=html_path))
    if getattr(store, '_file_mode', False):
        store._post_close_report = report

    # 5. 持久化日终摘要（复用上面已查到的 asset_snapshot，保证与 report 中的账户 P&L 完全一致）
    #    daily_pnl 以报告的「个股盈亏总和」口径为准（免疫未记账出入金），与飞书/HTML 一致；
    #    个股口径不可信（有漏记成交）时 per_stock_pnl=None，write_daily_summary 回退账户口径。
    per_stock_pnl = None
    if report_data:
        _s = report_data.get('summary', {})
        if _s.get('live_pnl_source') == 'per_stock':
            per_stock_pnl = _s.get('live_daily_pnl')
    if asset_snapshot and (per_stock_pnl is not None or not facts['summary_written']):
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
    elif not asset_snapshot:
        trading_logger.warning("[PostClose] 资产快照缺失，跳过日终摘要")

    # 6. 更新战报卡片 P&L 数据并锁定
    try:
        from trading.day_board import day_board, extract_bt_reference
        if day_board.is_active_for(trade_date):
            # 6.1 用盘后回测结果回灌「目标」对账参考。
            snaps = bt_result.get('daily_snapshots') or []
            day_board.feed_bt_reference(
                extract_bt_reference(bt_result),
                bt_daily_return=snaps[-1].get('daily_return_pct') if snaps else None)
            execution = report_data.get('execution') if report_data else None
            if execution and execution['available']:
                reference = {'buy': {}, 'sell': {}, 'positions': {
                    row['code']: row['bt_volume'] for row in execution['rows'] if row['bt_volume'] > 0}}
                for row in execution_targets.to_dict('records'):
                    if int(row['est_volume']) <= 0:
                        continue
                    slot = reference[row['direction']].setdefault(row['code'], {'shares': 0, 'amount': 0.0})
                    slot['shares'] += int(row['est_volume'])
                    slot['amount'] += int(row['est_volume']) * float(row['est_price'])
                day_board.feed_bt_reference(
                    reference, bt_daily_return=report_data['summary']['bt_daily_return_pct'],
                    comparison_basis='同计划零磨损')
            # 6.2 用 parquet 灌入实盘成交/持仓并定稿当天战报。
            day_board.feed_live_fills(live_fills)
            if live_positions_df is not None:
                day_board.feed_live_positions(live_positions_df)
            if report_data:
                s = report_data.get('summary', {})
                day_board.feed_close_data(
                    live_pnl=s.get('live_daily_pnl'),
                    live_return=s.get('live_daily_return_pct'))
            day_board.finalize()
        else:
            trading_logger.debug("[PostClose] 无对应实盘战报会话，跳过 finalize")
    except Exception as e:
        trading_logger.warning(f"[PostClose] 战报 finalize 失败: {e}")
        trading_logger.debug(traceback.format_exc())

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
            result = func()
            if result is False:
                raise RuntimeError(
                    f"{getattr(func, '__name__', func)} 显式返回 False")
            return result, True
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

def _expected_runtime_end(data_dir: Path, run_date: date) -> date:
    """Resolve the last exchange day that a Runtime for ``run_date`` must contain."""
    calendar_path = data_dir / "trading_calendar.parquet"
    if not calendar_path.exists():
        raise FileNotFoundError(f"交易日历不存在: {calendar_path}")
    calendar = pd.read_parquet(calendar_path)
    if 'trade_date' not in calendar.columns:
        raise RuntimeError(f"交易日历缺少 trade_date: {calendar_path}")
    dates = pd.to_datetime(calendar['trade_date'], errors='coerce').dropna()
    eligible = dates[dates.dt.date <= run_date]
    if eligible.empty:
        raise RuntimeError(f"交易日历无 {run_date} 及之前的日期")
    return eligible.max().date()


def _validate_runtime_date(runtime_path: Path, expected_end: date) -> Path:
    """Reject a corrupt or stale Runtime instead of publishing a false checkpoint."""
    import numpy as np

    runtime_path = Path(runtime_path)
    if not runtime_path.exists():
        raise FileNotFoundError(f"Runtime 不存在: {runtime_path}")
    with np.load(runtime_path, allow_pickle=False) as runtime:
        if 'trade_dates' not in runtime.files:
            raise RuntimeError(f"Runtime 缺少 trade_dates: {runtime_path.name}")
        trade_dates = np.asarray(runtime['trade_dates']).astype('datetime64[D]')
        if trade_dates.ndim != 1 or len(trade_dates) == 0:
            raise RuntimeError(f"Runtime trade_dates 为空: {runtime_path.name}")
        if len(trade_dates) > 1 and np.any(trade_dates[1:] < trade_dates[:-1]):
            raise RuntimeError(f"Runtime trade_dates 未排序: {runtime_path.name}")
        actual_end = pd.Timestamp(trade_dates[-1]).date()
    if actual_end != expected_end:
        raise RuntimeError(
            f"Runtime 日期不完整: actual={actual_end}, expected={expected_end}"
        )
    return runtime_path


def _write_final_checkpoint(path: Path, runtime_path: Path):
    """Atomically publish the final marker only after Runtime validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(runtime_path.name, encoding='utf-8')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run_update_all(store, *, history_end=None, history_days=3, allow_qmt=True):
    """16:00 全量数据更新：按日期断点续跑，且只在 Runtime 验证后完成。"""
    if getattr(store, '_skip_update', False):
        trading_logger.info("[UpdateAll] 跳过全量更新 (--skip 模式)")
        return True

    from data.update_all import (
        _update_stock_list, _update_stock_name, _update_kline,
        _update_balance, _update_issue_price,
        _update_financial_deep, _update_indices, _update_delist,
        _update_trading_calendar, _build_runtime,
        current_update_date, DATA_DIR,
    )

    run_date = current_update_date()
    yesterday = run_date - timedelta(days=1)
    t0 = time.time()
    trading_logger.info("=" * 50)
    trading_logger.info(f"全量更新 @ {run_date.isoformat()}")
    trading_logger.info("=" * 50)
    recorder.mark("全量更新")

    # 每个交易日使用独立目录，不删除上一次已成功步骤。这样失败后
    # 只重试失败步，也不会把前一天的 marker 误用到新交易日。
    checkpoint_root = DATA_DIR / "runtime" / ".update_checkpoints"
    run_checkpoint = checkpoint_root / run_date.isoformat()
    if history_end is not None:
        run_checkpoint = run_checkpoint / f"startup_{history_end.isoformat()}"
    if not allow_qmt:
        run_checkpoint = run_checkpoint / 'file_planner_no_qmt'
    run_checkpoint.mkdir(parents=True, exist_ok=True)
    final_marker = run_checkpoint / "final.done"

    # 已完成的当日任务仍重新验证 Runtime；文件被破坏/替换时不会永久跳过。
    if final_marker.exists():
        try:
            runtime_name = final_marker.read_text(encoding='utf-8').strip()
            runtime_path = DATA_DIR / "runtime" / runtime_name
            expected_end = history_end or _expected_runtime_end(DATA_DIR, run_date)
            _validate_runtime_date(runtime_path, expected_end)
            if history_end is not None:
                from trading.startup_data import validate_startup_runtime
                validate_startup_runtime(history_end, runtime_path)
            trading_logger.info(
                f"[Checkpoint] {run_date} 全量更新已完成，Runtime 验证通过")
            return True
        except Exception as e:
            trading_logger.warning(f"[Checkpoint] final 失效，重建 Runtime: {e}")
            final_marker.unlink(missing_ok=True)

    def _step_marker(key: str) -> Path:
        return run_checkpoint / f"step_{key}.done"

    def _run_step(key: str, label: str, func) -> bool:
        marker = _step_marker(key)
        if marker.exists():
            trading_logger.info(f">>> {label} <<< [跳过: 当日已完成]")
            return True
        t1 = time.time()
        trading_logger.info(f">>> {label} <<<")
        _, ok = _retry_with_backoff(func)
        if ok:
            marker.touch()
        trading_logger.info(f"<<< {label} {'OK' if ok else 'FAIL'} ({time.time()-t1:.0f}s) >>>")
        return ok

    def _clean_indices():
        failures = []
        for parquet_path in DATA_DIR.glob("index_*_daily.parquet"):
            try:
                df = pd.read_parquet(parquet_path)
                dates = pd.to_datetime(df['trade_date'])
                mask = dates.dt.date < yesterday
                if mask.sum() < len(df):
                    df[mask].reset_index(drop=True).to_parquet(parquet_path, index=False)
            except Exception as e:
                failures.append(f"{parquet_path.name}: {e}")
        if failures:
            raise RuntimeError("; ".join(failures))

    trading_logger.info("--- Phase 1: 清理指数最近数据 ---")
    steps_ok = _run_step("phase1_clean", "指数清理", _clean_indices)

    trading_logger.info("--- Phase 2: 下载更新 ---")
    steps = [
        ("stock_list", "股票列表", _update_stock_list, ()),
        ("delist", "退市列表", _update_delist, ()),
        ("stock_name", "股票名称/ST", _update_stock_name, ("stock_list",)),
        ("kline", "K线日线", _update_kline, ("stock_list", "delist")),
        ("balance", "资产负债表", _update_balance, ("stock_list",)),
        ("financial_deep", "深历史财务", _update_financial_deep, ()),
        ("issue_price", "发行价", _update_issue_price, ("kline",)),
        ("indices", "大盘指数", _update_indices, ()),
        ("trading_calendar", "交易日历", _update_trading_calendar, ()),
    ]
    if history_end is not None:
        replacements = {
            'kline': lambda: _update_kline(anchor_date=history_end, days=history_days),
            'indices': lambda: _update_indices(end_date=history_end),
            'stock_name': lambda: _update_stock_name(include_history=False),
        }
        steps = [('stock_status' if key == 'stock_name' else key,
                  label, replacements.get(key, func), deps)
                 for key, label, func, deps in steps]
    if not allow_qmt:
        replacements = {
            'stock_list': lambda: _update_stock_list(source='mootdx'),
            'kline': lambda: _update_kline(anchor_date=history_end, days=history_days, allow_qmt=False),
        }
        steps = [(key, label, replacements.get(key, func), deps)
                 for key, label, func, deps in steps]
    failed_steps = []
    step_results = {}
    if not steps_ok:
        failed_steps.append("phase1_clean")
    for key, label, func, dependencies in steps:
        blocked_by = [dep for dep in dependencies if not step_results.get(dep, False)]
        if blocked_by:
            # 上游本日失败时，下游不得用旧文件运行并写成功标记。
            _step_marker(key).unlink(missing_ok=True)
            trading_logger.warning(
                f">>> {label} <<< [跳过: 上游失败 {', '.join(blocked_by)}]")
            step_results[key] = False
        else:
            step_results[key] = _run_step(key, label, func)
        if not step_results[key]:
            steps_ok = False
            failed_steps.append(key)

    # 任一下载失败时严禁用新旧混合数据构建 Runtime。
    runtime_path = None
    runtime_ok = False
    if steps_ok:
        trading_logger.info("--- Phase 3: 构建 Runtime ---")

        def _build_and_validate():
            expected_end = history_end or _expected_runtime_end(DATA_DIR, run_date)
            built = (_build_runtime(end_date=history_end)
                     if history_end is not None else _build_runtime())
            if built is None:
                raise RuntimeError("build_runtime 未返回输出路径")
            validated = _validate_runtime_date(Path(built), expected_end)
            if history_end is not None:
                from trading.startup_data import validate_startup_runtime
                validate_startup_runtime(history_end, validated)
            return validated

        # Runtime 构建很重，单次失败留给 scheduler 五分钟后断点重试。
        runtime_path, runtime_ok = _retry_with_backoff(_build_and_validate, max_retries=1)
        if runtime_ok:
            _write_final_checkpoint(final_marker, runtime_path)
        else:
            # Runtime 日期不完整通常表明 K 线源尚未到收盘日；其他成功步保留。
            _step_marker("kline").unlink(missing_ok=True)
            _step_marker("issue_price").unlink(missing_ok=True)
    else:
        trading_logger.error(
            f"[UpdateAll] 下载步骤失败 ({', '.join(failed_steps)})，跳过 Runtime 构建")

    ok = steps_ok and runtime_ok
    elapsed = time.time() - t0
    trading_logger.info(f"全量更新{'完成' if ok else '失败'}! {elapsed:.0f}s")
    recorder.mark("全量更新完成" if ok else "全量更新失败")
    return ok

"""15:00 盘后工作流：拉收盘K线 → 单日回测 → 实盘vs回测对比 → 飞书报告。

同时负责 16:00 update-all（带重试）。
"""
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data.db.stock_list import allow_buy_stock_code_list
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


def _ensure_no_filter_factors(individual_config: dict):
    if individual_config.get('filter_factors'):
        raise ValueError("filter_factors 目前仅支持研究回测，盘后回放路径未启用，禁止静默忽略")


def _seed_replay_core(*, prev_date: date, seed_cash: float, seed_positions: dict,
                      y_positions_eod: list, data, all_scores, valid_dates, date_indices,
                      valid_stocks, stock_indices, individual_config: dict) -> dict | None:
    """用 T-1 种子 + 已算好的因子分数跑单日回放，并合成 [y_snap, t_snap]。

    盘前（before_trade，复用实时分数）与盘后（run_post_close，重算分数）共用此核心，
    使「开盘 diff」与「收盘 diff」的回测口径完全一致。
    """
    from core.backtest import _backtest_direct

    weights = individual_config['weights']
    buy_n = individual_config['buy_n']
    sell_m = individual_config.get('sell_m', buy_n)

    # 只保留 NPZ 覆盖到的种子持仓（其余无法估值/交易）。
    seed_in_npz = {c: v for c, v in seed_positions.items() if c in stock_indices}
    dropped = sorted(set(seed_positions) - set(seed_in_npz))
    if dropped:
        trading_logger.warning(
            f"[SeedReplay] {len(dropped)} 只种子持仓不在 NPZ，回放忽略: {dropped[:5]}")

    import numpy as np
    empty_months = individual_config.get('empty_months')
    cash_reserve = float(individual_config.get('cash_reserve_ratio', 0.0))
    invest_ratio = 1.0 - cash_reserve
    position_multipliers = None
    t_date = valid_dates[0].date() if valid_dates else None
    if t_date and empty_months and t_date.month in empty_months:
        position_multipliers = np.array([0.0])
    elif cash_reserve > 0:
        position_multipliers = np.array([invest_ratio])

    bt_result = _backtest_direct(
        data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
        weights=weights, buy_n=buy_n, sell_m=sell_m,
        init_cash=seed_cash, init_positions=seed_in_npz,
        position_multipliers=position_multipliers,
        limit_up_protection=individual_config.get('limit_up_protection', False),
        rebalance=individual_config.get('rebalance', True),
        prefilter_n=individual_config.get('prefilter_n'),
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
    from core.backtest import _compute_factor_scores

    _ensure_no_filter_factors(individual_config)
    loaded = _load_seed(trade_date)
    if loaded is None:
        trading_logger.info("[PostClose] 缺 T-1 种子(positions / daily_summary)，回退连续回测")
        return None
    prev_date, seed_cash, seed_positions, y_positions_eod = loaded

    all_stocks = allow_buy_stock_code_list(target_date=trade_date)
    weights = individual_config['weights']

    signal_dt = [datetime.combine(trade_date, datetime.min.time())]
    scores_result = _compute_factor_scores(signal_dt, all_stocks, weights, factor_classes,
                                            kline_data=kline_data)
    if scores_result is None:
        return None
    data, all_scores, _, valid_dates, date_indices, valid_stocks, stock_indices = scores_result
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
        f"[PostClose] 单日回放完成 (种子: {prev_date} 现金={seed_cash:,.0f} + "
        f"{len(seed_positions)}只持仓, 权益={y_total:,.0f}) · T日回测收益={t_snap['daily_return_pct']:+.2f}%"
    )
    return bt_result


def run_seed_replay_for_open(trade_date: date, individual_config: dict, *,
                             data, all_scores, date_idx: int,
                             valid_stocks, stock_indices) -> dict | None:
    """盘前/盘中：复用 before_trade 已算好的因子分数，跑 T-1 种子单日回放。

    供战报做「回测 vs 实盘」实时对账。回测端继承 T-1 实盘现金+持仓，与盘后口径一致；
    因子分数直接复用（不重算），几乎不增加盘前耗时。无 T-1 种子时返回 None（首日/缺快照）。
    """
    _ensure_no_filter_factors(individual_config)
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
        f"[SeedReplay@open] 完成 (种子 {prev_date} 现金={seed_cash:,.0f} + "
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
    from core.backtest import _compute_factor_scores, _backtest_direct
    from core.runtime import load_runtime_stock_codes
    from utils.stock.time import get_trading_date_span

    _ensure_no_filter_factors(individual_config)
    trading_days = get_trading_date_span(start_date, end_date)
    if not trading_days:
        trading_logger.warning(f"[PostClose] {start_date} → {end_date} 无交易日")
        return None

    signal_datetimes = [datetime.combine(d, datetime.min.time()) for d in trading_days]
    all_stocks = load_runtime_stock_codes()
    weights = individual_config['weights']
    buy_n = individual_config['buy_n']
    sell_m = individual_config.get('sell_m', buy_n)

    scores_result = _compute_factor_scores(
        signal_datetimes, all_stocks, weights, factor_classes,
        kline_data=kline_data)
    if scores_result is None:
        return None

    data, all_scores, _, valid_dates, date_indices, valid_stocks, stock_indices = scores_result
    if not valid_dates:
        return None

    seed_in_npz = ({c: v for c, v in init_positions.items() if c in stock_indices}
                   if init_positions else None)
    return _backtest_direct(
        data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
        weights=weights, buy_n=buy_n, sell_m=sell_m,
        lightweight=False, init_cash=init_cash, init_positions=seed_in_npz,
        limit_up_protection=individual_config.get('limit_up_protection', False),
        rebalance=individual_config.get('rebalance', True),
        prefilter_n=individual_config.get('prefilter_n'),
    )


# ================================================================
# 主入口
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

    individual_config = getattr(store, '_individual_config', None)
    factor_classes = getattr(store, '_factor_classes', None)
    if individual_config is None or factor_classes is None:
        trading_logger.error("[PostClose] 缺少 config/factor_classes, 跳过")
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

    # 2. 回测端：优先「单日回放」——用实盘 T-1 真实持仓+现金做种子，只回放 T 日多退少补，
    #    使持仓/下单手数与实盘可比（手数差归因执行层、价格差归因滑点）。
    #    无 T-1 种子（首个交易日 / 缺快照）时回退到多日连续回测。
    recorder.mark("盘后回测")
    bt_result = _run_seed_replay(trade_date, individual_config, factor_classes, kline_data=kline_data)
    if bt_result is None:
        # T-1 种子缺失：优先用「最近一个可用实盘快照」做真实资金种子，从该日演化到 T，
        # 使目标持仓与实盘资金体量对齐；无任何历史快照时才从默认空仓起步。
        from utils.stock.time import get_trading_date_span
        seed = _load_latest_seed(trade_date)
        if seed is not None:
            seed_date, seed_cash, seed_positions = seed
            span = get_trading_date_span(seed_date, trade_date)
            bt_start = span[1] if len(span) >= 2 else trade_date
            trading_logger.info(
                f"[PostClose] 缺 T-1 种子，改用最近实盘快照 {seed_date} "
                f"(现金={seed_cash:,.0f}+{len(seed_positions)}只) 连续演化 {bt_start} → {trade_date}")
            bt_result = _run_continuous_backtest(
                bt_start, trade_date, individual_config, factor_classes,
                kline_data=kline_data, init_cash=seed_cash, init_positions=seed_positions)
        else:
            bt_start = _resolve_backtest_start(trade_date)
            if bt_start < trade_date:
                trading_logger.info(f"[PostClose] 回退连续回测窗口: {bt_start} → {trade_date}")
            else:
                trading_logger.info(f"[PostClose] 回退单日回测: {trade_date} (实盘尚无历史调仓)")
            bt_result = _run_continuous_backtest(bt_start, trade_date,
                                                  individual_config, factor_classes, kline_data=kline_data)
    if bt_result is None:
        trading_logger.warning("[PostClose] 回测失败")
        return None
    snaps = bt_result.get('daily_snapshots') or []
    trading_logger.info(f"[PostClose] 回测完成: {len(snaps)}日快照")

    # 3. 构建对比报告。账户事实已在慢速数据/回测之前采集，
    #    这里仅复用快照，不再重复请求 QMT，避免时刻漂移。
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

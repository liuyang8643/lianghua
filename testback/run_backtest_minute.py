"""OvernightGap 权重=0, 过滤×信号 四版对比。

4 组: (无过滤/有过滤) × (信号=09:30 open / 信号=09:32 open)
每组内: 买入价 = 信号open(base) / 09:32 / 09:33 / 09:34 / 09:35 open

用法: uv run python -m testback.run_backtest_minute
"""

from __future__ import annotations

import json, sys, time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MINUTE_DIR = ROOT / "data" / "minute"
MINUTE_LABELS = ["09:31", "09:32", "09:33", "09:34", "09:35"]


def _log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _build_lookup() -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    _log("构建分钟价格查找表...")
    lookup: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for f in MINUTE_DIR.glob("*.parquet"):
        if f.stem == "slippage_stats": continue
        df = pd.read_parquet(f)
        code = f.stem
        cm: dict[int, dict[str, dict[str, float]]] = {}
        for _, row in df.iterrows():
            d_int = int(row["time"].strftime("%Y%m%d"))
            m = row["time"].strftime("%H:%M")
            if m in MINUTE_LABELS:
                cm.setdefault(d_int, {})[m] = {"open": float(row["open"]), "close": float(row["close"])}
        if cm: lookup[code] = cm
    _log(f"查找表: {len(lookup)} 只股票")
    return lookup


def _annual_sharpe(r: np.ndarray) -> float:
    a = r / 100.0
    if len(a) < 2: return 0.0
    m, s = np.mean(a), np.std(a, ddof=1)
    return float(m / s * np.sqrt(252)) if s else 0.0


def _annual_return(r: np.ndarray) -> float:
    a = r / 100.0
    if len(a) < 2: return 0.0
    return float(np.prod(1 + a) ** (252 / len(a)) - 1) * 100


def _nav_from_snaps(snaps, init=700_000.0) -> np.ndarray:
    nav = np.empty(len(snaps) + 1); nav[0] = init
    for i, s in enumerate(snaps):
        nav[i + 1] = nav[i] * (1 + s["daily_return_pct"] / 100.0)
    return nav


def _max_dd(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak) * 100)


def _metrics(nav: np.ndarray) -> dict:
    r = np.diff(nav) / nav[:-1] * 100
    return {"final": nav[-1], "total": (nav[-1]-700000)/700000*100,
            "ann_ret": _annual_return(r), "sharpe": _annual_sharpe(r), "max_dd": _max_dd(nav)}


def _daily_slip(buys, lookup, price_field: str, base_min: str, cmp_mins: list[str]):
    daily = {m: {} for m in cmp_mins}
    for b in buys:
        dm = lookup.get(b["code"], {}).get(int(b["trade_date"].replace("-", "")), {})
        bb = dm.get(base_min, {}); bp = bb.get(price_field, 0) if bb else 0
        if bp <= 0: continue
        for m in cmp_mins:
            bar = dm.get(m, {}); mp = bar.get(price_field, 0) if bar else 0
            if mp > 0:
                daily[m][b["trade_date"]] = daily[m].get(b["trade_date"], 0.0) + b["shares"] * (mp - bp)
    return daily


def _adjust_nav(nav_base: np.ndarray, snap_dates: list, slip: dict) -> np.ndarray:
    nav = nav_base.copy(); cum = 0.0
    for i, d in enumerate(snap_dates):
        cum += slip.get(d, 0.0); nav[i+1] = max(nav[i+1] - cum, 0.01)
    return nav


def _apply_amount_filter(all_scores, data, pct=10):
    """对 all_scores 做硬过滤: 每日剔除近20日均成交额最低 pct% 的股票"""
    import factor_db.factors.LiquidityFilter as lf
    lf.MIN_AMOUNT_PCT = pct
    liq = lf.LiquidityFilter()
    raw = liq.calc_batch(data)
    mask = np.isnan(raw)
    for fn in all_scores:
        all_scores[fn][mask] = 0.0
    return liq.thresholds


def _run_one(data, all_scores, valid_dates, date_indices, valid_stocks, stock_indices,
             kwa, lookup, label, buy_mins, sig_label):
    """跑一次回测，返回 baseline + 各分钟买入价的指标字典"""
    from core.backtest import _backtest_direct
    t0 = time.time()
    result = _backtest_direct(data, all_scores, valid_dates, date_indices,
                              valid_stocks, stock_indices, **kwa)
    _log(f"  {label} 回测: {time.time()-t0:.1f}s")

    snaps = result["daily_snapshots"]
    nav_base = _nav_from_snaps(snaps)
    snap_dates = [s["trade_date"] for s in snaps]

    buys = []; [buys.extend(s["executed_buy_details"]) for s in snaps]
    n_matched = 0
    for b in buys:
        dm = lookup.get(b["code"], {}).get(int(b["trade_date"].replace("-", "")), {})
        if dm.get("09:31", {}).get("open", 0) > 0: n_matched += 1

    # 信号 bar: 09:31=09:30 open, 09:32=09:31时刻
    signal_minute = "09:31" if sig_label == "09:30" else "09:32"
    slip = _daily_slip(buys, lookup, "open", signal_minute, buy_mins)

    out = {}
    out["base"] = _metrics(nav_base)
    for m in buy_mins:
        nav = _adjust_nav(nav_base, snap_dates, slip[m])
        out[m] = _metrics(nav)
    out["n_buys"] = len(buys)
    out["n_matched"] = n_matched
    return out


def run_all(config_path: str, start_date: str, end_date: str):
    from core.backtest import _compute_factor_scores, _compute_list_dates
    from core.factors.registry import get_factor_class
    from core.runtime import load_runtime_npz
    from data.db import allow_buy_stock_code_list
    from utils.stock.time import get_trading_date_span

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ic = cfg["individual_config"]

    bdt_list = [datetime.combine(d, datetime.min.time())
                for d in get_trading_date_span(
                    date(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8])),
                    date(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:8])))]

    all_stocks = list(allow_buy_stock_code_list())
    pool = ic.get("stock_pool") or ("60", "00", "30", "688")
    pool_t = tuple(pool) if isinstance(pool, list) else pool
    all_stocks = [s for s in all_stocks if s.startswith(pool_t)]
    _log(f"股票池: {len(all_stocks)}  区间: {start_date}~{end_date}  {len(bdt_list)} 交易日")

    # OvernightGap 权重置 0
    weights_no_og = dict(ic["weights"])
    weights_no_og["OvernightGap"] = 0
    temps = dict(ic["temperatures"])
    factor_classes = [get_factor_class(fn) for fn in ic["weights"]]

    # 首次加载 NPZ
    sr0 = _compute_factor_scores(bdt_list, all_stocks, weights=weights_no_og, factor_classes=factor_classes)
    if sr0 is None: _log("因子计算失败"); sys.exit(1)
    data_orig, _, valid_dates, date_indices, valid_stocks, stock_indices = sr0

    lookup = _build_lookup()
    trade_dates = data_orig["trade_dates"]
    d2r = {pd.Timestamp(d).strftime("%Y%m%d"): i for i, d in enumerate(trade_dates)}

    # 择时
    timing = None
    if ic.get("timing_enabled", True) and ic.get("timing_base") is not None:
        from testback.market_timing import load_index_open, compute_position_multiplier
        _, io = load_index_open(ic.get("timing_index", "sh000852"), valid_dates)
        if io is not None:
            timing = compute_position_multiplier(io, window=ic.get("timing_window", 20),
                base=ic["timing_base"], leverage=ic.get("timing_leverage", 10),
                direction=ic.get("timing_direction", 1))
    list_dates = _compute_list_dates(data_orig["stock_codes"], data_orig["open"], data_orig["trade_dates"])

    base_kwa = dict(weights=weights_no_og, buy_n=ic["buy_n"], sell_m=ic["sell_m"],
                    temperatures=temps, holding_period=ic.get("holding_period"),
                    position_multipliers=timing, list_dates_map=list_dates, lightweight=False)

    buy_minutes = ["09:32", "09:33", "09:34", "09:35"]

    # 构建 09:32 信号 data
    modified_open = data_orig["open"].copy()
    repl = 0
    for code, cm in lookup.items():
        si = stock_indices.get(code)
        if si is None: continue
        for d_int, mins in cm.items():
            row = d2r.get(str(d_int))
            if row is None: continue
            bar = mins.get("09:32", {})
            if bar and bar["open"] > 0:
                modified_open[row, si] = bar["open"]; repl += 1
    _log(f"09:32信号 open 替换: {repl} cells")
    data_0932 = dict(data_orig); data_0932["open"] = modified_open

    scenarios = [
        ("无过滤 + 09:30信号", data_orig, False, "09:30"),
        ("无过滤 + 09:32信号", data_0932, False, "09:32"),
        ("过滤底部10% + 09:30信号", data_orig, True, "09:30"),
        ("过滤底部10% + 09:32信号", data_0932, True, "09:32"),
    ]

    all_results = {}
    for title, d, do_filter, sig_label in scenarios:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

        sr = _compute_factor_scores(bdt_list, all_stocks, weights=weights_no_og,
                                    factor_classes=factor_classes, data=d)
        if sr is None: _log("因子计算失败"); continue
        data_new, all_scores, vd, di, vs, si = sr

        if do_filter:
            th = _apply_amount_filter(all_scores, data_new)
            print(f"  过滤阈值: min={min(th)/1e4:,.0f}万 median={np.median(th)/1e4:,.0f}万 max={max(th)/1e4:,.0f}万")

        ld = _compute_list_dates(data_new["stock_codes"], data_new["open"], data_new["trade_dates"])
        kwa = {**base_kwa, "list_dates_map": ld}

        r = _run_one(data_new, all_scores, vd, di, vs, si, kwa, lookup, title, buy_minutes, sig_label)
        all_results[title] = r

        # 打印
        print(f"  买入: {r['n_buys']}笔  匹配: {r['n_matched']}笔")
        print(f"  {'买入价':<12} {'最终资产':>13} {'总收益':>8} {'年化收益':>9} {'年化夏普':>9} {'最大回撤':>8}")
        print(f"  {'-'*58}")
        for lbl in ["base"] + buy_minutes:
            m = r[lbl]
            lbl_disp = {"base": sig_label+"(base)", **{x: x for x in buy_minutes}}[lbl]
            print(f"  {lbl_disp:<12} {m['final']:>13,.0f} {m['total']:>7.2f}% "
                  f"{m['ann_ret']:>8.2f}% {m['sharpe']:>9.3f} {m['max_dd']:>8.2f}%")

    # 汇总
    print(f"\n{'='*70}")
    print(f"  四版汇总")
    print(f"{'='*70}")
    print(f"  {'场景':<28} {'base年化':>9} {'base夏普':>9} {'09:32买年化':>11} {'09:35买年化':>11}")
    print(f"  {'-'*68}")
    for title, r in all_results.items():
        print(f"  {title:<28} {r['base']['ann_ret']:>8.2f}% {r['base']['sharpe']:>9.3f} "
              f"{r['09:32']['ann_ret']:>10.2f}% {r['09:35']['ann_ret']:>10.2f}%")


if __name__ == "__main__":
    run_all(config_path="configs/20260526.json", start_date="20251118", end_date="20260605")

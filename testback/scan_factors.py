"""批量单因子扫描：每个因子 weight=1，遍历 holding_period=[1,3,7,15,30]。

特性：增量保存（每因子完成后写 JSON），断点续跑（--resume）

用法:
    uv run python testback/scan_factors.py --start 2024-01-01 --end 2024-12-31
    uv run python testback/scan_factors.py --start 2024-01-01 --end 2024-12-31 --resume
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.factors.registry import get_factor_class, get_factor_names
from core.runtime import load_runtime_npz, load_runtime_stock_codes
from core.backtest import (
    _compute_factor_scores,
    _backtest_direct,
    _compute_list_dates,
    _ALL_A_SHARE_PREFIXES,
)
from core.metrics import compute_core_metrics
from utils.stock.time import get_trading_date_span

HOLDING_PERIODS = [1, 3, 7, 15, 30]
DEFAULT_BUY_N = 20
DEFAULT_OUTPUT = "results/factor_scan.json"


def _parse_date(s):
    s = s.replace("-", "")
    if len(s) == 8:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    raise ValueError(f"日期格式错误: {s}")


def _load_existing(output_path: Path) -> tuple[list[dict], set[str]]:
    """加载已有结果，返回 (results, completed_factors)"""
    if not output_path.exists():
        return [], set()
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        completed = {r["factor"] for r in existing}
        return existing, completed
    except (json.JSONDecodeError, KeyError):
        return [], set()


def _save_results(results: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _print_factor_results(factor_name: str, factor_results: list[dict]):
    print(f"  {'HP':<6} {'年化%':>8} {'回撤%':>8} {'夏普':>7} {'卡玛':>7} {'总收益%':>8}")
    for r in factor_results:
        print(
            f"  {r['holding_period']:<6} {r['annualized']:>8.2f} {r['max_drawdown']:>8.2f} "
            f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['total_return']:>8.2f}"
        )
    print()


def _print_summary(results: list[dict], factor_names: list[str]):
    print("=" * 75)
    print(f"{'因子':<42} {'最佳HP':>6} {'年化%':>8} {'回撤%':>8} {'夏普':>7}")
    print("-" * 75)

    factor_best = {}
    for r in results:
        key = r["factor"]
        prev = factor_best.get(key)
        if prev is None or r["sharpe"] > prev["sharpe"]:
            factor_best[key] = r

    for name in factor_names:
        r = factor_best.get(name)
        if r:
            print(
                f"{name:<42} {r['holding_period']:>6} {r['annualized']:>8.2f} "
                f"{r['max_drawdown']:>8.2f} {r['sharpe']:>7.2f}"
            )

    # 统计
    sharpes = [r["sharpe"] for r in factor_best.values()]
    if sharpes:
        pos = sum(1 for s in sharpes if s > 0)
        print(f"\n夏普>0: {pos}/{len(sharpes)}, "
              f"均值={np.mean(sharpes):.2f}, 中位数={np.median(sharpes):.2f}")


def scan_factors(
    factor_names: list[str],
    start_date: date,
    end_date: date,
    buy_n: int = DEFAULT_BUY_N,
    output_path: Path | None = None,
    resume: bool = False,
):
    output_path = output_path or Path(DEFAULT_OUTPUT)

    # 断点续跑
    results, completed = _load_existing(output_path) if resume else ([], set())
    pending = [n for n in factor_names if n not in completed]
    if completed:
        print(f"断点续跑: {len(completed)} 已完成, {len(pending)} 待处理")

    if not pending:
        print("所有因子已完成，跳过回测")
        return results

    backtest_datetime_list = [
        datetime.combine(d, datetime.min.time())
        for d in get_trading_date_span(start_date, end_date)
    ]
    all_stocks = load_runtime_stock_codes()
    pool_stocks = [s for s in all_stocks if s.startswith(_ALL_A_SHARE_PREFIXES)]

    print(f"股票池: {len(pool_stocks)} 只 | 区间: {start_date} ~ {end_date} | "
          f"buy_n={buy_n} | init_cash=100w")
    print(f"因子: {len(factor_names)} 个 × {len(HOLDING_PERIODS)} 持仓周期 = "
          f"{len(factor_names) * len(HOLDING_PERIODS)} 回测\n")

    # 预加载 NPZ
    all_factor_classes = [get_factor_class(n) for n in factor_names]
    max_lookback = max((c.hist_days for c in all_factor_classes), default=0)
    t0 = time.time()
    data = load_runtime_npz(backtest_datetime_list, max_lookback=max_lookback)
    if data is None:
        print("ERROR: 未找到覆盖日期的 runtime npz", file=sys.stderr)
        sys.exit(1)

    list_dates_map = _compute_list_dates(
        data["stock_codes"], data["open"], data["trade_dates"]
    )

    total_hp = len(pending) * len(HOLDING_PERIODS)
    completed_hp = len(completed) * len(HOLDING_PERIODS)

    for fi, factor_name in enumerate(pending):
        factor_cls = get_factor_class(factor_name)
        weights = {factor_name: 1.0}

        t_factor = time.time()
        scores_result = _compute_factor_scores(
            backtest_datetime_list, pool_stocks,
            weights=weights, factor_classes=[factor_cls], data=data,
        )
        if scores_result is None:
            print(f"[{fi+1}/{len(pending)}] {factor_name}: 因子计算失败，跳过")
            continue
        data2, all_scores, _, valid_dates, date_indices, valid_stocks, stock_indices = (
            scores_result
        )

        factor_results = []
        for hp in HOLDING_PERIODS:
            bt = _backtest_direct(
                data2, all_scores, valid_dates, date_indices,
                valid_stocks, stock_indices,
                weights=weights, buy_n=buy_n, sell_m=buy_n,
                holding_period=hp,
                list_dates_map=list_dates_map,
                lightweight=True,
            )
            m = compute_core_metrics(bt["daily_returns"])
            calmar = (
                round(m["annualized"] / abs(m["max_drawdown"]), 2)
                if m["max_drawdown"] != 0 else 0.0
            )
            factor_results.append({
                "factor": factor_name, "holding_period": hp,
                "annualized": round(m["annualized"], 2),
                "max_drawdown": round(m["max_drawdown"], 2),
                "sharpe": round(m["sharpe"], 2),
                "calmar": calmar,
                "total_return": round(bt["total_return"], 2),
                "total_trades": bt.get("total_trades", 0),
            })

        results.extend(factor_results)
        _save_results(results, output_path)

        elapsed = time.time() - t_factor
        done_count = completed_hp + (fi + 1) * len(HOLDING_PERIODS)
        eta = (total_hp - done_count) / len(HOLDING_PERIODS) * (elapsed)
        print(f"[{fi+1:3d}/{len(pending)}] {factor_name}  ({elapsed:.1f}s, "
              f"预计剩余 {eta/60:.0f}min)")
        _print_factor_results(factor_name, factor_results)

    total_elapsed = time.time() - t0
    print(f"\n总耗时: {total_elapsed/60:.1f}min")
    _print_summary(results, factor_names)
    print(f"\n结果已保存: {output_path.resolve()}")
    return results


def main():
    parser = argparse.ArgumentParser(description="WBR 单因子批量扫描")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--factors", type=str, default=None)
    parser.add_argument("--buy-n", type=int, default=DEFAULT_BUY_N)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    args = parser.parse_args()

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)

    if args.factors:
        factor_names = [n.strip() for n in args.factors.split(",") if n.strip()]
        valid_names = set(get_factor_names())
        for n in factor_names:
            if n not in valid_names:
                print(f"ERROR: 未知因子 '{n}'", file=sys.stderr)
                sys.exit(1)
    else:
        factor_names = sorted(get_factor_names())

    scan_factors(
        factor_names, start_date, end_date,
        buy_n=args.buy_n,
        output_path=Path(args.output),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

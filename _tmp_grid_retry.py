"""补跑失败的 OvernightGap 值"""
import json, re, subprocess, sys
from multiprocessing import Pool
from pathlib import Path

CONFIG_TEMPLATE = Path("configs/20260526.json")
START_DATE = "20100101"
END_DATE = "20260605"
# 之前失败的 10 个值
VALUES = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.9]


def run_one(val):
    config = json.loads(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    config["individual_config"]["temperatures"]["OvernightGap"] = val
    tmp_path = Path(f"_tmp_scan_og_{val:.1f}.json")
    tmp_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    try:
        result = subprocess.run([
            "uv", "run", "python", "testback/run_backtest.py",
            "--individual-config", str(tmp_path),
            "--start-date", START_DATE, "--end-date", END_DATE,
        ], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        return val, {"error": "timeout"}

    combined = result.stderr + result.stdout
    metrics = {}
    for line in combined.splitlines():
        if "年化=" in line and "夏普=" in line:
            m = re.search(
                r'年化=([\d.-]+)%?\s+夏普=([\d.-]+)\s+最大回撤=([\d.-]+)%?\s+卡玛=([\d.-]+)\s+胜率=([\d.-]+)%?\s+总成交=(\d+)',
                line,
            )
            if m:
                metrics = {
                    "annualized": float(m.group(1)),
                    "sharpe": float(m.group(2)),
                    "max_drawdown": float(m.group(3)),
                    "calmar": float(m.group(4)),
                    "win_rate": float(m.group(5)),
                    "total_trades": int(m.group(6)),
                }
            break
    if not metrics:
        metrics["error"] = f"rc={result.returncode}"
    tmp_path.unlink(missing_ok=True)
    return val, metrics


def main():
    print(f"补跑 {len(VALUES)} 个失败值: {VALUES}")
    with Pool(processes=len(VALUES)) as pool:
        results = pool.map(run_one, VALUES)
    for val, m in sorted(results, key=lambda x: x[0]):
        print(f"  OvernightGap={val:+.1f}  -> {m}")
    # 合并之前结果，按夏普排序全部输出
    previous = {
        -1.0: {"annualized": -3.53, "sharpe": -0.61, "max_drawdown": -50.68, "calmar": 0.07, "win_rate": 47.4, "total_trades": 14623},
        -0.9: {"annualized": -4.12, "sharpe": -0.68, "max_drawdown": -55.84, "calmar": 0.07, "win_rate": 48.6, "total_trades": 14909},
        -0.8: {"annualized": -3.42, "sharpe": -0.56, "max_drawdown": -50.14, "calmar": 0.07, "win_rate": 48.3, "total_trades": 14985},
        -0.7: {"annualized": -4.27, "sharpe": -0.73, "max_drawdown": -56.48, "calmar": 0.08, "win_rate": 47.1, "total_trades": 15150},
        -0.6: {"annualized": -2.59, "sharpe": -0.44, "max_drawdown": -42.80, "calmar": 0.06, "win_rate": 49.0, "total_trades": 15325},
        -0.5: {"annualized": -4.72, "sharpe": -0.85, "max_drawdown": -59.90, "calmar": 0.08, "win_rate": 46.8, "total_trades": 15368},
        0.4:  {"annualized": 186.99, "sharpe": 3.81, "max_drawdown": -41.11, "calmar": 4.55, "win_rate": 63.5, "total_trades": 126890},
        0.6:  {"annualized": 178.12, "sharpe": 3.70, "max_drawdown": -39.07, "calmar": 4.56, "win_rate": 63.7, "total_trades": 123419},
        0.7:  {"annualized": 171.86, "sharpe": 3.61, "max_drawdown": -40.42, "calmar": 4.25, "win_rate": 63.7, "total_trades": 121392},
        0.8:  {"annualized": 168.53, "sharpe": 3.59, "max_drawdown": -41.76, "calmar": 4.04, "win_rate": 63.8, "total_trades": 119620},
        1.0:  {"annualized": 159.81, "sharpe": 3.43, "max_drawdown": -42.81, "calmar": 3.73, "win_rate": 63.8, "total_trades": 116870},
    }
    for v, m in results:
        if "sharpe" in m:
            previous[v] = m
    all_valid = [(v, m) for v, m in previous.items()]
    all_valid.sort(key=lambda x: x[1]["sharpe"], reverse=True)

    print("\n" + "=" * 80)
    print("完整汇总 (按夏普排序)")
    print("=" * 80)
    print(f"{'OvernightGap':>12}  {'年化%':>8}  {'夏普':>7}  {'最大回撤%':>9}  {'卡玛':>7}  {'胜率%':>7}  {'成交':>6}")
    print("-" * 70)
    for v, m in all_valid:
        print(f"{v:>+12.1f}  {m['annualized']:>8.2f}  {m['sharpe']:>7.3f}  {m['max_drawdown']:>9.2f}  {m['calmar']:>7.3f}  {m['win_rate']:>7.1f}  {m['total_trades']:>6}")

    best = all_valid[0]
    print(f"\n最佳: OvernightGap={best[0]:+.1f}  夏普={best[1]['sharpe']:.3f}  年化={best[1]['annualized']:.2f}%  卡玛={best[1]['calmar']:.3f}")


if __name__ == "__main__":
    main()

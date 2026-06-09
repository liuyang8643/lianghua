"""临时脚本：OvernightGap weight 参数扫描 — 多进程并行"""
import json
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

CONFIG_TEMPLATE = Path("configs/20260526.json")
START_DATE = "20100101"
END_DATE = "20260605"
VALUES = [round(x / 10, 1) for x in range(-10, 11)]  # -1.0, -0.9, ..., 1.0


def run_one(val):
    config = json.loads(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    config["individual_config"]["weights"]["OvernightGap"] = val

    tmp_path = Path(f"_tmp_scan_og_w_{val:.1f}.json")
    tmp_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    cmd = [
        "uv", "run", "python", "-u", "-m", "testback.run_backtest",
        "--individual-config", str(tmp_path),
        "--start-date", START_DATE,
        "--end-date", END_DATE,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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
        # 找最后几行 stderr 报错
        tail = "\n".join((result.stderr).splitlines()[-5:])
        metrics["error"] = f"rc={result.returncode} stderr_tail={tail[:200]}"

    tmp_path.unlink(missing_ok=True)
    return val, metrics


def main():
    print(f"启动 {len(VALUES)} 进程并行扫描 OvernightGap weight [{VALUES[0]:.1f}, {VALUES[-1]:.1f}]")
    sys.stdout.flush()

    with Pool(processes=len(VALUES)) as pool:
        results = pool.map(run_one, VALUES)

    for val, m in sorted(results, key=lambda x: x[0]):
        print(f"  weight={val:+.1f}  -> {m}")

    valid = [(v, m) for v, m in results if m and "sharpe" in m]
    if not valid:
        print("\nERROR: 所有回测均失败！")
        return

    valid.sort(key=lambda x: x[1]["sharpe"], reverse=True)

    print("\n" + "=" * 80)
    print("完整汇总 (按夏普排序)")
    print("=" * 80)
    print(f"{'Weight':>8}  {'年化%':>8}  {'夏普':>7}  {'最大回撤%':>9}  {'卡玛':>7}  {'胜率%':>7}  {'成交':>6}")
    print("-" * 70)
    for v, m in valid:
        print(f"{v:>+8.1f}  {m['annualized']:>8.2f}  {m['sharpe']:>7.3f}  {m['max_drawdown']:>9.2f}  {m['calmar']:>7.3f}  {m['win_rate']:>7.1f}  {m['total_trades']:>6}")

    best = valid[0]
    print(f"\n最佳: OvernightGap weight={best[0]:+.1f}  夏普={best[1]['sharpe']:.3f}  年化={best[1]['annualized']:.2f}%  卡玛={best[1]['calmar']:.3f}")

    # 清理可能残留的临时文件
    for f in Path(".").glob("_tmp_scan_og_w_*.json"):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

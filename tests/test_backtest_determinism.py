"""回测确定性回归测试。

核心回测路径曾因遍历 Python set（哈希顺序随 PYTHONHASHSEED 变化）导致
现金受限买入 tie-break / 浮点累加顺序不稳定，同一因子两次回测结果不同。
修复后卖出候选按 topn 优先 + 持仓插入序遍历，结果应与进程哈希种子无关。
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_SCRIPT = '''
from datetime import date, datetime
from types import SimpleNamespace
from core.backtest import run_single_mode
from core.runtime import load_runtime_stock_codes
from utils.stock.time import get_trading_date_span

dts = [datetime.combine(d, datetime.min.time())
       for d in get_trading_date_span(date(2015, 1, 1), date(2015, 6, 30))]
all_stocks = load_runtime_stock_codes()
args = SimpleNamespace(individual_config="configs/config.json",
                       output_dir="results/_dettest_tmp",
                       start_date="20150101", end_date="20150630")
res = run_single_mode(args, {"desc": "det", "log_level": "INFO", "save_charts": False},
                      dts, all_stocks)
print("RESULT", repr(res["final_asset"]), repr(res["total_return"]),
      res["executed_buy_count"], res["executed_sell_count"])
'''


def _run(hashseed: int) -> str:
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0]


def test_backtest_deterministic_across_hashseeds():
    a = _run(1)
    b = _run(7)
    assert a == b, f"回测非确定性: {a!r} != {b!r}"

"""回测确定性回归测试。

核心回测路径曾因遍历 Python set（哈希顺序随 PYTHONHASHSEED 变化）导致
现金受限买入 tie-break / 浮点累加顺序不稳定，同一因子两次回测结果不同。
修复后卖出候选按 topn 优先 + 持仓插入序遍历，结果应与进程哈希种子无关。
"""
import os
import subprocess
import sys
from pathlib import Path

from rl_test_data import write_runtime

REPO = Path(__file__).resolve().parents[1]

_SCRIPT = '''
import os
from types import SimpleNamespace
from testback.backtest import run_single_mode

args = SimpleNamespace(individual_config="configs/config.json",
                       output_dir=os.environ["WBR_DETERMINISM_OUTPUT"],
                       runtime_path=os.environ["WBR_DETERMINISM_RUNTIME"],
                       start_date="20200620", end_date="20200624")
res = run_single_mode(args, {"desc": "det", "log_level": "INFO", "save_charts": False})
print("RESULT", repr(res["final_asset"]), repr(res["total_return"]),
      res["executed_buy_count"], res["executed_sell_count"])
'''


def _run(hashseed: int, runtime_path: Path, output_dir: Path) -> str:
    env = dict(
        os.environ,
        PYTHONHASHSEED=str(hashseed),
        PYTHONIOENCODING="utf-8",
        WBR_DETERMINISM_RUNTIME=str(runtime_path),
        WBR_DETERMINISM_OUTPUT=str(output_dir),
    )
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0]


def test_backtest_deterministic_across_hashseeds(tmp_path: Path):
    runtime_path = tmp_path / "runtime.npz"
    write_runtime(runtime_path)
    a = _run(1, runtime_path, tmp_path / "seed-1")
    b = _run(7, runtime_path, tmp_path / "seed-7")
    assert a == b, f"回测非确定性: {a!r} != {b!r}"

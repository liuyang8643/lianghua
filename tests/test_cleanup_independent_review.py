"""Independent failure and end-to-end checks for the September cleanup."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_backtest_lightweight import write_canonical_runtime
from utils.atomic_file import atomic_write_json


def test_failed_fsync_keeps_last_json_and_removes_temporary(tmp_path, monkeypatch):
    target = tmp_path / "current.json"
    target.write_bytes(b'{"previous":true}')

    def fail(_descriptor):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="injected fsync failure"):
        atomic_write_json(target, {"replacement": True})
    assert target.read_bytes() == b'{"previous":true}'
    assert list(tmp_path.iterdir()) == [target]


def test_concurrent_json_writers_publish_one_complete_payload(tmp_path):
    target = tmp_path / "current.json"
    payloads = [{"writer": n, "data": [n] * 2048} for n in range(8)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda payload: atomic_write_json(target, payload), payloads))
    assert json.loads(target.read_text("utf8")) in payloads
    assert list(tmp_path.iterdir()) == [target]


def test_real_fixed_backtest_cli_runs_complete_local_fixture(tmp_path):
    runtime = tmp_path / "runtime.npz"
    write_canonical_runtime(runtime, stocks=30)
    output = tmp_path / "cli"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-u", "-m", "testback.run_backtest",
         "--runtime", str(runtime), "--individual-config", "configs/config.json",
         "--start-date", "2020-06-01", "--end-date", "2020-06-28",
         "--output-dir", str(output), "--no-charts"],
        cwd=root, capture_output=True, text=True, encoding="utf8", timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf8"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((output / "record.json").read_text("utf8"))
    assert record["dates"][0] == "2020-06-01"
    assert record["dates"][-1] == "2020-06-27"
    assert len(record["dates"]) == 27
    assert record["period"]["settlement_end"] == "2020-06-28"
    assert record["metrics"]["executed_buy_count"] > 0
    assert record["full_investment_contract_satisfied"] is True
    assert (output / "single.log").stat().st_size > 0

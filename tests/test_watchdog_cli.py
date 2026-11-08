from pathlib import Path
import sys

import pytest

from trade import watchdog


def test_process_cleanup_rejects_unregistered_wrapper_before_process_lookup(monkeypatch):
    from types import SimpleNamespace
    from utils import sys as process_cleanup

    looked_up = []
    monkeypatch.setattr(
        process_cleanup.psutil,
        "Process",
        type("RecordedProcess", (), {"__init__": lambda self, pid: looked_up.append(pid)}),
    )
    wrapper = SimpleNamespace(pid=12345, poll=lambda: None)
    with pytest.raises(ValueError, match="不支持的进程类型"):
        process_cleanup.terminate_process_tree(wrapper)
    assert looked_up == []


def test_watchdog_builds_only_canonical_live_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watchdog, "REPO_ROOT", tmp_path)

    command = watchdog.build_main_command(
        bundle="models/policy",
        runtime="runtime/today.npz",
        decision_date="2026-08-30",
        account="test-account",
        journal="journal",
        initialize_new_chain=True,
    )

    assert command == [
        sys.executable,
        "-m",
        "trade.main",
        "--bundle",
        str((tmp_path / "models/policy").resolve()),
        "--runtime",
        str((tmp_path / "runtime/today.npz").resolve()),
        "--execute",
        "--date",
        "2026-08-30",
        "--account",
        "test-account",
        "--journal",
        str((tmp_path / "journal").resolve()),
        "--initialize-new-chain",
    ]
    assert "--individual-config" not in command


def test_watchdog_requires_explicit_execute_acknowledgement() -> None:
    with pytest.raises(SystemExit):
        watchdog.build_parser().parse_args(
            ["--bundle", "models/policy", "--runtime", "runtime/today.npz"]
        )


def test_watchdog_main_runs_exactly_one_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        watchdog,
        "run_once",
        lambda command: captured.append(command) or 17,
    )

    return_code = watchdog.main(
        [
            "--bundle",
            "models/policy",
            "--runtime",
            "runtime/today.npz",
            "--execute",
        ]
    )

    assert return_code == 17
    assert len(captured) == 1


def test_watchdog_prepares_complete_snapshot_before_starting_qmt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "runtime" / "old.npz"
    baseline.parent.mkdir()
    baseline.write_bytes(b"sealed")
    prepared = tmp_path / "runtime" / "today.npz"
    captured: list[list[str]] = []
    monkeypatch.setattr(watchdog, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        watchdog,
        "prepare_live_runtime",
        lambda **_: prepared,
    )
    monkeypatch.setattr(
        watchdog,
        "run_once",
        lambda command: captured.append(command) or 0,
    )

    result = watchdog.main(
        [
            "--bundle",
            "models/policy",
            "--runtime",
            str(baseline),
            "--date",
            "2026-08-28",
            "--prepare-snapshot",
            "--execute",
        ]
    )

    assert result == 0
    runtime_index = captured[0].index("--runtime") + 1
    assert captured[0][runtime_index] == str(prepared.resolve())

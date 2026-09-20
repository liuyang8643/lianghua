"""Start QMT and execute one canonical live decision exactly once.

Scheduling and deployment stay outside this process. In particular, a
successful or failed one-shot decision is never restarted automatically.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import signal
import subprocess
import sys
import time

from trade.broker.qmt import start_qmt
from utils.sys import terminate_process_tree


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_MODULE = "trade.main"


def _resolve_path(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def build_main_command(
    *,
    bundle: str,
    runtime: str,
    decision_date: str | None = None,
    account: str | None = None,
    journal: str | None = None,
    initialize_new_chain: bool = False,
) -> list[str]:
    """Build the sole supported ``trade.main`` invocation."""

    command = [
        sys.executable,
        "-m",
        MAIN_MODULE,
        "--bundle",
        str(_resolve_path(bundle)),
        "--runtime",
        str(_resolve_path(runtime)),
        "--execute",
    ]
    if decision_date:
        command.extend(("--date", decision_date))
    if account:
        command.extend(("--account", account))
    if journal:
        command.extend(("--journal", str(_resolve_path(journal))))
    if initialize_new_chain:
        command.append("--initialize-new-chain")
    return command


def run_once(command: list[str]) -> int:
    """Run QMT and one live subprocess, then clean up both process trees."""

    qmt_process = start_qmt()
    main_process: subprocess.Popen[bytes] | None = None

    def stop_processes() -> None:
        nonlocal main_process
        if main_process is not None and main_process.poll() is None:
            terminate_process_tree(main_process)
        main_process = None
        terminate_process_tree(qmt_process)

    def handle_sigterm(_signum: int, _frame: object) -> None:
        stop_processes()
        raise SystemExit(128 + signal.SIGTERM)

    previous_handler = signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        print(f"QMT platform ready (pid={qmt_process.pid})", flush=True)
        time.sleep(5)
        main_process = subprocess.Popen(command, cwd=REPO_ROOT)
        print(f"live decision started (pid={main_process.pid})", flush=True)
        return int(main_process.wait())
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        stop_processes()


def prepare_live_runtime(
    *,
    bundle: str,
    baseline_runtime: str,
    decision_date: date,
    journal_root: str | None,
    initialize_new_chain: bool,
) -> Path:
    """Build the T snapshot from the canonical T-1 prefilter state."""

    import json

    from ai.bundle import BundleManifest
    from data.update_live import build_live_runtime
    from env.action_schema import ActionSchema
    from env.observation import ObservationSchema
    from env.prefilter import prefilter_n_from_config
    from trade.journal import DecisionJournal
    from trade.runtime import live_prefilter_codes

    bundle_root = _resolve_path(bundle)
    manifest = BundleManifest.load(bundle_root, verify_files=True)
    manifest.require_deployable()
    action_schema = ActionSchema.from_dict(manifest.action_schema)
    observation_schema = ObservationSchema.from_dict(manifest.observation_schema)
    config_payload = json.loads(
        (bundle_root / manifest.config_file).read_text(encoding="utf-8")
    )
    if not isinstance(config_payload, dict):
        raise TypeError("bundled strategy config must be a JSON object")
    prefilter_n = prefilter_n_from_config(config_payload)
    journal = DecisionJournal(
        journal_root or "data/live_trades/canonical_journal",
        action_schema,
    )
    prior_date, prior_account, policy_memory = journal.load_prior_state_with_date(
        decision_date,
        initialize_new_chain=initialize_new_chain,
    )
    candidates = live_prefilter_codes(
        baseline_runtime,
        decision_date,
        policy_memory.previous_day_config,
        previous_decision_date=prior_date,
        prefilter_n=prefilter_n,
        held_codes=tuple(prior_account.positions) if prior_account else (),
        lookback=observation_schema.lookback,
    )
    return build_live_runtime(decision_date, candidate_codes=candidates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--date")
    parser.add_argument("--account")
    parser.add_argument("--journal")
    parser.add_argument("--initialize-new-chain", action="store_true")
    parser.add_argument(
        "--prepare-snapshot",
        action="store_true",
        help="download the complete T-open axis and seal a canonical runtime before QMT starts",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        required=True,
        help="required acknowledgement for the live QMT execution path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_path = str(_resolve_path(args.runtime))
    if args.prepare_snapshot:
        if not Path(runtime_path).is_file():
            raise FileNotFoundError(f"baseline runtime does not exist: {runtime_path}")
        decision_date = date.fromisoformat(args.date) if args.date else date.today()
        runtime_path = str(
            prepare_live_runtime(
                bundle=args.bundle,
                baseline_runtime=runtime_path,
                decision_date=decision_date,
                journal_root=args.journal,
                initialize_new_chain=args.initialize_new_chain,
            )
        )
    command = build_main_command(
        bundle=args.bundle,
        runtime=runtime_path,
        decision_date=args.date,
        account=args.account,
        journal=args.journal,
        initialize_new_chain=args.initialize_new_chain,
    )
    return run_once(command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_main_command",
    "build_parser",
    "main",
    "prepare_live_runtime",
    "run_once",
]

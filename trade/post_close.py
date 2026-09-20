"""Post-close audit from the immutable live journal; never rerun a policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

from ai.bundle import BundleManifest
from env.action_schema import ActionSchema
from trade.journal import DecisionJournal, JournalReplay
from trade.executor import BrokerExecutor


@dataclass(frozen=True)
class PostCloseSummary:
    decision_date: str
    planned_sell_quantity: int
    planned_buy_quantity: int
    filled_sell_quantity: int
    filled_buy_quantity: int
    total_fees: float
    model_sha256: str
    runtime_source_sha256: str


def summarize_replay(replay: JournalReplay) -> PostCloseSummary:
    if replay.fills is None:
        raise FileNotFoundError("decision journal has no broker Fill record")
    plan = replay.order_plan
    return PostCloseSummary(
        decision_date=plan.decision_date,
        planned_sell_quantity=sum(quantity for _, quantity in plan.sell_orders),
        planned_buy_quantity=sum(plan.buy_orders.values()),
        filled_sell_quantity=sum(
            fill.quantity for fill in replay.fills if fill.side == "sell"
        ),
        filled_buy_quantity=sum(
            fill.quantity for fill in replay.fills if fill.side == "buy"
        ),
        total_fees=sum(fill.fee for fill in replay.fills),
        model_sha256=str(replay.policy_identity["model_sha256"]),
        runtime_source_sha256=str(
            replay.snapshot_identity["runtime_source_sha256"]
        ),
    )


def run_post_close(
    *,
    trade_date: date,
    journal: DecisionJournal,
) -> PostCloseSummary:
    """Read the recorded Observation/action/plan/fills without policy access."""

    return summarize_replay(journal.replay(trade_date))


def reconcile_pending_execution(
    *,
    trade_date: date,
    journal: DecisionJournal,
    broker_executor: BrokerExecutor,
) -> PostCloseSummary:
    """Finalize a pending journal only after every broker order is terminal."""

    pending = journal.load_pending_execution(trade_date)
    plan = journal.load_order_plan_for_reconciliation(trade_date)
    fills = broker_executor.reconcile(plan, pending["accepted_orders"])
    journal.reconcile_pending(trade_date, fills)
    return summarize_replay(journal.replay(trade_date))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="bundle identity for schema decode")
    parser.add_argument("--journal", required=True)
    parser.add_argument("--date", required=True)
    return parser


def main(argv: list[str] | None = None) -> PostCloseSummary:
    args = build_parser().parse_args(argv)
    manifest = BundleManifest.load(Path(args.bundle).resolve(), verify_files=True)
    manifest.require_deployable()
    action_schema = ActionSchema.from_dict(manifest.action_schema)
    summary = run_post_close(
        trade_date=date.fromisoformat(args.date),
        journal=DecisionJournal(args.journal, action_schema),
    )
    print(
        json.dumps(summary.__dict__, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    main()


__all__ = [
    "PostCloseSummary",
    "build_parser",
    "main",
    "run_post_close",
    "reconcile_pending_execution",
    "summarize_replay",
]

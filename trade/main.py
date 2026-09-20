"""One-shot production entry for the canonical live decision chain.

The process accepts only a frozen deployable policy bundle and an already
sealed, complete local runtime snapshot.  Data download/snapshot construction
and scheduling stay outside this root assembly entry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

from env.contracts import Fill
from trade.executor import BrokerExecutionError, BrokerExecutor
from trade.journal import DecisionJournal
from trade.runtime import (
    BrokerAccountAdapter,
    DecisionSnapshot,
    LiveDecision,
    LiveDecisionRunner,
    SealedSnapshotAdapter,
)


DEFAULT_JOURNAL_ROOT = Path("data/live_trades/canonical_journal")


@dataclass(frozen=True)
class LiveRunResult:
    decision: LiveDecision
    fills: tuple[Fill, ...]


def execute_recorded_decision(
    *,
    decision: LiveDecision,
    journal: DecisionJournal,
    broker_executor: BrokerExecutor,
) -> tuple[Fill, ...]:
    """Persist a plan, then finalize only terminally confirmed broker reality."""

    journal.record_decision(decision)
    try:
        fills = tuple(broker_executor.execute(decision.order_plan))
    except BrokerExecutionError as exc:
        try:
            if exc.terminal_confirmed:
                journal.record_fills(decision.order_plan.decision_date, exc.fills)
            else:
                journal.record_pending_execution(
                    decision.order_plan.decision_date,
                    accepted_orders=exc.accepted_orders,
                    known_fills=exc.fills,
                    error=str(exc),
                )
        except Exception as journal_error:
            exc.add_note(f"execution journal persistence also failed: {journal_error}")
        raise
    journal.record_fills(decision.order_plan.decision_date, fills)
    return fills


def execute_live_once(
    *,
    decision_date: date,
    snapshot: DecisionSnapshot,
    policy: object,
    snapshot_adapter: SealedSnapshotAdapter,
    journal: DecisionJournal,
    trader: object,
    broker_executor: BrokerExecutor,
    initialize_new_chain: bool = False,
) -> LiveRunResult:
    """Run one serial account decision and persist reality before returning."""

    if snapshot.decision_date != decision_date.isoformat():
        raise ValueError("sealed snapshot date differs from requested live date")
    prior_date, prior_account, policy_memory = journal.load_prior_state_with_date(
        decision_date,
        initialize_new_chain=initialize_new_chain,
    )
    if prior_date is not None:
        previous_index = snapshot.prepared.decision_index - 1
        if previous_index < 0:
            raise RuntimeError("sealed live snapshot has no T-1 continuity row")
        expected_prior_date = str(
            snapshot.prepared.runtime.trade_dates[previous_index]
        )
        if prior_date != expected_prior_date:
            raise RuntimeError(
                "live journal is not continuous with the sealed T-1 runtime row"
            )
    asset = trader.query_asset()
    positions = trader.query_positions()
    if positions is None:
        raise RuntimeError("broker positions snapshot is missing")
    account = BrokerAccountAdapter().build(
        asset=asset,
        positions=positions,
        snapshot=snapshot,
        previous_account=prior_account,
    )
    decision = LiveDecisionRunner(snapshot_adapter.action_schema).decide(
        snapshot,
        account,
        policy_memory,
        policy,
    )
    fills = execute_recorded_decision(
        decision=decision,
        journal=journal,
        broker_executor=broker_executor,
    )
    return LiveRunResult(decision=decision, fills=fills)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="qualified frozen PPO bundle")
    parser.add_argument(
        "--runtime",
        required=True,
        help="complete sealed local runtime NPZ containing the decision row",
    )
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--account", help="QMT account; defaults to configs.TRADE_ACCOUNT")
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL_ROOT))
    parser.add_argument(
        "--initialize-new-chain",
        action="store_true",
        help="allow an atomic cold PolicyMemory only when the journal is empty",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required safety acknowledgement before connecting to QMT",
    )
    return parser


def main(argv: list[str] | None = None) -> LiveRunResult:
    args = build_parser().parse_args(argv)
    if not args.execute:
        raise SystemExit(
            "--execute is required; no legacy planner or simulation fallback is available"
        )

    # Heavy model and QMT imports remain at the assembly edge.  Snapshot
    # identity is checked before any broker order can be constructed.
    from ai.rl.policy import RLPolicy

    policy = RLPolicy.load(Path(args.bundle).resolve())
    snapshot_adapter = SealedSnapshotAdapter.from_policy(policy)
    journal = DecisionJournal(args.journal, policy.action_schema)
    decision_date = date.fromisoformat(args.date)
    # Complete-axis and append-only identity proof must pass before QMT is
    # imported or connected.
    snapshot = snapshot_adapter.load(args.runtime, decision_date)

    from configs import TRADE_ACCOUNT
    from trade.broker.trader import Trader
    from xtquant import xtconstant

    account_id = args.account or TRADE_ACCOUNT
    trader = Trader(account_id)
    broker_executor = BrokerExecutor(
        trader,
        buy_order_type=xtconstant.STOCK_BUY,
        sell_order_type=xtconstant.STOCK_SELL,
        terminal_statuses={
            xtconstant.ORDER_SUCCEEDED,
            xtconstant.ORDER_CANCELED,
            xtconstant.ORDER_JUNK,
            xtconstant.ORDER_PART_CANCEL,
        },
    )
    result = execute_live_once(
        decision_date=decision_date,
        snapshot=snapshot,
        policy=policy,
        snapshot_adapter=snapshot_adapter,
        journal=journal,
        trader=trader,
        broker_executor=broker_executor,
        initialize_new_chain=args.initialize_new_chain,
    )
    print(
        json.dumps(
            {
                "decision_date": result.decision.order_plan.decision_date,
                "sell_order_count": len(result.decision.order_plan.sell_orders),
                "buy_order_count": len(result.decision.order_plan.buy_orders),
                "fill_count": len(result.fills),
                "full_investment_contract_satisfied": bool(
                    result.decision.order_plan.diagnostics[
                        "full_investment_contract_satisfied"
                    ]
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "LiveRunResult",
    "build_parser",
    "execute_live_once",
    "execute_recorded_decision",
    "main",
]

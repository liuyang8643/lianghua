"""Multi-day audit replay from canonical journal records only."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path

from ai.bundle import BundleManifest
from env.action_schema import ActionSchema
from trade.journal import DecisionJournal
from trade.post_close import PostCloseSummary, summarize_replay


def replay_reports(
    start_date: date,
    end_date: date,
    *,
    journal: DecisionJournal,
) -> tuple[PostCloseSummary, ...]:
    """Read each recorded decision exactly once; no Policy is accepted."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    summaries: list[PostCloseSummary] = []
    current = start_date
    while current <= end_date:
        decision_path = journal.root / current.isoformat() / "decision.json"
        if decision_path.exists():
            summaries.append(summarize_replay(journal.replay(current)))
        current += timedelta(days=1)
    return tuple(summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser


def main(argv: list[str] | None = None) -> tuple[PostCloseSummary, ...]:
    args = build_parser().parse_args(argv)
    manifest = BundleManifest.load(Path(args.bundle).resolve(), verify_files=True)
    manifest.require_deployable()
    schema = ActionSchema.from_dict(manifest.action_schema)
    summaries = replay_reports(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        journal=DecisionJournal(args.journal, schema),
    )
    print(
        json.dumps(
            [summary.__dict__ for summary in summaries],
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return summaries


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main", "replay_reports"]

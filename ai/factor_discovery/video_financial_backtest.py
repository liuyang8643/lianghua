"""Offline published-rule financial research; no learning or holdout selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from env.calendar_replay import run_monthly_video_replay
from env.fees import DEFAULT_FEE_SCHEDULE
from factor.library.video_financial import calculate_financial_scores, VIDEO_FINANCIAL_NAMES, DECILE_NAMES
from offline_data import load_runtime_slice
from utils.atomic_file import atomic_write_json
from offline_data.runtime import load_runtime_calendar
from offline_data.financial_versions import load_financial_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--financial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2026-08-28")
    parser.add_argument("--names", nargs="+", choices=VIDEO_FINANCIAL_NAMES, default=list(VIDEO_FINANCIAL_NAMES))
    args = parser.parse_args()
    started = time.monotonic()
    sources = [Path("ai/factor_discovery/video_financial_backtest.py"), Path("offline_data/runtime.py"), Path("offline_data/contracts.py"), Path("offline_data/financial_versions.py"), Path("factor/library/video_financial.py"),
               Path("env/calendar_replay.py"), Path("env/planner.py"), Path("env/simulator.py"),
               Path("env/legality.py"), Path("env/fees.py"), Path("env/quantity.py"), Path("env/action_schema.py"), Path("utils/atomic_file.py")]
    protocol = {"schema": "video-financial-monthly-research-v2", "start": args.start, "end": args.end,
                "names": args.names, "fees": asdict(DEFAULT_FEE_SCHEDULE), "initial_cash": 1e9,
                "selection": "video threshold: all qualifying; directional group: floor(valid_count/10), stable stock-axis ties",
                "schedule": "month-end close signal, next-month first open buy, month-end scheduled close sell",
                "locked_exit": "retry only pending old-cohort quantities at following opens, rebased for company actions; freed cash waits for the next scheduled entry",
                "financial_time": "each source table's own announcement date strictly earlier than signal date",
                "financial_semantics": "ROE=TTM parent profit / same-report ending parent equity; PE=price*lagged total shares/TTM parent profit; PB=price*lagged total shares/equity",
                "field_assumptions": "other_payable excludes separately disclosed interest/dividends; cash-surplus flow uses current YTD; single-quarter YoY denominator abs(prior); 365 calendar-day listing minimum",
                "execution_differences": "original fees/rounding/limit handling were undisclosed; use shared WBR legality, lots, fees and synthetic total-return account. Selected count is frozen; blocked names are recorded and not replaced. 1bn notional reduces lot distortion but is not a capacity claim.",
                "source_approval_scope": "QMT white-listed raw amount fields; two-issuer original-PDF spot checks plus full snapshot integrity/coverage; not an exhaustive all-issuer PIT certification",
                "data_splits": "2012-2022/2023-2024/2025-2026 are descriptive calendar slices, not blind tests; no parameter fitting",
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    atomic_write_json(args.output / "protocol.json", protocol, sort_keys=False, allow_nan=False)
    for source in sources:
        destination = args.output / "source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    print("loading immutable runtime and report vintages", flush=True)
    runtime = load_runtime_slice(args.runtime, args.start, args.end, preload_rows=100000)
    events, identity = load_financial_events(args.financial, runtime.stock_codes)
    atomic_write_json(args.output / "financial_identity.json", identity, sort_keys=False, allow_nan=False)
    atomic_write_json(args.output / "runtime_manifest.json", runtime.manifest.as_dict(), sort_keys=False, allow_nan=False)
    months = runtime.trade_dates.astype("datetime64[M]")
    calendar = load_runtime_calendar(args.runtime)
    calendar_months = calendar.astype("datetime64[M]")
    month_end_dates = calendar[:-1][calendar_months[:-1] != calendar_months[1:]]
    signal_rows = np.flatnonzero(months[:-1] != months[1:])
    signal_rows = signal_rows[(signal_rows >= runtime.decision_start - 1) & (signal_rows < runtime.decision_stop)]
    # Runtime share-capital contract is lag one, also at a close decision.
    score_values = calculate_financial_scores(runtime.trade_dates[signal_rows], runtime.field("close")[signal_rows],
                                              runtime.field("total_share")[signal_rows - 1], events)
    np.savez_compressed(args.output / "monthly_scores.npz", dates=runtime.trade_dates[signal_rows],
                        stock_codes=runtime.stock_codes, **score_values)
    coverage = {}
    for name in args.names:
        covered = np.isfinite(score_values[name])
        member = (runtime.field("listing_age")[signal_rows] >= 0) & ~runtime.field("delisted_mask")[signal_rows]
        by_year = {}
        for year in np.unique(runtime.trade_dates[signal_rows].astype("datetime64[Y]")):
            take = runtime.trade_dates[signal_rows].astype("datetime64[Y]") == year
            denominator = int(member[take].sum())
            by_year[str(year)] = {"member_months": denominator, "available_member_months": int((covered[take] & member[take]).sum()),
                                  "coverage_or_threshold_pass_rate": float((covered[take] & member[take]).sum() / denominator) if denominator else None}
        coverage[name] = {"by_year": by_year, "all_missing_signal_dates": runtime.trade_dates[signal_rows][~covered.any(axis=1)].astype(str).tolist()}
    atomic_write_json(args.output / "coverage.json", coverage, sort_keys=False, allow_nan=False)
    results = []
    for name in args.names:
        print(json.dumps({"event": "replay", "name": name}), flush=True)
        panel = np.full((runtime.n_dates, runtime.n_stocks), np.nan, dtype=np.float32)
        panel[signal_rows] = score_values[name]
        result = run_monthly_video_replay(runtime, panel, name=name, selection_fraction=.1 if name in DECILE_NAMES else 1.0, output=args.output / name,
                                         month_end_dates=month_end_dates)
        del panel
        results.append(result)
        atomic_write_json(args.output / "partial_results.json", results, sort_keys=False, allow_nan=False)
        print(json.dumps({"name": name, "annualized_return": result["annualized_return"], "max_drawdown": result["max_drawdown"]}), flush=True)
    atomic_write_json(args.output / "summary.json", {"protocol": protocol, "results": results, "elapsed_seconds": time.monotonic() - started}, sort_keys=False, allow_nan=False)


if __name__ == "__main__":
    main()

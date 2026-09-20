"""Offline stock replay of the author's cash-scaled abnormal gross profit rule."""
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
from factor.library.abnormal_gross_profit import abnormal_gross_profit, ABNORMAL_GROSS_PROFIT_VERSION
from offline_data import load_runtime_slice
from offline_data.financial_versions import load_financial_events, iter_financial_fields, ABNORMAL_GROSS_PROFIT_FIELD_SET
from offline_data.industry_history import load_industry_history, build_industry_panel
from offline_data.runtime import load_runtime_calendar
from utils.atomic_file import atomic_write_json


INDUSTRIES = {"steel": 230000, "electronics": 270000, "power_equipment": 630000,
              "banks": 480000, "textiles": 350000, "home_appliances": 330000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--financial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2021-12-31")
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("Use a new evidence directory; completed runs are immutable")
    sources = [Path(__file__), Path("factor/library/abnormal_gross_profit.py"),
               Path("offline_data/financial_versions.py"), Path("offline_data/industry_history.py"),
               Path("offline_data/runtime.py"), Path("offline_data/contracts.py"),
               *[Path("env") / f"{name}.py" for name in
                 ("calendar_replay", "planner", "simulator", "legality", "fees", "quantity", "action_schema", "metrics")]]
    protocol = {"schema": "abnormal-gross-profit-monthly-stock-research-v1", "factor_version": ABNORMAL_GROSS_PROFIT_VERSION,
        "start": args.start, "end": args.end, "formula": "(GP_q - GP_q_minus_4 * sales_cash_q / sales_cash_q_minus_4) / assets_q",
        "selection": "month-end descending top floor(valid eligible count * .05); low uses negative score",
        "schedule": "month-end signal, next-month first open buy, scheduled month-end close sell",
        "financial_availability": "independent table announcements strictly before signal day; latest common quarter; YTD flows converted to single quarter; missing stays missing",
        "industry_codes": INDUSTRIES, "industry_timing": "historical dated membership, usable strictly after start_date; downloaded retrospective history is not independently certified vintage PIT",
        "industry_baseline": "all eligible historical industry members equal weight monthly, irrespective of financial coverage; not the author's undisclosed industry benchmark",
        "industry_version_warning": "keep historical codes; do not map old combined financial 440000 to banks 480000 or backfill modern sectors",
        "assumptions": "365 calendar-day listing minimum; exclude ST and delisted; prior sales cash and assets positive, current sales cash nonnegative; no minimum-one selection; shared WBR lots/fees/legality and synthetic total-return account",
        "video_comparison": "author original January 2023 publication shows 2010 onward, exact endpoint unspecified; predeclared 2010-2022 descriptive replay approximates it, not a blind holdout or PPO selection set",
        "fees": asdict(DEFAULT_FEE_SCHEDULE), "initial_cash": 1e9, "broker_exact": False,
        "source_approval_scope": "sealed QMT amounts and announcement rows, official dictionary plus limited original report spot checks; not exhaustive issuer PIT certification",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    atomic_write_json(args.output / "protocol.json", protocol, allow_nan=False)
    for p in sources:
        relative = p.resolve().relative_to(Path.cwd())
        destination = args.output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    print("Loading sealed stock, financial and industry data", flush=True)
    runtime = load_runtime_slice(args.runtime, args.start, args.end, preload_rows=100000)
    events, identity = load_financial_events(args.financial, runtime.stock_codes, field_set=ABNORMAL_GROSS_PROFIT_FIELD_SET)
    atomic_write_json(args.output / "financial_identity.json", identity, allow_nan=False)
    atomic_write_json(args.output / "runtime_manifest.json", runtime.manifest.as_dict(), allow_nan=False)
    industry_paths = [Path("data/industry_history/sw_industry_history.parquet"), Path("data/industry_history/sw_industry_history_metadata.json")]
    atomic_write_json(args.output / "industry_identity.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in industry_paths})
    months = runtime.trade_dates.astype("datetime64[M]")
    signal_rows = np.flatnonzero(months[:-1] != months[1:])
    signal_rows = signal_rows[(signal_rows >= runtime.decision_start - 1) & (signal_rows < runtime.decision_stop)]
    signal_dates = runtime.trade_dates[signal_rows]
    raw = np.stack([abnormal_gross_profit(fields) for _, fields in iter_financial_fields(
        signal_dates, runtime.n_stocks, events, field_set=ABNORMAL_GROSS_PROFIT_FIELD_SET)])
    industry = build_industry_panel(signal_dates, runtime.stock_codes, load_industry_history(), level=1)
    np.savez_compressed(args.output / "monthly_scores.npz", dates=signal_dates, stock_codes=runtime.stock_codes,
                        abnormal_gross_profit=raw, historical_industry=industry)
    member = (runtime.field("listing_age")[signal_rows] >= 0) & ~runtime.field("delisted_mask")[signal_rows]
    eligible = member & ~runtime.field("st_mask")[signal_rows] & ((signal_dates[:, None] - runtime.field("issue_date")) >= np.timedelta64(365, "D"))
    coverage = {}
    for name, code in {"all": None, **INDUSTRIES}.items():
        universe = np.ones_like(member) if code is None else industry == code
        valid = np.isfinite(raw) & eligible & universe
        counts = valid.sum(axis=1)
        by_year = {}
        for year in np.unique(signal_dates.astype("datetime64[Y]")):
            take = signal_dates.astype("datetime64[Y]") == year
            denominator = int((member[take] & universe[take]).sum())
            available = int((np.isfinite(raw[take]) & member[take] & universe[take]).sum())
            by_year[str(year)] = {"member_months": denominator, "available_member_months": available,
                                 "coverage": available / denominator if denominator else None}
        coverage[name] = {"by_year": by_year, "signal_dates": signal_dates.astype(str).tolist(),
            "valid_eligible_counts": counts.tolist(), "selected_counts": np.floor(counts * .05).astype(int).tolist(),
            "no_selection_dates": signal_dates[counts < 20].astype(str).tolist()}
    atomic_write_json(args.output / "coverage.json", coverage, allow_nan=False)
    calendar = load_runtime_calendar(args.runtime)
    month_ends = calendar[:-1][calendar[:-1].astype("datetime64[M]") != calendar[1:].astype("datetime64[M]")]
    strategies = [("all_high", raw, .05), ("all_low", -raw, .05)]
    for name, code in INDUSTRIES.items():
        strategies.extend([(name + "_high", np.where(industry == code, raw, np.nan), .05),
                           (name + "_baseline", np.where(industry == code, 1., np.nan), 1.)])
    results = []
    for name, values, fraction in strategies:
        print(json.dumps({"event": "replay", "name": name}), flush=True)
        panel = np.full((runtime.n_dates, runtime.n_stocks), np.nan, dtype=np.float32)
        panel[signal_rows] = values
        result = run_monthly_video_replay(runtime, panel, name=name, output=args.output / name,
            selection_fraction=fraction, month_end_dates=month_ends)
        del panel
        results.append(result)
        atomic_write_json(args.output / "partial_results.json", results, allow_nan=False)
        print(json.dumps({"name": name, "annualized_return": result["annualized_return"], "max_drawdown": result["max_drawdown"]}), flush=True)
    atomic_write_json(args.output / "summary.json", {"protocol": protocol, "results": results,
        "elapsed_seconds": time.monotonic() - started}, allow_nan=False)


if __name__ == "__main__":
    main()

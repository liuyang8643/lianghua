"""Offline, fixed-spec video-factor research through the canonical env session."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from env.backtest import PreparedEpisode, EpisodeSession, run_day_config_episode
from env.action_schema import ActionSchema
from env.fees import DEFAULT_FEE_SCHEDULE
from env.metrics import summarize_event_forward_returns
from env.simulator import settlement_economics
from factor import precompute_factors, factor_coverage
from factor.library.bilibili import prepare_bilibili_factor_definitions
from factor.library.bilibili_events import RESEARCH_FACTOR_DEFINITIONS
from factor.library.bilibili_smallcap import SMALLCAP_FACTOR_DEFINITIONS
from offline_data import load_runtime_slice
from utils.atomic_file import atomic_write_json


def run_lifetime_monthly(args) -> None:
    """Assemble published monthly lifetime signals into the shared env replay."""
    from env.calendar_replay import run_monthly_video_replay
    from env.fees import FeeSchedule
    from factor.library.bilibili import calculate_bilibili_scores, BILIBILI_FACTOR_NAMES
    from offline_data.runtime import load_runtime_calendar

    if args.family != "prices":
        raise ValueError("monthly-lifetime execution requires prices family")
    started = time.perf_counter()
    names = BILIBILI_FACTOR_NAMES[2:]
    fee_schedules = {
        "video_assumed": FeeSchedule(commission_rate=0.86 / 10000,
                                     minimum_commission=0, stamp_tax_rate=1 / 1000,
                                     transfer_fee_rate=0, slippage_rate=0),
        "wbr": DEFAULT_FEE_SCHEDULE,
    }
    sources = [Path(__file__), Path("factor/library/bilibili.py"),
               Path("env/calendar_replay.py"), Path("env/planner.py"),
               Path("env/simulator.py"), Path("env/legality.py"), Path("env/quantity.py"),
               Path("env/scoring.py"), Path("env/fees.py"), Path("env/action_schema.py"),
               Path("env/metrics.py"), Path("offline_data/runtime.py"),
               Path("offline_data/contracts.py"), Path("utils/atomic_file.py")]
    protocol = {
        "schema": "bilibili-lifetime-monthly-v1", "video": "BV1CNLQ6REu5",
        "start": args.start, "end": args.end, "split": args.split,
        "names": names, "initial_cash": 1e9,
        "signal": "completed-close phase, IPO-anchored adjusted lifetime high/low; no next-day membership or price input",
        "selection": "month-end PIT members, no ST/delisted, listed at least 365 calendar days; floor(valid eligible count * 0.1); stable stock-axis ties",
        "execution": "next-month first open equal-weight entry, calendar month-end scheduled close exit; shared env planner/simulator legality, lots and fees",
        "missing": "IPO day required; broken close/preClose chain remains invalid; absent whole bars skipped without asserting proven suspension; no synthetic IPO anchor",
        "price_adjustment": "IPO bar scale=1, subsequent scale=previous adjusted close/preClose; adjusted lifetime amplitude divided by actual issue price",
        "fees": {name: asdict(value) for name, value in fee_schedules.items()},
        "video_fee_assumption": "visible c_rate=.86/10000 and t_rate=1/1000 interpreted as both-side proportional commission and sell-only tax; undisclosed minimum, transfer and slippage set zero for this labeled comparison",
        "differences": "1bn synthetic total-return account, integer board lots, strict IPO/history validity, selected-but-blocked entries not replaced, locked exits retried at later opens; not author's undisclosed raw data/account source",
        "holdout": "only 2014-2021 eligible for current factor recommendation; full video interval descriptive reproduction, no PPO training or selection from validation/test returns",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    atomic_write_json(args.output / "protocol.json", protocol, sort_keys=False, allow_nan=False)
    for source in sources:
        relative = source.resolve().relative_to(Path.cwd())
        destination = args.output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    print(json.dumps({"event": "loading", "split": args.split}), flush=True)
    runtime = load_runtime_slice(args.runtime, args.start, args.end, preload_rows=100000)
    if runtime.manifest.actual_preload_rows >= runtime.manifest.requested_preload_rows:
        raise ValueError("lifetime monthly replay must retain full IPO history")
    atomic_write_json(args.output / "runtime_manifest.json", runtime.manifest.as_dict(), sort_keys=False, allow_nan=False)
    calendar = load_runtime_calendar(args.runtime)
    calendar_months = calendar.astype("datetime64[M]")
    month_end_dates = calendar[:-1][calendar_months[:-1] != calendar_months[1:]]
    months = runtime.trade_dates.astype("datetime64[M]")
    signal_rows = np.flatnonzero(months[:-1] != months[1:])
    signal_rows = signal_rows[(signal_rows >= runtime.decision_start - 1) & (signal_rows < runtime.decision_stop - 1)]
    print(json.dumps({"event": "completed_lifetime_scores", "signal_months": len(signal_rows)}), flush=True)
    all_scores = calculate_bilibili_scores(runtime.trade_dates, runtime.data, output_phase="completed_close")
    monthly_scores = {name: all_scores[name][signal_rows].copy() for name in names}
    del all_scores
    gc.collect()
    np.savez_compressed(args.output / "monthly_scores.npz", dates=runtime.trade_dates[signal_rows],
                        stock_codes=runtime.stock_codes, **monthly_scores)
    member = (runtime.field("listing_age")[signal_rows] >= 0) & ~runtime.field("delisted_mask")[signal_rows]
    coverage = {}
    for name in names:
        valid = np.isfinite(monthly_scores[name]) & member
        coverage[name] = {"by_year": {}, "all_missing_signal_dates": runtime.trade_dates[signal_rows][~valid.any(axis=1)].astype(str).tolist()}
        for year in np.unique(runtime.trade_dates[signal_rows].astype("datetime64[Y]")):
            take = runtime.trade_dates[signal_rows].astype("datetime64[Y]") == year
            denominator = int(member[take].sum())
            numerator = int(valid[take].sum())
            coverage[name]["by_year"][str(year)] = {"member_months": denominator, "available_member_months": numerator,
                                                        "coverage": numerator / denominator if denominator else None}
    atomic_write_json(args.output / "coverage.json", coverage, sort_keys=False, allow_nan=False)
    results = []
    for name in names:
        panel = np.full((runtime.n_dates, runtime.n_stocks), np.nan)
        panel[signal_rows] = monthly_scores[name]
        for fee_name, fees in fee_schedules.items():
            print(json.dumps({"event": "monthly_replay", "factor": name, "fees": fee_name}), flush=True)
            output = args.output / fee_name / name
            result = run_monthly_video_replay(runtime, panel, name=name, selection_fraction=.1,
                                             output=output, fees=fees, month_end_dates=month_end_dates)
            with np.load(output / "trace.npz", allow_pickle=False) as trace:
                years = trace["dates"][1:].astype("datetime64[Y]")
                log_returns = np.diff(np.log(trace["nav"]))
                by_year = {str(year): float(np.expm1(log_returns[years == year].sum())) for year in np.unique(years)}
            result = {**result, "fee_mode": fee_name, "by_year": by_year,
                      "normalized_ending_nav": result["ending_nav"] / result["initial_cash"]}
            atomic_write_json(output / "result.json", result, sort_keys=False, allow_nan=False)
            results.append(result)
            atomic_write_json(args.output / "partial_results.json", results, sort_keys=False, allow_nan=False)
            print(json.dumps({"name": name, "fees": fee_name, "annualized_return": result["annualized_return"],
                              "max_drawdown": result["max_drawdown"], "ending_nav": result["normalized_ending_nav"]}), flush=True)
        del panel
    atomic_write_json(args.output / "summary.json", {"protocol": protocol, "results": results,
                                                     "elapsed_seconds": time.perf_counter() - started}, sort_keys=False, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"))
    parser.add_argument("--family", choices=("prices", "events", "smallcap", "financial"), default="prices")
    parser.add_argument("--execution", choices=("daily", "monthly-lifetime"), default="daily")
    parser.add_argument("--financial", type=Path, help="sealed QMT financial versions, for financial family")
    args = parser.parse_args()
    if args.family == "financial" and args.financial is None:
        parser.error("financial family requires --financial")
    if (args.output / "summary.json").exists():
        raise FileExistsError("completed research output already exists; use a new output directory")
    if args.execution == "monthly-lifetime":
        run_lifetime_monthly(args)
        return
    started = time.perf_counter()
    payload = json.loads(args.config.read_text(encoding="utf-8"))["individual_config"]
    research_buy_n = 10 if args.family == "smallcap" else 30
    sources = [Path("ai/factor_discovery/bilibili_backtest.py"), Path("factor/library/bilibili.py"), Path("factor/library/bilibili_events.py"), Path("factor/library/bilibili_smallcap.py"), Path("factor/compute.py"),
               Path("env/backtest.py"), Path("env/planner.py"), Path("env/simulator.py"),
               Path("env/legality.py"), Path("env/fees.py"), Path("env/scoring.py"),
               Path("env/action_schema.py"), Path("env/metrics.py"), Path("offline_data/runtime.py"), Path("utils/atomic_file.py")]
    if args.family == "financial":
        sources.extend((Path("factor/library/video_financial.py"), Path("offline_data/financial_versions.py")))
    protocol = {
        "research_schema": "bilibili-wbr-fixed-research-v3", "family": args.family, "split": args.split,
        "start": args.start, "end": args.end, "initial_cash": 1_000_000,
        "research_buy_n": research_buy_n, "research_turnover_rate": 1.0,
        "prefilter_n": payload["prefilter_n"], "fees": asdict(DEFAULT_FEE_SCHEDULE),
        "baseline_config": payload, "selection": "video-fixed; no fitting, no performance-based parameter search",
        "execution": "WBR daily equalize + cash sweep; completed T-1 signal, T-open fills, next-open settlement",
        "comparison": "execution adaptation, not an exact monthly close-sale reproduction",
        "accounting": "total_return_reinvested-v1; broker_exact=false",
        "holdout": "independent descriptive research; 2025-2026 previously examined, not a blind test",
        "history": "All families use the same full available strictly prior history for precomputation; decision and settlement rows remain inside the named interval.",
        "event_adaptation": "Binary event ranks; non-event stocks remain eligible for mandatory full investment; ties use canonical rank order, not an economic preference. This is not an event-only trading strategy.",
        "smallcap_adaptation": "Full A-share PIT universe is a local assumption; source pure-smallcap universe unspecified. T-open times lagged share capital, daily equalize; rising MA20 requires 21 completed closes. Invalid signal rows remain fallback candidates under canonical full-investment rules.",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    if args.family == "financial":
        protocol["execution"] = "WBR daily equalize + cash sweep; announcement<T financials, T-open valuation times lagged shares, T-open fills, next-open settlement"
        protocol["history"] = "All prior disclosures are replayed; financial price scores cover decision rows plus the preceding prefilter row. Price filters and baseline retain full strictly prior history."
        protocol["availability_start"] = "Each financial account starts at its first PIT-member valid signal within the requested interval; determined by input availability, never returns. Full-investment fallback remains possible thereafter. Baseline covers the full requested interval and is not a matched-window comparator when starts differ."
        protocol["financial_adaptation"] = (
            "Same financial formulae and dated disclosures as monthly research, evaluated at T-open with lagged shares. "
            "Fixed 30-stock daily equalize replaces monthly variable-count selection. Threshold failures and missing "
            "signals can be canonical full-investment fallback candidates. This is not video stock-count reproduction. "
            "Only limited original-report spot checks support the financial field whitelist; not full-market PIT certification."
        )
    atomic_write_json(args.output / "protocol.json", protocol, sort_keys=False, allow_nan=False, trailing_newline=True)
    if args.family == "financial":
        for source in sources:
            destination = args.output / "source" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    print(json.dumps({"event": "loading", "split": args.split}), flush=True)
    runtime = load_runtime_slice(args.runtime, args.start, args.end, preload_rows=100000)
    atomic_write_json(args.output / "runtime_manifest.json", runtime.manifest.as_dict(), sort_keys=False, allow_nan=False, trailing_newline=True)
    if args.family == "financial":
        from offline_data.financial_versions import load_financial_events
        from factor.library.video_financial import prepare_financial_factor_definitions
        events, financial_identity = load_financial_events(args.financial, runtime.stock_codes)
        atomic_write_json(args.output / "financial_identity.json", financial_identity, sort_keys=False, allow_nan=False, trailing_newline=True)
        print(json.dumps({"event": "financial_scores", "rows": runtime.decision_stop - max(0, runtime.decision_start - 1)}), flush=True)
        definitions = prepare_financial_factor_definitions(runtime, events, financial_identity)
        del events
    else:
        definitions = (prepare_bilibili_factor_definitions(runtime) if args.family == "prices"
                       else RESEARCH_FACTOR_DEFINITIONS if args.family == "events" else SMALLCAP_FACTOR_DEFINITIONS)
    print(json.dumps({"event": "factor_batch"}), flush=True)
    factors = precompute_factors(runtime, definitions=definitions)
    del definitions
    coverage = factor_coverage(runtime, factors)
    atomic_write_json(args.output / "factor_coverage.json", coverage, sort_keys=False, allow_nan=False, trailing_newline=True)
    if args.family == "events":
        start, stop = runtime.decision_start, runtime.decision_stop
        opening, close, preclose = (runtime.field(k)[start:stop] for k in ("open", "close", "preClose"))
        economics = settlement_economics(current_mark=opening[:-1], current_close=close[:-1],
                                          next_preclose=preclose[1:], next_open=opening[1:], diagnostics=False)
        # Statistics require observed prices throughout the horizon; the account
        # backtest below instead retains canonical missing-mark settlement.
        complete = np.ones(economics.gross_return.shape, dtype=bool)
        for value in (opening[:-1], close[:-1], preclose[1:], opening[1:]):
            complete &= np.isfinite(value) & (value > 0)
        gross = np.where(complete, economics.gross_return, np.nan)
        member = ((runtime.field("listing_age")[start:stop] >= 0)
                  & ~runtime.field("delisted_mask")[start:stop])
        event_results = {}
        for index, name in enumerate(factors.factor_names):
            events = (factors.raw[start:stop, index] == 1) & member
            event_results[name] = summarize_event_forward_returns(events, gross, horizons=(1, 3, 5, 10))
        atomic_write_json(args.output / "event_statistics.json", {
            "schema": "bilibili-open-event-statistics-v1", "results": event_results,
            "semantics": "PIT members at entry; completed signal T-1; gross economic T-open to T+h-open; event equal weight, no fees/slippage/legality/portfolio allocation. Missing-price chains excluded and counted; no labels cross split end. Not executable strategy profit or video close-return replication.",
        }, sort_keys=False, allow_nan=False, trailing_newline=True)
        del economics, gross, complete, member, event_results
    results = []

    def run(name, batch, config_payload):
        run_started = time.perf_counter()
        first = runtime.decision_start
        if args.family == "financial" and name != "WBR_static_baseline":
            available_date = coverage["factors"][name]["first_available"]
            if available_date is None:
                raise ValueError(f"{name} has no available financial signal")
            first = max(first, int(np.searchsorted(runtime.trade_dates, np.datetime64(available_date))))
        episode = PreparedEpisode.build(runtime, batch, decision_start=first,
                                        encode_observations=False, prefilter_n=payload["prefilter_n"])
        research_schema = (ActionSchema(factor_names=batch.factor_names, filter_names=batch.filter_names,
                                        fixed_buy_n=10, turnover_maximum=1.0,
                                        schema_version="day-config-bilibili-smallcap-v3-continuous-turnover")
                           if args.family == "smallcap" and name != "WBR_static_baseline" else None)
        session = EpisodeSession(episode, action_schema=research_schema)
        config = session.action_schema.from_static_config(config_payload)
        counter = 0

        def provide(_):
            nonlocal counter
            counter += 1
            if counter % 500 == 0:
                print(json.dumps({"event": "progress", "split": args.split, "factor": name, "transitions": counter}), flush=True)
            return config

        trace = run_day_config_episode(session, provide)
        if not trace.full_investment_contract_satisfied or not np.isfinite(trace.nav).all():
            raise RuntimeError(f"{name} failed finite/full-investment contracts")
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "trace.npz", decision_dates=trace.decision_dates,
                            next_dates=trace.next_decision_dates, nav=trace.nav, cash=trace.cash,
                            exposure=trace.exposure, returns=trace.portfolio_returns,
                            full_investment_contract=trace.full_investment_contract,
                            residual_cash_reasons=trace.residual_cash_reasons)
        with (directory / "fills.jsonl").open("w", encoding="utf-8") as stream:
            for day, fills in zip(trace.decision_dates, trace.fills, strict=True):
                for fill in fills:
                    stream.write(json.dumps({"decision_date": day, **asdict(fill)}, ensure_ascii=False) + "\n")
        fees = [dict(fee) for day in trace.fee_breakdowns for fee in day]
        atomic_write_json(directory / "fee_breakdowns.json", fees, sort_keys=False, allow_nan=False, trailing_newline=True)
        result = {"name": name, "split": args.split, "first_open": trace.decision_dates[0],
                  "last_valuation_open": trace.next_decision_dates[-1],
                  **trace.metrics.as_dict(), "ending_nav": float(trace.nav[-1]),
                  "mean_exposure": float(np.mean(trace.exposure)),
                  "fills": sum(map(len, trace.fills)), "full_investment_contract_satisfied": True,
                  "factor_schema": batch.schema_version, "factor_schema_hash": batch.schema_hash,
                  "action_schema_hash": session.action_schema.schema_hash,
                  "config": session.action_schema.to_static_config(config),
                  "elapsed_seconds": time.perf_counter() - run_started}
        atomic_write_json(directory / "result.json", result, sort_keys=False, allow_nan=False, trailing_newline=True)
        results.append(result)
        atomic_write_json(args.output / "partial_results.json", results, sort_keys=False, allow_nan=False, trailing_newline=True)
        print(json.dumps({"event": "result", "name": name, "metrics": trace.metrics.as_dict()}, ensure_ascii=False), flush=True)

    for name in factors.factor_names:
        config = {**payload, "weights": {f: float(f == name) for f in factors.factor_names},
                  "buy_n": research_buy_n, "turnover_rate": 1.0, "single_buy_pct": 1 / research_buy_n}
        run(name, factors, config)
    del factors
    gc.collect()
    baseline = precompute_factors(runtime)
    run("WBR_static_baseline", baseline, payload)
    atomic_write_json(args.output / "summary.json", {"protocol": protocol, "results": results,
                                            "elapsed_seconds": time.perf_counter() - started}, sort_keys=False, allow_nan=False, trailing_newline=True)


if __name__ == "__main__":
    main()

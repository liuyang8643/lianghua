"""Training-only daily stock-account adaptation of the abnormal-profit video."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from env.backtest import PreparedEpisode, EpisodeSession, run_day_config_episode
from env.fees import DEFAULT_FEE_SCHEDULE
from factor import precompute_factors, factor_coverage
from factor.base import FactorDefinition, FactorMetadata
from factor.registry import PRODUCTION_FACTORS
from factor.library.abnormal_gross_profit import abnormal_gross_profit, ABNORMAL_GROSS_PROFIT_VERSION
from offline_data import load_runtime_slice
from offline_data.financial_versions import (
    ABNORMAL_GROSS_PROFIT_FIELD_SET, load_financial_events, iter_financial_fields,
)
from utils.atomic_file import atomic_write_json


def _write(path, value):
    atomic_write_json(path, value, sort_keys=False, allow_nan=False, trailing_newline=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--financial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"))
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise FileExistsError("completed research output exists")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.config.read_text(encoding="utf-8"))["individual_config"]
    fees = replace(DEFAULT_FEE_SCHEDULE, slippage_rate=.001)
    sources = [Path(__file__), args.config, Path("configs/strategy.yaml"),
               *[Path(p) for p in ("offline_data/financial_versions.py", "offline_data/runtime.py",
                 "offline_data/contracts.py", "factor/library/abnormal_gross_profit.py",
                 "factor/base.py", "factor/compute.py", "factor/registry.py",
                 "env/backtest.py", "env/action_schema.py", "env/planner.py", "env/simulator.py",
                 "env/scoring.py", "env/legality.py", "env/fees.py", "env/quantity.py", "env/metrics.py")]]
    sources.extend(sorted(Path("factor/library").glob("*.py")))
    sources = list(dict.fromkeys(sources))
    protocol = {
        "schema": "abnormal-gross-profit-daily-training-v1", "start": "2014-01-01", "end": "2021-12-31",
        "split": "train", "initial_cash": 1_000_000., "fees": asdict(fees), "static_config": payload,
        "factor_version": ABNORMAL_GROSS_PROFIT_VERSION, "financial_field_set": ABNORMAL_GROSS_PROFIT_FIELD_SET,
        "signal": "single-quarter abnormal gross profit, announcements strictly before each T-open; no monthly backfill",
        "execution": "canonical WBR daily equalize, T-open fills and next-open total-return synthetic settlement",
        "selection": "current config fixed buy_n/turnover_rate, filters, prefilter, limit protection and band; only weights change",
        "missing": "NaN operands stay NaN; canonical full-investment fallback retained and coverage reported",
        "comparison": "daily stock adaptation, not video monthly 5% selection or industry grouping",
        "holdout": "2014-2021 only; no validation/test account or PPO model is opened",
        "baseline_zero_weights": "Only nonzero-weight production definitions are precomputed; zero-weight factors contribute neither score nor available weight. Original full config and effective config are both sealed.",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    _write(args.output / "protocol.json", protocol)
    for source in sources:
        relative = source.resolve().relative_to(Path.cwd())
        destination = args.output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    print(json.dumps({"event": "load_runtime"}), flush=True)
    runtime = load_runtime_slice(args.runtime, "2014-01-01", "2021-12-31", preload_rows=100000)
    _write(args.output / "runtime_manifest.json", runtime.manifest.as_dict())
    print(json.dumps({"event": "load_financial"}), flush=True)
    events, identity = load_financial_events(args.financial, runtime.stock_codes,
                                            field_set=ABNORMAL_GROSS_PROFIT_FIELD_SET)
    _write(args.output / "financial_identity.json", identity)
    first, stop = max(0, runtime.decision_start - 1), runtime.decision_stop
    scores = np.full((runtime.n_dates, runtime.n_stocks), np.nan, dtype=np.float64)
    for row, (_, fields) in enumerate(iter_financial_fields(runtime.trade_dates[first:stop], runtime.n_stocks,
                                                           events, field_set=ABNORMAL_GROSS_PROFIT_FIELD_SET), first):
        scores[row] = abnormal_gross_profit(fields)
    scores.flags.writeable = False
    del events
    np.savez_compressed(args.output / "scores.npz", dates=runtime.trade_dates[first:stop],
                        stock_codes=runtime.stock_codes, raw=scores[first:stop])
    names = ("HighAbnormalGrossProfit", "LowAbnormalGrossProfit")
    binding = {"runtime": runtime.manifest.source_sha256, "financial": identity["sha256"],
               "source": protocol["source_sha256"], "computed_rows": [first, stop],
               "dates": runtime.trade_dates[[0, -1]].astype(str).tolist(), "stock_codes": runtime.stock_codes}

    def implementation(direction):
        bound = scores if direction == 1 else -scores
        bound.flags.writeable = False

        class BoundAbnormalGrossProfit:
            def calc_batch(self, panel):
                for field in ("listing_age", "delisted_mask"):
                    if panel[field] is not runtime.field(field) or panel[field].flags.writeable:
                        raise ValueError("abnormal factor bound to different or mutable runtime")
                return bound
        return BoundAbnormalGrossProfit

    definitions = tuple(FactorDefinition(
        metadata=FactorMetadata(name=name, version=ABNORMAL_GROSS_PROFIT_VERSION, hist_days=0,
            required_fields=("listing_age", "delisted_mask"), implementation_hash=hashlib.sha256(
                json.dumps({**binding, "direction": direction}, sort_keys=True).encode()).hexdigest()),
        implementation=implementation(direction), raw_runtime_view=True)
        for name, direction in zip(names, (1, -1), strict=True))
    print(json.dumps({"event": "factor_batch"}), flush=True)
    factors = precompute_factors(runtime, definitions=definitions)
    coverage = factor_coverage(runtime, factors)
    _write(args.output / "factor_coverage.json", coverage)
    for name in names:
        first_available = coverage["factors"][name]["first_available"]
        if first_available is None or np.datetime64(first_available) > runtime.trade_dates[runtime.decision_start]:
            raise ValueError("training start lacks a valid signal; matched-period comparison would be false")
    results = []

    def run(name, batch, config_payload):
        start = time.perf_counter()
        print(json.dumps({"event": "replay", "name": name}), flush=True)
        episode = PreparedEpisode.build(runtime, batch, encode_observations=False,
                                        prefilter_n=payload["prefilter_n"])
        session = EpisodeSession(episode, initial_cash=1_000_000., fees=fees)
        config = session.action_schema.from_static_config(config_payload)
        trace = run_day_config_episode(session, lambda _: config)
        if not trace.full_investment_contract_satisfied or not np.isfinite(trace.nav).all():
            raise RuntimeError("finite/full-investment contract failed")
        target = args.output / name
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / "trace.npz", decision_dates=trace.decision_dates,
                            next_dates=trace.next_decision_dates, nav=trace.nav, cash=trace.cash,
                            exposure=trace.exposure, returns=trace.portfolio_returns,
                            full_investment_contract=trace.full_investment_contract,
                            residual_cash_reasons=trace.residual_cash_reasons)
        with (target / "fills.jsonl").open("w", encoding="utf-8") as stream:
            for date, day_fills in zip(trace.decision_dates, trace.fills, strict=True):
                for fill in day_fills:
                    stream.write(json.dumps({"decision_date": date, **asdict(fill)}, ensure_ascii=False) + "\n")
        _write(target / "fee_breakdowns.json", [dict(fee) for day in trace.fee_breakdowns for fee in day])
        years = np.asarray(trace.next_decision_dates, dtype="datetime64[D]").astype("datetime64[Y]")
        log_returns = np.diff(np.log(trace.nav))
        result = {"name": name, "first_open": trace.decision_dates[0],
                  "last_valuation_open": trace.next_decision_dates[-1], **trace.metrics.as_dict(),
                  "ending_nav": float(trace.nav[-1]), "mean_exposure": float(np.mean(trace.exposure)),
                  "fills": sum(map(len, trace.fills)), "full_investment_contract_satisfied": True,
                  "by_year": {str(year): float(np.expm1(log_returns[years == year].sum())) for year in np.unique(years)},
                  "factor_schema": batch.schema_version, "factor_schema_hash": batch.schema_hash,
                  "action_schema_hash": session.action_schema.schema_hash,
                  "config": session.action_schema.to_static_config(config), "elapsed_seconds": time.perf_counter() - start}
        _write(target / "result.json", result)
        results.append(result)
        _write(args.output / "partial_results.json", results)
        print(json.dumps({"event": "result", **result}, ensure_ascii=False), flush=True)

    for name in names:
        run(name, factors, {**payload, "weights": {other: float(other == name) for other in names}})
    del factors, definitions, scores
    gc.collect()
    print(json.dumps({"event": "baseline_factors"}), flush=True)
    active = tuple(definition for definition in PRODUCTION_FACTORS if payload["weights"][definition.metadata.name] != 0)
    baseline = precompute_factors(runtime, definitions=active)
    run("WBR_static_baseline", baseline,
        {**payload, "weights": {name: payload["weights"][name] for name in baseline.factor_names}})
    _write(args.output / "summary.json", {"protocol": protocol, "results": results,
                                          "elapsed_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()

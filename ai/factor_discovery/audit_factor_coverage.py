"""Coverage-only audit of all production factors; no account or return evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from factor import precompute_factors, factor_coverage, PRODUCTION_FACTORS
from offline_data import load_runtime_slice
from offline_data.runtime import load_runtime_calendar


def audit(runtime_path: Path, output: Path) -> dict:
    calendar = load_runtime_calendar(runtime_path)
    runtime = load_runtime_slice(runtime_path, calendar[0], calendar[-1],
                                 preload_rows=max(item.metadata.hist_days for item in PRODUCTION_FACTORS))
    factors = precompute_factors(runtime)
    result = factor_coverage(runtime, factors)
    member = (runtime.field("listing_age") >= 0) & ~runtime.field("delisted_mask")
    complete = np.all(factors.validity, axis=1) & member
    counts = member.sum(axis=1)
    years = runtime.trade_dates.astype("datetime64[Y]").astype(str)

    def summarize(mask):
        coverage = np.divide(mask.sum(axis=1), counts, out=np.zeros(len(counts)), where=counts > 0)
        return {
            year: {
                "member_cells": int(counts[years == year].sum()),
                "available_cells": int(mask[years == year].sum()),
                "coverage": float(mask[years == year].sum() / counts[years == year].sum()),
                "daily_min": float(coverage[years == year].min()),
                "daily_p10": float(np.quantile(coverage[years == year], 0.1)),
                "all_missing_days": int(np.count_nonzero(mask[years == year].sum(axis=1) == 0)),
            } for year in np.unique(years) if counts[years == year].sum() > 0
        }

    result["simultaneously_valid"] = summarize(complete)
    result["daily_coverage_by_factor_year"] = {
        name: summarize(factors.validity[:, i, :] & member)
        for i, name in enumerate(factors.factor_names)
    }
    result["runtime_file_sha256"] = runtime.manifest.source_sha256
    result["purpose"] = "coverage_only_period_design_no_reward_nav_or_holdout_performance"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.runtime, args.output)
    print(json.dumps({"output": str(args.output), "intersection": report["simultaneously_valid"]}))

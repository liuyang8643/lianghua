"""Sealed train/validation/test research for industry-peer mean reversion.

The signal implementation lives in
``factor_db.factors.IndustryPeerReversalStrict`` and is passed to the existing
A-share score, legality, rebalance, account, fee, and next-open execution
pipeline.  The only extra in-memory field is a point-in-time ``industry_id``
panel built from the official Shenwan event history.  No core runtime schema or
trading rule is changed.

Parameter search is a small deterministic grid.  The runtime supplied to the
training calculation is physically cut at the training end date.  Exactly one
candidate is frozen from robust training Calmar; validation and test are then
run as diagnostics and never feed candidate selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtest import _backtest_direct, _compute_factor_scores
from core.metrics import compute_core_metrics
from core.runtime import latest_runtime_npz_path, load_runtime_npz
from data.update_industry_history import (
    build_industry_panel,
    load_industry_history,
)
from factor_db.factors.IndustryPeerReversalStrict import (
    IndustryPeerReversal20Strict,
    IndustryPeerReversal60Strict,
    IndustryPeerReversal120Strict,
    _leave_one_out_peer_returns,
    _official_log_returns,
)

DEFAULT_OUTPUT = ROOT / "results" / "stat_arb_industry" / "summary.json"
FACTOR_CLASSES = {
    cls.__name__: cls
    for cls in (
        IndustryPeerReversal20Strict,
        IndustryPeerReversal60Strict,
        IndustryPeerReversal120Strict,
    )
}
DEFAULT_POOL_PREFIXES = ("60", "00", "30", "688")


@dataclass(frozen=True)
class DatePeriod:
    start: np.datetime64
    end: np.datetime64

    def __post_init__(self) -> None:
        if np.isnat(self.start) or np.isnat(self.end) or self.start > self.end:
            raise ValueError("invalid date period")


@dataclass(frozen=True)
class IndustryCandidate:
    factor: str
    level: str
    buy_n: int
    sell_m: int
    holding_period: int

    def __post_init__(self) -> None:
        if self.factor not in FACTOR_CLASSES:
            raise ValueError(f"unknown factor: {self.factor}")
        if self.level not in {"l2", "l3"}:
            raise ValueError("level must be l2 or l3")
        if self.buy_n <= 0 or self.sell_m < self.buy_n:
            raise ValueError("require 0 < buy_n <= sell_m")
        if self.holding_period <= 0:
            raise ValueError("holding_period must be positive")


def candidate_grid() -> list[IndustryCandidate]:
    """Small auditable search space; trading costs are not optimized."""

    return [
        IndustryCandidate(
            factor=factor,
            level=level,
            buy_n=buy_n,
            sell_m=buy_n * 2,
            holding_period=holding,
        )
        for factor in FACTOR_CLASSES
        for level in ("l2", "l3")
        for buy_n in (10, 20)
        for holding in (5, 10)
    ]


def derive_periods(
    trade_dates: np.ndarray,
    membership_start: object,
) -> dict[str, DatePeriod]:
    """Use the repository protocol when coverage permits, else causal 60/20/20."""

    dates = np.unique(np.asarray(trade_dates, dtype="datetime64[D]"))
    dates = dates[~np.isnat(dates)]
    dates.sort()
    membership_day = np.datetime64(membership_start, "D")
    dates = dates[dates >= membership_day]
    if len(dates) < 30:
        raise ValueError("insufficient common kline/industry coverage")
    if dates[0] <= np.datetime64("2010-01-01") and dates[-1] >= np.datetime64(
        "2023-12-31"
    ):
        return {
            "train": DatePeriod(np.datetime64("2010-01-01"), np.datetime64("2018-12-31")),
            "validation": DatePeriod(
                np.datetime64("2019-01-01"), np.datetime64("2022-12-31")
            ),
            "test": DatePeriod(np.datetime64("2023-01-01"), dates[-1]),
        }
    train_stop = max(1, min(len(dates) - 2, int(math.floor(len(dates) * 0.60))))
    validation_stop = max(
        train_stop + 1,
        min(len(dates) - 1, int(math.floor(len(dates) * 0.80))),
    )
    return {
        "train": DatePeriod(dates[0], dates[train_stop - 1]),
        "validation": DatePeriod(dates[train_stop], dates[validation_stop - 1]),
        "test": DatePeriod(dates[validation_stop], dates[-1]),
    }


def _calmar(daily_returns_pct: np.ndarray) -> tuple[float, dict]:
    metrics = compute_core_metrics(daily_returns_pct)
    drawdown = abs(float(metrics["max_drawdown"]))
    calmar = float(metrics["annualized"]) / drawdown if drawdown > 0 else 0.0
    return calmar, metrics


def robust_training_fitness(
    daily_returns_pct: Iterable[float],
) -> tuple[float, float, list[float]]:
    """Half full-period Calmar plus half the worst chronological third."""

    daily = np.asarray(list(daily_returns_pct), dtype=np.float64)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 30:
        return float("-inf"), float("-inf"), []
    full, _ = _calmar(daily)
    folds = [chunk for chunk in np.array_split(daily, 3) if len(chunk) >= 10]
    fold_calmars = [_calmar(chunk)[0] for chunk in folds]
    if not fold_calmars:
        return full, full, []
    fitness = 0.5 * full + 0.5 * min(fold_calmars)
    return float(full), float(fitness), [float(value) for value in fold_calmars]


def summarize_backtest(result: dict) -> dict:
    daily = np.asarray(result.get("daily_returns", []), dtype=np.float64)
    calmar, metrics = _calmar(daily)
    terminal = (
        float(np.prod(1.0 + daily / 100.0) - 1.0) * 100.0 if len(daily) else 0.0
    )
    exposures = np.asarray(result.get("daily_exposures", []), dtype=np.float64)
    exposure = float(np.nanmean(exposures)) if len(exposures) else 0.0
    return {
        "trading_days": int(len(daily)),
        "total_return": terminal,
        "annualized": float(metrics["annualized"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "sharpe": float(metrics["sharpe"]),
        "calmar": float(calmar),
        "average_exposure": exposure,
    }


def select_training_candidate(records: Iterable[dict]) -> dict:
    """Deterministic train-only selection with an anti-cash exposure floor."""

    eligible = []
    for record in records:
        fitness = float(record.get("fitness", float("-inf")))
        exposure = float(record.get("average_exposure", 0.0))
        constrained = fitness if exposure >= 0.50 else -1000.0 + exposure
        candidate = IndustryCandidate(**record["candidate"])
        key = (
            constrained,
            fitness,
            min(record.get("fold_calmars") or [float("-inf")]),
            float(record.get("calmar", float("-inf"))),
            float(record.get("sharpe", float("-inf"))),
            exposure,
            -candidate.buy_n,
            -candidate.holding_period,
            candidate.factor,
            candidate.level,
        )
        eligible.append((key, record))
    if not eligible:
        raise ValueError("no training candidates")
    selected = max(eligible, key=lambda value: value[0])[1]
    if not np.isfinite(float(selected.get("fitness", float("-inf")))):
        raise ValueError("no candidate has a finite training fitness")
    return selected


def _runtime_dates() -> np.ndarray:
    with np.load(latest_runtime_npz_path(), allow_pickle=False) as runtime:
        return runtime["trade_dates"].astype("datetime64[D]")


def _period_datetimes(period: DatePeriod, runtime_dates: np.ndarray) -> list[datetime]:
    dates = runtime_dates[(runtime_dates >= period.start) & (runtime_dates <= period.end)]
    return [datetime.fromisoformat(str(value)) for value in dates]


def _load_sealed_period_data(
    period: DatePeriod,
    runtime_dates: np.ndarray,
    memberships,
    *,
    max_lookback: int,
) -> tuple[dict, list[datetime]]:
    backtest_dates = _period_datetimes(period, runtime_dates)
    if not backtest_dates:
        raise ValueError(f"period has no runtime dates: {period}")
    data = load_runtime_npz(
        backtest_dates,
        max_lookback=max_lookback,
        strict_end=True,
    )
    if data is None:
        raise FileNotFoundError("runtime data unavailable")
    # Future membership events are physically removed even though the panel
    # builder would not activate them before their effective date.
    membership_cutoff = np.datetime64(period.end, "D")
    sealed_memberships = memberships.loc[
        memberships["start_date"].to_numpy(dtype="datetime64[D]") <= membership_cutoff
    ].copy()
    data["_sealed_memberships"] = sealed_memberships
    return data, backtest_dates


def _score_context(
    data: dict,
    backtest_dates: list[datetime],
    candidate: IndustryCandidate,
) -> tuple:
    factor_class = FACTOR_CLASSES[candidate.factor]
    data["industry_id"] = build_industry_panel(
        data["trade_dates"],
        data["stock_codes"],
        data["_sealed_memberships"],
        level=candidate.level,
    )
    all_stocks = [
        str(code)
        for code in data["stock_codes"]
        if str(code).startswith(DEFAULT_POOL_PREFIXES)
    ]
    return _compute_factor_scores(
        backtest_dates,
        all_stocks,
        {candidate.factor: 1.0},
        [factor_class],
        data=data,
        enable_nan_filter=True,
    )


def _run_context(context: tuple, candidate: IndustryCandidate, *, slippage_bps: float) -> dict:
    (
        data,
        scores,
        filter_masks,
        valid_dates,
        date_indices,
        valid_stocks,
        stock_indices,
    ) = context
    return _backtest_direct(
        data,
        scores,
        valid_dates,
        date_indices,
        valid_stocks,
        stock_indices,
        {candidate.factor: 1.0},
        candidate.buy_n,
        candidate.sell_m,
        holding_period=candidate.holding_period,
        lightweight=True,
        filter_masks=filter_masks,
        slippage_bps=slippage_bps,
    )


def _evaluate_training(
    period: DatePeriod,
    runtime_dates: np.ndarray,
    memberships,
    candidates: list[IndustryCandidate],
    *,
    slippage_bps: float,
) -> list[dict]:
    max_lookback = max(FACTOR_CLASSES[c.factor].hist_days for c in candidates)
    data, dates = _load_sealed_period_data(
        period,
        runtime_dates,
        memberships,
        max_lookback=max_lookback,
    )
    records: list[dict] = []
    groups: dict[tuple[str, str], list[IndustryCandidate]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.factor, candidate.level), []).append(candidate)
    for group in groups.values():
        context = _score_context(data, dates, group[0])
        for candidate in group:
            result = _run_context(context, candidate, slippage_bps=slippage_bps)
            summary = summarize_backtest(result)
            full_calmar, fitness, folds = robust_training_fitness(
                result["daily_returns"]
            )
            records.append(
                {
                    "candidate": asdict(candidate),
                    **summary,
                    "calmar": full_calmar,
                    "fitness": fitness,
                    "fold_calmars": folds,
                }
            )
        del context
        gc.collect()
    return records


def _evaluate_frozen_period(
    period: DatePeriod,
    runtime_dates: np.ndarray,
    memberships,
    candidate: IndustryCandidate,
    *,
    cost_grid: Iterable[float],
) -> tuple[dict, tuple]:
    factor = FACTOR_CLASSES[candidate.factor]
    data, dates = _load_sealed_period_data(
        period,
        runtime_dates,
        memberships,
        max_lookback=factor.hist_days,
    )
    context = _score_context(data, dates, candidate)
    results = {
        f"slippage_{float(cost):g}bps": summarize_backtest(
            _run_context(context, candidate, slippage_bps=float(cost))
        )
        for cost in cost_grid
    }
    return results, context


def current_deviation_diagnostics(
    context: tuple,
    candidate: IndustryCandidate,
    *,
    top_n: int = 20,
) -> dict:
    """Rank next-open laggards/leaders and expose beta/correlation diagnostics."""

    data = context[0]
    factor_class = FACTOR_CLASSES[candidate.factor]
    factor = factor_class()
    close = np.asarray(data["close"])
    pre_close = np.asarray(data["preClose"])
    industry = np.asarray(data["industry_id"])
    extended = {
        "close": np.vstack([close, np.full((1, close.shape[1]), np.nan)]),
        "preClose": np.vstack([pre_close, np.full((1, close.shape[1]), np.nan)]),
        "industry_id": np.vstack([industry, industry[-1:]]),
    }
    current_scores = factor.calc_batch(extended)[-1].astype(np.float64)

    returns, return_valid = _official_log_returns(close, pre_close)
    peer, pair_valid = _leave_one_out_peer_returns(
        returns,
        return_valid,
        industry,
        factor.min_peers,
    )
    estimation = factor.estimation_window
    recent = factor.deviation_window
    start = len(close) - estimation - recent
    estimate_stop = len(close) - recent
    if start < 0:
        return {"completed_through": str(data["trade_dates"][-1]), "laggards": [], "leaders": []}

    codes = np.asarray(data["stock_codes"]).astype(str)
    names = np.asarray(data.get("stock_names", np.full(len(codes), ""))).astype(str)
    finite_columns = np.flatnonzero(np.isfinite(current_scores))

    def detail(column: int) -> dict:
        x = peer[start:estimate_stop, column]
        y = returns[start:estimate_stop, column]
        valid = pair_valid[start:estimate_stop, column]
        if int(valid.sum()) != estimation:
            correlation = float("nan")
            beta = float("nan")
        else:
            centered_x = x - x.mean()
            centered_y = y - y.mean()
            denom = float(np.sqrt(np.sum(centered_x**2) * np.sum(centered_y**2)))
            correlation = float(np.sum(centered_x * centered_y) / denom) if denom > 0 else float("nan")
            variance_x = float(np.sum(centered_x**2))
            beta = float(np.sum(centered_x * centered_y) / variance_x) if variance_x > 0 else float("nan")
        stock_name = names[column]
        if "\ufffd" in stock_name:
            stock_name = ""
        return {
            "stock_code": codes[column],
            "stock_name": stock_name,
            "industry_code": f"{int(industry[-1, column]):06d}",
            "reversal_score": float(current_scores[column]),
            "rolling_correlation": correlation,
            "beta": beta,
        }

    order = finite_columns[np.argsort(current_scores[finite_columns])]
    return {
        "completed_through": str(np.datetime64(data["trade_dates"][-1], "D")),
        "intended_execution": "next valid A-share open",
        "laggards": [detail(int(column)) for column in order[-top_n:][::-1]],
        "leaders": [detail(int(column)) for column in order[:top_n]],
    }


def _period_json(periods: Mapping[str, DatePeriod]) -> dict:
    return {
        name: {"start": str(period.start), "end": str(period.end)}
        for name, period in periods.items()
    }


def run_research(
    *,
    output: Path = DEFAULT_OUTPUT,
    slippage_bps: float = 20.0,
    cost_grid: Iterable[float] = (10.0, 20.0, 30.0, 50.0),
) -> dict:
    runtime_dates = _runtime_dates()
    memberships = load_industry_history()
    periods = derive_periods(runtime_dates, memberships["start_date"].min())
    candidates = candidate_grid()
    training = _evaluate_training(
        periods["train"],
        runtime_dates,
        memberships,
        candidates,
        slippage_bps=slippage_bps,
    )
    selected_record = select_training_candidate(training)
    selected = IndustryCandidate(**selected_record["candidate"])

    validation, validation_context = _evaluate_frozen_period(
        periods["validation"],
        runtime_dates,
        memberships,
        selected,
        cost_grid=cost_grid,
    )
    del validation_context
    gc.collect()
    test, test_context = _evaluate_frozen_period(
        periods["test"],
        runtime_dates,
        memberships,
        selected,
        cost_grid=cost_grid,
    )
    current = current_deviation_diagnostics(test_context, selected)
    del test_context
    gc.collect()

    summary = {
        "research_status": "sealed_out_of_sample",
        "contract": {
            "integration": "existing core score/legality/rebalance/account/fee engine",
            "runtime_change": "none; PIT industry_id injected in research memory only",
            "signal_time": "open T uses completed close/preClose and industry only through T-1",
            "industry_membership": (
                "official effective event becomes usable on first supplied trading day strictly after it"
            ),
            "peer_return": "daily equal-weight leave-one-out industry log return",
            "model": "OLS alpha/beta/correlation estimate then independent recent residual z-score",
            "portfolio": "A-share long-only laggard selection; not market-neutral short book",
            "selection": "training robust Calmar only; validation/test never tune or select",
            "sealed_data": "each period runtime and membership events physically end at period.end",
            "costs": "existing A-share fees plus stated slippage; cost is sensitivity, not search parameter",
            "known_limitations": [
                "official history is an effective-date file, not archived release vintages",
                "current engine is long-only and has no explicit industry-neutral exposure constraint",
            ],
        },
        "periods": _period_json(periods),
        "search_space": [asdict(candidate) for candidate in candidates],
        "search_slippage_bps": slippage_bps,
        "training_results": training,
        "selected_config": asdict(selected),
        "selected_training_result": selected_record,
        "validation_cost_sensitivity": validation,
        "test_cost_sensitivity": test,
        "current_deviations": current,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "selected": asdict(selected),
                "validation": validation,
                "test": test,
            },
            ensure_ascii=False,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    args = parser.parse_args()
    run_research(output=args.output, slippage_bps=args.slippage_bps)


if __name__ == "__main__":
    main()

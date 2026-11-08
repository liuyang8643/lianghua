"""Video financial formulae over explicitly dated, immutable report vintages."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from offline_data.financial_versions import FinancialEvents, FINANCIAL_REPLAY_VERSION, iter_financial_fields
from factor.library.financial import (
    FINANCIAL_SEMANTICS_VERSION, LowCashOutflowProfitGrowthSpread,
    HighOperatingProfitRevenueGrowthSpread,
)


VIDEO_FINANCIAL_NAMES = (
    "PRBelowOne", "LowPRDecile", "PBBelowTwoROEAbove10",
    "ROEAbove10", "ROEBelow10", "ROEAbove15", "ROEAbove20", "HighROEDecile",
    "LowCashOutflowProfitGrowthSpread", "HighOperatingProfitRevenueGrowthSpread",
    "HighInventoryOtherPayable", "HighOperatingCashSurplus", "HighReceivableOtherPayable",
)
DECILE_NAMES = frozenset(("LowPRDecile", "HighROEDecile", *VIDEO_FINANCIAL_NAMES[8:]))
SEMANTICS_VERSION = "video-qmt-financial-vintages-v2-shared-replay-and-scores"


def _ratio(numerator, denominator):
    result = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan)
    np.divide(numerator, denominator, out=result,
              where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0))
    return result


def calculate_financial_scores(dates: np.ndarray, prices: np.ndarray, shares: np.ndarray,
                               events: Mapping[str, FinancialEvents]) -> dict[str, np.ndarray]:
    """Use disclosures strictly before each date, with zero future backfill.

    Prices are supplied by the decision phase: month-end closes for the video
    protocol, opening prices for the later daily adaptation. Shares must be
    causal at that phase. ROE=TTM parent profit / latest matching ending parent
    equity; this denominator choice is explicit because the video omits it.
    All cash-flow level terms use the latest common YTD report. Growth terms
    are single-quarter YoY and use abs(prior quarter) as their denominator.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    if prices.shape != shares.shape or prices.shape[0] != len(dates):
        raise ValueError("financial score axes differ")
    if np.isnat(dates).any() or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("dates must be finite, sorted and unique")
    n = prices.shape[1]
    result = {name: np.full(prices.shape, np.nan, dtype=np.float32) for name in VIDEO_FINANCIAL_NAMES}
    for row, (_, fields) in enumerate(iter_financial_fields(dates, n, events)):
        profit = fields["financial_profit_ttm"]
        equity = fields["financial_equity"]
        cap = np.where((prices[row] > 0) & (shares[row] > 0), prices[row] * shares[row], np.nan)
        pe, pb, roe = _ratio(cap, profit), _ratio(cap, equity), _ratio(profit, equity)
        pr = _ratio(pe, roe * 100)
        result["PRBelowOne"][row] = np.where(pr < 1, -pr, np.nan)
        result["LowPRDecile"][row] = -pr
        result["PBBelowTwoROEAbove10"][row] = np.where((pb < 2) & (roe > .1), -pb, np.nan)
        for name, condition in (("ROEAbove10", roe > .1), ("ROEBelow10", roe < .1),
                                ("ROEAbove15", roe > .15), ("ROEAbove20", roe > .2)):
            result[name][row] = np.where(condition, roe, np.nan)
        result["HighROEDecile"][row] = roe
        result[VIDEO_FINANCIAL_NAMES[8]][row] = LowCashOutflowProfitGrowthSpread().calc_batch(fields)
        result[VIDEO_FINANCIAL_NAMES[9]][row] = HighOperatingProfitRevenueGrowthSpread().calc_batch(fields)
        payable = fields["financial_other_payable"]
        result[VIDEO_FINANCIAL_NAMES[10]][row] = _ratio(fields["financial_inventory"], payable)
        result[VIDEO_FINANCIAL_NAMES[12]][row] = _ratio(fields["financial_receivable"], payable)
        surplus = (fields["financial_operating_cash_flow"] - fields["financial_cash_taxes_payable"]
                   - fields["financial_cash_other_payable"] - fields["financial_cash_other_current_liability"])
        result[VIDEO_FINANCIAL_NAMES[11]][row] = _ratio(surplus, fields["financial_sales_cash"])
    for value in result.values():
        value[~np.isfinite(value)] = np.nan
        value.flags.writeable = False
    return result


def prepare_financial_factor_definitions(runtime, events, financial_identity):
    """Bind the same formulae to canonical daily-open factor evaluation."""
    import hashlib
    import inspect
    import json
    import sys
    from factor.base import FactorDefinition, FactorMetadata

    # The canonical prefilter needs exactly the preceding row. Earlier
    # disclosure history is replayed into the event state without computing
    # unused price scores throughout the runtime's price warm-up history.
    first, stop = max(0, runtime.decision_start - 1), runtime.decision_stop
    shares = np.full((stop - first, runtime.n_stocks), np.nan, dtype=np.float64)
    lag_start = max(1, first)
    shares[lag_start - first:] = runtime.field("total_share")[lag_start - 1:stop - 1]
    selected_scores = calculate_financial_scores(runtime.trade_dates[first:stop], runtime.field("open")[first:stop], shares, events)
    scores = {}
    for name, values in selected_scores.items():
        scores[name] = np.full((runtime.n_dates, runtime.n_stocks), np.nan, dtype=np.float32)
        scores[name][first:stop] = values
        scores[name].flags.writeable = False
    identity = {"source": inspect.getsource(sys.modules[__name__]), "financial_sha256": financial_identity["sha256"],
                "replay_version": FINANCIAL_REPLAY_VERSION,
                "replay_source": inspect.getsource(inspect.getmodule(iter_financial_fields)),
                "score_version": FINANCIAL_SEMANTICS_VERSION,
                "score_source": inspect.getsource(inspect.getmodule(LowCashOutflowProfitGrowthSpread)),
                "runtime_sha256": runtime.manifest.source_sha256, "dates": [str(runtime.trade_dates[0]), str(runtime.trade_dates[-1])],
                "stock_codes": runtime.stock_codes, "computed_row_range": [first, stop]}
    required = ("open", "total_share", "listing_age", "delisted_mask")

    def implementation(name):
        class BoundFinancialFactor:
            def calc_batch(self, panel):
                for field in required:
                    if panel[field] is not runtime.field(field) or panel[field].flags.writeable:
                        raise ValueError("financial factor received a different mutable/runtime panel")
                return scores[name]
        BoundFinancialFactor.__name__ = name
        return BoundFinancialFactor

    return tuple(FactorDefinition(metadata=FactorMetadata(name=name, version=SEMANTICS_VERSION,
                    hist_days=0, required_fields=required,
                    implementation_hash=hashlib.sha256(json.dumps({**identity, "factor": name}, sort_keys=True).encode()).hexdigest()),
                    implementation=implementation(name), raw_runtime_view=True) for name in VIDEO_FINANCIAL_NAMES)

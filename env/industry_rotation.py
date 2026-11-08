"""Offline theoretical index accounting for the video's scheduled close exit.

This is explicitly not the production stock planner or broker execution model.
Signals only consume completed closes; a close exit is fixed at entry, never
chosen using that exit day's prices. Index units are fractional and nontradable.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from env.metrics import performance_from_log_rewards


@dataclass(frozen=True)
class IndustryRotationResult:
    dates: np.ndarray
    nav: np.ndarray
    chain_nav: np.ndarray
    trades: tuple[dict, ...]
    signals: tuple[dict, ...]
    metrics: dict


def run_industry_rotation(dates, codes, open_prices, close_prices, *,
                          signal_start="2012-01-04", end="2022-12-31",
                          strategy="staggered_top3", cost_rate=0.0):
    """Start with cash at first signal close and report subsequent close NAVs.

    Ten-day momentum is close[S]/close[S-10]-1 (11 valid closes).
    Equal weights apply at each entry, with no cross-chain transfers/netting.
    Terminal holdings remain marked at the final close, not forcibly sold.
    Missing execution/valuation prices fail; missing signal windows cannot rank.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    codes = np.asarray(codes, dtype=str)
    op = np.asarray(open_prices, dtype=float)
    cl = np.asarray(close_prices, dtype=float)
    if dates.ndim != 1 or codes.ndim != 1 or op.shape != (len(dates), len(codes)) or cl.shape != op.shape:
        raise ValueError("incompatible index panel shapes")
    if np.any(np.diff(dates).astype(int) <= 0) or len(set(codes)) != len(codes) or np.isnat(dates).any():
        raise ValueError("dates must increase and codes must be unique")
    if strategy not in ("original_top1", "staggered_top3") or not np.isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError("invalid strategy or cost rate")
    selected_dates = np.flatnonzero((dates >= np.datetime64(signal_start)) & (dates <= np.datetime64(end)))
    if len(selected_dates) < 2 or selected_dates[0] < 10:
        raise ValueError("need ten warmup rows and at least two evaluation dates")
    first, last = selected_dates[0], selected_dates[-1]
    chains, topn = (1, 1) if strategy == "original_top1" else (2, 3)
    cash = np.full(chains, 1.0 / chains)
    units = np.zeros((chains, len(codes)))
    exits = np.full(chains, -1)
    history = [cash.copy()]
    trades, signals = [], []

    def prices_for(row, positions, phase):
        prices = op[row, positions] if phase == "open" else cl[row, positions]
        if not (np.isfinite(prices) & (prices > 0)).all():
            raise ValueError(f"missing {phase} price on {dates[row]} for {codes[positions]}")
        return prices

    def sell(row, chain, phase):
        held = np.flatnonzero(units[chain] != 0)
        prices = prices_for(row, held, phase)
        for col, price in zip(held, prices):
            gross = units[chain, col] * price
            cash[chain] += gross * (1 - cost_rate)
            trades.append(dict(date=str(dates[row]), phase=phase, chain=chain, side="sell", code=str(codes[col]),
                               price=float(price), units=float(units[chain, col]), notional=float(gross), cost=float(gross * cost_rate)))
        units[chain] = 0

    for row in range(first + 1, last + 1):
        signal = row - 1
        window = cl[signal - 10:signal + 1]
        valid = (np.isfinite(window) & (window > 0)).all(axis=0)
        score = np.full(len(codes), -np.inf)
        score[valid] = cl[signal, valid] / cl[signal - 10, valid] - 1
        order = np.lexsort((codes, -score))
        target = order[:topn]
        if valid.sum() < topn:
            raise ValueError(f"insufficient valid industry signals on {dates[signal]}")
        chain = 0 if chains == 1 else (row - first - 1) % 2
        signals.append(dict(signal_date=str(dates[signal]), entry_date=str(dates[row]), chain=chain,
                            codes=tuple(codes[target]), momentum=tuple(float(x) for x in score[target])))
        same = chains == 1 and np.array_equal(np.flatnonzero(units[chain] != 0), target)
        if not same:
            if chains == 1:
                sell(row, chain, "open")
            elif np.any(units[chain]):
                raise AssertionError("staggered chain must have exited before reentry")
            prices = prices_for(row, target, "open")
            budget = cash[chain] / topn
            bought = budget / ((1 + cost_rate) * prices)
            for col, price, quantity in zip(target, prices, bought):
                gross = quantity * price
                trades.append(dict(date=str(dates[row]), phase="open", chain=chain, side="buy", code=str(codes[col]),
                                   price=float(price), units=float(quantity), notional=float(gross), cost=float(gross * cost_rate)))
            units[chain, target] = bought
            cash[chain] = 0
            if chains == 2:
                exits[chain] = row + 1
        for sub in range(chains):
            if exits[sub] == row:
                sell(row, sub, "close")
        values = cash.copy()
        for sub in range(chains):
            held = np.flatnonzero(units[sub] != 0)
            values[sub] += np.dot(units[sub, held], prices_for(row, held, "close"))
        history.append(values)
    chain_nav = np.asarray(history)
    nav = chain_nav.sum(axis=1)
    metrics = performance_from_log_rewards(np.diff(np.log(nav))).as_dict()
    metrics.update(final_nav=float(nav[-1]), final_cash=float(cash.sum()),
                   terminal_marked_holdings=float(nav[-1] - cash.sum()))
    return IndustryRotationResult(dates[selected_dates], nav, chain_nav, tuple(trades), tuple(signals), metrics)

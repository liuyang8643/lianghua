"""Scheduled research replay using the canonical planner and account simulator.

This explicit close-phase adapter is for published monthly strategies. It is
not a PPO action or a new execution/accounting implementation. Exit dates are
fixed by the calendar before prices are observed. Normal daily inference still
uses the unchanged T-open episode session.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np

from env.action_schema import ActionSchema
from env.contracts import AccountState, OrderPlan
from env.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule
from env.metrics import performance_from_log_rewards
from env.planner import DayMarketData, DayPlanner
from env.simulator import DaySimulator
from offline_data.contracts import RuntimeSlice


def select_video_month(scores: np.ndarray, eligible: np.ndarray, *, selection_fraction: float = 1.0) -> np.ndarray:
    """Threshold scores mask failures; quantiles use floor(n * fraction).

    Stable full-axis order resolves ties. Invalid data never becomes a
    replacement security. Selection occurs before next-open legality checks.
    """
    if not np.isfinite(selection_fraction) or not 0 < selection_fraction <= 1:
        raise ValueError('selection_fraction must be finite and in (0, 1]')
    valid = np.flatnonzero(eligible & np.isfinite(scores))
    order = valid[np.argsort(-scores[valid], kind="stable")]
    chosen = order[:int(np.floor(len(order) * selection_fraction))]
    mask = np.zeros(len(scores), dtype=bool)
    mask[chosen] = True
    return mask


def mark_intraday(simulator: DaySimulator, account: AccountState, plan: OrderPlan,
                  opening: dict, closing: dict, date: str):
    """Execute the open plan, mark at close, and retain actual T+1 locks."""
    result = simulator.step(account, plan, opening, closing,
                            close_prices=opening, next_preclose_prices=opening,
                            next_decision_date=date)
    sellable = dict(account.sellable_positions)
    for fill in result.fills:
        if fill.side == "sell":
            sellable[fill.code] = max(0, sellable[fill.code] - fill.quantity)
    return replace(result, account_state=replace(result.account_state,
        sellable_positions={code: min(quantity, sellable.get(code, 0))
                            for code, quantity in result.account_state.positions.items()}))


def run_monthly_video_replay(runtime: RuntimeSlice, scores: np.ndarray, *, name: str,
                             output: Path, selection_fraction: float = 1.0, initial_cash: float = 1e9,
                             fees: FeeSchedule = DEFAULT_FEE_SCHEDULE,
                             month_end_dates: np.ndarray | None = None) -> dict:
    """Month-end signal, next-month open entry, month-end close liquidation.

    A large declared notional reduces board-lot distortion in broad threshold
    portfolios. It does not imply market capacity; no market-impact claim is
    made. Failed entry orders are not replaced outside the frozen selection.
    Locked exits are retried at following opens before the next entry.
    """
    if scores.shape != (runtime.n_dates, runtime.n_stocks):
        raise ValueError("score panel does not match runtime")
    if not np.isfinite(selection_fraction) or not 0 < selection_fraction <= 1:
        raise ValueError('selection_fraction must be finite and in (0, 1]')
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("completed calendar replay already exists")
    dates, codes = runtime.trade_dates, runtime.stock_codes
    months = dates.astype("datetime64[M]")
    if month_end_dates is None:
        # Callers with a truncated research interval must provide the full
        # calendar-derived dates. A final calendar month is never inferred
        # complete just because the available price snapshot ends there.
        month_end_dates = dates[:-1][months[:-1] != months[1:]]
    month_ends = set(np.asarray(month_end_dates, dtype="datetime64[D]").astype(str))
    first, stop = runtime.decision_start, runtime.decision_stop
    if first < 1 or stop - first < 2:
        raise ValueError("calendar replay requires prior signal history and a settlement row")
    planner, simulator = DayPlanner(fees=fees), DaySimulator(fees)
    account = AccountState(cash=initial_cash, nav=initial_cash, peak_nav=initial_cash)
    code_index = {code: index for index, code in enumerate(codes)}
    nav, valuation_dates, selected_counts, held_counts = [initial_cash], [str(dates[first])], [], []
    selections, all_fills, incomplete_entries, blocked_exits, daily_accounts = [], [], [], [], []
    costs = 0.0
    outstanding_exit: dict[str, int] = {}
    pending_selection = np.zeros(runtime.n_stocks, dtype=bool)
    unit_ranks = np.ones(runtime.n_stocks)
    unit_validity = np.ones(runtime.n_stocks, dtype=bool)

    def config(n):
        n = max(1, int(n))
        schema = ActionSchema(factor_names=(name,), fixed_buy_n=n, turnover_maximum=1.0,
                              fixed_filter_flags=(False, False), fixed_limit_up_protection=False,
                              fixed_rebalance_band_pct=0.0,
                              schema_version="video-monthly-selection-v4-continuous-turnover")
        return schema.from_static_config({"weights": {name: 1.0}, "buy_n": n, "turnover_rate": 1.0,
                                          "single_buy_pct": 1 / n, "limit_up_protection": False,
                                          "rebalance_band_pct": 0.0})

    def market(row, prices, candidates):
        return DayMarketData(decision_date=str(dates[row]), stock_codes=codes,
            factor_ranks={name: unit_ranks}, factor_validity={name: unit_validity}, filter_masks={},
            open_prices=prices, preclose_prices=runtime.field("preClose")[row],
            issue_prices=runtime.field("issue_price"), st_mask=runtime.field("st_mask")[row],
            delisted_mask=runtime.field("delisted_mask")[row], listing_age=runtime.field("listing_age")[row],
            candidate_mask=candidates, metadata={"research_execution_phase": "scheduled_video_replay"})

    def price_map(values, extra=()):
        indices = np.unique(np.concatenate((np.array([code_index[c] for c in account.positions], dtype=int), np.asarray(extra, dtype=int))))
        return {codes[i]: float(values[i]) for i in indices}

    for row in range(first, stop):
        date = str(dates[row])
        is_entry = months[row] != months[row - 1]
        is_exit = date in month_ends
        opens, closes = runtime.field("open")[row], runtime.field("close")[row]
        if is_entry:
            signal_row = row - 1
            eligible = ((dates[signal_row] - runtime.field("issue_date") >= np.timedelta64(365, "D"))
                        & (runtime.field("listing_age")[signal_row] >= 0)
                        & ~runtime.field("st_mask")[signal_row] & ~runtime.field("delisted_mask")[signal_row])
            pending_selection = select_video_month(scores[signal_row], eligible, selection_fraction=selection_fraction)
            selections.append({"signal_date": str(dates[signal_row]), "entry_date": date,
                               "selected": [codes[i] for i in np.flatnonzero(pending_selection)]})
        # The normal planner owns legality, sizing and cash use. A frozen
        # candidate mask prevents its full-investment fallback changing the
        # video's requested number or admitting threshold failures.
        if outstanding_exit:
            exit_plan = planner.plan(market(row, opens, np.zeros(runtime.n_stocks, bool)), account, config(1))
            exit_plan = replace(exit_plan, sell_orders=tuple((code, min(quantity, outstanding_exit[code]))
                                for code, quantity in exit_plan.sell_orders if code in outstanding_exit))
            # No price transition: only clear previously locked orders at open.
            cleared = mark_intraday(simulator, account, exit_plan, price_map(opens), price_map(opens), date)
            account = cleared.account_state
            costs += sum(fill.fee for fill in cleared.fills)
            all_fills.extend({**asdict(fill), "phase": "retry_open"} for fill in cleared.fills)
            for fill in cleared.fills:
                outstanding_exit[fill.code] -= fill.quantity
            outstanding_exit = {code: min(quantity, account.positions[code]) for code, quantity in outstanding_exit.items()
                                if quantity > 0 and code in account.positions}
        if is_entry:
            plan = planner.plan(market(row, opens, pending_selection), account, config(pending_selection.sum()))
        else:
            plan = OrderPlan(date)
        extra = np.array([code_index[code] for code in plan.buy_orders], dtype=int)
        opened = mark_intraday(simulator, account, plan, price_map(opens, extra), price_map(closes, extra), date)
        account = opened.account_state
        for fill in opened.fills:
            if fill.side == "sell" and fill.code in outstanding_exit:
                outstanding_exit[fill.code] = max(0, outstanding_exit[fill.code] - fill.quantity)
        outstanding_exit = {code: min(quantity, account.positions[code]) for code, quantity in outstanding_exit.items()
                            if quantity > 0 and code in account.positions}
        costs += sum(fill.fee for fill in opened.fills)
        all_fills.extend({**asdict(fill), "phase": "open"} for fill in opened.fills)
        if is_entry:
            held = set(account.positions)
            absent = [codes[i] for i in np.flatnonzero(pending_selection) if codes[i] not in held]
            if absent:
                incomplete_entries.append({"date": date, "selected_count": int(pending_selection.sum()), "unfilled": absent})
        selected_counts.append(int(pending_selection.sum()))
        if is_exit:
            exit_plan = planner.plan(market(row, closes, np.zeros(runtime.n_stocks, bool)), account, config(1))
        else:
            exit_plan = OrderPlan(date)
        has_next = row + 1 < stop
        following = simulator.step(account, exit_plan, price_map(closes),
            price_map(runtime.field("open")[row + 1] if has_next else closes),
            close_prices=price_map(closes),
            next_preclose_prices=price_map(runtime.field("preClose")[row + 1] if has_next else closes),
            next_delisted_codes=[codes[i] for i in np.flatnonzero(runtime.field("delisted_mask")[row + 1])] if has_next else (),
            next_decision_date=str(dates[row + 1] if has_next else dates[row]))
        closing_positions = dict(account.positions)
        for fill in following.fills:
            closing_positions[fill.code] -= fill.quantity
        closing_positions = {code: quantity for code, quantity in closing_positions.items() if quantity > 0}
        held_counts.append(len(closing_positions))
        account = following.account_state
        costs += sum(fill.fee for fill in following.fills)
        all_fills.extend({**asdict(fill), "phase": "close"} for fill in following.fills)
        if is_exit and account.positions:
            # Every remaining share belongs to this scheduled liquidation.
            # The simulator has already applied the overnight rebase.
            outstanding_exit = dict(account.positions)
            blocked_exits.append({"date": date, "positions": dict(account.positions)})
        elif is_exit:
            outstanding_exit = {}
        else:
            adjustments = following.diagnostics["corporate_action_adjustments"]
            for code in list(outstanding_exit):
                if code not in account.positions:
                    del outstanding_exit[code]
                elif code in adjustments:
                    quantity = round(outstanding_exit[code] * adjustments[code]["reference_ratio"])
                    outstanding_exit[code] = min(account.positions[code], int(quantity))
                    if outstanding_exit[code] <= 0:
                        del outstanding_exit[code]
        daily_accounts.append({"date": date, "close_positions": closing_positions,
                               "close_cash": float(following.diagnostics["post_fill_cash"]),
                               "close_nav": float(following.diagnostics["post_fill_nav"]),
                               "next_open_date": str(dates[row + 1]) if has_next else None,
                               "next_open_pending_exit": dict(outstanding_exit)})
        nav.append(float(following.diagnostics["post_fill_nav"]))
        valuation_dates.append(date)
    nav = np.array(nav)
    if not np.isfinite(nav).all() or np.any(nav <= 0) or account.cash < -1e-6:
        raise ValueError("calendar account violates finite/cash contracts")
    metrics = performance_from_log_rewards(np.diff(np.log(nav))).as_dict()
    np.savez_compressed(output / "trace.npz", dates=valuation_dates, nav=nav, selected_counts=selected_counts, held_counts=held_counts)
    for filename, records in (("fills.jsonl", all_fills), ("selections.jsonl", selections), ("accounts.jsonl", daily_accounts)):
        with (output / filename).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    result = {"name": name, "protocol": "video-monthly-open-entry-close-exit-v3-fraction", "selection_fraction": selection_fraction, "first_open": valuation_dates[0],
              "last_valuation_close": valuation_dates[-1], "initial_cash": initial_cash, "ending_nav": float(nav[-1]),
              "fees": asdict(fees), "total_costs": costs, "fills": len(all_fills),
              "terminal_positions": dict(account.positions), "terminal_cash": account.cash,
              "terminal_pending_exit": dict(outstanding_exit),
              "selection_months": len(selections), "incomplete_entries": incomplete_entries, "blocked_exits": blocked_exits,
              "integer_lot_execution_exact_video": False, "broker_exact": False, **metrics}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result

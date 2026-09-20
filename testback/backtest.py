"""Canonical fixed-policy backtest adapter.

The account timeline, stock selection, legality, order planning, fills,
settlement and reward all live in :mod:`env`. This module only prepares a
sealed offline episode, converts a static config through the one
``ActionSchema`` and presents the resulting trace in the historical report
shape. It intentionally contains no planner fallback.
"""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from offline_data import latest_runtime_npz_path
from ai.ga.config import canonicalize_individual_config
from env.action_schema import ActionSchema
from env.backtest import (
    EpisodeSession,
    PreparedEpisode,
    RolloutTrace,
    prepare_episode_from_runtime,
    run_day_config_episode,
)
from env.prefilter import prefilter_n_from_config
from loguru import logger as testback_logger
from utils.logger import LOG_FORMAT
from env.observation import DEFAULT_LOOKBACK
from factor import factor_coverage

def _trade_log(trace: RolloutTrace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for decision_date, fills, fee_breakdowns in zip(
        trace.decision_dates,
        trace.fills,
        trace.fee_breakdowns,
        strict=True,
    ):
        for fill, costs in zip(fills, fee_breakdowns, strict=True):
            rows.append(
                {
                    "date": fill.timestamp or decision_date,
                    "signal_date": decision_date,
                    "trade_date": fill.timestamp or decision_date,
                    "price_field": "open",
                    "action": fill.side,
                    "code": fill.code,
                    "price": float(fill.price),
                    "volume": int(fill.quantity),
                    "amount": float(fill.price * fill.quantity),
                    "total_fee": float(fill.fee),
                    "broker_commission": float(costs["commission"]),
                    "transfer_fee": float(costs["transfer_fee"]),
                    "stamp_tax": float(costs["stamp_tax"]),
                    "slippage": float(costs["slippage"]),
                }
            )
    return rows


def _daily_snapshots(trace: RolloutTrace) -> list[dict[str, object]]:
    cumulative = (trace.nav[1:] / trace.nav[0] - 1.0) * 100.0
    snapshots: list[dict[str, object]] = []
    for index, fills in enumerate(trace.fills):
        buys = [fill.code for fill in fills if fill.side == "buy"]
        sells = [fill.code for fill in fills if fill.side == "sell"]
        total_asset = float(trace.nav[index + 1])
        cash = float(trace.cash[index])
        market_value = total_asset - cash
        snapshots.append(
            {
                "date": trace.next_decision_dates[index],
                "signal_date": trace.decision_dates[index],
                "trade_date": trace.decision_dates[index],
                "settlement_date": trace.next_decision_dates[index],
                "daily_return_pct": float(trace.portfolio_returns[index] * 100.0),
                "cumulative_return_pct": float(cumulative[index]),
                "cash": cash,
                "total_asset": total_asset,
                "market_value": market_value,
                "exposure": market_value / total_asset,
                "rebalance_funds_ratio": 0.0,
                "executed_buy_list": buys,
                "executed_sell_list": sells,
                "entered_stocks": buys,
                "exited_stocks": sells,
                "day_config": dict(trace.day_configs[index]),
                "residual_cash_reason": trace.residual_cash_reasons[index],
                "full_investment_contract_satisfied": bool(
                    trace.full_investment_contract[index]
                ),
            }
        )
    return snapshots


def run_static_config_backtest(
    episode: PreparedEpisode,
    individual_config: Mapping[str, object],
    *,
    profile_name: str | None = None,
    initial_cash: float = 1_000_000.0,
) -> dict[str, object]:
    """Evaluate one static config on the canonical serial account session."""

    if not isinstance(episode, PreparedEpisode):
        raise TypeError(
            "canonical backtest requires PreparedEpisode; legacy data/scores "
            "arrays are no longer accepted"
        )
    if episode.prefilter_n is None:
        raise ValueError("canonical backtest episode is missing prefilter_n")
    schema = ActionSchema(
        factor_names=episode.factors.factor_names,
        filter_names=episode.factors.filter_names,
    )
    canonical_config, day_config = canonicalize_individual_config(
        individual_config,
        profile_name=profile_name,
        action_schema=schema,
    )
    session = EpisodeSession(
        episode,
        action_schema=schema,
        initial_cash=initial_cash,
    )
    trace = run_day_config_episode(
        session,
        lambda _observation: day_config,
    )
    if not trace.full_investment_contract_satisfied:
        failed_dates = [
            decision_date
            for decision_date, satisfied in zip(
                trace.decision_dates,
                trace.full_investment_contract,
                strict=True,
            )
            if not satisfied
        ]
        raise RuntimeError(
            "canonical backtest violated the full-investment contract on: "
            + ", ".join(failed_dates)
        )

    daily_returns = trace.portfolio_returns * 100.0
    cumulative_returns = (trace.nav[1:] / trace.nav[0] - 1.0) * 100.0
    final_account = session.current_account
    trade_log = _trade_log(trace)
    positions = [
        {
            "code": code,
            "volume": int(quantity),
            "avg_price": float(final_account.average_costs.get(code, 0.0)),
            "current_price": float(final_account.last_prices.get(code, 0.0)),
            "current_value": float(
                quantity * final_account.last_prices.get(code, 0.0)
            ),
            "cost": float(quantity * final_account.average_costs.get(code, 0.0)),
        }
        for code, quantity in sorted(final_account.positions.items())
    ]
    buy_count = sum(fill.side == "buy" for fills in trace.fills for fill in fills)
    sell_count = sum(fill.side == "sell" for fills in trace.fills for fill in fills)
    account_events = [dict(event) for event in trace.account_events]
    delist_events = [
        event
        for event in account_events
        if event.get("type") == "delist_write_off"
    ]
    return {
        "individual_config": canonical_config,
        "factor_coverage": factor_coverage(episode.runtime, episode.factors),
        "day_config": day_config,
        "trace": trace,
        "daily_returns": daily_returns.astype(float).tolist(),
        "cumulative_returns": cumulative_returns.astype(float).tolist(),
        "daily_assets": trace.nav[1:].astype(float).tolist(),
        "daily_exposures": trace.exposure.astype(float).tolist(),
        "daily_snapshots": _daily_snapshots(trace),
        "signal_dates": list(trace.decision_dates),
        "trade_dates": list(trace.decision_dates),
        "nav_dates": list(trace.next_decision_dates),
        "trade_log": trade_log,
        "positions": positions,
        "cleared_positions": [],
        "position_lot_analytics_available": False,
        "holding_period_available": False,
        "benchmark_available": False,
        "per_year_metrics_available": False,
        "account_events": account_events,
        "delist_events": delist_events,
        "stock_name_map": {},
        "holding_stats": {},
        "executed_buy_count": int(buy_count),
        "executed_sell_count": int(sell_count),
        "delist_count": len(delist_events),
        "round_trip_count": None,
        "cleared_positions_count": None,
        "current_positions_count": len(positions),
        "final_asset": float(trace.nav[-1]),
        "total_return": float(cumulative_returns[-1]),
        "full_investment_contract": trace.full_investment_contract.tolist(),
        "full_investment_contract_satisfied": (
            trace.full_investment_contract_satisfied
        ),
        "residual_cash_reasons": list(trace.residual_cash_reasons),
        "final_account": final_account,
    }


def _parse_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"invalid date: {value!r}")
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _resolve_output_dir(output_dir_arg: str | None, mode: str) -> Path:
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"{mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _core_report_metrics(result: Mapping[str, object]) -> dict[str, float]:
    trace = result["trace"]
    if not isinstance(trace, RolloutTrace):
        raise TypeError("canonical report requires a RolloutTrace")
    performance = trace.metrics
    trade_log = result["trade_log"]
    cost_totals = {
        name: float(sum(float(row[name]) for row in trade_log))
        for name in (
            "broker_commission",
            "transfer_fee",
            "stamp_tax",
            "slippage",
            "total_fee",
        )
    }
    return {
        "annualized": float(performance.annualized_return * 100.0),
        "max_drawdown": float(-performance.max_drawdown * 100.0),
        "sharpe_ratio": float(performance.sharpe),
        "calmar_ratio": float(performance.calmar),
        "average_exposure": float(np.mean(result["daily_exposures"])),
        "executed_buy_count": int(result["executed_buy_count"]),
        "executed_sell_count": int(result["executed_sell_count"]),
        "total_trades": len(result["trade_log"]),
        "total_broker_commission": cost_totals["broker_commission"],
        "total_transfer_fee": cost_totals["transfer_fee"],
        "total_stamp_tax": cost_totals["stamp_tax"],
        "total_slippage": cost_totals["slippage"],
        "total_fees": cost_totals["total_fee"],
    }


def run_single_mode(
    args,
    mode_config: Mapping[str, object],
) -> dict[str, object]:
    """Run a fixed config through the same env episode used by GA and PPO."""

    config_path = Path(args.individual_config)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    runtime_path = Path(
        getattr(args, "runtime_path", None)
        or getattr(args, "runtime", None)
        or latest_runtime_npz_path()
    )
    lookback = int(getattr(args, "lookback", DEFAULT_LOOKBACK))
    output_dir = _resolve_output_dir(getattr(args, "output_dir", None), "single")
    file_sink = testback_logger.add(
        output_dir / "single.log", format=LOG_FORMAT, level="DEBUG",
        colorize=False, rotation="50 MB", retention="7 days",
    )
    try:
        episode = prepare_episode_from_runtime(
            runtime_path,
            start,
            end,
            lookback=lookback,
            prefilter_n=prefilter_n_from_config(payload),
            encode_observations=False,
        )
        result = run_static_config_backtest(
            episode,
            payload,
            initial_cash=float(getattr(args, "initial_cash", 1_000_000.0)),
        )
        metrics = _core_report_metrics(result)
        testback_logger.info(
            "canonical env 回测: "
            f"年化={metrics['annualized']:.2f}% "
            f"夏普={metrics['sharpe_ratio']:.2f} "
            f"最大回撤={metrics['max_drawdown']:.2f}% "
            f"Calmar={metrics['calmar_ratio']:.2f} "
            "full-investment=PASS"
        )
        report_data = {
            **result,
            "metrics": metrics,
            "init_cash": float(getattr(args, "initial_cash", 1_000_000.0)),
            "period": {
                "signal_start": result["signal_dates"][0],
                "signal_end": result["signal_dates"][-1],
                "trade_start": result["trade_dates"][0],
                "trade_end": result["trade_dates"][-1],
                "settlement_start": result["nav_dates"][0],
                "settlement_end": result["nav_dates"][-1],
                "start": result["signal_dates"][0],
                "end": result["nav_dates"][-1],
            },
            "report_metadata": {
                "config_path": str(config_path.resolve()),
                "runtime_path": str(runtime_path.resolve()),
                "stock_pool_size": episode.runtime.n_stocks,
                "engine": "env.EpisodeSession",
            },
            "rebalance_rule": {
                "signal_timing": "T-open",
                "trade_timing": "T-open",
                "price_field": "open",
                "mode": "daily_equalize_then_cash_sweep",
            },
            "per_year_metrics": [],
            "hs300_returns": [],
            "_runtime_kline": {
                name: episode.runtime.field(name)
                for name in ("open", "high", "low", "close", "amount")
            }
            | {
                "trade_dates": episode.runtime.trade_dates,
                "stock_codes": np.asarray(episode.runtime.stock_codes),
            },
        }
        _save_single_record(report_data, output_dir)
        if bool(mode_config.get("save_charts", False)):
            from testback.reportor import generate_single_report

            html_path = generate_single_report(report_data, output_dir)
            testback_logger.info(f"可视化报告已保存至: {html_path}")
        return result
    finally:
        testback_logger.remove(file_sink)


def _save_single_record(report_data: Mapping[str, object], output_dir: Path) -> None:
    record = {
        "individual_config": report_data["individual_config"],
        "period": report_data["period"],
        "dates": report_data["signal_dates"],
        "daily_returns": report_data["daily_returns"],
        "cumulative_returns": report_data["cumulative_returns"],
        "daily_exposures": report_data["daily_exposures"],
        "full_investment_contract_satisfied": report_data[
            "full_investment_contract_satisfied"
        ],
        "metrics": report_data["metrics"],
    }
    path = output_dir / "record.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    testback_logger.info(f"回测明细记录已保存至: {path}")

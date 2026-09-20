"""Small deterministic runtime shared by PPO/env contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from env.action_schema import ActionSchema
from env.backtest import PreparedEpisode, required_runtime_preload_rows
from env.encoder import TrainOnlyNormalizer
from factor import precompute_factors
from offline_data import load_runtime_slice
from offline_data.financial_versions import RAW_FINANCIAL_VALUE_NAMES, RAW_FINANCIAL_TIME_NAMES, RAW_FINANCIAL_PERIOD_NAMES


ROOT = Path(__file__).resolve().parents[1]


def financial_runtime_arrays(total_share: np.ndarray) -> dict[str, np.ndarray]:
    """Synthetic as-of inputs, stable when future rows/stocks are appended."""
    shares = np.asarray(total_share, dtype=np.float64)
    day = np.arange(shares.shape[0], dtype=np.float64)[:, None]
    stock = np.arange(shares.shape[1], dtype=np.float64)[None, :]
    equity = shares * (6.0 + 2.0 * (stock % 3) + 0.002 * day)
    return {
        **{name: equity * (0.1 + i * 0.02) for i, name in enumerate(RAW_FINANCIAL_VALUE_NAMES)},
        **{name: np.broadcast_to(20 + day, shares.shape).copy() for name in RAW_FINANCIAL_TIME_NAMES},
        **{name: np.full(shares.shape, 2.0) for name in RAW_FINANCIAL_PERIOD_NAMES},
        "financial_profit_ttm": equity * (0.08 + 0.04 * (stock % 3)),
        "financial_equity": equity,
        "financial_cash_outflow_yoy": 0.08 + 0.0001 * day + 0.025 * (stock % 3),
        "financial_profit_yoy": 0.12 + 0.0001 * day + 0.015 * (stock % 5),
        "financial_operating_profit_yoy": 0.18 + 0.0002 * day + 0.02 * (stock % 4),
        "financial_revenue_yoy": 0.10 + 0.0001 * day + 0.01 * (stock % 3),
        "abnormal_revenue_quarter": equity * (0.6 + 0.02 * (stock % 3)),
        "abnormal_cost_quarter": equity * (0.4 + 0.01 * (stock % 4)),
        "abnormal_sales_cash_quarter": equity * 0.55,
        "abnormal_revenue_prior_year_quarter": equity * 0.52,
        "abnormal_cost_prior_year_quarter": equity * 0.36,
        "abnormal_sales_cash_prior_year_quarter": equity * 0.49,
        "abnormal_total_assets": equity * 2.,
    }


def write_runtime(path: Path, *, end: str = "2020-07-20") -> None:
    dates = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64(end),
        dtype="datetime64[D]",
    )
    codes = np.asarray(
        ("600001.SH", "000001.SZ", "300001.SZ", "688001.SH", "430001.BJ")
    )
    day = np.arange(len(dates), dtype=np.float64)[:, None]
    stock = np.arange(len(codes), dtype=np.float64)[None, :]
    close = 10.0 + day * 0.01 + stock
    opens = close * (1.0 + 0.001 * (stock - 1.5))
    preclose = np.empty_like(close)
    preclose[0] = close[0]
    preclose[1:] = close[:-1]
    volume = 1_000_000.0 + day * 1_000.0 + stock * 100.0
    panel = np.broadcast_to(1.0 + day * 0.001 + stock * 0.01, close.shape).copy()
    listing_age = np.broadcast_to(
        np.arange(len(dates), dtype=np.int32)[:, None],
        close.shape,
    ).copy()
    total_share = np.broadcast_to(
        np.asarray((1e9, 8e8, 5e8, 3e8, 2e8))[None, :],
        close.shape,
    )
    np.savez_compressed(
        path,
        stock_codes=codes,
        trade_dates=dates,
        open=opens,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=volume,
        amount=volume * close,
        preClose=preclose,
        issue_price=np.full(len(codes), 8.0),
        issue_date=np.full(len(codes), dates[0], dtype="datetime64[D]"),
        st_mask=np.zeros(close.shape, dtype=np.bool_),
        listing_age=listing_age,
        delisted_mask=np.zeros(close.shape, dtype=np.bool_),
        total_share=total_share,
        bps=panel,
        eps=panel,
        roe=panel,
        profit_yoy=panel,
        revenue_yoy=panel,
        operating_cf_ps=panel,
        gross_margin=panel,
        **financial_runtime_arrays(total_share),
    )


def build_episode(
    path: Path,
    *,
    start: str = "2020-06-20",
    end: str = "2020-07-10",
) -> PreparedEpisode:
    write_runtime(path)
    runtime = load_runtime_slice(
        path,
        start,
        end,
        preload_rows=required_runtime_preload_rows(64),
    )
    return PreparedEpisode.build(
        runtime,
        precompute_factors(runtime),
        lookback=64,
        prefilter_n=300,
    )


def static_config(schema: ActionSchema):
    payload = json.loads((ROOT / "configs" / "config.json").read_text("utf-8"))
    return schema.from_static_config(payload)


def fit_normalizer(episode: PreparedEpisode) -> TrainOnlyNormalizer:
    return TrainOnlyNormalizer.fit(
        episode.market_store,
        episode.encoder.output_schema,
        dataset_role="train",
        initial_cash=1_000_000.0,
    )

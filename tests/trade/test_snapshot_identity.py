from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rl_test_data import financial_runtime_arrays

from env.action_schema import ActionSchema
from env.backtest import PreparedDecision, required_runtime_preload_rows
from env.contracts import AccountState
from factor import precompute_factors
from offline_data import compute_runtime_lineage, load_runtime_slice
from trade.runtime import (
    BrokerAccountAdapter,
    SealedSnapshotAdapter,
    SnapshotIntegrityError,
)


def _runtime_npz(
    path: Path,
    *,
    rows: int,
    mutate_prefix: bool = False,
    reverse_codes: bool = False,
    missing_final_open: bool = False,
    missing_final_preclose: bool = False,
) -> None:
    dates = np.arange(
        np.datetime64("2019-01-01"),
        np.datetime64("2019-01-01") + rows,
        dtype="datetime64[D]",
    )
    codes = np.asarray(("600001.SH", "000001.SZ", "300001.SZ"))
    if reverse_codes:
        codes = codes[::-1]
    day = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(len(codes), dtype=np.float64)[None, :]
    close = 10.0 + day * 0.01 + stock
    if mutate_prefix:
        close[3, 0] += 0.25
    opens = close * (1.0 + 0.0001 * stock)
    if missing_final_open:
        opens[-1, 0] = np.nan
    preclose = np.empty_like(close)
    preclose[0] = close[0]
    preclose[1:] = close[:-1]
    if missing_final_preclose:
        preclose[-1, 0] = np.nan
    volume = 1_000_000.0 + day * 1_000.0 + stock * 100.0
    panel = np.broadcast_to(1.0 + day * 0.001 + stock * 0.01, close.shape).copy()
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
        listing_age=np.broadcast_to(
            np.arange(rows, dtype=np.int32)[:, None], close.shape
        ).copy(),
        delisted_mask=np.zeros(close.shape, dtype=np.bool_),
        total_share=np.broadcast_to(
            np.asarray((1e9, 8e8, 5e8))[None, :], close.shape
        ),
        bps=panel,
        eps=panel,
        roe=panel,
        profit_yoy=panel,
        revenue_yoy=panel,
        operating_cf_ps=panel,
        gross_margin=panel,
        **financial_runtime_arrays(np.broadcast_to(
            np.asarray((1e9, 8e8, 5e8))[None, :], close.shape
        )),
    )


def _adapter(base: Path, cutoff: str) -> SealedSnapshotAdapter:
    schema = ActionSchema()
    runtime = load_runtime_slice(
        base,
        cutoff,
        cutoff,
        preload_rows=required_runtime_preload_rows(64),
    )
    factors = precompute_factors(runtime)
    prepared = PreparedDecision.build(
        runtime,
        factors,
        action_schema=schema,
        lookback=64,
        prefilter_n=300,
    )
    expected_runtime = runtime.manifest.as_dict()
    expected_runtime["lineage"] = compute_runtime_lineage(
        base, cutoff=cutoff
    ).as_dict()
    return SealedSnapshotAdapter(
        action_schema=schema,
        observation_schema=prepared.observation_builder.schema,
        expected_runtime=expected_runtime,
        expected_factors={"schema_hash": factors.schema_hash},
        prefilter_n=300,
    )


def test_snapshot_exact_file_and_proven_append_only_file_pass(tmp_path: Path):
    base = tmp_path / "base.npz"
    appended = tmp_path / "appended.npz"
    _runtime_npz(base, rows=310)
    _runtime_npz(appended, rows=312)
    cutoff = str(np.datetime64("2019-01-01") + 309)
    next_date = str(np.datetime64("2019-01-01") + 311)
    adapter = _adapter(base, cutoff)

    exact = adapter.load(base, cutoff)
    appended_snapshot = adapter.load(appended, next_date)

    assert exact.decision_date == cutoff
    assert appended_snapshot.decision_date == next_date
    assert exact.stock_codes == appended_snapshot.stock_codes


def test_snapshot_rejects_changed_historical_prefix(tmp_path: Path):
    base = tmp_path / "base.npz"
    changed = tmp_path / "changed.npz"
    _runtime_npz(base, rows=310)
    _runtime_npz(changed, rows=312, mutate_prefix=True)
    cutoff = str(np.datetime64("2019-01-01") + 309)
    next_date = str(np.datetime64("2019-01-01") + 311)

    with pytest.raises(SnapshotIntegrityError, match="historical content prefix"):
        _adapter(base, cutoff).load(changed, next_date)


def test_snapshot_rejects_stock_axis_change(tmp_path: Path):
    base = tmp_path / "base.npz"
    changed = tmp_path / "axis.npz"
    _runtime_npz(base, rows=310)
    _runtime_npz(changed, rows=312, reverse_codes=True)
    cutoff = str(np.datetime64("2019-01-01") + 309)
    next_date = str(np.datetime64("2019-01-01") + 311)

    with pytest.raises(SnapshotIntegrityError, match="vocabulary/order"):
        _adapter(base, cutoff).load(changed, next_date)


def test_broker_account_requires_every_holding_on_complete_axis(tmp_path: Path):
    base = tmp_path / "base.npz"
    _runtime_npz(base, rows=310)
    cutoff = str(np.datetime64("2019-01-01") + 309)
    snapshot = _adapter(base, cutoff).load(base, cutoff)
    account = BrokerAccountAdapter().build(
        asset=SimpleNamespace(cash=1_000.0),
        positions=(
            SimpleNamespace(
                stock_code="600001.SH",
                volume=100,
                can_use_volume=100,
                avg_price=9.0,
            ),
        ),
        snapshot=snapshot,
    )
    assert account.positions == {"600001.SH": 100}
    assert account.nav == pytest.approx(
        1_000.0 + 100 * snapshot.prepared.market.open_prices[0]
    )
    assert account.mark_provenance == {"600001.SH": "runtime.open[T]"}

    with pytest.raises(SnapshotIntegrityError, match="outside"):
        BrokerAccountAdapter().build(
            asset=SimpleNamespace(cash=1_000.0),
            positions=(
                SimpleNamespace(
                    stock_code="OUTSIDE.SH",
                    volume=100,
                    can_use_volume=100,
                    avg_price=9.0,
                ),
            ),
            snapshot=snapshot,
        )


def test_broker_account_suspended_mark_fallback_order_and_provenance(
    tmp_path: Path,
):
    base = tmp_path / "suspended.npz"
    _runtime_npz(base, rows=310, missing_final_open=True)
    cutoff = str(np.datetime64("2019-01-01") + 309)
    snapshot = _adapter(base, cutoff).load(base, cutoff)
    position = SimpleNamespace(
        stock_code="600001.SH",
        volume=100,
        can_use_volume=100,
        avg_price=9.0,
        last_price=13.0,
        market_value=1_400.0,
    )
    previous = AccountState(
        cash=1_000.0,
        positions={"600001.SH": 100},
        sellable_positions={"600001.SH": 100},
        average_costs={"600001.SH": 9.0},
        last_prices={"600001.SH": 12.0},
        mark_provenance={"600001.SH": "runtime.open[T-1]"},
        nav=2_200.0,
        peak_nav=2_200.0,
    )

    from_previous = BrokerAccountAdapter().build(
        asset=SimpleNamespace(cash=1_000.0),
        positions=(position,),
        snapshot=snapshot,
        previous_account=previous,
    )
    runtime_preclose = snapshot.prepared.market.preclose_prices[0]
    assert from_previous.last_prices == {"600001.SH": runtime_preclose}
    assert from_previous.mark_provenance == {
        "600001.SH": "runtime.preClose[T]"
    }

    missing_reference = tmp_path / "missing-reference.npz"
    _runtime_npz(
        missing_reference,
        rows=310,
        missing_final_open=True,
        missing_final_preclose=True,
    )
    fallback_snapshot = _adapter(missing_reference, cutoff).load(
        missing_reference, cutoff
    )

    previous_fallback = BrokerAccountAdapter().build(
        asset=SimpleNamespace(cash=1_000.0),
        positions=(position,),
        snapshot=fallback_snapshot,
        previous_account=previous,
    )
    assert previous_fallback.last_prices == {"600001.SH": 12.0}
    assert previous_fallback.mark_provenance == {
        "600001.SH": "previous_account.last_prices"
    }

    from_last_price = BrokerAccountAdapter().build(
        asset=SimpleNamespace(cash=1_000.0),
        positions=(position,),
        snapshot=fallback_snapshot,
    )
    assert from_last_price.last_prices == {"600001.SH": 13.0}
    assert from_last_price.mark_provenance == {
        "600001.SH": "broker.last_price"
    }

    market_value_only = SimpleNamespace(
        stock_code="600001.SH",
        volume=100,
        can_use_volume=100,
        avg_price=9.0,
        market_value=1_400.0,
    )
    from_market_value = BrokerAccountAdapter().build(
        asset=SimpleNamespace(cash=1_000.0),
        positions=(market_value_only,),
        snapshot=fallback_snapshot,
    )
    assert from_market_value.last_prices == {"600001.SH": 14.0}
    assert from_market_value.mark_provenance == {
        "600001.SH": "broker.market_value/volume"
    }

    no_mark = SimpleNamespace(
        stock_code="600001.SH",
        volume=100,
        can_use_volume=100,
        avg_price=9.0,
    )
    with pytest.raises(SnapshotIntegrityError, match="no causal mark"):
        BrokerAccountAdapter().build(
            asset=SimpleNamespace(cash=1_000.0),
            positions=(no_mark,),
            snapshot=fallback_snapshot,
        )

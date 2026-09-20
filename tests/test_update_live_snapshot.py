from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import update_live


def _tencent_quote(symbol: str, code: str, *, open_price: str) -> str:
    fields = [""] * 31
    fields[1] = code
    fields[2] = code
    fields[4] = "10.00"
    fields[5] = open_price
    fields[30] = "20260828092500"
    return f'v_{symbol}="{"~".join(fields)}";'


def test_live_open_overlay_requires_and_preserves_complete_quote_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (
        _tencent_quote("sz000001", "000001", open_price="10.10")
        + _tencent_quote("sh600000", "600000", open_price="0")
    ).encode("gbk")

    class Response:
        content = payload

        @staticmethod
        def raise_for_status() -> None:
            return None

    monkeypatch.setattr(update_live, "LIVE_OPEN_DIR", tmp_path)
    monkeypatch.setattr(update_live.requests, "get", lambda *_, **__: Response())

    path = update_live._fetch_live_open_overlay(
        ("000001.SZ", "600000.SH"),
        date(2026, 8, 28),
    )
    frame = pd.read_parquet(path)

    assert frame["stock_code"].tolist() == ["000001.SZ", "600000.SH"]
    assert frame["preClose"].tolist() == [10.0, 10.0]
    assert frame.loc[0, "open"] == pytest.approx(10.1)
    assert pd.isna(frame.loc[1, "open"])


def test_live_snapshot_rebuilds_full_runtime_without_field_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data.build_runtime as runtime_builder
    import data.db.issue_price as issue_reference
    import data.db.stock_list as stock_list
    import data.kline_mootdx as kline

    target = date(2026, 8, 28)
    output = tmp_path / "runtime_2026-08-27_2026-08-28.npz"
    np.savez(output, trade_dates=np.array(["2026-08-27", "2026-08-28"], dtype="datetime64[D]"))
    calls = []
    overlay = tmp_path / "live_open.parquet"
    monkeypatch.setattr(kline, "resolve_recent_range", lambda *_: ("", "", target))
    monkeypatch.setattr(
        issue_reference,
        "resolve_terminal_active_codes",
        lambda current, *_args, **_kwargs: tuple(current),
    )
    monkeypatch.setattr(
        stock_list,
        "load_current_stock_codes",
        lambda: ["000001.SZ", "600000.SH"],
    )
    monkeypatch.setattr(kline, "update_recent", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(update_live, "_fetch_live_open_overlay", lambda *_: overlay)
    build_calls = []
    monkeypatch.setattr(
        runtime_builder,
        "build_runtime",
        lambda **kwargs: build_calls.append(kwargs) or output,
    )

    result = update_live.build_live_runtime(target)

    assert result == output.resolve()
    assert calls == [
        (
            (1,),
            {
                "anchor_date": target,
                "codes": ["000001.SZ", "600000.SH"],
                "strict": True,
            },
        )
    ]
    assert build_calls == [
        {
            "partial_live_candidates": ["000001.SZ", "600000.SH"],
            "live_open_overlay": overlay,
        }
    ]


def test_live_snapshot_fetches_only_explicit_prefilter_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data.build_runtime as runtime_builder
    import data.db.issue_price as issue_reference
    import data.db.stock_list as stock_list
    import data.kline_mootdx as kline

    target = date(2026, 8, 28)
    output = tmp_path / "runtime.npz"
    np.savez(
        output,
        trade_dates=np.array([target.isoformat()], dtype="datetime64[D]"),
    )
    calls = []
    overlay = tmp_path / "live_open.parquet"
    monkeypatch.setattr(kline, "resolve_recent_range", lambda *_: ("", "", target))
    monkeypatch.setattr(
        issue_reference,
        "resolve_terminal_active_codes",
        lambda current, *_args, **_kwargs: tuple(current),
    )
    monkeypatch.setattr(
        stock_list,
        "load_current_stock_codes",
        lambda: ["000001.SZ", "000002.SZ", "600000.SH"],
    )
    monkeypatch.setattr(kline, "update_recent", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(update_live, "_fetch_live_open_overlay", lambda *_: overlay)
    monkeypatch.setattr(runtime_builder, "build_runtime", lambda **_: output)

    update_live.build_live_runtime(
        target,
        candidate_codes=("600000.SH", "000002.SZ"),
    )

    assert calls[0][1]["codes"] == ["000002.SZ", "600000.SH"]


def test_live_snapshot_excludes_future_listing_from_k_pull_and_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data.build_runtime as runtime_builder
    import data.db.issue_price as issue_reference
    import data.db.stock_list as stock_list
    import data.kline_mootdx as kline

    target = date(2026, 8, 28)
    active, future = "000001.SZ", "001234.SZ"
    output = tmp_path / "runtime.npz"
    np.savez(
        output,
        trade_dates=np.array([target.isoformat()], dtype="datetime64[D]"),
    )
    monkeypatch.setattr(kline, "resolve_recent_range", lambda *_: ("", "", target))
    monkeypatch.setattr(
        stock_list, "load_current_stock_codes", lambda: [active, future]
    )
    resolver_calls = []

    def resolve(current, terminal, kline_codes):
        resolver_calls.append((tuple(current), terminal, tuple(kline_codes)))
        return (active,)

    monkeypatch.setattr(
        issue_reference, "resolve_terminal_active_codes", resolve
    )
    pulled = []
    monkeypatch.setattr(
        kline, "update_recent",
        lambda *_args, **kwargs: pulled.extend(kwargs["codes"]),
    )
    overlay_axes = []
    overlay = tmp_path / "live_open.parquet"
    monkeypatch.setattr(
        update_live,
        "_fetch_live_open_overlay",
        lambda codes, _target: overlay_axes.append(tuple(codes)) or overlay,
    )
    monkeypatch.setattr(runtime_builder, "build_runtime", lambda **_: output)

    update_live.build_live_runtime(target)

    assert resolver_calls[0][0] == (active, future)
    assert pulled == [active]
    assert overlay_axes == [(active,)]
    assert future not in pulled


def test_live_prefilter_forces_active_code_without_local_k_into_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data.build_runtime as runtime_builder
    import data.db.issue_price as issue_reference
    import data.db.stock_list as stock_list
    import data.kline_mootdx as kline

    target = date(2026, 8, 28)
    existing, listing_today = "000001.SZ", "920999.BJ"
    output = tmp_path / "runtime.npz"
    np.savez(
        output,
        trade_dates=np.array([target.isoformat()], dtype="datetime64[D]"),
    )
    monkeypatch.setattr(kline, "resolve_recent_range", lambda *_: ("", "", target))
    monkeypatch.setattr(
        stock_list,
        "load_current_stock_codes",
        lambda: [existing, listing_today],
    )
    monkeypatch.setattr(
        issue_reference,
        "resolve_terminal_active_codes",
        lambda *_args, **_kwargs: (existing, listing_today),
    )
    pulled = []
    monkeypatch.setattr(
        kline, "update_recent",
        lambda *_args, **kwargs: pulled.extend(kwargs["codes"]),
    )
    overlay_axes = []
    overlay = tmp_path / "live_open.parquet"
    monkeypatch.setattr(
        update_live,
        "_fetch_live_open_overlay",
        lambda codes, _target: overlay_axes.append(tuple(codes)) or overlay,
    )
    monkeypatch.setattr(runtime_builder, "build_runtime", lambda **_: output)

    update_live.build_live_runtime(target, candidate_codes=(existing,))

    assert pulled == [existing, listing_today]
    assert overlay_axes == [(existing, listing_today)]


@pytest.mark.parametrize(
    ("current", "active", "invalid_candidate"),
    [
        (("000001.SZ",), ("000001.SZ",), "999999.SZ"),
        (("000001.SZ", "920999.BJ"), ("000001.SZ",), "920999.BJ"),
    ],
)
def test_live_prefilter_rejects_unknown_or_prelisting_candidate(
    monkeypatch: pytest.MonkeyPatch,
    current,
    active,
    invalid_candidate,
) -> None:
    import data.build_runtime as runtime_builder
    import data.db.issue_price as issue_reference
    import data.db.stock_list as stock_list
    import data.kline_mootdx as kline

    target = date(2026, 8, 28)
    monkeypatch.setattr(kline, "resolve_recent_range", lambda *_: ("", "", target))
    monkeypatch.setattr(stock_list, "load_current_stock_codes", lambda: current)
    monkeypatch.setattr(
        issue_reference,
        "resolve_terminal_active_codes",
        lambda *_args, **_kwargs: active,
    )
    monkeypatch.setattr(
        kline,
        "update_recent",
        lambda *_args, **_kwargs: pytest.fail("非法 candidate 不得触发 K 下载"),
    )

    with pytest.raises(ValueError):
        update_live.build_live_runtime(
            target,
            candidate_codes=(active[0], invalid_candidate),
        )


def test_live_snapshot_rejects_non_trading_date(monkeypatch: pytest.MonkeyPatch) -> None:
    import data.kline_mootdx as kline

    requested = date(2026, 8, 30)
    monkeypatch.setattr(
        kline,
        "resolve_recent_range",
        lambda *_: ("", "", date(2026, 8, 28)),
    )

    with pytest.raises(RuntimeError, match="不是交易日"):
        update_live.build_live_runtime(requested)

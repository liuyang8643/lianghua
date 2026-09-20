import pandas as pd
import pytest

from data import update_financial_deep


def _valid_deep_frame(code: str = "000001.SZ") -> pd.DataFrame:
    values = {
        "stock_code": [code],
        "report_period": [20231231],
    }
    values.update({column: [1.0] for column in update_financial_deep._OUT_COLS})
    return pd.DataFrame(values)


def test_financial_deep_symbols_use_current_plus_delist_not_runtime(monkeypatch):
    import data.db.stock_list as stock_list

    monkeypatch.setattr(
        stock_list,
        "get_all_stock_code_list",
        lambda: ["600001.SH", "000003.SZ", "000001.SZ"],
    )

    assert update_financial_deep._all_symbols() == [
        ("000001.SZ", "000001"),
        ("000003.SZ", "000003"),
        ("600001.SH", "600001"),
    ]


def test_financial_deep_snapshot_rejects_unknown_or_duplicate_rows():
    frame = _valid_deep_frame()
    with pytest.raises(ValueError, match="股票全集之外"):
        update_financial_deep._validate_snapshot(frame, {"600001.SH"})
    with pytest.raises(ValueError, match="重复"):
        update_financial_deep._validate_snapshot(
            pd.concat([frame, frame], ignore_index=True),
            {"000001.SZ"},
        )


def test_financial_deep_atomic_save_validates_round_trip(tmp_path, monkeypatch):
    output = tmp_path / "financial" / "deep_indicators.parquet"
    monkeypatch.setattr(update_financial_deep, "OUT_PATH", output)

    update_financial_deep._save_snapshot_atomic(
        _valid_deep_frame(), {"000001.SZ"}
    )

    result = pd.read_parquet(output)
    assert result["stock_code"].tolist() == ["000001.SZ"]
    assert result["report_period"].tolist() == [20231231]
    assert not list(output.parent.glob("*.tmp.parquet"))


def test_financial_deep_refresh_replaces_existing_code_without_duplicates(
    tmp_path, monkeypatch,
):
    output = tmp_path / "financial" / "deep_indicators.parquet"
    monkeypatch.setattr(update_financial_deep, "OUT_PATH", output)
    monkeypatch.setattr(
        update_financial_deep,
        "_all_symbols",
        lambda: [("000001.SZ", "000001")],
    )
    monkeypatch.setattr(update_financial_deep.time, "sleep", lambda _seconds: None)
    update_financial_deep._save_snapshot_atomic(
        _valid_deep_frame(), {"000001.SZ"}
    )
    refreshed = _valid_deep_frame().drop(columns="stock_code")
    refreshed.loc[0, "eps"] = 2.0
    monkeypatch.setattr(update_financial_deep, "_parse_one", lambda _symbol: refreshed)

    update_financial_deep.main(refresh=True)

    result = pd.read_parquet(output)
    assert len(result) == 1
    assert result.loc[0, "eps"] == 2.0


def test_update_all_requests_full_financial_refresh(monkeypatch):
    import data.update_all as update_all

    calls = []
    monkeypatch.setattr(
        update_financial_deep,
        "main",
        lambda **kwargs: calls.append(kwargs),
    )

    update_all._update_financial_deep()

    assert calls == [{"refresh": True}]

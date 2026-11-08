from datetime import date

import pandas as pd
import pytest


def test_parse_tencent_index_kline_drops_nonpositive_opens():
    from data.update_all import _parse_tencent_index_kline

    payload = {"code": 0, "data": {"sh000300": {"day": [
        ["2026-09-16", "4449.88", "4480.27", "4483.36", "4417.60", "1"],
        ["2026-09-17", "0", "1", "1", "1", "1"],
        ["2026-09-18", "4492.32", "4507.39", "4523.09", "4480.59", "1"],
    ]}}}
    frame = _parse_tencent_index_kline(payload, "sh000300")
    assert frame["date"].dt.date.tolist() == [date(2026, 9, 16), date(2026, 9, 18)]
    assert frame["open"].tolist() == [4449.88, 4492.32]


def test_append_missing_index_days_keeps_existing_opens():
    from data.update_all import _append_missing_index_days

    base = pd.DataFrame({"date": ["2026-09-15", "2026-09-16"], "open": [1., 2.], "close": [1.1, 2.2]})
    extra = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-16", "2026-09-17"]),
        "open": [9., 3.],
    })
    merged = _append_missing_index_days(base, extra)
    assert pd.to_datetime(merged["date"]).dt.date.tolist() == [
        date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17)]
    assert merged["open"].tolist() == [1., 2., 3.]
    assert merged.loc[0, "close"] == 1.1


def test_sina_same_day_gap_is_filled_from_tencent(tmp_path, monkeypatch):
    from data import update_all

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    monkeypatch.setattr(update_all, "current_update_date", lambda: date(2026, 9, 17))
    calls = []

    def sina(symbol):
        return pd.DataFrame({"date": ["2026-09-15", "2026-09-16"], "open": [1., 2.]})

    def tencent(symbol):
        calls.append(symbol)
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-09-16", "2026-09-17"]),
            "open": [9., 3.],
        })

    monkeypatch.setattr(update_all.ak, "stock_zh_index_daily", sina)
    monkeypatch.setattr(update_all, "_fetch_tencent_index_daily", tencent)
    update_all._update_indices()

    assert calls == ["sh000300", "sh000905", "sh000852"]
    frame = pd.read_parquet(tmp_path / "index_sh000300_daily.parquet")
    dates = pd.to_datetime(frame["trade_date"])
    assert dates.dt.date.tolist() == [date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17)]
    assert frame["open"].tolist() == [1., 2., 3.]


def test_covered_sina_history_does_not_call_tencent(tmp_path, monkeypatch):
    from data import update_all

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    monkeypatch.setattr(update_all, "current_update_date", lambda: date(2026, 9, 17))
    monkeypatch.setattr(update_all.ak, "stock_zh_index_daily", lambda symbol: pd.DataFrame({
        "date": ["2026-09-16", "2026-09-17"], "open": [2., 3.]}))

    def fail(symbol):
        raise AssertionError("tencent should not be called")

    monkeypatch.setattr(update_all, "_fetch_tencent_index_daily", fail)
    update_all._update_indices()
    frame = pd.read_parquet(tmp_path / "index_sh000300_daily.parquet")
    assert pd.to_datetime(frame["trade_date"]).max().date() == date(2026, 9, 17)


def test_index_update_fails_when_tencent_still_misses_required_day(tmp_path, monkeypatch):
    from data import update_all

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    monkeypatch.setattr(update_all, "current_update_date", lambda: date(2026, 9, 17))
    monkeypatch.setattr(update_all.ak, "stock_zh_index_daily", lambda symbol: pd.DataFrame({
        "date": ["2026-09-16"], "open": [2.]}))
    monkeypatch.setattr(update_all, "_fetch_tencent_index_daily", lambda symbol: pd.DataFrame({
        "date": pd.to_datetime(["2026-09-16"]), "open": [2.]}))
    with pytest.raises(RuntimeError, match="未覆盖 2026-09-17"):
        update_all._update_indices()

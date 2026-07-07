from datetime import date
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


class _FakeStore:
    _skip_update = False


class _FakeLarkSender:
    def __init__(self):
        self.table_titles = []

    def send_table_card(self, title, level, tables):
        self.table_titles.append(title)

    def send_notification_card(self, level, title, sub_title, content):
        pass


class _FakeRecorder:
    def __init__(self):
        self.marks = []

    def mark(self, name):
        self.marks.append(name)


def test_post_close_update_all_failure_not_reported_success(tmp_path, monkeypatch):
    import data.update_all as update_all
    from trading import post_close

    today = date.today()
    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    monkeypatch.setattr(update_all, "TODAY", today)
    monkeypatch.setattr(update_all, "YESTERDAY", today)

    kline_path = tmp_path / "k-line" / "000001.SZ.parquet"
    kline_path.parent.mkdir(parents=True)
    kline_path.write_text("placeholder")

    called = []

    def noop():
        return None

    def fail_kline():
        called.append("kline")
        raise RuntimeError("kline failed")

    for name in [
        "_update_stock_list",
        "_update_delist",
        "_update_stock_name",
        "_update_balance",
        "_update_financial_deep",
        "_update_issue_price",
        "_update_indices",
        "_update_trading_calendar",
        "_build_runtime",
    ]:
        monkeypatch.setattr(update_all, name, noop)
    monkeypatch.setattr(update_all, "_update_kline", fail_kline)

    def retry_once(func, max_retries=3, base_delay=2.0):
        try:
            return func(), True
        except Exception:
            return None, False

    fake_sender = _FakeLarkSender()
    monkeypatch.setattr(post_close, "_retry_with_backoff", retry_once)
    monkeypatch.setattr(post_close, "lark_sender", fake_sender)
    monkeypatch.setattr(post_close, "recorder", _FakeRecorder())

    assert post_close.run_update_all(_FakeStore()) is False
    assert called == ["kline"]
    assert fake_sender.table_titles[-1].startswith("❌ 全量更新失败")


def test_delist_kline_missing_is_completed_by_mootdx(tmp_path, monkeypatch):
    import data.update_all as update_all
    import data.db.delist as delist_db
    import data.kline_mootdx as kline_mootdx

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    (tmp_path / "k-line").mkdir(parents=True)
    calls = []

    monkeypatch.setattr(
        delist_db,
        "get_delist_stock_info",
        lambda: {"000003.SZ": SimpleNamespace()},
    )

    def fake_update_full(codes):
        calls.append(list(codes))
        (tmp_path / "k-line" / "000003.SZ.parquet").write_text("ok")

    monkeypatch.setattr(kline_mootdx, "update_full", fake_update_full)

    update_all._ensure_delist_kline_mootdx()

    assert calls == [["000003.SZ"]]


def test_delist_kline_missing_after_mootdx_raises(tmp_path, monkeypatch):
    import data.update_all as update_all
    import data.db.delist as delist_db
    import data.kline_mootdx as kline_mootdx

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    (tmp_path / "k-line").mkdir(parents=True)
    monkeypatch.setattr(
        delist_db,
        "get_delist_stock_info",
        lambda: {"000003.SZ": SimpleNamespace()},
    )
    monkeypatch.setattr(kline_mootdx, "update_full", lambda codes: None)

    with pytest.raises(RuntimeError, match="mootdx 未补齐"):
        update_all._ensure_delist_kline_mootdx()


def test_update_indices_writes_close_column(tmp_path, monkeypatch):
    import akshare as ak
    import data.update_all as update_all

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)

    def fake_daily(symbol):
        return pd.DataFrame({
            "date": [date(2024, 1, 3), date(2024, 1, 2)],
            "open": [11.0, 10.0],
            "close": [11.5, 10.5],
        })

    monkeypatch.setattr(ak, "stock_zh_index_daily", fake_daily)

    update_all._update_indices(symbols=["sh000300"])

    table = pq.read_table(tmp_path / "index_sh000300_daily.parquet")
    assert table.schema.names == ["trade_date", "open", "close"]
    assert table.column("trade_date").to_pylist() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert table.column("close").to_pylist() == [10.5, 11.5]


def test_is_valid_index_parquet_requires_close(tmp_path):
    import data.update_all as update_all

    bad_path = tmp_path / "index_sh000300_daily.parquet"
    pq.write_table(pa.table({
        "trade_date": pa.array([date(2024, 1, 2)]),
        "open": pa.array([10.0]),
    }), bad_path)
    assert update_all._is_valid_index_parquet(bad_path) is False
    assert update_all._indices_ready_today(["sh000300"]) is False

    pq.write_table(pa.table({
        "trade_date": pa.array([date(2024, 1, 2)]),
        "open": pa.array([10.0]),
        "close": pa.array([10.5]),
    }), bad_path)
    assert update_all._is_valid_index_parquet(bad_path) is True

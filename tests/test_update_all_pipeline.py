from datetime import date
from types import SimpleNamespace

import pytest


class _FakeStore:
    _skip_update = False

    @staticmethod
    def _now():
        from datetime import datetime
        return datetime.now()


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


def test_delist_kline_missing_after_mootdx_warns_without_blocking(
    tmp_path, monkeypatch,
):
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

    missing = update_all._ensure_delist_kline_mootdx()

    assert missing == ["000003.SZ"]

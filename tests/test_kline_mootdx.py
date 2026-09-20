import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import kline_mootdx

_REAL_SECONDARY_EXACT_EXECUTABLE_DATES = (
    kline_mootdx._secondary_exact_executable_dates_by_code
)


@pytest.fixture(autouse=True)
def _isolate_secondary_snapshot(monkeypatch):
    """Unit tests never depend on whichever sealed secondary is in production."""

    monkeypatch.setattr(
        kline_mootdx,
        "_secondary_exact_executable_dates_by_code",
        lambda: {},
    )


class _EmptyMootdx:
    def __init__(self):
        self.calls = 0

    def xdxr(self, symbol):
        del symbol
        return pd.DataFrame()

    def bars(self, **kwargs):
        self.calls += 1
        return None


def test_download_skips_empty_kline_after_three_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(kline_mootdx, "RAW_DIR", tmp_path)
    warnings = []
    monkeypatch.setattr(kline_mootdx, "_warn_failed_codes",
                        lambda stage, failures: warnings.append((stage, failures)))
    mdx = _EmptyMootdx()

    result = kline_mootdx.download(mdx, ["000001.SZ"], "19900101", "20260708")

    assert result == {}
    assert mdx.calls == 3
    assert not (tmp_path / "000001.SZ.parquet").exists()
    failed_stage, failures = next(item for item in warnings if item[1])
    assert failed_stage == "全量下载"
    assert failures[0][0] == "000001.SZ"


def test_strict_download_rejects_partial_market_data(tmp_path, monkeypatch):
    monkeypatch.setattr(kline_mootdx, "RAW_DIR", tmp_path)
    monkeypatch.setattr(kline_mootdx, "_warn_failed_codes", lambda *_: None)

    with pytest.raises(kline_mootdx.KlineBatchError, match="全量下载不完整") as caught:
        kline_mootdx.download(
            _EmptyMootdx(),
            ["000001.SZ"],
            "19900101",
            "20260708",
            strict=True,
        )

    failure = caught.value.failures[0][1]
    assert isinstance(failure, kline_mootdx.KlineSourceError)
    assert failure.as_dict() == {
        "code": "000001.SZ",
        "operation": "bars",
        "kind": "empty_response",
        "detail": "mootdx 返回空 K 线；退市后源端不再保留代码历史是常见原因",
        "attempts": 3,
    }


class _RecentMootdx:
    def __init__(self):
        self.xdxr_calls = 0

    def xdxr(self, symbol):
        self.xdxr_calls += 1
        assert symbol == "000001"
        return pd.DataFrame(
            {
                "year": [2024],
                "month": [6],
                "day": [4],
                "category": [1],
                "fenhong": [1.0],
                "songzhuangu": [0.0],
                "peigu": [0.0],
                "peigujia": [0.0],
            }
        )

    def bars(self, **kwargs):
        return pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-06-03", "2024-06-04"]),
                "open": [10.0, 9.9],
                "high": [10.1, 10.0],
                "low": [9.9, 9.8],
                "close": [10.0, 9.95],
                "volume": [100.0, 120.0],
                "amount": [1000.0, 1194.0],
            }
        )


def test_recent_update_uses_xdxr_for_ex_right_preclose():
    mdx = _RecentMootdx()

    raw = kline_mootdx._fetch_recent_raw_with_retry(mdx, "000001.SZ", 3)

    assert mdx.xdxr_calls == 1
    assert raw.loc[1, "preClose"] == pytest.approx(9.9)


def test_nonstandard_xdxr_event_marks_preclose_unusable():
    closes = np.asarray([10.0, 20.0])
    times = pd.to_datetime(["2024-06-03", "2024-06-04"]).astype("int64") // 10**6
    xdxr = pd.DataFrame(
        {
            "year": [2024],
            "month": [6],
            "day": [4],
            "category": [9],
        }
    )

    result = kline_mootdx._compute_preclose(closes, np.asarray(times), xdxr)

    assert np.isnan(result[1])


def test_first_downloaded_bar_never_uses_its_own_close_as_preclose():
    result = kline_mootdx._compute_preclose(
        np.asarray([123.45]),
        np.asarray([pd.Timestamp("2024-06-03").value // 10**6]),
        None,
    )

    assert np.isnan(result[0])


def test_connect_mootdx_rejects_tcp_alive_server_with_empty_real_bar(
    monkeypatch,
):
    from mootdx.quotes import Quotes

    class _Client:
        def __init__(self, healthy):
            self.healthy = healthy

        def bars(self, **kwargs):
            del kwargs
            if not self.healthy:
                return pd.DataFrame()
            return pd.DataFrame({"close": [10.0]})

        def xdxr(self, **kwargs):
            del kwargs
            if not self.healthy:
                return pd.DataFrame()
            return pd.DataFrame(
                {
                    "year": [2024],
                    "month": [6],
                    "day": [14],
                    "category": [1],
                    "fenhong": [1.0],
                    "songzhuangu": [0.0],
                    "peigu": [0.0],
                    "peigujia": [0.0],
                }
            )

    calls = []

    def fake_factory(*, market, server=None, **kwargs):
        calls.append((market, server, kwargs))
        return _Client(server == ("good", 7709))

    monkeypatch.setattr(
        kline_mootdx,
        "TDX_SERVERS",
        (("empty", 7709), ("good", 7709)),
    )
    monkeypatch.setattr(kline_mootdx, "_probe_tdx_server", lambda *_: True)
    monkeypatch.setattr(Quotes, "factory", staticmethod(fake_factory))

    client = kline_mootdx._connect_mootdx()

    assert client.healthy is True
    assert [call[1] for call in calls] == [("empty", 7709), ("good", 7709)]


def test_xdxr_nonempty_response_requires_formula_columns():
    class _BadXdxr:
        def xdxr(self, symbol):
            del symbol
            return pd.DataFrame({"year": [2024]})

    with pytest.raises(RuntimeError, match="xdxr missing columns"):
        kline_mootdx._fetch_xdxr_with_retry(_BadXdxr(), "600000.SH")


def test_restore_backup_ignores_backup_preclose_and_uses_current_xdxr(
    tmp_path, monkeypatch,
):
    class _CurrentXdxr:
        def xdxr(self, symbol):
            assert symbol == "000003"
            return _RecentMootdx().xdxr("000001")

    backup_dir = tmp_path / "backup"
    production_dir = tmp_path / "production"
    backup_dir.mkdir()
    pd.DataFrame(
        {
            "time": (
                pd.to_datetime(["2024-06-03", "2024-06-04"])
                .astype("int64")
                // 10**6
            ),
            "open": [10.0, 9.9],
            "high": [10.1, 10.0],
            "low": [9.9, 9.8],
            "close": [10.0, 9.95],
            "volume": [100.0, 120.0],
            "amount": [1000.0, 1194.0],
            "preClose": [999.0, 999.0],
        }
    ).to_parquet(backup_dir / "000003.SZ.parquet", index=False)
    monkeypatch.setattr(kline_mootdx, "RAW_DIR", production_dir)
    monkeypatch.setattr(kline_mootdx, "_warn_failed_codes", lambda *_: None)

    restored = kline_mootdx.restore_backup_bars(
        ["000003.SZ"],
        backup_dir=backup_dir,
        mdx=_CurrentXdxr(),
    )

    assert set(restored) == {"000003.SZ"}
    result = pd.read_parquet(production_dir / "000003.SZ.parquet")
    assert np.isnan(result.loc[0, "preClose"])
    assert result.loc[1, "preClose"] == pytest.approx(9.9)
    assert 999.0 not in result["preClose"].to_numpy()
    assert not list(production_dir.glob("*.tmp.parquet"))


def test_restore_backup_failure_preserves_existing_production(
    tmp_path, monkeypatch,
):
    class _UnavailableXdxr:
        def xdxr(self, symbol):
            del symbol
            return None

    backup_dir = tmp_path / "backup"
    production_dir = tmp_path / "production"
    backup_dir.mkdir()
    production_dir.mkdir()
    source = pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-06-03").value // 10**6],
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [1.0],
            "amount": [10.0],
        }
    )
    source.to_parquet(backup_dir / "000003.SZ.parquet", index=False)
    output = production_dir / "000003.SZ.parquet"
    pd.DataFrame({"sentinel": [1]}).to_parquet(output, index=False)
    original = output.read_bytes()
    monkeypatch.setattr(kline_mootdx, "RAW_DIR", production_dir)
    monkeypatch.setattr(kline_mootdx, "_warn_failed_codes", lambda *_: None)

    with pytest.raises(RuntimeError, match="迁移不完整"):
        kline_mootdx.restore_backup_bars(
            ["000003.SZ"],
            backup_dir=backup_dir,
            mdx=_UnavailableXdxr(),
        )

    assert output.read_bytes() == original


def test_restore_rejects_empty_delisted_xdxr_as_ambiguous(
    tmp_path, monkeypatch,
):
    backup_dir = tmp_path / "backup"
    production_dir = tmp_path / "production"
    backup_dir.mkdir()
    pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-06-03").value // 10**6],
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [1.0],
            "amount": [10.0],
        }
    ).to_parquet(backup_dir / "000003.SZ.parquet", index=False)
    monkeypatch.setattr(kline_mootdx, "RAW_DIR", production_dir)
    monkeypatch.setattr(kline_mootdx, "_warn_failed_codes", lambda *_: None)

    with pytest.raises(kline_mootdx.KlineBatchError) as caught:
        kline_mootdx.restore_backup_bars(
            ["000003.SZ"],
            backup_dir=backup_dir,
            mdx=_EmptyMootdx(),
            use_baostock_fallback=False,
        )

    failure = caught.value.failures[0][1]
    assert isinstance(failure, kline_mootdx.KlineSourceError)
    assert failure.kind == "empty_response"
    assert failure.operation == "xdxr"
    assert not (production_dir / "000003.SZ.parquet").exists()


def test_connect_failure_is_explicitly_classified(monkeypatch):
    from mootdx.quotes import Quotes

    monkeypatch.setattr(kline_mootdx, "TDX_SERVERS", (("dead", 7709),))
    monkeypatch.setattr(kline_mootdx, "_probe_tdx_server", lambda *_: False)
    monkeypatch.setattr(
        Quotes,
        "factory",
        staticmethod(lambda **_kwargs: (_ for _ in ()).throw(OSError("offline"))),
    )

    with pytest.raises(kline_mootdx.KlineSourceError) as caught:
        kline_mootdx._connect_mootdx()

    assert caught.value.kind == "server_unavailable"
    assert caught.value.operation == "connect"


class _BaostockResult:
    def __init__(self, fields=(), rows=(), *, error_code="0", error_msg=""):
        self.fields = list(fields)
        self._rows = [list(row) for row in rows]
        self.error_code = error_code
        self.error_msg = error_msg
        self._index = -1

    def next(self):
        self._index += 1
        return self._index < len(self._rows)

    def get_row_data(self):
        return self._rows[self._index]


class _FakeBaostock:
    def __init__(self, history_rows, *, expire_calls=()):
        self.history_rows = list(history_rows)
        self.expire_calls = set(expire_calls)
        self.login_calls = 0
        self.query_calls = 0

    def login(self):
        self.login_calls += 1
        return _BaostockResult(error_code="0")

    def logout(self):
        return _BaostockResult(error_code="0")

    def _maybe_expire(self):
        self.query_calls += 1
        if self.query_calls in self.expire_calls:
            return _BaostockResult(
                error_code="10001001",
                error_msg="session expired",
            )
        return None

    def query_stock_basic(self):
        expired = self._maybe_expire()
        if expired is not None:
            return expired
        return _BaostockResult(
            ("code", "ipoDate", "outDate", "type", "status"),
            (("sz.000003", "2024-06-03", "2024-06-04", "1", "0"),),
        )

    def query_history_k_data_plus(self, code, fields, **kwargs):
        del kwargs
        assert code == "sz.000003"
        assert fields == "date,code,open,preclose,volume,amount,tradestatus"
        expired = self._maybe_expire()
        if expired is not None:
            return expired
        return _BaostockResult(fields.split(","), self.history_rows)


def _backup_frame():
    return pd.DataFrame(
        {
            "time": (
                pd.to_datetime(["2024-06-03", "2024-06-04"])
                .astype("int64")
                // 10**6
            ),
            "open": [10.0, 10.0],
            "high": [10.2, 10.0],
            "low": [9.8, 10.0],
            "close": [10.0, 10.0],
            "volume": [100.0, 0.0],
            "amount": [1000.0, 0.0],
            "preClose": [999.0, 999.0],
        }
    )


def _reference_rows():
    return (
        ("2024-06-03", "sz.000003", "10", "", "100", "1000", ""),
        ("2024-06-04", "sz.000003", "10", "10", "0", "0", "0"),
    )


def test_baostock_fallback_recomputes_preclose_masks_placeholder_and_seals_daily_evidence(
    tmp_path, monkeypatch,
):
    from datetime import date
    from data.db.delist import DelistStockInfo

    backup_dir = tmp_path / "backup"
    production_dir = tmp_path / "production"
    evidence_path = tmp_path / "evidence.parquet"
    backup_dir.mkdir()
    _backup_frame().to_parquet(backup_dir / "000003.SZ.parquet", index=False)
    monkeypatch.setattr(kline_mootdx, "RAW_DIR", production_dir)
    monkeypatch.setattr(kline_mootdx, "_warn_failed_codes", lambda *_: None)
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000003.SZ": DelistStockInfo(
                "退市样本", date(2024, 6, 3), date(2024, 6, 4)
            )
        },
    )

    restored = kline_mootdx.restore_backup_bars(
        ["000003.SZ"],
        backup_dir=backup_dir,
        evidence_path=evidence_path,
        mdx=_EmptyMootdx(),
        baostock_module=_FakeBaostock(_reference_rows()),
    )

    assert set(restored) == {"000003.SZ"}
    result = pd.read_parquet(production_dir / "000003.SZ.parquet")
    assert np.isnan(result.loc[0, "preClose"])
    assert np.isnan(result.loc[1, ["open", "high", "low", "close", "preClose"]]).all()
    assert result.loc[1, "volume"] == 0.0
    assert result.loc[1, "amount"] == 0.0
    assert 999.0 not in result["preClose"].dropna().to_numpy()
    evidence = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    assert len(evidence) == 2
    assert str(evidence.iloc[0]["listing_date"])[:10] == "2024-06-03"
    assert str(evidence.iloc[0]["first_executable_date"])[:10] == "2024-06-03"
    assert evidence["tradestatus"].tolist() == ["", "0"]
    assert kline_mootdx.find_placeholder_candidate_codes(
        kline_dir=production_dir,
        evidence_path=evidence_path,
    ) == ()


def test_restore_backup_baostock_evidence_publish_failure_rolls_back_kline(
    tmp_path, monkeypatch
):
    from datetime import date
    from data.db.delist import DelistStockInfo

    backup_dir = tmp_path / "backup"
    production_dir = tmp_path / "production"
    evidence_path = tmp_path / "evidence.parquet"
    backup_dir.mkdir()
    production_dir.mkdir()
    _backup_frame().to_parquet(backup_dir / "000003.SZ.parquet", index=False)
    production_path = production_dir / "000003.SZ.parquet"
    old_production = _backup_frame().iloc[:1].copy()
    old_production.to_parquet(production_path, index=False)
    original = production_path.read_bytes()
    monkeypatch.setattr(kline_mootdx, "RAW_DIR", production_dir)
    monkeypatch.setattr(kline_mootdx, "_warn_failed_codes", lambda *_: None)
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000003.SZ": DelistStockInfo(
                "退市样本", date(2024, 6, 3), date(2024, 6, 4)
            )
        },
    )
    original_replace = Path.replace

    def fail_evidence_publish(self, target):
        if Path(target) == evidence_path:
            raise OSError("evidence publish failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_evidence_publish)

    with pytest.raises(kline_mootdx.KlineBatchError):
        kline_mootdx.restore_backup_bars(
            ["000003.SZ"],
            backup_dir=backup_dir,
            evidence_path=evidence_path,
            mdx=_EmptyMootdx(),
            baostock_module=_FakeBaostock(_reference_rows()),
        )

    assert production_path.read_bytes() == original
    assert not evidence_path.exists()
    assert not kline_mootdx._baostock_evidence_manifest_path(
        evidence_path
    ).exists()


def test_baostock_fallback_rejects_incomplete_date_coverage_without_publish(
    tmp_path, monkeypatch,
):
    from datetime import date
    from data.db.delist import DelistStockInfo

    backup_dir = tmp_path / "backup"
    production_dir = tmp_path / "production"
    backup_dir.mkdir()
    _backup_frame().to_parquet(backup_dir / "000003.SZ.parquet", index=False)
    monkeypatch.setattr(kline_mootdx, "RAW_DIR", production_dir)
    monkeypatch.setattr(kline_mootdx, "_warn_failed_codes", lambda *_: None)
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000003.SZ": DelistStockInfo(
                "退市样本", date(2024, 6, 3), date(2024, 6, 4)
            )
        },
    )
    source = _FakeBaostock(_reference_rows()[:1])

    with pytest.raises(kline_mootdx.KlineBatchError) as caught:
        kline_mootdx.restore_backup_bars(
            ["000003.SZ"],
            backup_dir=backup_dir,
            evidence_path=tmp_path / "evidence.parquet",
            mdx=_EmptyMootdx(),
            baostock_module=source,
        )

    failure = caught.value.failures[0][1]
    assert isinstance(failure, kline_mootdx.KlineSourceError)
    assert failure.kind == "coverage_mismatch"
    assert not (production_dir / "000003.SZ.parquet").exists()


def test_baostock_delist_boundary_accepts_later_administrative_date(monkeypatch):
    from datetime import date
    from data.db.delist import DelistStockInfo

    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000003.SZ": DelistStockInfo(
                "退市样本", date(2024, 6, 3), date(2024, 6, 9)
            )
        },
    )
    basic = pd.Series(
        {
            "ipoDate": "2024-06-03",
            "outDate": "2024-06-04",
            "type": "1",
            "status": "0",
        }
    )
    history = pd.DataFrame(
        _reference_rows(),
        columns=(
            "date",
            "code",
            "open",
            "preclose",
            "volume",
            "amount",
            "tradestatus",
        ),
    )
    history["date"] = pd.to_datetime(history["date"]).dt.date
    for field in ("open", "preclose", "volume", "amount"):
        history[field] = pd.to_numeric(history[field], errors="coerce")

    production, _evidence = kline_mootdx._baostock_production_and_evidence(
        "000003.SZ",
        _backup_frame(),
        basic,
        history,
        require_delisted=True,
    )

    assert len(production) == 2

    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000003.SZ": DelistStockInfo(
                "退市样本", date(2024, 6, 3), date(2024, 6, 12)
            )
        },
    )
    with pytest.raises(kline_mootdx.KlineSourceError) as caught:
        kline_mootdx._baostock_production_and_evidence(
            "000003.SZ",
            _backup_frame(),
            basic,
            history,
            require_delisted=True,
        )
    assert caught.value.kind == "lifecycle_mismatch"


def test_baostock_reference_keeps_verified_price_but_masks_false_zero_turnover():
    backup = _backup_frame()
    backup.loc[0, ["volume", "amount"]] = 0.0
    basic = pd.Series(
        {
            "ipoDate": "2024-06-03",
            "outDate": "2024-06-04",
            "type": "1",
            "status": "0",
        }
    )
    history = pd.DataFrame(
        _reference_rows(),
        columns=(
            "date",
            "code",
            "open",
            "preclose",
            "volume",
            "amount",
            "tradestatus",
        ),
    )
    history["date"] = pd.to_datetime(history["date"]).dt.date
    for field in ("open", "preclose", "volume", "amount"):
        history[field] = pd.to_numeric(history[field], errors="coerce")

    production, _evidence = kline_mootdx._baostock_production_and_evidence(
        "000003.SZ",
        backup,
        basic,
        history,
        require_delisted=False,
    )

    assert production.loc[0, "open"] == 10.0
    assert np.isnan(production.loc[0, "volume"])
    assert np.isnan(production.loc[0, "amount"])


def test_listing_evidence_update_accepts_already_normalized_local_frame(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    local = _backup_frame()
    local.loc[1, ["open", "high", "low", "close", "preClose"]] = np.nan
    local.to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"

    succeeded = kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )

    assert succeeded == ("000003.SZ",)
    evidence = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    assert evidence["stock_code"].unique().tolist() == ["000003.SZ"]
    assert evidence["source_payload_sha256"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()
    assert kline_mootdx._baostock_evidence_manifest_path(evidence_path).exists()


def test_baostock_v4_rejects_source_tamper_before_manifest_check(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    _backup_frame().to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"
    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )
    tampered = pd.read_parquet(evidence_path)
    tampered.loc[1, "direct_preclose"] = 12345.0
    tampered.to_parquet(evidence_path, index=False)

    with pytest.raises(ValueError, match="source_payload_sha256"):
        kline_mootdx.load_baostock_daily_evidence(evidence_path)


def test_baostock_v4_manifest_rejects_self_rehashed_row_tamper(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    _backup_frame().to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"
    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )
    tampered = pd.read_parquet(evidence_path)
    tampered.loc[1, "direct_preclose"] = 12345.0
    tampered["source_payload_sha256"] = tampered.apply(
        kline_mootdx._source_payload_sha256,
        axis=1,
    )
    tampered.to_parquet(evidence_path, index=False)

    with pytest.raises(ValueError, match="snapshot manifest"):
        kline_mootdx.load_baostock_daily_evidence(evidence_path)


def test_baostock_v4_requires_manifest_sidecar(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    _backup_frame().to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"
    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )
    kline_mootdx._baostock_evidence_manifest_path(evidence_path).unlink()

    with pytest.raises(ValueError, match="manifest 缺失"):
        kline_mootdx.load_baostock_daily_evidence(evidence_path)


def test_baostock_loader_rejects_every_non_v4_schema(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    _backup_frame().to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"
    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )
    legacy = pd.read_parquet(evidence_path)
    legacy["schema_version"] = "baostock-daily-reference-v3"
    legacy.to_parquet(evidence_path, index=False)

    with pytest.raises(ValueError, match="schema_version 不兼容"):
        kline_mootdx.load_baostock_daily_evidence(evidence_path)


def test_baostock_v4_save_rejects_history_date_regression(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    _backup_frame().to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"
    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )
    previous = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    original = evidence_path.read_bytes()
    manifest_path = kline_mootdx._baostock_evidence_manifest_path(evidence_path)
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="历史日期回退"):
        kline_mootdx._save_baostock_evidence_atomic(
            [previous.iloc[:1].copy()], evidence_path
        )

    assert evidence_path.read_bytes() == original
    assert manifest_path.read_bytes() == original_manifest


def test_baostock_v4_incremental_merge_normalizes_retained_date_types(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    _backup_frame().to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"
    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )
    retained = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    incoming = retained.copy()
    incoming["stock_code"] = "000004.SZ"
    incoming["baostock_code"] = "sz.000004"
    incoming["source_payload_sha256"] = incoming.apply(
        kline_mootdx._source_payload_sha256,
        axis=1,
    )

    kline_mootdx._save_baostock_evidence_atomic([incoming], evidence_path)

    merged = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    assert merged["stock_code"].drop_duplicates().tolist() == [
        "000003.SZ",
        "000004.SZ",
    ]


def _seed_audited_second_row(evidence_path: Path, kline_path: Path) -> str:
    evidence = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    frame = pd.read_parquet(kline_path)
    day = pd.to_datetime(frame.loc[1, "time"], unit="ms").date()
    bound = kline_mootdx._bind_evidence_to_applied_dates(
        evidence,
        frame,
        {day},
    )
    kline_mootdx._write_baostock_evidence_stage(bound, evidence_path)
    return str(
        bound.loc[
            pd.to_datetime(bound["date"]).dt.date.eq(day),
            "applied_k_sha256",
        ].iloc[0]
    )


def test_evidence_only_incremental_refresh_preserves_audited_marker_and_hash(
    tmp_path,
):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    kline_path = kline_dir / "000003.SZ.parquet"
    _backup_frame().to_parquet(kline_path, index=False)
    evidence_path = tmp_path / "evidence.parquet"
    source = _FakeBaostock(_reference_rows())
    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=source,
    )
    expected_hash = _seed_audited_second_row(evidence_path, kline_path)

    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )

    refreshed = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    audited = refreshed[refreshed["normalization_applied"]]
    assert len(audited) == 1
    assert audited.iloc[0]["applied_k_sha256"] == expected_hash


@pytest.mark.parametrize("failure", ["missing_audited_date", "status_conflict"])
def test_incremental_merge_rejects_audited_date_loss_or_source_conflict(
    tmp_path, failure
):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    kline_path = kline_dir / "000003.SZ.parquet"
    _backup_frame().to_parquet(kline_path, index=False)
    evidence_path = tmp_path / "evidence.parquet"
    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )
    _seed_audited_second_row(evidence_path, kline_path)
    previous = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    original = evidence_path.read_bytes()
    manifest_path = kline_mootdx._baostock_evidence_manifest_path(evidence_path)
    original_manifest = manifest_path.read_bytes()
    incoming = previous.copy()
    target = incoming["normalization_applied"].to_numpy(
        dtype=np.bool_, copy=True
    )
    incoming.loc[target, "normalization_applied"] = False
    incoming.loc[target, "applied_k_sha256"] = kline_mootdx.UNAPPLIED_K_SHA256
    if failure == "missing_audited_date":
        incoming = incoming.loc[~target].reset_index(drop=True)
        message = "历史日期回退"
    else:
        incoming.loc[target, "source_tradestatus"] = "1"
        incoming.loc[target, "tradestatus"] = "1"
        incoming["source_payload_sha256"] = incoming.apply(
            kline_mootdx._source_payload_sha256,
            axis=1,
        )
        message = "已审计日期源状态/payload 冲突"

    with pytest.raises(ValueError, match=message):
        kline_mootdx._save_baostock_evidence_atomic([incoming], evidence_path)

    assert evidence_path.read_bytes() == original
    assert manifest_path.read_bytes() == original_manifest


def test_baostock_session_allows_one_relogin_for_each_failed_query():
    source = _FakeBaostock(_reference_rows(), expire_calls={1, 3})
    session = kline_mootdx._BaostockSession(source)
    try:
        first = session.query("*", "basic", source.query_stock_basic)
        second = session.query("*", "basic", source.query_stock_basic)
    finally:
        session.close()

    assert not first.empty and not second.empty
    assert source.login_calls == 3


def test_reconcile_existing_placeholder_is_atomic_and_does_not_use_old_preclose(
    tmp_path,
):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    path = kline_dir / "000003.SZ.parquet"
    _backup_frame().to_parquet(path, index=False)

    published = kline_mootdx.reconcile_baostock_placeholder_bars(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=tmp_path / "evidence.parquet",
        baostock_module=_FakeBaostock(_reference_rows()),
    )

    assert published == {"000003.SZ": path}
    result = pd.read_parquet(path)
    assert np.isnan(result.loc[1, "open"])
    assert 999.0 not in result["preClose"].dropna().to_numpy()
    assert not list(kline_dir.glob("*.tmp.parquet"))


def test_baostock_executability_uses_only_status_or_reference_open():
    history = pd.DataFrame(
        {
            "tradestatus": ["1", "0", "", ""],
            "open": [np.nan, 10.0, 10.0, np.nan],
            # These deliberately contradict the expected result.  They must
            # never decide T-open executability.
            "volume": [np.nan, 100.0, np.nan, 100.0],
            "amount": [np.nan, 1000.0, np.nan, 1000.0],
        }
    )

    assert kline_mootdx._baostock_executable_mask(history).tolist() == [
        True,
        False,
        True,
        False,
    ]


def _mini_status_correction_contract(history):
    source = history.copy()
    source["source_tradestatus"] = source["tradestatus"]
    target = source[source["date"].isin(
        (pd.Timestamp("2024-06-04").date(), pd.Timestamp("2024-06-05").date())
    )]
    return {
        "start": pd.Timestamp("2024-06-04").date(),
        "end": pd.Timestamp("2024-06-05").date(),
        "row_count": 2,
        "date_set_sha256": kline_mootdx._status_correction_date_set_sha256(
            target["date"]
        ),
        "source_rows_sha256": (
            kline_mootdx._status_correction_source_rows_sha256(target)
        ),
        "last_executable_boundary": pd.Timestamp("2024-06-03").date(),
        "termination_boundary": pd.Timestamp("2024-06-06").date(),
        "source_status": "1",
        "effective_status": "0",
        "correction_id": "test-fixed-status-correction-v1",
    }


def _mini_status_correction_history():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06"]
            ).date,
            "code": ["sz.000003"] * 4,
            "open": [10.0, 10.0, 10.0, 10.0],
            "preclose": [np.nan, 10.0, 10.0, 10.0],
            "volume": [100.0, 0.0, 0.0, np.nan],
            "amount": [1000.0, 0.0, 0.0, np.nan],
            "tradestatus": ["1", "1", "1", "0"],
        }
    )


def test_fixed_status_correction_preserves_source_and_is_idempotent(monkeypatch):
    history = _mini_status_correction_history()
    contract = _mini_status_correction_contract(history)
    monkeypatch.setattr(
        kline_mootdx,
        "BAOSTOCK_STATUS_CORRECTIONS",
        {"000003.SZ": contract},
    )
    monkeypatch.setattr(
        kline_mootdx,
        "_secondary_exact_executable_dates_by_code",
        lambda: {},
    )

    corrected = kline_mootdx._apply_baostock_status_corrections(
        "000003.SZ", history
    )
    replayed = kline_mootdx._apply_baostock_status_corrections(
        "000003.SZ", corrected
    )

    pd.testing.assert_frame_equal(corrected, replayed, check_exact=True)
    target = corrected["status_correction"].eq(contract["correction_id"])
    assert int(target.sum()) == 2
    assert corrected.loc[target, "source_tradestatus"].eq("1").all()
    assert corrected.loc[target, "tradestatus"].eq("0").all()
    assert corrected.loc[~target, "tradestatus"].eq(
        corrected.loc[~target, "source_tradestatus"]
    ).all()


@pytest.mark.parametrize("tamper", ["missing", "payload", "boundary"])
def test_fixed_status_correction_rejects_whole_changed_batch(monkeypatch, tamper):
    history = _mini_status_correction_history()
    contract = _mini_status_correction_contract(history)
    monkeypatch.setattr(
        kline_mootdx,
        "BAOSTOCK_STATUS_CORRECTIONS",
        {"000003.SZ": contract},
    )
    changed = history.copy()
    if tamper == "missing":
        changed = changed[changed["date"] != pd.Timestamp("2024-06-05").date()]
    elif tamper == "payload":
        changed.loc[
            changed["date"].eq(pd.Timestamp("2024-06-04").date()), "open"
        ] = 10.01
    else:
        changed.loc[
            changed["date"].eq(pd.Timestamp("2024-06-03").date()),
            "tradestatus",
        ] = "0"

    with pytest.raises(ValueError, match="status correction"):
        kline_mootdx._apply_baostock_status_corrections(
            "000003.SZ", changed
        )


def test_status_correction_masks_only_exact_dates_and_binds_evidence(
    monkeypatch, tmp_path
):
    history = _mini_status_correction_history()
    contract = _mini_status_correction_contract(history)
    monkeypatch.setattr(
        kline_mootdx,
        "BAOSTOCK_STATUS_CORRECTIONS",
        {"000003.SZ": contract},
    )
    basic = pd.Series(
        {
            "ipoDate": "2024-06-03",
            "outDate": "2024-06-06",
            "type": "1",
            "status": "0",
        }
    )
    backup = pd.DataFrame(
        {
            "time": pd.to_datetime(history["date"]).astype("int64") // 10**6,
            "open": [10.0] * 4,
            "high": [10.1] * 4,
            "low": [9.9] * 4,
            "close": [10.0] * 4,
            "volume": [100.0, np.nan, np.nan, 0.0],
            "amount": [1000.0, np.nan, np.nan, 0.0],
            "preClose": [np.nan, 10.0, 10.0, 10.0],
        }
    )

    production, evidence = kline_mootdx._baostock_production_and_evidence(
        "000003.SZ",
        backup,
        basic,
        history,
        require_delisted=False,
    )
    target = evidence["status_correction"].eq(contract["correction_id"])
    assert evidence.loc[target, "normalization_applied"].all()
    assert evidence.loc[target, "source_tradestatus"].eq("1").all()
    assert evidence.loc[target, "tradestatus"].eq("0").all()
    assert production.loc[[0], "open"].notna().all()
    assert production.loc[[1, 2, 3], "open"].isna().all()

    replayed, replayed_evidence = (
        kline_mootdx._baostock_production_and_evidence(
            "000003.SZ",
            production,
            basic,
            history,
            require_delisted=False,
        )
    )
    pd.testing.assert_frame_equal(replayed, production, check_exact=True)
    assert replayed_evidence.loc[
        replayed_evidence["status_correction"].eq(contract["correction_id"]),
        "normalization_applied",
    ].all()

    evidence_path = tmp_path / "corrected-evidence.parquet"
    kline_mootdx._write_baostock_evidence_stage(evidence, evidence_path)
    loaded = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    assert loaded.loc[
        loaded["status_correction"].eq(contract["correction_id"]),
        "source_tradestatus",
    ].eq("1").all()


def test_targeted_status_reconcile_is_transactional_and_repeatable(
    tmp_path, monkeypatch
):
    history = _mini_status_correction_history()
    contract = _mini_status_correction_contract(history)
    monkeypatch.setattr(
        kline_mootdx,
        "BAOSTOCK_STATUS_CORRECTIONS",
        {"000003.SZ": contract},
    )
    monkeypatch.setattr(
        kline_mootdx,
        "_secondary_exact_executable_dates_by_code",
        lambda: {},
    )

    class _CorrectionBaostock(_FakeBaostock):
        def query_stock_basic(self):
            expired = self._maybe_expire()
            if expired is not None:
                return expired
            return _BaostockResult(
                ("code", "ipoDate", "outDate", "type", "status"),
                (("sz.000003", "2024-06-03", "2024-06-06", "1", "0"),),
            )

    source_rows = tuple(
        tuple("" if pd.isna(value) else str(value) for value in row)
        for row in history[
            ["date", "code", "open", "preclose", "volume", "amount", "tradestatus"]
        ].itertuples(index=False, name=None)
    )
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    kline_path = kline_dir / "000003.SZ.parquet"
    local = pd.DataFrame(
        {
            "time": pd.to_datetime(history["date"]).astype("int64") // 10**6,
            "open": [10.0] * 4,
            "high": [10.1] * 4,
            "low": [9.9] * 4,
            "close": [10.0] * 4,
            "volume": [100.0, 0.0, 0.0, 0.0],
            "amount": [1000.0, 0.0, 0.0, 0.0],
            "preClose": [np.nan, 10.0, 10.0, 10.0],
        }
    )
    local.to_parquet(kline_path, index=False)
    evidence_path = tmp_path / "evidence.parquet"

    first = kline_mootdx.reconcile_baostock_status_corrections(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_CorrectionBaostock(source_rows),
    )
    first_frame = pd.read_parquet(kline_path)
    first_evidence = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    second = kline_mootdx.reconcile_baostock_status_corrections(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_CorrectionBaostock(source_rows),
    )

    assert first == second == {"000003.SZ": kline_path}
    pd.testing.assert_frame_equal(
        pd.read_parquet(kline_path), first_frame, check_exact=True
    )
    pd.testing.assert_frame_equal(
        kline_mootdx.load_baostock_daily_evidence(evidence_path),
        first_evidence,
        check_exact=True,
    )
    assert np.isnan(first_frame.loc[[1, 2, 3], "open"]).all()
    assert kline_mootdx._baostock_evidence_manifest_path(evidence_path).exists()


def test_baostock_history_preserves_empty_turnover_as_nan():
    rows = (
        ("2024-06-03", "sz.000003", "", "", "", "", "1"),
        ("2024-06-04", "sz.000003", "10", "10", "", "", "0"),
    )
    source = _FakeBaostock(rows)
    session = kline_mootdx._BaostockSession(source)
    try:
        history = kline_mootdx._baostock_history(
            session,
            "000003.SZ",
            pd.Timestamp("2024-06-03").date(),
            pd.Timestamp("2024-06-04").date(),
        )
    finally:
        session.close()

    assert np.isnan(history.loc[0, "open"])
    assert np.isnan(history.loc[:, ["volume", "amount"]].to_numpy()).all()
    assert kline_mootdx._baostock_executable_mask(history).tolist() == [True, False]


def test_baostock_basic_missing_symbol_isolated_from_other_code(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    _backup_frame().to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    _backup_frame().to_parquet(kline_dir / "000004.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"

    with pytest.raises(kline_mootdx.KlineBatchError) as caught:
        kline_mootdx.update_baostock_listing_evidence(
            ["000003.SZ", "000004.SZ"],
            kline_dir=kline_dir,
            evidence_path=evidence_path,
            baostock_module=_FakeBaostock(_reference_rows()),
        )

    assert caught.value.succeeded == ()
    assert [code for code, _exc in caught.value.failures] == ["000004.SZ"]
    assert caught.value.failures[0][1].kind == "missing_symbol"
    assert not evidence_path.exists()
    assert not kline_mootdx._baostock_evidence_manifest_path(
        evidence_path
    ).exists()


def test_listing_evidence_requires_every_local_kline_date(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    _backup_frame().to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"

    with pytest.raises(kline_mootdx.KlineBatchError) as caught:
        kline_mootdx.update_baostock_listing_evidence(
            ["000003.SZ"],
            kline_dir=kline_dir,
            evidence_path=evidence_path,
            baostock_module=_FakeBaostock(_reference_rows()[:1]),
        )

    assert caught.value.failures[0][1].kind == "coverage_mismatch"
    assert not evidence_path.exists()


def _coverage_local_frame(days):
    return pd.DataFrame(
        {
            "time": [pd.Timestamp(day).value // 10**6 for day in days],
        }
    )


def _validate_coverage(code, local_days, source_days, *, ipo_date):
    parsed_ipo = pd.Timestamp(ipo_date).date()
    return kline_mootdx._validate_baostock_local_coverage(
        code,
        _coverage_local_frame(local_days),
        pd.DataFrame(
            {"date": [pd.Timestamp(day).date() for day in source_days]}
        ),
        query_start=min(
            parsed_ipo,
            min(pd.Timestamp(day).date() for day in local_days),
        ),
        query_end=max(pd.Timestamp(day).date() for day in local_days),
        ipo_date=parsed_ipo,
        out_date=None,
    )


def test_secondary_coverage_contracts_are_fixed_in_primary_layer():
    assert set(kline_mootdx.SECONDARY_EXACT_DATE_LIMITS) == {
        "000004.SZ",
        "000012.SZ",
        "001872.SZ",
    }
    limit = kline_mootdx.SECONDARY_DATE_SET_LIMITS["600018.SH"]
    assert limit == {
        "count": 1448,
        "first": pd.Timestamp("2000-07-19").date(),
        "last": pd.Timestamp("2006-09-25").date(),
        "sha256": (
            "84e618e5115678c41b208c6766b14bc9"
            "be0424131eb086d256fddac6b5a7941f"
        ),
        "required": frozenset({pd.Timestamp("2001-08-16").date()}),
    }


def test_coverage_accepts_only_exact_600018_secondary_dates(monkeypatch):
    expected = frozenset(
        {
            pd.Timestamp("2000-07-19").date(),
            pd.Timestamp("2001-08-16").date(),
            pd.Timestamp("2006-09-25").date(),
        }
    )
    monkeypatch.setitem(
        kline_mootdx.SECONDARY_DATE_SET_LIMITS,
        "600018.SH",
        {
            "count": len(expected),
            "first": min(expected),
            "last": max(expected),
            "sha256": kline_mootdx._date_set_sha256(expected),
            "required": frozenset({pd.Timestamp("2001-08-16").date()}),
        },
    )
    module = types.ModuleType("data.kline_secondary_evidence")
    module.load_verified_daily_evidence = lambda: pd.DataFrame(
        {
            "stock_code": ["600018.SH"] * len(expected),
            "date": sorted(expected),
        }
    )
    module.covered_local_dates = lambda code, dates: frozenset(
        set(dates).intersection(expected) if code == "600018.SH" else ()
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)

    _validate_coverage(
        "600018.SH",
        ["2000-07-19", "2001-08-16", "2006-09-25", "2006-10-26"],
        ["2006-10-26"],
        ipo_date="2006-10-26",
    )

    with pytest.raises(kline_mootdx.KlineSourceError) as invented_date:
        _validate_coverage(
            "600018.SH",
            [
                "2000-07-19",
                "2001-08-16",
                "2001-08-17",
                "2006-09-25",
                "2006-10-26",
            ],
            ["2006-10-26"],
            ipo_date="2006-10-26",
        )
    assert invented_date.value.kind == "coverage_mismatch"

    with pytest.raises(kline_mootdx.KlineSourceError) as missing_real_date:
        _validate_coverage(
            "600018.SH",
            ["2000-07-19", "2006-09-25", "2006-10-26"],
            ["2006-10-26"],
            ipo_date="2006-10-26",
        )
    assert missing_real_date.value.kind == "coverage_mismatch"
    assert "2001-08-16" in str(missing_real_date.value)


def test_secondary_coverage_accepts_only_exact_hash_sealed_dates(monkeypatch):
    calls = []
    module = types.ModuleType("data.kline_secondary_evidence")

    def covered_local_dates(code, local_dates):
        calls.append((code, local_dates))
        return frozenset(local_dates)

    module.covered_local_dates = covered_local_dates
    monkeypatch.setitem(sys.modules, module.__name__, module)

    _validate_coverage(
        "001872.SZ",
        ["2012-09-10", "2012-09-11"],
        ["2012-09-11"],
        ipo_date="1993-05-05",
    )
    _validate_coverage(
        "000004.SZ",
        ["1991-01-19", "1991-01-20"],
        ["1991-01-20"],
        ipo_date="1991-01-14",
    )
    _validate_coverage(
        "000012.SZ",
        ["1992-05-03", "1992-05-04"],
        ["1992-05-04"],
        ipo_date="1992-02-28",
    )

    assert calls == [
        ("001872.SZ", frozenset({pd.Timestamp("2012-09-10").date()})),
        ("000004.SZ", frozenset({pd.Timestamp("1991-01-19").date()})),
        ("000012.SZ", frozenset({pd.Timestamp("1992-05-03").date()})),
    ]


def test_secondary_exact_date_accepts_finite_negative_adjusted_reference_open(
    monkeypatch,
):
    module = types.ModuleType("data.kline_secondary_evidence")
    module.load_verified_daily_evidence = lambda: pd.DataFrame(
        {
            "stock_code": ["000540.SZ", "001872.SZ"],
            "date": ["1995-04-28", "1993-07-15"],
            "tradestatus": ["1", "1"],
            # Evidence-only adjusted prices; never copied into canonical K.
            "reference_open": [-0.50, -8.44],
        }
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)

    result = _REAL_SECONDARY_EXACT_EXECUTABLE_DATES()

    assert result == {
        "000540.SZ": frozenset({pd.Timestamp("1995-04-28").date()}),
        "001872.SZ": frozenset({pd.Timestamp("1993-07-15").date()}),
    }


def test_secondary_coverage_cannot_broaden_code_or_date_whitelist(monkeypatch):
    module = types.ModuleType("data.kline_secondary_evidence")
    module.covered_local_dates = lambda _code, _dates: frozenset(
        {pd.Timestamp("2012-09-11").date()}
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(ValueError, match="非请求日期|精确日期白名单"):
        _validate_coverage(
            "001872.SZ",
            ["2012-09-10"],
            [],
            ipo_date="1993-05-05",
        )

    with pytest.raises(kline_mootdx.KlineSourceError) as unlisted:
        _validate_coverage(
            "000003.SZ",
            ["2024-06-03"],
            [],
            ipo_date="2024-06-03",
        )
    assert unlisted.value.kind == "coverage_mismatch"


def test_baostock_history_rejects_empty_internal_five_year_chunk():
    fields = "date,code,open,preclose,volume,amount,tradestatus".split(",")

    class _ChunkedBaostock:
        def login(self):
            return _BaostockResult(error_code="0")

        def logout(self):
            return _BaostockResult(error_code="0")

        def query_history_k_data_plus(self, code, requested_fields, **kwargs):
            assert code == "sz.000003"
            assert requested_fields.split(",") == fields
            if kwargs["start_date"] == "2010-01-01":
                return _BaostockResult(
                    fields,
                    (("2010-01-04", code, "10", "", "1", "10", "1"),),
                )
            return _BaostockResult(fields, ())

    source = _ChunkedBaostock()
    session = kline_mootdx._BaostockSession(source)
    try:
        with pytest.raises(kline_mootdx.KlineSourceError) as caught:
            kline_mootdx._baostock_history(
                session,
                "000003.SZ",
                pd.Timestamp("2010-01-01").date(),
                pd.Timestamp("2019-12-31").date(),
                lifecycle_start=pd.Timestamp("2010-01-01").date(),
                lifecycle_end=pd.Timestamp("2019-12-31").date(),
            )
    finally:
        session.close()

    assert caught.value.kind == "incomplete_response"


def _mootdx_page(dates):
    dates = pd.to_datetime(list(dates))
    size = len(dates)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": np.full(size, 10.0),
            "high": np.full(size, 10.1),
            "low": np.full(size, 9.9),
            "close": np.full(size, 10.0),
            "volume": np.full(size, 100.0),
            "amount": np.full(size, 1000.0),
        }
    )


def test_mootdx_full_pagination_rejects_empty_after_full_page():
    first = _mootdx_page(
        pd.date_range("2020-01-01", periods=kline_mootdx.PAGE_SIZE, freq="D")
    )

    class _Interrupted:
        def bars(self, **kwargs):
            return first if kwargs["start"] == 0 else pd.DataFrame()

    with pytest.raises(kline_mootdx.KlineSourceError) as caught:
        kline_mootdx._fetch_bars_all(_Interrupted(), "000001.SZ")

    assert caught.value.kind == "pagination_incomplete"


def test_mootdx_full_pagination_rejects_short_internal_page():
    first = _mootdx_page(["2024-06-04", "2024-06-03"])
    older = _mootdx_page(["2024-05-31"])

    class _ShortInternal:
        def bars(self, **kwargs):
            return first if kwargs["start"] == 0 else older

    with pytest.raises(kline_mootdx.KlineSourceError) as caught:
        kline_mootdx._fetch_bars_all(_ShortInternal(), "000001.SZ")

    assert caught.value.kind == "pagination_incomplete"


def test_full_download_preserves_old_file_when_new_history_regresses(
    tmp_path, monkeypatch
):
    old = _mootdx_page(["2024-06-03", "2024-06-04"])
    old_raw = kline_mootdx._mootdx_bars_to_df(old)
    assert old_raw is not None
    output = tmp_path / "000001.SZ.parquet"
    old_raw.to_parquet(output, index=False)
    original = output.read_bytes()

    class _Regressed:
        def xdxr(self, symbol):
            del symbol
            return pd.DataFrame()

        def bars(self, **kwargs):
            if kwargs["offset"] == 1:
                return pd.DataFrame()
            return _mootdx_page(["2024-06-04"])

    failures = []
    monkeypatch.setattr(kline_mootdx, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        kline_mootdx,
        "_warn_failed_codes",
        lambda _stage, batch: failures.extend(batch),
    )

    result = kline_mootdx.download(
        _Regressed(), ["000001.SZ"], "19900101", "20240604"
    )

    assert result == {}
    assert output.read_bytes() == original
    assert any(exc.kind == "history_regression" for _code, exc in failures)


def test_reconcile_kline_stage_failure_publishes_neither_evidence_nor_kline(
    tmp_path, monkeypatch
):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    path = kline_dir / "000003.SZ.parquet"
    local = _backup_frame()
    local.loc[0, ["volume", "amount"]] = 0.0
    local.to_parquet(path, index=False)
    original = path.read_bytes()
    evidence_path = tmp_path / "evidence.parquet"
    rows = (
        ("2024-06-03", "sz.000003", "", "", "", "", "1"),
        ("2024-06-04", "sz.000003", "10", "10", "0", "0", "0"),
    )
    monkeypatch.setattr(
        kline_mootdx,
        "_write_parquet_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("publish failed")),
    )

    with pytest.raises(kline_mootdx.KlineBatchError):
        kline_mootdx.reconcile_baostock_placeholder_bars(
            ["000003.SZ"],
            kline_dir=kline_dir,
            evidence_path=evidence_path,
            baostock_module=_FakeBaostock(rows),
        )

    assert path.read_bytes() == original
    assert not evidence_path.exists()
    assert not kline_mootdx._baostock_evidence_manifest_path(
        evidence_path
    ).exists()


def test_reconcile_rejects_concurrent_kline_change_without_overwriting_it(
    tmp_path, monkeypatch
):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    path = kline_dir / "000003.SZ.parquet"
    _backup_frame().to_parquet(path, index=False)
    evidence_path = tmp_path / "evidence.parquet"
    original_assert = kline_mootdx.assert_file_states_unchanged
    concurrent_bytes: dict[str, bytes] = {}

    def mutate_then_assert(expected, *, label):
        concurrent = _backup_frame()
        concurrent.loc[0, "close"] = 9.75
        concurrent.to_parquet(path, index=False)
        concurrent_bytes["value"] = path.read_bytes()
        original_assert(expected, label=label)

    monkeypatch.setattr(
        kline_mootdx,
        "assert_file_states_unchanged",
        mutate_then_assert,
    )

    with pytest.raises(kline_mootdx.KlineBatchError) as caught:
        kline_mootdx.reconcile_baostock_placeholder_bars(
            ["000003.SZ"],
            kline_dir=kline_dir,
            evidence_path=evidence_path,
            baostock_module=_FakeBaostock(_reference_rows()),
        )

    assert "并发变化" in str(caught.value.failures[0][1])
    assert path.read_bytes() == concurrent_bytes["value"]
    assert not evidence_path.exists()
    assert not kline_mootdx._baostock_evidence_manifest_path(
        evidence_path
    ).exists()


def test_reconcile_manifest_publish_failure_rolls_back_kline_and_evidence_pair(
    tmp_path, monkeypatch
):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    kline_path = kline_dir / "000003.SZ.parquet"
    _backup_frame().to_parquet(kline_path, index=False)
    evidence_path = tmp_path / "evidence.parquet"
    kline_mootdx.update_baostock_listing_evidence(
        ["000003.SZ"],
        kline_dir=kline_dir,
        evidence_path=evidence_path,
        baostock_module=_FakeBaostock(_reference_rows()),
    )
    manifest_path = kline_mootdx._baostock_evidence_manifest_path(evidence_path)
    originals = {
        kline_path: kline_path.read_bytes(),
        evidence_path: evidence_path.read_bytes(),
        manifest_path: manifest_path.read_bytes(),
    }
    original_replace = Path.replace

    def fail_final_manifest(self, target):
        if Path(target) == manifest_path and self != manifest_path:
            raise OSError("manifest publish failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_final_manifest)

    with pytest.raises(kline_mootdx.KlineBatchError):
        kline_mootdx.reconcile_baostock_placeholder_bars(
            ["000003.SZ"],
            kline_dir=kline_dir,
            evidence_path=evidence_path,
            baostock_module=_FakeBaostock(_reference_rows()),
        )

    for path, original in originals.items():
        assert path.read_bytes() == original
    assert not list(tmp_path.rglob("*.rollback"))


def test_transaction_keeps_backups_when_rollback_itself_fails(
    tmp_path, monkeypatch
):
    target_a = tmp_path / "a.txt"
    target_b = tmp_path / "b.txt"
    stage_a = tmp_path / "a.stage"
    stage_b = tmp_path / "b.stage"
    target_a.write_text("old-a", encoding="utf-8")
    target_b.write_text("old-b", encoding="utf-8")
    stage_a.write_text("new-a", encoding="utf-8")
    stage_b.write_text("new-b", encoding="utf-8")
    original_replace = Path.replace
    original_copy2 = kline_mootdx.shutil.copy2

    def fail_second_publish(self, target):
        if self == stage_b:
            raise OSError("second publish failed")
        return original_replace(self, target)

    def fail_restore(source, target, *args, **kwargs):
        if ".rollback" in Path(source).name and Path(target) == target_a:
            raise OSError("rollback copy failed")
        return original_copy2(source, target, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", fail_second_publish)
    monkeypatch.setattr(kline_mootdx.shutil, "copy2", fail_restore)

    with pytest.raises(RuntimeError, match="备份已保留"):
        kline_mootdx._replace_staged_files_transactionally(
            [(stage_a, target_a), (stage_b, target_b)],
            token="rollback-test",
        )

    backups = list(tmp_path.glob("*.rollback"))
    assert len(backups) == 2
    assert {path.read_text(encoding="utf-8") for path in backups} == {
        "old-a",
        "old-b",
    }


def test_transaction_rolls_back_keyboard_interrupt(tmp_path, monkeypatch):
    target_a = tmp_path / "a.txt"
    target_b = tmp_path / "b.txt"
    stage_a = tmp_path / "a.stage"
    stage_b = tmp_path / "b.stage"
    target_a.write_text("old-a", encoding="utf-8")
    target_b.write_text("old-b", encoding="utf-8")
    stage_a.write_text("new-a", encoding="utf-8")
    stage_b.write_text("new-b", encoding="utf-8")
    original_replace = Path.replace

    def interrupt_second_publish(self, target):
        if self == stage_b:
            raise KeyboardInterrupt()
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", interrupt_second_publish)

    with pytest.raises(KeyboardInterrupt):
        kline_mootdx._replace_staged_files_transactionally(
            [(stage_a, target_a), (stage_b, target_b)],
            token="interrupt-test",
        )

    assert target_a.read_text(encoding="utf-8") == "old-a"
    assert target_b.read_text(encoding="utf-8") == "old-b"
    assert not list(tmp_path.glob("*.rollback"))


def test_verified_nan_turnover_is_not_reclassified_as_zero_placeholder(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    frame = _backup_frame().iloc[:1].copy()
    frame.loc[0, ["volume", "amount"]] = np.nan
    frame.to_parquet(kline_dir / "000003.SZ.parquet", index=False)

    assert kline_mootdx.find_placeholder_candidate_codes(
        kline_dir=kline_dir,
        evidence_path=tmp_path / "absent-evidence.parquet",
    ) == ()


def test_candidate_scan_uses_audit_marker_and_secondary_exact_not_turnover(
    tmp_path, monkeypatch
):
    code = "000003.SZ"
    day = pd.Timestamp("2024-06-04").date()
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    kline_path = kline_dir / f"{code}.parquet"
    evidence_path = tmp_path / "evidence.parquet"
    basic = pd.Series(
        {
            "ipoDate": "2024-06-03",
            "outDate": "2024-06-04",
            "type": "1",
            "status": "0",
        }
    )
    history = pd.DataFrame(
        _reference_rows(),
        columns=(
            "date",
            "code",
            "open",
            "preclose",
            "volume",
            "amount",
            "tradestatus",
        ),
    )
    history["date"] = pd.to_datetime(history["date"]).dt.date
    for field in ("open", "preclose", "volume", "amount"):
        history[field] = pd.to_numeric(history[field], errors="coerce")
    masked, evidence = kline_mootdx._baostock_production_and_evidence(
        code,
        _backup_frame(),
        basic,
        history,
        require_delisted=False,
    )

    def publish(frame: pd.DataFrame) -> None:
        bound = kline_mootdx._bind_evidence_to_applied_dates(
            evidence,
            frame,
            {day},
        )
        frame.to_parquet(kline_path, index=False)
        kline_mootdx._write_baostock_evidence_stage(bound, evidence_path)

    old_primary_only = masked.copy()
    old_primary_only.loc[1, ["open", "high", "low", "close", "preClose"]] = [
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ]
    old_primary_only.loc[1, ["volume", "amount"]] = np.nan
    publish(old_primary_only)
    assert kline_mootdx.find_placeholder_candidate_codes(
        kline_dir=kline_dir,
        evidence_path=evidence_path,
    ) == (code,)

    publish(masked)
    assert kline_mootdx.find_placeholder_candidate_codes(
        kline_dir=kline_dir,
        evidence_path=evidence_path,
    ) == ()

    publish(old_primary_only)
    monkeypatch.setattr(
        kline_mootdx,
        "_secondary_exact_executable_dates_by_code",
        lambda: {code: frozenset({day})},
    )
    assert kline_mootdx.find_placeholder_candidate_codes(
        kline_dir=kline_dir,
        evidence_path=evidence_path,
    ) == ()


class _LiveFakeBaostock(_FakeBaostock):
    def query_stock_basic(self):
        expired = self._maybe_expire()
        if expired is not None:
            return expired
        return _BaostockResult(
            ("code", "ipoDate", "outDate", "type", "status"),
            (("sz.000003", "2024-06-03", "", "1", "1"),),
        )


def test_atomic_v4_rebuild_restores_only_masked_intersection_and_keeps_new_main_rows(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        kline_mootdx,
        "_secondary_exact_executable_dates_by_code",
        lambda: {"000003.SZ": frozenset({pd.Timestamp("2024-06-03").date()})},
    )
    kline_dir = tmp_path / "kline"
    backup_dir = tmp_path / "backup"
    kline_dir.mkdir()
    backup_dir.mkdir()
    backup = _backup_frame()
    backup.loc[0, ["volume", "amount"]] = 0.0
    backup.to_parquet(backup_dir / "000003.SZ.parquet", index=False)

    current = backup.copy()
    current.loc[0, ["open", "high", "low", "close", "preClose"]] = np.nan
    current.loc[1, ["open", "high", "low", "close", "preClose"]] = np.nan
    newer = pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-06-05").value // 10**6],
            "open": [11.0],
            "high": [11.2],
            "low": [10.8],
            "close": [11.1],
            "volume": [321.0],
            "amount": [3563.1],
            "preClose": [10.0],
        }
    )
    current = pd.concat([current, newer], ignore_index=True)
    current.to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"
    rows = (
        ("2024-06-03", "sz.000003", "", "", "", "", "1"),
        ("2024-06-04", "sz.000003", "10", "10", "0", "0", "0"),
        ("2024-06-05", "sz.000003", "11", "10", "321", "3563.1", "1"),
    )

    restored = kline_mootdx.rebuild_masked_kline_rows_from_backups(
        ["000003.SZ"],
        kline_dir=kline_dir,
        backup_dir=backup_dir,
        evidence_path=evidence_path,
        baostock_module=_LiveFakeBaostock(rows),
    )

    assert restored == {"000003.SZ": 1}
    result = pd.read_parquet(kline_dir / "000003.SZ.parquet")
    assert result.loc[0, "open"] == 10.0
    assert np.isnan(result.loc[0, "volume"])
    assert np.isnan(result.loc[0, "amount"])
    assert np.isnan(result.loc[1, "open"])
    pd.testing.assert_series_equal(
        result.loc[2], newer.loc[0], check_names=False, check_dtype=False
    )
    evidence = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    assert evidence["normalization_applied"].tolist() == [True, False, False]
    kline_mootdx._validate_evidence_application_hashes(result, evidence)
    assert not list(kline_dir.glob(".*.stage.parquet"))
    assert not list(kline_dir.glob(".*.rollback.parquet"))


def test_v4_rebuild_remasks_audited_candidate_without_secondary_exact(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        kline_mootdx,
        "_secondary_exact_executable_dates_by_code",
        lambda: {},
    )
    kline_dir = tmp_path / "kline"
    backup_dir = tmp_path / "backup"
    kline_dir.mkdir()
    backup_dir.mkdir()
    backup = _backup_frame()
    newer = pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-06-05").value // 10**6],
            "open": [11.0],
            "high": [11.2],
            "low": [10.8],
            "close": [11.1],
            "volume": [321.0],
            "amount": [3563.1],
            "preClose": [10.0],
        }
    )
    archived = pd.concat([backup, newer], ignore_index=True)
    archived.to_parquet(backup_dir / "000003.SZ.parquet", index=False)
    current = archived.copy()
    current.loc[1, ["open", "high", "low", "close", "preClose"]] = np.nan
    current.to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    history = pd.DataFrame(
        (
            ("2024-06-03", "sz.000003", "10", "", "0", "0", "1"),
            ("2024-06-04", "sz.000003", "10", "10", "0", "0", "0"),
            ("2024-06-05", "sz.000003", "11", "10", "321", "3563.1", "1"),
        ),
        columns="date,code,open,preclose,volume,amount,tradestatus".split(","),
    )
    history["date"] = pd.to_datetime(history["date"]).dt.date
    for field in ("open", "preclose", "volume", "amount"):
        history[field] = pd.to_numeric(history[field], errors="coerce")
    basic = pd.Series(
        {
            "ipoDate": "2024-06-03",
            "outDate": "",
            "type": "1",
            "status": "1",
        }
    )
    evidence, *_ = kline_mootdx._build_baostock_evidence(
        "000003.SZ", basic, history
    )
    evidence = kline_mootdx._bind_evidence_to_applied_dates(
        evidence,
        current,
        {pd.Timestamp("2024-06-03").date()},
    )
    evidence_path = tmp_path / "evidence.parquet"
    kline_mootdx._write_baostock_evidence_stage(evidence, evidence_path)

    restored = kline_mootdx.rebuild_masked_kline_rows_from_backups(
        ["000003.SZ"],
        kline_dir=kline_dir,
        backup_dir=backup_dir,
        evidence_path=evidence_path,
        baostock_module=_LiveFakeBaostock(tuple(
            tuple("" if pd.isna(value) else str(value) for value in row)
            for row in history[
                ["date", "code", "open", "preclose", "volume", "amount", "tradestatus"]
            ].itertuples(index=False, name=None)
        )),
    )

    assert restored == {"000003.SZ": 0}
    result = pd.read_parquet(kline_dir / "000003.SZ.parquet")
    assert np.isnan(result.loc[0, "open"])
    assert result.loc[0, "volume"] == 0.0
    assert result.loc[2, "open"] == 11.0
    refreshed = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    candidate = refreshed["date"].eq(np.datetime64("2024-06-03", "D"))
    assert refreshed.loc[candidate, "normalization_applied"].all()
    kline_mootdx._validate_evidence_application_hashes(result, refreshed)


def test_atomic_v4_rebuild_does_not_publish_any_code_when_basic_is_missing(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        kline_mootdx,
        "_secondary_exact_executable_dates_by_code",
        lambda: {},
    )
    kline_dir = tmp_path / "kline"
    backup_dir = tmp_path / "backup"
    kline_dir.mkdir()
    backup_dir.mkdir()
    originals = {}
    for code in ("000003.SZ", "000004.SZ"):
        path = kline_dir / f"{code}.parquet"
        _backup_frame().to_parquet(path, index=False)
        _backup_frame().to_parquet(backup_dir / f"{code}.parquet", index=False)
        originals[code] = path.read_bytes()
    evidence_path = tmp_path / "evidence.parquet"

    with pytest.raises(kline_mootdx.KlineBatchError) as caught:
        kline_mootdx.rebuild_masked_kline_rows_from_backups(
            ["000003.SZ", "000004.SZ"],
            kline_dir=kline_dir,
            backup_dir=backup_dir,
            evidence_path=evidence_path,
            baostock_module=_FakeBaostock(_reference_rows()),
        )

    assert any(code == "000004.SZ" for code, _exc in caught.value.failures)
    assert not evidence_path.exists()
    for code, original in originals.items():
        assert (kline_dir / f"{code}.parquet").read_bytes() == original


def test_v4_rebuild_allows_new_evidence_only_code_without_backup(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        kline_mootdx,
        "_secondary_exact_executable_dates_by_code",
        lambda: {},
    )
    kline_dir = tmp_path / "kline"
    backup_dir = tmp_path / "backup"
    kline_dir.mkdir()
    backup_dir.mkdir()
    _backup_frame().to_parquet(kline_dir / "000003.SZ.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"

    for _run in range(2):
        rebuilt = kline_mootdx.rebuild_masked_kline_rows_from_backups(
            ["000003.SZ"],
            kline_dir=kline_dir,
            backup_dir=backup_dir,
            evidence_path=evidence_path,
            baostock_module=_FakeBaostock(_reference_rows()),
        )
        assert rebuilt == {"000003.SZ": 0}

    evidence = kline_mootdx.load_baostock_daily_evidence(evidence_path)
    assert not evidence["normalization_applied"].any()

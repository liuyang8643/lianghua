from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import kline_history_repairs as repairs


class _Result:
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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _base_source_values(code: str, day: str) -> dict[str, str]:
    serial = sum(ord(value) for value in f"{code}{day}")
    base = 10.0 + (serial % 100) / 100.0
    return {
        "date": day,
        "code": repairs._baostock_symbol(code),
        "open": f"{base:.4f}",
        "high": f"{base + 1.0:.4f}",
        "low": f"{base - 1.0:.4f}",
        "close": f"{base + 0.5:.4f}",
        "preclose": f"{base - 0.5:.4f}",
        "volume": str(10_000 + serial),
        "amount": f"{1_000_000.0 + serial:.4f}",
        "tradestatus": "1",
    }


def _source_values(code: str, day: str) -> dict[str, str]:
    for operation in ("upsert", "patch_preclose"):
        sealed = repairs.SEALED_MULTISOURCE_RECORDS.get((code, day, operation))
        if sealed is not None:
            return {
                "date": day,
                "code": repairs._baostock_symbol(code),
                **{field: str(sealed[field]) for field in repairs.RAW_FIELDS},
                "tradestatus": str(sealed["tradestatus"]),
            }
    if (code, day) == ("300029.SZ", "2026-07-09"):
        return {
            "date": day,
            "code": repairs._baostock_symbol(code),
            "open": "0.13",
            "high": "0.14",
            "low": "0.12",
            "close": "0.12",
            "preclose": "0.14",
            "volume": "20594300",
            "amount": "2658657.09",
            "tradestatus": "1",
        }
    values = _base_source_values(code, day)
    predecessor = next(
        (
            target_day
            for (target_code, target_day), successor_day in (
                repairs.HISTORY_REPAIR_SUCCESSORS.items()
            )
            if target_code == code and successor_day == day
        ),
        None,
    )
    if predecessor is not None:
        values["preclose"] = _base_source_values(code, predecessor)["close"]
    return values


def _source_rows() -> dict[str, dict[str, dict[str, str]]]:
    rows: dict[str, dict[str, dict[str, str]]] = {
        code: {} for code in repairs.TARGET_CODES
    }
    for code, day, operation in repairs.EXPECTED_OPERATIONS:
        if operation in {"upsert", "patch_preclose"}:
            rows[code][day] = _source_values(code, day)
    for (code, _day), successor_day in repairs.HISTORY_REPAIR_SUCCESSORS.items():
        rows[code][successor_day] = _source_values(code, successor_day)
    return rows


class _FakeBaostock:
    def __init__(self, failure: str | None = None):
        self.failure = failure
        self.login_calls = 0
        self.history_calls = 0
        self.rows = _source_rows()

    def login(self):
        self.login_calls += 1
        return _Result()

    def logout(self):
        return _Result()

    def query_stock_basic(self):
        fields = ("code", "ipoDate", "outDate", "type", "status")
        rows = [
            (repairs._baostock_symbol(code), ipo, "", "1", "1")
            for code, ipo in repairs.EXPECTED_IPO_DATES.items()
        ]
        return _Result(fields, rows)

    def query_history_k_data_plus(
        self,
        code,
        fields,
        *,
        start_date,
        end_date,
        frequency,
        adjustflag,
    ):
        assert fields == repairs.QUERY_FIELDS
        assert frequency == "d"
        assert adjustflag == "3"
        self.history_calls += 1
        canonical = f"{code[3:]}.{code[:2].upper()}"
        selected = [
            row.copy()
            for day, row in sorted(self.rows[canonical].items())
            if start_date <= day <= end_date
        ]
        if self.history_calls == 1 and self.failure == "uncovered":
            selected = selected[1:]
        if selected and self.history_calls == 1:
            if self.failure == "nontrade":
                selected[0]["tradestatus"] = "0"
            elif self.failure == "wrong_code":
                selected[0]["code"] = "sz.999999"
            elif self.failure == "bad_price":
                selected[0]["preclose"] = "0"
        output_fields = fields.split(",")
        if self.history_calls == 1 and self.failure == "missing_field":
            output_fields.remove("preclose")
        return _Result(
            output_fields,
            [[row[field] for field in output_fields] for row in selected],
        )


class _MustStayOffline:
    def login(self):
        raise AssertionError("sealed snapshot replay must not contact Baostock")


@dataclass(frozen=True)
class _FakeSources:
    mootdx_dir: Path
    qmt_dir: Path
    secondary_path: Path
    secondary: pd.DataFrame
    tencent: dict[str, object]


def _secondary_evidence() -> pd.DataFrame:
    records = []
    for code, days in repairs.HISTORY_REPAIR_49.items():
        for day in days:
            records.append(
                {
                    "stock_code": code,
                    "date": day,
                    "payload_sha256": _digest(f"payload|{code}|{day}"),
                    "source_row_sha256": _digest(f"source|{code}|{day}"),
                }
            )
    frame = pd.DataFrame.from_records(records).sort_values(
        ["stock_code", "date"], kind="stable"
    ).reset_index(drop=True)
    assert len(frame) == 49
    return frame


def _archive_row(code: str, day: str) -> dict[str, float | np.int64]:
    source = _source_values(code, day)
    return {
        "time": repairs._time_ms(day),
        "open": float(source["open"]),
        "high": float(source["high"]),
        "low": float(source["low"]),
        "close": float(source["close"]),
        "volume": float(source["volume"]),
        "amount": float(source["amount"]),
        "preClose": float(source["preclose"]),
    }


def _write_archive(
    directory: Path,
    expected: frozenset[tuple[str, str]],
) -> None:
    directory.mkdir(parents=True)
    by_code: dict[str, list[str]] = {}
    for code, day in expected:
        by_code.setdefault(code, []).append(day)
    for code, days in sorted(by_code.items()):
        frame = pd.DataFrame(
            [_archive_row(code, day) for day in sorted(days)],
            columns=repairs.PRODUCTION_COLUMNS,
        )
        frame["time"] = frame["time"].astype(np.int64)
        for field in repairs.PRODUCTION_COLUMNS[1:]:
            frame[field] = frame[field].astype(np.float64)
        frame.to_parquet(directory / f"{code}.parquet", index=False)
    assert sum(
        len(pd.read_parquet(path)) for path in directory.glob("*.parquet")
    ) == len(expected)


@pytest.fixture
def fake_sources(tmp_path, monkeypatch) -> _FakeSources:
    mootdx_dir = tmp_path / "sources" / "mootdx"
    qmt_dir = tmp_path / "sources" / "qmt"
    secondary_path = tmp_path / "sources" / "secondary.parquet"
    _write_archive(mootdx_dir, repairs._ARCHIVED_MOOTDX_REPAIR_SET)
    _write_archive(qmt_dir, repairs._ARCHIVED_QMT_REPAIR_SET)
    secondary = _secondary_evidence()
    secondary_path.parent.mkdir(parents=True, exist_ok=True)
    secondary.to_parquet(secondary_path, index=False)

    def _verified(*, path: Path) -> pd.DataFrame:
        assert Path(path) == secondary_path
        return secondary.copy()

    monkeypatch.setattr(repairs, "verified_history_repair_evidence", _verified)
    target = secondary.loc[
        secondary["stock_code"].eq("300029.SZ")
        & secondary["date"].eq("2026-07-09")
    ].iloc[0]
    tencent = {
        "date": "2026-07-09",
        "open": 0.13,
        "close": 0.12,
        "high": 0.14,
        "low": 0.12,
        "volume": 205_943.0,
        "source_row_sha256": target["source_row_sha256"],
    }
    return _FakeSources(
        mootdx_dir=mootdx_dir,
        qmt_dir=qmt_dir,
        secondary_path=secondary_path,
        secondary=secondary,
        tencent=tencent,
    )


def _download_fake_snapshot(
    sources: _FakeSources,
    baostock: _FakeBaostock | None = None,
    *,
    tencent: dict[str, object] | None = None,
) -> pd.DataFrame:
    return repairs._download_snapshot(
        baostock or _FakeBaostock(),
        mootdx_archive_dir=sources.mootdx_dir,
        qmt_archive_dir=sources.qmt_dir,
        secondary_snapshot_path=sources.secondary_path,
        tencent_ohlcv=tencent or sources.tencent,
    )


def _seal_fake_manifest(
    sources: _FakeSources,
    monkeypatch,
) -> pd.DataFrame:
    evidence = _download_fake_snapshot(sources)
    monkeypatch.setattr(
        repairs,
        "EXPECTED_SNAPSHOT_MANIFEST_SHA256",
        repairs._snapshot_manifest_sha256(evidence),
    )
    return evidence


def _row(day: str, seed: float) -> dict[str, float | np.int64]:
    return {
        "time": repairs._time_ms(day),
        "open": seed,
        "high": seed + 1.0,
        "low": seed - 1.0,
        "close": seed + 0.5,
        "volume": seed * 10.0,
        "amount": seed * 1000.0,
        "preClose": seed - 0.5,
    }


def _stable_day(code: str) -> str:
    if code == "300029.SZ":
        return "2026-07-08"
    if code == "600018.SH":
        return "2002-01-03"
    return "2000-01-03"


def _write_local_klines(kline_dir: Path) -> dict[str, pd.DataFrame]:
    kline_dir.mkdir(parents=True)
    originals: dict[str, pd.DataFrame] = {}
    successors_by_code: dict[str, set[str]] = {}
    for (code, _target), successor in repairs.HISTORY_REPAIR_SUCCESSORS.items():
        successors_by_code.setdefault(code, set()).add(successor)
    for code in repairs.TARGET_CODES:
        rows: list[dict[str, float | np.int64]] = []
        for index, day in enumerate(repairs.DELETE_DATES.get(code, ())):
            rows.append(_row(day, 50.0 + index))
        for index, day in enumerate(repairs.DELETE_NONTRADING_DATES.get(code, ())):
            rows.append(_row(day, 55.0 + index))
        if code in repairs.DELETE_BEFORE:
            rows.extend([_row("1993-06-08", 61.0), _row("1993-08-05", 62.0)])
        upserts = repairs.UPSERT_DATES.get(code, ())
        if upserts:
            rows.append(_row(upserts[0], 70.0))
        for index, day in enumerate(sorted(successors_by_code.get(code, set()))):
            rows.append(_row(day, 75.0 + index))
        for index, day in enumerate(repairs.PATCH_PRECLOSE_DATES.get(code, ())):
            rows.append(_row(day, 80.0 + index))
        rows.append(_row(_stable_day(code), 100.0))
        frame = pd.DataFrame(rows, columns=repairs.PRODUCTION_COLUMNS)
        frame = frame.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        if code in {"600607.SH", "600652.SH", "600656.SH"}:
            frame["time"] -= 15 * 60 * 60 * 1000
        frame["time"] = frame["time"].astype(np.int64)
        for field in repairs.PRODUCTION_COLUMNS[1:]:
            frame[field] = frame[field].astype(np.float64)
        frame.to_parquet(kline_dir / f"{code}.parquet", index=False)
        originals[code] = frame
    return originals


def _ensure(
    *,
    kline_dir: Path,
    snapshot_path: Path,
    sources: _FakeSources,
    baostock=None,
    refresh_snapshot: bool = False,
    tencent: dict[str, object] | None = None,
) -> tuple[str, ...]:
    return repairs.ensure_verified_history_repairs(
        kline_dir=kline_dir,
        snapshot_path=snapshot_path,
        baostock_module=baostock,
        refresh_snapshot=refresh_snapshot,
        mootdx_archive_dir=sources.mootdx_dir,
        qmt_archive_dir=sources.qmt_dir,
        secondary_snapshot_path=sources.secondary_path,
        tencent_ohlcv=tencent or sources.tencent,
    )


def _date_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result["time"], unit="ms").dt.strftime("%Y-%m-%d")
    return result


def _kline_bytes(kline_dir: Path) -> dict[str, bytes]:
    return {
        code: (kline_dir / f"{code}.parquet").read_bytes()
        for code in repairs.TARGET_CODES
    }


def _assert_zero_publish(
    kline_dir: Path,
    before: dict[str, bytes],
    snapshot_path: Path,
) -> None:
    assert _kline_bytes(kline_dir) == before
    assert not snapshot_path.exists()
    assert not list(kline_dir.parent.rglob("*.tmp.parquet"))
    assert not list(kline_dir.parent.rglob("*.rollback"))


def test_download_then_offline_replay_obeys_complete_v5_contract(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    originals = _write_local_klines(kline_dir)
    expected_evidence = _seal_fake_manifest(fake_sources, monkeypatch)
    source = _FakeBaostock()

    changed = _ensure(
        kline_dir=kline_dir,
        snapshot_path=snapshot_path,
        sources=fake_sources,
        baostock=source,
    )

    assert changed == repairs.TARGET_CODES
    assert source.login_calls == 1
    assert source.history_calls > 0
    evidence = repairs._load_snapshot(snapshot_path)
    pd.testing.assert_frame_equal(evidence, expected_evidence, check_exact=True)
    assert evidence["source"].value_counts().to_dict() == dict(
        repairs.EXPECTED_SOURCE_COUNTS
    )
    assert evidence["operation"].value_counts().to_dict() == dict(
        repairs.EXPECTED_OPERATION_COUNTS
    )
    history_keys = set(
        (code, day)
        for code, days in repairs.HISTORY_REPAIR_49.items()
        for day in days
    )
    history = evidence.loc[
        [
            operation == "upsert" and (code, day) in history_keys
            for code, day, operation in evidence[
                ["code", "date", "operation"]
            ].itertuples(index=False, name=None)
        ]
    ]
    assert len(history) == 49
    assert history["turnover_policy"].value_counts().to_dict() == {
        repairs.TURNOVER_NAN: 34,
        repairs.TURNOVER_PRESERVE: 15,
    }
    assert history["source"].value_counts().to_dict() == {
        repairs.ARCHIVED_QMT_SOURCE: 42,
        repairs.ARCHIVED_MOOTDX_SOURCE: 6,
        repairs.SNAPSHOT_SOURCE: 1,
    }
    assert int(evidence["successor_date"].ne("").sum()) == 27

    indexed_evidence = evidence.set_index(["code", "date", "operation"])
    for code in repairs.TARGET_CODES:
        result = _date_rows(pd.read_parquet(kline_dir / f"{code}.parquet"))
        original = _date_rows(originals[code])
        assert not any(day in result.index for day in repairs.DELETE_DATES.get(code, ()))
        assert not any(
            day in result.index
            for day in repairs.DELETE_NONTRADING_DATES.get(code, ())
        )
        assert not (result.index < repairs.EXPECTED_IPO_DATES[code]).any()
        for day in repairs.UPSERT_DATES.get(code, ()):
            actual = result.loc[day]
            source_row = indexed_evidence.loc[(code, day, "upsert")]
            assert actual["open"] == float(source_row["open"])
            assert actual["preClose"] == float(source_row["preclose"])
            if source_row["turnover_policy"] == repairs.TURNOVER_NAN:
                assert np.isnan(actual["volume"])
                assert np.isnan(actual["amount"])
            else:
                divisor = repairs.UNIT_CONTRACTS[source_row["unit_contract"]][
                    "volume_divisor"
                ]
                assert actual["volume"] == float(source_row["volume"]) / float(divisor)
                assert actual["amount"] == float(source_row["amount"])
            successor = source_row["successor_date"]
            if successor:
                assert result.loc[successor, "preClose"] == float(
                    source_row["successor_preclose"]
                )
        for day in repairs.PATCH_PRECLOSE_DATES.get(code, ()):
            actual = result.loc[day]
            before = original.loc[day]
            source_row = indexed_evidence.loc[(code, day, "patch_preclose")]
            assert actual["preClose"] == float(source_row["preclose"])
            pd.testing.assert_series_equal(
                actual.drop("preClose"),
                before.drop("preClose"),
                check_names=False,
                check_exact=True,
            )
        stable_day = _stable_day(code)
        pd.testing.assert_series_equal(
            result.loc[stable_day],
            original.loc[stable_day],
            check_names=False,
            check_exact=True,
        )
        timestamps = pd.to_datetime(result["time"], unit="ms")
        expected_hour = (
            0 if code in {"600607.SH", "600652.SH", "600656.SH"} else 15
        )
        assert (timestamps.dt.hour == expected_hour).all()

    before_bytes = _kline_bytes(kline_dir)
    snapshot_bytes = snapshot_path.read_bytes()
    replayed = _ensure(
        kline_dir=kline_dir,
        snapshot_path=snapshot_path,
        sources=fake_sources,
        baostock=_MustStayOffline(),
        tencent={"offline": True},
    )
    assert replayed == ()
    assert snapshot_path.read_bytes() == snapshot_bytes
    assert _kline_bytes(kline_dir) == before_bytes


@pytest.mark.parametrize(
    "failure",
    ["uncovered", "nontrade", "wrong_code", "bad_price", "missing_field"],
)
def test_any_baostock_failure_rejects_whole_batch_without_publish(
    tmp_path,
    fake_sources,
    failure,
):
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    _write_local_klines(kline_dir)
    before = _kline_bytes(kline_dir)

    with pytest.raises((ValueError, KeyError)):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=_FakeBaostock(failure),
        )

    _assert_zero_publish(kline_dir, before, snapshot_path)


def test_archive_missing_one_required_row_rejects_batch_without_publish(
    tmp_path,
    fake_sources,
):
    path = fake_sources.mootdx_dir / "000005.SZ.parquet"
    frame = pd.read_parquet(path).iloc[1:].reset_index(drop=True)
    frame.to_parquet(path, index=False)
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    _write_local_klines(kline_dir)
    before = _kline_bytes(kline_dir)

    with pytest.raises(ValueError, match="archive 目标未覆盖"):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=_FakeBaostock(),
        )

    _assert_zero_publish(kline_dir, before, snapshot_path)


def test_archive_tamper_is_rehashed_but_rejected_by_fixed_manifest(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    _seal_fake_manifest(fake_sources, monkeypatch)
    path = fake_sources.mootdx_dir / "000005.SZ.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "open"] += 0.01
    frame.to_parquet(path, index=False)
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    _write_local_klines(kline_dir)
    before = _kline_bytes(kline_dir)

    with pytest.raises(ValueError, match="manifest_sha256"):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=_FakeBaostock(),
        )

    _assert_zero_publish(kline_dir, before, snapshot_path)


def test_secondary_incomplete_batch_rejects_without_publish(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    incomplete = fake_sources.secondary.iloc[1:].reset_index(drop=True)
    monkeypatch.setattr(
        repairs,
        "verified_history_repair_evidence",
        lambda *, path: incomplete.copy(),
    )
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    _write_local_klines(kline_dir)
    before = _kline_bytes(kline_dir)

    with pytest.raises(ValueError, match="49-row 覆盖不完整"):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=_FakeBaostock(),
        )

    _assert_zero_publish(kline_dir, before, snapshot_path)


@pytest.mark.parametrize("mismatch", ["hash", "ohlc", "volume", "amount"])
def test_targeted_tencent_mismatch_rejects_without_publish(
    tmp_path,
    fake_sources,
    mismatch,
):
    tencent = dict(fake_sources.tencent)
    baostock = _FakeBaostock()
    if mismatch == "hash":
        tencent["source_row_sha256"] = "f" * 64
    elif mismatch == "ohlc":
        tencent["open"] = 0.131
    elif mismatch == "volume":
        tencent["volume"] = float(tencent["volume"]) + 1.0
    else:
        baostock.rows["300029.SZ"]["2026-07-09"]["amount"] = "1.0"
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    _write_local_klines(kline_dir)
    before = _kline_bytes(kline_dir)

    with pytest.raises(ValueError, match="300029"):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=baostock,
            tencent=tencent,
        )

    _assert_zero_publish(kline_dir, before, snapshot_path)


def test_sealed_snapshot_missing_target_does_not_touch_kline(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "patches.parquet"
    _write_local_klines(kline_dir)
    evidence = _seal_fake_manifest(fake_sources, monkeypatch)
    evidence.iloc[1:].reset_index(drop=True).to_parquet(snapshot_path, index=False)
    before = _kline_bytes(kline_dir)

    with pytest.raises(ValueError, match="目标覆盖不完整"):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=_MustStayOffline(),
        )

    assert _kline_bytes(kline_dir) == before


def test_sealed_snapshot_rejects_payload_tampering(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    snapshot_path = tmp_path / "patches.parquet"
    evidence = _seal_fake_manifest(fake_sources, monkeypatch)
    source_row = evidence["operation"].eq("upsert").idxmax()
    evidence.loc[source_row, "payload_sha256"] = "0" * 64
    evidence.to_parquet(snapshot_path, index=False)

    with pytest.raises(ValueError, match="payload_sha256"):
        repairs._load_snapshot(snapshot_path)


def test_sealed_snapshot_rejects_self_rehashed_payload_tampering_by_manifest(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    snapshot_path = tmp_path / "patches.parquet"
    evidence = _seal_fake_manifest(fake_sources, monkeypatch)
    source_row = evidence["operation"].eq("upsert").idxmax()
    evidence.loc[source_row, "open"] += 0.01
    evidence.loc[source_row, "source_row_sha256"] = repairs._source_row_sha256(
        evidence.loc[source_row]
    )
    evidence.loc[source_row, "payload_sha256"] = repairs._payload_sha256(
        evidence.loc[source_row]
    )
    evidence.to_parquet(snapshot_path, index=False)

    with pytest.raises(ValueError, match="manifest_sha256"):
        repairs._load_snapshot(snapshot_path)


def test_missing_local_successor_rejects_entire_batch_without_publish(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    _seal_fake_manifest(fake_sources, monkeypatch)
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    _write_local_klines(kline_dir)
    (code, _target), successor = next(iter(repairs.HISTORY_REPAIR_SUCCESSORS.items()))
    path = kline_dir / f"{code}.parquet"
    frame = pd.read_parquet(path)
    dates = pd.to_datetime(frame["time"], unit="ms").dt.strftime("%Y-%m-%d")
    frame = frame.loc[~dates.eq(successor)].reset_index(drop=True)
    frame.to_parquet(path, index=False)
    before = _kline_bytes(kline_dir)

    with pytest.raises(ValueError, match="successor .*不存在"):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=_FakeBaostock(),
        )

    _assert_zero_publish(kline_dir, before, snapshot_path)


def test_secondary_source_cas_change_after_staging_blocks_publish(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    _seal_fake_manifest(fake_sources, monkeypatch)
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    _write_local_klines(kline_dir)
    before = _kline_bytes(kline_dir)
    original_stage = repairs._stage_kline
    original_secondary = fake_sources.secondary_path.read_bytes()
    mutated = False

    def _stage_then_mutate_source(frame: pd.DataFrame, path: Path) -> Path:
        nonlocal mutated
        staged = original_stage(frame, path)
        if not mutated:
            fake_sources.secondary_path.write_bytes(original_secondary + b"\n")
            mutated = True
        return staged

    monkeypatch.setattr(repairs, "_stage_kline", _stage_then_mutate_source)
    with pytest.raises(RuntimeError, match="source archive/secondary"):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=_FakeBaostock(),
        )

    assert mutated
    _assert_zero_publish(kline_dir, before, snapshot_path)


def test_kline_cas_change_after_staging_blocks_publish(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    _seal_fake_manifest(fake_sources, monkeypatch)
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    _write_local_klines(kline_dir)
    before = _kline_bytes(kline_dir)
    victim_code = repairs.TARGET_CODES[-1]
    victim_path = kline_dir / f"{victim_code}.parquet"
    original_stage = repairs._stage_kline
    mutated = False

    def _stage_then_mutate_kline(frame: pd.DataFrame, path: Path) -> Path:
        nonlocal mutated
        staged = original_stage(frame, path)
        if not mutated:
            concurrent = pd.read_parquet(victim_path)
            concurrent.loc[0, "amount"] += 0.125
            concurrent.to_parquet(victim_path, index=False)
            mutated = True
        return staged

    monkeypatch.setattr(repairs, "_stage_kline", _stage_then_mutate_kline)
    with pytest.raises(RuntimeError, match="history K/snapshot"):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=_FakeBaostock(),
        )

    assert mutated
    assert not snapshot_path.exists()
    assert victim_path.read_bytes() != before[victim_code]
    for code, expected in before.items():
        if code != victim_code:
            assert (kline_dir / f"{code}.parquet").read_bytes() == expected
    assert not list(tmp_path.rglob("*.tmp.parquet"))


def test_snapshot_and_all_klines_roll_back_if_final_publish_fails(
    tmp_path,
    monkeypatch,
    fake_sources,
):
    _seal_fake_manifest(fake_sources, monkeypatch)
    kline_dir = tmp_path / "kline"
    snapshot_path = tmp_path / "evidence" / "patches.parquet"
    _write_local_klines(kline_dir)
    snapshot_path.parent.mkdir(parents=True)
    previous_snapshot = b"preexisting-snapshot-bytes"
    snapshot_path.write_bytes(previous_snapshot)
    before = _kline_bytes(kline_dir)
    original_replace = Path.replace

    def _replace_then_fail_on_snapshot(self: Path, target: Path) -> Path:
        result = original_replace(self, target)
        if Path(target) == snapshot_path:
            raise OSError("injected final snapshot publish failure")
        return result

    monkeypatch.setattr(Path, "replace", _replace_then_fail_on_snapshot)
    with pytest.raises(OSError, match="injected final snapshot publish failure"):
        _ensure(
            kline_dir=kline_dir,
            snapshot_path=snapshot_path,
            sources=fake_sources,
            baostock=_FakeBaostock(),
            refresh_snapshot=True,
        )

    assert _kline_bytes(kline_dir) == before
    assert snapshot_path.read_bytes() == previous_snapshot
    assert not list(tmp_path.rglob("*.tmp.parquet"))
    assert not list(tmp_path.rglob("*.rollback"))


def test_same_code_predecessor_constant_is_read_only():
    assert repairs.SAME_CODE_PREDECESSOR_FIRST_EXECUTABLE["600018.SH"] == np.datetime64(
        "2000-07-19", "D"
    )
    with pytest.raises(TypeError):
        repairs.SAME_CODE_PREDECESSOR_FIRST_EXECUTABLE["600018.SH"] = np.datetime64(
            "2001-01-01", "D"
        )

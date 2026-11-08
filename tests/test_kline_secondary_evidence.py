from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

from data import kline_secondary_evidence as evidence


def _install_small_contract(monkeypatch):
    code = "600018.SH"
    dates = ("2020-01-02", "2020-01-03")
    monkeypatch.setattr(
        evidence,
        "TENCENT_EXPLICIT_DATES",
        MappingProxyType({code: dates}),
    )
    monkeypatch.setattr(
        evidence,
        "TENCENT_SOURCE_SYMBOLS",
        MappingProxyType({code: "sh600018"}),
    )
    monkeypatch.setattr(
        evidence, "TENCENT_DATE_SET_SUMMARIES", MappingProxyType({})
    )
    monkeypatch.setattr(
        evidence, "TENCENT_SUMMARY_SYMBOLS", MappingProxyType({})
    )
    monkeypatch.setattr(
        evidence, "BAIDU_EXPLICIT_DATES", MappingProxyType({})
    )
    monkeypatch.setattr(
        evidence, "BAIDU_SOURCE_SYMBOLS", MappingProxyType({})
    )
    monkeypatch.setattr(
        evidence, "LOCAL_PRECLOSE_CHAINS", MappingProxyType({})
    )
    monkeypatch.setattr(
        evidence, "REQUIRED_EXECUTABLE_DATES", MappingProxyType({})
    )
    records = []
    for index, day in enumerate(dates):
        row = [
            day,
            f"{10 + index:.3f}",
            f"{10.1 + index:.3f}",
            f"{10.2 + index:.3f}",
            f"{9.9 + index:.3f}",
            f"{1000 + index:.3f}",
        ]
        records.append(
            evidence._record(
                source=evidence.TENCENT_SOURCE,
                code=code,
                symbol="sh600018",
                day=day,
                reference_open=float(row[1]),
                source_hash=evidence._canonical_source_row_hash(row),
            )
        )
    frame = pd.DataFrame(records, columns=evidence.SNAPSHOT_COLUMNS)
    monkeypatch.setattr(
        evidence,
        "EXPECTED_CODE_DATE_SHA256",
        evidence._code_date_sha256(zip(frame["stock_code"], frame["date"])),
    )
    monkeypatch.setattr(
        evidence, "EXPECTED_SNAPSHOT_SHA256", evidence._manifest_sha256(frame)
    )
    return frame


def test_v3_contract_pins_all_sources_and_52_new_dates():
    assert evidence.SNAPSHOT_SCHEMA == "secondary-executable-dates-v3"
    assert evidence.EXPECTED_CODE_DATE_SHA256 == (
        "bb47992675decfdcc6a28fa6971c9ee2e41f9455cf3546e3cb189d4df19287b9"
    )
    assert evidence.EXPECTED_SNAPSHOT_SHA256 == (
        "9d55f5c3999724fd24ef8b36725017d404a585219b09580296d1bd6beb8df10e"
    )
    assert sum(map(len, evidence.CROSSCONFIRMED_MISSING_43.values())) == 43
    assert evidence._code_date_sha256(
        evidence._mapping_code_dates(evidence.CROSSCONFIRMED_MISSING_43)
    ) == evidence.CROSSCONFIRMED_MISSING_43_SHA256
    assert evidence.CROSSCONFIRMED_MISSING_43_SHA256 == (
        "88b39b7580c1bdd264510b76dc43d981890000a2f126ed23a945b0a29094eeec"
    )
    assert sum(map(len, evidence.BAIDU_RESTORED_6.values())) == 6
    assert len(evidence.LOCAL_PRECLOSE_CHAINS) == 3
    new_rows = {
        *evidence._mapping_code_dates(evidence.CROSSCONFIRMED_MISSING_43),
        *evidence._mapping_code_dates(evidence.BAIDU_RESTORED_6),
        *(
            (code, str(contract["date"]))
            for code, contract in evidence.LOCAL_PRECLOSE_CHAINS.items()
        ),
    }
    assert len(new_rows) == 52
    assert sum(map(len, evidence.HISTORY_REPAIR_49.values())) == 49
    assert {
        *evidence._mapping_code_dates(evidence.HISTORY_REPAIR_49)
    } == {
        *evidence._mapping_code_dates(evidence.CROSSCONFIRMED_MISSING_43),
        *evidence._mapping_code_dates(evidence.BAIDU_RESTORED_6),
    }

    assert sum(map(len, evidence.TENCENT_EXPLICIT_DATES.values())) == 95
    assert sum(map(len, evidence.BAIDU_EXPLICIT_DATES.values())) == 7
    assert dict(evidence.TENCENT_DATE_SET_SUMMARIES["600018.SH"]) == {
        "count": 1448,
        "first": "2000-07-19",
        "last": "2006-09-25",
        "sha256": (
            "84e618e5115678c41b208c6766b14bc9"
            "be0424131eb086d256fddac6b5a7941f"
        ),
    }
    assert 95 + 7 + 3 + 1448 == 1553


def test_multisource_contract_does_not_disguise_baidu_or_chain_rows():
    assert evidence._expected_source_contract(
        "001872.SZ", "2012-09-10"
    ) == (evidence.TENCENT_SOURCE, "sz000022", "")
    assert evidence._expected_source_contract(
        "001872.SZ", "1993-07-15"
    ) == (evidence.BAIDU_SOURCE, "001872", "")
    assert evidence._expected_source_contract(
        "000005.SZ", "1991-01-14"
    ) == (evidence.BAIDU_SOURCE, "000005", "")
    assert evidence._expected_source_contract(
        "600656.SH", "1996-07-29"
    ) == (evidence.CHAIN_SOURCE, "600656.SH", "sh.600656")


def test_public_history_repair_helper_returns_exact_49_with_provenance(
    monkeypatch, tmp_path
):
    records = []
    for code, day in evidence._mapping_code_dates(evidence.HISTORY_REPAIR_49):
        source, symbol, confirmation_symbol = evidence._expected_source_contract(
            code, day
        )
        records.append(
            evidence._record(
                source=source,
                code=code,
                symbol=symbol,
                confirmation_symbol=confirmation_symbol,
                confirmation_hash=("b" * 64 if source == evidence.CHAIN_SOURCE else ""),
                day=day,
                reference_open=1.0,
                source_hash="a" * 64,
            )
        )
    frame = pd.DataFrame(records, columns=evidence.SNAPSHOT_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    monkeypatch.setattr(
        evidence, "load_verified_daily_evidence", lambda _path: frame.copy()
    )

    selected = evidence.verified_history_repair_evidence(tmp_path / "unused")

    assert len(selected) == 49
    assert tuple(selected.columns) == evidence.SNAPSHOT_COLUMNS
    assert selected["source_row_sha256"].str.len().eq(64).all()

    monkeypatch.setattr(
        evidence,
        "load_verified_daily_evidence",
        lambda _path: frame.iloc[:-1].copy(),
    )
    with pytest.raises(ValueError, match="49-row contract"):
        evidence.verified_history_repair_evidence(tmp_path / "unused")


def test_loader_and_coverage_are_strictly_offline(monkeypatch, tmp_path):
    frame = _install_small_contract(monkeypatch)
    path = tmp_path / "secondary.parquet"
    frame.to_parquet(path, index=False)

    loaded = evidence.load_verified_daily_evidence(path)

    assert loaded.groupby("stock_code").size().to_dict() == {"600018.SH": 2}
    assert evidence.covered_local_dates(
        "600018.SH",
        {date(2020, 1, 2), date(2020, 1, 4)},
        path=path,
    ) == frozenset({date(2020, 1, 2)})
    assert evidence.covered_local_dates(
        "999999.SZ",
        {date(2020, 1, 2)},
        path=tmp_path / "must-not-be-read.parquet",
    ) == frozenset()


def test_loader_rejects_v2_missing_extra_and_wrong_source(monkeypatch, tmp_path):
    frame = _install_small_contract(monkeypatch)

    v2 = frame.copy()
    v2["schema_version"] = "secondary-executable-dates-v2"
    path = tmp_path / "v2.parquet"
    v2.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="schema_version"):
        evidence.load_verified_daily_evidence(path)

    missing = frame.iloc[:1].copy()
    path = tmp_path / "missing.parquet"
    missing.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="精确日期集合"):
        evidence.load_verified_daily_evidence(path)

    extra = frame.copy()
    row = extra.iloc[-1].copy()
    row["date"] = "2020-01-04"
    row["payload_sha256"] = evidence._payload_sha256(row)
    extra = pd.concat([extra, row.to_frame().T], ignore_index=True)
    path = tmp_path / "extra.parquet"
    extra.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="精确日期集合"):
        evidence.load_verified_daily_evidence(path)

    wrong_source = frame.copy()
    wrong_source.loc[0, "source"] = evidence.BAIDU_SOURCE
    wrong_source.loc[0, "source_symbol"] = "600018"
    wrong_source.loc[0, "payload_sha256"] = evidence._payload_sha256(
        wrong_source.loc[0]
    )
    path = tmp_path / "wrong-source.parquet"
    wrong_source.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="source/source_symbol"):
        evidence.load_verified_daily_evidence(path)


def test_loader_rejects_raw_hash_and_self_rehashed_payload_tampering(
    monkeypatch, tmp_path
):
    frame = _install_small_contract(monkeypatch)
    tampered = frame.copy()
    tampered.loc[0, "source_row_sha256"] = "a" * 64
    tampered.loc[0, "reference_open"] += 0.01
    path = tmp_path / "tampered.parquet"
    tampered.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="payload_sha256"):
        evidence.load_verified_daily_evidence(path)

    tampered.loc[0, "payload_sha256"] = evidence._payload_sha256(
        tampered.loc[0]
    )
    path = tmp_path / "self-rehashed.parquet"
    tampered.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="manifest SHA-256"):
        evidence.load_verified_daily_evidence(path)


def test_tencent_downloader_requires_none_day_and_exact_symbol(monkeypatch):
    captured = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "data": {
                        "sh600018": {
                            "day": [
                                [
                                    "2000-07-19", "20.000", "20.000",
                                    "20.880", "19.650", "469581.000",
                                ]
                            ]
                        }
                    }
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(evidence, "urlopen", fake_urlopen)
    rows = evidence._download_tencent_rows(
        "sh600018", "2000-07-19", "2000-07-19", timeout=7.0
    )

    assert rows[0][0] == "2000-07-19"
    assert captured["url"].startswith(
        "https://ifzq.gtimg.cn/appstock/app/fqkline/get?"
    )
    assert ",day,2000-07-19,2000-07-19,640,none" in captured["url"]
    assert captured["timeout"] == 7.0


@pytest.mark.parametrize("column", range(1, 6))
@pytest.mark.parametrize("bad_value", ["0", "-1", "nan"])
def test_tencent_contract_rejects_nonpositive_or_nonfinite_ohlcv(
    monkeypatch, column, bad_value
):
    code, day = "600018.SH", "2000-07-19"
    monkeypatch.setattr(
        evidence, "TENCENT_EXPLICIT_DATES",
        MappingProxyType({code: (day,)}),
    )
    monkeypatch.setattr(
        evidence, "TENCENT_SOURCE_SYMBOLS",
        MappingProxyType({code: "sh600018"}),
    )
    monkeypatch.setattr(
        evidence, "TENCENT_DATE_SET_SUMMARIES", MappingProxyType({}),
    )
    monkeypatch.setattr(evidence, "KNOWN_ABSENT_DATES", MappingProxyType({}))
    row = [day, "20", "20", "20.88", "19.65", "469581"]
    row[column] = bad_value
    monkeypatch.setattr(
        evidence, "_download_tencent_rows",
        lambda *_args, **_kwargs: [row],
    )

    with pytest.raises(ValueError, match="非正 OHLCV 行"):
        evidence._download_tencent_contract(timeout=1.0)


def _install_tencent_repair_row(monkeypatch, raw_row):
    code, day = "300029.SZ", "2026-07-09"
    sealed = pd.DataFrame(
        [
            {
                "stock_code": code,
                "date": day,
                "source": evidence.TENCENT_SOURCE,
                "source_row_sha256": evidence._canonical_source_row_hash(
                    raw_row
                ),
            }
        ]
    )
    monkeypatch.setattr(
        evidence, "verified_history_repair_evidence",
        lambda _path: sealed.copy(),
    )
    return code, day, sealed


def test_verified_tencent_history_repair_returns_hash_bound_raw_ohlcv(
    monkeypatch
):
    raw = ["2026-07-09", "18.00", "18.50", "18.88", "17.80", "123456"]
    code, day, _sealed = _install_tencent_repair_row(monkeypatch, raw)
    monkeypatch.setattr(
        evidence, "_download_tencent_rows",
        lambda *_args, **_kwargs: [raw.copy()],
    )

    result = evidence.verified_tencent_history_repair_ohlcv(
        code, day, evidence_path=Path("sealed-v3"), timeout=1.0
    )

    assert result == {
        "date": day,
        "open": "18.00",
        "close": "18.50",
        "high": "18.88",
        "low": "17.80",
        "volume": "123456",
        "source_symbol": "sz300029",
        "source_row_sha256": evidence._canonical_source_row_hash(raw),
    }


@pytest.mark.parametrize(
    ("downloaded", "message"),
    [
        ([], "下载日期不精确"),
        (["wrong-date"], "下载日期不精确"),
        (["zero-volume"], "非正 OHLCV 行"),
        (["tampered"], "hash 与 v3 封存不一致"),
    ],
)
def test_verified_tencent_history_repair_rejects_invalid_live_row(
    monkeypatch, downloaded, message
):
    original = [
        "2026-07-09", "18.00", "18.50", "18.88", "17.80", "123456"
    ]
    code, day, _sealed = _install_tencent_repair_row(monkeypatch, original)
    cases = {
        "wrong-date": [
            "2026-07-08", "18.00", "18.50", "18.88", "17.80", "123456"
        ],
        "zero-volume": [
            "2026-07-09", "18.00", "18.50", "18.88", "17.80", "0"
        ],
        "tampered": [
            "2026-07-09", "18.01", "18.50", "18.88", "17.80", "123456"
        ],
    }
    rows = [] if not downloaded else [cases[downloaded[0]]]
    monkeypatch.setattr(
        evidence, "_download_tencent_rows",
        lambda *_args, **_kwargs: rows,
    )

    with pytest.raises(ValueError, match=message):
        evidence.verified_tencent_history_repair_ohlcv(
            code, day, evidence_path=Path("sealed-v3"), timeout=1.0
        )


@pytest.mark.parametrize(
    ("code", "day", "message"),
    [
        ("300029.SZ", "2026-07-08", "不在固定 49-row contract"),
        ("001872.SZ", "1993-07-15", "不属于 Tencent source"),
    ],
)
def test_verified_tencent_history_repair_rejects_out_of_contract_request(
    code, day, message
):
    with pytest.raises(ValueError, match=message):
        evidence.verified_tencent_history_repair_ohlcv(code, day)


def test_baidu_downloader_hashes_only_stable_full_ohlcva_core(monkeypatch):
    core = [
        "742665600", "1993-07-15", "-8.44", "-8.64", "125100",
        "-8.44", "-8.64", "1257195.00", "-0.20", "+2.37", "0.31",
        "-8.44",
    ]
    payload = {
        "ResultCode": "0",
        "Result": {
            "newMarketData": {
                "keys": [*evidence.BAIDU_CORE_KEYS, "ma5avgprice"],
                "marketData": ",".join([*core, "-8.41"]),
            }
        },
    }

    class _Response:
        status_code = 200

        def json(self):
            return payload

    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr(evidence.requests, "get", fake_get)
    rows = evidence._download_baidu_rows(
        "001872", ("1993-07-15",), timeout=8.0
    )
    first_hash = evidence._canonical_source_row_hash(
        rows["1993-07-15"][: len(evidence.BAIDU_CORE_KEYS)]
    )
    payload["Result"]["newMarketData"]["marketData"] = ",".join(
        [*core, "999.99"]
    )
    rows = evidence._download_baidu_rows(
        "001872", ("1993-07-15",), timeout=8.0
    )
    second_hash = evidence._canonical_source_row_hash(
        rows["1993-07-15"][: len(evidence.BAIDU_CORE_KEYS)]
    )

    assert first_hash == second_hash
    assert captured["params"]["code"] == "001872"
    assert captured["params"]["all"] == "1"
    assert captured["timeout"] == 8.0


def test_baidu_negative_adjusted_open_is_existence_evidence_only(monkeypatch):
    monkeypatch.setattr(
        evidence,
        "BAIDU_EXPLICIT_DATES",
        MappingProxyType({"001872.SZ": ("1993-07-15",)}),
    )
    monkeypatch.setattr(
        evidence,
        "BAIDU_SOURCE_SYMBOLS",
        MappingProxyType({"001872.SZ": "001872"}),
    )
    row = [
        "742665600", "1993-07-15", "-8.44", "-8.64", "125100",
        "-8.44", "-8.64", "1257195.00", "-0.20", "+2.37", "0.31",
        "-8.44",
    ]
    monkeypatch.setattr(
        evidence,
        "_download_baidu_rows",
        lambda *_args, **_kwargs: {"1993-07-15": row},
    )

    records = evidence._download_baidu_contract(timeout=1.0)

    assert len(records) == 1
    assert records[0]["tradestatus"] == "1"
    assert records[0]["reference_open"] == -8.44
    assert records[0]["source"] == evidence.BAIDU_SOURCE


@pytest.mark.parametrize(
    ("volume", "amount"),
    [("0", "1257195.00"), ("-1", "1257195.00"),
     ("125100", "0"), ("125100", "-1")],
)
def test_baidu_batch_rejects_nonpositive_volume_or_amount(
    monkeypatch, volume, amount
):
    monkeypatch.setattr(
        evidence,
        "BAIDU_EXPLICIT_DATES",
        MappingProxyType({"001872.SZ": ("1993-07-15",)}),
    )
    monkeypatch.setattr(
        evidence,
        "BAIDU_SOURCE_SYMBOLS",
        MappingProxyType({"001872.SZ": "001872"}),
    )
    row = [
        "742665600", "1993-07-15", "-8.44", "-8.64", volume,
        "-8.44", "-8.64", amount, "-0.20", "+2.37", "0.31", "-8.44",
    ]
    monkeypatch.setattr(
        evidence,
        "_download_baidu_rows",
        lambda *_args, **_kwargs: {"1993-07-15": row},
    )
    with pytest.raises(ValueError, match="非正成交行"):
        evidence._download_baidu_contract(timeout=1.0)


def test_chain_seals_local_and_next_preclose_core(monkeypatch, tmp_path):
    code = "600656.SH"
    monkeypatch.setattr(
        evidence,
        "LOCAL_PRECLOSE_CHAINS",
        MappingProxyType(
            {
                code: MappingProxyType(
                    {"date": "1996-07-29", "next_date": "1996-07-30"}
                )
            }
        ),
    )
    kline_dir = tmp_path / "k"
    kline_dir.mkdir()
    pd.DataFrame(
        [
            {
                "time": int(pd.Timestamp("1996-07-29").timestamp() * 1000),
                "open": 5.50,
                "high": 5.58,
                "low": 5.23,
                "close": 5.58,
                "volume": 3_376_800.0,
                "amount": 18_381_048.0,
                "preClose": 5.97,
            }
        ]
    ).to_parquet(kline_dir / f"{code}.parquet", index=False)
    import data.kline_mootdx as primary_module

    local_target = pd.read_parquet(kline_dir / f"{code}.parquet").iloc[0]
    target_state_hash = primary_module._kline_row_state_sha256(
        pd.Timestamp("1996-07-29").date(), local_target
    )
    primary = pd.DataFrame(
        [
            {
                "stock_code": code,
                "baostock_code": "sh.600656",
                "date": "1996-07-29",
                "source_tradestatus": "1",
                "tradestatus": "1",
                "status_correction": "",
                "reference_open": 5.50,
                "reference_volume": 0.0,
                "reference_amount": 0.0,
                "direct_preclose": 5.97,
                "normalization_applied": True,
                "applied_k_sha256": target_state_hash,
            },
            {
                "stock_code": code,
                "baostock_code": "sh.600656",
                "date": "1996-07-30",
                "source_tradestatus": "1",
                "tradestatus": "1",
                "status_correction": "",
                "reference_open": 5.61,
                "reference_volume": 2_727_854.0,
                "reference_amount": 15_280_204.0,
                "direct_preclose": 5.58,
                "normalization_applied": False,
                "applied_k_sha256": primary_module.UNAPPLIED_K_SHA256,
            },
        ]
    )
    monkeypatch.setattr(
        primary_module,
        "load_baostock_daily_evidence",
        lambda _path: primary.copy(),
    )
    rows = evidence._download_chain_contract(
        kline_dir=kline_dir, primary_evidence_path=tmp_path / "primary.parquet"
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == evidence.CHAIN_SOURCE
    assert row["source_symbol"] == code
    assert row["confirmation_symbol"] == "sh.600656"
    assert len(row["source_row_sha256"]) == 64
    assert len(row["confirmation_row_sha256"]) == 64

    primary.loc[1, "direct_preclose"] = 5.57
    with pytest.raises(ValueError, match="chain 断裂"):
        evidence._download_chain_contract(
            kline_dir=kline_dir,
            primary_evidence_path=tmp_path / "primary.parquet",
        )


@pytest.mark.parametrize(
    ("row_index", "field", "value", "message"),
    [
        (0, "normalization_applied", False, "normalized candidate"),
        (1, "normalization_applied", True, "next row"),
        (0, "source_tradestatus", "0", "target 非原始"),
        (0, "tradestatus", "0", "target 非原始"),
        (1, "source_tradestatus", "0", "next row"),
        (1, "tradestatus", "0", "next row"),
        (0, "status_correction", "fixed", "target 非原始"),
        (1, "status_correction", "fixed", "next row"),
        (0, "applied_k_sha256", "0" * 64, "状态 hash 不一致"),
    ],
)
def test_chain_rejects_unbound_or_corrected_primary_rows(
    monkeypatch, row_index, field, value, message
):
    import data.kline_mootdx as primary_module

    code, day, next_day = "600656.SH", "1996-07-29", "1996-07-30"
    monkeypatch.setattr(
        evidence, "LOCAL_PRECLOSE_CHAINS",
        MappingProxyType(
            {code: MappingProxyType({"date": day, "next_date": next_day})}
        ),
    )
    target = pd.Series(
        {
            "time": int(pd.Timestamp(day).timestamp() * 1000),
            "open": 5.50, "high": 5.58, "low": 5.23, "close": 5.58,
            "volume": 3_376_800.0, "amount": 18_381_048.0,
            "preClose": 5.97, "date": day,
        }
    )
    local = pd.DataFrame([target]).set_index("date", drop=False)
    monkeypatch.setattr(evidence, "_local_day_frame", lambda _path: local)
    target_hash = primary_module._kline_row_state_sha256(
        pd.Timestamp(day).date(), target
    )
    primary = pd.DataFrame(
        [
            {
                "stock_code": code, "baostock_code": "sh.600656",
                "date": day, "source_tradestatus": "1", "tradestatus": "1",
                "status_correction": "", "reference_open": 5.50,
                "reference_volume": 0.0, "reference_amount": 0.0,
                "direct_preclose": 5.97, "normalization_applied": True,
                "applied_k_sha256": target_hash,
            },
            {
                "stock_code": code, "baostock_code": "sh.600656",
                "date": next_day, "source_tradestatus": "1",
                "tradestatus": "1", "status_correction": "",
                "reference_open": 5.61, "reference_volume": 2_727_854.0,
                "reference_amount": 15_280_204.0, "direct_preclose": 5.58,
                "normalization_applied": False,
                "applied_k_sha256": primary_module.UNAPPLIED_K_SHA256,
            },
        ]
    )
    primary.loc[row_index, field] = value
    monkeypatch.setattr(
        primary_module, "load_baostock_daily_evidence",
        lambda _path: primary.copy(),
    )

    with pytest.raises(ValueError, match=message):
        evidence._download_chain_contract(
            kline_dir=Path("unused"),
            primary_evidence_path=Path("unused-primary"),
        )


def test_batch_failure_publishes_nothing(monkeypatch, tmp_path):
    path = tmp_path / "secondary.parquet"
    sentinel = b"old-v2-snapshot"
    path.write_bytes(sentinel)
    calls = []
    monkeypatch.setattr(
        evidence,
        "_download_tencent_contract",
        lambda **_kwargs: calls.append("tencent") or [],
    )

    def fail_baidu(**_kwargs):
        calls.append("baidu")
        raise RuntimeError("source failed")

    monkeypatch.setattr(evidence, "_download_baidu_contract", fail_baidu)
    monkeypatch.setattr(
        evidence,
        "_download_chain_contract",
        lambda **_kwargs: calls.append("chain") or [],
    )

    with pytest.raises(RuntimeError, match="source failed"):
        evidence.refresh_verified_daily_evidence(path, timeout=1.0)

    assert path.read_bytes() == sentinel
    assert calls == ["tencent", "baidu"]
    assert not list(tmp_path.glob(".*.tmp.parquet"))


@pytest.mark.parametrize(
    ("failed_batch", "expected_calls"),
    [("tencent", ["tencent"]),
     ("chain", ["tencent", "baidu", "chain"])],
)
def test_tencent_or_chain_failure_publishes_nothing(
    monkeypatch, tmp_path, failed_batch, expected_calls
):
    path = tmp_path / "secondary.parquet"
    sentinel = b"previous-valid-snapshot"
    path.write_bytes(sentinel)
    calls = []

    def batch(name):
        def run(**_kwargs):
            calls.append(name)
            if name == failed_batch:
                raise RuntimeError(f"{name} failed")
            return []
        return run

    monkeypatch.setattr(
        evidence, "_download_tencent_contract", batch("tencent")
    )
    monkeypatch.setattr(evidence, "_download_baidu_contract", batch("baidu"))
    monkeypatch.setattr(evidence, "_download_chain_contract", batch("chain"))

    with pytest.raises(RuntimeError, match=f"{failed_batch} failed"):
        evidence.refresh_verified_daily_evidence(path, timeout=1.0)

    assert path.read_bytes() == sentinel
    assert calls == expected_calls
    assert not list(tmp_path.glob(".*.tmp.parquet"))


def test_atomic_replace_failure_preserves_previous_snapshot(
    monkeypatch, tmp_path
):
    frame = _install_small_contract(monkeypatch)
    path = tmp_path / "secondary.parquet"
    sentinel = b"previous-snapshot"
    path.write_bytes(sentinel)
    monkeypatch.setattr(evidence, "_download_snapshot", lambda **_kwargs: frame)

    original_replace = Path.replace

    def fail_replace(self, target):
        if Path(target) == path:
            raise OSError("replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        evidence.refresh_verified_daily_evidence(path)

    assert path.read_bytes() == sentinel
    assert not list(tmp_path.glob(".*.tmp.parquet"))

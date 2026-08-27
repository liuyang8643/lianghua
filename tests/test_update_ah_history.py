import io
import json
from pathlib import Path

import pandas as pd
import pytest

from data import update_ah_history as ah


class _Response:
    def __init__(self, *, payload=None, text=None, content=None):
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})
        self.content = content if content is not None else self.text.encode()

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append((url, params, headers, timeout))
        return self.responder(url, params)


def _pairs():
    return ah.normalize_ah_pairs(
        [{"f191": "600000", "f12": "00005", "f193": "示例"}],
        snapshot_date="2026-08-26",
    )


def _hk_prices():
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "h_code": ["00005.HK", "00005.HK"],
            "raw_open": [10.0, 10.2],
            "raw_close": [10.1, 10.3],
            "raw_high": [10.2, 10.4],
            "raw_low": [9.9, 10.1],
            "qfq_open": [10.0, 10.2],
            "qfq_close": [10.1, 10.3],
            "qfq_high": [10.2, 10.4],
            "qfq_low": [9.9, 10.1],
            "hfq_open": [10.0, 10.2],
            "hfq_close": [10.1, 10.3],
            "volume": [1000.0, 1100.0],
        }
    )
    return ah.attach_h_preclose(source)


def _sse_fx():
    return ah.normalize_fx_rows(
        [{"validDate": "2024-01-02", "buyPrice": "0.91", "sellPrice": "0.93"}],
        exchange="SH",
    )


def test_hk_validation_allows_fully_missing_optional_hfq_rows():
    prices = _hk_prices()
    prices.loc[0, ["hfq_open", "hfq_close"]] = float("nan")
    ah._validate_hk_prices(prices)

    partial = prices.copy()
    partial.loc[0, "hfq_open"] = 10.0
    with pytest.raises(RuntimeError, match="either complete or absent"):
        ah._validate_hk_prices(partial)


def test_code_normalization_and_current_cohort_leakage_labels():
    assert ah.canonical_a_code("600000") == "600000.SH"
    assert ah.canonical_a_code("000001.SZ") == "000001.SZ"
    assert ah.canonical_h_code(5) == "00005.HK"
    with pytest.raises(ValueError, match="mismatched"):
        ah.canonical_a_code("600000.SZ")

    frame = ah.normalize_ah_pairs(
        [
            {
                "f191": "600000",
                "f12": "5",
                "f193": "浦发示例",
                "f188": 99,
                "f189": 101,
            },
            {
                "f191": "000001",
                "f12": "00001",
                "f193": "平安示例",
            },
        ],
        snapshot_date="2026-08-26",
    )
    assert frame["a_code"].tolist() == ["000001.SZ", "600000.SH"]
    assert frame["h_code"].tolist() == ["00001.HK", "00005.HK"]
    assert "valid_from" not in frame.columns
    assert set(frame["universe_cohort"]) == {"current_live_cohort"}
    assert not frame["point_in_time_complete"].any()
    assert frame["survivorship_bias"].all()
    assert set(frame["share_unit_assumption"]) == {
        "ordinary_share_1_to_1_unverified"
    }
    assert "f188" not in frame.columns
    assert "f189" not in frame.columns


def test_eastmoney_serial_pagination_uses_reported_total():
    rows = [
        {"f191": "600000", "f12": "00005", "f193": "一"},
        {"f191": "000001", "f12": "00001", "f193": "二"},
        {"f191": "601000", "f12": "00002", "f193": "三"},
    ]

    def responder(url, params):
        page = int(params["pn"])
        assert url == ah.EASTMONEY_AH_URL
        return _Response(
            payload={"data": {"total": 201, "diff": {"0": rows[page - 1]}}}
        )

    session = _Session(responder)
    frame, pages = ah.download_current_ah_pairs(
        session, snapshot_date="2026-08-26", delay_seconds=0
    )
    assert pages == 3
    assert len(frame) == 3
    assert [int(call[1]["pn"]) for call in session.calls] == [1, 2, 3]
    assert all(call[1]["fs"] == "b:DLMK0101" for call in session.calls)


def test_tencent_candidate_and_hkex_official_mapping_parsers():
    tencent = (
        'list_data={"data":{"page_count":2,"page_data":['
        '["00939~建设银行~1~2"],"00042~东北电气~1~2"]}};'
    )
    rows, pages = ah.parse_tencent_ah_candidates(tencent)
    assert pages == 2
    assert rows == [
        {"h_code": "00939.HK", "name": "建设银行"},
        {"h_code": "00042.HK", "name": "东北电气"},
    ]

    token = "evLtsLsBNAUVTPxtGqVeG7AykO6IsC8EuxyabulHa5gmMeOeny5%2bzmZ"
    html = f'LabCI.getToken = function() {{ return "{token}"; }};'
    assert ah.extract_hkex_token(html) == token
    widget = (
        'jQuery351({"data":{"responsecode":"000","quote":'
        '{"ric":"0939.HK","nm":"CCB H Shares",'
        '"underlying_ric":"601939.SS"}}})'
    )
    assert ah.parse_hkex_underlying_quote(widget) == {
        "a_code": "601939.SH",
        "h_code": "00939.HK",
        "name": "CCB H Shares",
    }


def test_hkex_parser_does_not_invent_mapping_when_underlying_is_absent():
    widget = (
        'cb({"data":{"responsecode":"000","quote":'
        '{"ric":"0941.HK","nm":"China Mobile",'
        '"underlying_ric":""}}})'
    )
    assert ah.parse_hkex_underlying_quote(widget) is None


def test_open_source_registry_keeps_only_active_rows_and_discloses_provenance():
    rows = [
        "hk_code,a_code,name,status,is_red_chip,is_restricted,source,first_seen"
    ]
    for value in range(1, 102):
        rows.append(
            f"{value:05d},60{value:04d},示例{value},active,false,false,hkex,2026-04-16"
        )
    rows.append("00999,600999,退市示例,delisted,false,false,hkex,2026-04-16")
    frame = ah.normalize_github_ah_pairs(
        ("\n".join(rows) + "\n").encode(),
        snapshot_date="2026-08-26",
    )
    assert len(frame) == 101
    assert "00999.HK" not in set(frame["h_code"])
    assert frame.attrs["mapping_source"] == "open_source_current_registry"
    assert frame.attrs["registry_source_breakdown"] == {"hkex": 101}
    assert len(frame.attrs["registry_sha256"]) == 64


def test_tencent_jsonp_parses_day_and_qfqday_then_outer_aligns():
    raw = (
        'kline_day={"data":{"hk00005":{"day":'
        '[["2024-01-02","10","10.2","10.3","9.9","1000"],'
        '["2024-01-03","10.2","10.4","10.5","10.1","1100"],'
        '["2024-01-03","10.3","10.5","10.6","10.2","1200"]]}}};'
    )
    qfq = {
        "data": {
            "hk00005": {
                "qfqday": [
                    ["2024-01-03", "9.8", "10.0", "10.1", "9.7", "1100", "x"],
                    ["2024-01-04", "10.0", "10.2", "10.3", "9.9", "900", "x"],
                ]
            }
        }
    }
    raw_frame = ah.parse_tencent_hk_kline(raw, h_code="00005.HK", adjusted=False)
    qfq_frame = ah.parse_tencent_hk_kline(qfq, h_code="5", adjusted=True)
    aligned = ah.align_hk_price_frames(
        raw_frame,
        qfq_frame,
        qfq_frame,
        h_code="00005",
    )

    assert raw_frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert raw_frame["open"].tolist() == [10.0, 10.3]
    assert aligned["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert pd.isna(aligned.loc[0, "qfq_open"])
    assert pd.isna(aligned.loc[2, "raw_open"])
    assert aligned.loc[1, "volume"] == 1200


def test_tencent_empty_history_and_negative_qfq_synthetic_row_are_auditable():
    empty = ah.parse_tencent_hk_kline(
        {"code": 0, "msg": "", "data": []}, h_code="00005", adjusted=False
    )
    assert empty.empty
    qfq = ah.parse_tencent_hk_kline(
        {
            "code": 0,
            "data": {
                "hk00005": {
                    "qfqday": [
                        ["2000-01-03", "-1", "-0.5", "1", "-1", "0", {}]
                    ]
                }
            },
        },
        h_code="00005",
        adjusted=True,
    )
    assert qfq.loc[0, "open"] == -1
    hfq = ah.parse_tencent_hk_kline(
        {
            "code": 0,
            "data": {
                "hk00005": {
                    "hfqday": [
                        ["2024-01-03", "10", "10.2", "10.3", "9.9", "1000"]
                    ]
                }
            },
        },
        h_code="00005",
        adjusted=True,
        adjustment="hfq",
    )
    assert hfq.loc[0, "close"] == pytest.approx(10.2)
    with pytest.raises(RuntimeError, match="non-positive"):
        ah.parse_tencent_hk_kline(
            {
                "code": 0,
                "data": {
                    "hk00005": {
                        "hfqday": [
                            ["2024-01-03", "-1", "1", "1", "-1", "0"]
                        ]
                    }
                },
            },
            h_code="00005",
            adjusted=True,
            adjustment="hfq",
        )
    with pytest.raises(RuntimeError, match="non-positive"):
        ah.parse_tencent_hk_kline(
            {
                "code": 0,
                "data": {
                    "hk00005": {
                        "day": [["2000-01-03", "-1", "-0.5", "1", "-1", "0"]]
                    }
                },
            },
            h_code="00005",
            adjusted=False,
        )


def _corporate_action_source(
    previous_raw,
    current_raw,
    *,
    previous_scale,
    previous_intercept,
    current_scale,
    current_intercept,
):
    rows = []
    for date, raw, scale, intercept in (
        ("2025-06-02", previous_raw, previous_scale, previous_intercept),
        ("2025-06-03", current_raw, current_scale, current_intercept),
    ):
        raw_open, raw_close, raw_high, raw_low = raw
        rows.append(
            {
                "date": pd.Timestamp(date),
                "h_code": "00300.HK",
                "raw_open": raw_open,
                "raw_close": raw_close,
                "raw_high": raw_high,
                "raw_low": raw_low,
                "qfq_open": scale * raw_open + intercept,
                "qfq_close": scale * raw_close + intercept,
                "qfq_high": scale * raw_high + intercept,
                "qfq_low": scale * raw_low + intercept,
                "hfq_open": raw_open,
                "hfq_close": raw_close,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows, columns=ah.HK_SOURCE_COLUMNS)


def test_qfq_affine_preclose_reconstructs_midea_cash_dividend():
    source = _corporate_action_source(
        (81.85, 82.85, 83.00, 81.00),
        (79.50, 77.60, 80.00, 77.00),
        previous_scale=1.0,
        previous_intercept=-8.729,
        current_scale=1.0,
        current_intercept=-4.914,
    )
    result = ah.attach_h_preclose(source)
    assert result["qfq_affine_valid"].all()
    assert result.loc[1, "h_pre_close"] == pytest.approx(79.035, abs=0.002)
    assert result.loc[1, "raw_close"] / result.loc[1, "h_pre_close"] - 1 == (
        pytest.approx(-0.01816, abs=0.0001)
    )


def test_qfq_affine_preclose_handles_byd_split_and_cash_together():
    source = _corporate_action_source(
        (406.8, 396.6, 410.0, 395.0),
        (132.4, 135.6, 137.0, 130.0),
        previous_scale=1.0 / 3.0,
        previous_intercept=-1.857,
        current_scale=1.0,
        current_intercept=-0.411,
    )
    result = ah.attach_h_preclose(source)
    assert result.loc[1, "h_pre_close"] == pytest.approx(130.754, abs=0.002)
    assert result.loc[1, "raw_close"] / result.loc[1, "h_pre_close"] - 1 == (
        pytest.approx(0.0370, abs=0.0002)
    )


def test_preclose_is_invariant_to_future_affine_qfq_rebasing_and_negative_levels():
    source = _corporate_action_source(
        (81.85, 82.85, 83.00, 81.00),
        (79.50, 77.60, 80.00, 77.00),
        previous_scale=1.0,
        previous_intercept=-108.729,
        current_scale=1.0,
        current_intercept=-104.914,
    )
    base = ah.attach_h_preclose(source)
    rebased = source.copy()
    for column in ("qfq_open", "qfq_close", "qfq_high", "qfq_low"):
        rebased[column] = 2.5 * rebased[column] + 37.0
    changed = ah.attach_h_preclose(rebased)
    assert (base["qfq_open"] < 0).all()
    assert base.loc[1, "h_pre_close"] == pytest.approx(
        changed.loc[1, "h_pre_close"], abs=1e-9
    )


def test_non_affine_or_underdetermined_qfq_rows_fail_closed():
    source = _corporate_action_source(
        (10.0, 10.2, 10.3, 9.9),
        (10.2, 10.4, 10.5, 10.1),
        previous_scale=1.0,
        previous_intercept=0.0,
        current_scale=1.0,
        current_intercept=0.0,
    )
    source.loc[1, "qfq_high"] += 1.0
    result = ah.attach_h_preclose(source)
    assert not result.loc[1, "qfq_affine_valid"]
    assert pd.isna(result.loc[1, "h_pre_close"])

    flat = source.iloc[[0]].copy()
    for column in ("raw_open", "raw_close", "raw_high", "raw_low"):
        flat[column] = 10.0
    for column in ("qfq_open", "qfq_close", "qfq_high", "qfq_low"):
        flat[column] = 9.0
    flat_result = ah.attach_h_preclose(flat)
    assert not flat_result.loc[0, "qfq_affine_valid"]


def test_hfq_audit_flags_midea_provider_reset_not_raw_market_move():
    source = _corporate_action_source(
        (81.85, 82.85, 83.00, 81.00),
        (79.50, 77.60, 80.00, 77.00),
        previous_scale=1.0,
        previous_intercept=-8.729,
        current_scale=1.0,
        current_intercept=-4.914,
    )
    source["hfq_close"] = [180.576, 81.415]
    assert ah.hfq_discontinuity_mask(source).tolist() == [False, True]


def test_hk_checkpoint_resume_performs_no_request(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = (
        checkpoint_dir
        / "00005_20240101_20241231_raw_qfq_hfq_ohlc_v3.parquet"
    )
    _hk_prices().to_parquet(checkpoint, index=False)

    def no_network(url, params):
        raise AssertionError("checkpoint resume must not access the network")

    frame, resumed = ah.download_hk_histories(
        _Session(no_network),
        _pairs(),
        checkpoint_dir=checkpoint_dir,
        start_date="2024-01-01",
        end_date="2024-12-31",
        resume=True,
        delay_seconds=0,
    )
    assert resumed == 1
    assert len(frame) == 2


def test_sse_fx_paginates_to_pagehelp_pagecount_and_deduplicates():
    page_rows = {
        1: [
            {"validDate": "2024-01-02", "buyPrice": "0.91", "sellPrice": "0.93"},
            {"validDate": "2024-01-02", "buyPrice": "0.92", "sellPrice": "0.94"},
        ],
        2: [
            {"validDate": "2024-01-03", "buyPrice": "0.90", "sellPrice": "0.92"}
        ],
    }

    def responder(url, params):
        page = int(params["pageHelp.pageNo"])
        assert url == ah.SSE_FX_URL
        assert params["sqlId"] == "FW_HGT_JSHDBL"
        return _Response(
            payload={"pageHelp": {"pageCount": "2"}, "result": page_rows[page]}
        )

    session = _Session(responder)
    frame, pages = ah.download_sse_settlement_fx(
        session, end_date="2026-08-26", delay_seconds=0
    )
    assert pages == 2
    assert len(frame) == 2
    assert frame["exchange"].tolist() == ["SH", "SH"]
    assert frame["unit"].tolist() == ["CNY/HKD", "CNY/HKD"]
    assert frame.iloc[0]["buy_rate"] == pytest.approx(0.92)
    assert frame.iloc[0]["mid_rate"] == pytest.approx(0.93)
    assert [int(call[1]["pageHelp.pageNo"]) for call in session.calls] == [1, 2]


def test_fx_unit_guard_rejects_percentage_scaled_values():
    with pytest.raises(RuntimeError, match="plausible CNY/HKD"):
        ah.normalize_fx_rows(
            [{"validDate": "2024-01-02", "buyPrice": 91, "sellPrice": 93}],
            exchange="SH",
        )


def test_szse_workbook_schema_and_download_are_exchange_specific():
    source = pd.DataFrame(
        {
            "适用日期": ["2024-01-03", "2024-01-02"],
            "买入结算汇兑比率": [0.90, 0.91],
            "卖出结算汇兑比率": [0.92, 0.93],
            "货币种类": ["HKD", "HKD"],
        }
    )
    stream = io.BytesIO()
    with pd.ExcelWriter(stream, engine="openpyxl") as writer:
        source.to_excel(writer, index=False)
    workbook = stream.getvalue()

    def responder(url, params):
        assert url == ah.SZSE_FX_URL
        assert params["CATALOGID"] == "SGT_LSHL"
        assert params["TABKEY"] == "tab2"
        return _Response(content=workbook)

    frame, digest = ah.download_szse_settlement_fx(
        _Session(responder), delay_seconds=0
    )
    assert frame["exchange"].tolist() == ["SZ", "SZ"]
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert digest == ah._sha256_bytes(workbook)


def test_metadata_marks_survivorship_share_units_and_exchange_routing():
    pairs = _pairs()
    hk_prices = _hk_prices()
    fx = pd.concat(
        [
            _sse_fx(),
            ah.normalize_fx_rows(
                [
                    {
                        "validDate": "2024-01-02",
                        "buyPrice": 0.90,
                        "sellPrice": 0.92,
                    }
                ],
                exchange="SZ",
            ),
        ],
        ignore_index=True,
    )
    hashes = {
        ah.PAIR_FILENAME: "a" * 64,
        ah.HK_PRICE_FILENAME: "b" * 64,
        ah.FX_FILENAME: "c" * 64,
    }
    metadata = ah.build_metadata(
        pairs=pairs,
        hk_prices=hk_prices,
        fx=fx,
        hashes=hashes,
        snapshot_date="2026-08-26",
        requested_start_date="2014-11-17",
        requested_end_date="2026-08-26",
        eastmoney_pages=3,
        sse_pages=2,
        resumed_h_codes=1,
        szse_source_sha256="d" * 64,
        szse_limitation=None,
    )
    assert metadata["universe"] == {
        "cohort": "current_live_cohort",
        "point_in_time_complete": False,
        "survivorship_bias": True,
        "warning": metadata["universe"]["warning"],
    }
    assert metadata["share_unit_assumption"]["label"] == (
        "ordinary_share_1_to_1_unverified"
    )
    assert not metadata["share_unit_assumption"]["eastmoney_price_ratio_fields_used"]
    assert metadata["fx_contract"]["routing"] == {
        "SH": "SSE settlement rate",
        "SZ": "SZSE settlement rate",
    }
    assert metadata["fx_contract"]["available_exchanges"] == ["SH", "SZ"]


def test_szse_failure_is_never_silently_replaced_with_sse(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ah,
        "download_current_ah_pairs",
        lambda *args, **kwargs: (_pairs(), 1),
    )
    monkeypatch.setattr(
        ah,
        "download_hk_histories",
        lambda *args, **kwargs: (_hk_prices(), 0),
    )
    monkeypatch.setattr(
        ah,
        "download_sse_settlement_fx",
        lambda *args, **kwargs: (_sse_fx(), 1),
    )
    monkeypatch.setattr(
        ah,
        "download_szse_settlement_fx",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("SZSE down")),
    )

    with pytest.raises(RuntimeError, match="no SSE rate was silently substituted"):
        ah.update_ah_history(
            output_dir=tmp_path / "strict",
            start_date="2024-01-01",
            end_date="2024-12-31",
            session=object(),
        )

    metadata = ah.update_ah_history(
        output_dir=tmp_path / "limited",
        start_date="2024-01-01",
        end_date="2024-12-31",
        allow_sse_only=True,
        session=object(),
    )
    assert metadata["fx_contract"]["available_exchanges"] == ["SH"]
    assert "SZ pairs must be excluded" in metadata["fx_contract"]["limitation"]
    _, _, fx, loaded_metadata = ah.load_ah_history(tmp_path / "limited")
    assert set(fx["exchange"]) == {"SH"}
    assert loaded_metadata["fx_contract"]["limitation"] is not None


def test_loader_detects_artifact_hash_tampering(tmp_path):
    pairs = _pairs()
    hk_prices = _hk_prices()
    fx = _sse_fx()
    hashes = {
        ah.PAIR_FILENAME: ah._atomic_write_parquet(pairs, tmp_path / ah.PAIR_FILENAME),
        ah.HK_PRICE_FILENAME: ah._atomic_write_parquet(
            hk_prices, tmp_path / ah.HK_PRICE_FILENAME
        ),
        ah.FX_FILENAME: ah._atomic_write_parquet(fx, tmp_path / ah.FX_FILENAME),
    }
    metadata = ah.build_metadata(
        pairs=pairs,
        hk_prices=hk_prices,
        fx=fx,
        hashes=hashes,
        snapshot_date="2026-08-26",
        requested_start_date="2024-01-01",
        requested_end_date="2024-12-31",
        eastmoney_pages=1,
        sse_pages=1,
        resumed_h_codes=0,
        szse_source_sha256=None,
        szse_limitation="test-only SH artifact",
    )
    ah._atomic_write_json(metadata, tmp_path / ah.METADATA_FILENAME)
    (tmp_path / ah.HK_PRICE_FILENAME).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        ah.load_ah_history(tmp_path)

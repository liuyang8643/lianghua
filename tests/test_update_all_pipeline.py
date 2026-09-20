import sys
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest


def _index_snapshot(*days: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(list(days)),
            "open": [10.0 + index for index in range(len(days))],
            "close": [10.5 + index for index in range(len(days))],
        }
    )


def _valid_st_records():
    return [
        {
            "VARYDATE": "1991-01-01",
            "F006V": "新股上市",
            "F002V": "正常上市",
            "SECCODE": "000001",
        },
        {
            "VARYDATE": "2000-01-01",
            "F006V": "戴帽",
            "F002V": "特别处理",
            "SECCODE": "600001",
        },
    ]


def _valid_delist_snapshot():
    return pd.DataFrame([
        {
            "exchange": "SH",
            "公司代码": "600001",
            "公司简称": "上证退市",
            "上市日期": "1992-01-01",
            "暂停上市日期": "2024-01-02",
        },
        {
            "exchange": "SZ",
            "证券代码": "000001",
            "证券简称": "深证退市",
            "上市日期": "1991-01-01",
            "终止上市日期": "2024-01-03",
        },
    ])


def _valid_stock_list():
    return pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "600001.SH"],
            "exchange": ["SZ", "SH"],
        }
    )


def test_stock_list_snapshot_rejects_empty_or_bad_schema():
    from data.db.stock_list import validate_stock_list_frame

    with pytest.raises(ValueError, match="不得为空"):
        validate_stock_list_frame(
            pd.DataFrame(columns=["stock_code", "exchange"])
        )
    with pytest.raises(ValueError, match="缺少列"):
        validate_stock_list_frame(pd.DataFrame({"stock_code": ["000001.SZ"]}))
    broken = _valid_stock_list()
    broken.loc[0, "exchange"] = "SH"
    with pytest.raises(ValueError, match="后缀不一致"):
        validate_stock_list_frame(broken)


def test_index_snapshot_rejects_history_shrink_without_overwrite(tmp_path):
    import data.update_all as update_all

    output_path = tmp_path / "index_sh000001_daily.parquet"
    update_all._save_index_snapshot_atomic(
        _index_snapshot("2024-01-02", "2024-01-03"), output_path
    )
    original = output_path.read_bytes()

    with pytest.raises(ValueError, match="历史响应收缩"):
        update_all._save_index_snapshot_atomic(
            _index_snapshot("2024-01-03"), output_path
        )

    assert output_path.read_bytes() == original


def test_stock_list_atomic_save_rejects_loss_of_still_listed_code(
    tmp_path, monkeypatch,
):
    import data.db.delist as delist_db
    import data.update_all as update_all

    output_path = tmp_path / "stock_list" / "stock_list.parquet"
    monkeypatch.setattr(delist_db, "get_delist_stock_info", lambda: {})
    update_all._save_stock_list_snapshot_atomic(
        _valid_stock_list(), output_path, as_of=date(2024, 1, 2)
    )
    original = output_path.read_bytes()

    with pytest.raises(ValueError, match="丢失仍在市股票"):
        update_all._save_stock_list_snapshot_atomic(
            _valid_stock_list().iloc[:1],
            output_path,
            as_of=date(2024, 1, 3),
        )

    assert output_path.read_bytes() == original


def test_stock_list_atomic_save_allows_locally_confirmed_delist(
    tmp_path, monkeypatch,
):
    import data.db.delist as delist_db
    import data.update_all as update_all

    output_path = tmp_path / "stock_list" / "stock_list.parquet"
    monkeypatch.setattr(delist_db, "get_delist_stock_info", lambda: {})
    update_all._save_stock_list_snapshot_atomic(
        _valid_stock_list(), output_path, as_of=date(2024, 1, 2)
    )
    monkeypatch.setattr(
        delist_db,
        "get_delist_stock_info",
        lambda: {"600001.SH": SimpleNamespace(delist_date=date(2024, 1, 3))},
    )

    update_all._save_stock_list_snapshot_atomic(
        _valid_stock_list().iloc[:1],
        output_path,
        as_of=date(2024, 1, 3),
    )

    assert pd.read_parquet(output_path)["stock_code"].tolist() == ["000001.SZ"]


def test_delist_snapshot_requires_both_exchanges():
    import data.update_all as update_all

    with pytest.raises(ValueError, match="同时包含 SH/SZ"):
        update_all._validate_delist_snapshot(_valid_delist_snapshot().iloc[:1])


def test_st_snapshot_requires_complete_stock_coverage():
    import data.update_all as update_all

    with pytest.raises(ValueError, match="未覆盖本地股票全集"):
        update_all._parse_st_records(
            _valid_st_records(), {"000001", "600001", "688001"}
        )


def test_st_snapshot_rejects_conflicting_events_on_same_stock_date():
    from data.db.stock_name import validate_st_changes

    frame = pd.DataFrame(
        {
            "bare_code": ["600001", "600001"],
            "date": [date(2024, 1, 2), date(2024, 1, 2)],
            "event": ["戴帽", "摘帽"],
            "status": ["特别处理", "正常上市"],
        }
    )
    with pytest.raises(ValueError, match="同日事件必须唯一"):
        validate_st_changes(frame)


def test_st_snapshot_atomic_save_rejects_history_shrink(tmp_path):
    import data.update_all as update_all

    output_path = tmp_path / "stock_name" / "st_changes.parquet"
    previous = update_all._parse_st_records(
        _valid_st_records(), {"000001", "600001"}
    )
    update_all._save_st_snapshot_atomic(previous, output_path)
    previous_bytes = output_path.read_bytes()

    current = previous.iloc[:1].copy()
    with pytest.raises(ValueError, match="历史收缩"):
        update_all._save_st_snapshot_atomic(current, output_path)

    assert output_path.read_bytes() == previous_bytes

    replacement = previous.copy()
    replacement.loc[0, "event"] = "摘帽"
    with pytest.raises(ValueError, match="历史收缩"):
        update_all._save_st_snapshot_atomic(replacement, output_path)

    assert output_path.read_bytes() == previous_bytes


def test_build_st_mask_fails_closed_without_snapshot(tmp_path, monkeypatch):
    import data.build_runtime as build_runtime

    monkeypatch.setattr(build_runtime, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="拒绝生成 runtime"):
        build_runtime.build_st_mask(
            pd.Series(["000001.SZ"]).to_numpy(),
            pd.Series([date(2024, 1, 2)], dtype="datetime64[ns]").to_numpy(),
        )


def test_delist_snapshot_atomic_save_revalidates_and_invalidates_cache(
    tmp_path, monkeypatch,
):
    import data.db.delist as delist_db
    import data.update_all as update_all

    output_path = tmp_path / "delist" / "delist.parquet"
    monkeypatch.setattr(delist_db, "_DELIST_CACHE", {"stale": object()})

    update_all._save_delist_snapshot_atomic(
        _valid_delist_snapshot(), output_path,
    )

    assert set(pd.read_parquet(output_path)["exchange"]) == {"SH", "SZ"}
    assert delist_db._DELIST_CACHE is None
    assert not list(output_path.parent.glob("*.tmp.parquet"))


def test_delist_snapshot_atomic_save_rejects_partial_two_market_response(
    tmp_path,
):
    import data.update_all as update_all

    output_path = tmp_path / "delist" / "delist.parquet"
    base = _valid_delist_snapshot()
    extra = base.copy()
    extra.loc[extra["exchange"] == "SH", "公司代码"] = "600002"
    extra.loc[extra["exchange"] == "SZ", "证券代码"] = "000002"
    previous = pd.concat([base, extra], ignore_index=True)
    update_all._save_delist_snapshot_atomic(previous, output_path)
    previous_bytes = output_path.read_bytes()

    with pytest.raises(ValueError, match="丢失或改写历史记录"):
        update_all._save_delist_snapshot_atomic(base, output_path)

    assert output_path.read_bytes() == previous_bytes


def test_delist_update_fails_closed_without_overwriting_previous_snapshot(
    tmp_path, monkeypatch,
):
    import data.update_all as update_all

    output_path = tmp_path / "delist" / "delist.parquet"
    output_path.parent.mkdir(parents=True)
    previous = _valid_delist_snapshot()
    previous.to_parquet(output_path, index=False)
    previous_bytes = output_path.read_bytes()
    old_mtime = output_path.stat().st_mtime - 86_400
    output_path.touch()
    import os
    os.utime(output_path, (old_mtime, old_mtime))

    sh = previous[previous["exchange"] == "SH"].drop(columns="exchange")
    fake_akshare = SimpleNamespace(
        stock_info_sh_delist=lambda: sh,
        stock_info_sz_delist=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)
    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="SZ 请求失败"):
        update_all._update_delist()

    assert output_path.read_bytes() == previous_bytes

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

    def fake_update_full(codes, *, strict=False):
        assert strict is True
        calls.append(list(codes))
        (tmp_path / "k-line" / "000003.SZ.parquet").write_text("ok")

    monkeypatch.setattr(kline_mootdx, "update_full", fake_update_full)

    update_all._ensure_delist_kline_mootdx()

    assert calls == [["000003.SZ"]]


def test_delist_kline_missing_after_mootdx_blocks_incomplete_runtime(
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
    monkeypatch.setattr(kline_mootdx, "update_full", lambda codes, **_kwargs: None)

    with pytest.raises(
        update_all.DelistedKlineUnavailableError,
        match="拒绝继续生成不完整 runtime",
    ) as caught:
        update_all._ensure_delist_kline_mootdx()

    assert caught.value.diagnostics == (
        {
            "code": "000003.SZ",
            "backup_exists": False,
            "restore_failure": None,
            "live_failure": None,
        },
    )


def test_kline_update_restores_delisted_before_strict_current_pull(monkeypatch):
    import data.db.issue_price as issue_reference
    import data.db.stock_list as stock_list
    import data.kline_mootdx as kline_mootdx
    import data.update_all as update_all

    calls = []
    monkeypatch.setattr(
        update_all,
        "_ensure_delist_kline_mootdx",
        lambda: calls.append("restore"),
    )
    monkeypatch.setattr(
        stock_list,
        "load_current_stock_codes",
        lambda: ("000001.SZ",),
    )
    monkeypatch.setattr(
        issue_reference,
        "resolve_terminal_active_codes",
        lambda current, *_args, **_kwargs: tuple(current),
    )

    def fake_recent(days, **kwargs):
        calls.append(("recent", days, kwargs))

    monkeypatch.setattr(kline_mootdx, "update_recent", fake_recent)

    update_all._update_kline(anchor_date=date(2024, 1, 3))

    assert calls[0] == "restore"
    assert calls[1] == (
        "recent",
        update_all.REPULL_TRADING_DAYS,
        {
            "anchor_date": date(2024, 1, 3),
            "codes": ["000001.SZ"],
            "strict": True,
        },
    )


def _install_old_axis_missing_issue_reference(tmp_path, monkeypatch):
    import data.db.issue_price as issue_reference
    import data.db.stock_list as stock_list
    import data.update_all as update_all

    output = tmp_path / "issue_price" / "issue_price.parquet"
    manifest = output.with_name(f"{output.name}.manifest.json")
    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    monkeypatch.setattr(issue_reference, "ISSUE_REFERENCE_PATH", output)
    monkeypatch.setattr(
        issue_reference, "ISSUE_REFERENCE_MANIFEST_PATH", manifest
    )
    monkeypatch.setattr(
        stock_list, "load_current_stock_codes", lambda: ("301688.SZ",)
    )
    (tmp_path / "k-line").mkdir()
    issue_reference.save_issue_reference_atomic(
        pd.DataFrame(
            {
                "stock_code": ["301688"],
                "issue_price": [10.0],
                "list_date": [date(2026, 9, 3)],
                "source": ["old-source"],
                "source_as_of": [date(2026, 8, 30)],
            }
        ),
        path=output,
        manifest_path=manifest,
    )
    return issue_reference, update_all, output, manifest


def test_issue_update_refreshes_and_upserts_old_axis_missing_record(
    tmp_path, monkeypatch
):
    issue_reference, update_all, output, manifest = (
        _install_old_axis_missing_issue_reference(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(
        update_all,
        "_fetch_sina_issue_record",
        lambda bare: {
            "stock_code": bare,
            "issue_price": 12.34,
            "list_date": date(2026, 9, 1),
            "source": update_all.ISSUE_REFERENCE_SOURCE,
            "source_as_of": date(2026, 8, 31),
        },
    )

    update_all._update_issue_price()

    refreshed = issue_reference.load_issue_reference(
        path=output, manifest_path=manifest
    )
    assert refreshed["stock_code"].tolist() == ["301688"]
    assert refreshed["list_date"].tolist() == [date(2026, 9, 1)]
    assert refreshed["issue_price"].tolist() == [12.34]
    assert refreshed["source"].tolist() == [update_all.ISSUE_REFERENCE_SOURCE]


def test_issue_update_axis_missing_refresh_failure_publishes_nothing(
    tmp_path, monkeypatch
):
    _issue_reference, update_all, output, manifest = (
        _install_old_axis_missing_issue_reference(tmp_path, monkeypatch)
    )
    original_parquet = output.read_bytes()
    original_manifest = manifest.read_bytes()
    monkeypatch.setattr(
        update_all,
        "_fetch_sina_issue_record",
        lambda _bare: (_ for _ in ()).throw(RuntimeError("source offline")),
    )

    with pytest.raises(RuntimeError, match="刷新失败"):
        update_all._update_issue_price()

    assert output.read_bytes() == original_parquet
    assert manifest.read_bytes() == original_manifest


def test_runtime_builder_reconciles_placeholders_then_fetches_conflict_evidence_once(
    tmp_path, monkeypatch,
):
    import data.build_runtime as runtime_builder
    import data.kline_history_repairs as history_repairs
    import data.kline_mootdx as kline_mootdx
    import data.kline_secondary_evidence as secondary_evidence
    import data.update_all as update_all

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        secondary_evidence,
        "refresh_verified_daily_evidence",
        lambda path: calls.append(("secondary_evidence", path)),
    )
    monkeypatch.setattr(
        update_all,
        "_ensure_baostock_daily_evidence",
        lambda **kwargs: calls.append(("daily_evidence", kwargs)),
    )
    monkeypatch.setattr(
        kline_mootdx,
        "find_placeholder_candidate_codes",
        lambda **kwargs: calls.append(("scan", kwargs)) or ("000003.SZ",),
    )
    monkeypatch.setattr(
        kline_mootdx,
        "reconcile_baostock_placeholder_bars",
        lambda codes, **kwargs: calls.append(("reconcile", codes, kwargs)),
    )
    monkeypatch.setattr(
        kline_mootdx,
        "update_baostock_listing_evidence",
        lambda codes, **kwargs: calls.append(("evidence", codes, kwargs)),
    )
    monkeypatch.setattr(
        history_repairs,
        "ensure_verified_history_repairs",
        lambda **kwargs: calls.append(("history_repairs", kwargs)) or (),
    )
    conflict = runtime_builder.ListingDateAlignmentError(
        [
            {
                "code": "600001.SH",
                "first_kline_date": "2020-01-03",
                "first_executable_date": "2020-01-02",
                "expected_listing_dates": ["2020-01-01"],
                "expected_first_executable_dates": ["2020-01-02"],
                "evidence": {},
            }
        ]
    )
    attempts = iter((conflict, tmp_path / "runtime.npz"))

    def fake_build():
        result = next(attempts)
        calls.append("build")
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runtime_builder, "build_runtime", fake_build)

    update_all._build_runtime()

    assert [item if isinstance(item, str) else item[0] for item in calls] == [
        "secondary_evidence",
        "history_repairs",
        "daily_evidence",
        "scan",
        "reconcile",
        "build",
        "evidence",
        "build",
    ]
    evidence_call = next(item for item in calls if isinstance(item, tuple) and item[0] == "evidence")
    assert evidence_call[1] == ["600001.SH"]


def test_daily_evidence_adds_new_delisted_code_without_rebuilding_v2(
    tmp_path, monkeypatch,
):
    import data.db.delist as delist_db
    import data.kline_mootdx as kline_mootdx
    import data.update_all as update_all
    import utils.stock.info as stock_info

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    kline_dir = tmp_path / "k-line"
    kline_dir.mkdir()
    for code in ("000001.SZ", "000002.SZ"):
        (kline_dir / f"{code}.parquet").touch()
    evidence_path = tmp_path / "evidence.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001.SZ"],
            "schema_version": [kline_mootdx.BAOSTOCK_EVIDENCE_SCHEMA],
        }
    ).to_parquet(evidence_path, index=False)
    monkeypatch.setattr(
        delist_db,
        "get_delist_stock_info",
        lambda: {"000001.SZ": object(), "000002.SZ": object()},
    )
    monkeypatch.setattr(stock_info, "is_b_stock", lambda _code: False)
    state = {"updated": False}
    monkeypatch.setattr(
        kline_mootdx,
        "load_baostock_daily_evidence",
        lambda _path: pd.DataFrame(
            {
                "stock_code": (
                    ["000001.SZ", "000002.SZ"]
                    if state["updated"]
                    else ["000001.SZ"]
                )
            }
        ),
    )
    calls = []

    def fake_update(codes, **kwargs):
        calls.append((codes, kwargs))
        state["updated"] = True

    monkeypatch.setattr(
        kline_mootdx, "update_baostock_listing_evidence", fake_update
    )
    monkeypatch.setattr(
        kline_mootdx,
        "rebuild_masked_kline_rows_from_backups",
        lambda *_args, **_kwargs: pytest.fail("v2 incremental add must not rebuild"),
    )

    update_all._ensure_baostock_daily_evidence(
        kline_dir=kline_dir,
        evidence_path=evidence_path,
    )

    assert calls[0][0] == ["000002.SZ"]


def test_daily_evidence_refreshes_active_code_when_local_tail_advances(
    tmp_path, monkeypatch,
):
    import data.db.delist as delist_db
    import data.kline_mootdx as kline_mootdx
    import data.update_all as update_all

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    kline_dir = tmp_path / "k-line"
    kline_dir.mkdir()
    code = "000001.SZ"
    pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2026-08-28").value // 10**6,
                pd.Timestamp("2026-08-31").value // 10**6,
            ],
            "open": [10.0, 10.1],
        }
    ).to_parquet(kline_dir / f"{code}.parquet", index=False)
    evidence_path = tmp_path / "evidence.parquet"
    pd.DataFrame(
        {
            "stock_code": [code],
            "schema_version": [kline_mootdx.BAOSTOCK_EVIDENCE_SCHEMA],
        }
    ).to_parquet(evidence_path, index=False)
    monkeypatch.setattr(delist_db, "get_delist_stock_info", lambda: {})
    state = {"updated": False}

    def fake_load(_path):
        return pd.DataFrame(
            {
                "stock_code": [code],
                "date": ["2026-08-31" if state["updated"] else "2026-08-28"],
                "out_date": [pd.NaT],
            }
        )

    monkeypatch.setattr(kline_mootdx, "load_baostock_daily_evidence", fake_load)
    calls = []

    def fake_update(codes, **kwargs):
        calls.append((codes, kwargs))
        state["updated"] = True

    monkeypatch.setattr(
        kline_mootdx, "update_baostock_listing_evidence", fake_update
    )
    monkeypatch.setattr(
        kline_mootdx,
        "rebuild_masked_kline_rows_from_backups",
        lambda *_args, **_kwargs: pytest.fail("current schema must refresh incrementally"),
    )

    update_all._ensure_baostock_daily_evidence(
        kline_dir=kline_dir,
        evidence_path=evidence_path,
    )

    assert calls[0][0] == [code]


def test_active_daily_evidence_same_calendar_date_is_not_stale(tmp_path):
    import data.update_all as update_all

    kline_dir = tmp_path / "k-line"
    kline_dir.mkdir()
    code = "000001.SZ"
    pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-08-28 15:00:00").value // 10**6],
            "open": [10.0],
        }
    ).to_parquet(kline_dir / f"{code}.parquet", index=False)
    evidence = pd.DataFrame(
        {
            "stock_code": [code],
            "date": ["2026-08-28"],
            "out_date": [pd.NaT],
        }
    )

    assert update_all._active_evidence_codes_needing_refresh(
        evidence,
        kline_dir=kline_dir,
    ) == ()


def test_daily_evidence_v1_upgrade_rebuilds_old_and_required_union(
    tmp_path, monkeypatch,
):
    import data.db.delist as delist_db
    import data.kline_mootdx as kline_mootdx
    import data.update_all as update_all
    import utils.stock.info as stock_info

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    kline_dir = tmp_path / "k-line"
    kline_dir.mkdir()
    for code in ("000001.SZ", "000002.SZ"):
        (kline_dir / f"{code}.parquet").touch()
    evidence_path = tmp_path / "evidence.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001.SZ"],
            "schema_version": ["baostock-daily-reference-v1"],
        }
    ).to_parquet(evidence_path, index=False)
    monkeypatch.setattr(
        delist_db,
        "get_delist_stock_info",
        lambda: {"000002.SZ": object()},
    )
    monkeypatch.setattr(stock_info, "is_b_stock", lambda _code: False)
    calls = []
    monkeypatch.setattr(
        kline_mootdx,
        "rebuild_masked_kline_rows_from_backups",
        lambda codes, **kwargs: calls.append((codes, kwargs)) or {},
    )
    monkeypatch.setattr(
        kline_mootdx,
        "update_baostock_listing_evidence",
        lambda *_args, **_kwargs: pytest.fail("v1 must use the atomic rebuild"),
    )
    monkeypatch.setattr(
        kline_mootdx,
        "load_baostock_daily_evidence",
        lambda _path: pd.DataFrame(
            {"stock_code": ["000001.SZ", "000002.SZ"]}
        ),
    )

    update_all._ensure_baostock_daily_evidence(
        kline_dir=kline_dir,
        evidence_path=evidence_path,
    )

    assert calls[0][0] == ["000001.SZ", "000002.SZ"]


@pytest.mark.parametrize("missing", ["snapshot", "manifest"])
def test_issue_update_rejects_incomplete_snapshot_without_migrating(
    tmp_path, monkeypatch, missing
):
    _issue_reference, update_all, output, manifest = (
        _install_old_axis_missing_issue_reference(tmp_path, monkeypatch)
    )
    absent, preserved = (output, manifest) if missing == "snapshot" else (manifest, output)
    absent.unlink()
    original = preserved.read_bytes()

    def forbidden_fetch(_bare):
        pytest.fail("incomplete local snapshot must fail before network access")

    monkeypatch.setattr(update_all, "_fetch_sina_issue_record", forbidden_fetch)
    with pytest.raises(FileNotFoundError, match="issue_price"):
        update_all._update_issue_price()

    assert preserved.read_bytes() == original
    assert not absent.exists()

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from offline_data import financial_snapshot as snapshot
from offline_data.financial_versions import FINANCIAL_PANEL_FIELDS, iter_financial_fields
from test_video_financial_versions import events


def test_snapshot_preserves_base_axes_and_authenticates_enrichment(tmp_path, monkeypatch):
    base = tmp_path / "base.npz"
    dates = np.array(["2021-04-20", "2021-04-21", "2021-04-22"], dtype="datetime64[D]")
    original = {"trade_dates": dates, "stock_codes": np.array(["000001.SZ"]),
                "open": np.arange(3., dtype=np.float32)[:, None]}
    np.savez_compressed(base, **original)
    financial = tmp_path / "financial"
    financial.mkdir()
    (financial / "request.json").write_text("{}")
    identity = {"sha256": "test-sealed-events"}
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    versions = events()
    monkeypatch.setattr(snapshot, "load_financial_events", lambda *args, **kwargs: (versions, identity))
    target = tmp_path / "enriched.npz"
    metadata = snapshot.build_financial_snapshot(
        base, financial, target, expected_base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        financial_identity_path=identity_path,
    )
    assert snapshot.read_financial_snapshot_manifest(target) == metadata
    with np.load(target) as values:
        assert set(values.files) == set(original) | set(FINANCIAL_PANEL_FIELDS)
        for name, expected in original.items():
            np.testing.assert_array_equal(values[name], expected)
        np.testing.assert_array_equal(values["financial_profit_ttm"], np.full((3, 1), 400.))
        for name in FINANCIAL_PANEL_FIELDS:
            assert values[name].dtype == np.float64
    with pytest.raises(FileExistsError):
        snapshot.build_financial_snapshot(base, financial, target, expected_base_sha256="bad",
                                          financial_identity_path=identity_path)
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="file hash"):
        snapshot.read_financial_snapshot_manifest(target)


def test_financial_fields_exclude_same_day_revision_and_do_not_forward_fill_period_gap():
    versions = events()
    income = versions["Income"]
    values = income.values.copy()
    values[income.quarters == 2021 * 4, 0] = 200.
    versions["Income"] = replace(income, values=values)
    dates = np.array(["2021-04-20", "2021-04-21"], dtype="datetime64[D]")
    rows = [fields for _, fields in iter_financial_fields(dates, 1, versions)]
    assert rows[0]["financial_profit_ttm"][0] == 400.
    assert rows[1]["financial_profit_ttm"][0] == 500.
    assert rows[1]["financial_profit_yoy"][0] == 1.
    assert rows[1]["financial_cash_outflow_yoy"][0] == 0.
    assert not rows[0]["financial_equity"].flags.writeable
    # Removing the prior matching quarter must make the new TTM unavailable.
    keep = income.quarters != 2020 * 4
    versions["Income"] = replace(income, announcement_dates=income.announcement_dates[keep],
                                 quarters=income.quarters[keep], stock_columns=income.stock_columns[keep],
                                 values=values[keep])
    last = list(iter_financial_fields(dates, 1, versions))[-1][1]
    assert np.isnan(last["financial_profit_ttm"][0])
    assert np.isnan(last["financial_profit_yoy"][0])


def test_old_six_panel_snapshot_identity_is_explicitly_rejected(tmp_path):
    path = tmp_path / "old.npz"
    path.with_suffix(".manifest.json").write_text(json.dumps({"schema": "financial-primitives-snapshot-v1"}))
    with pytest.raises(ValueError, match="unsupported financial snapshot schema"):
        snapshot.read_financial_snapshot_manifest(path)

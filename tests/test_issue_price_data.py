from datetime import date
import json

import pandas as pd
import pytest

from data.db import issue_price


def _reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001", "600000"],
            "issue_price": [3.20, 10.00],
            "list_date": [date(1991, 4, 3), date(1999, 11, 10)],
            "source": ["sina-vISSUE_NewStock", "sina-vISSUE_NewStock"],
            "source_as_of": [date(2026, 8, 31), date(2026, 8, 31)],
        }
    )


def test_issue_price_v2_atomic_save_and_strict_load(tmp_path):
    path = tmp_path / "issue_price.parquet"
    manifest_path = tmp_path / "issue_price.manifest.json"

    issue_price.save_issue_reference_atomic(
        _reference_frame(), path=path, manifest_path=manifest_path
    )
    loaded = issue_price.load_issue_reference(
        path=path, manifest_path=manifest_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pd.testing.assert_frame_equal(
        loaded, issue_price.validate_issue_reference_frame(_reference_frame())
    )
    assert manifest["schema_version"] == issue_price.ISSUE_REFERENCE_SCHEMA
    assert manifest["row_count"] == 2
    assert manifest["sources"] == ["sina-vISSUE_NewStock"]
    assert manifest["source_as_of_max"] == "2026-08-31"
    assert not list(tmp_path.glob("*.tmp*"))


def test_issue_price_loader_rejects_parquet_hash_tampering(tmp_path):
    path = tmp_path / "issue_price.parquet"
    manifest_path = tmp_path / "issue_price.manifest.json"
    issue_price.save_issue_reference_atomic(
        _reference_frame(), path=path, manifest_path=manifest_path
    )
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA256"):
        issue_price.load_issue_reference(
            path=path, manifest_path=manifest_path
        )


def test_issue_price_validator_rejects_extra_column_and_nan_source():
    extra = _reference_frame().assign(untrusted="duplicate-default")
    with pytest.raises(ValueError, match="列集合不匹配.*extra"):
        issue_price.validate_issue_reference_frame(extra)

    missing_source = _reference_frame()
    missing_source.loc[0, "source"] = None
    with pytest.raises(ValueError, match="source 为空"):
        issue_price.validate_issue_reference_frame(missing_source)


def test_issue_price_loader_rejects_source_as_of_manifest_tampering(tmp_path):
    path = tmp_path / "issue_price.parquet"
    manifest_path = tmp_path / "issue_price.manifest.json"
    issue_price.save_issue_reference_atomic(
        _reference_frame(), path=path, manifest_path=manifest_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_as_of_max"] = "2099-12-31"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source_as_of_max"):
        issue_price.load_issue_reference(
            path=path, manifest_path=manifest_path
        )


def test_sina_issue_price_parser_accepts_minimal_unique_html(monkeypatch):
    import data.update_all as update_all

    html = """
    <html><body><div>股票代码：301688</div><table>
      <tr><td>发行价(元)</td><td>12.34</td></tr>
      <tr><td>上市日期</td><td>2026-09-01</td></tr>
    </table></body></html>
    """

    class Response:
        content = html.encode("gbk")

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(
        update_all.requests, "get", lambda *_args, **_kwargs: Response()
    )
    monkeypatch.setattr(update_all, "TODAY", date(2026, 8, 31))

    record = update_all._fetch_sina_issue_record("301688")

    assert record == {
        "stock_code": "301688",
        "issue_price": 12.34,
        "list_date": date(2026, 9, 1),
        "source": update_all.ISSUE_REFERENCE_SOURCE,
        "source_as_of": date(2026, 8, 31),
    }

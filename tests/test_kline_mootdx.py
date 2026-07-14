import pytest

from data import kline_mootdx


class _EmptyMootdx:
    def __init__(self):
        self.calls = 0

    def xdxr(self, symbol):
        return None

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

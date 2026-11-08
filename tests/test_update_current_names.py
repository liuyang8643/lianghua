from datetime import date

import pandas as pd


def test_update_current_names_uses_tencent_fetcher(tmp_path, monkeypatch):
    import data.update_all as update_all

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    monkeypatch.setattr(update_all, "TODAY", date.today())

    monkeypatch.setattr(
        update_all,
        "_fetch_current_stock_names",
        lambda codes: pd.DataFrame({
            "code": ["000001", "600000", "430001", "900901"],
            "name": ["平安银行", "浦发银行", "北交测试", "云赛B股"],
        }),
    )

    update_all._update_current_names(["000001.SZ", "600000.SH", "430001.BJ", "900901.SH"])

    df = pd.read_parquet(tmp_path / "stock_name" / "current_names.parquet")
    assert df.to_dict("records") == [
        {"bare_code": "000001", "stock_code": "000001.SZ", "name": "平安银行"},
        {"bare_code": "600000", "stock_code": "600000.SH", "name": "浦发银行"},
        {"bare_code": "430001", "stock_code": "430001.BJ", "name": "北交测试"},
        {"bare_code": "900901", "stock_code": "900901.SH", "name": "云赛B股"},
    ]


def test_update_current_names_rejects_partial_response_without_overwrite(
    tmp_path, monkeypatch,
):
    import data.update_all as update_all

    monkeypatch.setattr(update_all, "DATA_DIR", tmp_path)
    path = tmp_path / "stock_name" / "current_names.parquet"
    path.parent.mkdir(parents=True)
    previous = pd.DataFrame(
        [{"bare_code": "000001", "stock_code": "000001.SZ", "name": "旧简称"}]
    )
    previous.to_parquet(path, index=False)
    original = path.read_bytes()
    monkeypatch.setattr(
        update_all,
        "_fetch_current_stock_names",
        lambda _codes: pd.DataFrame({"code": ["000001"], "name": ["平安银行"]}),
    )
    monkeypatch.setattr(update_all, "TODAY", date(2000, 1, 1))

    import pytest

    with pytest.raises(RuntimeError, match="未覆盖完整股票轴"):
        update_all._update_current_names(["000001.SZ", "600000.SH"])
    assert path.read_bytes() == original


def test_tencent_quote_symbol_routes_boards():
    import data.update_all as update_all

    assert update_all._tencent_quote_symbol("000001.SZ") == "sz000001"
    assert update_all._tencent_quote_symbol("600000.SH") == "sh600000"
    assert update_all._tencent_quote_symbol("688001.SH") == "sh688001"
    assert update_all._tencent_quote_symbol("430001.BJ") == "bj430001"
    assert update_all._tencent_quote_symbol("830001.BJ") == "bj830001"
    assert update_all._tencent_quote_symbol("870001.BJ") == "bj870001"
    assert update_all._tencent_quote_symbol("920001.BJ") == "bj920001"


def test_fetch_current_stock_names_parses_tencent_response(monkeypatch):
    import data.update_all as update_all

    calls = []

    class _Resp:
        def __init__(self, text):
            self.content = text.encode("gbk")

        def raise_for_status(self):
            return None

    def fake_get(url, headers, timeout):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        symbols = url.split("q=", 1)[1].split(",")
        lines = []
        for sym in symbols:
            bare = sym[2:]
            name = {"000001": "平安银行", "600000": "浦发银行", "430001": "北交测试"}[bare]
            lines.append(f'v_{sym}="51~{name}~{bare}~0";')
        return _Resp("".join(lines))

    monkeypatch.setattr(update_all, "CURRENT_NAMES_CHUNK_SIZE", 2)
    monkeypatch.setattr(update_all.requests, "get", fake_get)

    df = update_all._fetch_current_stock_names(["000001.SZ", "600000.SH", "430001.BJ"])

    assert calls == [
        {
            "url": "https://qt.gtimg.cn/q=sz000001,bj430001",
            "headers": {"User-Agent": "Mozilla/5.0"},
            "timeout": update_all.CURRENT_NAMES_REQUEST_TIMEOUT,
        },
        {
            "url": "https://qt.gtimg.cn/q=sh600000",
            "headers": {"User-Agent": "Mozilla/5.0"},
            "timeout": update_all.CURRENT_NAMES_REQUEST_TIMEOUT,
        },
    ]
    assert df.to_dict("records") == [
        {"code": "000001", "name": "平安银行"},
        {"code": "430001", "name": "北交测试"},
        {"code": "600000", "name": "浦发银行"},
    ]

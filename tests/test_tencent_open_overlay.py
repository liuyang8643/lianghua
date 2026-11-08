from datetime import date

import numpy as np
import pandas as pd
import pytest


TRADE_DATE = date(2026, 8, 31)


def _quote(
        symbol: str, *, open_price: str = "10.20", pre_close: str = "10.00",
        quote_date: date = TRADE_DATE, response_code: str | None = None,
) -> str:
    values = [""] * 31
    values[0] = "51"
    values[1] = "测试股"
    values[2] = response_code or symbol[2:]
    values[3] = "99.99"  # 当日现价：因果契约要求不读取
    values[4] = pre_close
    values[5] = open_price
    values[30] = quote_date.strftime("%Y%m%d") + "092500"
    return f'v_{symbol}="{"~".join(values)}";'


class _Response:
    def __init__(self, text: str):
        self.content = text.encode("gbk")
        self.status_checked = False

    def raise_for_status(self):
        self.status_checked = True


def test_tencent_overlay_batches_routes_and_exposes_only_t_open(monkeypatch):
    import data.update_live as update_live

    calls = []

    def fake_get(url, headers, timeout):
        symbols = url.split("q=", 1)[1].split(",")
        response = _Response("".join(_quote(symbol) for symbol in symbols))
        calls.append((url, headers, timeout, response))
        return response

    monkeypatch.setattr(update_live, "TENCENT_OPEN_CHUNK_SIZE", 2)
    monkeypatch.setattr(update_live.requests, "get", fake_get)

    overlay = update_live.fetch_tencent_open_overlay(
        ["000001.SZ", "600000.SH", "430001.BJ"],
        TRADE_DATE,
        min_coverage=1.0,
    )

    assert [call[:3] for call in calls] == [
        (
            "https://qt.gtimg.cn/q=sz000001,sh600000",
            {"User-Agent": "Mozilla/5.0"},
            update_live.TENCENT_OPEN_REQUEST_TIMEOUT,
        ),
        (
            "https://qt.gtimg.cn/q=bj430001",
            {"User-Agent": "Mozilla/5.0"},
            update_live.TENCENT_OPEN_REQUEST_TIMEOUT,
        ),
    ]
    assert all(call[3].status_checked for call in calls)
    assert list(overlay) == ["000001.SZ", "600000.SH", "430001.BJ"]

    expected_time = pd.Timestamp(TRADE_DATE).value // 1_000_000
    for bars in overlay.values():
        assert bars["time"].tolist() == [expected_time]
        assert bars["open"].tolist() == [10.2]
        assert bars["preClose"].tolist() == [10.0]
        for forbidden in ("high", "low", "close", "volume", "amount"):
            assert bars[forbidden].shape == (1,)
            assert np.isnan(bars[forbidden][0])


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("000001.SZ", "sz000001"),
        ("600000.SH", "sh600000"),
        ("688001.SH", "sh688001"),
        ("430001.BJ", "bj430001"),
        ("830001", "bj830001"),
        ("920001", "bj920001"),
    ],
)
def test_tencent_symbol_routes_sh_sz_bj(code, expected):
    import data.update_live as update_live

    symbol, canonical = update_live._tencent_quote_symbol(code)

    assert symbol == expected
    assert canonical == f"{expected[2:]}.{expected[:2].upper()}"


def test_tencent_overlay_rejects_duplicate_request_before_network(monkeypatch):
    import data.update_live as update_live

    monkeypatch.setattr(
        update_live.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("duplicate must fail before network"),
    )

    with pytest.raises(ValueError, match="重复代码"):
        update_live.fetch_tencent_open_overlay(
            ["000001.SZ", "000001.SZ"], TRADE_DATE
        )


def test_tencent_overlay_rejects_duplicate_response(monkeypatch):
    import data.update_live as update_live

    monkeypatch.setattr(
        update_live.requests,
        "get",
        lambda *args, **kwargs: _Response(
            _quote("sz000001") + _quote("sz000001")
        ),
    )

    with pytest.raises(RuntimeError, match="响应代码重复"):
        update_live.fetch_tencent_open_overlay(
            ["000001.SZ"], TRADE_DATE, min_coverage=1.0
        )


def test_tencent_overlay_rejects_wrong_date(monkeypatch):
    import data.update_live as update_live

    monkeypatch.setattr(
        update_live.requests,
        "get",
        lambda *args, **kwargs: _Response(
            _quote("sh600000", quote_date=date(2026, 8, 28))
        ),
    )

    with pytest.raises(RuntimeError, match=r"错日.*2026-08-28.*2026-08-31"):
        update_live.fetch_tencent_open_overlay(
            ["600000.SH"], TRADE_DATE, min_coverage=1.0
        )


@pytest.mark.parametrize(
    "payload",
    [
        'v_sz000001="51~too~short";',
        _quote("sz000001", response_code="000002"),
        'not_a_tencent_quote;',
    ],
)
def test_tencent_overlay_rejects_malformed_response(monkeypatch, payload):
    import data.update_live as update_live

    monkeypatch.setattr(
        update_live.requests,
        "get",
        lambda *args, **kwargs: _Response(payload),
    )

    with pytest.raises(RuntimeError, match=r"(畸形|代码不一致)"):
        update_live.fetch_tencent_open_overlay(
            ["000001.SZ"], TRADE_DATE, min_coverage=1.0
        )


def test_tencent_overlay_fails_fast_with_exact_coverage_diagnostics(monkeypatch):
    import data.update_live as update_live

    monkeypatch.setattr(
        update_live.requests,
        "get",
        lambda *args, **kwargs: _Response(
            _quote("sz000001")
            + _quote("sh600000", open_price="0", pre_close="8.88")
            # bj430001 故意无响应
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        update_live.fetch_tencent_open_overlay(
            ["000001.SZ", "600000.SH", "430001.BJ"],
            TRADE_DATE,
            min_coverage=0.95,
        )

    message = str(exc_info.value)
    assert "覆盖率不足: 2/3=66.67%" in message
    assert "无响应=[bj430001]" in message
    assert "非正价格=[-]" in message


def test_zero_open_is_observed_untradable_not_missing_response(monkeypatch):
    import data.update_live as update_live
    monkeypatch.setattr(update_live.requests, 'get', lambda *a, **k: _Response(
        _quote('sz000001') + _quote('sh600000', open_price='0')))
    overlay = update_live.fetch_tencent_open_overlay(
        ['000001.SZ', '600000.SH'], TRADE_DATE, min_coverage=1)
    assert overlay['600000.SH']['open'][0] == 0
    assert overlay['600000.SH']['preClose'][0] == 10
    assert np.isnan(overlay['600000.SH']['close'][0])


def test_all_zero_open_is_not_a_ready_market(monkeypatch):
    import data.update_live as update_live
    monkeypatch.setattr(update_live.requests, 'get', lambda *a, **k: _Response(
        _quote('sz000001', open_price='0')))
    with pytest.raises(RuntimeError, match='尚未就绪'):
        update_live.fetch_tencent_open_overlay(['000001.SZ'], TRADE_DATE)


def test_update_live_quick_default_is_memory_only(monkeypatch, tmp_path):
    import data.kline_mootdx as kline_mootdx
    import data.update_live as update_live

    expected = {"000001.SZ": {"sentinel": True}}
    calls = []

    monkeypatch.setattr(update_live, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        kline_mootdx,
        "resolve_recent_range",
        lambda days, anchor: ("20260831", "20260831", TRADE_DATE),
    )
    monkeypatch.setattr(
        update_live,
        "fetch_tencent_open_overlay",
        lambda codes, trade_date: calls.append((codes, trade_date)) or expected,
    )
    monkeypatch.setattr(
        update_live,
        "_download_kline_all",
        lambda *args, **kwargs: pytest.fail("default path must not call mootdx"),
    )
    monkeypatch.setattr(
        update_live,
        "_patch_npz_incremental",
        lambda *args, **kwargs: pytest.fail("default path must not write NPZ"),
    )

    result = update_live.update_live_quick(
        patch_npz=False,
        anchor_date=TRADE_DATE,
        codes=["000001.SZ"],
    )

    assert result is expected
    assert calls == [(["000001.SZ"], TRADE_DATE)]
    assert list(tmp_path.rglob("*")) == []


def test_local_historical_overlay_reads_only_open_and_preclose(monkeypatch, tmp_path):
    import data.update_live as update_live

    kline_dir = tmp_path / "k-line"
    kline_dir.mkdir()
    pd.DataFrame({
        "time": [pd.Timestamp(TRADE_DATE).value // 1_000_000],
        "open": [10.2],
        "preClose": [10.0],
        "close": [99.0],
    }).to_parquet(kline_dir / "000001.SZ.parquet", index=False)
    monkeypatch.setattr(update_live, "DATA_DIR", tmp_path)

    overlay = update_live.load_local_open_overlay(
        ["000001.SZ"], TRADE_DATE, min_coverage=1.0,
    )

    assert overlay["000001.SZ"]["open"].tolist() == [10.2]
    assert overlay["000001.SZ"]["preClose"].tolist() == [10.0]
    assert np.isnan(overlay["000001.SZ"]["close"]).all()
    assert np.isnan(overlay["000001.SZ"]["volume"]).all()


def test_local_historical_overlay_rejects_missing_date(monkeypatch, tmp_path):
    import data.update_live as update_live

    kline_dir = tmp_path / "k-line"
    kline_dir.mkdir()
    pd.DataFrame({
        "time": [pd.Timestamp("2026-08-28").value // 1_000_000],
        "open": [10.2],
        "preClose": [10.0],
    }).to_parquet(kline_dir / "000001.SZ.parquet", index=False)
    monkeypatch.setattr(update_live, "DATA_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="覆盖率不足"):
        update_live.load_local_open_overlay(
            ["000001.SZ"], TRADE_DATE, min_coverage=1.0,
        )

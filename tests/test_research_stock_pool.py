import numpy as np

import core.runtime as runtime


def test_runtime_stock_pool_keeps_delisted_codes(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    np.savez_compressed(
        runtime_dir / "runtime_20200101_20201231.npz",
        stock_codes=np.array(["000001.SZ", "600999.SH"], dtype="U12"),
        trade_dates=np.array(["2020-01-01"], dtype="datetime64[D]"),
    )
    np.savez_compressed(
        runtime_dir / "runtime_20210101_20211231.npz",
        stock_codes=np.array(["000001.SZ", "000003.SZ"], dtype="U12"),
        trade_dates=np.array(["2021-01-01"], dtype="datetime64[D]"),
    )

    monkeypatch.setattr(runtime, "_RUNTIME_DIR", runtime_dir)

    assert runtime.load_runtime_stock_codes() == ["000001.SZ", "000003.SZ"]

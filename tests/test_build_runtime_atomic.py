import numpy as np
import pytest

from data.build_runtime import save_runtime_npz_atomic


def test_atomic_runtime_publish_is_readable_and_leaves_no_temp(tmp_path):
    destination = tmp_path / "runtime_2026-09-03_2026-09-03.npz"

    result = save_runtime_npz_atomic(
        destination,
        trade_dates=np.array(['2026-09-03'], dtype='datetime64[D]'),
        stock_codes=np.array(['000001.SZ']),
    )

    assert result == destination
    with np.load(destination, allow_pickle=False) as runtime:
        assert str(runtime['trade_dates'][-1]) == '2026-09-03'
    assert list(tmp_path.glob(f".{destination.name}.tmp.*")) == []


def test_atomic_runtime_failure_preserves_previous_archive(tmp_path, monkeypatch):
    destination = tmp_path / "runtime_2026-09-03_2026-09-03.npz"
    previous = b"previous-runtime-remains-readable-by-owner"
    destination.write_bytes(previous)

    def fail_compression(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail_compression)

    with pytest.raises(OSError, match="disk full"):
        save_runtime_npz_atomic(
            destination,
            trade_dates=np.array(['2026-09-03'], dtype='datetime64[D]'),
        )

    assert destination.read_bytes() == previous
    assert list(tmp_path.glob(f".{destination.name}.tmp.*")) == []

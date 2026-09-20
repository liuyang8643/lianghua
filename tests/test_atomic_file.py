import pytest
import os
import hashlib
import json
from pathlib import Path

from utils import atomic_file


@pytest.mark.parametrize("options", [
    {}, {"sort_keys": False, "allow_nan": False},
    {"indent": None, "separators": (",", ":")},
])
@pytest.mark.parametrize("trailing_newline", [False, True])
def test_atomic_json_keeps_existing_checkpoint_ga_and_journal_bytes(tmp_path, options, trailing_newline):
    payload = {"z": [0.1, None], "中文": "真实记录", "a": {"valid": True}}
    expected = tmp_path / "old.json"
    serialization = {"ensure_ascii": False, "sort_keys": True, "indent": 2, **options}
    expected.write_text(json.dumps(payload, **serialization) + ("\n" if trailing_newline else ""),
                        encoding="utf8")
    actual = tmp_path / "nested" / "new.json"
    atomic_file.atomic_write_json(actual, payload, **options, trailing_newline=trailing_newline)
    assert actual.read_bytes() == expected.read_bytes()
    assert atomic_file.file_sha256(actual) == hashlib.sha256(expected.read_bytes()).hexdigest()
    assert list(actual.parent.iterdir()) == [actual]


def test_atomic_json_rejects_nan_or_failed_replace_without_losing_previous(tmp_path, monkeypatch):
    target = tmp_path / "current.json"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError):
        atomic_file.atomic_write_json(target, {"value": float("nan")}, allow_nan=False)
    assert target.read_bytes() == b"previous"
    calls = []
    monkeypatch.setattr(atomic_file.os, "fsync", lambda fd: calls.append("fsync"))

    def fail(source, destination):
        assert calls == ["fsync"]
        assert destination == target
        assert json.loads(Path(source).read_text("utf8")) == {"next": 1}
        raise OSError("disk unavailable")

    monkeypatch.setattr(atomic_file, "replace_file", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        atomic_file.atomic_write_json(target, {"next": 1})
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_conflict_retries_same_replace(tmp_path, monkeypatch, code):
    source, target = tmp_path / "pending", tmp_path / "current"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    original = atomic_file._replace
    attempts = []

    def replace(a, b):
        attempts.append((a, b))
        if len(attempts) < 3:
            assert target.read_bytes() == b"old"
            error = PermissionError("sharing")
            error.winerror = code
            raise error
        return original(a, b)

    monkeypatch.setattr(atomic_file, "_replace", replace)
    monkeypatch.setattr(atomic_file.time, "sleep", lambda seconds: None)
    atomic_file.replace_file(source, target)
    assert target.read_bytes() == b"new"
    assert len(attempts) == 3


@pytest.mark.parametrize("code, count", [(5, 51), (112, 1), (None, 1)])
def test_persistent_or_unrelated_error_preserves_files(tmp_path, monkeypatch, code, count):
    source, target = tmp_path / "pending", tmp_path / "current"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    attempts = []
    error = OSError("failure")
    if code is not None:
        error.winerror = code

    def replace(a, b):
        attempts.append(1)
        raise error

    monkeypatch.setattr(atomic_file, "_replace", replace)
    monkeypatch.setattr(atomic_file.time, "sleep", lambda seconds: None)
    with pytest.raises(OSError):
        atomic_file.replace_file(source, target)
    assert len(attempts) == count
    assert source.read_bytes() == b"new"
    assert target.read_bytes() == b"old"


@pytest.mark.skipif(os.name != "nt", reason="Windows handle sharing")
def test_real_windows_reader_lock_is_retried(tmp_path):
    import ctypes
    import threading

    source, target = tmp_path / "pending", tmp_path / "current"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong,
                                  ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.CreateFileW(str(target), 0x80000000, 1, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    release = threading.Timer(0.3, kernel.CloseHandle, args=(handle,))
    release.start()
    try:
        atomic_file.replace_file(source, target)
        assert target.read_bytes() == b"new"
    finally:
        release.join()

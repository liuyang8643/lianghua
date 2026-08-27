from pathlib import Path

import pytest

import testback.run_ga as run_ga


def test_cleanup_memmap_removes_registered_directory(tmp_path):
    memmap_dir = tmp_path / "memmap_case"
    memmap_dir.mkdir()
    (memmap_dir / "array.bin").write_bytes(b"temporary")
    run_ga._MEMMAP_DIRS.append(str(memmap_dir))

    run_ga._cleanup_memmap()

    assert not memmap_dir.exists()
    assert str(memmap_dir) not in run_ga._MEMMAP_DIRS


def test_cleanup_memmap_exposes_failure_and_keeps_retry_target(
    monkeypatch, tmp_path
):
    memmap_dir = tmp_path / "memmap_locked"
    memmap_dir.mkdir()
    run_ga._MEMMAP_DIRS.append(str(memmap_dir))

    def fail_remove(path):
        raise PermissionError(f"locked: {Path(path).name}")

    monkeypatch.setattr("shutil.rmtree", fail_remove)

    with pytest.raises(RuntimeError, match="memmap_locked"):
        run_ga._cleanup_memmap()

    assert run_ga._MEMMAP_DIRS == [str(memmap_dir)]

    run_ga._MEMMAP_DIRS.clear()


class _FakeSharedMemory:
    def __init__(self, name, *, fail_unlink=False):
        self.name = name
        self.fail_unlink = fail_unlink
        self.closed = False
        self.unlinked = False

    def close(self):
        self.closed = True

    def unlink(self):
        if self.fail_unlink:
            raise PermissionError(f"locked: {self.name}")
        self.unlinked = True


def test_cleanup_shm_releases_registered_blocks():
    shared = _FakeSharedMemory("shared_ok")
    run_ga._SHM_BLOCKS.append((shared, object()))

    run_ga._cleanup_shm()

    assert shared.closed is True
    assert shared.unlinked is True
    assert run_ga._SHM_BLOCKS == []


def test_cleanup_shm_exposes_failure_and_keeps_retry_target():
    shared = _FakeSharedMemory("shared_locked", fail_unlink=True)
    block = (shared, object())
    run_ga._SHM_BLOCKS.append(block)

    with pytest.raises(RuntimeError, match="shared_locked"):
        run_ga._cleanup_shm()

    assert run_ga._SHM_BLOCKS == [block]
    run_ga._SHM_BLOCKS.clear()


def test_cleanup_ga_resources_attempts_both_and_surfaces_failure(
    monkeypatch,
):
    calls = []

    def fail_shm():
        calls.append("shm")
        raise RuntimeError("shared_locked")

    def finish_memmap():
        calls.append("memmap")

    monkeypatch.setattr(run_ga, "_cleanup_shm", fail_shm)
    monkeypatch.setattr(run_ga, "_cleanup_memmap", finish_memmap)

    with pytest.raises(RuntimeError, match="shared_locked"):
        run_ga._cleanup_ga_resources()

    assert calls == ["shm", "memmap"]

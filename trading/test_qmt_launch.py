from pathlib import Path
from types import SimpleNamespace

from trading import qmt


def test_current_client_without_legacy_link_file_can_be_launched(tmp_path, monkeypatch):
    exe = tmp_path / 'XtMiniQmt.exe'
    exe.write_bytes(b'fixture-only-not-executable')
    monkeypatch.setattr(qmt, 'QMT_ROOT_DIR', [str(tmp_path)])
    monkeypatch.setattr(qmt, 'get_qmt_process', lambda: None)
    launched = []
    process = SimpleNamespace(pid=123)
    monkeypatch.setattr(qmt.psutil, 'Popen', lambda cmd, **kw: launched.append((cmd, kw)) or process)
    assert qmt.start_qmt() is process
    assert launched == [([str(exe)], {'cwd': str(tmp_path)})]
    assert not (tmp_path / 'linkMini').exists()


def test_legacy_client_link_file_keeps_its_existing_startup_semantics(tmp_path, monkeypatch):
    exe = tmp_path / 'XtMiniQmt.exe'
    exe.write_bytes(b'fixture-only-not-executable')
    (tmp_path / '_linkMini').write_bytes(b'legacy-fixture')
    monkeypatch.setattr(qmt, 'QMT_ROOT_DIR', [str(tmp_path)])
    assert Path(qmt.find_valid_qmt_dir()) == exe
    assert (tmp_path / 'linkMini').read_bytes() == b'legacy-fixture'

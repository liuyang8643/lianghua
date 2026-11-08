"""The review checkout must reconstruct exact bytes without remote services."""
import json

import pytest

from offline_data.portable_snapshot import pack, restore


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / "runtime.npz"
    source.write_bytes(bytes(range(256)) * 5)
    sidecar = source.with_suffix(".manifest.json")
    sidecar.write_text('{"provenance": "test"}', encoding="utf-8")
    manifest = pack(source, tmp_path / "parts", chunk_bytes=200, sidecar=sidecar)
    return source, manifest, tmp_path / "restored"


def test_round_trip_and_repeated_restore_preserve_source_and_sidecar(snapshot):
    source, manifest, output = snapshot
    restored = restore(manifest, output)
    assert restored.read_bytes() == source.read_bytes()
    assert restored.with_suffix('.manifest.json').read_bytes() == source.with_suffix('.manifest.json').read_bytes()
    assert restore(manifest, output) == restored
    assert not list(output.glob('*.tmp'))


@pytest.mark.parametrize('failure', ['corrupt', 'missing', 'reordered', 'sidecar'])
def test_invalid_archive_leaves_no_partial_runtime(snapshot, failure):
    source, manifest, output = snapshot
    metadata = json.loads(manifest.read_text())
    part = manifest.parent / metadata['parts'][0]['file']
    if failure == 'corrupt':
        part.write_bytes(b'wrong')
    elif failure == 'missing':
        part.unlink()
    elif failure == 'sidecar':
        (manifest.parent / metadata['sidecars'][0]['file']).write_bytes(b'wrong')
    else:
        metadata['parts'].reverse()
        manifest.write_text(json.dumps(metadata))
    with pytest.raises((ValueError, FileNotFoundError)):
        restore(manifest, output)
    assert not (output / source.name).exists()
    assert not list(output.glob('*.tmp'))


def test_existing_different_runtime_is_never_overwritten(snapshot):
    source, manifest, output = snapshot
    output.mkdir()
    target = output / source.name
    target.write_bytes(b'user data')
    with pytest.raises(ValueError, match='refusing to replace'):
        restore(manifest, output)
    assert target.read_bytes() == b'user data'


@pytest.mark.parametrize('field', ['filename', 'part'])
def test_manifest_cannot_escape_output_or_archive_directory(snapshot, field):
    _, manifest, output = snapshot
    metadata = json.loads(manifest.read_text())
    if field == 'filename':
        metadata['filename'] = '../escape'
    else:
        metadata['parts'][0]['file'] = '../escape'
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='plain filenames'):
        restore(manifest, output)

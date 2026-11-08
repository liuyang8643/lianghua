"""Shared file persistence and byte hashing at application I/O boundaries."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

_replace = os.replace


def replace_file(source, target):
    """Retry only Windows access/sharing errors; never remove the destination."""
    for attempt in range(51):
        try:
            return _replace(source, target)
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 50:
                raise
            time.sleep(0.1)


def atomic_write_json(path, payload, *, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=None, allow_nan=True, trailing_newline=False):
    """Durably write JSON without changing the caller's serialization contract."""
    path = Path(path)
    content = json.dumps(payload, ensure_ascii=ensure_ascii, sort_keys=sort_keys,
                         indent=indent, separators=separators, allow_nan=allow_nan)
    if trailing_newline:
        content += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path):
    """Hash raw file bytes; source newline normalization remains separate."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

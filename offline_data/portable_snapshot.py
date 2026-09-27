"""Pack/restore an unchanged runtime using small, checksummed Git files.

Run directly with Python's standard library; no market-data service is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "snapshots/runtime/manifest.json"
FORMAT = "wbr-portable-runtime-v1"
CHUNK_BYTES = 48 * 1024 * 1024


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_name(value: str) -> str:
    if not value or "/" in value or "\\" in value or ":" in value or value in (".", ".."):
        raise ValueError("snapshot entries must be plain filenames")
    return value


def pack(source: Path, destination: Path, *, chunk_bytes: int = CHUNK_BYTES,
         sidecar: Path | None = None) -> Path:
    """Copy source bytes, preserving the original NPZ and its embedded arrays."""
    if not 0 < chunk_bytes <= CHUNK_BYTES:
        raise ValueError("chunk size must be between 1 byte and 48 MiB")
    destination.mkdir(parents=True, exist_ok=False)
    parts = []
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            name = f"runtime.part{len(parts):03d}"
            (destination / name).write_bytes(block)
            digest.update(block)
            parts.append({"file": name, "bytes": len(block),
                          "sha256": hashlib.sha256(block).hexdigest()})
    if not parts:
        raise ValueError("source snapshot is empty")
    sidecars = []
    if sidecar is not None:
        content = sidecar.read_bytes()
        (destination / sidecar.name).write_bytes(content)
        sidecars.append({"file": sidecar.name, "bytes": len(content),
                         "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"format": FORMAT, "filename": source.name,
                "bytes": sum(part["bytes"] for part in parts),
                "sha256": digest.hexdigest(), "parts": parts, "sidecars": sidecars}
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def restore(manifest_path: Path, destination: Path) -> Path:
    """Validate each part and atomically install a byte-identical local NPZ."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format"] != FORMAT:
        raise ValueError("unsupported portable snapshot format")
    filename = local_name(manifest["filename"])
    parts = manifest["parts"]
    names = [local_name(part["file"]) for part in parts]
    if not names or len(set(names)) != len(names):
        raise ValueError("snapshot parts must be nonempty and unique")
    destination.mkdir(parents=True, exist_ok=True)
    sidecars = []
    for item in manifest["sidecars"]:
        name = local_name(item["file"])
        content = (manifest_path.parent / name).read_bytes()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"snapshot sidecar failed checksum: {name}")
        sidecar_path = destination / name
        if sidecar_path.exists() and sidecar_path.read_bytes() != content:
            raise ValueError(f"existing sidecar differs; refusing to replace {sidecar_path}")
        sidecars.append((sidecar_path, content))
    target = destination / filename
    if target.exists():
        if target.stat().st_size != manifest["bytes"] or digest_file(target) != manifest["sha256"]:
            raise ValueError(f"existing runtime differs; refusing to replace {target}")
        for path, content in sidecars:
            if not path.exists():
                path.write_bytes(content)
        return target
    # The temporary file shares the target filesystem for atomic link publication.
    fd, temporary = tempfile.mkstemp(prefix=filename + ".", suffix=".tmp", dir=destination)
    temporary = Path(temporary)
    try:
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "wb") as output:
            for part, name in zip(parts, names, strict=True):
                source = manifest_path.parent / name
                part_digest = hashlib.sha256()
                part_size = 0
                with source.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        part_digest.update(block)
                        digest.update(block)
                        output.write(block)
                        part_size += len(block)
                if part_size != part["bytes"] or part_digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"snapshot part failed checksum: {name}")
                size += part_size
        if size != manifest["bytes"] or digest.hexdigest() != manifest["sha256"]:
            raise ValueError("restored snapshot failed checksum")
        # No clobber: a concurrent restore must not replace an existing file.
        os.link(temporary, target)
        for path, content in sidecars:
            if not path.exists():
                path.write_bytes(content)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    unpack = commands.add_parser("restore", help="restore the bundled full-history runtime offline")
    unpack.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    unpack.add_argument("--output", type=Path, default=ROOT / "data/runtime")
    export = commands.add_parser("pack", help="maintainer: split one local runtime without changing it")
    export.add_argument("source", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument("--sidecar", type=Path, help="original runtime provenance manifest")
    args = parser.parse_args()
    path = (restore(args.manifest, args.output) if args.command == "restore"
            else pack(args.source, args.destination, sidecar=args.sidecar))
    print(path)


if __name__ == "__main__":
    main()

"""Build an immutable full-axis financial enrichment without loading factor code.

The NPZ contains exactly the original members plus registered float64 financial
panels. Its ZIP comment binds the provenance in <stem>.manifest.json. The
sidecar also seals the final NPZ SHA256; read_financial_snapshot_manifest is
the public verifier for training identity assembly. Building never selects a
split, computes a strategy score, or reads a revised fundamentals fallback.
"""
from __future__ import annotations

from utils.atomic_file import file_sha256
import argparse
from contextlib import ExitStack
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from offline_data.financial_versions import (
    FINANCIAL_PANEL_FIELDS, FINANCIAL_REPLAY_VERSION,
    iter_financial_fields, load_financial_events,
)


FINANCIAL_SNAPSHOT_SCHEMA = "financial-primitives-snapshot-v3-source-state"
PANEL_BUILDER_VERSION = "financial-full-axis-float64-v3-source-state"
_CHUNK_BYTES = 1024 * 1024


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _stream_sha256(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()




def financial_snapshot_manifest_path(npz_path: str | Path) -> Path:
    return Path(npz_path).with_suffix(".manifest.json")


def _member_hashes(archive: zipfile.ZipFile) -> dict[str, str]:
    names = archive.namelist()
    if len(set(names)) != len(names) or any(not name.endswith(".npy") for name in names):
        raise ValueError("runtime must contain unique NPY members only")
    result = {}
    for name in names:
        with archive.open(name) as stream:
            result[name] = _stream_sha256(stream)
    return result


def _read_manifest(npz_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["schema"] != FINANCIAL_SNAPSHOT_SCHEMA:
        raise ValueError("unsupported financial snapshot schema")
    if (manifest["fields"] != list(FINANCIAL_PANEL_FIELDS) or manifest["dtype"] != "float64"
            or manifest["financial_replay_version"] != FINANCIAL_REPLAY_VERSION):
        raise ValueError("financial snapshot primitive schema differs")
    provenance = {k: v for k, v in manifest.items() if k not in ("manifest_sha256", "snapshot_sha256")}
    digest = hashlib.sha256(_json_bytes(provenance)).hexdigest()
    if digest != manifest["manifest_sha256"]:
        raise ValueError("financial snapshot manifest hash mismatch")
    with zipfile.ZipFile(npz_path) as archive:
        if archive.comment != _json_bytes({"schema": FINANCIAL_SNAPSHOT_SCHEMA, "manifest_sha256": digest}):
            raise ValueError("financial snapshot does not bind this manifest")
        expected = {*manifest["base"]["member_sha256"], *(f"{name}.npy" for name in FINANCIAL_PANEL_FIELDS)}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError("financial snapshot members differ from manifest")
    if file_sha256(npz_path) != manifest["snapshot_sha256"]:
        raise ValueError("financial snapshot file hash mismatch")
    return manifest


def read_financial_snapshot_manifest(npz_path: str | Path) -> dict:
    """Verify the sidecar, its NPZ binding, and full file SHA before returning.

    Identity owners may embed this result under `financial_snapshot`. Reading
    needs no original financial archive and never loads price/return arrays.
    Legacy, unenriched snapshots have no such manifest and are not accepted.
    """
    path = Path(npz_path).resolve()
    return _read_manifest(path, financial_snapshot_manifest_path(path))


def build_financial_snapshot(base_npz: str | Path, financial_directory: str | Path,
                             output_path: str | Path, *, expected_base_sha256: str,
                             financial_identity_path: str | Path) -> dict:
    """Append registered causal panels to a hash-pinned base NPZ, refusing overwrite.

    No date/stock subset arguments exist. Original compressed entries are
    copied as bytes, with each uncompressed NPY member hashed before/after.
    Disk-backed panels share one serial event replay; other runtime fields
    are never materialised. The final NPZ is published only after verification.
    """
    base = Path(base_npz).resolve()
    financial = Path(financial_directory).resolve()
    output = Path(output_path).resolve()
    identity_path = Path(financial_identity_path).resolve()
    sidecar = financial_snapshot_manifest_path(output)
    if output.suffix != ".npz":
        raise ValueError("financial snapshot output must end in .npz")
    if output == base or output.exists() or sidecar.exists():
        raise FileExistsError("immutable financial output or manifest already exists")
    if file_sha256(base) != expected_base_sha256:
        raise ValueError("base runtime file hash mismatch")
    identity_bytes = identity_path.read_bytes()
    financial_identity = json.loads(identity_bytes)
    with np.load(base, allow_pickle=False) as arrays:
        dates, codes = arrays["trade_dates"], arrays["stock_codes"]
        if (dates.ndim != 1 or not len(dates) or dates.dtype != np.dtype("datetime64[D]")
                or np.isnat(dates).any() or np.any(dates[1:] <= dates[:-1])):
            raise ValueError("base runtime needs its full sorted daily date axis")
        if (codes.ndim != 1 or not len(codes) or codes.dtype.kind not in ("U", "S")
                or len(set(codes.tolist())) != len(codes) or np.any(codes == "")):
            raise ValueError("base runtime needs its full unique stock axis")
        if set(FINANCIAL_PANEL_FIELDS).intersection(arrays.files):
            raise ValueError("base runtime is already financially enriched")
    with zipfile.ZipFile(base) as archive:
        base_hashes = _member_hashes(archive)
    events, identity = load_financial_events(financial, tuple(codes.astype(str)), expected_identity=financial_identity)
    source_paths = (Path(__file__), Path(__file__).with_name("financial_versions.py"))
    source_hashes = {path.name: file_sha256(path) for path in source_paths}
    provenance = {
        "schema": FINANCIAL_SNAPSHOT_SCHEMA,
        "panel_builder_version": PANEL_BUILDER_VERSION,
        "financial_replay_version": FINANCIAL_REPLAY_VERSION,
        "panel_builder_source_sha256": source_hashes,
        "base": {"path": str(base), "source_sha256": expected_base_sha256, "member_sha256": base_hashes},
        "financial_directory": str(financial),
        "financial_identity": identity,
        "financial_identity_source_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "financial_request_source_sha256": file_sha256(financial / "request.json"),
        "axes": {"n_dates": len(dates), "n_stocks": len(codes), "first_date": str(dates[0]), "last_date": str(dates[-1]),
                 "trade_dates_sha256": base_hashes["trade_dates.npy"], "stock_codes_sha256": base_hashes["stock_codes.npy"]},
        "fields": list(FINANCIAL_PANEL_FIELDS), "dtype": "float64", "decision_lag": 0,
        "raw_source_state": "nine unchanged source YTD/balance values; latest disclosed quarter per table; quarter-of-year and selected-version report/announcement ages; no ratios or ranks",
        "availability": "announcement_date < decision_date; no future backfill",
        "period_alignment": {"profit_ttm_equity": "min(latest Income, latest Balance)",
                             "cash_outflow_profit_yoy": "min(latest Income, latest CashFlow)",
                             "operating_profit_revenue_yoy": "latest Income",
                             "abnormal_gross_profit_operands": "min(latest Income, latest CashFlow, latest Balance); current and prior-year single quarters; ending assets"},
        "missing": "NaN; selected-period gaps and zero YoY denominator remain unavailable",
        "pit_evidence_limit": "announcement-vintage archive and field samples; not a whole-database original-filing certification",
        "legacy_fundamentals": "preserved byte-for-byte; not consumed or substituted by this builder",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Same-volume staging permits atomic, no-clobber publication by hard link.
    with tempfile.TemporaryDirectory(prefix=".financial-snapshot-", dir=output.parent) as temporary:
        staging = Path(temporary)
        staged_npz = staging / output.name
        staged_manifest = staging / sidecar.name
        panel_paths = {name: staging / f"{name}.npy" for name in FINANCIAL_PANEL_FIELDS}
        shape = (len(dates), len(codes))
        with ExitStack() as stack:
            panels = {}
            for name, path in panel_paths.items():
                stream = stack.enter_context(path.open("w+b"))
                np.lib.format.write_array_header_1_0(stream, {"descr": "<f8", "fortran_order": False, "shape": shape})
                offset = stream.tell()
                stream.truncate(offset + len(dates) * len(codes) * 8)
                stream.flush()
                mapping = stack.enter_context(mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE))
                panels[name] = np.ndarray(shape, dtype="<f8", buffer=mapping, offset=offset)
            for row, (_, fields) in enumerate(iter_financial_fields(dates, len(codes), events)):
                for name in FINANCIAL_PANEL_FIELDS:
                    if np.isinf(fields[name]).any():
                        raise ValueError(f"non-finite financial arithmetic in {name} on {dates[row]}")
                    panels[name][row] = fields[name]
            panels.clear()
        shutil.copyfile(base, staged_npz)
        # Detect a source replacement during event replay/copy before appending.
        if file_sha256(staged_npz) != expected_base_sha256:
            raise ValueError("base runtime changed during snapshot build")
        with zipfile.ZipFile(staged_npz, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for name, path in panel_paths.items():
                archive.write(path, arcname=f"{name}.npy")
        with zipfile.ZipFile(staged_npz) as archive:
            enriched_hashes = _member_hashes(archive)
        if any(enriched_hashes[name] != digest for name, digest in base_hashes.items()):
            raise ValueError("original runtime content hashes changed")
        provenance["panel_member_sha256"] = {name: enriched_hashes[f"{name}.npy"] for name in FINANCIAL_PANEL_FIELDS}
        if any(file_sha256(path) != source_hashes[path.name] for path in source_paths):
            raise ValueError("financial panel builder source changed during build")
        manifest_digest = hashlib.sha256(_json_bytes(provenance)).hexdigest()
        with zipfile.ZipFile(staged_npz, "a") as archive:
            archive.comment = _json_bytes({"schema": FINANCIAL_SNAPSHOT_SCHEMA, "manifest_sha256": manifest_digest})
        manifest = {**provenance, "manifest_sha256": manifest_digest, "snapshot_sha256": file_sha256(staged_npz)}
        staged_manifest.write_bytes(_json_bytes(manifest))
        _read_manifest(staged_npz, staged_manifest)
        # Publish the verified sidecar first; consumers never see an NPZ without
        # provenance. Both links fail if another writer claimed either target.
        os.link(staged_manifest, sidecar)
        try:
            os.link(staged_npz, output)
        except BaseException:
            sidecar.unlink()
            raise
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--financial", type=Path, required=True)
    parser.add_argument("--financial-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_financial_snapshot(args.base, args.financial, args.output,
                                       expected_base_sha256=args.base_sha256, financial_identity_path=args.financial_identity)
    print(json.dumps({"path": str(args.output.resolve()), "schema": manifest["schema"],
                      "snapshot_sha256": manifest["snapshot_sha256"], "axes": manifest["axes"]}, indent=2))


if __name__ == "__main__":
    main()

"""Cold deterministic inference from a frozen PPO policy bundle."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ai.bundle import BundleManifest
from ai.rl.policy import RLPolicy
from env.contracts import AccountState
from env.gym_adapter import DEFAULT_UNIVERSE_PREFIXES, stock_universe_mask
from env.observation import ObservationBuilder
from factor import precompute_factors
from offline_data import compute_runtime_lineage, load_runtime_slice


def _validate_runtime_identity(
    current,
    trained: dict[str, object],
    current_lineage: Mapping[str, object] | None = None,
) -> None:
    """Allow only a semantically identical prefix followed by new dates."""

    if current.schema_hash != str(trained["schema_hash"]):
        raise ValueError("runtime schema differs from the trained bundle")
    if current.stock_vocabulary_sha256 != str(
        trained["stock_vocabulary_sha256"]
    ):
        raise ValueError("runtime stock vocabulary/order differs from the trained bundle")
    trained_lineage = trained.get("lineage")
    if not isinstance(trained_lineage, Mapping):
        raise ValueError("trained bundle has no runtime prefix lineage")
    if str(trained_lineage.get("runtime_schema_hash")) != str(
        trained["schema_hash"]
    ):
        raise ValueError("trained runtime lineage conflicts with runtime schema")
    if str(trained_lineage.get("stock_vocabulary_sha256")) != str(
        trained["stock_vocabulary_sha256"]
    ):
        raise ValueError("trained runtime lineage conflicts with stock vocabulary")

    if current.source_sha256 == str(trained["source_sha256"]):
        return
    if current_lineage is None:
        raise ValueError(
            "changed runtime file requires append-only prefix verification"
        )
    comparisons = (
        ("lineage_version", "runtime lineage version"),
        ("runtime_schema_hash", "runtime lineage schema"),
        ("generation_semantics_sha256", "runtime generation semantics"),
        ("prefix_start", "runtime prefix start"),
        ("prefix_end", "runtime prefix end"),
        ("prefix_rows", "runtime prefix row count"),
        ("stock_vocabulary_sha256", "runtime lineage stock vocabulary"),
        ("date_axis_sha256", "runtime historical date axis"),
        ("semantic_sha256", "runtime historical content prefix"),
    )
    for field, label in comparisons:
        if current_lineage.get(field) != trained_lineage.get(field):
            raise ValueError(f"{label} differs from the trained bundle")


def _account(path: str | None, initial_cash: float) -> AccountState:
    if path is None:
        return AccountState(cash=initial_cash, nav=initial_cash, peak_nav=initial_cash)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AccountState(
        cash=float(payload["cash"]),
        positions={str(k): int(v) for k, v in payload["positions"].items()},
        sellable_positions={
            str(k): int(v) for k, v in payload["sellable_positions"].items()
        },
        average_costs={str(k): float(v) for k, v in payload["average_costs"].items()},
        last_prices={str(k): float(v) for k, v in payload["last_prices"].items()},
        nav=float(payload["nav"]),
        peak_nav=float(payload["peak_nav"]),
    )


def infer(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    bundle_dir = Path(args.bundle).resolve()
    manifest = BundleManifest.load(bundle_dir, verify_files=True)
    config_path = None if args.config is None else Path(args.config).resolve()
    policy = RLPolicy.load(bundle_dir, config_path=config_path)
    runtime_path = Path(args.runtime).resolve()
    runtime = load_runtime_slice(
        runtime_path,
        args.date,
        args.date,
        lookback=policy.encoder.observation_schema.lookback,
        expected_stock_codes=None,
    )
    # The local runtime may be append-only after training.  Bind inference to
    # semantics and stock order, not to the bytes of the old full NPZ file.
    trained_runtime = dict(manifest.runtime)
    current_lineage = None
    if runtime.manifest.source_sha256 != str(trained_runtime["source_sha256"]):
        trained_lineage = trained_runtime.get("lineage")
        if not isinstance(trained_lineage, Mapping):
            raise ValueError("trained bundle has no runtime prefix lineage")
        cutoff = trained_lineage.get("prefix_end")
        if not isinstance(cutoff, str) or not cutoff:
            raise ValueError("trained runtime lineage has no prefix cutoff")
        current_lineage = compute_runtime_lineage(
            runtime_path,
            cutoff=cutoff,
        ).as_dict()
    _validate_runtime_identity(
        runtime.manifest,
        trained_runtime,
        current_lineage,
    )
    universe = stock_universe_mask(runtime.stock_codes, DEFAULT_UNIVERSE_PREFIXES)
    factors = precompute_factors(runtime, rank_universe_mask=universe)
    if factors.schema_hash != str(manifest.factors["schema_hash"]):
        raise ValueError("factor schema differs from the trained bundle")
    if factors.rank_universe_sha256 != str(
        manifest.factors["rank_universe_sha256"]
    ):
        raise ValueError("rank universe differs from the trained bundle")
    builder = ObservationBuilder(
        runtime,
        factors,
        lookback=policy.encoder.observation_schema.lookback,
    )
    if builder.schema.identifier != policy.encoder.observation_schema.identifier:
        raise ValueError("runtime observation schema differs from the model bundle")
    observation = builder.build(runtime.decision_start, _account(args.account, args.initial_cash))
    first_action = policy.predict_action(observation, deterministic=True)
    second_action = policy.predict_action(observation, deterministic=True)
    if not np.array_equal(first_action, second_action):
        raise RuntimeError("deterministic inference changed for the same observation")
    config = policy.action_schema.decode(first_action)
    encoded = policy.encoder.encode(observation, normalizer=policy.normalizer)
    result: dict[str, object] = {
        "bundle": str(bundle_dir),
        "decision_date": observation.decision_date,
        "model_sha256": manifest.model_sha256,
        "observation_schema": observation.schema_version,
        "encoded_observation_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
        "raw_action": first_action.astype(float).tolist(),
        "day_config": policy.action_schema.to_static_config(config),
        "same_observation_exact": True,
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--config",
        help="optional external config identity check; defaults to the bundled config",
    )
    parser.add_argument(
        "--runtime",
        default="data/runtime/runtime_1990-12-19_2026-08-27.npz",
    )
    parser.add_argument("--date", required=True)
    parser.add_argument("--account")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--output")
    return parser


def main() -> None:
    result = infer(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

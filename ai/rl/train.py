"""Train and freeze the typed policy with standard Stable-Baselines3 PPO.

The learner sees one candidate account per environment. Resets use either a
random contiguous window or the complete sealed training period. Dense annualized
net-return and maximum-drawdown increments are scaled by H/252, changing
random-window weighting without asserting equal gradients across horizons.
Static configuration backtests are independent external reference benchmarks;
they never enter rollout state, reward, loss, or checkpoint ranking.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping

from configs.training import DEFAULT_EVALUATION_EVERY, DEFAULT_ROLLOUT_WORKERS, read_evaluation_splits
import numpy as np
import torch as th
from offline_data.financial_snapshot import read_financial_snapshot_manifest
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.env_checker import check_env

from ai.bundle import (
    policy_source_sha256,
)
from utils.atomic_file import atomic_write_json, file_sha256
from ai.rl.advantage import ADVANTAGE_BASELINE_VERSION, SynchronizedEnvBaselineRolloutBuffer
from ai.rl.evaluation import (
    BacktestRequest,
    FixedConfigProvider,
    FrozenPPOProvider,
    MAX_PARALLEL_BACKTEST_WORKERS,
    parallel_backtests,
    parallel_backtests_for_episode,
)
from ai.rl.device import PPO_DEVICE, load_cuda_ppo, require_cuda_device, require_cuda_model
from ai.rl.checkpoint import (
    FrozenEvaluationCheckpoint,
    capture_evaluation_checkpoint,
    promote_evaluation_checkpoint,
    EVALUATION_CHECKPOINT_VERSION,
    LATEST_TRAIN_MODEL_FILE,
    save_latest_train_checkpoint,
    validate_latest_train_checkpoint,
)
from ai.rl.rollout import (
    RolloutEnvironment,
    build_rollout_env,
    build_worker_assignments,
    configure_single_thread_runtime,
)
from ai.rl.diagnostics import TrainingDiagnostics
from ai.reporting import mark_training_failed, DIAGNOSTIC_SCHEMA_VERSION, append_ppo_update_diagnostic, write_ppo_report, write_preparing_report
from ai.rl.typed_policy import (
    RAW_PANEL_CONFIG,
    RAW_PANEL_NETWORK_VERSION,
    TYPED_ACTION_DISTRIBUTION_VERSION,
    TypedActorCriticPolicy,
)
from env.action_schema import ActionSchema
from env.backtest import (
    PreparedEpisode,
    RolloutTrace,
    dynamic_config_summary,
    environment_schema_manifest,
    prepare_episode_from_runtime,
)
from env.encoder import TrainOnlyNormalizer
from env.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule
from env.gym_adapter import WBRGymEnv
from env.metrics import REWARD_SCHEMA_VERSION
from env.observation import DEFAULT_LOOKBACK
from env.prefilter import prefilter_n_from_config
from env.shared_episode import (
    SharedPreparedEpisodeDescriptor,
    ResidentPreparedEpisode,
)
from factor import FactorBatch
from offline_data import RuntimeSlice, compute_runtime_lineage


ACTOR_NETWORK = RAW_PANEL_NETWORK_VERSION
ACTOR_NET_ARCH: tuple[int, ...] = ()
CRITIC_NET_ARCH = (64, 32)
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
DEFAULT_LOG_STD_INIT = 0.0
ADVANTAGE_BASELINES = ("none", "synchronized_env_row_mean")
DEFAULT_ADVANTAGE_BASELINE = "none"


def scheduled_learning_rate(initial: float, end_fraction: float, decay_start: float,
                            completed: int, total: int) -> float:
    """Global training-budget schedule, unaffected by per-rollout learn() calls."""
    progress = min(1.0, max(0.0, completed / total))
    phase = max(0.0, (progress - decay_start) / (1.0 - decay_start))
    return initial * (1.0 - (1.0 - end_fraction) * phase)
RUN_IDENTITY_VERSION = "wbr-ppo-run-identity-v88-exploration-scale-advantage-baseline"
EVALUATION_CACHE_PROTOCOL = {"prepared_splits": "lazy_once_shared_readonly_until_exit",
                             "evaluation_execution": "serial", "release": "ExitStack",
                             "replay_storage": "full_history_precompute_then_causal_history_projection"}
EVALUATION_PROTOCOL = {
    "test_usage": "repeated_diagnostic_only",
    "selection": "maximum_validation_calmar",
    "deployment_bundle": False,
    "random_initialization_eligible": False,
    "resume": "identical_contract_canonical_latest_only",
}
EVALUATION_COMPLETION_FILE = "evaluation_completion.json"
RUN_IDENTITY_FILE = "run_identity.json"
LIFECYCLE_CLAIM_VERSION = "wbr-ppo-lifecycle-claim-v3-continuation"
LIFECYCLE_CLAIM_FILE = "lifecycle_claim.json"
CHECKPOINT_SELECTION_OBJECTIVE = "full_validation_calmar"
MODEL_FILE = "model.zip"
INITIAL_MODEL_FILE = "initial_model.zip"
TRAIN_BEST_MODEL_FILE = "train_best_model.zip"
DEFAULT_ROLLOUT_COUNT = 100000
DEFAULT_BATCH_SIZE_TARGET = 640
STATIC_BENCHMARK_VERSION = "wbr-static-benchmark-v3-periodic-evaluation"
STATIC_BENCHMARK_FILE = "static_benchmark.json"


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal_run_identity(payload: Mapping[str, object]) -> dict[str, object]:
    identity = dict(payload)
    identity.pop("identity_sha256", None)
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _validate_run_identity(payload: Mapping[str, object]) -> dict[str, object]:
    identity = dict(payload)
    if identity.get("identity_version") != RUN_IDENTITY_VERSION:
        raise ValueError("unsupported PPO run identity version")
    expected = str(identity.pop("identity_sha256", ""))
    if len(expected) != 64 or _canonical_sha256(identity) != expected:
        raise ValueError("PPO run identity SHA mismatch")
    identity["identity_sha256"] = expected
    return identity


def _load_run_identity(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("PPO run identity must be a JSON object")
    return _validate_run_identity(payload)


def _seal_static_benchmark(
    *,
    contract_sha256: str,
    train: Mapping[str, object],
    validation: Mapping[str, object] | None,
) -> dict[str, object]:
    if len(contract_sha256) != 64:
        raise ValueError("static benchmark contract must be a SHA-256 digest")
    payload: dict[str, object] = {
        "schema_version": STATIC_BENCHMARK_VERSION,
        "contract_sha256": contract_sha256,
        "role": "external_benchmark_not_rollout_loss_or_ranking",
        "train": dict(train),
        "validation": None if validation is None else dict(validation),
    }
    payload["benchmark_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_static_benchmark(
    payload: Mapping[str, object],
    *,
    contract_sha256: str,
) -> dict[str, object]:
    benchmark = dict(payload)
    expected_fields = {
        "schema_version",
        "contract_sha256",
        "role",
        "train",
        "validation",
        "benchmark_sha256",
    }
    if set(benchmark) != expected_fields:
        raise ValueError("static benchmark fields are invalid")
    if benchmark["schema_version"] != STATIC_BENCHMARK_VERSION:
        raise ValueError("unsupported static benchmark schema")
    if benchmark["contract_sha256"] != contract_sha256:
        raise ValueError("static benchmark frozen contract mismatch")
    if benchmark["role"] != "external_benchmark_not_rollout_loss_or_ranking":
        raise ValueError("static benchmark role is invalid")
    expected_sha256 = str(benchmark.pop("benchmark_sha256"))
    if len(expected_sha256) != 64 or _canonical_sha256(benchmark) != expected_sha256:
        raise ValueError("static benchmark SHA mismatch")
    for split in ("train", "validation"):
        summary = benchmark.get(split)
        if split == "validation" and summary is None:
            continue
        if not isinstance(summary, Mapping):
            raise ValueError(f"static {split} benchmark summary is invalid")
        _summary_calmar(summary)
        if summary.get("reward_schema_version") != REWARD_SCHEMA_VERSION:
            raise ValueError(f"static {split} benchmark reward schema mismatch")
    benchmark["benchmark_sha256"] = expected_sha256
    return benchmark


def _load_static_benchmark(
    path: Path,
    *,
    contract_sha256: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("static benchmark cache must be a JSON object")
    return _validate_static_benchmark(payload, contract_sha256=contract_sha256)


def _claim_lifecycle(
    path: Path,
    *,
    mode: str,
    run_identity_sha256: str,
    child_identity_sha256: str | None = None,
) -> dict[str, object]:
    if mode != "continuation":
        raise ValueError("unsupported PPO lifecycle mode")
    if len(run_identity_sha256) != 64:
        raise ValueError("lifecycle identity must be a SHA-256 digest")
    claim: dict[str, object] = {
        "claim_version": LIFECYCLE_CLAIM_VERSION,
        "mode": mode,
        "run_identity_sha256": run_identity_sha256,
        "child_identity_sha256": child_identity_sha256,
    }
    if mode == "continuation" and (
        child_identity_sha256 is None or len(child_identity_sha256) != 64
    ):
        raise ValueError("continuation claim needs a child identity")
    claim["claim_sha256"] = _canonical_sha256(claim)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(claim, ensure_ascii=False, sort_keys=True, indent=2)
            )
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("PPO lineage was already continued") from exc
    return claim


def _validate_resume_arguments(args: argparse.Namespace) -> Path | None:
    checkpoint = None if args.resume_from is None else Path(args.resume_from).resolve()
    if checkpoint is None:
        return None
    if args.learning_rate_end_fraction != 1.0:
        raise ValueError("decaying learning rate requires a new root with a sealed transition horizon")
    if not checkpoint.is_file() or checkpoint.name != LATEST_TRAIN_MODEL_FILE:
        raise ValueError(
            "verified resume requires canonical latest_train_model.zip"
        )
    if Path(args.output).resolve() == checkpoint.parent:
        raise ValueError("resume output must differ from the parent directory")
    return checkpoint


def _validate_warm_start_arguments(args: argparse.Namespace) -> Path | None:
    value = args.warm_start_from
    if value is None:
        return None
    if args.resume_from is not None:
        raise ValueError("warm-start-from and resume-from are mutually exclusive")
    checkpoint = Path(value).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"warm-start checkpoint is missing: {checkpoint}")
    if checkpoint.name != MODEL_FILE:
        raise ValueError("warm-start requires the validation-selected model.zip")
    if not (checkpoint.parent / RUN_IDENTITY_FILE).is_file() or not checkpoint.with_suffix(".json").is_file():
        raise ValueError(
            "warm-start donor is missing identity or selection provenance"
        )
    return checkpoint


def _load_warm_start_provenance(
    checkpoint: Path,
    model: PPO,
) -> dict[str, object]:
    identity_payload = json.loads(
        (checkpoint.parent / RUN_IDENTITY_FILE).read_text(encoding="utf-8")
    )
    if not isinstance(identity_payload, dict):
        raise TypeError("warm-start donor identity must be an object")
    identity = _validate_run_identity(identity_payload)
    identity_sha256 = identity["identity_sha256"]
    if getattr(model, "wbr_run_identity_version", None) != identity.get(
        "identity_version"
    ) or getattr(model, "wbr_run_identity_sha256", None) != identity_sha256:
        raise ValueError("warm-start model does not match donor identity")
    selection = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise TypeError("warm-start donor selection sidecar must be an object")
    required_selection = {
        "schema_version",
        "file",
        "sha256",
        "run_identity_version",
        "run_identity_sha256",
        "timesteps",
        "ppo_updates",
        "role",
        "metrics",
    }
    if set(selection) != required_selection:
        raise ValueError("warm-start donor selection sidecar fields are invalid")
    if (
        selection["schema_version"] != EVALUATION_CHECKPOINT_VERSION
        or selection["file"] != MODEL_FILE
        or selection["role"] != "validation_calmar_selected_checkpoint"
        or selection["run_identity_version"] != identity.get("identity_version")
        or selection["run_identity_sha256"] != identity_sha256
        or selection["sha256"] != file_sha256(checkpoint)
    ):
        raise ValueError("warm-start donor is not the authenticated validation selection")
    metrics = selection["metrics"]
    if not isinstance(metrics, dict) or not math.isfinite(
        float(metrics.get("validation_calmar", math.nan))
    ):
        raise ValueError("warm-start donor has no finite validation Calmar")
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_run_identity_version": identity.get("identity_version"),
        "source_run_identity_sha256": identity_sha256,
        "source_validation_calmar": float(metrics["validation_calmar"]),
        "source_selection_sidecar": str(checkpoint.with_suffix(".json")),
        "role": "validation_selected_policy_weights_only_optimizer_reset",
    }


def _prepare_empty_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("PPO output directory must be empty")


def _validate_split_boundaries(args: argparse.Namespace) -> None:
    train_start = np.datetime64(args.train_start, "D")
    train_end = np.datetime64(args.train_end, "D")
    validation_start = np.datetime64(args.validation_start, "D")
    validation_end = np.datetime64(args.validation_end, "D")
    test_start = np.datetime64(args.test_start, "D")
    test_end = np.datetime64(args.test_end, "D")
    if not (
        train_start <= train_end < validation_start <= validation_end
        < test_start <= test_end
    ):
        raise ValueError("train, validation and test must be ordered and disjoint")


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("strategy config must be a JSON object")
    return payload


def _prepare_split(
    runtime_path: Path,
    start: str,
    end: str,
    *,
    lookback: int,
    prefilter_n: int,
    action_schema: ActionSchema,
) -> tuple[RuntimeSlice, FactorBatch, PreparedEpisode]:
    begin = time.perf_counter()
    episode = prepare_episode_from_runtime(
        runtime_path,
        start,
        end,
        lookback=lookback,
        prefilter_n=prefilter_n,
        action_schema=action_schema,
    )
    print(
        json.dumps(
            {
                "event": "prepared_split",
                "start": start,
                "end": end,
                "observations": episode.observation_count,
                "transitions": episode.transition_count,
                "encoded_dimension": episode.encoder.output_dimension,
                "elapsed_seconds": round(time.perf_counter() - begin, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return episode.runtime, episode.factors, episode


def _fit_train_normalizer(
    episode: PreparedEpisode,
    *,
    initial_cash: float,
) -> TrainOnlyNormalizer:
    return TrainOnlyNormalizer.fit(
        episode.market_store,
        episode.encoder.output_schema,
        dataset_role="train",
        initial_cash=initial_cash,
    )


def _trace_summary(trace: RolloutTrace) -> dict[str, object]:
    summary = trace.as_summary()
    summary["reward_schema_version"] = REWARD_SCHEMA_VERSION
    return summary


def _request(
    task_id: str,
    provider: FrozenPPOProvider | FixedConfigProvider,
    action_schema: ActionSchema,
    normalizer: TrainOnlyNormalizer,
    *,
    initial_cash: float,
) -> BacktestRequest:
    return BacktestRequest(
        task_id=task_id,
        provider=provider,
        action_schema_payload=action_schema.to_dict(),
        normalizer_payload=normalizer.to_dict(),
        initial_cash=initial_cash,
    )


def _backtest_provider(
    episode: PreparedEpisode | SharedPreparedEpisodeDescriptor,
    action_schema: ActionSchema,
    normalizer: TrainOnlyNormalizer,
    *,
    task_id: str,
    provider: FrozenPPOProvider | FixedConfigProvider,
    initial_cash: float,
    workers: int,
) -> RolloutTrace:
    request = _request(
        task_id,
        provider,
        action_schema,
        normalizer,
        initial_cash=initial_cash,
    )
    execute = parallel_backtests if isinstance(episode, SharedPreparedEpisodeDescriptor) else parallel_backtests_for_episode
    return execute(
        episode,
        (request,),
        max_workers=workers,
    )[request.task_id]


def _summary_calmar(summary: Mapping[str, object]) -> float:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("benchmark summary has no metrics")
    try:
        value = float(metrics["calmar"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("benchmark summary has no valid Calmar") from exc
    if not math.isfinite(value):
        raise ValueError("benchmark summary Calmar must be finite")
    return value


def _trace_is_finite(trace: RolloutTrace) -> bool:
    return all(
        math.isfinite(float(value))
        for value in trace.metrics.as_dict().values()
    )


def _evaluate_frozen_on_descriptor(
    snapshot: FrozenEvaluationCheckpoint,
    descriptor: SharedPreparedEpisodeDescriptor,
    *,
    task_id: str,
    action_schema: ActionSchema,
    normalizer: TrainOnlyNormalizer,
    initial_cash: float,
    workers: int,
) -> RolloutTrace:
    snapshot.validate()
    request = _request(
        task_id,
        FrozenPPOProvider(str(snapshot.path.resolve()), snapshot.sha256),
        action_schema,
        normalizer,
        initial_cash=initial_cash,
    )
    return parallel_backtests(
        descriptor,
        (request,),
        max_workers=workers,
    )[request.task_id]


class FrozenEvaluationQueue:
    """One frozen train replay in flight; selection stays on the learner thread."""

    def __init__(self, replay: Callable, consume: Callable):
        self.replay = replay
        self.consume = consume
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frozen-replay")
        self.pending: tuple[FrozenEvaluationCheckpoint, Future] | None = None

    def _timed_replay(self, snapshot: FrozenEvaluationCheckpoint) -> tuple[RolloutTrace, float]:
        started = time.perf_counter()
        trace = self.replay(snapshot)
        return trace, time.perf_counter() - started

    def submit(self, snapshot: FrozenEvaluationCheckpoint) -> None:
        if self.pending is not None:
            raise RuntimeError("consume the previous frozen evaluation before submitting another")
        self.pending = (snapshot, self.executor.submit(self._timed_replay, snapshot))

    def poll(self, *, wait: bool = False) -> bool:
        if self.pending is None:
            return False
        snapshot, future = self.pending
        if not wait and not future.done():
            return False
        trace, replay_elapsed = future.result()  # The process runner enforces its own finite timeout.
        self.consume(snapshot, trace, replay_elapsed)
        snapshot.release()
        self.pending = None
        return True

    def close(self) -> None:
        # A running process-backed task cannot be cancelled via Future.cancel.
        # Join before releasing its shared episode; failed snapshots stay auditable.
        self.executor.shutdown(wait=True, cancel_futures=True)


def _validate_evaluation_completion(parent_dir: Path, identity: Mapping[str, object]) -> None:
    contract = identity.get("contract", {})
    algorithm = contract.get("algorithm", {}) if isinstance(contract, Mapping) else {}
    if algorithm.get("evaluation_execution", "blocking") != "overlap":
        return
    try:
        completion = json.loads((parent_dir / EVALUATION_COMPLETION_FILE).read_text(encoding="utf-8"))
        latest = json.loads((parent_dir / "latest_train_model.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("overlap resume requires an orderly evaluation drain") from exc
    if (completion.get("complete") is not True
            or completion.get("run_identity_sha256") != identity["identity_sha256"]
            or completion.get("latest") != latest
            or latest.get("sha256") != file_sha256(parent_dir / LATEST_TRAIN_MODEL_FILE)):
        raise ValueError("overlap resume has incomplete or mismatched evaluation state")


@dataclass
class TrainingCalmarTracker:
    descriptor: SharedPreparedEpisodeDescriptor
    action_schema: ActionSchema
    normalizer: TrainOnlyNormalizer
    output_dir: Path
    initial_cash: float
    workers: int
    best_train_calmar: float = -math.inf
    best_train_trace: RolloutTrace | None = None
    best_train_timesteps: int | None = None
    best_train_updates: int | None = None

    def evaluate(
        self,
        model: PPO,
        *,
        eligible_for_selection: bool,
    ) -> dict[str, object]:
        evaluation_started = time.perf_counter()
        snapshot = capture_evaluation_checkpoint(model, self.output_dir)
        train_trace = self.replay(snapshot)
        record = self.record(snapshot, train_trace,
                             eligible_for_selection=eligible_for_selection,
                             elapsed_seconds=time.perf_counter() - evaluation_started)
        snapshot.release()
        return record

    def replay(self, snapshot: FrozenEvaluationCheckpoint) -> RolloutTrace:
        return _evaluate_frozen_on_descriptor(
            snapshot,
            self.descriptor,
            task_id="train_candidate",
            action_schema=self.action_schema,
            normalizer=self.normalizer,
            initial_cash=self.initial_cash,
            workers=self.workers,
        )

    def record(self, snapshot: FrozenEvaluationCheckpoint, train_trace: RolloutTrace, *,
               eligible_for_selection: bool, elapsed_seconds: float) -> dict[str, object]:
        train_calmar = float(train_trace.metrics.calmar)
        if not math.isfinite(train_calmar):
            raise RuntimeError("training Calmar is non-finite")

        train_improved = (
            eligible_for_selection and train_calmar > self.best_train_calmar
        )
        if train_improved:
            promote_evaluation_checkpoint(
                snapshot,
                self.output_dir,
                file_name=TRAIN_BEST_MODEL_FILE,
                role="best_complete_training_calmar_diagnostic_only",
                metrics={"train_calmar": train_calmar},
            )
            self.best_train_calmar = train_calmar
            self.best_train_trace = train_trace
            self.best_train_timesteps = snapshot.timesteps
            self.best_train_updates = snapshot.updates
        record = {
            "timesteps": snapshot.timesteps,
            "ppo_updates": snapshot.updates,
            "checkpoint_sha256": snapshot.sha256,
            "run_identity_sha256": snapshot.identity,
            "eligible_for_selection": bool(eligible_for_selection),
            "train_calmar": train_calmar,
            "train_improved": train_improved,
            "best_train_calmar": (
                None
                if not math.isfinite(self.best_train_calmar)
                else self.best_train_calmar
            ),
            "evaluation_elapsed_seconds": elapsed_seconds,
            "train_metrics": train_trace.metrics.as_dict(),
            "dynamic_config": dynamic_config_summary(train_trace),
        }
        print(json.dumps({"event": "training_evaluation", **record}), flush=True)
        return record

    def train_selection(self) -> dict[str, object]:
        if self.best_train_timesteps is None:
            raise RuntimeError("no trained checkpoint was evaluated on training")
        return {
            "objective": "full_training_calmar_diagnostic_only",
            "model_file": TRAIN_BEST_MODEL_FILE,
            "timesteps": self.best_train_timesteps,
            "ppo_updates": self.best_train_updates,
            "train_calmar": self.best_train_calmar,
            "deployment_selection": False,
        }


@dataclass
class ValidationCalmarSelector:
    descriptor: SharedPreparedEpisodeDescriptor
    action_schema: ActionSchema
    normalizer: TrainOnlyNormalizer
    output_dir: Path
    initial_cash: float
    workers: int
    checkpoint_role: str = "validation_calmar_selected_checkpoint"
    best_calmar: float = -math.inf
    best_trace: RolloutTrace | None = None
    best_timesteps: int | None = None
    best_updates: int | None = None

    def evaluate(self, model: PPO) -> dict[str, object]:
        snapshot = capture_evaluation_checkpoint(model, self.output_dir)
        record = self.evaluate_snapshot(snapshot)
        snapshot.release()
        return record

    def evaluate_snapshot(self, snapshot: FrozenEvaluationCheckpoint, *,
                          eligible_for_selection: bool = True) -> dict[str, object]:
        evaluation_started = time.perf_counter()
        trace = _evaluate_frozen_on_descriptor(
            snapshot,
            self.descriptor,
            task_id="validation_candidate",
            action_schema=self.action_schema,
            normalizer=self.normalizer,
            initial_cash=self.initial_cash,
            workers=self.workers,
        )
        calmar = float(trace.metrics.calmar)
        if not math.isfinite(calmar):
            raise RuntimeError("validation Calmar is non-finite")
        improved = eligible_for_selection and calmar > self.best_calmar
        if improved:
            promote_evaluation_checkpoint(
                snapshot,
                self.output_dir,
                file_name=MODEL_FILE,
                role=self.checkpoint_role,
                metrics={"validation_calmar": calmar},
            )
            self.best_calmar = calmar
            self.best_trace = trace
            self.best_timesteps = snapshot.timesteps
            self.best_updates = snapshot.updates
        record = {
            "timesteps": snapshot.timesteps,
            "ppo_updates": snapshot.updates,
            "checkpoint_sha256": snapshot.sha256,
            "run_identity_sha256": snapshot.identity,
            "validation_calmar": calmar,
            "validation_metrics": trace.metrics.as_dict(),
            "evaluation_elapsed_seconds": time.perf_counter() - evaluation_started,
            "eligible_for_selection": eligible_for_selection,
            "improved": improved,
            "best_validation_calmar": self.best_calmar if math.isfinite(self.best_calmar) else None,
        }
        print(json.dumps({"event": "validation_evaluation", **record}), flush=True)
        return record

    def selection(self) -> dict[str, object]:
        if self.best_timesteps is None:
            raise RuntimeError("no validation-selected checkpoint reached validation")
        return {
            "objective": CHECKPOINT_SELECTION_OBJECTIVE,
            "model_file": MODEL_FILE,
            "timesteps": self.best_timesteps,
            "ppo_updates": self.best_updates,
            "validation_calmar": self.best_calmar,
        }


def _build_run_identity(
    *,
    source_sha256: str,
    runtime_sha256: str,
    config_sha256: str,
    runtime_lineage: Mapping[str, object],
    args: argparse.Namespace,
    episode: PreparedEpisode,
    factors: FactorBatch,
    action_schema: ActionSchema,
    normalizer_path: Path,
    rollout_manifest: Mapping[str, object],
    n_steps: int,
    batch_size: int,
    learner_device: str,
    parent_identity_sha256: str | None,
    warm_start: Mapping[str, object] | None,
    financial_snapshot: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract: dict[str, object] = {
        "source_sha256": source_sha256,
        "runtime": {
            "file_sha256": runtime_sha256,
            "lineage": dict(runtime_lineage),
            "financial_snapshot": dict(financial_snapshot) if financial_snapshot is not None else None,
        },
        "config": {
            "file_sha256": config_sha256,
            "static_day_config_role": (
                "external_benchmark_not_initialization_loss_or_ranking"
            ),
            "prefilter_n": episode.prefilter_n,
            "prefilter_role": "fixed_environment_used_by_all_account_chains",
        },
        "splits": {
            "train": [args.train_start, args.train_end],
            "validation": [args.validation_start, args.validation_end],
            "test": [args.test_start, args.test_end],
        },
        "factor_schema_hash": factors.schema_hash,
        "financial_data_protocol": {
            "inputs": "immutable_full_axis_runtime_with_announced_financial_vintages",
            "effective_time": "announcement_date_strictly_before_decision_T",
            "missing": "NaN_distinct_from_valid_binary_rejection_zero",
            "pit_evidence": "quarantined_timestamps_and_finite_original_report_sample_not_exhaustive_certification",
            "holdout_prior_exposure": "2022_2026_already_viewed_in_prior_training_or_research_not_blind",
        },
        "observation_schema": episode.observation_builder.schema.identifier,
        "encoded_schema": episode.encoder.output_schema.identifier,
        "action_schema_hash": action_schema.schema_hash,
        "environment": environment_schema_manifest(
            action_schema,
            episode.encoder,
            prefilter_n=episode.prefilter_n,
        ),
        "normalizer_sha256": file_sha256(normalizer_path),
        "algorithm": {
            "implementation": "stable_baselines3.PPO",
            "evaluation_protocol": "periodic_three_split_validation_selection",
            "policy": "ai.rl.typed_policy.TypedActorCriticPolicy",
            "objective": "horizon_scaled_annualized_return_minus_max_drawdown",
            "checkpoint_selection": CHECKPOINT_SELECTION_OBJECTIVE,
            "gamma": PPO_GAMMA,
            "gae_lambda": PPO_GAE_LAMBDA,
            "ent_coef": 0.0,
            "n_steps": n_steps,
            "collection_scope": "complete_episode" if args.n_steps == 0 else "fixed_transition_count",
            "partial_final_minibatch": args.n_steps == 0,
            "diagnostics": {"version": DIAGNOSTIC_SCHEMA_VERSION, "sample_limit": 128,
                            "detail_every_rollouts": args.eval_every_rollouts, "archive_evaluation_checkpoints": True},
            "batch_size": batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": {"end_fraction": args.learning_rate_end_fraction,
                                       "decay_start": args.learning_rate_decay_start,
                                       "total_transitions": (args.rollouts * n_steps * args.n_envs
                                                             if args.learning_rate_end_fraction != 1.0 else None),
                                       "clock": "global_requested_transition_budget"},
            "target_kl": args.target_kl,
            "complete_train_evaluation_every_rollouts": args.eval_every_rollouts,
            "evaluation_execution": args.evaluation_execution,
            "maximum_pending_evaluations": 1,
            "learner_device": learner_device,
            "model_execution": "cuda_only_training_sampling_frozen_and_live_inference",
            "actor_network": ACTOR_NETWORK,
            "actor_net": list(ACTOR_NET_ARCH),
            "raw_panel_network": dict(RAW_PANEL_CONFIG),
            "market_transport": "sealed_raw_store_reference_resolved_before_learnable_layers",
            "critic_net": list(CRITIC_NET_ARCH),
            "typed_distribution": TYPED_ACTION_DISTRIBUTION_VERSION,
            "weight_mode": "(clip(gaussian_mean,-1,1)+1)/2",
            "state_head": "SB3 Linear Gaussian mean head; orthogonal gain 0.01, zero bias",
            "exploration": "SB3 DiagGaussianDistribution; trainable state-independent log_std",
            "log_std_init": args.log_std_init,
            "advantage_baseline": {
                "mode": args.advantage_baseline,
                "version": ADVANTAGE_BASELINE_VERSION if args.advantage_baseline != "none" else None,
                "scope": ("after_gae_before_minibatch_normalization; returns and clipped surrogate unchanged"
                          if args.advantage_baseline != "none" else "sb3_default"),
            },
            "action_execution": "SB3 clips samples to Box; rollout buffer retains raw Gaussian samples and likelihoods",
            "training_execution_fees": {
                "commission_rate": DEFAULT_FEE_SCHEDULE.commission_rate,
                "minimum_commission": DEFAULT_FEE_SCHEDULE.minimum_commission,
                "stamp_tax_rate": DEFAULT_FEE_SCHEDULE.stamp_tax_rate,
                "transfer_fee_rate": DEFAULT_FEE_SCHEDULE.transfer_fee_rate,
                "slippage_rate": args.training_slippage_rate,
            },
            "complete_period_evaluation_fees": {
                "commission_rate": DEFAULT_FEE_SCHEDULE.commission_rate,
                "minimum_commission": DEFAULT_FEE_SCHEDULE.minimum_commission,
                "stamp_tax_rate": DEFAULT_FEE_SCHEDULE.stamp_tax_rate,
                "transfer_fee_rate": DEFAULT_FEE_SCHEDULE.transfer_fee_rate,
                "slippage_rate": args.evaluation_slippage_rate,
            },
        },
        "rollout": dict(rollout_manifest),
    }
    contract["evaluation_cache"] = dict(EVALUATION_CACHE_PROTOCOL)
    contract["evaluation_protocol"] = {
        **EVALUATION_PROTOCOL, "evaluation_every_rollouts": args.eval_every_rollouts,
        "initial_evaluation": True, "final_evaluation": True,
        "numerical_runtime": {"torch_version": str(th.__version__),
            "torch_cuda_build": th.version.cuda, "stable_baselines3_version": package_version("stable-baselines3")},
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return _seal_run_identity(
        {
            "identity_version": RUN_IDENTITY_VERSION,
            "contract": contract,
            "lineage": {
                "mode": (
                    "warm_start_root"
                    if warm_start is not None
                    else ("root" if parent_identity_sha256 is None else "resume")
                ),
                "parent_identity_sha256": parent_identity_sha256,
                "warm_start": None if warm_start is None else dict(warm_start),
            },
        }
    )


def _assert_verified_resume_compatible(
    parent: Mapping[str, object],
    child: Mapping[str, object],
) -> None:
    validated_parent = _validate_run_identity(parent)
    validated_child = _validate_run_identity(child)
    parent_contract = validated_parent.get("contract")
    child_contract = validated_child.get("contract")
    if not isinstance(parent_contract, Mapping) or not isinstance(
        child_contract, Mapping
    ):
        raise ValueError("resume identity has no frozen contract")
    if parent_contract != child_contract:
        raise ValueError("verified resume changed the frozen PPO contract")
    child_lineage = validated_child.get("lineage")
    if not isinstance(child_lineage, Mapping) or child_lineage.get(
        "parent_identity_sha256"
    ) != validated_parent["identity_sha256"]:
        raise ValueError("verified resume parent identity mismatch")


def _bind_model_identity(model: PPO, identity: Mapping[str, object]) -> None:
    model.wbr_run_identity_version = str(identity["identity_version"])
    model.wbr_run_identity_sha256 = str(identity["identity_sha256"])


def _assert_model_identity(
    model: PPO,
    identity: Mapping[str, object],
    *,
    label: str,
) -> None:
    if (
        getattr(model, "wbr_run_identity_version", None) != identity["identity_version"]
        or getattr(model, "wbr_run_identity_sha256", None)
        != identity["identity_sha256"]
    ):
        raise ValueError(f"{label} embedded identity mismatch")


def _training_timesteps_from_rollouts(
    rollout_count: int,
    rollout_size: int,
) -> int:
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if rollout_size <= 0:
        raise ValueError("rollout_size must be positive")
    return rollout_count * rollout_size


def _resolve_batch_size(requested: int, rollout_size: int, *, allow_partial: bool = False) -> int:
    batch_size = requested
    if allow_partial:
        batch_size = min(DEFAULT_BATCH_SIZE_TARGET, rollout_size) if requested == 0 else requested
        if batch_size <= 1 or batch_size > rollout_size or rollout_size % batch_size == 1:
            raise ValueError("partial batch requires at least two samples in every minibatch")
        return batch_size
    if requested == 0:
        batch_size = next(
            (size for size in range(min(DEFAULT_BATCH_SIZE_TARGET, rollout_size), 1, -1)
             if rollout_size % size == 0),
            rollout_size,
        )
    if batch_size <= 1 or rollout_size % batch_size != 0:
        raise ValueError("batch_size must be greater than one and divide the vector rollout")
    return batch_size


def train(args: argparse.Namespace) -> Path:
    report_path = Path(args.output) / "training_report.json"
    report_existed = report_path.exists()
    try:
        with ExitStack() as resources:
            return _train(args, resources)
    except BaseException as error:
        if not report_existed and report_path.exists():
            mark_training_failed(Path(args.output), error)
        raise


def _train(args: argparse.Namespace, resources: ExitStack) -> Path:
    configure_single_thread_runtime()
    learner_device = str(require_cuda_device(args.device))
    _validate_split_boundaries(args)
    resume_checkpoint = _validate_resume_arguments(args)
    warm_start_checkpoint = _validate_warm_start_arguments(args)
    training_fees = FeeSchedule(slippage_rate=args.training_slippage_rate)
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve()
    runtime_path = Path(args.runtime).resolve()
    output_dir = Path(args.output).resolve()
    _prepare_empty_output_directory(output_dir)
    write_preparing_report(output_dir, algorithm="PPO", total=args.rollouts,
                           splits={name: [getattr(args, f"{name}_start"), getattr(args, f"{name}_end")]
                                   for name in ("train", "validation", "test")})

    source_sha256 = policy_source_sha256(repo_root)
    financial_manifest = read_financial_snapshot_manifest(runtime_path)
    runtime_sha256 = financial_manifest["snapshot_sha256"]
    financial_snapshot_identity = {
        "manifest_sha256": financial_manifest["manifest_sha256"],
        "snapshot_sha256": financial_manifest["snapshot_sha256"],
        "financial_identity_sha256": financial_manifest["financial_identity"]["sha256"],
        "panel_builder_version": financial_manifest["panel_builder_version"],
        "financial_replay_version": financial_manifest["financial_replay_version"],
        "availability": financial_manifest["availability"],
        "pit_evidence_limit": financial_manifest["pit_evidence_limit"],
    }
    config_sha256 = file_sha256(config_path)
    config_payload = _load_config(config_path)
    prefilter_n = prefilter_n_from_config(config_payload)
    runtime_lineage = compute_runtime_lineage(runtime_path)
    strategy_payload = config_payload.get("individual_config", config_payload)
    if not isinstance(strategy_payload, Mapping):
        raise TypeError("strategy config payload must be a mapping")
    fixed_filters = strategy_payload.get("filter_factors", {})
    if not isinstance(fixed_filters, Mapping):
        raise TypeError("filter_factors must be a mapping")
    action_schema = ActionSchema(
        fixed_filter_flags=tuple(
            bool(fixed_filters.get(name, True))
            for name in ActionSchema().filter_names
        ),
        fixed_limit_up_protection=bool(
            strategy_payload.get("limit_up_protection", True)
        ),
        fixed_rebalance_band_pct=float(
            strategy_payload.get("rebalance_band_pct", 0.01)
        ),
    )
    train_runtime, train_factors, train_episode = _prepare_split(
        runtime_path,
        args.train_start,
        args.train_end,
        lookback=args.lookback,
        prefilter_n=prefilter_n,
        action_schema=action_schema,
    )
    atomic_write_json(output_dir / "factor_coverage_train.json", train_episode.factor_coverage)
    static_config = action_schema.from_static_config(config_payload)

    parent_identity: dict[str, object] | None = None
    parent_model: PPO | None = None
    parent_train_best_model: PPO | None = None
    parent_selected_model: PPO | None = None
    parent_dir: Path | None = None
    warm_start_model: PPO | None = None
    warm_start_manifest: dict[str, object] | None = None
    if resume_checkpoint is not None:
        parent_dir = resume_checkpoint.parent
        parent_identity = _load_run_identity(parent_dir / RUN_IDENTITY_FILE)
        _validate_evaluation_completion(parent_dir, parent_identity)
        validate_latest_train_checkpoint(
            resume_checkpoint,
            run_identity_version=parent_identity["identity_version"],
            run_identity_sha256=str(parent_identity["identity_sha256"]),
        )
        if (parent_dir / LIFECYCLE_CLAIM_FILE).exists():
            raise ValueError("parent PPO lineage was already continued")
        parent_model = load_cuda_ppo(resume_checkpoint, device=learner_device)
        _assert_model_identity(
            parent_model,
            parent_identity,
            label="resume checkpoint",
        )
        parent_train_best_path = parent_dir / TRAIN_BEST_MODEL_FILE
        if parent_train_best_path.is_file():
            parent_train_best_model = load_cuda_ppo(parent_train_best_path, device=learner_device)
            _assert_model_identity(
                parent_train_best_model,
                parent_identity,
                label="parent training-best checkpoint",
            )
        parent_selected_path = parent_dir / MODEL_FILE
        if parent_selected_path.is_file():
            parent_selected_model = load_cuda_ppo(parent_selected_path, device=learner_device)
            _assert_model_identity(
                parent_selected_model,
                parent_identity,
                label="parent selected checkpoint",
            )
    elif warm_start_checkpoint is not None:
        warm_start_model = load_cuda_ppo(warm_start_checkpoint, device=learner_device)
        donor_schema = getattr(warm_start_model.policy, "typed_action_schema", None)
        if donor_schema is None:
            raise ValueError("donor checkpoint has no typed ActionSchema")
        if donor_schema.schema_hash != action_schema.schema_hash:
            raise ValueError("warm-start checkpoint ActionSchema differs")
        if int(getattr(warm_start_model.policy, "actor_observation_dim", -1)) != (
            train_episode.encoder.output_dimension
        ):
            raise ValueError("warm-start actor observation dimension differs")
        warm_start_manifest = _load_warm_start_provenance(warm_start_checkpoint, warm_start_model)

    bundled_config = output_dir / "strategy_config.json"
    shutil.copyfile(config_path, bundled_config)
    normalizer_path = output_dir / "normalizer.json"
    if parent_dir is None:
        normalizer = _fit_train_normalizer(
            train_episode,
            initial_cash=args.initial_cash,
        )
        normalizer.save(normalizer_path)
    else:
        parent_normalizer = parent_dir / "normalizer.json"
        normalizer = TrainOnlyNormalizer.load(
            parent_normalizer,
            expected_schema=train_episode.encoder.output_schema,
        )
        shutil.copyfile(parent_normalizer, normalizer_path)

    probe = WBRGymEnv(
        train_episode,
        action_schema=action_schema,
        normalizer=normalizer,
        initial_cash=args.initial_cash,
        include_critic_context=True,
        fees=training_fees,
    )
    check_env(probe, warn=True)
    probe.close()
    del probe

    worker_seed = (int(parent_identity["contract"]["rollout"]["assignments"][0]["seed"])
                   if parent_identity is not None else
                   int(np.random.SeedSequence().generate_state(1)[0]) if args.seed is None else args.seed)
    assignments = build_worker_assignments(n_envs=args.n_envs, base_seed=worker_seed)
    rollout_environment: RolloutEnvironment = build_rollout_env(
        train_episode,
        action_schema,
        normalizer,
        initial_cash=args.initial_cash,
        assignments=assignments,
        backend=args.rollout_backend,
        random_window_min_transitions=(None if args.episode_scope == "full" else args.min_episode_transitions),
        fees=training_fees,
    )
    resources.callback(rollout_environment.close)
    n_steps = train_episode.transition_count if args.n_steps == 0 else args.n_steps
    rollout_size = n_steps * args.n_envs
    batch_size = _resolve_batch_size(args.batch_size, rollout_size, allow_partial=args.n_steps == 0)
    rollout_count = int(args.rollouts)
    actual_timesteps = _training_timesteps_from_rollouts(
        rollout_count,
        rollout_size,
    )

    run_identity = _build_run_identity(
        source_sha256=source_sha256,
        runtime_sha256=runtime_sha256,
        config_sha256=config_sha256,
        runtime_lineage=runtime_lineage.as_dict(),
        args=args,
        episode=train_episode,
        factors=train_factors,
        action_schema=action_schema,
        normalizer_path=normalizer_path,
        rollout_manifest=rollout_environment.manifest(),
        n_steps=n_steps,
        batch_size=batch_size,
        learner_device=learner_device,
        parent_identity_sha256=(
            None if parent_identity is None else str(parent_identity["identity_sha256"])
        ),
        warm_start=warm_start_manifest,
        financial_snapshot=financial_snapshot_identity,
    )
    inherited_static_benchmark: dict[str, object] | None = None
    if parent_identity is not None:
        _assert_verified_resume_compatible(parent_identity, run_identity)
        contract = run_identity.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError("run identity has no frozen contract")
        inherited_static_benchmark = _load_static_benchmark(
            parent_dir / STATIC_BENCHMARK_FILE,  # type: ignore[operator]
            contract_sha256=str(parent_identity["contract"]["contract_sha256"]),
        )
        _claim_lifecycle(
            parent_dir / LIFECYCLE_CLAIM_FILE,  # type: ignore[operator]
            mode="continuation",
            run_identity_sha256=str(parent_identity["identity_sha256"]),
            child_identity_sha256=str(run_identity["identity_sha256"]),
        )
    atomic_write_json(output_dir / RUN_IDENTITY_FILE, run_identity)
    if args.evaluation_execution == "overlap":
        atomic_write_json(output_dir / EVALUATION_COMPLETION_FILE, {
            "complete": False, "run_identity_sha256": run_identity["identity_sha256"],
        })
    if inherited_static_benchmark is None:
        fixed_provider = FixedConfigProvider(action_schema.to_static_config(static_config))
        static_train_trace = _backtest_provider(
            rollout_environment.borrowed_shared_descriptor or train_episode,
            action_schema,
            normalizer,
            task_id="static_train",
            provider=fixed_provider,
            initial_cash=args.initial_cash,
            workers=args.backtest_workers,
        )
        contract = run_identity.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError("run identity has no frozen contract")
        static_benchmark = _seal_static_benchmark(
            contract_sha256=str(contract["contract_sha256"]),
            train=_trace_summary(static_train_trace),
            validation=None,
        )
    else:
        static_benchmark = inherited_static_benchmark
    atomic_write_json(output_dir / STATIC_BENCHMARK_FILE, static_benchmark)
    static_train = static_benchmark["train"]
    if not isinstance(static_train, Mapping):
        raise RuntimeError("validated static training benchmark is missing")

    policy_kwargs = {
        "net_arch": {"pi": list(ACTOR_NET_ARCH), "vf": list(CRITIC_NET_ARCH)},
        "raw_panel_config": dict(RAW_PANEL_CONFIG),
        "action_schema": action_schema.to_dict(),
        "encoded_schema": train_episode.encoder.output_schema.to_dict(),
        "log_std_init": args.log_std_init,
    }
    rollout_buffer_class = (SynchronizedEnvBaselineRolloutBuffer
                            if args.advantage_baseline == "synchronized_env_row_mean" else None)
    if parent_model is None:
        model = PPO(
            TypedActorCriticPolicy,
            rollout_environment.vec_env,
            learning_rate=args.learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            ent_coef=0.0,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
            rollout_buffer_class=rollout_buffer_class,
            verbose=args.verbose,
            seed=args.seed,
            device=learner_device,
        )
        if warm_start_model is not None:
            model.policy.load_state_dict(warm_start_model.policy.state_dict(), strict=True)

    else:
        model = parent_model
        if type(model.rollout_buffer) is not (rollout_buffer_class or RolloutBuffer):
            raise ValueError("resumed checkpoint uses a different advantage baseline than requested")
        model.set_env(rollout_environment.vec_env, force_reset=True)
        model.learning_rate = args.learning_rate
        model.n_steps = n_steps
        model.batch_size = batch_size
        model.n_epochs = args.n_epochs
        model.gamma = PPO_GAMMA
        model.gae_lambda = PPO_GAE_LAMBDA
        model.ent_coef = 0.0
        model.target_kl = args.target_kl
    require_cuda_model(model)
    _bind_model_identity(model, run_identity)

    train_evaluation_descriptor = rollout_environment.borrowed_shared_descriptor
    if train_evaluation_descriptor is None:
        train_evaluation_owner = ResidentPreparedEpisode(train_episode)
        resources.callback(train_evaluation_owner.close)
        train_evaluation_descriptor = train_evaluation_owner.descriptor
    if rollout_environment.borrowed_shared_descriptor is not None:
        attached_train = resources.enter_context(train_evaluation_descriptor.attach())
        train_episode = attached_train.episode
        train_runtime, train_factors = train_episode.runtime, train_episode.factors
    model.policy.bind_market_store(train_episode.market_store, normalizer)
    train_tracker = TrainingCalmarTracker(
        descriptor=train_evaluation_descriptor,
        action_schema=action_schema,
        normalizer=normalizer,
        output_dir=output_dir,
        initial_cash=args.initial_cash,
        workers=args.backtest_workers,
    )
    training_curve: list[dict[str, object]] = []
    iteration_timings: list[dict[str, object]] = []
    validation_curve: list[dict[str, object]] = []
    test_curve: list[dict[str, object]] = []
    static_test: Mapping[str, object] | None = None
    validation_owner: ResidentPreparedEpisode | None = None
    test_owner: ResidentPreparedEpisode | None = None
    validation_selector: ValidationCalmarSelector | None = None
    static_validation: Mapping[str, object] | None = None
    initial_evaluation: dict[str, object] | None = None
    resident_splits = {"train": train_evaluation_descriptor.shared_memory_bytes}
    def prepare_validation() -> None:
        nonlocal validation_selector, static_validation, validation_owner
        if validation_selector is not None:
            return
        _, validation_factors, prepared_validation = _prepare_split(
            runtime_path,
            args.validation_start,
            args.validation_end,
            lookback=args.lookback,
            prefilter_n=prefilter_n,
            action_schema=action_schema,
        )
        atomic_write_json(output_dir / "factor_coverage_validation.json", prepared_validation.factor_coverage)
        if validation_factors.schema_hash != train_factors.schema_hash:
            raise ValueError("validation factor schema differs from training")
        validation_owner = resources.enter_context(ResidentPreparedEpisode(prepared_validation))
        del _, validation_factors, prepared_validation
        resident_splits["validation"] = validation_owner.descriptor.shared_memory_bytes
        cached_validation = static_benchmark.get("validation")
        if isinstance(cached_validation, Mapping):
            static_validation = cached_validation
        else:
            trace = _backtest_provider(
                validation_owner.descriptor,
                action_schema,
                normalizer,
                task_id="static_validation",
                provider=FixedConfigProvider(action_schema.to_static_config(static_config)),
                initial_cash=args.initial_cash,
                workers=args.backtest_workers,
            )
            static_validation = _trace_summary(trace)
            contract = run_identity.get("contract")
            if not isinstance(contract, Mapping):
                raise ValueError("run identity has no frozen contract")
            updated_benchmark = _seal_static_benchmark(
                contract_sha256=str(contract["contract_sha256"]),
                train=static_train,
                validation=static_validation,
            )
            static_benchmark.clear()
            static_benchmark.update(updated_benchmark)
            atomic_write_json(output_dir / STATIC_BENCHMARK_FILE, static_benchmark)

        validation_selector = ValidationCalmarSelector(
            descriptor=validation_owner.descriptor, action_schema=action_schema,
            normalizer=normalizer, output_dir=output_dir, initial_cash=args.initial_cash,
            workers=args.backtest_workers,
            checkpoint_role="validation_calmar_selected_checkpoint")

    def publish_evaluation_curves(*, pending: Mapping[str, object] | None = None) -> None:
        atomic_write_json(output_dir / "evaluation_curves.json", {
            "run_identity_sha256": run_identity["identity_sha256"],
            "protocol": run_identity["contract"]["evaluation_protocol"],
            "train": training_curve, "validation": validation_curve, "test": test_curve,
            "baselines": {"train": static_train, "validation": static_validation, "test": static_test},
            "resident_shared_memory_bytes": dict(resident_splits),
            "pending": pending,
        })

    def evaluate_test(snapshot: FrozenEvaluationCheckpoint, phase: str) -> dict[str, object]:
        nonlocal static_test, test_owner
        publish_evaluation_curves(pending={"split": "test", "checkpoint_sha256": snapshot.sha256})
        if test_owner is None:
            _, test_factors, test_episode = _prepare_split(
                runtime_path, args.test_start, args.test_end, lookback=args.lookback, prefilter_n=prefilter_n,
                action_schema=action_schema)
            if test_factors.schema_hash != train_factors.schema_hash:
                raise ValueError("diagnostic test factor schema differs from training")
            atomic_write_json(output_dir / "factor_coverage_test.json", test_episode.factor_coverage)
            test_owner = resources.enter_context(ResidentPreparedEpisode(test_episode))
            del _, test_factors, test_episode
            resident_splits["test"] = test_owner.descriptor.shared_memory_bytes
        if static_test is None:
            static_test = _trace_summary(_backtest_provider(
                test_owner.descriptor, action_schema, normalizer, task_id="static_diagnostic_test",
                provider=FixedConfigProvider(action_schema.to_static_config(static_config)),
                initial_cash=args.initial_cash, workers=args.backtest_workers))
        started = time.perf_counter()
        test_trace = _evaluate_frozen_on_descriptor(
            snapshot, test_owner.descriptor, task_id="diagnostic_test", action_schema=action_schema,
            normalizer=normalizer, initial_cash=args.initial_cash, workers=args.backtest_workers)
        if not test_trace.full_investment_contract_satisfied or not _trace_is_finite(test_trace):
            raise RuntimeError("diagnostic test violated finite/full-investment contract")
        record = {
            "timesteps": snapshot.timesteps, "ppo_updates": snapshot.updates,
            "checkpoint_sha256": snapshot.sha256, "run_identity_sha256": snapshot.identity,
            "test_calmar": float(test_trace.metrics.calmar), "test_metrics": test_trace.metrics.as_dict(),
            "eligible_for_selection": False, "role": "diagnostic_only_not_blind_test", "phase": phase,
            "evaluation_elapsed_seconds": time.perf_counter() - started,
        }
        print(json.dumps({"event": "test_evaluation", **record}), flush=True)
        return record

    def publish_report(*, complete: bool = False) -> None:
        write_ppo_report(output_dir, total_rollouts=rollout_count,
                         timesteps=int(model.num_timesteps),
                         curves={"train": training_curve, "validation": validation_curve, "test": test_curve},
                         baselines={"train": static_train, "validation": static_validation, "test": static_test},
                         complete=complete)

    def consume_training_candidate(snapshot: FrozenEvaluationCheckpoint, trace: RolloutTrace,
                                   elapsed: float, *, phase: str = "post_update",
                                   eligible_for_selection: bool = True) -> None:
        record = train_tracker.record(snapshot, trace, eligible_for_selection=eligible_for_selection,
                                      elapsed_seconds=elapsed)
        record["phase"] = phase
        training_curve.append(record)
        publish_evaluation_curves(pending={"split": "validation", "checkpoint_sha256": snapshot.sha256})
        prepare_validation()
        assert validation_selector is not None
        validation_record = validation_selector.evaluate_snapshot(
            snapshot, eligible_for_selection=eligible_for_selection)
        validation_record["phase"] = phase
        validation_curve.append(validation_record)
        test_curve.append(evaluate_test(snapshot, phase))
        publish_evaluation_curves()
        publish_report()

    def evaluate_training_candidate(candidate: PPO, *, phase: str,
                                    eligible_for_selection: bool = True) -> None:
        started = time.perf_counter()
        snapshot = capture_evaluation_checkpoint(candidate, output_dir)
        trace = train_tracker.replay(snapshot)
        consume_training_candidate(snapshot, trace, time.perf_counter() - started, phase=phase,
                                   eligible_for_selection=eligible_for_selection)
        if phase in ("pre_update_random_initialization", "pre_update_warm_start"):
            promote_evaluation_checkpoint(
                snapshot, output_dir, file_name=INITIAL_MODEL_FILE,
                role=phase,
                metrics={"train_calmar": float(trace.metrics.calmar)},
            )
        snapshot.release()

    if parent_dir is None:
        evaluate_training_candidate(model, phase=("pre_update_warm_start" if warm_start_model is not None
                                                 else "pre_update_random_initialization"), eligible_for_selection=False)
        initial_evaluation = training_curve[-1]
        initial_evaluation["phase"] = (
            "pre_update_warm_start"
            if warm_start_model is not None
            else "pre_update_random_initialization"
        )
        save_latest_train_checkpoint(model, output_dir,
            run_identity_sha256=str(run_identity["identity_sha256"]),
            timesteps=int(model.num_timesteps), ppo_updates=int(model._n_updates),
            phase="post_initial_evaluation")

    if parent_dir is not None:
        if parent_selected_model is not None:
            _bind_model_identity(parent_selected_model, run_identity)
            evaluate_training_candidate(parent_selected_model, phase="resumed_parent_validation_best")
        if parent_train_best_model is not None:
            _bind_model_identity(parent_train_best_model, run_identity)
            evaluate_training_candidate(parent_train_best_model, phase="resumed_parent_train_best")
        if train_tracker.best_train_timesteps != int(model.num_timesteps):
            evaluate_training_candidate(model, phase="resumed_parent_latest")
    evaluation_queue = None
    if args.evaluation_execution == "overlap":
        evaluation_queue = FrozenEvaluationQueue(train_tracker.replay, consume_training_candidate)
        resources.callback(evaluation_queue.close)
    publish_evaluation_curves()
    publish_report()
    schedule_total = (run_identity["contract"]["algorithm"]["learning_rate_schedule"]["total_transitions"]
                      if args.learning_rate_end_fraction != 1.0 else rollout_count * rollout_size)
    for rollout_index in range(rollout_count):
        if evaluation_queue is not None:
            evaluation_queue.poll()
        before = int(model.num_timesteps)
        model.lr_schedule = FloatSchedule(scheduled_learning_rate(
            args.learning_rate, args.learning_rate_end_fraction, args.learning_rate_decay_start,
            before + rollout_size, schedule_total))
        learn_started = time.perf_counter()
        model.learn(
            total_timesteps=rollout_size,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=TrainingDiagnostics(
                output_dir, detailed=(rollout_index == 0 or (rollout_index + 1) % args.eval_every_rollouts == 0),
            ),
        )
        diagnostics = {
            "event": "ppo_update_diagnostics",
            "timesteps": int(model.num_timesteps),
            "train": {key: float(value) for key, value in model.logger.name_to_value.items() if key.startswith("train/") and np.isscalar(value)},
            "rollout_quantiles": {
                name: np.quantile(np.asarray(getattr(model.rollout_buffer, name)), [0.0, 0.01, 0.5, 0.99, 1.0]).tolist()
                for name in ("rewards", "advantages", "returns", "values")
            },
        }
        append_ppo_update_diagnostic(output_dir, diagnostics, rollout_size)
        print(json.dumps(diagnostics), flush=True)
        learn_elapsed = time.perf_counter() - learn_started
        if int(model.num_timesteps) - before != rollout_size:
            raise RuntimeError("SB3 did not consume exactly one complete rollout")
        _bind_model_identity(model, run_identity)
        checkpoint_started = time.perf_counter()
        save_latest_train_checkpoint(
            model,
            output_dir,
            run_identity_sha256=str(run_identity["identity_sha256"]),
            timesteps=int(model.num_timesteps),
            ppo_updates=int(model._n_updates),
            phase="post_update",
        )
        checkpoint_elapsed = time.perf_counter() - checkpoint_started
        timing_record = {
            "rollout": rollout_index + 1,
            "timesteps": int(model.num_timesteps),
            "learn_elapsed_seconds": learn_elapsed,
            "checkpoint_elapsed_seconds": checkpoint_elapsed,
        }
        iteration_timings.append(timing_record)
        publish_report()
        print(json.dumps({"event": "training_iteration_timing", **timing_record}), flush=True)
        if (
            (rollout_index + 1) % args.eval_every_rollouts == 0
            or rollout_index + 1 == rollout_count
        ):
            if evaluation_queue is None:
                evaluate_training_candidate(model, phase="post_update")
            else:
                evaluation_queue.poll(wait=True)
                evaluation_queue.submit(capture_evaluation_checkpoint(model, output_dir))

    if evaluation_queue is not None:
        evaluation_queue.poll(wait=True)
        latest = save_latest_train_checkpoint(
            model, output_dir, run_identity_sha256=str(run_identity["identity_sha256"]),
            timesteps=int(model.num_timesteps), ppo_updates=int(model._n_updates),
            phase="post_evaluation_drain",
        )
        atomic_write_json(output_dir / EVALUATION_COMPLETION_FILE, {
            "complete": True, "run_identity_sha256": run_identity["identity_sha256"],
            "latest": latest,
        })

    publish_report(complete=True)
    train_selection = train_tracker.train_selection()
    def _save_output_schemas() -> None:
        atomic_write_json(output_dir / "action_schema.json", action_schema.to_dict())
        atomic_write_json(
            output_dir / "observation_schema.json",
            train_episode.observation_builder.schema.to_dict(),
        )
        atomic_write_json(
            output_dir / "encoded_schema.json",
            train_episode.encoder.output_schema.to_dict(),
        )

    def _verify_training_inputs() -> None:
        if policy_source_sha256(repo_root) != source_sha256:
            raise RuntimeError("policy source changed while PPO training was running")
        if file_sha256(runtime_path) != runtime_sha256:
            raise RuntimeError("runtime changed while PPO training was running")
        if file_sha256(config_path) != config_sha256:
            raise RuntimeError("static benchmark config changed during training")

    training_metadata = {
        'objective': 'horizon_scaled_annualized_return_minus_max_drawdown',
        'financial_data_protocol': run_identity['contract']['financial_data_protocol'],
        'reward_schema_version': REWARD_SCHEMA_VERSION,
        'typed_distribution': TYPED_ACTION_DISTRIBUTION_VERSION,
        'training_checkpoint_selection': train_selection,
        'iteration_timings': iteration_timings,
        'initial_evaluation': initial_evaluation,
        'training_curve': training_curve,
        'validation_curve': validation_curve,
        'requested_rollouts': args.rollouts,
        'actual_timesteps': actual_timesteps,
        'rollout_count': rollout_count,
        'n_envs': args.n_envs,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': args.n_epochs,
        'learning_rate': args.learning_rate,
        'target_kl': args.target_kl,
        'log_std_init': args.log_std_init,
        'advantage_baseline': args.advantage_baseline,
        'learner_device': learner_device,
        'complete_train_evaluation_every_rollouts': args.eval_every_rollouts,
        'evaluation_execution': args.evaluation_execution,
        'gamma': PPO_GAMMA,
        'gae_lambda': PPO_GAE_LAMBDA,
        'ent_coef': 0.0,
    }
    if validation_selector is None:
        raise RuntimeError("training completed without validation")
    selection = validation_selector.selection()
    selection.update(checkpoint_sha256=file_sha256(output_dir / MODEL_FILE), deployment_bundle=False)
    training_metadata.update(
        status="training_complete", protocol=run_identity["contract"]["evaluation_protocol"],
        checkpoint_selection=selection, test_curve=test_curve,
        rollout=rollout_environment.manifest(), deployment_bundle=False,
        resident_shared_memory_bytes=dict(resident_splits),
        run_identity={"identity_version": run_identity["identity_version"],
                      "identity_sha256": run_identity["identity_sha256"]},
    )
    _save_output_schemas()
    _verify_training_inputs()
    final_latest = save_latest_train_checkpoint(model, output_dir,
        run_identity_sha256=str(run_identity["identity_sha256"]),
        timesteps=int(model.num_timesteps), ppo_updates=int(model._n_updates),
        phase="training_complete")
    if evaluation_queue is not None:
        atomic_write_json(output_dir / EVALUATION_COMPLETION_FILE, {
            "complete": True, "run_identity_sha256": run_identity["identity_sha256"],
            "latest": final_latest,
        })
    publish_evaluation_curves()
    atomic_write_json(output_dir / "training.json", training_metadata)
    print(json.dumps({"event": "training_complete", "timesteps": int(model.num_timesteps),
                      "selected": selection, "deployment_bundle": False}), flush=True)
    return output_dir

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", default="artifacts/rl/ppo")
    default_splits, _ = read_evaluation_splits()
    parser.add_argument("--train-start", default=default_splits["train"][0])
    parser.add_argument("--train-end", default=default_splits["train"][1])
    parser.add_argument("--validation-start", default=default_splits["validation"][0])
    parser.add_argument("--validation-end", default=default_splits["validation"][1])
    parser.add_argument("--test-start", default=default_splits["test"][0])
    parser.add_argument("--test-end", default=default_splits["test"][1])
    parser.add_argument(
        "--rollouts",
        type=int,
        default=DEFAULT_ROLLOUT_COUNT,
        help="number of vector rollouts; each collects n_envs * n_steps transitions",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=DEFAULT_ROLLOUT_WORKERS,
        help="independent rollout workers (GA-style shared memory)",
    )
    parser.add_argument(
        "--rollout-backend",
        choices=("auto", "dummy", "subproc"),
        default="auto",
    )
    parser.add_argument(
        "--backtest-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE_TARGET,
        help=f"0 chooses a divisor of the vector rollout targeting at most {DEFAULT_BATCH_SIZE_TARGET}; if none exists, uses the full rollout",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=64,
        help="transitions per env before updating; 0 collects the complete episode (requires --episode-scope full)",
    )
    parser.add_argument(
        "--eval-every-rollouts",
        type=int,
        default=DEFAULT_EVALUATION_EVERY,
        help="replay complete train, validation and test periods at this cadence",
    )
    parser.add_argument(
        "--device",
        choices=(PPO_DEVICE,),
        default=PPO_DEVICE,
        help="CUDA is required for all PPO training, evaluation and inference",
    )
    parser.add_argument("--evaluation-execution", choices=("blocking", "overlap"),
                        default="blocking", help="overlap one frozen CUDA train replay with the CUDA learner")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate-end-fraction", type=float, default=1.0)
    parser.add_argument("--learning-rate-decay-start", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--log-std-init", type=float, default=DEFAULT_LOG_STD_INIT,
                        help="initial log standard deviation of the SB3 Gaussian head in Box coordinates")
    parser.add_argument("--advantage-baseline", choices=ADVANTAGE_BASELINES, default=DEFAULT_ADVANTAGE_BASELINE,
                        help="synchronized_env_row_mean centers GAE advantages across the lockstep "
                             "environments of each buffer row (requires --episode-scope full)")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--episode-scope", choices=("random", "full"), default="full",
                        help="random contiguous windows or repeated complete training-period episodes")
    parser.add_argument(
        "--min-episode-transitions",
        type=int,
        default=20,
        help="minimum random contiguous training-window length",
    )
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument(
        "--training-slippage-rate",
        type=float,
        default=DEFAULT_FEE_SCHEDULE.slippage_rate,
        help="single-side slippage used by rollout reward execution",
    )
    parser.add_argument(
        "--evaluation-slippage-rate",
        type=float,
        default=DEFAULT_FEE_SCHEDULE.slippage_rate,
        help="single-side slippage used by complete train/validation/test backtests",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--resume-from")
    parser.add_argument("--warm-start-from")
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    if args.rollouts <= 0 or args.n_envs <= 0:
        raise ValueError("rollouts and n_envs must be positive")
    if args.min_episode_transitions <= 0:
        raise ValueError("min_episode_transitions must be positive")
    if not 1 <= args.backtest_workers <= MAX_PARALLEL_BACKTEST_WORKERS:
        raise ValueError(
            f"backtest_workers must be in [1,{MAX_PARALLEL_BACKTEST_WORKERS}]"
        )
    if (
        args.batch_size < 0
        or args.n_steps < 0 or args.n_steps == 1
        or (args.n_steps == 0 and args.episode_scope != "full")
        or args.n_epochs <= 0
        or args.eval_every_rollouts <= 0
    ):
        raise ValueError("batch_size and n_epochs are invalid")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not 0 < args.learning_rate_end_fraction <= 1 or not 0 <= args.learning_rate_decay_start < 1:
        raise ValueError("learning-rate schedule fractions are invalid")
    if args.target_kl is not None and (not math.isfinite(args.target_kl) or args.target_kl <= 0.0):
        raise ValueError("target_kl must be finite and positive")
    if not math.isfinite(args.log_std_init):
        raise ValueError("log_std_init must be finite")
    if args.advantage_baseline == "synchronized_env_row_mean" and (
        args.episode_scope != "full" or args.n_envs < 2
    ):
        raise ValueError("synchronized_env_row_mean baseline requires --episode-scope full and at least two envs")
    if (
        not math.isfinite(args.training_slippage_rate)
        or args.training_slippage_rate < 0.0
        or not math.isfinite(args.evaluation_slippage_rate)
        or args.evaluation_slippage_rate < 0.0
    ):
        raise ValueError("slippage rates must be finite and non-negative")
    if args.evaluation_slippage_rate != DEFAULT_FEE_SCHEDULE.slippage_rate:
        raise ValueError(
            f"complete-period evaluation requires the canonical {DEFAULT_FEE_SCHEDULE.slippage_rate:.2%} slippage"
        )
    _validate_resume_arguments(args)
    _validate_warm_start_arguments(args)
    _validate_split_boundaries(args)


def main() -> None:
    args = build_parser().parse_args()
    _validate_cli(args)
    train(args)


if __name__ == "__main__":
    main()

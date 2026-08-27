"""Train, evaluate, and export the config4 Stable-Baselines3 PPO policy."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import time
from typing import Iterable, Mapping

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from ai.bundle import (
    DEPLOYMENT_GATE_NAMES,
    BundleManifest,
    file_sha256,
    policy_source_sha256,
)
from env.action_schema import ActionSchema
from env.backtest import RolloutTrace, dynamic_config_summary, run_episode
from env.encoder import TrainOnlyNormalizer
from env.gym_adapter import (
    DEFAULT_UNIVERSE_PREFIXES,
    PreparedEpisode,
    WBRGymEnv,
    environment_schema_manifest,
    stock_universe_mask,
)
from env.metrics import robust_calmar
from factor import FactorBatch, precompute_factors
from offline_data import (
    RuntimeSlice,
    compute_runtime_lineage,
    load_runtime_slice,
)


DEFAULT_RUNTIME = "data/runtime/runtime_1990-12-19_2026-08-27.npz"
DEFAULT_TRAIN_START = "2010-01-01"
DEFAULT_TRAIN_END = "2018-12-31"
DEFAULT_VALIDATION_START = "2019-01-01"
DEFAULT_VALIDATION_END = "2022-12-31"
DEFAULT_TEST_START = "2023-01-01"
DEFAULT_TEST_END = "2026-08-27"
DEFAULT_TRAIN_FOLDS = (
    ("2010-01-01", "2012-12-31"),
    ("2013-01-01", "2015-12-31"),
    ("2016-01-01", "2018-12-31"),
)
MIN_EXPOSURE = 0.45
MIN_CONTINUOUS_STD = 0.002
MIN_CONTINUOUS_RANGE = 0.01
MIN_CATEGORY_COVERAGE = 0.01
MIN_SCORE_IMPROVEMENT = 0.001
MIN_SCORE_RELATIVE_IMPROVEMENT = 0.001
MATERIAL_SCORE_IMPROVEMENT = 0.02
PLATEAU_EVALUATIONS = 5
MIN_VALIDATION_ABSOLUTE_TOLERANCE = 0.05
VALIDATION_RELATIVE_TOLERANCE = 0.15


def _validate_split_boundaries(args: argparse.Namespace) -> None:
    train_start = np.datetime64(args.train_start, "D")
    train_end = np.datetime64(args.train_end, "D")
    if train_start > train_end:
        raise ValueError("training split start must not be later than end")
    if args.skip_holdout:
        return
    validation_start = np.datetime64(args.validation_start, "D")
    validation_end = np.datetime64(args.validation_end, "D")
    test_start = np.datetime64(args.test_start, "D")
    test_end = np.datetime64(args.test_end, "D")
    if not (
        train_end < validation_start <= validation_end < test_start <= test_end
    ):
        raise ValueError(
            "train, validation, and test splits must be strictly ordered and disjoint"
        )


def _json_write(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


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
) -> tuple[RuntimeSlice, FactorBatch, PreparedEpisode]:
    begin = time.perf_counter()
    runtime = load_runtime_slice(runtime_path, start, end, lookback=lookback)
    universe = stock_universe_mask(
        runtime.stock_codes,
        DEFAULT_UNIVERSE_PREFIXES,
    )
    factors = precompute_factors(runtime, rank_universe_mask=universe)
    episode = PreparedEpisode.build(
        runtime,
        factors,
        lookback=lookback,
        universe_prefixes=DEFAULT_UNIVERSE_PREFIXES,
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
    return runtime, factors, episode


def _fit_train_normalizer(
    episode: PreparedEpisode,
    action_schema: ActionSchema,
    fixed_action: np.ndarray,
    *,
    initial_cash: float,
) -> TrainOnlyNormalizer:
    env = WBRGymEnv(
        episode,
        action_schema=action_schema,
        initial_cash=initial_cash,
        normalizer=None,
    )
    observation, _ = env.reset(seed=0)
    samples = [observation.copy()]
    terminated = False
    while not terminated:
        observation, _, terminated, truncated, _ = env.step(fixed_action)
        if truncated:
            raise RuntimeError("normalizer rollout unexpectedly truncated")
        samples.append(observation.copy())
    values = np.stack(samples)
    return TrainOnlyNormalizer.fit(
        values,
        episode.encoder.output_schema,
        dataset_role="train",
    )


def _model_provider(model: PPO):
    def provide(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return provide


def _random_provider(action_dim: int, seed: int):
    generator = np.random.default_rng(seed)

    def provide(observation: np.ndarray) -> np.ndarray:
        del observation
        return generator.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)

    return provide


def _folds_for_episode(
    episode: PreparedEpisode,
    train_start: str,
    train_end: str,
) -> tuple[tuple[str, str], ...]:
    if train_start == DEFAULT_TRAIN_START and train_end == DEFAULT_TRAIN_END:
        return DEFAULT_TRAIN_FOLDS
    dates = episode.runtime.trade_dates[episode.decision_start : episode.decision_stop]
    boundaries = np.linspace(0, len(dates) - 1, 4, dtype=int)
    if np.any(np.diff(boundaries) < 1):
        raise ValueError("training split is too short for three robust folds")
    return tuple(
        (
            np.datetime_as_string(dates[boundaries[index]], unit="D"),
            np.datetime_as_string(dates[boundaries[index + 1]], unit="D"),
        )
        for index in range(3)
    )


def _trace_evaluation(
    trace: RolloutTrace,
    folds: Iterable[tuple[str, str]] | None = None,
) -> dict[str, object]:
    result = trace.as_summary()
    if folds is not None:
        result["robust"] = robust_calmar(
            trace.rewards,
            trace.decision_dates,
            trace.next_decision_dates,
            folds,
        )
    return result


def _dynamicity(dynamic_summary: Mapping[str, object]) -> dict[str, bool]:
    continuous = dynamic_summary["continuous"]
    categorical = dynamic_summary["categorical"]
    continuous_dynamic = any(
        float(values["std"]) >= MIN_CONTINUOUS_STD
        and float(values["maximum"]) - float(values["minimum"])
        >= MIN_CONTINUOUS_RANGE
        for values in continuous.values()
    )

    def category_is_dynamic(name: str) -> bool:
        values = categorical[name]
        return bool(
            int(values["unique_count"]) >= 2
            and sum(
                1
                for coverage in values["coverage"].values()
                if float(coverage) >= MIN_CATEGORY_COVERAGE
            )
            >= 2
        )

    binary_names = tuple(
        name
        for name in categorical
        if name.startswith(("factor_enabled.", "filter_flag."))
        or name in ("rebalance_now", "limit_up_protection")
    )
    return {
        "continuous_parameter_dynamic": continuous_dynamic,
        "binary_parameter_dynamic": any(
            category_is_dynamic(name) for name in binary_names
        ),
        "discrete_parameter_dynamic": any(
            category_is_dynamic(name) for name in ("buy_n", "sell_m")
        ),
        "enum_parameter_dynamic": category_is_dynamic("rebalance"),
    }


class _TrainCheckpointState:
    """Training-only checkpoint and plateau state over actual evaluations."""

    def __init__(self, initial_score: float) -> None:
        score = float(initial_score)
        if not np.isfinite(score):
            raise ValueError("initial training checkpoint score must be finite")
        self.best_score = score
        self.material_best_score = score
        self.best_trained_step: int | None = None
        self.best_trained_phase: str | None = None
        self.last_material_improvement_eval = 0

    def observe(
        self,
        *,
        score: float,
        eligible: bool,
        timesteps: int,
        phase: str,
        evaluation_index: int,
    ) -> tuple[bool, bool]:
        score = float(score)
        if not np.isfinite(score):
            raise RuntimeError("training robust Calmar became non-finite")
        if evaluation_index <= 0:
            raise ValueError("training evaluation index must be positive")

        improved = bool(eligible and score > self.best_score)
        material_improved = bool(
            eligible
            and score > self.material_best_score + MATERIAL_SCORE_IMPROVEMENT
        )
        if improved:
            self.best_score = score
            self.best_trained_step = int(timesteps)
            self.best_trained_phase = str(phase)
        if material_improved:
            # This anchor deliberately does not drift on smaller improvements:
            # several incremental gains can cumulatively cross the threshold.
            self.material_best_score = score
            self.last_material_improvement_eval = int(evaluation_index)
        return improved, material_improved

    def plateau_reached(self, evaluation_count: int) -> bool:
        return bool(
            evaluation_count >= PLATEAU_EVALUATIONS
            and evaluation_count - self.last_material_improvement_eval
            >= PLATEAU_EVALUATIONS
        )


class TrainEvaluationCallback(BaseCallback):
    """Evaluate only on sealed training folds and retain the best checkpoint."""

    def __init__(
        self,
        episode: PreparedEpisode,
        action_schema: ActionSchema,
        normalizer: TrainOnlyNormalizer,
        folds: tuple[tuple[str, str], ...],
        output_dir: Path,
        eval_freq: int,
        initial_score: float,
        *,
        initial_cash: float,
        seed: int,
    ) -> None:
        super().__init__(verbose=0)
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        self.episode = episode
        self.action_schema = action_schema
        self.normalizer = normalizer
        self.folds = folds
        self.output_dir = output_dir
        self.eval_freq = int(eval_freq)
        self.initial_cash = float(initial_cash)
        self.seed = int(seed)
        self.history: list[dict[str, object]] = []
        self.checkpoints = _TrainCheckpointState(initial_score)
        self.best_path = output_dir / "best_train_model"

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        self.evaluate_current_model(phase="scheduled")
        return True

    def evaluate_current_model(self, *, phase: str) -> dict[str, object]:
        if phase not in ("scheduled", "post_update"):
            raise ValueError(f"unsupported training evaluation phase: {phase}")
        env = WBRGymEnv(
            self.episode,
            action_schema=self.action_schema,
            normalizer=self.normalizer,
            initial_cash=self.initial_cash,
        )
        trace = run_episode(env, _model_provider(self.model), seed=self.seed)
        evaluation = _trace_evaluation(trace, self.folds)
        dynamicity = _dynamicity(dynamic_config_summary(trace))
        score = float(evaluation["robust"]["robust_calmar"])
        return self._record_evaluation(
            timesteps=int(self.num_timesteps),
            phase=phase,
            ppo_updates=int(self.model._n_updates),
            score=score,
            average_exposure=float(trace.average_exposure),
            dynamicity=dynamicity,
            optimizer=self._optimizer_snapshot(),
            evaluation=evaluation,
        )

    def _record_evaluation(
        self,
        *,
        timesteps: int,
        phase: str,
        ppo_updates: int,
        score: float,
        average_exposure: float,
        dynamicity: Mapping[str, bool],
        optimizer: Mapping[str, float],
        evaluation: Mapping[str, object],
    ) -> dict[str, object]:
        eligible = bool(
            ppo_updates > 0
            and average_exposure >= MIN_EXPOSURE
            and dynamicity["continuous_parameter_dynamic"]
            and dynamicity["binary_parameter_dynamic"]
            and dynamicity["discrete_parameter_dynamic"]
        )
        evaluation_index = len(self.history) + 1
        improved, material_improved = self.checkpoints.observe(
            score=score,
            eligible=eligible,
            timesteps=timesteps,
            phase=phase,
            evaluation_index=evaluation_index,
        )
        if improved:
            self.model.save(self.best_path)
        row = {
            "evaluation_index": evaluation_index,
            "phase": phase,
            "timesteps": int(timesteps),
            "ppo_updates": int(ppo_updates),
            "score": float(score),
            "average_exposure": float(average_exposure),
            "eligible_exposure": average_exposure >= MIN_EXPOSURE,
            "eligible_checkpoint": eligible,
            "dynamicity": dict(dynamicity),
            "improved": improved,
            "material_improved": material_improved,
            "material_best_score": self.checkpoints.material_best_score,
            "best_trained_step": self.checkpoints.best_trained_step,
            "best_trained_phase": self.checkpoints.best_trained_phase,
            "optimizer": dict(optimizer),
            "evaluation": dict(evaluation),
        }
        self.history.append(row)
        _json_write(self.output_dir / "training_curve.json", self.history)
        print(json.dumps({"event": "train_eval", **row}, ensure_ascii=False), flush=True)
        return row

    def _optimizer_snapshot(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, value in self.model.logger.name_to_value.items():
            if not name.startswith("train/"):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(number):
                raise RuntimeError(f"PPO optimizer metric became non-finite: {name}")
            values[name] = number
        return values

    @property
    def plateau_reached(self) -> bool:
        return self.checkpoints.plateau_reached(len(self.history))

    @property
    def best_score(self) -> float:
        return self.checkpoints.best_score

    @property
    def best_trained_step(self) -> int | None:
        return self.checkpoints.best_trained_step

    @property
    def best_trained_phase(self) -> str | None:
        return self.checkpoints.best_trained_phase


def _record_post_update_evaluation(
    callback: TrainEvaluationCallback,
) -> dict[str, object]:
    """Evaluate the final PPO parameters after the last optimizer update."""

    return callback.evaluate_current_model(phase="post_update")


def _select_model_after_training(
    final_model: PPO,
    callback: TrainEvaluationCallback,
    train_env: WBRGymEnv,
    output_dir: Path,
) -> tuple[PPO, dict[str, object]]:
    """Select a real trained checkpoint or fail closed to checkpoint zero."""

    if callback.best_trained_step is None:
        diagnostic_path = output_dir / "trained_final_model"
        final_model.save(diagnostic_path)
        selected_path = output_dir / "checkpoint_0.zip"
        selection = {
            "source": "checkpoint_0",
            "selected_step": 0,
            "selected_phase": "initial",
            "best_trained_step": None,
            "best_trained_phase": None,
            "trained_final_model": diagnostic_path.with_suffix(".zip").name,
        }
    else:
        selected_path = callback.best_path.with_suffix(".zip")
        selection = {
            "source": "best_trained_model",
            "selected_step": callback.best_trained_step,
            "selected_phase": callback.best_trained_phase,
            "best_trained_step": callback.best_trained_step,
            "best_trained_phase": callback.best_trained_phase,
            "trained_final_model": None,
        }
    if not selected_path.exists():
        raise RuntimeError(f"selected PPO checkpoint is missing: {selected_path}")
    selection["selected_file"] = selected_path.name
    return PPO.load(selected_path, env=train_env, device="cpu"), selection


def _technical_convergence(convergence: Mapping[str, object]) -> bool:
    """Require every deployment gate, including a real trained checkpoint."""

    return all(bool(convergence[name]) for name in DEPLOYMENT_GATE_NAMES)


def _factor_manifest(factors: FactorBatch) -> dict[str, object]:
    return {
        "schema_version": factors.schema_version,
        "schema_hash": factors.schema_hash,
        "rank_universe_sha256": factors.rank_universe_sha256,
        "factor_names": list(factors.factor_names),
        "filter_names": list(factors.filter_names),
        "factors": [item.as_dict() for item in factors.factor_metadata],
        "filters": [item.as_dict() for item in factors.filter_metadata],
    }


def _initialize_policy_from_static(
    model: PPO,
    action_schema: ActionSchema,
    fixed_action: np.ndarray,
) -> np.ndarray:
    """Start PPO at the current config without freezing its state dependence."""

    initial_action = np.asarray(fixed_action, dtype=np.float32).copy()
    for field in action_schema.layout:
        if field.kind == "binary":
            # Stay on the same side of the decoder threshold while leaving
            # useful exploration probability for binary decisions.
            distance = 0.02 if field.name == "limit_up_protection" else 0.10
            initial_action[field.index] = (
                distance if initial_action[field.index] >= 0.0 else -distance
            )
        elif field.kind in ("discrete", "enum"):
            count = len(field.choices)
            coordinate = float(initial_action[field.index])
            bucket = min(int(((coordinate + 1.0) / 2.0) * count), count - 1)
            if bucket < count - 1:
                initial_action[field.index] = -1.0 + 2.0 * (bucket + 1) / count - 1e-3
            else:
                initial_action[field.index] = -1.0 + 2.0 * bucket / count + 1e-3
    if action_schema.decode(initial_action) != action_schema.decode(fixed_action):
        raise RuntimeError("softened PPO initialization changed the static config")
    action_net = model.policy.action_net
    with torch.no_grad():
        action_net.weight.zero_()
        action_net.bias.copy_(torch.as_tensor(initial_action, device=action_net.bias.device))
    return initial_action


def _config_comparison(
    static_config: Mapping[str, object],
    dynamic_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "static_config": dict(static_config),
        "rl_validation_distribution": dict(dynamic_summary),
        "semantic_changes": {
            "weights": "static four weights become normalized per-day weights",
            "factor_enabled": "new explicit per-day binary controls",
            "cash_reserve_ratio": "replaced by per-day target_exposure=1-reserve",
            "holding_period": "replaced by per-day rebalance_now",
            "rebalance": "remains equalize versus replace-only mode, independent of rebalance_now",
            "prefilter_n": "excluded because the legacy loader discarded it",
            "timing_enabled": "excluded; target_exposure is the direct learned timing control",
            "environment_constants": "fees, slippage, universe, causality and split boundaries stay fixed",
        },
    }


def train(args: argparse.Namespace) -> Path:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _validate_split_boundaries(args)
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve()
    runtime_path = Path(args.runtime).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_source_sha256 = policy_source_sha256(repo_root)
    initial_config_sha256 = file_sha256(config_path)
    initial_runtime_sha256 = file_sha256(runtime_path)
    config_payload = _load_config(config_path)
    bundle_config_path = output_dir / "strategy_config.json"
    if config_path != bundle_config_path:
        shutil.copyfile(config_path, bundle_config_path)
    if file_sha256(bundle_config_path) != initial_config_sha256:
        raise RuntimeError("bundled strategy config differs from the training config")
    runtime_lineage = compute_runtime_lineage(runtime_path)
    if file_sha256(runtime_path) != initial_runtime_sha256:
        raise RuntimeError("runtime data changed while lineage was computed")

    train_runtime, train_factors, train_episode = _prepare_split(
        runtime_path,
        args.train_start,
        args.train_end,
        lookback=args.lookback,
    )
    if train_runtime.manifest.source_sha256 != initial_runtime_sha256:
        raise RuntimeError("runtime data changed before training data was sealed")
    action_schema = ActionSchema(
        factor_names=train_factors.factor_names,
        filter_names=train_factors.filter_names,
    )
    static_day_config = action_schema.from_static_config(config_payload)
    fixed_action = action_schema.encode(static_day_config)
    normalizer = _fit_train_normalizer(
        train_episode,
        action_schema,
        fixed_action,
        initial_cash=args.initial_cash,
    )
    normalizer_path = output_dir / "normalizer.json"
    normalizer.save(normalizer_path)

    train_env = WBRGymEnv(
        train_episode,
        action_schema=action_schema,
        normalizer=normalizer,
        initial_cash=args.initial_cash,
    )
    check_env(train_env, warn=True)
    policy_kwargs = {
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "log_std_init": args.log_std_init,
    }
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
        seed=args.seed,
        device="cpu",
    )
    _initialize_policy_from_static(
        model,
        action_schema,
        fixed_action,
    )
    folds = _folds_for_episode(train_episode, args.train_start, args.train_end)
    fixed_trace = run_episode(
        WBRGymEnv(
            train_episode,
            action_schema=action_schema,
            normalizer=normalizer,
            initial_cash=args.initial_cash,
        ),
        lambda observation: fixed_action,
        seed=args.seed,
    )
    initial_trace = run_episode(
        WBRGymEnv(
            train_episode,
            action_schema=action_schema,
            normalizer=normalizer,
            initial_cash=args.initial_cash,
        ),
        _model_provider(model),
        seed=args.seed,
    )
    random_trace = run_episode(
        WBRGymEnv(
            train_episode,
            action_schema=action_schema,
            normalizer=normalizer,
            initial_cash=args.initial_cash,
        ),
        _random_provider(action_schema.action_dim, args.seed),
        seed=args.seed,
    )
    baseline = {
        "fixed_config": _trace_evaluation(fixed_trace, folds),
        "untrained_ppo": _trace_evaluation(initial_trace, folds),
        "seeded_random": _trace_evaluation(random_trace, folds),
    }
    initial_score = float(baseline["untrained_ppo"]["robust"]["robust_calmar"])
    fixed_score = float(baseline["fixed_config"]["robust"]["robust_calmar"])
    model.save(output_dir / "checkpoint_0")
    callback = TrainEvaluationCallback(
        train_episode,
        action_schema,
        normalizer,
        folds,
        output_dir,
        args.eval_freq,
        max(initial_score, fixed_score),
        initial_cash=args.initial_cash,
        seed=args.seed,
    )
    # Checkpoint zero is diagnostic fallback. A trained checkpoint replaces it
    # only when score, exposure, and all required output-type dynamics pass.
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    trained_num_timesteps = int(model.num_timesteps)
    # Always evaluate after learn() returns: an SB3 callback at the same
    # timestep observes parameters before the last PPO optimizer update.
    _record_post_update_evaluation(callback)
    model, checkpoint_selection = _select_model_after_training(
        model,
        callback,
        train_env,
        output_dir,
    )
    model_path_without_suffix = output_dir / "model"
    model.save(model_path_without_suffix)
    model_path = model_path_without_suffix.with_suffix(".zip")

    selected_train_trace = run_episode(
        WBRGymEnv(
            train_episode,
            action_schema=action_schema,
            normalizer=normalizer,
            initial_cash=args.initial_cash,
        ),
        _model_provider(model),
        seed=args.seed,
    )
    train_evaluation = _trace_evaluation(selected_train_trace, folds)
    evaluation: dict[str, object] = {"train": train_evaluation}
    dynamic_trace = selected_train_trace

    if not args.skip_holdout:
        _, validation_factors, validation_episode = _prepare_split(
            runtime_path,
            args.validation_start,
            args.validation_end,
            lookback=args.lookback,
        )
        if validation_factors.schema_hash != train_factors.schema_hash:
            raise ValueError("validation factor schema differs from training")
        validation_trace = run_episode(
            WBRGymEnv(
                validation_episode,
                action_schema=action_schema,
                normalizer=normalizer,
                initial_cash=args.initial_cash,
            ),
            _model_provider(model),
            seed=args.seed,
        )
        evaluation["validation"] = _trace_evaluation(validation_trace)
        validation_fixed_trace = run_episode(
            WBRGymEnv(
                validation_episode,
                action_schema=action_schema,
                normalizer=normalizer,
                initial_cash=args.initial_cash,
            ),
            lambda observation: fixed_action,
            seed=args.seed,
        )
        baseline["fixed_config_validation"] = _trace_evaluation(
            validation_fixed_trace
        )
        dynamic_trace = validation_trace

    dynamic_summary = dynamic_config_summary(dynamic_trace)
    dynamicity = _dynamicity(dynamic_summary)
    initial_threshold = max(
        MIN_SCORE_IMPROVEMENT,
        MIN_SCORE_RELATIVE_IMPROVEMENT * abs(initial_score),
    )
    random_score = float(baseline["seeded_random"]["robust"]["robust_calmar"])
    random_threshold = max(0.05, 0.10 * abs(random_score))
    selected_score = float(train_evaluation["robust"]["robust_calmar"])
    fixed_threshold = max(
        MIN_SCORE_IMPROVEMENT,
        MIN_SCORE_RELATIVE_IMPROVEMENT * abs(fixed_score),
    )
    if args.skip_holdout:
        validation_gate = False
        validation_score = None
        fixed_validation_score = None
        validation_tolerance = None
    else:
        validation_score = float(evaluation["validation"]["metrics"]["calmar"])
        fixed_validation_score = float(
            baseline["fixed_config_validation"]["metrics"]["calmar"]
        )
        validation_tolerance = max(
            MIN_VALIDATION_ABSOLUTE_TOLERANCE,
            VALIDATION_RELATIVE_TOLERANCE * abs(fixed_validation_score),
        )
        validation_gate = bool(
            np.isfinite(validation_score)
            and validation_score >= fixed_validation_score - validation_tolerance
            and float(evaluation["validation"]["metrics"]["total_return"]) > 0.0
        )
    convergence = {
        "trained_checkpoint_selected": callback.best_trained_step is not None,
        "finite": bool(
            np.isfinite(selected_train_trace.rewards).all()
            and np.isfinite(selected_train_trace.actions).all()
            and np.isfinite(selected_train_trace.nav).all()
            and all(
                bool(torch.isfinite(parameter).all().item())
                for parameter in model.policy.parameters()
            )
            and all(
                np.isfinite(list(row["optimizer"].values())).all()
                for row in callback.history
            )
        ),
        "beats_untrained_threshold": selected_score >= initial_score + initial_threshold,
        "beats_random_threshold": selected_score >= random_score + random_threshold,
        "beats_fixed_config_threshold": selected_score >= fixed_score + fixed_threshold,
        "validation_noninferiority_and_positive_return": validation_gate,
        "plateau_reached": callback.plateau_reached,
        "average_exposure_at_least_0_45": dynamic_trace.average_exposure
        >= MIN_EXPOSURE,
        **dynamicity,
        "selected_robust_calmar": selected_score,
        "best_trained_step": callback.best_trained_step,
        "best_trained_phase": callback.best_trained_phase,
        "material_best_score": callback.checkpoints.material_best_score,
        "untrained_robust_calmar": initial_score,
        "random_robust_calmar": random_score,
        "fixed_config_robust_calmar": fixed_score,
        "fixed_config_threshold": fixed_threshold,
        "validation_calmar": validation_score,
        "fixed_config_validation_calmar": fixed_validation_score,
        "validation_tolerance": validation_tolerance,
        "gate_thresholds": {
            "minimum_exposure": MIN_EXPOSURE,
            "minimum_continuous_std": MIN_CONTINUOUS_STD,
            "minimum_continuous_range": MIN_CONTINUOUS_RANGE,
            "minimum_category_coverage": MIN_CATEGORY_COVERAGE,
            "minimum_score_improvement": MIN_SCORE_IMPROVEMENT,
            "minimum_score_relative_improvement": MIN_SCORE_RELATIVE_IMPROVEMENT,
            "material_score_improvement": MATERIAL_SCORE_IMPROVEMENT,
            "plateau_evaluations": PLATEAU_EVALUATIONS,
            "minimum_validation_absolute_tolerance": (
                MIN_VALIDATION_ABSOLUTE_TOLERANCE
            ),
            "validation_relative_tolerance": VALIDATION_RELATIVE_TOLERANCE,
        },
    }
    convergence["technical_convergence"] = _technical_convergence(convergence)

    # The sealed test interval is opened exactly once, only after the model is
    # frozen and every train/validation deployment gate has passed.  Test
    # results are reporting-only and can never select or rescue a checkpoint.
    if bool(convergence["technical_convergence"]):
        _, test_factors, test_episode = _prepare_split(
            runtime_path,
            args.test_start,
            args.test_end,
            lookback=args.lookback,
        )
        if test_factors.schema_hash != train_factors.schema_hash:
            raise ValueError("test factor schema differs from training")
        test_trace = run_episode(
            WBRGymEnv(
                test_episode,
                action_schema=action_schema,
                normalizer=normalizer,
                initial_cash=args.initial_cash,
            ),
            _model_provider(model),
            seed=args.seed,
        )
        evaluation["test"] = _trace_evaluation(test_trace)
        test_fixed_trace = run_episode(
            WBRGymEnv(
                test_episode,
                action_schema=action_schema,
                normalizer=normalizer,
                initial_cash=args.initial_cash,
            ),
            lambda observation: fixed_action,
            seed=args.seed,
        )
        baseline["fixed_config_test"] = _trace_evaluation(test_fixed_trace)

    _json_write(output_dir / "action_schema.json", action_schema.to_dict())
    _json_write(
        output_dir / "observation_schema.json",
        train_episode.observation_builder.schema.to_dict(),
    )
    _json_write(
        output_dir / "encoded_schema.json",
        train_episode.encoder.output_schema.to_dict(),
    )
    _json_write(output_dir / "baseline.json", baseline)
    _json_write(output_dir / "evaluation.json", evaluation)
    _json_write(output_dir / "dynamic_config.json", dynamic_summary)
    _json_write(output_dir / "convergence.json", convergence)
    static_export = action_schema.to_static_config(static_day_config)
    _json_write(
        output_dir / "config_comparison.json",
        _config_comparison(static_export, dynamic_summary),
    )
    training_metadata = {
        "seed": args.seed,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "total_timesteps": trained_num_timesteps,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "log_std_init": args.log_std_init,
        "eval_freq": args.eval_freq,
        "folds": [list(item) for item in folds],
        "baseline": baseline,
        "training_curve": callback.history,
        "checkpoint_selection": checkpoint_selection,
        "convergence": convergence,
    }
    runtime_manifest = train_runtime.manifest.as_dict()
    runtime_manifest["lineage"] = runtime_lineage.as_dict()
    if policy_source_sha256(repo_root) != initial_source_sha256:
        raise RuntimeError("policy source changed while PPO training was running")
    if file_sha256(config_path) != initial_config_sha256:
        raise RuntimeError("strategy config changed while PPO training was running")
    if file_sha256(bundle_config_path) != initial_config_sha256:
        raise RuntimeError("bundled strategy config changed while PPO training was running")
    if file_sha256(runtime_path) != initial_runtime_sha256:
        raise RuntimeError("runtime data changed while PPO training was running")
    manifest = BundleManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        algorithm="stable_baselines3.PPO",
        model_file=model_path.name,
        model_sha256=file_sha256(model_path),
        normalizer_file=normalizer_path.name,
        normalizer_sha256=file_sha256(normalizer_path),
        config_file=bundle_config_path.name,
        config_sha256=initial_config_sha256,
        source_sha256=initial_source_sha256,
        runtime=runtime_manifest,
        factors=_factor_manifest(train_factors),
        observation_schema=train_episode.observation_builder.schema.to_dict(),
        encoded_schema=train_episode.encoder.output_schema.to_dict(),
        action_schema=action_schema.to_dict(),
        environment=environment_schema_manifest(action_schema),
        training=training_metadata,
        evaluation=evaluation,
    )
    manifest.save(output_dir)
    print(
        json.dumps(
            {
                "event": "training_complete",
                "bundle": str(output_dir),
                "convergence": convergence,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    gc.collect()
    if not bool(convergence["technical_convergence"]):
        failed = [
            name
            for name in DEPLOYMENT_GATE_NAMES
            if not convergence[name]
        ]
        raise RuntimeError(
            "PPO artifact is diagnostic-only; convergence gates failed: "
            + ", ".join(failed)
        )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME)
    parser.add_argument("--output", default="artifacts/rl/config4_final")
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=DEFAULT_TEST_END)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--log-std-init", type=float, default=-1.0)
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--skip-holdout", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.total_timesteps <= 0 or args.eval_freq <= 0:
        raise ValueError("timesteps and eval frequency must be positive")
    if args.total_timesteps < 5 * args.eval_freq:
        raise ValueError("total timesteps must permit at least five convergence evaluations")
    if args.n_steps <= 1 or args.batch_size <= 1 or args.n_epochs <= 0:
        raise ValueError("invalid PPO rollout/update settings")
    train(args)


if __name__ == "__main__":
    main()

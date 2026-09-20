"""Multiprocess production-factor GA over the canonical offline env."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from itertools import islice
from typing import Mapping

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from loguru import logger as ga_logger

from configs.training import DEFAULT_EVALUATION_EVERY, DEFAULT_EVALUATION_SPLITS_PATH, DEFAULT_ROLLOUT_WORKERS, read_evaluation_splits
from env.shared_episode import (
    AttachedPreparedEpisode,
    SharedPreparedEpisodeDescriptor,
    ResidentPreparedEpisode,
)
from ai.ga import (
    DEFAULT_GA_PROFILE,
    build_individual_config,
    generate_initial_configs,
    get_mode_configs,
    get_profile,
    get_profile_factor_classes,
    get_profile_filter_factor_classes,
    get_profile_preload_range,
    get_profile_weight_search_spaces,
    resolve_profile_name,
    sample_turnover_rate,
)
from ai.ga.config import canonicalize_ga_genes
from ai.reporting import append_training_diagnostic, write_ga_report, write_preparing_report, mark_training_failed
from env.action_schema import ActionSchema
from env.backtest import (
    EpisodeSession,
    PreparedEpisode,
    prepare_episode_from_runtime,
    run_day_config_episode,
)
from env.prefilter import prefilter_n_from_config
from env.observation import DEFAULT_LOOKBACK
from factor.registry import PRODUCTION_FACTOR_NAMES, PRODUCTION_FILTER_NAMES
from offline_data import latest_runtime_npz_path
from utils.stock.time import get_trading_date_span
from utils.atomic_file import atomic_write_json, file_sha256
from utils.windows_awake import keep_windows_awake


def _resolve_output_dir(output_dir_arg: str | None, mode: str) -> Path:
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"{mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

DEFAULT_GA_SEED = 20260720
_HOLDOUT_RESULT_FIELDS = (
    "val_calmar",
    "val_sharpe",
    "val_annualized",
    "val_max_drawdown",
    "test_calmar",
    "test_sharpe",
    "test_annualized",
    "test_max_drawdown",
)
_LEGACY_OBJECTIVE_FIELDS = ("fitness", "raw_fitness", "fold_calmars")


def _seed_ga_randomness(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _config_key(config: Mapping[str, object]) -> tuple:
    def freeze(value):
        if isinstance(value, Mapping):
            return tuple(sorted((key, freeze(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        return value

    return freeze(config)


def _ga_cache_key(config: Mapping[str, object]) -> str:
    return json.dumps(_config_key(config), ensure_ascii=False)


def _training_candidate_rank_key(entry: Mapping[str, object]) -> tuple:
    if entry.get("calmar") is None or not math.isfinite(float(entry["calmar"])):
        raise ValueError("training candidate is missing calmar")
    # Calmar is the only optimization metric. Canonical config is only a
    # deterministic tie-breaker and does not introduce a second objective.
    return float(entry["calmar"]), _ga_cache_key(entry["individual_config"])


def _select_training_candidate(ga_cache: Mapping[str, dict]) -> dict | None:
    eligible = [entry for entry in ga_cache.values() if entry.get("calmar") is not None]
    return max(eligible, key=_training_candidate_rank_key) if eligible else None


def _save_training_candidate(
    output_dir: Path,
    profile_name: str,
    ga_cache: Mapping[str, dict],
    *,
    prefilter_n: int,
) -> dict | None:
    winner = _select_training_candidate(ga_cache)
    if winner is None:
        return None
    payload = {"ga_profile": profile_name, "individual_config": {
        **winner["individual_config"], "prefilter_n": prefilter_n}}
    atomic_write_json(Path(output_dir) / "best_individual_config.json", payload,
                      sort_keys=False, allow_nan=False)
    return winner


def _append_jsonl_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _load_candidate_configs(
    path: str | Path,
    profile_name: str | None = None,
) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    resolved = resolve_profile_name(fallback=profile_name)
    if not isinstance(payload, dict) or set(payload) != {'configs'}:
        raise ValueError('candidate file must contain exactly the configs field')
    if not isinstance(payload['configs'], list) or not payload['configs']:
        raise ValueError('configs must be a non-empty canonical config list')

    configs: list[dict] = []
    for config in payload['configs']:
        if not isinstance(config, Mapping):
            raise ValueError("--candidate-configs contains an invalid config")
        canonical, _ = canonicalize_ga_genes(
            config,
            profile_name=resolved,
        )
        configs.append(canonical)
    return configs



class _TrainingParentPool:
    """Incremental score order over one append-only evaluated-candidate cache."""

    def __init__(self, cache: dict[str, dict]) -> None:
        self.cache = cache
        self.indexed_count = len(cache)
        self.rows = []
        for entry in cache.values():
            config, calmar = entry.get("individual_config"), entry.get("calmar")
            if config is not None and calmar is not None:
                self.rows.append((config, float(calmar), _ga_cache_key(config)))
        self.rows.sort(key=lambda row: (row[1], row[2]), reverse=True)

    def update(self, cache: dict[str, dict]) -> list[tuple[dict, float, str]]:
        if cache is not self.cache or len(cache) < self.indexed_count:
            raise ValueError("GA parent index requires the same append-only cache")
        added = len(cache) - self.indexed_count
        # Dict reverse iteration reaches only newly appended candidates.
        for key in reversed(list(islice(reversed(cache), added))):
            entry = cache[key]
            config, calmar = entry.get("individual_config"), entry.get("calmar")
            if config is None or calmar is None:
                continue
            item = (config, float(calmar), _ga_cache_key(config))
            rank = item[1:]
            left, right = 0, len(self.rows)
            while left < right:
                middle = (left + right) // 2
                if self.rows[middle][1:] >= rank:
                    left = middle + 1
                else:
                    right = middle
            self.rows.insert(left, item)
        self.indexed_count = len(cache)
        return self.rows


def ga_optimizer(
    results,
    state,
    population_size: int = 24,
    hall_of_fame_size: int = 24,
    profile_name: str = DEFAULT_GA_PROFILE,
    ga_cache: dict[str, dict] | None = None,
    gen: int = 0,
) -> list[dict]:
    """Select, cross and mutate current4 configs using Calmar only."""

    profile_name = resolve_profile_name(fallback=profile_name)
    results_list = list(results)
    score_cache = state.setdefault("score_cache", {})
    state.setdefault("population", [])
    state.setdefault("hall_of_fame", [])
    for result in results_list:
        score_cache[_config_key(result["individual_config"])] = float(result["calmar"])

    if not state["population"]:
        results_list.sort(key=_training_candidate_rank_key, reverse=True)
        state["population"] = [
            result["individual_config"] for result in results_list[:population_size]
        ]

    def score(config: Mapping[str, object]) -> float | None:
        return score_cache.get(_config_key(config))

    elite_fraction = 0.10
    tournament_size = 5
    immigrant_fraction = 0.15
    if ga_cache and len(ga_cache) >= population_size:
        if "parent_pool" not in state:
            state["parent_pool"] = _TrainingParentPool(ga_cache)
        pool = state["parent_pool"].update(ga_cache)
        elite_count = min(
            max(2, round(elite_fraction * population_size)),
            len(pool),
        )
        parents = [row[0] for row in pool[:elite_count]]
        for _ in range(population_size - elite_count):
            aspirants = random.sample(pool, min(tournament_size, len(pool)))
            parents.append(
                max(
                    aspirants,
                    key=lambda item: (item[1], item[2]),
                )[0]
            )
    else:
        ranked = [
            (config, value)
            for config in state["population"]
            if (value := score(config)) is not None
        ]
        if not ranked:
            ranked = [
                (result["individual_config"], float(result["calmar"]))
                for result in results_list
            ]
        ranked.sort(
            key=lambda item: (item[1], _ga_cache_key(item[0])),
            reverse=True,
        )
        parents = [config for config, _ in ranked[:population_size]]
    if not parents:
        raise ValueError("GA cannot breed without evaluated parents")

    unique_hof: dict[tuple, tuple[dict, float]] = {}
    for config in [*state["hall_of_fame"], *parents]:
        value = score(config)
        if value is None:
            continue
        key = _config_key(config)
        if key not in unique_hof or value > unique_hof[key][1]:
            unique_hof[key] = (config, value)
    state["hall_of_fame"] = [
        config
        for config, _ in sorted(
            unique_hof.values(),
            key=lambda item: (item[1], _ga_cache_key(item[0])),
            reverse=True,
        )[:hall_of_fame_size]
    ]

    weight_spaces = get_profile_weight_search_spaces(profile_name)
    dimensions = (
        "turnover_rate",
        *(f"weight.{name}" for name in weight_spaces),
    )

    def crossover(left: dict, right: dict) -> dict:
        turnover_rate = random.choice([left["turnover_rate"], right["turnover_rate"]])
        weights = {
            name: random.choice([left["weights"][name], right["weights"][name]])
            for name in weight_spaces
        }
        return build_individual_config(
            turnover_rate=turnover_rate,
            weights=weights,
            profile_name=profile_name,
        )

    def mutate(config: dict) -> dict:
        progress = gen / max(gen + 20, 1)
        low_rate = 0.25 - 0.17 * progress
        high_rate = 0.35 - 0.20 * progress
        mutate_count = max(
            1,
            min(
                math.ceil(random.uniform(low_rate, high_rate) * len(dimensions)),
                len(dimensions),
            ),
        )
        selected = set(random.sample(dimensions, mutate_count))
        turnover_rate = config["turnover_rate"]
        if "turnover_rate" in selected:
            turnover_rate = sample_turnover_rate(profile_name)
        weights = dict(config["weights"])
        for name, space in weight_spaces.items():
            if f"weight.{name}" in selected:
                weights[name] = random.uniform(0.0, 1.0)
        return build_individual_config(
            turnover_rate=turnover_rate,
            weights=weights,
            profile_name=profile_name,
        )

    immigrant_count = min(
        max(1, round(immigrant_fraction * population_size)),
        population_size,
    )
    crossover_count = population_size - immigrant_count
    children: list[dict] = []
    while len(children) < crossover_count:
        if len(parents) == 1:
            child = mutate(parents[0])
        else:
            left, right = random.sample(parents, 2)
            child = mutate(crossover(left, right))
        children.append(child)
    children.extend(generate_initial_configs(immigrant_count, profile_name))
    state["population"] = [*parents, *children]
    return [*parents, *children]


def _find_latest_ga_dir() -> Path | None:
    results_dir = Path("results")
    if not results_dir.is_dir():
        return None
    candidates = [
        path
        for path in results_dir.iterdir()
        if path.is_dir()
        and path.name.startswith(("ga_", "debug_"))
        and (path / "all_results.jsonl").is_file()
        and (path / "run_metadata.json").is_file()
    ]
    return max(candidates, key=lambda path: path.name) if candidates else None


def write_generation_diagnostics(output_dir, generation, results, proposed, evaluated, elapsed):
    """Persist population diversity and training-only fitness without extra evaluations."""
    names = tuple(results[0]['individual_config']['weights'])
    weights = np.asarray([[row['individual_config']['weights'][name] for name in names]
                          for row in results], dtype=np.float64)
    turnover, counts = np.unique([row['individual_config']['turnover_rate'] for row in results], return_counts=True)
    record = {
        'generation': generation + 1,
        'proposed_candidates': proposed, 'unique_candidates': len(results),
        'new_evaluations': evaluated, 'cache_hits': len(results) - evaluated,
        'evaluation_wall_seconds_excludes_holdouts_and_breeding': elapsed,
        'calmar_quantiles': np.quantile([row['calmar'] for row in results], [0, .25, .5, .75, 1]).tolist(),
        'factor_names': names, 'factor_weight_mean': weights.mean(axis=0).tolist(),
        'factor_weight_std': weights.std(axis=0).tolist(),
        'factor_weight_min': weights.min(axis=0).tolist(), 'factor_weight_max': weights.max(axis=0).tolist(),
        'turnover_counts': {str(value): int(count) for value, count in zip(turnover, counts)},
    }
    scalars = {key: {'label': label, 'value': record[key]} for key, label in (
        ('new_evaluations', '本代新增评估数'), ('cache_hits', '本代缓存命中数'),
        ('unique_candidates', '本代不同候选数'),
        ('evaluation_wall_seconds_excludes_holdouts_and_breeding', '本代训练评估秒数'))}
    scalars['fitness_median'] = {'label': '种群训练 Calmar 中位数', 'value': record['calmar_quantiles'][2]}
    scalars['fitness_best'] = {'label': '种群训练 Calmar 最大值', 'value': record['calmar_quantiles'][4]}
    for index, name in enumerate(names):
        scalars[f'diversity/{name}'] = {'label': f'{name} · 种群权重标准差', 'value': float(weights[:, index].std())}
    recorded = [row['execution_diagnostics'] for row in results if row['execution_diagnostics'] is not None]
    for name, label in (('mean_gross_turnover_ratio', '候选平均每日实际总换手率'),
                        ('mean_total_cost_ratio', '候选平均每日成本率'), ('total_fees', '候选平均总费用')):
        scalars[name] = {'label': label, 'value': float(np.mean([row[name] for row in recorded])) if recorded else None}
    record['execution_diagnostic_candidates'] = len(recorded)
    append_training_diagnostic(output_dir, algorithm='GA', step=generation + 1,
                               timesteps=None, scalars=scalars, details=record)


def _rebuild_from_jsonl(output_dir: Path) -> tuple[dict, int]:
    ga_cache: dict[str, dict] = {}
    last_generation = -1
    path = output_dir / "all_results.jsonl"
    if not path.is_file():
        return ga_cache, last_generation
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        present_holdout = [name for name in _HOLDOUT_RESULT_FIELDS if name in row]
        if present_holdout:
            raise ValueError(
                "GA resume row contains forbidden holdout fields: "
                + ", ".join(present_holdout)
            )
        present_legacy = [name for name in _LEGACY_OBJECTIVE_FIELDS if name in row]
        if present_legacy:
            raise ValueError(
                "GA resume row uses the removed multi-objective schema: "
                + ", ".join(present_legacy)
            )
        required = {"generation", "config", "calmar", "metrics", "total_return", "sharpe",
                    "annualized", "max_drawdown", "average_exposure", "full_investment_contract_satisfied",
                    "evaluation_elapsed_seconds", "execution_diagnostics"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError("GA resume row is missing: " + ", ".join(missing))
        config = row["config"]
        entry = {
            "individual_config": config,
            "total_return": float(row["total_return"]),
            "calmar": float(row["calmar"]),
            "sharpe": float(row["sharpe"]),
            "annualized": float(row["annualized"]),
            "max_drawdown": float(row["max_drawdown"]),
            "average_exposure": float(row["average_exposure"]),
            "full_investment_contract_satisfied": row['full_investment_contract_satisfied'],
            "metrics": row['metrics'],
            "evaluation_elapsed_seconds": row['evaluation_elapsed_seconds'],
            "execution_diagnostics": row['execution_diagnostics'],
        }
        key = _ga_cache_key(config)
        previous = ga_cache.get(key)
        if previous is None or entry["calmar"] > previous["calmar"]:
            ga_cache[key] = entry
        last_generation = max(last_generation, int(row["generation"]))
    return ga_cache, last_generation


def _rebuild_ga_state(ga_cache: Mapping[str, dict]) -> dict:
    ranked = sorted(ga_cache.values(), key=_training_candidate_rank_key, reverse=True)
    return {
        "population": [],
        "hall_of_fame": [entry["individual_config"] for entry in ranked[:100]],
        "score_cache": {
            _config_key(entry["individual_config"]): float(entry["calmar"])
            for entry in ranked
        },
    }


_worker_episode_attachment: AttachedPreparedEpisode | None = None
_worker_episode: PreparedEpisode | None = None


def _worker_initializer(descriptor: SharedPreparedEpisodeDescriptor) -> None:
    if not isinstance(descriptor, SharedPreparedEpisodeDescriptor):
        raise TypeError("GA worker requires SharedPreparedEpisodeDescriptor")
    global _worker_episode_attachment, _worker_episode
    _worker_episode_attachment = descriptor.attach()
    _worker_episode = _worker_episode_attachment.episode


def _evaluate_individual(episode: PreparedEpisode, config: Mapping[str, object]) -> dict:
    started = time.perf_counter()
    schema = ActionSchema(
        factor_names=episode.factors.factor_names,
        filter_names=episode.factors.filter_names,
    )
    canonical, day_config = canonicalize_ga_genes(
        config, action_schema=schema
    )
    session = EpisodeSession(episode, action_schema=schema)
    trace = run_day_config_episode(
        session,
        lambda _observation: day_config,
        record_details=False,
    )
    full_investment = trace.full_investment_contract_satisfied
    if not full_investment:
        raise RuntimeError("GA trajectory violated the full-investment contract")
    metrics = trace.metrics
    if not all(math.isfinite(float(v)) for v in metrics.as_dict().values()):
        raise ValueError("non-finite GA metrics")
    return {
        "individual_config": canonical,
        "metrics": metrics.as_dict(),
        "total_return": float((trace.nav[-1] / trace.nav[0] - 1.0) * 100.0),
        "calmar": float(metrics.calmar),
        "average_exposure": float(np.mean(trace.exposure)),
        "full_investment_contract_satisfied": True,
        "sharpe": float(metrics.sharpe),
        "annualized": float(metrics.annualized_return * 100.0),
        "max_drawdown": float(-metrics.max_drawdown * 100.0),
        "evaluation_elapsed_seconds": time.perf_counter() - started,
        "execution_diagnostics": {
            "executed_sell_count": int(trace.executed_sell_count),
            "total_fees": trace.total_fees,
            "mean_gross_turnover_ratio": trace.sum_gross_turnover_ratio / len(trace.rewards),
            "mean_total_cost_ratio": trace.sum_total_cost_ratio / len(trace.rewards),
            "reward_sum": float(trace.rewards.sum()),
        },
    }


def _worker_evaluate(config: Mapping[str, object]) -> dict:
    if _worker_episode is None:
        raise RuntimeError("GA worker canonical episode is not initialized")
    return _evaluate_individual(_worker_episode, config)


def _eval_parallel(
    configs: list[dict],
    results: list[dict],
    ga_cache: dict[str, dict],
    pool,
    progress=None,
) -> None:
    for result in pool.imap_unordered(_worker_evaluate, configs, chunksize=1):
        results.append(result)
        ga_cache[_ga_cache_key(result["individual_config"])] = result
        if progress is not None:
            progress(len(ga_cache))


def _validate_canonical_ga_profile(profile_name: str) -> None:
    profile_name = resolve_profile_name(fallback=profile_name)
    factor_names = tuple(
        factor.__name__ for factor in get_profile_factor_classes(profile_name)
    )
    filter_names = tuple(
        factor.__name__ for factor in get_profile_filter_factor_classes(profile_name)
    )
    if factor_names != PRODUCTION_FACTOR_NAMES:
        raise ValueError("current4 factor vocabulary differs from ActionSchema")
    if filter_names != PRODUCTION_FILTER_NAMES:
        raise ValueError("current4 filter vocabulary differs from ActionSchema")


def _canonical_ga_config(
    config: Mapping[str, object],
    profile_name: str,
) -> dict:
    canonical, _ = canonicalize_ga_genes(
        config,
        profile_name=profile_name,
        action_schema=ActionSchema(),
    )
    return canonical


def _canonical_ga_metadata(
    *,
    profile_name: str,
    seed: int,
    runtime_path: Path,
    episode: PreparedEpisode,
) -> dict[str, object]:
    schema = ActionSchema()
    profile = get_profile(profile_name)
    return {
        "schema_version": "canonical-env-selected11-ga-v4-continuous",
        "profile": profile_name,
        "search_space_version": profile["search_space_version"],
        "seed": seed,
        "runtime_path": str(runtime_path.resolve()),
        "runtime_schema_hash": episode.runtime.manifest.schema_hash,
        "runtime_source_sha256": episode.runtime.manifest.source_sha256,
        "factor_schema_hash": episode.factors.schema_hash,
        "action_schema_hash": schema.schema_hash,
        "decision_start": str(episode.runtime.decision_dates[0]),
        "decision_end": str(episode.runtime.decision_dates[-1]),
        "objective": "complete-training-period-net-nav-calmar-only",
        "account_chain": "one-serial-empty-account-chain-per-individual",
        "parallelism": "independent-individuals-only",
        "full_investment_required": True,
        "holdout_access": "none",
    }


def _load_canonical_ga_resume(
    output_dir: Path,
    expected_metadata: Mapping[str, object],
) -> tuple[dict, int]:
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("GA resume directory is missing run_metadata.json")
    actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    if actual != dict(expected_metadata):
        raise ValueError("GA resume identity does not match the sealed run")
    return _rebuild_from_jsonl(output_dir)


def _canonical_result_row(generation: int, result: Mapping[str, object]) -> dict:
    return {
        "generation": generation,
        "config": result["individual_config"],
        "metrics": result["metrics"],
        "total_return": result["total_return"],
        "calmar": result["calmar"],
        "sharpe": result["sharpe"],
        "annualized": result["annualized"],
        "max_drawdown": result["max_drawdown"],
        "average_exposure": result["average_exposure"],
        "evaluation_elapsed_seconds": result["evaluation_elapsed_seconds"],
        "execution_diagnostics": result["execution_diagnostics"],
        "full_investment_contract_satisfied": True,
    }


def training_date_range(args, profile_name=DEFAULT_GA_PROFILE):
    """Resolve the full requested period from the selected evaluation protocol."""
    if getattr(args, "evaluation_splits", None):
        splits, _ = read_evaluation_splits(args.evaluation_splits)
        bounds = splits["train"]
    else:
        return get_profile_preload_range(profile_name)
    return tuple(map(date.fromisoformat, bounds))


def _run_ga(
    args,
    mode_config,
    backtest_datetime_list,
    profile_name: str = DEFAULT_GA_PROFILE,
    resume_dir: Path | None = None,
):
    """Search static current4 DayConfigs on one sealed training episode."""

    from multiprocessing import get_context

    if args.warm_start and (resume_dir is not None or args.continue_from):
        raise ValueError("--warm-start cannot be combined with resume/continue-from")
    profile_name = resolve_profile_name(fallback=profile_name)
    _validate_canonical_ga_profile(profile_name)
    if not backtest_datetime_list:
        raise ValueError("GA requires a non-empty sealed training calendar")
    population_size = int(mode_config["population_size"])
    generations = int(mode_config["generations"])
    if population_size <= 0 or generations <= 0:
        raise ValueError("population_size and generations must be positive")
    args.population_size, args.generations = population_size, generations
    split_path = getattr(args, "evaluation_splits", None)
    comparison_requested = bool(split_path)
    if args.workers is None:
        args.workers = DEFAULT_ROLLOUT_WORKERS
    if args.workers is not None and args.workers <= 0:
        raise ValueError("workers must be positive")
    seed = int(args.seed)
    _seed_ga_randomness(seed)
    runtime_path = Path(args.runtime_path or latest_runtime_npz_path())
    config_path = Path(args.config)
    args.runtime_path = str(runtime_path.resolve())
    requested_start, requested_end = backtest_datetime_list[0].date(), backtest_datetime_list[-1].date()
    if comparison_requested:
        requested_start, requested_end = training_date_range(args, profile_name)
    mode_name = "debug" if args.mode == "debug" else "ga"
    output_dir = resume_dir or _resolve_output_dir(args.output_dir, mode_name)
    if resume_dir is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"GA output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    owns_report = False
    if resume_dir is None:
        report_splits = read_evaluation_splits(split_path)[0] if comparison_requested else {
            'train': [requested_start.isoformat(), requested_end.isoformat()], 'validation': None, 'test': None}
        write_preparing_report(output_dir, algorithm='GA', total=generations, splits=report_splits)
        owns_report = True
    comparison = None
    resources = ExitStack()
    log_sink = (ga_logger.add(str(output_dir / "ga.log"), level="INFO")
                if owns_report else None)
    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        episode = prepare_episode_from_runtime(
            runtime_path,
            requested_start,
            requested_end,
            lookback=int(args.lookback),
            prefilter_n=prefilter_n_from_config(config_payload),
            encode_observations=False,
        )
        metadata = _canonical_ga_metadata(
            profile_name=profile_name,
            seed=seed,
            runtime_path=runtime_path,
            episode=episode,
        )
        if comparison_requested:
            if resume_dir is not None or args.candidate_configs:
                raise ValueError("comparison requires a new GA run or explicit continuation")
            from ai.ga.comparison import Comparison
            comparison = Comparison(args, output_dir, episode, _evaluate_individual, resources=resources)
            metadata["comparison_identity"] = comparison.identity
            metadata["holdout_access"] = "serial_generation_champion_validation_and_diagnostic_test"
        if resume_dir is None:
            atomic_write_json(output_dir / "run_metadata.json", metadata, sort_keys=False)
            ga_cache: dict[str, dict] = {}
            start_generation = 0
            if args.continue_from:
                parent_dir = Path(args.continue_from)
                parent_metadata = json.loads((parent_dir / "run_metadata.json").read_text(encoding="utf8"))
                metadata_changes = {"comparison_identity", "holdout_access"}
                for key, value in metadata.items():
                    if key not in metadata_changes and parent_metadata[key] != value:
                        raise ValueError(f"GA continuation changed {key}")
                ga_cache, last_generation = _rebuild_from_jsonl(parent_dir)
                if not ga_cache:
                    raise ValueError("empty GA continuation")
                start_generation = last_generation + 1
                if comparison is None or start_generation != comparison.completed_generation:
                    raise ValueError("continuation generation identity mismatch")
                if start_generation >= int(mode_config["generations"]):
                    raise ValueError("continuation budget already completed")
                (output_dir / "all_results.jsonl").write_bytes((parent_dir / "all_results.jsonl").read_bytes())
        else:
            ga_cache, last_generation = _load_canonical_ga_resume(
                output_dir,
                metadata,
            )
            start_generation = last_generation + 1
            owns_report = True
            log_sink = ga_logger.add(str(output_dir / "ga.log"), level="INFO")

        candidate_path = args.candidate_configs
        if candidate_path:
            next_configs = _load_candidate_configs(candidate_path, profile_name)
            generations = start_generation + 1
        elif resume_dir is not None or args.continue_from:
            next_configs = ga_optimizer(
                [],
                state=_rebuild_ga_state(ga_cache),
                population_size=population_size,
                hall_of_fame_size=population_size,
                profile_name=profile_name,
                ga_cache=ga_cache,
                gen=max(start_generation - 1, 0),
            )
        elif args.warm_start:
            next_configs = _load_candidate_configs(args.warm_start, profile_name)
            if comparison is not None:
                if file_sha256(args.warm_start) != comparison.identity['initialization']['candidate_file_sha256']:
                    raise ValueError("warm-start candidate file changed after identity sealing")
        else:
            next_configs = generate_initial_configs(2 * population_size, profile_name)

        state = _rebuild_ga_state(ga_cache)
        worker_count = int(args.workers)
        if worker_count <= 0:
            raise ValueError("workers must be positive")

        with ResidentPreparedEpisode(episode) as owner:
            episode = owner.episode
            if comparison is not None:
                comparison.episodes['train'] = episode
            context = get_context("spawn")
            with context.Pool(
                processes=worker_count,
                initializer=_worker_initializer,
                initargs=(owner.descriptor,),
            ) as pool:
                for generation in range(start_generation, generations):
                    generation_started = time.perf_counter()
                    canonical_configs = [
                        _canonical_ga_config(config, profile_name)
                        for config in next_configs
                    ]
                    unique_configs = {
                        _ga_cache_key(config): config for config in canonical_configs
                    }
                    missing = [
                        config
                        for key, config in unique_configs.items()
                        if key not in ga_cache
                    ]
                    results = [
                        ga_cache[key] for key in unique_configs if key in ga_cache
                    ]
                    if missing:
                        _eval_parallel(missing, results, ga_cache, pool,
                            comparison.candidate_progress if comparison else None)
                    if not results:
                        raise RuntimeError("GA generation produced no results")
                    if not all(
                        result.get("full_investment_contract_satisfied") is True
                        for result in results
                    ):
                        raise RuntimeError(
                            "GA generation contains a non-full-investment result"
                        )
                    _append_jsonl_rows(
                        output_dir / "all_results.jsonl",
                        [
                            _canonical_result_row(generation, result)
                            for result in results
                        ],
                    )
                    best = max(results, key=_training_candidate_rank_key)
                    write_generation_diagnostics(output_dir, generation, results,
                                                 len(canonical_configs), len(missing),
                                                 time.perf_counter() - generation_started)
                    if comparison is not None:
                        comparison.evaluate(generation, best, len(ga_cache))
                    write_ga_report(output_dir, total_generations=generations,
                                    generation=generation + 1, best=best)
                    ga_logger.info(
                        f"generation={generation} candidates={len(results)} "
                        f"calmar={best['calmar']:.4f} full-investment=PASS"
                    )
                    if candidate_path:
                        break
                    next_configs = ga_optimizer(
                        results,
                        state=state,
                        population_size=population_size,
                        hall_of_fame_size=population_size,
                        profile_name=profile_name,
                        ga_cache=ga_cache,
                        gen=generation,
                    )

        winner = _save_training_candidate(
            output_dir,
            profile_name,
            ga_cache,
            prefilter_n=prefilter_n_from_config(config_payload),
        )
        if winner is None:
            raise RuntimeError("GA finished without a training winner")
        if comparison is not None:
            comparison.finish()
        write_ga_report(output_dir, total_generations=generations,
                        generation=generation + 1, best=best, complete=True)
        return winner
    except BaseException as error:
        if owns_report:
            mark_training_failed(output_dir, error)
        if comparison is not None:
            comparison.fail(error)
        raise
    finally:
        if comparison is not None:
            comparison.episodes.clear()
        resources.close()
        if log_sink is not None:
            ga_logger.remove(log_sink)


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="WBR production-factor GA parameter search")
    parser.add_argument("--mode", default="ga", choices=("debug", "ga"))
    parser.add_argument("--output-dir")
    parser.add_argument("--evaluation-splits", help="JSON with explicit train/validation/test ISO date pairs")
    parser.add_argument("--eval-every-generations", type=int, default=DEFAULT_EVALUATION_EVERY)
    parser.add_argument("--continue-from", help="Explicit training-cache migration; breeder RNG restarts, not exact resume")
    parser.add_argument("--runtime", dest="runtime_path")
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--workers", type=int, default=DEFAULT_ROLLOUT_WORKERS)
    parser.add_argument("--warm-start")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--profile", choices=(DEFAULT_GA_PROFILE,), default=DEFAULT_GA_PROFILE)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_GA_SEED)
    parser.add_argument(
        "--candidate-configs",
        help="evaluate a complete current4 config list in debug mode",
    )
    args = parser.parse_args(argv)
    if args.warm_start and (args.resume or args.continue_from):
        parser.error("--warm-start cannot be combined with resume/continue-from")
    if args.candidate_configs and args.mode != "debug":
        parser.error("--candidate-configs requires --mode debug")
    if args.candidate_configs and (args.resume or args.warm_start):
        parser.error("--candidate-configs cannot be combined with resume/warm-start")
    if args.generations is not None and args.generations < 1:
        parser.error("--generations must be positive")
    if args.population_size is not None and args.population_size < 2:
        parser.error("--population-size must be at least two")
    if args.mode == "ga" and not args.evaluation_splits:
        args.evaluation_splits = str(DEFAULT_EVALUATION_SPLITS_PATH)
    return args


def main():
    started = datetime.now()
    args = parse_args()

    profile_name = resolve_profile_name(fallback=args.profile)
    mode_config = get_mode_configs(profile_name)[args.mode].copy()
    if args.generations is not None:
        mode_config["generations"] = args.generations
    if args.population_size is not None:
        mode_config["population_size"] = args.population_size

    ga_logger.remove()
    ga_logger.add(sys.stderr, level=mode_config["log_level"])
    ga_logger.info(f"mode={args.mode}: {mode_config['desc']}")
    start_date, end_date = training_date_range(args, profile_name)
    ga_logger.info(
        f"GA sealed range: {start_date:%Y%m%d} - {end_date:%Y%m%d}"
    )
    dates = [
        datetime.combine(day, datetime.min.time())
        for day in get_trading_date_span(start_date, end_date)
    ]
    histories = {
        factor.__name__: factor().hist_days
        for factor in get_profile_factor_classes(profile_name)
    }
    ga_logger.info(
        "factor history: "
        + ", ".join(f"{name}={days}" for name, days in histories.items())
    )

    resume_dir = None
    if args.resume:
        resume_dir = _find_latest_ga_dir() if args.resume == "auto" else Path(args.resume)
        if resume_dir is None or not resume_dir.is_dir():
            raise ValueError("--resume did not resolve to a GA run directory")

    with keep_windows_awake() as awake:
        if awake:
            ga_logger.info("Windows sleep prevention enabled")
        result = _run_ga(
            args,
            mode_config,
            dates,
            profile_name=profile_name,
            resume_dir=resume_dir,
        )
    ga_logger.info(
        f"elapsed={(datetime.now() - started).total_seconds():.2f}s"
    )
    return result


if __name__ == "__main__":
    main()

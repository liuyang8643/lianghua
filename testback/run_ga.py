import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

DEFAULT_GA_SEED = 20260720
TRAIN_METRIC_VERSION = 'initial_nav_1_calendar_folds_v2'
FACTOR_RANKING_VERSION = 'strategy_pool_v1'
SELECTION_SLEEVES_VERSION = 'ordered_fixed_slots_v1'
EXECUTION_COST_VERSION = 'simulated_slippage_bps_v1'
EXECUTION_SEMANTIC_DEFAULTS = {
    'slippage_bps': 10.0,
    'rebalance_band_pct': 0.01,
}
CANDIDATE_SEMANTIC_FIELDS = list(EXECUTION_SEMANTIC_DEFAULTS)
_HOLDOUT_RESULT_FIELDS = (
    'val_calmar', 'val_sharpe', 'val_annualized', 'val_max_drawdown',
    'test_calmar', 'test_sharpe', 'test_annualized', 'test_max_drawdown',
)


from core.scoring import (
    FactorScoreMatrices,
    factor_scores_to_rank_matrix,
    scores_to_ranks,
    top_level_factor_filter_masks,
)
from core.runtime import load_runtime_npz
from utils.stock.time import get_trading_date_span
from utils.windows_awake import keep_windows_awake
from core.ga import (
    DEFAULT_GA_PROFILE,
    build_individual_config,
    generate_initial_configs,
    get_intrinsic_params,
    get_mode_configs,
    get_profile,
    get_profile_factor_classes,
    get_profile_preload_range,
    get_profile_search_spaces,
    get_profile_weight_search_spaces,
    resolve_profile_name,
    sample_factor_choice,
    get_profile_filter_factor_classes,
    get_config_param,
)

from datetime import date, datetime
from testback.logger import testback_logger
from core.backtest import (
  _backtest_direct,
  _compute_list_dates, _compute_timing_multipliers,
  _format_pool, _format_timing,
  _resolve_output_dir,
  stock_pool_prefixes,
)
from core.metrics import compute_core_metrics
from core.runtime import load_runtime_stock_codes
from core.strategy_config import strategy_config_payload


# ========== GA 核心函数 ==========

def _seed_ga_randomness(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _seed_ga_generation(seed: int, generation: int) -> None:
    """Seed breeding after a generation so resume rebuilds the same children."""
    _seed_ga_randomness(seed + generation + 1)


def _validate_resume_metadata(
    metadata: dict, *, profile_name: str, seed: int,
    sealed_holdout: bool, split_period_results: bool,
    training_objective: dict,
    ranking_pool: tuple[str, ...] | None = None,
    selection_sleeves: list[dict] | None = None,
) -> None:
    """Reject resumes that change the experiment identity or holdout contract."""
    if metadata.get('profile') != profile_name:
        raise ValueError(
            f"恢复目录 profile={metadata.get('profile')}，当前 profile={profile_name}，禁止混用"
        )
    if metadata.get('seed') != seed:
        raise ValueError(
            f"恢复目录随机种子={metadata.get('seed')}，当前 --seed={seed}，禁止混用"
        )
    if metadata.get('training_objective') != training_objective:
        raise ValueError('恢复目录 training_objective 与当前 profile 不一致')
    if metadata.get('sealed_holdout') != bool(sealed_holdout):
        raise ValueError(
            '恢复目录 sealed_holdout 与当前 --sealed-holdout 状态不一致'
        )
    if (
        metadata.get('test_period_is_strictly_sealed')
        != bool(sealed_holdout)
    ):
        raise ValueError(
            '恢复目录 test_period_is_strictly_sealed 与 sealed_holdout 不一致'
        )
    if metadata.get('split_period_results') != bool(split_period_results):
        raise ValueError(
            '恢复目录 split_period_results 与当前 --split-period-results 状态不一致'
        )
    if metadata.get('train_metric_version') != TRAIN_METRIC_VERSION:
        raise ValueError(
            '恢复目录训练指标口径不是 initial-NAV-1 当前版本，禁止混用'
        )
    if metadata.get('execution_cost_version') != EXECUTION_COST_VERSION:
        raise ValueError('恢复目录模拟滑点成本口径不是当前版本，禁止混用')
    if metadata.get('candidate_semantic_fields') != CANDIDATE_SEMANTIC_FIELDS:
        raise ValueError('恢复目录候选回测语义字段与当前版本不一致')
    if metadata.get('semantic_defaults') != EXECUTION_SEMANTIC_DEFAULTS:
        raise ValueError('恢复目录候选回测语义默认值与当前版本不一致')
    if ranking_pool is not None:
        if metadata.get('factor_ranking_version') != FACTOR_RANKING_VERSION:
            raise ValueError(
                '恢复目录因子排名口径不是当前策略股票池版本，禁止混用'
            )
        metadata_pool = metadata.get('factor_ranking_pool')
        if metadata_pool is None:
            raise ValueError('恢复目录缺少 factor_ranking_pool，禁止混用')
        if _normalize_ranking_pool(metadata_pool) != _normalize_ranking_pool(
            ranking_pool
        ):
            raise ValueError(
                '恢复目录 factor_ranking_pool 与当前 profile 不一致'
            )
    if selection_sleeves is not None:
        if (
            metadata.get('selection_sleeves_version')
            != SELECTION_SLEEVES_VERSION
        ):
            raise ValueError('恢复目录选股袖套口径不是当前固定槽位版本')
        if metadata.get('selection_sleeves') != selection_sleeves:
            raise ValueError('恢复目录 selection_sleeves 与当前 profile 不一致')

def _config_key(config: dict) -> tuple:
    def freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((k, freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze(v) for v in value)
        return value

    semantic_config = {
        key: value for key, value in config.items()
        if key != 'neighborhood_change'
    }
    for key, default in EXECUTION_SEMANTIC_DEFAULTS.items():
        value = semantic_config.get(key, default)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
        semantic_config[key] = value
    return freeze(semantic_config)


def _ga_cache_key(config: dict) -> str:
    """Return the canonical key used by live and resumed GA caches."""
    import json as json_mod

    return json_mod.dumps(_config_key(config), ensure_ascii=False)


def _without_holdout_results(entry: dict) -> dict:
    """Keep a training result free of validation/test diagnostics."""
    return {
        key: value for key, value in entry.items()
        if key not in _HOLDOUT_RESULT_FIELDS
    }


def _training_jsonl_entry(
    entry: dict, *, training_objective: dict,
    split_period_results: bool,
) -> dict:
    """Enforce that robust/split training rows contain training metrics only."""
    if (
        training_objective.get('mode') == 'robust_calmar'
        or split_period_results
    ):
        return _without_holdout_results(entry)
    return entry


def _build_run_metadata(
    *, profile_name: str, seed: int, sealed_holdout: bool,
    split_period_results: bool, training_objective: dict,
    ranking_pool: tuple[str, ...] | None = None,
    selection_sleeves: list[dict] | None = None,
) -> dict:
    metadata = {
        'profile': profile_name,
        'seed': seed,
        'sealed_holdout': sealed_holdout,
        'split_period_results': split_period_results,
        'period_result_files': (
            {
                'training': 'training_results.jsonl',
                'validation': 'validation_results.jsonl',
                'test': 'test_results.jsonl',
            }
            if split_period_results else None
        ),
        'final_candidate_diagnostic_files': (
            None
            if split_period_results
            else {
                'training': 'training_diagnostics.json',
                'validation': 'validation_diagnostics.json',
                'test': 'test_diagnostics.json',
            }
        ),
        'selection_uses_training_only': True,
        'test_period_is_strictly_sealed': sealed_holdout,
        'train_metric_version': TRAIN_METRIC_VERSION,
        'execution_cost_version': EXECUTION_COST_VERSION,
        'candidate_semantic_fields': list(CANDIDATE_SEMANTIC_FIELDS),
        'semantic_defaults': dict(EXECUTION_SEMANTIC_DEFAULTS),
        'execution_cost_semantics': {
            'slippage_bps': (
                'total simulated slippage charged independently on each '
                'buy and sell side; 10 means 10 bps per side'
            ),
            'rebalance_band_pct': (
                'relative no-trade band around each target position; '
                '0.01 means a 1% band'
            ),
            'candidate_values_source': (
                'each candidate config; semantic_defaults apply when absent'
            ),
        },
        'training_objective': training_objective,
        'legacy_resume_without_seed': False,
    }
    if ranking_pool is not None:
        metadata['factor_ranking_version'] = FACTOR_RANKING_VERSION
        metadata['factor_ranking_pool'] = list(
            _normalize_ranking_pool(ranking_pool)
        )
    if selection_sleeves is not None:
        metadata['selection_sleeves_version'] = SELECTION_SLEEVES_VERSION
        metadata['selection_sleeves'] = selection_sleeves
    return metadata


def _load_resume_metadata(
    output_dir: Path, *, profile_name: str, seed: int,
    sealed_holdout: bool, split_period_results: bool,
    training_objective: dict,
    ranking_pool: tuple[str, ...] | None = None,
    selection_sleeves: list[dict] | None = None,
) -> dict:
    """Load and validate experiment identity before any JSONL is restored."""
    import json as json_mod

    metadata_path = Path(output_dir) / 'run_metadata.json'
    if not metadata_path.is_file():
        raise ValueError(
            f'恢复目录缺少 run_metadata.json，禁止恢复: {output_dir}'
        )
    metadata = json_mod.loads(metadata_path.read_text(encoding='utf-8'))
    _validate_resume_metadata(
        metadata,
        profile_name=profile_name,
        seed=seed,
        sealed_holdout=sealed_holdout,
        split_period_results=split_period_results,
        training_objective=training_objective,
        ranking_pool=ranking_pool,
        selection_sleeves=selection_sleeves,
    )
    return metadata


def _write_frozen_candidate_diagnostics(
    output_dir: Path, *, frozen_entry: dict,
    validation_result: dict | None, test_result: dict | None,
) -> None:
    """Persist one train-selected candidate in three period-specific files."""
    import json as json_mod

    candidate_id = _ga_cache_key(frozen_entry['individual_config'])
    metric_fields = (
        'fitness', 'raw_fitness', 'calmar', 'fold_calmars', 'sharpe',
        'annualized', 'max_drawdown', 'total_return', 'average_exposure',
        'exposure_constraint_passed',
    )

    def payload(period: str, result: dict | None) -> dict:
        metrics = None
        if result is not None:
            metrics = {
                field: result[field]
                for field in metric_fields
                if field in result
            }
        return {
            'period': period,
            'candidate_id': candidate_id,
            'metrics': metrics,
        }

    period_results = {
        'training': frozen_entry,
        'validation': validation_result,
        'test': test_result,
    }
    for period, result in period_results.items():
        path = Path(output_dir) / f'{period}_diagnostics.json'
        path.write_text(
            json_mod.dumps(
                payload(period, result), ensure_ascii=False, indent=2,
            ),
            encoding='utf-8',
        )


def _run_frozen_holdout_reports(
    output_dir: Path,
    config_path: Path,
    individual_config: dict,
) -> dict[str, dict]:
    """Run the two holdout periods only after the train candidate is fixed."""
    import json as json_mod
    import subprocess
    import sys

    from core.runtime import latest_runtime_npz_path

    with np.load(latest_runtime_npz_path(), allow_pickle=False) as runtime:
        runtime_dates = runtime['trade_dates']
        if runtime_dates.ndim != 1 or runtime_dates.size == 0:
            raise ValueError('runtime 缺少测试集末日')
        test_end = str(runtime_dates[-1].astype('datetime64[D]'))

    periods = {
        'validation': ('2019-01-01', '2022-12-31'),
        'test': ('2023-01-01', test_end),
    }
    results = {}
    for period, (start_date, end_date) in periods.items():
        period_dir = Path(output_dir) / f'{period}_report'
        command = [
            sys.executable,
            '-u',
            '-m',
            'testback.run_backtest',
            '--individual-config',
            str(config_path),
            '--start-date',
            start_date,
            '--end-date',
            end_date,
            '--no-live-sim',
            '--output-dir',
            str(period_dir),
        ]
        subprocess.run(command, check=True)
        record_path = period_dir / 'record.json'
        if not record_path.is_file():
            raise FileNotFoundError(
                f'{period} 单回测未生成 record.json: {record_path}'
            )
        record = json_mod.loads(record_path.read_text(encoding='utf-8'))
        daily_returns = np.asarray(record.get('daily_returns', []), dtype=float)
        if daily_returns.size < 2 or not np.all(np.isfinite(daily_returns)):
            raise ValueError(f'{period} 单回测日收益为空或包含非有限值')
        calmar, metrics = _calmar_from_returns(daily_returns)
        total_return = (
            float(np.prod(1.0 + daily_returns / 100.0) - 1.0) * 100.0
        )
        results[period] = {
            'calmar': calmar,
            'sharpe': float(metrics['sharpe']),
            'annualized': float(metrics['annualized']),
            'max_drawdown': float(metrics['max_drawdown']),
            'total_return': total_return,
            'individual_config': individual_config,
            'artifact': str(period_dir),
            'start': record['period']['start'],
            'end': record['period']['end'],
            'trading_days': int(daily_returns.size),
        }
    return results


def _training_candidate_rank_key(entry: dict) -> tuple:
    """Rank candidates using training metrics and canonical config only."""
    fitness = entry.get('fitness')
    if fitness is None:
        fitness = entry.get('calmar')
    if fitness is None:
        fitness = entry.get('sharpe')
    if fitness is None:
        raise ValueError('训练候选缺少 fitness/calmar/sharpe')
    fold_calmars = entry.get('fold_calmars') or ()
    worst_fold = min(fold_calmars) if fold_calmars else float('-inf')

    def metric(name: str) -> float:
        value = entry.get(name)
        return float(value) if value is not None else float('-inf')

    return (
        float(fitness),
        float(worst_fold),
        metric('calmar'),
        metric('sharpe'),
        metric('annualized'),
        metric('max_drawdown'),
        _ga_cache_key(entry['individual_config']),
    )


def _select_training_candidate(ga_cache: dict) -> dict | None:
    """Select one deterministic deployable candidate from training only."""
    eligible = [
        entry for entry in ga_cache.values()
        if entry.get('fitness', entry.get('calmar')) is not None
    ]
    return max(eligible, key=_training_candidate_rank_key) if eligible else None


def _save_training_candidate(
    output_dir: Path,
    profile_name: str,
    ga_cache: dict,
) -> dict | None:
    """Persist the deterministic global training winner."""
    import json as json_mod

    training_candidate = _select_training_candidate(ga_cache)
    if training_candidate is None:
        return None
    best_result = strategy_config_payload(
        profile_name,
        training_candidate['individual_config'],
    )
    with open(
        Path(output_dir) / 'best_individual_config.json',
        'w',
        encoding='utf-8',
    ) as f:
        json_mod.dump(best_result, f, indent=2, ensure_ascii=False)
    return training_candidate


def _period_result_entry(generation: int, result: dict) -> dict:
    """Return one diagnostics-only validation/test JSONL row."""
    required = (
        'calmar', 'sharpe', 'annualized', 'max_drawdown', 'total_return',
        'individual_config',
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f'周期结果缺少字段: {missing}')
    return {
        'generation': int(generation),
        'calmar': result['calmar'],
        'sharpe': result['sharpe'],
        'annualized': result['annualized'],
        'max_drawdown': result['max_drawdown'],
        'total_return': result['total_return'],
        'average_exposure': result.get('average_exposure'),
        'config': result['individual_config'],
    }


def _append_jsonl_rows(path: Path, rows: list[dict]) -> None:
    """Append JSONL rows without mixing metrics from different periods."""
    import json as json_mod

    with open(path, 'a', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json_mod.dumps(row, ensure_ascii=False) + '\n')


def _validate_split_result_files(output_dir: Path) -> None:
    """Fail if the three period JSONLs are absent or misaligned in part."""
    import json as json_mod

    paths = [
        Path(output_dir) / 'training_results.jsonl',
        Path(output_dir) / 'validation_results.jsonl',
        Path(output_dir) / 'test_results.jsonl',
    ]
    exists = [path.exists() for path in paths]
    if not any(exists):
        return
    if not all(exists):
        raise ValueError('训练/验证/测试结果文件必须同时存在')

    identities = []
    for path in paths:
        rows = [
            json_mod.loads(line)
            for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        current = []
        for row in rows:
            if set(('generation', 'config')) - set(row):
                raise ValueError(f'{path.name} 缺少 generation/config')
            current.append((int(row['generation']), _config_key(row['config'])))
        identities.append(current)
    if not (identities[0] == identities[1] == identities[2]):
        raise ValueError('训练/验证/测试结果未按 generation/config 一一对齐')


def _period_results_for_configs(
    configs: list[dict], *, evaluation_cache: dict,
    info: dict, score_keys: set, valid_dates: list,
    date_indices: list, stock_indices: dict, all_valid_stocks: list,
    index_data: dict, list_dates_full: dict, logger, pool,
) -> list[dict]:
    """Evaluate and return one diagnostics row per config in input order."""
    missing_configs = []
    for config in configs:
        key = _ga_cache_key(config)
        if key not in evaluation_cache:
            missing_configs.append(config)

    if missing_configs:
        worker_args = [
            (
                info, score_keys, valid_dates, date_indices, stock_indices,
                all_valid_stocks, config, index_data, list_dates_full, {},
            )
            for config in missing_configs
        ]
        evaluated = []
        evaluated_cache = {}
        _eval_parallel(
            worker_args, evaluated, evaluated_cache, logger, pool=pool,
        )
        for result in evaluated:
            key = _ga_cache_key(result['individual_config'])
            evaluation_cache[key] = result

    ordered = []
    for config in configs:
        key = _ga_cache_key(config)
        if key not in evaluation_cache:
            raise RuntimeError('周期诊断结果未与配置一一对应')
        ordered.append(evaluation_cache[key])
    return ordered


def _load_candidate_configs(
    path: str | Path,
    profile_name: str | None = None,
) -> list[dict]:
    payload = __import__('json').loads(Path(path).read_text(encoding='utf-8'))
    inherit_profile_defaults = False
    if isinstance(payload, dict):
        inherit_profile_defaults = bool(
            payload.get('inherit_profile_defaults', False)
        )
        payload = payload.get('configs')
    if not isinstance(payload, list) or not payload:
        raise ValueError('--candidate-configs 必须是非空配置列表或包含 configs 的对象')
    configs = []
    for item in payload:
        config = item.get('individual_config') if isinstance(item, dict) else None
        config = config if config is not None else item
        if inherit_profile_defaults:
            if profile_name is None:
                raise ValueError('继承 profile 默认值时必须指定 profile')
            config = build_individual_config(
                profile_name=profile_name,
                **config,
            )
        configs.append(config)
    if any(not isinstance(config, dict) or 'weights' not in config for config in configs):
        raise ValueError('--candidate-configs 存在无效 individual_config')
    return configs


def _calmar_from_returns(daily_returns) -> tuple[float, dict]:
    metrics = compute_core_metrics(daily_returns)
    drawdown = abs(metrics['max_drawdown'])
    calmar = metrics['annualized'] / drawdown if drawdown != 0 else 0.0
    return float(calmar), metrics


def _training_fitness(
    daily_returns,
    objective: dict | None = None,
    dates=None,
) -> tuple[float, float, list[float]]:
    """Return full Calmar and a train-only robustness score."""
    daily = np.asarray(daily_returns, dtype=float)
    full_calmar, _ = _calmar_from_returns(daily)
    settings = objective or {}
    if settings.get('mode') != 'robust_calmar':
        return full_calmar, full_calmar, []

    calendar_folds = settings.get('calendar_folds')
    if calendar_folds is not None:
        if dates is None:
            raise ValueError('calendar_folds requires training dates')
        date_values = np.asarray(
            [
                np.datetime64(value.date() if hasattr(value, 'date') else value, 'D')
                for value in dates
            ],
            dtype='datetime64[D]',
        )
        if date_values.shape != daily.shape:
            raise ValueError('training dates must align one-to-one with daily returns')
        fold_chunks = []
        for bounds in calendar_folds:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError('each calendar fold must contain [start, end]')
            start, end = (np.datetime64(value, 'D') for value in bounds)
            if start > end:
                raise ValueError('calendar fold start must not exceed end')
            mask = (date_values >= start) & (date_values <= end)
            if int(mask.sum()) < 2:
                raise ValueError(
                    f'calendar fold {start}..{end} has fewer than two training days'
                )
            fold_chunks.append(daily[mask])
    else:
        folds = max(2, int(settings.get('folds', 3)))
        fold_chunks = [
            chunk for chunk in np.array_split(daily, folds) if len(chunk) >= 2
        ]
    fold_calmars = [_calmar_from_returns(chunk)[0] for chunk in fold_chunks]
    worst_fold = min(fold_calmars) if fold_calmars else full_calmar
    full_weight = float(settings.get('full_weight', 0.5))
    if not 0.0 <= full_weight <= 1.0:
        raise ValueError('training objective full_weight must be in [0, 1]')
    fitness = full_weight * full_calmar + (1.0 - full_weight) * worst_fold
    return float(full_calmar), float(fitness), [float(value) for value in fold_calmars]


def _apply_exposure_constraint(raw_fitness, average_exposure, objective):
    minimum = (objective or {}).get('min_average_exposure')
    if minimum is None or average_exposure >= float(minimum):
        return float(raw_fitness), True
    return float(-1000.0 + average_exposure), False


def _format_config_params(config: dict, profile_name: str) -> str:
    """动态格式化所有搜索参数，驱动自 intrinsic_params + search_spaces。"""
    spaces = get_profile_search_spaces(profile_name)
    parts = []
    for pdef in get_intrinsic_params():
        key = pdef['key']
        if key not in spaces:
            continue
        val = get_config_param(config, pdef)
        if val is None or (pdef['type'] == 'int' and val == 0):
            continue
        display = pdef['display']
        if isinstance(val, bool):
            parts.append(f"{display}={'ON' if val else 'OFF'}")
        elif isinstance(val, float):
            parts.append(f"{display}={val:.2f}")
        elif isinstance(val, list):
            parts.append(f"{display}={_format_pool(tuple(val))}")
        else:
            parts.append(f"{display}={val}")
    return ', '.join(parts)

def _format_best_log(best_cfg: dict, profile_name: str, prefix: str, train_fitness: float, avg_fitness: float,
                     sharpe=None, val_sharpe=None, test_sharpe=None, live_total=None, live_test=None):
    sorted_w = sorted(best_cfg['weights'].items(), key=lambda x: -abs(x[1]))
    w_str = ', '.join(f'{k}={v:.1f}' for k, v in sorted_w)
    params_str = _format_config_params(best_cfg, profile_name)
    msg = f"{prefix}: 训练Calmar={train_fitness:.3f}/{avg_fitness:.3f}"
    if sharpe is not None:
        msg += f" S={sharpe:.2f}"
    if val_sharpe is not None:
        msg += f" | 验证Calmar={val_sharpe:.3f}"
    if test_sharpe is not None:
        msg += f" | 测试Calmar={test_sharpe:.3f}"
    msg += f" | {params_str} | {w_str}"
    if live_total is not None:
        msg += f" | 实盘: total={live_total:.3f}"
        if live_test is not None:
            msg += f" test={live_test:.3f}"
    return msg


def ga_optimizer(results, state, population_size=24, hall_of_fame_size=24, profile_name=DEFAULT_GA_PROFILE, ga_cache=None, gen=0):
  import random

  results_list = list(results) if not isinstance(results, list) else results

  for r in results_list:
    key = _config_key(r['individual_config'])
    state['fitness_cache'][key] = r.get('fitness', r.get('calmar', r.get('sharpe', -1000.0)))

  if not state['population']:
    results_list.sort(key=_training_candidate_rank_key, reverse=True)
    state['population'] = [r['individual_config'] for r in results_list[:population_size]]

  def get_fitness(config):
    key = _config_key(config)
    return state.get('fitness_cache', {}).get(key)

  # GA 多样性参数
  elite_frac = 0.10
  tournament_k = 5          # 中等选择压力 (3太弱/7过强在欺骗区)
  immigrant_frac = 0.15     # 中等移民率：每代保留足够随机移民，避免早熟收敛

  # 父代选择：少量真精英 + 锦标赛选择（从历史全局池抽，打破当代近亲繁殖、降低选择压力）
  if ga_cache and len(ga_cache) >= population_size:
    unique_cfgs = {}
    for key, val in ga_cache.items():
      cfg = val.get('individual_config')
      fit = val.get('fitness', val.get('calmar', val.get('sharpe', -1000.0)))
      if cfg is not None and fit > -900:
        unique_cfgs[key] = (cfg, fit)
    pool = sorted(
      unique_cfgs.values(),
      key=lambda x: (x[1], _ga_cache_key(x[0])),
      reverse=True,
    )
    total = len(pool)
    n_elite = min(max(2, int(round(elite_frac * population_size))), total)
    elite_cfgs = [cfg for cfg, _ in pool[:n_elite]]
    # 剩余父代用锦标赛选择：每次随机取 k 个，留适应度最高者，给次优个体繁殖机会
    n_tournament = population_size - n_elite
    selected = []
    for _ in range(n_tournament):
      aspirants = random.sample(pool, min(tournament_k, total))
      selected.append(max(
        aspirants,
        key=lambda x: (x[1], _ga_cache_key(x[0])),
      )[0])
    parents = elite_cfgs + selected
  else:
    population_with_fitness = [(ind, fit) for ind in state['population']
                               if (fit := get_fitness(ind)) is not None]
    if not population_with_fitness:
      population_with_fitness = [
        (r['individual_config'], r.get('fitness', r.get('calmar', r.get('sharpe', -1000.0))))
        for r in results_list
      ]
    population_with_fitness.sort(
      key=lambda x: (x[1], _ga_cache_key(x[0])),
      reverse=True,
    )
    parents = [ind for ind, _ in population_with_fitness[:population_size]]

  all_individuals = state['hall_of_fame'] + parents
  unique_dict = {}
  for ind in all_individuals:
    key = _config_key(ind)
    fitness = get_fitness(ind)
    if fitness is None:
      continue
    if key not in unique_dict or fitness > unique_dict[key][1]:
      unique_dict[key] = (ind, fitness)
  sorted_hof = sorted(
    unique_dict.values(),
    key=lambda x: (x[1], _ga_cache_key(x[0])),
    reverse=True,
  )
  state['hall_of_fame'] = [ind for ind, _ in sorted_hof[:hall_of_fame_size]]

  has_weight_search = get_profile_weight_search_spaces(profile_name) is not None
  profile = get_profile(profile_name)
  has_factor_choice = profile.get('factor_choice_space') is not None
  search_spaces = get_profile_search_spaces(profile_name)

  def _crossover_field(p1, p2, key, default=None):
    return (p1.get(key, default) if random.random() < 0.5 else p2.get(key, default))

  def crossover_config(p1, p2):
    kwargs = {}
    for pdef in get_intrinsic_params():
        key = pdef['key']
        if key in search_spaces:
            left = get_config_param(p1, pdef)
            right = get_config_param(p2, pdef)
            kwargs[key] = left if random.random() < 0.5 else right
    if has_factor_choice:
        kwargs['factor_choice'] = _crossover_field(p1, p2, 'factor_choice')
    crossed_weights = None
    if has_weight_search:
      crossed_weights = {}
      all_keys = sorted(set(p1.get('weights', {})) | set(p2.get('weights', {})))
      for k in all_keys:
        w1 = p1['weights'].get(k, 0.0)
        w2 = p2['weights'].get(k, 0.0)
        crossed_weights[k] = w1 if random.random() < 0.5 else w2
    kwargs['weights'] = crossed_weights
    kwargs['profile_name'] = profile_name
    return build_individual_config(**kwargs)

  weight_spaces = get_profile_weight_search_spaces(profile_name) if has_weight_search else {}

  def _collect_dims():
    dims = [p['key'] for p in get_intrinsic_params() if p['key'] in search_spaces]
    if has_factor_choice: dims.append('factor_choice')
    for k in weight_spaces: dims.append(f'weight_{k}')
    return dims

  import math
  _all_dims = _collect_dims()

  def mutate_config(config):
    if not _all_dims:
      return build_individual_config(config['buy_n'],
                                      weights=config.get('weights'), profile_name=profile_name)
    # 变异率退火：前期 25-35% 广泛探索，后期 8-15% 精细搜索
    t = gen / max(gen + 20, 1)
    lo = 0.25 - 0.17 * t
    hi = 0.35 - 0.20 * t
    n_mutate = max(1, min(math.ceil(random.uniform(lo, hi) * len(_all_dims)), len(_all_dims)))
    mutate_dims = set(random.sample(_all_dims, n_mutate))
    kwargs = {}
    for pdef in get_intrinsic_params():
        key = pdef['key']
        if key in mutate_dims:
            space = search_spaces.get(key)
            kwargs[key] = random.choice(space) if space else config.get(pdef['config_key'])
        else:
            kwargs[key] = get_config_param(config, pdef)
    if has_factor_choice:
        kwargs['factor_choice'] = sample_factor_choice(profile_name=profile_name) if 'factor_choice' in mutate_dims else config.get('factor_choice')
    mutated_weights = dict(config.get('weights', {}))
    for k in weight_spaces:
      if f'weight_{k}' in mutate_dims:
        mutated_weights[k] = random.choice(weight_spaces[k])
      elif k not in mutated_weights:
        mutated_weights[k] = 0.0
    kwargs['weights'] = mutated_weights
    kwargs['profile_name'] = profile_name
    return build_individual_config(**kwargs)

  # 随机移民：每代注入全新随机个体，维持探索、逃离局部盆地（不依赖现有父代）
  n_immigrants = min(max(1, int(round(immigrant_frac * population_size))), population_size)
  n_crossover_children = population_size - n_immigrants

  children = []
  while len(children) < n_crossover_children:
    if len(parents) == 1:
      child = mutate_config(parents[0])
    else:
      p1, p2 = random.sample(parents, 2)
      child = mutate_config(crossover_config(p1, p2))
    children.append(child)
  children.extend(generate_initial_configs(n_immigrants, profile_name=profile_name))
  state['population'] = parents + children

  next_configs = parents + children
  return next_configs


# ========== 工具函数 ==========


def _find_latest_ga_dir() -> Path | None:
  results_dir = Path('results')
  if not results_dir.is_dir():
    return None
  candidates = []
  for d in results_dir.iterdir():
    if not d.is_dir():
      continue
    if not (d.name.startswith('ga_') or d.name.startswith('debug_')):
      continue
    if (d / 'all_results.jsonl').exists():
      candidates.append((d.name, d))
  if not candidates:
    return None
  candidates.sort(reverse=True)
  return candidates[0][1]


def _trim_runtime_to_training_end(
  data: dict,
  training_end: np.datetime64,
) -> dict:
  """Remove every runtime row after the sealed training endpoint."""
  runtime_dates = np.asarray(data['trade_dates'])
  if runtime_dates.ndim != 1 or runtime_dates.size == 0:
    raise ValueError('training-only debug 的 runtime 交易日为空或不是一维')
  strict_end = np.datetime64(training_end, 'D')
  strict_end_index = int(np.searchsorted(
    runtime_dates,
    strict_end,
    side='right',
  ))
  if strict_end_index <= 0:
    raise ValueError('training-only debug 的 runtime 不包含训练日期')
  original_rows = len(runtime_dates)
  protected_stock_fields = {'stock_codes', 'stock_names', 'issue_price'}
  trimmed = {
    key: (
      value[:strict_end_index].copy()
      if (
        isinstance(value, np.ndarray)
        and value.ndim >= 1
        and value.shape[0] == original_rows
        and key not in protected_stock_fields
      )
      else value
    )
    for key, value in data.items()
  }
  if np.any(trimmed['trade_dates'] > strict_end):
    raise RuntimeError('training-only debug runtime 仍包含 holdout 日期')
  return trimmed


def _load_ga_index_data(valid_dates, profile_name: str) -> dict:
  """Load only benchmark/timing indexes that exist for the requested period."""
  from core.timing import load_index_open

  symbols = {'sh000001'}
  spaces = get_profile_search_spaces(profile_name)
  if 'timing_index' in spaces:
    symbols.update(spaces['timing_index'])
  result = {}
  for symbol in symbols:
    try:
      _, result[symbol] = load_index_open(symbol, valid_dates)
    except ValueError as exc:
      testback_logger.warning(f"跳过区间不完整的指数 {symbol}: {exc}")
  return result


_GA_BENCHMARKS = ('sh000905', 'sh000852')


def _compute_benchmark_metric(symbol: str, valid_dates) -> dict | None:
  """Compute an Open-to-Open Calmar diagnostic over available overlap."""
  from core.timing import load_index_open

  index_dates, index_open = load_index_open(symbol)
  requested = {np.datetime64(d.date(), 'D') for d in valid_dates}
  mask = np.array([d in requested for d in index_dates], dtype=bool)
  mask &= np.isfinite(index_open)
  values = index_open[mask].astype(float)
  dates = index_dates[mask]
  if len(values) < 2:
    return None
  daily = np.diff(values) / values[:-1] * 100.0
  calmar, _ = _calmar_from_returns(daily[np.isfinite(daily)])
  return {
    'calmar': calmar,
    'available_days': len(values),
    'requested_days': len(valid_dates),
    'start': str(dates[0]),
    'end': str(dates[-1]),
  }


def _log_benchmark_metrics(period_dates: dict[str, list]) -> None:
  """Log diagnostics once; benchmark values never enter GA selection."""
  from core.timing import INDEX_INFO

  testback_logger.info('同期指数 Calmar（Open-to-Open，仅展示，不参与 GA 适应度或候选选择）')
  for symbol in _GA_BENCHMARKS:
    period_parts = []
    for period_name, dates in period_dates.items():
      metric = _compute_benchmark_metric(symbol, dates)
      if metric is None:
        period_parts.append(f'{period_name}=N/A')
        continue
      coverage = f"{metric['available_days']}/{metric['requested_days']}天"
      if metric['available_days'] < metric['requested_days']:
        coverage += f", {metric['start']}~{metric['end']}"
      period_parts.append(f"{period_name}={metric['calmar']:.3f}({coverage})")
    testback_logger.info(f"  {INDEX_INFO[symbol]}: " + ' | '.join(period_parts))


def _rebuild_from_jsonl(output_dir: Path, require_train_fitness: bool = False) -> tuple[dict, int]:
  import json as _json
  ga_cache = {}
  last_gen = -1
  jsonl_path = output_dir / 'all_results.jsonl'
  if jsonl_path.exists():
    with open(jsonl_path, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        r = _json.loads(line)
        if require_train_fitness and ('fitness' not in r or 'fold_calmars' not in r):
          raise ValueError(
            f'恢复目录使用旧适应度口径，不能恢复到训练稳健目标: {output_dir}'
          )
        if require_train_fitness:
          contaminated = [
            field for field in _HOLDOUT_RESULT_FIELDS
            if r.get(field) is not None
          ]
          if contaminated:
            raise ValueError(
              f"恢复训练记录包含 holdout 结果: {', '.join(contaminated)}"
            )
        cfg = r['config']
        key = _ga_cache_key(cfg)
        fitness = r.get('fitness', r.get('calmar', r['sharpe']))
        previous = ga_cache.get(key)
        if previous is None or fitness > previous.get('fitness', previous['calmar']):
          ga_cache[key] = {
            'individual_config': cfg,
            'calmar': r.get('calmar', fitness),
            'fitness': fitness,
            'raw_fitness': r.get('raw_fitness', fitness),
            'average_exposure': r.get('average_exposure'),
            'exposure_constraint_passed': r.get('exposure_constraint_passed'),
            'fold_calmars': r.get('fold_calmars', []),
            'sharpe': r['sharpe'],
            'annualized': r.get('annualized', 0.0),
            'max_drawdown': r.get('max_drawdown', 0.0),
          }
        if r['generation'] > last_gen:
          last_gen = r['generation']
  return ga_cache, last_gen


def _rebuild_ga_state(ga_cache: dict) -> dict:
  """从 ga_cache 重建 ga_state（population + hall_of_fame + fitness_cache）。"""
  sorted_entries = sorted(
    ga_cache.values(),
    key=_training_candidate_rank_key,
    reverse=True,
  )
  hall_of_fame = [v['individual_config'] for v in sorted_entries[:100]]
  fitness_cache = {
    _config_key(v['individual_config']): v.get('fitness', v['calmar'])
    for v in sorted_entries
  }
  return {
    'population': [],
    'hall_of_fame': hall_of_fame,
    'fitness_cache': fitness_cache,
  }



# === SharedMemory 数据共享（Windows spawn，替代 memmap 文件 I/O） ===
_MEMMAP_DIRS: list = []
_SHM_BLOCKS: list = []  # (SharedMemory, np.ndarray) 引用，防止被 GC 回收

# Per-worker cache: pool initializer 打开一次，所有 eval 复用
_worker_shm_cache: dict = {}

def _worker_initializer(shm_entries):
  """Pool initializer：每个 worker 进程启动时打开 SharedMemory，缓存在模块全局。
  shm_entries: [(name, shm_name, shape, dtype_str), ...]"""
  from multiprocessing.shared_memory import SharedMemory
  global _worker_shm_cache
  _worker_shm_cache = {}
  for name, shm_name, shape, dtype_str in shm_entries:
    shm = SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    _worker_shm_cache[name] = (shm, arr)


def _normalize_ranking_pool(stock_pool) -> tuple[str, ...]:
  """Return a canonical, non-empty stock-pool prefix tuple."""
  prefixes = stock_pool_prefixes(stock_pool)
  normalized = tuple(sorted({str(prefix) for prefix in prefixes}))
  if not normalized or any(not prefix for prefix in normalized):
    raise ValueError('stock_pool 必须包含至少一个非空代码前缀')
  return normalized


def _resolve_profile_ranking_pool(profile_name: str) -> tuple[str, ...]:
  """Resolve the one pool for which GA may safely pre-rank factor scores."""
  profile = get_profile(profile_name)
  fixed_pool = profile.get('fixed_parameters', {}).get('stock_pool')
  if fixed_pool is not None:
    return _normalize_ranking_pool(fixed_pool)

  pool_space = profile.get('search_spaces', {}).get('stock_pool')
  if pool_space:
    distinct_pools = {
      _normalize_ranking_pool(candidate) for candidate in pool_space
    }
    if len(distinct_pools) != 1:
      raise ValueError(
        f'GA profile {profile_name} 会搜索多个 stock_pool，'
        'pool-aware 多池预排名尚未实现；请拆成 '
        'fixed_parameters.stock_pool 固定池 profile，禁止复用错误排名'
      )
    return next(iter(distinct_pools))

  return _normalize_ranking_pool(None)


def _validate_config_ranking_pool(
    config: dict, ranking_pool: tuple[str, ...],
) -> None:
  """Reject a candidate that does not match the shared pre-ranked pool."""
  candidate_pool = _normalize_ranking_pool(config.get('stock_pool'))
  expected_pool = _normalize_ranking_pool(ranking_pool)
  if candidate_pool != expected_pool:
    raise ValueError(
      f'候选 stock_pool={candidate_pool} 与 GA 共享预排名池='
      f'{expected_pool} 不一致，禁止复用错误因子排名'
    )

def _arrays_from_memmap(info: dict) -> dict:
  return {name: np.memmap(filepath, dtype=np.dtype(dtype_str), mode='r', shape=shape)
          for name, (filepath, shape, dtype_str) in info.items()}

def _cleanup_memmap():
  import shutil
  failures = []
  for d in list(_MEMMAP_DIRS):
    try:
      shutil.rmtree(d)
    except Exception as error:
      failures.append((d, error))
  _MEMMAP_DIRS[:] = [path for path, _ in failures]
  if failures:
    details = '; '.join(f'{path}: {error}' for path, error in failures)
    raise RuntimeError(f'GA memmap 临时目录清理失败: {details}')

def _cleanup_shm():
  """Clean every SharedMemory block and fail closed on any leak."""
  failures = []
  retry_blocks = []
  for block in list(_SHM_BLOCKS):
    shm, _ = block
    try:
      shm.close()
      shm.unlink()
    except Exception as error:
      failures.append((getattr(shm, 'name', '<unknown>'), error))
      retry_blocks.append(block)
  _SHM_BLOCKS[:] = retry_blocks
  if failures:
    details = '; '.join(
      f'{name}: {error}' for name, error in failures
    )
    raise RuntimeError(f'GA SharedMemory 清理失败: {details}')


def _cleanup_ga_resources():
  """Attempt both cleanup layers and surface every failure."""
  failures = []
  for label, cleanup in (
    ('SharedMemory', _cleanup_shm),
    ('memmap', _cleanup_memmap),
  ):
    try:
      cleanup()
    except Exception as error:
      failures.append((label, error))
  if failures:
    details = '; '.join(
      f'{label}: {error}' for label, error in failures
    )
    raise RuntimeError(f'GA 训练资源未完全释放: {details}')


def _factor_worker(args):
  """Compute one score matrix or one boolean filter mask into a memmap."""
  factor_cls, base_info, stock_codes, trade_dates, tmpdir, row_slice, is_filter, rank_cols = args
  data = _arrays_from_memmap(base_info)
  n_full = len(trade_dates)
  if row_slice is not None:
    r0, r1 = row_slice
    data = {k: (v[r0:r1] if v.ndim >= 2 else v) for k, v in data.items()}
    trade_dates = trade_dates[r0:r1]
  data['stock_codes'] = stock_codes
  data['trade_dates'] = trade_dates
  f = factor_cls()
  name = f.__class__.__name__
  with np.errstate(all='ignore'):
      raw = f.calc_batch(data)
  raw_valid = ~np.isnan(raw)
  if is_filter:
      values = np.isfinite(raw) & (raw > 0)
  else:
      rank_cols = np.asarray(rank_cols, dtype=np.intp)
      if rank_cols.ndim != 1 or rank_cols.size == 0:
        raise ValueError('GA 因子预排名股票池不能为空')
      if (
        np.unique(rank_cols).size != rank_cols.size
        or rank_cols.min() < 0
        or rank_cols.max() >= raw.shape[1]
      ):
        raise ValueError('GA 因子预排名股票列必须唯一且位于 runtime 范围内')
      values = factor_scores_to_rank_matrix(
          raw,
          rank_cols,
          scores_are_ranks=bool(getattr(f, 'scores_are_ranks', False)),
      )
  if row_slice is not None:
    fill_value = False if is_filter else np.nan
    full = np.full((n_full, values.shape[1]), fill_value, dtype=values.dtype)
    full[r0:r1] = values
    values = full
    full_valid = np.zeros((n_full, raw_valid.shape[1]), dtype=bool)
    full_valid[r0:r1] = raw_valid
    raw_valid = full_valid
  filepath = Path(tmpdir) / f'factor_{name}.bin'
  values.tofile(str(filepath))
  return name, (str(filepath), values.shape, str(values.dtype)), raw_valid


def _build_worker_filter_masks(all_arrays, score_keys, config):
  result = {}
  if config.get('selection_sleeves') is None:
    active_validity = [
        all_arrays[f'_factor_valid_{name}']
        for name in score_keys
        if config['weights'].get(name, 0.0) != 0.0
        and f'_factor_valid_{name}' in all_arrays
    ]
    if active_validity:
        result['_active_factor_intersection'] = np.logical_and.reduce(active_validity)
  for name, enabled in config.get('filter_factors', {}).items():
    if enabled and name in all_arrays:
      result[name] = np.asarray(all_arrays[name], dtype=bool)
  return result


def _worker_evaluate(args):
  global _worker_shm_cache
  train_info, score_keys, valid_dates, date_indices, stock_indices, \
      all_stocks_list, config, index_data, list_dates_map, training_objective = args

  ranking_pool = train_info['_ranking_pool_prefixes']
  _validate_config_ranking_pool(config, ranking_pool)
  factor_validity_keys = {f'_factor_valid_{name}' for name in score_keys}
  needed = score_keys | factor_validity_keys | set(config.get('filter_factors', {})) | {'open', 'close', 'high', 'low', 'preClose', 'volume', 'amount', 'total_share', 'st_mask', 'issue_price', 'stock_codes', 'trade_dates', '_market_open_index'}
  all_arrays = {k: arr for k, (shm, arr) in _worker_shm_cache.items() if k in needed}
  data = {
      k: v for k, v in all_arrays.items()
      if k not in score_keys
      and k not in factor_validity_keys
      and k not in config.get('filter_factors', {})
  }
  score_values = {k: v for k, v in all_arrays.items() if k in score_keys}
  factor_validity = {
      name: all_arrays[f'_factor_valid_{name}']
      for name in score_keys
      if f'_factor_valid_{name}' in all_arrays
  }
  all_scores = FactorScoreMatrices(
      score_values,
      factor_validity=(
          factor_validity if config.get('selection_sleeves') is not None
          else None
      ),
  )
  filter_masks = _build_worker_filter_masks(all_arrays, score_keys, config)

  pool_stocks = list(all_stocks_list)

  def _timing_for(dates, indices):
      base = _compute_timing_multipliers(config, dates, index_data)
      from core.trend_timing import compute_configured_timing_multipliers
      timing_filter_masks = top_level_factor_filter_masks(
          all_scores, filter_masks, config['weights'],
      )
      return compute_configured_timing_multipliers(
          data=data, all_scores=all_scores, valid_dates=dates,
          date_indices=indices, valid_stocks=pool_stocks,
          stock_indices=stock_indices, config=config,
          filter_masks=timing_filter_masks,
          base_multipliers=base,
      )

  hp = config.get('holding_period', 1)
  n_starts = hp if hp > 1 else 1

  calmar_list, raw_fitness_list, fold_lists = [], [], []
  sharpe_list, ann_list, dd_list, tr_list = [], [], [], []
  exposure_list = []
  for offset in range(n_starts):
      od = valid_dates[offset:]
      oi = date_indices[offset:]
      om = _timing_for(od, oi)
      r = _backtest_direct(
          data, all_scores, od, oi, pool_stocks, stock_indices,
          weights=config['weights'], buy_n=config['buy_n'], sell_m=config['sell_m'],
          holding_period=hp,
          position_multipliers=om if om is not None and len(om) == len(od) else None,
          list_dates_map=list_dates_map,
          lightweight=True, limit_up_protection=config.get('limit_up_protection', False),
          rebalance=config.get('rebalance', True),
          slippage_bps=config.get(
              'slippage_bps',
              EXECUTION_SEMANTIC_DEFAULTS['slippage_bps'],
          ),
          rebalance_band_pct=config.get(
              'rebalance_band_pct',
              EXECUTION_SEMANTIC_DEFAULTS['rebalance_band_pct'],
          ),
          filter_masks=filter_masks,
          selection_sleeves=config.get('selection_sleeves'))
      calmar, fitness, fold_calmars = _training_fitness(
          r['daily_returns'], training_objective, dates=od,
      )
      m = compute_core_metrics(r['daily_returns'])
      calmar_list.append(calmar)
      raw_fitness_list.append(fitness)
      fold_lists.append(fold_calmars)
      sharpe_list.append(m['sharpe'])
      ann_list.append(m['annualized'])
      dd_list.append(m['max_drawdown'])
      tr_list.append(r['total_return'])
      exposure_list.append(float(np.mean(r['daily_exposures'])))

  raw_fitness = float(np.mean(raw_fitness_list))
  average_exposure = float(np.mean(exposure_list))
  constrained_fitness, exposure_constraint_passed = _apply_exposure_constraint(
      raw_fitness, average_exposure, training_objective,
  )

  return {
    'individual_config': config,
    'total_return': float(np.mean(tr_list)),
    'calmar': float(np.mean(calmar_list)),
    'fitness': constrained_fitness,
    'raw_fitness': raw_fitness,
    'average_exposure': average_exposure,
    'exposure_constraint_passed': exposure_constraint_passed,
    'fold_calmars': (
      np.mean(np.asarray(fold_lists), axis=0).astype(float).tolist()
      if fold_lists and fold_lists[0] else []
    ),
    'sharpe': float(np.mean(sharpe_list)),
    'annualized': float(np.mean(ann_list)),
    'max_drawdown': float(np.mean(dd_list)),
    'cleared_positions_count': r['cleared_positions_count'],
  }


def _eval_parallel(worker_args, results_list, ga_cache, logger, pool):
  import gc

  for result in pool.imap_unordered(_worker_evaluate, worker_args, chunksize=1):
    results_list.append(result)
    key = _ga_cache_key(result['individual_config'])
    ga_cache[key] = result
  gc.collect()


def _run_ga(args, mode_config, backtest_datetime_list, all_stocks, profile_name=DEFAULT_GA_PROFILE, resume_dir=None):
  """GA/调试模式：多进程并行回测。预计算共享内存复用。"""
  import json as json_mod
  import os
  import pickle
  import time

  is_debug = args.mode == 'debug'
  generations = mode_config['generations']

  if is_debug:
    population_size = mode_config['population_size']
  else:
    population_size = mode_config['population_size'] or 50

  mode_name = 'debug' if is_debug else 'ga'
  output_dir = resume_dir or _resolve_output_dir(args.output_dir, mode_name)
  seed = int(getattr(args, 'seed', DEFAULT_GA_SEED))
  split_period_results = bool(
    getattr(args, 'split_period_results', False)
  )
  _seed_ga_randomness(seed)
  factor_classes = get_profile_factor_classes(profile_name)
  filter_factor_classes = get_profile_filter_factor_classes(profile_name)
  profile = get_profile(profile_name)
  ranking_pool = _resolve_profile_ranking_pool(profile_name)
  training_objective = profile.get('training_objective', {})
  profile_selection_sleeves = profile.get('fixed_parameters', {}).get(
    'selection_sleeves'
  )

  metadata_path = output_dir / 'run_metadata.json'
  sealed_holdout = bool(getattr(args, 'sealed_holdout', False))
  training_only_debug = bool(is_debug and sealed_holdout)
  training_only_runtime = sealed_holdout
  if resume_dir:
    metadata = _load_resume_metadata(
      output_dir,
      profile_name=profile_name,
      seed=seed,
      sealed_holdout=sealed_holdout,
      split_period_results=split_period_results,
      training_objective=training_objective,
      ranking_pool=ranking_pool,
      selection_sleeves=profile_selection_sleeves,
    )
    if bool(metadata.get('training_only_debug', False)) != training_only_debug:
      raise ValueError(
        '恢复目录 training_only_debug 与当前 debug/sealed 状态不一致'
      )
  else:
    metadata = _build_run_metadata(
      profile_name=profile_name,
      seed=seed,
      sealed_holdout=sealed_holdout,
      split_period_results=split_period_results,
      training_objective=training_objective,
      ranking_pool=ranking_pool,
      selection_sleeves=profile_selection_sleeves,
    )
    metadata['training_only_debug'] = training_only_debug
    metadata['holdout_access_mode'] = (
      'training_dates_only'
      if training_only_debug
      else (
        'candidate_metrics_after_training_freeze'
        if sealed_holdout
        else 'period_diagnostics_enabled'
      )
    )
    if training_only_debug:
      metadata['final_candidate_diagnostic_files'] = None
    metadata_path.write_text(
      json_mod.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8',
    )

  # 添加文件日志 sink，将所有控制台日志同步写入 log 文件
  log_path = output_dir / 'ga.log'
  testback_logger.add_file_sink(str(log_path))
  testback_logger.info(f"日志文件: {log_path}")
  testback_logger.info(f"GA 随机种子: {seed}")

  if split_period_results:
    _validate_split_result_files(output_dir)

  from multiprocessing import get_context

  ga_state = {'population': [], 'hall_of_fame': [], 'fitness_cache': {}}
  start_generation = 0
  ga_cache = {}

  val_cache_path = output_dir / 'val_cache.json'
  val_eval_cache = {}
  if not sealed_holdout and val_cache_path.exists():
    with open(val_cache_path, 'r', encoding='utf-8') as f:
      val_eval_cache = json_mod.load(f)
    testback_logger.info(f"加载验证集缓存: {len(val_eval_cache)} 条")

  test_cache_path = output_dir / 'test_cache.json'
  test_eval_cache = {}
  if not sealed_holdout and test_cache_path.exists():
    with open(test_cache_path, 'r', encoding='utf-8') as f:
      test_eval_cache = json_mod.load(f)
    testback_logger.info(f"加载测试集缓存: {len(test_eval_cache)} 条")


  t0 = time.time()

  def _build_date_subset(date_list, npz_dates, date_to_idx):
    """从全量 npz_dates 中提取目标日期段的 indices。纯索引，无 NPZ 加载无因子计算。"""
    date_indices = []
    valid_dates = []
    for dt in date_list:
      d = dt.date()
      di = date_to_idx.get(d)
      if di is None:
        continue
      date_indices.append(di)
      valid_dates.append(dt)
    return valid_dates, date_indices

  # === NPZ 加载 + 因子计算（一次性）===
  max_hist = max(
    f.hist_days
    for f in [*factor_classes, *filter_factor_classes]
  )
  runtime_lookback = max(max_hist, 512) if training_only_runtime else None
  data = load_runtime_npz(
    backtest_datetime_list,
    max_lookback=runtime_lookback,
    strict_end=training_only_runtime,
  )
  if data is None:
    first_d = backtest_datetime_list[0].strftime('%Y%m%d')
    last_d = backtest_datetime_list[-1].strftime('%Y%m%d')
    raise FileNotFoundError(f"未找到覆盖 {first_d}~{last_d} 的 runtime npz 文件")
  if training_only_runtime:
    strict_end = np.datetime64(
      max(item.date() for item in backtest_datetime_list),
      'D',
    )
    data = _trim_runtime_to_training_end(data, strict_end)
    sealed_mode_name = 'debug' if training_only_debug else 'GA'
    testback_logger.info(
      f'training-only {sealed_mode_name} runtime 截止 {strict_end}: '
      f'{len(data["trade_dates"])} 天，不构建 holdout 日期索引'
    )

  npz_stocks = [str(s) for s in data['stock_codes']]
  stock_indices = {c: i for i, c in enumerate(npz_stocks)}
  all_valid_stocks = [s for s in all_stocks if s in stock_indices]
  runtime_stock_count = len(all_valid_stocks)
  all_valid_stocks = [
    stock for stock in all_valid_stocks
    if stock.startswith(ranking_pool)
  ]
  if not all_valid_stocks:
    raise ValueError(
      f'GA 预排名池 {ranking_pool} 在 runtime 中没有任何股票'
    )
  testback_logger.info(
    f"GA 因子排名池={_format_pool(ranking_pool)}: "
    f"{len(all_valid_stocks)}/{runtime_stock_count} 只"
  )
  npz_dates = data['trade_dates']
  py_dates = [d.astype('datetime64[D]').item() for d in npz_dates]

  date_to_idx = {}
  for i, d in enumerate(npz_dates):
    date_to_idx[d.astype('datetime64[D]').item()] = i

  import tempfile
  tmpdir = tempfile.mkdtemp(prefix='memmap_', dir=str(output_dir))
  _MEMMAP_DIRS.append(tmpdir)
  scores_dir = Path(tmpdir) / "scores"
  scores_dir.mkdir(exist_ok=True)

  info = {}
  for k, v in data.items():
    if k == 'stock_names':
      continue
    arr = np.ascontiguousarray(v)
    filepath = Path(tmpdir) / f'{k}.bin'
    arr.tofile(str(filepath))
    info[k] = (str(filepath), arr.shape, str(arr.dtype))

  # 提前计算三段时间索引，确定因子计算所需的最小天数范围
  valid_dates, date_indices = _build_date_subset(backtest_datetime_list, npz_dates, date_to_idx)

  val_start = date(2019, 1, 1)
  val_end = date(2022, 12, 31)
  test_start = date(2023, 1, 1)
  test_end = None
  val_valid_dates = []
  val_date_indices = []
  test_valid_dates = []
  test_date_indices = []
  if not training_only_runtime:
    val_datetime_list = [
      datetime.combine(d, datetime.min.time())
      for d in get_trading_date_span(val_start, val_end)
    ]
    val_valid_dates, val_date_indices = _build_date_subset(
      val_datetime_list,
      npz_dates,
      date_to_idx,
    )

    # 测试集末日 = runtime npz 实际数据末日，避免硬编码过期。
    test_end = py_dates[-1]
    test_datetime_list = [
      datetime.combine(d, datetime.min.time())
      for d in get_trading_date_span(test_start, test_end)
    ]
    test_valid_dates, test_date_indices = _build_date_subset(
      test_datetime_list,
      npz_dates,
      date_to_idx,
    )

  all_idx = date_indices + val_date_indices + test_date_indices
  row_start = max(0, min(all_idx) - max_hist)
  row_end = max(all_idx) + 1
  n_needed = row_end - row_start
  testback_logger.info(f"因子计算范围: [{row_start}:{row_end}] = {n_needed} 天 (全量 {len(npz_dates)} 天, 截断 {len(npz_dates)-n_needed} 天)")

  # 统一打印当前 profile 搜索空间
  spaces = get_profile_search_spaces(profile_name)
  weight_spaces = get_profile_weight_search_spaces(profile_name)
  search_info = []
  for pdef in get_intrinsic_params():
      space = spaces.get(pdef['key'])
      if space:
          vals = [str(v) for v in space]
          search_info.append(f"{pdef['display']}=[{','.join(vals[:8])}{'...' if len(vals) > 8 else ''}]" if len(vals) <= 10 else f"{pdef['display']}={len(vals)}值")
  if profile.get('factor_choice_space'):
      search_info.append(f"fc={len(profile['factor_choice_space'])}值")
  if weight_spaces:
      for k, vals in weight_spaces.items():
          search_info.append(f"w_{k}=[{min(vals):.1f}-{max(vals):.1f}]")
  testback_logger.info(f"搜索空间: " + " | ".join(search_info))

  score_keys = set()
  t_f_all = time.time()
  base_info = {k: v for k, v in info.items() if k not in ('stock_codes', 'trade_dates')}
  row_slice = (row_start, row_end)
  all_worker_args = [
    (
      f_cls, base_info, npz_stocks, py_dates, str(scores_dir), row_slice,
      f_cls in filter_factor_classes,
      np.asarray([stock_indices[s] for s in all_valid_stocks], dtype=np.intp),
    )
    for f_cls in [*factor_classes, *filter_factor_classes]
  ]
  factor_names = {f.__name__ for f in factor_classes}
  for wargs in all_worker_args:
    name, entry, raw_valid = _factor_worker(wargs)
    info[name] = entry
    if name in factor_names:
      score_keys.add(name)
      validity_path = scores_dir / f'_factor_valid_{name}.bin'
      raw_valid.astype(bool, copy=False).tofile(str(validity_path))
      info[f'_factor_valid_{name}'] = (
        str(validity_path), raw_valid.shape, str(raw_valid.dtype),
      )
  info['_ranking_pool_prefixes'] = ranking_pool
  testback_logger.info(f"{len(factor_classes)} 因子计算完成 ({time.time() - t_f_all:.1f}s)")

  # 共用 list_dates
  list_dates_full = _compute_list_dates(npz_stocks, data['open'], npz_dates)
  from core.trend_timing import market_open_index
  market_index = market_open_index(data)
  market_index_path = Path(tmpdir) / '_market_open_index.bin'
  market_index.tofile(str(market_index_path))
  info['_market_open_index'] = (
    str(market_index_path), market_index.shape, str(market_index.dtype),
  )
  del data

  # === 构建 SharedMemory 供 worker 高速访问（替代 memmap 磁盘 I/O）===
  from multiprocessing.shared_memory import SharedMemory
  filter_mask_keys = {
    *(f.__name__ for f in filter_factor_classes),
    *(f'_factor_valid_{name}' for name in score_keys),
  }
  needed_shm = score_keys | filter_mask_keys | {'open', 'close', 'high', 'low', 'preClose', 'volume', 'amount', 'total_share', 'st_mask', 'issue_price', 'stock_codes', 'trade_dates', '_market_open_index'}
  shm_entries = []
  for name in needed_shm:
    if name not in info:
      continue
    filepath, shape, dtype_str = info[name]
    arr = np.memmap(filepath, dtype=np.dtype(dtype_str), mode='r', shape=shape)
    contig = np.ascontiguousarray(arr)
    shm = SharedMemory(create=True, size=contig.nbytes)
    shared_arr = np.ndarray(contig.shape, dtype=contig.dtype, buffer=shm.buf)
    shared_arr[:] = contig
    _SHM_BLOCKS.append((shm, shared_arr))
    shm_entries.append((name, shm.name, shape, dtype_str))
    del arr, contig, shared_arr
  shm_gb = sum(int(np.prod(e[2])) * np.dtype(e[3]).itemsize for e in shm_entries) / 1024**3
  testback_logger.info(f"SharedMemory 就绪: {len(shm_entries)} 数组 ({shm_gb:.1f} GB)")

  testback_logger.info(f"训练集就绪 ({time.time() - t0:.1f}s), {len(all_valid_stocks)} 只, {len(valid_dates)} 天")

  index_data = _load_ga_index_data(valid_dates, profile_name)

  import gc
  gc.collect()

  n_workers = 20
  ctx = get_context('spawn')
  ga_pool = ctx.Pool(processes=n_workers, initializer=_worker_initializer, initargs=(shm_entries,))
  testback_logger.info(f"多进程池已创建: {n_workers} workers")

  val_index_data = None
  test_index_data = None
  if training_only_debug:
    _log_benchmark_metrics({'训练': valid_dates})
  elif sealed_holdout:
    _log_benchmark_metrics({'训练': valid_dates})
    testback_logger.info(
      'sealed holdout: 验证/测试指数与候选指标将在训练候选冻结后加载'
    )
  else:
    val_index_data = _load_ga_index_data(val_valid_dates, profile_name)
    testback_logger.info(
      f"验证集就绪: {val_start} - {val_end}, {len(val_valid_dates)} 天"
    )
    test_index_data = _load_ga_index_data(test_valid_dates, profile_name)
    _log_benchmark_metrics({
      '训练': valid_dates,
      '验证': val_valid_dates,
      '测试': test_valid_dates,
    })
    testback_logger.info(
      f"测试集就绪: {test_start} - {test_end}, {len(test_valid_dates)} 天"
    )

  generation_results = []
  candidate_config_path = getattr(args, 'candidate_configs', None)
  if candidate_config_path:
    next_configs = _load_candidate_configs(
      candidate_config_path, profile_name=profile_name,
    )
    generations = 1
    testback_logger.info(f"训练候选批量评估: {len(next_configs)} 个配置")
  elif resume_dir:
    ga_cache, last_gen = _rebuild_from_jsonl(
      resume_dir,
      require_train_fitness=training_objective.get('mode') == 'robust_calmar',
    )
    gr_path = resume_dir / 'generation_results.pkl'
    if gr_path.exists():
      generation_results = pickle.loads(gr_path.read_bytes())
    ga_state = _rebuild_ga_state(ga_cache)
    _seed_ga_generation(seed, last_gen)
    next_configs = ga_optimizer([], state=ga_state, population_size=population_size,
                                hall_of_fame_size=population_size, profile_name=profile_name,
                                ga_cache=ga_cache, gen=last_gen)
    start_generation = last_gen + 1
    testback_logger.info(f"从 JSONL 恢复: {len(ga_cache)} 个唯一配置, 第 {start_generation} 代开始")
  elif args.warm_start:
    if training_objective.get('mode') == 'robust_calmar':
      raise ValueError('训练稳健目标禁止信任外部 warm-start 分数，请使用新运行或同口径 --resume')
    with open(args.warm_start, 'r', encoding='utf-8') as f:
      cache_data = json_mod.load(f)
    if isinstance(cache_data, list):
      cache_data = {_ga_cache_key(v['individual_config']): v for v in cache_data}
    for v in cache_data.values():
      v.setdefault('calmar', v.get('fitness', v['sharpe']))
    sorted_results = sorted(
      cache_data.values(),
      key=_training_candidate_rank_key,
      reverse=True,
    )
    next_configs = [v['individual_config'] for v in sorted_results[:2 * population_size]]
    testback_logger.info(f"热启动: 从 {args.warm_start} 加载 top {len(next_configs)} 个种子配置 (共 {len(cache_data)} 条缓存)")
  else:
    next_configs = generate_initial_configs(2 * population_size, profile_name=profile_name)

  sealed_frozen_entry = None
  try:
    for generation in range(start_generation, generations):
      generation_start_ts = time.time()
      n_configs = len(next_configs)
      # 分离缓存命中与待执行配置
      cached_results = []
      uncached_configs = []
      for cfg in next_configs:
        key = _ga_cache_key(cfg)
        if key in ga_cache:
          cached_results.append(ga_cache[key])
        else:
          uncached_configs.append(cfg)

      worker_args = [
        (info, score_keys, valid_dates, date_indices, stock_indices,
         all_valid_stocks, config, index_data, list_dates_full, training_objective)
        for config in uncached_configs
      ]

      results_list = list(cached_results)
      if worker_args:
        _eval_parallel(worker_args, results_list, ga_cache, testback_logger, pool=ga_pool)

      split_validation_results = []
      split_test_results = []
      if split_period_results:
        period_configs = [result['individual_config'] for result in results_list]
        split_validation_results = _period_results_for_configs(
          period_configs,
          evaluation_cache=val_eval_cache,
          info=info,
          score_keys=score_keys,
          valid_dates=val_valid_dates,
          date_indices=val_date_indices,
          stock_indices=stock_indices,
          all_valid_stocks=all_valid_stocks,
          index_data=val_index_data,
          list_dates_full=list_dates_full,
          logger=testback_logger,
          pool=ga_pool,
        )
        split_test_results = _period_results_for_configs(
          period_configs,
          evaluation_cache=test_eval_cache,
          info=info,
          score_keys=score_keys,
          valid_dates=test_valid_dates,
          date_indices=test_date_indices,
          stock_indices=stock_indices,
          all_valid_stocks=all_valid_stocks,
          index_data=test_index_data,
          list_dates_full=list_dates_full,
          logger=testback_logger,
          pool=ga_pool,
        )
        val_cache_path.write_text(
          json_mod.dumps(val_eval_cache, ensure_ascii=False), encoding='utf-8',
        )
        test_cache_path.write_text(
          json_mod.dumps(test_eval_cache, ensure_ascii=False), encoding='utf-8',
        )

      if not is_debug:
        # 训练集统计
        calmars = [r['calmar'] for r in results_list]
        fitnesses = [r.get('fitness', r['calmar']) for r in results_list]
        best_idx = max(
          range(len(results_list)),
          key=lambda i: _training_candidate_rank_key(results_list[i]),
        )
        best = results_list[best_idx]
        best_cfg = best['individual_config']
        best_m = {'calmar': best['calmar'],
                  'fitness': best.get('fitness', best['calmar']),
                  'fold_calmars': best.get('fold_calmars', []),
                  'sharpe': best['sharpe'],
                  'annualized': best['annualized'], 'max_drawdown': best['max_drawdown'],
                  'average_exposure': best.get('average_exposure', float('nan'))}
        avg_calmar = sum(calmars) / len(calmars)
        avg_ann = sum(r['annualized'] for r in results_list) / len(results_list)
        avg_dd = sum(r['max_drawdown'] for r in results_list) / len(results_list)

        # 上证指数训练基线
        gen_time = time.time() - generation_start_ts
        sorted_w = sorted(best_cfg['weights'].items(), key=lambda x: -abs(x[1]))
        w_str = ', '.join(f'{k}={v:.2f}' for k, v in sorted_w)
        timing_str = _format_timing(best_cfg)

        # 验证集+测试集评估：第1代 + 每10代，仅评估训练最优个体
        base_report_gen = (generation == 0 or (generation + 1) % 10 == 0)
        report_gen = (
          base_report_gen
          and not args.sealed_holdout
          and not split_period_results
        )
        if report_gen:
          # 验证集评估训练最优个体
          val_best_key = _ga_cache_key(best_cfg)
          if val_best_key in val_eval_cache:
            train_best_val_m = dict(val_eval_cache[val_best_key])
          else:
            val_worker_args = [
              (info, score_keys, val_valid_dates, val_date_indices, stock_indices,
               all_valid_stocks, best_cfg, val_index_data, list_dates_full, {})
            ]
            val_res = []
            _eval_parallel(val_worker_args, val_res, {}, testback_logger, pool=ga_pool)
            train_best_val_m = {'calmar': val_res[0]['calmar'], 'sharpe': val_res[0]['sharpe'], 'annualized': val_res[0]['annualized'], 'max_drawdown': val_res[0]['max_drawdown']} if val_res else {'calmar': 0, 'sharpe': 0, 'annualized': 0, 'max_drawdown': 0}
            val_eval_cache[val_best_key] = dict(train_best_val_m)
            with open(val_cache_path, 'w', encoding='utf-8') as f:
              json_mod.dump(val_eval_cache, f, ensure_ascii=False)
          # 训练最优个体的 val metrics 写入 ga_cache
          ga_cache[val_best_key]['val_calmar'] = train_best_val_m['calmar']
          ga_cache[val_best_key]['val_sharpe'] = train_best_val_m['sharpe']
          ga_cache[val_best_key]['val_annualized'] = train_best_val_m['annualized']
          ga_cache[val_best_key]['val_max_drawdown'] = train_best_val_m['max_drawdown']

          # 测试集评估训练最优个体
          best_test_key = _ga_cache_key(best_cfg)
          if best_test_key in test_eval_cache:
            _test_train_best_m = dict(test_eval_cache[best_test_key])
          else:
            test_worker_args = [
              (info, score_keys, test_valid_dates, test_date_indices, stock_indices,
               all_valid_stocks, best_cfg, test_index_data, list_dates_full, {})
            ]
            test_res = []
            _eval_parallel(test_worker_args, test_res, {}, testback_logger, pool=ga_pool)
            _test_train_best_m = {'calmar': test_res[0]['calmar'], 'sharpe': test_res[0]['sharpe'], 'annualized': test_res[0]['annualized'], 'max_drawdown': test_res[0]['max_drawdown']} if test_res else {'calmar': 0, 'sharpe': 0, 'annualized': 0, 'max_drawdown': 0}
            test_eval_cache[best_test_key] = dict(_test_train_best_m)
            with open(test_cache_path, 'w', encoding='utf-8') as f:
              json_mod.dump(test_eval_cache, f, ensure_ascii=False)

          sorted_w = sorted(best_cfg['weights'].items(), key=lambda x: -abs(x[1]))
          fold_text = ','.join(f'{value:.2f}' for value in best_m['fold_calmars'])
          testback_logger.info(
            f"GA gen{generation + 1}: 训练Calmar={best_m['calmar']:.3f}/{avg_calmar:.3f} "
            f"稳健适应度={best_m['fitness']:.3f} folds=[{fold_text}] S={best_m['sharpe']:.2f} "
            f"E={best_m['average_exposure']:.1%} | "
            f"验证Calmar={train_best_val_m['calmar']:.3f} | "
            f"测试Calmar={_test_train_best_m['calmar']:.3f} | "
            f"{_format_config_params(best_cfg, profile_name)} | "
            f"{', '.join(f'{k}={v:.1f}' for k, v in sorted_w)}")
          training_candidate = _select_training_candidate(ga_cache)
          if training_candidate is not None:
            candidate_cfg = training_candidate['individual_config']
            candidate_key = _ga_cache_key(candidate_cfg)
            candidate_val = val_eval_cache.get(candidate_key, {}).get('calmar')
            candidate_test = test_eval_cache.get(candidate_key, {}).get('calmar')
            val_text = f'{candidate_val:.3f}' if candidate_val is not None else 'N/A'
            test_text = f'{candidate_test:.3f}' if candidate_test is not None else 'N/A'
            testback_logger.info(
              f"实盘候选(仅训练选择): train={training_candidate['calmar']:.3f} "
              f"fitness={training_candidate.get('fitness', training_candidate['calmar']):.3f} "
              f"E={training_candidate.get('average_exposure', float('nan')):.1%} "
              f"val={val_text} test={test_text} | "
              f"{_format_config_params(candidate_cfg, profile_name)} | "
              f"{', '.join(f'{k}={v:.1f}' for k, v in sorted(candidate_cfg['weights'].items(), key=lambda x: -abs(x[1])))}")
            best_result = strategy_config_payload(profile_name, candidate_cfg)
            with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
              json_mod.dump(best_result, f, indent=2, ensure_ascii=False)

        if base_report_gen and args.sealed_holdout:
          fold_text = ','.join(f'{value:.2f}' for value in best_m['fold_calmars'])
          testback_logger.info(
            f"GA gen{generation + 1}: 训练Calmar={best_m['calmar']:.3f}/{avg_calmar:.3f} "
            f"稳健适应度={best_m['fitness']:.3f} folds=[{fold_text}] S={best_m['sharpe']:.2f} "
            f"E={best_m['average_exposure']:.1%} | "
            f"{_format_config_params(best_cfg, profile_name)} | "
            f"{', '.join(f'{k}={v:.1f}' for k, v in sorted_w)}")
          training_candidate = _select_training_candidate(ga_cache)
          if training_candidate is not None:
            candidate_cfg = training_candidate['individual_config']
            testback_logger.info(
              f"实盘候选(仅训练选择, holdout封存): "
              f"train={training_candidate['calmar']:.3f} "
              f"fitness={training_candidate.get('fitness', training_candidate['calmar']):.3f} "
              f"E={training_candidate.get('average_exposure', float('nan')):.1%} | "
              f"{_format_config_params(candidate_cfg, profile_name)} | "
              f"{', '.join(f'{k}={v:.1f}' for k, v in sorted(candidate_cfg['weights'].items(), key=lambda x: -abs(x[1])))}")
            best_result = strategy_config_payload(profile_name, candidate_cfg)
            with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
              json_mod.dump(best_result, f, indent=2, ensure_ascii=False)


      if not results_list:
        raise RuntimeError(f"{'调试模式' if is_debug else '第 ' + str(generation + 1) + ' 代'}未获得任何有效回测结果")

      generation_time = time.time() - generation_start_ts
      for result in results_list:
        result['generation'] = generation
        result.setdefault('fitness', result['calmar'])


      best_in_gen = max(results_list, key=_training_candidate_rank_key)

      if not is_debug:
        fitnesses = [ind['fitness'] for ind in results_list]
        generation_stats = {
          'generation': generation, 'generation_time': generation_time,
          'population_size': len(results_list),
          'max_fitness': max(fitnesses), 'mean_fitness': sum(fitnesses) / len(fitnesses),
          'min_fitness': min(fitnesses),
        }
        generation_stats['best_weights'] = dict(best_in_gen['individual_config']['weights'])
        generation_stats['best_buy_n'] = best_in_gen['individual_config']['buy_n']
        generation_stats['best_average_exposure'] = best_in_gen.get('average_exposure')
        generation_stats['best_stock_pool'] = best_in_gen['individual_config'].get('stock_pool')
        generation_stats['best_timing_enabled'] = best_in_gen['individual_config'].get('timing_enabled')
        generation_stats['best_timing_base'] = best_in_gen['individual_config'].get('timing_base')
        generation_stats['best_timing_leverage'] = best_in_gen['individual_config'].get('timing_leverage')
        if report_gen:
          generation_stats['val_best_calmar'] = float(train_best_val_m['calmar'])
          generation_stats['val_best_sharpe'] = float(train_best_val_m['sharpe'])
          generation_stats['val_best_annualized'] = float(train_best_val_m['annualized'])
          generation_stats['val_best_mdd'] = float(train_best_val_m['max_drawdown'])
          generation_stats['test_train_best_calmar'] = float(_test_train_best_m['calmar'])
          generation_stats['test_train_best_sharpe'] = float(_test_train_best_m['sharpe'])
          generation_stats['test_train_best_annualized'] = float(_test_train_best_m['annualized'])
          generation_stats['test_train_best_mdd'] = float(_test_train_best_m['max_drawdown'])
        generation_results.append(generation_stats)

      import json as _json
      best_key = _config_key(best_in_gen['individual_config'])
      training_entries = []
      with open(output_dir / 'all_results.jsonl', 'a', encoding='utf-8') as _f:
        for _r in results_list:
          entry = {
            'generation': _r['generation'],
            'fitness': _r['fitness'], 'fold_calmars': _r.get('fold_calmars', []),
            'raw_fitness': _r.get('raw_fitness', _r['fitness']),
            'average_exposure': _r.get('average_exposure'),
            'exposure_constraint_passed': _r.get('exposure_constraint_passed'),
            'calmar': _r['calmar'], 'sharpe': _r['sharpe'],
            'annualized': _r['annualized'],
            'max_drawdown': _r['max_drawdown'], 'total_return': _r['total_return'],
            'val_calmar': _r.get('val_calmar'), 'val_sharpe': _r.get('val_sharpe'), 'config': _r['individual_config'],
          }
          if _config_key(_r['individual_config']) == best_key and not is_debug:
            if report_gen:
              entry['val_calmar'] = train_best_val_m['calmar']
              entry['val_sharpe'] = train_best_val_m['sharpe']
              entry['val_annualized'] = train_best_val_m['annualized']
              entry['val_max_drawdown'] = train_best_val_m['max_drawdown']
              entry['test_calmar'] = _test_train_best_m['calmar']
              entry['test_sharpe'] = _test_train_best_m['sharpe']
              entry['test_annualized'] = _test_train_best_m['annualized']
              entry['test_max_drawdown'] = _test_train_best_m['max_drawdown']
          entry = _training_jsonl_entry(
            entry,
            training_objective=training_objective,
            split_period_results=split_period_results,
          )
          _f.write(_json.dumps(entry, ensure_ascii=False) + '\n')
          training_entries.append(entry)

      if split_period_results:
        _append_jsonl_rows(
          output_dir / 'training_results.jsonl', training_entries,
        )
        _append_jsonl_rows(
          output_dir / 'validation_results.jsonl',
          [
            _period_result_entry(generation, result)
            for result in split_validation_results
          ],
        )
        _append_jsonl_rows(
          output_dir / 'test_results.jsonl',
          [
            _period_result_entry(generation, result)
            for result in split_test_results
          ],
        )
        _validate_split_result_files(output_dir)

      if not is_debug:
        with open(output_dir / 'generation_results.pkl', 'wb') as f:
          pickle.dump(generation_results, f)

      _seed_ga_generation(seed, generation)
      next_configs = ga_optimizer(results_list, state=ga_state, population_size=population_size,
                                  hall_of_fame_size=population_size, profile_name=profile_name,
                                  ga_cache=ga_cache, gen=generation)

    if not is_debug and not split_period_results and sealed_holdout:
      sealed_frozen_entry = _select_training_candidate(ga_cache)
      if sealed_frozen_entry is None:
        raise RuntimeError('训练结束后没有可冻结候选')
      sealed_config = sealed_frozen_entry['individual_config']
      sealed_result = strategy_config_payload(profile_name, sealed_config)
      with open(
        output_dir / 'best_individual_config.json',
        'w',
        encoding='utf-8',
      ) as f:
        json_mod.dump(sealed_result, f, indent=2, ensure_ascii=False)
      testback_logger.info(f"\n{'=' * 60}")
      testback_logger.info(
        '训练候选已冻结；先释放训练数据，再由独立 single '
        '进程各运行一次验证/测试'
      )

    # 非 sealed 兼容模式沿用内存内 holdout 诊断。
    if not is_debug and not split_period_results and not sealed_holdout:
      frozen_entry = _select_training_candidate(ga_cache)
      if frozen_entry is None:
        raise RuntimeError('训练结束后没有可冻结候选')
      train_best_config = frozen_entry['individual_config']
      frozen_result = strategy_config_payload(profile_name, train_best_config)
      with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
        json_mod.dump(frozen_result, f, indent=2, ensure_ascii=False)
      testback_logger.info(f"\n{'=' * 60}")
      testback_logger.info('训练候选已冻结，开始一次性 holdout 诊断')
      frozen_val_args = [(
        info, score_keys, val_valid_dates, val_date_indices, stock_indices,
        all_valid_stocks, train_best_config, val_index_data, list_dates_full, {}
      )]
      frozen_val_results = []
      _eval_parallel(frozen_val_args, frozen_val_results, {}, testback_logger, pool=ga_pool)
      train_test_args = [(
        info, score_keys, test_valid_dates, test_date_indices, stock_indices,
        all_valid_stocks, train_best_config, test_index_data, list_dates_full, {}
      )]
      train_test_results = []
      _eval_parallel(train_test_args, train_test_results, {}, testback_logger, pool=ga_pool)
      if frozen_val_results:
        vr = frozen_val_results[0]
        testback_logger.info(
          f"  [冻结训练候选] 验证Calmar={vr['calmar']:.3f}, "
          f"夏普={vr['sharpe']:.3f}, 年化={vr['annualized']:.1f}%, 回撤={vr['max_drawdown']:.1f}%")
      if train_test_results:
        tr = train_test_results[0]
        testback_logger.info(
          f"  [冻结训练候选] 测试Calmar={tr['calmar']:.3f}, "
          f"夏普={tr['sharpe']:.3f}, 年化={tr['annualized']:.1f}%, 回撤={tr['max_drawdown']:.1f}%")
      frozen_validation_result = (
        frozen_val_results[0] if frozen_val_results else None
      )
      frozen_test_result = (
        train_test_results[0] if train_test_results else None
      )
      _write_frozen_candidate_diagnostics(
        output_dir,
        frozen_entry=frozen_entry,
        validation_result=frozen_validation_result,
        test_result=frozen_test_result,
      )
      holdout_payload = {
        'selection_scope': 'train_only',
        'candidate_id': _ga_cache_key(train_best_config),
        'training_fitness': frozen_entry.get('fitness', frozen_entry['calmar']),
        'training_calmar': frozen_entry['calmar'],
        'fold_calmars': frozen_entry.get('fold_calmars', []),
        'validation': frozen_validation_result,
        'test': frozen_test_result,
      }
      for period in ('validation', 'test'):
        if holdout_payload[period] is not None:
          holdout_payload[period] = {
            key: value for key, value in holdout_payload[period].items()
            if key != 'individual_config'
          }
      with open(output_dir / 'holdout_diagnostics.json', 'w', encoding='utf-8') as f:
        json_mod.dump(holdout_payload, f, indent=2, ensure_ascii=False)
      testback_logger.info(f"{'=' * 60}")

  finally:
    ga_pool.terminate()
    ga_pool.join()
    _cleanup_ga_resources()

  if sealed_frozen_entry is not None:
    frozen_config = sealed_frozen_entry['individual_config']
    frozen_config_path = output_dir / 'best_individual_config.json'
    holdout_results = _run_frozen_holdout_reports(
      output_dir,
      frozen_config_path,
      frozen_config,
    )
    frozen_validation_result = holdout_results['validation']
    frozen_test_result = holdout_results['test']
    _write_frozen_candidate_diagnostics(
      output_dir,
      frozen_entry=sealed_frozen_entry,
      validation_result=frozen_validation_result,
      test_result=frozen_test_result,
    )
    holdout_payload = {
      'selection_scope': 'train_only',
      'candidate_id': _ga_cache_key(frozen_config),
      'training_fitness': sealed_frozen_entry.get(
        'fitness',
        sealed_frozen_entry['calmar'],
      ),
      'training_calmar': sealed_frozen_entry['calmar'],
      'fold_calmars': sealed_frozen_entry.get('fold_calmars', []),
      'validation': {
        key: value
        for key, value in frozen_validation_result.items()
        if key != 'individual_config'
      },
      'test': {
        key: value
        for key, value in frozen_test_result.items()
        if key != 'individual_config'
      },
    }
    with open(
      output_dir / 'holdout_diagnostics.json',
      'w',
      encoding='utf-8',
    ) as f:
      json_mod.dump(holdout_payload, f, indent=2, ensure_ascii=False)
    testback_logger.info(
      'sealed holdout 独立 single 诊断完成: '
      f"验证Calmar={frozen_validation_result['calmar']:.3f}, "
      f"测试Calmar={frozen_test_result['calmar']:.3f}"
    )

  training_candidate = _save_training_candidate(
    output_dir,
    profile_name,
    ga_cache,
  )
  if is_debug:
    if training_candidate is None:
      raise RuntimeError('训练结束后没有可保存候选')
    testback_logger.info(
      '调试模式全局训练最优参数已保存: '
      f"{output_dir / 'best_individual_config.json'}"
    )
  else:
    if training_candidate is not None:
      best_cfg = training_candidate['individual_config']
      w_str = ', '.join(f'{k}={v:.2f}' for k, v in best_cfg['weights'].items())
      overlay_enabled = bool((best_cfg.get('trend_risk_overlay') or {}).get('enabled'))
      timing_str = ', trend_overlay=ON' if overlay_enabled else _format_timing(best_cfg)
      testback_logger.info(f"\n最优参数已保存:")
      testback_logger.info(f"  - {output_dir / 'best_individual_config.json'}")
      testback_logger.info(f"实盘候选仅按训练稳健适应度={training_candidate.get('fitness', training_candidate['calmar']):.3f}, 全训练Calmar={training_candidate['calmar']:.3f}: [{w_str}], pool={_format_pool(best_cfg.get('stock_pool'))}, buy={best_cfg['buy_n']},sell={best_cfg['sell_m']}{timing_str}, rebal={'ON' if best_cfg.get('rebalance') else 'OFF'}")
    else:
      testback_logger.warning("无可用训练候选，回退到训练 hall-of-fame")
      best_config = ga_state['hall_of_fame'][0]
      best_result = strategy_config_payload(profile_name, best_config)
      with open(output_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
        json_mod.dump(best_result, f, indent=2, ensure_ascii=False)
      testback_logger.info(f"\n最优参数已保存（训练最优）: {output_dir / 'best_individual_config.json'}")

    calmars = [v['calmar'] for v in ga_cache.values()]
    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info("回测执行完成")
    testback_logger.info(f"  总回测次数: {sum(g.get('population_size', len(g.get('best_weights', {}))) for g in generation_results)}")
    testback_logger.info(f"  唯一配置数: {len(ga_cache)}")
    testback_logger.info(f"  平均Calmar: {sum(calmars) / len(calmars):.3f}")
    testback_logger.info(f"  最大Calmar: {max(calmars):.3f}")
    testback_logger.info(f"{'=' * 60}")

  testback_logger.info(f"\n{'调试' if is_debug else 'GA'}模式执行完成，结果目录: {output_dir}")
  testback_logger.remove_file_sink()
  return None


# ========== 主入口 ==========

def main():
  import argparse

  from loguru import logger as loguru_logger

  ts = datetime.now()

  parser = argparse.ArgumentParser(description='WBR GA 参数搜索')
  parser.add_argument('--mode', type=str, default='ga', choices=['debug', 'ga'])
  parser.add_argument('--output-dir', type=str, default=None)
  parser.add_argument('--warm-start', type=str, default=None, help='热启动种群 JSON 文件路径')
  parser.add_argument('--resume', type=str, nargs='?', const='auto', default=None,
                      help='从 JSONL 恢复: 不传则自动找最新, 或指定目录路径')
  parser.add_argument('--profile', type=str, default=None)
  parser.add_argument('--generations', type=int, default=None,
                      help='覆盖当前模式的代数，仅影响本次运行')
  parser.add_argument('--population-size', type=int, default=None,
                      help='覆盖当前模式的种群大小，仅影响本次运行')
  parser.add_argument('--sealed-holdout', action='store_true',
                      help='代内不评估验证/测试，训练候选冻结后各评估一次')
  parser.add_argument(
      '--split-period-results', action='store_true',
      help=(
          '逐个体评估训练/验证/测试并分别写入三个 JSONL；'
          '验证/测试不参与遗传选择'
      ),
  )
  parser.add_argument('--seed', type=int, default=DEFAULT_GA_SEED,
                      help=f'GA 随机种子（默认 {DEFAULT_GA_SEED}）')
  parser.add_argument('--candidate-configs', type=str,
                      help='debug模式下批量评估训练候选JSON，不运行遗传或holdout')
  parser.add_argument('--live-sim', action='store_true', default=True, help='启用实盘模拟 (默认开启)')
  args = parser.parse_args()
  _seed_ga_randomness(args.seed)
  if args.candidate_configs and args.mode != 'debug':
    parser.error('--candidate-configs 只能与 --mode debug 一起使用')
  if args.candidate_configs and (args.resume or args.warm_start):
    parser.error('--candidate-configs 不能与 --resume/--warm-start 同时使用')
  if args.split_period_results and args.sealed_holdout:
    parser.error(
      '--split-period-results 会逐个体观察验证/测试，不能与 '
      '--sealed-holdout 同时使用'
    )
  if args.split_period_results and args.mode != 'ga':
    parser.error('--split-period-results 只能用于 --mode ga')

  profile_name = args.profile or DEFAULT_GA_PROFILE
  mode_configs = get_mode_configs(profile_name)
  mode_config = mode_configs[args.mode].copy()
  if args.generations is not None:
    if args.generations < 1:
      parser.error('--generations 必须大于0')
    mode_config['generations'] = args.generations
  if args.population_size is not None:
    if args.population_size < 2:
      parser.error('--population-size 必须至少为2')
    mode_config['population_size'] = args.population_size

  loguru_logger.remove()
  loguru_logger.add(sys.stderr, level=mode_config['log_level'])

  testback_logger.info(f"运行模式: {args.mode} - {mode_config['desc']}")

  start_date, end_date = get_profile_preload_range(profile_name)
  testback_logger.info(f"GA 预加载区间: {start_date.strftime('%Y%m%d')} - {end_date.strftime('%Y%m%d')}")

  backtest_datetime_list = [
    datetime.combine(d, datetime.min.time())
    for d in get_trading_date_span(start_date, end_date)
  ]

  factor_classes = get_profile_factor_classes(profile_name)
  factor_histories = {factor_cls.__name__: factor_cls().hist_days for factor_cls in factor_classes}
  max_hist_days = max(factor_histories.values(), default=0)
  hist_detail = ', '.join(f'{name}={days}天' for name, days in factor_histories.items())
  testback_logger.info(f"因子历史需求: {hist_detail}，最大需求={max_hist_days}天")

  # GA/debug: 从最新 runtime NPZ 取历史全集，避免当前可买池带来幸存者偏差
  filtered_stocks = load_runtime_stock_codes()
  testback_logger.info(f"股票池(runtime历史全集): {len(filtered_stocks)} 只")

  resume_dir = None
  if args.resume:
    if args.resume == 'auto':
      resume_dir = _find_latest_ga_dir()
      if resume_dir is None:
        testback_logger.error('--resume 未找到可恢复的 GA 目录（需含 all_results.jsonl）')
        sys.exit(1)
    else:
      resume_dir = Path(args.resume)
      if (
        not resume_dir.is_dir()
        or not (resume_dir / 'all_results.jsonl').is_file()
        or not (resume_dir / 'run_metadata.json').is_file()
      ):
        testback_logger.error(
          f'--resume 指定目录无效或缺少 all_results.jsonl/run_metadata.json: {resume_dir}'
        )
        sys.exit(1)
    testback_logger.info(f'从 JSONL 恢复: {resume_dir}')

  with keep_windows_awake() as keep_awake_enabled:
    if keep_awake_enabled:
      testback_logger.info('已启用 Windows 防休眠，任务结束后自动恢复')

    result = _run_ga(args, mode_config, backtest_datetime_list, filtered_stocks, profile_name=profile_name, resume_dir=resume_dir)

  te = datetime.now()
  testback_logger.info(f"总耗时: {(te - ts).total_seconds():.2f} 秒")
  return result


if __name__ == "__main__":
  main()

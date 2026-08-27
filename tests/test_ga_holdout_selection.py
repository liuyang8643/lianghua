import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from testback.run_ga import (
    TRAIN_METRIC_VERSION,
    _append_jsonl_rows, _build_run_metadata, _calmar_from_returns,
    _ga_cache_key, _load_candidate_configs, _load_resume_metadata,
    _period_result_entry,
    _rebuild_from_jsonl,
    _run_frozen_holdout_reports,
    _save_training_candidate,
    _select_training_candidate, _training_fitness,
    _training_jsonl_entry,
    _trim_runtime_to_training_end,
    _validate_resume_metadata, _validate_split_result_files,
    _write_frozen_candidate_diagnostics,
)


def test_training_only_runtime_trim_excludes_holdout_rows():
    data = {
        'trade_dates': np.array(
            ['2018-12-27', '2018-12-28', '2019-01-02'],
            dtype='datetime64[D]',
        ),
        'open': np.arange(6, dtype=float).reshape(3, 2),
        'stock_codes': np.array(['000001.SZ', '000002.SZ']),
        'issue_price': np.array([10.0, 20.0]),
    }

    trimmed = _trim_runtime_to_training_end(
        data,
        np.datetime64('2018-12-28'),
    )

    np.testing.assert_array_equal(
        trimmed['trade_dates'],
        np.array(['2018-12-27', '2018-12-28'], dtype='datetime64[D]'),
    )
    assert trimmed['open'].shape == (2, 2)
    assert not np.shares_memory(trimmed['open'], data['open'])
    assert not np.shares_memory(
        trimmed['trade_dates'],
        data['trade_dates'],
    )
    np.testing.assert_array_equal(trimmed['stock_codes'], data['stock_codes'])
    np.testing.assert_array_equal(trimmed['issue_price'], data['issue_price'])


def test_frozen_holdout_reports_run_as_separate_single_processes(
    tmp_path,
    monkeypatch,
):
    runtime_path = tmp_path / 'runtime.npz'
    np.savez(
        runtime_path,
        trade_dates=np.array(
            ['2026-07-23', '2026-07-24'],
            dtype='datetime64[D]',
        ),
    )
    monkeypatch.setattr(
        'core.runtime.latest_runtime_npz_path',
        lambda: runtime_path,
    )
    calls = []

    def fake_run(command, check):
        assert check is True
        calls.append(command)
        output_dir = command[command.index('--output-dir') + 1]
        start_date = command[command.index('--start-date') + 1]
        end_date = command[command.index('--end-date') + 1]
        path = Path(output_dir)
        path.mkdir(parents=True)
        (path / 'record.json').write_text(
            json.dumps({
                'daily_returns': [1.0, -0.5, 0.2],
                'period': {
                    'start': start_date,
                    'end': end_date,
                },
            }),
            encoding='utf-8',
        )

    monkeypatch.setattr('subprocess.run', fake_run)
    config_path = tmp_path / 'best.json'
    config_path.write_text('{}', encoding='utf-8')

    results = _run_frozen_holdout_reports(
        tmp_path,
        config_path,
        {'weights': {'Demo': 1.0}},
    )

    assert len(calls) == 2
    assert results['validation']['start'] == '2019-01-01'
    assert results['validation']['end'] == '2022-12-31'
    assert results['test']['start'] == '2023-01-01'
    assert results['test']['end'] == '2026-07-24'
    assert results['validation']['trading_days'] == 3
    assert results['test']['calmar'] == results['validation']['calmar']


def test_deployable_candidate_is_selected_only_by_training_calmar():
    cache = {
        'high_validation': {
            'calmar': 1.0,
            'val_calmar': 9.0,
            'test_calmar': 9.0,
            'individual_config': {'name': 'validation_winner'},
        },
        'high_training': {
            'calmar': 2.0,
            'val_calmar': 0.1,
            'test_calmar': 0.1,
            'individual_config': {'name': 'training_winner'},
        },
    }

    selected = _select_training_candidate(cache)

    assert selected['individual_config']['name'] == 'training_winner'


def test_deployable_candidate_prefers_train_only_robust_fitness():
    cache = {
        'high_full_calmar': {
            'calmar': 3.0, 'fitness': 1.0,
            'individual_config': {'name': 'fragile'},
        },
        'stable': {
            'calmar': 2.0, 'fitness': 1.5,
            'individual_config': {'name': 'stable'},
        },
    }

    selected = _select_training_candidate(cache)

    assert selected['individual_config']['name'] == 'stable'


def test_training_candidate_tie_break_is_deterministic_and_holdout_blind():
    candidates = [
        {
            'fitness': 1.5,
            'fold_calmars': [0.8, 1.0, 1.2],
            'calmar': 2.0,
            'sharpe': 1.4,
            'annualized': 25.0,
            'max_drawdown': -12.0,
            'val_calmar': 99.0,
            'test_calmar': 99.0,
            'individual_config': {
                'name': 'a',
                'weights': {'Demo': 1.0},
                'buy_n': 10,
                'sell_m': 10,
            },
        },
        {
            'fitness': 1.5,
            'fold_calmars': [0.8, 1.0, 1.2],
            'calmar': 2.0,
            'sharpe': 1.4,
            'annualized': 25.0,
            'max_drawdown': -12.0,
            'val_calmar': -99.0,
            'test_calmar': -99.0,
            'individual_config': {
                'name': 'b',
                'weights': {'Demo': 1.0},
                'buy_n': 10,
                'sell_m': 10,
            },
        },
    ]
    expected_key = max(
        _ga_cache_key(entry['individual_config'])
        for entry in candidates
    )

    forward = _select_training_candidate({
        'first': candidates[0],
        'second': candidates[1],
    })
    reverse = _select_training_candidate({
        'second': candidates[1],
        'first': candidates[0],
    })

    assert _ga_cache_key(forward['individual_config']) == expected_key
    assert _ga_cache_key(reverse['individual_config']) == expected_key


def test_training_tie_prefers_worst_fold_before_holdout_diagnostics():
    stronger_fold = {
        'fitness': 1.5,
        'fold_calmars': [0.9, 1.0, 1.1],
        'calmar': 2.0,
        'sharpe': 1.4,
        'annualized': 25.0,
        'max_drawdown': -12.0,
        'val_calmar': -50.0,
        'test_calmar': -50.0,
        'individual_config': {'name': 'stronger_fold'},
    }
    weaker_fold = {
        'fitness': 1.5,
        'fold_calmars': [0.7, 1.2, 1.2],
        'calmar': 2.0,
        'sharpe': 1.4,
        'annualized': 25.0,
        'max_drawdown': -12.0,
        'val_calmar': 50.0,
        'test_calmar': 50.0,
        'individual_config': {'name': 'weaker_fold'},
    }

    selected = _select_training_candidate({
        'weaker': weaker_fold,
        'stronger': stronger_fold,
    })

    assert selected['individual_config']['name'] == 'stronger_fold'


def test_global_training_winner_is_saved_after_debug_generations(tmp_path):
    global_winner = {
        'generation': 3,
        'fitness': 2.0,
        'fold_calmars': [1.5, 1.6, 1.7],
        'calmar': 2.4,
        'sharpe': 1.8,
        'annualized': 30.0,
        'max_drawdown': -10.0,
        'individual_config': {
            'name': 'global',
            'weights': {'Demo': 1.0},
            'buy_n': 10,
            'sell_m': 10,
        },
    }
    last_generation_winner = {
        'generation': 49,
        'fitness': 1.8,
        'fold_calmars': [1.3, 1.4, 1.5],
        'calmar': 2.1,
        'sharpe': 1.6,
        'annualized': 28.0,
        'max_drawdown': -11.0,
        'val_calmar': 100.0,
        'test_calmar': 100.0,
        'individual_config': {
            'name': 'last_generation',
            'weights': {'Demo': 1.0},
            'buy_n': 10,
            'sell_m': 10,
        },
    }

    selected = _save_training_candidate(
        tmp_path,
        'v9_dual_shadow',
        {
            'global': global_winner,
            'last': last_generation_winner,
        },
    )
    saved = json.loads(
        (tmp_path / 'best_individual_config.json').read_text(
            encoding='utf-8',
        )
    )

    assert selected is global_winner
    assert saved['individual_config']['name'] == 'global'


def test_negative_training_return_has_negative_calmar():
    calmar, metrics = _calmar_from_returns(np.full(252, -0.1))

    assert metrics['annualized'] < 0
    assert calmar < 0


def test_robust_fitness_penalizes_the_worst_training_fold():
    daily = np.concatenate([
        np.full(252, 0.10),
        np.full(252, 0.10),
        np.full(252, -0.05),
    ])

    full, fitness, folds = _training_fitness(
        daily, {'mode': 'robust_calmar', 'folds': 3, 'full_weight': 0.5},
    )

    assert len(folds) == 3
    assert folds[-1] < 0
    assert fitness < full


def test_robust_fitness_supports_explicit_calendar_folds():
    dates = [
        datetime(2010, 1, 4),
        datetime(2010, 1, 5),
        datetime(2011, 1, 4),
        datetime(2011, 1, 5),
    ]
    daily = np.array([-10.0, 0.0, 1.0, -0.5])

    _, _, folds = _training_fitness(
        daily,
        {
            'mode': 'robust_calmar',
            'full_weight': 0.5,
            'calendar_folds': [
                ['2010-01-01', '2010-12-31'],
                ['2011-01-01', '2011-12-31'],
            ],
        },
        dates=dates,
    )

    assert len(folds) == 2
    assert folds[0] < 0
    assert folds[1] > 0


def test_calendar_folds_require_aligned_training_dates():
    with pytest.raises(ValueError, match='training dates'):
        _training_fitness(
            np.array([0.1, 0.1]),
            {
                'mode': 'robust_calmar',
                'calendar_folds': [['2010-01-01', '2010-12-31']],
            },
        )


def test_candidate_grid_can_inherit_fixed_profile_semantics(tmp_path):
    path = tmp_path / 'candidates.json'
    path.write_text(
        json.dumps({
            'inherit_profile_defaults': True,
            'configs': [{
                'weights': {
                    'PreCloseMarketCap': 1.0,
                    'AmihudIlliquidityStrict': 0.4,
                    'TrendReversalPreCloseStrict': 0.1,
                    'VolumeCVStrict': 0.2,
                    'CompletedSmallCapTrendConsistency60Strict': 0.1,
                },
                'buy_n': 30,
            }],
        }),
        encoding='utf-8',
    )

    config = _load_candidate_configs(
        path, profile_name='v38_smallcap_trend_consistency',
    )[0]

    assert config['sell_m'] == 30
    assert config['stock_pool'] == ['60', '00', '30']
    assert config['filter_factors'] == {
        'FilterST': True,
        'FilterStarST': True,
        'FilterLowPrice': True,
    }
    assert config['trend_risk_overlay']['mode'] == 'dual_completed'


def test_robust_resume_rejects_legacy_metric_scope(tmp_path):
    path = tmp_path / 'all_results.jsonl'
    path.write_text(
        '{"generation":0,"calmar":1.0,"sharpe":1.0,'
        '"config":{"weights":{"Demo":1.0}}}\n',
        encoding='utf-8',
    )

    try:
        _rebuild_from_jsonl(tmp_path, require_train_fitness=True)
    except ValueError as exc:
        assert '旧适应度口径' in str(exc)
    else:
        raise AssertionError('legacy resume should be rejected')


def test_robust_resume_rejects_holdout_results(tmp_path):
    path = tmp_path / 'all_results.jsonl'
    path.write_text(
        '{"generation":0,"fitness":1.0,"fold_calmars":[1.0,1.0,1.0],'
        '"calmar":1.0,"sharpe":1.0,"val_calmar":2.0,'
        '"config":{"weights":{"Demo":1.0}}}\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='holdout'):
        _rebuild_from_jsonl(tmp_path, require_train_fitness=True)


def test_resume_cache_uses_same_canonical_string_key_as_live_execution(tmp_path):
    config = {
        'weights': {'Second': 0.5, 'First': 1.0},
        'buy_n': 30,
        'sell_m': 30,
        'stock_pool': ['60', '00', '30'],
    }
    equivalent = {
        'stock_pool': ['60', '00', '30'],
        'sell_m': 30,
        'buy_n': 30,
        'weights': {'First': 1.0, 'Second': 0.5},
    }
    _append_jsonl_rows(
        tmp_path / 'all_results.jsonl',
        [{
            'generation': 4,
            'fitness': 1.2,
            'fold_calmars': [1.0, 1.1, 1.2],
            'calmar': 1.4,
            'sharpe': 1.1,
            'config': config,
        }],
    )

    cache, last_generation = _rebuild_from_jsonl(
        tmp_path, require_train_fitness=True,
    )
    live_key = _ga_cache_key(equivalent)

    assert isinstance(live_key, str)
    assert list(cache) == [live_key]
    assert cache[live_key]['individual_config'] == config
    assert last_generation == 4


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'profile': 'other'}, 'profile'),
        ({'seed': 7}, '随机种子'),
        ({'sealed_holdout': False}, 'sealed_holdout'),
        ({'training_objective': {'mode': 'other'}}, 'training_objective'),
    ],
)
def test_resume_metadata_must_match_experiment_identity(override, message):
    metadata = {
        'profile': 'v9_dual_shadow',
        'seed': 20260720,
        'sealed_holdout': True,
        'test_period_is_strictly_sealed': True,
        'split_period_results': False,
        'train_metric_version': TRAIN_METRIC_VERSION,
        'training_objective': {'mode': 'robust_calmar'},
    }
    metadata.update(override)

    with pytest.raises(ValueError, match=message):
        _validate_resume_metadata(
            metadata,
            profile_name='v9_dual_shadow',
            seed=20260720,
            sealed_holdout=True,
            split_period_results=False,
            training_objective={'mode': 'robust_calmar'},
        )


def test_resume_requires_metadata_before_results_can_be_restored(tmp_path):
    (tmp_path / 'all_results.jsonl').write_text(
        'not valid jsonl\n', encoding='utf-8',
    )

    with pytest.raises(ValueError, match='run_metadata.json'):
        _load_resume_metadata(
            tmp_path,
            profile_name='v9_dual_shadow',
            seed=20260720,
            sealed_holdout=True,
            split_period_results=False,
            training_objective={'mode': 'robust_calmar'},
        )


def test_resume_rejects_missing_false_sealed_state():
    metadata = {
        'profile': 'v9_dual_shadow',
        'seed': 20260720,
        'split_period_results': False,
        'train_metric_version': TRAIN_METRIC_VERSION,
        'training_objective': {'mode': 'robust_calmar'},
    }

    with pytest.raises(ValueError, match='sealed_holdout'):
        _validate_resume_metadata(
            metadata,
            profile_name='v9_dual_shadow',
            seed=20260720,
            sealed_holdout=False,
            split_period_results=False,
            training_objective={'mode': 'robust_calmar'},
        )


def test_resume_metadata_is_validated_before_malformed_jsonl(tmp_path):
    (tmp_path / 'all_results.jsonl').write_text(
        'not valid jsonl\n', encoding='utf-8',
    )
    (tmp_path / 'run_metadata.json').write_text(
        json.dumps({
            'profile': 'v9_dual_shadow',
            'seed': 7,
            'sealed_holdout': True,
            'test_period_is_strictly_sealed': True,
            'split_period_results': False,
            'train_metric_version': TRAIN_METRIC_VERSION,
            'training_objective': {'mode': 'robust_calmar'},
        }),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='随机种子'):
        _load_resume_metadata(
            tmp_path,
            profile_name='v9_dual_shadow',
            seed=20260720,
            sealed_holdout=True,
            split_period_results=False,
            training_objective={'mode': 'robust_calmar'},
        )


def test_period_result_entry_has_no_training_selection_fields():
    config = {'weights': {'Demo': 1.0}, 'buy_n': 30, 'sell_m': 30}
    row = _period_result_entry(7, {
        'calmar': 2.1,
        'sharpe': 1.7,
        'annualized': 32.0,
        'max_drawdown': -15.0,
        'total_return': 80.0,
        'average_exposure': 0.6,
        'fitness': 999.0,
        'fold_calmars': [9.0, 9.0, 9.0],
        'individual_config': config,
    })

    assert row == {
        'generation': 7,
        'calmar': 2.1,
        'sharpe': 1.7,
        'annualized': 32.0,
        'max_drawdown': -15.0,
        'total_return': 80.0,
        'average_exposure': 0.6,
        'config': config,
    }
    assert 'fitness' not in row
    assert 'fold_calmars' not in row


def test_robust_training_jsonl_entry_never_contains_holdout_results():
    row = _training_jsonl_entry(
        {
            'generation': 3,
            'fitness': 1.2,
            'fold_calmars': [1.0, 1.1, 1.2],
            'val_calmar': 9.0,
            'test_calmar': 8.0,
            'config': {'weights': {'Demo': 1.0}},
        },
        training_objective={'mode': 'robust_calmar'},
        split_period_results=False,
    )

    assert row['fitness'] == 1.2
    assert all(field not in row for field in ('val_calmar', 'test_calmar'))


@pytest.mark.parametrize('sealed_holdout', [False, True])
def test_metadata_strict_seal_flag_tracks_sealed_holdout(sealed_holdout):
    metadata = _build_run_metadata(
        profile_name='v9_dual_shadow',
        seed=20260720,
        sealed_holdout=sealed_holdout,
        split_period_results=True,
        training_objective={'mode': 'robust_calmar'},
    )

    assert metadata['test_period_is_strictly_sealed'] is sealed_holdout
    assert metadata['period_result_files'] is not None
    assert metadata['final_candidate_diagnostic_files'] is None


def test_frozen_candidate_diagnostics_are_split_by_period(tmp_path):
    config = {'weights': {'Demo': 1.0}, 'buy_n': 30, 'sell_m': 30}
    training = {
        'individual_config': config,
        'fitness': 1.2,
        'raw_fitness': 1.2,
        'calmar': 1.4,
        'fold_calmars': [1.0, 1.1, 1.2],
        'sharpe': 1.1,
        'val_calmar': 99.0,
        'test_calmar': 98.0,
    }
    validation = {
        'individual_config': config,
        'calmar': 0.9,
        'sharpe': 0.8,
        'annualized': 12.0,
        'max_drawdown': -13.0,
    }
    test = {
        'individual_config': config,
        'calmar': 0.7,
        'sharpe': 0.6,
        'annualized': 10.0,
        'max_drawdown': -15.0,
    }

    _write_frozen_candidate_diagnostics(
        tmp_path,
        frozen_entry=training,
        validation_result=validation,
        test_result=test,
    )

    saved = {
        period: json.loads(
            (tmp_path / f'{period}_diagnostics.json').read_text(
                encoding='utf-8',
            )
        )
        for period in ('training', 'validation', 'test')
    }
    assert {row['candidate_id'] for row in saved.values()} == {
        _ga_cache_key(config)
    }
    assert saved['training']['period'] == 'training'
    assert saved['training']['metrics']['calmar'] == 1.4
    assert 'val_calmar' not in saved['training']['metrics']
    assert 'test_calmar' not in saved['training']['metrics']
    assert saved['validation']['metrics']['calmar'] == 0.9
    assert saved['test']['metrics']['calmar'] == 0.7
    assert all(
        'individual_config' not in row['metrics']
        for row in saved.values()
    )


def test_resume_rejects_pre_anchor_metric_version():
    metadata = {
        'profile': 'v9_dual_shadow',
        'seed': 20260720,
        'sealed_holdout': True,
        'test_period_is_strictly_sealed': True,
        'split_period_results': False,
        'training_objective': {'mode': 'robust_calmar'},
    }

    with pytest.raises(ValueError, match='initial-NAV-1'):
        _validate_resume_metadata(
            metadata,
            profile_name='v9_dual_shadow',
            seed=20260720,
            sealed_holdout=True,
            split_period_results=False,
            training_objective={'mode': 'robust_calmar'},
        )


def test_three_period_jsonl_rows_can_be_aligned_by_generation_and_config(
    tmp_path,
):
    config = {'weights': {'Demo': 1.0}, 'buy_n': 30, 'sell_m': 30}
    training = {
        'generation': 2,
        'fitness': 1.2,
        'fold_calmars': [1.0, 1.1, 1.2],
        'calmar': 1.4,
        'config': config,
    }
    validation = _period_result_entry(2, {
        'calmar': 1.1, 'sharpe': 1.0, 'annualized': 20.0,
        'max_drawdown': -18.0, 'total_return': 40.0,
        'average_exposure': 0.5, 'individual_config': config,
    })
    test = _period_result_entry(2, {
        'calmar': 0.9, 'sharpe': 0.8, 'annualized': 18.0,
        'max_drawdown': -20.0, 'total_return': 30.0,
        'average_exposure': 0.5, 'individual_config': config,
    })

    paths = {
        'training': tmp_path / 'training_results.jsonl',
        'validation': tmp_path / 'validation_results.jsonl',
        'test': tmp_path / 'test_results.jsonl',
    }
    for name, row in (
        ('training', training), ('validation', validation), ('test', test),
    ):
        _append_jsonl_rows(paths[name], [row])

    saved = {
        name: json.loads(path.read_text(encoding='utf-8'))
        for name, path in paths.items()
    }
    keys = {
        (row['generation'], json.dumps(row['config'], sort_keys=True))
        for row in saved.values()
    }
    assert len(keys) == 1
    assert 'fitness' in saved['training']
    assert 'fitness' not in saved['validation']
    assert 'fitness' not in saved['test']


def test_resume_metadata_rejects_split_period_mode_change():
    metadata = {
        'profile': 'v9_dual_shadow',
        'seed': 20260720,
        'sealed_holdout': False,
        'test_period_is_strictly_sealed': False,
        'split_period_results': True,
        'train_metric_version': TRAIN_METRIC_VERSION,
        'training_objective': {'mode': 'robust_calmar'},
    }

    with pytest.raises(ValueError, match='split_period_results'):
        _validate_resume_metadata(
            metadata,
            profile_name='v9_dual_shadow',
            seed=20260720,
            sealed_holdout=False,
            split_period_results=False,
            training_objective={'mode': 'robust_calmar'},
        )


def test_split_period_files_fail_loudly_when_one_file_is_missing(tmp_path):
    _append_jsonl_rows(
        tmp_path / 'training_results.jsonl',
        [{'generation': 0, 'config': {'weights': {'Demo': 1.0}}}],
    )

    with pytest.raises(ValueError, match='必须同时存在'):
        _validate_split_result_files(tmp_path)


def test_split_period_files_fail_loudly_when_configs_are_misaligned(tmp_path):
    for name, weight in (
        ('training', 1.0), ('validation', 1.0), ('test', 2.0),
    ):
        _append_jsonl_rows(
            tmp_path / f'{name}_results.jsonl',
            [{
                'generation': 0,
                'config': {'weights': {'Demo': weight}},
            }],
        )

    with pytest.raises(ValueError, match='一一对齐'):
        _validate_split_result_files(tmp_path)

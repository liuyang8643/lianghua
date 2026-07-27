import json

import numpy as np
import pytest

from testback.run_ga import (
    _append_jsonl_rows, _calmar_from_returns, _period_result_entry,
    _rebuild_from_jsonl,
    _select_training_candidate, _training_fitness,
    _validate_resume_metadata, _validate_split_result_files,
)


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
        'split_period_results': True,
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

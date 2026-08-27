from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest

import core.trend_timing as trend_timing
import testback.run_ga as run_ga
from testback.run_ga import (
    EXECUTION_COST_VERSION,
    EXECUTION_SEMANTIC_DEFAULTS,
    _build_run_metadata,
    _config_key,
    _ga_cache_key,
    _validate_resume_metadata,
    _worker_evaluate,
)


def _metadata_kwargs():
    return {
        'profile_name': 'fixed_cost_profile',
        'seed': 7,
        'sealed_holdout': True,
        'split_period_results': False,
        'training_objective': {'mode': 'robust_calmar'},
    }


def test_config_key_canonicalizes_execution_semantic_defaults():
    implicit = {
        'weights': {'Demo': 1.0},
        'buy_n': 30,
        'sell_m': 30,
    }
    explicit = {
        **implicit,
        'slippage_bps': 10,
        'rebalance_band_pct': 0.01,
    }

    assert _config_key(implicit) == _config_key(explicit)
    assert _ga_cache_key(implicit) == _ga_cache_key(explicit)
    assert _ga_cache_key(implicit) != _ga_cache_key({
        **explicit,
        'slippage_bps': 30,
    })
    assert _ga_cache_key(implicit) != _ga_cache_key({
        **explicit,
        'rebalance_band_pct': 0.02,
    })

def test_run_metadata_records_candidate_execution_semantics():
    kwargs = _metadata_kwargs()
    metadata = _build_run_metadata(**kwargs)

    assert metadata['execution_cost_version'] == EXECUTION_COST_VERSION
    assert metadata['candidate_semantic_fields'] == [
        'slippage_bps',
        'rebalance_band_pct',
    ]
    assert metadata['semantic_defaults'] == EXECUTION_SEMANTIC_DEFAULTS
    assert 'each candidate config' in metadata[
        'execution_cost_semantics'
    ]['candidate_values_source']
    _validate_resume_metadata(metadata, **kwargs)


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (
            lambda metadata: metadata.update(
                execution_cost_version='legacy_cost_model'
            ),
            '模拟滑点成本口径',
        ),
        (
            lambda metadata: metadata.update(
                candidate_semantic_fields=['slippage_bps']
            ),
            '候选回测语义字段',
        ),
        (
            lambda metadata: metadata.update(
                semantic_defaults={
                    **EXECUTION_SEMANTIC_DEFAULTS,
                    'slippage_bps': 20.0,
                }
            ),
            '候选回测语义默认值',
        ),
    ],
)
def test_resume_rejects_execution_semantic_mismatch(mutation, message):
    kwargs = _metadata_kwargs()
    metadata = deepcopy(_build_run_metadata(**kwargs))
    mutation(metadata)

    with pytest.raises(ValueError, match=message):
        _validate_resume_metadata(metadata, **kwargs)


@pytest.mark.parametrize(
    (
        'semantic_values',
        'expected_slippage',
        'expected_band',
    ),
    [
        ({}, 10.0, 0.01),
        (
            {
                'slippage_bps': 35.0,
                'rebalance_band_pct': 0.025,
            },
            35.0,
            0.025,
        ),
    ],
)
def test_worker_passes_candidate_cost_semantics_to_backtest(
    monkeypatch,
    semantic_values,
    expected_slippage,
    expected_band,
):
    captured = []

    def fake_backtest(*args, **kwargs):
        captured.append(kwargs)
        return {
            'daily_returns': [0.1, 0.2],
            'daily_exposures': [0.5, 0.5],
            'total_return': 0.3,
            'cleared_positions_count': 0,
        }

    monkeypatch.setattr(run_ga, '_backtest_direct', fake_backtest)
    monkeypatch.setattr(
        run_ga,
        '_compute_timing_multipliers',
        lambda config, dates, index_data: None,
    )
    monkeypatch.setattr(
        trend_timing,
        'compute_configured_timing_multipliers',
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        run_ga,
        '_worker_shm_cache',
        {
            'Demo': (
                None,
                np.asarray([[1.0], [1.0]], dtype=np.float32),
            ),
        },
    )

    dates = [datetime(2024, 1, 2), datetime(2024, 1, 3)]
    config = {
        'weights': {'Demo': 1.0},
        'buy_n': 1,
        'sell_m': 1,
        'stock_pool': ['60'],
        **semantic_values,
    }
    result = _worker_evaluate((
        {'_ranking_pool_prefixes': ('60',)},
        {'Demo'},
        dates,
        [0, 1],
        {'600001.SH': 0},
        ['600001.SH'],
        config,
        {},
        {},
        {},
    ))

    assert result['individual_config'] is config
    assert len(captured) == 1
    assert captured[0]['slippage_bps'] == expected_slippage
    assert captured[0]['rebalance_band_pct'] == expected_band

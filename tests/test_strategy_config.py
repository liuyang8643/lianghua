from core.ga import (
    generate_initial_configs, get_intrinsic_params, get_mode_configs,
    get_profile, get_profile_factor_names, get_profile_search_spaces,
    repair_config,
)
from core.strategy_config import (
    load_strategy_config, normalize_individual_config, strategy_config_payload,
)


def test_strategy_yaml_drives_ga_profiles():
    params = {p['key'] for p in get_intrinsic_params()}
    assert {'buy_n', 'sell_m', 'stock_pool', 'timing_base'} <= params
    assert 'ga' in get_mode_configs('core')
    assert 'TrueMarketCap' in get_profile_factor_names('core6')
    assert get_profile_search_spaces('core6')['buy_n'][0] == 20


def test_final_config_only_uses_weights_as_runtime_factors():
    cfg = load_strategy_config('configs/config.json')
    assert cfg['profile_name'] == 'core6'
    assert 'prefilter_n' not in cfg['individual_config']
    assert cfg['factor_names'] == list(cfg['individual_config']['weights'])
    assert [c.__name__ for c in cfg['factor_classes']] == cfg['factor_names']
    assert cfg['individual_config']['sell_m'] >= cfg['individual_config']['buy_n']
    assert cfg['individual_config']['cash_reserve_ratio'] == 0.25
    assert cfg['filter_factor_names'] == ['FilterST', 'FilterStarST', 'FilterLowPrice']


def test_causal_trend_config_ranks_the_full_universe():
    cfg = load_strategy_config('results/smallcap_trend_causal_config.json')
    assert 'prefilter_n' not in cfg['individual_config']


def test_legacy_prefilter_is_not_part_of_runtime_config():
    cfg = normalize_individual_config({
        'weights': {'Trend': 1.0},
        'buy_n': 2,
        'prefilter_n': 1,
    })
    assert 'prefilter_n' not in cfg


def test_v9_dual_shadow_ga_materializes_all_nested_search_parameters():
    cfg = generate_initial_configs(1, profile_name='v9_dual_shadow')[0]
    spaces = get_profile_search_spaces('v9_dual_shadow')
    definitions = {p['key']: p for p in get_intrinsic_params()}
    overlay = cfg['trend_risk_overlay']

    assert overlay['enabled'] is True
    assert overlay['mode'] == 'dual_strategy'
    assert 'sell_m' not in spaces
    assert cfg['sell_m'] == cfg['buy_n']
    for key in spaces:
        definition = definitions.get(key)
        if definition and definition.get('config_group') == 'trend_risk_overlay':
            assert overlay[definition['config_key']] in spaces[key]
    assert set(cfg['weights']) == {
        'AmihudIlliquidity', 'TrueMarketCap', 'VolumeCV',
        'AmountBasedSmallCap', 'TrendReversalV7',
    }


def test_v9_dual_shadow_repair_forces_sell_m_to_buy_n():
    cfg = generate_initial_configs(1, profile_name='v9_dual_shadow')[0]
    cfg['sell_m'] = cfg['buy_n'] + 10

    assert repair_config(cfg, profile_name='v9_dual_shadow') is True
    assert cfg['sell_m'] == cfg['buy_n']


def test_v9_dual_shadow_saved_payload_forces_sell_m_to_buy_n():
    payload = strategy_config_payload('v9_dual_shadow', {
        'weights': {'AmihudIlliquidity': 1.0},
        'buy_n': 30,
        'sell_m': 50,
    })

    assert payload['individual_config']['sell_m'] == 30


def test_v11_strict_volume_profile_is_low_dimensional_and_frozen():
    profile_name = 'v11_strict_volume_low_dim'
    profile = get_profile(profile_name)
    configs = generate_initial_configs(20, profile_name=profile_name)

    assert get_profile_search_spaces(profile_name) == {
        'buy_n': [20, 30, 40, 50],
    }
    assert profile['training_objective'] == {
        'mode': 'robust_calmar',
        'folds': 3,
        'full_weight': 0.5,
        'min_average_exposure': 0.45,
    }
    expected_weights = {
        'AmihudIlliquidity', 'TrueMarketCap', 'VolumeCVStrict',
        'AmountBasedSmallCap', 'TrendReversalV7',
    }
    for cfg in configs:
        assert set(cfg['weights']) == expected_weights
        assert cfg['weights']['TrueMarketCap'] == 1.0
        assert all(
            weight in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
            for name, weight in cfg['weights'].items()
            if name != 'TrueMarketCap'
        )
        assert cfg['buy_n'] in {20, 30, 40, 50}
        assert cfg['sell_m'] == cfg['buy_n']
        assert cfg['timing_enabled'] is False
        assert cfg['trend_risk_overlay'] == {
            'floor': 0.0,
            'ceiling': 1.0,
            'momentum_center': -0.055,
            'momentum_scale': 0.012,
            'ma_center': 1.01,
            'ma_scale': 0.009,
            'softmin_sharpness': 4.0,
            'slope': 2.0,
            'momentum_window': 10,
            'ma_window': 20,
            'strategy_weight': 0.8,
            'strategy_momentum_window': 5,
            'strategy_momentum_center': -0.044,
            'strategy_momentum_scale': 0.015,
            'strategy_ma_window': 20,
            'strategy_ma_center': 1.014,
            'strategy_ma_scale': 0.009,
            'strategy_softmin_sharpness': 4.0,
            'strategy_slope': 2.0,
            'enabled': True,
            'mode': 'dual_strategy',
        }


def test_v11_nine_fold_profile_only_changes_training_diagnostic():
    baseline = get_profile('v11_strict_volume_low_dim')
    diagnostic = get_profile('v11_strict_volume_low_dim_9fold')

    assert diagnostic['factor_classes'] == baseline['factor_classes']
    assert diagnostic['fixed_weights'] == baseline['fixed_weights']
    assert diagnostic['weight_search_spaces'] == baseline['weight_search_spaces']
    assert diagnostic['search_spaces'] == baseline['search_spaces']
    assert diagnostic['fixed_parameters'] == baseline['fixed_parameters']
    assert diagnostic['constraints'] == baseline['constraints']
    assert diagnostic['training_objective'] == {
        'mode': 'robust_calmar',
        'folds': 9,
        'full_weight': 0.5,
        'min_average_exposure': 0.45,
    }


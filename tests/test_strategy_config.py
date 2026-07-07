from core.ga import get_intrinsic_params, get_mode_configs, get_profile_factor_names, get_profile_search_spaces
from core.strategy_config import load_strategy_config


def test_strategy_yaml_drives_ga_profiles():
    params = {p['key'] for p in get_intrinsic_params()}
    assert {'buy_n', 'sell_m', 'stock_pool', 'timing_base'} <= params
    assert 'ga' in get_mode_configs('core')
    assert 'TrueMarketCap' in get_profile_factor_names('core6')
    assert get_profile_search_spaces('core6')['buy_n'][0] == 20


def test_final_config_only_uses_weights_as_runtime_factors():
    cfg = load_strategy_config('configs/config.json')
    assert cfg['profile_name'] == 'core6'
    assert cfg['factor_names'] == list(cfg['individual_config']['weights'])
    assert [c.__name__ for c in cfg['factor_classes']] == cfg['factor_names']
    assert cfg['individual_config']['sell_m'] == cfg['individual_config']['buy_n']
    assert cfg['individual_config']['cash_reserve_ratio'] == 0.25

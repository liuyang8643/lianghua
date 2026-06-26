"""最终策略配置读取入口。

运行配置只保留 `ga_profile` 和 `individual_config`；GA 过程指标写入 results。
"""
import json
from pathlib import Path

from core.factors.registry import get_factor_class
from core.ga import resolve_profile_name


def normalize_individual_config(config: dict) -> dict:
    cfg = dict(config)
    if 'sell_m' not in cfg:
        cfg['sell_m'] = cfg['buy_n']
    if 'timing_enabled' not in cfg:
        cfg['timing_enabled'] = 'timing_base' in cfg and cfg['timing_base'] is not None
    if 'rebalance' not in cfg:
        cfg['rebalance'] = True
    if 'filter_factors' not in cfg:
        cfg['filter_factors'] = {}
    elif isinstance(cfg['filter_factors'], list):
        cfg['filter_factors'] = {name: True for name in cfg['filter_factors']}
    if 'empty_months' not in cfg:
        cfg['empty_months'] = None
    if 'cash_reserve_ratio' not in cfg:
        cfg['cash_reserve_ratio'] = 0.0
    return cfg


def strategy_config_payload(profile_name: str, individual_config: dict) -> dict:
    return {
        'ga_profile': profile_name,
        'individual_config': normalize_individual_config(individual_config),
    }


def load_strategy_config(path: str | Path) -> dict:
    config_data = json.loads(Path(path).read_text(encoding='utf-8'))
    profile_name = resolve_profile_name(config_data)
    individual_config = normalize_individual_config(config_data['individual_config'])
    factor_names = list(individual_config['weights'])
    filter_factor_names = [name for name, enabled in individual_config.get('filter_factors', {}).items() if enabled]
    return {
        'profile_name': profile_name,
        'individual_config': individual_config,
        'factor_names': factor_names,
        'factor_classes': [get_factor_class(name) for name in factor_names],
        'filter_factor_names': filter_factor_names,
        'filter_factor_classes': [get_factor_class(name) for name in filter_factor_names],
    }

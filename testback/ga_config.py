"""GA profile config — loads YAML, resolves factor classes by name."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from core.factors.TrueMarketCap import TrueMarketCap
from core.factors.AmountBasedSmallCap import AmountBasedSmallCap
from core.factors.MarginExpansion import MarginExpansion
from core.factors.CashFlowQuality import CashFlowQuality

_FACTOR_REGISTRY = {
    cls.__name__: cls
    for cls in [
        TrueMarketCap, AmountBasedSmallCap,
        MarginExpansion, CashFlowQuality,
    ]
}

_CFG = yaml.safe_load((Path(__file__).parent.parent / 'configs' / 'ga_profiles.yaml').read_text())

DEFAULT_GA_PROFILE: str = _CFG['default_profile']
SEARCH_SPACE_VERSION: str = _CFG['version']

_BOARD_PREFIXES = [str(p) for p in _CFG['search_spaces']['board_prefixes']]
_STOCK_POOL_SPACE = [tuple(c) for r in range(1, 5) for c in combinations(_BOARD_PREFIXES, r)]
_SEARCH_SPACES = {**_CFG['search_spaces'], 'stock_pool': _STOCK_POOL_SPACE}
del _SEARCH_SPACES['board_prefixes']

_COMMON_MODE_CONFIGS: dict[str, dict] = _CFG['mode_configs']

_GA_PROFILES_RAW: dict[str, dict] = _CFG['profiles']

GA_PROFILES: dict[str, dict] = {}
for _name, _p in _GA_PROFILES_RAW.items():
    _classes = [_FACTOR_REGISTRY[n] for n in _p['factor_classes']]
    _factors = [c.__name__ for c in _classes]
    _spaces = {}
    for _k in _p['search_spaces']:
        _v = _SEARCH_SPACES.get(_k)
        if _v is not None:
            _spaces[_k] = _v
        elif _k == 'factor_choice':
            _spaces[_k] = _p['factor_choice_space']
    _profile = {
        'desc': _p['desc'],
        'factor_classes': _classes,
        'fixed_weights': {n: 1.0 for n in _factors} if _p.get('factor_choice_space') is None else None,
        'fixed_temperatures': {n: 1.0 for n in _factors},
        'search_spaces': _spaces,
        'preload_start_date': _p['preload_start'],
        'preload_end_date': _p['preload_end'],
        'mode_configs': deepcopy(_COMMON_MODE_CONFIGS),
    }
    if _p.get('factor_choice_space'):
        _profile['factor_choice_space'] = _p['factor_choice_space']
    if _p.get('weight_range'):
        min_w, max_w, step = _p['weight_range']
        vals = [0.0 if abs(x) < 1e-9 else round(float(x), 2) for x in np.arange(min_w, max_w + step / 2, step)]
        _profile['weight_search_spaces'] = {n: vals for n in _factors}
        _profile['fixed_weights'] = None
    GA_PROFILES[_name] = _profile


def get_profile(profile_name: str | None = None) -> dict:
    resolved = profile_name or DEFAULT_GA_PROFILE
    p = GA_PROFILES.get(resolved)
    if p is None:
        raise ValueError(f'未知 GA profile: {resolved}，可选值: {", ".join(sorted(GA_PROFILES))}')
    return p


def resolve_profile_name(config_data: dict | None = None, fallback: str = DEFAULT_GA_PROFILE) -> str:
    name = fallback
    if config_data:
        candidate = config_data.get('ga_profile')
        if isinstance(candidate, str) and candidate.strip():
            name = candidate.strip()
    get_profile(name)
    return name


def get_mode_configs(profile_name: str | None = None) -> dict:
    return deepcopy(get_profile(profile_name)['mode_configs'])


def get_profile_factor_classes(profile_name: str | None = None) -> list:
    return list(get_profile(profile_name)['factor_classes'])


def get_profile_factor_names(profile_name: str | None = None) -> list[str]:
    return [c.__name__ for c in get_profile_factor_classes(profile_name)]


def get_profile_fixed_weights(profile_name: str | None = None) -> dict:
    profile = get_profile(profile_name)
    fixed = profile.get('fixed_weights')
    if fixed is not None:
        return dict(fixed)
    spaces = profile.get('weight_search_spaces')
    if spaces:
        return {k: 1.0 for k in spaces}
    return {c.__name__: 1.0 for c in profile['factor_classes']}


def get_profile_fixed_temperatures(profile_name: str | None = None) -> dict:
    return dict(get_profile(profile_name)['fixed_temperatures'])


def get_profile_search_spaces(profile_name: str | None = None) -> dict:
    return deepcopy(get_profile(profile_name)['search_spaces'])


def get_profile_preload_range(profile_name: str | None = None) -> tuple:
    p = get_profile(profile_name)
    return p['preload_start_date'], p['preload_end_date']


def get_profile_metadata(profile_name: str | None = None) -> dict:
    resolved = resolve_profile_name({'ga_profile': profile_name} if profile_name else None)
    return {
        'ga_profile': resolved,
        'all_factor_names': get_profile_factor_names(resolved),
        'search_space_version': SEARCH_SPACE_VERSION,
    }


def get_profile_weight_search_spaces(profile_name: str | None = None) -> dict | None:
    return deepcopy(get_profile(profile_name).get('weight_search_spaces'))


def _sample_from_space(space: list):
    import random
    return random.choice(space)


def _sample_from_search_space(key: str, profile_name: str | None = None):
    space = get_profile_search_spaces(profile_name).get(key)
    return _sample_from_space(space) if space else None


def sample_weights(profile_name: str | None = None) -> dict:
    import random
    spaces = get_profile_weight_search_spaces(profile_name)
    if spaces:
        return {k: random.choice(v) for k, v in spaces.items()}
    return get_profile_fixed_weights(profile_name)


def sample_position_count(profile_name: str | None = None):
    return _sample_from_search_space('position_count', profile_name)


def sample_stock_pool(profile_name: str | None = None):
    return _sample_from_search_space('stock_pool', profile_name)


def sample_holding_period(profile_name: str | None = None):
    return _sample_from_search_space('holding_period', profile_name)


def sample_timing_base(profile_name: str | None = None):
    return _sample_from_search_space('timing_base', profile_name)


def sample_timing_leverage(profile_name: str | None = None):
    return _sample_from_search_space('timing_leverage', profile_name)


def sample_timing_direction(profile_name: str | None = None):
    return _sample_from_search_space('timing_direction', profile_name)


def sample_timing_enabled(profile_name: str | None = None):
    return _sample_from_search_space('timing_enabled', profile_name)


def sample_timing_window(profile_name: str | None = None):
    return _sample_from_search_space('timing_window', profile_name)


def sample_timing_index(profile_name: str | None = None):
    return _sample_from_search_space('timing_index', profile_name)


def sample_factor_choice(profile_name: str | None = None):
    space = get_profile_search_spaces(profile_name).get('factor_choice')
    return _sample_from_space(space) if space else None


def repair_config(config: dict, profile_name: str | None = None) -> bool:
    """校验并修复 checkpoint 恢复的配置，将不在当前搜索空间内的参数重新随机化。"""
    import random
    spaces = get_profile_search_spaces(profile_name)
    weight_spaces = get_profile_weight_search_spaces(profile_name)
    fixed_weights = get_profile_fixed_weights(profile_name)

    changed = False

    pos_space = spaces.get('position_count')
    if pos_space and config.get('buy_n') not in pos_space:
        config['buy_n'] = random.choice(pos_space)
        config['sell_m'] = config['buy_n']
        changed = True

    field_map = {
        'stock_pool': 'stock_pool',
        'holding_period': 'holding_period',
        'timing_base': 'timing_base',
        'timing_leverage': 'timing_leverage',
        'timing_direction': 'timing_direction',
        'timing_enabled': 'timing_enabled',
        'timing_window': 'timing_window',
        'timing_index': 'timing_index',
    }
    for space_key, cfg_key in field_map.items():
        space = spaces.get(space_key)
        if space and cfg_key in config and config[cfg_key] not in space:
            config[cfg_key] = random.choice(space)
            changed = True

    old_weights = config.get('weights', {})
    if weight_spaces:
        new_weights = {}
        for k, vals in weight_spaces.items():
            if k in old_weights and old_weights[k] in vals:
                new_weights[k] = old_weights[k]
            else:
                new_weights[k] = random.choice(vals)
                changed = True
        if set(old_weights) != set(weight_spaces):
            changed = True
        config['weights'] = new_weights
    elif fixed_weights:
        if old_weights != fixed_weights:
            config['weights'] = dict(fixed_weights)
            changed = True

    return changed


def build_individual_config(
    position_count: int | None = None,
    weights: dict | None = None,
    factor_choice: str | None = None,
    stock_pool: list | None = None,
    holding_period: int | None = None,
    timing_base: float | None = None,
    timing_leverage: float | None = None,
    timing_direction: int | None = None,
    timing_enabled: bool | None = None,
    timing_window: int | None = None,
    timing_index: str | None = None,
    profile_name: str | None = None,
) -> dict:
    resolved_choice = factor_choice or None
    if weights is None:
        if resolved_choice and isinstance(resolved_choice, str):
            weights = {resolved_choice: 1.0}
        else:
            weights = get_profile_fixed_weights(profile_name)
    else:
        weights = dict(weights)

    if factor_choice:
        filtered = {}
        for k, v in weights.items():
            if k == factor_choice:
                filtered[k] = v
        weights = filtered

    config: dict = {
        'weights': weights,
        'buy_n': position_count if position_count is not None else sample_position_count(profile_name=profile_name),
        'sell_m': position_count if position_count is not None else sample_position_count(profile_name=profile_name),
        'temperatures': get_profile_fixed_temperatures(profile_name),
    }
    if stock_pool:
        config['stock_pool'] = stock_pool
    if holding_period:
        config['holding_period'] = holding_period
    config['timing_enabled'] = bool(timing_enabled)
    if timing_enabled:
        if timing_base is not None:
            config['timing_base'] = timing_base
        if timing_leverage is not None:
            config['timing_leverage'] = timing_leverage
        if timing_direction is not None:
            config['timing_direction'] = timing_direction
        if timing_window is not None:
            config['timing_window'] = timing_window
        if timing_index is not None:
            config['timing_index'] = timing_index
    config['rebalance'] = True
    return config


def generate_initial_configs(count: int, profile_name: str | None = None) -> list[dict]:
    import random
    profile = get_profile(profile_name)
    has_weight = profile.get('weight_search_spaces') is not None
    has_factor_choice = profile.get('factor_choice_space') is not None
    search_spaces = get_profile_search_spaces(profile_name)

    configs = []
    for _ in range(count):
        pc = sample_position_count(profile_name=profile_name)
        fc = sample_factor_choice(profile_name=profile_name) if has_factor_choice else None
        sp = sample_stock_pool(profile_name=profile_name) if 'stock_pool' in search_spaces else None
        hp = sample_holding_period(profile_name=profile_name) if 'holding_period' in search_spaces else None
        te = sample_timing_enabled(profile_name=profile_name) if 'timing_enabled' in search_spaces else None
        tb = sample_timing_base(profile_name=profile_name) if te else None
        tl = sample_timing_leverage(profile_name=profile_name) if te else None
        td = sample_timing_direction(profile_name=profile_name) if te else None
        tw = sample_timing_window(profile_name=profile_name) if te else None
        ti = sample_timing_index(profile_name=profile_name) if te else None
        w = sample_weights(profile_name=profile_name) if has_weight else None
        configs.append(build_individual_config(pc, weights=w, factor_choice=fc, stock_pool=sp,
                                                holding_period=hp, timing_base=tb, timing_leverage=tl,
                                                timing_direction=td, timing_enabled=te,
                                                timing_window=tw, timing_index=ti,
                                                profile_name=profile_name))
    return configs

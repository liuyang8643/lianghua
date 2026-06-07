"""GA 搜索空间采样 + individual config 构建"""
import random
from ._profiles import (
    get_profile, get_profile_search_spaces, get_profile_weight_search_spaces,
    get_profile_fixed_weights, get_profile_fixed_temperatures,
)


def _sample_from_space(space: list):
    return random.choice(space)


def _sample_space_key(key: str, profile_name: str | None = None):
    space = get_profile_search_spaces(profile_name).get(key)
    return _sample_from_space(space) if space else None


def sample_weights(profile_name: str | None = None) -> dict:
    spaces = get_profile_weight_search_spaces(profile_name)
    fixed = get_profile_fixed_weights(profile_name)
    if spaces:
        sampled = {k: random.choice(v) for k, v in spaces.items()}
        return {**fixed, **sampled} if fixed else sampled
    return fixed


def sample_position_count(profile_name: str | None = None):
    return _sample_space_key('position_count', profile_name)


def sample_stock_pool(profile_name: str | None = None):
    return _sample_space_key('stock_pool', profile_name)


def sample_holding_period(profile_name: str | None = None):
    return _sample_space_key('holding_period', profile_name)


def sample_timing_base(profile_name: str | None = None):
    return _sample_space_key('timing_base', profile_name)


def sample_timing_leverage(profile_name: str | None = None):
    return _sample_space_key('timing_leverage', profile_name)


def sample_timing_direction(profile_name: str | None = None):
    return _sample_space_key('timing_direction', profile_name)


def sample_timing_window(profile_name: str | None = None):
    return _sample_space_key('timing_window', profile_name)


def sample_timing_index(profile_name: str | None = None):
    return _sample_space_key('timing_index', profile_name)


def sample_factor_choice(profile_name: str | None = None):
    space = get_profile_search_spaces(profile_name).get('factor_choice')
    return _sample_from_space(space) if space else None


def build_individual_config(
    position_count: int | None = None,
    weights: dict | None = None,
    factor_choice: str | None = None,
    stock_pool: list | None = None,
    holding_period: int | None = None,
    timing_base: float | None = None,
    timing_leverage: float | None = None,
    timing_direction: int | None = None,
    timing_window: int | None = None,
    timing_index: str | None = None,
    profile_name: str | None = None,
) -> dict:
    if weights is None:
        weights = {factor_choice: 1.0} if factor_choice else get_profile_fixed_weights(profile_name)
    else:
        weights = dict(weights)
    if factor_choice:
        weights = {k: v for k, v in weights.items() if k == factor_choice}

    n = position_count if position_count is not None else sample_position_count(profile_name=profile_name)
    cfg: dict = {
        'weights': weights,
        'buy_n': n,
        'sell_m': n,
        'temperatures': get_profile_fixed_temperatures(profile_name),
    }
    if stock_pool:
        cfg['stock_pool'] = stock_pool
    if holding_period:
        cfg['holding_period'] = holding_period
    cfg['timing_enabled'] = timing_base is not None
    if timing_base is not None:
        cfg['timing_base'] = timing_base
    if timing_leverage is not None:
        cfg['timing_leverage'] = timing_leverage
    if timing_direction is not None:
        cfg['timing_direction'] = timing_direction
    if timing_window is not None:
        cfg['timing_window'] = timing_window
    if timing_index is not None:
        cfg['timing_index'] = timing_index
    cfg['rebalance'] = True
    return cfg


def repair_config(config: dict, profile_name: str | None = None) -> bool:
    spaces = get_profile_search_spaces(profile_name)
    weight_spaces = get_profile_weight_search_spaces(profile_name)
    fixed_weights = get_profile_fixed_weights(profile_name)
    changed = False

    pos_space = spaces.get('position_count')
    if pos_space and config.get('buy_n') not in pos_space:
        config['buy_n'] = random.choice(pos_space)
        config['sell_m'] = config['buy_n']
        changed = True

    for space_key, cfg_key in [
        ('stock_pool', 'stock_pool'), ('holding_period', 'holding_period'),
        ('timing_base', 'timing_base'), ('timing_leverage', 'timing_leverage'),
        ('timing_direction', 'timing_direction'), ('timing_window', 'timing_window'),
        ('timing_index', 'timing_index'),
    ]:
        space = spaces.get(space_key)
        if space and cfg_key in config and config[cfg_key] not in space:
            config[cfg_key] = random.choice(space)
            changed = True

    old_weights = config.get('weights', {})
    if weight_spaces:
        new_weights = {}
        if fixed_weights:
            for k, v in fixed_weights.items():
                new_weights[k] = v
                if old_weights.get(k) != v:
                    changed = True
        for k, vals in weight_spaces.items():
            new_weights[k] = old_weights[k] if k in old_weights and old_weights[k] in vals else random.choice(vals)
            if new_weights[k] != old_weights.get(k):
                changed = True
        expected = set(weight_spaces) | (set(fixed_weights) if fixed_weights else set())
        if set(old_weights) != expected:
            changed = True
        config['weights'] = new_weights
    elif fixed_weights and old_weights != fixed_weights:
        config['weights'] = dict(fixed_weights)
        changed = True

    return changed


def generate_initial_configs(count: int, profile_name: str | None = None) -> list[dict]:
    p = get_profile(profile_name)
    has_weight = p.get('weight_search_spaces') is not None
    has_fc = p.get('factor_choice_space') is not None
    spaces = get_profile_search_spaces(profile_name)

    configs = []
    for _ in range(count):
        pc = sample_position_count(profile_name=profile_name)
        fc = sample_factor_choice(profile_name=profile_name) if has_fc else None
        sp = sample_stock_pool(profile_name=profile_name) if 'stock_pool' in spaces else None
        hp = sample_holding_period(profile_name=profile_name) if 'holding_period' in spaces else None
        tb = sample_timing_base(profile_name=profile_name) if 'timing_base' in spaces else None
        tl = sample_timing_leverage(profile_name=profile_name) if 'timing_leverage' in spaces else None
        td = sample_timing_direction(profile_name=profile_name) if 'timing_direction' in spaces else None
        tw = sample_timing_window(profile_name=profile_name) if 'timing_window' in spaces else None
        ti = sample_timing_index(profile_name=profile_name) if 'timing_index' in spaces else None
        w = sample_weights(profile_name=profile_name) if has_weight else None
        configs.append(build_individual_config(pc, weights=w, factor_choice=fc, stock_pool=sp,
                                                holding_period=hp, timing_base=tb, timing_leverage=tl,
                                                timing_direction=td, timing_window=tw, timing_index=ti,
                                                profile_name=profile_name))
    return configs

"""GA 搜索空间采样 + individual config 构建

所有可搜索参数由 INTRINSIC_PARAMS 注册表驱动，新增参数只需在注册表加一条。
"""
import random
from ._profiles import (
    get_profile, get_profile_search_spaces, get_profile_weight_search_spaces,
    get_profile_fixed_weights, get_profile_fixed_temperatures,
    get_intrinsic_params,
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


# ---- 向后兼容的具名采样函数 (由 INTRINSIC_PARAMS 注册) ----
_sample_registry = {p['key']: p for p in get_intrinsic_params()}

for _key, _def in _sample_registry.items():
    _fn_name = f'sample_{_key}'
    _fn = (lambda k=_key: lambda profile_name=None: _sample_space_key(k, profile_name))()
    _fn.__name__ = _fn_name
    _fn.__qualname__ = _fn_name
    globals()[_fn_name] = _fn


def sample_factor_choice(profile_name: str | None = None):
    space = get_profile_search_spaces(profile_name).get('factor_choice')
    return _sample_from_space(space) if space else None


# ---- 核心构建函数 ----

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
    amount_filter_pct: int | None = None,
    market_cap_filter_pct: int | None = None,
    profile_name: str | None = None,
) -> dict:
    if weights is None:
        weights = get_profile_fixed_weights(profile_name)
    else:
        weights = dict(weights)
    if factor_choice:
        weights = {k: (v if k == factor_choice else 0.0) for k, v in weights.items()}

    n = position_count if position_count is not None else sample_position_count(profile_name=profile_name)
    cfg: dict = {
        'weights': weights,
        'buy_n': n,
        'sell_m': n,
        'temperatures': get_profile_fixed_temperatures(profile_name),
    }

    # 内置参数：从注册表驱动
    kwargs = locals()
    for pdef in get_intrinsic_params():
        key = pdef['key']
        ck = pdef['config_key']
        val = kwargs.get(key)
        if val is None and key in _sample_registry:
            space = get_profile_search_spaces(profile_name).get(key)
            if space:
                val = _sample_from_space(space)
        if val is not None:
            if pdef['type'] == 'int' and val == 0:
                continue  # 零值不写入 config, 视为关闭
            cfg[ck] = val

    # timing_enabled 特殊处理
    cfg['timing_enabled'] = cfg.get('timing_base') is not None
    cfg['rebalance'] = True
    return cfg


def repair_config(config: dict, profile_name: str | None = None) -> bool:
    spaces = get_profile_search_spaces(profile_name)
    weight_spaces = get_profile_weight_search_spaces(profile_name)
    fixed_weights = get_profile_fixed_weights(profile_name)
    changed = False

    # position_count 特殊处理(buy_n/sell_m 联动)
    pos_space = spaces.get('position_count')
    if pos_space and config.get('buy_n') not in pos_space:
        config['buy_n'] = random.choice(pos_space)
        config['sell_m'] = config['buy_n']
        changed = True

    # 内置参数
    for pdef in get_intrinsic_params():
        if pdef['key'] == 'position_count':
            continue  # 上面已处理
        space = spaces.get(pdef['key'])
        ck = pdef['config_key']
        if space and ck in config and config[ck] not in space:
            config[ck] = random.choice(space)
            changed = True

    # 权重
    old_weights = config['weights']
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

        # 内置参数
        extra = {}
        for pdef in get_intrinsic_params():
            key = pdef['key']
            if key == 'position_count':
                continue
            if key in spaces:
                extra[key] = _sample_space_key(key, profile_name)

        w = sample_weights(profile_name=profile_name) if has_weight else None
        configs.append(build_individual_config(
            pc, weights=w, factor_choice=fc,
            stock_pool=extra.get('stock_pool'),
            holding_period=extra.get('holding_period'),
            timing_base=extra.get('timing_base'),
            timing_leverage=extra.get('timing_leverage'),
            timing_direction=extra.get('timing_direction'),
            timing_window=extra.get('timing_window'),
            timing_index=extra.get('timing_index'),
            amount_filter_pct=extra.get('amount_filter_pct'),
            market_cap_filter_pct=extra.get('market_cap_filter_pct'),
            profile_name=profile_name))
    return configs

"""GA 搜索空间采样 + individual config 构建

所有可搜索参数由 INTRINSIC_PARAMS 注册表驱动，新增参数只需在注册表加一条。
"""
import random
from ._profiles import (
    get_profile, get_profile_search_spaces, get_profile_weight_search_spaces,
    get_profile_fixed_weights, get_intrinsic_params,
    get_profile_fixed_parameters,
    get_config_param,
)


def _sample_from_space(space: list):
    return random.choice(space)


def _sample_space_key(key: str, profile_name: str | None = None):
    space = get_profile_search_spaces(profile_name).get(key)
    if space:
        return _sample_from_space(space)
    return _sample_registry.get(key, {}).get('default')


def sample_buy_n(profile_name: str | None = None) -> int:
    space = get_profile_search_spaces(profile_name).get('buy_n')
    return _sample_from_space(space) if space else 20


def sample_weights(profile_name: str | None = None) -> dict:
    spaces = get_profile_weight_search_spaces(profile_name)
    fixed = get_profile_fixed_weights(profile_name)
    if spaces:
        sampled = {k: random.choice(v) for k, v in spaces.items()}
        return {**fixed, **sampled} if fixed else sampled
    return fixed


# ---- 参数采样函数 (由 INTRINSIC_PARAMS 注册表自动生成) ----
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
    buy_n: int | None = None,
    sell_m: int | None = None,
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
    rebalance: bool | None = None,
    limit_up_protection: bool | None = None,
    cash_reserve_ratio: float | None = None,
    filter_factors: dict | None = None,
    profile_name: str | None = None,
    **search_values,
) -> dict:
    if weights is None:
        weights = get_profile_fixed_weights(profile_name)
    else:
        weights = dict(weights)
    if factor_choice:
        weights = {k: (v if k == factor_choice else 0.0) for k, v in weights.items()}

    if filter_factors is None:
        filter_factors = _sample_space_key('filter_factors', profile_name)
    n = buy_n if buy_n is not None else sample_buy_n(profile_name=profile_name)
    cfg: dict = {
        'weights': weights,
        'buy_n': n,
        'sell_m': n,
        'filter_factors': dict(filter_factors or {}),
    }

    # 内置参数：从注册表驱动（buy_n 已在上面处理，跳过）
    kwargs = locals()
    for pdef in get_intrinsic_params():
        key = pdef['key']
        if key == 'buy_n':
            continue
        ck = pdef['config_key']
        val = search_values.get(key, kwargs.get(key))
        if val is None and key in _sample_registry:
            val = _sample_space_key(key, profile_name)
        if val is not None:
            if pdef['type'] == 'int' and val == 0:
                continue  # 零值不写入 config, 视为关闭
            group = pdef.get('config_group')
            if group:
                cfg.setdefault(group, {})[ck] = val
            else:
                cfg[ck] = val

    # sell_m >= buy_n 约束
    if cfg['sell_m'] < cfg['buy_n']:
        cfg['sell_m'] = cfg['buy_n']

    # timing_enabled 特殊处理
    cfg['timing_enabled'] = cfg.get('timing_base') is not None
    # rebalance 兜底默认值（不在 search_spaces 时不会经过注册表采样循环）
    if 'rebalance' not in cfg:
        cfg['rebalance'] = True
    fixed = get_profile_fixed_parameters(profile_name)
    for key, value in fixed.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    if get_profile(profile_name).get('constraints', {}).get('sell_m_equals_buy_n'):
        cfg['sell_m'] = cfg['buy_n']
    return cfg


def repair_config(config: dict, profile_name: str | None = None) -> bool:
    spaces = get_profile_search_spaces(profile_name)
    weight_spaces = get_profile_weight_search_spaces(profile_name)
    fixed_weights = get_profile_fixed_weights(profile_name)
    changed = False

    # buy_n / sell_m 修复
    buy_n_space = spaces.get('buy_n')
    sell_m_space = spaces.get('sell_m')
    if buy_n_space and config['buy_n'] not in buy_n_space:
        config['buy_n'] = random.choice(buy_n_space)
        changed = True
    if sell_m_space and config['sell_m'] not in sell_m_space:
        config['sell_m'] = random.choice(sell_m_space)
        changed = True

    # 内置参数
    for pdef in get_intrinsic_params():
        if pdef['key'] == 'buy_n':
            continue  # 上面已处理
        space = spaces.get(pdef['key'])
        value = get_config_param(config, pdef)
        if space and value not in space:
            replacement = random.choice(space)
            group = pdef.get('config_group')
            if group:
                config.setdefault(group, {})[pdef['config_key']] = replacement
            else:
                config[pdef['config_key']] = replacement
            changed = True

    filters = config.get('filter_factors', {})
    if isinstance(filters, dict):
        filter_space = spaces.get('filter_factors')
        if filter_space and filters not in filter_space:
            replacement = random.choice(filter_space)
            config['filter_factors'] = dict(replacement)
            changed = True

    # sell_m >= buy_n 约束
    if config['sell_m'] < config['buy_n']:
        config['sell_m'] = config['buy_n']
        changed = True
    if (get_profile(profile_name).get('constraints', {}).get('sell_m_equals_buy_n')
            and config['sell_m'] != config['buy_n']):
        config['sell_m'] = config['buy_n']
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
        buy_n_val = sample_buy_n(profile_name=profile_name)
        fc = sample_factor_choice(profile_name=profile_name) if has_fc else None

        # 内置参数（不在 search_spaces 中的参数使用 INTRINSIC_PARAMS default）
        extra = {}
        for pdef in get_intrinsic_params():
            key = pdef['key']
            if key == 'buy_n':
                continue
            extra[key] = _sample_space_key(key, profile_name)

        w = sample_weights(profile_name=profile_name) if has_weight else None
        configs.append(build_individual_config(
            buy_n=buy_n_val, weights=w, factor_choice=fc,
            profile_name=profile_name, **extra))
    return configs

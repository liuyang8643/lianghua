"""策略总配置加载与 GA profile 查询。"""
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from core.factors.registry import get_factor_class

_YAML_PATH: Path | None = None
_loaded = False

DEFAULT_GA_PROFILE = ''
SEARCH_SPACE_VERSION = ''
GA_PROFILES: dict[str, dict] = {}

# ============================================================
# 统一参数注册表 — 所有 GA 可搜索的配置维度
# ============================================================
# type: 'int' | 'float' | 'categorical' | 'stock_pool' | 'weights'
# config_key: 在 individual_config dict 中的键名
# display: 日志缩写
# default: 当不在 profile search_spaces 中时的值
# crossover: True=参与交叉, False=不参与(如 buy_n, stock_pool 等)
# config_key_for_mutate: 变异时从搜索空间重新采样的键(可不同于 config_key)
INTRINSIC_PARAMS: list[dict] = []

def get_intrinsic_params() -> list[dict]:
    return deepcopy(INTRINSIC_PARAMS)


def set_yaml_path(path: Path):
    global _YAML_PATH
    _YAML_PATH = path


def _get_yaml_path() -> Path:
    if _YAML_PATH:
        return _YAML_PATH
    return Path(__file__).parent.parent.parent / 'configs' / 'strategy.yaml'


def _expand_space(raw):
    if not isinstance(raw, dict):
        return raw
    start, end, step = raw['range']
    vals = [0.0 if abs(x) < 1e-9 else round(float(x), 10) for x in np.arange(start, end + step / 2, step)]
    if all(float(v).is_integer() for v in vals):
        vals = [int(v) for v in vals]
    if 'include' in raw:
        vals = list(raw['include']) + vals
    return sorted(set(vals))


def _load():
    global _loaded, DEFAULT_GA_PROFILE, SEARCH_SPACE_VERSION, GA_PROFILES, INTRINSIC_PARAMS
    if _loaded:
        return
    _loaded = True

    cfg = yaml.safe_load((_get_yaml_path()).read_text())
    INTRINSIC_PARAMS = deepcopy(cfg['strategy_parameters'])
    DEFAULT_GA_PROFILE = cfg['default_profile']
    SEARCH_SPACE_VERSION = cfg['version']

    expanded_spaces = {k: _expand_space(v) for k, v in cfg['search_spaces'].items()}
    board_prefixes = [str(p) for p in expanded_spaces['board_prefixes']]
    stock_pool_space = [tuple(c) for r in range(1, 5) for c in combinations(board_prefixes, r)]
    search_spaces = {**expanded_spaces, 'stock_pool': stock_pool_space}
    del search_spaces['board_prefixes']

    mode_configs = cfg['mode_configs']

    for name, raw in cfg['profiles'].items():
        classes = [get_factor_class(n) for n in raw['factor_classes']]
        factor_names = [c.__name__ for c in classes]
        spaces = {}
        for k in raw['search_spaces']:
            v = search_spaces.get(k)
            if v is not None:
                spaces[k] = v
            elif k == 'factor_choice':
                spaces[k] = raw['factor_choice_space']

        profile = {
            'desc': raw['desc'],
            'factor_classes': classes,
            'fixed_weights': {n: 1.0 for n in factor_names} if raw.get('factor_choice_space') is None else None,
            'search_spaces': spaces,
            'preload_start_date': raw['preload_start'],
            'preload_end_date': raw['preload_end'],
            'mode_configs': deepcopy(mode_configs),
        }
        if raw.get('factor_choice_space'):
            profile['factor_choice_space'] = raw['factor_choice_space']
        if raw.get('weight_range'):
            mn, mx, step = raw['weight_range']
            vals = [0.0 if abs(x) < 1e-9 else round(float(x), 2) for x in np.arange(mn, mx + step / 2, step)]
            yaml_fixed = raw.get('fixed_weights')
            if yaml_fixed:
                profile['fixed_weights'] = {n: float(yaml_fixed[n]) for n in yaml_fixed if n in factor_names}
                searchable = [n for n in factor_names if n not in yaml_fixed]
                profile['weight_search_spaces'] = {n: vals for n in searchable} if searchable else None
            else:
                profile['weight_search_spaces'] = {n: vals for n in factor_names}
                profile['fixed_weights'] = None
        GA_PROFILES[name] = profile


def get_profile(name: str | None = None) -> dict:
    _load()
    resolved = name or DEFAULT_GA_PROFILE
    p = GA_PROFILES.get(resolved)
    if p is None:
        raise ValueError(f'未知 GA profile: {resolved}，可选值: {", ".join(sorted(GA_PROFILES))}')
    return p


def resolve_profile_name(config_data: dict | None = None, fallback: str | None = None) -> str:
    _load()
    name = fallback or DEFAULT_GA_PROFILE
    if config_data:
        candidate = config_data.get('ga_profile')
        if isinstance(candidate, str) and candidate.strip():
            name = candidate.strip()
    get_profile(name)
    return name


def get_mode_configs(name: str | None = None) -> dict:
    return deepcopy(get_profile(name)['mode_configs'])


def get_profile_factor_classes(name: str | None = None) -> list:
    return list(get_profile(name)['factor_classes'])


def get_profile_factor_names(name: str | None = None) -> list[str]:
    return [c.__name__ for c in get_profile_factor_classes(name)]


def get_profile_fixed_weights(name: str | None = None) -> dict:
    p = get_profile(name)
    if p.get('fixed_weights'):
        return dict(p['fixed_weights'])
    if p.get('weight_search_spaces'):
        return {k: 1.0 for k in p['weight_search_spaces']}
    return {c.__name__: 1.0 for c in p['factor_classes']}


def get_profile_search_spaces(name: str | None = None) -> dict:
    return deepcopy(get_profile(name)['search_spaces'])


def get_profile_weight_search_spaces(name: str | None = None) -> dict | None:
    return deepcopy(get_profile(name).get('weight_search_spaces'))


def get_profile_preload_range(name: str | None = None) -> tuple:
    p = get_profile(name)
    return p['preload_start_date'], p['preload_end_date']


def get_profile_metadata(name: str | None = None) -> dict:
    resolved = resolve_profile_name({'ga_profile': name} if name else None)
    return {
        'ga_profile': resolved,
        'all_factor_names': get_profile_factor_names(resolved),
        'search_space_version': SEARCH_SPACE_VERSION,
    }


_load()

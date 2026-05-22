"""GA profile config — loads YAML, resolves factor classes by name."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path


import numpy as np
import yaml

from core.factors.SmallCap import SmallCap
from core.factors.TrueMarketCap import TrueMarketCap
from core.factors.SmallCapMarginExpansion import SmallCapMarginExpansion
from core.factors.PureProfitYoyAccel import PureProfitYoyAccel
from core.factors.QualityReversal10D import QualityReversal10D
from core.factors.TMC_GARP_Broad import TMC_GARP_Broad
from core.factors.TMC_GARP_Mult import TMC_GARP_Mult
from core.factors.TMC_ProfitYoy_25_LowVol import TMC_ProfitYoy_25_LowVol
from core.factors.ShortTermReversal import ShortTermReversal
from core.factors.CashFlowQuality import CashFlowQuality
from core.factors.LowTurnover20D import LowTurnover20D
from core.factors.ADX14Trend import ADX14Trend
from core.factors.CloseMom21D import CloseMom21D
from core.factors.CCI14 import CCI14
from core.factors.BB20Position import BB20Position
from core.factors.OvernightGap1D import OvernightGap1D
from core.factors.OBVSlope import OBVSlope
from core.factors.ATR14 import ATR14
from core.factors.PricePosition256D import PricePosition256D
from core.factors.EWMADivergence import EWMADivergence
from core.factors.Aroon14 import Aroon14
from core.factors.ProfitYoy import ProfitYoy
from core.factors.ROE import ROE
from core.factors.EPValuation import EPValuation
from core.factors.MACD import MACD
from core.factors.TRIX import TRIX
from core.factors.KDJ import KDJ
from core.factors.Turnover import Turnover
from core.factors.SAR import SAR

_FACTOR_REGISTRY = {
    cls.__name__: cls
    for cls in [
        SmallCap, TrueMarketCap,
        ShortTermReversal, CashFlowQuality,
        SmallCapMarginExpansion, PureProfitYoyAccel,
        QualityReversal10D, TMC_GARP_Broad, TMC_GARP_Mult,
        TMC_ProfitYoy_25_LowVol,
        LowTurnover20D,
        ADX14Trend, CloseMom21D, CCI14, BB20Position,
        OvernightGap1D, OBVSlope, ATR14, PricePosition256D, EWMADivergence,
        Aroon14, ProfitYoy, ROE, EPValuation,
        MACD, TRIX, KDJ, Turnover, SAR,
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


def sample_holding_period(profile_name: str | None = None, current_value=None):
    return _sample_from_search_space('holding_period', profile_name)


def sample_stock_pool(profile_name: str | None = None, current_value=None):
    return _sample_from_search_space('stock_pool', profile_name)


def sample_factor_choice(profile_name: str | None = None, current_value=None):
    return _sample_from_search_space('factor_choice', profile_name)


def sample_timing_base(profile_name: str | None = None, current_value=None):
    return _sample_from_search_space('timing_base', profile_name)


def sample_timing_leverage(profile_name: str | None = None, current_value=None):
    return _sample_from_search_space('timing_leverage', profile_name)


def sample_timing_direction(profile_name: str | None = None, current_value=None):
    return _sample_from_search_space('timing_direction', profile_name)


def sample_timing_enabled(profile_name: str | None = None, current_value=None):
    return _sample_from_search_space('timing_enabled', profile_name)


def sample_timing_window(profile_name: str | None = None, current_value=None):
    return _sample_from_search_space('timing_window', profile_name)


def sample_timing_index(profile_name: str | None = None, current_value=None):
    return _sample_from_search_space('timing_index', profile_name)


def build_individual_config(position_count: int, freeze_days: int = 0, weights: dict | None = None,
                            factor_choice: str | None = None, stock_pool=None,
                            holding_period=None,
                            timing_base=None, timing_leverage=None, timing_direction=None,
                            timing_enabled=None, timing_window=None, timing_index=None,
                            profile_name: str | None = None) -> dict:
    profile = get_profile(profile_name)
    if factor_choice:
        other = [f for f in profile['factor_classes'] if f.__name__ != factor_choice][0]
        weights = {factor_choice: 1.0, other.__name__: 0.0}
    elif weights is None:
        spaces = get_profile_weight_search_spaces(profile_name)
        weights = sample_weights(profile_name) if spaces else get_profile_fixed_weights(profile_name)
    cfg = {
        'weights': weights,
        'buy_n': position_count,
        'sell_m': position_count,
        'temperatures': get_profile_fixed_temperatures(profile_name),
    }
    for k, v in {'factor_choice': factor_choice, 'stock_pool': stock_pool,
                 'holding_period': holding_period,
                 'timing_base': timing_base, 'timing_leverage': timing_leverage, 'timing_direction': timing_direction,
                 'timing_enabled': timing_enabled, 'timing_window': timing_window, 'timing_index': timing_index}.items():
        if v is not None:
            cfg[k] = v
    if freeze_days:
        cfg['freeze_days'] = freeze_days
    return cfg


def generate_initial_configs(size: int, profile_name: str | None = None) -> list[dict]:
    return [build_individual_config(
        position_count=sample_position_count(profile_name),
        factor_choice=sample_factor_choice(profile_name),
        stock_pool=sample_stock_pool(profile_name),
        holding_period=sample_holding_period(profile_name),
        timing_base=sample_timing_base(profile_name),
        timing_leverage=sample_timing_leverage(profile_name),
        timing_direction=sample_timing_direction(profile_name),
        timing_enabled=sample_timing_enabled(profile_name),
        timing_window=sample_timing_window(profile_name),
        timing_index=sample_timing_index(profile_name),
        profile_name=profile_name,
    ) for _ in range(size)]

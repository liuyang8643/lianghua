from copy import deepcopy
from datetime import date
from typing import Any, Dict

from core.factors import SmallCap, SmallCapKeepST, WMACross

DEFAULT_GA_PROFILE = 'smallcap_only'
SEARCH_SPACE_VERSION = '2026-03-22-smallcap-v1'

GA_POSITION_SPACE = list(range(10, 501, 10))
GA_PRELOAD_START_DATE = date(2006, 1, 1)
GA_PRELOAD_END_DATE = date(2026, 1, 1)

GA_WEIGHT_SPACE = [round(x * 0.1, 1) for x in range(-20, 21)]  # -2.0 ~ 2.0, step 0.1

_COMMON_MODE_CONFIGS = {
  'single': {
    'population_size': 1,
    'generations': 1,
    'period_span': 30,
    'log_level': 'DEBUG',
    'save_charts': True,
    'desc': '单次回测模式',
  },
  'debug': {
    'population_size': 2,
    'generations': 3,
    'period_span': 30,
    'window_days': 60,
    'log_level': 'DEBUG',
    'save_charts': False,
    'desc': '调试模式（最小GA）',
  },
  'ga': {
    'population_size': None,
    'generations': 300,
    'period_span': 250,
    'window_days': 250,
    'log_level': 'INFO',
    'save_charts': False,
    'desc': 'GA优化模式（默认）',
  },
}

GA_PROFILES = {
  DEFAULT_GA_PROFILE: {
    'desc': 'SmallCap-only GA profile',
    'factor_classes': [SmallCap],
    'fixed_weights': {
      'SmallCap': 1.0,
    },
    'fixed_temperatures': {
      'SmallCap': 1.0,
    },
    'search_spaces': {
      'position_count': GA_POSITION_SPACE,
    },
    'preload_start_date': GA_PRELOAD_START_DATE,
    'preload_end_date': GA_PRELOAD_END_DATE,
    'mode_configs': deepcopy(_COMMON_MODE_CONFIGS),
  },
  'smallcap_st_wmacross': {
    'desc': 'SmallCapKeepST + WMACross 双因子权重搜索',
    'factor_classes': [SmallCapKeepST, WMACross],
    'fixed_weights': None,
    'weight_search_spaces': {
      'SmallCapKeepST': GA_WEIGHT_SPACE,
      'WMACross': GA_WEIGHT_SPACE,
    },
    'fixed_temperatures': {
      'SmallCapKeepST': 1.0,
      'WMACross': 1.0,
    },
    'search_spaces': {
      'position_count': [30],
    },
    'preload_start_date': GA_PRELOAD_START_DATE,
    'preload_end_date': date(2025, 12, 31),
    'mode_configs': deepcopy(_COMMON_MODE_CONFIGS),
  },
}


def get_profile(profile_name: str | None = None) -> Dict[str, Any]:
  resolved_name = profile_name or DEFAULT_GA_PROFILE
  profile = GA_PROFILES.get(resolved_name)
  if profile is None:
    available = ', '.join(sorted(GA_PROFILES))
    raise ValueError(f'未知 GA profile: {resolved_name}，可选值: {available}')
  return profile



def resolve_profile_name(config_data: Dict[str, Any] | None = None, fallback: str = DEFAULT_GA_PROFILE) -> str:
  profile_name = fallback
  if config_data:
    candidate = config_data.get('ga_profile')
    if isinstance(candidate, str) and candidate.strip():
      profile_name = candidate.strip()
  get_profile(profile_name)
  return profile_name



def get_mode_configs(profile_name: str | None = None) -> Dict[str, Dict[str, Any]]:
  return deepcopy(get_profile(profile_name)['mode_configs'])



def get_profile_factor_classes(profile_name: str | None = None) -> list[type]:
  return list(get_profile(profile_name)['factor_classes'])



def get_profile_factor_names(profile_name: str | None = None) -> list[str]:
  return [factor_cls.__name__ for factor_cls in get_profile_factor_classes(profile_name)]



def get_profile_fixed_weights(profile_name: str | None = None) -> Dict[str, float]:
  profile = get_profile(profile_name)
  fixed = profile.get('fixed_weights')
  if fixed is not None:
    return dict(fixed)
  weight_spaces = profile.get('weight_search_spaces')
  if weight_spaces:
    return {k: 1.0 for k in weight_spaces}
  return {cls.__name__: 1.0 for cls in profile['factor_classes']}



def get_profile_fixed_temperatures(profile_name: str | None = None) -> Dict[str, float]:
  return dict(get_profile(profile_name)['fixed_temperatures'])



def get_profile_search_spaces(profile_name: str | None = None) -> Dict[str, list]:
  return deepcopy(get_profile(profile_name)['search_spaces'])



def get_profile_preload_range(profile_name: str | None = None) -> tuple[date, date]:
  profile = get_profile(profile_name)
  return profile['preload_start_date'], profile['preload_end_date']



def get_profile_metadata(profile_name: str | None = None) -> Dict[str, Any]:
  resolved_name = resolve_profile_name({'ga_profile': profile_name} if profile_name else None)
  return {
    'ga_profile': resolved_name,
    'all_factor_names': get_profile_factor_names(resolved_name),
    'search_space_version': SEARCH_SPACE_VERSION,
  }



def _sample_from_space(space: list, current_value=None):
  import random

  if current_value is None:
    return random.choice(space)

  try:
    idx = space.index(current_value)
    if idx == 0:
      return random.choice(space[:2])
    if idx == len(space) - 1:
      return random.choice(space[-2:])
    return random.choice(space[idx - 1:idx + 2])
  except ValueError:
    return random.choice(space)



def get_profile_weight_search_spaces(profile_name: str | None = None) -> Dict[str, list] | None:
  return deepcopy(get_profile(profile_name).get('weight_search_spaces'))


def sample_weights(current_weights: Dict[str, float] | None = None, mutation_rate: float = 1.0, profile_name: str | None = None) -> Dict[str, float]:
  """从 weight_search_spaces 采样权重。如有 current_weights 则按 mutation_rate 概率变异。"""
  import random
  weight_spaces = get_profile_weight_search_spaces(profile_name)
  if not weight_spaces:
    return get_profile_fixed_weights(profile_name)
  weights = {}
  for factor_name, space in weight_spaces.items():
    current = current_weights.get(factor_name) if current_weights else None
    if current is not None and random.random() > mutation_rate:
      weights[factor_name] = current
    else:
      weights[factor_name] = _sample_from_space(space, current_value=current)
  return weights


def build_individual_config(position_count: int, freeze_days: int = 0, weights: Dict[str, float] | None = None, profile_name: str | None = None) -> dict:
  if weights is None:
    weight_spaces = get_profile_weight_search_spaces(profile_name)
    if weight_spaces:
      weights = sample_weights(profile_name=profile_name)
    else:
      weights = get_profile_fixed_weights(profile_name)
  config = {
    'weights': weights,
    'buy_n': position_count,
    'sell_m': position_count,
    'temperatures': get_profile_fixed_temperatures(profile_name),
  }
  if freeze_days:
    config['freeze_days'] = freeze_days
  return config



def sample_position_count(current_value=None, profile_name: str | None = None):
  search_spaces = get_profile_search_spaces(profile_name)
  return _sample_from_space(search_spaces['position_count'], current_value=current_value)



def generate_initial_configs(size: int, profile_name: str | None = None) -> list[dict]:
  return [
    build_individual_config(sample_position_count(profile_name=profile_name), profile_name=profile_name)
    for _ in range(size)
  ]

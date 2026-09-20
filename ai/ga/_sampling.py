"""Continuous unit-weight GA sampling through the canonical action schema."""
import random
from env.action_schema import ActionSchema
from env.contracts import DayConfig
from ._profiles import resolve_profile_name

_SCHEMA = ActionSchema()

def sample_turnover_rate(profile_name=None):
    resolve_profile_name(fallback=profile_name)
    field = _SCHEMA.layout[-1]
    return random.uniform(field.minimum, field.maximum)

def sample_weights(profile_name=None):
    resolve_profile_name(fallback=profile_name)
    return {name: random.uniform(0.0, 1.0) for name in _SCHEMA.factor_names}

def build_individual_config(turnover_rate=None, weights=None, profile_name=None):
    resolve_profile_name(fallback=profile_name)
    buy_n = _SCHEMA.fixed_buy_n
    turnover_rate = sample_turnover_rate(profile_name) if turnover_rate is None else turnover_rate
    weights = sample_weights(profile_name) if weights is None else dict(weights)
    config = DayConfig(factor_weights=weights,
        factor_enabled={k: v != 0.0 for k,v in weights.items()},
        filter_flags=dict(zip(_SCHEMA.filter_names, _SCHEMA.fixed_filter_flags)),
        buy_n=buy_n, turnover_rate=turnover_rate,
        limit_up_protection=_SCHEMA.fixed_limit_up_protection,
        rebalance_band_pct=_SCHEMA.fixed_rebalance_band_pct, single_buy_pct=1.0/buy_n)
    return _SCHEMA.to_static_config(_SCHEMA.canonicalize_day_config(config))

def generate_initial_configs(count, profile_name=None):
    if type(count) is not int or count < 0:
        raise ValueError('count must be a nonnegative integer')
    return [build_individual_config(profile_name=profile_name) for _ in range(count)]

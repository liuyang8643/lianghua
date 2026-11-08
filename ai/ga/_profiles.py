"""The single production GA search profile.

``configs/strategy.yaml`` owns run sizes; shared training config owns periods; ActionSchema
owns continuous weights, turnover and fixed controls. GA no
longer supports a registry of historical profiles. A name argument remains at
the CLI boundary so old invocations fail explicitly instead of silently
running another strategy.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path

import yaml
from configs.training import read_evaluation_splits
from env.action_schema import ActionSchema

from factor.registry import (
    PRODUCTION_FACTOR_NAMES,
    PRODUCTION_FILTER_NAMES,
    get_factor_class,
)


DEFAULT_GA_PROFILE = "current4"
_YAML_PATH = Path(__file__).resolve().parents[2] / "configs" / "strategy.yaml"


def _require_current4(name: str | None) -> str:
    resolved = name or DEFAULT_GA_PROFILE
    if resolved != DEFAULT_GA_PROFILE:
        raise ValueError(
            f"GA only supports {DEFAULT_GA_PROFILE!r}; got {resolved!r}"
        )
    return resolved


def _load_profile() -> dict:
    payload = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
    if payload.get("default_profile") != DEFAULT_GA_PROFILE:
        raise ValueError("strategy.yaml default_profile must be current4")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {DEFAULT_GA_PROFILE}:
        raise ValueError("strategy.yaml must contain exactly the current4 profile")
    raw = profiles[DEFAULT_GA_PROFILE]
    factor_names = tuple(raw["factor_classes"])
    filter_names = tuple(raw["filter_factor_classes"])
    if factor_names != PRODUCTION_FACTOR_NAMES:
        raise ValueError("current4 factors must match the production ActionSchema")
    if filter_names != PRODUCTION_FILTER_NAMES:
        raise ValueError("current4 filters must match the production ActionSchema")

    schema = ActionSchema()
    splits, _ = read_evaluation_splits()
    search_spaces = {"turnover_rate": [schema.layout[-1].minimum, schema.layout[-1].maximum]}
    return {
        "name": DEFAULT_GA_PROFILE,
        "desc": str(raw["desc"]),
        "search_space_version": "schema-selected11-continuous-turnover-v6-20pct",
        "factor_classes": tuple(get_factor_class(name) for name in factor_names),
        "filter_factor_classes": tuple(
            get_factor_class(name) for name in filter_names
        ),
        "search_spaces": search_spaces,
        "weight_search_spaces": {
            name: [0.0, 1.0] for name in factor_names
        },
        "preload_start_date": date.fromisoformat(splits["train"][0]),
        "preload_end_date": date.fromisoformat(splits["train"][1]),
        "mode_configs": deepcopy(payload["mode_configs"]),
    }


_PROFILE = _load_profile()


def resolve_profile_name(
    *,
    fallback: str | None = None,
) -> str:
    return _require_current4(fallback)


def get_profile(name: str | None = None) -> dict:
    _require_current4(name)
    return deepcopy(_PROFILE)


def get_mode_configs(name: str | None = None) -> dict:
    _require_current4(name)
    return deepcopy(_PROFILE["mode_configs"])


def get_profile_factor_classes(name: str | None = None) -> list[type]:
    _require_current4(name)
    return list(_PROFILE["factor_classes"])


def get_profile_filter_factor_classes(name: str | None = None) -> list[type]:
    _require_current4(name)
    return list(_PROFILE["filter_factor_classes"])


def get_profile_weight_search_spaces(
    name: str | None = None,
) -> dict[str, list[float]]:
    _require_current4(name)
    return deepcopy(_PROFILE["weight_search_spaces"])


def get_profile_preload_range(name: str | None = None) -> tuple:
    _require_current4(name)
    return _PROFILE["preload_start_date"], _PROFILE["preload_end_date"]

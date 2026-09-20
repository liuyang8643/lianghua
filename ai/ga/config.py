"""Static current4 config loading through the canonical ActionSchema."""

from __future__ import annotations

from typing import Mapping

from ai.ga import resolve_profile_name
from env.action_schema import ActionSchema
from env.contracts import DayConfig


def _unwrap_individual_config(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], str | None]:
    if not isinstance(payload, Mapping):
        raise TypeError("strategy config must be a mapping")
    if "individual_config" not in payload:
        return dict(payload), None
    unexpected = sorted(set(payload) - {"ga_profile", "individual_config"})
    if unexpected:
        raise ValueError(
            "strategy config wrapper contains unexpected fields: "
            + ", ".join(unexpected)
        )
    individual = payload["individual_config"]
    if not isinstance(individual, Mapping):
        raise TypeError("individual_config must be a mapping")
    profile = payload.get("ga_profile")
    if profile is not None and not isinstance(profile, str):
        raise TypeError("ga_profile must be a string")
    return dict(individual), profile


def canonicalize_individual_config(
    payload: Mapping[str, object],
    *,
    profile_name: str | None = None,
    action_schema: ActionSchema | None = None,
) -> tuple[dict[str, object], DayConfig]:
    """Unwrap and parse one static config through the canonical schema."""

    individual, wrapped_profile = _unwrap_individual_config(payload)
    if profile_name is not None and wrapped_profile is not None:
        if profile_name != wrapped_profile:
            raise ValueError("explicit and wrapped ga_profile values disagree")
    resolve_profile_name(fallback=profile_name or wrapped_profile)
    payload = individual
    # prefilter_n is T-1 candidate-fetch metadata, not a DayConfig field.
    payload.pop("prefilter_n", None)
    schema = action_schema or ActionSchema()
    day_config = schema.from_static_config(payload)
    return schema.to_static_config(day_config), day_config


def canonicalize_ga_genes(payload, *, profile_name=None, action_schema=None):
    """Decode exact GA genes without the legacy static benchmark normalization."""
    resolve_profile_name(fallback=profile_name)
    schema = action_schema or ActionSchema()
    config = dict(payload)
    config.pop("prefilter_n", None)
    day = schema.canonicalize_day_config(schema.from_serialized_day_config(config))
    return schema.to_static_config(day), day

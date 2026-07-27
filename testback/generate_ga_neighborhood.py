"""Generate discrete one-parameter neighbors and structural ablations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from core.ga import (
    get_config_param, get_intrinsic_params, get_profile_search_spaces,
    get_profile_weight_search_spaces,
)
from core.strategy_config import load_strategy_config


def _set_config_param(config: dict, definition: dict, value) -> None:
    group = definition.get('config_group')
    if group:
        config.setdefault(group, {})[definition['config_key']] = value
    else:
        config[definition['config_key']] = value
    if definition['key'] == 'buy_n':
        config['sell_m'] = value


def generate_neighborhood(config: dict, profile_name: str) -> list[dict]:
    spaces = get_profile_search_spaces(profile_name)
    weight_spaces = get_profile_weight_search_spaces(profile_name) or {}
    configs = []

    base = copy.deepcopy(config)
    base['neighborhood_change'] = 'base'
    configs.append(base)

    for definition in get_intrinsic_params():
        key = definition['key']
        space = spaces.get(key)
        if not space:
            continue
        current = get_config_param(config, definition)
        for value in space:
            if value == current:
                continue
            candidate = copy.deepcopy(config)
            _set_config_param(candidate, definition, value)
            candidate['neighborhood_change'] = f'{key}:{current}->{value}'
            configs.append(candidate)

    for name, space in weight_spaces.items():
        current = config['weights'][name]
        for value in space:
            if value == current:
                continue
            candidate = copy.deepcopy(config)
            candidate['weights'][name] = value
            candidate['neighborhood_change'] = f'weight_{name}:{current}->{value}'
            configs.append(candidate)

    overlay = config.get('trend_risk_overlay') or {}
    if overlay.get('enabled', False):
        candidate = copy.deepcopy(config)
        candidate['trend_risk_overlay']['enabled'] = False
        candidate['neighborhood_change'] = (
            'ablation_trend_risk_overlay:enabled->false'
        )
        configs.append(candidate)

    unique = {}
    for candidate in configs:
        key_payload = copy.deepcopy(candidate)
        key_payload.pop('neighborhood_change', None)
        key = json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
        unique[key] = candidate
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description='生成 GA 配置的离散单参数邻域')
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    loaded = load_strategy_config(args.config)
    profile_name = loaded['profile_name']
    configs = generate_neighborhood(loaded['individual_config'], profile_name)
    payload = {'profile': profile_name, 'configs': configs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(configs)} configs -> {args.output}')


if __name__ == '__main__':
    main()

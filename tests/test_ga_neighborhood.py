import copy
import json

import pytest

from core.ga import generate_initial_configs
from testback.generate_ga_neighborhood import generate_neighborhood
from testback.run_ga import _config_key, _load_candidate_configs


def test_neighborhood_changes_exactly_one_search_dimension():
    base = generate_initial_configs(1, profile_name='v9_dual_shadow')[0]
    neighbors = generate_neighborhood(base, 'v9_dual_shadow')
    base_key = _config_key(base)

    assert neighbors[0]['neighborhood_change'] == 'base'
    assert len(neighbors) > 20
    for candidate in neighbors[1:]:
        candidate_key = _config_key(candidate)
        changed_fields = sum(left != right for left, right in zip(base_key, candidate_key))
        expected = 2 if candidate['neighborhood_change'].startswith('buy_n:') else 1
        assert changed_fields == expected


def test_neighborhood_does_not_mutate_base_config():
    base = generate_initial_configs(1, profile_name='v9_dual_shadow')[0]
    expected = copy.deepcopy(base)

    generate_neighborhood(base, 'v9_dual_shadow')

    assert base == expected


def test_neighborhood_includes_full_timing_ablation():
    base = generate_initial_configs(1, profile_name='v9_dual_shadow')[0]

    neighbors = generate_neighborhood(base, 'v9_dual_shadow')
    timing_off = [
        candidate for candidate in neighbors
        if candidate['neighborhood_change']
        == 'ablation_trend_risk_overlay:enabled->false'
    ]

    assert len(timing_off) == 1
    assert timing_off[0]['trend_risk_overlay']['enabled'] is False
    assert _config_key(timing_off[0]) != _config_key(base)


def test_config_key_includes_fixed_behavior_fields():
    base = generate_initial_configs(1, profile_name='v9_dual_shadow')[0]
    changed = copy.deepcopy(base)
    changed['limit_up_protection'] = not base['limit_up_protection']

    assert _config_key(changed) != _config_key(base)


def test_candidate_file_round_trip(tmp_path):
    base = generate_initial_configs(1, profile_name='v9_dual_shadow')[0]
    path = tmp_path / 'neighbors.json'
    path.write_text(json.dumps({'profile': 'v9_dual_shadow', 'configs': [base]}), encoding='utf-8')

    assert _load_candidate_configs(path) == [base]


def test_candidate_file_rejects_empty_list(tmp_path):
    path = tmp_path / 'neighbors.json'
    path.write_text('{"configs": []}', encoding='utf-8')

    with pytest.raises(ValueError, match='非空配置列表'):
        _load_candidate_configs(path)

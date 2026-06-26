"""ga_optimizer 选择/多样性策略单测：锦标赛选择 + 真精英保留 + 随机移民。

只验证 GA 算子行为，不触发回测、不读 runtime npz、不参与验证集。
"""
from testback import run_ga
from testback.run_ga import ga_optimizer, _config_key
from core.ga import build_individual_config, get_profile_factor_names

PROFILE = 'core'
_NAMES = get_profile_factor_names(PROFILE)
_POS = [20, 30, 40, 50]


def _make_cfg(i: int) -> dict:
    """构造第 i 个互异配置：首因子权重随 i 单调，保证 config_key 唯一。"""
    w = {n: 0.0 for n in _NAMES}
    w[_NAMES[0]] = round(0.01 * i, 2)
    return build_individual_config(_POS[i % 4], weights=w, profile_name=PROFILE)


def _ga_cache(n: int) -> dict:
    """n 个唯一配置，sharpe = i（i 越大越优）。"""
    return {str(i): {'individual_config': _make_cfg(i), 'sharpe': float(i)} for i in range(n)}


def _empty_state() -> dict:
    return {'population': [], 'hall_of_fame': [], 'fitness_cache': {}}


def test_returns_two_population_size():
    pop = 20
    cache = _ga_cache(2 * pop)
    nxt = ga_optimizer([], state=_empty_state(), population_size=pop,
                       hall_of_fame_size=pop, profile_name=PROFILE, ga_cache=cache)
    assert len(nxt) == 2 * pop


def test_true_elites_preserved():
    """全局历史 top n_elite 必须出现在父代中（精英保留）。"""
    pop = 20
    n = 2 * pop
    cache = _ga_cache(n)
    nxt = ga_optimizer([], state=_empty_state(), population_size=pop,
                       hall_of_fame_size=pop, profile_name=PROFILE, ga_cache=cache)
    n_elite = max(2, round(0.10 * pop))
    # sharpe = i，故 top n_elite 是 i 最大的若干个
    elite_keys = {_config_key(_make_cfg(i)) for i in range(n - n_elite, n)}
    nxt_keys = {_config_key(c) for c in nxt}
    assert elite_keys <= nxt_keys


def test_random_immigrants_injected(monkeypatch):
    """每代注入固定比例随机移民（独立于父代）。"""
    pop = 20
    cache = _ga_cache(2 * pop)
    expected = min(max(1, round(0.15 * pop)), pop)

    def fake_gen(count, profile_name=None):
        assert count == expected
        out = []
        for _ in range(count):
            cfg = _make_cfg(0)
            cfg['_immigrant'] = True
            out.append(cfg)
        return out

    monkeypatch.setattr(run_ga, 'generate_initial_configs', fake_gen)
    nxt = ga_optimizer([], state=_empty_state(), population_size=pop,
                       hall_of_fame_size=pop, profile_name=PROFILE, ga_cache=cache)
    n_immi = sum(1 for c in nxt if c.get('_immigrant'))
    assert n_immi == expected


def test_tournament_only_draws_from_pool():
    """父代全部来自历史池（锦标赛不会凭空造个体）。"""
    pop = 20
    n = 2 * pop
    cache = _ga_cache(n)
    pool_keys = {_config_key(_make_cfg(i)) for i in range(n)}
    nxt = ga_optimizer([], state=_empty_state(), population_size=pop,
                       hall_of_fame_size=pop, profile_name=PROFILE, ga_cache=cache)
    parents = nxt[:pop]
    for p in parents:
        assert _config_key(p) in pool_keys


def test_small_cache_falls_back_without_crash():
    """ga_cache 不足 population_size 时走回退分支：父代=可用个体数，子代补满 pop。"""
    pop = 20
    n_results = 5
    results = [{'individual_config': _make_cfg(i), 'sharpe': float(i)} for i in range(n_results)]
    state = _empty_state()
    nxt = ga_optimizer(results, state=state, population_size=pop,
                       hall_of_fame_size=pop, profile_name=PROFILE, ga_cache=None)
    # 父代上限 = 可用个体数(5)，子代补满 pop 个 → 共 n_results + pop
    assert len(nxt) == n_results + pop

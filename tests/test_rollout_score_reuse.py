"""Independent public-contract and complete-replay checks for score reuse."""
from dataclasses import asdict, replace
import pickle

import numpy as np
import pytest

from env.action_schema import ActionSchema, CORE_FACTOR_NAMES, CORE_FILTER_NAMES
from env.backtest import EpisodeSession, prepare_episode_from_runtime
from env.contracts import DayConfig, Fill
from env.fees import FeeSchedule
from env.planner import DayPlanner
from env.prefilter import PrefilterUniverse, candidate_mask_from_previous_indices, rank_complete_universe_indices, rank_scored_universe_indices
from env.scoring import score_factor_ranks
from test_backtest_lightweight import write_canonical_runtime
from test_rl_day_planner import _cash_account, _config, _market


def simple_config():
    return DayConfig(
        factor_weights={"a": 0.25, "b": 0.75},
        factor_enabled={"a": True, "b": True}, filter_flags={},
        buy_n=1, turnover_rate=1.0, limit_up_protection=False,
        rebalance_band_pct=0.0, single_buy_pct=1.0,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mixed_storage", ["writeability", "layout", "dtype"])
def test_scoring_accepts_mixed_public_array_storage_without_mutating_inputs(dtype, mixed_storage):
    a = np.array([0.125, 0.5, 0.875, np.nan], dtype=dtype)
    b = np.array([0.875, 0.5, np.nan, np.nan], dtype=dtype)
    a_mask = np.array([True, True, True, False])
    b_mask = np.array([True, True, False, False])
    if mixed_storage == "writeability":
        a.flags.writeable = False
        b_mask.flags.writeable = False
    elif mixed_storage == "layout":
        storage = np.empty(8, dtype=dtype)
        storage[::2] = a
        a = storage[::2]
        mask_storage = np.empty(8, dtype=bool)
        mask_storage[::2] = b_mask
        b_mask = mask_storage[::2]
    else:
        b = b.astype(np.float64 if dtype == np.float32 else np.float32)
    arrays = (a, b, a_mask, b_mask)
    before = [value.copy() for value in arrays]
    flags = [value.flags.writeable for value in arrays]
    scores = score_factor_ranks({"a": a, "b": b}, {"a": a_mask, "b": b_mask}, simple_config(), 4)
    np.testing.assert_array_equal(scores.view(np.uint64), np.array([0.6875, 0.5, 0.875, -0.5]).view(np.uint64))
    assert scores.dtype == np.float64
    for actual, expected, writeable in zip(arrays, before, flags):
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.writeable == writeable


@pytest.mark.parametrize("bad_input", ["rank_shape", "mask_shape", "missing_factor", "mask_keys", "valid_nan", "valid_inf"])
def test_scoring_rejects_invalid_enabled_factor_inputs(bad_input):
    ranks = {"a": np.zeros(4), "b": np.ones(4)}
    masks = {"a": np.ones(4, dtype=bool), "b": np.ones(4, dtype=bool)}
    if bad_input == "rank_shape":
        ranks["a"] = np.zeros(3)
    elif bad_input == "mask_shape":
        masks["a"] = np.ones(3, dtype=bool)
    elif bad_input == "missing_factor":
        ranks.pop("a")
        masks.pop("a")
    elif bad_input == "mask_keys":
        masks.pop("b")
    else:
        ranks["b"][0] = np.nan if bad_input == "valid_nan" else np.inf
    with pytest.raises(ValueError):
        score_factor_ranks(ranks, masks, simple_config(), 4)


def test_zero_binary_signal_and_missing_signal_have_different_tail_semantics():
    config = replace(simple_config(), factor_weights={"a": 1.0, "b": 0.0}, factor_enabled={"a": True, "b": False})
    ranks = {"a": np.array([0.0, 1.0, np.nan]), "b": np.zeros(3)}
    masks = {"a": np.array([True, True, False]), "b": np.zeros(3, dtype=bool)}
    np.testing.assert_array_equal(score_factor_ranks(ranks, masks, config, 3), [0.0, 1.0, -1.0])


def test_combined_plan_preserves_full_pit_ranking_and_scores_exactly_once(monkeypatch):
    import env.scoring as scoring_module

    codes = tuple(f"60000{i}.SH" for i in range(7))
    rank = np.linspace(1.0, 0.4, 7, dtype=np.float32)
    ranks = {name: rank.copy() for name in CORE_FACTOR_NAMES}
    filters = {name: np.ones(7, dtype=bool) for name in CORE_FILTER_NAMES}
    filters[CORE_FILTER_NAMES[0]][3] = False
    market = _market(
        codes=codes, ranks=ranks, filters=filters,
        opens=np.array([10.0, 10.0, np.nan, 10.0, 11.0, 10.0, 10.0]),
        listing_age=np.array([-1, 10, 10, 10, 10, 10, 10], dtype=np.int32),
        delisted=np.array([False, True, False, False, False, False, False]),
    ).seal().with_candidate_mask(np.array([False, False, False, False, False, True, False]))
    config = _config(filter_flags={name: index == 0 for index, name in enumerate(CORE_FILTER_NAMES)})
    universe = PrefilterUniverse(codes)
    expected_plan = DayPlanner().plan(market, _cash_account(), config)
    expected_rank = rank_complete_universe_indices(
        universe, market.factor_ranks, market.factor_validity, config,
        pit_universe_mask=(market.listing_age >= 0) & ~market.delisted_mask,
    )
    real_score = scoring_module._score_rows
    calls = []

    def counted_score(*args):
        calls.append(1)
        return real_score(*args)

    monkeypatch.setattr(scoring_module, "_score_rows", counted_score)
    actual_plan, actual_rank = DayPlanner().plan_and_rank(market, _cash_account(), config, universe)
    assert actual_plan == expected_plan
    np.testing.assert_array_equal(actual_rank, expected_rank)
    np.testing.assert_array_equal(actual_rank, [2, 3, 4, 5, 6, 0, 1])
    assert set(actual_plan.buy_orders) == {codes[5]}
    assert calls == [1]
    assert not actual_rank.flags.writeable
    assert all(row.dtype == np.float32 for row in market.factor_ranks.values())


def test_combined_plan_rejects_reordered_stock_axis():
    market = _market()
    universe = PrefilterUniverse(market.stock_codes[::-1])
    with pytest.raises(ValueError, match="universe differs"):
        DayPlanner().plan_and_rank(market, _cash_account(), _config(), universe)


@pytest.mark.parametrize("bad_scores,bad_mask", [
    ([1.0], [True, True]), ([0.0, np.inf], [True, True]),
    ([0.0, np.nan], [True, True]), ([1.0, 0.0], [True]),
])
def test_precomputed_ranking_rejects_nonfinite_or_wrong_axis(bad_scores, bad_mask):
    universe = PrefilterUniverse(("600001.SH", "600002.SH"))
    with pytest.raises(ValueError):
        rank_scored_universe_indices(universe, np.asarray(bad_scores), pit_universe_mask=np.asarray(bad_mask))


def test_position_only_sort_preserves_preference_axis_and_unknown_tail():
    positions = {"outside-z": 1, "300001.SZ": 2, "600099.SH": 3, "600001.SH": 4, "outside-a": 5}
    codes = ("600099.SH", "600001.SH", "300001.SZ")
    result = DayPlanner()._ordered_position_codes(positions, ("600001.SH", "600001.SH", "outside-z"), codes)
    assert result == ["600001.SH", "outside-z", "600099.SH", "300001.SZ", "outside-a"]


@pytest.mark.parametrize("fees", [FeeSchedule(), FeeSchedule(commission_rate=0.0003, minimum_commission=5.0, slippage_rate=0.01)])
def test_analytic_affordability_is_maximal_at_exact_float_cost_boundaries(fees):
    for price in (0.01, 1.5, 123.4):
        for target in (0, 1, 99, 100, 101, 200, 201, 10000):
            boundary = fees.buy_total_cost(target * price)
            for cash in (np.nextafter(boundary, 0.0), boundary, np.nextafter(boundary, np.inf)):
                quantity = fees.affordable_buy_shares(cash, price)
                assert quantity >= 0
                if quantity:
                    assert fees.buy_total_cost(quantity * price) <= cash
                assert fees.buy_total_cost((quantity + 1) * price) > cash


class TwoScorePlanner(DayPlanner):
    """The pre-reuse orchestration, still using the single domain scoring law."""

    def plan_and_rank(self, market, account, config, universe, *, prefilter_n=None):
        plan = self.plan(market, account, config)
        order = rank_complete_universe_indices(
            universe, market.factor_ranks, market.factor_validity, config,
            pit_universe_mask=(market.listing_age >= 0) & ~market.delisted_mask,
        )
        return plan, order if prefilter_n is None else order[:prefilter_n]


@pytest.fixture
def episode(tmp_path):
    path = tmp_path / "score-reuse-runtime.npz"
    write_canonical_runtime(path, stocks=30)
    return prepare_episode_from_runtime(path, "2020-06-08", "2020-06-27", lookback=12, prefilter_n=21)


def test_complete_dynamic_account_and_next_day_prefilter_match_two_score_path(episode):
    schema = ActionSchema()
    base = schema.decode(np.zeros(schema.action_dim))
    combined = EpisodeSession(episode, action_schema=schema)
    separate = EpisodeSession(episode, action_schema=schema)
    separate._planner = TwoScorePlanner()
    combined.reset()
    separate.reset()
    steps = 0
    while not combined.terminated:
        weights = {name: ((steps + index) % 9 + 1) / 9 for index, name in enumerate(CORE_FACTOR_NAMES)}
        turnover_rate = (0.05, 0.2)[steps % 2]
        config = replace(base, factor_weights=weights, factor_enabled={name: True for name in weights}, turnover_rate=turnover_rate)
        actual = combined.step(config)
        expected = separate.step(config)
        assert actual.order_plan == expected.order_plan
        assert actual.step_result == expected.step_result
        assert actual.info == expected.info
        np.testing.assert_array_equal(actual.observation, expected.observation)
        np.testing.assert_array_equal(combined._previous_prefilter_indices, separate._previous_prefilter_indices)
        assert len(combined._previous_prefilter_indices) == episode.prefilter_n
        assert len(combined.episode.runtime.stock_codes) == 30
        assert combined.episode.factors.ranks.shape[-1] == 30
        assert combined.current_policy_memory == separate.current_policy_memory
        assert actual.info["full_investment_contract_satisfied"]
        steps += 1
    assert steps == 19
    assert separate.terminated


def test_failed_settlement_does_not_commit_next_day_prefilter(episode, monkeypatch):
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim))
    session = EpisodeSession(episode, action_schema=schema)
    session.reset()
    session.step(config)
    old_index = session.current_index
    old_account = session.current_account
    previous_rank = session._previous_prefilter_indices.copy()

    def fail_settlement(*args, **kwargs):
        raise ValueError("invalid settlement fixture")

    monkeypatch.setattr(session._simulator, "step", fail_settlement)
    different_weights = {name: (index + 1) / len(CORE_FACTOR_NAMES) for index, name in enumerate(CORE_FACTOR_NAMES)}
    changed = replace(config, factor_weights=different_weights, factor_enabled={name: True for name in different_weights})
    market = episode.market_at(session.current_index)
    changed_rank = rank_complete_universe_indices(
        session._prefilter_universe, market.factor_ranks, market.factor_validity, changed,
        pit_universe_mask=(market.listing_age >= 0) & ~market.delisted_mask,
    )
    assert not np.array_equal(changed_rank[:episode.prefilter_n], previous_rank)
    with pytest.raises(ValueError, match="invalid settlement fixture"):
        session.step(changed)
    assert session.current_index == old_index
    assert session.current_account == old_account
    np.testing.assert_array_equal(session._previous_prefilter_indices, previous_rank)


@pytest.mark.parametrize("size", [1, 7, 64, 137])
@pytest.mark.parametrize("membership", ["none", "all", "sparse", "mixed"])
def test_topk_is_exact_full_stable_ranking_prefix_for_ties_and_membership(size, membership):
    rng = np.random.default_rng(840 + size)
    codes = tuple(f"{index:06d}.SH" for index in range(size))
    universe = PrefilterUniverse(codes)
    member = {
        "none": np.zeros(size, dtype=bool),
        "all": np.ones(size, dtype=bool),
        "sparse": np.arange(size) % 7 == 0,
        "mixed": rng.random(size) < 0.65,
    }[membership]
    score_sets = (
        np.zeros(size), np.linspace(-1.0, 1.0, size),
        rng.choice([-1.0, -0.0, 0.0, 0.5, 1.0], size=size),
        rng.normal(size=size),
    )
    for scores in score_sets:
        full_order = np.argsort(-scores, kind="stable")
        expected = np.concatenate((full_order[member[full_order]], np.flatnonzero(~member)))
        for count in sorted({1, max(1, size // 3), max(1, int(member.sum())), size, size + 5}):
            actual = rank_scored_universe_indices(universe, scores, pit_universe_mask=member, limit=count)
            np.testing.assert_array_equal(actual, expected[:count])
            assert not actual.flags.writeable
            assert actual.dtype == np.intp
            np.testing.assert_array_equal(
                candidate_mask_from_previous_indices(universe, actual, count, ranking_is_prefix=True),
                candidate_mask_from_previous_indices(universe, expected, count),
            )


def test_topk_threshold_ties_use_original_axis_order():
    universe = PrefilterUniverse(tuple(str(index) for index in range(8)))
    scores = np.array([0.7, 0.9, 0.7, 0.9, 0.7, 0.7, 0.9, 0.7])
    member = np.ones(8, dtype=bool)
    np.testing.assert_array_equal(
        rank_scored_universe_indices(universe, scores, pit_universe_mask=member, limit=5),
        [1, 3, 6, 0, 2],
    )


def test_prefix_cold_start_and_held_names_preserve_full_stock_axis():
    universe = PrefilterUniverse(("A", "B", "C", "D", "E"))
    cold = candidate_mask_from_previous_indices(universe, None, 2, ranking_is_prefix=True)
    np.testing.assert_array_equal(cold, [True] * 5)
    prefix = np.array([3, 1])
    actual = candidate_mask_from_previous_indices(universe, prefix, 2, held_codes=("A", "E", "E"), ranking_is_prefix=True)
    np.testing.assert_array_equal(actual, [True, True, False, True, True])
    np.testing.assert_array_equal(prefix, [3, 1])
    with pytest.raises(ValueError, match="outside"):
        candidate_mask_from_previous_indices(universe, prefix, 2, held_codes=("Z",), ranking_is_prefix=True)


@pytest.mark.parametrize("invalid", [
    np.array([1]), np.array([1, 1]), np.array([-1, 2]),
    np.array([1, 5]), np.array([1.0, 2.0]), np.array([[1, 2]]),
])
def test_explicit_prefix_rejects_short_duplicate_noninteger_and_invalid_indices(invalid):
    universe = PrefilterUniverse(("A", "B", "C", "D", "E"))
    with pytest.raises(ValueError):
        candidate_mask_from_previous_indices(universe, invalid, 2, ranking_is_prefix=True)


def test_full_permutation_default_does_not_silently_accept_a_prefix():
    universe = PrefilterUniverse(("A", "B", "C", "D", "E"))
    with pytest.raises(ValueError):
        candidate_mask_from_previous_indices(universe, np.array([3, 1]), 2)
    with pytest.raises(TypeError, match="must be bool"):
        candidate_mask_from_previous_indices(universe, np.array([3, 1]), 2, ranking_is_prefix=1)


@pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.0])
def test_topk_rejects_invalid_limits(invalid_limit):
    universe = PrefilterUniverse(("A", "B"))
    with pytest.raises(ValueError, match="ranking limit"):
        rank_scored_universe_indices(universe, np.array([1.0, 0.0]), pit_universe_mask=np.ones(2, dtype=bool), limit=invalid_limit)


def test_single_scan_preserves_sequential_filter_and_missing_counters():
    codes = tuple(f"60000{i}.SH" for i in range(8))
    ranks = {name: np.linspace(1.0, 0.1, 8) for name in CORE_FACTOR_NAMES}
    validity = {name: np.ones(8, dtype=bool) for name in CORE_FACTOR_NAMES}
    validity[CORE_FACTOR_NAMES[0]][[4, 7]] = False
    validity[CORE_FACTOR_NAMES[1]][[2, 6]] = False
    filters = {
        CORE_FILTER_NAMES[0]: np.array([True, True, True, True, False, True, True, False]),
        CORE_FILTER_NAMES[1]: np.array([True, True, True, True, False, False, True, True]),
    }
    market = _market(
        codes=codes, ranks=ranks, validity=validity, filters=filters,
        opens=np.array([10.0, 10.0, np.nan, 10.0, 10.0, 10.0, 10.0, 10.0]),
        listing_age=np.array([-1, 10, 10, 10, 10, 10, 10, 10], dtype=np.int32),
        delisted=np.array([False, True, False, False, False, False, False, False]),
    ).with_candidate_mask(np.array([True, True, True, False, True, True, True, True]))
    weights = {name: (0.5 if index < 2 else 0.0) for index, name in enumerate(CORE_FACTOR_NAMES)}
    config = _config(factor_weights=weights, filter_flags={name: True for name in CORE_FILTER_NAMES}, buy_n=2, turnover_rate=1.0)
    plan = DayPlanner(diagnostics="full").plan(market, _cash_account(), config)
    assert plan.diagnostics["market_rejection_counts"] == {
        "not_listed": 1, "delisted": 1, "suspended_or_missing_open": 1, "outside_t1_prefilter": 1,
    }
    assert plan.diagnostics["factor_missing"] == {CORE_FACTOR_NAMES[0]: 2, CORE_FACTOR_NAMES[1]: 1}
    assert plan.diagnostics["filter_rejected"] == {CORE_FILTER_NAMES[0]: 2, CORE_FILTER_NAMES[1]: 1}
    assert plan.diagnostics["eligible_count"] == 1
    assert plan.diagnostics["buy_n_stocks"][0] == codes[6]


def test_vectorized_legal_prefix_counts_only_rejections_before_original_stop():
    codes = tuple(f"60000{i}.SH" for i in range(8))
    market = _market(codes=codes, opens=np.array([11.0, 10.0, 9.0, 11.0, 10.0, 11.0, 10.0, 10.0]))
    plan = DayPlanner(diagnostics="full").plan(market, _cash_account(), _config(buy_n=2, turnover_rate=1.0))
    assert plan.diagnostics["buy_n_stocks"] == (codes[1], codes[2])
    assert plan.diagnostics["full_rank_top_stocks"] == (codes[0], codes[1])
    assert plan.diagnostics["buy_legality_rejection_counts"] == {"limit_up": 1}
    assert plan.diagnostics["buy_legality_rejections"] == {codes[0]: "limit_up"}


def test_fill_slots_preserve_serialization_and_worker_transport():
    fill = Fill("600001.SH", "buy", 100, 10.0, 1.1, "2026-09-14T09:30:00")
    assert pickle.loads(pickle.dumps(fill)) == fill
    assert asdict(fill) == {
        "code": "600001.SH", "side": "buy", "quantity": 100,
        "price": 10.0, "fee": 1.1, "timestamp": "2026-09-14T09:30:00",
    }


def test_sealed_score_layout_reuses_rows_but_never_policy_scores():
    market = _market().seal()
    row = market._factor_score_row
    first = _config()
    weights = {name: (index + 1) / len(CORE_FACTOR_NAMES) for index, name in enumerate(CORE_FACTOR_NAMES)}
    second = _config(factor_weights=weights)
    for config in (first, second, first):
        actual = market.factor_scores(config)
        expected = score_factor_ranks(market.factor_ranks, market.factor_validity, config, len(market.stock_codes))
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        actual[:] = 999.0
        assert market._factor_score_row is row
    assert not np.any(market.factor_scores(first) == 999.0)
    clone = market.with_candidate_mask(np.zeros(len(market.stock_codes), dtype=bool))
    assert clone._factor_score_row is row
    np.testing.assert_array_equal(clone.factor_scores(second), market.factor_scores(second))
    for name in market.factor_ranks:
        assert np.shares_memory(row.factor_ranks[name], market.factor_ranks[name])
        assert np.shares_memory(row.factor_validity[name], market.factor_validity[name])
        assert not row.factor_ranks[name].flags.writeable
        assert not row.factor_validity[name].flags.writeable


def test_live_score_inputs_update_while_sealed_scores_keep_snapshot():
    market = _market()
    frozen = market.seal()
    config = _config()
    original = frozen.factor_scores(config)
    name = CORE_FACTOR_NAMES[0]
    market.factor_ranks[name][0] = 0.125
    market.factor_validity[name][1] = False
    live = market.factor_scores(config)
    expected = score_factor_ranks(market.factor_ranks, market.factor_validity, config, len(market.stock_codes))
    np.testing.assert_array_equal(live.view(np.uint64), expected.view(np.uint64))
    assert not np.array_equal(live, original)
    np.testing.assert_array_equal(frozen.factor_scores(config).view(np.uint64), original.view(np.uint64))
    market.factor_ranks[name][0] = np.nan
    with pytest.raises(ValueError, match="valid factor ranks must be finite"):
        market.factor_scores(config)


def test_cached_score_layout_keeps_configuration_factor_order():
    from env.scoring import FactorScoreRow

    config = simple_config()
    ranks = {"b": np.array([0.125, 1.0, np.nan], dtype=np.float32), "a": np.array([1.0, 0.125, 0.5])}
    masks = {"a": np.array([True, True, True]), "b": np.array([True, True, False])}
    row = FactorScoreRow(ranks, masks, 3)
    expected = np.array([0.34375, 0.78125, 0.5])
    np.testing.assert_array_equal(row.score(config).view(np.uint64), expected.view(np.uint64))
    with pytest.raises(TypeError):
        row.factor_ranks["a"] = np.zeros(3)


@pytest.mark.parametrize("value", [np.finfo(float).max, -np.finfo(float).max])
def test_overflow_from_valid_finite_ranks_does_not_fabricate_finite_missing_tail(value):
    config = replace(simple_config(), factor_weights={"a": 1.0, "b": 1.0})
    ranks = {"a": np.array([value, np.nan]), "b": np.array([value, np.nan])}
    masks = {name: np.array([True, False]) for name in ranks}
    with np.errstate(over="ignore"):
        expected = value + value
    actual = score_factor_ranks(ranks, masks, config, 2)
    np.testing.assert_array_equal(actual, [expected, expected])
    assert np.isinf(actual).all()


def test_overflow_preserves_factor_order_instead_of_reassociation():
    largest = np.finfo(float).max
    ranks = {"a": np.array([largest, np.nan]), "b": np.array([largest, np.nan]),
        "c": np.array([-largest, np.nan]), "d": np.array([-largest, np.nan])}
    masks = {name: np.array([True, False]) for name in ranks}
    config = DayConfig(factor_weights={name: 1.0 for name in ranks}, factor_enabled={name: True for name in ranks},
        filter_flags={}, buy_n=1, turnover_rate=1.0, single_buy_pct=1.0, limit_up_protection=False, rebalance_band_pct=0.0)
    actual = score_factor_ranks(ranks, masks, config, 2)
    # Preserve factor-order additions: max+max overflows before subtraction.
    assert np.isposinf(actual).all()


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int32, np.uint64, np.dtype(">i8")])
def test_checked_prefilter_mask_keeps_strided_readonly_integer_inputs(dtype):
    from env.prefilter import PrefilterUniverse, candidate_mask_from_previous_indices
    universe = PrefilterUniverse(tuple("abcde"))
    storage = np.zeros(10, dtype=dtype)
    storage[::2] = [4, 2, 0, 3, 1]
    ranking = storage[::2]
    ranking.flags.writeable = False
    for prefix in (False, True):
        result = candidate_mask_from_previous_indices(
            universe, ranking[:3] if prefix else ranking, 3, held_codes=("b",), ranking_is_prefix=prefix)
        np.testing.assert_array_equal(result, [True, True, True, False, True])


@pytest.mark.parametrize("bad", [np.uint64(2**63), np.uint64(2**64 - 1), -1, 5])
def test_checked_prefilter_mask_rejects_extreme_indices_before_indexing(bad):
    from env.prefilter import PrefilterUniverse, candidate_mask_from_previous_indices
    dtype = np.uint64 if isinstance(bad, np.uint64) else np.int64
    ranking = np.array([0, 1, bad], dtype=dtype)
    with pytest.raises(ValueError, match="outside"):
        candidate_mask_from_previous_indices(PrefilterUniverse(tuple("abcde")), ranking, 3, ranking_is_prefix=True)


@pytest.mark.parametrize("limit", [2**63, 2**80])
def test_checked_prefilter_accepts_huge_positive_limit_as_complete_axis(limit):
    universe = PrefilterUniverse(tuple("abcde"))
    for prefix in (False, True):
        mask = candidate_mask_from_previous_indices(universe, np.array([4, 2, 0, 3, 1]), limit, ranking_is_prefix=prefix)
        np.testing.assert_array_equal(mask, np.ones(5, dtype=bool))


def test_schema_and_successful_config_certificate_have_no_mutable_alias_or_replace_bypass():
    from env.action_schema import ActionSchema
    schema = ActionSchema()
    factors, filters = list(schema.factor_names), list(schema.fixed_filter_flags)
    owned = replace(schema, factor_names=factors, fixed_filter_flags=filters)
    original_hash = owned.schema_hash
    factors.reverse()
    filters[0] = not filters[0]
    assert owned.factor_names == schema.factor_names
    assert owned.fixed_filter_flags == schema.fixed_filter_flags and owned.schema_hash == original_hash
    assert owned.schema_hash == ActionSchema.from_dict(owned.to_dict()).schema_hash
    config = owned.decode(np.zeros(owned.action_dim))
    before = owned.to_static_config(config)
    owned.validate_day_config(config)
    assert owned.to_static_config(config) == before
    with pytest.raises(ValueError, match="buy_n"):
        owned.validate_day_config(replace(config, buy_n=21, single_buy_pct=1 / 21))
    changed = replace(owned, fixed_rebalance_band_pct=0.02)
    with pytest.raises(ValueError, match="rebalance"):
        changed.validate_day_config(config)
    changed_config = replace(config, rebalance_band_pct=0.02)
    changed.validate_day_config(changed_config)
    with pytest.raises(ValueError, match="rebalance"):
        owned.validate_day_config(changed_config)

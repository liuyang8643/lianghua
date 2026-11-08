from __future__ import annotations

import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.prefilter import (
    PrefilterUniverse,
    candidate_mask_from_previous_indices,
    candidate_mask_from_previous_ranking,
    prefilter_n_from_config,
    rank_complete_universe,
    rank_scored_universe_indices,
)


def _config():
    schema = ActionSchema(
        factor_names=("Size", "Trend"),
        filter_names=("FilterST",),
        fixed_filter_flags=(True,),
        fixed_buy_n=2,
    )
    return schema.from_static_config(
        {
            "weights": {"Size": 1.0, "Trend": 1.0},
            "filter_factors": {"FilterST": True},
            "buy_n": 2,
            "turnover_rate": 1.0,
            "single_buy_pct": 0.5,
            "limit_up_protection": True,
        }
    )


def test_complete_ranking_keeps_negative_values_and_stable_invalid_tail():
    codes = ("A", "B", "C", "D")
    ranking = rank_complete_universe(
        codes,
        {
            "Size": np.array([-0.5, -0.1, -0.3, 99.0]),
            "Trend": np.array([0.0, 0.0, 0.0, 99.0]),
        },
        {
            "Size": np.array([True, True, True, False]),
            "Trend": np.array([True, True, True, False]),
        },
        _config(),
        pit_universe_mask=np.ones(len(codes), dtype=bool),
    )

    assert ranking == ("B", "C", "A", "D")


@pytest.mark.parametrize("available", [True, False])
def test_nonmembers_never_precede_missing_members(available):
    schema = ActionSchema(factor_names=("Size",))
    config = schema.decode(np.array([0.0, -1.0]))
    ranking = rank_complete_universe(
        ("active_A", "active_B", "delisted_C"),
        {"Size": np.array([1.0, 0.5, 0.0])},
        {"Size": np.array([available, available, True])},
        config,
        pit_universe_mask=np.array([True, True, False]),
    )
    assert ranking[-1] == "delisted_C"
    assert ranking[0].startswith("active_")


def test_candidate_mask_is_t1_top_n_plus_holdings_in_full_axis_order():
    codes = ("A", "B", "C", "D")
    mask = candidate_mask_from_previous_ranking(
        codes,
        ("D", "B", "C", "A"),
        2,
        held_codes=("A",),
    )

    np.testing.assert_array_equal(mask, [True, True, False, True])
    np.testing.assert_array_equal(
        candidate_mask_from_previous_ranking(codes, None, 2),
        np.ones(4, dtype=bool),
    )


def test_prefilter_is_required_fixed_metadata_not_a_policy_default():
    assert prefilter_n_from_config(
        {"individual_config": {"prefilter_n": 300}}
    ) == 300
    with pytest.raises(ValueError, match="positive int"):
        prefilter_n_from_config({"individual_config": {}})


@pytest.mark.parametrize("bad", [
    np.array([0, 0, 2]), np.array([-1, 1, 2]), np.array([0, 1, 3]),
    np.array([0, 1]), np.array([[0, 1, 2]]), np.array([0., 1., 2.]),
    np.array([False, True, True]), np.array([0, 1, 2], dtype=object),
])
def test_integer_prefilter_rejects_corrupt_permutations(bad):
    with pytest.raises(ValueError):
        candidate_mask_from_previous_indices(PrefilterUniverse(("A", "B", "C")), bad, 2)


def test_integer_prefilter_preserves_axis_and_held_names():
    codes = ["A", "B", "C", "D"]
    universe = PrefilterUniverse(codes)
    codes[0] = "changed"
    assert universe.stock_codes == ("A", "B", "C", "D")
    with pytest.raises(TypeError):
        universe.code_to_index["outside"] = 4
    np.testing.assert_array_equal(
        candidate_mask_from_previous_indices(universe, np.array([3, 1, 2, 0]), 2, held_codes=("A",)),
        [True, True, False, True],
    )
    with pytest.raises(ValueError, match="held codes"):
        candidate_mask_from_previous_indices(universe, np.arange(4), 2, held_codes=("outside",))


@pytest.mark.parametrize("dtype", [np.dtype(">f8"), np.dtype("f8"), np.dtype("f4")])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("membership", ["all", "none", "alternating"])
@pytest.mark.parametrize("limit", [None, 1, 17, 60])
def test_score_ranking_preserves_stability_endianness_and_pit(dtype, strided, membership, limit):
    values = np.resize(np.array([0., -0., 3., -3., 3., np.finfo(np.float32).tiny]), 100).astype(dtype)
    scores = values[::2] if strided else values[:50]
    before = scores.copy()
    member = np.ones(50, dtype=bool)
    if membership == "none":
        member[:] = False
    elif membership == "alternating":
        member[::2] = False
    indices = np.flatnonzero(member)
    expected = np.concatenate((
        indices[np.argsort(-scores[indices].astype(np.float64), kind="stable")],
        np.flatnonzero(~member),
    ))
    if limit is not None:
        expected = expected[:limit]
    actual = rank_scored_universe_indices(
        PrefilterUniverse(tuple(map(str, range(50)))), scores,
        pit_universe_mask=member, limit=limit,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(scores, before)
    assert not actual.flags.writeable


def test_score_ranking_preserves_finite_float64_extremes():
    values = np.array([
        np.finfo(float).max, -np.finfo(float).max, np.nextafter(0., 1.),
        -np.nextafter(0., 1.), 0., -0., np.finfo(float).tiny,
        -np.finfo(float).tiny, np.finfo(float).max,
    ])
    result = rank_scored_universe_indices(
        PrefilterUniverse(tuple(map(str, range(len(values))))), values,
        pit_universe_mask=np.ones(len(values), dtype=bool),
    )
    np.testing.assert_array_equal(result, np.argsort(-values, kind="stable"))

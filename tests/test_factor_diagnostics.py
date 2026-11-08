import numpy as np
import pytest

from env.factor_diagnostics import daily_rank_ic, rank_similarity, summarize_ic


def test_daily_rank_ic_monotonic_constant_and_count():
    returns = np.arange(40.0)
    scores = np.vstack([returns, -returns, np.ones(40), returns])
    scores[3, :11] = np.nan
    ic, counts = daily_rank_ic(scores, returns, np.ones(40, dtype=bool))
    np.testing.assert_allclose(ic[:2], [1, -1])
    assert np.isnan(ic[2:]).all()
    np.testing.assert_array_equal(counts, [40, 40, 40, 29])
    ic, _ = daily_rank_ic(scores, np.ones(40), np.ones(40, dtype=bool))
    assert np.isnan(ic).all()


def test_binary_ties_match_manual_average_ranks_and_permutation():
    event = np.repeat([0.0, 1.0], 20)
    returns = np.arange(40.0)
    expected = np.corrcoef(np.repeat([10.5, 30.5], 20), returns + 1)[0, 1]
    actual, _ = daily_rank_ic(event[None, :], returns, np.ones(40, bool))
    assert actual[0] == pytest.approx(expected)
    assert actual[0] < 0.9  # Stable ordinal ranks would fabricate perfect IC.
    permutation = np.random.default_rng(7).permutation(40)
    reordered, _ = daily_rank_ic(event[None, permutation], returns[permutation], np.ones(40, bool))
    np.testing.assert_allclose(actual, reordered)


def test_daily_ic_reranks_common_finite_universe():
    scores = np.tile(np.arange(50.0), (1, 1))
    returns = np.arange(50.0) ** 2
    scores[0, [1, 3, 7, 15, 22]] = np.nan
    returns[[6, 14, 26]] = np.nan
    eligible = np.ones(50, bool)
    eligible[40:] = False
    ic, counts = daily_rank_ic(scores, returns, eligible)
    np.testing.assert_allclose(ic, [1])
    np.testing.assert_array_equal(counts, [32])


def test_similarity_uses_own_ranks_on_common_samples():
    a = np.arange(60.0)
    b = np.roll(a, 9)
    a[::5] = np.nan
    b[::7] = np.nan
    scores = np.vstack([a, b, np.ones(60)])
    result, counts = rank_similarity(scores, np.ones(60, bool))
    # Manual ordinal ranks suffice here: both finite rows are unique.
    ranks = []
    for row in (a, b):
        valid = np.isfinite(row)
        ranked = np.full(60, np.nan)
        ranked[valid] = np.argsort(np.argsort(row[valid])) + 1.0
        ranks.append(ranked)
    common = np.isfinite(a) & np.isfinite(b)
    expected = np.corrcoef(ranks[0][common], ranks[1][common])[0, 1]
    assert result[0, 1] == pytest.approx(expected)
    assert counts[0, 1] == common.sum()
    assert np.isnan(result[2]).all()
    np.testing.assert_allclose(result, result.T)


def test_similarity_ties_eligibility_and_insufficient_overlap():
    binary = np.repeat([0.0, 1.0], 30)
    scores = np.vstack([binary, 1 - binary])
    eligible = np.ones(60, bool)
    eligible[:10] = False
    result, counts = rank_similarity(scores, eligible)
    np.testing.assert_allclose(result, [[1, -1], [-1, 1]])
    np.testing.assert_array_equal(counts, np.full((2, 2), 50))
    eligible[29:] = False
    result, _ = rank_similarity(scores, eligible)
    assert np.isnan(result).all()


def test_hac_matches_explicit_bartlett_covariance_with_missing_days():
    values = np.array([0.4, 0.1, np.nan, -0.2, 0.3, -0.1])
    dates = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-07"))
    row = summarize_ic(dates, values[:, None], 30)[0]
    valid = np.isfinite(values)
    mean = values[valid].mean()
    centered = np.where(valid, values - mean, 0)
    kernel = np.fromfunction(lambda i, j: np.maximum(1 - np.abs(i-j) / 30, 0), (6, 6))
    expected_se = np.sqrt(centered @ kernel @ centered) / valid.sum()
    assert row["mean_ic"] == pytest.approx(0.1)
    assert row["positive_fraction"] == pytest.approx(3 / 5)
    assert row["nw_standard_error"] == pytest.approx(expected_se)
    assert row["nw_t"] == pytest.approx(mean / expected_se)
    assert row["nw_lag"] == 29
    assert not row["phase_eligible"]


def test_fixed_calendar_phases_minimum_days_and_constant_hac():
    dates = np.arange(np.datetime64("2000-01-01"), np.datetime64("2001-02-01"))
    values = np.full((dates.size, 2), 0.125)
    values[:, 1] = np.nan
    rows = summarize_ic(dates, values, 1)
    lookup = {(row["period"], row["factor_index"]): row for row in rows}
    assert lookup[("2000-H1", 0)]["n"] == 182
    assert lookup[("2000-H2", 0)]["n"] == 184
    assert lookup[("2001-H1", 0)]["n"] == 31
    assert lookup[("2000", 0)]["phase_eligible"]
    assert not lookup[("2001", 0)]["phase_eligible"]
    assert lookup[("all", 0)]["nw_standard_error"] == 0
    assert np.isnan(lookup[("all", 0)]["nw_t"])
    assert lookup[("all", 1)]["n"] == 0
    assert np.isnan(lookup[("all", 1)]["mean_ic"])


def test_input_contracts():
    with pytest.raises(ValueError, match="scores"):
        daily_rank_ic(np.ones(30), np.ones(30), np.ones(30, bool))
    with pytest.raises(ValueError, match="forward_returns"):
        daily_rank_ic(np.ones((2, 30)), np.ones(29), np.ones(30, bool))
    with pytest.raises(ValueError, match="strictly increasing"):
        summarize_ic(["2000-01-02", "2000-01-01"], np.ones((2, 1)), 1)
    with pytest.raises(ValueError, match="horizon"):
        summarize_ic(["2000-01-01"], np.ones((1, 1)), 0)

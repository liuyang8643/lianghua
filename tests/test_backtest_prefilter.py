import numpy as np

from core.backtest import _build_t1_ranking
from core.scoring import FactorScoreMatrices, candidate_local_score_matrices


def test_t1_ranking_uses_the_complete_universe_before_t_day_filters():
    stocks = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    stock_indices = {stock: index for index, stock in enumerate(stocks)}
    scores = {"Trend": np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)}

    ranking = _build_t1_ranking(
        scores, 0, stocks, stock_indices, {"Trend": 1.0}
    )

    assert ranking == stocks


def test_t1_ranking_keeps_negative_near_candidates_for_prefilter_discovery():
    stocks = ["000001.SZ", "000002.SZ"]
    scores = {"Trend": np.array([[-0.5, -0.1]], dtype=np.float32)}

    ranking = _build_t1_ranking(
        scores,
        0,
        stocks,
        {stock: index for index, stock in enumerate(stocks)},
        {"Trend": 1.0},
    )

    assert ranking == ["000002.SZ", "000001.SZ"]


def test_regular_factors_are_reranked_inside_the_t_day_candidate_pool():
    raw = {
        "Size": np.array([[40.0, 10.0, 30.0, 20.0]], dtype=np.float32),
        "Trend": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
    }
    ranked = FactorScoreMatrices(
        {
            "Size": np.array([[1.0, 0.25, 0.75, 0.5]], dtype=np.float32),
            "Trend": raw["Trend"].copy(),
        },
        raw_scores=raw,
        pre_ranked_names={"Trend"},
    )

    local, score_idx = candidate_local_score_matrices(
        ranked, 0, np.array([1, 3], dtype=np.intp)
    )

    assert score_idx == 0
    np.testing.assert_array_equal(
        local["Size"], np.array([[0.0, 0.5, 0.0, 1.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        local["Trend"], np.array([[0.0, 0.2, 0.0, 0.4]], dtype=np.float32)
    )

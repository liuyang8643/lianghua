from __future__ import annotations

import ast
import inspect
import time
from pathlib import Path

import numpy as np
import pytest

from factor_db.factors.IndustryPeerReversalStrict import (
    IndustryPeerReversal20Strict,
    IndustryPeerReversal60Strict,
    IndustryPeerReversal120Strict,
    _leave_one_out_peer_returns,
)


ESTIMATION = 20
RECENT = 5
TOTAL = ESTIMATION + RECENT
MIN_PEERS = 3
MIN_CORRELATION = 0.20


def _panel(rows: int = 53) -> dict:
    stocks = 9
    row = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    group = np.asarray([0] * 5 + [1] * 4, dtype=np.float64)[None, :]
    common = (
        0.007 * np.sin(row / 3.7 + group * 0.4)
        + 0.002 * np.cos(row / 8.3 - group * 0.2)
    )
    loading = 0.82 + 0.055 * stock
    idiosyncratic = (
        0.00075 * np.sin(row * (0.31 + stock / 230.0) + stock)
        + 0.00035 * np.cos(row / 2.9 - stock * 0.7)
    )
    log_return = common * loading + idiosyncratic
    if rows > TOTAL:
        log_return[ESTIMATION:TOTAL, 0] -= 0.006

    pre_close = 8.0 + 0.013 * row + 0.27 * stock
    close = pre_close * np.exp(log_return)
    industry_id = np.tile(
        np.asarray([101] * 5 + [202] * 4, dtype=np.int32),
        (rows, 1),
    )
    shape = close.shape
    return {
        "close": close,
        "preClose": pre_close,
        "industry_id": industry_id,
        "open": pre_close * 1.001,
        "high": np.maximum(close, pre_close) * 1.01,
        "low": np.minimum(close, pre_close) * 0.99,
        "volume": np.full(shape, 1.0e6),
        "amount": np.full(shape, 1.0e7),
    }


def _copy_panel(panel: dict) -> dict:
    return {
        name: value.copy() if hasattr(value, "copy") else value
        for name, value in panel.items()
    }


def _oracle_row(panel: dict, output_row: int) -> np.ndarray:
    close = np.asarray(panel["close"], dtype=np.float64)
    pre_close = np.asarray(panel["preClose"], dtype=np.float64)
    industry = np.asarray(panel["industry_id"])
    valid_return = (
        np.isfinite(close)
        & (close > 0.0)
        & np.isfinite(pre_close)
        & (pre_close > 0.0)
    )
    log_return = np.full(close.shape, np.nan, dtype=np.float64)
    log_return[valid_return] = np.log(
        close[valid_return] / pre_close[valid_return]
    )
    valid_return &= np.isfinite(log_return)

    expected = np.full(close.shape[1], np.nan, dtype=np.float32)
    start = output_row - TOTAL
    estimation_end = output_row - RECENT
    for stock_index in range(close.shape[1]):
        stock_industry = industry[start:output_row, stock_index]
        if (
            not np.all(np.isfinite(stock_industry))
            or np.any(stock_industry < 0)
            or np.any(stock_industry != stock_industry[0])
        ):
            continue

        peer = np.full(TOTAL, np.nan, dtype=np.float64)
        own = log_return[start:output_row, stock_index]
        for local_day, source_day in enumerate(range(start, output_row)):
            peer_mask = (
                (industry[source_day] == industry[source_day, stock_index])
                & valid_return[source_day]
            )
            peer_mask[stock_index] = False
            if np.count_nonzero(peer_mask) >= MIN_PEERS:
                peer[local_day] = np.mean(
                    log_return[source_day, peer_mask],
                    dtype=np.float64,
                )
        if not np.all(np.isfinite(own)) or not np.all(np.isfinite(peer)):
            continue

        x = peer[:ESTIMATION]
        y = own[:ESTIMATION]
        centered_x = x - np.mean(x)
        centered_y = y - np.mean(y)
        xx = float(np.dot(centered_x, centered_x))
        yy = float(np.dot(centered_y, centered_y))
        xy = float(np.dot(centered_x, centered_y))
        if xx <= 0.0 or yy <= 0.0:
            continue
        correlation = xy / np.sqrt(xx * yy)
        beta = xy / xx
        alpha = float(np.mean(y) - beta * np.mean(x))
        residual_variance = (yy - xy * xy / xx) / (ESTIMATION - 2)
        if (
            correlation < MIN_CORRELATION
            or not np.isfinite(residual_variance)
            or residual_variance <= 0.0
        ):
            continue
        recent_residual = (
            own[ESTIMATION:]
            - alpha
            - beta * peer[ESTIMATION:]
        )
        score = -float(np.sum(recent_residual)) / np.sqrt(
            residual_variance * RECENT
        )
        if np.isfinite(score):
            expected[stock_index] = score

    assert estimation_end == start + ESTIMATION
    return expected


def _assert_same_scores(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=3e-6,
        atol=3e-6,
        equal_nan=True,
    )


def _peer_oracle(
    log_return: np.ndarray,
    return_valid: np.ndarray,
    industry_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected = np.zeros(log_return.shape, dtype=np.float64)
    expected_valid = np.zeros(log_return.shape, dtype=bool)
    for row_index in range(log_return.shape[0]):
        for stock_index in range(log_return.shape[1]):
            group = industry_id[row_index, stock_index]
            if (
                not return_valid[row_index, stock_index]
                or not np.isfinite(group)
                or group < 0
            ):
                continue
            peer_mask = (
                return_valid[row_index]
                & np.isfinite(industry_id[row_index])
                & (industry_id[row_index] == group)
            )
            peer_mask[stock_index] = False
            if np.count_nonzero(peer_mask) < MIN_PEERS:
                continue
            expected[row_index, stock_index] = np.mean(
                log_return[row_index, peer_mask],
                dtype=np.float64,
            )
            expected_valid[row_index, stock_index] = True
    return expected, expected_valid


def test_fixed_variants_are_no_arg_registry_compatible():
    variants = (
        (IndustryPeerReversal20Strict, 20, 5),
        (IndustryPeerReversal60Strict, 60, 10),
        (IndustryPeerReversal120Strict, 120, 20),
    )
    for factor_class, estimation, recent in variants:
        factor = factor_class()
        assert factor.estimation_window == estimation
        assert factor.deviation_window == recent
        assert factor.hist_days == estimation + recent
        assert factor.min_peers == MIN_PEERS
        assert factor.min_correlation == MIN_CORRELATION
        assert factor.update_frequency == "daily"
        assert factor.pre_ranked is False
        assert factor.requires_full_history is False
        assert callable(factor.calc_batch)


def test_independent_leave_one_out_ols_oracle_and_score_direction_match():
    panel = _panel()
    actual = IndustryPeerReversal20Strict().calc_batch(panel)
    assert actual.dtype == np.float32
    assert np.isnan(actual[:TOTAL]).all()

    for output_row in (TOTAL, TOTAL + 3, len(actual) - 1):
        _assert_same_scores(actual[output_row], _oracle_row(panel, output_row))

    assert np.isfinite(actual[TOTAL]).all()
    assert actual[TOTAL, 0] > 0.0
    assert actual[TOTAL, 0] == np.max(actual[TOTAL, :5])


def test_output_t_ignores_all_t_row_inputs():
    panel = _panel(rows=47)
    factor = IndustryPeerReversal20Strict()
    expected = factor.calc_batch(panel)
    output_row = 35
    changed = _copy_panel(panel)
    changed["close"][output_row] = np.resize(
        np.asarray([np.nan, np.inf, 0.0, -1.0, 1.0e200]),
        changed["close"][output_row].shape,
    )
    changed["preClose"][output_row] = np.resize(
        np.asarray([np.inf, np.nan, -1.0, 0.0, 1.0e-200]),
        changed["preClose"][output_row].shape,
    )
    changed["industry_id"][output_row] = np.arange(
        changed["industry_id"].shape[1]
    )
    changed["open"][output_row] = 1.0e100
    changed["high"][output_row] = 1.0e200
    changed["low"][output_row] = -1.0e200
    changed["volume"][output_row] = 0.0
    changed["amount"][output_row] = np.nan

    actual = factor.calc_batch(changed)
    np.testing.assert_array_equal(actual[output_row], expected[output_row])


def test_appending_arbitrary_future_rows_preserves_prefix_bit_for_bit():
    panel = _panel(rows=43)
    factor = IndustryPeerReversal20Strict()
    expected = factor.calc_batch(panel)
    extra = 7
    extended = {}
    for name, values in panel.items():
        tail = np.resize(values[-1:], (extra, values.shape[1])).copy()
        if name == "close":
            tail[:] = np.resize(
                np.asarray([np.nan, np.inf, 0.0, -1.0]),
                tail.shape,
            )
        elif name == "preClose":
            tail[:] = np.resize(
                np.asarray([np.inf, np.nan, -1.0, 0.0]),
                tail.shape,
            )
        elif name == "industry_id":
            tail[:] = np.arange(values.shape[1])
        else:
            tail[:] = 0
        extended[name] = np.concatenate((values, tail), axis=0)

    actual = factor.calc_batch(extended)
    np.testing.assert_array_equal(actual[: len(expected)], expected)


def test_other_industries_cannot_affect_target_industry_scores():
    panel = _panel(rows=49)
    factor = IndustryPeerReversal20Strict()
    expected = factor.calc_batch(panel)[:, :5]
    changed = _copy_panel(panel)
    changed["close"][:, 5:] = np.resize(
        np.asarray([np.nan, np.inf, 0.0, -1.0]),
        changed["close"][:, 5:].shape,
    )
    changed["preClose"][:, 5:] = np.resize(
        np.asarray([np.inf, np.nan, -1.0, 0.0]),
        changed["preClose"][:, 5:].shape,
    )
    changed["industry_id"][:, 5:] = np.asarray([901, 902, 903, 904])

    actual = factor.calc_batch(changed)[:, :5]
    np.testing.assert_array_equal(actual, expected)


def test_changed_unknown_and_too_small_industries_are_strictly_nan():
    factor = IndustryPeerReversal20Strict()
    panel = _panel(rows=TOTAL + 3)
    baseline = factor.calc_batch(panel)
    assert np.isfinite(baseline[TOTAL]).all()

    changed = _copy_panel(panel)
    changed["industry_id"][9, 0] = 202
    changed_score = factor.calc_batch(changed)
    assert np.isnan(changed_score[TOTAL, 0])

    unknown = _copy_panel(panel)
    unknown["industry_id"][:TOTAL, 1] = -1
    unknown_score = factor.calc_batch(unknown)
    assert np.isnan(unknown_score[TOTAL, 1])

    too_small = _copy_panel(panel)
    too_small["industry_id"][:, :3] = 303
    too_small["industry_id"][:, 3:5] = 404
    too_small_score = factor.calc_batch(too_small)
    assert np.isnan(too_small_score[TOTAL, :5]).all()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_invalid_completed_return_has_no_fill_or_shortened_window(
    bad_value: float,
):
    panel = _panel(rows=TOTAL * 2 + 2)
    panel["close"][12, 0] = bad_value
    actual = IndustryPeerReversal20Strict().calc_batch(panel)
    assert np.isnan(actual[TOTAL : 12 + TOTAL + 1, 0]).all()
    assert np.isfinite(actual[12 + TOTAL + 1, 0])


def test_independent_positive_price_rescaling_is_invariant():
    panel = _panel(rows=45)
    factor = IndustryPeerReversal20Strict()
    expected = factor.calc_batch(panel)
    scaled = _copy_panel(panel)
    row_power = (np.arange(45) % 7 - 3)[:, None]
    stock_power = (np.arange(9) % 5 - 2)[None, :]
    scale = np.exp2(row_power + stock_power)
    scaled["close"] *= scale
    scaled["preClose"] *= scale
    actual = factor.calc_batch(scaled)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=3e-6,
        atol=3e-6,
        equal_nan=True,
    )


def test_only_required_fields_are_accessed_and_shapes_fail_loudly():
    class RecordingPanel(dict):
        def __init__(self, values):
            super().__init__(values)
            self.accesses = []

        def __getitem__(self, key):
            self.accesses.append(key)
            return super().__getitem__(key)

    recorded = RecordingPanel(_panel(rows=TOTAL + 1))
    IndustryPeerReversal20Strict().calc_batch(recorded)
    assert recorded.accesses == ["close", "preClose", "industry_id"]

    panel = _panel(rows=TOTAL + 1)
    for missing in ("close", "preClose", "industry_id"):
        invalid = _copy_panel(panel)
        invalid.pop(missing)
        with pytest.raises(KeyError):
            IndustryPeerReversal20Strict().calc_batch(invalid)

    invalid = _copy_panel(panel)
    invalid["close"] = invalid["close"][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        IndustryPeerReversal20Strict().calc_batch(invalid)

    invalid = _copy_panel(panel)
    invalid["industry_id"] = invalid["industry_id"][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        IndustryPeerReversal20Strict().calc_batch(invalid)

    invalid = _copy_panel(panel)
    invalid["industry_id"] = invalid["industry_id"].astype("U")
    with pytest.raises(ValueError, match="numeric encoded"):
        IndustryPeerReversal20Strict().calc_batch(invalid)


def test_flattened_peer_aggregation_matches_independent_float_id_oracle():
    rng = np.random.default_rng(20260826)
    rows = 17
    stocks = 15
    groups = np.asarray(
        [10.5] * 5 + [203.25] * 6 + [9001.75] * 4,
        dtype=np.float64,
    )
    industry_id = np.broadcast_to(groups, (rows, stocks)).copy()
    industry_id[:3, 0] = np.nan
    industry_id[6:9, 5] = -1.0
    industry_id[11:, 14] = 203.25
    log_return = rng.normal(0.0, 0.012, size=(rows, stocks))
    return_valid = rng.random((rows, stocks)) > 0.14
    log_return[~return_valid] = rng.normal(
        0.25,
        0.03,
        size=np.count_nonzero(~return_valid),
    )

    expected, expected_valid = _peer_oracle(
        log_return,
        return_valid,
        industry_id,
    )
    actual, actual_valid = _leave_one_out_peer_returns(
        log_return,
        return_valid,
        industry_id,
        MIN_PEERS,
    )

    np.testing.assert_array_equal(actual_valid, expected_valid)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_medium_panel_vectorized_runtime_guard():
    rows = 720
    stocks = 1800
    row = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    group = np.arange(stocks, dtype=np.int32) // 30 + 1000
    industry_id = np.broadcast_to(group, (rows, stocks)).copy()
    activation = (np.arange(stocks, dtype=np.int32) % 260)[None, :]
    industry_id[row.astype(np.int32) < activation] = -1
    common = 0.006 * np.sin(row / 13.0 + group[None, :] / 17.0)
    log_return = common + 0.0012 * np.cos(row / 7.0 + stock / 19.0)
    pre_close = 8.0 + (stock % 80) * 0.05 + row * 0.0002
    close = pre_close * np.exp(log_return)

    started = time.perf_counter()
    score = IndustryPeerReversal120Strict().calc_batch(
        {
            "close": close,
            "preClose": pre_close,
            "industry_id": industry_id,
        }
    )
    elapsed = time.perf_counter() - started

    assert np.isfinite(score[400:]).any()
    # Wide enough to catch accidental nested/date-wise regrouping while
    # retaining generous headroom for shared and virtualized CI runners.
    assert elapsed < 1.5


def test_factor_source_has_no_per_stock_loop_or_noncausal_fill_path():
    source_path = Path(inspect.getsourcefile(IndustryPeerReversal20Strict))
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loop_targets = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.comprehension))
        and isinstance(node.target, ast.Name)
    }
    assert not (
        loop_targets
        & {
            "stock",
            "stocks",
            "stock_index",
            "column",
            "columns",
            "position",
            "positions",
        }
    )
    assert "range(stocks)" not in source
    assert "sliding_window" not in source
    assert "np.nan_to_num" not in source
    assert "np.clip" not in source
    assert "panel.get(" not in source

    peer_source = inspect.getsource(_leave_one_out_peer_returns)
    peer_tree = ast.parse(peer_source)
    assert not any(
        isinstance(node, (ast.For, ast.While, ast.comprehension))
        for node in ast.walk(peer_tree)
    )

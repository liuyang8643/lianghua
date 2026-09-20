"""Independent numerical boundaries for the sole settlement authority."""
from dataclasses import fields
import math

import numpy as np
import pytest

from env.contracts import AccountState, OrderPlan
from env.simulator import DaySimulator, settlement_economics


pytestmark = pytest.mark.filterwarnings("error::numba.core.errors.NumbaIRAssumptionWarning")


def test_scalar_boundary_matches_math_isclose_and_keeps_ratio_source():
    boundaries = (1.0 - 1e-8, 1.0 + 1e-8, 1.0 / (1.0 - 1e-8))
    ratios = [np.nextafter(value, direction) for value in boundaries for direction in (-np.inf, 0.0, np.inf)]
    ratios += [1.0, 1.000000009, 1.000000015, 0.999999985]
    for ratio in ratios:
        result = settlement_economics(current_mark=1.0, current_close=ratio, next_preclose=1.0, next_open=2.0)
        expected = 1.0 if math.isclose(ratio, 1.0, rel_tol=1e-8, abs_tol=1e-8) else ratio
        assert result.effective_corporate_action_ratio.item() == expected
        assert result.reference_ratio.item() == ratio
        assert result.gross_return.item() == expected * 2.0
        assert result.ratio_source.item() == "close[T]/preClose[T+1]"
        assert result.gross_return.shape == ()


def test_price_overflow_underflow_preserves_flags_and_operation_order():
    tiny = np.nextafter(0.0, 1.0)
    largest = np.finfo(float).max
    result = settlement_economics(
        current_mark=[1.0, 1.0, largest, largest],
        current_close=[largest, tiny, 2.0, 0.5],
        next_preclose=[tiny, largest, 1.0, 1.0],
        next_open=[1.0, 1.0, largest, tiny],
    )
    np.testing.assert_array_equal(result.reference_ratio, [np.inf, 0.0, 2.0, 0.5])
    np.testing.assert_array_equal(result.gross_return_valid, [False, False, True, True])
    np.testing.assert_array_equal(result.gross_return, [np.nan, np.nan, np.inf, 0.0])
    # Validity records valid inputs; the historical law does not divide first
    # to hide intermediate overflow or convert this flag into output finiteness.
    assert result.gross_return[2] != 2.0


def test_all_fallback_sources_and_invalid_prices_are_explicit():
    result = settlement_economics(
        current_mark=[10.0, 10.0, 10.0, 10.0, np.nan],
        current_close=[10.0, np.inf, 12.0, -1.0, -0.0],
        next_preclose=[5.0, 5.0, np.inf, 0.0, np.nan],
        next_open=[6.0, np.nan, -1.0, np.inf, -np.inf],
    )
    np.testing.assert_array_equal(result.settlement_mark, [6.0, 5.0, 12.0, 10.0, np.nan])
    np.testing.assert_array_equal(result.reference_ratio, [2.0, 2.0, 1.0, 1.0, 1.0])
    assert result.mark_source.tolist() == ["open[T+1]", "preClose[T+1]", "close[T]", "current_mark[T]", "unavailable"]
    assert result.ratio_source.tolist() == ["close[T]/preClose[T+1]", "current_mark[T]_fallback/preClose[T+1]", "unavailable; ratio=1", "unavailable; ratio=1", "unavailable; ratio=1"]
    np.testing.assert_array_equal(result.gross_return_valid, [True, True, True, True, False])


@pytest.mark.parametrize("shape", [(), (5,), (7, 5), (0,), (0, 5), (7, 0)])
def test_chunked_and_complete_economics_preserve_shape_flags_and_bytes(shape):
    current = np.full(shape, 10.0)
    close = np.full(shape, 11.0)
    preclose = np.full(shape, 10.0)
    following = np.full(shape, 12.0)
    if current.size:
        current.flat[0] = np.nan
    inputs = dict(current_mark=current, current_close=close, next_preclose=preclose, next_open=following)
    complete = settlement_economics(**inputs)
    for chunk_rows in (1, 3, 64):
        light = settlement_economics(**inputs, diagnostics=False, chunk_rows=chunk_rows)
        assert light.gross_return.shape == shape
        assert light.gross_return.tobytes() == complete.gross_return.tobytes()
        assert light.gross_return_valid.tobytes() == complete.gross_return_valid.tobytes()
        for item in fields(light):
            if item.name not in ("gross_return", "gross_return_valid"):
                assert getattr(light, item.name) is None


def test_broadcast_noncontiguous_readonly_inputs_are_not_mutated():
    current = np.arange(1, 13, dtype=np.float32).reshape(3, 4)[:, ::2]
    close = np.array([[2.0, 4.0]])
    preclose = np.array([[1.0], [2.0], [3.0]])
    current.flags.writeable = False
    inputs = (current, close, preclose)
    before = [value.tobytes() for value in inputs]
    result = settlement_economics(current_mark=current, current_close=close, next_preclose=preclose, next_open=6.0)
    expected = (close / preclose) * 6.0 / current.astype(np.float64)
    assert result.gross_return.tobytes() == expected.tobytes()
    assert result.gross_return.shape == (3, 2)
    assert [value.tobytes() for value in inputs] == before
    assert not current.flags.writeable


@pytest.mark.parametrize("cash,expected_quantities,expected_cash", [
    (0.0, {"A": 151, "B": 152}, 0.0),
    (1.0, {"A": 152, "B": 151}, 1.0),
    (2.0, {"A": 152, "B": 152}, 0.0),
])
def test_cash_safe_rounding_uses_sorted_codes_and_preserves_total_value(cash, expected_quantities, expected_cash):
    results = []
    for codes in (("A", "B"), ("B", "A")):
        account = AccountState(cash=cash, positions={code: 101 for code in codes},
            sellable_positions={code: 101 for code in codes}, last_prices={code: 3.0 for code in codes},
            average_costs={code: 3.0 for code in codes}, nav=606.0 + cash, peak_nav=606.0 + cash)
        result = DaySimulator().step(account, OrderPlan("2026-07-03", (), {}),
            {code: 3.0 for code in codes}, {code: 2.0 for code in codes},
            close_prices={code: 3.0 for code in codes}, next_preclose_prices={code: 2.0 for code in codes},
            next_decision_date="2026-07-06")
        assert result.account_state.positions == expected_quantities
        assert result.account_state.cash == expected_cash
        assert result.account_state.nav == account.nav
        assert result.portfolio_return == 0.0
        assert result.fills == ()
        adjustments = result.diagnostics["corporate_action_adjustments"]
        for code in codes:
            assert adjustments[code]["cash_safe_floor_applied"] == (expected_quantities[code] == 151)
            assert adjustments[code]["nearest_ties_to_even_quantity"] == 152
        results.append(result)
    assert results[0] == results[1]


@pytest.mark.parametrize("quantity", [1, 101, 2**53 - 1, 2**53, 2**53 + 1])
def test_ordinary_ratio_shortcut_respects_float_integer_exactness_boundary(quantity):
    account = AccountState(cash=1.0, positions={"A": quantity}, sellable_positions={"A": quantity},
        last_prices={"A": 1.0}, average_costs={"A": 1.0}, nav=float(quantity) + 1.0, peak_nav=float(quantity) + 1.0)
    result = DaySimulator().step(account, OrderPlan("2026-07-03", (), {}), {"A": 1.0}, {"A": 1.0},
        close_prices={"A": 1.0}, next_preclose_prices={"A": 1.0})
    assert result.account_state.positions == {"A": int(round(quantity * 1.0))}
    assert result.account_state.cash == 1.0
    assert result.diagnostics["corporate_action_adjustments"] == {}


@pytest.mark.parametrize("invalid_chunk", [0, -1, True, 1.5])
def test_lightweight_settlement_rejects_invalid_chunk_contract(invalid_chunk):
    with pytest.raises(ValueError, match="positive int"):
        settlement_economics(current_mark=1, current_close=1, next_preclose=1, next_open=1,
            diagnostics=False, chunk_rows=invalid_chunk)


def test_incompatible_broadcast_shapes_are_rejected():
    with pytest.raises(ValueError):
        settlement_economics(current_mark=np.ones(2), current_close=np.ones(3), next_preclose=1, next_open=1)

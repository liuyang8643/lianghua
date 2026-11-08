from dataclasses import replace

import numpy as np
import pytest

from factor import PRODUCTION_FACTORS, precompute_factors
from factor.library.bilibili import (
    BILIBILI_FACTOR_NAMES as NAMES, calculate_bilibili_scores,
    prepare_bilibili_factor_definitions,
)
from offline_data import load_runtime_slice
from test_rl_runtime_slice import _runtime_arrays


def panel():
    data = _runtime_arrays(rows=7, stocks=2)
    close = np.array([10, 5, 6, 6, 7, 8, 9], dtype=float)[:, None]
    for name, offset in (("open", 0), ("close", 0), ("high", 1), ("low", -1)):
        data[name] = np.repeat(close + offset, 2, axis=1)
    data["preClose"] = np.repeat(np.array([8, 5, 5, 6, 6, 7, 8])[:, None], 2, axis=1).astype(float)
    data["issue_price"][:] = 8
    return data


def scores(data):
    return calculate_bilibili_scores(data["trade_dates"], data)


def test_exact_ipo_anchoring_and_split_adjusted_ohlc():
    result = scores(panel())
    assert all(np.isnan(values[0]).all() for values in result.values())
    # IPO close 10, issue price 8; split halves raw prices but not return chain.
    np.testing.assert_allclose(result[NAMES[0]][1:3, 0], [-1.25, -0.625])
    np.testing.assert_allclose(result[NAMES[1]][1:4, 0], [-1.25, -1.25, -1.5])
    assert result[NAMES[3]][1, 0] == pytest.approx(11 / 9)
    assert result[NAMES[3]][2, 0] == pytest.approx(12 / 8)
    assert result[NAMES[4]][2, 0] == pytest.approx(-4 / 8)
    np.testing.assert_array_equal(result[NAMES[2]], -result[NAMES[3]])
    assert all(not values.flags.writeable for values in result.values())


@pytest.mark.parametrize("phase",["next_open","completed_close"])
def test_selective_outputs_equal_complete_common_price_chain(phase):
    data=panel()
    data["preClose"][2,0]=np.nan
    complete=calculate_bilibili_scores(data["trade_dates"],data,output_phase=phase)
    for selection in ((NAMES[1],NAMES[3]),*(tuple([name]) for name in NAMES)):
        selected=calculate_bilibili_scores(data["trade_dates"],data,output_phase=phase,factor_names=selection)
        assert tuple(selected)==selection
        for name in selection:
            np.testing.assert_array_equal(selected[name],complete[name])
            assert not selected[name].flags.writeable


@pytest.mark.parametrize("names",[(),("unknown",),(NAMES[0],NAMES[0])])
def test_selective_outputs_require_declared_unique_vocabulary(names):
    data=panel()
    with pytest.raises(ValueError,match="vocabulary"):
        calculate_bilibili_scores(data["trade_dates"],data,factor_names=names)


def test_t_bar_and_all_future_bars_do_not_affect_t_scores():
    data = panel()
    original = scores(data)
    for name in ("open", "high", "low", "close", "preClose"):
        data[name][3:] *= 100
    changed = scores(data)
    for name in NAMES:
        np.testing.assert_array_equal(original[name][:4], changed[name][:4])


def test_completed_close_phase_is_current_bar_and_not_next_membership():
    data = panel()
    default = scores(data)
    close_phase = calculate_bilibili_scores(data["trade_dates"], data, output_phase="completed_close")
    for name in NAMES:
        np.testing.assert_array_equal(close_phase[name][:-1], default[name][1:])
    data["delisted_mask"][3:, 0] = True
    data["listing_age"][3:, 1] = -1
    for name in ("open", "high", "low", "close", "preClose"):
        data[name][3:] *= 100
    changed = calculate_bilibili_scores(data["trade_dates"], data, output_phase="completed_close")
    for name in NAMES:
        np.testing.assert_array_equal(close_phase[name][:3], changed[name][:3])
    assert np.isfinite(changed[NAMES[3]][2]).all()
    assert np.isnan(scores(data)[NAMES[3]][3]).all()


def test_completed_close_phase_computes_final_completed_bar():
    data = panel()
    expected = calculate_bilibili_scores(data["trade_dates"], data, output_phase="completed_close")
    data["high"][-1] *= 10
    changed = calculate_bilibili_scores(data["trade_dates"], data, output_phase="completed_close")
    np.testing.assert_array_equal(changed[NAMES[3]][:-1], expected[NAMES[3]][:-1])
    assert np.all(changed[NAMES[3]][-1] > expected[NAMES[3]][-1])
    with pytest.raises(ValueError, match="output_phase"):
        calculate_bilibili_scores(data["trade_dates"], data, output_phase="unknown")


def test_missing_preclose_permanently_breaks_adjusted_chain_but_not_raw():
    data = panel()
    data["preClose"][2, 0] = np.nan
    result = scores(data)
    assert np.isfinite(result[NAMES[0]][3:, 0]).all()
    for name in NAMES[1:]:
        assert np.isnan(result[name][3:, 0]).all()
        assert np.isfinite(result[name][3:, 1]).all()


def test_absent_bar_skips_extremes_without_inventing_a_close():
    data = panel()
    for name in ("open", "high", "low", "close", "preClose"):
        data[name][2] = np.nan
    data["preClose"][3] = 5  # Resume references the last actual raw close.
    result = scores(data)
    assert np.isnan(result[NAMES[0]][3]).all()
    assert np.isnan(result[NAMES[1]][3]).all()
    for name in NAMES[2:]:
        np.testing.assert_array_equal(result[name][3], result[name][2])
    assert np.isfinite(result[NAMES[1]][4]).all()


@pytest.mark.parametrize("field", ["close", "preClose"])
def test_partial_or_malformed_bars_fail_closed(field):
    data = panel()
    data[field][2, 0] = np.nan
    for name in NAMES[1:]:
        assert np.isnan(scores(data)[name][3:, 0]).all()


@pytest.mark.parametrize("field", ["high", "low"])
def test_missing_extreme_does_not_poison_independent_close_chain(field):
    data = panel()
    data[field][2, 0] = np.nan
    result = scores(data)
    assert np.isfinite(result[NAMES[1]][3:, 0]).all()
    for name in NAMES[2:]:
        assert np.isnan(result[name][3:, 0]).all()


def test_open_is_not_a_signal_formula_input_when_completed_hlc_exist():
    data = panel()
    expected = scores(data)
    data["open"][2, 0] = np.nan
    actual = scores(data)
    for name in NAMES:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_missing_ipo_and_prelisting_stocks_do_not_gain_lifetime_history():
    data = panel()
    data["issue_date"][0] -= np.timedelta64(1, "D")
    data["issue_date"][1] = data["trade_dates"][3]
    data["listing_age"][:3, 1] = -1
    data["listing_age"][3:, 1] = np.arange(4)
    result = scores(data)
    for name in NAMES[1:]:
        assert np.isnan(result[name][:, 0]).all()
        assert np.isnan(result[name][:4, 1]).all()
        assert np.isfinite(result[name][4:, 1]).all()


def test_definition_factory_shares_one_calculation_and_binds_snapshot(tmp_path, monkeypatch):
    import factor.library.bilibili as module
    data = panel()
    path = tmp_path / "runtime.npz"
    np.savez(path, **data)
    runtime = load_runtime_slice(path, data["trade_dates"][3], data["trade_dates"][-1], preload_rows=1000)
    count = 0
    original = module.calculate_bilibili_scores

    def counted(*args):
        nonlocal count
        count += 1
        return original(*args)

    monkeypatch.setattr(module, "calculate_bilibili_scores", counted)
    definitions = prepare_bilibili_factor_definitions(runtime)
    batch = precompute_factors(runtime, definitions=definitions)
    assert count == 1
    assert batch.factor_names == NAMES
    assert batch.schema_version.startswith("wbr.research-")
    assert len({d.metadata.implementation_hash for d in definitions}) == 5
    assert all(len(d.metadata.implementation_hash) == 64 for d in definitions)
    assert all(d.raw_runtime_view for d in definitions)
    different = {name: value.astype(np.float64) for name, value in runtime.data.items()}
    different["close"][0, 0] += 1
    with pytest.raises(ValueError, match="different runtime panel"):
        definitions[0].implementation().calc_batch(different)
    shallow = replace(runtime, manifest=replace(runtime.manifest, requested_preload_rows=3))
    with pytest.raises(ValueError, match="first row"):
        prepare_bilibili_factor_definitions(shallow)


def test_bound_inputs_preserve_identity_and_reject_same_value_copies(tmp_path):
    from factor.library.bilibili import REQUIRED_FIELDS
    data = panel()
    path = tmp_path / "runtime.npz"
    np.savez(path, **data)
    runtime = load_runtime_slice(path, data["trade_dates"][3], data["trade_dates"][-1], preload_rows=1000)
    definitions = prepare_bilibili_factor_definitions(runtime)
    instance = definitions[0].implementation()
    expected = instance.calc_batch(runtime.data)
    assert expected is instance.calc_batch(dict(runtime.data))
    for field in REQUIRED_FIELDS:
        copied = dict(runtime.data)
        copied[field] = copied[field].copy()
        copied[field].flags.writeable = False
        with pytest.raises(ValueError, match="different runtime panel"):
            instance.calc_batch(copied)
    other = load_runtime_slice(path, data["trade_dates"][3], data["trade_dates"][-1], preload_rows=1000)
    with pytest.raises(ValueError, match="different runtime panel"):
        precompute_factors(other, definitions=definitions)
    runtime.field("close").flags.writeable = True
    try:
        with pytest.raises(ValueError, match="read-only"):
            precompute_factors(runtime, definitions=definitions)
    finally:
        runtime.field("close").flags.writeable = False


def test_view_input_mode_matches_legacy_float64_outputs_and_preserves_production(tmp_path):
    data = panel()
    path = tmp_path / "runtime.npz"
    np.savez(path, **data)
    runtime = load_runtime_slice(path, data["trade_dates"][3], data["trade_dates"][-1], preload_rows=1000)
    definitions = prepare_bilibili_factor_definitions(runtime)
    reference = calculate_bilibili_scores(runtime.trade_dates, runtime.data)

    def legacy_class(name):
        class LegacyBound:
            def calc_batch(self, panel):
                assert panel["close"].dtype == np.float64
                assert panel["close"] is not runtime.field("close")
                return reference[name]
        return LegacyBound

    legacy = tuple(replace(d, raw_runtime_view=False, implementation=legacy_class(d.metadata.name)) for d in definitions)
    views = precompute_factors(runtime, definitions=definitions)
    converted = precompute_factors(runtime, definitions=legacy)
    assert views.schema_hash != converted.schema_hash
    for name in ("raw", "ranks", "validity", "filters"):
        np.testing.assert_array_equal(getattr(views, name), getattr(converted, name))
    assert not any(d.raw_runtime_view for d in PRODUCTION_FACTORS)
    default = precompute_factors(runtime)
    explicit = precompute_factors(runtime, definitions=tuple(replace(d, raw_runtime_view=False) for d in PRODUCTION_FACTORS))
    assert default.schema_hash == explicit.schema_hash
    for name in ("raw", "ranks", "validity", "filters"):
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))
    with pytest.raises(ValueError, match="lagged fields"):
        replace(definitions[0], metadata=replace(definitions[0].metadata, lagged_fields=("close",)))

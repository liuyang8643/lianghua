import numpy as np
import pytest

from factor.library.bilibili_events import (
    Bili2560Events, BiliBigBear, BiliBigBull, BiliHighDoji,
    BiliLongLowerShadow, BiliLongUpperShadow, BiliLowDoji,
    BiliVolumeRatio25To5Inclusive, BiliVolumeRatioAbove5, RESEARCH_FACTOR_DEFINITIONS,
)


def _panel(rows=64, stocks=1):
    close = np.full((rows, stocks), 10.0)
    volume = np.full((rows, stocks), 100.0)
    return {"close": close, "volume": volume}


def test_crosses_score_one_and_non_event_zero():
    panel = _panel()
    # d=62: close crosses MA25; volume crosses MA5 over MA60, observed at T=63.
    panel["close"][38:62] = 9.0
    panel["close"][62] = 11.0
    panel["volume"][0:5] = 200.0
    panel["volume"][59:62] = 100.0
    panel["volume"][62] = 500.0
    result = Bili2560Events().calc_batch(panel)
    assert result[63, 0] == 1.0
    assert result[62, 0] == 0.0


def test_volume_mean_is_distinct_from_price_mean():
    panel = _panel()
    panel["close"][38:62] = 9.0
    panel["close"][62] = 11.0
    # Volume remains below its 60-day mean, so no event despite price cross.
    panel["volume"][0:5] = 200.0
    panel["volume"][59:63] = 90.0
    result = Bili2560Events().calc_batch(panel)
    assert result[63, 0] == 0.0


def test_same_day_and_future_perturbations_do_not_change_score():
    panel = _panel(rows=66)
    panel["close"][38:62] = 9.0
    panel["close"][62] = 11.0
    panel["volume"][0:5] = 200.0
    panel["volume"][59:62] = 100.0
    panel["volume"][62] = 500.0
    baseline = Bili2560Events().calc_batch(panel)
    changed = {k: v.copy() for k, v in panel.items()}
    changed["close"][63] = 9999.0
    changed["volume"][63] = 9999.0
    changed["close"][64] = -9999.0
    changed["volume"][64] = -9999.0
    np.testing.assert_array_equal(Bili2560Events().calc_batch(changed)[:64], baseline[:64])


def test_missing_window_is_nan_and_strict_equal_is_not_cross():
    panel = _panel()
    panel["close"][38, 0] = np.nan
    result = Bili2560Events().calc_batch(panel)
    assert np.isnan(result[63, 0])
    equal = _panel()
    # Flat series makes all comparisons equal, therefore no strict signal.
    assert Bili2560Events().calc_batch(equal)[63, 0] == 0.0


def test_candle_events_have_hand_checked_examples_and_t_minus_one_alignment():
    cases = (
        (BiliLongUpperShadow, (10, 10.05, 10.60, 9.99), 1),
        (BiliLongLowerShadow, (10, 10.01, 10.01, 9.40), 1),
        (BiliBigBull, (10, 10.60, 10.70, 9.90), 1),
        (BiliBigBear, (10, 9.40, 10.10, 9.30), 1),
        (BiliHighDoji, (11, 11.05, 11.30, 10.75), 1),
        (BiliLowDoji, (9, 9.05, 9.30, 8.75), 1),
    )
    for factor, bar, expected in cases:
        panel = {k: np.full((22, 1), 10.0, dtype=float) for k in ("open", "close", "high", "low")}
        for key, value in zip(("open", "close", "high", "low"), bar):
            panel[key][20, 0] = value
        actual = factor().calc_batch(panel)
        assert actual[20, 0] == 0.0
        assert actual[21, 0] == expected


@pytest.mark.parametrize("factor,bar", [
    (BiliHighDoji, (10, 10.60, 10.80, 9.80)),
    (BiliLowDoji, (10, 9.40, 10.20, 9.00)),
])
def test_ma_deviation_without_doji_is_not_an_event(factor, bar):
    panel = {key: np.full((21, 1), 10.0) for key in ("open", "close", "high", "low")}
    for key, value in zip(("open", "close", "high", "low"), bar):
        panel[key][19, 0] = value
    assert factor().calc_batch(panel)[20, 0] == 0.0


@pytest.mark.parametrize("factor,opening", [(BiliHighDoji, 11.0), (BiliLowDoji, 9.0)])
def test_doji_uses_exactly_twenty_completed_prices_and_no_prior_ma(factor, opening):
    panel = {key: np.full((22, 1), 10.0) for key in ("open", "close", "high", "low")}
    for key, value in {"open": opening, "close": opening, "high": opening + .2, "low": opening - .2}.items():
        panel[key][19:21, 0] = value
    actual = factor().calc_batch(panel)
    assert np.isnan(actual[:20]).all()
    assert actual[20, 0] == 1
    panel["close"][0] = np.nan
    changed = factor().calc_batch(panel)
    assert np.isnan(changed[20, 0])
    assert changed[21, 0] == 1  # Invalid row zero has left the sole MA window.
    assert factor.hist_days == 20


@pytest.mark.parametrize("factor", [BiliLongUpperShadow, BiliLongLowerShadow, BiliBigBull, BiliBigBear, BiliHighDoji, BiliLowDoji])
def test_candle_signals_reject_illegal_ohlc_and_are_causal(factor):
    panel = {key: np.full((24, 1), 10.0) for key in ("open", "close", "high", "low")}
    baseline = factor().calc_batch(panel)
    for key in panel:
        changed = {name: values.copy() for name, values in panel.items()}
        changed[key][21:] = 999
        np.testing.assert_array_equal(factor().calc_batch(changed)[:22], baseline[:22])
    for key, bad in (("open", 0), ("close", -1), ("low", 0), ("low", 11), ("high", 9), ("close", np.inf)):
        changed = {name: values.copy() for name, values in panel.items()}
        changed[key][20] = bad
        assert np.isnan(factor().calc_batch(changed)[21, 0])


@pytest.mark.parametrize("field,bad,row", [("close", 0, 50), ("close", -1, 50), ("volume", -1, 10), ("volume", np.inf, 10)])
def test_2560_rejects_malformed_values_anywhere_in_required_windows(field, bad, row):
    panel = _panel()
    panel[field][row] = bad
    assert np.isnan(Bili2560Events().calc_batch(panel)[63, 0])


def test_zero_volume_is_a_valid_non_event_and_first_cross_row_is_61():
    panel = _panel(rows=62)
    panel["volume"][:] = 0
    result = Bili2560Events().calc_batch(panel)
    assert np.isnan(result[:61]).all()
    assert result[61, 0] == 0


def test_volume_ratio_uses_five_prior_bars_and_assumed_inclusive_bounds():
    volume = np.full((8, 6), 100.0)
    volume[5] = [249, 250, 499, 500, 501, 0]
    expected = ([0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0])
    for factor, scores in zip((BiliVolumeRatio25To5Inclusive, BiliVolumeRatioAbove5), expected):
        result = factor().calc_batch({"volume": volume})
        assert np.isnan(result[:6]).all()
        np.testing.assert_array_equal(result[6], scores)
        changed = volume.copy()
        changed[6:] = np.nan
        np.testing.assert_array_equal(factor().calc_batch({"volume": changed})[:7], result[:7])


@pytest.mark.parametrize("factor", [BiliVolumeRatio25To5Inclusive, BiliVolumeRatioAbove5])
def test_volume_ratio_missing_negative_and_zero_denominator_are_unavailable(factor):
    for row, bad in ((0, np.nan), (4, -1), (5, np.inf), (5, -1)):
        volume = np.full((7, 1), 100.0)
        volume[row] = bad
        assert np.isnan(factor().calc_batch({"volume": volume})[6, 0])
    volume = np.zeros((7, 1))
    volume[5] = 500
    assert np.isnan(factor().calc_batch({"volume": volume})[6, 0])


def test_research_metadata_records_actual_windows_and_assumption_versions():
    definitions = {item.metadata.name: item for item in RESEARCH_FACTOR_DEFINITIONS}
    assert len(definitions) == 9
    for name in ("BiliHighDoji", "BiliLowDoji"):
        assert definitions[name].metadata.hist_days == 20
        assert definitions[name].metadata.version == "bilibili-candle-events-v2-doji-ma20"
    for name in ("BiliVolumeRatio25To5Inclusive", "BiliVolumeRatioAbove5"):
        metadata = definitions[name].metadata
        assert metadata.hist_days == 6
        assert metadata.required_fields == ("volume",)
        assert "assumed-bounds" in metadata.version

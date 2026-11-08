from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from offline_data.financial_versions import FINANCIAL_FIELDS, normalize_financial_events
from factor.library.video_financial import calculate_financial_scores, prepare_financial_factor_definitions
from env.calendar_replay import mark_intraday, select_video_month, run_monthly_video_replay
from env.contracts import AccountState, OrderPlan
from env.fees import FeeSchedule
from env.simulator import DaySimulator


def events():
    result = {}
    for table, fields in FINANCIAL_FIELDS.items():
        records = []
        for year in (2019, 2020, 2021):
            for quarter, month in enumerate((3, 6, 9, 12), 1):
                period = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd()
                value = {field: float(quarter * 100) for field in fields}
                if table == "Balance":
                    value = {field: 1000.0 for field in fields}
                records.append({"stock_code": "000001.SZ", "m_timetag": str(period.date()),
                                "m_anntime": str((period + pd.Timedelta(days=20)).date()), **value})
        result[table] = normalize_financial_events(pd.DataFrame(records), ("000001.SZ",), fields)
    return result


def test_future_revision_does_not_change_past_scores():
    original = events()
    dates = np.array(["2021-04-20", "2021-04-21", "2021-05-10"], dtype="datetime64[D]")
    before = calculate_financial_scores(dates, np.ones((3, 1)), np.full((3, 1), 1000.), original)
    event = original["Income"]
    revised = replace(event, announcement_dates=np.append(event.announcement_dates, np.datetime64("2030-01-01")),
                      quarters=np.append(event.quarters, 2020 * 4 + 3), stock_columns=np.append(event.stock_columns, 0),
                      values=np.vstack((event.values, np.full(len(event.fields), 1e10))))
    after = calculate_financial_scores(dates, np.ones((3, 1)), np.full((3, 1), 1000.), {**original, "Income": revised})
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    # TTM remains 400, not a partial-year cumulative 100.
    np.testing.assert_allclose(before["HighROEDecile"], .4)


def test_conflicts_are_unavailable_and_keep_other_fields():
    frame = pd.DataFrame([{"stock_code": "000001.SZ", "m_timetag": "20201231", "m_anntime": "20210301", "x": x, "y": 2}
                          for x in (1, 3)])
    result = normalize_financial_events(frame, ("000001.SZ",), ("x", "y"))
    assert np.isnan(result.values[0, 0]) and result.values[0, 1] == 2
    assert result.audit["conflicting_fields_unavailable"] == {"x": 1, "y": 0}


def test_period_end_placeholder_is_quarantined_not_shifted_to_next_day():
    frame = pd.DataFrame([
        {"stock_code": "000001.SZ", "m_timetag": "20201231", "m_anntime": "20201231", "x": 99.},
        {"stock_code": "000001.SZ", "m_timetag": "20201231", "m_anntime": "20210301", "x": 2.},
    ])
    result = normalize_financial_events(frame, ("000001.SZ",), ("x",))
    np.testing.assert_array_equal(result.announcement_dates, np.array(["2021-03-01"], dtype="datetime64[D]"))
    assert result.values[0, 0] == 2.
    assert result.audit["quarantined_period_end_announcement_rows"] == 1


def test_new_disclosure_activates_only_after_announcement_day():
    original = events()
    income = original["Income"]
    values = income.values.copy()
    values[income.quarters == 2021 * 4, 0] = 200.0
    updated = {**original, "Income": replace(income, values=values)}
    dates = np.array(["2021-04-20", "2021-04-21"], dtype="datetime64[D]")
    scores = calculate_financial_scores(dates, np.ones((2, 1)), np.full((2, 1), 1000.), updated)
    np.testing.assert_allclose(scores["HighROEDecile"][:, 0], [.4, .5])


def test_daily_binding_skips_unused_prices_but_preserves_disclosure_history(monkeypatch):
    dates = np.array(["2021-04-19", "2021-04-20", "2021-04-21"], dtype="datetime64[D]")
    fields = {"open": np.ones((3, 1)), "total_share": np.full((3, 1), 1000.),
              "listing_age": np.full((3, 1), 1000), "delisted_mask": np.zeros((3, 1), bool)}
    for array in fields.values():
        array.flags.writeable = False
    runtime = SimpleNamespace(trade_dates=dates, n_dates=3, n_stocks=1, stock_codes=("000001.SZ",),
                              decision_start=2, decision_stop=3, field=fields.__getitem__,
                              manifest=SimpleNamespace(source_sha256="test"))
    definitions = prepare_financial_factor_definitions(runtime, events(), {"sha256": "test"})
    full = calculate_financial_scores(dates, fields["open"], fields["total_share"], events())
    for definition in definitions:
        bound = definition.implementation().calc_batch(fields)
        assert np.isnan(bound[0]).all() and not bound.flags.writeable
        np.testing.assert_array_equal(bound[1:], full[definition.metadata.name][1:])

    # A dependency change must invalidate cached research bindings as well.
    import factor.library.video_financial as financial_module
    original_replay = financial_module.iter_financial_fields

    def changed_replay(*args, **kwargs):
        for row, replay_fields in original_replay(*args, **kwargs):
            yield row, {**replay_fields, "financial_profit_ttm": replay_fields["financial_profit_ttm"] * 2}

    monkeypatch.setattr(financial_module, "iter_financial_fields", changed_replay)
    changed = prepare_financial_factor_definitions(runtime, events(), {"sha256": "test"})
    assert all(a.metadata.implementation_hash != b.metadata.implementation_hash
               for a, b in zip(definitions, changed))


def test_decile_is_selected_before_execution_and_missing_not_backfilled():
    score = np.arange(21., dtype=float)
    score[-1] = np.nan
    chosen = select_video_month(score, np.ones(21, bool), selection_fraction=.1)
    np.testing.assert_array_equal(np.flatnonzero(chosen), [18, 19])


def test_five_percent_and_small_industry_selection():
    scores = np.arange(42, dtype=float)
    scores[-1] = np.nan
    selected = select_video_month(scores, np.ones(42, bool), selection_fraction=.05)
    np.testing.assert_array_equal(np.flatnonzero(selected), [39, 40])
    assert not select_video_month(scores[:19], np.ones(19, bool), selection_fraction=.05).any()
    counts = np.arange(5545)
    np.testing.assert_array_equal(np.floor(counts*.1).astype(int), counts//10)


def test_same_day_mark_does_not_unlock_new_buys():
    simulator = DaySimulator(FeeSchedule(0, 0, 0, 0, 0))
    state = AccountState(cash=10000., nav=10000.)
    marked = mark_intraday(simulator, state, OrderPlan("2021-02-01", buy_orders={"000001.SZ": 100}),
                           {"000001.SZ": 10.}, {"000001.SZ": 11.}, "2021-02-01")
    assert marked.account_state.sellable_positions["000001.SZ"] == 0
    assert marked.account_state.nav == 10100.


def test_monthly_schedule_buys_once_and_exits_at_month_end_close(tmp_path):
    dates = np.array(["2021-01-29", "2021-02-01", "2021-02-02", "2021-02-26", "2021-03-01"], dtype="datetime64[D]")
    prices = np.full((5, 1), 10.)
    data = {"open": prices, "close": prices + .2, "preClose": prices,
            "issue_price": np.array([10.]), "issue_date": np.array(["2000-01-01"], dtype="datetime64[D]"),
            "st_mask": np.zeros((5, 1), bool), "delisted_mask": np.zeros((5, 1), bool), "listing_age": np.full((5, 1), 1000)}
    runtime = SimpleNamespace(n_dates=5, n_stocks=1, stock_codes=("000001.SZ",), trade_dates=dates,
                              decision_start=1, decision_stop=5, field=data.__getitem__)
    run_monthly_video_replay(runtime, np.ones((5, 1)), name="test", selection_fraction=1.0, output=tmp_path,
                              initial_cash=100000., fees=FeeSchedule(0, 0, 0, 0, 0))
    import json
    fills = [json.loads(line) for line in (tmp_path / "fills.jsonl").read_text().splitlines()]
    assert {f["timestamp"] for f in fills if f["side"] == "buy"} == {"2021-02-01", "2021-03-01"}
    sells = [f for f in fills if f["side"] == "sell"]
    assert len(sells) == 1 and sells[0]["timestamp"] == "2021-02-26" and sells[0]["phase"] == "close"
    assert sells[0]["price"] == 10.2


def test_truncated_last_month_end_still_liquidates(tmp_path):
    dates = np.array(["2024-02-29", "2024-03-01", "2024-03-29"], dtype="datetime64[D]")
    prices = np.full((3, 1), 10.)
    data = {"open": prices, "close": prices, "preClose": prices, "issue_price": np.array([10.]),
            "issue_date": np.array(["2000-01-01"], dtype="datetime64[D]"), "st_mask": np.zeros((3, 1), bool),
            "delisted_mask": np.zeros((3, 1), bool), "listing_age": np.full((3, 1), 1000)}
    runtime = SimpleNamespace(n_dates=3, n_stocks=1, stock_codes=("000001.SZ",), trade_dates=dates,
                              decision_start=1, decision_stop=3, field=data.__getitem__)
    result = run_monthly_video_replay(runtime, np.ones((3, 1)), name="test", selection_fraction=1.0, output=tmp_path,
                                     initial_cash=100000., month_end_dates=np.array(["2024-03-29"], dtype="datetime64[D]"))
    import json
    fills = [json.loads(line) for line in (tmp_path / "fills.jsonl").read_text().splitlines()]
    assert fills[-1]["side"] == "sell" and fills[-1]["timestamp"] == "2024-03-29"
    assert result["terminal_positions"] == {} and result["terminal_pending_exit"] == {}
    assert np.load(tmp_path / "trace.npz")["held_counts"][-1] == 0


@pytest.mark.parametrize("split_ratio", [1.0, 2.0])
def test_retry_sells_only_old_cohort_even_after_corporate_action(tmp_path, split_ratio):
    dates = np.array(["2021-01-29", "2021-02-01", "2021-02-26", "2021-03-01", "2021-03-02"], dtype="datetime64[D]")
    opening = np.array([[10, 10], [10, 10], [10, 10], [8.1, 10], [8.2 / split_ratio, 10]], float)
    closing = np.array([[10, 10], [10, 10], [9, 10], [8.1, 10], [8.2 / split_ratio, 10]], float)
    preclose = np.array([[10, 10], [10, 10], [10, 10], [9, 10], [8.1 / split_ratio, 10]], float)
    data = {"open": opening, "close": closing, "preClose": preclose, "issue_price": np.array([10., 10.]),
            "issue_date": np.array(["2000-01-01"] * 2, dtype="datetime64[D]"), "st_mask": np.zeros((5, 2), bool),
            "delisted_mask": np.zeros((5, 2), bool), "listing_age": np.full((5, 2), 1000)}
    runtime = SimpleNamespace(n_dates=5, n_stocks=2, stock_codes=("000001.SZ", "000002.SZ"), trade_dates=dates,
                              decision_start=1, decision_stop=5, field=data.__getitem__)
    scores = np.ones((5, 2)); scores[2, 1] = np.nan
    run_monthly_video_replay(runtime, scores, name="test", selection_fraction=1.0, output=tmp_path,
                              initial_cash=100000., fees=FeeSchedule(0, 0, 0, 0, 0))
    import json
    fills = [json.loads(line) for line in (tmp_path / "fills.jsonl").read_text().splitlines()]
    initial = next(f["quantity"] for f in fills if f["side"] == "buy" and f["code"] == "000001.SZ")
    new = [f for f in fills if f["side"] == "buy" and f["timestamp"] == "2021-03-01"]
    assert new
    retry = [f for f in fills if f["phase"] == "retry_open"]
    assert len(retry) == 1 and retry[0]["quantity"] == initial * split_ratio
    assert np.load(tmp_path / "trace.npz")["held_counts"][-1] == 1

"""Original financial values, source availability and factor-free actor inputs."""
from dataclasses import replace
import numpy as np
import pandas as pd
import torch as th

from offline_data.financial_versions import FINANCIAL_FIELDS, normalize_financial_events, iter_financial_fields
from env.observation import RAW_MISSING_VALUE, HISTORICAL_STOCK_FEATURE_NAMES, LATEST_STOCK_FEATURE_NAMES
from ai.rl.raw_features import RawPanelFeatures, RAW_PANEL_CONFIG
from factor.library.bilibili import calculate_lifetime_state, calculate_bilibili_scores
from rl_test_data import build_episode, fit_normalizer
from ai.rl.train import scheduled_learning_rate


def test_raw_financial_negative_same_day_and_old_quarter_revision():
    events = {}
    for table, fields in FINANCIAL_FIELDS.items():
        records = []
        for period, announcement, value in [('2020-12-31','2021-03-20',-100.),
                                             ('2021-03-31','2021-04-20',-20.),
                                             ('2020-12-31','2021-04-25',-999.)]:
            records.append(dict(stock_code='000001.SZ',m_timetag=period,m_anntime=announcement,
                                **{field:value for field in fields}))
        events[table] = normalize_financial_events(pd.DataFrame(records), ('000001.SZ',), fields)
    dates = np.array(['2021-03-20','2021-03-21','2021-04-20','2021-04-21','2021-04-26'],dtype='datetime64[D]')
    result = [fields for _,fields in iter_financial_fields(dates,1,events)]
    key = 'financial_raw.Income.net_profit_excl_min_int_inc'
    assert np.isnan(result[0][key][0])
    assert [r[key][0] for r in result[1:]] == [-100.,-100.,-20.,-20.]
    assert result[-1]['financial_raw.Income.announcement_age_days'][0] == 6
    assert result[-1]['financial_raw.Income.report_age_days'][0] == 26
    assert result[-1]['financial_raw.Income.report_quarter'][0] == 1


def test_factor_scores_do_not_enter_observation_and_signed_values_survive(tmp_path):
    episode = build_episode(tmp_path/'runtime.npz')
    builder = episode.observation_builder
    row = episode.decision_start
    first = builder.build_static_rows(row,row+1).stock_panel
    assert not any(n.startswith(('factor_rank.','filter_pass.')) for n in builder.schema.stock_feature_names)
    changed = replace(episode.factors, ranks=np.zeros_like(episode.factors.ranks), raw=np.zeros_like(episode.factors.raw))
    other = type(builder)(episode.runtime,changed,day_markets=builder.day_markets,lookback=builder.lookback)
    np.testing.assert_array_equal(first,other.build_static_rows(row,row+1).stock_panel)
    store = episode.market_store
    values = store.raw_rows.copy()
    field = builder.schema.stock_feature_names.index('financial_raw.Income.net_profit_excl_min_int_inc')
    values[store.decision_start, :3, field] = [-1.,0.,RAW_MISSING_VALUE]
    values.flags.writeable=False
    model=RawPanelFeatures(episode.encoder.output_schema.to_dict(),RAW_PANEL_CONFIG).cuda()
    model.bind_market_store(replace(store,raw_rows=values),fit_normalizer(episode))
    actual=model.raw_bank[store.decision_start,:3,field]
    assert actual[0] <= -2 and actual[1] == 0 and actual[2] == -1


def test_latest_fields_have_no_historical_path_and_primitives_share_chain(tmp_path):
    episode=build_episode(tmp_path/'runtime.npz')
    normalizer=fit_normalizer(episode)
    model=RawPanelFeatures(episode.encoder.output_schema.to_dict(),RAW_PANEL_CONFIG).cuda().eval()
    store=episode.market_store
    ref=store.decision_start+2
    with th.no_grad():
        model.bind_market_store(store,normalizer)
        expected=model._market_many(np.array([ref]))
        rows=store.raw_rows.copy()
        indices=[store.schema.stock_feature_names.index(n) for n in LATEST_STOCK_FEATURE_NAMES]
        rows[:ref,:,indices] *= 2
        rows.flags.writeable=False
        model.bind_market_store(replace(store,raw_rows=rows),normalizer)
        th.testing.assert_close(model._market_many(np.array([ref])),expected,rtol=0,atol=0)
    assert set(HISTORICAL_STOCK_FEATURE_NAMES).isdisjoint(LATEST_STOCK_FEATURE_NAMES)
    primitive=calculate_lifetime_state(episode.runtime.trade_dates,episode.runtime.data)
    scores=calculate_bilibili_scores(episode.runtime.trade_dates,episode.runtime.data)
    np.testing.assert_allclose(primitive['ipo_lifetime_high']/primitive['ipo_lifetime_low'],
                               scores['BiliHighLifetimeRangeRatio'],equal_nan=True)
    np.testing.assert_allclose(-primitive['ipo_adjusted_close']/episode.runtime.data['issue_price'],
                               scores['BiliAdjustedIssueDiscount'],equal_nan=True)


def test_public_cached_and_shared_observation_keep_complete_ipo_history(tmp_path):
    from env.backtest import EpisodeSession
    from env.shared_episode import SharedPreparedEpisodeOwner
    episode=build_episode(tmp_path/'runtime.npz')
    compact=episode.compact_for_replay()
    for candidate in (episode,compact):
        session=EpisodeSession(candidate,initial_cash=1_000_000.)
        session.reset()
        public=session.current_observation
        ref=candidate.market_store.row_reference(public.decision_date)
        np.testing.assert_array_equal(public.stock_panel,candidate.market_store.window(ref)[0])
    with SharedPreparedEpisodeOwner.create(compact) as owner:
        attached=owner.descriptor.attach()
        try:
            session=EpisodeSession(attached.episode,initial_cash=1_000_000.)
            session.reset()
            public=session.current_observation
            expected=EpisodeSession(episode,initial_cash=1_000_000.)
            expected.reset()
            np.testing.assert_array_equal(public.stock_panel,expected.current_observation.stock_panel)
        finally:
            attached.close()


def test_learning_rate_decay_uses_global_budget_not_each_learn_call():
    assert scheduled_learning_rate(3e-4,.1,.2,20,100)==3e-4
    assert np.isclose(scheduled_learning_rate(3e-4,.1,.2,60,100),1.65e-4)
    assert np.isclose(scheduled_learning_rate(3e-4,.1,.2,100,100),3e-5)
    whole=[scheduled_learning_rate(3e-4,.1,.2,i,100) for i in range(1,101)]
    resumed=[scheduled_learning_rate(3e-4,.1,.2,i,100) for i in range(41,101)]
    assert whole[40:]==resumed


def test_decay_resume_cannot_reset_sealed_horizon(tmp_path):
    import pytest
    from ai.rl.train import build_parser, _validate_resume_arguments
    args=build_parser().parse_args(["--runtime", "runtime.npz", "--resume-from",
        str(tmp_path/"latest_train_model.zip"), "--learning-rate-end-fraction", "0.1"])
    with pytest.raises(ValueError, match="sealed transition horizon"):
        _validate_resume_arguments(args)

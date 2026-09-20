from __future__ import annotations
from dataclasses import replace
import json
import numpy as np
import pytest
from env.encoder import ObservationEncoder, EncodedObservationSchema, RawMarketStore, TrainOnlyNormalizer
from env.observation import RAW_MISSING_VALUE
from offline_data.financial_versions import RAW_FINANCIAL_VALUE_NAMES
from env.shared_episode import SharedPreparedEpisodeOwner
from test_rl_observation import synthetic_inputs, sample_account, make_builder
from rl_test_data import build_episode, fit_normalizer


def fixture_raw():
    runtime,factors=synthetic_inputs(date_count=10)
    builder=make_builder(runtime,factors,lookback=4)
    encoder=ObservationEncoder(builder.schema)
    store=RawMarketStore.precompute(builder,range(4,9),chunk_rows=2)
    return runtime,factors,builder,encoder,store


def test_transport_is_lossless_account_and_local_ref_only():
    runtime,_,builder,encoder,store=fixture_raw()
    obs=builder.build(5,sample_account(runtime))
    encoded=encoder.encode(obs,store=store)
    schema=encoder.output_schema
    assert schema.dimension==1+3*4+4+4*15
    assert encoded[0]==4
    np.testing.assert_array_equal(encoded[schema.position_slice].reshape(3,4),obs.position_panel)
    np.testing.assert_array_equal(encoded[schema.portfolio_slice],obs.portfolio)
    np.testing.assert_array_equal(encoded[schema.history_slice].reshape(4,15),obs.policy_history)
    assert len(schema.feature_names)==schema.dimension
    assert EncodedObservationSchema.from_dict(schema.to_dict())==schema
    assert schema.feature_names[0]=="transport.raw_row_ref"


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_account_encoding_owns_output_and_keeps_input_dtype_coercion(dtype):
    runtime,_,builder,encoder,store=fixture_raw()
    account=builder.build_account(5,sample_account(runtime))
    account=replace(account,**{name:getattr(account,name).astype(dtype)
        for name in ("position_panel","portfolio","policy_history")})
    expected=np.concatenate(([store.row_reference(account.decision_date)],
        account.position_panel.ravel(),account.portfolio,account.policy_history.ravel())).astype(np.float32)
    first=encoder.encode_account(account,store)
    second=encoder.encode_account(account,store)
    np.testing.assert_array_equal(first,expected)
    assert first.dtype==np.float32 and first.flags.owndata
    assert not np.shares_memory(first,second)
    for name in ("position_panel","portfolio","policy_history"):
        assert not np.shares_memory(first,getattr(account,name))
    first[:]=-99
    np.testing.assert_array_equal(second,expected)
    np.testing.assert_array_equal(encoder.encode_account(account,store),expected)


def test_raw_store_windows_match_same_live_observation_path():
    runtime,_,builder,encoder,store=fixture_raw()
    for day in range(4,9):
        obs=builder.build(day,sample_account(runtime,day))
        raw,pit,valid=store.window(store.row_reference(obs.decision_date))
        np.testing.assert_array_equal(raw,obs.stock_panel)
        np.testing.assert_array_equal(pit,obs.pit_universe_mask)
        np.testing.assert_array_equal(valid,obs.time_mask)
        live=RawMarketStore.from_observation(obs,encoder)
        compact=encoder.encode(obs,store=live)
        live_raw,live_pit,live_valid=live.window(int(compact[0]))
        np.testing.assert_array_equal(live_raw,raw)
        np.testing.assert_array_equal(live_pit,pit)
        np.testing.assert_array_equal(live_valid,valid)
    with pytest.raises(ValueError,match="outside"):
        store.window(store.decision_start-1)
    with pytest.raises(ValueError,match="outside"):
        store.window(store.decision_stop)
    with pytest.raises(ValueError,match="outside"):
        store.window(float(store.decision_start))


def test_raw_store_preserves_early_padding_and_readonly_arrays():
    runtime,factors=synthetic_inputs()
    builder=make_builder(runtime,factors,lookback=12)
    store=RawMarketStore.precompute(builder,range(0,5))
    raw,pit,valid=store.window(0)
    np.testing.assert_array_equal(raw,builder.build_static(0).stock_panel)
    assert valid.tolist()==[False]*11+[True]
    assert not pit[:-1].any()
    for array in (store.raw_rows,store.pit_universe_mask,store.row_valid):
        assert not array.flags.writeable


@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
def test_raw_store_rejects_undeclared_missing_and_nonfinite(bad):
    *_,store=fixture_raw()
    raw=store.raw_rows.copy()
    raw[0,0,0]=bad
    with pytest.raises(ValueError,match="sentinel"):
        replace(store,raw_rows=raw)


def test_signed_raw_financial_values_and_reserved_missing_survive_transport():
    *_,store=fixture_raw()
    raw=store.raw_rows.copy()
    column=store.schema.stock_feature_names.index(RAW_FINANCIAL_VALUE_NAMES[0])
    raw[0,:,column]=[-0.5,-2.0,RAW_MISSING_VALUE]
    restored=replace(store,raw_rows=raw)
    np.testing.assert_array_equal(restored.raw_rows[0,:,column],raw[0,:,column])


def test_normalizer_only_training_per_field_no_center_clip_or_stock_scales(tmp_path):
    *_,encoder,store=fixture_raw()
    norm=TrainOnlyNormalizer.fit(store,encoder.output_schema,dataset_role="train",initial_cash=1e6)
    assert norm.stock_scale.shape==(37,)
    assert norm.position_scale.shape==(4,)
    assert norm.portfolio_scale.tolist()==[1e6,1e6,1e6,1]
    assert norm.history_scale.shape==(15,)
    assert not hasattr(norm,"mean") and not hasattr(norm,"clip") and not hasattr(norm,"transform")
    for i,name in enumerate(encoder.output_schema.stock_feature_names):
        if name in ("st_mask","price_buy_allowed","price_sell_allowed"):
            assert norm.stock_scale[i]==1
    expected=np.sqrt(np.square(store.raw_rows[store.decision_start:store.decision_stop,:,0].astype(np.float64)).mean())
    assert norm.stock_scale[0]==pytest.approx(expected)
    with pytest.raises(ValueError,match="sealed train"):
        TrainOnlyNormalizer.fit(store,encoder.output_schema,dataset_role="validation",initial_cash=1e6)
    path=tmp_path/"normalizer.json"
    norm.save(path)
    restored=TrainOnlyNormalizer.load(path,expected_schema=encoder.output_schema)
    assert restored.state_hash==norm.state_hash
    payload=json.loads(path.read_text())
    payload["stock_scale"][0]+=1
    with pytest.raises(ValueError,match="hash"):
        TrainOnlyNormalizer.from_dict(payload)


def test_normalizer_excludes_pretraining_rows_missing_and_future_nonmembers():
    runtime,factors=synthetic_inputs(date_count=10)
    runtime.data["listing_age"][:,2]=-1
    runtime.data["open"][:4]=99999
    runtime.data["open"][4:,2]=1e20
    runtime.data["amount"][:]=np.nan
    runtime.data["volume"][:]=0
    builder=make_builder(runtime,factors,lookback=4)
    encoder=ObservationEncoder(builder.schema)
    store=RawMarketStore.precompute(builder,range(4,9))
    norm=TrainOnlyNormalizer.fit(store,encoder.output_schema,dataset_role="train",initial_cash=1e6)
    expected=np.sqrt(np.square(runtime.data["open"][4:9,:2].astype(np.float64)).mean())
    assert norm.stock_scale[builder.schema.stock_feature_names.index("open")]==pytest.approx(expected)
    assert norm.stock_scale[builder.schema.stock_feature_names.index("amount_lag1")]==1
    assert norm.stock_scale[builder.schema.stock_feature_names.index("volume_lag1")]==1


def test_compact_shared_store_references_and_raw_rows_remain_identical(tmp_path):
    episode=build_episode(tmp_path/"runtime.npz")
    compact=episode.compact_for_replay()
    assert compact.market_store.decision_start==episode.market_store.decision_start
    assert compact.market_store.decision_stop==episode.market_store.decision_stop
    np.testing.assert_array_equal(compact.market_store.raw_rows,episode.market_store.raw_rows)
    assert compact.encoder.output_schema.identifier==episode.encoder.output_schema.identifier
    with SharedPreparedEpisodeOwner.create(episode) as owner:
        assert owner.descriptor.raw_encoder_schema==episode.encoder.output_schema.identifier
        with owner.descriptor.attach() as attached:
            shared=attached.episode
            assert shared.market_store.decision_start==episode.market_store.decision_start
            np.testing.assert_array_equal(shared.market_store.raw_rows,episode.market_store.raw_rows)
            assert not shared.market_store.raw_rows.flags.writeable
            for date in episode.market_store.decision_dates:
                assert shared.market_store.row_reference(date)==episode.market_store.row_reference(date)
            assert fit_normalizer(shared).state_hash==fit_normalizer(episode).state_hash


def test_raw_store_size_is_rows_not_overlapping_windows():
    runtime,_,builder,encoder,store=fixture_raw()
    assert store.raw_rows.shape==(8,3,37)
    assert encoder.output_dimension<4*3*37
    assert not hasattr(store,"market_encodings")


def test_future_listings_do_not_change_raw_members_or_field_scales():
    runtime,factors=synthetic_inputs(date_count=10)
    extended,extended_factors=synthetic_inputs(date_count=10,
        stock_codes=runtime.stock_codes+("600099.SH",))
    for name,array in runtime.data.items():
        extended.data[name][...,:3]=array
    extended.data["listing_age"][:,-1]=-1
    for name in ("ranks","validity","filters"):
        getattr(extended_factors,name)[...,:3]=getattr(factors,name)
    base_builder=make_builder(runtime,factors)
    extra_builder=make_builder(extended,extended_factors)
    base=RawMarketStore.precompute(base_builder,range(4,9))
    extra=RawMarketStore.precompute(extra_builder,range(4,9))
    np.testing.assert_array_equal(base.raw_rows,extra.raw_rows[:,:3])
    assert not extra.raw_rows[:,-1].any()
    normalizers=[TrainOnlyNormalizer.fit(store,ObservationEncoder(store.schema).output_schema,
        dataset_role="train",initial_cash=1e6) for store in (base,extra)]
    for field in ("stock_scale","position_scale","portfolio_scale","history_scale"):
        np.testing.assert_array_equal(getattr(normalizers[0],field),getattr(normalizers[1],field))


def test_prepared_episode_rejects_misbound_store_schema_and_dates(tmp_path):
    episode=build_episode(tmp_path/"runtime.npz")
    store=episode.market_store
    with pytest.raises(ValueError,match="schemas differ"):
        replace(episode,market_store=replace(store,schema=replace(store.schema,action_schema_hash="f"*64)))
    with pytest.raises(ValueError,match="calendar"):
        replace(episode,market_store=replace(store,decision_dates=store.decision_dates[::-1]))

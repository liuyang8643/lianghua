"""Raw sequence learning, causal routing and full-stock attention invariants."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
import torch as th

from ai.rl.raw_features import RAW_PANEL_CONFIG, RawPanelFeatures, TemporalPatchEncoder, _WindowMemberIndex
from env.encoder import ObservationEncoder
from env.observation import RAW_MISSING_VALUE
from env.gym_adapter import WBRGymEnv
from rl_test_data import build_episode, fit_normalizer


def _readonly(values):
    values = np.ascontiguousarray(values)
    values.flags.writeable = False
    return values


@pytest.mark.parametrize("span", [1, 3, 21, 504])
@pytest.mark.parametrize("stocks", [0, 5])
def test_window_member_index_matches_direct_causal_windows(span, stocks):
    member = np.random.default_rng(59).random((9, stocks)) > 0.6
    member[0:3] = False
    prefix = np.vstack((np.zeros((1, stocks), dtype=np.int32),
                        np.cumsum(member, axis=0, dtype=np.int32)))
    state_cpu, state_cuda = th.get_rng_state(), th.cuda.get_rng_state()
    index = _WindowMemberIndex().cuda()
    index.bind(prefix, span, th.device("cuda"))
    assert index.offsets.dtype == np.int64 and len(index.offsets) == len(member) + 2
    assert not index.offsets.flags.writeable
    assert not list(index.parameters()) and not index.state_dict()
    assert th.equal(state_cpu, th.get_rng_state()) and th.equal(state_cuda, th.cuda.get_rng_state())
    index.to(dtype=th.float64)
    assert index.membership.dtype == th.bool and index.stocks.dtype == th.int64
    for ends in (np.array([], dtype=np.int64), np.array([-100, -1, 0, 2]),
                 np.array([8, 4, -1, 0, 4, 2, 8])):
        expected = np.zeros((len(ends), stocks), dtype=bool)
        for row, end in enumerate(ends):
            if end >= 0:
                expected[row] = member[max(0, end - span + 1):end + 1].any(axis=0)
        dates, symbols = index.pairs(ends)
        expected_dates, expected_symbols = np.nonzero(expected)
        np.testing.assert_array_equal(dates.cpu(), expected_dates)
        np.testing.assert_array_equal(symbols.cpu(), expected_symbols)
        np.testing.assert_array_equal(index.mask(ends).cpu(), expected)
    with pytest.raises(IndexError):
        index.pairs(np.array([len(member)]))
    member[:] = False
    index.bind(np.zeros_like(prefix), span, th.device("cuda"))
    assert index.pairs(np.array([8, 0, -1]))[0].numel() == 0
    assert not index.mask(np.array([8, 0, -1])).any()


def test_all_padding_temporal_sequence_has_zero_features_and_gradients():
    temporal = TemporalPatchEncoder(3, 42, RAW_PANEL_CONFIG)
    rows = th.randn(5, 42, 3, requires_grad=True)
    valid = th.zeros((5, 42), dtype=th.bool)
    valid[-1] = True
    result = temporal(rows, valid)
    assert th.count_nonzero(result[:4]) == 0
    result[:4].sum().backward()
    assert th.count_nonzero(rows.grad) == 0
    for parameter in temporal.parameters():
        assert parameter.grad is not None and th.count_nonzero(parameter.grad) == 0


@pytest.fixture
def raw_inputs(tmp_path):
    th.manual_seed(493)
    episode = build_episode(tmp_path / "runtime.npz")
    normalizer = fit_normalizer(episode)
    env = WBRGymEnv(episode, normalizer=normalizer)
    public, _ = env.reset()
    # Give the account branch one held stock and real action history.
    schema = episode.encoder.output_schema.to_dict()
    public[1:5] = (100.0, 10.0, 100.0, 10.0)
    history_start = 1 + schema["stock_count"] * len(schema["position_feature_names"]) + len(schema["portfolio_feature_names"])
    history = public[history_start:].reshape(schema["lookback"], -1)
    history[-3:, 0] = 1
    history[-3:, 1:13] = 0.2
    history[-3:, 13:] = 0.01
    return episode, normalizer, th.tensor(public[None], device="cuda")


def _network(schema, store, normalizer, *, sequence_batch_size=2):
    model = RawPanelFeatures(schema, dict(RAW_PANEL_CONFIG, sequence_batch_size=sequence_batch_size)).cuda()
    model.bind_market_store(store, normalizer)
    return model


def test_future_dates_cannot_change_current_features_but_past_raw_values_can(raw_inputs):
    episode, normalizer, public = raw_inputs
    schema = episode.encoder.output_schema.to_dict()
    store = episode.market_store
    model = _network(schema, store, normalizer)
    model.eval()
    reference = int(public[0, 0])
    with th.no_grad():
        expected = model(public)
        future_rows = store.raw_rows.copy()
        future_rows[reference + 1:] += 100_000 * store.pit_universe_mask[reference + 1:, :, None]
        model.bind_market_store(replace(store, raw_rows=_readonly(future_rows)), normalizer)
        th.testing.assert_close(model(public), expected, rtol=0, atol=0)
        past_rows = store.raw_rows.copy()
        past_rows[reference - 2, 0, 0] += float(normalizer.stock_scale[0]) * 100
        model.bind_market_store(replace(store, raw_rows=_readonly(past_rows)), normalizer)
        assert not th.equal(model(public), expected)


def test_future_stocks_and_stock_permutation_preserve_values_and_parameter_gradients(raw_inputs):
    episode, normalizer, public = raw_inputs
    original_schema = episode.encoder.output_schema.to_dict()
    store = episode.market_store
    original = _network(original_schema, store, normalizer)
    expected = original(public)
    objective_weights = th.arange(expected.shape[-1], dtype=expected.dtype, device="cuda")
    (expected * objective_weights).sum().backward()
    expected_gradients = {name: value.grad.clone() for name, value in original.named_parameters() if value.grad is not None}
    stocks = original_schema["stock_count"]
    width = len(original_schema["position_feature_names"])
    for order in (np.arange(stocks + 1), np.arange(stocks - 1, -1, -1)):
        appending = len(order) > stocks
        rows = (np.concatenate((store.raw_rows, np.zeros_like(store.raw_rows[:, :1])), axis=1)
                if appending else store.raw_rows[:, order])
        membership = (np.concatenate((store.pit_universe_mask, np.zeros_like(store.pit_universe_mask[:, :1])), axis=1)
                      if appending else store.pit_universe_mask[:, order])
        observation_schema = replace(store.schema, stock_count=len(order),
            stock_codes_hash=sha256(order.tobytes()).hexdigest())
        encoded = ObservationEncoder(observation_schema).output_schema
        changed_store = replace(store, schema=observation_schema,
            raw_rows=_readonly(rows), pit_universe_mask=_readonly(membership))
        changed_normalizer = replace(normalizer, encoder_schema=encoded.identifier)
        positions = public[:, 1:1 + stocks * width].reshape(1, stocks, width)
        positions = (th.cat((positions, th.zeros_like(positions[:, :1])), dim=1)
                     if appending else positions[:, order.copy()])
        changed_public = th.cat((public[:, :1], positions.flatten(1), public[:, 1 + stocks * width:]), dim=1)
        changed = _network(encoded.to_dict(), changed_store, changed_normalizer)
        changed.load_state_dict(original.state_dict())
        actual = changed(changed_public)
        th.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        (actual * objective_weights).sum().backward()
        for name, value in changed.named_parameters():
            if name in expected_gradients:
                th.testing.assert_close(value.grad, expected_gradients[name], rtol=3e-4, atol=3e-6)


def test_chunking_and_duplicate_dates_preserve_forward_backward_without_embedding_cache(raw_inputs, monkeypatch):
    episode, normalizer, public = raw_inputs
    schema = episode.encoder.output_schema.to_dict()
    small = _network(schema, episode.market_store, normalizer, sequence_batch_size=1)
    whole = _network(schema, episode.market_store, normalizer, sequence_batch_size=schema["stock_count"])
    whole.load_state_dict(small.state_dict())
    seen = []
    resolve = small._project_chunk
    monkeypatch.setattr(small, "_project_chunk", lambda *args: (seen.append(1), resolve(*args))[1])
    samples = public.repeat(3, 1).requires_grad_()
    actual, expected = small(samples), whole(samples)
    forward_calls = len(seen)
    assert forward_calls > 0
    th.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    (actual * th.arange(actual.shape[-1], dtype=actual.dtype, device="cuda")).sum().backward()
    assert len(seen) > forward_calls  # Shared patch projection is recomputed in backward.
    assert samples.grad[:, 0].eq(0).all()
    assert th.isfinite(samples.grad).all()
    assert small.market_temporal.projection.weight.grad.abs().sum() > 0
    optimizer = th.optim.SGD(small.parameters(), lr=0.1)
    optimizer.step()
    assert not th.equal(small(public), actual[:1])


@pytest.mark.parametrize("reference", [-1.0, 0.5, float("nan"), 1e7])
def test_raw_routing_requires_bound_sealed_integer_reference(raw_inputs, reference):
    episode, normalizer, public = raw_inputs
    model = RawPanelFeatures(episode.encoder.output_schema.to_dict(), RAW_PANEL_CONFIG).cuda()
    with pytest.raises(RuntimeError, match="bind_market_store"):
        model(public)
    model.bind_market_store(episode.market_store, normalizer)
    invalid = public.clone()
    invalid[:, 0] = reference
    with pytest.raises(ValueError, match="reference"):
        model(invalid)


def test_missing_sentinel_and_true_zero_reach_distinct_patch_inputs(raw_inputs, monkeypatch):
    episode, normalizer, public = raw_inputs
    store = episode.market_store
    reference = int(public[0, 0])
    rows = store.raw_rows.copy()
    rows[reference, :2, 0] = (RAW_MISSING_VALUE, 0)
    model = _network(episode.encoder.output_schema.to_dict(),
        replace(store, raw_rows=_readonly(rows)), normalizer)
    assert model.raw_bank[reference, 0, 0] == -1
    assert model.raw_bank[reference, 1, 0] == 0



def test_checkpoint_backward_keeps_the_forward_store_snapshot(raw_inputs):
    episode, normalizer, public = raw_inputs
    schema = episode.encoder.output_schema.to_dict()
    expected = _network(schema, episode.market_store, normalizer)
    actual = _network(schema, episode.market_store, normalizer)
    actual.load_state_dict(expected.state_dict())
    # Both retained and recomputed sequence chunks must keep the original
    # forward's raw data after the module is rebound.
    quota = 2 * expected.market_temporal.patch_count * expected.width * 4
    expected.config['retained_token_input_bytes'] = quota
    actual.config['retained_token_input_bytes'] = quota
    first, second = expected(public), actual(public)
    modified = episode.market_store.raw_rows.copy()
    modified += 1000 * episode.market_store.pit_universe_mask[:, :, None]
    membership = episode.market_store.pit_universe_mask.copy()
    membership[:, ::2] = False
    modified[~membership] = 0
    actual.bind_market_store(replace(episode.market_store, raw_rows=_readonly(modified),
        pit_universe_mask=_readonly(membership)), normalizer)
    weights = th.arange(first.shape[-1], dtype=first.dtype, device="cuda")
    (first * weights).sum().backward()
    (second * weights).sum().backward()
    for (_, left), (_, right) in zip(expected.named_parameters(), actual.named_parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None
        else:
            th.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize('lookback', [1, 22, 504])
@pytest.mark.parametrize('quota_offset', [-1, 0, 1])
def test_retained_sequence_budget_preserves_double_gradients_and_adam(
        raw_inputs, monkeypatch, lookback, quota_offset):
    import ai.rl.raw_features as raw_module

    episode, normalizer, public = raw_inputs
    schema = replace(episode.market_store.schema, lookback=lookback)
    encoded = ObservationEncoder(schema).output_schema
    store = replace(episode.market_store, schema=schema)
    normalizer = replace(normalizer, encoder_schema=encoded.identifier)
    original = _network(encoded.to_dict(), store, normalizer)
    retained = _network(encoded.to_dict(), store, normalizer)
    retained.load_state_dict(original.state_dict())
    original.config['retained_token_input_bytes'] = 1
    chunk_bytes = 2 * retained.market_temporal.patch_count * retained.width * 4
    retained.config['retained_token_input_bytes'] = chunk_bytes + quota_offset
    opts = [th.optim.Adam(m.market_temporal.parameters(), lr=1e-4) for m in (original, retained)]
    reference = np.asarray([int(public[0, 0])])
    seen = []
    checkpoint = raw_module.checkpoint

    def recorded(function, **kwargs):
        if getattr(function.func, '__self__', None) is retained.market_temporal:
            seen.append(function.args[0].shape[0])
        return checkpoint(function, **kwargs)

    monkeypatch.setattr(raw_module, 'checkpoint', recorded)
    for _ in range(2):
        rng_cpu, rng_cuda = th.get_rng_state(), th.cuda.get_rng_state()
        results = []
        for model, optimizer in zip((original, retained), opts):
            optimizer.zero_grad(set_to_none=True)
            seen.clear()
            output = model._market_many(reference)
            if model is retained:
                # Five stocks give 2/2/1 sequences; retain only a full prefix.
                assert seen == ([2, 2, 1] if quota_offset < 0 else [2, 1])
            params = tuple(model.market_temporal.parameters())
            weights = th.arange(model.width, device='cuda')
            first = th.autograd.grad((output * weights).sum(), params, retain_graph=True)
            second = th.autograd.grad(output.square().sum(), params)
            for param, a, b in zip(params, first, second):
                param.grad = a + b
            results.append((output.detach(), first, second))
            optimizer.step()
        for left, right in zip(results[0], results[1]):
            th.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(original.parameters(), retained.parameters()):
            th.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(opts[0].state.values(), opts[1].state.values()):
            th.testing.assert_close(left, right, rtol=0, atol=0)
        assert th.equal(rng_cpu, th.get_rng_state())
        assert th.equal(rng_cuda, th.cuda.get_rng_state())


def test_raw_device_cache_is_input_only_and_never_serialized(raw_inputs, tmp_path):
    episode, normalizer, public = raw_inputs
    model = _network(episode.encoder.output_schema.to_dict(), episode.market_store, normalizer)
    for name, expected in (("raw_bank", episode.market_store.raw_rows),
                           ("pit_bank", episode.market_store.pit_universe_mask),
                           ("row_valid_bank", episode.market_store.row_valid)):
        value = getattr(model, name)
        if name == "raw_bank":
            missing = expected == RAW_MISSING_VALUE
            expected = np.where(missing, -1.0, expected / normalizer.stock_scale - ((expected < 0) & ~missing) * 2)
            expected = expected * (episode.market_store.pit_universe_mask & episode.market_store.row_valid[:, None])[..., None]
        np.testing.assert_allclose(value.cpu().numpy(), expected, rtol=1e-6, atol=1e-7)
        assert not value.requires_grad
        assert name not in model.state_dict()
    path = tmp_path / "raw_network.pt"
    th.save(model.state_dict(), path)
    restored = RawPanelFeatures(episode.encoder.output_schema.to_dict(), model.config).cuda()
    restored.load_state_dict(th.load(path, weights_only=True))
    assert restored.raw_bank.numel() == restored.pit_bank.numel() == restored.row_valid_bank.numel() == 0
    with pytest.raises(RuntimeError, match="bind_market_store"):
        restored(public)
    restored.bind_market_store(episode.market_store, normalizer)
    th.testing.assert_close(restored(public), model(public), rtol=0, atol=0)


def test_raw_public_model_rejects_cpu_execution(raw_inputs):
    episode, normalizer, public = raw_inputs
    model = RawPanelFeatures(episode.encoder.output_schema.to_dict(), RAW_PANEL_CONFIG)
    with pytest.raises(ValueError, match="CUDA"):
        model.bind_market_store(episode.market_store, normalizer)
    model.cuda().bind_market_store(episode.market_store, normalizer)
    with pytest.raises(ValueError, match="inputs"):
        model(public.cpu())


def test_frozen_market_reuse_is_invalidated_by_updates_load_and_binding(raw_inputs):
    episode, normalizer, public = raw_inputs
    model = _network(episode.encoder.output_schema.to_dict(), episode.market_store, normalizer)
    model.eval()
    with th.no_grad():
        before = model(public)
        bank = model.prepare_frozen_market()
        key = model.frozen_market_token
        assert model.prepare_frozen_market() is bank
        th.testing.assert_close(model(public), before, rtol=0, atol=0)
    assert "_frozen_market" not in model.state_dict()
    model.train()
    assert model._frozen_market.numel() == 0
    optimizer = th.optim.SGD(model.parameters(), lr=0.01)
    (model(public) * th.arange(model.width, device="cuda")).sum().backward()
    assert model.market_temporal.projection.weight.grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    with th.no_grad():
        after = model(public)
    assert model.frozen_market_token != key
    assert not th.equal(before, after)
    key = model.frozen_market_token
    model.load_state_dict(model.state_dict())
    with th.no_grad():
        th.testing.assert_close(model(public), after, rtol=0, atol=0)
    assert model.frozen_market_token != key
    model.bind_market_store(episode.market_store, normalizer)
    assert model._frozen_market.numel() == 0
    with th.no_grad():
        th.testing.assert_close(model(public), after, rtol=0, atol=0)


@pytest.mark.parametrize("lookback", [1, 20, 21, 22, 63, 64, 504])
def test_shared_patch_algebra_matches_direct_windows_and_gradients(raw_inputs, lookback):
    episode, normalizer, public = raw_inputs
    store = episode.market_store
    observation_schema = replace(store.schema, lookback=lookback)
    encoded = ObservationEncoder(observation_schema).output_schema
    store = replace(store, schema=observation_schema)
    normalizer = replace(normalizer, encoder_schema=encoded.identifier)
    model = _network(encoded.to_dict(), store, normalizer, sequence_batch_size=16)
    reference = int(public[0, 0])
    refs = np.asarray([reference, reference + 1, reference + 2])
    actual = model._market_many(refs)
    objective = th.arange(model.width, device="cuda", dtype=actual.dtype)
    (actual * objective).sum().backward()
    gradients = {n: v.grad.clone() for n, v in model.market_temporal.named_parameters()}
    model.zero_grad(set_to_none=True)
    expected = []
    for ref in refs:
        raw, member, row_valid = store.window(int(ref))
        rows = th.tensor(raw, device="cuda").permute(1, 0, 2)
        missing = rows == float(RAW_MISSING_VALUE)
        rows = th.where(missing, -1.0, rows / model.stock_scale - ((rows < 0) & ~missing) * 2)
        rows = rows[..., model.historical_columns]
        valid = th.tensor(member & row_valid[:, None], device="cuda").T
        expected.append(model.market_temporal(rows, valid))
    expected = th.stack(expected)
    th.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    (expected * objective).sum().backward()
    for n, v in model.market_temporal.named_parameters():
        th.testing.assert_close(v.grad, gradients[n], rtol=3e-3, atol=1e-4)

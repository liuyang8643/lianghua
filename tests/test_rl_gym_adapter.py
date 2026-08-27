import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ai.bundle import (
    DEPLOYMENT_GATE_NAMES,
    BundleManifest,
    file_sha256,
    policy_source_sha256,
)
from ai.rl.policy import RLPolicy
from env.action_schema import ActionSchema
from env.backtest import run_episode
from env.contracts import AccountState
from env.encoder import TrainOnlyNormalizer
from env.gym_adapter import (
    PreparedEpisode,
    WBRGymEnv,
    environment_schema_manifest,
    fixed_config_action,
)
from factor import precompute_factors
from offline_data import compute_runtime_lineage, load_runtime_slice


ROOT = Path(__file__).resolve().parents[1]


def _runtime_npz(path: Path) -> None:
    dates = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-06-29"),
        dtype="datetime64[D]",
    )
    codes = np.asarray(("600001.SH", "000001.SZ", "300001.SZ", "688001.SH"))
    day = np.arange(len(dates), dtype=np.float64)[:, None]
    stock = np.arange(len(codes), dtype=np.float64)[None, :]
    close = 10.0 + day * 0.01 + stock
    open_prices = close * (1.0 + 0.001 * (stock - 1.5))
    preclose = np.empty_like(close)
    preclose[0] = close[0]
    preclose[1:] = close[:-1]
    volume = 1_000_000.0 + day * 1_000.0 + stock * 100.0
    amount = volume * close
    panel = np.broadcast_to(1.0 + day * 0.001 + stock * 0.01, close.shape).copy()
    np.savez_compressed(
        path,
        stock_codes=codes,
        trade_dates=dates,
        open=open_prices,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=volume,
        amount=amount,
        preClose=preclose,
        issue_price=np.full(len(codes), 8.0),
        stock_names=np.asarray(("A", "B", "C", "D")),
        st_mask=np.zeros(close.shape, dtype=np.bool_),
        total_share=np.broadcast_to(
            np.asarray((1e9, 8e8, 5e8, 3e8))[None, :], close.shape
        ),
        bps=panel,
        eps=panel,
        roe=panel,
        profit_yoy=panel,
        revenue_yoy=panel,
        operating_cf_ps=panel,
        gross_margin=panel,
    )


@pytest.fixture
def prepared_episode(tmp_path: Path) -> PreparedEpisode:
    path = tmp_path / "runtime.npz"
    _runtime_npz(path)
    runtime = load_runtime_slice(path, "2020-06-20", "2020-06-24")
    factors = precompute_factors(runtime)
    return PreparedEpisode.build(runtime, factors, lookback=64)


def test_gym_adapter_passes_sb3_checker(prepared_episode: PreparedEpisode):
    env = WBRGymEnv(prepared_episode)
    check_env(env, warn=True)


def test_episode_has_d_observations_and_exactly_d_minus_one_actions(
    prepared_episode: PreparedEpisode,
):
    schema = ActionSchema()
    payload = json.loads((ROOT / "configs" / "config.json").read_text("utf-8"))
    action = fixed_config_action(schema, schema.from_static_config(payload))
    env = WBRGymEnv(prepared_episode, action_schema=schema)

    observation, info = env.reset(seed=20260827)
    assert observation.shape == env.observation_space.shape
    assert info["decision_date"] == "2020-06-20"
    transitions = 0
    terminated = False
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(action)
        transitions += 1
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert truncated is False

    assert transitions == prepared_episode.transition_count == 4
    assert info["next_decision_date"] == "2020-06-24"
    with pytest.raises(RuntimeError, match="after termination"):
        env.step(action)


def test_shared_rollout_uses_the_same_sealed_gym_path(prepared_episode: PreparedEpisode):
    schema = ActionSchema()
    payload = json.loads((ROOT / "configs" / "config.json").read_text("utf-8"))
    action = schema.encode_static_config(payload)
    trace = run_episode(
        WBRGymEnv(prepared_episode, action_schema=schema),
        lambda observation: action,
        seed=7,
    )

    assert len(trace.rewards) == prepared_episode.transition_count
    assert trace.nav[-1] == pytest.approx(trace.nav[0] * np.exp(trace.rewards.sum()))
    assert np.isfinite(trace.metrics.calmar)


def test_real_sb3_bundle_roundtrips_through_strict_policy_loader(
    prepared_episode: PreparedEpisode,
    tmp_path: Path,
):
    schema = ActionSchema()
    payload = json.loads((ROOT / "configs" / "config.json").read_text("utf-8"))
    action = schema.encode_static_config(payload)
    raw_env = WBRGymEnv(prepared_episode, action_schema=schema)
    observation, _ = raw_env.reset(seed=7)
    samples = [observation]
    terminated = False
    while not terminated:
        observation, _, terminated, _, _ = raw_env.step(action)
        samples.append(observation)
    normalizer = TrainOnlyNormalizer.fit(
        np.stack(samples),
        prepared_episode.encoder.output_schema,
        dataset_role="train",
    )
    model = PPO(
        "MlpPolicy",
        WBRGymEnv(
            prepared_episode,
            action_schema=schema,
            normalizer=normalizer,
        ),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=7,
        device="cpu",
    )
    model.save(tmp_path / "model")
    normalizer.save(tmp_path / "normalizer.json")
    bundled_config = tmp_path / "strategy_config.json"
    bundled_config.write_bytes((ROOT / "configs" / "config.json").read_bytes())
    model_path = tmp_path / "model.zip"
    runtime_manifest = prepared_episode.runtime.manifest.as_dict()
    runtime_manifest["lineage"] = compute_runtime_lineage(
        runtime_manifest["source_path"]
    ).as_dict()
    manifest = BundleManifest(
        created_at="2026-08-27T00:00:00+00:00",
        algorithm="stable_baselines3.PPO",
        model_file=model_path.name,
        model_sha256=file_sha256(model_path),
        normalizer_file="normalizer.json",
        normalizer_sha256=file_sha256(tmp_path / "normalizer.json"),
        config_file=bundled_config.name,
        config_sha256=file_sha256(bundled_config),
        source_sha256=policy_source_sha256(ROOT),
        runtime=runtime_manifest,
        factors={"schema_hash": prepared_episode.factors.schema_hash},
        observation_schema=prepared_episode.observation_builder.schema.to_dict(),
        encoded_schema=prepared_episode.encoder.output_schema.to_dict(),
        action_schema=schema.to_dict(),
        environment=environment_schema_manifest(schema),
        training={
            "convergence": {
                "technical_convergence": True,
                **{name: True for name in DEPLOYMENT_GATE_NAMES},
            }
        },
        evaluation={"train": {}, "validation": {}, "test": {}},
    )
    manifest.save(tmp_path)

    cold = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ai.rl.policy import RLPolicy; "
                f"p=RLPolicy.load(r'{tmp_path}'); "
                "print(p.action_schema.action_dim)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert cold.returncode == 0, cold.stderr
    assert cold.stdout.strip() == str(schema.action_dim)

    loaded = RLPolicy.load(tmp_path)
    raw_observation = prepared_episode.observation_builder.build(
        prepared_episode.decision_start,
        AccountState(cash=1_000_000.0, nav=1_000_000.0, peak_nav=1_000_000.0),
    )
    predicted = loaded.predict_action(raw_observation)

    assert predicted.shape == (schema.action_dim,)
    assert np.isfinite(predicted).all()

    wrong_config = tmp_path / "wrong-config.json"
    wrong_config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="strategy config"):
        RLPolicy.load(tmp_path, config_path=wrong_config)

    manifest_payload = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    manifest_payload["environment"]["planner"]["lot_size"] = 1
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="environment semantics"):
        RLPolicy.load(tmp_path)

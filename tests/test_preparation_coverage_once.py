"""Preparation logging and audit consumers share the same expensive scan."""
import json

import env.backtest as backtest
from rl_test_data import write_runtime


def test_preparation_logs_and_audit_share_one_coverage_scan(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'runtime.npz'
    write_runtime(path)
    original = backtest.factor_coverage
    calls = []

    def counted(runtime, factors):
        calls.append(runtime.manifest.requested_start)
        return original(runtime, factors)

    monkeypatch.setattr(backtest, 'factor_coverage', counted)
    episode = backtest.prepare_episode_from_runtime(
        path, '2020-06-01', '2020-06-10', lookback=4,
        prefilter_n=3, encode_observations=False,
    )
    payload = episode.factor_coverage
    assert episode.factor_coverage is payload
    assert calls == ['2020-06-01']
    emitted = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(emitted) == len(payload['factors'])
    for row in emitted:
        factor = row.pop('factor')
        assert row.pop('event') == 'factor_coverage'
        assert row.pop('start') == payload['start']
        assert row.pop('end') == payload['end']
        assert row == payload['factors'][factor]

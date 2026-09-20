"""Current report contract only; historical conversion belongs to one-off artifacts."""
import json
from pathlib import Path

import pytest

from ai.reporting import (ReportReader, read_json, clean_json, process_alive,
    write_ppo_report, write_ga_report, append_training_diagnostic, append_ppo_update_diagnostic)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def ppo(root, *, curves=None, selected=None, baselines=None):
    run = root / 'run'
    write(run / 'run_identity.json', {'identity_sha256': 'identity', 'contract': {
        'evaluation_protocol': {'test_usage': 'repeated_diagnostic_only'}, 'algorithm': {'n_steps': 8, 'checkpoint_selection': 'validation_calmar',
                      'complete_train_evaluation_every_rollouts': 2},
        'rollout': {'assignments': [{}, {}]},
        'splits': {'train': ['2004-01-01', '2017-12-31'], 'validation': ['2018-01-01', '2022-12-31'],
                   'test': ['2023-01-01', '2026-08-28']}}})
    if selected is not None:
        write(run / 'model.json', selected)
    write_ppo_report(run, total_rollouts=100, timesteps=32,
                     curves=curves or dict(train=[], validation=[], test=[]),
                     baselines=baselines or dict(train=None, validation=None, test=None))
    return run


def point(split, artifact='chosen', calmar=.7, eligible=True):
    return {'timesteps': 16, 'checkpoint_sha256': artifact, f'{split}_metrics': {
        'calmar': calmar, 'annualized_return': .21, 'max_drawdown': .3},
        'eligible_for_selection': eligible, 'run_identity_sha256': 'identity', 'evaluation_elapsed_seconds': 1.2}


def test_json_missing_distinct_from_corrupt():
    assert process_alive(None) is None
    assert process_alive(False) is None
    assert clean_json([float('nan'), float('inf'), 1]) == [None, None, 1]


def test_reader_only_consumes_current_schema_and_does_not_write(tmp_path):
    run = ppo(tmp_path, baselines={'train': {'metrics': {'calmar': .9}}, 'validation': None, 'test': None})
    before = {p: p.read_bytes() for p in run.iterdir()}
    report = ReportReader(tmp_path / 'run', 'ppo', 'PPO', log_path=tmp_path / 'stdout.log').snapshot()
    assert report['progress']['current'] == 2
    assert report['progress']['total'] == 100
    assert report['selection'] is None
    assert report['baseline'] == {'train': {'calmar': .9}}
    assert report['evaluations'] == []
    assert {p: p.read_bytes() for p in run.iterdir()} == before
    payload = read_json(run / 'training_report.json')
    payload['schema_version'] = 'old'
    write(run / 'training_report.json', payload)
    with pytest.raises(ValueError, match='schema'):
        ReportReader(tmp_path / 'run', 'ppo', 'PPO', log_path=tmp_path / 'stdout.log').snapshot()


def test_selection_is_joined_by_checkpoint_not_step_or_score(tmp_path):
    curves = {'train': [], 'validation': [point('validation'), point('validation', 'other', 99, False)],
              'test': [point('test', 'other', 999, True)]}
    ppo(tmp_path, curves=curves, selected={'sha256': 'chosen', 'timesteps': 16})
    report = ReportReader(tmp_path / 'run', 'ppo', 'PPO', log_path=tmp_path / 'stdout.log').snapshot()
    assert len(report['evaluations']) == 3
    assert report['selection']['metrics']['validation']['calmar'] == .7
    assert report['selection']['metrics']['test'] is None
    assert report['evaluations'][-1]['eligible'] is False


def test_incremental_diagnostics_partial_lines_and_extrema(tmp_path, monkeypatch):
    run = ppo(tmp_path)
    for i in range(3000):
        append_training_diagnostic(run, algorithm='PPO', step=i, timesteps=i * 16,
            scalars={'loss': {'label': 'Loss', 'value': 999 if i == 777 else -999 if i == 888 else i % 3}}, details={})
    reader = ReportReader(tmp_path / 'run', 'ppo', 'PPO', log_path=tmp_path / 'stdout.log')
    points = reader.snapshot()['diagnostics']['loss']['points']
    assert len(points) <= 1000
    assert [777, 999] in points and [888, -999] in points
    assert points[0][0] == 0 and points[-1][0] == 2999
    path = run / 'training_diagnostics.jsonl'
    original = Path.open
    def guarded(self, *args, **kwargs):
        if self == path or self == run / 'training_report.json':
            pytest.fail('unchanged input must be cached')
        return original(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', guarded)
        reader.snapshot()
    with path.open('a') as stream:
        stream.write('{"schema_version":')
    assert reader.snapshot()['diagnostics']['loss']['points'] == points
    with path.open('a') as stream:
        stream.write('"training-diagnostics-v3","algorithm":"PPO","step":3000,"timesteps":48000,"scalars":{"loss":{"label":"Loss","value":4}},"details":{}}\n')
    assert reader.snapshot()['diagnostics']['loss']['points'][-1] == [3000, 4]


def test_bad_json_and_unexpected_diagnostic_version_fail(tmp_path):
    run = ppo(tmp_path)
    (run / 'training_report.json').write_text('broken')
    with pytest.raises(json.JSONDecodeError):
        ReportReader(tmp_path / 'run', 'ppo', 'PPO', log_path=tmp_path / 'stdout.log').snapshot()
    ppo(tmp_path)
    (run / 'training_diagnostics.jsonl').write_text('{"schema_version":"old","algorithm":"PPO"}\n')
    with pytest.raises(ValueError, match='schema'):
        ReportReader(tmp_path / 'run', 'ppo', 'PPO', log_path=tmp_path / 'stdout.log').snapshot()


def test_ga_train_only_uses_same_schema_and_ratio_metrics(tmp_path):
    run = tmp_path / 'run'
    write(run / 'run_metadata.json', {'seed': 1, 'decision_start': '2004', 'decision_end': '2017', 'objective': 'calmar'})
    best = {'individual_config': {'weights': {'factor': .5}, 'buy_n': 50, 'turnover_rate': .173},
            'metrics': {'calmar': 1.5, 'annualized_return': .3, 'max_drawdown': .2}}
    write_ga_report(run, total_generations=10, generation=1, best=best)
    write_ga_report(run, total_generations=10, generation=2, best=best)
    write_ga_report(run, total_generations=10, generation=2, best=best, complete=True)
    report = ReportReader(tmp_path / 'run', 'ga', 'GA', log_path=tmp_path / 'stdout.log').snapshot()
    assert report['schema_version'] == 'training-report-v2'
    assert len(report['evaluations']) == 2
    assert report['evaluations'][0]['metrics'] == best['metrics']
    assert report['selection'] is None
    assert report['state'] == 'complete'
    assert report['actions'][0]['controls']['turnover_rate'] == .173


def test_update_diagnostic_undefined_metric_is_explicit_null(tmp_path):
    run = ppo(tmp_path)
    append_ppo_update_diagnostic(run, {'timesteps': 16, 'train': {'train/loss': .5, 'train/explained_variance': float('nan')},
                                     'rollout_quantiles': {'rewards': [-1, -.5, 0, .5, 1]}}, 16)
    report = ReportReader(tmp_path / 'run', 'ppo', 'PPO', log_path=tmp_path / 'stdout.log').snapshot()
    assert report['diagnostics']['train/loss']['points'] == [[1, .5]]
    assert 'train/explained_variance' not in report['diagnostics']
    assert report['diagnostic_records'][0]['scalars']['train/explained_variance']['value'] is None


def test_report_path_is_explicit_even_when_nested_run_exists(tmp_path):
    ppo(tmp_path)
    with pytest.raises(FileNotFoundError):
        ReportReader(tmp_path, 'ppo', 'PPO', log_path=tmp_path / 'stdout.log').snapshot()


def test_frontend_only_references_existing_controls():
    import re
    assets = Path(__file__).resolve().parents[1] / 'ai' / 'report_assets'
    html = (assets / 'index.html').read_text(encoding='utf-8')
    script = (assets / 'app.js').read_text(encoding='utf-8')
    ids = set(re.findall(r'id="([^"]+)"', html))
    references = set(re.findall(r"\$\('([^']+)'\)", script))
    assert references <= ids


def test_unchanged_diagnostics_do_not_resample_and_reset_rebuilds(tmp_path, monkeypatch):
    import ai.reporting as reporting
    run = ppo(tmp_path)
    append_training_diagnostic(run, algorithm='PPO', step=1, timesteps=16,
        scalars={'loss': {'label': 'Loss', 'value': 2}}, details={})
    reader = ReportReader(run, 'ppo', 'PPO', log_path=tmp_path / 'stdout.log')
    assert reader.snapshot()['diagnostics']['loss']['points'] == [[1, 2]]
    with monkeypatch.context() as patch:
        patch.setattr(reporting, '_sample', lambda *a: pytest.fail('unchanged series resampled'))
        reader.snapshot()
    path = run / 'training_diagnostics.jsonl'
    path.unlink()
    assert reader.snapshot()['diagnostics'] == {}
    append_training_diagnostic(run, algorithm='PPO', step=7, timesteps=112,
        scalars={'loss': {'label': 'Loss', 'value': 4}}, details={})
    assert reader.snapshot()['diagnostics']['loss']['points'] == [[7, 4]]
    replacement = run / 'replacement.jsonl'
    replacement.write_text(path.read_text().replace('"value": 4', '"value": 9'))
    replacement.replace(path)
    assert reader.snapshot()['diagnostics']['loss']['points'] == [[7, 9]]


def test_failure_report_retains_results(tmp_path):
    from ai.reporting import mark_training_failed, read_json
    run = ppo(tmp_path)
    previous = read_json(run / 'training_report.json')
    mark_training_failed(run, MemoryError('prepare test'))
    report = read_json(run / 'training_report.json')
    assert report['state'] == 'failed'
    assert report['evaluations'] == previous['evaluations']
    assert report['issues'][-1] == 'MemoryError: prepare test'

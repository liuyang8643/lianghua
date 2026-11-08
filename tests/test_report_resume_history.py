import copy
import json
from pathlib import Path

import pytest

from ai import report_server


def setup_history(monkeypatch, tmp_path):
    contract = {'algorithm': {'n_steps': 64}, 'rollout': {'assignments': [0, 1]}}
    for key, lineage in [('parent', {'mode': 'root'}), ('child', {'mode': 'resume', 'parent_identity_sha256': 'parent'})]:
        folder = tmp_path / key
        folder.mkdir()
        (folder / 'run_identity.json').write_text(json.dumps({'identity_sha256': key, 'contract': contract, 'lineage': lineage}))
    (tmp_path / 'parent/latest_train_model.json').write_text(json.dumps({
        'phase': 'post_update', 'run_identity_sha256': 'parent', 'timesteps': 1280}))
    def point(step):
        return {'step': step, 'split': 'train', 'metrics': {'calmar': step}, 'artifact_id': str(step),
                'role': 'training_evaluation', 'eligible': False, 'elapsed_seconds': None}
    reports = {key: {'id': key, 'algorithm': 'PPO', 'protocol': {}, 'progress': {'unit': '轮'},
                     'selection': {'step': 2}, 'evaluations': [point(step) for step in steps]}
               for key, steps in [('parent', [0, 2, 4, 6, 8, 10]), ('child', [2, 4, 10, 12])]}
    class Reader:
        def __init__(self, output_dir, key, *args, **kwargs):
            self.run_dir, self.key = Path(output_dir), key
        def snapshot(self, *, detail=True):
            return reports[self.key]
    monkeypatch.setattr(report_server, 'ReportReader', Reader)
    config = {'runs': [{'id': key, 'algorithm': 'PPO', 'output_dir': key, 'log_path': f'{key}/stdout.log',
                        'trace_dir': f'{key}/cache', **({'history_parent': 'parent'} if key == 'child' else {})}
                       for key in ('child', 'parent')]}
    return config, reports


def test_composes_full_history_without_duplicates_or_mutating_sources(monkeypatch, tmp_path):
    config, raw = setup_history(monkeypatch, tmp_path)
    before = copy.deepcopy(raw)
    service = report_server.Reports(config, tmp_path, max_runs=1)
    child = service.snapshot()['runs'][0]
    assert [p['step'] for p in child['evaluations']] == [0, 2, 4, 6, 8, 10, 12]
    assert [p['source_run_id'] for p in child['evaluations']] == ['parent'] * 5 + ['child'] * 2
    assert child['protocol']['evaluation_history']['resume_steps'] == [10]
    assert child['selection'] == before['child']['selection']
    assert raw == before
    csv = service.csv('child').decode('utf-8-sig')
    assert 'parent,PPO,0,' in csv and 'child,PPO,10,' in csv


@pytest.mark.parametrize('kind', ['identity', 'contract', 'warm_start', 'checkpoint'])
def test_rejects_unrelated_history(monkeypatch, tmp_path, kind):
    config, _ = setup_history(monkeypatch, tmp_path)
    path = tmp_path / ('parent/latest_train_model.json' if kind == 'checkpoint' else 'child/run_identity.json')
    data = json.loads(path.read_text())
    if kind == 'identity':
        data['lineage']['parent_identity_sha256'] = 'other'
    elif kind == 'contract':
        data['contract']['algorithm']['n_steps'] = 32
    elif kind == 'warm_start':
        data['lineage']['mode'] = 'warm_start'
    else:
        data['phase'] = 'pre_update'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report_server.Reports(config, tmp_path)

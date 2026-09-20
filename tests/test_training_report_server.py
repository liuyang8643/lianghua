from __future__ import annotations

from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer

import pytest

from ai import report_server
from ai.report_traces import TraceStore


def test_report_import_does_not_load_environment_or_learner():
    subprocess.run([sys.executable, '-c',
        "import ai.report_server, sys; assert not any(n in sys.modules for n in ('env.backtest', 'offline_data.runtime', 'torch', 'stable_baselines3'))"],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)


class FakeReader:
    def __init__(self, output_dir, run_id, algorithm, *, log_path):
        self.id, self.algorithm = run_id, algorithm

    def snapshot(self):
        return {'id': self.id, 'algorithm': self.algorithm, 'progress': {'unit': '轮'},
                'evaluations': [{'step': 12, 'split': 'validation', 'artifact_id': 'model-a',
                    'role': 'diagnostic', 'eligible': False, 'elapsed_seconds': None,
                    'metrics': {'calmar': .65, 'max_drawdown': .2}}]}


@contextmanager
def running_server(monkeypatch, tmp_path):
    monkeypatch.setattr(report_server, 'ReportReader', FakeReader)
    reports = report_server.Reports({'runs': [{'id': 'ppo', 'algorithm': 'PPO', 'output_dir': 'experiment/run', 'log_path': 'experiment/stdout.log', 'trace_dir': 'experiment/monitor/cache'}]}, tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), report_server.make_handler(reports))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, reports
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request(server, method, url, payload=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    connection.request(method, url, body=json.dumps(payload) if payload is not None else None, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_read_only_http_and_csv_preserve_missing_diagnostic_values(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, _):
        status, headers, body = request(server, 'GET', '/api/status')
        assert status == 200
        assert json.loads(body)['schema_version'] == 'training-report-v2'
        assert 'nosniff' == headers['X-Content-Type-Options']
        status, _, body = request(server, 'GET', '/api/evaluations.csv?run=ppo')
        assert status == 200
        text = body.decode('utf-8-sig')
        assert 'diagnostic,False,0.65,,0.2,,' in text
        status, _, body = request(server, 'GET', '/api/checkpoints?run=ppo')
        assert status == 200
        assert json.loads(body)['checkpoints'] == []
        assert not (tmp_path / 'experiment').exists()


def test_http_paths_and_post_inputs_cannot_choose_sources(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, _):
        assert request(server, 'GET', '/../../configs/config.json')[0] == 404
        assert request(server, 'GET', '/api/checkpoints?run=../../outside')[0] == 400
        assert request(server, 'POST', '/api/replay', {'run': 'ppo', 'ids': [], 'source': 'evil'})[0] == 405
        assert request(server, 'POST', '/api/replay', {'run': 'ppo', 'ids': ['x']}, {'Origin': 'https://unrelated.invalid'})[0] == 405
        assert request(server, 'POST', '/api/replay', {'run': 'ppo', 'ids': ['x']})[0] == 405


def test_duplicate_run_ids_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(report_server, 'ReportReader', FakeReader)
    entry = {'id': 'same', 'algorithm': 'PPO', 'output_dir': '.', 'log_path': 'stdout.log', 'trace_dir': 'monitor/cache'}
    with pytest.raises(ValueError, match='unique'):
        report_server.Reports({'runs': [entry, entry]}, tmp_path)


def test_status_only_reads_eight_newest_reports(monkeypatch, tmp_path):
    visited = []
    class Reader(FakeReader):
        def __init__(self, output_dir, key, algorithm, **kwargs):
            self.key = key
        def snapshot(self):
            visited.append(self.key)
            return {'id': self.key}
    monkeypatch.setattr(report_server, 'ReportReader', Reader)
    entries = [{'id': f'run{i}', 'algorithm': 'PPO', 'output_dir': f'run{i}',
                'log_path': f'run{i}.log', 'trace_dir': f'cache{i}'} for i in range(12)]
    reports = report_server.Reports({'runs': entries}, tmp_path)
    assert len(reports.snapshot()['runs']) == 8
    assert visited == [f'run{i}' for i in range(8)]


def test_preparing_report_has_live_elapsed_and_truncated_stage_log(tmp_path):
    from ai.reporting import ReportReader, write_preparing_report
    write_preparing_report(tmp_path, algorithm='PPO', total=100, splits={})
    log = tmp_path / 'stdout.log'
    log.write_text('{"event": "validation_evaluation", "payload": "' + 'x' * 2000 + '"}\n')
    report = ReportReader(tmp_path, 'test', 'PPO', log_path=log).snapshot()
    assert report['progress']['wall_elapsed_seconds'] >= 0
    assert report['state_label'] == '初始验证集回测已完成，继续初始化'


def test_trace_get_rejects_wrong_checkpoint_or_split(tmp_path):
    directory = tmp_path / 'monitor' / 'cache'
    directory.mkdir(parents=True)
    key = '000012-' + 'a' * 16
    (directory / f'{key}.meta.json').write_text(json.dumps({'id': key, 'sha256': 'a' * 64, 'rollout': 12}))
    trace_path = directory / f'{key}.trace.json'
    store = TraceStore(tmp_path / 'run', directory=directory)
    assert store.trace(key) is None
    trace_path.write_text(json.dumps({'id': key, 'sha256': 'b' * 64, 'split': 'train'}))
    with pytest.raises(ValueError, match='身份'):
        store.trace(key)
    trace_path.write_text(json.dumps({'id': key, 'sha256': 'a' * 64, 'split': 'test'}))
    with pytest.raises(ValueError, match='身份'):
        store.trace(key)
    with pytest.raises(ValueError):
        store.trace('../../outside')



def test_catalog_caches_validation_but_invalidates_changed_files(tmp_path, monkeypatch):
    import hashlib
    import ai.report_traces as traces
    directory = tmp_path / 'traces'
    directory.mkdir()
    key = '000012-' + 'a' * 16
    metadata = {'id': key, 'sha256': 'a' * 64, 'rollout': 12}
    meta_path = directory / f'{key}.meta.json'
    meta_path.write_text(json.dumps(metadata))
    contract = {'source_sha256': 'source', 'runtime': {'file_sha256': 'runtime'}, 'normalizer_sha256': 'normalizer'}
    identity = {'contract': contract}
    identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    identity_path = tmp_path / 'run_identity.json'
    identity_path.write_text(json.dumps(identity))
    payload = {**metadata, 'split': 'train', 'source_sha256': 'source', 'runtime_sha256': 'runtime',
               'normalizer_sha256': 'normalizer', 'controls': {}, 'dates': []}
    trace_path = directory / f'{key}.trace.json'
    trace_path.write_text(json.dumps(payload))
    store = TraceStore(tmp_path, directory=directory)
    assert store.catalog()['checkpoints'][0]['job']['state'] == 'ready'
    with monkeypatch.context() as patch:
        patch.setattr(traces, 'read_json', lambda *a: pytest.fail('unchanged catalog reparsed JSON'))
        assert store.catalog()['checkpoints'][0]['job']['state'] == 'ready'
        assert store.trace(key)['id'] == key
    trace_path.unlink()
    assert store.catalog()['checkpoints'][0]['job']['state'] == 'missing'
    trace_path.write_text(json.dumps(payload))
    assert store.catalog()['checkpoints'][0]['job']['state'] == 'ready'
    identity['contract']['source_sha256'] = 'corrupt'
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match='指纹'):
        store.catalog()

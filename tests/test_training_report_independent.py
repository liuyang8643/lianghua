"""Independent HTTP and immutable-artifact boundaries for shared reports."""

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from ai.reporting import ReportReader
from ai.report_server import make_handler
from ai.report_traces import TraceStore


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def test_legacy_files_are_not_runtime_compatibility_inputs(tmp_path):
    run = tmp_path / 'run'
    write_json(run / 'training.json', {'n_envs': 2, 'n_steps': 64})
    write_json(run / 'evaluation_curves.json', {'test': [{'test_calmar': 999}]})
    with pytest.raises(FileNotFoundError):
        ReportReader(tmp_path / 'run', 'ppo', 'PPO', log_path=tmp_path / 'stdout.log').snapshot()


@pytest.fixture
def trace_store(tmp_path):
    identity = {"lineage": {"parent_identity_sha256": None}, "contract": {
        "source_sha256": "s" * 64, "runtime": {"file_sha256": "r" * 64},
        "normalizer_sha256": "n" * 64, "splits": {"train": ["2014-01-01", "2021-12-31"]},
    }}
    identity["identity_sha256"] = hashlib.sha256(json.dumps(
        identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(tmp_path / "run" / "run_identity.json", identity)
    store = TraceStore(tmp_path / 'run', directory=tmp_path / 'monitor' / 'cache')
    checkpoint = b"frozen-model"
    sha = hashlib.sha256(checkpoint).hexdigest()
    key = "000010-" + sha[:16]
    write_json(store.directory / f"{key}.meta.json", {
        "id": key, "rollout": 10, "sha256": sha,
        "run_identity_sha256": identity["identity_sha256"],
    })
    (store.directory / f"{key}.zip").write_bytes(checkpoint)
    return store, key, identity


def test_trace_rejects_modified_identity(trace_store):
    store, key, identity = trace_store
    identity["contract"]["normalizer_sha256"] = "changed"
    write_json(store.output_dir / "run_identity.json", identity)
    with pytest.raises(ValueError, match="指纹"):
        store._identity()


def test_catalog_cannot_label_mismatched_trace_ready(trace_store):
    store, key, identity = trace_store
    meta = json.loads((store.directory / f"{key}.meta.json").read_text("utf8"))
    trace = {"id": key, "sha256": meta["sha256"], "split": "train", "dates": [], "controls": {},
             "source_sha256": "wrong", "runtime_sha256": "r" * 64, "normalizer_sha256": "n" * 64}
    write_json(store.directory / f"{key}.trace.json", trace)
    with pytest.raises(ValueError):
        store.trace(key)
    with pytest.raises(ValueError):
        store.catalog()
    trace["source_sha256"] = identity["contract"]["source_sha256"]
    write_json(store.directory / f"{key}.trace.json", trace)
    assert store.trace(key) == trace
    assert store.catalog()["checkpoints"][0]["job"]["state"] == "ready"


@pytest.fixture
def http_report():
    submitted = []
    store = SimpleNamespace(
        catalog=lambda: {"checkpoints": []},
        trace=lambda key: None,
        markets=lambda: {"markets": []},
    )

    def require(key):
        if key != "ppo":
            raise ValueError("unknown run")

    reports = SimpleNamespace(
        snapshot=lambda: {"runs": [], "schema_version": "training-report-v2"},
        require_run=require, traces={"ppo": store},
        csv=lambda key: b"run,split,calmar\r\nppo,train,0.9\r\n",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reports))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_port, submitted
    server.shutdown()
    server.server_close()
    thread.join(5)
    assert not thread.is_alive()


def request(port, method, path, payload=None, headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    body = None if payload is None else json.dumps(payload).encode()
    connection.request(method, path, body, headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_get_report_catalog_market_and_csv_cannot_submit_jobs(http_report):
    port, submitted = http_report
    for path in ("/api/status", "/api/checkpoints?run=ppo", "/api/markets?run=ppo", "/api/evaluations.csv?run=ppo"):
        status, headers, body = request(port, "GET", path)
        assert status == 200
        assert body
        assert headers["Cache-Control"] == "no-store"
    assert request(port, "GET", "/api/trace?run=ppo&id=missing")[0] == 404
    assert request(port, "GET", "/../../configs/env.py")[0] == 404
    assert request(port, "GET", "/api/checkpoints?run=unknown")[0] == 400
    assert submitted == []


@pytest.mark.parametrize("headers", [
    {"Origin": "https://external.example"},
    {"Sec-Fetch-Site": "cross-site"},
    {"Host": "external.example"},
])
def test_cross_origin_post_cannot_request_replay(http_report, headers):
    port, submitted = http_report
    status, _, _ = request(port, "POST", "/api/replay", {"run": "ppo", "ids": ["sample"]}, headers)
    assert status == 405
    assert submitted == []


def test_even_explicit_local_post_cannot_execute_historical_models(http_report):
    port, submitted = http_report
    status, _, body = request(port, "POST", "/api/replay", {"run": "ppo", "ids": ["sample"]},
                              {"Origin": f"http://127.0.0.1:{port}"})
    assert status == 405
    assert "error" in json.loads(body)
    assert submitted == []

"""One local report service for GA and PPO; training remains an independent CLI."""
from __future__ import annotations

import argparse
import csv
import gzip
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlsplit

from ai.reporting import ReportReader, clean_json, REPORT_SCHEMA_VERSION
from ai.report_traces import TraceStore

ASSETS = Path(__file__).with_name('report_assets')


class Reports:
    def __init__(self, config: dict, base: Path, *, max_runs: int | None = None):
        if max_runs is not None and max_runs <= 0:
            raise ValueError('max_runs must be positive when given')
        self.max_runs = max_runs
        self.readers = {}
        self.traces = {}
        self.lock = threading.RLock()
        for entry in config['runs']:
            key = entry['id']
            if not re.fullmatch(r'[a-zA-Z0-9_-]+', key) or key in self.readers:
                raise ValueError('Run ID must be unique and contain only letters, digits, underscore or hyphen')
            output_dir = (base / entry['output_dir']).resolve()
            reader = ReportReader(output_dir, key, entry['algorithm'], log_path=base / entry['log_path'])
            self.readers[key] = reader
            self.traces[key] = TraceStore(output_dir, directory=base / entry['trace_dir'])
        if not self.readers:
            raise ValueError('Configure at least one run')

    def require_run(self, key: str):
        if key not in self.readers:
            raise ValueError('未找到指定运行')
        return self.readers[key]

    def snapshot(self, detail: str | None = None) -> dict:
        with self.lock:
            # Registry order is newest first; snapshots are mtime-cached per run, so every
            # registered run is shown unless --max-runs narrows the dashboard. Only the
            # selected run carries its heavy detail sections; the others are card/curve-sized.
            items = list(self.readers.items())
            if self.max_runs is not None:
                items = items[:self.max_runs]
            rows = [reader.snapshot(detail=(key == detail)) for key, reader in items]
            return {'schema_version': REPORT_SCHEMA_VERSION, 'updated_at': datetime.now(timezone.utc).isoformat(), 'runs': rows}

    def csv(self, key: str | None) -> bytes:
        with self.lock:
            readers = [self.require_run(key)] if key else self.readers.values()
            stream = io.StringIO(newline='')
            writer = csv.writer(stream)
            writer.writerow(('run', 'algorithm', 'step', 'unit', 'split', 'artifact_id', 'role', 'eligible',
                             'calmar', 'annualized_return', 'max_drawdown', 'sharpe', 'elapsed_seconds'))
            for reader in readers:
                report = reader.snapshot()
                for point in report['evaluations']:
                    metrics = point['metrics']
                    writer.writerow((report['id'], report['algorithm'], point['step'], report['progress']['unit'],
                        point['split'], point['artifact_id'], point['role'], point['eligible'],
                        *(metrics.get(key) for key in ('calmar', 'annualized_return', 'max_drawdown', 'sharpe')),
                        point['elapsed_seconds']))
            return stream.getvalue().encode('utf-8-sig')


def make_handler(reports: Reports):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, value, status=200, mime='application/json; charset=utf-8', download=False):
            data = value if isinstance(value, bytes) else json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False).encode('utf-8')
            compressed = len(data) > 1024 and 'gzip' in self.headers.get('Accept-Encoding', '')
            if compressed:
                data = gzip.compress(data, compresslevel=3)
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Vary', 'Accept-Encoding')
            if compressed:
                self.send_header('Content-Encoding', 'gzip')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'")
            if download:
                self.send_header('Content-Disposition', 'attachment; filename="training-evaluations.csv"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path
            try:
                if path == '/api/status':
                    return self.respond(reports.snapshot(detail=query.get('detail', [None])[0]))
                if path == '/api/evaluations.csv':
                    return self.respond(reports.csv(query.get('run', [None])[0]), mime='text/csv; charset=utf-8', download=True)
                if path in ('/api/checkpoints', '/api/trace', '/api/markets'):
                    key = query.get('run', [''])[0]
                    reports.require_run(key)
                    store = reports.traces[key]
                    if path == '/api/checkpoints':
                        value = store.catalog()
                    elif path == '/api/markets':
                        value = store.markets()
                    else:
                        value = store.trace(query.get('id', [''])[0])
                        if value is None:
                            return self.respond({'error': '尚未生成逐日记录'}, status=404)
                    return self.respond(value)
                assets = {'/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js', '/charts.js': 'charts.js', '/style.css': 'style.css'}
                if path in assets:
                    name = assets[path]
                    mime = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[name.rsplit('.', 1)[1]]
                    return self.respond((ASSETS / name).read_bytes(), mime=mime + '; charset=utf-8')
                return self.respond({'error': '未找到页面'}, status=404)
            except (ValueError, KeyError) as error:
                return self.respond({'error': str(error)}, status=400)
            except OSError as error:
                return self.respond({'error': f'读取报告失败：{error}'}, status=503)

        def do_POST(self):
            length = self.headers.get('Content-Length', '0')
            if length.isdecimal() and int(length) <= 4096:
                self.rfile.read(int(length))
            return self.respond({'error': '报告服务只读取已保存记录，不执行历史模型'}, status=405)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='JSON run list; relative paths resolve against this file')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--max-runs', type=int, default=None,
                        help='show only the first N registered runs (default: all)')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    reports = Reports(config, config_path.parent, max_runs=args.max_runs)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(reports))
    print(json.dumps({'url': f'http://127.0.0.1:{server.server_port}', 'runs': list(reports.readers)}), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()

"""Read saved train-only traces without importing or executing model code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import threading

from ai.reporting import read_json


class TraceStore:
    def __init__(self, output_dir: Path, *, directory: Path):
        self.output_dir = Path(output_dir).resolve()
        self.directory = Path(directory).resolve()
        self.lock = threading.RLock()
        self._json_cache = {}
        self._verified_traces = {}
        self._identity_signature = None
        self._trace_cache = None

    @staticmethod
    def _signature(path: Path):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def _read_cached(self, path: Path):
        signature = self._signature(path)
        cached = self._json_cache.get(path)
        if cached is None or cached[0] != signature:
            value = read_json(path)
            self._json_cache[path] = (signature, value)
            return value
        return cached[1]

    def _meta(self, key: str) -> dict:
        if not isinstance(key, str) or not re.fullmatch(r'\d{6,}-[a-f0-9]{16}', key):
            raise ValueError('无效的模型记录编号')
        record = self._read_cached(self.directory / f'{key}.meta.json')
        if record is None or record['id'] != key:
            raise ValueError('未找到对应模型记录')
        if not key.endswith(record['sha256'][:16]):
            raise ValueError('模型记录编号与指纹不一致')
        return record

    def _identity(self) -> dict:
        path = self.output_dir / 'run_identity.json'
        identity = self._read_cached(path)
        if identity is None:
            raise ValueError('缺少冻结运行身份')
        signature = self._json_cache[path][0]
        if signature != self._identity_signature:
            self._verify_identity(identity)
            self._identity_signature = signature
        return identity

    @staticmethod
    def _verify_identity(identity: dict) -> None:
        payload = {key: value for key, value in identity.items() if key != 'identity_sha256'}
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != identity['identity_sha256']:
            raise ValueError('冻结运行身份指纹不一致')

    def catalog(self) -> dict:
        with self.lock:
            entries = []
            for path in self.directory.glob('*.meta.json'):
                record = self._meta(path.name.removesuffix('.meta.json'))
                key = record['id']
                signature = (self._signature(path), self._signature(self.directory / f'{key}.trace.json'),
                             self._signature(self.output_dir / 'run_identity.json'))
                cached = self._verified_traces.get(key)
                if cached is None or cached[0] != signature:
                    cached = (signature, self.trace(key) is not None)
                    self._verified_traces[key] = cached
                job = {'state': 'ready' if cached[1] else 'missing'}
                entries.append({**record, 'step': record['rollout'], 'job': job})
            return {'checkpoints': sorted(entries, key=lambda row: (row['step'], row['id'])),
                    'split': 'train'}

    def trace(self, key: str) -> dict | None:
        record = self._meta(key)
        signature = (self._signature(self.directory / f'{key}.meta.json'),
                     self._signature(self.directory / f'{key}.trace.json'),
                     self._signature(self.output_dir / 'run_identity.json'))
        if self._trace_cache is not None and self._trace_cache[:2] == (key, signature):
            return self._trace_cache[2]
        trace = read_json(self.directory / f'{key}.trace.json')
        if trace is not None:
            if trace['id'] != key or trace['sha256'] != record['sha256'] or trace['split'] != 'train':
                raise ValueError('逐日记录与训练期模型身份不一致')
            contract = self._identity()['contract']
            expected = {'source_sha256': contract['source_sha256'],
                        'runtime_sha256': contract['runtime']['file_sha256'],
                        'normalizer_sha256': contract['normalizer_sha256']}
            if any(trace.get(name) != value for name, value in expected.items()):
                raise ValueError('逐日记录的数据、源码或归一化身份不一致')
            if not isinstance(trace['controls'], dict):
                raise ValueError('逐日控制记录必须使用统一字段字典')
            if any(not isinstance(control['label'], str) or len(control['values']) != len(trace['dates'])
                   for control in trace['controls'].values()):
                raise ValueError('逐日控制记录长度或标签不合法')
        self._trace_cache = (key, signature, trace)
        return trace

    def markets(self) -> dict:
        return self._read_cached(self.directory / 'markets.json') or {'markets': []}


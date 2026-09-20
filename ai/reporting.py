"""Read existing GA/PPO artifacts into one presentation contract.

Producers write the presentation contract; readers consume saved files only.
This module has no training or market-data dependencies.
Metrics use ratios (including positive drawdown magnitude), not percentages.
"""
from __future__ import annotations

from collections import deque
import ctypes
import json
import math
import os
from pathlib import Path
import threading
import copy
import time
import psutil
import re

from utils.atomic_file import atomic_write_json
from typing import Any

SPLITS = ("train", "validation", "test")
_DIAGNOSTICS = {
    "train/loss": "Loss", "train/policy_gradient_loss": "Policy loss",
    "train/value_loss": "Value loss", "train/approx_kl": "KL",
    "train/entropy_loss": "Entropy loss", "train/clip_fraction": "Clip fraction",
    "train/explained_variance": "Explained variance", "train/learning_rate": "Learning rate",
    "train/n_updates": "Update epochs",
}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def clean_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def process_alive(pid: int | None) -> bool | None:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False if ctypes.get_last_error() == 87 else None
        try:
            code = ctypes.c_ulong()
            return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


def _sample(points: list[list], limit: int = 1000) -> list[list]:
    """Bound diagnostic payload while retaining endpoints and bucket extrema."""
    if len(points) <= limit:
        return points
    width = math.ceil((len(points) - 2) / ((limit - 2) // 2))
    indices = [0]
    for start in range(1, len(points) - 1, width):
        bucket = range(start, min(start + width, len(points) - 1))
        low = min(bucket, key=lambda i: points[i][1])
        high = max(bucket, key=lambda i: points[i][1])
        indices.extend(sorted({low, high}))
    return [points[i] for i in indices + [len(points) - 1]]


class _Log:
    """Append-only JSONL/plain log reader; unfinished final lines wait for append."""
    def __init__(self, path: Path, *, plain: bool = False):
        self.path, self.plain = path, plain
        self.offset = 0
        self.pending = b""
        self.rows: list[dict] = []
        self.tail: deque[str] = deque(maxlen=60)
        self.signature = None
        self.generation = 0

    def update(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self.signature is not None:
                self.offset, self.pending, self.signature = 0, b"", None
                self.rows.clear()
                self.tail.clear()
                self.generation += 1
            return
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if signature == self.signature:
            return
        if (self.signature and stat.st_size <= self.offset) or (self.signature and stat.st_ino != self.signature[2]):
            self.offset, self.pending = 0, b""
            self.rows.clear()
            self.tail.clear()
            self.generation += 1
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        for raw in lines:
            line = raw.decode("utf-8-sig").strip()
            if not line:
                continue
            self.tail.append(line[:1500])
            if self.plain:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSONL objects: {self.path}")
            self.rows.append(value)
        self.offset += len(data)
        self.signature = signature


REPORT_SCHEMA_VERSION = 'training-report-v2'
DIAGNOSTIC_SCHEMA_VERSION = 'training-diagnostics-v3'


def append_training_diagnostic(output_dir: Path, *, algorithm: str, step: float,
                               timesteps: int | None, scalars: dict, details: dict) -> None:
    """One append-only presentation contract for measured PPO and GA diagnostics."""
    record = {'schema_version': DIAGNOSTIC_SCHEMA_VERSION, 'algorithm': algorithm,
              'step': step, 'timesteps': timesteps, 'scalars': scalars, 'details': details}
    with (Path(output_dir) / 'training_diagnostics.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def append_ppo_update_diagnostic(output_dir: Path, diagnostic: dict, rollout_size: int) -> None:
    scalars = {name: {'label': label, 'value': diagnostic['train'][name]}
               for name, label in _DIAGNOSTICS.items() if name in diagnostic['train']}
    for name, quantiles in diagnostic['rollout_quantiles'].items():
        for suffix, index in (('p01', 1), ('median', 2), ('p99', 3)):
            scalars[f'{name}_{suffix}'] = {'label': f'{name} · {suffix}', 'value': quantiles[index]}
    append_training_diagnostic(output_dir, algorithm='PPO',
        step=diagnostic['timesteps'] / rollout_size, timesteps=diagnostic['timesteps'],
        scalars=clean_json(scalars), details={'scope': 'after_optimizer_update',
                                            'optimizer': clean_json(diagnostic)})


def _report(output_dir: Path, algorithm: str, total: int, current: float,
            timesteps: int | None, elapsed_seconds: float | None, complete: bool) -> dict:
    return {'schema_version': REPORT_SCHEMA_VERSION, 'id': '', 'label': output_dir.parent.name
            if output_dir.name == 'run' else output_dir.name, 'algorithm': algorithm,
            'process_pid': os.getpid(), 'state': 'complete' if complete else 'running',
            'state_label': '训练已结束' if complete else '训练进程运行中',
            'progress': {'current': current, 'total': total, 'unit': '轮' if algorithm == 'PPO' else '代',
                         'timesteps': timesteps, 'elapsed_seconds': elapsed_seconds,
                         'wall_elapsed_seconds': max(0.0, time.time() - psutil.Process().create_time())},
            'splits': dict.fromkeys(SPLITS), 'parameters': {}, 'factors': [],
            'evaluations': [], 'selection': None, 'baseline': {}, 'diagnostics': {},
            'diagnostic_records': [], 'latest_distribution': None, 'diagnostic_status': 'recording',
            'actions': [], 'logs': [], 'issues': [], 'protocol': {}, 'identity': None}


def _point(step, split, metrics, artifact, role, eligible, identity, seconds):
    return {'step': step, 'split': split, 'metrics': metrics, 'artifact_id': artifact,
            'role': role, 'eligible': eligible if split == 'validation' else False,
            'identity': identity, 'elapsed_seconds': seconds}


def write_preparing_report(output_dir: Path, *, algorithm: str, total: int, splits: dict) -> None:
    """Publish an explicit preparation state before expensive snapshot assembly."""
    output_dir = Path(output_dir)
    report = _report(output_dir, algorithm, total, 0, 0 if algorithm == 'PPO' else None, None, False)
    report['state'], report['state_label'] = 'preparing', '正在准备离线数据与模型输入'
    report['diagnostic_status'] = 'awaiting_first_rollout'
    report['splits'] = splits
    atomic_write_json(output_dir / 'training_report.json', report, allow_nan=False)


def mark_training_failed(output_dir: Path, error: BaseException) -> None:
    """Persist a failed lifecycle without discarding the last measured results."""
    path = Path(output_dir) / 'training_report.json'
    report = read_json(path)
    if report is None:
        raise ValueError('Training failure requires an initialized report')
    report['state'], report['state_label'] = 'failed', '训练失败'
    report['issues'].append(f'{type(error).__name__}: {error}')
    report['progress']['wall_elapsed_seconds'] = max(0.0, time.time() - psutil.Process().create_time())
    atomic_write_json(path, report, allow_nan=False)


def _selection(selected: dict | None, report: dict, *, artifact_key: str, step: float | None):
    if selected is None:
        return None
    artifact = selected[artifact_key]
    metrics = dict.fromkeys(SPLITS)
    for point in report['evaluations']:
        if point['artifact_id'] == artifact:
            metrics[point['split']] = point['metrics']
    return {'step': step, 'artifact_id': artifact, 'metrics': metrics, 'config': None}


def write_ppo_report(output_dir: Path, *, total_rollouts: int, timesteps: int,
                     curves: dict, baselines: dict, elapsed_seconds: float | None = None,
                     complete: bool = False) -> None:
    """Publish already-opened PPO results; never opens a dataset or infers selection."""
    output_dir = Path(output_dir)
    identity = read_json(output_dir / 'run_identity.json')
    if identity is None:
        raise ValueError('PPO report requires a sealed run identity')
    contract = identity['contract']
    algorithm = contract['algorithm']
    n_envs = len(contract['rollout']['assignments'])
    size = algorithm['n_steps'] * n_envs
    report = _report(output_dir, 'PPO', total_rollouts, timesteps / size, timesteps, elapsed_seconds, complete)
    report['identity'] = identity['identity_sha256']
    report['splits'] = contract['splits']
    report['parameters'] = {**algorithm, 'n_envs': n_envs, 'rollout_size': size}
    report['protocol'] = {'selection': algorithm['checkpoint_selection'],
                          'test_role': contract['evaluation_protocol']['test_usage'],
                          'evaluation_interval': algorithm['complete_train_evaluation_every_rollouts'],
                          'elapsed_scope': 'recorded_collection_update_and_checkpoint_seconds_excludes_evaluation_and_startup'}
    for split in SPLITS:
        baseline = baselines[split]
        if baseline is not None:
            report['baseline'][split] = baseline['metrics']
        for row in curves[split]:
            step = row['timesteps'] / size
            report['evaluations'].append(_point(step, split, row[f'{split}_metrics'],
                row['checkpoint_sha256'], 'training_evaluation' if split == 'train' else
                'formal_validation' if split == 'validation' else 'test_evaluation',
                row['eligible_for_selection'], row['run_identity_sha256'], row['evaluation_elapsed_seconds']))
            if split == 'train':
                dynamic = row['dynamic_config']
                weights = {name.removeprefix('factor_weight.'): {'min': value['minimum'],
                           'mean': value['mean'], 'max': value['maximum']}
                           for name, value in dynamic['continuous'].items() if name.startswith('factor_weight.')}
                controls = {name: value for name, value in dynamic['continuous'].items()
                            if not name.startswith('factor_weight.')}
                controls['buy_n'] = dynamic['categorical']['buy_n']
                report['actions'].append({'step': step, 'artifact_id': row['checkpoint_sha256'],
                                          'weights': weights, 'controls': controls})
                report['factors'] = list(weights)
    selected = read_json(output_dir / 'model.json')
    report['selection'] = _selection(selected, report, artifact_key='sha256',
                                      step=None if selected is None else selected['timesteps'] / size)
    atomic_write_json(output_dir / 'training_report.json', report, allow_nan=False)


def write_ga_report(output_dir: Path, *, total_generations: int, generation: int,
                    best: dict, complete: bool = False) -> None:
    """Publish GA generation champions, using the same report contract as PPO."""
    output_dir = Path(output_dir)
    metadata = read_json(output_dir / 'run_metadata.json')
    if metadata is None:
        raise ValueError('GA report requires run metadata')
    report = _report(output_dir, 'GA', total_generations, generation, None, None, complete)
    report['parameters'] = {'generations': total_generations, 'seed': metadata['seed']}
    report['splits']['train'] = [metadata['decision_start'], metadata['decision_end']]
    report['factors'] = list(best['individual_config']['weights'])
    report['protocol']['objective'] = metadata['objective']
    comparison = read_json(output_dir / 'comparison.json')
    if comparison is None:
        # Train-only GA is an explicit lifecycle, not an alternative artifact schema.
        previous = read_json(output_dir / 'training_report.json')
        if previous is not None:
            if previous['schema_version'] != REPORT_SCHEMA_VERSION:
                raise ValueError('Training report schema mismatch')
            report['evaluations'] = [row for row in previous['evaluations'] if row['step'] != generation]
            report['actions'] = [row for row in previous['actions'] if row['step'] != generation]
        config = best['individual_config']
        import hashlib
        artifact = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        rows = [{'generation': generation, 'config_sha256': artifact, 'config': config,
                 'train': best['metrics'], 'scheduled_evaluation': False, 'elapsed_seconds': None}]
    else:
        identity = metadata['comparison_identity']
        report['identity'] = identity['sha256']
        report['splits'] = identity['evaluation_contract']['splits']
        report['parameters'].update({key: comparison[key] for key in ('population', 'workers', 'unique_candidates')})
        report['protocol'].update({key: identity[key] for key in ('selection', 'test_role', 'objective', 'deployment_bundle')})
        report['protocol']['evaluation_interval'] = comparison['eval_every_generations']
        report['progress']['elapsed_seconds'] = comparison['elapsed_seconds']
        report['baseline'] = comparison['baselines']
        rows = comparison['rows']
    for row in rows:
        step, artifact = row['generation'], row['config_sha256']
        for split in SPLITS:
            if split in row:
                report['evaluations'].append(_point(step, split, row[split], artifact,
                    'generation_champion' if split == 'train' else 'formal_validation' if split == 'validation'
                    else 'diagnostic_test', row['scheduled_evaluation'], report['identity'], row['elapsed_seconds']))
        config = row['config']
        report['actions'].append({'step': step, 'artifact_id': artifact,
            'weights': {name: {'min': value, 'mean': value, 'max': value} for name, value in config['weights'].items()},
            'controls': {'buy_n': config['buy_n'], 'turnover_rate': config['turnover_rate']}})
    selected = read_json(output_dir / 'validation_selected.json')
    report['selection'] = _selection(selected, report, artifact_key='config_sha256',
                                      step=None if selected is None else selected['generation'])
    if selected is not None:
        report['selection']['config'] = selected['config']
    atomic_write_json(output_dir / 'training_report.json', report, allow_nan=False)


class ReportReader:
    """Consume only the current canonical report and diagnostic schemas."""
    def __init__(self, output_dir: Path, run_id: str, algorithm: str, *, log_path: Path):
        self.run_dir = Path(output_dir).resolve()
        self.run_id, self.algorithm = run_id, algorithm.upper()
        if self.algorithm not in ('GA', 'PPO'):
            raise ValueError('algorithm must be GA or PPO')
        self._signature = None
        self._report = None
        self._diagnostics = _Log(self.run_dir / 'training_diagnostics.jsonl')
        self._stdout = _Log(Path(log_path).resolve(), plain=True)
        self._lock = threading.RLock()
        self._diagnostic_generation = 0
        self._diagnostic_count = 0
        self._series = {}
        self._sampled_series = {}
        self._latest_distribution = None

    def snapshot(self) -> dict:
        with self._lock:
            path = self.run_dir / 'training_report.json'
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            if signature != self._signature:
                report = read_json(path)
                if report['schema_version'] != REPORT_SCHEMA_VERSION or report['algorithm'] != self.algorithm:
                    raise ValueError('Training report schema or algorithm mismatch; migrate historical artifacts once')
                self._report, self._signature = report, signature
            report = copy.deepcopy(self._report)
            report['id'] = self.run_id
            if report['state'] in ('running', 'preparing') and process_alive(report['process_pid']) is False:
                report['state'], report['state_label'] = 'stopped', '训练已停止'
            if report['state'] in ('running', 'preparing'):
                try:
                    report['progress']['wall_elapsed_seconds'] = max(
                        0.0, time.time() - psutil.Process(report['process_pid']).create_time())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self._diagnostics.update()
            if self._diagnostic_generation != self._diagnostics.generation:
                self._diagnostic_generation = self._diagnostics.generation
                self._diagnostic_count = 0
                self._series.clear()
                self._sampled_series.clear()
                self._latest_distribution = None
            new_rows = self._diagnostics.rows[self._diagnostic_count:]
            changed = set()
            for row in new_rows:
                if row['schema_version'] != DIAGNOSTIC_SCHEMA_VERSION or row['algorithm'] != self.algorithm:
                    raise ValueError('Training diagnostic schema or algorithm mismatch')
                if 'distribution_sample' in row['details']:
                    self._latest_distribution = {'step': row['step'],
                        'action_names': row['details']['action_names'], **row['details']['distribution_sample']}
                for name, scalar in row['scalars'].items():
                    if scalar['value'] is not None:
                        if not math.isfinite(scalar['value']):
                            raise ValueError('Non-finite diagnostic scalar must be recorded as unavailable')
                        self._series.setdefault(name, {'label': scalar['label'], 'points': []})['points'].append(
                            [row['step'], scalar['value']])
                        changed.add(name)
                self._diagnostic_count += 1
            for name in changed:
                item = self._series[name]
                self._sampled_series[name] = {'label': item['label'], 'points': _sample(item['points'])}
            report['diagnostics'].update(copy.deepcopy(self._sampled_series))
            report['latest_distribution'] = copy.deepcopy(self._latest_distribution)
            report['diagnostic_records'] = self._diagnostics.rows[-8:]
            report['protocol']['diagnostic_sampling'] = 'bucket_minmax_max1000_points_per_series_evaluations_not_sampled'
            self._stdout.update()
            if self._stdout.tail:
                report['logs'] = list(self._stdout.tail)
            if report['state'] == 'preparing':
                stages = {'prepared_split': '输入已准备，正在完成初始回测',
                          'training_evaluation': '初始训练集回测已完成，继续初始化',
                          'validation_evaluation': '初始验证集回测已完成，继续初始化',
                          'test_evaluation': '初始测试集回测已完成，正在启动采样'}
                for line in self._stdout.tail:
                    match = re.match(r'\{"event":\s*"([a-z_]+)"', line)
                    if match and match[1] in stages:
                        report['state_label'] = stages[match[1]]
            return report



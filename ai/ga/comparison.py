"""Serial holdout diagnostics for the existing canonical GA training loop."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from configs.training import read_evaluation_splits
from env.action_schema import ActionSchema
from env.backtest import ENVIRONMENT_SCHEMA_VERSION, prepare_episode_from_runtime
from env.fees import DEFAULT_FEE_SCHEDULE
from env.shared_episode import ResidentPreparedEpisode
from env.quantity import quantity_schema_manifest
from env.simulator import accounting_schema_manifest
from offline_data.financial_snapshot import (
    financial_snapshot_manifest_path, read_financial_snapshot_manifest,
)
from utils.atomic_file import atomic_write_json, file_sha256 as digest


def atomic_json(path, value):
    atomic_write_json(path, value, sort_keys=False, allow_nan=False)


def config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Comparison:
    def __init__(self, args, output_dir, train_episode, evaluate, *, resources):
        if args.warm_start and args.continue_from:
            raise ValueError('--warm-start cannot be combined with continue-from')
        self.resources = resources
        self.root, self.args, self.evaluator = Path(output_dir), args, evaluate
        self.started = time.monotonic()
        if args.eval_every_generations <= 0:
            raise ValueError("evaluation interval must be positive")
        self.completed_generation = 0
        self.parent_elapsed = 0.0
        self.episodes = {'train': train_episode}
        self.cache = {}
        self.rows = []
        self.unique_candidates = 0
        self.selected = None
        self.baselines = {}
        self.phase = 'checking_contract'
        self.failure = None
        self.opened = {'validation': False, 'test': False}
        schema = ActionSchema()
        splits, split_sha = read_evaluation_splits(args.evaluation_splits)
        manifest = train_episode.runtime.manifest
        financial_snapshot = read_financial_snapshot_manifest(manifest.source_path)
        if financial_snapshot['snapshot_sha256'] != manifest.source_sha256:
            raise ValueError('financial snapshot differs from the loaded training runtime')
        contract = {
            'action_schema_hash': schema.schema_hash,
            'action_schema': schema.to_dict(),
            'factor_schema_hash': train_episode.factors.schema_hash,
            'runtime': {'file_sha256': manifest.source_sha256, 'manifest': manifest.as_dict(),
                'financial_snapshot': financial_snapshot,
                'financial_manifest_file_sha256': digest(financial_snapshot_manifest_path(manifest.source_path))},
            'config': {'file_sha256': digest(args.config), 'prefilter_n': train_episode.prefilter_n},
            'algorithm': {'complete_period_evaluation_fees': asdict(DEFAULT_FEE_SCHEDULE),
                          'training_execution_fees': asdict(DEFAULT_FEE_SCHEDULE)},
            'lookback': args.lookback,
            'environment': {'schema_version': ENVIRONMENT_SCHEMA_VERSION,
                'accounting': accounting_schema_manifest(),
                'quantity_rules': quantity_schema_manifest(),
                'fees': asdict(DEFAULT_FEE_SCHEDULE)},
            'policy_inputs': 'static_DayConfig; actor_encoding_and_normalizer_not_used',
            'splits': splits,
            'split_file': {'path': str(Path(args.evaluation_splits).resolve()), 'sha256': split_sha},
        }
        if [manifest.requested_start, manifest.requested_end] != splits['train']:
            raise ValueError('training requested range mismatch with evaluation splits')
        if (train_episode.decision_start != train_episode.runtime.decision_start
                or train_episode.decision_stop != train_episode.runtime.decision_stop):
            raise ValueError('training episode must use the entire requested split')
        self.splits = contract['splits']
        dates = train_episode.runtime.decision_dates
        if str(dates[0]) < self.splits['train'][0] or str(dates[-1]) > self.splits['train'][1]:
            raise ValueError('training dates cross the sealed split')
        source = Path(__file__).resolve().parents[2]
        # Bind application modules without traversing archived experiments,
        # local environments or user artifacts when invoked from the workspace.
        files = {p.relative_to(source).as_posix(): digest(p)
            for module in ('ai', 'env', 'factor', 'offline_data', 'trade', 'utils')
            for p in sorted((source / module).rglob('*.py')) if '__pycache__' not in p.parts}
        files['configs/training.py'] = digest(source / 'configs/training.py')
        self.identity = {'version': 'ga-production-three-split-v9-canonical-continuous-actions',
            'evaluation_contract': contract,
            'source_files': files, 'seed': args.seed, 'population': args.population_size,
            'source_scope': 'ai/env/factor/offline_data/trade/utils Python modules and configs/training.py; config and runtime separately hashed',
            'holdout_episode_cache': 'each_split_prepared_once; compact_read_only_shared_memory; scalar_results_cached_by_config_and_split',
            'generations': args.generations, 'workers': args.workers,
            'genes': list(schema.action_names), 'weight_support': '[0,1] continuous independent',
            'policy': 'one fixed DayConfig for every decision date',
            'objective': 'complete_train_calmar', 'selection': 'maximum_validation_calmar_of_generation_champions',
            'test_role': 'repeated_diagnostic_only_never_selection', 'deployment_bundle': False,
            'holdout_scope': 'scheduled_generation_champion_only',
            'eval_every_generations': args.eval_every_generations, 'final_evaluation': True,
            'initialization': 'random; no PPO or static benchmark warm start',
            'initial_cash': 1000000.0}
        if args.warm_start:
            self.identity['initialization'] = {
                'mode': 'candidate_warm_start_new_root',
                'candidate_file': str(Path(args.warm_start).resolve()),
                'candidate_file_sha256': digest(args.warm_start),
                'evaluation': 'all inherited candidates re-evaluated on current training contract',
                'training_cache_reused': False,
                'holdout_metrics_reused': False,
                'optimizer_rng_restored': False,
            }
        if args.continue_from:
            parent_dir = Path(args.continue_from)
            parent = json.loads((parent_dir/'comparison_identity.json').read_text(encoding='utf8'))
            signed = dict(parent); parent_sha = signed.pop('sha256')
            if config_sha(signed) != parent_sha:
                raise ValueError('parent identity corrupt')
            derived_fields = {'initialization', 'continuation', 'sha256'}
            for key in (set(parent) | set(self.identity)) - derived_fields:
                if key not in parent or key not in self.identity or parent[key] != self.identity[key]:
                    raise ValueError(f'GA continuation changed {key}; start a new search')
            history = json.loads((parent_dir/'comparison.json').read_text(encoding='utf8'))
            if history['identity_sha256'] != parent_sha:
                raise ValueError('history parent identity mismatch')
            self.rows = history['rows']
            if [r['generation'] for r in self.rows] != list(range(1, len(self.rows)+1)):
                raise ValueError('history generations are not contiguous')
            cached_rows = [json.loads(line) for line in (parent_dir/'all_results.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
            if not self.rows or {r['generation'] for r in cached_rows} != set(range(len(self.rows))):
                raise ValueError('cache/history completed generation mismatch')
            self.opened = dict(history['opened'])
            self.baselines = dict(history['baselines'])
            for row in self.rows:
                expected_evaluation = (row['generation'] % args.eval_every_generations == 0
                                       or row['generation'] == args.generations)
                if (row['scheduled_evaluation'] != expected_evaluation
                        or ('validation' in row) != expected_evaluation
                        or ('test' in row) != expected_evaluation):
                    raise ValueError('history evaluation schedule mismatch')
                if config_sha(row['config']) != row['config_sha256']:
                    raise ValueError('history configuration hash mismatch')
                schema.from_serialized_day_config(row['config'])
                for split in ('validation','test'):
                    if split in row:
                        self.cache[(split,row['config_sha256'])] = row[split]
            self.completed_generation = max(r['generation'] for r in self.rows)
            self.parent_elapsed = history['elapsed_seconds']
            self.unique_candidates = len({config_sha(row['config']) for row in cached_rows})
            for row in self.rows:
                if 'validation' in row:
                    if self.selected is None or row['validation']['calmar'] > self.selected['validation']['calmar']:
                        self.selected = {k:v for k,v in row.items() if k != 'test'}
            self.identity['continuation'] = {'parent_identity_sha256':parent_sha,
                'training_cache_sha256':digest(parent_dir/'all_results.jsonl'),
                'history_sha256':digest(parent_dir/'comparison.json'),
                'mode':'same_contract_training_cache_continuation_reseeded_breeder',
                'optimizer_rng_restored': False,
                'start_generation':self.completed_generation+1}
            self.identity['initialization'] = 'retain training cache; breeder RNG restarts from declared seed'
        self.identity['sha256'] = config_sha(self.identity)
        atomic_json(self.root / 'comparison_identity.json', self.identity)
        self.publish()
        if self.selected is not None:
            atomic_json(self.root/'validation_selected.json',self.selected)
            atomic_json(self.root/'validation_selected_config.json',self.selected['config'])
        raw = json.loads(Path(args.config).read_text(encoding='utf-8'))
        self.baseline_config = schema.to_static_config(schema.from_static_config(raw))
        if 'train' not in self.baselines:
            self.baselines['train'] = self.evaluator(train_episode, self.baseline_config)['metrics']
        self.phase = 'training'
        self.publish()

    def publish(self):
        atomic_json(self.root / 'comparison.json', {'state': self.phase, 'failure': self.failure,
            'elapsed_seconds': self.parent_elapsed+time.monotonic()-self.started,
            'session_elapsed_seconds': time.monotonic()-self.started,
            'completed_generation':self.completed_generation,
            'eval_every_generations':self.args.eval_every_generations, 'rows': self.rows,
            'selected': self.selected, 'baselines': self.baselines, 'opened': self.opened,
            'identity_sha256': self.identity['sha256'], 'generations': self.args.generations,
            'population': self.args.population_size, 'workers': self.args.workers,
            'unique_candidates': self.unique_candidates})

    def candidate_progress(self, count):
        self.unique_candidates = count
        self.publish()

    def holdout(self, split, config):
        key = (split, config_sha(config))
        if key in self.cache:
            return self.cache[key]
        if split not in self.episodes:
            self.opened[split] = True
            self.phase = f'preparing_{split}'
            self.publish()
            prepared = prepare_episode_from_runtime(self.args.runtime_path,
                *self.splits[split], lookback=self.args.lookback,
                prefilter_n=self.episodes['train'].prefilter_n, encode_observations=False)
            if prepared.factors.schema_hash != self.episodes['train'].factors.schema_hash:
                raise ValueError('holdout factor schema differs from training')
            resident = self.resources.enter_context(ResidentPreparedEpisode(prepared))
            del prepared
            self.episodes[split] = resident.episode
            if split not in self.baselines:
                self.baselines[split] = self.evaluator(self.episodes[split], self.baseline_config)['metrics']
        self.phase = f'evaluating_{split}'
        self.publish()
        self.cache[key] = self.evaluator(self.episodes[split], config)['metrics']
        return self.cache[key]

    def evaluate(self, generation, best, unique_candidates):
        # Only copied frozen champions enter diagnostics. No result returns to GA breeding.
        config = json.loads(json.dumps(best['individual_config'], allow_nan=False))
        row = {'generation': generation + 1, 'config_sha256': config_sha(config),
            'config': config, 'train': best['metrics'], 'unique_candidates': unique_candidates}
        scheduled = (generation + 1) % self.args.eval_every_generations == 0 or generation + 1 == self.args.generations
        row['scheduled_evaluation'] = scheduled
        self.completed_generation = generation + 1
        if not scheduled:
            row['elapsed_seconds'] = self.parent_elapsed + time.monotonic()-self.started
            self.rows.append(row)
            self.phase = 'training'
            self.publish()
            return
        row['validation'] = self.holdout('validation', config)
        # Selection occurs before any test metric is obtained.
        if self.selected is None or row['validation']['calmar'] > self.selected['validation']['calmar']:
            self.selected = json.loads(json.dumps(row))
            atomic_json(self.root / 'validation_selected.json', self.selected)
            atomic_json(self.root / 'validation_selected_config.json', config)
        row['test'] = self.holdout('test', config)
        row['elapsed_seconds'] = self.parent_elapsed + time.monotonic()-self.started
        self.rows.append(row)
        self.phase = 'training'
        self.publish()
        print(json.dumps({'event': 'ga_three_split', **row}, allow_nan=False), flush=True)

    def fail(self, error):
        self.phase = 'failed'
        self.failure = {'type': type(error).__name__, 'message': str(error)}
        self.publish()

    def finish(self):
        self.phase = 'complete'
        self.publish()

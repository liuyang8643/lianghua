"""Prepare quote-independent factors before the publication window.

This caches outputs of the same factor implementations; it contains no model
selection, alternative policy, or intraday market-data persistence.
"""
from datetime import datetime,timedelta
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np

from offline_data import load_runtime_slice,latest_runtime_npz_path
from strategy_release.runtime import FrozenStrategyRuntime,FeatureInputs,OpeningCache
from data.strategy_inputs import with_open_overlay
from utils.atomic_file import atomic_write_json,file_sha256
from utils.stock.time import get_last_trading_day

_READY=None


def implementation_hash():
    root=Path(__file__).resolve().parents[1]
    paths=set()
    for folder in ('factor','factor_db/factors','strategy_release','offline_data','env'):
        paths.update((root/folder).rglob('*.py'))
    paths.update(root/p for p in ('ai/supervised/feature_data.py','ai/supervised/local_preflight.py',
        'factor/research.py','data/strategy_inputs.py','data/db/stock_name.py',
        'utils/stock/quantity.py','utils/fees.py','ai/supervised/portfolio.py',
        'trading/strategy_plan.py','trading/plan_publisher.py'))
    digest=hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def identity(trade_date,config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf8'))
    if set(config)!={'strategy_inputs','unmanaged_codes'}:
        raise ValueError('The sole strategy configuration accepts only input location and unmanaged assets')
    inputs=Path(config['strategy_inputs'])
    runtime=latest_runtime_npz_path().resolve();stat=runtime.stat()
    root=Path(__file__).resolve().parents[1]
    default=root/'configs/default_supervised_strategy.json'
    selected=json.loads(default.read_text(encoding='utf8'))
    bundle=root/selected['bundle']
    manifest=file_sha256(bundle/'manifest.json')
    if manifest!=selected['manifest_sha256']:
        raise ValueError('Frozen model manifest changed')
    return dict(day=trade_date.isoformat(),config=file_sha256(config_path),
        inputs=file_sha256(inputs/'manifest.json'),financial=file_sha256(inputs/'financial_manifest.json'),
        implementation=implementation_hash(),runtime_path=str(runtime),
        runtime_size=stat.st_size,runtime_modified_ns=stat.st_mtime_ns,
        model_config=file_sha256(default),model_manifest=manifest,
        model=file_sha256(bundle/'model.txt'),features=file_sha256(bundle/'features.json'))


@dataclass
class PreparedOpen:
    identity: dict
    assembly: object
    inputs: object
    cache: object


def prepare(trade_date,config_path,directory,*,allow_calculation):
    global _READY
    signature=identity(trade_date,config_path)
    if _READY is not None and _READY.identity==signature:
        return _READY
    _READY=None
    root=Path(directory)/'preopen';root.mkdir(parents=True,exist_ok=True)
    receipt_path=root/('features_'+trade_date.isoformat()+'.json')
    cache_path=receipt_path.with_suffix('.npz')
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    usable=receipt is not None and receipt['identity']==signature and file_sha256(cache_path)==receipt['cache_sha256']
    if not usable and not allow_calculation:
        raise RuntimeError('Quote-independent factor preflight is not ready; no late full-history calculation')
    config=json.loads(Path(config_path).read_text(encoding='utf8'))
    previous=get_last_trading_day(trade_date-timedelta(days=1))
    history=load_runtime_slice(signature['runtime_path'],previous,previous,preload_rows=100001,
                               cache_dir=Path('data/runtime_cache'))
    if receipt is not None and usable and receipt['history_sha256']!=history.manifest.source_sha256:
        raise ValueError('Prepared factor history changed')
    # These placeholders never enter a plan. Every cached family is certified
    # to exclude open[T]/preClose[T], and live replaces both complete rows.
    quotes={code:dict(open=np.asarray([history.field('close')[-1,j]]),
                      preClose=np.asarray([history.field('close')[-1,j]]))
            for j,code in enumerate(history.stock_codes)}
    context=with_open_overlay(history,trade_date,quotes,config['strategy_inputs'])
    assembly=FrozenStrategyRuntime();inputs=FeatureInputs.load(config['strategy_inputs'])
    if usable:
        with np.load(cache_path,allow_pickle=False) as payload:
            columns={name:payload[name] for name in payload.files}
        if set(columns)!=set(assembly.preopen_names()):raise ValueError('Prepared factor vocabulary changed')
        for value in columns.values():value.flags.writeable=False
        cache=OpeningCache(context,context.decision_start,inputs.identity,columns)
    else:
        cache=assembly.prepare_open(context,context.decision_start,inputs)
        if identity(trade_date,config_path)!=signature:
            raise ValueError('Inputs changed during factor preflight')
        temporary=cache_path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**cache.columns);temporary.replace(cache_path)
        atomic_write_json(receipt_path,dict(identity=signature,history_sha256=history.manifest.source_sha256,
            cache_sha256=file_sha256(cache_path),created_at=datetime.now().isoformat(),
            factor_count=len(cache.columns),contains_intraday_market_data=False))
    _READY=PreparedOpen(signature,assembly,inputs,cache)
    return _READY

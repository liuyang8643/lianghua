"""Offline release replay against sealed research evidence; never publishes targets.

Run from WBR-live with --research-root and a fresh --output directory.
The archived research engine is isolated from both repositories' working code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--research-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--support-archive', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.research_root.resolve(), args.output.resolve()
    support_archive = args.support_archive.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(REPO))
    from strategy_release import load_default
    strategy = load_default()
    # Preserve the exact release bytes while enforcing the live repository's
    # single-thread rule on this offline invocation.
    booster = strategy.model
    class SingleThreadModel:
        def predict(self, values, **kwargs):
            kwargs['num_threads'] = 1
            return booster.predict(values, **kwargs)
    strategy.model = SingleThreadModel()
    frozen = out / 'frozen_source'
    with zipfile.ZipFile(strategy.root / 'research_source.zip') as archive:
        for name in archive.namelist():
            if not (frozen / name).resolve().is_relative_to(frozen):
                raise ValueError('Unsafe archive path')
        archive.extractall(frozen)
    sources = read(strategy.root / 'research_source_manifest.json')
    for name, digest in sources.items():
        if sha(frozen / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    with zipfile.ZipFile(support_archive) as archive:
        for name in archive.namelist():
            if name.endswith('/'):
                continue
            if not (frozen / name).resolve().is_relative_to(frozen) or (frozen/name).exists():
                raise ValueError('Support archive must only add safe missing files')
        archive.extractall(frozen)
    save(out/'support_sources.json', {'archive_sha256':sha(support_archive), 'files':{name:sha(frozen/name) for name in archive.namelist() if not name.endswith('/')}})
    # Reporting imports a static-GA config adapter at module load even though
    # trace conversion never calls it. Defer only that unused import; record
    # the exact derivative. No execution or reporting arithmetic changes.
    report_adapter = frozen/'testback/backtest.py'
    before = sha(report_adapter)
    text = report_adapter.read_text(encoding='utf8')
    eager = 'from ai.ga.config import canonicalize_individual_config'
    if text.count(eager) != 1:
        raise ValueError('Unexpected report adapter source')
    text = text.replace(eager, 'def canonicalize_individual_config(*args, **kwargs):\n    from ai.ga.config import canonicalize_individual_config as impl\n    return impl(*args, **kwargs)')
    report_adapter.write_text(text, encoding='utf8')
    save(out/'report_adapter_derivative.json', {'before':before, 'after':sha(report_adapter), 'change':'defer unused static GA import; no replay arithmetic changes'})
    sys.path = [str(frozen)] + [p for p in sys.path if p and Path(p).resolve() not in (REPO, REPO / 'scripts')]
    os.chdir(frozen)
    import numpy as np
    import pandas as pd
    from dataclasses import asdict
    from scipy.stats import rankdata
    from offline_data import load_runtime_slice
    from ai.supervised.portfolio import PreparedModelPortfolio
    from ai.supervised.reporting import write_detailed_observation
    from env.qlib_strategy import RankBufferPortfolio
    from env.fees import DEFAULT_FEE_SCHEDULE
    from env.simulator import DaySimulator
    from testback.economic_positions import EconomicPositionLedger

    protocol = read(root / 'protocol.json')
    active = read(root / 'active_data.json')
    done = read(root / 'models/final/complete.json')
    if sha(strategy.root / 'model.txt') != done['model_sha256']:
        raise ValueError('Reference uses a different model')
    if asdict(DEFAULT_FEE_SCHEDULE) != strategy.manifest['fees'] or protocol['fees'] != strategy.manifest['fees']:
        raise ValueError('Fee schedules differ')
    identities = {}
    def verify(path, expected):
        actual = sha(path)
        if actual != expected:
            raise ValueError('Evidence changed: ' + str(path))
        identities[str(path)] = actual
    verify(Path(active['runtime_path']), active['runtime_sha256'])
    verify(Path(active['manifest_path']), active['manifest_sha256'])
    for name, key in [('predictions.npy', 'predictions_sha256'), ('rows.npy', 'rows_sha256')]:
        verify(root / 'models/final' / name, done[key])
    axes = read(root / 'features/data_manifest.json')
    for name in ('dates.npy', 'codes.npy', 'eligible.npy'):
        verify(root / 'features' / name, axes[name])
    dates = np.load(root / 'features/dates.npy')
    codes = np.load(root / 'features/codes.npy').astype(str)
    eligible = np.load(root / 'features/eligible.npy', mmap_mode='r')
    saved_predictions = np.load(root / 'models/final/predictions.npy', mmap_mode='r')
    np.testing.assert_array_equal(np.load(root / 'models/final/rows.npy'), np.arange(len(dates)))
    inventory = read(root / 'features/inventory.json')
    columns = []
    print('Verifying 284 frozen normalized feature columns', flush=True)
    for index, name in enumerate(strategy.names):
        item = inventory[name]
        verify(Path(item['path']), item['sha256'])
        columns.append(np.load(item['path'], mmap_mode='r'))
        if index % 50 == 0:
            print('Verified columns', index + 1, flush=True)
    # Re-predict the entire external observation interval; training uses the
    # hash-verified saved predictions plus spaced full-cross-section checks.
    validation_rows = np.flatnonzero((dates >= np.datetime64(protocol['validation'][0])) & (dates <= np.datetime64(protocol['validation'][1])))
    train_rows = np.flatnonzero((dates >= np.datetime64(protocol['train'][0])) & (dates <= np.datetime64(protocol['train'][1])))
    samples = train_rows[np.linspace(0, len(train_rows)-1, 17, dtype=int)]
    scores = np.full(saved_predictions.shape[:2], np.nan)
    pool = eligible & np.isfinite(saved_predictions).all(axis=2)
    for start in range(0, len(dates), 32):
        block = saved_predictions[start:start+32]
        mask = pool[start:start+32]
        ranks = np.full(block.shape, np.nan, dtype=np.float32)
        for k in range(5):
            values = np.where(mask, block[:, :, k], np.nan)
            count = mask.sum(axis=1)[:, None]
            rr = rankdata(values, axis=1, method='average', nan_policy='omit')
            ranks[:, :, k] = np.divide(rr-.5, count, out=np.full_like(rr, np.nan), where=count>0)
        scores[start:start+32] = np.sum(ranks*np.asarray(strategy.manifest['weights']), axis=2)
    max_prediction_diff = 0.
    for step, row in enumerate(np.r_[samples, validation_rows]):
        matrix = np.column_stack([c[row] for c in columns])
        prediction, fused = strategy.predict(matrix, feature_names=strategy.names, codes=codes, eligible=eligible[row], complete_universe=True)
        np.testing.assert_allclose(prediction, saved_predictions[row], rtol=0, atol=1e-12, equal_nan=True)
        # Exact ranks are decisive for selection, regardless of tiny arithmetic.
        np.testing.assert_array_equal(fused, scores[row])
        max_prediction_diff = max(max_prediction_diff, float(np.nanmax(np.abs(prediction-saved_predictions[row]))))
        scores[row] = fused
        if step % 50 == 0:
            print('Predicted cross sections', step + 1, '/', len(samples)+len(validation_rows), str(dates[row]), flush=True)
    save(out / 'prediction_verification.json', dict(validation_days=len(validation_rows), training_sample_days=[str(dates[r]) for r in samples], max_abs_prediction_difference=max_prediction_diff, raw_features_recomputed=False, normalized_feature_hashes_verified=len(columns)))
    save(out / 'protocol.json', {**protocol, 'observation_splits': ['train', 'validation']})
    save(out / 'active_data.json', active)
    original_target = RankBufferPortfolio.targets
    original_step = DaySimulator.step
    rows = []
    for split in ('validation', 'train'):
        print('Replaying', split, flush=True)
        rt = load_runtime_slice(active['runtime_path'], *protocol[split], preload_rows=252, cache_dir=out/'runtime_cache')
        np.testing.assert_array_equal(rt.stock_codes, codes)
        at = np.searchsorted(dates, rt.trade_dates)
        valid = at < len(dates)
        valid[valid] &= dates[at[valid]] == rt.trade_dates[valid]
        signal = np.full((rt.n_dates, rt.n_stocks), np.nan)
        signal[valid] = scores[at[valid]]
        prepared = PreparedModelPortfolio.from_scores(rt, signal, initial_cash=protocol['initial_cash'], source_hash=sha(strategy.root/'model.txt'), split=split)
        ledger = EconomicPositionLedger()
        parity = {'days': 0, 'max_weight_difference': 0.}
        def target(adapter, market, account, day, daily_scores):
            # The archive supplies account marking and ages; release supplies
            # the actual allocation. Compare against the archived allocator.
            held, sellable, ages, current, locked, buyable, equity = adapter.context(market, account, day, daily_scores)
            weights, exits = strategy.targets(daily_scores, held=held, current_weights=current, buyable=buyable, sellable=sellable, holding_days=ages)
            adapter.last_day = day - 1  # context is read again without resetting ages
            expected = original_target(adapter, market, account, day, daily_scores)
            actual = {adapter.codes[i]: float(weights[i]) for i in np.flatnonzero(weights>0)}
            if actual != expected:
                raise AssertionError('Release allocation differs on ' + str(rt.trade_dates[day]))
            parity['days'] += 1
            return actual
        def observe(simulator, account, plan, opening, next_opening, **kwargs):
            result = original_step(simulator, account, plan, opening, next_opening, **kwargs)
            ledger.observe(account, result.account_state, plan, result.fills, result.diagnostics)
            return result
        RankBufferPortfolio.targets, DaySimulator.step = target, observe
        try:
            trace, metrics, diagnostics = prepared.evaluate('rank_buffer', strategy.manifest['parameters'], details=True)
        finally:
            RankBufferPortfolio.targets, DaySimulator.step = original_target, original_step
        baseline = read(root / 'final' / (split+'.json'))
        verify(root/'final'/(split+'.npz'), baseline['trace_sha256'])
        with np.load(root/'final'/(split+'.npz')) as reference:
            for key, value in [('nav',trace.nav), ('returns',trace.portfolio_returns), ('cash',trace.cash), ('exposure',trace.exposure), ('decision_dates',trace.decision_dates), ('next_dates',trace.next_decision_dates)]:
                np.testing.assert_array_equal(value, reference[key])
        fills = [dict(date=str(date), code=f.code, side=f.side, quantity=int(f.quantity), price=f.price, fee=f.fee) for date, day in zip(trace.decision_dates,trace.fills) for f in day]
        frame = pd.DataFrame(fills)
        reference_fills = pd.read_csv(root/'final'/(split+'_fills.csv'))[list(frame.columns)]
        pd.testing.assert_frame_equal(frame, reference_fills, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-10)
        frame.to_csv(out/(split+'_fills.csv'), index=False)
        metrics['total_fees'] = sum(f['fee'] for f in fills)
        np.savez_compressed(out/(split+'.npz'), nav=trace.nav, returns=trace.portfolio_returns, cash=trace.cash, exposure=trace.exposure, decision_dates=trace.decision_dates, next_dates=trace.next_decision_dates)
        config = {**strategy.manifest['parameters'], 'weights': {'MultiHorizonScore':1.}, 'horizon_weights_pct': dict(zip(map(str,strategy.manifest['horizons']),[20,30,10,30,10]))}
        report = write_detailed_observation(out,'ml',split,rt,trace,prepared.final_account,config,ledger)
        result = dict(split=split, metrics=metrics, reference_metrics=baseline['metrics'], daily_arrays_exact=True, fills_equal=True, fills=len(fills), allocation_parity=parity, report=str(report))
        save(out/(split+'.json'), result)
        rows.append(result)
        print('Completed', split, metrics, flush=True)
    save(out/'input_hashes.json', identities)
    save(out/'complete.json', dict(state='complete', release='98075c2', results=rows, scope='Frozen research data and archived engine with release allocation; not WBR-live feature-provider acceptance', raw_features_recomputed=False, production_targets_written=False, broker_connections=0))


if __name__ == '__main__':
    main()

"""Training-only fixed-calendar style diagnostics and canonical factor replays."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from env.action_schema import ActionSchema
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from env.backtest import EpisodeSession, PreparedEpisode, run_day_config_episode
from env.factor_diagnostics import daily_rank_ic, rank_similarity, summarize_ic
from env.fees import DEFAULT_FEE_SCHEDULE
from env.simulator import settlement_economics
from factor import precompute_factors, PRODUCTION_FACTORS
from factor.library.bilibili import prepare_bilibili_factor_definitions
from factor.library.bilibili_events import RESEARCH_FACTOR_DEFINITIONS
from factor.library.bilibili_smallcap import SMALLCAP_FACTOR_DEFINITIONS
from offline_data import load_runtime_slice
from utils.atomic_file import atomic_write_json

START, END = '2000-01-01', '2022-12-31'
HORIZONS = (1, 5, 20)


def save_ic_summary(output, dates, correlations, names):
    summaries = {}
    for horizon in HORIZONS:
        rows = summarize_ic(dates, correlations[horizon], horizon)
        for row in rows:
            row['factor_name'] = names[row['factor_index']]
            # Undefined correlations/standard errors remain NaN in the binary
            # diagnostics, and are explicitly null in portable JSON.
            for key, value in row.items():
                if isinstance(value, float) and not np.isfinite(value):
                    row[key] = None
        summaries[str(horizon)] = rows
    atomic_write_json(output / 'ic_summary.json', summaries, sort_keys=False, allow_nan=False, trailing_newline=True)


def build(args):
    if (args.output / 'build_complete.json').exists():
        raise FileExistsError('build already complete; use --analyze-only or a new directory')
    payload = json.loads(Path('configs/config.json').read_text(encoding='utf-8'))['individual_config']
    source_files = set(json.loads(Path('artifacts/bilibili_133578883_20260910/source_snapshot_v7.json').read_text(encoding='utf-8')))
    source_files.update((str(Path(__file__).relative_to(Path.cwd())), 'env/factor_diagnostics.py', 'tests/test_factor_diagnostics.py', 'utils/atomic_file.py'))
    manifest = {str(Path(name)): hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sorted(source_files)}
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / 'source_snapshot.json', manifest, sort_keys=False, allow_nan=False, trailing_newline=True)
    with zipfile.ZipFile(args.output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:
            archive.write(name, name)
    atomic_write_json(args.output / 'protocol.json', {
        'schema': 'train-style-diagnostics-v1', 'start': START, 'end': END,
        'holdout': 'No validation/test rows or results used. Labels end inside training slice.',
        'periods': 'Predetermined calendar years and half-years; no optimized endpoints.',
        'horizons': HORIZONS, 'primary_horizon': 1, 'secondary_horizons': [5, 20],
        'stage_reference': '>=100 valid days; mean RankIC>0 and HAC t>1.96 is descriptive evidence, not multiplicity-adjusted significance.',
        'ic_universe': 'T PIT member, configured ST and low-price filters; complete positive forward-price chains. No future membership filtering.',
        'rank_ic': 'Average-tie Spearman on per-factor jointly valid stocks, min 30; float32 public raw cache.',
        'similarity': 'Daily Pearson of each factor own available cross-section average ranks on pairwise common valid stocks.',
        'fees': asdict(DEFAULT_FEE_SCHEDULE), 'single_factor_buy_n': 30, 'single_factor_turnover_rate': 1.0,
        'baseline': payload, 'baseline_limitation': 'Existing Amihud amount/cumulative-precision issue retained and flagged.',
        'execution': 'Canonical env daily full investment; invalid or absent signals may be tail fallback; no custom event-only portfolio.',
        'preload': 'Full available history strictly before training start for equal initialization.',
    }, sort_keys=False, allow_nan=False, trailing_newline=True)
    print('Loading sealed training runtime', flush=True)
    runtime = load_runtime_slice(args.runtime, START, END, preload_rows=100000)
    atomic_write_json(args.output / 'runtime_manifest.json', runtime.manifest.as_dict(), sort_keys=False, allow_nan=False, trailing_newline=True)
    start, stop = runtime.decision_start, runtime.decision_stop
    dates = runtime.trade_dates[start:stop]
    assert dates[-1] <= np.datetime64(END)
    np.save(args.output / 'dates.npy', dates)
    names = [d.metadata.name for d in PRODUCTION_FACTORS]
    names += ['BiliRawIssueDiscount', 'BiliAdjustedIssueDiscount', 'BiliLowLifetimeRangeRatio',
              'BiliHighLifetimeRangeRatio', 'BiliLowLifetimeAmplitude']
    names += [d.metadata.name for d in RESEARCH_FACTOR_DEFINITIONS]
    names += [SMALLCAP_FACTOR_DEFINITIONS[1].metadata.name]
    assert len(names) == len(set(names)) == 23
    atomic_write_json(args.output / 'factor_names.json', names, sort_keys=False, allow_nan=False, trailing_newline=True)
    scores = np.lib.format.open_memmap(args.output / 'scores.npy', mode='w+', dtype=np.float32,
                                       shape=(len(dates), len(names), runtime.n_stocks))
    member = (runtime.field('listing_age')[start:stop] >= 0) & ~runtime.field('delisted_mask')[start:stop]
    opening, closing, preclose = [runtime.field(k)[start:stop] for k in ('open', 'close', 'preClose')]
    economic = settlement_economics(current_mark=opening[:-1], current_close=closing[:-1],
                                    next_preclose=preclose[1:], next_open=opening[1:], diagnostics=False)
    complete = np.ones(economic.gross_return.shape, bool)
    for values in (opening[:-1], closing[:-1], preclose[1:], opening[1:]):
        complete &= np.isfinite(values) & (values > 0)
    gross = np.where(complete, economic.gross_return, np.nan)
    del economic, complete
    for horizon in HORIZONS:
        labels = np.lib.format.open_memmap(args.output / f'forward_{horizon}.npy', mode='w+', dtype=np.float32,
                                           shape=(len(dates), runtime.n_stocks))
        labels[:] = np.nan
        rows = len(dates) - horizon
        product = np.ones((rows, runtime.n_stocks), dtype=np.float64)
        for offset in range(horizon):
            product *= gross[offset:offset + rows]
        labels[:rows] = product - 1.0
        labels.flush()
        del labels, product
    del gross
    results = []

    def replay(name, batch, config_payload):
        episode = PreparedEpisode.build(runtime, batch, encode_observations=False, prefilter_n=payload['prefilter_n'])
        schema = ActionSchema(factor_names=batch.factor_names, filter_names=batch.filter_names,
            fixed_buy_n=config_payload["buy_n"], turnover_maximum=1.0)
        session = EpisodeSession(episode, action_schema=schema)
        config = session.action_schema.from_static_config(config_payload)
        trace = run_day_config_episode(session, lambda _: config)
        assert trace.full_investment_contract_satisfied and np.isfinite(trace.nav).all()
        folder = args.output / 'backtests' / name
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'trace.npz', decision_dates=trace.decision_dates, next_dates=trace.next_decision_dates,
                            nav=trace.nav, returns=trace.portfolio_returns, cash=trace.cash, exposure=trace.exposure,
                            full_investment_contract=trace.full_investment_contract)
        result = {'name': name, **trace.metrics.as_dict(), 'first_open': trace.decision_dates[0],
                  'last_valuation_open': trace.next_decision_dates[-1], 'config': session.action_schema.to_static_config(config),
                  'fills': sum(map(len, trace.fills)), 'full_investment_contract_satisfied': True,
                  'factor_schema_hash': batch.schema_hash, 'action_schema_hash': session.action_schema.schema_hash}
        atomic_write_json(folder / 'result.json', result, sort_keys=False, allow_nan=False, trailing_newline=True)
        results.append(result)
        atomic_write_json(args.output / 'partial_backtests.json', results, sort_keys=False, allow_nan=False, trailing_newline=True)
        print(json.dumps({'backtest': name, 'annual': trace.metrics.annualized_return}, ensure_ascii=False), flush=True)

    offset = 0
    for family in ('production', 'prices', 'events', 'smallcap'):
        definitions = (PRODUCTION_FACTORS if family == 'production' else
                       prepare_bilibili_factor_definitions(runtime) if family == 'prices' else
                       RESEARCH_FACTOR_DEFINITIONS if family == 'events' else (SMALLCAP_FACTOR_DEFINITIONS[1],))
        batch = precompute_factors(runtime, definitions=definitions)
        del definitions
        assert list(batch.factor_names) == names[offset:offset + len(batch.factor_names)]
        scores[:, offset:offset + len(batch.factor_names)] = batch.raw[start:stop]
        scores.flush()
        atomic_write_json(args.output / f'{family}_factor_metadata.json', [d.as_dict() for d in batch.factor_metadata], sort_keys=False, allow_nan=False, trailing_newline=True)
        if family == 'production':
            for i, filter_name in enumerate(batch.filter_names):
                if payload['filter_factors'][filter_name]:
                    member &= batch.filters[start:stop, i]
            np.save(args.output / 'eligible.npy', member)
        for name in batch.factor_names:
            configured = {**payload, 'weights': {f: float(f == name) for f in batch.factor_names},
                          'buy_n': 30, 'turnover_rate': 1.0, 'single_buy_pct': 1 / 30}
            replay(name, batch, configured)
        if family == 'production':
            replay('WBR_static_config', batch, payload)
        offset += len(batch.factor_names)
        del batch
        gc.collect()
    atomic_write_json(args.output / 'build_complete.json', {'factor_names': names, 'backtests': results}, sort_keys=False, allow_nan=False, trailing_newline=True)


def analyze(args):
    names = json.loads((args.output / 'factor_names.json').read_text(encoding='utf-8'))
    dates = np.load(args.output / 'dates.npy', allow_pickle=False)
    assert dates[0] >= np.datetime64(START) and dates[-1] <= np.datetime64(END)
    # This also permits completing JSON export after a serialization failure
    # without repeating already saved, source-identified daily calculations.
    daily_path = args.output / 'daily_diagnostics.npz'
    if daily_path.exists():
        source = json.loads((args.output / 'source_snapshot.json').read_text(encoding='utf-8'))
        diagnostics_file = Path('env/factor_diagnostics.py')
        assert hashlib.sha256(diagnostics_file.read_bytes()).hexdigest() == source[str(diagnostics_file)]
        with np.load(daily_path, allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached['dates'], dates)
            save_ic_summary(args.output, dates, {h: cached[f'ic_{h}'] for h in HORIZONS}, names)
            mean_similarity = np.nanmean(cached['similarity'], axis=0)
        atomic_write_json(args.output / 'similarity.json', {'factor_names': names,
            'mean_daily_rank_correlation': [[None if not np.isfinite(v) else float(v) for v in row] for row in mean_similarity]}, sort_keys=False, allow_nan=False, trailing_newline=True)
        files = (Path(__file__), diagnostics_file, Path('utils/atomic_file.py'))
        atomic_write_json(args.output / 'summary_export_sources.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, sort_keys=False, allow_nan=False, trailing_newline=True)
        print('Completed summary export from saved daily calculations', flush=True)
        return
    scores = np.load(args.output / 'scores.npy', mmap_mode='r', allow_pickle=False)
    eligible = np.load(args.output / 'eligible.npy', mmap_mode='r', allow_pickle=False)
    labels = {h: np.load(args.output / f'forward_{h}.npy', mmap_mode='r', allow_pickle=False) for h in HORIZONS}
    correlations = {h: np.full((len(dates), len(names)), np.nan) for h in HORIZONS}
    observations = {h: np.zeros((len(dates), len(names)), dtype=np.int32) for h in HORIZONS}
    similarity = np.full((len(dates), len(names), len(names)), np.nan, dtype=np.float32)
    for t in range(len(dates)):
        row = scores[t]
        for h in HORIZONS:
            correlations[h][t], observations[h][t] = daily_rank_ic(row, labels[h][t], eligible[t])
        similarity[t], _ = rank_similarity(row, eligible[t])
        if t % 250 == 0:
            print(json.dumps({'ic_rows': t, 'total': len(dates)}), flush=True)
    np.savez_compressed(args.output / 'daily_diagnostics.npz', dates=dates, similarity=similarity,
                        **{f'ic_{h}': correlations[h] for h in HORIZONS},
                        **{f'n_{h}': observations[h] for h in HORIZONS})
    save_ic_summary(args.output, dates, correlations, names)
    with np.errstate(invalid='ignore'):
        mean_similarity = np.nanmean(similarity, axis=0)
    atomic_write_json(args.output / 'similarity.json', {'factor_names': names,
        'mean_daily_rank_correlation': [[None if not np.isfinite(v) else float(v) for v in row] for row in mean_similarity]}, sort_keys=False, allow_nan=False, trailing_newline=True)
    print('Training-only diagnostics complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path('data/runtime/runtime_1990-12-19_2026-08-28.npz'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if not args.analyze_only:
        build(args)
    analyze(args)


if __name__ == '__main__':
    main()

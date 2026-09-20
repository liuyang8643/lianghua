"""Seal a finite research-only source-version correction for mislabeled coal history."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def correct(primary, official, archive, analysis, calendar, output):
    p = pd.read_parquet(primary)[['date', 'code', 'open', 'close']].copy()
    s = pd.read_parquet(archive)
    s = s[(s.code == '801950') & (s.date < '2022-11-10')][p.columns].copy()
    raw = json.loads(analysis.read_text(encoding='utf-8'))
    records = pd.DataFrame(raw['data']['results'])
    records['date'] = pd.to_datetime(records.bargaindate.str[:10])
    records['analysis_close'] = records.closeindex.astype(float)
    suspect = p[(p.code == '801950') & p.date.between('2014-02-21', '2015-12-31')]
    if len(suspect) != 455 or suspect.duplicated(['date', 'code']).any():
        raise ValueError('unexpected finite correction key set')
    proof = suspect.merge(records[['date', 'swindexname', 'analysis_close']], on='date', how='left')
    labeled = proof.swindexname.eq('Imp_国防军工') & proof.close.eq(proof.analysis_close)
    if int(labeled.sum()) != 454 or proof.loc[~labeled, 'date'].tolist() != [pd.Timestamp('2015-10-16')]:
        raise ValueError('official mislabeled-close evidence does not match original prices')
    reference = pd.read_parquet(official)
    overlap = s.merge(reference[reference.code == '801950'], on=['date', 'code'], suffixes=('_archive', '_primary'))
    later = overlap[overlap.date >= '2016-01-01']
    if len(later) < 1000 or not np.array_equal(later[['open_archive','close_archive']].to_numpy(), later[['open_primary','close_primary']].to_numpy()):
        raise ValueError('coal archive must exactly agree outside the finite contaminated interval')
    replacement = s[s.date.between('2014-02-21', '2015-12-31')].copy()
    with np.load(calendar, allow_pickle=False) as stored:
        expected = pd.DatetimeIndex(stored['trade_dates'])
    expected = expected[(expected >= '2014-02-21') & (expected <= '2015-12-31')]
    if not np.array_equal(replacement.date.sort_values().to_numpy(dtype='datetime64[D]'), expected.to_numpy(dtype='datetime64[D]')) or replacement.date.duplicated().any():
        raise ValueError('archive correction must cover every real session exactly once')
    if not np.isfinite(replacement[['open','close']]).all().all() or (replacement[['open','close']] <= 0).any().any():
        raise ValueError('invalid archive prices')
    changes = replacement.merge(suspect, on=['date','code'], how='left', suffixes=('_new','_old'))
    mask = (p.code == '801950') & p.date.between('2014-02-21', '2015-12-31')
    corrected = pd.concat([p[~mask], replacement], ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    corrected.to_parquet(output / 'indices.parquet', index=False)
    changes.to_parquet(output / 'coal_changes.parquet', index=False)
    boundary = corrected[(corrected.code == '801950') & corrected.date.between('2015-12-15','2016-01-08')].copy()
    boundary['close_return'] = boundary.close.pct_change()
    boundary['overnight_return'] = boundary.open / boundary.close.shift() - 1
    boundary.to_parquet(output / 'coal_boundary.parquet', index=False)
    evidence = dict(schema='finite-coal-research-source-correction-v1',
        inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (primary, official, archive, analysis, calendar)},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        replaced_rows=455, added_rows=len(replacement)-455, official_mislabeled_close_matches=454,
        exact_overlap_after_2015=len(later), interval=['2014-02-21','2015-12-31'],
        output_sha256=hashlib.sha256((output/'indices.parquet').read_bytes()).hexdigest(),
        semantics='Research source-version correction, not an independently price-certified historical coal series. Official contamination is evidenced on 454 days; 2015-10-16 belongs to the same divergent interval but lacks analysis evidence. Archive supplies all 458 sessions of this interval. All other prices unchanged. No interpolation. Not author-exact or PIT certified.')
    (output/'correction_manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary','official','archive','analysis','calendar','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    correct(args.primary,args.official,args.archive,args.analysis,args.calendar,args.output)

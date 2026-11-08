"""Bind the completed pre-fix correlation run to its frozen upstream evidence.

This does not recompute the all-pairs kernel or amend its original identity.
It verifies historical axes, all stored training IC means, and every selection
against the saved correlation matrix. Future runs bind axes at admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from env.forward_returns import label_split_mask
from .selection import greedy_select
from .ic import summary
from utils.atomic_file import atomic_write_json,file_sha256


def run(root):
    begun=time.monotonic();stage=root/'selection'
    if json.loads((stage/'status.json').read_text())['state']!='complete':raise ValueError('selection is not complete')
    identity=json.loads((stage/'identity.json').read_text())
    assert file_sha256(stage/'executed_selection.py')==identity['kernel']
    assert file_sha256(root/'protocol.json')==identity['protocol']
    config=json.loads((root/'protocol.json').read_text(encoding='utf8'));legacy=Path(config['legacy_ic_root'])
    axes={name:file_sha256(legacy/name) for name in ['dates.npy','eligible.npy']}
    # These receipts predate the actual correlation computation, and independently
    # bind the exact same two files. No new declared hash substitutes for them.
    for source,key in [('qlib_factors_v2','source'),('local_factors_v2','sources')]:
        previous=json.loads((root/source/'identity.json').read_text())[key]
        for name,sha in axes.items():
            hits=[value for path,value in previous.items() if Path(path).resolve()==(legacy/name).resolve()]
            assert hits==[sha],(source,name)
    dates=np.load(legacy/'dates.npy');which=np.flatnonzero((dates>=np.datetime64(config['train'][0]))&(dates<=np.datetime64(config['train'][1])))
    records=[];daily_hashes={};maximum=0.
    for source,sha in identity['inventories'].items():
        assert file_sha256(root/source/'inventory.json')==sha
        records.extend(json.loads((root/source/'inventory.json').read_text(encoding='utf8')))
    for record in records:
        digest=file_sha256(record['daily_path']);assert digest==record['daily_sha256']
        daily_hashes[record['name']]=digest
        with np.load(record['daily_path']) as daily:
            for h in (1,5,10,20):
                actual=summary(daily[str(h)],label_split_mask(dates,*config['train'],h))
                expected=record['statistics'][str(h)]['train']
                for key,value in actual.items():
                    if value is None:assert expected[key] is None
                    else:
                        error=abs(value-expected[key]);maximum=max(maximum,error)
                        assert error<1e-12,(record['name'],h,key,error)
    with np.load(stage/'correlations.npz') as z:
        names=list(z['names'].astype(str));corr=z['correlation'];days=z['valid_days'];pairs=z['pair_observations']
    assert names==identity['names'];assert corr.shape==(len(names),len(names))
    np.testing.assert_allclose(corr,corr.T,rtol=0,atol=1e-14,equal_nan=True)
    saved=json.loads((stage/'correlation_checkpoint.json').read_text())
    assert saved['offset']==len(which) and file_sha256(stage/saved['file'])==saved['sha256']
    with np.load(stage/saved['file']) as z:
        assert int(z['offset'])==len(which)
        np.testing.assert_array_equal(days,z['days']);np.testing.assert_array_equal(pairs,z['pairs'])
        np.testing.assert_allclose(corr,np.divide(z['sums'],days,out=np.full(corr.shape,np.nan),where=days>0),rtol=0,atol=0,equal_nan=True)
    canonical={n:i for i,n in enumerate(names)};means={h:{r['name']:r['statistics'][str(h)]['train']['rank_ic'] for r in records} for h in (1,5,10)}
    positive=[r['name'] for r in records if any(means[h][r['name']] is not None and means[h][r['name']]>0 for h in means)]
    indexes=[canonical[identity['aliases'].get(n,n)] for n in positive];full=corr[np.ix_(indexes,indexes)]
    selections={}
    for h in (1,5,10):
        keep,removed=greedy_select(positive,means[h],full)
        selection=json.loads((stage/f'selected_{h}d.json').read_text(encoding='utf8'))
        assert [positive[i] for i in keep]==[r['name'] for r in selection['selected']]
        assert removed==selection['removed']
        with np.load(stage/f'selected_{h}d_correlation.npz') as z:
            np.testing.assert_array_equal(z['names'],np.asarray([positive[i] for i in keep]))
            np.testing.assert_allclose(z['correlation'],full[np.ix_(keep,keep)],rtol=0,atol=0,equal_nan=True)
        selections[str(h)]={'count':len(keep),'max_correlation':selection['max_absolute_mean_correlation'],'unestimated_pairs':selection['unestimated_retained_pairs']}
    bound=['identity.json','executed_selection.py','correlations.npz','correlation_checkpoint.json',saved['file']]
    bound += [f'selected_{h}d{suffix}' for h in (1,5,10) for suffix in ['.json','_correlation.npz']]
    result={'state':'pass','scope':'existing completed correlation input/result binding; not a new kernel run',
            'axes':axes,'which_sha256':hashlib.sha256(which.astype('<i8').tobytes()).hexdigest(),'training_days':len(which),
            'upstream_identities':{s:file_sha256(root/s/'identity.json') for s in identity['inventories']},
            'daily_sha256':daily_hashes,'verified_factor_count':len(records),'verified_horizons':[1,5,10,20],
            'max_summary_error':maximum,'selections':selections,'artifacts':{name:file_sha256(stage/name) for name in bound},
            'elapsed_seconds':time.monotonic()-begun,'audit_source_sha256':file_sha256(__file__)}
    atomic_write_json(stage/'binding_audit.json',result,allow_nan=False)
    print({k:v for k,v in result.items() if k not in ['daily_sha256','artifacts']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)

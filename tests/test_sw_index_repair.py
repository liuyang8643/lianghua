import json
import numpy as np
import pandas as pd
import pytest
from offline_data.sw_index_repair import repair


def source_files(tmp_path):
    frame=pd.DataFrame({'code':['801010']*8,'date':pd.date_range('2015-01-01',periods=8),
                        'open':np.arange(8,dtype=float)+100,'close':np.arange(8,dtype=float)+101})
    primary=tmp_path/'primary.parquet'
    supplement=tmp_path/'supplement.parquet'
    frame.drop(index=4).to_parquet(primary,index=False)
    frame.to_parquet(supplement,index=False)
    return primary,supplement,frame


def test_real_gap_added_original_rows_preserved(tmp_path):
    primary,supplement,frame=source_files(tmp_path)
    out=tmp_path/'out'
    repair(primary,[supplement],out)
    pd.testing.assert_frame_equal(pd.read_parquet(out/'repaired_indices.parquet'),frame)
    proof=json.loads((out/'repair_manifest.json').read_text())
    assert proof['supplements'][0]['added_rows']==1
    assert proof['supplements'][0]['max_absolute_open_close_difference']==0


def test_revised_source_prices_rejected_before_write(tmp_path):
    primary,supplement,frame=source_files(tmp_path)
    frame.loc[1,'open']-=1
    frame.to_parquet(supplement,index=False)
    with pytest.raises(ValueError,match='convention differs'):
        repair(primary,[supplement],tmp_path/'out')
    assert not (tmp_path/'out'/'repaired_indices.parquet').exists()


def test_unverified_earlier_history_is_not_created(tmp_path):
    primary,supplement,frame=source_files(tmp_path)
    extra=frame.iloc[[0]].copy()
    extra['date']=pd.Timestamp('2014-12-31')
    pd.concat([extra,frame],ignore_index=True).to_parquet(supplement,index=False)
    repair(primary,[supplement],tmp_path/'out')
    actual=pd.read_parquet(tmp_path/'out'/'repaired_indices.parquet')
    assert actual.date.min()==frame.date.min()


def test_sparse_overlap_does_not_certify_patch(tmp_path):
    primary,supplement,frame=source_files(tmp_path)
    frame.iloc[:5].to_parquet(supplement,index=False)
    with pytest.raises(ValueError,match='>=5'):
        repair(primary,[supplement],tmp_path/'out')


def test_precision_difference_requires_explicit_source_and_both_bounds(tmp_path):
    primary,supplement,frame=source_files(tmp_path)
    frame[['open','close']] *= 100
    frame.drop(index=4).to_parquet(primary,index=False)
    frame.loc[1,'open'] += 0.04
    frame.to_parquet(supplement,index=False)
    with pytest.raises(ValueError,match='convention differs'):
        repair(primary,[supplement],tmp_path/'out')
    repair(primary,[supplement],tmp_path/'out',[supplement])
    proof=json.loads((tmp_path/'out'/'repair_manifest.json').read_text())
    assert proof['supplements'][0]['nonexact_price_cells']==1
    actual=pd.read_parquet(tmp_path/'out'/'repaired_indices.parquet')
    assert actual.loc[1,'open']==10100
    frame.loc[1,'open'] += 0.02
    frame.to_parquet(supplement,index=False)
    with pytest.raises(ValueError,match='convention differs'):
        repair(primary,[supplement],tmp_path/'other',[supplement])
    frame.loc[1,['open','close']]=[100,101]
    frame.drop(index=4).to_parquet(primary,index=False)
    frame.loc[1,'open']+=0.01
    frame.to_parquet(supplement,index=False)
    with pytest.raises(ValueError,match='convention differs'):
        repair(primary,[supplement],tmp_path/'other',[supplement])

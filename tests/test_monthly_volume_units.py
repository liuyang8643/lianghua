import gzip,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ai.supervised import monthly_data
from utils.atomic_file import file_sha256


def test_monthly_adapter_equivalent_share_and_lot_bars(tmp_path,monkeypatch):
    """The actual producer must give equal turnover to equal traded shares."""
    root=tmp_path/'study';root.mkdir();source=tmp_path/'source';source.mkdir()
    dates=np.arange(np.datetime64('2013-12-01'),np.datetime64('2014-02-01'))
    codes=('000001.SZ','000002.SZ','000003.SZ');n=len(dates)
    volume=np.full((n,3),100.);volume[:,1]=10000.;volume[::2,2]=10000.
    fields={'volume':volume,'amount':np.full((n,3),100000.),'low':np.full((n,3),9.),'high':np.full((n,3),11.),'close':np.full((n,3),10.),'listing_age':np.full((n,3),1000),'delisted_mask':np.zeros((n,3),bool),'st_mask':np.zeros((n,3),bool),'issue_date':np.full(3,np.datetime64('2010-01-01'))}
    rt=SimpleNamespace(trade_dates=dates,stock_codes=codes,manifest=SimpleNamespace(source_sha256='fixture'),field=fields.__getitem__)
    for code in codes:
        folder=source/code;folder.mkdir();(folder/'shares.raw.gz').write_bytes(gzip.compress(b'fixture'))
        np.savez(folder/'flow.npz',dates=dates,fields=np.array(['fixture']),values=np.ones((n,1)))
        (folder/'receipt.json').write_text(json.dumps({'files_sha256':{name:file_sha256(folder/name) for name in ['shares.raw.gz','flow.npz']}}))
    (source/'terminal.json').write_text(json.dumps({'all_processed':True}))
    (root/'protocol.json').write_text(json.dumps({'monthly_source_root':str(source),'monthly_signal_path':str(root/'monthly/monthly.npz'),'data_end':'2014-01-31'}))
    (root/'active_data.json').write_text(json.dumps({'runtime_path':'fixture','runtime_sha256':'fixture'}))
    import scripts.run_research_guarded as guard
    monkeypatch.setattr(guard,'require_guarded_job',lambda:None)
    monkeypatch.setattr(monthly_data,'load_runtime_slice',lambda *a,**k:rt)
    monkeypatch.setattr(monthly_data,'parse_share_structure',lambda *a:None)
    monkeypatch.setattr(monthly_data,'align_float_shares',lambda *a:(np.full(n,1e6),np.ones(n,bool)))
    monkeypatch.setattr(monthly_data,'align_main_flow',lambda *a:(np.full(n,1e4),np.full(n,2e4),{'cross_provider_zero_conflict_dates':[]}))
    monthly_data.run(root)
    with np.load(root/'monthly/monthly.npz') as z:
        for k in range(3):np.testing.assert_allclose(z['scores'][:,:,k],z['scores'][:,:,0],rtol=0,atol=0)
        assert np.isclose(z['scores'][0,0,0],-.01)
    config=json.loads((root/'protocol.json').read_text())
    assert monthly_data.verified_monthly_path(config)==root/'monthly/monthly.npz'
    manifest=json.loads((root/'monthly/manifest.json').read_text());manifest['implementation_sha256'].pop('offline_data/volume_units.py')
    (root/'monthly/manifest.json').write_text(json.dumps(manifest))
    import pytest
    with pytest.raises(ValueError,match='implementation changed'):monthly_data.verified_monthly_path(config)

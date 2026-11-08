"""Post-selection fold contributions from the actual continuous OOF account."""
import csv,json
from pathlib import Path
import numpy as np
from env.metrics import performance_from_log_rewards
from utils.atomic_file import atomic_write_json,file_sha256
root=Path(__file__).resolve().parent
assert (root/'search/winner.json').exists(),'Do not analyze selected strategy before selection is sealed'
record=json.loads((root/'final/oof.json').read_text(encoding='utf8'));path=root/'final/oof.npz'
assert file_sha256(path)==record['trace_sha256']
rows=[]
with np.load(path) as z:
    decisions=z['decision_dates'].astype(str);next_dates=z['next_dates'].astype(str);nav=z['nav'];returns=z['returns'];years=np.array([int(x[:4]) for x in decisions])
    assert set(years)==set(json.loads((root/'protocol.json').read_text(encoding='utf8'))['fold_years'])
    np.testing.assert_array_equal(next_dates[:-1],decisions[1:])
    reconstructed=1.
    for year in sorted(set(years)):
        index=np.flatnonzero(years==year);assert np.array_equal(index,np.arange(index[0],index[-1]+1))
        a,b=index[0],index[-1]+1;m=performance_from_log_rewards(np.log1p(returns[a:b])).as_dict()
        np.testing.assert_allclose(m['total_return'],nav[b]/nav[a]-1,rtol=1e-10,atol=1e-12)
        reconstructed*=1+m['total_return']
        rows.append({'fold':f'fold{year}','decision_start':decisions[a],'last_decision':decisions[b-1],'last_mark':next_dates[b-1],'starting_nav':float(nav[a]),'ending_nav':float(nav[b]),**m})
    np.testing.assert_allclose(reconstructed,nav[-1]/nav[0],rtol=1e-10,atol=1e-12)
with (root/'final/oof_years.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
atomic_write_json(root/'oof_year_export_receipt.json',{'state':'pass','folds':len(rows),'nav_source_sha256':file_sha256(path),'csv_sha256':file_sha256(root/'final/oof_years.csv'),'scope':'Grouped by decision/model year; last mark can be first open of next year. Each segment starts with actual carried NAV, not a reset. Segment drawdown uses its local peak; selection uses full continuous-account Calmar, not averaged segment Calmar.','products_match_full_nav':True})
print({'oof_years':len(rows),'compound_nav_consistent':True})

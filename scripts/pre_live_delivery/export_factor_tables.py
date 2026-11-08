"""Read-only export of completed selections; never influences selection/training."""
import csv,json
from pathlib import Path
import numpy as np
from utils.atomic_file import atomic_write_json,file_sha256
root=Path(__file__).resolve().parent;rows=[];sources={}
for p in sorted((root/'fold_selection').glob('*/complete.json')):
    record=json.loads(p.read_text(encoding='utf8'));folder=p.parent;sources[str(p.relative_to(root))]=file_sha256(p)
    with np.load(folder/'correlation.npz') as z:
        names=z['names'].astype(str);lookup={n:i for i,n in enumerate(names)};corr=z['correlation'];days=z['valid_days']
        for h in [3,7,15,30,60]:
            q=folder/f'selected_{h}d.json';selection=json.loads(q.read_text(encoding='utf8'));sources[str(q.relative_to(root))]=file_sha256(q)
            with (folder/f'selected_{h}d.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['name','rank_ic','valid_days']);writer.writeheader();writer.writerows(selection['selected'])
            ii=[lookup[r['name']] for r in selection['selected']];matrix=corr[np.ix_(ii,ii)];n=len(ii)
            triangle=matrix[np.triu_indices(n,1)];count=days[np.ix_(ii,ii)][np.triu_indices(n,1)]
            assert np.isfinite(triangle).all() and (np.abs(triangle)<.8).all() and (count>=60).all()
            rows.append({'fold':folder.name,'fit_start':selection['fit'][0],'fit_end':selection['fit'][1],'last_feature_date':selection['last_feature_date'],'horizon':h,'selected_count':n,'union_count':len(names),'max_abs_mean_correlation':float(np.max(np.abs(triangle))) if len(triangle) else None,'minimum_pair_days':int(count.min()) if len(count) else None})
with (root/'fold_selection/summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
outputs={'fold_selection/summary.csv':file_sha256(root/'fold_selection/summary.csv')}
final=root/'fold_selection/final'
if (final/'complete.json').exists():
    record=json.loads((final/'complete.json').read_text(encoding='utf8'));names=record['features']
    inventory=json.loads((root/'features/inventory.json').read_text(encoding='utf8'))
    dates=np.load(root/'features/dates.npy');selected={};stats={};ic_sources={}
    for h in [3,7,15,30,60]:
        selection=json.loads((final/f'selected_{h}d.json').read_text(encoding='utf8'));selected[h]={x['name']:x for x in selection['selected']}
        mask=(dates>=np.datetime64(selection['fit'][0]))&(dates<=np.datetime64(selection['last_feature_date']))
        for name in names:
            source=inventory[name]['daily_ic'][str(h)];p=Path(source['path']);actual=file_sha256(p);assert actual==source['sha256']
            values=np.load(p)[mask,1];values=values[np.isfinite(values)];n=len(values);mean=float(values.mean()) if n>=60 else None
            stats[name,h]=(mean,n);ic_sources[str(p)]=actual
            if name in selected[h]:
                s=selected[h][name];assert n==s['valid_days'] and np.isclose(mean,s['rank_ic'],rtol=0,atol=1e-12)
    fields=['name','selected_horizons']+[f'{kind}_{h}d' for h in [3,7,15,30,60] for kind in ['rank_ic','valid_days','selected']]
    with (final/'union_factors.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name in names:
            row={'name':name,'selected_horizons':';'.join(str(h) for h in selected if name in selected[h])}
            for h in selected:row.update({f'rank_ic_{h}d':stats[name,h][0],f'valid_days_{h}d':stats[name,h][1],f'selected_{h}d':int(name in selected[h])})
            writer.writerow(row)
    with np.load(final/'correlation.npz') as z:
        np.testing.assert_array_equal(z['names'].astype(str),names);matrix=z['correlation'];counts=z['valid_days']
        for filename,values in [('correlation.csv',matrix),('correlation_valid_days.csv',counts)]:
            with (final/filename).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f);writer.writerow(['factor',*names]);writer.writerows([[name,*values[i]] for i,name in enumerate(names)])
        pairs=[]
        for i,j in zip(*np.triu_indices(len(names),1)):
            value=float(matrix[i,j]);shared=[h for h in selected if names[i] in selected[h] and names[j] in selected[h]]
            if shared:assert np.isfinite(value) and abs(value)<.8 and counts[i,j]>=60
            pairs.append({'factor_a':names[i],'factor_b':names[j],'mean_daily_spearman':value,'abs_correlation':abs(value),'valid_days':int(counts[i,j]),'shared_selected_horizons':';'.join(map(str,shared))})
        pairs.sort(key=lambda x:-(x['abs_correlation'] if np.isfinite(x['abs_correlation']) else -1))
        with (final/'correlation_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    for name in ['union_factors.csv','correlation.csv','correlation_valid_days.csv','correlation_pairs.csv']:
        outputs[str((final/name).relative_to(root))]=file_sha256(final/name)
    atomic_write_json(final/'export_receipt.json',{'state':'exported_saved_training_evidence','factors':len(names),'pairs':len(pairs),'pairs_abs_ge_08':sum(x['abs_correlation']>=.8 for x in pairs),'scope':'Each horizon is separately deduplicated. High-correlation pairs across the horizon union are retained as approved; exports do not change selection or training.','selection_sha256':file_sha256(final/'complete.json'),'correlation_npz_sha256':file_sha256(final/'correlation.npz'),'ic_sources':ic_sources,'outputs':outputs})
atomic_write_json(root/'factor_export_receipt.json',{'state':'exported_completed_prefixes','fold_count':len(rows)//5,'sources':sources,'outputs':outputs,'summary_sha256':file_sha256(root/'fold_selection/summary.csv'),'scope':'Each horizon separately deduplicated; cross-horizon union can still contain high correlations.'})
print({'folds_exported':len(rows)//5,'rows':len(rows)})

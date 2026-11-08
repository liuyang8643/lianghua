"""Read-only arithmetic audit of saved search results and frozen selection."""
import json,gzip,hashlib,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf8'))
records=[read(p) for p in sorted((R/'trials').glob('*.json'))]
checks=[]
for r in records:
    if r['state']!='complete':continue
    path=R/'trials'/f"{r['id']:02d}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['trace_sha256']
    with np.load(path) as z:
        nav=z['nav'];ret=z['returns'];np.testing.assert_allclose(ret,nav[1:]/nav[:-1]-1,atol=1e-12)
        annual=float(math.expm1(math.log(float(nav[-1]/nav[0]))*252/len(ret)));dd=float(max(1-nav/np.maximum.accumulate(nav)));calmar=annual/max(dd,1e-6)
        for k,v in [('annualized_return',annual),('max_drawdown',dd),('calmar',calmar)]:assert math.isclose(v,r['metrics'][k],rel_tol=1e-9,abs_tol=1e-10),(r['id'],k)
        with gzip.open(path.with_suffix('.diagnostics.json.gz'),'rt',encoding='utf8') as f:d=json.load(f)
        assert len(d)==len(ret) and all(x['status'] in ['Solved','AlmostSolved'] for x in d)
        assert max(x['max_constraint_violation'] for x in d)<=2e-6
        checks.append({'id':r['id'],'dates':len(ret),'calmar':calmar,'annual_rows':len(r['annual']),'drift_over_cap_days':sum(x.get('drift_over_cap',0)>0 for x in d)})
w=read(R/'winner.json');valid=[r for r in records if r['state']=='complete']
key=lambda r:(r['metrics']['annualized_return']>0,r['metrics']['calmar'] if r['metrics']['annualized_return']>0 else r['metrics']['annualized_return'],-r['metrics']['mean_gross_turnover'],-r['id'])
assert max(valid,key=key)==w['winner'];assert len(records)==w['trials']
assert read(R/'complete.json')['winner_sha256']==hashlib.sha256((R/'winner.json').read_bytes()).hexdigest()
result={'state':'pass','scope':'saved full OOF trace arithmetic, solver residual receipts, fixed selection; not a second solver rerun for every market day','checks':checks,'invalid_trials':[r['id'] for r in records if r['state']!='complete'],'winner':w['winner']['id'],'full_grid':w['complete_grid']}
(R/'result_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))

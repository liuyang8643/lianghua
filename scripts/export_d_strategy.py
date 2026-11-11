"""Build the approved frozen D portable release without fitting or broker actions."""
from pathlib import Path
import hashlib,json,shutil,zipfile
REPO=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 root=REPO/'artifacts/research/d_supervised_20261110/D';model=root/'models/D';target=REPO/'strategy_release/d_model_topk_20261111';target.mkdir(exist_ok=True)
 done=read(model/'complete.json');meta=read(model/'model.json');assert done['state']=='complete' and done['reload_parity'];assert sha(model/'model.txt')==done['model_sha256'];assert meta['best_iteration']==583 and len(meta['feature_names'])==278
 losses=meta['evaluation']['valid']['l2'];assert len(losses)==783 and min(range(len(losses)),key=losses.__getitem__)+1==583
 assert sha(REPO/'research/d_supervised_20261110/model.py')==read(model/'identity.json')['source']
 for name in ('model.txt','model.json'):shutil.copyfile(model/name,target/name)
 prior=REPO/'strategy_release/new_model_topk_20261108';shutil.copyfile(prior/'QLIB_LICENSE.txt',target/'QLIB_LICENSE.txt')
 inventory=read(root/'features/inventory.json');names=meta['feature_names'];selected={str(h):read(root/f'fold_selection/D/selected_{h}d.json')['selected'] for h in (3,7,15,30,60)}
 template=read(prior/'features.json');features={'feature_names':names,'normalization':{n:inventory[n]['normalization'] for n in names},'factors':{n:{'name':n,'rank_ic_by_horizon':{h:next(x['rank_ic'] for x in rows if x['name']==n) for h,rows in selected.items() if any(x['name']==n for x in rows)}} for n in names},'timing':template['timing'],'universe':template['universe']};write(target/'features.json',features)
 assert set(names)=={x['name'] for rows in selected.values() for x in rows};write(target/'factor_selection.json',{'fit':['2005-01-01','2025-12-31'],'horizons':selected,'correlation_threshold':.8,'selection_receipt_sha256':sha(root/'fold_selection/D/complete.json')})
 # Retain the existing reviewed feature-provider source snapshot; verify every research feature implementation.
 with zipfile.ZipFile(prior/'research_source.zip') as z:payload={n:z.read(n) for n in z.namelist()}
 sources=read(REPO/'artifacts/research/multihorizon_pre_live_20261108_volume_fix/features/identity.json')['factor_sources'];checked={}
 for name,expected in sources.items():
  p=Path(name);p=p if p.is_absolute() else REPO/p
  relative=p.relative_to(REPO).as_posix()
  if relative not in payload:continue
  assert sha(p)==expected and hashlib.sha256(payload[relative]).hexdigest()==expected;checked[relative]=expected
 for relative in ('ai/supervised/model.py','ai/supervised/multihorizon_model.py','scripts/run_research_guarded.py'):
  payload[relative]=(REPO/relative).read_bytes()
 for name in ('__init__.py','prepare.py','extend_ic.py','select_d.py','prepare_training.py','model.py','train.py'):
  relative='research/d_supervised_20261110/'+name;payload[relative]=(REPO/relative).read_bytes()
 payload['research/__init__.py']=b''
 with zipfile.ZipFile(target/'research_source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for n,data in sorted(payload.items()):
   info=zipfile.ZipInfo(n,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
 write(target/'research_source_manifest.json',{n:hashlib.sha256(v).hexdigest() for n,v in sorted(payload.items())})
 endpoint=read(root/'endpoint_1110.json');write(target/'training_provenance.json',{'spec':done['spec'],'actual_rounds':783,'best_iteration':583,'max_rounds':2000,'early_stopping_rounds':200,'best_valid_l2':min(losses),'model_sha256':done['model_sha256'],'training_identity_sha256':sha(model/'identity.json'),'label_endpoint':endpoint,'feature_source_sha256':checked,'notes':['2026usedearlystop;historicalcomparisonsalreadyobserved','Feature-provider archive requires original local data to reproduce training; no raw data packaged','Fixed score inference usescomplete eligiblecrosssection, not holdings-only ranking']})
 study=REPO/'artifacts/research/d_40_turnover2_20261111';results=read(study/'annual.json');write(target/'evaluation_summary.json',{'parameters':{'buy_n':40,'sell_m':40,'hold_thresh':3,'max_replacements':2},'annual':{y:v['h3_r2'] for y,v in results.items()},'source_sha256':sha(study/'annual.json'),'basis':'Independent cash start eachyear;2026 endsNov10open.2005-2025fit,2026earlystop;not new blind evidence. Includes25bp one-way slippage and nativefees.'})
 manifest=read(prior/'manifest.json');manifest.update(release_id='d_model_topk_20261111',display_name='D模型 + TopK 40/40，最短3日，每日最多换出2只',weights=[.2]*5,parameters={'buy_n':40,'sell_m':40,'hold_thresh':3,'max_replacements':2},training_interval=['2005-01-01','2025-12-31'],early_stopping_interval=['2026-01-01','2026-11-10'],label_endpoint='2026-11-10 open',features=278,trees=583,actual_training_rounds=783)
 manifest['files']={p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file() and p.name!='manifest.json'};write(target/'manifest.json',manifest)
 write(REPO/'configs/default_supervised_strategy.json',{'strategy':'new_model_topk','bundle':'strategy_release/d_model_topk_20261111','manifest_sha256':sha(target/'manifest.json'),'mvo_enabled':False})
 print({'release':str(target),'model_sha256':done['model_sha256'],'features':len(names),'feature_sources_verified':len(checked),'parameters':manifest['parameters']})
if __name__=='__main__':main()

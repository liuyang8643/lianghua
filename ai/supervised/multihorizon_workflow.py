"""Resume the approved stages with immutable per-stage source and input receipts.

Run only through scripts/run_research_guarded.py. Incomplete model fits fail
closed; CPU factor/selection and GA checkpoints resume with identical inputs.
"""
import argparse,json,shutil,sys,time,traceback,os
os.environ['NUMBA_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from utils.atomic_file import atomic_write_json,file_sha256

def wait_for_data(root):
    while True:
        status=root/'features/status.json'
        if status.exists() and json.loads(status.read_text()).get('state')=='complete':return
        if (root/'STOP').exists():raise RuntimeError('STOP requested while awaiting data')
        log=root/'data_logs/guard.jsonl'
        if log.exists():
            lines=log.read_text(encoding='utf8').splitlines()
            for line in reversed(lines):
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                if event.get('event')=='child_exit':raise RuntimeError('data process exited before complete receipt: '+line)
        time.sleep(10)

def run(root,start='selection',wait_data=False):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    if wait_data:
        inputs={str(p):file_sha256(p) for p in Path('ai/supervised').glob('*.py')}
        atomic_write_json(root/'waiting_source_manifest.json',inputs)
        wait_for_data(root)
        for p,sha in inputs.items():assert file_sha256(p)==sha,'source changed while waiting: '+p
    from . import feature_data,multihorizon_select,multihorizon_model,multihorizon_search,multihorizon_finish,multihorizon_static
    stages=[('data',lambda:feature_data.run(root),root/'features/status.json'),
        ('selection',lambda:multihorizon_select.run(root),root/'fold_selection/status.json'),
        ('fold_models',lambda:multihorizon_model.run(root),root/'models/oof_complete.json'),
        ('search',lambda:multihorizon_search.run(root),root/'search/winner.json'),
        ('final_model',lambda:multihorizon_model.run(root,final=True),root/'models/final/complete.json'),
        ('observations',lambda:multihorizon_finish.run(root),root/'final/complete.json'),
        ('static_baseline',lambda:multihorizon_static.run(root),root/'static_baseline/complete.json')]
    enabled=False
    for name,action,receipt in stages:
        enabled=enabled or name==start
        if not enabled:continue
        stage=root/'stages'/name;stage.mkdir(parents=True,exist_ok=True)
        manifest={str(p):file_sha256(p) for d in ['ai/supervised','factor','factor_db','offline_data','env','utils','configs'] for p in Path(d).rglob('*.py')}
        report_sources=[Path('testback/backtest.py'),Path('testback/economic_positions.py'),Path('testback/__init__.py'),*Path('testback/reportor').rglob('*.py'),*Path('testback/reportor/templates').glob('*.js'),*Path('testback/reportor/templates').glob('*.css')]
        for p in report_sources:manifest[str(p)]=file_sha256(p)
        for p in [Path('factor/library/qlib_catalog.json'),Path('scripts/run_research_guarded.py')]:manifest[str(p)]=file_sha256(p)
        manifest['protocol.json']=file_sha256(root/'protocol.json')
        path=stage/'inputs.json'
        if path.exists():assert json.loads(path.read_text())==manifest,'stage implementation changed; preserve prior attempt before recovery'
        else:
            atomic_write_json(path,manifest)
            for p in [Path(k) for k in manifest if k!='protocol.json']:
                dest=stage/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        if receipt.exists() and json.loads(receipt.read_text(encoding='utf8')).get('state') in ['complete','qualified','no_qualified_candidate']:
            if (stage/'complete.json').exists():
                sealed=json.loads((stage/'complete.json').read_text());assert sealed['output_sha256']==file_sha256(receipt)
            elif name=='data':
                from .multihorizon_admission import verify_data
                verify_data(root)
                producer=json.loads((root/'features/identity.json').read_text());assert producer['source']==file_sha256(producer.get('source_path','ai/supervised/feature_data.py'))
                atomic_write_json(stage/'complete.json',{'state':'complete','external_stage':True,'output_sha256':file_sha256(receipt),'inputs_sha256':file_sha256(path)})
            else:raise RuntimeError('stage outputs exist without successful post-check receipt; audit required: '+name)
            continue
        t=time.monotonic();atomic_write_json(root/'task_state.json',{'state':'running','stage':name,'started_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
        try:action()
        except BaseException as e:
            atomic_write_json(stage/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'seconds':time.monotonic()-t})
            atomic_write_json(root/'task_state.json',{'state':'failed','stage':name,'error':repr(e)});raise
        for p,sha in manifest.items():assert file_sha256(root/p if p=='protocol.json' else p)==sha,p
        atomic_write_json(stage/'complete.json',{'state':'complete','seconds':time.monotonic()-t,'output_receipt':str(receipt),'output_sha256':file_sha256(receipt),'inputs_sha256':file_sha256(path)})
    multihorizon_finish.report(root)
    config=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    main_split='validation' if config.get('validation') else 'train'
    detailed=root/'reports'/f'ml_{main_split}'/'single_report.html'
    assert detailed.exists(),'Native detailed report missing'
    atomic_write_json(root/'task_state.json',{'state':'awaiting_independent_final_review','report':str(detailed),'summary_report':str(root/'final/report.html')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--start',default='selection',choices=['data','selection','fold_models','search','final_model','observations','static_baseline']);p.add_argument('--wait-for-data',action='store_true');a=p.parse_args();run(a.root,a.start,a.wait_for_data)

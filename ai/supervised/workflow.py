"""Run the committed research stages sequentially under the existing job guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from utils.atomic_file import atomic_write_json,file_sha256


def run(root,start,wait_feature_guard=None):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();root=root.resolve();begun=time.monotonic()
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('selection_split')!='train':raise ValueError('this workflow now requires training-only selection; legacy validation studies are archived')
    if json.loads((root/'selection/status.json').read_text())['state']!='complete':raise ValueError('factor selection is not complete')
    stages=['feature_data','train','search','final_evaluation']
    if start not in stages:raise ValueError('requested stage is not configured for this study')
    stages=stages[stages.index(start):]
    directory=root/'workflow';directory.mkdir(exist_ok=True)
    if wait_feature_guard is not None:
        if start!='train':raise ValueError('waiting for features is only valid before training')
        atomic_write_json(directory/'status.json',{'state':'waiting_for_features','feature_guard':str(wait_feature_guard)})
        while True:
            if (directory/'STOP').exists():raise RuntimeError('workflow STOP')
            events=[json.loads(line) for line in wait_feature_guard.read_text().splitlines() if line.strip()]
            last=events[-1] if events else {}
            if last.get('event') in ('stopped_for_game','blocked_before_launch','blocked_before_resume'):
                raise RuntimeError('feature preparation was stopped by its guard')
            if last.get('event')=='child_exit':
                if last['returncode']!=0 or json.loads((root/'features/status.json').read_text())['state']!='complete':
                    raise RuntimeError('feature preparation did not complete successfully')
                break
            time.sleep(5)
    environment=dict(os.environ);environment['PYTHONPATH']=protocol.get('research_packages_dir',str(root/'packages'))+os.pathsep+str(Path.cwd())
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:environment[key]='1'
    for stage in stages:
        if (directory/'STOP').exists():raise RuntimeError('workflow STOP')
        command=[sys.executable,'-u','-m','ai.supervised.'+stage,'--root',str(root)]
        atomic_write_json(directory/'status.json',{'state':'running','stage':stage,'command':command,'elapsed_seconds':time.monotonic()-begun})
        print({'event':'stage_start','stage':stage},flush=True)
        prefix=directory/stage
        if prefix.with_suffix('.stdout.log').exists():raise FileExistsError('stage log exists; inspect prior outcome before an explicit resume')
        with prefix.with_suffix('.stdout.log').open('wb') as stdout,prefix.with_suffix('.stderr.log').open('wb') as stderr:
            result=subprocess.run(command,env=environment,stdout=stdout,stderr=stderr)
        atomic_write_json(prefix.with_suffix('.receipt.json'),{'returncode':result.returncode,'elapsed_seconds':time.monotonic()-begun,
            'source_sha256':file_sha256(Path('ai/supervised')/(stage+'.py'))})
        if result.returncode:
            atomic_write_json(directory/'status.json',{'state':'failed','stage':stage,'returncode':result.returncode})
            raise RuntimeError('stage failed: '+stage)
        print({'event':'stage_complete','stage':stage},flush=True)
    atomic_write_json(directory/'status.json',{'state':'complete_pending_audit','elapsed_seconds':time.monotonic()-begun})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--start',choices=['feature_data','train','search','final_evaluation'],default='feature_data')
    p.add_argument('--wait-feature-guard',type=Path)
    args=p.parse_args();run(args.root,args.start,args.wait_feature_guard)

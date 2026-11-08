"""Explicitly stop only the verified local planner, never the brokerage app."""
import argparse,json
from pathlib import Path
import sys
import psutil


def _option(args,name,default):
    values=[]
    for index,value in enumerate(args):
        if value==name:
            if index+1>=len(args):raise ValueError('Missing planner argument')
            values.append(args[index+1])
        elif value.startswith(name+'='):values.append(value.split('=',1)[1])
    if len(values)>1:raise ValueError('Ambiguous planner arguments')
    return values[0] if values else default


def stop_planner(root,*,processes=None,process_factory=psutil.Process):
    root=Path(root).resolve();directory=root/'data/live_file'
    status=directory/'planner_status.json'
    if status.exists() and json.loads(status.read_text(encoding='utf8')).get('stage') in (
            'checking_history','preparing_completed_factors','calculating_target'):
        raise RuntimeError('Planner is busy calculating or updating; retry after the current stage finishes')
    allowed={Path(sys.executable).resolve(),Path(sys._base_executable).resolve(),root/'.venv/Scripts/python.exe'}
    candidates=[]
    for process in psutil.process_iter() if processes is None else processes:
        try:
            args=process.cmdline()
            if '-m' not in args or args[args.index('-m')+1] not in ('trading.planner_watchdog','trading.main'):continue
            if any(value.split('=',1)[0] in ('--skip','--dry-run','--confirm','--individual-config') for value in args):
                continue
            if Path(process.cwd()).resolve()!=root or Path(process.exe()).resolve() not in allowed:continue
            config=Path(_option(args,'--config','configs/config.json'))
            folder=Path(_option(args,'--directory','data/live_file'))
            if not config.is_absolute():config=root/config
            if not folder.is_absolute():folder=root/folder
            if config.resolve()!=root/'configs/config.json' or folder.resolve()!=directory:continue
            candidates.append((process.pid,process.create_time(),args))
        except (psutil.NoSuchProcess,psutil.AccessDenied,IndexError,ValueError):
            continue
    stopped=[]
    for pid,created,args in sorted(candidates,key=lambda row:row[1],reverse=True):
        try:
            process=process_factory(pid)
            if not process.is_running():continue
            if (abs(process.create_time()-created)>.01 or process.cmdline()!=args
                    or Path(process.cwd()).resolve()!=root or Path(process.exe()).resolve() not in allowed):
                raise RuntimeError('Planner identity changed before stop; refusing reused PID')
            process.terminate();process.wait(timeout=10);stopped.append(pid)
        except psutil.NoSuchProcess:
            continue
    return stopped


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(dict(stopped_planner_pids=stop_planner(args.root))))

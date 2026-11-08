"""Serial, throttled archive for the local dividend-factor supplement."""
import argparse
import json
from pathlib import Path
import pandas as pd
from data.update_eastmoney_research_common import em_get
from utils.atomic_file import atomic_write_json,file_sha256

FIELDS=('SECUCODE','SECURITY_CODE','PRETAX_BONUS_RMB','EX_DIVIDEND_DATE','EQUITY_RECORD_DATE',
        'PLAN_NOTICE_DATE','NOTICE_DATE','PUBLISH_DATE','REPORT_DATE','ASSIGN_PROGRESS','IMPL_PLAN_PROFILE','TOTAL_SHARES')


def fetch(output, *, start='2010-01-01', end='2026-10-31'):
    from datetime import date
    if date.fromisoformat(start)>date.fromisoformat(end):raise ValueError('invalid dividend interval')
    output.mkdir(parents=True,exist_ok=True)
    request={'reportName':'RPT_SHAREBONUS_DET','columns':','.join(FIELDS),
             'filter':f"(EX_DIVIDEND_DATE>='{start}')(EX_DIVIDEND_DATE<='{end}')",
             'pageSize':'500','sortColumns':'EX_DIVIDEND_DATE,SECURITY_CODE','sortTypes':'1,1','source':'WEB','client':'WEB'}
    record=output/'request.json'
    if record.exists() and json.loads(record.read_text(encoding='utf-8'))!=request:raise ValueError('dividend request changed')
    atomic_write_json(record,request)
    page=1;pages=None;all_rows=[];hashes={}
    while pages is None or page<=pages:
        path=output/f'page_{page:04d}.json'
        if path.exists():payload=json.loads(path.read_text(encoding='utf-8'))
        else:
            response=em_get({**request,'pageNumber':str(page)});response.raise_for_status();payload=response.json()
            if not payload.get('success'):raise RuntimeError(str(payload.get('message')))
            atomic_write_json(path,payload)
        result=payload.get('result') or {};pages=int(result.get('pages') or 0)
        if pages<1:
            if result.get('count')==0 and result.get('data') in (None,[]):
                hashes[str(page)]=file_sha256(path);break
            raise ValueError('ambiguous empty dividend source')
        all_rows.extend(result['data']);hashes[str(page)]=file_sha256(path)
        atomic_write_json(output/'status.json',{'state':'fetching','page':page,'pages':pages,'rows':len(all_rows)})
        if page%10==0 or page==pages:print({'page':page,'pages':pages,'rows':len(all_rows)},flush=True)
        page+=1
    if len(all_rows)!=int(result['count']):raise ValueError('dividend count mismatch')
    frame=pd.DataFrame(all_rows,columns=list(FIELDS));frame.to_parquet(output/'events.parquet',index=False)
    atomic_write_json(output/'status.json',{'state':'complete','pages':pages,'rows':len(frame),'response_sha256':hashes,
                                         'parquet_sha256':file_sha256(output/'events.parquet')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--start',default='2010-01-01');p.add_argument('--end',default='2026-10-31');a=p.parse_args();fetch(a.output,start=a.start,end=a.end)

"""Bounded Tencent unadjusted quote check; never replaces research inputs."""
import datetime,json,time
from pathlib import Path
from urllib.parse import quote
import requests
import numpy as np
from scripts.run_research_guarded import require_guarded_job
from utils.atomic_file import atomic_write_json,file_sha256
from delivery_inputs import admit,bind,verify,read

require_guarded_job();root=Path(__file__).resolve().parent;c,bindings,_=admit(root)
source=root/'extreme_audit/ml_validation.json';bind(bindings,source);bind(bindings,Path(__file__))
extremes=[d for d in read(source)['selected_extreme_days'] if d['highest_gain_day'] or d['worst_day']]
expected={}
for day in extremes:
    for stock in day['stocks']:
        code=stock['code'];target=expected.setdefault(code,{})
        for date,values in [(day['decision_date'],{k:stock[k] for k in ('open','high','low','close')}),(day['settlement_date'],{'open':stock['next_open']})]:
            row=target.setdefault(date,{})
            for field,value in values.items():
                if field in row:assert row[field]==value
                row[field]=value
out=root/'public_extreme_quotes';out.mkdir(exist_ok=True);(out/'raw').mkdir(exist_ok=True)
request={'source':'Tencent fqkline/get day with adjustment none','expected':expected,'purpose':'Post-hoc maximum/minimum validation-day quote validation, no training input replacement','skill':'D:/coding/WBR/.agents/skills/a-stock-data/SKILL.md'}
if (out/'request.json').exists():assert read(out/'request.json')==request
else:atomic_write_json(out/'request.json',request)
checks=[];errors=[];sources={};started=time.monotonic();session=requests.Session()
try:
    for n,(code,points) in enumerate(sorted(expected.items()),1):
        bare,market=code.split('.');symbol=market.lower()+bare;dates=sorted(points)
        # Split distant endpoints to avoid silently dropping them under640bars.
        ranges=[dates] if (datetime.date.fromisoformat(dates[-1])-datetime.date.fromisoformat(dates[0])).days<700 else [[d] for d in dates]
        received={}
        for group in ranges:
            start,end=group[0],group[-1];path=out/'raw'/f'{code}_{start}_{end}.json';url='https://ifzq.gtimg.cn/appstock/app/fqkline/get?param='+quote(f'{symbol},day,{start},{end},640,none',safe=',')
            try:
                if path.exists():payload=read(path)
                else:
                    response=session.get(url,timeout=(10,25),headers={'User-Agent':'Mozilla/5.0 WBR-readonly-quote-audit/1','Referer':'https://gu.qq.com/'})
                    response.raise_for_status();path.write_bytes(response.content);payload=response.json()
                assert payload.get('code',0)==0,payload.get('msg');rows=(payload.get('data') or {}).get(symbol,{}).get('day');assert isinstance(rows,list) and rows,'No unadjusted day rows'
                sources[str(path.relative_to(out))]={'url':url,'sha256':file_sha256(path),'rows':len(rows)}
                received.update({str(row[0]):{'open':float(row[1]),'close':float(row[2]),'high':float(row[3]),'low':float(row[4])} for row in rows})
            except Exception as exc:errors.append({'code':code,'start':start,'end':end,'url':url,'error':repr(exc)})
            time.sleep(.2)
        for date,values in points.items():
            for field,value in values.items():
                if value is None:continue
                actual=received.get(date,{}).get(field);matched=actual is not None and abs(actual-value)<=max(.0051,abs(value)*2e-7)
                checks.append({'code':code,'date':date,'field':field,'snapshot':value,'tencent_unadjusted':actual,'matched':bool(matched)})
        if n%10==0:print({'quote_codes':n,'total':len(expected),'errors':len(errors)},flush=True)
finally:session.close()
verify(bindings);mismatches=[x for x in checks if not x['matched']];atomic_write_json(out/'checks.json',checks)
atomic_write_json(out/'complete.json',{'state':'matched' if not errors and not mismatches else 'needs_review','codes':len(expected),'price_checks':len(checks),'matched':len(checks)-len(mismatches),'mismatches':mismatches,'errors':errors,'extreme_days':[{'decision_date':d['decision_date'],'settlement_date':d['settlement_date'],'return_pct':d['return_pct']} for d in extremes],'sources':sources,'inputs':bindings,'seconds':time.monotonic()-started,'source_data_replaced':False,'model_training_calls':0,'scope':'Additional public unadjusted OHLC only for all stock rows on two extreme validation decisions. Not full-history or auction-execution certification.'})
print({'state':'matched' if not errors and not mismatches else 'needs_review','codes':len(expected),'checks':len(checks),'mismatches':len(mismatches),'errors':len(errors)},flush=True)

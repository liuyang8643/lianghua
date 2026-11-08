# coding: utf-8
import sys,json
ROOT='D:/coding/WBR-live'
R=ROOT+'/data/qmt_bridge_runtime'
sys.path[:0]=[R+'/stdlib',R,R+'/site-packages']
import encodings
encodings.__path__.append(R+'/stdlib/encodings')
r={'mode':'CLI_DEPENDENCY_PROBE_ONLY','modules':{}}
for name in ['importlib','asyncio','sqlite3','xtquant.xttrader','xtquant.xttype']:
 try:
  m=__import__(name,fromlist=['*']);r['modules'][name]=str(getattr(m,'__file__',''))
 except Exception as e:r['modules'][name]=repr(e)
with open(ROOT+'/tmp/qmt_bridge_20261102/import_probe_result.json','w',encoding='utf-8') as f:json.dump(r,f,indent=2)

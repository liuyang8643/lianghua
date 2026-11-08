import json,gzip,hashlib,ast,time
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from env.metrics import performance_from_log_rewards
from testback.economic_positions import EconomicPositionLedger
from utils.atomic_file import atomic_write_json,file_sha256
from run import annual
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf8"))
a=ast.parse((R/"mvo_adapter.py").read_text(encoding="utf8"));b=ast.parse((R/"mvo_adapter_report.py").read_text(encoding="utf8"));assert ast.dump(next(n for n in a.body if isinstance(n,ast.ClassDef)))==ast.dump(next(n for n in b.body if isinstance(n,ast.ClassDef)))
fa=next(n for n in a.body if isinstance(n,ast.FunctionDef));fb=next(n for n in b.body if isinstance(n,ast.FunctionDef));assert all(ast.dump(x)==ast.dump(y) for x,y in zip(fa.body[:6],fb.body[:6]))
ledger=EconomicPositionLedger();nav=[];returns=[];cash=[];exposure=[];dates=[];next_dates=[];fees=0.;turnover=0.;count=0
with gzip.open(R/"external_steps.jsonl.gz","rt",encoding="utf8") as f:
 for line in f:
  z=json.loads(line);before=z["before"];after=z["after"];info=z["info"]
  ledger.observe(before,after,z["plan"],z["fills"],z["diagnostics"])
  if not nav:nav.append(before["nav"])
  assert before["nav"]==nav[-1];nav.append(after["nav"]);returns.append(info["portfolio_return"]);cash.append(info["cash"]);exposure.append(info["exposure"]);dates.append(info["decision_date"]);next_dates.append(info["next_decision_date"])
  ff=sum(x["fee"] for x in z["fills"]);tt=sum(x["quantity"]*x["price"] for x in z["fills"])/z["plan"]["diagnostics"]["balance_sheet_nav"]
  assert np.isclose(ff,info["total_fees"],atol=1e-7) and np.isclose(tt,info["gross_turnover_ratio"],atol=1e-12);fees+=ff;turnover+=tt;count+=1
P=R.with_name("multihorizon_pre_live_20261108_volume_fix")
with np.load(P/"final/validation.npz") as z:np.testing.assert_array_equal(dates,z["decision_dates"]);np.testing.assert_array_equal(next_dates,z["next_dates"])
np.testing.assert_allclose(returns,np.array(nav[1:])/nav[:-1]-1,atol=1e-12)
np.savez_compressed(R/"external.npz",nav=nav,returns=returns,cash=cash,exposure=exposure,decision_dates=dates,next_dates=next_dates)
m=performance_from_log_rewards(np.log1p(returns)).as_dict();m.update(total_fees=fees,mean_gross_turnover=turnover/count,mean_exposure=float(np.mean(exposure)))
w=read(R/"winner.json");atomic_write_json(R/"external.json",{"metrics":m,"annual":annual(SimpleNamespace(decision_dates=dates,portfolio_returns=np.array(returns))),"ledger":ledger.summary(after),"parameters":w["winner"]["parameters"],"reconstructed_from_original_saved_native_journal":True})
for name,sha in read(R/"inputs.json")["sources"].items():assert file_sha256(name)==sha,name
atomic_write_json(R/"metadata_recovery.json",{"state":"pass_saved_full_path_recovered_pending_replay","rows":count,"ledger":ledger.summary(after),"original_guard_exit":1,"error":"Detailed RolloutTrace lacks summary-only total_fees; financial path already complete","fix":"Only post-rollout statistics accessor; Strategy AST and pre-rollout evaluate body unchanged","journal_sha256":file_sha256(R/"external_steps.jsonl.gz"),"original_adapter_sha256":file_sha256(R/"mvo_adapter.py"),"report_adapter_sha256":file_sha256(R/"mvo_adapter_report.py")})
atomic_write_json(R/"complete.json",{"state":"complete","completion_method":"saved full native account path metadata recovery; original process exit1 preserved; delivery replay still required","seconds":time.time()-read(R/"protocol.json")["created_epoch"],"trials":w["trials"],"valid":w["trials"],"planned":w["planned"],"winner_sha256":file_sha256(R/"winner.json"),"external_sha256":file_sha256(R/"external.npz"),"source_unchanged":True,"no_training":True})
print(m)

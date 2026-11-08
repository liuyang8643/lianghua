from pathlib import Path
import ast,sys
root=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root))
p=Path(__file__).with_name('audit.py')
# Reuse only audited imports/setup/streaming loader, never rerun the ledger loop.
src=p.read_text(encoding='utf-8-sig'); prefix=src.split('calendar=get_trading_date_span(START,END)')[0]
exec(compile(prefix,str(p),'exec'),globals())
from types import SimpleNamespace
args=SimpleNamespace(individual_config=str(ROOT/'configs/config.json'),output_dir=str(OUT/'formal_backtest'),filter_enabled=None)
dts=[datetime.combine(d,datetime.min.time()) for d in get_trading_date_span(START,END)]
result=backtest.run_single_mode(args,{'desc':'migration audit','save_charts':True},dts,runtime.load_runtime_stock_codes(),live_sim=False)
(OUT/'formal_backtest_resources.json').write_text(json.dumps({'loads':loads,'memory':psutil.Process().memory_info()._asdict(),'total_return_pct':result['total_return'],'final_asset':result['final_asset'],'network_calls':0,'minute_price_simulation':False},indent=2),encoding='utf-8')

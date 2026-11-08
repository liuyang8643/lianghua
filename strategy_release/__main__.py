"""Verify the selected release; no data downloads, scheduling, or broker calls."""
import json
from . import load_default

if __name__=='__main__':
    strategy=load_default()
    print(json.dumps({'release':strategy.manifest['release_id'],'method':strategy.manifest['default_method'],'features':len(strategy.names),'trees':strategy.model.num_trees(),'horizons':strategy.manifest['horizons'],'weights':strategy.manifest['weights'],'parameters':strategy.manifest['parameters'],'broker_orders_sent':False},ensure_ascii=False,indent=2))

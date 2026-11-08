"""One output directory convention for GA and static backtests."""
from datetime import datetime
from pathlib import Path

def resolve_output_dir(explicit, mode):
    result = Path(explicit) if explicit else Path('results') / f'{mode}_{datetime.now():%Y%m%d_%H%M%S}'
    result.mkdir(parents=True, exist_ok=True)
    return result

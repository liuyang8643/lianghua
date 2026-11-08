"""One immutable, source-neutral store of dated financial input tables."""
import json
from pathlib import Path
import pandas as pd
from utils.atomic_file import file_sha256


def read_tables(directory):
    root=Path(directory)
    manifest=json.loads((root/'financial_manifest.json').read_text(encoding='utf8'))
    frames={}
    for table,item in manifest['tables'].items():
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path)!=item['sha256']:
            raise ValueError('Dated financial input changed: '+table)
        frames[table]=pd.read_parquet(path)
    return frames,manifest

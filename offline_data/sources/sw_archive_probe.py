"""Archive public independent industry data for explicit gap verification."""
import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote
import hashlib

def main():
    root=Path('artifacts/industry_rotation_20260914/independent_source')
    root.mkdir(parents=True,exist_ok=True)
    repo='18June96/stock-analysis-system'
    tree=json.load(urlopen(f'https://api.github.com/repos/{repo}/git/trees/main?recursive=1',timeout=30))
    item=next(x for x in tree['tree'] if x['path']=='指数交易数据.csv')
    url=f'https://raw.githubusercontent.com/{repo}/{tree["sha"]}/'+quote(item['path'])
    # Fixed byte range located by read-only probes around the missing 2023-09-08.
    # Preserve the fragment verbatim; do not claim a full-file archive.
    parts=[]
    for begin in range(4_850_000,5_050_000,25_000):
        end=begin+25_000-1
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0','Range':f'bytes={begin}-{end}'}),timeout=30) as response:
            if response.status!=206 or not response.headers['Content-Range'].startswith(f'bytes {begin}-{end}/'):
                raise ValueError('Range request not honored')
            chunk=response.read()
        if len(chunk)!=end-begin+1:
            raise ValueError('Truncated range')
        parts.append(chunk)
        print(f'Archived {end+1}/{item["size"]} bytes',flush=True)
    data=b''.join(parts)
    (root/'fragment.bin').write_bytes(data)
    (root/'manifest.json').write_text(json.dumps({'repository':repo,'commit':tree['sha'],'url':url,'full_blob_sha':item['sha'],'byte_range':[4850000,5049999],'fragment_sha256':hashlib.sha256(data).hexdigest(),'source_claim':'repository README attributes data to Tushare; independent validation required'},ensure_ascii=False,indent=2),encoding='utf-8')
    import pandas as pd
    import io
    whole_lines=data.split(b'\n')[1:-1]
    p=pd.read_csv(io.BytesIO(b'ts_code,name,trade_date,open,close,vol,pe,pb\n'+b'\n'.join(whole_lines)))
    p.to_parquet(root/'fragment.parquet',index=False)
    print(p.shape,p.columns.tolist())
    print(p.head(3).to_string(index=False))
    print(p.tail(3).to_string(index=False))

if __name__=='__main__':
    main()

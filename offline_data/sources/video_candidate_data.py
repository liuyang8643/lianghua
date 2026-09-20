"""Archive free-source responses for video-factor feasibility research.

Network access stays at the data boundary. A successful response is evidence of
availability, never automatic approval of historical point-in-time semantics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests


def download_qmt_financial_versions(output: Path, runtime: Path, start: str, end: str) -> None:
    """Preserve every returned announcement vintage, including delisted codes.

    Serial batches only; missing source rows are recorded, never fabricated.
    Existing batch artifacts make an interrupted download resumable.
    """
    from xtquant import xtdata
    from offline_data import load_runtime_stock_codes
    import pandas as pd

    codes = list(map(str, load_runtime_stock_codes(runtime)))
    tables = ["PershareIndex", "Balance", "Income", "CashFlow"]
    output.mkdir(parents=True, exist_ok=True)
    identity = {"schema": "qmt-financial-vintages-research-v1", "codes": codes,
                "tables": tables, "start": start, "end": end,
                "report_type": "report_time", "source": "xtquant financial data",
                "pit_approval": "requires independent source and original-filing audit"}
    identity_path = output / "request.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("download identity changed")
    else:
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    for offset in range(0, len(codes), 20):
        batch_dir = output / f"batch_{offset:05d}"
        if (batch_dir / "manifest.json").exists():
            continue
        batch_dir.mkdir(exist_ok=True)
        batch = codes[offset:offset + 20]
        # The newer asynchronous wrapper can wait forever on unknown delisted
        # instruments. This public serial API completes each requested pair.
        xtdata.download_financial_data2(batch, tables, start_time=start, end_time=end)
        fetched = xtdata.get_financial_data(batch, tables, start_time=start, end_time=end, report_type="report_time")
        counts, hashes = {}, {}
        for table in tables:
            parts = []
            for code in batch:
                frame = fetched[code][table]
                counts[f"{code}:{table}"] = len(frame)
                if not frame.empty:
                    parts.append(frame.assign(stock_code=code))
            frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            path = batch_dir / f"{table}.parquet"
            temporary = path.with_suffix(".tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
            hashes[table] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "counts": counts, "sha256": hashes}
        temporary = batch_dir / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(batch_dir / "manifest.json")
        print(json.dumps({"event": "qmt_batch", "completed_codes": min(offset + 20, len(codes)),
                          "total_codes": len(codes), "rows": sum(counts.values())}), flush=True)


def archive_response(directory: Path, name: str, url: str, *, params=None, data=None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / f"{name}.json").exists():
        raise FileExistsError(name)
    metadata = {"name": name, "requested_url": url, "params": params, "form": data,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "pit_verified": False}
    started = time.monotonic()
    try:
        response = requests.request("POST" if data is not None else "GET", url,
                                    params=params, data=data, timeout=(10, 25),
                                    headers={"User-Agent": "Mozilla/5.0", "Referer": url})
        body = response.content
        (directory / f"{name}.body").write_bytes(body)
        metadata.update(status_code=response.status_code, final_url=response.url,
                        content_type=response.headers.get("Content-Type", ""),
                        size=len(body), sha256=hashlib.sha256(body).hexdigest())
        response.raise_for_status()
        metadata["status"] = "received"
    except requests.RequestException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    metadata["elapsed_seconds"] = time.monotonic() - started
    (directory / f"{name}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probes = [
        ("sw_catalog", "https://www.swsresearch.com/institute-sw/api/index_publish/current/",
         {"page": 1, "page_size": 50, "indextype": "一级行业"}),
        ("sw_801010", "https://www.swsresearch.com/institute-sw/api/index_publish/trend/",
         {"swindexcode": "801010", "period": "DAY"}),
        ("sina_600519_fzb", "https://quotes.sina.cn/cn/api/openapi.php/CompanyFinanceService.getFinanceReport2022",
         {"paperCode": "sh600519", "source": "fzb", "type": "0", "page": 1, "num": 200}),
        ("em_600519_balance", "https://datacenter-web.eastmoney.com/api/data/v1/get",
         {"reportName": "RPT_DMSK_FN_BALANCE", "columns": "ALL", "filter": '(SECURITY_CODE="600519")',
          "pageSize": 200, "pageNumber": 1, "sortColumns": "REPORT_DATE", "sortTypes": "-1"}),
    ]
    for name, url, params in probes:
        archive_response(args.output, name, url, params=params)
        time.sleep(1.1)
    archive_response(args.output, "cninfo_2012_reports", "https://www.cninfo.com.cn/new/hisAnnouncement/query", data={
        "pageNum": "1", "pageSize": "30", "column": "", "tabName": "fulltext", "plate": "",
        "stock": "", "searchkey": "2011年年度报告", "secid": "", "category": "",
        "trade": "", "seDate": "2012-01-01~2012-05-01", "sortName": "time", "sortType": "asc", "isHLtitle": "false"})


if __name__ == "__main__":
    main()

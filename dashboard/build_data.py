from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "data.json"

def build():
    frames = []
    for path in (ROOT / "data" / "k-line").glob("*.parquet"):
        frame = pd.read_parquet(path, columns=["time"])
        frame["trade_date"] = pd.to_datetime(frame["time"], unit="ms").dt.date
        frame["stock_code"] = path.stem
        frames.append(frame[["trade_date", "stock_code"]])
    kline = pd.concat(frames, ignore_index=True)
    kline_sets = kline.groupby("trade_date")["stock_code"].agg(set)
    financial = pd.read_parquet(ROOT / "data" / "financial" / "deep_indicators.parquet", columns=["stock_code", "report_period", "eps", "roe"]).dropna(subset=["report_period"])
    financial["report_date"] = pd.to_datetime(financial["report_period"].astype(str), format="%Y%m%d").dt.date
    financial = financial.sort_values("report_date")
    counts = kline.groupby("trade_date")["stock_code"].nunique()
    dates = [d for d in sorted(counts.index) if d >= pd.Timestamp("2000-01-01").date()]
    rows = []
    for date in dates:
        latest = financial[financial["report_date"] <= date].drop_duplicates("stock_code", keep="last")
        universe = kline_sets[date]
        latest = latest[latest.stock_code.isin(universe)]
        rows.append({"date": date.isoformat(), "kline": int(counts.get(date, 0)), "financial": int(latest.stock_code.nunique()), "eps": int(latest.eps.notna().sum()), "roe": int(latest.roe.notna().sum())})
    latest = rows[-1]
    payload = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"), "source": "data/k-line + data/financial/pershare_index.parquet", "rows": rows, "summary": {"latest": latest, "stock_files": len(frames), "financial_records": int(len(financial)), "coverage_gap": latest["kline"] - latest["financial"]}}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT.write_text(serialized, encoding="utf-8")
    (OUT.parent / "data.js").write_text(f"window.WBR_DATA = {serialized};", encoding="utf-8")

if __name__ == "__main__":
    build()

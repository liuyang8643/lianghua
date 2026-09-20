"""Build the complete sealed T-open runtime used by the live decision path.

The live snapshot is never patched field-by-field. Today's full active stock
axis is downloaded first, then the canonical full-history runtime builder
recomputes every PIT field and atomically promotes one schema-valid NPZ.
"""

from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests


logger = logging.getLogger("update_live")
LIVE_OPEN_DIR = Path(__file__).resolve().parent / "live_open"
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="
TENCENT_QUOTE_CHUNK_SIZE = 250
TENCENT_QUOTE_TIMEOUT_SECONDS = 10


def _tencent_symbol(stock_code: str) -> str:
    bare, exchange = stock_code.split(".", 1)
    prefix = "bj" if exchange == "BJ" else exchange.lower()
    if prefix not in {"sh", "sz", "bj"} or len(bare) != 6 or not bare.isdigit():
        raise ValueError(f"invalid A-share code: {stock_code!r}")
    return prefix + bare


def _fetch_live_open_overlay(
    stock_codes: Sequence[str],
    decision_date: date,
) -> Path:
    """Fetch one complete batch-quote axis and atomically seal T open/preClose."""

    codes = tuple(str(code).strip().upper() for code in stock_codes)
    if not codes or len(set(codes)) != len(codes):
        raise ValueError("live quote stock axis must be non-empty and unique")
    symbol_to_code = {_tencent_symbol(code): code for code in codes}
    rows: list[dict[str, object]] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    symbols = tuple(symbol_to_code)
    for start in range(0, len(symbols), TENCENT_QUOTE_CHUNK_SIZE):
        chunk = symbols[start : start + TENCENT_QUOTE_CHUNK_SIZE]
        response = requests.get(
            TENCENT_QUOTE_URL + ",".join(chunk),
            headers=headers,
            timeout=TENCENT_QUOTE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        text = response.content.decode("gbk", errors="strict")
        for item in text.split(";"):
            if '="' not in item:
                continue
            variable, quoted = item.split("=", 1)
            symbol = variable.removeprefix("v_").strip()
            code = symbol_to_code.get(symbol)
            if code is None:
                continue
            payload = quoted.strip().strip('"')
            fields = payload.split("~")
            if len(fields) <= 30:
                raise RuntimeError(f"腾讯行情字段不足: {code}")
            quote_date = fields[30].strip()[:8]
            if quote_date != decision_date.strftime("%Y%m%d"):
                raise RuntimeError(
                    f"腾讯行情日期不是决策日: {code}={quote_date}"
                )
            try:
                preclose = float(fields[4])
                open_price = float(fields[5])
            except ValueError as exc:
                raise RuntimeError(f"腾讯开盘行情不是数值: {code}") from exc
            if not np.isfinite(preclose) or preclose <= 0.0:
                raise RuntimeError(f"腾讯前收无效: {code}")
            rows.append(
                {
                    "trade_date": decision_date.isoformat(),
                    "stock_code": code,
                    "open": (
                        open_price
                        if np.isfinite(open_price) and open_price > 0.0
                        else np.nan
                    ),
                    "preClose": preclose,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["stock_code"].duplicated().any():
        raise RuntimeError("腾讯全轴开盘行情为空或股票重复")
    actual = set(frame["stock_code"])
    if actual != set(codes):
        missing = sorted(set(codes) - actual)
        unexpected = sorted(actual - set(codes))
        raise RuntimeError(
            f"腾讯全轴开盘行情覆盖不完整: missing={missing}, unexpected={unexpected}"
        )
    frame = frame.set_index("stock_code").loc[list(codes)].reset_index()
    LIVE_OPEN_DIR.mkdir(parents=True, exist_ok=True)
    target = LIVE_OPEN_DIR / f"{decision_date.isoformat()}.parquet"
    temporary = target.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(target)
    return target


def build_live_runtime(
    decision_date: date | None = None,
    *,
    candidate_codes: Sequence[str] | None = None,
) -> Path:
    """Materialise a full-axis T-open runtime from explicit fetch candidates."""
    from data.build_runtime import build_runtime
    from data.db.issue_price import resolve_terminal_active_codes
    from data.db.stock_list import load_current_stock_codes
    from data.kline_mootdx import resolve_recent_range, update_recent

    target = decision_date or date.today()
    _, _, resolved = resolve_recent_range(1, target)
    if resolved != target:
        raise RuntimeError(f"{target.isoformat()} 不是交易日，拒绝构建实盘快照")
    kline_dir = Path(__file__).resolve().parent / "k-line"
    current_codes = tuple(load_current_stock_codes())
    local_kline_codes = {
        path.stem for path in kline_dir.glob("*.parquet")
    }
    complete_active_codes = list(
        resolve_terminal_active_codes(
            current_codes,
            target,
            local_kline_codes,
        )
    )
    if not complete_active_codes:
        raise RuntimeError("实盘快照的完整活跃股票轴为空")
    if candidate_codes is None:
        fetch_codes = complete_active_codes
    else:
        normalized = tuple(
            str(code).strip().upper() for code in candidate_codes
        )
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("candidate_codes 必须非空且不得重复")
        selected = set(normalized)
        unknown = sorted(selected.difference(current_codes))
        if unknown:
            raise ValueError(
                "candidate_codes 包含当前股票列表之外代码: "
                + ", ".join(unknown[:20])
            )
        active_set = set(complete_active_codes)
        prelisting = sorted(selected.difference(active_set))
        if prelisting:
            raise ValueError(
                "candidate_codes 包含决策日尚未上市代码: "
                + ", ".join(prelisting[:20])
            )
        first_day_missing_axis = active_set.difference(local_kline_codes)
        required = selected | first_day_missing_axis
        fetch_codes = [
            code for code in complete_active_codes if code in required
        ]
        if not fetch_codes:
            raise RuntimeError("prefilter 后的实盘 K 线候选为空")

    logger.info(
        "下载 %s 的 T-open K 线候选：%d/%d 只",
        target,
        len(fetch_codes),
        len(complete_active_codes),
    )
    update_recent(
        1,
        anchor_date=target,
        codes=fetch_codes,
        strict=True,
    )
    live_open_overlay = _fetch_live_open_overlay(complete_active_codes, target)
    runtime_path = Path(
        build_runtime(
            partial_live_candidates=fetch_codes,
            live_open_overlay=live_open_overlay,
        )
    ).resolve()
    with np.load(runtime_path, allow_pickle=False) as payload:
        latest = np.asarray(payload["trade_dates"], dtype="datetime64[D]")[-1]
    if latest != np.datetime64(target, "D"):
        raise RuntimeError(
            f"实盘 runtime 最后一行是 {latest}，不是决策日 {target.isoformat()}"
        )
    logger.info("完整 T-open runtime 已封存：%s", runtime_path)
    return runtime_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args(argv)
    path = build_live_runtime(date.fromisoformat(args.date))
    print(path, flush=True)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


__all__ = ["build_live_runtime", "main"]

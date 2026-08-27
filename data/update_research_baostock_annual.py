"""Checkpointed BaoStock annual-data download for vintage-PIT research."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "strategy_opt_20260730"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime import load_runtime_stock_codes  # noqa: E402


OUTPUT_PATH = RESULT_DIR / "baostock_annual_growth_2010_2018.parquet"
STATUS_PATH = RESULT_DIR / "baostock_annual_growth_2010_2018_status.json"
DATASETS = {
    "growth": {
        "query": "query_growth_data",
        "fields": (
            "code",
            "pubDate",
            "statDate",
            "YOYEquity",
            "YOYAsset",
            "YOYNI",
            "YOYEPSBasic",
            "YOYPNI",
        ),
    },
    "profit": {
        "query": "query_profit_data",
        "fields": (
            "code",
            "pubDate",
            "statDate",
            "roeAvg",
            "npMargin",
            "gpMargin",
            "netProfit",
            "epsTTM",
            "MBRevenue",
            "totalShare",
            "liqaShare",
        ),
    },
}
POOL_PREFIXES = ("60", "00", "30")
_SOCKET_TIMEOUT = 10.0


def baostock_code(stock_code: str) -> str:
    bare, suffix = str(stock_code).split(".", 1)
    return f"{suffix.lower()}.{bare}"


def _close_worker_socket() -> None:
    import baostock.common.context as context

    sock = getattr(context, "default_socket", None)
    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass
    if hasattr(context, "default_socket"):
        delattr(context, "default_socket")


def _connect_worker(timeout: float | None = None) -> None:
    import baostock as bs
    import baostock.common.context as context

    global _SOCKET_TIMEOUT
    if timeout is not None:
        _SOCKET_TIMEOUT = float(timeout)
    _close_worker_socket()
    login = bs.login()
    if login.error_code != "0":
        raise ConnectionError(
            f"BaoStock login failed: {login.error_code} {login.error_msg}"
        )
    sock = getattr(context, "default_socket")
    sock.settimeout(_SOCKET_TIMEOUT)


def _worker_init(timeout: float) -> None:
    _connect_worker(timeout)


def _source_preflight(timeout: float) -> bool:
    """Fail once in the parent instead of endlessly respawning pool workers."""
    try:
        _connect_worker(timeout)
    except (OSError, ConnectionError, socket.timeout) as exc:
        print(f"SOURCE_BLOCKED: {type(exc).__name__}: {exc}", flush=True)
        _close_worker_socket()
        return False
    _close_worker_socket()
    return True


def _query_one(task: tuple[str, str, int]) -> dict:
    import baostock as bs

    dataset, code, year = task
    last_error = ""
    for attempt in range(3):
        try:
            result = getattr(bs, DATASETS[dataset]["query"])(
                code=code,
                year=year,
                quarter=4,
            )
            if result.error_code != "0":
                if result.error_code == "10001011":
                    return {
                        "task": [dataset, code, year],
                        "status": "blocked",
                        "rows": [],
                        "attempts": attempt + 1,
                        "error": (
                            f"{result.error_code} {result.error_msg}"
                        ),
                    }
                raise RuntimeError(
                    f"{result.error_code} {result.error_msg}"
                )
            rows = []
            while result.next():
                values = result.get_row_data()
                rows.append(
                    {
                        **dict(zip(result.fields, values)),
                        "query_year": year,
                        "query_quarter": 4,
                        "source_status": "ok",
                    }
                )
            if not rows:
                rows.append(
                    {
                        "code": code,
                        "query_year": year,
                        "query_quarter": 4,
                        "source_status": "empty",
                    }
                )
            return {
                "task": [dataset, code, year],
                "status": rows[0]["source_status"],
                "rows": rows,
                "attempts": attempt + 1,
            }
        except (OSError, RuntimeError, socket.timeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            try:
                _connect_worker()
            except (OSError, ConnectionError, socket.timeout) as reconnect:
                last_error += (
                    f"; reconnect {type(reconnect).__name__}: {reconnect}"
                )
    return {
        "task": [dataset, code, year],
        "status": "error",
        "rows": [],
        "attempts": 3,
        "error": last_error,
    }


def dataset_paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "growth":
        return OUTPUT_PATH, STATUS_PATH
    stem = f"baostock_annual_{dataset}_2010_2018"
    return RESULT_DIR / f"{stem}.parquet", RESULT_DIR / f"{stem}_status.json"


def _existing_rows(dataset: str, output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame(
            columns=[
                *DATASETS[dataset]["fields"],
                "query_year",
                "query_quarter",
                "source_status",
            ]
        )
    return pd.read_parquet(output_path)


def _write_checkpoint(
    dataset: str,
    output_path: Path,
    status_path: Path,
    rows: list[dict],
    completed: set[tuple[str, int]],
    errors: list[dict],
    *,
    started: float,
    total_tasks: int,
) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(
            subset=["code", "query_year", "query_quarter"],
            keep="last",
        ).sort_values(["code", "query_year"])
    temporary = output_path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)

    status = {
        "source": (
            f"BaoStock 0.9.2 {DATASETS[dataset]['query']} quarter=4"
        ),
        "vintage_contract": (
            "retain source pubDate/statDate and activate no earlier than the "
            "next trading day in later factor research"
        ),
        "output_path": str(output_path),
        "total_tasks": total_tasks,
        "completed_tasks": len(completed),
        "rows": len(frame),
        "errors": errors[-200:],
        "error_count": len(errors),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "complete": len(completed) == total_tasks and not errors,
    }
    temporary_status = status_path.with_suffix(".tmp.json")
    temporary_status.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_status, status_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="growth",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--years",
        default="2010-2018",
        help="inclusive range such as 2010-2018",
    )
    parser.add_argument("--checkpoint-every", type=int, default=250)
    return parser.parse_args()


def year_range(value: str) -> list[int]:
    start_text, end_text = value.split("-", 1)
    start, end = int(start_text), int(end_text)
    if start > end:
        raise ValueError("year range start must not exceed end")
    return list(range(start, end + 1))


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.timeout <= 0.0:
        raise ValueError("timeout must be positive")

    years = year_range(args.years)
    output_path, status_path = dataset_paths(args.dataset)
    codes = sorted(
        baostock_code(code)
        for code in load_runtime_stock_codes()
        if str(code).startswith(POOL_PREFIXES)
    )
    if args.limit is not None:
        codes = codes[: args.limit]
    all_tasks = [
        (args.dataset, code, year) for code in codes for year in years
    ]
    total_tasks = len(all_tasks)

    existing = _existing_rows(args.dataset, output_path)
    completed = {
        (str(row.code), int(row.query_year))
        for row in existing.itertuples()
    }
    rows = existing.to_dict("records")
    pending = [
        task for task in all_tasks if (task[1], task[2]) not in completed
    ]
    errors: list[dict] = []
    started = time.perf_counter()
    print(
        f"dataset={args.dataset} codes={len(codes)} "
        f"years={years[0]}-{years[-1]} "
        f"tasks={total_tasks} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    if not pending:
        _write_checkpoint(
            args.dataset,
            output_path,
            status_path,
            rows,
            completed,
            errors,
            started=started,
            total_tasks=total_tasks,
        )
        return
    if not _source_preflight(args.timeout):
        return

    context = mp.get_context("spawn")
    with context.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.timeout,),
        maxtasksperchild=500,
    ) as pool:
        for index, result in enumerate(
            pool.imap_unordered(_query_one, pending, chunksize=1),
            1,
        ):
            task = (str(result["task"][1]), int(result["task"][2]))
            if result["status"] in {"error", "blocked"}:
                errors.append(result)
            else:
                completed.add(task)
                rows.extend(result["rows"])
            if (
                index % args.checkpoint_every == 0
                or index == len(pending)
            ):
                _write_checkpoint(
                    args.dataset,
                    output_path,
                    status_path,
                    rows,
                    completed,
                    errors,
                    started=started,
                    total_tasks=total_tasks,
                )
                elapsed = time.perf_counter() - started
                print(
                    f"processed={index}/{len(pending)} "
                    f"completed={len(completed)}/{total_tasks} "
                    f"rows={len(rows)} errors={len(errors)} "
                    f"rate={index / max(elapsed, 1e-9):.2f} tasks/s",
                    flush=True,
                )
            if result["status"] == "blocked":
                _write_checkpoint(
                    args.dataset,
                    output_path,
                    status_path,
                    rows,
                    completed,
                    errors,
                    started=started,
                    total_tasks=total_tasks,
                )
                print(
                    "SOURCE_BLOCKED: BaoStock returned 10001011; "
                    "checkpoint saved and remaining tasks left pending",
                    flush=True,
                )
                pool.terminate()
                break


if __name__ == "__main__":
    main()
